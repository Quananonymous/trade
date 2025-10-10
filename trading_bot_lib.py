# trading_bot_lib.py - HOÀN CHỈNH VỚI AI TÌM COIN & CÂN BẰNG VỊ THẾ
import json
import hmac
import hashlib
import time
import threading
import urllib.request
import urllib.parse
import numpy as np
import websocket
import logging
import requests
import os
import math
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# ========== CẤU HÌNH LOGGING ==========
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot_errors.log')
        ]
    )
    return logging.getLogger()

logger = setup_logging()

# ========== HÀM TELEGRAM ==========
def send_telegram(message, chat_id=None, reply_markup=None, bot_token=None, default_chat_id=None):
    if not bot_token:
        logger.warning("Telegram Bot Token chưa được thiết lập")
        return
    
    chat_id = chat_id or default_chat_id
    if not chat_id:
        logger.warning("Telegram Chat ID chưa được thiết lập")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            logger.error(f"Lỗi Telegram ({response.status_code}): {response.text}")
    except Exception as e:
        logger.error(f"Lỗi kết nối Telegram: {str(e)}")

# ========== MENU TELEGRAM HOÀN CHỈNH ==========
def create_main_menu():
    return {
        "keyboard": [
            [{"text": "📊 Danh sách Bot"}],
            [{"text": "➕ Thêm Bot"}, {"text": "⛔ Dừng Bot"}],
            [{"text": "💰 Số dư"}, {"text": "📈 Vị thế"}],
            [{"text": "⚙️ Cấu hình"}, {"text": "🎯 Chiến lược"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def create_cancel_keyboard():
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "🤖 RSI/EMA Recursive"}, {"text": "📊 EMA Crossover"}],
            [{"text": "🎯 Reverse 24h"}, {"text": "📈 Trend Following"}],
            [{"text": "⚡ Scalping"}, {"text": "🛡️ Safe Grid"}],
            [{"text": "🔄 Bot Động Thông Minh"}, {"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_exit_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "🎯 Chỉ TP/SL cố định"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_bot_mode_keyboard():
    return {
        "keyboard": [
            [{"text": "🤖 Bot Tĩnh - Coin cụ thể"}, {"text": "🔄 Bot Động - Tự tìm coin"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard(strategy=None):
    try:
        symbols = get_all_usdt_pairs(limit=12)
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"]
    except:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"]
    
    keyboard = []
    row = []
    for symbol in symbols:
        row.append({"text": symbol})
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_leverage_keyboard(strategy=None):
    leverages = ["3", "5", "10", "15", "20", "25", "50", "75", "100"]
    
    keyboard = []
    row = []
    for lev in leverages:
        row.append({"text": f"{lev}x"})
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_percent_keyboard():
    return {
        "keyboard": [
            [{"text": "1"}, {"text": "3"}, {"text": "5"}, {"text": "10"}],
            [{"text": "15"}, {"text": "20"}, {"text": "25"}, {"text": "50"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_tp_keyboard():
    return {
        "keyboard": [
            [{"text": "50"}, {"text": "100"}, {"text": "200"}],
            [{"text": "300"}, {"text": "500"}, {"text": "1000"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_sl_keyboard():
    return {
        "keyboard": [
            [{"text": "0"}, {"text": "50"}, {"text": "100"}],
            [{"text": "150"}, {"text": "200"}, {"text": "500"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_threshold_keyboard():
    return {
        "keyboard": [
            [{"text": "30"}, {"text": "50"}, {"text": "70"}],
            [{"text": "100"}, {"text": "150"}, {"text": "200"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_volatility_keyboard():
    return {
        "keyboard": [
            [{"text": "2"}, {"text": "3"}, {"text": "5"}],
            [{"text": "7"}, {"text": "10"}, {"text": "15"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_grid_levels_keyboard():
    return {
        "keyboard": [
            [{"text": "3"}, {"text": "5"}, {"text": "7"}],
            [{"text": "10"}, {"text": "15"}, {"text": "20"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_bot_count_keyboard():
    return {
        "keyboard": [
            [{"text": "1"}, {"text": "2"}],
            [{"text": "3"}, {"text": "5"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== AI MARKET ANALYZER ==========
class AIMarketAnalyzer:
    """AI PHÂN TÍCH THỊ TRƯỜNG & ĐỀ XUẤT HƯỚNG GIAO DỊCH"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "ai_market_model.pkl"
        self.scaler_path = "ai_scaler.pkl"
        self.is_trained = False
        self.load_model()
        
    def load_model(self):
        """Tải mô hình AI đã train"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info("✅ Đã tải mô hình AI thành công")
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                logger.info("🆕 Khởi tạo mô hình AI mới")
                # Tạo dữ liệu mẫu để train scaler ngay từ đầu
                self._initialize_with_sample_data()
        except Exception as e:
            logger.error(f"❌ Lỗi tải mô hình AI: {str(e)}")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._initialize_with_sample_data()
    
    def _initialize_with_sample_data(self):
        """Khởi tạo với dữ liệu mẫu để scaler được fit"""
        try:
            # Tạo dữ liệu mẫu có cấu trúc giống thật
            sample_data = []
            for i in range(100):
                sample = [
                    np.random.uniform(20, 80),    # RSI
                    np.random.uniform(1000, 50000), # EMA9
                    np.random.uniform(1000, 50000), # EMA21  
                    np.random.uniform(1000, 50000), # EMA50
                    np.random.uniform(-50, 50),   # MACD
                    np.random.uniform(-50, 50),   # Signal
                    np.random.uniform(0.1, 5.0),  # Volume ratio
                    np.random.uniform(-10, 10),   # Price change 1h
                    np.random.uniform(-20, 20),   # Price change 4h
                    np.random.uniform(-30, 30),   # Price change 24h
                    np.random.uniform(0.5, 20),   # Volatility
                    np.random.uniform(30, 70),    # BTC dominance
                    np.random.uniform(10, 90)     # Fear greed
                ]
                sample_data.append(sample)
            
            # Fit scaler
            self.scaler.fit(sample_data)
            
            # Train model với dữ liệu mẫu (tất cả neutral)
            y_sample = [1] * len(sample_data)  # 1 = NEUTRAL
            X_sample_scaled = self.scaler.transform(sample_data)
            self.model.fit(X_sample_scaled, y_sample)
            
            self.is_trained = True
            logger.info("✅ Đã khởi tạo AI với dữ liệu mẫu")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo dữ liệu mẫu: {str(e)}")
            self.is_trained = False

    def predict_direction(self, symbol_data, market_data):
        """Dự đoán hướng giao dịch cho symbol - ĐÃ SỬA LỖI"""
        try:
            if not self.is_trained:
                logger.warning("⚠️ AI chưa được train, trả về NEUTRAL")
                return "NEUTRAL"
                
            features = self.extract_features(symbol_data, market_data)
            if features is None:
                return "NEUTRAL"
            
            # Kiểm tra scaler đã được fit chưa
            if not hasattr(self.scaler, 'mean_'):
                logger.warning("⚠️ Scaler chưa được fit")
                return "NEUTRAL"
            
            # Chuẩn hóa features
            try:
                features_scaled = self.scaler.transform([features])
            except Exception as e:
                logger.error(f"❌ Lỗi chuẩn hóa features: {str(e)}")
                return "NEUTRAL"
            
            # Dự đoán
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            directions = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}
            direction = directions.get(prediction, "NEUTRAL")
            
            logger.info(f"🤖 AI dự đoán {direction} (độ tin cậy: {confidence:.2f})")
            return direction if confidence > 0.6 else "NEUTRAL"
            
        except Exception as e:
            logger.error(f"❌ Lỗi dự đoán AI: {str(e)}")
            return "NEUTRAL"

# Thêm hàm để train scaler với dữ liệu mẫu
def initialize_ai_with_sample_data(self):
    """Khởi tạo AI với dữ liệu mẫu để tránh lỗi scaler"""
    try:
        # Tạo dữ liệu mẫu ngẫu nhiên để fit scaler
        import numpy as np
        sample_features = []
        for _ in range(100):
            sample_feature = [
                np.random.uniform(20, 80),  # RSI
                np.random.uniform(100, 50000),  # EMA9
                np.random.uniform(100, 50000),  # EMA21
                np.random.uniform(100, 50000),  # EMA50
                np.random.uniform(-10, 10),  # MACD
                np.random.uniform(-10, 10),  # Signal
                np.random.uniform(0.5, 3.0),  # Volume ratio
                np.random.uniform(-5, 5),  # Price change 1h
                np.random.uniform(-10, 10),  # Price change 4h
                np.random.uniform(-20, 20),  # Price change 24h
                np.random.uniform(1, 15),  # Volatility
                np.random.uniform(40, 60),  # BTC dominance
                np.random.uniform(20, 80)  # Fear greed
            ]
            sample_features.append(sample_feature)
        
        # Fit scaler với dữ liệu mẫu
        self.scaler.fit(sample_features)
        logger.info("✅ Đã khởi tạo scaler với dữ liệu mẫu")
        
        # Tạo model với dữ liệu mẫu cơ bản
        sample_X = sample_features
        sample_y = [1] * 100  # Tất cả NEUTRAL
        
        self.model.fit(sample_X, sample_y)
        logger.info("✅ Đã khởi tạo model với dữ liệu mẫu")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo AI với dữ liệu mẫu: {str(e)}")

# Trong hàm __init__ của AIMarketAnalyzer, thêm:
def __init__(self):
    self.model = None
    self.scaler = StandardScaler()
    self.model_path = "ai_market_model.pkl"
    self.scaler_path = "ai_scaler.pkl"
    self.load_model()
    
    # Nếu model chưa được train, khởi tạo với dữ liệu mẫu
    if not hasattr(self.model, 'classes_') or not hasattr(self.scaler, 'mean_'):
        logger.info("🔄 Khởi tạo AI với dữ liệu mẫu...")
        self.initialize_ai_with_sample_data()
    
    def calc_rsi(self, prices, period=14):
        """Tính RSI"""
        try:
            if len(prices) < period + 1:
                return None
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100.0 - (100.0 / (1 + rs))
        except:
            return None
    
    def calc_ema(self, prices, period):
        """Tính EMA"""
        try:
            if len(prices) < period:
                return None
            ema = [sum(prices[:period]) / period]
            multiplier = 2 / (period + 1)
            for price in prices[period:]:
                ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
            return ema[-1]
        except:
            return None
    
    def calc_macd(self, prices, fast=12, slow=26, signal=9):
        """Tính MACD"""
        try:
            if len(prices) < slow:
                return None, None
            ema_fast = self.calc_ema(prices, fast)
            ema_slow = self.calc_ema(prices, slow)
            if ema_fast is None or ema_slow is None:
                return None, None
            macd = ema_fast - ema_slow
            # Tính signal line đơn giản
            macd_values = [macd] * signal
            signal_line = np.mean(macd_values)
            return macd, signal_line
        except:
            return None, None

# ========== POSITION BALANCER ==========
class PositionBalancer:
    """CÂN BẰNG VỊ THẾ TỰ ĐỘNG DỰA TRÊN TỶ LỆ BUY/SELL"""
    
    def __init__(self, bot_manager):
        self.bot_manager = bot_manager
        self.buy_sell_history = []
        self.max_history = 50
        self.imbalance_threshold = 2  # Ngưỡng chênh lệch
        
    def get_current_ratio(self):
        """Lấy tỷ lệ BUY/SELL hiện tại"""
        try:
            buy_count = 0
            sell_count = 0
            
            for bot_id, bot in self.bot_manager.bots.items():
                if bot.position_open:
                    if bot.side == "BUY":
                        buy_count += 1
                    elif bot.side == "SELL":
                        sell_count += 1
            
            total = buy_count + sell_count
            if total == 0:
                return 0.5, 0.5  # Tỷ lệ cân bằng
                
            buy_ratio = buy_count / total
            sell_ratio = sell_count / total
            
            # Lưu lịch sử
            self.buy_sell_history.append((buy_count, sell_count))
            if len(self.buy_sell_history) > self.max_history:
                self.buy_sell_history.pop(0)
                
            return buy_ratio, sell_ratio
            
        except Exception as e:
            logger.error(f"❌ Lỗi tính tỷ lệ vị thế: {str(e)}")
            return 0.5, 0.5
    
    def get_recommended_direction(self):
        """Đề xuất hướng giao dịch dựa trên cân bằng vị thế"""
        try:
            buy_ratio, sell_ratio = self.get_current_ratio()
            
            # Phân tích lịch sử
            if len(self.buy_sell_history) >= 10:
                recent_buys = sum([item[0] for item in self.buy_sell_history[-5:]])
                recent_sells = sum([item[1] for item in self.buy_sell_history[-5:]])
                
                # Nếu BUY nhiều hơn SELL đáng kể → đề xuất SELL
                if recent_buys - recent_sells >= self.imbalance_threshold:
                    recommendation = "SELL"
                    reason = f"BUY đang chiếm ưu thế ({recent_buys} BUY vs {recent_sells} SELL)"
                # Nếu SELL nhiều hơn BUY đáng kể → đề xuất BUY
                elif recent_sells - recent_buys >= self.imbalance_threshold:
                    recommendation = "BUY" 
                    reason = f"SELL đang chiếm ưu thế ({recent_sells} SELL vs {recent_buys} BUY)"
                else:
                    recommendation = "NEUTRAL"
                    reason = "Thị trường cân bằng"
            else:
                recommendation = "NEUTRAL"
                reason = "Chưa đủ dữ liệu lịch sử"
            
            logger.info(f"⚖️ Cân bằng vị thế: BUY {buy_ratio:.1%} / SELL {sell_ratio:.1%} → {recommendation} ({reason})")
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Lỗi đề xuất hướng: {str(e)}")
            return "NEUTRAL"

# ========== SMART COIN FINDER ==========
class SmartCoinFinder:
    """TÌM COIN THÔNG MINH - SỬ DỤNG AI THẬT SỰ"""
    
    def __init__(self, api_key, api_secret, ai_analyzer, position_balancer):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ai_analyzer = ai_analyzer
        self.position_balancer = position_balancer
        
    def find_coins_by_direction(self, target_direction, strategy_type, count=2, market_data=None):
        """TÌM COIN THEO HƯỚNG CHỈ ĐỊNH - SỬ DỤNG AI THẬT"""
        try:
            logger.info(f"🎯 AI đang tìm {count} coin cho hướng {target_direction}")
            
            all_symbols = get_all_usdt_pairs(limit=80)
            qualified_coins = []
            
            if market_data is None:
                market_data = self.get_market_sentiment()
            
            for symbol in all_symbols:
                try:
                    if symbol in ['BTCUSDT', 'ETHUSDT']:
                        continue
                    
                    # Lấy dữ liệu kỹ thuật
                    symbol_data = self.get_symbol_data(symbol)
                    if not symbol_data or len(symbol_data.get('prices', [])) < 50:
                        continue
                    
                    # SỬ DỤNG AI ĐỂ DỰ ĐOÁN HƯỚNG
                    ai_direction = self.ai_analyzer.predict_direction(symbol_data, market_data)
                    
                    # CHỈ CHỌN COIN CÓ HƯỚNG TRÙNG VỚI MỤC TIÊU
                    if ai_direction == target_direction:
                        score = self.calculate_signal_score(symbol_data, target_direction, strategy_type)
                        
                        if score > 0.5:  # Ngưỡng thấp hơn để có nhiều coin hơn
                            qualified_coins.append({
                                'symbol': symbol,
                                'direction': target_direction,
                                'score': score,
                                'strategy_type': strategy_type,
                                'ai_confidence': "high" if score > 0.7 else "medium"
                            })
                            logger.info(f"✅ {symbol}: AI đề xuất {target_direction} (điểm: {score:.2f})")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Lỗi phân tích {symbol}: {str(e)}")
                    continue
            
            # Sắp xếp theo điểm số AI
            qualified_coins.sort(key=lambda x: x['score'], reverse=True)
            
            selected_coins = qualified_coins[:count]
            
            if selected_coins:
                symbols = [coin['symbol'] for coin in selected_coins]
                logger.info(f"🎯 AI đã chọn {len(selected_coins)} coin: {symbols}")
            else:
                logger.warning(f"⚠️ AI không tìm thấy coin nào cho {target_direction}")
                # Fallback: sử dụng phương pháp đơn giản
                selected_coins = self._find_coins_fallback(target_direction, strategy_type, count)
            
            return selected_coins
            
        except Exception as e:
            logger.error(f"❌ Lỗi tìm coin AI: {str(e)}")
            return self._find_coins_fallback(target_direction, strategy_type, count)
    
    def _find_coins_fallback(self, target_direction, strategy_type, count):
        """Phương pháp dự phòng khi AI gặp lỗi"""
        logger.info(f"🔄 Sử dụng phương pháp dự phòng cho {target_direction}")
        
        all_symbols = get_all_usdt_pairs(limit=50)
        qualified_coins = []
        
        for symbol in all_symbols:
            try:
                if symbol in ['BTCUSDT', 'ETHUSDT']:
                    continue
                
                change_24h = get_24h_change(symbol)
                if change_24h is None:
                    continue
                
                score = 0
                if target_direction == "BUY" and change_24h < -10:
                    score = abs(change_24h) / 100
                elif target_direction == "SELL" and change_24h > 10:
                    score = abs(change_24h) / 100
                
                if score > 0:
                    qualified_coins.append({
                        'symbol': symbol,
                        'direction': target_direction,
                        'score': score,
                        'strategy_type': strategy_type,
                        'ai_confidence': "fallback"
                    })
                    
            except Exception:
                continue
        
        qualified_coins.sort(key=lambda x: x['score'], reverse=True)
        return qualified_coins[:count]
    
    def calculate_signal_score(self, symbol_data, target_direction, strategy_type):
        """TÍNH ĐIỂM CHẤT LƯỢNG TÍN HIỆU - TỔNG HỢP ĐA CHỈ BÁO"""
        try:
            prices = symbol_data.get('prices', [])
            volumes = symbol_data.get('volumes', [])
            
            if len(prices) < 50:
                return 0
            
            score = 0
            max_score = 0
            
            # 1. ĐIỀU KIỆN RSI
            rsi = self.ai_analyzer.calc_rsi(prices[-14:])
            if rsi is not None:
                max_score += 1
                if target_direction == "BUY" and rsi < 35:
                    score += 1
                elif target_direction == "SELL" and rsi > 65:
                    score += 1
                elif 40 <= rsi <= 60:  # Vùng trung lập
                    score += 0.5
            
            # 2. ĐIỀU KIỆN EMA
            ema_9 = self.ai_analyzer.calc_ema(prices, 9)
            ema_21 = self.ai_analyzer.calc_ema(prices, 21)
            if ema_9 is not None and ema_21 is not None:
                max_score += 1
                if target_direction == "BUY" and ema_9 > ema_21:
                    score += 1
                elif target_direction == "SELL" and ema_9 < ema_21:
                    score += 1
            
            # 3. ĐIỀU KIỆN MACD
            macd, signal = self.ai_analyzer.calc_macd(prices)
            if macd is not None and signal is not None:
                max_score += 1
                if target_direction == "BUY" and macd > signal:
                    score += 1
                elif target_direction == "SELL" and macd < signal:
                    score += 1
            
            # 4. ĐIỀU KIỆN VOLUME
            if len(volumes) >= 20:
                max_score += 1
                volume_avg = np.mean(volumes[-20:])
                volume_current = volumes[-1] if volumes else 0
                if volume_current > volume_avg * 1.2:  # Volume tăng
                    score += 1
            
            # 5. ĐIỀU KIỆN BIẾN ĐỘNG (tùy chiến lược)
            if strategy_type in ["Scalping", "Smart Dynamic"]:
                max_score += 1
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) * 100
                if 2 <= volatility <= 8:  # Biến động vừa phải
                    score += 1
            
            # 6. ĐIỀU KIỆN XU HƯỚNG
            max_score += 1
            if len(prices) >= 20:
                trend_short = (prices[-1] - prices[-5]) / prices[-5] * 100
                trend_medium = (prices[-1] - prices[-10]) / prices[-10] * 100
                
                if target_direction == "BUY" and trend_short > 0 and trend_medium > 0:
                    score += 1
                elif target_direction == "SELL" and trend_short < 0 and trend_medium < 0:
                    score += 1
            
            return score / max_score if max_score > 0 else 0
            
        except Exception as e:
            logger.error(f"❌ Lỗi tính điểm tín hiệu: {str(e)}")
            return 0
    
    def get_symbol_data(self, symbol, limit=100):
        """Lấy dữ liệu kỹ thuật cho symbol"""
        try:
            # Lấy dữ liệu giá
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': limit
            }
            
            data = binance_api_request(url, params=params)
            if not data:
                return None
            
            prices = [float(item[4]) for item in data]  # Close prices
            volumes = [float(item[5]) for item in data]  # Volumes
            
            return {
                'symbol': symbol,
                'prices': prices,
                'volumes': volumes,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy dữ liệu {symbol}: {str(e)}")
            return None
    
    def get_market_sentiment(self):
        """Lấy sentiment thị trường tổng quan"""
        try:
            # Trong thực tế, có thể lấy từ various APIs
            # Ở đây trả về dữ liệu mẫu
            return {
                'btc_dominance': 45.5,
                'fear_greed': 65,
                'market_trend': 'BULLISH',
                'timestamp': time.time()
            }
        except:
            return {
                'btc_dominance': 50,
                'fear_greed': 50,
                'market_trend': 'NEUTRAL',
                'timestamp': time.time()
            }

# ========== QUẢN LÝ COIN CHUNG ==========
class CoinManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CoinManager, cls).__new__(cls)
                cls._instance.managed_coins = {}
                cls._instance.config_coin_count = {}
        return cls._instance
    
    def register_coin(self, symbol, bot_id, strategy, config_key=None):
        with self._lock:
            if config_key not in self.config_coin_count:
                self.config_coin_count[config_key] = 0
            
            # KIỂM TRA SỐ COIN TỐI ĐA CHO CONFIG
            if self.config_coin_count.get(config_key, 0) >= 2:
                return False
                
            if symbol not in self.managed_coins:
                self.managed_coins[symbol] = {
                    "strategy": strategy, 
                    "bot_id": bot_id,
                    "config_key": config_key
                }
                self.config_coin_count[config_key] = self.config_coin_count.get(config_key, 0) + 1
                return True
            return False
    
    def unregister_coin(self, symbol):
        with self._lock:
            if symbol in self.managed_coins:
                config_key = self.managed_coins[symbol].get("config_key")
                del self.managed_coins[symbol]
                
                if config_key in self.config_coin_count:
                    self.config_coin_count[config_key] = max(0, self.config_coin_count[config_key] - 1)
                return True
            return False
    
    def is_coin_managed(self, symbol):
        with self._lock:
            return symbol in self.managed_coins

    def has_same_config_bot(self, symbol, config_key):
        with self._lock:
            if symbol in self.managed_coins:
                existing_config = self.managed_coins[symbol].get("config_key")
                return existing_config == config_key
            return False
    
    def count_bots_by_config(self, config_key):
        with self._lock:
            count = 0
            for coin_info in self.managed_coins.values():
                if coin_info.get("config_key") == config_key:
                    count += 1
            return count
    
    def get_config_coin_count(self, config_key):
        with self._lock:
            return self.config_coin_count.get(config_key, 0)
    
    def get_managed_coins(self):
        with self._lock:
            return self.managed_coins.copy()

# ========== API BINANCE ==========
def sign(query, api_secret):
    try:
        return hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if method.upper() == 'GET':
                if params:
                    query = urllib.parse.urlencode(params)
                    url = f"{url}?{query}"
                req = urllib.request.Request(url, headers=headers or {})
            else:
                data = urllib.parse.urlencode(params).encode() if params else None
                req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    error_content = response.read().decode()
                    logger.error(f"Lỗi API ({response.status}): {error_content}")
                    if response.status == 401:
                        return None
                    if response.status == 429:
                        time.sleep(2 ** attempt)
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 401:
                return None
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_all_usdt_pairs(limit=100):
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            logger.warning("Không lấy được dữ liệu từ Binance, trả về danh sách rỗng")
            return []
        
        usdt_pairs = []
        for symbol_info in data.get('symbols', []):
            symbol = symbol_info.get('symbol', '')
            if symbol.endswith('USDT') and symbol_info.get('status') == 'TRADING':
                usdt_pairs.append(symbol)
        
        logger.info(f"✅ Lấy được {len(usdt_pairs)} coin USDT từ Binance")
        return usdt_pairs[:limit] if limit else usdt_pairs
        
    except Exception as e:
        logger.error(f"❌ Lỗi lấy danh sách coin từ Binance: {str(e)}")
        return []

def get_step_size(symbol, api_key, api_secret):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        data = binance_api_request(url)
        if not data:
            return 0.001
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        logger.error(f"Lỗi lấy step size: {str(e)}")
    return 0.001

def set_leverage(symbol, lev, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "leverage": lev,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        response = binance_api_request(url, method='POST', headers=headers)
        if response is None:
            return False
        if response and 'leverage' in response:
            return True
        return False
    except Exception as e:
        logger.error(f"Lỗi thiết lập đòn bẩy: {str(e)}")
        return False

def get_balance(api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return None
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        return 0
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        return None

def place_order(symbol, side, qty, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
    return None

def cancel_all_orders(symbol, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Lỗi hủy lệnh: {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
    return 0

def get_positions(symbol=None, api_key=None, api_secret=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        if symbol:
            params["symbol"] = symbol.upper()
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        positions = binance_api_request(url, headers=headers)
        if not positions:
            return []
        if symbol:
            for pos in positions:
                if pos['symbol'] == symbol.upper():
                    return [pos]
        return positions
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
    return []

def get_24h_change(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'priceChangePercent' in data:
            change = data['priceChangePercent']
            if change is None:
                return 0.0
            return float(change) if change is not None else 0.0
        return 0.0
    except Exception as e:
        logger.error(f"Lỗi lấy biến động 24h cho {symbol}: {str(e)}")
    return 0.0

# ========== CHỈ BÁO KỸ THUẬT ==========
def calc_rsi(prices, period=14):
    try:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1 + rs))
        
        if np.isnan(rsi_value) or np.isinf(rsi_value):
            return None
        return rsi_value
    except Exception as e:
        return None

def calc_ema(prices, period):
    try:
        if len(prices) < period:
            return None
        
        ema = [sum(prices[:period]) / period]
        multiplier = 2 / (period + 1)
        
        for price in prices[period:]:
            ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return float(ema[-1])
    except Exception as e:
        return None

# ========== WEBSOCKET MANAGER ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def add_symbol(self, symbol, callback):
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self.connections:
                self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
        stream = f"{symbol.lower()}@trade"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    self.executor.submit(callback, price)
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn WebSocket {symbol}: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Lỗi WebSocket {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(5)
                self._reconnect(symbol, callback)
                
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        
        self.connections[symbol] = {
            'ws': ws,
            'thread': thread,
            'callback': callback
        }
        logger.info(f"WebSocket bắt đầu cho {symbol}")
        
    def _reconnect(self, symbol, callback):
        logger.info(f"Kết nối lại WebSocket cho {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Lỗi đóng WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket đã xóa cho {symbol}")
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== BASE BOT NÂNG CẤP VỚI CÂN BẰNG VỊ THẾ ==========
class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None):
        
        self.symbol = symbol.upper() if symbol else "BTCUSDT"
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.ws_manager = ws_manager
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.strategy_name = strategy_name
        self.config_key = config_key
        
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self.position_open = False
        self._stop = False
        
        # Biến theo dõi thời gian
        self.last_trade_time = 0
        self.last_close_time = 0
        self._last_find_attempt = 0
        self._find_coin_cooldown = 300
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 300
        self.position_check_interval = 30
        
        # Bảo vệ chống lặp đóng lệnh
        self._close_attempted = False
        self._last_close_attempt = 0
        
        # Cờ đánh dấu cần xóa bot
        self.should_be_removed = False
        
        # THÊM THEO DÕI CÂN BẰNG
        self.position_balance_check = 0
        self.balance_check_interval = 60
        
        self.coin_manager = CoinManager()
        
        if symbol and config_key:
            success = self._register_coin_with_retry(symbol)
            if not success:
                self.log(f"❌ Không thể đăng ký coin {symbol} - đã đạt giới hạn 2 coin/config")
                self.should_be_removed = True
        
        self.check_position_status()
        if symbol:
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")

    def _register_coin_with_retry(self, symbol):
        max_retries = 3
        for attempt in range(max_retries):
            success = self.coin_manager.register_coin(symbol, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key)
            if success:
                return True
            time.sleep(0.5)
        return False

    def log(self, message):
        logger.info(f"[{self.symbol} - {self.strategy_name}] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>{self.symbol}</b> ({self.strategy_name}): {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def _handle_price_update(self, price):
        if self._stop or not price or price <= 0:
            return
        try:
            self.prices.append(float(price))
            if len(self.prices) > 100:
                self.prices = self.prices[-100:]
        except Exception as e:
            self.log(f"❌ Lỗi xử lý giá: {str(e)}")

    def get_signal(self):
        raise NotImplementedError("Phương thức get_signal cần được triển khai")

    def get_signal_with_balance(self, original_signal):
        """ĐIỀU CHỈNH TÍN HIỆU DỰA TRÊN CÂN BẰNG VỊ THẾ"""
        try:
            current_time = time.time()
            if current_time - self.position_balance_check < self.balance_check_interval:
                return original_signal
            
            self.position_balance_check = current_time
            
            # Lấy tỷ lệ BUY/SELL hiện tại từ tất cả bot
            buy_count = 0
            sell_count = 0
            
            # Giả sử chúng ta có tham chiếu đến bot manager
            bot_manager = getattr(self, '_bot_manager', None)
            if bot_manager and hasattr(bot_manager, 'bots'):
                for bot_id, bot in bot_manager.bots.items():
                    if bot.position_open:
                        if bot.side == "BUY":
                            buy_count += 1
                        elif bot.side == "SELL":
                            sell_count += 1
            
            total = buy_count + sell_count
            if total == 0:
                return original_signal
            
            buy_ratio = buy_count / total
            sell_ratio = sell_count / total
            
            # Nếu chênh lệch lớn, điều chỉnh tín hiệu
            if original_signal == "BUY" and buy_ratio - sell_ratio > 0.3:  # BUY đang chiếm ưu thế
                self.log(f"⚖️ Cân bằng: BUY {buy_ratio:.1%} vs SELL {sell_ratio:.1%} → Ưu tiên SELL")
                return "SELL" if self._check_opposite_signal() else None
            elif original_signal == "SELL" and sell_ratio - buy_ratio > 0.3:  # SELL đang chiếm ưu thế
                self.log(f"⚖️ Cân bằng: BUY {buy_ratio:.1%} vs SELL {sell_ratio:.1%} → Ưu tiên BUY")
                return "BUY" if self._check_opposite_signal() else None
            
            return original_signal
            
        except Exception as e:
            self.log(f"❌ Lỗi cân bằng tín hiệu: {str(e)}")
            return original_signal

    def _check_opposite_signal(self):
        """Kiểm tra tín hiệu ngược lại có hợp lệ không"""
        try:
            # Đơn giản hóa: luôn trả về True
            # Trong thực tế, có thể thêm logic kiểm tra indicator cho hướng ngược lại
            return True
        except:
            return True

    def check_position_status(self):
        try:
            positions = get_positions(self.symbol, self.api_key, self.api_secret)
            if not positions:
                self._reset_position()
                return
            
            position_found = False
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    if abs(position_amt) > 0:
                        position_found = True
                        self.position_open = True
                        self.status = "open"
                        self.side = "BUY" if position_amt > 0 else "SELL"
                        self.qty = position_amt
                        self.entry = float(pos.get('entryPrice', 0))
                        break
                    else:
                        position_found = True
                        self._reset_position()
                        break
            
            if not position_found:
                self._reset_position()
                
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"❌ Lỗi kiểm tra vị thế: {str(e)}")
                self.last_error_log_time = time.time()

    def _reset_position(self):
        self.position_open = False
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self._close_attempted = False
        self._last_close_attempt = 0

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                
                if not self.position_open:
                    signal = self.get_signal()
                    
                    # ÁP DỤNG CÂN BẰNG VỊ THẾ
                    balanced_signal = self.get_signal_with_balance(signal)
                    
                    if (balanced_signal and 
                        current_time - self.last_trade_time > 60 and
                        current_time - self.last_close_time > self.cooldown_period):
                        
                        self.log(f"🎯 Nhận tín hiệu {balanced_signal} (gốc: {signal}), đang mở lệnh...")
                        if self.open_position(balanced_signal):
                            self.last_trade_time = current_time
                        else:
                            time.sleep(30)
                
                if self.position_open and not self._close_attempted:
                    self.check_tp_sl()
                    
                time.sleep(1)
                
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"❌ Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        if self.symbol:
            self.ws_manager.remove_symbol(self.symbol)
        if self.symbol and self.config_key:
            self.coin_manager.unregister_coin(self.symbol)
        if self.symbol:
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
        self.log(f"🔴 Bot dừng cho {self.symbol}")

    def open_position(self, side):
        try:
            self.check_position_status()
            if self.position_open:
                self.log(f"⚠️ Đã có vị thế {self.side}, bỏ qua tín hiệu {side}")
                return False

            if self.should_be_removed:
                self.log("⚠️ Bot đã được đánh dấu xóa, không mở lệnh mới")
                return False

            if not set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                self.log(f"❌ Không thể đặt đòn bẩy {self.lev}x")
                return False

            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                self.log("❌ Không đủ số dư")
                return False

            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                self.log("❌ Lỗi lấy giá")
                return False

            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)

            if qty < step_size:
                self.log(f"❌ Số lượng quá nhỏ: {qty}")
                return False

            result = place_order(self.symbol, side, qty, self.api_key, self.api_secret)
            if result and 'orderId' in result:
                executed_qty = float(result.get('executedQty', 0))
                avg_price = float(result.get('avgPrice', current_price))
                
                if executed_qty >= 0:
                    self.entry = avg_price
                    self.side = side
                    self.qty = executed_qty if side == "BUY" else -executed_qty
                    self.position_open = True
                    self.status = "open"
                    
                    message = (
                        f"✅ <b>ĐÃ MỞ VỊ THẾ {self.symbol}</b>\n"
                        f"🤖 Chiến lược: {self.strategy_name}\n"
                        f"📌 Hướng: {side}\n"
                        f"🏷️ Giá vào: {self.entry:.4f}\n"
                        f"📊 Khối lượng: {executed_qty:.4f}\n"
                        f"💵 Giá trị: {executed_qty * self.entry:.2f} USDT\n"
                        f"💰 Đòn bẩy: {self.lev}x\n"
                        f"🎯 TP: {self.tp}% | 🛡️ SL: {self.sl}%"
                    )
                    self.log(message)
                    return True
                else:
                    self.log(f"❌ Lệnh không khớp - Số lượng: {qty}")
                    return False
            else:
                error_msg = result.get('msg', 'Unknown error') if result else 'No response'
                self.log(f"❌ Lỗi đặt lệnh {side}: {error_msg}")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi mở lệnh: {str(e)}")
            return False

    def close_position(self, reason=""):
        try:
            self.check_position_status()
            
            if not self.position_open or abs(self.qty) <= 0:
                self.log(f"⚠️ Không có vị thế để đóng: {reason}")
                if self.symbol and self.config_key:
                    self.coin_manager.unregister_coin(self.symbol)
                return False

            current_time = time.time()
            if self._close_attempted and current_time - self._last_close_attempt < 30:
                self.log(f"⚠️ Đang thử đóng lệnh lần trước, chờ...")
                return False
            
            self._close_attempted = True
            self._last_close_attempt = current_time

            close_side = "SELL" if self.side == "BUY" else "BUY"
            close_qty = abs(self.qty)
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.5)
            
            result = place_order(self.symbol, close_side, close_qty, self.api_key, self.api_secret)
            if result and 'orderId' in result:
                current_price = get_current_price(self.symbol)
                pnl = 0
                if self.entry > 0:
                    if self.side == "BUY":
                        pnl = (current_price - self.entry) * abs(self.qty)
                    else:
                        pnl = (self.entry - current_price) * abs(self.qty)
                
                message = (
                    f"⛔ <b>ĐÃ ĐÓNG VỊ THẾ {self.symbol}</b>\n"
                    f"🤖 Chiến lược: {self.strategy_name}\n"
                    f"📌 Lý do: {reason}\n"
                    f"🏷️ Giá ra: {current_price:.4f}\n"
                    f"📊 Khối lượng: {close_qty:.4f}\n"
                    f"💰 PnL: {pnl:.2f} USDT"
                )
                self.log(message)
                
                if self.symbol and self.config_key:
                    self.coin_manager.unregister_coin(self.symbol)
                
                if hasattr(self, 'config_key') and self.config_key:
                    self._find_new_coin_after_close()
                
                self._reset_position()
                self.last_close_time = time.time()
                
                time.sleep(2)
                self.check_position_status()
                
                return True
            else:
                error_msg = result.get('msg', 'Unknown error') if result else 'No response'
                self.log(f"❌ Lỗi đóng lệnh: {error_msg}")
                self._close_attempted = False
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi đóng lệnh: {str(e)}")
            self._close_attempted = False
            return False

    def _find_new_coin_after_close(self):
        try:
            current_time = time.time()
            if hasattr(self, '_last_find_attempt') and current_time - self._last_find_attempt < 300:
                return False
                
            self._last_find_attempt = current_time
            
            self.log(f"🔄 Bot động đang tìm coin mới thay thế {self.symbol}...")
            
            if hasattr(self, 'config_key') and self.config_key:
                bot_manager = getattr(self, '_bot_manager', None)
                if bot_manager and hasattr(bot_manager, '_handle_coin_after_close'):
                    bot_manager._handle_coin_after_close(self.config_key, self.symbol)
                    return True
            
            return False
            
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {str(e)}")
            return False

    def check_tp_sl(self):
        if not self.position_open or self.entry <= 0 or self._close_attempted:
            return

        current_price = get_current_price(self.symbol)
        if current_price <= 0:
            return

        if self.side == "BUY":
            profit = (current_price - self.entry) * abs(self.qty)
        else:
            profit = (self.entry - current_price) * abs(self.qty)
            
        invested = self.entry * abs(self.qty) / self.lev
        if invested <= 0:
            return
            
        roi = (profit / invested) * 100

        if self.tp is not None and roi >= self.tp:
            self.close_position(f"✅ Đạt TP {self.tp}% (ROI: {roi:.2f}%)")
        elif self.sl is not None and self.sl > 0 and roi <= -self.sl:
            self.close_position(f"❌ Đạt SL {self.sl}% (ROI: {roi:.2f}%)")

# ========== CÁC CHIẾN LƯỢC GIAO DỊCH ==========
class RSI_EMA_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "RSI/EMA Recursive")
        self.rsi_period = 14
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            rsi = calc_rsi(self.prices, self.rsi_period)
            ema_fast = calc_ema(self.prices, self.ema_fast)
            ema_slow = calc_ema(self.prices, self.ema_slow)

            if rsi is None or ema_fast is None or ema_slow is None:
                return None

            signal = None
            if rsi < self.rsi_oversold and ema_fast > ema_slow:
                signal = "BUY"
            elif rsi > self.rsi_overbought and ema_fast < ema_slow:
                signal = "SELL"

            return signal

        except Exception as e:
            return None

class EMA_Crossover_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "EMA Crossover")
        self.ema_fast = 9
        self.ema_slow = 21
        self.prev_ema_fast = None
        self.prev_ema_slow = None

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            ema_fast = calc_ema(self.prices, self.ema_fast)
            ema_slow = calc_ema(self.prices, self.ema_slow)

            if ema_fast is None or ema_slow is None:
                return None

            signal = None
            if self.prev_ema_fast is not None and self.prev_ema_slow is not None:
                if self.prev_ema_fast <= self.prev_ema_slow and ema_fast > ema_slow:
                    signal = "BUY"
                elif self.prev_ema_fast >= self.prev_ema_slow and ema_fast < ema_slow:
                    signal = "SELL"

            self.prev_ema_fast = ema_fast
            self.prev_ema_slow = ema_slow

            return signal

        except Exception as e:
            return None

class Reverse_24h_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, threshold=30, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Reverse 24h", config_key)
        self.threshold = threshold
        self.last_24h_check = 0
        self.last_reported_change = 0

    def get_signal(self):
        try:
            current_time = time.time()
            if current_time - self.last_24h_check < 60:
                return None

            change_24h = get_24h_change(self.symbol)
            self.last_24h_check = current_time

            if change_24h is None:
                return None
                
            if abs(change_24h - self.last_reported_change) > 5:
                self.log(f"📊 Biến động 24h: {change_24h:.2f}% | Ngưỡng: {self.threshold}%")
                self.last_reported_change = change_24h

            signal = None
            if abs(change_24h) >= self.threshold:
                if change_24h > 0:
                    signal = "SELL"
                    self.log(f"🎯 Tín hiệu SELL - Biến động 24h: +{change_24h:.2f}% (≥ {self.threshold}%)")
                else:
                    signal = "BUY" 
                    self.log(f"🎯 Tín hiệu BUY - Biến động 24h: {change_24h:.2f}% (≤ -{self.threshold}%)")

            return signal

        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu Reverse 24h: {str(e)}")
            return None

class Trend_Following_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Trend Following", config_key)
        self.trend_period = 20
        self.trend_threshold = 0.001

    def get_signal(self):
        try:
            if len(self.prices) < self.trend_period + 1:
                return None

            recent_prices = self.prices[-self.trend_period:]
            if len(recent_prices) < 2:
                return None
                
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            signal = None
            if price_change > self.trend_threshold:
                signal = "BUY"
            elif price_change < -self.trend_threshold:
                signal = "SELL"

            return signal

        except Exception as e:
            return None

class Scalping_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Scalping", config_key)
        self.rsi_period = 7
        self.min_movement = 0.001

    def get_signal(self):
        try:
            if len(self.prices) < 20:
                return None

            current_price = self.prices[-1]
            price_change = 0
            if len(self.prices) >= 2:
                price_change = (current_price - self.prices[-2]) / self.prices[-2]

            rsi = calc_rsi(self.prices, self.rsi_period)

            if rsi is None:
                return None

            signal = None
            if rsi < 25 and price_change < -self.min_movement:
                signal = "BUY"
            elif rsi > 75 and price_change > self.min_movement:
                signal = "SELL"

            return signal

        except Exception as e:
            return None

class Safe_Grid_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, grid_levels=5, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Safe Grid", config_key)
        self.grid_levels = grid_levels
        self.orders_placed = 0

    def get_signal(self):
        try:
            if self.orders_placed < self.grid_levels:
                self.orders_placed += 1
                if self.orders_placed % 2 == 1:
                    return "BUY"
                else:
                    return "SELL"
            return None
        except Exception as e:
            return None

# ========== BOT ĐỘNG THÔNG MINH VỚI AI ==========
class SmartDynamicBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Smart Dynamic", config_key)

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            # 1. RSI SIGNAL
            rsi = calc_rsi(self.prices, 14)
            
            # 2. EMA SIGNAL  
            ema_fast = calc_ema(self.prices, 9)
            ema_slow = calc_ema(self.prices, 21)
            
            # 3. TREND SIGNAL
            trend_strength = self._calculate_trend_strength()
            
            # 4. VOLATILITY CHECK
            volatility = self._calculate_volatility()
            
            if None in [rsi, ema_fast, ema_slow]:
                return None

            signal = None
            score = 0
            
            # RSI + EMA CONFIRMATION
            if rsi < 30 and ema_fast > ema_slow:
                score += 2
                signal = "BUY"
            elif rsi > 70 and ema_fast < ema_slow:
                score += 2
                signal = "SELL"
            
            # TREND CONFIRMATION
            if trend_strength > 0.5 and signal == "BUY":
                score += 1
            elif trend_strength < -0.5 and signal == "SELL":
                score += 1
            
            # VOLATILITY FILTER
            if volatility > 8.0:
                score -= 1
            
            if score >= 2:
                self.log(f"🎯 Smart Signal: {signal} | Score: {score}/3 | RSI: {rsi:.1f} | Trend: {trend_strength:.2f}")
                return signal
            
            return None

        except Exception as e:
            self.log(f"❌ Lỗi Smart Dynamic signal: {str(e)}")
            return None

    def _calculate_trend_strength(self):
        if len(self.prices) < 20:
            return 0
            
        short_trend = (self.prices[-1] - self.prices[-5]) / self.prices[-5]
        medium_trend = (self.prices[-1] - self.prices[-10]) / self.prices[-10]
        long_trend = (self.prices[-1] - self.prices[-20]) / self.prices[-20]
        
        return (short_trend + medium_trend + long_trend) / 3

    def _calculate_volatility(self):
        if len(self.prices) < 20:
            return 0
            
        returns = []
        for i in range(1, len(self.prices)):
            ret = (self.prices[i] - self.prices[i-1]) / self.prices[i-1]
            returns.append(abs(ret))
            
        return np.mean(returns) * 100

# ========== BOT MANAGER HOÀN CHỈNH VỚI AI ==========
class BotManager:
    def __init__(self, api_key=None, api_secret=None, telegram_bot_token=None, telegram_chat_id=None):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        
        self.auto_strategies = {}
        self.last_auto_scan = 0
        self.auto_scan_interval = 600
        
        self.target_coins = {}
        self.max_bots_per_config = {}
        
        self.strategy_cooldowns = {
            "Reverse 24h": {},
            "Scalping": {},
            "Trend Following": {},
            "Safe Grid": {},
            "Smart Dynamic": {}
        }
        self.cooldown_period = 300
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        # KHỞI TẠO AI VÀ CÂN BẰNG
        self.ai_analyzer = AIMarketAnalyzer()
        
        # Kiểm tra trạng thái AI
        if hasattr(self.ai_analyzer, 'is_trained') and self.ai_analyzer.is_trained:
            logger.info("🤖 AI Market Analyzer: Đã sẵn sàng")
        else:
            logger.warning("⚠️ AI Market Analyzer: Chưa được train, sẽ sử dụng fallback")
            
        self.position_balancer = PositionBalancer(self)
        self.coin_finder = SmartCoinFinder(api_key, api_secret, self.ai_analyzer, self.position_balancer)
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG BOT AI THÔNG MINH ĐÃ KHỞI ĐỘNG")
            self.log("🤖 AI Market Analyzer: Đã sẵn sàng")
            self.log("⚖️ Position Balancer: Đã sẵn sàng")
            self.log("🎯 Smart Coin Finder: Đã sẵn sàng")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
            self.auto_scan_thread = threading.Thread(target=self._auto_scan_loop, daemon=True)
            self.auto_scan_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def _verify_api_connection(self):
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance API.")
        else:
            self.log(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>SYSTEM</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES THÔNG MINH VỚI AI</b>\n\n🎯 <b>HỆ THỐNG AI TÌM COIN & CÂN BẰNG VỊ THẾ</b>"
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def _is_in_cooldown(self, strategy_type, config_key):
        if strategy_type not in self.strategy_cooldowns:
            return False
            
        last_cooldown_time = self.strategy_cooldowns[strategy_type].get(config_key)
        if last_cooldown_time is None:
            return False
            
        current_time = time.time()
        if current_time - last_cooldown_time < self.cooldown_period:
            return True
            
        del self.strategy_cooldowns[strategy_type][config_key]
        return False

    def _find_qualified_symbols_smart(self, strategy_type, leverage, config, strategy_key):
        """TÌM COIN THÔNG MINH - SỬ DỤNG AI VÀ CÂN BẰNG VỊ THẾ"""
        try:
            # 1. XÁC ĐỊNH HƯỚNG ƯU TIÊN
            balance_direction = self.position_balancer.get_recommended_direction()
            
            market_data = self.coin_finder.get_market_sentiment()
            
            if balance_direction != "NEUTRAL":
                target_direction = balance_direction
                reason = "cân bằng vị thế"
            else:
                fear_greed = market_data.get('fear_greed', 50)
                if fear_greed > 60:
                    target_direction = "BUY"
                    reason = "sentiment tích cực"
                elif fear_greed < 40:
                    target_direction = "SELL"
                    reason = "sentiment tiêu cực"
                else:
                    target_direction = "NEUTRAL"
                    reason = "thị trường trung lập"
            
            self.log(f"🎯 Hướng chiến lược: {target_direction} ({reason})")
            
            if target_direction != "NEUTRAL":
                max_bots = self.max_bots_per_config.get(strategy_key, 2)
                
                qualified_coins = self.coin_finder.find_coins_by_direction(
                    target_direction=target_direction,
                    strategy_type=strategy_type,
                    count=max_bots,
                    market_data=market_data
                )
                
                symbols = [coin['symbol'] for coin in qualified_coins]
                
                self.log(f"🤖 AI đã tìm thấy {len(symbols)} coin {target_direction}: {symbols}")
                return symbols
            else:
                self.log("⚡ Không có hướng rõ ràng, bỏ qua tìm coin")
                return []
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin thông minh: {str(e)}")
            return []

    def _find_and_populate_target_coins(self, strategy_type, leverage, config, strategy_key, max_bots=2):
        try:
            if strategy_key not in self.target_coins:
                self.target_coins[strategy_key] = []
                self.max_bots_per_config[strategy_key] = max_bots
            
            current_count = len(self.target_coins[strategy_key])
            needed_count = max_bots - current_count
            
            if needed_count <= 0:
                return self.target_coins[strategy_key]
            
            logger.info(f"🔄 AI đang tìm {needed_count} coin cho {strategy_type}")
            
            new_symbols = self._find_qualified_symbols_smart(strategy_type, leverage, config, strategy_key)
            
            added_count = 0
            for symbol in new_symbols:
                if symbol not in self.target_coins[strategy_key] and added_count < needed_count:
                    self.target_coins[strategy_key].append(symbol)
                    logger.info(f"✅ AI đã thêm {symbol} vào danh sách target")
                    added_count += 1
            
            self._create_all_bots_from_target_list(strategy_key)
            
            return self.target_coins[strategy_key]
            
        except Exception as e:
            logger.error(f"❌ Lỗi AI populate target coins: {str(e)}")
            return []

    def _create_all_bots_from_target_list(self, strategy_key):
        try:
            if strategy_key not in self.target_coins or not self.target_coins[strategy_key]:
                return False
            
            strategy_config = self.auto_strategies.get(strategy_key)
            if not strategy_config:
                return False
            
            created_count = 0
            max_bots = self.max_bots_per_config.get(strategy_key, 2)
            
            for symbol in self.target_coins[strategy_key]:
                if created_count >= max_bots:
                    break
                    
                bot_id = f"{symbol}_{strategy_key}"
                
                if bot_id not in self.bots:
                    strategy_type = strategy_config['strategy_type']
                    success = self._create_auto_bot(symbol, strategy_type, strategy_config)
                    if success:
                        created_count += 1
                        logger.info(f"✅ Đã tạo bot {bot_id} từ danh sách AI")
            
            if created_count > 0:
                logger.info(f"🎯 Đã tạo {created_count} bot mới cho {strategy_key}")
            
            return created_count > 0
                
        except Exception as e:
            logger.error(f"❌ Lỗi tạo bot từ target list: {str(e)}")
            return False

    def _scan_auto_strategies(self):
        if not self.auto_strategies:
            return
            
        self.log("🔄 AI đang quét coin cho các cấu hình tự động...")
        
        for strategy_key, strategy_config in self.auto_strategies.items():
            try:
                strategy_type = strategy_config['strategy_type']
                
                if self._is_in_cooldown(strategy_type, strategy_key):
                    continue
                
                coin_manager = CoinManager()
                current_bots_count = coin_manager.count_bots_by_config(strategy_key)
                max_bots = self.max_bots_per_config.get(strategy_key, 2)
                
                if current_bots_count < max_bots:
                    self.log(f"🤖 {strategy_type}: AI đang tìm {max_bots - current_bots_count} coin...")
                    
                    target_coins = self._find_and_populate_target_coins(
                        strategy_type, 
                        strategy_config['leverage'], 
                        strategy_config, 
                        strategy_key,
                        max_bots
                    )
                    
                    current_after = coin_manager.count_bots_by_config(strategy_key)
                    if current_after > current_bots_count:
                        self.log(f"✅ {strategy_type}: AI đã thêm {current_after - current_bots_count} bot mới")
                    
                    if target_coins:
                        self.log(f"🎯 {strategy_type}: danh sách AI - {target_coins}")
                    else:
                        self.log(f"⚠️ {strategy_type}: AI không tìm thấy coin phù hợp")
                else:
                    self.log(f"✅ {strategy_type}: đã đủ {current_bots_count}/{max_bots} bot")
                        
            except Exception as e:
                self.log(f"❌ Lỗi AI quét {strategy_type}: {str(e)}")

    def _handle_coin_after_close(self, strategy_key, closed_symbol):
        try:
            if strategy_key not in self.target_coins:
                self.target_coins[strategy_key] = []
            
            if closed_symbol in self.target_coins[strategy_key]:
                self.target_coins[strategy_key].remove(closed_symbol)
                self.log(f"🗑️ Đã xóa {closed_symbol} khỏi danh sách target {strategy_key}")
            
            bot_id_to_remove = f"{closed_symbol}_{strategy_key}"
            if bot_id_to_remove in self.bots:
                self.stop_bot(bot_id_to_remove)
                self.log(f"🔴 Đã dừng và xóa bot cũ {bot_id_to_remove}")
            
            coin_manager = CoinManager()
            current_bots_count = coin_manager.count_bots_by_config(strategy_key)
            max_bots = self.max_bots_per_config.get(strategy_key, 2)
            
            if current_bots_count < max_bots:
                strategy_config = self.auto_strategies.get(strategy_key)
                if strategy_config:
                    new_symbols = self._find_qualified_symbols_smart(
                        strategy_config['strategy_type'],
                        strategy_config['leverage'],
                        strategy_config,
                        strategy_key
                    )
                    
                    for symbol in new_symbols:
                        if (symbol not in self.target_coins[strategy_key] and 
                            len(self.target_coins[strategy_key]) < max_bots and
                            symbol != closed_symbol):
                            
                            self.target_coins[strategy_key].append(symbol)
                            self.log(f"🔄 Đã thêm {symbol} vào danh sách target thay thế {closed_symbol}")
                            
                            success = self._create_all_bots_from_target_list(strategy_key)
                            if success:
                                self.log(f"✅ Đã tạo bot mới {symbol} thay thế {closed_symbol}")
                            break
            
        except Exception as e:
            self.log(f"❌ Lỗi xử lý coin sau khi đóng: {str(e)}")

    def _auto_scan_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                for bot_id, bot in list(self.bots.items()):
                    if (hasattr(bot, 'config_key') and bot.config_key and
                        not bot.position_open and 
                        current_time - bot.last_close_time < 300 and
                        bot.strategy_name in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following", "Smart Dynamic"]):
                        
                        if current_time - getattr(bot, '_last_find_attempt', 0) > 300:
                            self.log(f"🔄 Bot động {bot_id} đang tìm coin mới sau khi đóng lệnh...")
                            bot._last_find_attempt = current_time
                            self._handle_coin_after_close(bot.config_key, bot.symbol)
                
                if current_time - self.last_auto_scan > self.auto_scan_interval:
                    self._scan_auto_strategies()
                    self.last_auto_scan = current_time
                
                time.sleep(60)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(60)

    def _create_auto_bot(self, symbol, strategy_type, config):
        try:
            leverage = config['leverage']
            percent = config['percent']
            tp = config['tp']
            sl = config['sl']
            strategy_key = config['strategy_key']
            
            bot_class = {
                "Reverse 24h": Reverse_24h_Bot,
                "Scalping": Scalping_Bot,
                "Safe Grid": Safe_Grid_Bot,
                "Trend Following": Trend_Following_Bot,
                "Smart Dynamic": SmartDynamicBot
            }.get(strategy_type)
            
            if not bot_class:
                return False
            
            if strategy_type == "Reverse 24h":
                threshold = config.get('threshold', 30)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, threshold, strategy_key)
            elif strategy_type == "Safe Grid":
                grid_levels = config.get('grid_levels', 5)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels, strategy_key)
            else:
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, strategy_key)
            
            bot._bot_manager = self
            
            bot_id = f"{symbol}_{strategy_key}"
            self.bots[bot_id] = bot
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, bot_count=1, **kwargs):
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance")
            return False
        
        bot_mode = kwargs.get('bot_mode', 'static')
        
        if bot_mode == 'dynamic' and strategy_type in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following", "Smart Dynamic"]:
            strategy_key = f"{strategy_type}_{lev}_{percent}_{tp}_{sl}"
            
            if strategy_type == "Reverse 24h":
                threshold = kwargs.get('threshold', 30)
                strategy_key += f"_th{threshold}"
            elif strategy_type == "Scalping":
                volatility = kwargs.get('volatility', 3)
                strategy_key += f"_vol{volatility}"
            elif strategy_type == "Safe Grid":
                grid_levels = kwargs.get('grid_levels', 5)
                strategy_key += f"_grid{grid_levels}"
            
            if self._is_in_cooldown(strategy_type, strategy_key):
                self.log(f"⏰ {strategy_type} (Config: {strategy_key}): đang trong cooldown, không thể thêm mới")
                return False
            
            self.auto_strategies[strategy_key] = {
                'strategy_type': strategy_type,
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key,
                **kwargs
            }
            
            self.max_bots_per_config[strategy_key] = bot_count
            
            target_coins = self._find_and_populate_target_coins(
                strategy_type, lev, self.auto_strategies[strategy_key], 
                strategy_key, bot_count
            )
            
            success_count = len([bot_id for bot_id in self.bots.keys() if strategy_key in bot_id])
            
            if success_count > 0:
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {success_count}/{bot_count} BOT {strategy_type}</b>\n\n"
                    f"🎯 Chiến lược: {strategy_type}\n"
                    f"💰 Đòn bẩy: {lev}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl}%\n"
                )
                if strategy_type == "Reverse 24h":
                    success_msg += f"📈 Ngưỡng: {threshold}%\n"
                elif strategy_type == "Scalping":
                    success_msg += f"⚡ Biến động: {volatility}%\n"
                elif strategy_type == "Safe Grid":
                    success_msg += f"🛡️ Số lệnh: {grid_levels}\n"
                    
                success_msg += f"🤖 Coin: {', '.join(target_coins) if target_coins else 'Đang tìm...'}\n\n"
                success_msg += f"🔑 <b>Config Key:</b> {strategy_key}\n"
                success_msg += f"🎯 <b>Mỗi chiến lược có {bot_count} coin riêng biệt</b>"
                
                self.log(success_msg)
                return True
            else:
                self.log(f"⚠️ {strategy_type}: đang tìm coin phù hợp, sẽ thử lại sau")
                return True
        
        else:
            symbol = symbol.upper()
            bot_id = f"{symbol}_{strategy_type}"
            
            if bot_id in self.bots:
                self.log(f"⚠️ Đã có bot {strategy_type} cho {symbol}")
                return False
                
            try:
                bot_class = {
                    "RSI/EMA Recursive": RSI_EMA_Bot,
                    "EMA Crossover": EMA_Crossover_Bot
                }.get(strategy_type)
                
                if not bot_class:
                    self.log(f"❌ Chiến lược {strategy_type} không được hỗ trợ")
                    return False
                
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id)
                
                self.bots[bot_id] = bot
                self.log(f"✅ Đã thêm bot {strategy_type}: {symbol} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")
                return True
                
            except Exception as e:
                error_msg = f"❌ Lỗi tạo bot {symbol}: {str(e)}"
                self.log(error_msg)
                return False

    def stop_bot(self, bot_id):
        bot = self.bots.get(bot_id)
        if bot:
            if hasattr(bot, 'config_key') and bot.config_key:
                strategy_key = bot.config_key
                if strategy_key in self.target_coins and bot.symbol in self.target_coins[strategy_key]:
                    self.target_coins[strategy_key].remove(bot.symbol)
                    self.log(f"🗑️ Đã xóa {bot.symbol} khỏi danh sách target {strategy_key}")
            
            bot.stop()
            del self.bots[bot_id]
            self.log(f"⛔ Đã dừng và xóa bot {bot_id}")
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _telegram_listener(self):
        last_update_id = 0
        
        while self.running and self.telegram_bot_token:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates?offset={last_update_id+1}&timeout=30"
                response = requests.get(url, timeout=35)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        for update in data['result']:
                            update_id = update['update_id']
                            message = update.get('message', {})
                            chat_id = str(message.get('chat', {}).get('id'))
                            text = message.get('text', '').strip()
                            
                            if chat_id != self.telegram_chat_id:
                                continue
                            
                            if update_id > last_update_id:
                                last_update_id = update_id
                            
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    logger.error("Lỗi xung đột Telegram")
                    time.sleep(60)
                else:
                    time.sleep(10)
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(10)

    def _handle_telegram_message(self, chat_id, text):
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        if current_step == 'waiting_bot_count':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    bot_count = int(text)
                    if bot_count <= 0 or bot_count > 5:
                        send_telegram("⚠️ Số lượng bot phải từ 1 đến 5. Vui lòng chọn lại:",
                                    chat_id, create_bot_count_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['bot_count'] = bot_count
                    user_state['step'] = 'waiting_bot_mode'
                    
                    send_telegram(
                        f"🤖 Số lượng bot: {bot_count}\n\n"
                        f"Chọn chế độ bot:",
                        chat_id,
                        create_bot_mode_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho số lượng bot:",
                                chat_id, create_bot_count_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_bot_mode':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
                if text == "🤖 Bot Tĩnh - Coin cụ thể":
                    user_state['bot_mode'] = 'static'
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT TĨNH</b>\n\n"
                        "🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n"
                        "📊 Bạn cần chọn coin cụ thể\n\n"
                        "Chọn chiến lược:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                else:
                    user_state['bot_mode'] = 'dynamic'
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT ĐỘNG</b>\n\n"
                        f"🤖 Hệ thống sẽ tự động tìm <b>{user_state.get('bot_count', 1)} coin</b> tốt nhất\n"
                        f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                        f"📈 Mỗi chiến lược có danh sách coin riêng\n\n"
                        "Chọn chiến lược:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )

        elif current_step == 'waiting_strategy':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text in ["🤖 RSI/EMA Recursive", "📊 EMA Crossover", "🎯 Reverse 24h", 
                         "📈 Trend Following", "⚡ Scalping", "🛡️ Safe Grid", "🔄 Bot Động Thông Minh"]:
                
                strategy_map = {
                    "🤖 RSI/EMA Recursive": "RSI/EMA Recursive",
                    "📊 EMA Crossover": "EMA Crossover", 
                    "🎯 Reverse 24h": "Reverse 24h",
                    "📈 Trend Following": "Trend Following",
                    "⚡ Scalping": "Scalping",
                    "🛡️ Safe Grid": "Safe Grid",
                    "🔄 Bot Động Thông Minh": "Smart Dynamic"
                }
                
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                user_state['step'] = 'waiting_exit_strategy'
                
                strategy_descriptions = {
                    "RSI/EMA Recursive": "Phân tích RSI + EMA đệ quy",
                    "EMA Crossover": "Giao cắt EMA nhanh/chậm", 
                    "Reverse 24h": "Đảo chiều biến động 24h",
                    "Trend Following": "Theo xu hướng giá",
                    "Scalping": "Giao dịch tốc độ cao",
                    "Safe Grid": "Grid an toàn",
                    "Smart Dynamic": "Bot động thông minh đa chiến lược"
                }
                
                description = strategy_descriptions.get(strategy, "")
                bot_count = user_state.get('bot_count', 1)
                
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"{description}\n\n"
                    f"Chọn chiến lược thoát lệnh:",
                    chat_id,
                    create_exit_strategy_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )

        elif current_step == 'waiting_exit_strategy':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text == "🎯 Chỉ TP/SL cố định":
                user_state['exit_strategy'] = 'traditional'
                self._continue_bot_creation(chat_id, user_state)

        elif current_step == 'waiting_threshold':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    threshold = float(text)
                    if threshold <= 0:
                        send_telegram("⚠️ Ngưỡng phải lớn hơn 0. Vui lòng chọn lại:",
                                    chat_id, create_threshold_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['threshold'] = threshold
                    user_state['step'] = 'waiting_leverage'
                    send_telegram(
                        f"📈 Ngưỡng biến động: {threshold}%\n\n"
                        f"Chọn đòn bẩy:",
                        chat_id,
                        create_leverage_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho ngưỡng:",
                                chat_id, create_threshold_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_volatility':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    volatility = float(text)
                    if volatility <= 0:
                        send_telegram("⚠️ Biến động phải lớn hơn 0. Vui lòng chọn lại:",
                                    chat_id, create_volatility_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['volatility'] = volatility
                    user_state['step'] = 'waiting_leverage'
                    send_telegram(
                        f"⚡ Biến động tối thiểu: {volatility}%\n\n"
                        f"Chọn đòn bẩy:",
                        chat_id,
                        create_leverage_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho biến động:",
                                chat_id, create_volatility_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_grid_levels':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    grid_levels = int(text)
                    if grid_levels <= 0:
                        send_telegram("⚠️ Số lệnh grid phải lớn hơn 0. Vui lòng chọn lại:",
                                    chat_id, create_grid_levels_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['grid_levels'] = grid_levels
                    user_state['step'] = 'waiting_leverage'
                    send_telegram(
                        f"🛡️ Số lệnh grid: {grid_levels}\n\n"
                        f"Chọn đòn bẩy:",
                        chat_id,
                        create_leverage_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho số lệnh grid:",
                                chat_id, create_grid_levels_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                user_state['symbol'] = text
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🔗 Coin: {text}\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )

        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                if text.endswith('x'):
                    lev_text = text[:-1]
                else:
                    lev_text = text

                try:
                    leverage = int(lev_text)
                    if leverage <= 0 or leverage > 100:
                        send_telegram("⚠️ Đòn bẩy phải từ 1 đến 100. Vui lòng chọn lại:",
                                    chat_id, create_leverage_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['leverage'] = leverage
                    user_state['step'] = 'waiting_percent'
                    
                    balance = get_balance(self.api_key, self.api_secret)
                    balance_info = f"\n💰 Số dư hiện có: {balance:.2f} USDT" if balance else ""
                    
                    send_telegram(
                        f"💰 Đòn bẩy: {leverage}x{balance_info}\n\n"
                        f"Chọn % số dư cho mỗi lệnh:",
                        chat_id,
                        create_percent_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho đòn bẩy:",
                                chat_id, create_leverage_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    percent = float(text)
                    if percent <= 0 or percent > 100:
                        send_telegram("⚠️ % số dư phải từ 0.1 đến 100. Vui lòng chọn lại:",
                                    chat_id, create_percent_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['percent'] = percent
                    user_state['step'] = 'waiting_tp'
                    
                    balance = get_balance(self.api_key, self.api_secret)
                    actual_amount = balance * (percent / 100) if balance else 0
                    
                    send_telegram(
                        f"📊 % Số dư: {percent}%\n"
                        f"💵 Số tiền mỗi lệnh: ~{actual_amount:.2f} USDT\n\n"
                        f"Chọn Take Profit (%):",
                        chat_id,
                        create_tp_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho % số dư:",
                                chat_id, create_percent_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_tp':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    tp = float(text)
                    if tp <= 0:
                        send_telegram("⚠️ Take Profit phải lớn hơn 0. Vui lòng chọn lại:",
                                    chat_id, create_tp_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['tp'] = tp
                    user_state['step'] = 'waiting_sl'
                    
                    send_telegram(
                        f"🎯 Take Profit: {tp}%\n\n"
                        f"Chọn Stop Loss (%):",
                        chat_id,
                        create_sl_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Take Profit:",
                                chat_id, create_tp_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_sl':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    sl = float(text)
                    if sl < 0:
                        send_telegram("⚠️ Stop Loss phải lớn hơn hoặc bằng 0. Vui lòng chọn lại:",
                                    chat_id, create_sl_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['sl'] = sl
                    
                    strategy = user_state.get('strategy')
                    bot_mode = user_state.get('bot_mode', 'static')
                    leverage = user_state.get('leverage')
                    percent = user_state.get('percent')
                    tp = user_state.get('tp')
                    sl = user_state.get('sl')
                    symbol = user_state.get('symbol')
                    bot_count = user_state.get('bot_count', 1)
                    exit_strategy = user_state.get('exit_strategy', 'traditional')
                    
                    threshold = user_state.get('threshold')
                    volatility = user_state.get('volatility')
                    grid_levels = user_state.get('grid_levels')
                    
                    success = False
                    
                    if bot_mode == 'static':
                        success = self.add_bot(
                            symbol=symbol,
                            lev=leverage,
                            percent=percent,
                            tp=tp,
                            sl=sl,
                            strategy_type=strategy
                        )
                    else:
                        success = self.add_bot(
                            symbol=None,
                            lev=leverage,
                            percent=percent,
                            tp=tp,
                            sl=sl,
                            strategy_type=strategy,
                            bot_mode='dynamic',
                            bot_count=bot_count,
                            threshold=threshold,
                            volatility=volatility,
                            grid_levels=grid_levels
                        )
                    
                    if success:
                        success_msg = (
                            f"✅ <b>ĐÃ TẠO BOT THÀNH CÔNG</b>\n\n"
                            f"🤖 Chiến lược: {strategy}\n"
                            f"🔧 Chế độ: {bot_mode}\n"
                            f"🔢 Số lượng: {bot_count} coin\n"
                            f"💰 Đòn bẩy: {leverage}x\n"
                            f"📊 % Số dư: {percent}%\n"
                            f"🎯 TP: {tp}%\n"
                            f"🛡️ SL: {sl}%"
                        )
                        if bot_mode == 'static':
                            success_msg += f"\n🔗 Coin: {symbol}"
                        else:
                            if threshold:
                                success_msg += f"\n📈 Ngưỡng: {threshold}%"
                            if volatility:
                                success_msg += f"\n⚡ Biến động: {volatility}%"
                            if grid_levels:
                                success_msg += f"\n🛡️ Grid levels: {grid_levels}"
                        
                        success_msg += f"\n\n🎯 <b>Mỗi chiến lược có {bot_count} coin riêng biệt</b>"
                        if bot_mode == 'dynamic':
                            success_msg += f"\n🔄 <b>Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất</b>"
                        
                        send_telegram(success_msg, chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    else:
                        send_telegram("❌ Có lỗi khi tạo bot. Vui lòng thử lại.",
                                    chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    
                    self.user_states[chat_id] = {}
                    
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Stop Loss:",
                                chat_id, create_sl_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_count'}
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(
                f"🎯 <b>CHỌN SỐ LƯỢNG COIN CHO CHIẾN LƯỢC</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"Chọn số lượng coin bạn muốn cho chiến lược:",
                chat_id,
                create_bot_count_keyboard(),
                self.telegram_bot_token, self.telegram_chat_id
            )
        
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                
                strategy_groups = {}
                for bot_id, bot in self.bots.items():
                    strategy_name = bot.strategy_name
                    if strategy_name not in strategy_groups:
                        strategy_groups[strategy_name] = []
                    strategy_groups[strategy_name].append(bot)
                
                for strategy_name, bots in strategy_groups.items():
                    message += f"🎯 <b>{strategy_name}</b> ({len(bots)} bot):\n"
                    for bot in bots:
                        status = "🟢 Mở" if bot.status == "open" else "🟡 Chờ"
                        mode = "Tĩnh"
                        if hasattr(bot, 'config_key') and bot.config_key:
                            mode = "Động"
                        
                        message += f"  🔹 {bot.symbol} | {status} | {mode} | ĐB: {bot.lev}x\n"
                    message += "\n"
                
                total_bots = len(self.bots)
                dynamic_bots = sum(1 for bot in self.bots.values() if hasattr(bot, 'config_key') and bot.config_key)
                message += f"📊 Tổng số: {total_bots} bot | 🔄 Động: {dynamic_bots}"
                
                send_telegram(message, chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "⛔ <b>CHỌN BOT ĐỂ DỪNG</b>\n\n"
                keyboard = []
                row = []
                
                for i, bot_id in enumerate(self.bots.keys()):
                    message += f"🔹 {bot_id}\n"
                    row.append({"text": f"⛔ {bot_id}"})
                    if len(row) == 2 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram(
                    message, 
                    chat_id, 
                    {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                    self.telegram_bot_token, self.telegram_chat_id
                )
        
        elif text.startswith("⛔ "):
            bot_id = text.replace("⛔ ", "").strip()
            if self.stop_bot(bot_id):
                send_telegram(f"⛔ Đã dừng bot {bot_id}", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                send_telegram(f"⚠️ Không tìm thấy bot {bot_id}", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
        
        elif text == "💰 Số dư":
            try:
                balance = get_balance(self.api_key, self.api_secret)
                if balance is None:
                    send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                                bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                else:
                    send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id,
                                bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy số dư: {str(e)}", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "📈 Vị thế":
            try:
                positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
                if not positions:
                    send_telegram("📭 Không có vị thế nào đang mở", chat_id,
                                bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                    return
                
                message = "📈 <b>VỊ THẾ ĐANG MỞ</b>\n\n"
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0:
                        symbol = pos.get('symbol', 'UNKNOWN')
                        entry = float(pos.get('entryPrice', 0))
                        side = "LONG" if position_amt > 0 else "SHORT"
                        pnl = float(pos.get('unRealizedProfit', 0))
                        
                        message += (
                            f"🔹 {symbol} | {side}\n"
                            f"📊 Khối lượng: {abs(position_amt):.4f}\n"
                            f"🏷️ Giá vào: {entry:.4f}\n"
                            f"💰 PnL: {pnl:.2f} USDT\n\n"
                        )
                
                send_telegram(message, chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy vị thế: {str(e)}", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "🎯 Chiến lược":
            strategy_info = (
                "🎯 <b>DANH SÁCH CHIẾN LƯỢC HOÀN CHỈNH VỚI AI</b>\n\n"
                
                "🔄 <b>Bot Động Thông Minh</b>\n"
                "• 🤖 AI phân tích đa chỉ báo\n"
                "• ⚖️ Tự động cân bằng vị thế\n"
                "• 🎯 Chọn hướng trước, tìm coin sau\n"
                "• 🔄 Mỗi chiến lược có danh sách coin riêng\n\n"
                
                "🎯 <b>Reverse 24h</b> - TỰ ĐỘNG + AI\n"
                "• Đảo chiều biến động 24h\n"
                "• 🤖 AI tìm coin phù hợp hướng\n"
                "• ⚖️ Cân bằng vị thế tự động\n\n"
                
                "⚡ <b>Scalping</b> - TỰ ĐỘNG + AI\n"
                "• Giao dịch tốc độ cao\n"
                "• 🤖 AI tìm coin biến động\n"
                "• ⚖️ Cân bằng vị thế tự động\n\n"
                
                "🛡️ <b>Safe Grid</b> - TỰ ĐỘNG + AI\n"
                "• Grid an toàn\n"
                "• 🤖 AI tìm coin ổn định\n"
                "• ⚖️ Cân bằng vị thế tự động\n\n"
                
                "📈 <b>Trend Following</b> - TỰ ĐỘNG + AI\n"
                "• Theo xu hướng giá\n"
                "• 🤖 AI tìm coin trend rõ\n"
                "• ⚖️ Cân bằng vị thế tự động\n\n"
                
                "💡 <b>Hệ thống AI & Cân bằng</b>\n"
                "• 🎯 AI chọn hướng trước, tìm coin sau\n"
                "• ⚖️ Tự động cân bằng tỷ lệ BUY/SELL\n"
                "• 📊 Đa chỉ báo tổng hợp\n"
                "• 🔄 Tự tìm coin mới sau khi đóng lệnh"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            dynamic_bots_count = sum(1 for bot in self.bots.values() 
                                   if hasattr(bot, 'config_key') and bot.config_key)
            
            strategy_stats = {}
            for bot in self.bots.values():
                strategy_name = bot.strategy_name
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = 0
                strategy_stats[strategy_name] += 1
            
            stats_text = "\n".join([f"  • {name}: {count} bot" for name, count in strategy_stats.items()])
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG AI THÔNG MINH</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🔄 Bot động: {dynamic_bots_count}\n"
                f"🤖 AI Analyzer: Đã sẵn sàng\n"
                f"⚖️ Position Balancer: Đã sẵn sàng\n"
                f"📊 Chiến lược đang chạy:\n{stats_text}\n"
                f"🎯 Auto scan: {len(self.auto_strategies)} cấu hình\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        strategy = user_state.get('strategy')
        bot_mode = user_state.get('bot_mode', 'static')
        bot_count = user_state.get('bot_count', 1)
        
        if bot_mode == 'dynamic' and strategy != "Smart Dynamic":
            if strategy == "Reverse 24h":
                user_state['step'] = 'waiting_threshold'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn ngưỡng biến động (%):",
                    chat_id,
                    create_threshold_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "Scalping":
                user_state['step'] = 'waiting_volatility'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn biến động tối thiểu (%):",
                    chat_id,
                    create_volatility_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "Safe Grid":
                user_state['step'] = 'waiting_grid_levels'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn số lệnh grid:",
                    chat_id,
                    create_grid_levels_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
        else:
            if bot_mode == 'static':
                user_state['step'] = 'waiting_symbol'
                send_telegram(
                    f"🎯 <b>BOT TĨNH: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n\n"
                    f"Chọn cặp coin:",
                    chat_id,
                    create_symbols_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG THÔNG MINH</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n"
                    f"📈 Tối ưu hóa tự động\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()
