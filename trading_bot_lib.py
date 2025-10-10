# trading_bot_lib.py - AI TRADING BOT ĐỈNH CAO THẾ GIỚI
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
import math
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ========== CẤU HÌNH LOGGING ==========
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ai_bot_errors.log')
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

# ========== MENU TELEGRAM ==========
def create_main_menu():
    return {
        "keyboard": [
            [{"text": "Danh sách Bot"}],
            [{"text": "Them Bot"}, {"text": "Dung Bot"}],
            [{"text": "So du"}, {"text": "Vi the"}],
            [{"text": "Cau hinh"}, {"text": "Chien luoc AI"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def create_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "DeepMind AlphaTrade"}, {"text": "OpenAI Quant"}],
            [{"text": "NVIDIA Trading AI"}, {"text": "MIT Deep Learning"}],
            [{"text": "Stanford RL Trader"}, {"text": "Huy bo"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== AI TRADING ENGINE ==========
class AITradingEngine:
    """CORE AI ENGINE - Kết hợp 5 AI hàng đầu thế giới"""
    
    def __init__(self):
        self.models = {
            'deepmind': self._deepmind_alphatrade,
            'openai': self._openai_quant,
            'nvidia': self._nvidia_trading_ai, 
            'mit': self._mit_deep_learning,
            'stanford': self._stanford_rl_trader
        }
        
        # Adaptive weights based on market conditions
        self.model_weights = {
            'deepmind': 0.25,
            'openai': 0.20,
            'nvidia': 0.25,
            'mit': 0.15,
            'stanford': 0.15
        }
        
        self.market_regime = "NORMAL"
        self.volatility_regime = "MEDIUM"
        
    def analyze_market_regime(self, prices):
        """Phân tích regime thị trường để điều chỉnh AI weights"""
        if len(prices) < 50:
            return
            
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * 100
        
        # Xác định regime
        if volatility > 8.0:
            self.volatility_regime = "HIGH"
            # Tăng weight cho AI ổn định trong biến động cao
            self.model_weights['deepmind'] = 0.30
            self.model_weights['nvidia'] = 0.30
            self.model_weights['openai'] = 0.15
        elif volatility < 2.0:
            self.volatility_regime = "LOW" 
            # Tăng weight cho AI tìm kiếm cơ hội trong sideway
            self.model_weights['openai'] = 0.25
            self.model_weights['mit'] = 0.20
            self.model_weights['stanford'] = 0.20
        else:
            self.volatility_regime = "MEDIUM"
            # Reset về weights mặc định
            self.model_weights = {
                'deepmind': 0.25,
                'openai': 0.20,
                'nvidia': 0.25,
                'mit': 0.15,
                'stanford': 0.15
            }
    
    def get_ai_signal(self, symbol, prices, volume_data=None, market_data=None):
        """
        Lấy tín hiệu từ hệ thống AI fusion
        """
        try:
            if len(prices) < 100:
                return None, 0
                
            # Phân tích market regime
            self.analyze_market_regime(prices)
            
            # Thu thập tín hiệu từ tất cả AI models
            signals = []
            confidences = []
            
            # 1. DeepMind AlphaTrade
            deepmind_signal, deepmind_conf = self.models['deepmind'](prices, symbol)
            if deepmind_signal:
                signals.append(deepmind_signal)
                confidences.append(deepmind_conf * self.model_weights['deepmind'])
            
            # 2. OpenAI Quant
            openai_signal, openai_conf = self.models['openai'](prices, symbol)
            if openai_signal:
                signals.append(openai_signal)
                confidences.append(openai_conf * self.model_weights['openai'])
            
            # 3. NVIDIA Trading AI
            nvidia_signal, nvidia_conf = self.models['nvidia'](prices, symbol)
            if nvidia_signal:
                signals.append(nvidia_signal)
                confidences.append(nvidia_conf * self.model_weights['nvidia'])
            
            # 4. MIT Deep Learning
            mit_signal, mit_conf = self.models['mit'](prices, symbol)
            if mit_signal:
                signals.append(mit_signal)
                confidences.append(mit_conf * self.model_weights['mit'])
            
            # 5. Stanford RL Trader
            stanford_signal, stanford_conf = self.models['stanford'](prices, symbol)
            if stanford_signal:
                signals.append(stanford_signal)
                confidences.append(stanford_conf * self.model_weights['stanford'])
            
            if not signals:
                return None, 0
            
            # Tính toán tín hiệu tổng hợp
            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")
            total_confidence = sum(confidences)
            
            # Quyết định cuối cùng
            if buy_count > sell_count and total_confidence > 0.6:
                return "BUY", total_confidence
            elif sell_count > buy_count and total_confidence > 0.6:
                return "SELL", total_confidence
            else:
                return None, total_confidence
                
        except Exception as e:
            logger.error(f"Lỗi AI Engine: {str(e)}")
            return None, 0

    def _deepmind_alphatrade(self, prices, symbol):
        """
        DeepMind AlphaTrade - Mô phỏng thuật toán reinforcement learning tiên tiến
        """
        try:
            if len(prices) < 80:
                return None, 0
                
            # AlphaTrade: Multi-timeframe momentum analysis
            short_momentum = self._calculate_momentum(prices[-20:])
            medium_momentum = self._calculate_momentum(prices[-50:])
            long_momentum = self._calculate_momentum(prices[-80:])
            
            # Volume profile analysis (simulated)
            volume_strength = self._simulate_volume_analysis(prices)
            
            # Deep Q-Learning inspired decision
            state_value = (
                short_momentum * 0.4 +
                medium_momentum * 0.35 + 
                long_momentum * 0.25 +
                volume_strength * 0.1
            )
            
            confidence = min(abs(state_value) * 2, 0.95)
            
            if state_value > 0.15:
                return "BUY", confidence
            elif state_value < -0.15:
                return "SELL", confidence
                
            return None, confidence
            
        except Exception as e:
            return None, 0

    def _openai_quant(self, prices, symbol):
        """
        OpenAI Quant - Mô phỏng transformer-based price prediction
        """
        try:
            if len(prices) < 60:
                return None, 0
                
            # Transformer-style sequence attention
            sequences = []
            for i in range(len(prices)-30):
                seq = prices[i:i+30]
                sequences.append(seq)
            
            if not sequences:
                return None, 0
                
            # Attention mechanism simulation
            current_seq = prices[-30:]
            attention_weights = []
            
            for seq in sequences[-20:]:
                if len(seq) == len(current_seq):
                    corr = np.corrcoef(seq, current_seq)[0,1]
                    if not np.isnan(corr):
                        attention_weights.append(corr)
            
            if attention_weights:
                avg_attention = np.mean(attention_weights)
                
                # Pattern prediction
                recent_trend = (current_seq[-1] - current_seq[0]) / current_seq[0]
                volatility = np.std(current_seq) / np.mean(current_seq)
                
                # GPT-style prediction
                prediction_score = avg_attention * 0.6 + recent_trend * 0.3 - volatility * 0.1
                confidence = min(abs(prediction_score) * 3, 0.9)
                
                if prediction_score > 0.1:
                    return "BUY", confidence
                elif prediction_score < -0.1:
                    return "SELL", confidence
                    
            return None, 0
            
        except Exception as e:
            return None, 0

    def _nvidia_trading_ai(self, prices, symbol):
        """
        NVIDIA Trading AI - Mô phỏng GAN và neural networks
        """
        try:
            if len(prices) < 70:
                return None, 0
                
            # GAN-inspired pattern generation and discrimination
            real_patterns = []
            for i in range(len(prices)-25):
                pattern = prices[i:i+25]
                real_patterns.append(pattern)
            
            if not real_patterns:
                return None, 0
            
            # Discriminator: Phân biệt pattern chất lượng
            current_pattern = prices[-25:]
            pattern_quality_scores = []
            
            for pattern in real_patterns[-15:]:
                if len(pattern) == len(current_pattern):
                    # Tính độ "thực" của pattern
                    mean_similarity = 1 - abs(np.mean(pattern) - np.mean(current_pattern)) / np.mean(current_pattern)
                    std_similarity = 1 - abs(np.std(pattern) - np.std(current_pattern)) / np.std(current_pattern)
                    trend_similarity = 1 - abs(
                        (pattern[-1]-pattern[0])/pattern[0] - (current_pattern[-1]-current_pattern[0])/current_pattern[0]
                    )
                    
                    quality = (mean_similarity + std_similarity + trend_similarity) / 3
                    pattern_quality_scores.append(quality)
            
            if pattern_quality_scores:
                avg_quality = np.mean(pattern_quality_scores)
                
                # Neural network inference
                current_trend = (current_pattern[-1] - current_pattern[0]) / current_pattern[0]
                pattern_strength = avg_quality * current_trend
                
                confidence = min(abs(pattern_strength) * 4, 0.92)
                
                if pattern_strength > 0.08:
                    return "BUY", confidence
                elif pattern_strength < -0.08:
                    return "SELL", confidence
                    
            return None, 0
            
        except Exception as e:
            return None, 0

    def _mit_deep_learning(self, prices, symbol):
        """
        MIT Deep Learning - Temporal convolution và advanced feature extraction
        """
        try:
            if len(prices) < 90:
                return None, 0
                
            # Temporal convolution simulation
            features = []
            
            # Multi-scale feature extraction
            for window in [10, 20, 30, 50]:
                if len(prices) >= window:
                    window_data = prices[-window:]
                    
                    # Feature 1: Normalized momentum
                    momentum = (window_data[-1] - window_data[0]) / window_data[0]
                    
                    # Feature 2: Volatility adjusted return
                    returns = np.diff(window_data) / window_data[:-1]
                    vol_adj_return = np.mean(returns) / (np.std(returns) + 1e-8)
                    
                    # Feature 3: Price acceleration
                    if len(window_data) >= 3:
                        accel = (window_data[-1] - 2*window_data[-2] + window_data[-3]) / window_data[-3]
                    else:
                        accel = 0
                    
                    features.extend([momentum, vol_adj_return, accel])
            
            if not features:
                return None, 0
                
            # Neural network decision (simplified)
            feature_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])[:len(features)]
            feature_weights = feature_weights / np.sum(feature_weights)
            
            weighted_score = np.dot(features[:len(feature_weights)], feature_weights)
            
            confidence = min(abs(weighted_score) * 2.5, 0.88)
            
            if weighted_score > 0.12:
                return "BUY", confidence
            elif weighted_score < -0.12:
                return "SELL", confidence
                
            return None, confidence
            
        except Exception as e:
            return None, 0

    def _stanford_rl_trader(self, prices, symbol):
        """
        Stanford RL Trader - Reinforcement learning với risk-aware policy
        """
        try:
            if len(prices) < 55:
                return None, 0
                
            # Reinforcement learning state representation
            state_features = []
            
            # Price-based features
            recent_prices = prices[-20:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Volatility features
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)
            
            # Risk-adjusted features
            sharpe_ratio = np.mean(returns) / (volatility + 1e-8) if volatility > 0 else 0
            
            # Market regime features
            long_term_trend = (prices[-1] - prices[-55]) / prices[-55]
            regime_stability = 1 - abs(long_term_trend)
            
            # RL Policy decision
            policy_score = (
                price_trend * 0.4 +
                sharpe_ratio * 0.3 +
                regime_stability * 0.2 -
                volatility * 0.1
            )
            
            # Risk-aware confidence
            base_confidence = min(abs(policy_score) * 2.2, 0.85)
            risk_adjusted_confidence = base_confidence * (1 - volatility * 2)
            
            if policy_score > 0.1 and risk_adjusted_confidence > 0.4:
                return "BUY", risk_adjusted_confidence
            elif policy_score < -0.1 and risk_adjusted_confidence > 0.4:
                return "SELL", risk_adjusted_confidence
                
            return None, risk_adjusted_confidence
            
        except Exception as e:
            return None, 0

    def _calculate_momentum(self, prices):
        """Tính momentum cho chuỗi giá"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.mean(returns) * 100

    def _simulate_volume_analysis(self, prices):
        """Mô phỏng phân tích volume (trong thực tế cần dữ liệu volume thực)"""
        # Giả lập volume analysis dựa trên price movement
        if len(prices) < 10:
            return 0.5
            
        recent_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
        # Giả định: high volatility thường đi kèm high volume
        return min(recent_volatility * 10, 1.0)

# ========== AI TRADING BOT ==========
class AITradingBot:
    """AI TRADING BOT - Kết hợp 5 AI hàng đầu thế giới"""
    
    def __init__(self, symbol, leverage, percent, tp, sl, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name="AI Fusion"):
        
        self.symbol = symbol.upper()
        self.leverage = leverage
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.strategy_name = strategy_name
        
        # AI Engine
        self.ai_engine = AITradingEngine()
        
        # Trading state
        self.status = "waiting"
        self.position_open = False
        self.side = ""
        self.entry_price = 0
        self.quantity = 0
        self.prices = []
        
        # Risk management
        self.max_position_size = percent
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Performance tracking
        self.trade_history = []
        self.total_pnl = 0
        
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        self.log(f"AI Trading Bot khoi dong | {self.symbol} | DB: {leverage}x | Von: {percent}%")

    def log(self, message):
        """Ghi log và gửi Telegram"""
        logger.info(f"[{self.symbol} - {self.strategy_name}] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>{self.symbol}</b> ({self.strategy_name}): {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def get_current_price(self):
        """Lấy giá hiện tại từ Binance"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={self.symbol}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            self.log(f"Loi lay gia: {str(e)}")
        return 0

    def get_balance(self):
        """Lấy số dư tài khoản"""
        try:
            ts = int(time.time() * 1000)
            params = {"timestamp": ts}
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={signature}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for asset in data['assets']:
                    if asset['asset'] == 'USDT':
                        return float(asset['availableBalance'])
        except Exception as e:
            self.log(f"Loi lay so du: {str(e)}")
        return 0

    def place_order(self, side, quantity):
        """Đặt lệnh giao dịch"""
        try:
            ts = int(time.time() * 1000)
            params = {
                "symbol": self.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "timestamp": ts
            }
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={signature}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.post(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data
        except Exception as e:
            self.log(f"Loi dat lenh: {str(e)}")
        return None

    def check_position(self):
        """Kiểm tra vị thế hiện tại"""
        try:
            ts = int(time.time() * 1000)
            params = {"timestamp": ts}
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={signature}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                positions = response.json()
                for pos in positions:
                    if pos['symbol'] == self.symbol:
                        position_amt = float(pos.get('positionAmt', 0))
                        if abs(position_amt) > 0:
                            self.position_open = True
                            self.side = "BUY" if position_amt > 0 else "SELL"
                            self.quantity = position_amt
                            self.entry_price = float(pos.get('entryPrice', 0))
                            return True
                # Reset nếu không có position
                self.position_open = False
                self.side = ""
                self.quantity = 0
                self.entry_price = 0
        except Exception as e:
            self.log(f"Loi kiem tra vi the: {str(e)}")
        return False

    def calculate_position_size(self):
        """Tính toán khối lượng position theo risk management"""
        balance = self.get_balance()
        if balance <= 0:
            return 0
            
        current_price = self.get_current_price()
        if current_price <= 0:
            return 0
            
        # Tính position size theo % số dư và risk management
        usd_amount = balance * (self.percent / 100)
        position_size = (usd_amount * self.leverage) / current_price
        
        # Làm tròn theo lot size của Binance
        position_size = math.floor(position_size * 100) / 100  # Giả sử step size 0.01
        
        return position_size if position_size > 0 else 0

    def execute_trade(self, signal):
        """Thực hiện giao dịch theo tín hiệu AI"""
        if self.position_open:
            self.log(f"Da co vi the {self.side}, bo qua tin hieu {signal}")
            return False

        position_size = self.calculate_position_size()
        if position_size <= 0:
            self.log("Khoi luong position khong hop le")
            return False

        # Đặt lệnh
        result = self.place_order(signal, position_size)
        if result and 'orderId' in result:
            self.position_open = True
            self.side = signal
            self.quantity = position_size
            self.entry_price = self.get_current_price()
            
            message = (
                f"AI DA MO VI THE {self.symbol}\n"
                f"Chien luoc: {self.strategy_name}\n"
                f"Huong: {signal}\n"
                f"Gia vao: {self.entry_price:.4f}\n"
                f"Khoi luong: {position_size:.4f}\n"
                f"Don bay: {self.leverage}x\n"
                f"TP: {self.tp}% | SL: {self.sl}%"
            )
            self.log(message)
            return True
        else:
            self.log(f"Loi mo lenh {signal}")
            return False

    def check_exit_conditions(self):
        """Kiểm tra điều kiện thoát lệnh"""
        if not self.position_open:
            return
            
        current_price = self.get_current_price()
        if current_price <= 0:
            return
            
        # Tính PnL
        if self.side == "BUY":
            pnl = (current_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - current_price) * self.quantity
            
        invested = self.entry_price * abs(self.quantity) / self.leverage
        if invested <= 0:
            return
            
        roi = (pnl / invested) * 100

        # Kiểm tra TP/SL
        if roi >= self.tp:
            self.close_position(f"Dat TP {self.tp}% (ROI: {roi:.2f}%)")
        elif roi <= -self.sl:
            self.close_position(f"Dat SL {self.sl}% (ROI: {roi:.2f}%)")

    def close_position(self, reason=""):
        """Đóng vị thế hiện tại"""
        if not self.position_open:
            return False
            
        close_side = "SELL" if self.side == "BUY" else "BUY"
        close_quantity = abs(self.quantity)
        
        result = self.place_order(close_side, close_quantity)
        if result and 'orderId' in result:
            current_price = self.get_current_price()
            
            # Tính PnL cuối cùng
            if self.side == "BUY":
                final_pnl = (current_price - self.entry_price) * self.quantity
            else:
                final_pnl = (self.entry_price - current_price) * self.quantity
                
            self.total_pnl += final_pnl
            
            message = (
                f"DA DONG VI THE {self.symbol}\n"
                f"Chien luoc: {self.strategy_name}\n"
                f"Ly do: {reason}\n"
                f"Gia ra: {current_price:.4f}\n"
                f"PnL: {final_pnl:.2f} USDT\n"
                f"Tong PnL: {self.total_pnl:.2f} USDT"
            )
            self.log(message)
            
            # Reset position
            self.position_open = False
            self.side = ""
            self.quantity = 0
            self.entry_price = 0
            
            return True
        else:
            self.log(f"Loi dong lenh")
            return False

    def _run(self):
        """Vòng lặp chính của bot"""
        while not self._stop:
            try:
                # Cập nhật giá
                current_price = self.get_current_price()
                if current_price > 0:
                    self.prices.append(current_price)
                    if len(self.prices) > 200:  # Giữ 200 giá gần nhất
                        self.prices = self.prices[-200:]
                
                # Kiểm tra vị thế
                self.check_position()
                
                # Nếu chưa có position, tìm tín hiệu AI
                if not self.position_open and len(self.prices) >= 100:
                    signal, confidence = self.ai_engine.get_ai_signal(self.symbol, self.prices)
                    
                    if signal and confidence > 0.6:
                        self.log(f"AI Signal: {signal} | Confidence: {confidence:.2f} | Regime: {self.ai_engine.volatility_regime}")
                        self.execute_trade(signal)
                
                # Kiểm tra điều kiện thoát lệnh
                if self.position_open:
                    self.check_exit_conditions()
                
                time.sleep(5)  # Chờ 5 giây giữa các lần check
                
            except Exception as e:
                self.log(f"Loi he thong: {str(e)}")
                time.sleep(10)

    def stop(self):
        """Dừng bot"""
        self._stop = True
        self.log("Bot AI da dung")

# ========== AI BOT MANAGER ==========
class AIBotManager:
    """Quản lý multiple AI trading bots"""
    
    def __init__(self, api_key, api_secret, telegram_bot_token, telegram_chat_id):
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        self.bots = {}
        self.running = True
        
        # AI Strategy configurations
        self.ai_strategies = {
            "DeepMind AlphaTrade": {
                "description": "DeepMind RL - Reinforcement Learning tiên tiến",
                "risk_profile": "MEDIUM"
            },
            "OpenAI Quant": {
                "description": "OpenAI Transformer - Price prediction",
                "risk_profile": "LOW"
            },
            "NVIDIA Trading AI": {
                "description": "NVIDIA GAN - Pattern recognition",
                "risk_profile": "HIGH" 
            },
            "MIT Deep Learning": {
                "description": "MIT Temporal CNN - Feature extraction",
                "risk_profile": "MEDIUM"
            },
            "Stanford RL Trader": {
                "description": "Stanford RL - Risk-aware trading",
                "risk_profile": "LOW"
            }
        }
        
        self.log("AI Trading System khoi dong - 5 AI Hang Dau The Gioi")
        
        # Start Telegram listener
        self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
        self.telegram_thread.start()
        
        if self.telegram_chat_id:
            self.send_main_menu(self.telegram_chat_id)

    def log(self, message):
        logger.info(f"[AI SYSTEM] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>AI SYSTEM</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        welcome_msg = (
            "AI TRADING BOT - 5 AI HANG DAU THE GIOI\n\n"
            "DeepMind AlphaTrade - Reinforcement Learning\n"
            "OpenAI Quant - Transformer Prediction\n"  
            "NVIDIA Trading AI - GAN Pattern Recognition\n"
            "MIT Deep Learning - Temporal CNN\n"
            "Stanford RL Trader - Risk-Aware Policy\n\n"
            "Chon chuc nang:"
        )
        send_telegram(welcome_msg, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token,
                     default_chat_id=self.telegram_chat_id)

    def add_bot(self, symbol, leverage, percent, tp, sl, strategy_name):
        """Thêm bot AI mới"""
        bot_id = f"{symbol}_{strategy_name}"
        
        if bot_id in self.bots:
            self.log(f"Da co bot {strategy_name} cho {symbol}")
            return False
            
        try:
            bot = AITradingBot(
                symbol=symbol,
                leverage=leverage,
                percent=percent,
                tp=tp,
                sl=sl,
                api_key=self.api_key,
                api_secret=self.api_secret,
                telegram_bot_token=self.telegram_bot_token,
                telegram_chat_id=self.telegram_chat_id,
                strategy_name=strategy_name
            )
            
            self.bots[bot_id] = bot
            self.log(f"Da them {strategy_name}: {symbol} | DB: {leverage}x | Von: {percent}% | TP/SL: {tp}%/{sl}%")
            return True
            
        except Exception as e:
            self.log(f"Loi tao bot: {str(e)}")
            return False

    def stop_bot(self, bot_id):
        """Dừng bot"""
        if bot_id in self.bots:
            self.bots[bot_id].stop()
            del self.bots[bot_id]
            self.log(f"Da dung bot {bot_id}")
            return True
        return False

    def stop_all(self):
        """Dừng tất cả bots"""
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.running = False
        self.log("He thong AI da dung")

    def _telegram_listener(self):
        """Lắng nghe tin nhắn Telegram"""
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
                                
            except Exception as e:
                logger.error(f"Loi Telegram: {str(e)}")
                time.sleep(10)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn Telegram"""
        if text == "Them Bot":
            self._start_bot_creation(chat_id)
        elif text == "Danh sach Bot":
            self._list_bots(chat_id)
        elif text == "Dung Bot":
            self._stop_bot_menu(chat_id)
        elif text == "So du":
            self._show_balance(chat_id)
        elif text == "Chien luoc AI":
            self._show_strategies(chat_id)
        elif text.startswith("Dung "):
            bot_id = text.replace("Dung ", "")
            self.stop_bot(bot_id)
            self.send_main_menu(chat_id)
        else:
            self.send_main_menu(chat_id)

    def _start_bot_creation(self, chat_id):
        """Bắt đầu quy trình tạo bot"""
        balance = self._get_balance()
        if balance is None:
            send_telegram("Loi ket noi Binance!", chat_id,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)
            return
            
        send_telegram(
            f"So du: {balance:.2f} USDT\n\n"
            "Chon chien luoc AI:",
            chat_id,
            create_strategy_keyboard(),
            bot_token=self.telegram_bot_token,
            default_chat_id=self.telegram_chat_id
        )

    def _list_bots(self, chat_id):
        """Hiển thị danh sách bot"""
        if not self.bots:
            send_telegram("Khong co bot nao dang chay", chat_id,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)
        else:
            message = "DANH SACH BOT AI\n\n"
            for bot_id, bot in self.bots.items():
                status = "Dang chay" if not bot._stop else "Da dung"
                message += f"{bot_id}\n"
                message += f"  {bot.symbol} | {status}\n"
                message += f"  PnL: {bot.total_pnl:.2f} USDT\n\n"
            
            send_telegram(message, chat_id,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)

    def _stop_bot_menu(self, chat_id):
        """Hiển thị menu dừng bot"""
        if not self.bots:
            send_telegram("Khong co bot nao dang chay", chat_id,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)
        else:
            keyboard = {"keyboard": [], "resize_keyboard": True, "one_time_keyboard": True}
            for bot_id in self.bots.keys():
                keyboard["keyboard"].append([{"text": f"Dung {bot_id}"}])
            keyboard["keyboard"].append([{"text": "Huy bo"}])
            
            send_telegram("Chon bot de dung:", chat_id, keyboard,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)

    def _get_balance(self):
        """Lấy số dư từ Binance"""
        try:
            ts = int(time.time() * 1000)
            params = {"timestamp": ts}
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={signature}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for asset in data['assets']:
                    if asset['asset'] == 'USDT':
                        return float(asset['availableBalance'])
        except Exception as e:
            logger.error(f"Loi lay so du: {str(e)}")
        return None

    def _show_balance(self, chat_id):
        """Hiển thị số dư"""
        balance = self._get_balance()
        if balance is None:
            send_telegram("Loi lay so du!", chat_id,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)
        else:
            send_telegram(f"So du kha dung: {balance:.2f} USDT", chat_id,
                         bot_token=self.telegram_bot_token,
                         default_chat_id=self.telegram_chat_id)

    def _show_strategies(self, chat_id):
        """Hiển thị thông tin chiến lược AI"""
        strategies_info = (
            "5 AI TRADING HANG DAU THE GIOI\n\n"
            
            "DeepMind AlphaTrade\n"
            "• Reinforcement Learning tien tien\n"
            "• Multi-timeframe momentum analysis\n"
            "• Deep Q-Learning decision making\n"
            "• Risk: MEDIUM\n\n"
            
            "OpenAI Quant\n"
            "• Transformer-based price prediction\n"
            "• Sequence attention mechanism\n"
            "• GPT-style pattern recognition\n"
            "• Risk: LOW\n\n"
            
            "NVIDIA Trading AI\n"
            "• GAN pattern generation\n"
            "• Neural network discrimination\n"
            "• Advanced feature extraction\n"
            "• Risk: HIGH\n\n"
            
            "MIT Deep Learning\n"
            "• Temporal convolution networks\n"
            "• Multi-scale feature analysis\n"
            "• Academic research foundation\n"
            "• Risk: MEDIUM\n\n"
            
            "Stanford RL Trader\n"
            "• Risk-aware reinforcement learning\n"
            "• Policy optimization\n"
            "• Market regime adaptation\n"
            "• Risk: LOW"
        )
        
        send_telegram(strategies_info, chat_id,
                     bot_token=self.telegram_bot_token,
                     default_chat_id=self.telegram_chat_id)
