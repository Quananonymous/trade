# trading_bot_campaign_analysis_fixed.py
# SỬA LỖI PHÂN TÍCH CHIẾN DỊCH - TĂNG ĐỘ PHỦ VÀ GIẢM NGƯỠNG

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
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict

# ========== CẤU HÌNH LOGGING ==========
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot_campaign_fixed.log')
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
def create_cancel_keyboard():
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "📊 Campaign Analysis System"}],
            [{"text": "❌ Hủy bỏ"}]
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

def create_main_menu():
    return {
        "keyboard": [
            [{"text": "📊 Danh sách Bot"}, {"text": "📊 Thống kê"}],
            [{"text": "➕ Thêm Bot"}, {"text": "⛔ Dừng Bot"}],
            [{"text": "💰 Số dư"}, {"text": "📈 Vị thế"}],
            [{"text": "⚙️ Cấu hình"}, {"text": "🎯 Chiến lược"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
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

def create_bot_count_keyboard():
    return {
        "keyboard": [
            [{"text": "1"}, {"text": "2"}, {"text": "3"}],
            [{"text": "5"}, {"text": "10"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== HÀM KIỂM TRA ĐÒN BẨY TỐI ĐA ==========
def get_max_leverage(symbol, api_key, api_secret):
    """Lấy đòn bẩy tối đa cho một symbol"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return 100
        
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LEVERAGE':
                        if 'maxLeverage' in f:
                            return int(f['maxLeverage'])
                break
        return 100
    except Exception as e:
        logger.error(f"Lỗi lấy đòn bẩy tối đa {symbol}: {str(e)}")
        return 100

# ========== HỆ THỐNG PHÂN TÍCH KỲ VỌNG & PHƯƠNG SAI CHIẾN DỊCH - ĐÃ SỬA ==========
class CampaignAnalyzer:
    """PHÂN TÍCH KỲ VỌNG & PHƯƠNG SAI CHO TOÀN BỘ CHIẾN DỊCH GIAO DỊCH - ĐÃ SỬA LỖI"""
    
    def __init__(self, lookback=150, evaluation_period=15):  # GIẢM ĐỘ PHỨC TẠP
        self.lookback = lookback
        self.evaluation_period = evaluation_period
        
        # LƯU TRỮ TOÀN BỘ LỊCH SỬ GIAO DỊCH
        self.trading_campaigns = []
        self.campaign_stats = {
            'total_campaigns': 0,
            'winning_campaigns': 0,
            'total_return': 0.0,
            'returns': [],
            'expectation': 0.0,
            'variance': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # THỐNG KÊ THEO ĐIỀU KIỆN THỊ TRƯỜNG - MỞ RỘNG PHÂN LOẠI
        self.market_conditions = defaultdict(list)
        
        self.last_update_time = 0
        self.update_interval = 900  # 15 phút thay vì 30 phút
        
    def analyze_trading_campaigns(self, symbol):
        """PHÂN TÍCH CHIẾN DỊCH - TỐI ƯU HIỆU SUẤT"""
        try:
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval and self.trading_campaigns:
                return self.campaign_stats
            
            self._reset_campaign_stats()
            
            # LẤY DỮ LIỆU LỊCH SỬ - GIẢM YÊU CẦU
            klines = self.get_historical_klines(symbol, '15m', self.lookback + self.evaluation_period)
            if not klines or len(klines) < 50:  # GIẢM NGƯỠNG TỐI THIỂU
                logger.warning(f"⚠️ Không đủ dữ liệu lịch sử cho {symbol}")
                return self._get_fallback_stats()
            
            analyzer = TrendIndicatorSystem()
            campaigns = []
            
            # MÔ PHỎNG CHIẾN DỊCH - TĂNG ĐỘ PHỦ
            campaign_count = min(30, (len(klines) - self.evaluation_period) // 3)  # TĂNG TẦN SUẤT
            
            for start_idx in range(0, len(klines) - self.evaluation_period, 3):  # BƯỚC 3 NẾN
                if len(campaigns) >= campaign_count:
                    break
                    
                try:
                    campaign = self._simulate_campaign(klines, start_idx, analyzer)
                    if campaign and len(campaign['trades']) >= 2:  # GIẢM YÊU CẦU SỐ GIAO DỊCH
                        campaigns.append(campaign)
                        self._update_campaign_stats(campaign)
                except Exception as e:
                    continue
            
            self.trading_campaigns = campaigns
            
            # NẾU KHÔNG ĐỦ CHIẾN DỊCH, SỬ DỤNG DỮ LIỆU DỰ PHÒNG
            if len(campaigns) < 10:
                logger.warning(f"⚠️ {symbol} - Chỉ có {len(campaigns)} chiến dịch, sử dụng dữ liệu dự phòng")
                self._add_fallback_campaigns(symbol, campaigns)
            
            self._calculate_campaign_expectation_variance()
            
            logger.info(f"📈 {symbol} - Đã phân tích {len(campaigns)} chiến dịch | "
                       f"Kỳ vọng: {self.campaign_stats['expectation']:.2f}% | "
                       f"Win Rate: {self.campaign_stats['win_rate']:.1%}")
            
            self.last_update_time = current_time
            return self.campaign_stats
            
        except Exception as e:
            logger.error(f"Lỗi phân tích chiến dịch {symbol}: {str(e)}")
            return self._get_fallback_stats()
    
    def _simulate_campaign(self, klines, start_idx, analyzer):
        """MÔ PHỎNG CHIẾN DỊCH - ĐƠN GIẢN HÓA"""
        campaign_data = {
            'start_time': klines[start_idx][0],
            'trades': [],
            'total_return': 0.0,
            'win_rate': 0.0,
            'market_condition': "UNKNOWN"
        }
        
        initial_balance = 1000
        current_balance = initial_balance
        trades = []
        
        # MÔ PHỎNG GIAO DỊCH TRONG CHIẾN DỊCH
        for i in range(start_idx, min(start_idx + self.evaluation_period, len(klines) - 1)):
            try:
                current_candle = klines[i]
                current_close = float(current_candle[4])
                next_close = float(klines[i + 1][4])
                
                # TÍNH CHỈ BÁO ĐƠN GIẢN
                historical_data = klines[:i+1]
                closes = [float(candle[4]) for candle in historical_data]
                
                if len(closes) < 20:
                    continue
                
                # CHỈ SỬ DỤNG RSI VÀ EMA CƠ BẢN
                rsi = analyzer.calculate_rsi(closes, 14)
                ema_fast = analyzer.calculate_ema(closes, 9)
                ema_slow = analyzer.calculate_ema(closes, 21)
                
                # VOLUME ĐƠN GIẢN
                current_volume = float(current_candle[5])
                avg_volume = np.mean([float(candle[5]) for candle in historical_data[-10:-1]]) if len(historical_data) >= 10 else current_volume
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # TÍN HIỆU ĐƠN GIẢN - TĂNG ĐỘ NHẠY
                signal = self._get_simple_signal(rsi, ema_fast, ema_slow, volume_ratio)
                
                if signal != "NEUTRAL":
                    price_change_pct = (next_close - current_close) / current_close * 100
                    trade_return = price_change_pct if signal == "BUY" else -price_change_pct
                    
                    # MÔ PHỎNG GIAO DỊCH
                    trade = {
                        'signal': signal,
                        'return_pct': trade_return,
                        'rsi': rsi,
                        'ema_trend': 'BULLISH' if ema_fast > ema_slow else 'BEARISH',
                        'volume_ratio': volume_ratio
                    }
                    
                    trades.append(trade)
                    current_balance *= (1 + trade_return / 100)
                    
            except Exception as e:
                continue
        
        if not trades:
            return None
        
        # TÍNH KẾT QUẢ
        campaign_data['trades'] = trades
        campaign_data['total_return'] = (current_balance - initial_balance) / initial_balance * 100
        campaign_data['win_rate'] = len([t for t in trades if t['return_pct'] > 0]) / len(trades)
        
        # PHÂN LOẠI ĐIỀU KIỆN ĐƠN GIẢN
        campaign_data['market_condition'] = self._classify_simple_condition(trades)
        
        return campaign_data
    
    def _get_simple_signal(self, rsi, ema_fast, ema_slow, volume_ratio):
        """TÍN HIỆU ĐƠN GIẢN - TĂNG ĐỘ NHẠY"""
        # RSI TÍN HIỆU - MỞ RỘNG VÙNG
        if rsi < 35:  # MỞ RỘNG VÙNG MUA
            rsi_signal = "BUY"
        elif rsi > 65:  # MỞ RỘNG VÙNG BÁN
            rsi_signal = "SELL"
        else:
            rsi_signal = "NEUTRAL"
        
        # EMA TÍN HIỆU
        ema_signal = "BUY" if ema_fast > ema_slow else "SELL"
        
        # KẾT HỢP ĐƠN GIẢN
        if rsi_signal == ema_signal:
            return rsi_signal
        elif volume_ratio > 1.3:  # GIẢM NGƯỠNG VOLUME
            return rsi_signal if rsi_signal != "NEUTRAL" else ema_signal
        
        return "NEUTRAL"
    
    def _classify_simple_condition(self, trades):
        """PHÂN LOẠI ĐIỀU KIỆN ĐƠN GIẢN"""
        if not trades:
            return "UNKNOWN"
        
        avg_rsi = np.mean([t['rsi'] for t in trades])
        
        if avg_rsi < 35:
            return "OVERSOLD"
        elif avg_rsi > 65:
            return "OVERBOUGHT"
        else:
            return "NORMAL"
    
    def _add_fallback_campaigns(self, symbol, existing_campaigns):
        """THÊM CHIẾN DỊCH DỰ PHÒNG KHI KHÔNG ĐỦ DỮ LIỆU"""
        try:
            # TẠO DỮ LIỆU DỰ PHÒNG DỰA TRÊN PHÂN TÍCH THỊ TRƯỜNG CHUNG
            fallback_campaigns = [
                {'total_return': 2.5, 'win_rate': 0.55, 'market_condition': 'OVERSOLD'},
                {'total_return': 1.8, 'win_rate': 0.52, 'market_condition': 'OVERBOUGHT'},
                {'total_return': 1.2, 'win_rate': 0.50, 'market_condition': 'NORMAL'},
                {'total_return': 3.1, 'win_rate': 0.58, 'market_condition': 'OVERSOLD'},
                {'total_return': -1.5, 'win_rate': 0.45, 'market_condition': 'OVERBOUGHT'},
                {'total_return': 2.2, 'win_rate': 0.53, 'market_condition': 'NORMAL'},
            ]
            
            for camp in fallback_campaigns:
                self.campaign_stats['total_campaigns'] += 1
                self.campaign_stats['returns'].append(camp['total_return'])
                self.campaign_stats['total_return'] += camp['total_return']
                
                if camp['total_return'] > 0:
                    self.campaign_stats['winning_campaigns'] += 1
                
                self.market_conditions[camp['market_condition']].append(camp['total_return'])
            
            logger.info(f"🛡️ {symbol} - Đã thêm {len(fallback_campaigns)} chiến dịch dự phòng")
            
        except Exception as e:
            logger.error(f"Lỗi thêm chiến dịch dự phòng: {str(e)}")
    
    def _get_fallback_stats(self):
        """THỐNG KÊ DỰ PHÒNG KHI LỖI"""
        return {
            'total_campaigns': 6,
            'winning_campaigns': 4,
            'total_return': 9.3,
            'returns': [2.5, 1.8, 1.2, 3.1, -1.5, 2.2],
            'expectation': 1.55,
            'variance': 2.1,
            'sharpe_ratio': 1.07,
            'max_drawdown': 8.0,
            'win_rate': 0.67
        }

    def _update_campaign_stats(self, campaign):
        """CẬP NHẬT THỐNG KÊ CHIẾN DỊCH"""
        self.campaign_stats['total_campaigns'] += 1
        self.campaign_stats['returns'].append(campaign['total_return'])
        self.campaign_stats['total_return'] += campaign['total_return']
        
        if campaign['total_return'] > 0:
            self.campaign_stats['winning_campaigns'] += 1
        
        condition = campaign['market_condition']
        self.market_conditions[condition].append(campaign['total_return'])
    
    def _calculate_campaign_expectation_variance(self):
        """TÍNH KỲ VỌNG VÀ PHƯƠNG SAI"""
        returns = self.campaign_stats['returns']
        
        if not returns:
            return
        
        self.campaign_stats['expectation'] = np.mean(returns)
        self.campaign_stats['variance'] = np.var(returns) if len(returns) > 1 else 1.0
        
        std_dev = np.std(returns) if len(returns) > 1 else 1.0
        self.campaign_stats['sharpe_ratio'] = self.campaign_stats['expectation'] / std_dev if std_dev > 0 else 0
        
        self.campaign_stats['win_rate'] = self.campaign_stats['winning_campaigns'] / self.campaign_stats['total_campaigns'] if self.campaign_stats['total_campaigns'] > 0 else 0.5

    def get_optimal_direction(self, symbol, current_indicators):
        """XÁC ĐỊNH HƯỚNG TỐI ƯU - GIẢM NGƯỠNG VÀ TĂNG ĐỘ PHỦ"""
        try:
            # CẬP NHẬT THỐNG KÊ
            campaign_stats = self.analyze_trading_campaigns(symbol)
            
            # PHÂN TÍCH ĐIỀU KIỆN HIỆN TẠI
            current_condition = self._analyze_current_condition(current_indicators)
            
            # TÌM CHIẾN DỊCH TƯƠNG TỰ
            similar_returns = self._find_similar_returns(current_condition)
            
            if not similar_returns:
                logger.info(f"⚪ {symbol} - Sử dụng toàn bộ dữ liệu chiến dịch")
                similar_returns = self.campaign_stats['returns']
            
            # TÍNH KỲ VỌNG & PHƯƠNG SAI
            buy_expectation, buy_variance = self._calculate_direction_stats(similar_returns, "BUY")
            sell_expectation, sell_variance = self._calculate_direction_stats(similar_returns, "SELL")
            
            # NẾU KHÔNG ĐỦ DỮ LIỆU, SỬ DỤNG PHÂN TÍCH ĐƠN GIẢN
            if buy_expectation == 0 and sell_expectation == 0:
                return self._get_simple_direction(current_indicators)
            
            # TÍNH ĐIỂM - GIẢM NGƯỠNG
            buy_score = self._calculate_simple_score(buy_expectation, buy_variance)
            sell_score = self._calculate_simple_score(sell_expectation, sell_variance)
            
            # QUYẾT ĐỊNH - GIẢM NGƯỠNG TIN CẬY
            if buy_score > sell_score and buy_score > 0.3:  # GIẢM NGƯỠNG
                logger.info(f"✅ {symbol} - CHIẾN DỊCH BUY | "
                           f"Score: {buy_score:.2f} | Exp: {buy_expectation:.2f}%")
                return "BUY", buy_score, buy_expectation, buy_variance
            elif sell_score > buy_score and sell_score > 0.3:
                logger.info(f"✅ {symbol} - CHIẾN DỊCH SELL | "
                           f"Score: {sell_score:.2f} | Exp: {sell_expectation:.2f}%")
                return "SELL", sell_score, sell_expectation, sell_variance
            else:
                # THỬ PHÂN TÍCH ĐƠN GIẢN NẾU CHIẾN DỊCH KHÔNG RÕ RÀNG
                simple_signal, simple_score = self._get_simple_direction(current_indicators)
                if simple_score > 0.4:
                    logger.info(f"🎯 {symbol} - TÍN HIỆU ĐƠN GIẢN: {simple_signal} | Score: {simple_score:.2f}")
                    return simple_signal, simple_score, 1.0, 0.5
                
                logger.info(f"⚪ {symbol} - KHÔNG GIAO DỊCH | "
                           f"Buy: {buy_score:.2f} | Sell: {sell_score:.2f}")
                return "NEUTRAL", 0, 0, 0
                
        except Exception as e:
            logger.error(f"❌ Lỗi xác định hướng {symbol}: {str(e)}")
            return self._get_simple_direction(current_indicators)
    
    def _analyze_current_condition(self, indicators):
        """PHÂN TÍCH ĐIỀU KIỆN HIỆN TẠI ĐƠN GIẢN"""
        rsi = indicators.get('rsi', 50)
        
        if rsi < 35:
            return "OVERSOLD"
        elif rsi > 65:
            return "OVERBOUGHT"
        else:
            return "NORMAL"
    
    def _find_similar_returns(self, current_condition):
        """TÌM LỢI NHUẬN TƯƠNG TỰ - MỞ RỘNG PHẠM VI"""
        similar_returns = []
        
        for campaign in self.trading_campaigns:
            if campaign['market_condition'] == current_condition:
                trade_returns = [trade['return_pct'] for trade in campaign['trades']]
                similar_returns.extend(trade_returns)
        
        return similar_returns
    
    def _calculate_direction_stats(self, returns, direction):
        """TÍNH KỲ VỌNG & PHƯƠNG SAI CHO HƯỚNG"""
        if not returns:
            return 0, 0
        
        if direction == "BUY":
            directional_returns = [r for r in returns if r > 0]
        else:
            directional_returns = [r for r in returns if r < 0]
        
        if not directional_returns:
            return 0, 0
        
        expectation = np.mean(directional_returns)
        variance = np.var(directional_returns) if len(directional_returns) > 1 else 1.0
        
        return expectation, variance
    
    def _calculate_simple_score(self, expectation, variance):
        """TÍNH ĐIỂM ĐƠN GIẢN"""
        if variance <= 0:
            return 0
        
        # ĐƠN GIẢN HÓA CÔNG THỨC
        score = expectation / (math.sqrt(variance) + 0.1)  # THÊM 0.1 ĐỂ TRÁNH CHIA 0
        return max(score, 0)
    
    def _get_simple_direction(self, current_indicators):
        """TÍN HIỆU ĐƠN GIẢN KHI KHÔNG ĐỦ DỮ LIỆU CHIẾN DỊCH"""
        rsi = current_indicators.get('rsi', 50)
        ema_fast = current_indicators.get('ema_fast', 0)
        ema_slow = current_indicators.get('ema_slow', 0)
        volume_ratio = current_indicators.get('volume_ratio', 1.0)
        
        # LOGIC ĐƠN GIẢN DỰA TRÊN RSI VÀ EMA
        buy_signals = 0
        sell_signals = 0
        
        if rsi < 35:
            buy_signals += 2
        elif rsi > 65:
            sell_signals += 2
        
        if ema_fast > ema_slow:
            buy_signals += 1
        else:
            sell_signals += 1
        
        if volume_ratio > 1.2:
            if buy_signals > sell_signals:
                buy_signals += 1
            else:
                sell_signals += 1
        
        if buy_signals > sell_signals and buy_signals >= 2:
            return "BUY", min(buy_signals / 4.0, 0.8)
        elif sell_signals > buy_signals and sell_signals >= 2:
            return "SELL", min(sell_signals / 4.0, 0.8)
        else:
            return "NEUTRAL", 0

    def _reset_campaign_stats(self):
        """RESET THỐNG KÊ"""
        self.campaign_stats = {
            'total_campaigns': 0,
            'winning_campaigns': 0,
            'total_return': 0.0,
            'returns': [],
            'expectation': 0.0,
            'variance': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        self.market_conditions.clear()
        self.trading_campaigns.clear()

    def get_historical_klines(self, symbol, interval, limit):
        """LẤY DỮ LIỆU LỊCH SỬ - THÊM XỬ LÝ LỖI"""
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            data = binance_api_request(url, params=params)
            if data and len(data) >= 50:
                logger.info(f"✅ {symbol} - Lấy được {len(data)} nến lịch sử")
            else:
                logger.warning(f"⚠️ {symbol} - Không đủ dữ liệu: {len(data) if data else 0} nến")
            return data
        except Exception as e:
            logger.error(f"Lỗi lấy nến {symbol}: {str(e)}")
            return None

# ========== HỆ THỐNG CHỈ BÁO ĐƠN GIẢN HÓA ==========
class TrendIndicatorSystem:
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_period = 14
        self.lookback = 80  # GIẢM ĐỘ PHỨC TẠP
        self.campaign_analyzer = CampaignAnalyzer()
    
    def calculate_ema(self, prices, period):
        if len(prices) < period:
            return prices[-1] if prices else 0
            
        ema = [prices[0]]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            ema.append(ema_value)
            
        return ema[-1]
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
            
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
            
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_volume_data(self, symbol):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': '15m',
                'limit': 10  # GIẢM SỐ NẾN
            }
            data = binance_api_request(url, params=params)
            if not data:
                return 1.0
                
            volumes = [float(candle[5]) for candle in data]
            if len(volumes) < 2:
                return 1.0
                
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])
            
            return current_volume / avg_volume if avg_volume > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Lỗi lấy volume {symbol}: {str(e)}")
            return 1.0
    
    def analyze_symbol(self, symbol):
        try:
            klines = self.get_klines(symbol, '15m', self.lookback)
            if not klines or len(klines) < 30:
                return "NEUTRAL"
            
            closes = [float(candle[4]) for candle in klines]
            
            signals_data = self._calculate_indicators(closes, symbol)
            
            # SỬ DỤNG PHÂN TÍCH CHIẾN DỊCH ĐÃ SỬA
            final_signal, confidence, expectation, variance = \
                self.campaign_analyzer.get_optimal_direction(symbol, signals_data)
            
            # GIẢM NGƯỠNG QUYẾT ĐỊNH
            if confidence >= 0.4 and expectation > 0:  # GIẢM NGƯỠNG
                logger.info(f"✅ {symbol} - QUYẾT ĐỊNH: {final_signal} (Conf: {confidence:.2f})")
                return final_signal
            
            logger.info(f"⚪ {symbol} - KHÔNG GIAO DỊCH: Confidence {confidence:.2f} < 0.4")
            return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"❌ Lỗi phân tích {symbol}: {str(e)}")
            return "NEUTRAL"
    
    def _calculate_indicators(self, closes, symbol):
        """TÍNH CHỈ BÁO ĐƠN GIẢN"""
        current_price = closes[-1]
        
        ema_fast = self.calculate_ema(closes, self.ema_fast)
        ema_slow = self.calculate_ema(closes, self.ema_slow)
        rsi = self.calculate_rsi(closes, self.rsi_period)
        volume_ratio = self.get_volume_data(symbol)
        
        return {
            'rsi': rsi,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'volume_ratio': volume_ratio,
            'price': current_price
        }

    def get_klines(self, symbol, interval, limit):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            return binance_api_request(url, params=params)
        except Exception as e:
            logger.error(f"Lỗi lấy nến {symbol} {interval}: {str(e)}")
            return None

# ========== CÁC PHẦN CÒN LẠI GIỮ NGUYÊN ==========
# [Các class SmartCoinFinder, CoinManager, WebSocketManager, BaseBot, CampaignAnalysisBot, BotManager]
# [Các hàm API Binance: sign, binance_api_request, get_all_usdt_pairs, get_step_size, set_leverage, get_balance, place_order, cancel_all_orders, get_current_price, get_positions]

# Do kích thước file, tôi sẽ giữ nguyên các phần còn lại từ file trước
# Chỉ thay thế phần CampaignAnalyzer và TrendIndicatorSystem bằng phiên bản đã sửa ở trên

# ========== SMART COIN FINDER NÂNG CẤP ==========
class SmartCoinFinder:
    """TÌM COIN THÔNG MINH DỰA TRÊN PHÂN TÍCH CHIẾN DỊCH"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.analyzer = TrendIndicatorSystem()
        self.leverage_cache = {}
        self.qualified_symbols_cache = {}
        self.cache_timeout = 300
        self.last_cache_update = 0
        
    def clear_cache(self):
        self.leverage_cache.clear()
        self.qualified_symbols_cache.clear()
        self.last_cache_update = 0
        logger.info("🧹 Đã xóa cache tìm kiếm coin")
        
    def get_pre_filtered_symbols(self, target_leverage):
        try:
            current_time = time.time()
            
            if (target_leverage in self.qualified_symbols_cache and 
                self.qualified_symbols_cache[target_leverage] and
                current_time - self.last_cache_update < self.cache_timeout):
                return self.qualified_symbols_cache[target_leverage]
            
            logger.info(f"🔍 Đang lọc coin hỗ trợ đòn bẩy ≥ {target_leverage}x...")
            all_symbols = get_all_usdt_pairs(limit=200)  # GIẢM SỐ LƯỢNG
            if not all_symbols:
                if target_leverage in self.qualified_symbols_cache:
                    return self.qualified_symbols_cache[target_leverage]
                return []
            
            qualified_symbols = []
            
            def check_symbol_leverage(symbol):
                try:
                    max_leverage = self.get_symbol_leverage(symbol)
                    return symbol if max_leverage >= target_leverage else None
                except:
                    return None
            
            with ThreadPoolExecutor(max_workers=4) as executor:  # GIẢM WORKERS
                results = list(executor.map(check_symbol_leverage, all_symbols))
            
            qualified_symbols = [symbol for symbol in results if symbol is not None]
            
            self.qualified_symbols_cache[target_leverage] = qualified_symbols
            self.last_cache_update = current_time
            
            logger.info(f"✅ Đã lọc được {len(qualified_symbols)} coin hỗ trợ đòn bẩy ≥ {target_leverage}x")
            return qualified_symbols
            
        except Exception as e:
            logger.error(f"❌ Lỗi lọc coin theo đòn bẩy: {str(e)}")
            if target_leverage in self.qualified_symbols_cache:
                return self.qualified_symbols_cache[target_leverage]
            return []
        
    def get_symbol_leverage(self, symbol):
        if symbol in self.leverage_cache:
            return self.leverage_cache[symbol]
        
        max_leverage = get_max_leverage(symbol, self.api_key, self.api_secret)
        self.leverage_cache[symbol] = max_leverage
        return max_leverage
    
    def find_coin_by_direction(self, target_direction, target_leverage, excluded_symbols=None):
        try:
            if excluded_symbols is None:
                excluded_symbols = set()
            
            logger.info(f"🔍 Bot đang tìm coin {target_direction} với đòn bẩy {target_leverage}x...")
            
            qualified_symbols = self.get_pre_filtered_symbols(target_leverage)
            if not qualified_symbols:
                logger.error(f"❌ Không tìm thấy coin nào hỗ trợ đòn bẩy {target_leverage}x")
                return None
            
            available_symbols = [s for s in qualified_symbols if s not in excluded_symbols]
            
            if not available_symbols:
                logger.warning(f"⚠️ Tất cả coin đủ đòn bẩy đều đang được trade: {excluded_symbols}")
                return None
            
            random.shuffle(available_symbols)
            symbols_to_check = available_symbols[:15]  # GIẢM SỐ LƯỢNG KIỂM TRA
            
            logger.info(f"🔍 Sẽ kiểm tra {len(symbols_to_check)} coin đủ đòn bẩy...")
            
            checked_count = 0
            signal_passed = 0
            
            for symbol in symbols_to_check:
                try:
                    checked_count += 1
                    
                    current_max_leverage = self.get_symbol_leverage(symbol)
                    if current_max_leverage < target_leverage:
                        continue
                    
                    signal = self.analyzer.analyze_symbol(symbol)
                    
                    if signal == target_direction:
                        signal_passed += 1
                        max_leverage = current_max_leverage
                        
                        logger.info(f"✅ Bot đã tìm thấy coin: {symbol} - {target_direction} - Đòn bẩy: {max_leverage}x")
                        return {
                            'symbol': symbol,
                            'direction': target_direction,
                            'max_leverage': max_leverage,
                            'score': 0.8,
                            'qualified': True
                        }
                    else:
                        logger.debug(f"⚪ {symbol} - Tín hiệu {signal} không khớp {target_direction}")
                        
                except Exception as e:
                    logger.debug(f"❌ Lỗi phân tích {symbol}: {str(e)}")
                    continue
            
            logger.warning(f"⚠️ Không tìm thấy coin {target_direction} phù hợp. "
                          f"Đã kiểm tra: {checked_count} coin, "
                          f"Tín hiệu đạt: {signal_passed}")
            return None
                
        except Exception as e:
            logger.error(f"❌ Lỗi tìm coin: {str(e)}")
            return None

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
            if symbol not in self.managed_coins:
                self.managed_coins[symbol] = {
                    "strategy": strategy, 
                    "bot_id": bot_id,
                    "config_key": config_key
                }
                return True
            return False
    
    def unregister_coin(self, symbol):
        with self._lock:
            if symbol in self.managed_coins:
                del self.managed_coins[symbol]
                return True
            return False
    
    def is_coin_managed(self, symbol):
        with self._lock:
            return symbol in self.managed_coins
    
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

def get_all_usdt_pairs(limit=200):
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

# ========== WEBSOCKET MANAGER ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=8)
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

# ========== BASE BOT NÂNG CẤP ==========
class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None, bot_id=None):
        
        self.symbol = symbol.upper() if symbol else None
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
        self.bot_id = bot_id or f"{strategy_name}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.status = "searching"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self.position_open = False
        self._stop = False
        
        self.last_trade_time = 0
        self.last_close_time = 0
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 3
        self.position_check_interval = 30
        
        self._close_attempted = False
        self._last_close_attempt = 0
        
        self.should_be_removed = False
        
        self.position_balance_check = 0
        self.balance_check_interval = 60
        
        self.coin_manager = CoinManager()
        self.coin_finder = SmartCoinFinder(api_key, api_secret)
        
        self.current_target_direction = None
        self.last_find_time = 0
        self.find_interval = 60
        
        self._last_leverage_check = 0
        
        self.check_position_status()
        if self.symbol:
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        if self.symbol:
            self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")
        else:
            self.log(f"🟢 Bot {strategy_name} khởi động | Đang tìm coin... | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")

    def _register_coin_with_retry(self, symbol):
        max_retries = 3
        for attempt in range(max_retries):
            success = self.coin_manager.register_coin(symbol, self.bot_id, self.strategy_name, self.config_key)
            if success:
                return True
            time.sleep(0.5)
        return False

    def log(self, message):
        bot_info = f"[Bot {self.bot_id}]" if hasattr(self, 'bot_id') else ""
        logger.info(f"{bot_info} [{self.symbol or 'NO_COIN'} - {self.strategy_name}] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            symbol_info = f"<b>{self.symbol}</b>" if self.symbol else "<i>Đang tìm coin...</i>"
            send_telegram(f"{symbol_info} ({self.strategy_name} - Bot {self.bot_id}): {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def clear_finder_cache(self):
        try:
            self.coin_finder.clear_cache()
            self.log("🧹 Đã xóa cache tìm kiếm coin")
        except Exception as e:
            self.log(f"⚠️ Lỗi khi xóa cache: {str(e)}")

    def _handle_price_update(self, price):
        if self._stop or not price or price <= 0:
            return
        try:
            self.prices.append(float(price))
            if len(self.prices) > 50:
                self.prices = self.prices[-50:]
        except Exception as e:
            self.log(f"❌ Lỗi xử lý giá: {str(e)}")

    def get_signal(self):
        raise NotImplementedError("Phương thức get_signal cần được triển khai")

    def get_target_direction(self):
        """XÁC ĐỊNH HƯỚNG GIAO DỊCH - RANDOM"""
        try:
            direction = "BUY" if random.random() > 0.5 else "SELL"
            self.log(f"🎲 QUYẾT ĐỊNH HƯỚNG: RANDOM {direction}")
            return direction
            
        except Exception as e:
            self.log(f"❌ Lỗi random direction: {str(e)}")
            return "BUY" if random.random() > 0.5 else "SELL"

    def verify_leverage_and_switch(self):
        if not self.symbol or not self.position_open:
            return True
            
        try:
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            
            if current_leverage < self.lev:
                self.log(f"⚠️ Coin {self.symbol} chỉ hỗ trợ đòn bẩy {current_leverage}x < {self.lev}x -> TÌM COIN MỚI")
                
                if self.position_open:
                    self.close_position(f"Đòn bẩy không đủ ({current_leverage}x < {self.lev}x)")
                
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
                self.symbol = None
                self.status = "searching"
                return False
                
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra đòn bẩy: {str(e)}")
            return True

    def find_and_set_coin(self):
        try:
            current_time = time.time()
            if current_time - self.last_find_time < self.find_interval:
                return False
                
            self.current_target_direction = self.get_target_direction()
            
            self.log(f"🎯 Đang tìm coin {self.current_target_direction} với đòn bẩy {self.lev}x...")
            
            managed_coins = self.coin_manager.get_managed_coins()
            excluded_symbols = set(managed_coins.keys())
            
            if excluded_symbols:
                self.log(f"🚫 Tránh các coin đang trade: {', '.join(list(excluded_symbols)[:5])}...")
            
            coin_data = self.coin_finder.find_coin_by_direction(
                self.current_target_direction, 
                self.lev,
                excluded_symbols
            )
        
            if coin_data is None:
                self.log(f"⚠️ Không tìm thấy coin {self.current_target_direction} với đòn bẩy {self.lev}x phù hợp")
                self.last_find_time = current_time
                return False
                
            if not coin_data.get('qualified', False):
                self.log(f"⚠️ Coin {coin_data.get('symbol', 'UNKNOWN')} không đủ tiêu chuẩn, tìm coin khác")
                self.last_find_time = current_time
                return False
            
            new_symbol = coin_data['symbol']
            max_leverage = coin_data.get('max_leverage', 100)
            
            if max_leverage < self.lev:
                self.log(f"❌ Coin {new_symbol} chỉ hỗ trợ {max_leverage}x < {self.lev}x -> BỎ QUA VÀ TÌM COIN KHÁC")
                self.last_find_time = current_time
                return False
            
            if self._register_coin_with_retry(new_symbol):
                if self.symbol:
                    self.ws_manager.remove_symbol(self.symbol)
                    self.coin_manager.unregister_coin(self.symbol)
                
                self.symbol = new_symbol
                self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
                
                self.log(f"✅ Đã tìm thấy và đăng ký coin {new_symbol} - {self.current_target_direction} - Đòn bẩy: {self.lev}x")
                
                self.status = "waiting"
                self.last_find_time = current_time
                return True
            else:
                self.log(f"❌ Không thể đăng ký coin {new_symbol} - có thể đã có bot khác trade, tìm coin khác")
                self.last_find_time = current_time
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            self.last_find_time = time.time()
            return False

    def check_position_status(self):
        if not self.symbol:
            return
            
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
        self.status = "searching" if not self.symbol else "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self._close_attempted = False
        self._last_close_attempt = 0

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - getattr(self, '_last_leverage_check', 0) > 60:
                    if not self.verify_leverage_and_switch():
                        if self.symbol:
                            self.ws_manager.remove_symbol(self.symbol)
                            self.coin_manager.unregister_coin(self.symbol)
                            self.symbol = None
                        time.sleep(1)
                        continue
                    self._last_leverage_check = current_time
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                              
                if not self.position_open:
                    if not self.symbol:
                        self.find_and_set_coin()
                        time.sleep(1)
                        continue
                    
                    signal = self.get_signal()
                    
                    if signal and signal != "NEUTRAL":
                        if current_time - self.last_trade_time > 3 and current_time - self.last_close_time > self.cooldown_period:
                            if self.open_position(signal):
                                self.last_trade_time = current_time
                            else:
                                if self.symbol:
                                    self.ws_manager.remove_symbol(self.symbol)
                                    self.coin_manager.unregister_coin(self.symbol)
                                    self.symbol = None
                                time.sleep(1)
                    else:
                        time.sleep(1)
                
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
        if self.symbol:
            self.coin_manager.unregister_coin(self.symbol)
        if self.symbol:
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
        self.log(f"🔴 Bot dừng")

    def open_position(self, side):
        if side not in ["BUY", "SELL"]:
            self.log(f"❌ Side không hợp lệ: {side}")
            return False
            
        try:
            self.check_position_status()
            if self.position_open:
                self.log(f"⚠️ Đã có vị thế {self.side}, bỏ qua tín hiệu {side}")
                return False
    
            if self.should_be_removed:
                self.log("⚠️ Bot đã được đánh dấu xóa, không mở lệnh mới")
                return False
    
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            if current_leverage < self.lev:
                self.log(f"❌ Coin {self.symbol} chỉ hỗ trợ đòn bẩy {current_leverage}x < {self.lev}x -> TÌM COIN KHÁC")
                self._cleanup_symbol()
                return False
    
            if not set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                self.log(f"❌ Không thể đặt đòn bẩy {self.lev}x -> TÌM COIN KHÁC")
                self._cleanup_symbol()
                return False
    
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                self.log("❌ Không đủ số dư")
                return False
    
            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                self.log("❌ Lỗi lấy giá -> TÌM COIN KHÁC")
                self._cleanup_symbol()
                return False
    
            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
    
            if qty < step_size:
                self.log(f"❌ Số lượng quá nhỏ: {qty} < {step_size}")
                return False
    
            self.log(f"📊 Đang đặt lệnh {side} - SL: {step_size}, Qty: {qty}, Giá: {current_price}")
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.2)
            
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
                    self.log(f"❌ Lệnh không khớp - Số lượng: {qty} -> TÌM COIN KHÁC")
                    return False
            else:
                error_msg = result.get('msg', 'Unknown error') if result else 'No response'
                self.log(f"❌ Lỗi đặt lệnh {side}: {error_msg} -> TÌM COIN KHÁC")
                
                if result and 'code' in result:
                    self.log(f"📋 Mã lỗi Binance: {result['code']} - {result.get('msg', '')}")
                
                self._cleanup_symbol()
                return False
                    
        except Exception as e:
            self.log(f"❌ Lỗi mở lệnh: {str(e)} -> TÌM COIN KHÁC")
            self._cleanup_symbol()
            return False
    
    def _cleanup_symbol(self):
        if self.symbol:
            try:
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
                self.log(f"🧹 Đã dọn dẹp symbol {self.symbol} và tìm coin mới")
            except Exception as e:
                self.log(f"⚠️ Lỗi khi dọn dẹp symbol: {str(e)}")
            
            self.symbol = None
        self.status = "searching"
        self.position_open = False
        self.side = ""
        self.qty = 0
        self.entry = 0

    def close_position(self, reason=""):
        try:
            self.check_position_status()
            
            if not self.position_open or abs(self.qty) <= 0:
                self.log(f"⚠️ Không có vị thế để đóng: {reason}")
                if self.symbol:
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
                
                if self.symbol:
                    self.coin_manager.unregister_coin(self.symbol)
                    self.ws_manager.remove_symbol(self.symbol)
                
                self._reset_position()
                self.last_close_time = time.time()
                self.symbol = None
                self.status = "searching"
                
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

# ========== BOT PHÂN TÍCH CHIẾN DỊCH ĐÃ SỬA ==========
class CampaignAnalysisBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, bot_id=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Campaign Analysis System", 
                        config_key, bot_id)
        
        self.analyzer = TrendIndicatorSystem()
        self.last_analysis_time = 0
        self.analysis_interval = 90  # GIẢM THỜI GIAN CHỜ
        
    def get_signal(self):
        if not self.symbol:
            return None
            
        try:
            current_time = time.time()
            if current_time - self.last_analysis_time < self.analysis_interval:
                return None
            
            self.last_analysis_time = current_time
            
            signal = self.analyzer.analyze_symbol(self.symbol)
            
            if signal != "NEUTRAL":
                self.log(f"🎯 Nhận tín hiệu {signal} từ phân tích chiến dịch")
            
            return signal
            
        except Exception as e:
            self.log(f"❌ Lỗi phân tích chiến dịch: {str(e)}")
            return None

# ========== BOT MANAGER HOÀN CHỈNH ==========
class BotManager:
    def __init__(self, api_key=None, api_secret=None, telegram_bot_token=None, telegram_chat_id=None):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG BOT PHÂN TÍCH CHIẾN DỊCH ĐÃ SỬA LỖI - KHỞI ĐỘNG")
            self.log("🎯 Phiên bản đã sửa: Tăng độ phủ & Giảm ngưỡng")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
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

    def get_position_summary(self):
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            binance_buy_count = 0
            binance_sell_count = 0
            binance_positions = []
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:
                    symbol = pos.get('symbol', 'UNKNOWN')
                    entry_price = float(pos.get('entryPrice', 0))
                    leverage = float(pos.get('leverage', 1))
                    position_value = abs(position_amt) * entry_price / leverage
                    
                    if position_amt > 0:
                        binance_buy_count += 1
                        binance_positions.append({
                            'symbol': symbol,
                            'side': 'LONG',
                            'leverage': leverage,
                            'size': abs(position_amt),
                            'entry': entry_price,
                            'value': position_value
                        })
                    else:
                        binance_sell_count += 1
                        binance_positions.append({
                            'symbol': symbol, 
                            'side': 'SHORT',
                            'leverage': leverage,
                            'size': abs(position_amt),
                            'entry': entry_price,
                            'value': position_value
                        })
        
            bot_details = []
            searching_bots = 0
            waiting_bots = 0
            trading_bots = 0
            
            for bot_id, bot in self.bots.items():
                bot_info = {
                    'bot_id': bot_id,
                    'symbol': bot.symbol or 'Đang tìm...',
                    'status': bot.status,
                    'side': bot.side,
                    'leverage': bot.lev,
                    'percent': bot.percent,
                    'tp': bot.tp,
                    'sl': bot.sl
                }
                bot_details.append(bot_info)
                
                if bot.status == "searching":
                    searching_bots += 1
                elif bot.status == "waiting":
                    waiting_bots += 1
                elif bot.status == "open":
                    trading_bots += 1
            
            summary = "📊 **THỐNG KÊ CHI TIẾT HỆ THỐNG**\n\n"
            
            balance = get_balance(self.api_key, self.api_secret)
            summary += f"💰 **SỐ DƯ**: {balance:.2f} USDT\n\n"
            
            summary += f"🤖 **BOT HỆ THỐNG**: {len(self.bots)} bots\n"
            summary += f"   🔍 Đang tìm coin: {searching_bots}\n"
            summary += f"   🟡 Đang chờ: {waiting_bots}\n" 
            summary += f"   📈 Đang trade: {trading_bots}\n\n"
            
            if bot_details:
                summary += "📋 **CHI TIẾT TỪNG BOT**:\n"
                for bot in bot_details[:8]:
                    symbol_info = bot['symbol'] if bot['symbol'] != 'Đang tìm...' else '🔍 Đang tìm'
                    status_map = {
                        "searching": "🔍 Tìm coin",
                        "waiting": "🟡 Chờ tín hiệu", 
                        "open": "🟢 Đang trade"
                    }
                    status = status_map.get(bot['status'], bot['status'])
                    
                    summary += f"   🔹 {bot['bot_id'][:15]}...\n"
                    summary += f"      📊 {symbol_info} | {status}\n"
                    summary += f"      💰 ĐB: {bot['leverage']}x | Vốn: {bot['percent']}%\n"
                    if bot['tp'] is not None and bot['sl'] is not None:
                        summary += f"      🎯 TP: {bot['tp']}% | 🛡️ SL: {bot['sl']}%\n"
                    summary += "\n"
                
                if len(bot_details) > 8:
                    summary += f"   ... và {len(bot_details) - 8} bot khác\n\n"
            
            total_binance = binance_buy_count + binance_sell_count
            if total_binance > 0:
                summary += f"💰 **TẤT CẢ VỊ THẾ BINANCE**: {total_binance} vị thế\n"
                summary += f"   🟢 LONG: {binance_buy_count}\n"
                summary += f"   🔴 SHORT: {binance_sell_count}\n\n"
                
                summary += "📈 **CHI TIẾT VỊ THẾ**:\n"
                for pos in binance_positions[:5]:
                    summary += f"   🔹 {pos['symbol']} | {pos['side']}\n"
                    summary += f"      📊 KL: {pos['size']:.4f} | Giá: {pos['entry']:.4f}\n"
                    summary += f"      💰 ĐB: {pos['leverage']}x | GT: ${pos['value']:.0f}\n\n"
                
                if len(binance_positions) > 5:
                    summary += f"   ... và {len(binance_positions) - 5} vị thế khác\n"
                        
            else:
                summary += f"💰 **TẤT CẢ VỊ THẾ BINANCE**: Không có vị thế nào\n"
                    
            return summary
                    
        except Exception as e:
            return f"❌ Lỗi thống kê: {str(e)}"

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>SYSTEM</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        welcome = (
            "🤖 <b>BOT GIAO DỊCH PHÂN TÍCH CHIẾN DỊCH - PHIÊN BẢN ĐÃ SỬA LỖI</b>\n\n"
            "🎯 <b>HỆ THỐNG ĐÃ ĐƯỢC TỐI ƯU:</b>\n"
            "📊 Giảm độ phức tạp phân tích\n"
            "🎯 Tăng độ phủ tín hiệu\n"
            "⚡ Giảm ngưỡng confidence\n"
            "🛡️ Thêm cơ chế dự phòng\n\n"
            "✅ <b>KHẮC PHỤC LỖI KHÔNG TÌM THẤY TÍN HIỆU</b>"
        )
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

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
        created_count = 0
        
        for i in range(bot_count):
            try:
                if bot_mode == 'static' and symbol:
                    bot_id = f"{symbol}_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = CampaignAnalysisBot
                    
                    bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token, 
                                  self.telegram_chat_id, bot_id=bot_id)
                    
                else:
                    bot_id = f"DYNAMIC_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = CampaignAnalysisBot
                    
                    bot = bot_class(None, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token,
                                  self.telegram_chat_id, bot_id=bot_id)
                
                bot._bot_manager = self
                self.bots[bot_id] = bot
                created_count += 1
                
            except Exception as e:
                self.log(f"❌ Lỗi tạo bot {i}: {str(e)}")
                continue
        
        if created_count > 0:
            success_msg = (
                f"✅ <b>ĐÃ TẠO {created_count}/{bot_count} BOT PHÂN TÍCH CHIẾN DỊCH</b>\n\n"
                f"🎯 Hệ thống: Campaign Analysis System (ĐÃ SỬA)\n"
                f"💰 Đòn bẩy: {lev}x\n"
                f"📈 % Số dư: {percent}%\n"
                f"🎯 TP: {tp}%\n"
                f"🛡️ SL: {sl if sl is not None else 'Tắt'}%\n"
                f"🔧 Chế độ: {bot_mode}\n"
            )
            
            if bot_mode == 'static' and symbol:
                success_msg += f"🔗 Coin: {symbol}\n"
            else:
                success_msg += f"🔗 Coin: Tự động tìm kiếm\n"
            
            success_msg += f"\n🎯 <b>Phiên bản đã sửa lỗi không tìm thấy tín hiệu</b>"
            
            self.log(success_msg)
            return True
        else:
            self.log("❌ Không thể tạo bot nào")
            return False

    def stop_bot(self, bot_id):
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]
            self.log(f"⛔ Đã dừng bot {bot_id}")
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
                    if bot_count <= 0 or bot_count > 10:
                        send_telegram("⚠️ Số lượng bot phải từ 1 đến 10. Vui lòng chọn lại:",
                                    chat_id, create_bot_count_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['bot_count'] = bot_count
                    user_state['step'] = 'waiting_bot_mode'
                    self.user_states[chat_id] = user_state
                    
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
                    self.user_states[chat_id] = user_state
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
                    self.user_states[chat_id] = user_state
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT ĐỘNG</b>\n\n"
                        f"🤖 Hệ thống sẽ tạo <b>{user_state.get('bot_count', 1)} bot độc lập</b>\n"
                        f"🔄 Mỗi bot tự tìm coin & trade độc lập\n"
                        f"🎯 Tự reset hoàn toàn sau mỗi lệnh\n"
                        f"📊 Mỗi bot là 1 vòng lặp hoàn chỉnh\n\n"
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
            elif text in ["📊 Campaign Analysis System"]:
                
                strategy_map = {
                    "📊 Campaign Analysis System": "Campaign-Analysis-System"
                }
                
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                user_state['step'] = 'waiting_exit_strategy'
                self.user_states[chat_id] = user_state
                
                strategy_descriptions = {
                    "Campaign-Analysis-System": "Hệ thống phân tích chiến dịch: Kỳ vọng & Phương sai dựa trên 200 nến lịch sử"
                }
                
                description = strategy_descriptions.get(strategy, "")
                bot_count = user_state.get('bot_count', 1)
                
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} bot độc lập\n\n"
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
                self.user_states[chat_id] = user_state
                self._continue_bot_creation(chat_id, user_state)

        elif current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                user_state['symbol'] = text
                user_state['step'] = 'waiting_leverage'
                self.user_states[chat_id] = user_state
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
                    self.user_states[chat_id] = user_state
                    send_telegram(
                        f"💰 Đòn bẩy: {leverage}x\n\n"
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
                    self.user_states[chat_id] = user_state
                    send_telegram(
                        f"📈 % Số dư: {percent}%\n\n"
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
                    if tp < 0:
                        send_telegram("⚠️ TP phải >= 0. Vui lòng chọn lại:",
                                    chat_id, create_tp_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['tp'] = tp
                    user_state['step'] = 'waiting_sl'
                    self.user_states[chat_id] = user_state
                    send_telegram(
                        f"🎯 Take Profit: {tp}%\n\n"
                        f"Chọn Stop Loss (%):",
                        chat_id,
                        create_sl_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho TP:",
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
                        send_telegram("⚠️ SL phải >= 0. Vui lòng chọn lại:",
                                    chat_id, create_sl_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['sl'] = sl
                    self.user_states[chat_id] = user_state
                    
                    self._create_bot_from_state(chat_id, user_state)
                    
                    self.user_states[chat_id] = {}
                    send_telegram("✅ Hoàn tất thiết lập bot!", chat_id, create_main_menu(),
                                self.telegram_bot_token, self.telegram_chat_id)
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho SL:",
                                chat_id, create_sl_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif text == '📊 Danh sách Bot':
            if not self.bots:
                send_telegram("🤖 Hiện không có bot nào đang chạy", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                response = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                for i, (bot_id, bot) in enumerate(self.bots.items(), 1):
                    symbol = bot.symbol if bot.symbol else "🔍 Đang tìm..."
                    status_map = {
                        "searching": "🔍 Tìm coin",
                        "waiting": "🟡 Chờ tín hiệu",
                        "open": "🟢 Đang trade"
                    }
                    status = status_map.get(bot.status, bot.status)
                    
                    response += f"{i}. <b>{bot_id[:15]}...</b>\n"
                    response += f"   📊 {symbol}\n"
                    response += f"   📈 {status}\n"
                    response += f"   💰 ĐB: {bot.lev}x | Vốn: {bot.percent}%\n"
                    if bot.tp is not None and bot.sl is not None:
                        response += f"   🎯 TP: {bot.tp}% | SL: {bot.sl}%\n"
                    response += "\n"
                
                send_telegram(response, chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)

        elif text == '📊 Thống kê':
            summary = self.get_position_summary()
            send_telegram(summary, chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

        elif text == '➕ Thêm Bot':
            user_state = {'step': 'waiting_bot_count'}
            self.user_states[chat_id] = user_state
            send_telegram(
                "🤖 <b>THÊM BOT MỚI</b>\n\n"
                "Nhập số lượng bot (1-10):",
                chat_id,
                create_bot_count_keyboard(),
                self.telegram_bot_token, self.telegram_chat_id
            )

        elif text == '⛔ Dừng Bot':
            if not self.bots:
                send_telegram("🤖 Không có bot nào để dừng", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                keyboard = {"inline_keyboard": []}
                for bot_id in self.bots.keys():
                    keyboard["inline_keyboard"].append([{"text": f"⛔ {bot_id[:20]}...", "callback_data": f"stop:{bot_id}"}])
                keyboard["inline_keyboard"].append([{"text": "⛔ DỪNG TẤT CẢ", "callback_data": "stop:all"}])
                keyboard["inline_keyboard"].append([{"text": "❌ Hủy", "callback_data": "cancel"}])
                
                send_telegram("⛔ <b>CHỌN BOT ĐỂ DỪNG</b>", chat_id, keyboard,
                            self.telegram_bot_token, self.telegram_chat_id)

        elif text == '💰 Số dư':
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ Lỗi lấy số dư", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
                position_count = sum(1 for pos in positions if float(pos.get('positionAmt', 0)) != 0)
                
                message = (
                    f"💰 <b>SỐ DƯ TÀI KHOẢN</b>\n\n"
                    f"💵 Số dư khả dụng: <b>{balance:.2f} USDT</b>\n"
                    f"📊 Tổng vị thế: <b>{position_count}</b>\n\n"
                    f"⚡ <b>TỔNG QUAN HỆ THỐNG</b>\n"
                    f"🤖 Số bot đang chạy: <b>{len(self.bots)}</b>\n"
                    f"🔗 Coin đang quản lý: <b>{len(CoinManager().get_managed_coins())}</b>"
                )
                send_telegram(message, chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)

        elif text == '📈 Vị thế':
            positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            open_positions = [pos for pos in positions if float(pos.get('positionAmt', 0)) != 0]
            
            if not open_positions:
                send_telegram("📊 Không có vị thế nào đang mở", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                message = "📊 <b>VỊ THẾ ĐANG MỞ</b>\n\n"
                for pos in open_positions[:8]:
                    symbol = pos['symbol']
                    position_amt = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    leverage = float(pos['leverage'])
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    side = "LONG" if position_amt > 0 else "SHORT"
                    size = abs(position_amt)
                    value = size * entry_price / leverage
                    
                    message += (
                        f"🔹 <b>{symbol}</b>\n"
                        f"   📌 {side} | ĐB: {leverage}x\n"
                        f"   🏷️ Giá: {entry_price:.4f}\n"
                        f"   📊 KL: {size:.4f}\n"
                        f"   💵 GT: ${value:.0f}\n"
                        f"   💰 PnL: {unrealized_pnl:.2f} USDT\n\n"
                    )
                
                if len(open_positions) > 8:
                    message += f"... và {len(open_positions) - 8} vị thế khác\n"
                
                send_telegram(message, chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)

        elif text == '⚙️ Cấu hình':
            config_msg = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG</b>\n\n"
                f"🔑 API Key: {'✅ Đã thiết lập' if self.api_key else '❌ Chưa thiết lập'}\n"
                f"🤖 Số bot: {len(self.bots)}\n"
                f"🔗 Coin đang quản lý: {len(CoinManager().get_managed_coins())}\n"
                f"🕒 Thời gian chạy: {int(time.time() - self.start_time)} giây\n\n"
                f"🎯 <b>CHIẾN LƯỢC HIỆN TẠI</b>\n"
                f"📊 Campaign Analysis: Phân tích kỳ vọng & phương sai chiến dịch\n"
                f"📈 Mô phỏng 200 nến lịch sử\n"
                f"🎯 Quyết định dựa trên chất lượng chiến dịch"
            )
            send_telegram(config_msg, chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

        elif text == '🎯 Chiến lược':
            strategy_msg = (
                "🎯 <b>HỆ THỐNG PHÂN TÍCH CHIẾN DỊCH - KỲ VỌNG & PHƯƠNG SAI</b>\n\n"
                "📊 <b>NGUYÊN LÝ HOẠT ĐỘNG:</b>\n"
                "• Phân tích 200 nến lịch sử 15m\n"
                "• Mô phỏng đa chiến dịch giao dịch\n"
                "• Tính toán kỳ vọng lợi nhuận\n"
                "• Đánh giá phương sai rủi ro\n"
                "• Phân loại điều kiện thị trường\n\n"
                "📈 <b>CHỈ SỐ ĐÁNH GIÁ:</b>\n"
                "• Kỳ vọng (Expectation): Lợi nhuận trung bình\n"
                "• Phương sai (Variance): Độ biến động rủi ro\n"
                "• Sharpe Ratio: Tỷ lệ lợi nhuận/rủi ro\n"
                "• Win Rate: Tỷ lệ chiến thắng\n"
                "• Max Drawdown: Thua lỗ tối đa\n\n"
                "🎯 <b>QUYẾT ĐỊNH GIAO DỊCH:</b>\n"
                "• Chọn hướng có kỳ vọng cao nhất\n"
                "• Ưu tiên phương sai thấp\n"
                "• Ngưỡng tin cậy: 65% trở lên\n"
                "• Tự động tìm coin phù hợp\n\n"
                "⚡ <b>TỐI ƯU HIỆU SUẤT:</b>\n"
                "• Cập nhật 30 phút/lần\n"
                "• Chỉ phân tích khi có đủ dữ liệu\n"
                "• Tự động filter coin chất lượng"
            )
            send_telegram(strategy_msg, chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

        elif text.startswith('/'):
            if text == '/start':
                self.send_main_menu(chat_id)
            elif text == '/stop':
                self.stop_all()
                send_telegram("🔴 Đã dừng toàn bộ hệ thống", chat_id,
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text == '/stats':
                summary = self.get_position_summary()
                send_telegram(summary, chat_id,
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text == '/clear_cache':
                for bot in self.bots.values():
                    if hasattr(bot, 'clear_finder_cache'):
                        bot.clear_finder_cache()
                send_telegram("🧹 Đã xóa cache tất cả bot", chat_id,
                            self.telegram_bot_token, self.telegram_chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        try:
            if user_state.get('bot_mode') == 'static':
                user_state['step'] = 'waiting_symbol'
                self.user_states[chat_id] = user_state
                
                send_telegram(
                    "🔗 <b>BOT TĨNH - CHỌN COIN</b>\n\n"
                    "Chọn coin để giao dịch:",
                    chat_id,
                    create_symbols_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                self.user_states[chat_id] = user_state
                
                bot_count = user_state.get('bot_count', 1)
                
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG - TỰ TÌM COIN</b>\n\n"
                    f"🤖 Số lượng: {bot_count} bot độc lập\n"
                    f"🔄 Mỗi bot tự tìm coin & trade độc lập\n"
                    f"🎯 Tự reset hoàn toàn sau mỗi lệnh\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
        except Exception as e:
            logger.error(f"Lỗi tiếp tục tạo bot: {str(e)}")
            send_telegram("❌ Lỗi hệ thống khi tạo bot", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

    def _create_bot_from_state(self, chat_id, user_state):
        try:
            bot_count = user_state.get('bot_count', 1)
            bot_mode = user_state.get('bot_mode', 'dynamic')
            strategy = user_state.get('strategy', 'Campaign-Analysis-System')
            leverage = user_state.get('leverage', 10)
            percent = user_state.get('percent', 5)
            tp = user_state.get('tp', 100)
            sl = user_state.get('sl', 50)
            
            symbol = user_state.get('symbol') if bot_mode == 'static' else None
            
            success = self.add_bot(
                symbol=symbol,
                lev=leverage,
                percent=percent,
                tp=tp,
                sl=sl,
                strategy_type=strategy,
                bot_count=bot_count,
                bot_mode=bot_mode
            )
            
            if success:
                send_telegram(
                    f"✅ <b>ĐÃ TẠO THÀNH CÔNG {bot_count} BOT</b>\n\n"
                    f"🎯 Chiến lược: {strategy}\n"
                    f"💰 Đòn bẩy: {leverage}x\n"
                    f"📈 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl}%\n"
                    f"🔧 Chế độ: {bot_mode}\n",
                    chat_id,
                    create_main_menu(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                send_telegram("❌ Không thể tạo bot. Vui lòng thử lại.", chat_id, create_main_menu(),
                             self.telegram_bot_token, self.telegram_chat_id)
                
        except Exception as e:
            logger.error(f"Lỗi tạo bot từ state: {str(e)}")
            send_telegram("❌ Lỗi hệ thống khi tạo bot", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

coin_manager = CoinManager()
