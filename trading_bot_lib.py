# trading_bot_volume_fixed_part1.py - PHẦN 1: CORE SYSTEM & BASE BOT
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
        return
    
    chat_id = chat_id or default_chat_id
    if not chat_id:
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
    except Exception:
        pass

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
            [{"text": "📊 Volume & MACD System"}],
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

# ========== HỆ THỐNG PHÂN TÍCH VOLUME, MACD, RSI & EMA ==========
class VolumeMACDStrategy:
    """HỆ THỐNG PHÂN TÍCH DỰA TRÊN VOLUME, MACD, RSI & EMA"""
    
    def __init__(self):
        self.volume_threshold = 2
        self.volume_decrease_threshold = 0.5
        self.small_body_threshold = 0.1
        
    def get_klines(self, symbol, interval, limit):
        """Lấy dữ liệu nến từ Binance"""
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            return binance_api_request(url, params=params)
        except Exception as e:
            return None
    
    def calculate_ema(self, prices, period):
        """Tính EMA (Exponential Moving Average)"""
        if len(prices) < period:
            return None
        
        ema_values = []
        multiplier = 2 / (period + 1)
        
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Tính MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow_period:
            return None, None, None
        
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        if not ema_fast or not ema_slow:
            return None, None, None
        
        min_length = min(len(ema_fast), len(ema_slow))
        ema_fast = ema_fast[-min_length:]
        ema_slow = ema_slow[-min_length:]
        prices = prices[-min_length:]
        
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(ema_fast))]
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        if not signal_line:
            return None, None, None
        
        histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, prices, period=14):
        """Tính RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        rsi_values = []
        for i in range(period, len(gains)):
            avg_gain = sum(gains[i-period:i]) / period
            avg_loss = sum(losses[i-period:i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    
    def is_doji(self, open_price, high, low, close):
        """Kiểm tra nến doji (thân nến nhỏ)"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return False
        
        body_ratio = body_size / total_range
        return body_ratio < self.small_body_threshold
    
    def analyze_volume_macd(self, symbol):
        """PHÂN TÍCH VOLUME, MACD, RSI & EMA THEO 3 KHUNG 1m, 5m, 15m"""
        try:
            intervals = ['1m', '5m', '15m']
            signals = []
            
            for interval in intervals:
                klines = self.get_klines(symbol, interval, 100)
                if not klines or len(klines) < 50:
                    continue
                
                current_candle = klines[-2]
                prev_candles = klines[-30:-2]
                
                open_price = float(current_candle[1])
                close_price = float(current_candle[4])
                high_price = float(current_candle[2])
                low_price = float(current_candle[3])
                current_volume = float(current_candle[5])
                
                volumes = [float(candle[5]) for candle in prev_candles]
                avg_volume = np.mean(volumes) if volumes else current_volume
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                volume_increase = volume_ratio > self.volume_threshold
                volume_decrease = volume_ratio < self.volume_decrease_threshold
                
                is_green = close_price > open_price
                is_red = close_price < open_price
                is_doji_candle = self.is_doji(open_price, high_price, low_price, close_price)
                
                close_prices = [float(candle[4]) for candle in klines]
                
                macd_line, signal_line, histogram = self.calculate_macd(close_prices)
                rsi_values = self.calculate_rsi(close_prices)
                ema_20 = self.calculate_ema(close_prices, 20)
                
                macd_bullish = False
                macd_bearish = False
                rsi_overbought = False
                rsi_oversold = False
                price_above_ema = False
                price_below_ema = False
                
                if macd_line and signal_line and len(macd_line) > 1 and len(signal_line) > 1:
                    macd_bullish = macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]
                    macd_bearish = macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]
                
                if rsi_values and len(rsi_values) > 0:
                    current_rsi = rsi_values[-1]
                    rsi_overbought = current_rsi > 75
                    rsi_oversold = current_rsi < 25
                
                if ema_20 and len(ema_20) > 0:
                    current_ema_20 = ema_20[-1]
                    price_above_ema = close_price > current_ema_20
                    price_below_ema = close_price < current_ema_20
                
                signal = "NEUTRAL"
                
                if volume_increase and macd_bullish and is_green:
                    signal = "BUY"
                elif volume_increase and macd_bearish and is_red:
                    signal = "SELL"
                elif volume_decrease and is_doji_candle:
                    signal = "BUY"
                
                signals.append((interval, signal, {
                    'volume_ratio': volume_ratio,
                    'macd_bullish': macd_bullish,
                    'macd_bearish': macd_bearish,
                    'rsi': rsi_values[-1] if rsi_values else 50,
                    'price_above_ema': price_above_ema,
                    'price_below_ema': price_below_ema
                }))
            
            if not signals:
                return "NEUTRAL"
                
            buy_count = sum(1 for _, s, _ in signals if s == "BUY")
            sell_count = sum(1 for _, s, _ in signals if s == "SELL")
            
            if buy_count > sell_count:
                final_signal = "BUY"
            elif sell_count > buy_count:
                final_signal = "SELL"
            else:
                final_signal = "NEUTRAL"
            
            return final_signal
            
        except Exception as e:
            return "NEUTRAL"
    
    def check_exit_signal(self, symbol, current_side):
        """KIỂM TRA TÍN HIỆU THOÁT LỆNH KHI ĐANG CÓ VỊ THẾ"""
        try:
            if not current_side:
                return False
                
            klines = self.get_klines(symbol, '5m', 50)
            if not klines or len(klines) < 30:
                return False
            
            current_candle = klines[-2]
            open_price = float(current_candle[1])
            close_price = float(current_candle[4])
            high_price = float(current_candle[2])
            low_price = float(current_candle[3])
            current_volume = float(current_candle[5])
            
            prev_candles = klines[-30:-2]
            volumes = [float(candle[5]) for candle in prev_candles]
            avg_volume = np.mean(volumes) if volumes else current_volume
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_decrease = volume_ratio < self.volume_decrease_threshold
            
            close_prices = [float(candle[4]) for candle in klines]
            
            rsi_values = self.calculate_rsi(close_prices)
            ema_20 = self.calculate_ema(close_prices, 20)
            
            rsi_overbought = False
            rsi_oversold = False
            price_above_ema = False
            price_below_ema = False
            
            if rsi_values and len(rsi_values) > 0:
                current_rsi = rsi_values[-1]
                rsi_overbought = current_rsi > 85
                rsi_oversold = current_rsi < 15
            
            if ema_20 and len(ema_20) > 0:
                current_ema_20 = ema_20[-1]
                price_above_ema = close_price > current_ema_20
                price_below_ema = close_price < current_ema_20
            
            if current_side == "BUY":
                if (rsi_overbought or (rsi_oversold and price_below_ema)) and volume_decrease:
                    return True
            
            elif current_side == "SELL":
                if (rsi_oversold or (rsi_overbought and price_above_ema)) and volume_decrease:
                    return True
            
            return False
            
        except Exception as e:
            return False

# ========== SMART COIN FINDER NÂNG CẤP ==========
class SmartCoinFinder:
    """TÌM COIN THÔNG MINH - LỌC ĐÒN BẨY TRƯỚC, VOLUME SAU"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.analyzer = VolumeMACDStrategy()
        self.leverage_cache = {}
        self.volume_cache = {}
        self.qualified_symbols_cache = {}
        self.cache_timeout = 300
        self.last_cache_update = 0
        
    def get_24h_volume(self, symbol):
        """Lấy volume giao dịch 24h của symbol"""
        try:
            if symbol in self.volume_cache:
                return self.volume_cache[symbol]
                
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            params = {'symbol': symbol.upper()}
            data = binance_api_request(url, params=params)
            if data and 'volume' in data:
                volume = float(data['volume'])
                self.volume_cache[symbol] = volume
                return volume
        except Exception:
            pass
        return 0

    def get_pre_filtered_symbols(self, target_leverage):
        """LẤY DANH SÁCH COIN ĐÃ LỌC THEO ĐÒN BẨY TRƯỚC"""
        try:
            current_time = time.time()
            
            cache_key = f"lev_{target_leverage}"
            if (cache_key in self.qualified_symbols_cache and 
                self.qualified_symbols_cache[cache_key] and
                current_time - self.last_cache_update < self.cache_timeout):
                return self.qualified_symbols_cache[cache_key]
            
            all_symbols = get_all_usdt_pairs(limit=600)
            if not all_symbols:
                if cache_key in self.qualified_symbols_cache:
                    return self.qualified_symbols_cache[cache_key]
                return []
            
            qualified_by_leverage = []
            
            def check_leverage_only(symbol):
                try:
                    max_leverage = self.get_symbol_leverage(symbol)
                    return symbol if max_leverage >= target_leverage else None
                except:
                    return None
            
            with ThreadPoolExecutor(max_workers=15) as executor:
                results = list(executor.map(check_leverage_only, all_symbols))
            
            qualified_by_leverage = [symbol for symbol in results if symbol is not None]
            
            final_qualified = []
            for symbol in qualified_by_leverage:
                try:
                    volume_24h = self.get_24h_volume(symbol)
                    if 1000000 <= volume_24h <= 1000000000:
                        final_qualified.append(symbol)
                except:
                    continue
            
            self.qualified_symbols_cache[cache_key] = final_qualified
            self.last_cache_update = current_time
            
            return final_qualified
            
        except Exception as e:
            logger.error(f"Lỗi lọc coin theo đòn bẩy: {str(e)}")
            if cache_key in self.qualified_symbols_cache:
                return self.qualified_symbols_cache[cache_key]
            return []

    def get_symbol_leverage(self, symbol):
        if symbol in self.leverage_cache:
            return self.leverage_cache[symbol]
        
        max_leverage = get_max_leverage(symbol, self.api_key, self.api_secret)
        self.leverage_cache[symbol] = max_leverage
        return max_leverage

    def find_coin_by_direction(self, target_direction, target_leverage, excluded_symbols=None):
        """TÌM 1 COIN DUY NHẤT - LỌC ĐÒN BẨY TRƯỚC, TÍN HIỆU SAU"""
        try:
            if excluded_symbols is None:
                excluded_symbols = set()
            
            qualified_symbols = self.get_pre_filtered_symbols(target_leverage)
            if not qualified_symbols:
                return None
            
            available_symbols = [s for s in qualified_symbols if s not in excluded_symbols]
            
            if not available_symbols:
                return None
            
            random.shuffle(available_symbols)
            symbols_to_check = available_symbols[:30]
            
            for symbol in symbols_to_check:
                try:
                    signal = self.analyzer.analyze_volume_macd(symbol)
                    
                    if signal == target_direction:
                        max_leverage = self.get_symbol_leverage(symbol)
                        volume_24h = self.get_24h_volume(symbol)
                        
                        return {
                            'symbol': symbol,
                            'direction': target_direction,
                            'max_leverage': max_leverage,
                            'volume_24h': volume_24h,
                            'score': 0.8,
                            'qualified': True
                        }
                        
                except Exception:
                    continue
            
            return None
                
        except Exception as e:
            logger.error(f"Lỗi tìm coin: {str(e)}")
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
                    if response.status == 401:
                        return None
                    if response.status == 429:
                        time.sleep(2 ** attempt)
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return None
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            time.sleep(1)
    
    return None

def get_all_usdt_pairs(limit=600):
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return []
        
        usdt_pairs = []
        for symbol_info in data.get('symbols', []):
            symbol = symbol_info.get('symbol', '')
            if symbol.endswith('USDT') and symbol_info.get('status') == 'TRADING':
                usdt_pairs.append(symbol)
        
        return usdt_pairs[:limit] if limit else usdt_pairs
        
    except Exception as e:
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
    except Exception:
        pass
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
            except Exception:
                pass
                
        def on_error(ws, error):
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
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
        
    def _reconnect(self, symbol, callback):
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception:
                    pass
                del self.connections[symbol]
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()

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
        self.current_price = 0
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
        
        # Thêm biến quản lý nhồi lệnh
        self.entry_base = 0
        self.average_down_count = 0
        self.last_average_down_time = 0
        self.average_down_cooldown = 60
        
        self.check_position_status()
        if self.symbol:
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

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

    def _handle_price_update(self, price):
        if self._stop or not price or price <= 0:
            return
        try:
            price_float = float(price)
            self.current_price = price_float
            self.prices.append(price_float)
            if len(self.prices) > 100:
                self.prices = self.prices[-100:]
        except Exception:
            pass

    def get_signal(self):
        raise NotImplementedError("Phương thức get_signal cần được triển khai")

    def get_target_direction(self):
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            buy_count = 0
            sell_count = 0
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:
                    if position_amt > 0:
                        buy_count += 1
                    else:
                        sell_count += 1
            
            total = buy_count + sell_count
            
            if total == 0:
                direction = "BUY" if random.random() > 0.5 else "SELL"
                return direction
            
            if buy_count > sell_count:
                return "SELL"
            elif sell_count > buy_count:
                return "BUY"
            else:
                direction = "BUY" if random.random() > 0.5 else "SELL"
                return direction
                
        except Exception as e:
            return "BUY" if random.random() > 0.5 else "SELL"

    def verify_leverage_and_switch(self):
        if not self.symbol or not self.position_open:
            return True
            
        try:
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            
            if current_leverage < self.lev:
                if self.position_open:
                    self.close_position(f"Đòn bẩy không đủ ({current_leverage}x < {self.lev}x)")
                
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
                self.symbol = None
                self.status = "searching"
                return False
                
            return True
            
        except Exception as e:
            return True

    def find_and_set_coin(self):
        try:
            self.current_target_direction = self.get_target_direction()
            managed_coins = self.coin_manager.get_managed_coins()
            excluded_symbols = set(managed_coins.keys())
            
            coin_data = self.coin_finder.find_coin_by_direction(
                self.current_target_direction, 
                self.lev,
                excluded_symbols
            )
        
            if coin_data is None:
                return False
                
            if not coin_data.get('qualified', False):
                return False
            
            new_symbol = coin_data['symbol']
            max_leverage = coin_data.get('max_leverage', 100)
            
            if max_leverage < self.lev:
                return False
            
            if self._register_coin_with_retry(new_symbol):
                if self.symbol:
                    self.ws_manager.remove_symbol(self.symbol)
                    self.coin_manager.unregister_coin(self.symbol)
                
                self.symbol = new_symbol
                self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
                
                self.status = "waiting"
                return True
            else:
                return False
                
        except Exception as e:
            return False

    def get_actual_pnl(self):
        """Lấy PnL thực tế từ Binance"""
        try:
            positions = get_positions(self.symbol, self.api_key, self.api_secret)
            if not positions:
                return 0, 0
                
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    position_amt = float(pos.get('positionAmt', 0))
                    leverage = float(pos.get('leverage', 1))
                    
                    if position_amt == 0:
                        return 0, 0
                        
                    # Tính ROI thực tế
                    position_size = abs(position_amt) * entry_price
                    invested = position_size / leverage
                    roi = (unrealized_pnl / invested) * 100 if invested > 0 else 0
                    
                    return unrealized_pnl, roi
                    
            return 0, 0
        except Exception as e:
            return 0, 0

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
                        
                        # Cập nhật PnL thực tế
                        unrealized_pnl, roi = self.get_actual_pnl()
                        self.log(f"📊 Vị thế thực tế: {self.side} {abs(self.qty):.4f} | Entry: {self.entry:.4f} | PnL: {unrealized_pnl:.2f} USDT")
                        break
                    else:
                        position_found = True
                        self._reset_position()
                        break
            
            if not position_found:
                self._reset_position()
                
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"💥 Lỗi check position status: {str(e)}")
                self.last_error_log_time = time.time()

     def check_averaging_down(self):
        if not self.position_open or self.entry_base <= 0:
            return False
            
        try:
            current_price = self.current_price or get_current_price(self.symbol)
            if current_price <= 0:
                return False
            
            if self.side == "BUY":
                profit = (current_price - self.entry_base) * abs(self.qty)
            else:
                profit = (self.entry_base - current_price) * abs(self.qty)
                
            invested = self.entry_base * abs(self.qty) / self.lev
            if invested <= 0:
                return False
                
            roi = (profit / invested) * 100
            
            current_time = time.time()
            
            # CẤP SỐ CỘNG: Lần 1: -100%, Lần 2: -200%, Lần 3: -300%, ...
            required_roi_threshold = -100 * (self.average_down_count + 1)
            
            if (roi <= required_roi_threshold and 
                current_time - self.last_average_down_time > self.average_down_cooldown):
                
                self.log(f"📉 Đạt {required_roi_threshold}% ROI từ mốc neo, thực hiện nhồi lệnh lần {self.average_down_count + 1}. ROI: {roi:.2f}%")
                return self.average_down()
                
            return False
            
        except Exception as e:
            return False

    def average_down(self):
        try:
            if not self.position_open:
                return False
                
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                return False

            current_price = self.current_price or get_current_price(self.symbol)
            if current_price <= 0:
                return False

            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            additional_qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                additional_qty = math.floor(additional_qty / step_size) * step_size
                additional_qty = round(additional_qty, 8)

            if additional_qty < step_size:
                return False

            result = place_order(self.symbol, self.side, additional_qty, self.api_key, self.api_secret)

            if result and 'orderId' in result:
                executed_qty = float(result.get('executedQty', 0))
                avg_price = float(result.get('avgPrice', current_price))

                if executed_qty > 0:
                    total_qty = abs(self.qty) + executed_qty
                    self.entry = (self.entry * abs(self.qty) + avg_price * executed_qty) / total_qty
                    self.qty = total_qty if self.side == "BUY" else -total_qty
                    
                    # GIỮ NGUYÊN entry_base (mốc neo ban đầu)
                    self.average_down_count += 1
                    self.last_average_down_time = time.time()

                    message = (
                        f"📈 <b>ĐÃ NHỒI LỆNH LẦN {self.average_down_count} - {self.symbol}</b>\n"
                        f"📌 Hướng: {self.side}\n"
                        f"🏷️ Giá vào ban đầu: {self.entry_base:.4f}\n"
                        f"🏷️ Giá trung bình mới: {self.entry:.4f}\n"
                        f"📊 Khối lượng thêm: {executed_qty:.4f}\n"
                        f"📊 Tổng khối lượng: {total_qty:.4f}\n"
                        f"💵 Giá trị nhồi: {executed_qty * avg_price:.2f} USDT\n"
                        f"💰 Đòn bẩy: {self.lev}x\n"
                        f"🎯 Ngưỡng nhồi tiếp theo: {-100 * (self.average_down_count + 1)}%"
                    )
                    self.log(message)
                    return True
                else:
                    return False
            else:
                return False
                
        except Exception as e:
            return False

    def _reset_position(self):
        self.position_open = False
        self.status = "searching" if not self.symbol else "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self._close_attempted = False
        self._last_close_attempt = 0
        self.entry_base = 0
        self.average_down_count = 0  # Reset số lần nhồi về 0

    def _force_reset(self):
        """Reset mạnh tay, không phụ thuộc vào trạng thái hiện tại"""
        if self.symbol:
            self.coin_manager.unregister_coin(self.symbol)
            self.ws_manager.remove_symbol(self.symbol)
        
        self.position_open = False
        self.status = "searching"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.entry_base = 0
        self.average_down_count = 0  # Reset số lần nhồi về 0
        self._close_attempted = False
        self.symbol = None

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
                    self.last_position_check = time.time()
                              
                if not self.position_open:
                    if not self.symbol:
                        if current_time - self.last_find_time > self.find_interval:
                            self.log("🔍 Đang tìm coin mới...")
                            success = self.find_and_set_coin()
                            if success:
                                self.log(f"✅ Đã tìm thấy coin: {self.symbol}")
                            else:
                                self.log("❌ Không tìm thấy coin phù hợp")
                            self.last_find_time = current_time
                        time.sleep(1)
                        continue
                    
                    # THÊM: Luôn gọi get_signal, không check time
                    signal = self.get_signal()
                    
                    if signal and signal != "NEUTRAL":
                        self.log(f"🚀 Nhận được tín hiệu {signal}, chuẩn bị mở lệnh...")
                        if current_time - self.last_trade_time > 3 and current_time - self.last_close_time > self.cooldown_period:
                            if self.open_position(signal):
                                self.last_trade_time = current_time
                            else:
                                self.log("❌ Không thể mở lệnh, reset symbol...")
                                self._cleanup_symbol()
                        else:
                            self.log(f"⏳ Đang trong thời gian chờ giữa các lệnh")
                    else:
                        # THÊM: Log trạng thái chờ
                        if current_time - getattr(self, 'last_analysis_time', 0) > 30:
                            self.log("🟡 Đang chờ tín hiệu từ Volume MACD...")
                            self.last_analysis_time = current_time
                        time.sleep(1)
                
                else:
                    # LUÔN check TP/SL và averaging_down, không phụ thuộc vào SL
                    self.check_averaging_down()
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
            return False
            
        try:
            self.check_position_status()
            if self.position_open:
                return False
    
            if self.should_be_removed:
                return False
    
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            if current_leverage < self.lev:
                self._cleanup_symbol()
                return False
    
            volume_24h = self.coin_finder.get_24h_volume(self.symbol)
            if volume_24h < 5000000 or volume_24h > 1000000000:
                self._cleanup_symbol()
                return False
    
            if not set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                self._cleanup_symbol()
                return False
    
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                return False
    
            current_price = self.current_price or get_current_price(self.symbol)
            if current_price <= 0:
                self._cleanup_symbol()
                return False
    
            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
    
            if qty < step_size:
                return False
    
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.5)
            
            max_retries = 3
            for attempt in range(max_retries):
                result = place_order(self.symbol, side, qty, self.api_key, self.api_secret)
                
                if result and 'orderId' in result:
                    executed_qty = float(result.get('executedQty', 0))
                    avg_price = float(result.get('avgPrice', current_price))
                    
                    if executed_qty > 0:
                        self.entry = avg_price
                        self.entry_base = avg_price
                        self.average_down_count = 0
                        self.side = side
                        self.qty = executed_qty if side == "BUY" else -executed_qty
                        self.position_open = True
                        self.status = "open"
                        self.position_open_time = time.time() 
                        
                        message = (
                            f"✅ <b>ĐÃ MỞ VỊ THẾ {self.symbol}</b>\n"
                            f"🤖 Chiến lược: {self.strategy_name}\n"
                            f"📌 Hướng: {side}\n"
                            f"🏷️ Giá vào: {self.entry:.4f}\n"
                            f"📊 Khối lượng: {executed_qty:.4f}\n"
                            f"💵 Giá trị: {executed_qty * self.entry:.2f} USDT\n"
                            f"💰 Đòn bẩy: {self.lev}x\n"
                            f"🎯 TP: {self.tp}% | 🛡️ SL: {'Tắt' if self.sl == 0 else f'{self.sl}%'}"
                        )
                        self.log(message)
                        return True
                    else:
                        time.sleep(1)
                else:
                    time.sleep(1)
            
            self._cleanup_symbol()
            return False
                    
        except Exception as e:
            self._cleanup_symbol()
            return False
    
    def _cleanup_symbol(self):
        if self.symbol:
            try:
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
            except Exception:
                pass
            
            self.symbol = None
        self.status = "searching"
        self.position_open = False
        self.side = ""
        self.qty = 0
        self.entry = 0
        
    def close_position(self, reason=""):
        try:
            current_time = time.time()
            if self._close_attempted and current_time - self._last_close_attempt < 30:
                return False
            
            self._close_attempted = True
            self._last_close_attempt = current_time

            # Lấy vị thế THỰC TẾ từ Binance
            positions = get_positions(self.symbol, self.api_key, self.api_secret)
            actual_position_amt = 0
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    actual_position_amt = float(pos.get('positionAmt', 0))
                    break
            
            # Nếu không có vị thế thực tế, vẫn reset
            if actual_position_amt == 0:
                self._force_reset()
                self.log(f"🔄 Reset trạng thái (không có vị thế thực tế): {reason}")
                return True
                
            close_side = "SELL" if actual_position_amt > 0 else "BUY"
            close_qty = abs(actual_position_amt)
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.5)
            
            result = place_order(self.symbol, close_side, close_qty, self.api_key, self.api_secret)
            
            if result and 'orderId' in result:
                # Lấy PnL thực tế cuối cùng
                unrealized_pnl, roi = self.get_actual_pnl()
                current_price = self.current_price or get_current_price(self.symbol)
                
                message = (
                    f"⛔ <b>ĐÃ ĐÓNG VỊ THẾ {self.symbol}</b>\n"
                    f"🤖 Chiến lược: {self.strategy_name}\n"
                    f"📌 Lý do: {reason}\n"
                    f"🏷️ Giá ra: {current_price:.4f}\n"
                    f"📊 Khối lượng: {close_qty:.4f}\n"
                    f"💰 PnL thực tế: {unrealized_pnl:.2f} USDT (ROI: {roi:.2f}%)"
                )
                self.log(message)
                
                self._force_reset()
                self.last_close_time = time.time()
                
                time.sleep(2)
                self.check_position_status()
                
                return True
            else:
                self._close_attempted = False
                return False
                
        except Exception as e:
            self._close_attempted = False
            self.log(f"💥 Lỗi khi đóng lệnh: {str(e)}")
            return False

    def check_tp_sl(self):
        if self._close_attempted:
            return
        
        try:
            # Lấy vị thế THỰC TẾ từ Binance
            positions = get_positions(self.symbol, self.api_key, self.api_secret)
            if not positions:
                return
                
            current_position = None
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    if abs(position_amt) > 0:
                        current_position = pos
                        break
            
            if not current_position:
                return
                
            # Lấy dữ liệu THỰC TẾ từ Binance
            position_amt = float(current_position.get('positionAmt', 0))
            entry_price = float(current_position.get('entryPrice', 0))
            leverage = float(current_position.get('leverage', 1))
            
            if position_amt == 0 or entry_price <= 0:
                return
            
            # Sử dụng giá hiện tại từ WebSocket
            current_price = self.current_price
            if current_price <= 0:
                current_price = get_current_price(self.symbol)
            if current_price <= 0:
                return
            
            # Tính toán PnL THỰC TẾ
            if position_amt > 0:  # LONG position
                price_diff = current_price - entry_price
            else:  # SHORT position  
                price_diff = entry_price - current_price
                
            # Tính ROI dựa trên vốn thực tế
            position_size = abs(position_amt) * entry_price
            invested = position_size / leverage
            unrealized_pnl = price_diff * abs(position_amt)
            
            if invested <= 0:
                return
                
            roi = (unrealized_pnl / invested) * 100

            # Log theo dõi
            if abs(roi) > max(self.tp, self.sl if self.sl > 0 else 0) * 0.7:
                self.log(f"📊 Theo dõi PnL: ROI={roi:.2f}%, Giá vào={entry_price:.4f}, Giá hiện tại={current_price:.4f}")

            # Kiểm tra TP/SL
            if self.tp is not None and roi >= self.tp:
                self.log(f"🎯 ĐẠT TP: {roi:.2f}% >= {self.tp}%")
                self.close_position(f"✅ Đạt TP {self.tp}% (ROI thực tế: {roi:.2f}%)")
            elif self.sl is not None and self.sl > 0 and roi <= -self.sl:
                self.log(f"🛑 ĐẠT SL: {roi:.2f}% <= -{self.sl}%")
                self.close_position(f"❌ Đạt SL {self.sl}% (ROI thực tế: {roi:.2f}%)")
                
        except Exception as e:
            self.log(f"💥 Lỗi check_tp_sl thực tế: {str(e)}")

# ========== BOT VOLUME & MACD ==========
class VolumeMACDBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, bot_id=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Volume MACD System", 
                        config_key, bot_id)
        
        self.analyzer = VolumeMACDStrategy()
        self.last_analysis_time = 0
        self.analysis_interval = 10  # Giảm từ 60s xuống 10s
        
    def get_signal(self):
        if not self.symbol:
            self.log("❌ Không có symbol để phân tích")
            return None
            
        try:
            current_time = time.time()
            if current_time - self.last_analysis_time < self.analysis_interval:
                return None
            
            self.last_analysis_time = current_time
            
            self.log(f"🔍 Đang phân tích tín hiệu cho {self.symbol}")
            signal = self.analyzer.analyze_volume_macd(self.symbol)
            
            # THÊM LOG CHI TIẾT
            if signal == "NEUTRAL":
                self.log(f"⚪ Không có tín hiệu rõ ràng từ Volume MACD")
            else:
                self.log(f"🎯 PHÁT HIỆN TÍN HIỆU: {signal} cho {self.symbol}")
                
            return signal
            
        except Exception as e:
            self.log(f"💥 Lỗi phân tích tín hiệu: {str(e)}")
            return None

# trading_bot_volume_fixed_part2.py - PHẦN 2: BOT MANAGER & SYSTEM LAUNCHER
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
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)

    def _verify_api_connection(self):
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None:
            self.log("❌ LỖI KẾT NỐI BINANCE - Kiểm tra API Key!")

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
                    
                if binance_buy_count > binance_sell_count:
                    summary += f"\n⚖️ **ĐỀ XUẤT**: Nhiều LONG hơn → ƯU TIÊN TÌM SHORT"
                elif binance_sell_count > binance_buy_count:
                    summary += f"\n⚖️ **ĐỀ XUẤT**: Nhiều SHORT hơn → ƯU TIÊN TÌM LONG"
                else:
                    summary += f"\n⚖️ **TRẠNG THÁI**: Cân bằng tốt"
                        
            else:
                summary += f"💰 **TẤT CẢ VỊ THẾ BINANCE**: Không có vị thế nào\n"
                    
            return summary
                    
        except Exception as e:
            return f"❌ Lỗi thống kê: {str(e)}"

    def log(self, message):
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>SYSTEM</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES ĐA LUỒNG</b>\n\n🎯 <b>HỆ THỐNG VOLUME, MACD, RSI & EMA</b>"
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, bot_count=1, **kwargs):
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            return False
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None:
            return False
        
        bot_mode = kwargs.get('bot_mode', 'static')
        created_count = 0
        
        for i in range(bot_count):
            try:
                if bot_mode == 'static' and symbol:
                    bot_id = f"{symbol}_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = VolumeMACDBot
                    
                    if not bot_class:
                        continue
                    
                    bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token, 
                                  self.telegram_chat_id, bot_id=bot_id)
                    
                else:
                    bot_id = f"DYNAMIC_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = VolumeMACDBot
                    
                    if not bot_class:
                        continue
                    
                    bot = bot_class(None, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token,
                                  self.telegram_chat_id, bot_id=bot_id)
                
                bot._bot_manager = self
                self.bots[bot_id] = bot
                created_count += 1
                
            except Exception as e:
                continue
        
        if created_count > 0:
            success_msg = (
                f"✅ <b>ĐÃ TẠO {created_count}/{bot_count} BOT VOLUME & MACD</b>\n\n"
                f"🎯 Hệ thống: Volume, MACD, RSI & EMA\n"
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
            
            success_msg += f"\n🎯 <b>Mỗi bot là 1 vòng lặp độc lập</b>"
            
            self.log(success_msg)
            return True
        else:
            return False

    def stop_bot(self, bot_id):
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]
            return True
        return False

    def stop_all(self):
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.running = False

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
                    time.sleep(60)
                else:
                    time.sleep(10)
                
            except Exception as e:
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
            elif text in ["📊 Volume & MACD System"]:
                
                strategy_map = {
                    "📊 Volume & MACD System": "Volume-MACD"
                }
                
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                user_state['step'] = 'waiting_exit_strategy'
                
                strategy_descriptions = {
                    "Volume-MACD": "Phân tích Volume, MACD, RSI & EMA trên 3 khung thời gian"
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
                self._continue_bot_creation(chat_id, user_state)

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

                    warning_msg = ""
                    if leverage > 50:
                        warning_msg = f"\n\n⚠️ <b>CẢNH BÁO RỦI RO CAO</b>\nĐòn bẩy {leverage}x rất nguy hiểm!"
                    elif leverage > 20:
                        warning_msg = f"\n\n⚠️ <b>CẢNH BÁO RỦI RO</b>\nĐòn bẩy {leverage}x có rủi ro cao!"

                    user_state['leverage'] = leverage
                    user_state['step'] = 'waiting_percent'
                    
                    balance = get_balance(self.api_key, self.api_secret)
                    balance_info = f"\n💰 Số dư hiện có: {balance:.2f} USDT" if balance else ""
                    
                    send_telegram(
                        f"💰 Đòn bẩy: {leverage}x{balance_info}{warning_msg}\n\n"
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
                    
                    success = self.add_bot(
                        symbol=symbol,
                        lev=leverage,
                        percent=percent,
                        tp=tp,
                        sl=sl,
                        strategy_type=strategy,
                        bot_mode=bot_mode,
                        bot_count=bot_count
                    )
                    
                    if success:
                        success_msg = (
                            f"✅ <b>ĐÃ TẠO {bot_count} BOT THÀNH CÔNG</b>\n\n"
                            f"🤖 Chiến lược: {strategy}\n"
                            f"🔧 Chế độ: {bot_mode}\n"
                            f"🔢 Số lượng: {bot_count} bot độc lập\n"
                            f"💰 Đòn bẩy: {leverage}x\n"
                            f"📊 % Số dư: {percent}%\n"
                            f"🎯 TP: {tp}%\n"
                            f"🛡️ SL: {sl}%"
                        )
                        if bot_mode == 'static' and symbol:
                            success_msg += f"\n🔗 Coin: {symbol}"
                        
                        success_msg += f"\n\n🎯 <b>Mỗi bot là 1 vòng lặp độc lập</b>\n"
                        success_msg += f"🔄 <b>Tự reset hoàn toàn sau mỗi lệnh</b>\n"
                        success_msg += f"📊 <b>Tự tìm coin & trade độc lập</b>"
                        
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
                f"🎯 <b>CHỌN SỐ LƯỢNG BOT ĐỘC LẬP</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"Chọn số lượng bot độc lập bạn muốn tạo:\n"
                f"<i>Mỗi bot sẽ tự tìm coin & trade độc lập</i>",
                chat_id,
                create_bot_count_keyboard(),
                self.telegram_bot_token, self.telegram_chat_id
            )
        
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐỘC LẬP ĐANG CHẠY</b>\n\n"
                
                active_bots = 0
                searching_bots = 0
                trading_bots = 0
                
                for bot_id, bot in self.bots.items():
                    if bot.status == "searching":
                        status = "🔍 Đang tìm coin"
                        searching_bots += 1
                    elif bot.status == "waiting":
                        status = "🟡 Chờ tín hiệu"
                        trading_bots += 1
                    elif bot.status == "open":
                        status = "🟢 Đang trade"
                        trading_bots += 1
                    else:
                        status = "⚪ Unknown"
                    
                    symbol_info = bot.symbol if bot.symbol else "Đang tìm..."
                    message += f"🔹 {bot_id}\n"
                    message += f"   📊 {symbol_info} | {status}\n"
                    message += f"   💰 ĐB: {bot.lev}x | Vốn: {bot.percent}%\n\n"
                
                message += f"📈 Tổng số: {len(self.bots)} bot\n"
                message += f"🔍 Đang tìm coin: {searching_bots} bot\n"
                message += f"📊 Đang trade: {trading_bots} bot"
                
                send_telegram(message, chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "📊 Thống kê":
            summary = self.get_position_summary()
            send_telegram(summary, chat_id,
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
                    bot = self.bots[bot_id]
                    symbol_info = bot.symbol if bot.symbol else "No Coin"
                    message += f"🔹 {bot_id} - {symbol_info}\n"
                    row.append({"text": f"⛔ {bot_id}"})
                    if len(row) == 1 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                
                keyboard.append([{"text": "⛔ DỪNG TẤT CẢ"}])
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram(
                    message, 
                    chat_id, 
                    {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                    self.telegram_bot_token, self.telegram_chat_id
                )
        
        elif text.startswith("⛔ "):
            bot_id = text.replace("⛔ ", "").strip()
            if bot_id == "DỪNG TẤT CẢ":
                self.stop_all()
                send_telegram("⛔ Đã dừng tất cả bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif self.stop_bot(bot_id):
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
                pass
        
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
                pass
        
        elif text == "🎯 Chiến lược":
            strategy_info = (
                "🎯 <b>HỆ THỐNG PHÂN TÍCH VOLUME, MACD, RSI & EMA</b>\n\n"
                
                "📊 <b>Nguyên tắc giao dịch:</b>\n"
                "• Volume tăng + MACD bullish + nến xanh → MUA\n"
                "• Volume tăng + MACD bearish + nến đỏ → BÁN\n"  
                "• Volume giảm + nến doji → CHỈ MUA\n"
                "• RSI quá mua/quá bán + EMA + volume giảm → THOÁT LỆNH\n\n"
                
                "⏰ <b>Khung thời gian phân tích:</b>\n"
                "• 1 phút - Tín hiệu nhanh\n"
                "• 5 phút - Trung hạn\n"
                "• 15 phút - Xu hướng chính\n\n"
                
                "📈 <b>Chỉ báo kỹ thuật:</b>\n"
                "• MACD (12,26,9) - Xác định xu hướng\n"
                "• RSI (14) - Xác định quá mua/quá bán\n"
                "• EMA (20) - Xác định xu hướng trung hạn\n"
                "• Volume - Xác định sức mạnh\n\n"
                
                "⚖️ <b>Cân bằng vị thế:</b>\n"
                "• Đếm tổng số LONG/SHORT trên Binance\n"
                "• Ưu tiên hướng NGƯỢC với số lượng nhiều hơn\n"
                "• Đảm bảo đa dạng hóa rủi ro"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            searching_bots = sum(1 for bot in self.bots.values() if bot.status == "searching")
            trading_bots = sum(1 for bot in self.bots.values() if bot.status in ["waiting", "open"])
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG ĐA LUỒNG</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🔍 Đang tìm coin: {searching_bots} bot\n"
                f"📊 Đang trade: {trading_bots} bot\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n\n"
                f"🎯 <b>Mỗi bot độc lập - Tự reset hoàn toàn</b>"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        strategy = user_state.get('strategy')
        bot_mode = user_state.get('bot_mode', 'static')
        bot_count = user_state.get('bot_count', 1)
        
        if bot_mode == 'static':
            user_state['step'] = 'waiting_symbol'
            send_telegram(
                f"🎯 <b>BOT TĨNH: {strategy}</b>\n"
                f"🤖 Số lượng: {bot_count} bot độc lập\n\n"
                f"🤖 Mỗi bot sẽ trade coin CỐ ĐỊNH\n\n"
                f"Chọn cặp coin:",
                chat_id,
                create_symbols_keyboard(strategy),
                self.telegram_bot_token, self.telegram_chat_id
            )
        else:
            user_state['step'] = 'waiting_leverage'
            send_telegram(
                f"🎯 <b>BOT ĐỘNG ĐA LUỒNG</b>\n"
                f"🤖 Số lượng: {bot_count} bot độc lập\n\n"
                f"🤖 Mỗi bot sẽ tự tìm coin & trade độc lập\n"
                f"🔄 Tự reset hoàn toàn sau mỗi lệnh\n"
                f"📊 Mỗi bot là 1 vòng lặp hoàn chỉnh\n"
                f"⚖️ Tự cân bằng với các bot khác\n\n"
                f"Chọn đòn bẩy:",
                chat_id,
                create_leverage_keyboard(strategy),
                self.telegram_bot_token, self.telegram_chat_id
            )
