# trading_bot_lib.py - HOÀN CHỈNH VỚI CƠ CHẾ TÌM BOT THÔNG MINH & QUẢN LÝ n BOT
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
from collections import deque

# ========== CẤU HÌNH LOGGING ==========
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot_enhanced.log')
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
            symbols = ["ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT", "SOLUSDT"]
    except:
        symbols = ["ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT", "SOLUSDT"]
    
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
                cls._instance.symbol_cooldowns = {}
                cls._instance.cooldown_seconds = 1200  # 20 phút
        return cls._instance
    
    def register_coin(self, symbol, bot_id, strategy, config_key=None):
        with self._lock:
            if config_key not in self.config_coin_count:
                self.config_coin_count[config_key] = 0
            
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

    def set_cooldown(self, symbol, seconds=None):
        ts = time.time() + (seconds or self.cooldown_seconds)
        with self._lock:
            self.symbol_cooldowns[symbol.upper()] = ts

    def is_in_cooldown(self, symbol):
        with self._lock:
            ts = self.symbol_cooldowns.get(symbol.upper())
            if not ts:
                return False
            if time.time() >= ts:
                del self.symbol_cooldowns[symbol.upper()]
                return False
            return True

# ========== API BINANCE NÂNG CẤP ==========
def sign(query, api_secret):
    try:
        return hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None, timeout=15):
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
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
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

def get_all_usdt_pairs(limit=None, max_volume=1000000000):  # THÊM THAM SỐ max_volume = 1 tỷ
    """Lấy danh sách coin USDT với volume 24h NHỎ HƠN 1B - LOẠI BỎ BTC"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            logger.warning("Không lấy được dữ liệu từ Binance, trả về danh sách rỗng")
            return []
        
        # Lấy volume 24h để sắp xếp
        ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        ticker_data = binance_api_request(ticker_url)
        volume_map = {}
        if ticker_data:
            for ticker in ticker_data:
                if 'symbol' in ticker and 'quoteVolume' in ticker:
                    volume_map[ticker['symbol']] = float(ticker['quoteVolume'])
        
        usdt_pairs = []
        for symbol_info in data.get('symbols', []):
            symbol = symbol_info.get('symbol', '')
            # LOẠI BỎ BTCUSDT VÀ CÁC COIN KHÔNG PHÙ HỢP
            if (symbol.endswith('USDT') and 
                symbol != 'BTCUSDT' and  # LOẠI BỎ BTC
                symbol_info.get('status') == 'TRADING' and
                symbol_info.get('contractType') == 'PERPETUAL'):
                
                volume = volume_map.get(symbol, 0)
                # THÊM ĐIỀU KIỆN: CHỈ LẤY COIN CÓ VOLUME NHỎ HƠN 1 TỶ USDT
                if volume < max_volume:  # < 1,000,000,000
                    usdt_pairs.append((symbol, volume))
        
        # Sắp xếp theo volume giảm dần
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # NẾU CÓ LIMIT THÌ CẮT, KHÔNG THÌ TRẢ VỀ TẤT CẢ
        if limit:
            symbols = [pair[0] for pair in usdt_pairs[:limit]]
        else:
            symbols = [pair[0] for pair in usdt_pairs]  # TRẢ VỀ TẤT CẢ
        
        logger.info(f"✅ Lấy được {len(symbols)} coin USDT (volume < {max_volume:,.0f} USDT) - đã loại BTC")
        return symbols
        
    except Exception as e:
        logger.error(f"❌ Lỗi lấy danh sách coin từ Binance: {str(e)}")
        # DANH SÁCH MẶC ĐỊNH KHÔNG CÓ BTC
        return ["ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT", "SOLUSDT"]

def get_symbol_info(symbol, api_key=None, api_secret=None):
    """Lấy thông tin chi tiết về symbol"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return None
            
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                info = {'symbol': symbol}
                
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        info['step_size'] = float(f['stepSize'])
                        info['min_qty'] = float(f['minQty'])
                    elif f['filterType'] == 'MARKET_LOT_SIZE':
                        info['max_qty'] = float(f['maxQty'])
                        info['min_qty'] = max(info.get('min_qty', 0), float(f['minQty']))
                    elif f['filterType'] == 'PRICE_FILTER':
                        info['tick_size'] = float(f['tickSize'])
                
                return info
                
    except Exception as e:
        logger.error(f"Lỗi lấy symbol info: {str(e)}")
    
    return {'step_size': 0.001, 'min_qty': 0.001, 'tick_size': 0.01}

def get_klines_with_cache(symbol, interval="5m", limit=100):
    """Lấy dữ liệu klines với cache"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        data = binance_api_request(url, params=params)
        
        if not data:
            return None
            
        opens, highs, lows, closes, volumes = [], [], [], [], []
        for kline in data:
            opens.append(float(kline[1]))
            highs.append(float(kline[2]))
            lows.append(float(kline[3]))
            closes.append(float(kline[4]))
            volumes.append(float(kline[5]))
            
        return opens, highs, lows, closes, volumes
    except Exception as e:
        logger.error(f"Lỗi lấy klines {symbol}: {e}")
        return None

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
    return 0

def get_24h_ticker(symbol):
    """Lấy thông tin ticker 24h chi tiết"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data:
            return {
                'symbol': data.get('symbol'),
                'price_change_percent': float(data.get('priceChangePercent', 0)),
                'volume': float(data.get('quoteVolume', 0)),
                'high_price': float(data.get('highPrice', 0)),
                'low_price': float(data.get('lowPrice', 0)),
                'count': int(data.get('count', 0))
            }
    except Exception as e:
        logger.error(f"Lỗi lấy ticker 24h: {str(e)}")
    return None

def get_batch_ticker_info(symbols):
    """Lấy thông tin ticker cho nhiều coin cùng lúc"""
    try:
        # Binance API cho phép lấy nhiều symbol cùng lúc
        symbol_param = '["' + '","'.join(symbols) + '"]'
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbols={symbol_param}"
        data = binance_api_request(url)
        if data:
            return {item['symbol']: item for item in data}
        return {}
    except Exception as e:
        logger.error(f"Lỗi lấy batch ticker: {str(e)}")
        return {}

# ========== CƠ CHẾ TÌM BOT THÔNG MINH ==========
def get_qualified_symbol(strategy_config, excluded_symbols=None):
    """
    Tìm 1 symbol duy nhất đủ điều kiện theo chiến lược
    """
    if excluded_symbols is None:
        excluded_symbols = []
    
    try:
        # Lấy tất cả symbol có volume cao từ Binance
        all_symbols = get_all_usdt_pairs(limit=100)
        
        # Lọc theo chiến lược và loại bỏ các symbol đã có
        qualified_symbols = []
        
        for symbol in all_symbols:
            if symbol in excluded_symbols:
                continue
                
            # Kiểm tra điều kiện theo chiến lược
            if check_strategy_conditions(symbol, strategy_config):
                qualified_symbols.append(symbol)
        
        # Trả về symbol đầu tiên đủ điều kiện
        return qualified_symbols[0] if qualified_symbols else None
        
    except Exception as e:
        logger.error(f"Lỗi khi tìm symbol: {e}")
        return None

def check_strategy_conditions(symbol, strategy_config):
    """
    Kiểm tra điều kiện chiến lược cho symbol
    """
    try:
        # Lấy thông tin klines và ticker
        klines = get_klines_with_cache(symbol, "5m", 50)
        ticker_info = get_24h_ticker(symbol)
        
        if not klines or not ticker_info:
            return False
            
        closes = klines[3]
        price_change = ticker_info['price_change_percent']
        volume = ticker_info['volume']
        
        # Điều kiện cơ bản
        if volume < 1000000:  # Volume tối thiểu 1M USDT
            return False
            
        # Kiểm tra theo chiến lược
        strategy_type = strategy_config.get('strategy_type', 'Smart Dynamic')
        
        if strategy_type == "Reverse 24h":
            threshold = strategy_config.get('threshold', 25)
            return abs(price_change) >= threshold
            
        elif strategy_type == "Scalping":
            volatility_std = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
            volatility_threshold = strategy_config.get('volatility', 3)
            return (2 <= abs(price_change) <= 15 and 
                    volatility_std >= volatility_threshold)
                    
        elif strategy_type == "Trend Following":
            if len(closes) >= 50:
                short_trend = (closes[-1] - closes[-10]) / closes[-10] * 100
                medium_trend = (closes[-1] - closes[-25]) / closes[-25] * 100
                trend_strength = (abs(short_trend) + abs(medium_trend)) / 2
                return trend_strength >= 3
                
        elif strategy_type == "Smart Dynamic":
            rsi = calc_rsi(closes, 14)
            volatility_std = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
            return (1 <= abs(price_change) <= 12 and
                    volatility_std >= 2 and
                    rsi and 20 <= rsi <= 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi kiểm tra điều kiện {symbol}: {e}")
        return False

def get_qualified_symbols(api_key, api_secret, strategy_type, leverage, threshold=None, volatility=None, grid_levels=None, max_candidates=50, final_limit=2, strategy_key=None):
    """TÌM COIN THÔNG MINH - LẤY TẤT CẢ COIN - PHÂN LOẠI THEO CHIẾN LƯỢC - LOẠI BỎ BTC"""
    try:
        coin_manager = CoinManager()
        
        # Lấy danh sách TẤT CẢ coin có volume cao - BỎ LIMIT
        all_symbols = get_all_usdt_pairs(limit=None)  # LẤY TẤT CẢ COIN
        if not all_symbols:
            logger.error("❌ Không lấy được danh sách coin từ Binance")
            return []
        
        # Lọc bớt: chỉ xét coin có volume > 1M USDT để tối ưu hiệu năng
        filtered_symbols = []
        for symbol in all_symbols:
            ticker_info = get_24h_ticker(symbol)
            if ticker_info and ticker_info['volume'] > 1000000:  # Volume > 1M USDT
                filtered_symbols.append(symbol)
            else:
                continue
        
        logger.info(f"🔍 Quét {len(filtered_symbols)} coin có volume > 1M USDT")
        
        qualified_symbols = []
        processed_count = 0
        
        for symbol in filtered_symbols:
            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"⏳ Đã xử lý {processed_count}/{len(filtered_symbols)} coin...")
            
            # LOẠI BỎ BTC NGAY TỪ ĐẦU
            if symbol == 'BTCUSDT':
                continue
                
            # Kiểm tra coin đã được quản lý bởi config này chưa
            if strategy_key and coin_manager.has_same_config_bot(symbol, strategy_key):
                continue
                
            # Kiểm tra cooldown
            if coin_manager.is_in_cooldown(symbol):
                continue
            
            # Lấy thông tin chi tiết
            ticker_info = get_24h_ticker(symbol)
            klines = get_klines_with_cache(symbol, "5m", 50)
            
            if not ticker_info or not klines:
                continue
                
            price_change = ticker_info['price_change_percent']
            volume = ticker_info['volume']
            high_price = ticker_info['high_price']
            low_price = ticker_info['low_price']
            
            if low_price > 0:
                price_range = ((high_price - low_price) / low_price) * 100
            else:
                price_range = 0
            
            closes = klines[3]
            if len(closes) < 20:
                continue
                
            # Tính toán chỉ báo
            current_rsi = calc_rsi(closes, 14)
            volatility_std = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
            
            # ĐÁNH GIÁ PHÙ HỢP THEO CHIẾN LƯỢC
            score = 0
            reasons = []
            
            if strategy_type == "Reverse 24h":
                if abs(price_change) >= (threshold or 25):
                    score = abs(price_change) * min(volume / 10000000, 5)
                    reasons.append(f"Biến động: {price_change:.1f}%")
                    
            elif strategy_type == "Scalping":
                if (2 <= abs(price_change) <= 15 and 
                    volatility_std >= (volatility or 3) and
                    volume > 5000000):
                    score = volume * volatility_std
                    reasons.append(f"Volatility: {volatility_std:.1f}%")
                    
            elif strategy_type == "Safe Grid":
                if (0.5 <= abs(price_change) <= 8 and 
                    volatility_std <= 5 and
                    volume > 2000000):
                    score = volume * (1 / (abs(price_change - 3) + 0.1))
                    reasons.append(f"Ổn định: {price_change:.1f}%")
                    
            elif strategy_type == "Trend Following":
                if len(closes) >= 50:
                    short_trend = (closes[-1] - closes[-10]) / closes[-10] * 100
                    medium_trend = (closes[-1] - closes[-25]) / closes[-25] * 100
                    trend_strength = (abs(short_trend) + abs(medium_trend)) / 2
                    
                    if trend_strength >= 3 and volume > 3000000:
                        score = volume * trend_strength
                        reasons.append(f"Trend: {trend_strength:.1f}%")
                        
            elif strategy_type == "Smart Dynamic":
                if (1 <= abs(price_change) <= 12 and
                    volatility_std >= 2 and
                    volume > 3000000 and
                    current_rsi and 20 <= current_rsi <= 80):
                    
                    volume_score = min(volume / 5000000, 3)
                    volatility_score = min(volatility_std / 5, 2)
                    rsi_score = 1 if 30 <= current_rsi <= 70 else 0.5
                    
                    score = volume_score + volatility_score + rsi_score
                    reasons.append(f"RSI: {current_rsi:.1f}")
            
            # THÊM ĐIỂM THƯỞNG CHO COIN KHÔNG PHẢI BTC/ETH
            if symbol not in ['BTCUSDT', 'ETHUSDT']:
                score *= 1.5
                reasons.append("Coin alt")
                
            if score > 0:
                qualified_symbols.append({
                    'symbol': symbol,
                    'score': score,
                    'price_change': price_change,
                    'volume': volume,
                    'volatility': volatility_std,
                    'rsi': current_rsi,
                    'reasons': reasons
                })
        
        # SẮP XẾP VÀ LỌC
        qualified_symbols.sort(key=lambda x: x['score'], reverse=True)
        
        final_symbols = []
        for candidate in qualified_symbols[:max_candidates]:
            if len(final_symbols) >= final_limit:
                break
                
            symbol = candidate['symbol']
            
            try:
                # Kiểm tra leverage và số lượng
                leverage_success = set_leverage(symbol, leverage, api_key, api_secret)
                symbol_info = get_symbol_info(symbol, api_key, api_secret)
                min_qty = symbol_info.get('min_qty', 0.001)
                
                # Tính toán số lượng test
                balance = get_balance(api_key, api_secret)
                if balance:
                    current_price = get_current_price(symbol)
                    if current_price > 0:
                        usd_amount = balance * 0.02  # 2% để test
                        test_qty = (usd_amount * leverage) / current_price
                        
                        if test_qty >= min_qty and leverage_success:
                            final_symbols.append(symbol)
                            logger.info(f"✅ {symbol}: {strategy_type} - Score: {candidate['score']:.2f}, " +
                                      f"Change: {candidate['price_change']:.1f}%, " +
                                      f"Vol: {candidate['volatility']:.1f}%, " +
                                      f"RSI: {candidate['rsi']:.1f}")
                
                time.sleep(0.2)  # Tránh rate limit
            except Exception as e:
                logger.error(f"❌ Lỗi kiểm tra {symbol}: {str(e)}")
                continue
        
        logger.info(f"🎯 {strategy_type}: Tìm thấy {len(final_symbols)} coin phù hợp từ {len(filtered_symbols)} coin được quét")
        return final_symbols[:final_limit]
        
    except Exception as e:
        logger.error(f"❌ Lỗi tìm coin {strategy_type}: {str(e)}")
        return []

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
        # Làm tròn số lượng theo quy tắc Binance
        symbol_info = get_symbol_info(symbol, api_key, api_secret)
        if symbol_info:
            step_size = symbol_info.get('step_size', 0.001)
            min_qty = symbol_info.get('min_qty', 0.001)
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
            
            if qty < min_qty:
                logger.error(f"Số lượng {qty} nhỏ hơn mức tối thiểu {min_qty}")
                return None
                
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": float(qty),
            "timestamp": ts
        }
        
        logger.info(f"Đặt lệnh {side} {symbol}: {qty}")
        
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        result = binance_api_request(url, method='POST', headers=headers)
        
        if result is None:
            logger.error(f"Lỗi khi đặt lệnh {side} cho {symbol}")
        elif 'code' in result:
            logger.error(f"Binance API error: {result}")
            
        return result
        
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh {symbol} {side}: {str(e)}")
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

# ========== CHỈ BÁO KỸ THUẬT NÂNG CAO ==========
def calc_rsi(prices, period=14):
    """RSI với xử lý lỗi tốt hơn"""
    try:
        if len(prices) < period + 1:
            return None
            
        prices = np.array(prices)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period+1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        rs = avg_gains[period:] / (avg_losses[period:] + 1e-10)  # Tránh chia 0
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi[-1]) if len(rsi) > 0 else None
        
    except Exception as e:
        return None

def calc_ema(prices, period):
    """EMA với smoothing factor"""
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

def calc_macd(prices, fast=12, slow=26, signal=9):
    """Tính MACD"""
    try:
        if len(prices) < slow + signal:
            return None, None, None
            
        ema_fast = calc_ema(prices, fast)
        ema_slow = calc_ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
            
        macd_line = ema_fast - ema_slow
        
        # Tính signal line (EMA của MACD)
        macd_prices = [macd_line] * len(prices)  # Giả định cho đơn giản
        signal_line = calc_ema(macd_prices, signal)
        
        histogram = macd_line - signal_line if signal_line else None
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        return None, None, None

def calc_bollinger_bands(prices, period=20, std_dev=2):
    """Tính Bollinger Bands"""
    try:
        if len(prices) < period:
            return None, None, None
            
        recent_prices = prices[-period:]
        middle_band = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
        
    except Exception as e:
        return None, None, None

def calc_stochastic(highs, lows, closes, k_period=14, d_period=3):
    """Tính Stochastic Oscillator"""
    try:
        if len(closes) < k_period:
            return None, None
            
        current_close = closes[-1]
        lowest_low = min(lows[-k_period:])
        highest_high = max(highs[-k_period:])
        
        if highest_high == lowest_low:
            return None, None
            
        k_value = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
        
        # Tính D line (SMA của K)
        if len(closes) >= k_period + d_period - 1:
            k_values = []
            for i in range(len(closes) - k_period + 1):
                period_low = min(lows[i:i+k_period])
                period_high = max(highs[i:i+k_period])
                if period_high != period_low:
                    k_val = 100 * (closes[i+k_period-1] - period_low) / (period_high - period_low)
                    k_values.append(k_val)
            
            if len(k_values) >= d_period:
                d_value = np.mean(k_values[-d_period:])
                return k_value, d_value
        
        return k_value, None
        
    except Exception as e:
        return None, None

def calc_atr(highs, lows, closes, period=14):
    """Tính Average True Range"""
    try:
        if len(highs) < period + 1:
            return None
            
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        atr = np.mean(true_ranges[-period:])
        return atr
        
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

# ========== BASE BOT NÂNG CẤP ==========
class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None):
        
        self.symbol = symbol.upper() if symbol else "ETHUSDT"
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
        self.prices = deque(maxlen=200)  # Dùng deque để giới hạn bộ nhớ
        self.highs = deque(maxlen=200)
        self.lows = deque(maxlen=200)
        self.position_open = False
        self._stop = False
        
        # Biến theo dõi thời gian
        self.last_trade_time = 0
        self.last_close_time = 0
        self.last_signal_time = 0
        self._last_find_attempt = 0
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 300
        self.position_check_interval = 30
        self.signal_cooldown = 60  # 60 giây giữa các tín hiệu
        
        # Bảo vệ chống lặp
        self._close_attempted = False
        self._last_close_attempt = 0
        
        self.should_be_removed = False
        
        self.coin_manager = CoinManager()
        
        # ĐĂNG KÝ COIN
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
            # Cập nhật highs/lows cho các chỉ báo nâng cao
            if len(self.highs) == 0 or price > self.highs[-1]:
                self.highs.append(float(price))
            else:
                self.highs.append(self.highs[-1] if self.highs else float(price))
                
            if len(self.lows) == 0 or price < self.lows[-1]:
                self.lows.append(float(price))
            else:
                self.lows.append(self.lows[-1] if self.lows else float(price))
        except Exception as e:
            self.log(f"❌ Lỗi xử lý giá: {str(e)}")

    def get_signal(self):
        raise NotImplementedError("Phương thức get_signal cần được triển khai")

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
                    # Kiểm tra cooldown giữa các tín hiệu
                    if current_time - self.last_signal_time > self.signal_cooldown:
                        signal = self.get_signal()
                        
                        if (signal and 
                            current_time - self.last_trade_time > 60 and
                            current_time - self.last_close_time > self.cooldown_period):
                            
                            self.log(f"🎯 Nhận tín hiệu {signal}, đang mở lệnh...")
                            if self.open_position(signal):
                                self.last_trade_time = current_time
                                self.last_signal_time = current_time
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

            # TÍNH TOÁN SỐ LƯỢNG CHÍNH XÁC
            symbol_info = get_symbol_info(self.symbol, self.api_key, self.api_secret)
            step_size = symbol_info.get('step_size', 0.001)
            min_qty = symbol_info.get('min_qty', 0.001)
            
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            # KIỂM TRA SỐ LƯỢNG TỐI THIỂU
            if qty < min_qty:
                self.log(f"❌ Số lượng quá nhỏ: {qty:.8f} (tối thiểu: {min_qty})")
                return False

            # LÀM TRÒN THEO STEP SIZE
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)

            # KIỂM TRA LẠI SAU KHI LÀM TRÒN
            if qty < min_qty:
                self.log(f"❌ Số lượng sau làm tròn quá nhỏ: {qty:.8f} (tối thiểu: {min_qty})")
                return False

            self.log(f"💰 Tính toán: balance={balance:.2f}, amount={usd_amount:.2f}, qty={qty:.6f}")

            result = place_order(self.symbol, side, qty, self.api_key, self.api_secret)
            if result and 'orderId' in result:
                executed_qty = float(result.get('executedQty', 0))
                avg_price = float(result.get('avgPrice', current_price))
                
                if executed_qty > 0:
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
                
                # ĐẶT COOLDOWN CHO COIN
                self.coin_manager.set_cooldown(self.symbol)
                
                # XÓA COIN KHỎI DANH SÁCH QUẢN LÝ
                if self.symbol and self.config_key:
                    self.coin_manager.unregister_coin(self.symbol)
                
                # BOT ĐỘNG: TÌM COIN MỚI
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
            if current_time - self._last_find_attempt < 300:  # 5 phút cooldown
                return False
                
            self._last_find_attempt = current_time
            
            self.log(f"🔄 Bot động đang tìm coin mới thay thế {self.symbol}...")
            
            # Kiểm tra số coin hiện tại của config
            current_count = self.coin_manager.get_config_coin_count(self.config_key)
            if current_count >= 2:
                self.log(f"⚠️ Đã đủ {current_count}/2 coin cho config, không tìm thêm")
                return False
            
            # Tìm coin mới phù hợp
            new_symbols = get_qualified_symbols(
                self.api_key, self.api_secret,
                self.strategy_name, self.lev,
                getattr(self, 'threshold', None),
                getattr(self, 'volatility', None),
                getattr(self, 'grid_levels', None),
                max_candidates=5, final_limit=1,
                strategy_key=self.config_key
            )
            
            if new_symbols:
                new_symbol = new_symbols[0]
                
                if new_symbol != self.symbol:
                    self.log(f"🔄 Chuyển từ {self.symbol} → {new_symbol}")
                    
                    # Cập nhật symbol mới
                    old_symbol = self.symbol
                    self.symbol = new_symbol
                    
                    # Đăng ký coin mới
                    success = self._register_coin_with_retry(self.symbol)
                    if not success:
                        self.log(f"❌ Không thể đăng ký coin mới {self.symbol}")
                        # Khôi phục lại symbol cũ
                        self.symbol = old_symbol
                        self._register_coin_with_retry(self.symbol)
                        return False
                    
                    # Khởi động lại WebSocket với coin mới
                    self.ws_manager.remove_symbol(old_symbol)
                    self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
                    
                    self.log(f"✅ Đã chuyển sang coin mới: {self.symbol}")
                    return True
                else:
                    self.log(f"ℹ️ Vẫn giữ coin {self.symbol} (phù hợp nhất)")
            else:
                self.log(f"⚠️ Không tìm thấy coin mới phù hợp, giữ {self.symbol}")
            
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

    def has_active_orders(self):
        """Kiểm tra bot có orders active không"""
        try:
            orders = get_positions(self.symbol, self.api_key, self.api_secret)
            for order in orders:
                position_amt = float(order.get('positionAmt', 0))
                if abs(position_amt) > 0:
                    return True
            return False
        except Exception as e:
            self.log(f"Lỗi kiểm tra active orders: {e}")
            return False

    def check_status(self):
        """
        Kiểm tra trạng thái bot với cơ chế tránh trùng lặp
        """
        try:
            # Kiểm tra trạng thái orders hiện tại
            current_orders = self.has_active_orders()
            
            if not current_orders:
                if self.entry_price:  # Đã có lệnh entry trước đó
                    return "completed"
                else:
                    return "waiting"
            
            # Phân loại trạng thái
            if self.position_open:
                return "active"
            else:
                return "waiting"
                
        except Exception as e:
            self.log(f"Lỗi khi check status: {e}")
            return "error"

# ========== CÁC CHIẾN LƯỢC VỚI CHỈ BÁO NÂNG CAO ==========
class RSI_EMA_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "RSI/EMA Recursive")
        self.rsi_period = 14
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            # Lấy dữ liệu klines để có high/low
            klines = get_klines_with_cache(self.symbol, "5m", 100)
            if not klines:
                return None
                
            highs, lows, closes = klines[2], klines[3], klines[4]
            
            # Tính các chỉ báo
            rsi = calc_rsi(closes, self.rsi_period)
            ema_fast = calc_ema(closes, self.ema_fast)
            ema_slow = calc_ema(closes, self.ema_slow)
            macd, signal, histogram = calc_macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
            stoch_k, stoch_d = calc_stochastic(highs, lows, closes)

            if None in [rsi, ema_fast, ema_slow, macd, signal]:
                return None

            signal = None
            score = 0
            
            # RSI + EMA CONFIRMATION
            if rsi < self.rsi_oversold and ema_fast > ema_slow:
                score += 2
                signal = "BUY"
            elif rsi > self.rsi_overbought and ema_fast < ema_slow:
                score += 2
                signal = "SELL"
            
            # MACD CONFIRMATION
            if macd > signal and signal == "BUY":
                score += 1
            elif macd < signal and signal == "SELL":
                score += 1
            
            # STOCHASTIC CONFIRMATION
            if stoch_k and stoch_d:
                if stoch_k < 20 and stoch_d < 20 and signal == "BUY":
                    score += 1
                elif stoch_k > 80 and stoch_d > 80 and signal == "SELL":
                    score += 1

            # CHỈ VÀO LỆNH KHI SCORE ĐỦ CAO
            if score >= 3:
                self.log(f"🎯 RSI/EMA Signal: {signal} | Score: {score}/4 | RSI: {rsi:.1f} | MACD: {macd:.4f}")
                return signal

            return None

        except Exception as e:
            return None

class EMA_Crossover_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "EMA Crossover")
        self.ema_fast = 9
        self.ema_slow = 21
        self.ema_trend = 50
        self.prev_ema_fast = None
        self.prev_ema_slow = None

    def get_signal(self):
        try:
            if len(self.prices) < 100:
                return None

            klines = get_klines_with_cache(self.symbol, "5m", 100)
            if not klines:
                return None
                
            closes = klines[4]
            
            ema_fast = calc_ema(closes, self.ema_fast)
            ema_slow = calc_ema(closes, self.ema_slow)
            ema_trend = calc_ema(closes, self.ema_trend)
            bb_upper, bb_middle, bb_lower = calc_bollinger_bands(closes)

            if None in [ema_fast, ema_slow, ema_trend, bb_upper]:
                return None

            signal = None
            current_price = closes[-1] if closes else self.prices[-1] if self.prices else 0
            
            if self.prev_ema_fast is not None and self.prev_ema_slow is not None:
                # EMA CROSSOVER
                if (self.prev_ema_fast <= self.prev_ema_slow and ema_fast > ema_slow and
                    current_price > ema_trend and current_price > bb_middle):
                    signal = "BUY"
                elif (self.prev_ema_fast >= self.prev_ema_slow and ema_fast < ema_slow and
                      current_price < ema_trend and current_price < bb_middle):
                    signal = "SELL"
                
                if signal:
                    self.log(f"🎯 EMA Crossover: {signal} | Fast: {ema_fast:.4f} | Slow: {ema_slow:.4f}")

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
            if current_time - self.last_24h_check < 300:  # 5 phút
                return None

            change_24h = get_24h_change(self.symbol)
            self.last_24h_check = current_time

            if change_24h is None:
                return None
                
            # Chỉ log khi có thay đổi đáng kể
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
        self.trend_period_short = 10
        self.trend_period_medium = 25
        self.trend_period_long = 50
        self.trend_threshold = 0.5

    def get_signal(self):
        try:
            if len(self.prices) < 100:
                return None

            klines = get_klines_with_cache(self.symbol, "5m", 100)
            if not klines:
                return None
                
            closes = klines[4]
            highs = klines[2]
            lows = klines[3]
            
            # Tính trend strength
            short_trend = (closes[-1] - closes[-self.trend_period_short]) / closes[-self.trend_period_short] * 100
            medium_trend = (closes[-1] - closes[-self.trend_period_medium]) / closes[-self.trend_period_medium] * 100
            long_trend = (closes[-1] - closes[-self.trend_period_long]) / closes[-self.trend_period_long] * 100
            
            # Tính ATR để xác định độ mạnh của trend
            atr = calc_atr(highs, lows, closes)
            
            signal = None
            trend_strength = (short_trend + medium_trend + long_trend) / 3
            
            if trend_strength > self.trend_threshold and atr and atr > 0.001:
                signal = "BUY"
                self.log(f"🎯 Trend Following BUY | Strength: {trend_strength:.2f}% | ATR: {atr:.4f}")
            elif trend_strength < -self.trend_threshold and atr and atr > 0.001:
                signal = "SELL"
                self.log(f"🎯 Trend Following SELL | Strength: {trend_strength:.2f}% | ATR: {atr:.4f}")

            return signal

        except Exception as e:
            return None

class Scalping_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Scalping", config_key)
        self.rsi_period = 7
        self.min_movement = 0.001
        self.volatility_threshold = 3

    def get_signal(self):
        try:
            if len(self.prices) < 20:
                return None

            klines = get_klines_with_cache(self.symbol, "5m", 50)
            if not klines:
                return None
                
            closes = klines[4]
            highs = klines[2]
            lows = klines[3]
            
            current_price = closes[-1] if closes else self.prices[-1]
            price_change = 0
            if len(closes) >= 2:
                price_change = (closes[-1] - closes[-2]) / closes[-2]

            rsi = calc_rsi(closes, self.rsi_period)
            atr = calc_atr(highs, lows, closes)
            volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) * 100 if len(closes) >= 10 else 0

            if rsi is None or atr is None:
                return None

            signal = None
            if (rsi < 25 and price_change < -self.min_movement and 
                volatility >= self.volatility_threshold and atr > 0.001):
                signal = "BUY"
                self.log(f"🎯 Scalping BUY | RSI: {rsi:.1f} | Volatility: {volatility:.1f}%")
            elif (rsi > 75 and price_change > self.min_movement and 
                  volatility >= self.volatility_threshold and atr > 0.001):
                signal = "SELL"
                self.log(f"🎯 Scalping SELL | RSI: {rsi:.1f} | Volatility: {volatility:.1f}%")

            return signal

        except Exception as e:
            return None

class Safe_Grid_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, grid_levels=5, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Safe Grid", config_key)
        self.grid_levels = grid_levels
        self.orders_placed = 0
        self.min_position_value = 10  # Giá trị vị thế tối thiểu 10 USDT

    def get_signal(self):
        try:
            # KIỂM TRA GIÁ TRỊ VỊ THẾ TỐI THIỂU
            if self.orders_placed < self.grid_levels:
                current_price = get_current_price(self.symbol)
                balance = get_balance(self.api_key, self.api_secret)
                
                if current_price > 0 and balance:
                    usd_amount = balance * (self.percent / 100)
                    position_value = usd_amount * self.lev
                    
                    if position_value < self.min_position_value:
                        self.log(f"⚠️ Bỏ qua grid - giá trị vị thế {position_value:.2f} < {self.min_position_value} USDT")
                        return None
                
                self.orders_placed += 1
                if self.orders_placed % 2 == 1:
                    return "BUY"
                else:
                    return "SELL"
            return None
        except Exception as e:
            return None

# ========== BOT ĐỘNG THÔNG MINH NÂNG CAO ==========
class SmartDynamicBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Smart Dynamic", config_key)

    def get_signal(self):
        try:
            if len(self.prices) < 100:
                return None

            klines = get_klines_with_cache(self.symbol, "5m", 100)
            if not klines:
                return None
                
            highs, lows, closes = klines[2], klines[3], klines[4]
            
            # TÍNH TOÁN ĐA CHỈ BÁO
            rsi = calc_rsi(closes, 14)
            ema_fast = calc_ema(closes, 9)
            ema_slow = calc_ema(closes, 21)
            ema_trend = calc_ema(closes, 50)
            macd, signal, histogram = calc_macd(closes, 12, 26, 9)
            bb_upper, bb_middle, bb_lower = calc_bollinger_bands(closes)
            stoch_k, stoch_d = calc_stochastic(highs, lows, closes)
            atr = calc_atr(highs, lows, closes)
            
            if None in [rsi, ema_fast, ema_slow, macd, signal, bb_upper]:
                return None

            signal_direction = None
            score = 0
            reasons = []
            
            current_price = closes[-1]
            
            # 1. XU HƯỚNG CHÍNH (30%)
            if current_price > ema_trend and ema_fast > ema_slow:
                score += 3
                reasons.append("Xu hướng tăng")
                signal_direction = "BUY"
            elif current_price < ema_trend and ema_fast < ema_slow:
                score += 3
                reasons.append("Xu hướng giảm")
                signal_direction = "SELL"
            
            # 2. RSI + MOMENTUM (25%)
            if signal_direction == "BUY" and rsi < 40:
                score += 2.5
                reasons.append(f"RSI oversold: {rsi:.1f}")
            elif signal_direction == "SELL" and rsi > 60:
                score += 2.5
                reasons.append(f"RSI overbought: {rsi:.1f}")
            
            # 3. MACD CONFIRMATION (20%)
            if signal_direction == "BUY" and macd > signal:
                score += 2
                reasons.append("MACD bullish")
            elif signal_direction == "SELL" and macd < signal:
                score += 2
                reasons.append("MACD bearish")
            
            # 4. BOLLINGER BANDS (15%)
            if signal_direction == "BUY" and current_price < bb_lower:
                score += 1.5
                reasons.append("BB oversold")
            elif signal_direction == "SELL" and current_price > bb_upper:
                score += 1.5
                reasons.append("BB overbought")
            
            # 5. STOCHASTIC CONFIRMATION (10%)
            if stoch_k and stoch_d:
                if signal_direction == "BUY" and stoch_k < 30 and stoch_d < 30:
                    score += 1
                    reasons.append("Stoch oversold")
                elif signal_direction == "SELL" and stoch_k > 70 and stoch_d > 70:
                    score += 1
                    reasons.append("Stoch overbought")
            
            # FILTER: ATR để tránh market quá bình ổn
            if atr and atr < 0.0005:  # ATR quá thấp
                score -= 2
                reasons.append("Market quá bình ổn")
            
            # CHỈ VÀO LỆNH KHI SCORE ĐỦ CAO
            if score >= 6 and signal_direction:
                reason_str = " | ".join(reasons)
                self.log(f"🎯 Smart Dynamic: {signal_direction} | Score: {score}/10 | {reason_str}")
                return signal_direction
            
            return None

        except Exception as e:
            self.log(f"❌ Lỗi Smart Dynamic signal: {str(e)}")
            return None

# ========== BOT MANAGER HOÀN CHỈNH VỚI CƠ CHẾ TÌM BOT THÔNG MINH ==========
class BotManager:
    def __init__(self, api_key=None, api_secret=None, telegram_bot_token=None, telegram_chat_id=None, max_bots=5):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        
        # CẤU HÌNH MỚI: Quản lý n bot
        self.max_bots = max_bots
        self.pending_symbols = []
        self.excluded_symbols = set()
        
        self.auto_strategies = {}
        self.last_auto_scan = 0
        self.auto_scan_interval = 600
        
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
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG BOT THÔNG MINH ĐÃ KHỞI ĐỘNG")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
            self.auto_scan_thread = threading.Thread(target=self._auto_scan_loop, daemon=True)
            self.auto_scan_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def update_bot_list(self, strategy_config):
        """
        Cập nhật danh sách bot - chỉ thêm bot mới khi có bot đặt lệnh
        """
        try:
            # Bước 1: Kiểm tra và xử lý các bot đang active
            self._process_active_bots()
            
            # Bước 2: Nếu chưa đủ bot và có bot đã đặt lệnh, tìm bot mới
            if len(self.bots) < self.max_bots and self._has_active_orders():
                new_symbol = self._find_new_symbol(strategy_config)
                if new_symbol:
                    self._add_new_bot(new_symbol, strategy_config)
            
            # Bước 3: Hiển thị trạng thái
            self._display_status()
            
        except Exception as e:
            self.log(f"Lỗi khi cập nhật bot list: {e}")
    
    def _process_active_bots(self):
        """Xử lý các bot đang active"""
        bots_to_remove = []
        
        for bot_id, bot in self.bots.items():
            try:
                # Kiểm tra trạng thái bot
                status = bot.check_status()
                
                # Nếu bot đã kết thúc hoặc lỗi, loại bỏ
                if status in ['completed', 'error', 'cancelled']:
                    bots_to_remove.append(bot_id)
                    self.excluded_symbols.add(bot.symbol)
                    
            except Exception as e:
                self.log(f"Lỗi khi kiểm tra bot {bot_id}: {e}")
                bots_to_remove.append(bot_id)
        
        # Xóa các bot không còn active
        for bot_id in bots_to_remove:
            bot = self.bots[bot_id]
            self.excluded_symbols.add(bot.symbol)
            del self.bots[bot_id]
    
    def _has_active_orders(self):
        """
        Kiểm tra xem có ít nhất một bot đã đặt lệnh thành công chưa
        """
        for bot in self.bots.values():
            if bot.has_active_orders():
                return True
        return False
    
    def _find_new_symbol(self, strategy_config):
        """
        Tìm symbol mới không trùng lặp
        """
        return get_qualified_symbol(
            strategy_config, 
            excluded_symbols=list(self.excluded_symbols)
        )
    
    def _add_new_bot(self, symbol, strategy_config):
        """
        Thêm bot mới vào hệ thống
        """
        try:
            if symbol not in [bot.symbol for bot in self.bots.values()] and symbol not in self.excluded_symbols:
                # Tạo bot mới dựa trên chiến lược
                strategy_type = strategy_config.get('strategy_type', 'Smart Dynamic')
                leverage = strategy_config.get('leverage', 10)
                percent = strategy_config.get('percent', 5)
                tp = strategy_config.get('tp', 100)
                sl = strategy_config.get('sl', 50)
                
                bot_class = {
                    "Reverse 24h": Reverse_24h_Bot,
                    "Scalping": Scalping_Bot,
                    "Safe Grid": Safe_Grid_Bot,
                    "Trend Following": Trend_Following_Bot,
                    "Smart Dynamic": SmartDynamicBot
                }.get(strategy_type, SmartDynamicBot)
                
                if strategy_type == "Reverse 24h":
                    threshold = strategy_config.get('threshold', 30)
                    bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token, 
                                  self.telegram_chat_id, threshold)
                elif strategy_type == "Safe Grid":
                    grid_levels = strategy_config.get('grid_levels', 5)
                    bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token,
                                  self.telegram_chat_id, grid_levels)
                else:
                    bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token,
                                  self.telegram_chat_id)
                
                bot_id = f"{symbol}_{strategy_type}"
                self.bots[bot_id] = bot
                self.excluded_symbols.add(symbol)
                self.log(f"✅ Đã thêm bot mới: {symbol} - {strategy_type}")
                
        except Exception as e:
            self.log(f"Lỗi khi thêm bot {symbol}: {e}")
            self.excluded_symbols.add(symbol)
    
    def _display_status(self):
        """Hiển thị trạng thái hiện tại"""
        active_count = len(self.bots)
        self.log(f"🎯 Trạng thái Bot Manager: {active_count}/{self.max_bots} bot đang active")
        
        for bot_id, bot in self.bots.items():
            status = bot.check_status()
            self.log(f"   - {bot_id}: {status}")

    def scan_and_allocate_bots(self, strategy_name, allocation_rules):
        """
        Duyệt bot theo yêu cầu chiến lược và phân bổ vào list
        
        Args:
            strategy_name: Tên chiến lược
            allocation_rules: Quy tắc phân bổ
        """
        try:
            # Lấy config chiến lược
            strategy_config = self._get_strategy_config(strategy_name, allocation_rules)
            
            # Duyệt theo khối lượng giảm dần từ Binance
            high_volume_symbols = get_all_usdt_pairs(limit=500)
            
            allocated_count = 0
            
            for symbol in high_volume_symbols:
                # Kiểm tra nếu đã đủ số lượng bot
                if len(self.bots) >= self.max_bots:
                    break
                    
                # Kiểm tra symbol không trùng lặp
                if symbol in self.excluded_symbols:
                    continue
                
                # Kiểm tra điều kiện chiến lược
                if self._meets_strategy_requirements(symbol, strategy_config, allocation_rules):
                    # Thêm bot mới
                    self._add_new_bot(symbol, strategy_config)
                    allocated_count += 1
                    
                    # Chờ bot này đặt lệnh trước khi tiếp tục
                    time.sleep(2)  # Delay ngắn để tránh rate limit
                    
            self.log(f"✅ Đã phân bổ {allocated_count} bot theo chiến lược {strategy_name}")
            
        except Exception as e:
            self.log(f"Lỗi khi scan và allocate bots: {e}")

    def _get_strategy_config(self, strategy_name, allocation_rules):
        """Lấy config chiến lược"""
        return {
            'strategy_type': strategy_name,
            'leverage': allocation_rules.get('leverage', 10),
            'percent': allocation_rules.get('percent', 5),
            'tp': allocation_rules.get('tp', 100),
            'sl': allocation_rules.get('sl', 50),
            'threshold': allocation_rules.get('threshold', 30),
            'volatility': allocation_rules.get('volatility', 3),
            'grid_levels': allocation_rules.get('grid_levels', 5)
        }

    def _meets_strategy_requirements(self, symbol, strategy_config, allocation_rules):
        """
        Kiểm tra symbol có đáp ứng yêu cầu chiến lược không
        """
        try:
            # Kiểm tra volume
            volume_ok = self._check_volume_requirement(symbol, allocation_rules.get('min_volume', 1000000))
            
            # Kiểm tra biến động giá
            volatility_ok = self._check_volatility(symbol, allocation_rules.get('max_volatility', 10))
            
            # Kiểm tra điều kiện chiến lược cụ thể
            strategy_ok = check_strategy_conditions(symbol, strategy_config)
            
            return volume_ok and volatility_ok and strategy_ok
            
        except Exception as e:
            self.log(f"Lỗi kiểm tra requirements {symbol}: {e}")
            return False

    def _check_volume_requirement(self, symbol, min_volume):
        """Kiểm tra volume"""
        ticker_info = get_24h_ticker(symbol)
        if ticker_info and ticker_info['volume'] >= min_volume:
            return True
        return False

    def _check_volatility(self, symbol, max_volatility):
        """Kiểm tra biến động"""
        klines = get_klines_with_cache(symbol, "5m", 20)
        if klines:
            closes = klines[3]
            volatility_std = np.std(closes) / np.mean(closes) * 100
            return volatility_std <= max_volatility
        return False

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
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES THÔNG MINH</b>\n\n🎯 <b>HỆ THỐNG ĐA CHIẾN LƯỢC + BOT ĐỘNG TỰ TÌM COIN</b>"
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

    def _find_qualified_symbols(self, strategy_type, leverage, config, strategy_key):
        try:
            threshold = config.get('threshold', 30)
            volatility = config.get('volatility', 3)
            grid_levels = config.get('grid_levels', 5)
            
            qualified_symbols = get_qualified_symbols(
                self.api_key, self.api_secret, strategy_type, leverage,
                threshold, volatility, grid_levels, 
                max_candidates=20, 
                final_limit=2,
                strategy_key=strategy_key
            )
            
            return qualified_symbols
            
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            return []

    def _auto_scan_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                # CẬP NHẬT: Sử dụng cơ chế mới để quản lý bot
                for strategy_key, strategy_config in self.auto_strategies.items():
                    self.update_bot_list(strategy_config)
                
                # AUTO SCAN CHO CÁC CHIẾN LƯỢC TỰ ĐỘNG
                if current_time - self.last_auto_scan > self.auto_scan_interval:
                    self._scan_auto_strategies()
                    self.last_auto_scan = current_time
                
                time.sleep(60)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(60)

    def _scan_auto_strategies(self):
        if not self.auto_strategies:
            return
            
        self.log("🔄 Đang quét coin cho các cấu hình tự động...")
        
        for strategy_key, strategy_config in self.auto_strategies.items():
            try:
                strategy_type = strategy_config['strategy_type']
                
                if self._is_in_cooldown(strategy_type, strategy_key):
                    continue
                
                # SỬ DỤNG CƠ CHẾ MỚI: Quét và phân bổ bot
                allocation_rules = {
                    'min_volume': 1000000,
                    'max_volatility': 10,
                    'leverage': strategy_config.get('leverage', 10),
                    'percent': strategy_config.get('percent', 5),
                    'tp': strategy_config.get('tp', 100),
                    'sl': strategy_config.get('sl', 50)
                }
                
                self.scan_and_allocate_bots(strategy_type, allocation_rules)
                        
            except Exception as e:
                self.log(f"❌ Lỗi quét {strategy_type}: {str(e)}")

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
            
            bot_id = f"{symbol}_{strategy_key}"
            self.bots[bot_id] = bot
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, **kwargs):
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        # KIỂM TRA SỐ DƯ TỐI THIỂU
        min_balance_required = 10
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None or test_balance < min_balance_required:
            self.log(f"❌ Số dư {test_balance:.2f} USDT không đủ (tối thiểu: {min_balance_required} USDT)")
            return False
        
        bot_mode = kwargs.get('bot_mode', 'static')
        
        # BOT ĐỘNG THÔNG MINH - SỬ DỤNG CƠ CHẾ MỚI
        if strategy_type == "Smart Dynamic":
            strategy_key = f"SmartDynamic_{lev}_{percent}_{tp}_{sl}"
            
            if self._is_in_cooldown("Smart Dynamic", strategy_key):
                self.log(f"⏰ Smart Dynamic (Config: {strategy_key}): đang trong cooldown")
                return False
            
            self.auto_strategies[strategy_key] = {
                'strategy_type': "Smart Dynamic",
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key
            }
            
            # SỬ DỤNG CƠ CHẾ MỚI: Quét và phân bổ
            allocation_rules = {
                'min_volume': 1000000,
                'max_volatility': 10,
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl
            }
            
            self.scan_and_allocate_bots("Smart Dynamic", allocation_rules)
            return True
        
        # CÁC CHIẾN LƯỢC ĐỘNG KHÁC
        elif bot_mode == 'dynamic' and strategy_type in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following"]:
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
                self.log(f"⏰ {strategy_type} (Config: {strategy_key}): đang trong cooldown")
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
            
            # SỬ DỤNG CƠ CHẾ MỚI: Quét và phân bổ
            allocation_rules = {
                'min_volume': 1000000,
                'max_volatility': 10,
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'threshold': kwargs.get('threshold', 30),
                'volatility': kwargs.get('volatility', 3),
                'grid_levels': kwargs.get('grid_levels', 5)
            }
            
            self.scan_and_allocate_bots(strategy_type, allocation_rules)
            return True
        
        # CHIẾN LƯỢC TĨNH
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
            bot.stop()
            self.log(f"⛔ Đã dừng bot {bot_id}")
            del self.bots[bot_id]
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
        
        # XỬ LÝ CÁC BƯỚC TẠO BOT
        if current_step:
            if current_step == 'waiting_bot_mode':
                if text in ['🤖 Bot Tĩnh - Coin cụ thể', '🔄 Bot Động - Tự tìm coin']:
                    user_state['bot_mode'] = 'static' if text == '🤖 Bot Tĩnh - Coin cụ thể' else 'dynamic'
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 Chọn chiến lược:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                elif text == '❌ Hủy bỏ':
                    self.user_states.pop(chat_id, None)
                    self.send_main_menu(chat_id)
                return

            elif current_step == 'waiting_strategy':
                strategy_map = {
                    '🤖 RSI/EMA Recursive': 'RSI/EMA Recursive',
                    '📊 EMA Crossover': 'EMA Crossover', 
                    '🎯 Reverse 24h': 'Reverse 24h',
                    '📈 Trend Following': 'Trend Following',
                    '⚡ Scalping': 'Scalping',
                    '🛡️ Safe Grid': 'Safe Grid',
                    '🔄 Bot Động Thông Minh': 'Smart Dynamic'
                }
                
                if text in strategy_map:
                    user_state['strategy'] = strategy_map[text]
                    user_state['step'] = 'waiting_leverage'
                    send_telegram(
                        f"🎯 {user_state['strategy']}\n\nChọn đòn bẩy:",
                        chat_id,
                        create_leverage_keyboard(user_state['strategy']),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                elif text == '❌ Hủy bỏ':
                    self.user_states.pop(chat_id, None)
                    self.send_main_menu(chat_id)
                return

            elif current_step == 'waiting_leverage':
                if text.endswith('x') and text[:-1].isdigit():
                    user_state['leverage'] = int(text[:-1])
                    user_state['step'] = 'waiting_percent'
                    send_telegram(
                        f"💰 Đòn bẩy: {user_state['leverage']}x\n\nChọn % số dư mỗi lệnh:",
                        chat_id,
                        create_percent_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                elif text == '❌ Hủy bỏ':
                    self.user_states.pop(chat_id, None)
                    self.send_main_menu(chat_id)
                return

            elif current_step == 'waiting_percent':
                if text.isdigit():
                    user_state['percent'] = int(text)
                    user_state['step'] = 'waiting_tp'
                    send_telegram(
                        f"📊 % số dư: {user_state['percent']}%\n\nChọn Take Profit (%):",
                        chat_id,
                        create_tp_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                elif text == '❌ Hủy bỏ':
                    self.user_states.pop(chat_id, None)
                    self.send_main_menu(chat_id)
                return

            elif current_step == 'waiting_tp':
                if text.isdigit():
                    user_state['tp'] = int(text)
                    user_state['step'] = 'waiting_sl'
                    send_telegram(
                        f"🎯 TP: {user_state['tp']}%\n\nChọn Stop Loss (%):",
                        chat_id,
                        create_sl_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                elif text == '❌ Hủy bỏ':
                    self.user_states.pop(chat_id, None)
                    self.send_main_menu(chat_id)
                return

            elif current_step == 'waiting_sl':
                if text.isdigit():
                    user_state['sl'] = int(text)
                    
                    # HOÀN THÀNH CẤU HÌNH - TẠO BOT
                    strategy = user_state['strategy']
                    lev = user_state['leverage']
                    percent = user_state['percent']
                    tp = user_state['tp']
                    sl = user_state['sl']
                    bot_mode = user_state.get('bot_mode', 'static')
                    
                    success = False
                    if bot_mode == 'static' and strategy in ['RSI/EMA Recursive', 'EMA Crossover']:
                        user_state['step'] = 'waiting_symbol'
                        send_telegram(
                            f"✅ Cấu hình hoàn tất!\n\nChiến lược: {strategy}\nĐòn bẩy: {lev}x\n% số dư: {percent}%\nTP: {tp}% | SL: {sl}%\n\nChọn coin:",
                            chat_id,
                            create_symbols_keyboard(strategy),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                        return
                    else:
                        # BOT ĐỘNG - SỬ DỤNG CƠ CHẾ MỚI
                        success = self.add_bot(
                            symbol=None,  # Bot động tự tìm coin
                            lev=lev,
                            percent=percent,
                            tp=tp,
                            sl=sl,
                            strategy_type=strategy,
                            bot_mode=bot_mode,
                            threshold=user_state.get('threshold'),
                            volatility=user_state.get('volatility'),
                            grid_levels=user_state.get('grid_levels')
                        )
                    
                    if success:
                        send_telegram(
                            f"✅ <b>ĐÃ TẠO BOT THÀNH CÔNG!</b>\n\n"
                            f"🎯 Chiến lược: {strategy}\n"
                            f"🤖 Chế độ: {bot_mode}\n"
                            f"💰 Đòn bẩy: {lev}x\n"
                            f"📊 % số dư: {percent}%\n"
                            f"🎯 TP: {tp}%\n"
                            f"🛡️ SL: {sl}%",
                            chat_id,
                            create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    else:
                        send_telegram(
                            "❌ <b>LỖI TẠO BOT!</b>\nVui lòng kiểm tra cấu hình và thử lại.",
                            chat_id,
                            create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    
                    self.user_states.pop(chat_id, None)
                    
                elif text == '❌ Hủy bỏ':
                    self.user_states.pop(chat_id, None)
                    self.send_main_menu(chat_id)
                return

            elif current_step == 'waiting_symbol':
                if text != '❌ Hủy bỏ':
                    # TẠO BOT TĨNH VỚI COIN CỤ THỂ
                    strategy = user_state['strategy']
                    lev = user_state['leverage']
                    percent = user_state['percent']
                    tp = user_state['tp']
                    sl = user_state['sl']
                    
                    success = self.add_bot(
                        symbol=text,
                        lev=lev,
                        percent=percent,
                        tp=tp,
                        sl=sl,
                        strategy_type=strategy,
                        bot_mode='static'
                    )
                    
                    if success:
                        send_telegram(
                            f"✅ <b>ĐÃ TẠO BOT {text} THÀNH CÔNG!</b>\n\n"
                            f"🎯 Chiến lược: {strategy}\n"
                            f"💰 Đòn bẩy: {lev}x\n"
                            f"📊 % số dư: {percent}%\n"
                            f"🎯 TP: {tp}%\n"
                            f"🛡️ SL: {sl}%",
                            chat_id,
                            create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    else:
                        send_telegram(
                            f"❌ <b>LỖI TẠO BOT {text}!</b>\nCoin có thể không khả dụng hoặc đã có bot chạy.",
                            chat_id,
                            create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                
                self.user_states.pop(chat_id, None)
                return

        # XỬ LÝ CÁC LỆNH CHÍNH
        if text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_mode'}
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(
                f"🎯 <b>CHỌN CHẾ ĐỘ BOT</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"🤖 <b>Bot Tĩnh:</b>\n• Giao dịch coin CỐ ĐỊNH\n• Bạn chọn coin cụ thể\n\n"
                f"🔄 <b>Bot Động:</b>\n• TỰ ĐỘNG tìm coin tốt nhất\n• Tự tìm coin mới sau khi đóng lệnh",
                chat_id,
                create_bot_mode_keyboard(),
                self.telegram_bot_token, self.telegram_chat_id
            )
        
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                dynamic_bots = 0
                for bot_id, bot in self.bots.items():
                    status = "🟢 Mở" if bot.status == "open" else "🟡 Chờ"
                    mode = "Tĩnh"
                    if hasattr(bot, 'config_key') and bot.config_key:
                        mode = "Động"
                        dynamic_bots += 1
                    
                    message += f"🔹 {bot_id} | {status} | {mode} | ĐB: {bot.lev}x\n"
                
                message += f"\n📊 Tổng số: {len(self.bots)}/{self.max_bots} bot | 🔄 Động: {dynamic_bots}"
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
                "🎯 <b>DANH SÁCH CHIẾN LƯỢC NÂNG CAO</b>\n\n"
                
                "🔄 <b>Bot Động Thông Minh</b>\n"
                "• Kết hợp đa chỉ báo: RSI, EMA, MACD, Bollinger Bands\n"
                "• Tự động tìm coin tốt nhất\n"
                "• Hệ thống điểm đánh giá thông minh\n\n"
                
                "🎯 <b>Reverse 24h</b> - TỰ ĐỘNG\n"
                "• Đảo chiều biến động 24h mạnh\n"
                "• Tìm coin có biến động ≥ 25%\n"
                "• Quản lý rủi ro thông minh\n\n"
                
                "⚡ <b>Scalping</b> - TỰ ĐỘNG\n"
                "• Giao dịch tốc độ cao với RSI ngắn\n"
                "• Lọc theo volatility và ATR\n"
                "• Tối ưu cho coin biến động\n\n"
                
                "🛡️ <b>Safe Grid</b> - TỰ ĐỘNG\n"
                "• Grid an toàn với số lệnh cố định\n"
                "• Tìm coin ổn định, biến động thấp\n"
                "• Phân bổ vốn thông minh\n\n"
                
                "📈 <b>Trend Following</b> - TỰ ĐỘNG\n"
                "• Theo xu hướng đa khung thời gian\n"
                "• Xác nhận bằng ATR\n"
                "• Bắt trend sớm và an toàn\n\n"
                
                "🤖 <b>RSI/EMA Recursive</b> - TĨNH\n"
                "• Kết hợp RSI, EMA, MACD, Stochastic\n"
                "• Hệ thống xác nhận đa tầng\n"
                "• Tối ưu cho swing trading\n\n"
                
                "📊 <b>EMA Crossover</b> - TĨNH\n"
                "• Giao cắt EMA với Bollinger Bands\n"
                "• Xác nhận xu hướng dài hạn\n"
                "• Filter nhiễu hiệu quả"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            dynamic_bots_count = sum(1 for bot in self.bots.values() 
                                   if hasattr(bot, 'config_key') and bot.config_key)
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG NÂNG CAO</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}/{self.max_bots}\n"
                f"🔄 Bot động: {dynamic_bots_count}\n"
                f"📊 Chiến lược: {len(set(bot.strategy_name for bot in self.bots.values()))}\n"
                f"🔄 Auto scan: {len(self.auto_strategies)} cấu hình\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n"
                f"⏰ Cooldown: {self.cooldown_period//60} phút\n"
                f"💡 Phiên bản: Enhanced Bot Manager v2.0"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()
