# ==============================================================
# TRADING BOT HOÀN CHỈNH - PHIÊN BẢN TỐI ƯU (PHẦN 1)
# ==============================================================
# Gộp 3 file thành 1 file duy nhất, tối ưu hiệu năng và tính năng
# Bao gồm: Bot động thông minh, Smart Exit, Scanner, Menu Telegram
# ==============================================================

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
from heapq import nlargest
from concurrent.futures import ThreadPoolExecutor

# ==============================================================
# CẤU HÌNH LOGGING & TELEGRAM
# ==============================================================

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

# ==============================================================
# MENU TELEGRAM TỐI ƯU
# ==============================================================

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
            [{"text": "🔄 Thoát lệnh thông minh"}, {"text": "⚡ Thoát lệnh cơ bản"}],
            [{"text": "🎯 Chỉ TP/SL cố định"}, {"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_smart_exit_config_keyboard():
    return {
        "keyboard": [
            [{"text": "Trailing: 30/15"}, {"text": "Trailing: 50/20"}],
            [{"text": "Time Exit: 4h"}, {"text": "Time Exit: 8h"}],
            [{"text": "Kết hợp Full"}, {"text": "Cơ bản"}],
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

# ==============================================================
# BINANCE API HELPERS TỐI ƯU
# ==============================================================

BASE_FAPI = "https://fapi.binance.com"

def signed_request(url_path, params, api_key, secret_key, method='GET'):
    query = urllib.parse.urlencode(params)
    signature = hmac.new(secret_key.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    url = f"{BASE_FAPI}{url_path}?{query}&signature={signature}"
    req = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def binance_api_request(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        logger.error(f"GET {url} {r.status_code}: {r.text}")
        return None
    except Exception as e:
        logger.error(f"GET error {url}: {e}")
        return None

def get_all_usdt_pairs(limit=200):
    url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
    data = binance_api_request(url)
    out = []
    if data and 'symbols' in data:
        for sym in data['symbols']:
            if sym.get('contractType') in ("PERPETUAL","CURRENT_QUARTER","NEXT_QUARTER") and sym.get('quoteAsset') == 'USDT':
                out.append(sym['symbol'])
    return out[:limit]

def get_balance(api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v2/balance"
        params = {"timestamp": ts}
        data = signed_request(url_path, params, api_key, api_secret, 'GET')
        if data:
            for asset in data:
                if asset.get('asset') == 'USDT':
                    return float(asset.get('availableBalance', 0))
        return 0
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        return None

def set_leverage(symbol, leverage, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v1/leverage"
        params = {"symbol": symbol.upper(), "leverage": leverage, "timestamp": ts}
        signed_request(url_path, params, api_key, api_secret, 'POST')
        return True
    except Exception as e:
        logger.error(f"set_leverage({symbol}) lỗi: {e}")
        return False

def get_step_size(symbol, api_key, api_secret):
    try:
        url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data: return 0
        for s in data.get('symbols', []):
            if s.get('symbol') == symbol:
                for f in s.get('filters', []):
                    if f.get('filterType') == 'LOT_SIZE':
                        return float(f.get('stepSize', '0'))
        return 0
    except Exception:
        return 0

def cancel_all_orders(symbol, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v1/allOpenOrders"
        params = {"symbol": symbol.upper(), "timestamp": ts}
        signed_request(url_path, params, api_key, api_secret, 'DELETE')
        return True
    except Exception:
        return False

def place_order(symbol, side, quantity, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v1/order"
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "timestamp": ts
        }
        return signed_request(url_path, params, api_key, api_secret, 'POST')
    except Exception as e:
        logger.error(f"place_order lỗi: {e}")
        return None

def get_current_price(symbol):
    try:
        url = f"{BASE_FAPI}/fapi/v1/ticker/price"
        data = binance_api_request(url, {"symbol": symbol.upper()})
        if data and 'price' in data:
            return float(data['price'])
        return 0.0
    except Exception:
        return 0.0

def get_positions(symbol=None, api_key=None, api_secret=None):
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v2/account"
        params = {"timestamp": ts}
        data = signed_request(url_path, params, api_key, api_secret, 'GET')
        if not data:
            return []
        
        positions = []
        for pos in data.get('positions', []):
            if float(pos.get('positionAmt', 0)) != 0:
                if symbol and pos.get('symbol') != symbol.upper():
                    continue
                positions.append({
                    'symbol': pos.get('symbol'),
                    'positionAmt': float(pos.get('positionAmt', 0)),
                    'entryPrice': float(pos.get('entryPrice', 0)),
                    'unRealizedProfit': float(pos.get('unRealizedProfit', 0))
                })
        return positions
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
    return []

def get_24h_change(symbol):
    try:
        url = f"{BASE_FAPI}/fapi/v1/ticker/24hr?symbol={symbol.upper()}"
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

def get_klines(symbol, interval, limit):
    try:
        url = f"{BASE_FAPI}/fapi/v1/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        data = binance_api_request(url, params)
        if not data:
            return None
        
        # Trả về OHLCV
        opens = [float(candle[1]) for candle in data]
        highs = [float(candle[2]) for candle in data]
        lows = [float(candle[3]) for candle in data]
        closes = [float(candle[4]) for candle in data]
        volumes = [float(candle[5]) for candle in data]
        
        return (opens, highs, lows, closes, volumes)
    except Exception as e:
        logger.error(f"Lỗi lấy klines {symbol}: {str(e)}")
        return None

# ==============================================================
# CHỈ BÁO KỸ THUẬT NÂNG CAO
# ==============================================================

def rsi_wilder_last(prices, period=14):
    if len(prices) < period + 1: return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    for i in range(period, len(deltas)):
        gain, loss = gains[i], losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - 100 / (1 + rs)
    return float(rsi)

def ema_last(values, period):
    if len(values) < period: return None
    k = 2/(period+1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v*k + ema*(1-k)
    return float(ema)

def atr_last(highs, lows, closes, period=14, return_pct=True):
    n = len(closes)
    if min(len(highs), len(lows), len(closes)) < period+1: return None
    trs = []
    for i in range(1, n):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr*(period-1) + trs[i]) / period
    return float(atr / closes[-1] * 100.0) if return_pct else float(atr)

def adx_last(highs, lows, closes, period=14):
    n = len(closes)
    if min(len(highs), len(lows), len(closes)) < period+1: return None, None, None
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        up = highs[i]-highs[i-1]
        down = lows[i-1]-lows[i]
        plus_dm.append(up if (up>down and up>0) else 0.0)
        minus_dm.append(down if (down>up and down>0) else 0.0)
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    def wilder(arr):
        a = sum(arr[:period]) / period
        for x in arr[period:]:
            a = (a*(period-1) + x) / period
        return a
    atr = wilder(trs)
    if atr == 0: return None, None, None
    plus_di = 100 * (wilder(plus_dm) / atr)
    minus_di = 100 * (wilder(minus_dm) / atr)
    dx = 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9)
    adx = dx
    return float(adx), float(plus_di), float(minus_di)

def bbands_last(closes, period=20, std=2):
    if len(closes) < period: return None, None, None
    w = np.array(closes[-period:])
    mid = w.mean(); dev = w.std(ddof=0)
    return float(mid), float(mid+std*dev), float(mid-std*dev)

def mfi_last(highs, lows, closes, volumes, period=14):
    n = len(closes)
    if min(len(highs), len(lows), len(closes), len(volumes)) < period+1: return None
    tp = (np.array(highs) + np.array(lows) + np.array(closes)) / 3.0
    raw = tp * np.array(volumes)
    pos, neg = [], []
    for i in range(1, n):
        if tp[i] > tp[i-1]: pos.append(raw[i]); neg.append(0.0)
        elif tp[i] < tp[i-1]: pos.append(0.0); neg.append(raw[i])
        else: pos.append(0.0); neg.append(0.0)
    if len(pos) < period: return None
    p = sum(pos[-period:]); q = sum(neg[-period:])
    if q == 0: return 100.0
    mr = p/q
    return float(100 - 100/(1+mr))

def obv_last(closes, volumes):
    if len(closes) < 2 or len(volumes) < 2: return 0.0
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]: obv += volumes[i]
        elif closes[i] < closes[i-1]: obv -= volumes[i]
    return float(obv)

def detect_regime(highs, lows, closes):
    adx, _, _ = adx_last(highs, lows, closes, 14)
    atrp = atr_last(highs, lows, closes, 14, True)
    regime = 'unknown'
    if adx is not None:
        if adx >= 25: regime = 'trend'
        elif adx < 20: regime = 'range'
        else: regime = 'transition'
    if atrp is not None and atrp >= 5.0: regime += '|hi-vol'
    return regime, (adx or 0.0), (atrp or 0.0)

# ==============================================================
# COIN MANAGER VỚI COOLDOWN NÂNG CAO
# ==============================================================

class CoinManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._instance.symbol_to_bot = {}
            cls._instance.symbol_configs = {}
            cls._instance.active_configs = {}
            # COOLDOWN nâng cấp
            cls._instance.symbol_cooldowns = {}
            cls._instance.cooldown_seconds = 20*60
        return cls._instance

    def register_coin(self, symbol, bot_id, strategy_name, config_key=None):
        with self._lock:
            if symbol in self.symbol_to_bot:
                return False
            self.symbol_to_bot[symbol] = bot_id
            self.symbol_configs[symbol] = (strategy_name, config_key)
            key = (strategy_name, config_key)
            if key not in self.active_configs:
                self.active_configs[key] = set()
            self.active_configs[key].add(symbol)
            return True

    def unregister_coin(self, symbol):
        with self._lock:
            if symbol in self.symbol_to_bot:
                del self.symbol_to_bot[symbol]
            if symbol in self.symbol_configs:
                tup = self.symbol_configs.pop(symbol)
                key = (tup[0], tup[1])
                if key in self.active_configs and symbol in self.active_configs[key]:
                    self.active_configs[key].remove(symbol)

    def has_same_config_bot(self, symbol, config_key):
        with self._lock:
            if symbol not in self.symbol_configs: return False
            _, cfg = self.symbol_configs.get(symbol, (None, None))
            return cfg == config_key

    def is_coin_managed(self, symbol):
        with self._lock:
            return symbol in self.symbol_to_bot

    def count_bots_by_config(self, config_key):
        with self._lock:
            count = 0
            for coin_info in self.symbol_configs.values():
                if coin_info[1] == config_key:
                    count += 1
            return count

    def get_managed_coins(self):
        with self._lock:
            return self.symbol_to_bot.copy()

    # COOLDOWN API
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

    def cooldown_left(self, symbol):
        with self._lock:
            ts = self.symbol_cooldowns.get(symbol.upper())
            return max(0, int(ts - time.time())) if ts else 0

# ==============================================================
# WEBSOCKET MANAGER TỐI ƯU
# ==============================================================

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

# ==============================================================
# SMART EXIT MANAGER NÂNG CAO
# ==============================================================

class SmartExitManager:
    """QUẢN LÝ THÔNG MINH 4 CƠ CHẾ ĐÓNG LỆNH + NÂNG CẤP"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.config = {
            'enable_trailing': False,
            'enable_time_exit': False,
            'enable_volume_exit': False,
            'enable_support_resistance': False,
            'trailing_activation': 30,
            'trailing_distance': 15,
            'max_hold_time': 6,
            'min_profit_for_exit': 10,
            # Nâng cấp
            'breakeven_at': 12,
            'trail_adaptive': True,
            'tp_ladder': [
                {'roi': 15, 'pct': 0.30},
                {'roi': 25, 'pct': 0.30},
                {'roi': 40, 'pct': 0.40},
            ]
        }
        
        self.trailing_active = False
        self.peak_price = 0
        self.position_open_time = 0
        self.volume_history = []
        self._breakeven_active = False
        self._tp_hit = set()
        
    def update_config(self, **kwargs):
        """Cập nhật cấu hình từ người dùng"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        self.bot.log(f"⚙️ Cập nhật Smart Exit: {self.config}")
    
    def on_position_opened(self):
        """Khi mở position mới"""
        self.trailing_active = False
        self.peak_price = self.bot.entry
        self.position_open_time = time.time()
        self.volume_history = []
        self._breakeven_active = False
        self._tp_hit.clear()
    
    def _calculate_roi(self, current_price):
        """Tính ROI hiện tại"""
        if not self.bot.position_open or self.bot.entry <= 0 or abs(self.bot.qty) <= 0:
            return 0.0
            
        if self.bot.side == "BUY":
            profit = (current_price - self.bot.entry) * abs(self.bot.qty)
        else:
            profit = (self.bot.entry - current_price) * abs(self.bot.qty)
            
        invested = self.bot.entry * abs(self.bot.qty) / self.bot.lev
        if invested <= 0: return 0.0
        return (profit / invested) * 100.0

    def check_all_exit_conditions(self, current_price, current_volume=None):
        """KIỂM TRA TẤT CẢ ĐIỀU KIỆN ĐÓNG LỆNH"""
        if not self.bot.position_open:
            return None
            
        exit_reasons = []
        current_roi = self._calculate_roi(current_price)
        
        # 1. TRAILING STOP EXIT
        if self.config['enable_trailing']:
            reason = self._check_trailing_stop(current_price, current_roi)
            if reason:
                exit_reasons.append(reason)
        
        # 2. TIME-BASED EXIT
        if self.config['enable_time_exit']:
            reason = self._check_time_exit()
            if reason:
                exit_reasons.append(reason)
        
        # 3. VOLUME-BASED EXIT  
        if self.config['enable_volume_exit'] and current_volume:
            reason = self._check_volume_exit(current_volume)
            if reason:
                exit_reasons.append(reason)
        
        # 4. SUPPORT/RESISTANCE EXIT
        if self.config['enable_support_resistance']:
            reason = self._check_support_resistance(current_price, current_roi)
            if reason:
                exit_reasons.append(reason)

        # 5. BREAKEVEN + TP LADDER (NÂNG CẤP)
        if not self._breakeven_active and current_roi >= self.config.get('breakeven_at', 12):
            self._breakeven_active = True
            self.config['min_profit_for_exit'] = max(self.config.get('min_profit_for_exit', 10), 0)
            self.bot.log(f"🟩 Kích hoạt Breakeven tại ROI {current_roi:.1f}%")
        
        # TP LADDER - CHỐT LỜI TỪNG PHẦN
        for step in self.config.get('tp_ladder', []):
            roi_lv = step.get('roi', 0)
            pct = step.get('pct', 0)
            key = f"tp_{roi_lv}"
            
            if current_roi >= roi_lv and key not in self._tp_hit:
                ok = self.bot.partial_close(pct, reason=f"TP ladder {roi_lv}%")
                if ok:
                    self._tp_hit.add(key)
        
        # Chỉ đóng lệnh nếu đang có lãi đạt ngưỡng tối thiểu
        if exit_reasons and current_roi >= self.config['min_profit_for_exit']:
            return f"Smart Exit: {' + '.join(exit_reasons)} | Lãi: {current_roi:.1f}%"
        
        return None
    
    def _check_trailing_stop(self, current_price, current_roi):
        """Trailing Stop - Bảo vệ lợi nhuận nâng cao"""
        
        # Kích hoạt trailing khi đạt ngưỡng
        if current_roi >= self.config['trailing_activation'] and not self.trailing_active:
            self.trailing_active = True
            self.peak_price = current_price
            self.bot.log(f"🟢 Kích hoạt Trailing Stop | Lãi {current_roi:.1f}%")
        
        # Cập nhật đỉnh mới
        if self.trailing_active:
            if (self.bot.side == "BUY" and current_price > self.peak_price) or \
               (self.bot.side == "SELL" and current_price < self.peak_price):
                self.peak_price = current_price
            
            # Tính trailing distance (có thể adaptive)
            distance = self.config['trailing_distance']
            if self.config.get('trail_adaptive'):
                try:
                    closes = (self.bot.prices or [])[-120:]
                    if len(closes) >= 20:
                        highs = [p*1.0006 for p in closes]
                        lows  = [p*0.9994 for p in closes]
                        atrp = atr_last(highs, lows, closes, 14, True)
                        if atrp is not None:
                            distance = 8 + min(max(atrp, 1.0), 10.0) * 1.2
                except Exception:
                    pass
            
            # Tính drawdown từ đỉnh
            if self.bot.side == "BUY":
                trigger = self.peak_price * (1 - distance/100.0)
                if current_price <= trigger:
                    return f"🔻 Trailing({distance:.1f}%)"
            else:
                trigger = self.peak_price * (1 + distance/100.0)
                if current_price >= trigger:
                    return f"🔻 Trailing({distance:.1f}%)"
        
        return None
    
    def _check_time_exit(self):
        """Time-based Exit - Giới hạn thời gian giữ lệnh"""
        if self.position_open_time == 0:
            return None
            
        holding_hours = (time.time() - self.position_open_time) / 3600
        
        if holding_hours >= self.config['max_hold_time']:
            return f"⏰ Time({holding_hours:.1f}h)"
        
        return None
    
    def _check_volume_exit(self, current_volume):
        """Volume-based Exit - Theo dấu hiệu volume"""
        if len(self.volume_history) < 5:
            self.volume_history.append(current_volume)
            return None
        
        avg_volume = sum(self.volume_history[-5:]) / 5
        
        if current_volume < avg_volume * 0.4:
            return "📊 Volume(giảm 60%)"
        
        self.volume_history.append(current_volume)
        if len(self.volume_history) > 10:
            self.volume_history.pop(0)
            
        return None
    
    def _check_support_resistance(self, current_price, current_roi):
        """Support/Resistance Exit - Theo key levels"""
        if self.bot.side == "BUY" and current_roi >= 5.0:
            return f"🎯 Resistance(+{current_roi:.1f}%)"
        elif self.bot.side == "SELL" and current_roi >= 5.0:
            return f"🎯 Support(+{current_roi:.1f}%)"
        
        return None

# ==============================================================
# BASE BOT NÂNG CẤP VỚI TÍNH NĂNG TÌM COIN MỚI
# ==============================================================

class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None,
                 smart_exit_config=None, dynamic_mode=False):
        
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
        self.dynamic_mode = dynamic_mode  # 🔄 CHẾ ĐỘ ĐỘNG
        
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
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 300
        self.position_check_interval = 30
        
        # Bảo vệ chống lặp đóng lệnh
        self._close_attempted = False
        self._last_close_attempt = 0
        
        # Cờ đánh dấu cần xóa bot
        self.should_be_removed = False
        
        # Ứng viên coin dự phòng
        self._candidate_pool = []
        
        # HỆ THỐNG SMART EXIT
        self.smart_exit = SmartExitManager(self)
        if smart_exit_config:
            self.smart_exit.update_config(**smart_exit_config)
        
        self.coin_manager = CoinManager()
        if symbol:
            success = self.coin_manager.register_coin(self.symbol, f"{strategy_name}_{id(self)}", strategy_name, config_key)
            if not success:
                self.log(f"⚠️ Cảnh báo: {self.symbol} đã được quản lý bởi bot khác")
        
        self.check_position_status()
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        mode_text = "ĐỘNG - Tự tìm coin" if dynamic_mode else "TĨNH - Coin cố định"
        self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | Chế độ: {mode_text} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")

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
            if len(self.prices) > 1000:
                self.prices.pop(0)
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
        self.smart_exit.trailing_active = False
        self.smart_exit._breakeven_active = False
        self.smart_exit._tp_hit.clear()

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                
                if self.should_be_removed:
                    self.log("🛑 Bot đã được đánh dấu xóa, dừng hoạt động")
                    time.sleep(1)
                    continue
                
                if not self.position_open:
                    signal = self.get_signal()
                    
                    if (signal and 
                        current_time - self.last_trade_time > 60 and
                        current_time - self.last_close_time > self.cooldown_period and
                        not self.should_be_removed):
                        
                        self.log(f"🎯 Nhận tín hiệu {signal}, đang mở lệnh...")
                        if self.open_position(signal):
                            self.last_trade_time = current_time
                        else:
                            time.sleep(30)
                
                if self.position_open and not self._close_attempted and not self.should_be_removed:
                    self.check_tp_sl()
                    
                time.sleep(1)
                
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"❌ Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        self.ws_manager.remove_symbol(self.symbol)
        self.coin_manager.unregister_coin(self.symbol)
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

            balance_data = get_balance(self.api_key, self.api_secret)
            if balance_data is None:
                self.log("❌ Không đủ số dư")
                return False

            usdt_balance = 0
            for asset in balance_data:
                if asset.get('asset') == 'USDT':
                    usdt_balance = float(asset.get('availableBalance', 0))
                    break

            if usdt_balance <= 0:
                self.log("❌ Không đủ số dư USDT")
                return False

            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                self.log("❌ Lỗi lấy giá")
                return False

            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = usdt_balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                precision = int(round(-math.log10(step_size))) if step_size < 1 else 0
                qty = float(f"{qty:.{precision}f}")

            if qty <= 0:
                self.log(f"❌ Số lượng quá nhỏ: {qty}")
                return False

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
                    
                    # KÍCH HOẠT SMART EXIT KHI MỞ LỆNH
                    self.smart_exit.on_position_opened()
                    
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

    def partial_close(self, fraction, reason=""):
        """Chốt một phần vị thế - TÍNH NĂNG NÂNG CAO"""
        try:
            if not self.position_open or fraction <= 0 or fraction >= 1:
                return False
                
            qty_to_close = abs(self.qty) * fraction
            if qty_to_close <= 0:
                return False
                
            side = "SELL" if self.side == "BUY" else "BUY"
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.2)
            
            result = place_order(self.symbol, side, qty_to_close, self.api_key, self.api_secret)
            if result and 'orderId' in result:
                remain = abs(self.qty) - qty_to_close
                self.qty = remain if self.side == "BUY" else -remain
                self.log(f"🔹 Chốt {fraction*100:.0f}% vị thế | {reason}")
                
                if remain <= 0:
                    self._reset_position()
                    self.last_close_time = time.time()
                    
                return True
            return False
            
        except Exception as e:
            self.log(f"❌ Lỗi partial_close: {str(e)}")
            return False

    def close_position(self, reason=""):
        if not self.position_open or self._close_attempted:
            return False
            
        try:
            self._close_attempted = True
            self._last_close_attempt = time.time()

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
                
                old_symbol = self.symbol
                self._reset_position()
                self.last_close_time = time.time()
                
                # SET COOLDOWN CHO COIN CŨ
                try:
                    self.coin_manager.set_cooldown(old_symbol)
                    self.log(f"⏳ COOLDOWN {old_symbol} ({self.coin_manager.cooldown_left(old_symbol)}s)")
                except Exception:
                    pass
                
                # 🔄 QUAN TRỌNG: BOT ĐỘNG TÌM COIN MỚI SAU KHI ĐÓNG LỆNH
                if self.dynamic_mode:
                    self.log("🔄 Bot động: Đang tìm coin mới...")
                    threading.Thread(target=self._find_new_coin_after_exit, daemon=True).start()
                else:
                    self.should_be_removed = True
                
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

    def _find_new_coin_after_exit(self):
        """🔄 TÌM COIN MỚI CHO BOT ĐỘNG SAU KHI ĐÓNG LỆNH - TỐI ƯU"""
        try:
            self.log("🔄 Bot động đang tìm coin mới (2 ứng viên)...")
            
            # Ưu tiên dùng ứng viên từ pool trước
            if self._candidate_pool:
                cached_symbol = self._candidate_pool.pop(0)
                if not self.coin_manager.is_in_cooldown(cached_symbol):
                    old_symbol = self.symbol
                    self.coin_manager.unregister_coin(old_symbol)
                    self.symbol = cached_symbol
                    
                    if self.coin_manager.register_coin(cached_symbol, f"{self.strategy_name}_{id(self)}", 
                                                     self.strategy_name, self.config_key):
                        self._restart_websocket_for_new_coin()
                        self.log(f"🔁 Dùng ứng viên sẵn có: {old_symbol} → {cached_symbol}")
                        return
            
            # Tìm coin phù hợp mới
            new_symbols = get_qualified_symbols(
                self.api_key, 
                self.api_secret,
                self.strategy_name,
                self.lev,
                threshold=getattr(self, 'threshold', None),
                volatility=getattr(self, 'volatility', None),
                grid_levels=getattr(self, 'grid_levels', None),
                max_candidates=12,
                final_limit=2,
                strategy_key=self.config_key
            )
            
            if new_symbols:
                primary_symbol = new_symbols[0]
                backup_symbol = new_symbols[1] if len(new_symbols) > 1 else None
                
                old_symbol = self.symbol
                self.coin_manager.unregister_coin(old_symbol)
                self.symbol = primary_symbol
                
                registered = self.coin_manager.register_coin(
                    primary_symbol, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key
                )
                
                if registered:
                    self._restart_websocket_for_new_coin()
                    
                    msg = f"🔄 Chuyển {old_symbol} → {primary_symbol}"
                    if backup_symbol:
                        self._candidate_pool = [backup_symbol]
                        msg += f" | Backup: {backup_symbol}"
                    
                    self.log(msg)
                    
                    # KHÔNG đánh dấu xóa bot - BOT VẪN TIẾP TỤC CHẠY
                    self.should_be_removed = False
                else:
                    self.log(f"❌ Không thể đăng ký coin mới {primary_symbol}")
                    # Quay lại coin cũ nếu không đăng ký được
                    self.symbol = old_symbol
                    self.coin_manager.register_coin(old_symbol, f"{self.strategy_name}_{id(self)}", 
                                                  self.strategy_name, self.config_key)
            else:
                self.log("❌ Không tìm thấy coin mới phù hợp, giữ nguyên coin hiện tại")
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {str(e)}")
            traceback.print_exc()

    def _restart_websocket_for_new_coin(self):
        """Khởi động lại WebSocket cho coin mới"""
        try:
            self.ws_manager.remove_symbol(self.symbol)
            time.sleep(2)
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
            self.log(f"🔗 Khởi động lại WebSocket cho {self.symbol}")
        except Exception as e:
            self.log(f"❌ Lỗi khởi động lại WebSocket: {str(e)}")

    def check_tp_sl(self):
        """KIỂM TRA SMART EXIT + TP/SL TRUYỀN THỐNG"""
        # 1. KIỂM TRA SMART EXIT TRƯỚC
        if self.position_open and self.entry > 0:
            current_price = get_current_price(self.symbol)
            if current_price > 0:
                exit_reason = self.smart_exit.check_all_exit_conditions(current_price)
                if exit_reason:
                    self.close_position(exit_reason)
                    return
        
        # 2. KIỂM TRA TP/SL TRUYỀN THỐNG
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

# ==============================================================
# SCANNER TÌM COIN PHÙ HỢP - TỐI ƯU
# ==============================================================

def _score_symbol_for_strategy(symbol, k5, k15, ticker, strategy_type):
    """Đánh giá và chấm điểm coin cho từng chiến lược"""
    o5, h5, l5, c5, v5 = k5
    o15, h15, l15, c15, v15 = k15
    
    # Tính các chỉ báo
    atr5 = atr_last(h5, l5, c5, 14, True) or 0.0
    adx15, pdi, mdi = adx_last(h15, l15, c15, 14)
    adx15 = adx15 or 0.0
    ema9_15 = ema_last(c15, 9) or c15[-1]
    ema21_15 = ema_last(c15, 21) or c15[-1]
    trend_bias = 1 if ema9_15 > ema21_15 else -1 if ema9_15 < ema21_15 else 0
    
    # Volume analysis
    vol_avg = np.mean(v5[-30:]) if len(v5) >= 30 else (sum(v5)/max(len(v5),1))
    vol_surge = (v5[-1]/max(vol_avg, 1e-9))
    
    # Thông tin từ ticker 24h
    abs_change = abs(float(ticker.get('priceChangePercent', 0.0) or 0.0))
    qvol = float(ticker.get('quoteVolume', 0.0) or 0.0)

    # Penalty cho BTC/ETH (tránh biến động quá cao)
    base_penalty = 0.0
    if symbol in ('BTCUSDT', 'ETHUSDT'):
        base_penalty = 0.5

    # Tính điểm theo chiến lược
    if strategy_type == "Reverse 24h":
        score = (abs_change/10.0) + min(vol_surge, 3.0) - base_penalty
        if atr5 >= 8.0: score -= 1.0  # Trừ điểm nếu biến động quá cao
        
    elif strategy_type == "Scalping":
        score = min(vol_surge, 3.0) + min(atr5/2.0, 2.0) - base_penalty
        if atr5 > 10.0: score -= 1.0
        
    elif strategy_type == "Safe Grid":
        mid_vol = 2.0 <= atr5 <= 6.0
        score = (1.5 if mid_vol else 0.5) + min(qvol/5_000_000, 3.0) - base_penalty
        
    elif strategy_type == "Trend Following":
        score = (adx15/25.0) + (1.0 if trend_bias > 0 else 0.0) + min(vol_surge, 3.0) - base_penalty
        
    elif strategy_type == "Smart Dynamic":
        base = (min(adx15, 40)/40.0) + min(vol_surge, 3.0)
        if trend_bias != 0 and 3.0 <= atr5 <= 8.0: base += 1.0
        score = base - base_penalty
        
    else:
        score = min(vol_surge, 3.0) + (abs_change/20.0)

    return score, {
        "atr5%": atr5, 
        "adx15": adx15, 
        "vol_surge": vol_surge, 
        "trend_bias": trend_bias,
        "24h_change": abs_change
    }

def get_qualified_symbols(api_key, api_secret, strategy_type, leverage, 
                          threshold=None, volatility=None, grid_levels=None, 
                          max_candidates=20, final_limit=2, strategy_key=None):
    """Tìm coin phù hợp từ TOÀN BỘ Binance - PHÂN BIỆT THEO CẤU HÌNH"""
    try:
        # Kiểm tra kết nối Binance
        test_balance = get_balance(api_key, api_secret)
        if test_balance is None:
            logger.error("❌ KHÔNG THỂ KẾT NỐI BINANCE")
            return []
        
        coin_manager = CoinManager()
        
        # Lấy danh sách coin từ Binance
        all_symbols = get_all_usdt_pairs(limit=300)
        if not all_symbols:
            logger.error("❌ Không lấy được danh sách coin từ Binance")
            return []
        
        # Lấy dữ liệu 24h
        url = f"{BASE_FAPI}/fapi/v1/ticker/24hr"
        all_tickers = binance_api_request(url)
        if not all_tickers:
            return []
            
        ticker_dict = {ticker['symbol']: ticker for ticker in all_tickers if 'symbol' in ticker}
        
        scored_symbols = []
        
        # Đánh giá từng coin
        for symbol in all_symbols:
            if symbol not in ticker_dict:
                continue
                
            # Loại trừ BTC và ETH để tránh biến động quá cao
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                continue
            
            # Kiểm tra coin đã được quản lý bởi config này chưa
            if strategy_key and coin_manager.has_same_config_bot(symbol, strategy_key):
                continue
            
            # Kiểm tra cooldown
            if coin_manager.is_in_cooldown(symbol):
                continue
                
            ticker = ticker_dict[symbol]
            
            try:
                # Kiểm tra volume tối thiểu
                qvol = float(ticker.get('quoteVolume', 0.0) or 0.0)
                if qvol < 1_000_000: 
                    continue
                
                # Lấy dữ liệu klines
                k5 = get_klines(symbol, "5m", 210)
                k15 = get_klines(symbol, "15m", 210)
                if not k5 or not k15:
                    continue
                
                # Tính điểm
                score, features = _score_symbol_for_strategy(symbol, k5, k15, ticker, strategy_type)
                
                # Áp dụng threshold cho Reverse 24h
                if strategy_type == "Reverse 24h" and threshold is not None:
                    abs_change = abs(float(ticker.get('priceChangePercent', 0.0) or 0.0))
                    if abs_change < threshold:
                        continue
                
                scored_symbols.append((score, symbol, features))
                
            except Exception as e:
                continue
        
        # Sắp xếp theo điểm
        top_symbols = nlargest(max_candidates, scored_symbols, key=lambda x: x[0])
        
        # Kiểm tra leverage và step size
        final_symbols = []
        for score, symbol, features in top_symbols:
            try:
                if not set_leverage(symbol, leverage, api_key, api_secret):
                    continue
                    
                step_size = get_step_size(symbol, api_key, api_secret)
                if step_size <= 0:
                    continue
                    
                final_symbols.append(symbol)
                logger.info(f"✅ {symbol}: score={score:.2f} | features={features}")
                
                if len(final_symbols) >= final_limit:
                    break
                    
            except Exception:
                continue
        
        # BACKUP SYSTEM: Nếu không tìm thấy coin phù hợp
        if not final_symbols:
            logger.warning(f"⚠️ {strategy_type}: không tìm thấy coin phù hợp, sử dụng backup method")
            backup_symbols = []
            
            for symbol in all_symbols:
                if symbol not in ticker_dict:
                    continue
                    
                # Kiểm tra coin đã được quản lý bởi config này chưa
                if strategy_key and coin_manager.has_same_config_bot(symbol, strategy_key):
                    continue
                    
                ticker = ticker_dict[symbol]
                try:
                    volume = float(ticker.get('quoteVolume', 0.0) or 0.0)
                    price_change = abs(float(ticker.get('priceChangePercent', 0.0) or 0.0))
                    
                    # Điều kiện backup: volume cao, biến động vừa phải
                    if (volume > 3_000_000 and 
                        0.5 <= price_change <= 10.0 and
                        symbol not in ['BTCUSDT', 'ETHUSDT']):
                        backup_symbols.append((volume, symbol))
                except:
                    continue
            
            # Sắp xếp theo volume giảm dần
            backup_symbols.sort(reverse=True)
            
            for volume, symbol in backup_symbols[:final_limit]:
                try:
                    if set_leverage(symbol, leverage, api_key, api_secret) and get_step_size(symbol, api_key, api_secret) > 0:
                        final_symbols.append(symbol)
                        logger.info(f"🔄 {symbol}: backup coin (Volume: {volume:.0f})")
                        if len(final_symbols) >= final_limit:
                            break
                except Exception:
                    continue
        
        logger.info(f"🎯 {strategy_type}: Kết quả cuối - {len(final_symbols)} coin: {final_symbols}")
        return final_symbols[:final_limit]
        
    except Exception as e:
        logger.error(f"❌ Lỗi get_qualified_symbols: {str(e)}")
        return []
# ==============================================================
# TRADING BOT HOÀN CHỈNH - PHIÊN BẢN TỐI ƯU (PHẦN 2)
# ==============================================================
# Tiếp nối PHẦN 1 - Hoàn thiện hệ thống Bot
# ==============================================================

# ==============================================================
# CÁC CHIẾN LƯỢC GIAO DỊCH CỤ THỂ
# ==============================================================

class RSI_EMA_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "RSI/EMA Recursive", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.rsi_period = 14
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def get_signal(self):
        try:
            if len(self.prices) < 60: 
                return None
                
            closes = self.prices[-210:]
            if len(closes) < 50:
                return None
                
            highs = [p * 1.0005 for p in closes]
            lows = [p * 0.9995 for p in closes]
            
            rsi = rsi_wilder_last(closes, self.rsi_period)
            ema_f = ema_last(closes, self.ema_fast)
            ema_s = ema_last(closes, self.ema_slow)
            mid, upper, lower = bbands_last(closes, 20, 2)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            
            if None in (rsi, ema_f, ema_s, mid, upper, lower, adx): 
                return None
                
            last_price = closes[-1]
            
            # Tín hiệu BUY: RSI quá bán + EMA tăng + (ADX mạnh hoặc giá ở dưới Bollinger)
            if (rsi < self.rsi_oversold and 
                ema_f > ema_s and 
                (adx >= 20 or last_price <= lower)):
                return "BUY"
                
            # Tín hiệu SELL: RSI quá mua + EMA giảm + (ADX mạnh hoặc giá ở trên Bollinger)
            if (rsi > self.rsi_overbought and 
                ema_f < ema_s and 
                (adx >= 20 or last_price >= upper)):
                return "SELL"
                
            return None
            
        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu RSI/EMA: {str(e)}")
            return None

class EMA_Crossover_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "EMA Crossover", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.ema_fast = 9
        self.ema_slow = 21
        self.prev_ema_fast = None
        self.prev_ema_slow = None

    def get_signal(self):
        try:
            if len(self.prices) < 60: 
                return None
                
            closes = self.prices[-210:]
            if len(closes) < 50:
                return None
                
            highs = [p * 1.0005 for p in closes]
            lows = [p * 0.9995 for p in closes]
            
            ema_f = ema_last(closes, self.ema_fast)
            ema_s = ema_last(closes, self.ema_slow)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            
            if None in (ema_f, ema_s, adx): 
                return None
                
            signal = None
            
            # Chỉ giao dịch khi trend rõ ràng (ADX >= 20)
            if (self.prev_ema_fast is not None and 
                self.prev_ema_slow is not None and 
                adx >= 20):
                    
                # EMA fast cắt lên EMA slow -> BUY
                if (self.prev_ema_fast <= self.prev_ema_slow and 
                    ema_f > ema_s):
                    signal = "BUY"
                    
                # EMA fast cắt xuống EMA slow -> SELL  
                elif (self.prev_ema_fast >= self.prev_ema_slow and 
                      ema_f < ema_s):
                    signal = "SELL"

            self.prev_ema_fast = ema_f
            self.prev_ema_slow = ema_s
            
            return signal
            
        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu EMA Crossover: {str(e)}")
            return None

class Reverse_24h_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, threshold=30, config_key=None, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Reverse 24h", config_key, smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.threshold = threshold
        self.last_24h_check = 0
        self.last_reported_change = 0

    def get_signal(self):
        try:
            current_time = time.time()
            if current_time - self.last_24h_check < 60:
                return None

            # Lấy biến động 24h
            change_24h = get_24h_change(self.symbol)
            self.last_24h_check = current_time

            if change_24h is None:
                return None
                
            # Log khi có thay đổi đáng kể
            if abs(change_24h - self.last_reported_change) > 5:
                self.log(f"📊 Biến động 24h: {change_24h:.2f}% | Ngưỡng: {self.threshold}%")
                self.last_reported_change = change_24h

            # Kiểm tra điều kiện kỹ thuật bổ sung
            if len(self.prices) < 40:
                return None
                
            closes = self.prices[-210:]
            highs = [p * 1.0006 for p in closes]
            lows = [p * 0.9994 for p in closes]
            
            mid, upper, lower = bbands_last(closes, 20, 2)
            rsi = rsi_wilder_last(closes, 14)
            
            if None in (mid, upper, lower, rsi): 
                return None
                
            last_price = closes[-1]
            signal = None

            # Tăng mạnh 24h + RSI quá mua/giá ở band trên -> SELL (đảo chiều)
            if (change_24h >= self.threshold and 
                (last_price >= upper or rsi > 70)):
                signal = "SELL"
                self.log(f"🎯 Tín hiệu SELL - Biến động 24h: +{change_24h:.2f}% (≥ {self.threshold}%)")
                
            # Giảm mạnh 24h + RSI quá bán/giá ở band dưới -> BUY (đảo chiều)
            elif (change_24h <= -self.threshold and 
                  (last_price <= lower or rsi < 30)):
                signal = "BUY"
                self.log(f"🎯 Tín hiệu BUY - Biến động 24h: {change_24h:.2f}% (≤ -{self.threshold}%)")

            return signal

        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu Reverse 24h: {str(e)}")
            return None

class Trend_Following_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Trend Following", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)

    def get_signal(self):
        try:
            if len(self.prices) < 100: 
                return None
                
            closes = self.prices[-240:]
            if len(closes) < 50:
                return None
                
            highs = [p * 1.0006 for p in closes]
            lows = [p * 0.9994 for p in closes]
            
            adx, _, _ = adx_last(highs, lows, closes, 14)
            ema_fast = ema_last(closes, 20)
            ema_slow = ema_last(closes, 50)
            
            if None in (adx, ema_fast, ema_slow): 
                return None
                
            signal = None
            
            # Xu hướng tăng mạnh (ADX >= 25) + EMA nhanh > EMA chậm -> BUY
            if adx >= 25 and ema_fast > ema_slow:
                signal = "BUY"
                
            # Xu hướng giảm mạnh (ADX >= 25) + EMA nhanh < EMA chậm -> SELL
            elif adx >= 25 and ema_fast < ema_slow:
                signal = "SELL"

            return signal
            
        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu Trend Following: {str(e)}")
            return None

class Scalping_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Scalping", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.rsi_period = 7
        self.min_movement = 0.001

    def get_signal(self):
        try:
            if len(self.prices) < 60: 
                return None
                
            closes = self.prices[-120:]
            if len(closes) < 20:
                return None
                
            highs = [p * 1.0007 for p in closes]
            lows = [p * 0.9993 for p in closes]
            volumes = [1.0] * len(closes)  # Mock volume
            
            rsi = rsi_wilder_last(closes, self.rsi_period)
            mid, upper, lower = bbands_last(closes, 20, 2)
            mfi = mfi_last(highs, lows, closes, volumes, 14)
            
            if None in (rsi, mid, upper, lower, mfi): 
                return None
                
            last_price = closes[-1]
            price_change = (last_price - closes[-2]) / max(closes[-2], 1e-9)
            
            signal = None
            
            # BUY: RSI quá bán + MFI quá bán + Giá ở band dưới + Giá giảm
            if (rsi < 25 and 
                mfi < 20 and 
                last_price <= lower and 
                price_change < -self.min_movement):
                signal = "BUY"
                
            # SELL: RSI quá mua + MFI quá mua + Giá ở band trên + Giá tăng  
            elif (rsi > 75 and 
                  mfi > 80 and 
                  last_price >= upper and 
                  price_change > self.min_movement):
                signal = "SELL"

            return signal
            
        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu Scalping: {str(e)}")
            return None

class Safe_Grid_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, grid_levels=5, config_key=None, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Safe Grid", config_key, smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.grid_levels = grid_levels
        self.orders_placed = 0

    def get_signal(self):
        try:
            # Grid trading: luân phiên BUY/SELL theo số level
            if self.orders_placed < self.grid_levels:
                self.orders_placed += 1
                if self.orders_placed % 2 == 1:
                    return "BUY"
                else:
                    return "SELL"
            return None
        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu Safe Grid: {str(e)}")
            return None

# ==============================================================
# BOT ĐỘNG THÔNG MINH - KẾT HỢP ĐA CHIẾN LƯỢC
# ==============================================================

class SmartDynamicBot(BaseBot):
    """BOT ĐỘNG THÔNG MINH - KẾT HỢP NHIỀU CHIẾN LƯỢC"""
    
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, smart_exit_config=None, dynamic_mode=True):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Smart Dynamic", config_key, smart_exit_config, dynamic_mode)
        
        # KÍCH HOẠT SMART EXIT MẶC ĐỊNH
        default_smart_config = {
            'enable_trailing': True,
            'enable_time_exit': True,
            'enable_support_resistance': True,
            'trailing_activation': 30,
            'trailing_distance': 15,
            'max_hold_time': 4,
            'min_profit_for_exit': 15,
            'breakeven_at': 12,
            'trail_adaptive': True
        }
        self.smart_exit.update_config(**default_smart_config)

    def get_signal(self):
        """KẾT HỢP NHIỀU CHIẾN LƯỢC ĐỂ RA TÍN HIỆU TỐI ƯU"""
        try:
            if len(self.prices) < 120:
                return None

            closes = self.prices[-240:]
            if len(closes) < 50:
                return None
                
            highs = [p * 1.0006 for p in closes]
            lows = [p * 0.9994 for p in closes]
            
            # 1. RSI SIGNAL
            rsi = rsi_wilder_last(closes, 14)
            
            # 2. EMA SIGNAL  
            ema_fast = ema_last(closes, 9)
            ema_slow = ema_last(closes, 21)
            
            # 3. BOLLINGER BANDS
            mid, upper, lower = bbands_last(closes, 20, 2)
            
            # 4. ADX TREND STRENGTH
            adx, plus_di, minus_di = adx_last(highs, lows, closes, 14)
            
            if None in [rsi, ema_fast, ema_slow, mid, upper, lower, adx]:
                return None

            signal = None
            buy_score = 0
            sell_score = 0
            
            # PHÂN TÍCH ĐA CHIẾN LƯỢC
            
            # Chiến lược 1: RSI + EMA
            if rsi < 30 and ema_fast > ema_slow:
                buy_score += 2
            if rsi > 70 and ema_fast < ema_slow:
                sell_score += 2
                
            # Chiến lược 2: Bollinger Bands
            last_price = closes[-1]
            if last_price <= lower:
                buy_score += 1
            if last_price >= upper:
                sell_score += 1
                
            # Chiến lược 3: ADX Trend Following
            if adx >= 25:  # Trend mạnh
                if ema_fast > ema_slow:
                    buy_score += 1
                elif ema_fast < ema_slow:
                    sell_score += 1
            
            # Chiến lược 4: Plus DI/Minue DI
            if plus_di and minus_di:
                if plus_di > minus_di and adx >= 20:
                    buy_score += 1
                elif minus_di > plus_di and adx >= 20:
                    sell_score += 1
            
            # Chiến lược 5: Volatility Filter (tránh market quá biến động)
            try:
                atrp = atr_last(highs, lows, closes, 14, True)
                if atrp and atrp > 8.0:  # Biến động quá cao
                    buy_score -= 1
                    sell_score -= 1
            except Exception:
                pass
            
            # RA QUYẾT ĐỊNH CUỐI CÙNG
            if buy_score >= 3 and buy_score > sell_score:
                signal = "BUY"
                self.log(f"🎯 Smart Signal BUY | Score: {buy_score}/5 | RSI: {rsi:.1f} | ADX: {adx:.1f}")
                
            elif sell_score >= 3 and sell_score > buy_score:
                signal = "SELL" 
                self.log(f"🎯 Smart Signal SELL | Score: {sell_score}/5 | RSI: {rsi:.1f} | ADX: {adx:.1f}")
            
            return signal

        except Exception as e:
            self.log(f"❌ Lỗi Smart Dynamic signal: {str(e)}")
            return None

# ==============================================================
# BOT MANAGER HOÀN CHỈNH VỚI TÍNH NĂNG BOT ĐỘNG
# ==============================================================

# ==============================================================
# BOT MANAGER TỐI ƯU - FIX LỖI TELEGRAM & TĂNG HIỆU NĂNG
# ==============================================================

class BotManager:
    def __init__(self, api_key=None, api_secret=None, telegram_bot_token=None, telegram_chat_id=None):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        
        # 🔥 TỐI ƯU: Cache cho các request lặp lại
        self._balance_cache = {"value": None, "timestamp": 0}
        self._positions_cache = {"value": None, "timestamp": 0}
        self._symbols_cache = {"value": None, "timestamp": 0}
        self.cache_ttl = 30  # seconds
        
        # 🔥 TỐI ƯU: Connection pooling cho requests
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=100, max_retries=3))
        
        self.auto_strategies = {}
        self.last_auto_scan = 0
        self.auto_scan_interval = 600
        
        # 🔥 TỐI ƯU: Giảm số lần scan auto
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
            self.log("🟢 HỆ THỐNG BOT THÔNG MINH ĐÃ KHỞI ĐỘNG - PHIÊN BẢN TỐI ƯU")
            
            # 🔥 TỐI ƯU: Dùng daemon threads để tự động cleanup
            self.telegram_thread = threading.Thread(target=self._optimized_telegram_listener, daemon=True)
            self.telegram_thread.start()
            
            self.auto_scan_thread = threading.Thread(target=self._optimized_auto_scan, daemon=True)
            self.auto_scan_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    # 🔥 TỐI ƯU: Cache cho balance
    def get_cached_balance(self):
        current_time = time.time()
        if (self._balance_cache["value"] is None or 
            current_time - self._balance_cache["timestamp"] > self.cache_ttl):
            balance = get_balance(self.api_key, self.api_secret)
            self._balance_cache = {"value": balance, "timestamp": current_time}
        return self._balance_cache["value"]

    # 🔥 TỐI ƯU: Cache cho positions
    def get_cached_positions(self):
        current_time = time.time()
        if (self._positions_cache["value"] is None or 
            current_time - self._positions_cache["timestamp"] > self.cache_ttl):
            positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            self._positions_cache = {"value": positions, "timestamp": current_time}
        return self._positions_cache["value"]

    def _verify_api_connection(self):
        balance = self.get_cached_balance()
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
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES THÔNG MINH - TỐI ƯU</b>\n\n🎯 <b>HỆ THỐNG ĐA CHIẾN LƯỢC + SMART EXIT + BOT ĐỘNG</b>"
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    # 🔥 TỐI ƯU: Telegram listener với connection pooling
    def _optimized_telegram_listener(self):
        """Lắng nghe Telegram với timeout và retry tối ưu"""
        last_update_id = 0
        error_count = 0
        max_errors = 5
        
        while self.running and self.telegram_bot_token:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
                params = {"offset": last_update_id + 1, "timeout": 25}
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    error_count = 0  # Reset error count
                    
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
                    logger.error("Lỗi xung đột Telegram - nhiều instance đang chạy?")
                    time.sleep(30)
                else:
                    error_count += 1
                    if error_count >= max_errors:
                        self.log("⚠️ Quá nhiều lỗi Telegram, tạm dừng 60s")
                        time.sleep(60)
                    else:
                        time.sleep(10)
                
            except requests.exceptions.Timeout:
                continue  # Timeout là bình thường, tiếp tục ngay
            except Exception as e:
                error_count += 1
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                if error_count >= max_errors:
                    time.sleep(30)
                else:
                    time.sleep(5)

    # 🔥 TỐI ƯU: Auto scan hiệu quả hơn
    def _optimized_auto_scan(self):
        """Vòng lặp tự động quét coin với tối ưu hiệu năng"""
        scan_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # 🔥 TỐI ƯU: Chỉ scan mỗi 10 phút hoặc khi có bot bị xóa
                should_scan = (
                    current_time - self.last_auto_scan > self.auto_scan_interval or
                    any(getattr(bot, 'should_be_removed', False) for bot in self.bots.values())
                )
                
                if not should_scan:
                    time.sleep(30)
                    continue
                
                # Dọn dẹp bot đã đóng lệnh
                removed_bots = []
                for bot_id, bot in list(self.bots.items()):
                    if (hasattr(bot, 'should_be_removed') and bot.should_be_removed and
                        bot.strategy_name in self.strategy_cooldowns):
                        
                        # Thêm cooldown cho chiến lược
                        strategy_type = bot.strategy_name
                        config_key = getattr(bot, 'config_key', None)
                        if config_key:
                            self.strategy_cooldowns[strategy_type][config_key] = current_time
                        
                        self.stop_bot(bot_id)
                        removed_bots.append(bot_id)
                
                if removed_bots:
                    self.log(f"🔄 Đã xóa {len(removed_bots)} bot: {', '.join(removed_bots)}")
                    time.sleep(5)  # Chờ ổn định trước khi scan mới
                
                # Quét và bổ sung coin
                self._scan_auto_strategies()
                self.last_auto_scan = current_time
                scan_count += 1
                
                # 🔥 TỐI ƯU: Giảm log sau mỗi 10 lần scan
                if scan_count % 10 == 0:
                    self.log(f"📊 Đã thực hiện {scan_count} lần auto scan")
                
                time.sleep(30)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(60)

    def _is_in_cooldown(self, strategy_type, config_key):
        """Kiểm tra cooldown với cache"""
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

    def _scan_auto_strategies(self):
        """Quét và bổ sung coin cho các chiến thuật tự động"""
        if not self.auto_strategies:
            return
            
        self.log("🔄 Đang quét coin cho các cấu hình tự động...")
        
        for strategy_key, strategy_config in self.auto_strategies.items():
            try:
                strategy_type = strategy_config['strategy_type']
                
                # Kiểm tra cooldown
                if self._is_in_cooldown(strategy_type, strategy_key):
                    continue
                
                coin_manager = CoinManager()
                current_bots_count = coin_manager.count_bots_by_config(strategy_key)
                
                if current_bots_count < 2:
                    self.log(f"🔄 {strategy_type}: đang có {current_bots_count}/2 bot, tìm thêm coin...")
                    
                    qualified_symbols = self._find_qualified_symbols(strategy_type, strategy_config, strategy_key)
                    
                    added_count = 0
                    for symbol in qualified_symbols:
                        if added_count >= (2 - current_bots_count):
                            break
                            
                        bot_id = f"{symbol}_{strategy_key}"
                        if bot_id not in self.bots:
                            success = self._create_auto_bot(symbol, strategy_type, strategy_config)
                            if success:
                                added_count += 1
                                time.sleep(0.3)  # 🔥 TỐI ƯU: Giảm delay
                    
                    if added_count > 0:
                        self.log(f"✅ {strategy_type}: đã thêm {added_count} bot mới")
                        
            except Exception as e:
                self.log(f"❌ Lỗi quét {strategy_type}: {str(e)}")

    def _find_qualified_symbols(self, strategy_type, config, strategy_key):
        """Tìm coin phù hợp với timeout"""
        try:
            leverage = config['leverage']
            threshold = config.get('threshold', 30)
            volatility = config.get('volatility', 3)
            grid_levels = config.get('grid_levels', 5)
            
            # 🔥 TỐI ƯU: Timeout cho việc tìm symbol
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_qualified_symbols,
                    self.api_key, self.api_secret, strategy_type, leverage,
                    threshold, volatility, grid_levels, 
                    max_candidates=15,  # 🔥 Giảm số candidate để tăng tốc
                    final_limit=2,
                    strategy_key=strategy_key
                )
                try:
                    return future.result(timeout=30)  # Timeout 30 giây
                except TimeoutError:
                    self.log(f"⚠️ {strategy_type}: timeout khi tìm coin")
                    return []
                    
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            return []

    def _create_auto_bot(self, symbol, strategy_type, config):
        """Tạo bot tự động với error handling"""
        try:
            leverage = config['leverage']
            percent = config['percent']
            tp = config['tp']
            sl = config['sl']
            strategy_key = config['strategy_key']
            smart_exit_config = config.get('smart_exit_config', {})
            dynamic_mode = config.get('dynamic_mode', False)
            
            bot_class = {
                "Reverse 24h": Reverse_24h_Bot,
                "Scalping": Scalping_Bot,
                "Safe Grid": Safe_Grid_Bot,
                "Trend Following": Trend_Following_Bot,
                "Smart Dynamic": SmartDynamicBot
            }.get(strategy_type)
            
            if not bot_class:
                return False
            
            # 🔥 TỐI ƯU: Validate symbol trước khi tạo bot
            current_price = get_current_price(symbol)
            if current_price <= 0:
                self.log(f"⚠️ {symbol}: không lấy được giá, bỏ qua")
                return False
            
            if strategy_type == "Reverse 24h":
                threshold = config.get('threshold', 30)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, threshold, strategy_key, smart_exit_config, dynamic_mode)
            elif strategy_type == "Safe Grid":
                grid_levels = config.get('grid_levels', 5)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels, strategy_key, smart_exit_config, dynamic_mode)
            else:
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, strategy_key, smart_exit_config, dynamic_mode)
            
            bot_id = f"{symbol}_{strategy_key}"
            self.bots[bot_id] = bot
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, **kwargs):
        """THÊM BOT MỚI - PHIÊN BẢN TỐI ƯU"""
        try:
            # Validation cơ bản
            if sl == 0:
                sl = None
                
            if not self.api_key or not self.api_secret:
                self.log("❌ Chưa thiết lập API Key")
                return False
            
            # 🔥 TỐI ƯU: Dùng cached balance
            test_balance = self.get_cached_balance()
            if test_balance is None:
                self.log("❌ LỖI: Không thể kết nối Binance")
                return False
            
            smart_exit_config = kwargs.get('smart_exit_config', {})
            dynamic_mode = kwargs.get('dynamic_mode', False)
            threshold = kwargs.get('threshold')
            volatility = kwargs.get('volatility')
            grid_levels = kwargs.get('grid_levels')
            
            bot_created = False
            
            if strategy_type == "Smart Dynamic":
                bot_created = self._create_smart_dynamic_bot(
                    lev, percent, tp, sl, smart_exit_config, dynamic_mode
                )
            elif dynamic_mode and strategy_type in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following"]:
                bot_created = self._create_dynamic_bot(
                    strategy_type, lev, percent, tp, sl, 
                    smart_exit_config, threshold, volatility, grid_levels
                )
            else:
                bot_created = self._create_static_bot(
                    symbol, strategy_type, lev, percent, tp, sl, smart_exit_config
                )
            
            # 🔥 TỐI ƯU: Clear cache sau khi thêm bot
            if bot_created:
                self._balance_cache = {"value": None, "timestamp": 0}
                self._positions_cache = {"value": None, "timestamp": 0}
            
            return bot_created
            
        except Exception as e:
            self.log(f"❌ Lỗi nghiêm trọng trong add_bot: {str(e)}")
            return False

    def _create_smart_dynamic_bot(self, lev, percent, tp, sl, smart_exit_config, dynamic_mode):
        """TẠO BOT ĐỘNG THÔNG MINH"""
        try:
            strategy_key = f"SmartDynamic_{lev}_{percent}_{tp}_{sl}"
            
            if self._is_in_cooldown("Smart Dynamic", strategy_key):
                self.log(f"⏰ Smart Dynamic: đang trong cooldown")
                return False
            
            self.auto_strategies[strategy_key] = {
                'strategy_type': "Smart Dynamic",
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key,
                'smart_exit_config': smart_exit_config,
                'dynamic_mode': True
            }
            
            qualified_symbols = self._find_qualified_symbols(
                "Smart Dynamic", self.auto_strategies[strategy_key], strategy_key
            )
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, "Smart Dynamic", self.auto_strategies[strategy_key])
                    if success:
                        success_count += 1
                        if success_count >= 2:  # 🔥 Giới hạn số bot
                            break
            
            if success_count > 0:
                success_msg = f"✅ ĐÃ TẠO {success_count} BOT ĐỘNG THÔNG MINH"
                self.log(success_msg)
                return True
            else:
                self.log("⚠️ Smart Dynamic: chưa tìm thấy coin phù hợp")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tạo Smart Dynamic bot: {str(e)}")
            return False

    def _create_dynamic_bot(self, strategy_type, lev, percent, tp, sl, smart_exit_config, threshold, volatility, grid_levels):
        """TẠO BOT ĐỘNG CHO CÁC CHIẾN LƯỢC"""
        try:
            strategy_key = f"{strategy_type}_{lev}_{percent}_{tp}_{sl}"
            
            if strategy_type == "Reverse 24h":
                strategy_key += f"_th{threshold or 30}"
            elif strategy_type == "Scalping":
                strategy_key += f"_vol{volatility or 3}"
            elif strategy_type == "Safe Grid":
                strategy_key += f"_grid{grid_levels or 5}"
            
            if self._is_in_cooldown(strategy_type, strategy_key):
                self.log(f"⏰ {strategy_type}: đang trong cooldown")
                return False
            
            config = {
                'strategy_type': strategy_type,
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key,
                'smart_exit_config': smart_exit_config,
                'dynamic_mode': True
            }
            
            if threshold: config['threshold'] = threshold
            if volatility: config['volatility'] = volatility
            if grid_levels: config['grid_levels'] = grid_levels
            
            self.auto_strategies[strategy_key] = config
            
            qualified_symbols = self._find_qualified_symbols(strategy_type, config, strategy_key)
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, strategy_type, config)
                    if success:
                        success_count += 1
                        if success_count >= 2:
                            break
            
            if success_count > 0:
                success_msg = f"✅ ĐÃ TẠO {success_count} BOT {strategy_type}"
                self.log(success_msg)
                return True
            else:
                self.log(f"⚠️ {strategy_type}: chưa tìm thấy coin phù hợp")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tạo {strategy_type} bot: {str(e)}")
            return False

    def _create_static_bot(self, symbol, strategy_type, lev, percent, tp, sl, smart_exit_config):
        """TẠO BOT TĨNH TRUYỀN THỐNG"""
        try:
            symbol = symbol.upper() if symbol else "BTCUSDT"
            bot_id = f"{symbol}_{strategy_type}"
            
            if bot_id in self.bots:
                self.log(f"⚠️ Đã có bot {strategy_type} cho {symbol}")
                return False
            
            bot_class = {
                "RSI/EMA Recursive": RSI_EMA_Bot,
                "EMA Crossover": EMA_Crossover_Bot,
                "Reverse 24h": Reverse_24h_Bot,
                "Trend Following": Trend_Following_Bot,
                "Scalping": Scalping_Bot,
                "Safe Grid": Safe_Grid_Bot
            }.get(strategy_type)
            
            if not bot_class:
                self.log(f"❌ Chiến lược {strategy_type} không được hỗ trợ")
                return False
            
            # 🔥 TỐI ƯU: Validate symbol trước khi tạo bot
            current_price = get_current_price(symbol)
            if current_price <= 0:
                self.log(f"❌ {symbol}: không lấy được giá")
                return False
            
            if strategy_type == "Reverse 24h":
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, threshold=30, smart_exit_config=smart_exit_config)
            elif strategy_type == "Safe Grid":
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels=5, smart_exit_config=smart_exit_config)
            else:
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, smart_exit_config=smart_exit_config)
            
            self.bots[bot_id] = bot
            self.log(f"✅ Đã thêm bot {strategy_type}: {symbol}")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot tĩnh {symbol}: {str(e)}")
            return False

    def stop_bot(self, bot_id):
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]
            # 🔥 TỐI ƯU: Clear cache khi có thay đổi
            self._positions_cache = {"value": None, "timestamp": 0}
            self.log(f"⛔ Đã dừng bot {bot_id}")
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.running = False
        # 🔥 TỐI ƯU: Đóng session
        self.session.close()
        self.log("🔴 Hệ thống đã dừng")

    # 🔥 TỐI ƯU: Phiên bản hoàn chỉnh của Telegram handler
    def _handle_telegram_message(self, chat_id, text):
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        # XỬ LÝ CÁC BƯỚC TẠO BOT - ĐẦY ĐỦ
        step_handlers = {
            'waiting_bot_mode': self._handle_bot_mode,
            'waiting_strategy': self._handle_strategy,
            'waiting_exit_strategy': self._handle_exit_strategy,
            'waiting_smart_config': self._handle_smart_config,
            'waiting_threshold': self._handle_threshold,
            'waiting_volatility': self._handle_volatility,
            'waiting_grid_levels': self._handle_grid_levels,
            'waiting_symbol': self._handle_symbol,
            'waiting_leverage': self._handle_leverage,
            'waiting_percent': self._handle_percent,
            'waiting_tp': self._handle_tp,
            'waiting_sl': self._handle_sl
        }
        
        if current_step in step_handlers:
            step_handlers[current_step](chat_id, text, user_state)
            return
        
        # XỬ LÝ CÁC LỆNH CHÍNH
        command_handlers = {
            "➕ Thêm Bot": self._handle_add_bot,
            "📊 Danh sách Bot": self._handle_list_bots,
            "⛔ Dừng Bot": self._handle_stop_bot,
            "💰 Số dư": self._handle_balance,
            "📈 Vị thế": self._handle_positions,
            "🎯 Chiến lược": self._handle_strategies,
            "⚙️ Cấu hình": self._handle_config
        }
        
        if text in command_handlers:
            command_handlers[text](chat_id)
        elif text.startswith("⛔ "):
            self._handle_stop_specific_bot(chat_id, text)
        else:
            self.send_main_menu(chat_id)

    # 🔥 CÁC HANDLER CHO TỪNG BƯỚC
    def _handle_bot_mode(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        elif text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
            user_state['dynamic_mode'] = (text == "🔄 Bot Động - Tự tìm coin")
            user_state['step'] = 'waiting_strategy'
            
            mode_text = "ĐỘNG - Tự tìm coin" if user_state['dynamic_mode'] else "TĨNH - Coin cố định"
            send_telegram(f"✅ Đã chọn: {mode_text}\n\nChọn chiến lược:", chat_id,
                         create_strategy_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state

    def _handle_strategy(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            strategy_map = {
                "🤖 RSI/EMA Recursive": "RSI/EMA Recursive",
                "📊 EMA Crossover": "EMA Crossover", 
                "🎯 Reverse 24h": "Reverse 24h",
                "📈 Trend Following": "Trend Following",
                "⚡ Scalping": "Scalping",
                "🛡️ Safe Grid": "Safe Grid",
                "🔄 Bot Động Thông minh": "Smart Dynamic"
            }
            
            if text in strategy_map:
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                
                if strategy == "Smart Dynamic":
                    user_state['dynamic_mode'] = True
                
                user_state['step'] = 'waiting_exit_strategy'
                
                mode_text = "ĐỘNG" if user_state.get('dynamic_mode') else "TĨNH"
                send_telegram(f"✅ Chiến lược: {strategy}\nChế độ: {mode_text}\n\nChọn chiến lược thoát lệnh:", chat_id,
                             create_exit_strategy_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state

    def _handle_exit_strategy(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            exit_configs = {
                "🔄 Thoát lệnh thông minh": {'step': 'waiting_smart_config', 'config': {}},
                "⚡ Thoát lệnh cơ bản": {'step': None, 'config': {
                    'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': False
                }},
                "🎯 Chỉ TP/SL cố định": {'step': None, 'config': {
                    'enable_trailing': False, 'enable_time_exit': False, 'enable_support_resistance': False
                }}
            }
            
            if text in exit_configs:
                config_info = exit_configs[text]
                user_state['smart_exit_config'] = config_info['config']
                
                if config_info['step']:
                    user_state['step'] = config_info['step']
                    send_telegram("Chọn cấu hình Smart Exit:", chat_id,
                                 create_smart_exit_config_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                else:
                    self._continue_bot_creation(chat_id, user_state)
                
                self.user_states[chat_id] = user_state

    def _handle_smart_config(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            smart_configs = {
                "Trailing: 30/15": {'enable_trailing': True, 'trailing_activation': 30, 'trailing_distance': 15},
                "Trailing: 50/20": {'enable_trailing': True, 'trailing_activation': 50, 'trailing_distance': 20},
                "Time Exit: 4h": {'enable_time_exit': True, 'max_hold_time': 4},
                "Time Exit: 8h": {'enable_time_exit': True, 'max_hold_time': 8},
                "Kết hợp Full": {'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True},
                "Cơ bản": {'enable_trailing': True, 'enable_time_exit': True}
            }
            
            if text in smart_configs:
                user_state['smart_exit_config'].update(smart_configs[text])
                self._continue_bot_creation(chat_id, user_state)
                self.user_states[chat_id] = user_state

    def _handle_threshold(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                threshold = float(text)
                user_state['threshold'] = threshold
                user_state['step'] = 'waiting_leverage'
                send_telegram(f"✅ Ngưỡng: {threshold}%\n\nChọn đòn bẩy:", chat_id,
                             create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_threshold_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_volatility(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                volatility = float(text)
                user_state['volatility'] = volatility
                user_state['step'] = 'waiting_leverage'
                send_telegram(f"✅ Biến động: {volatility}%\n\nChọn đòn bẩy:", chat_id,
                             create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_volatility_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_grid_levels(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                grid_levels = int(text)
                user_state['grid_levels'] = grid_levels
                user_state['step'] = 'waiting_leverage'
                send_telegram(f"✅ Số lệnh: {grid_levels}\n\nChọn đòn bẩy:", chat_id,
                             create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_grid_levels_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_symbol(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            symbol = text.upper()
            user_state['symbol'] = symbol
            user_state['step'] = 'waiting_leverage'
            send_telegram(f"✅ Coin: {symbol}\n\nChọn đòn bẩy:", chat_id,
                         create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state

    def _handle_leverage(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                leverage = int(text.replace('x', '').strip())
                if 1 <= leverage <= 100:
                    user_state['leverage'] = leverage
                    user_state['step'] = 'waiting_percent'
                    send_telegram(f"✅ Đòn bẩy: {leverage}x\n\nChọn % số dư:", chat_id,
                                 create_percent_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                    self.user_states[chat_id] = user_state
                else:
                    send_telegram("⚠️ Đòn bẩy phải từ 1-100x", chat_id,
                                 create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_percent(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                percent = float(text)
                if 0 < percent <= 100:
                    user_state['percent'] = percent
                    user_state['step'] = 'waiting_tp'
                    send_telegram(f"✅ Số dư: {percent}%\n\nChọn Take Profit (%):", chat_id,
                                 create_tp_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                    self.user_states[chat_id] = user_state
                else:
                    send_telegram("⚠️ % số dư phải từ 0.1-100", chat_id,
                                 create_percent_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_percent_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_tp(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                tp = float(text)
                if tp >= 0:
                    user_state['tp'] = tp
                    user_state['step'] = 'waiting_sl'
                    send_telegram(f"✅ TP: {tp}%\n\nChọn Stop Loss (%):", chat_id,
                                 create_sl_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                    self.user_states[chat_id] = user_state
                else:
                    send_telegram("⚠️ TP phải >= 0", chat_id,
                                 create_tp_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_tp_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_sl(self, chat_id, text, user_state):
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
        else:
            try:
                sl = float(text)
                if sl >= 0:
                    user_state['sl'] = sl
                    
                    # TẠO BOT CUỐI CÙNG
                    success = self._create_bot_from_state(chat_id, user_state)
                    
                    if success:
                        strategy = user_state.get('strategy')
                        leverage = user_state.get('leverage')
                        percent = user_state.get('percent')
                        dynamic_mode = user_state.get('dynamic_mode', False)
                        
                        success_msg = (
                            f"✅ <b>BOT ĐÃ ĐƯỢC TẠO THÀNH CÔNG!</b>\n\n"
                            f"🤖 {strategy}\n"
                            f"💰 {leverage}x | {percent}% số dư\n"
                            f"🎯 TP: {user_state.get('tp')}% | SL: {sl}%\n"
                            f"🔧 Chế độ: {'ĐỘNG' if dynamic_mode else 'TĨNH'}\n\n"
                            f"📈 Theo dõi trong danh sách bot!"
                        )
                    else:
                        success_msg = "❌ <b>LỖI TẠO BOT!</b>\n\nKiểm tra API Key và số dư!"
                    
                    send_telegram(success_msg, chat_id, create_main_menu(),
                                self.telegram_bot_token, self.telegram_chat_id)
                    
                    # RESET STATE
                    self.user_states[chat_id] = {}
                    
                else:
                    send_telegram("⚠️ SL phải >= 0", chat_id,
                                 create_sl_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                             create_sl_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        """Tiếp tục quy trình tạo bot"""
        strategy = user_state.get('strategy')
        dynamic_mode = user_state.get('dynamic_mode', False)
        
        next_steps = {
            ("Reverse 24h", True): 'waiting_threshold',
            ("Scalping", True): 'waiting_volatility',
            ("Safe Grid", True): 'waiting_grid_levels',
        }
        
        next_step = next_steps.get((strategy, dynamic_mode), 
                                 'waiting_symbol' if not dynamic_mode else 'waiting_leverage')
        
        user_state['step'] = next_step
        
        step_messages = {
            'waiting_threshold': ("Chọn ngưỡng biến động (%):", create_threshold_keyboard()),
            'waiting_volatility': ("Chọn biến động tối thiểu (%):", create_volatility_keyboard()),
            'waiting_grid_levels': ("Chọn số lệnh grid:", create_grid_levels_keyboard()),
            'waiting_symbol': ("Chọn cặp coin:", create_symbols_keyboard(strategy)),
            'waiting_leverage': ("Chọn đòn bẩy:", create_leverage_keyboard(strategy))
        }
        
        if next_step in step_messages:
            message, keyboard = step_messages[next_step]
            send_telegram(message, chat_id, keyboard, self.telegram_bot_token, self.telegram_chat_id)
        
        self.user_states[chat_id] = user_state

    def _create_bot_from_state(self, chat_id, user_state):
        """Tạo bot từ state đã thu thập"""
        try:
            return self.add_bot(
                symbol=user_state.get('symbol'),
                lev=user_state.get('leverage'),
                percent=user_state.get('percent'),
                tp=user_state.get('tp'),
                sl=user_state.get('sl'),
                strategy_type=user_state.get('strategy'),
                dynamic_mode=user_state.get('dynamic_mode', False),
                smart_exit_config=user_state.get('smart_exit_config', {}),
                threshold=user_state.get('threshold'),
                volatility=user_state.get('volatility'),
                grid_levels=user_state.get('grid_levels')
            )
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot từ state: {str(e)}")
            return False

    # 🔥 TỐI ƯU: Các command handler chính
    def _handle_add_bot(self, chat_id):
        self.user_states[chat_id] = {'step': 'waiting_bot_mode'}
        balance = self.get_cached_balance()
        
        if balance is None:
            send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            return
        
        send_telegram(f"💰 Số dư: {balance:.2f} USDT\n\nChọn chế độ bot:", chat_id,
                     create_bot_mode_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_list_bots(self, chat_id):
        if not self.bots:
            send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
            message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
            for bot_id, bot in self.bots.items():
                status = "🟢 Mở" if bot.position_open else "🟡 Chờ"
                mode = "🔄 Động" if getattr(bot, 'dynamic_mode', False) else "🤖 Tĩnh"
                message += f"🔹 {bot_id} | {status} | {mode}\n"
            
            message += f"\n📊 Tổng số: {len(self.bots)} bot"
            send_telegram(message, chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

    def _handle_stop_bot(self, chat_id):
        if not self.bots:
            send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
            keyboard = []
            for bot_id in self.bots.keys():
                keyboard.append([{"text": f"⛔ {bot_id}"}])
            keyboard.append([{"text": "❌ Hủy bỏ"}])
            
            send_telegram("⛔ <b>CHỌN BOT ĐỂ DỪNG</b>", chat_id,
                         {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_stop_specific_bot(self, chat_id, text):
        bot_id = text.replace("⛔ ", "").strip()
        if self.stop_bot(bot_id):
            send_telegram(f"⛔ Đã dừng bot {bot_id}", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)
        else:
            send_telegram(f"⚠️ Không tìm thấy bot {bot_id}", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_balance(self, chat_id):
        balance = self.get_cached_balance()
        if balance is None:
            send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
            send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

    def _handle_positions(self, chat_id):
        positions = self.get_cached_positions()
        if not positions:
            send_telegram("📭 Không có vị thế nào đang mở", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
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

    def _handle_strategies(self, chat_id):
        strategy_info = "🎯 <b>DANH SÁCH CHIẾN LƯỢC</b>\n\n" \
                       "🔄 <b>Bot Động Thông minh</b>\n• Tự động tìm coin\n• Đa chiến lược\n\n" \
                       "🎯 <b>Reverse 24h</b>\n• Đảo chiều biến động\n\n" \
                       "⚡ <b>Scalping</b>\n• Giao dịch tốc độ cao\n\n" \
                       "🛡️ <b>Safe Grid</b>\n• Grid an toàn\n\n" \
                       "📈 <b>Trend Following</b>\n• Theo xu hướng"
        
        send_telegram(strategy_info, chat_id,
                     bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

    def _handle_config(self, chat_id):
        balance = self.get_cached_balance()
        api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
        
        config_info = (
            f"⚙️ <b>CẤU HÌNH HỆ THỐNG</b>\n\n"
            f"🔑 Binance API: {api_status}\n"
            f"🤖 Tổng số bot: {len(self.bots)}\n"
            f"🔄 Bot động: {sum(1 for b in self.bots.values() if getattr(b, 'dynamic_mode', False))}\n"
            f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối"
        )
        send_telegram(config_info, chat_id,
                     bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

coin_manager = CoinManager()
