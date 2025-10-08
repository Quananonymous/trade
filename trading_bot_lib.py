# trading_bot_lib.py - HOÀN CHỈNH VỚI BOT ĐỘNG TỰ TÌM COIN MỚI
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
from heapq import nlargest

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

# ========== CHỈ BÁO KỸ THUẬT NÂNG CAO ==========
def rsi_wilder_last(prices, period=14):
    """RSI Wilder - chính xác hơn"""
    if len(prices) < period + 1: 
        return None
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
    """EMA nhanh và chính xác"""
    if len(values) < period: 
        return None
    k = 2/(period+1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v*k + ema*(1-k)
    return float(ema)

def atr_last(highs, lows, closes, period=14, return_pct=True):
    """ATR với % biến động"""
    n = len(closes)
    if min(len(highs), len(lows), len(closes)) < period+1: 
        return None
    trs = []
    for i in range(1, n):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        trs.append(tr)
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr*(period-1) + trs[i]) / period
    return float(atr / closes[-1] * 100.0) if return_pct else float(atr)

def adx_last(highs, lows, closes, period=14):
    """ADX với +DI và -DI"""
    n = len(closes)
    if min(len(highs), len(lows), len(closes)) < period+1: 
        return None, None, None
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
    if atr == 0: 
        return None, None, None
    plus_di = 100 * (wilder(plus_dm) / atr)
    minus_di = 100 * (wilder(minus_dm) / atr)
    dx = 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9)
    adx = dx
    return float(adx), float(plus_di), float(minus_di)

def bbands_last(closes, period=20, std=2):
    """Bollinger Bands"""
    if len(closes) < period: 
        return None, None, None
    w = np.array(closes[-period:])
    mid = w.mean(); dev = w.std(ddof=0)
    return float(mid), float(mid+std*dev), float(mid-std*dev)

def mfi_last(highs, lows, closes, volumes, period=14):
    """Money Flow Index"""
    n = len(closes)
    if min(len(highs), len(lows), len(closes), len(volumes)) < period+1: 
        return None
    tp = (np.array(highs) + np.array(lows) + np.array(closes)) / 3.0
    raw = tp * np.array(volumes)
    pos, neg = [], []
    for i in range(1, n):
        if tp[i] > tp[i-1]: 
            pos.append(raw[i]); neg.append(0.0)
        elif tp[i] < tp[i-1]: 
            pos.append(0.0); neg.append(raw[i])
        else: 
            pos.append(0.0); neg.append(0.0)
    if len(pos) < period: 
        return None
    p = sum(pos[-period:]); q = sum(neg[-period:])
    if q == 0: 
        return 100.0
    mr = p/q
    return float(100 - 100/(1+mr))

def obv_last(closes, volumes):
    """On-Balance Volume"""
    if len(closes) < 2 or len(volumes) < 2: 
        return 0.0
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]: 
            obv += volumes[i]
        elif closes[i] < closes[i-1]: 
            obv -= volumes[i]
    return float(obv)

def detect_regime(highs, lows, closes):
    """Phát hiện regime thị trường"""
    adx, _, _ = adx_last(highs, lows, closes, 14)
    atrp = atr_last(highs, lows, closes, 14, True)
    regime = 'unknown'
    if adx is not None:
        if adx >= 25: 
            regime = 'trend'
        elif adx < 20: 
            regime = 'range'
        else: 
            regime = 'transition'
    if atrp is not None and atrp >= 5.0: 
        regime += '|hi-vol'
    return regime, (adx or 0.0), (atrp or 0.0)

# ========== SMART EXIT MANAGER NÂNG CẤP ==========
class SmartExitManager:
    """Quản lý đóng lệnh thông minh: trailing, time, volume, SR + NÂNG CẤP"""
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.config = {
            'enable_trailing': True,
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
        changed = {}
        for k,v in kwargs.items():
            if k in self.config:
                self.config[k] = v
                changed[k] = v
        if changed:
            self.bot.log(f"⚙️ Cập nhật Smart Exit: {changed}")

    def _calculate_roi(self, current_price):
        if not self.bot.position_open or self.bot.entry <= 0 or abs(self.bot.qty) <= 0:
            return 0.0
        if self.bot.side == "BUY":
            profit = (current_price - self.bot.entry) * abs(self.bot.qty)
        else:
            profit = (self.bot.entry - current_price) * abs(self.bot.qty)
        invested = self.bot.entry * abs(self.bot.qty) / self.bot.lev
        if invested <= 0: return 0.0
        return (profit / invested) * 100.0

    def _check_trailing_stop(self, current_price):
        roi = self._calculate_roi(current_price)
        if not self.trailing_active and roi >= self.config['trailing_activation']:
            self.trailing_active = True
            self.peak_price = current_price
            return None
        if self.trailing_active:
            self.peak_price = max(self.peak_price, current_price)
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
            trigger = self.peak_price * (1 - distance/100.0)
            if current_price <= trigger:
                return f"🔻 Trailing hit ({distance:.1f}%)"
        return None

    def check_all_exit_conditions(self, current_price, current_volume=None):
        if not self.bot.position_open:
            return None
        exit_reasons = []
        if self.config['enable_trailing']:
            r = self._check_trailing_stop(current_price)
            if r: exit_reasons.append(r)
        if self.config['enable_time_exit'] and self.position_open_time>0:
            hold_hours = (time.time() - self.position_open_time)/3600
            if hold_hours >= self.config['max_hold_time']:
                exit_reasons.append(f"⏰ Quá thời gian giữ {self.config['max_hold_time']}h")
        if self.config['enable_volume_exit'] and current_volume is not None:
            self.volume_history.append(current_volume)
            if len(self.volume_history) > 60:
                self.volume_history.pop(0)
        if self.config['enable_support_resistance']:
            pass

        # Breakeven + TP ladder
        current_roi = self._calculate_roi(current_price)
        if (not self._breakeven_active) and (current_roi >= self.config.get('breakeven_at', 12)):
            self._breakeven_active = True
            self.config['min_profit_for_exit'] = max(self.config.get('min_profit_for_exit', 10), 0)
            self.bot.log(f"🟩 Kích hoạt Breakeven tại ROI {current_roi:.1f}%")
        for step in self.config.get('tp_ladder', []):
            roi_lv = step.get('roi', 0); pct = step.get('pct', 0)
            key = f"tp_{roi_lv}"
            if current_roi >= roi_lv and key not in self._tp_hit:
                ok = self.bot.partial_close(pct, reason=f"TP ladder {roi_lv}%")
                if ok:
                    self._tp_hit.add(key)

        return exit_reasons[0] if exit_reasons else None

    def on_position_opened(self):
        """Khi mở position mới"""
        self.trailing_active = False
        self.peak_price = self.bot.entry
        self.position_open_time = time.time()
        self.volume_history = []

# ========== COIN MANAGER NÂNG CẤP ==========
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
            # === COOLDOWN nâng cấp ===
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

    # ===== COOLDOWN API =====
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

# ========== API BINANCE ==========
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

def get_klines(symbol, interval, limit):
    """Lấy dữ liệu kline từ Binance"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return binance_api_request(url, params)
    except Exception:
        return None

def get_balance(api_key, api_secret):
    try:
        ts = int(time.time()*1000)
        url_path = "/fapi/v2/balance"
        params = {"timestamp": ts}
        data = signed_request(url_path, params, api_key, api_secret, 'GET')
        return data
    except Exception:
        return None

def set_leverage(symbol, leverage, api_key, api_secret):
    try:
        ts = int(time.time()*1000)
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
        ts = int(time.time()*1000)
        url_path = "/fapi/v1/allOpenOrders"
        params = {"symbol": symbol.upper(), "timestamp": ts}
        signed_request(url_path, params, api_key, api_secret, 'DELETE')
        return True
    except Exception:
        return False

def place_order(symbol, side, quantity, api_key, api_secret):
    try:
        ts = int(time.time()*1000)
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
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        data = binance_api_request(url, headers=headers)
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

def sign(query, api_secret):
    try:
        return hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        return ""

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

# ========== BASE BOT NÂNG CẤP VỚI TÍNH NĂNG TÌM COIN MỚI ==========
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
        self.dynamic_mode = dynamic_mode  # 🔄 THÊM CHẾ ĐỘ ĐỘNG
        
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
        
        # HỆ THỐNG SMART EXIT NÂNG CẤP
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
            if len(self.prices) > 100:
                self.prices = self.prices[-100:]
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

    def close_position(self, reason=""):
        try:
            self.check_position_status()
            
            if not self.position_open or abs(self.qty) <= 0:
                self.log(f"⚠️ Không có vị thế để đóng: {reason}")
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
                
                # 🔄 QUAN TRỌNG: BOT ĐỘNG TÌM COIN MỚI SAU KHI ĐÓNG LỆNH
                if self.dynamic_mode:
                    self.log("🔄 Bot động: Đang tìm coin mới...")
                    # Chạy trong thread riêng để không block main thread
                    threading.Thread(target=self._find_new_coin_after_exit, daemon=True).start()
                else:
                    self.should_be_removed = True
                
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

    def partial_close(self, fraction, reason=""):
        try:
            if not self.position_open or fraction <= 0 or fraction >= 1: 
                return False
            qty_to_close = abs(self.qty) * fraction
            if qty_to_close <= 0: 
                return False
            side = "SELL" if self.side == "BUY" else "BUY"
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.2)
            res = place_order(self.symbol, side, qty_to_close, self.api_key, self.api_secret)
            if res and 'orderId' in res:
                remain = abs(self.qty) - qty_to_close
                self.qty = remain if self.side == "BUY" else -remain
                self.log(f"🔹 Chốt {fraction*100:.0f}% vị thế | {reason}")
                if remain <= 0:
                    self._reset_position()
                    self.last_close_time = time.time()
                return True
            return False
        except Exception as e:
            self.log(f"❌ partial_close lỗi: {e}")
            return False

    def _find_new_coin_after_exit(self):
        """🔄 TÌM COIN MỚI CHO BOT ĐỘNG SAU KHI ĐÓNG LỆNH"""
        try:
            self.log("🔄 Bot động đang tìm coin mới...")
            
            # Tìm coin phù hợp
            new_symbols = get_qualified_symbols(
                self.api_key, 
                self.api_secret,
                self.strategy_name,
                self.lev,
                threshold=getattr(self, 'threshold', None),
                volatility=getattr(self, 'volatility', None),
                grid_levels=getattr(self, 'grid_levels', None),
                max_candidates=10,
                final_limit=1,
                strategy_key=self.config_key
            )
            
            if new_symbols:
                new_symbol = new_symbols[0]
                
                # Hủy đăng ký coin cũ
                old_symbol = self.symbol
                self.coin_manager.unregister_coin(old_symbol)
                
                # Cập nhật symbol mới
                self.symbol = new_symbol
                
                # Đăng ký coin mới
                registered = self.coin_manager.register_coin(
                    new_symbol, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key
                )
                
                if registered:
                    # Khởi động lại WebSocket với coin mới
                    self._restart_websocket_for_new_coin()
                    
                    message = f"🔄 Bot động chuyển từ {old_symbol} → {new_symbol}"
                    self.log(message)
                    
                    # KHÔNG đánh dấu xóa bot - BOT VẪN TIẾP TỤC CHẠY
                    self.should_be_removed = False
                else:
                    self.log(f"❌ Không thể đăng ký coin mới {new_symbol}")
                    # Quay lại coin cũ nếu không đăng ký được
                    self.symbol = old_symbol
                    self.coin_manager.register_coin(old_symbol, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key)
            else:
                self.log("❌ Không tìm thấy coin mới phù hợp, giữ nguyên coin hiện tại")
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {str(e)}")
            traceback.print_exc()

    def _restart_websocket_for_new_coin(self):
        """Khởi động lại WebSocket cho coin mới"""
        try:
            # Đóng kết nối cũ
            self.ws_manager.remove_symbol(self.symbol)
            
            # Đợi một chút
            time.sleep(2)
            
            # Mở kết nối mới
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

# ========== CÁC CHIẾN LƯỢC GIAO DỊCH VỚI CHỈ BÁO NÂNG CAO ==========
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
            
            # Sử dụng chỉ báo nâng cao
            closes = self.prices[-210:]
            highs = [p*1.0005 for p in closes]
            lows  = [p*0.9995 for p in closes]
            
            rsi = rsi_wilder_last(closes, self.rsi_period)
            ema_f = ema_last(closes, self.ema_fast)
            ema_s = ema_last(closes, self.ema_slow)
            mid, upper, lower = bbands_last(closes, 20, 2)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            
            if None in (rsi, ema_f, ema_s, mid, upper, lower, adx): 
                return None
            
            last = closes[-1]
            
            # Tín hiệu mua: RSI quá bán + EMA tăng + (ADX mạnh hoặc giá chạm lower BB)
            if rsi < self.rsi_oversold and ema_f > ema_s and (adx >= 20 or last <= lower):
                return "BUY"
            
            # Tín hiệu bán: RSI quá mua + EMA giảm + (ADX mạnh hoặc giá chạm upper BB)  
            if rsi > self.rsi_overbought and ema_f < ema_s and (adx >= 20 or last >= upper):
                return "SELL"
            
            return None
        except Exception:
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
            highs = [p*1.0005 for p in closes]
            lows  = [p*0.9995 for p in closes]
            
            ema_f = ema_last(closes, self.ema_fast)
            ema_s = ema_last(closes, self.ema_slow)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            
            if None in (ema_f, ema_s, adx): 
                return None
            
            signal = None
            if self.prev_ema_fast is not None and self.prev_ema_slow is not None and adx >= 20:
                if self.prev_ema_fast <= self.prev_ema_slow and ema_f > ema_s:
                    signal = "BUY"
                elif self.prev_ema_fast >= self.prev_ema_slow and ema_f < ema_s:
                    signal = "SELL"
            
            self.prev_ema_fast = ema_f
            self.prev_ema_slow = ema_s
            return signal
        except Exception:
            return None

class Reverse_24h_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, threshold=30, config_key=None, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Reverse 24h", config_key, smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.threshold = threshold
        self.last_24h_check = 0
        self.last_reported_change = 0

    def get_signal(self):
        try:
            now = time.time()
            if now - self.last_24h_check < 60: 
                return None
            
            url = f"{BASE_FAPI}/fapi/v1/ticker/24hr"
            data = binance_api_request(url, {"symbol": self.symbol})
            self.last_24h_check = now
            if not data: 
                return None
            
            ch = float(data.get('priceChangePercent', 0.0) or 0.0)
            if abs(ch - self.last_reported_change) > 5:
                self.log(f"📊 24h: {ch:.2f}% | Ngưỡng: {self.threshold}%")
                self.last_reported_change = ch
            
            closes = self.prices[-210:]
            if len(closes) < 40: 
                return None
            
            highs = [p*1.0006 for p in closes]
            lows  = [p*0.9994 for p in closes]
            
            mid, up, low = bbands_last(closes, 20, 2)
            rsi = rsi_wilder_last(closes, 14)
            
            if None in (mid, up, low, rsi): 
                return None
            
            last = closes[-1]
            
            if ch >= self.threshold and (last >= up or rsi > 70):
                return "SELL"
            if ch <= -self.threshold and (last <= low or rsi < 30):
                return "BUY"
            
            return None
        except Exception:
            return None

class Trend_Following_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Trend Following", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)

    def get_signal(self):
        try:
            if len(self.prices) < 100: 
                return None
            
            closes = self.prices[-240:]
            highs = [p*1.0006 for p in closes]
            lows  = [p*0.9994 for p in closes]
            
            adx, _, _ = adx_last(highs, lows, closes, 14)
            ema_fast = ema_last(closes, 20)
            ema_slow = ema_last(closes, 50)
            
            if None in (adx, ema_fast, ema_slow): 
                return None
            
            if adx >= 25 and ema_fast > ema_slow:
                return "BUY"
            if adx >= 25 and ema_fast < ema_slow:
                return "SELL"
            
            return None
        except Exception:
            return None

class Scalping_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Scalping", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)
        self.rsi_period = 7
        self.min_move = 0.001

    def get_signal(self):
        try:
            if len(self.prices) < 60: 
                return None
            
            closes = self.prices[-120:]
            highs = [p*1.0007 for p in closes]
            lows  = [p*0.9993 for p in closes]
            vols  = [1.0]*len(closes)
            
            rsi = rsi_wilder_last(closes, self.rsi_period)
            mid, up, low = bbands_last(closes, 20, 2)
            mfi = mfi_last(highs, lows, closes, vols, 14)
            
            if None in (rsi, mid, up, low, mfi): 
                return None
            
            last = closes[-1]
            pc = (last - closes[-2]) / max(closes[-2], 1e-9)
            
            if rsi < 25 and (mfi < 20) and (last <= low) and pc < -self.min_move:
                return "BUY"
            if rsi > 75 and (mfi > 80) and (last >= up) and pc > self.min_move:
                return "SELL"
            
            return None
        except Exception:
            return None

class Smart_Dynamic_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Smart Dynamic", smart_exit_config=smart_exit_config, dynamic_mode=dynamic_mode)

    def get_signal(self):
        try:
            if len(self.prices) < 120: 
                return None
            
            closes = self.prices[-240:]
            highs = [p*1.0006 for p in closes]
            lows  = [p*0.9994 for p in closes]
            
            rsi = rsi_wilder_last(closes, 14)
            ema9 = ema_last(closes, 9)
            ema21 = ema_last(closes, 21)
            mid, up, low = bbands_last(closes, 20, 2)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            
            if None in (rsi, ema9, ema21, mid, up, low, adx): 
                return None
            
            score_buy = 0; score_sell = 0
            
            if rsi < 30 and ema9 > ema21: 
                score_buy += 2
            if rsi > 70 and ema9 < ema21: 
                score_sell += 2
            
            # ưu tiên thuận trend
            if adx >= 25 and ema9 > ema21: 
                score_buy += 1
            if adx >= 25 and ema9 < ema21: 
                score_sell += 1
            
            # giảm điểm nếu quá hi-vol
            try:
                atrp = atr_last(highs, lows, closes, 14, True)
                if atrp and atrp > 8.0:
                    score_buy -= 1; score_sell -= 1
            except Exception:
                pass
            
            if score_buy >= 2 and score_buy > score_sell:
                return "BUY"
            if score_sell >= 2 and score_sell > score_buy:
                return "SELL"
            
            return None
        except Exception:
            return None

# ========== SCANNER NÂNG CẤP ==========
def _score_symbol_for_strategy(symbol, k5, k15, ticker, strategy_type):
    """Đánh giá coin cho chiến lược với chỉ báo nâng cao"""
    o5,h5,l5,c5,v5 = k5
    o15,h15,l15,c15,v15 = k15
    
    # Tính chỉ báo nâng cao
    atr5 = atr_last(h5, l5, c5, 14, True) or 0.0
    adx15, pdi, mdi = adx_last(h15, l15, c15, 14)
    adx15 = adx15 or 0.0
    ema9_15 = ema_last(c15, 9) or c15[-1]
    ema21_15 = ema_last(c15, 21) or c15[-1]
    trend_bias = 1 if ema9_15>ema21_15 else -1 if ema9_15<ema21_15 else 0
    
    vol_avg = np.mean(v5[-30:]) if len(v5) >= 30 else (sum(v5)/max(len(v5),1))
    vol_surge = (v5[-1]/max(vol_avg,1e-9))
    abs_change = abs(float(ticker.get('priceChangePercent', 0.0) or 0.0))
    qvol = float(ticker.get('quoteVolume', 0.0) or 0.0)

    base_penalty = 0.0
    if symbol in ('BTCUSDT','ETHUSDT'):
        base_penalty = 0.5

    if strategy_type == "Reverse 24h":
        score = (abs_change/10.0) + min(vol_surge, 3.0) - base_penalty
        if atr5 >= 8.0: score -= 1.0
    elif strategy_type == "Scalping":
        score = min(vol_surge, 3.0) + min(atr5/2.0, 2.0) - base_penalty
        if atr5 > 10.0: score -= 1.0
    elif strategy_type == "Safe Grid":
        mid_vol = 2.0 <= atr5 <= 6.0
        score = (1.5 if mid_vol else 0.5) + min(qvol/5_000_000, 3.0) - base_penalty
    elif strategy_type == "Trend Following":
        score = (adx15/25.0) + (1.0 if trend_bias>0 else 0.0) + min(vol_surge, 3.0) - base_penalty
    elif strategy_type == "Smart Dynamic":
        base = (min(adx15, 40)/40.0) + min(vol_surge, 3.0)
        if trend_bias != 0 and 3.0 <= atr5 <= 8.0: base += 1.0
        score = base - base_penalty
    else:
        score = min(vol_surge, 3.0) + (abs_change/20.0)

    return score, {"atr5%": atr5, "adx15": adx15, "vol_surge": vol_surge, "trend_bias": trend_bias}

def get_qualified_symbols(api_key, api_secret, strategy_type, leverage, 
                          threshold=None, volatility=None, grid_levels=None, 
                          max_candidates=20, final_limit=2, strategy_key=None):
    """Tìm coin phù hợp với chỉ báo nâng cao"""
    try:
        test_balance = get_balance(api_key, api_secret)
        if test_balance is None:
            logger.error("❌ KHÔNG THỂ KẾT NỐI BINANCE")
            return []
        
        coin_manager = CoinManager()
        all_symbols = get_all_usdt_pairs(limit=300)
        if not all_symbols:
            return []
        
        url = f"{BASE_FAPI}/fapi/v1/ticker/24hr"
        all_tickers = binance_api_request(url)
        if not all_tickers:
            return []
        
        tickmap = {t['symbol']: t for t in all_tickers if 'symbol' in t}

        scored = []
        for sym in all_symbols:
            if sym not in tickmap: continue
            if sym in ('BTCUSDT','ETHUSDT'): continue
            if strategy_key and coin_manager.has_same_config_bot(sym, strategy_key):
                continue
            if coin_manager.is_in_cooldown(sym):
                continue
            
            t = tickmap[sym]
            try:
                qvol = float(t.get('quoteVolume', 0.0) or 0.0)
                if qvol < 1_000_000: 
                    continue
                
                # Lấy dữ liệu kline với chỉ báo nâng cao
                k5 = get_klines(sym, "5m", 210); k15 = get_klines(sym, "15m", 210)
                if not k5 or not k15:
                    continue
                
                score, feat = _score_symbol_for_strategy(sym, k5, k15, t, strategy_type)
                
                if strategy_type == "Reverse 24h" and threshold is not None:
                    abs_change = abs(float(t.get('priceChangePercent', 0.0) or 0.0))
                    if abs_change < threshold:
                        continue
                
                scored.append((score, sym, feat))
            except Exception:
                continue

        top = nlargest(max_candidates, scored, key=lambda x: x[0])
        final_syms = []
        for score, sym, feat in top:
            try:
                if not set_leverage(sym, leverage, api_key, api_secret):
                    continue
                step = get_step_size(sym, api_key, api_secret)
                if step <= 0:
                    continue
                final_syms.append(sym)
                logger.info(f"✅ {sym}: score={score:.2f} | feats={feat}")
                if len(final_syms) >= final_limit:
                    break
            except Exception:
                continue

        if not final_syms:
            backup = []
            for sym in all_symbols:
                t = tickmap.get(sym); 
                if not t: continue
                try:
                    qv = float(t.get('quoteVolume', 0.0) or 0.0)
                    ch = abs(float(t.get('priceChangePercent', 0.0) or 0.0))
                    if qv > 3_000_000 and 0.5 <= ch <= 10.0 and sym not in ('BTCUSDT','ETHUSDT'):
                        backup.append((qv, sym))
                except: 
                    continue
            backup.sort(reverse=True)
            for _, sym in backup[:final_limit]:
                if set_leverage(sym, leverage, api_key, api_secret) and get_step_size(sym, api_key, api_secret) > 0:
                    final_syms.append(sym)
                    logger.info(f"🔄 {sym}: backup coin")

        logger.info(f"🎯 {strategy_type}: cuối cùng {len(final_syms)} coin: {final_syms}")
        return final_syms[:final_limit]
    except Exception as e:
        logger.error(f"❌ Lỗi get_qualified_symbols (new): {str(e)}")
        return []

# ========== BOT MANAGER HOÀN CHỈNH ==========
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
        
        # THÊM: Dictionary theo dõi thời gian chờ cho mỗi chiến lược
        self.strategy_cooldowns = {
            "Reverse 24h": {},
            "Scalping": {},
            "Trend Following": {},
            "Safe Grid": {},
            "Smart Dynamic": {}
        }
        self.cooldown_period = 300  # 5 phút cooldown
        
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
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES THÔNG MINH</b>\n\n🎯 <b>HỆ THỐNG ĐA CHIẾN LƯỢC + SMART EXIT + BOT ĐỘNG TỰ TÌM COIN</b>"
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def _is_in_cooldown(self, strategy_type, config_key):
        """Kiểm tra xem chiến lược có đang trong thời gian chờ không"""
        if strategy_type not in self.strategy_cooldowns:
            return False
            
        last_cooldown_time = self.strategy_cooldowns[strategy_type].get(config_key)
        if last_cooldown_time is None:
            return False
            
        current_time = time.time()
        if current_time - last_cooldown_time < self.cooldown_period:
            return True
            
        # Hết cooldown, xóa khỏi danh sách
        del self.strategy_cooldowns[strategy_type][config_key]
        return False

    def _auto_scan_loop(self):
        """VÒNG LẶP TỰ ĐỘNG QUÉT COIN VỚI COOLDOWN"""
        while self.running:
            try:
                current_time = time.time()
                
                # KIỂM TRA BOT ĐỘNG CẦN TÌM COIN MỚI
                for bot_id, bot in list(self.bots.items()):
                    if (hasattr(bot, 'config_key') and bot.config_key and
                        not bot.position_open and 
                        bot.strategy_name in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following", "Smart Dynamic"]):
                        
                        # Bot động đang chờ, tìm coin mới
                        self.log(f"🔄 Bot động {bot_id} đang tìm coin mới...")
                        bot._find_new_coin_after_close()
                
                if current_time - self.last_auto_scan > self.auto_scan_interval:
                    self._scan_auto_strategies()
                    self.last_auto_scan = current_time
                
                time.sleep(30)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(30)

    def _scan_auto_strategies(self):
        """Quét và bổ sung coin cho các chiến thuật tự động với COOLDOWN"""
        if not self.auto_strategies:
            return
            
        self.log("🔄 Đang quét coin cho các cấu hình tự động...")
        
        for strategy_key, strategy_config in self.auto_strategies.items():
            try:
                strategy_type = strategy_config['strategy_type']
                leverage = strategy_config['leverage']
                percent = strategy_config['percent']
                tp = strategy_config['tp']
                sl = strategy_config['sl']
                
                # KIỂM TRA COOLDOWN - QUAN TRỌNG
                if self._is_in_cooldown(strategy_type, strategy_key):
                    self.log(f"⏰ {strategy_type} (Config: {strategy_key}): đang trong cooldown, bỏ qua")
                    continue
                
                coin_manager = CoinManager()
                current_bots_count = coin_manager.count_bots_by_config(strategy_key)
                
                if current_bots_count < 2:
                    self.log(f"🔄 {strategy_type} (Config: {strategy_key}): đang có {current_bots_count}/2 bot, tìm thêm coin...")
                    
                    qualified_symbols = self._find_qualified_symbols(strategy_type, leverage, strategy_config, strategy_key)
                    
                    added_count = 0
                    for symbol in qualified_symbols:
                        bot_id = f"{symbol}_{strategy_key}"
                        if bot_id not in self.bots and added_count < (2 - current_bots_count):
                            success = self._create_auto_bot(symbol, strategy_type, strategy_config)
                            if success:
                                added_count += 1
                                self.log(f"✅ Đã thêm {symbol} cho {strategy_type} (Config: {strategy_key})")
                    
                    if added_count > 0:
                        self.log(f"🎯 {strategy_type}: đã thêm {added_count} bot mới cho config {strategy_key}")
                    else:
                        self.log(f"⚠️ {strategy_type}: không tìm thấy coin mới phù hợp cho config {strategy_key}")
                else:
                    self.log(f"✅ {strategy_type} (Config: {strategy_key}): đã đủ 2 bot, không tìm thêm")
                        
            except Exception as e:
                self.log(f"❌ Lỗi quét {strategy_type}: {str(e)}")

    def _find_qualified_symbols(self, strategy_type, leverage, config, strategy_key):
        """Tìm coin phù hợp cho chiến lược"""
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

    def _create_auto_bot(self, symbol, strategy_type, config):
        """Tạo bot tự động"""
        try:
            leverage = config['leverage']
            percent = config['percent']
            tp = config['tp']
            sl = config['sl']
            strategy_key = config['strategy_key']
            smart_exit_config = config.get('smart_exit_config', {})
            
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
                              self.telegram_chat_id, threshold, strategy_key, smart_exit_config)
            elif strategy_type == "Safe Grid":
                grid_levels = config.get('grid_levels', 5)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels, strategy_key, smart_exit_config)
            else:
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, strategy_key, smart_exit_config)
            
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
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance")
            return False
        
        # LẤY CẤU HÌNH SMART EXIT
        smart_exit_config = kwargs.get('smart_exit_config', {})
        bot_mode = kwargs.get('bot_mode', 'static')  # static or dynamic
        
        # BOT ĐỘNG THÔNG MINH - CHỈ KHI CHỌN ĐÚNG CHIẾN LƯỢC SMART DYNAMIC
        if strategy_type == "Smart Dynamic":
            strategy_key = f"SmartDynamic_{lev}_{percent}_{tp}_{sl}"
            
            # KIỂM TRA COOLDOWN TRƯỚC KHI THÊM
            if self._is_in_cooldown("Smart Dynamic", strategy_key):
                self.log(f"⏰ Smart Dynamic (Config: {strategy_key}): đang trong cooldown, không thể thêm mới")
                return False
            
            self.auto_strategies[strategy_key] = {
                'strategy_type': "Smart Dynamic",
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key,
                'smart_exit_config': smart_exit_config
            }
            
            qualified_symbols = self._find_qualified_symbols("Smart Dynamic", lev, 
                                                           self.auto_strategies[strategy_key], strategy_key)
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, "Smart Dynamic", self.auto_strategies[strategy_key])
                    if success:
                        success_count += 1
            
            if success_count > 0:
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {success_count} BOT ĐỘNG THÔNG MINH</b>\n\n"
                    f"🎯 Chiến lược: Smart Dynamic\n"
                    f"💰 Đòn bẩy: {lev}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl}%\n"
                    f"🤖 Coin: {', '.join(qualified_symbols[:success_count])}\n\n"
                    f"🔑 <b>Config Key:</b> {strategy_key}\n"
                    f"🔄 <i>Hệ thống sẽ tự động tìm coin mới sau khi đóng lệnh</i>\n"
                    f"⏰ <i>Cooldown: {self.cooldown_period//60} phút sau khi đóng lệnh</i>"
                )
                self.log(success_msg)
                return True
            else:
                self.log("⚠️ Smart Dynamic: chưa tìm thấy coin phù hợp, sẽ thử lại sau")
                return True
        
        # CÁC CHIẾN LƯỢC ĐỘNG KHÁC - KHI CHỌN BOT ĐỘNG VỚI CHIẾN LƯỢC CỤ THỂ
        elif bot_mode == 'dynamic' and strategy_type in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following"]:
            strategy_key = f"{strategy_type}_{lev}_{percent}_{tp}_{sl}"
            
            # Thêm tham số đặc biệt
            if strategy_type == "Reverse 24h":
                threshold = kwargs.get('threshold', 30)
                strategy_key += f"_th{threshold}"
            elif strategy_type == "Scalping":
                volatility = kwargs.get('volatility', 3)
                strategy_key += f"_vol{volatility}"
            elif strategy_type == "Safe Grid":
                grid_levels = kwargs.get('grid_levels', 5)
                strategy_key += f"_grid{grid_levels}"
            
            # KIỂM TRA COOLDOWN TRƯỚC KHI THÊM
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
                'smart_exit_config': smart_exit_config,
                **kwargs
            }
            
            qualified_symbols = self._find_qualified_symbols(strategy_type, lev, 
                                                           self.auto_strategies[strategy_key], strategy_key)
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, strategy_type, self.auto_strategies[strategy_key])
                    if success:
                        success_count += 1
            
            if success_count > 0:
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {success_count} BOT {strategy_type}</b>\n\n"
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
                    
                success_msg += f"🤖 Coin: {', '.join(qualified_symbols[:success_count])}\n\n"
                success_msg += f"🔑 <b>Config Key:</b> {strategy_key}\n"
                success_msg += f"🔄 <i>Bot sẽ tự động tìm coin mới sau khi đóng lệnh</i>\n"
                success_msg += f"⏰ <i>Cooldown: {self.cooldown_period//60} phút sau khi đóng lệnh</i>"
                
                self.log(success_msg)
                return True
            else:
                self.log(f"⚠️ {strategy_type}: chưa tìm thấy coin phù hợp, sẽ thử lại sau")
                return True
        
        # CHIẾN LƯỢC THỦ CÔNG
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
                              self.telegram_chat_id, smart_exit_config)
                
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
        
        # XỬ LÝ CÁC BƯỚC TẠO BOT THEO THỨ TỰ
        if current_step == 'waiting_bot_mode':
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
                        "🤖 Bot sẽ TỰ ĐỘNG tìm coin phù hợp\n"
                        "🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                        "📈 Tối ưu hóa tự động\n\n"
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
                
                # Map tên hiển thị sang tên chiến lược thực tế
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
                
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {strategy}</b>\n\n"
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
            elif text in ["🔄 Thoát lệnh thông minh", "⚡ Thoát lệnh cơ bản", "🎯 Chỉ TP/SL cố định"]:
                if text == "🔄 Thoát lệnh thông minh":
                    user_state['exit_strategy'] = 'smart_full'
                    user_state['step'] = 'waiting_smart_config'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: THOÁT LỆNH THÔNG MINH</b>\n\n"
                        "Hệ thống sẽ tự động:\n"
                        "• 🔄 Trailing Stop bảo vệ lợi nhuận\n"
                        "• ⏰ Time Exit giới hạn thời gian\n"
                        "• 📊 Support/Resistance Exit\n"
                        "• 🎯 Tối ưu hóa lợi nhuận\n\n"
                        "Chọn cấu hình Smart Exit:",
                        chat_id,
                        create_smart_exit_config_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                elif text == "⚡ Thoát lệnh cơ bản":
                    user_state['exit_strategy'] = 'smart_basic'
                    user_state['smart_exit_config'] = {
                        'enable_trailing': True,
                        'enable_time_exit': True,
                        'enable_support_resistance': False,
                        'trailing_activation': 30,
                        'trailing_distance': 15,
                        'max_hold_time': 6
                    }
                    self._continue_bot_creation(chat_id, user_state)
                else:
                    user_state['exit_strategy'] = 'traditional'
                    user_state['smart_exit_config'] = {
                        'enable_trailing': False,
                        'enable_time_exit': False,
                        'enable_support_resistance': False
                    }
                    self._continue_bot_creation(chat_id, user_state)

        elif current_step == 'waiting_smart_config':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                smart_config = {}
                if text == "Trailing: 30/15":
                    smart_config = {
                        'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True,
                        'trailing_activation': 30, 'trailing_distance': 15, 'max_hold_time': 4
                    }
                elif text == "Trailing: 50/20":
                    smart_config = {
                        'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True,
                        'trailing_activation': 50, 'trailing_distance': 20, 'max_hold_time': 6
                    }
                elif text == "Time Exit: 4h":
                    smart_config = {
                        'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True,
                        'trailing_activation': 25, 'trailing_distance': 12, 'max_hold_time': 4
                    }
                elif text == "Time Exit: 8h":
                    smart_config = {
                        'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True,
                        'trailing_activation': 40, 'trailing_distance': 18, 'max_hold_time': 8
                    }
                elif text == "Kết hợp Full":
                    smart_config = {
                        'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True,
                        'trailing_activation': 35, 'trailing_distance': 15, 'max_hold_time': 6
                    }
                elif text == "Cơ bản":
                    smart_config = {
                        'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': False,
                        'trailing_activation': 30, 'trailing_distance': 15, 'max_hold_time': 6
                    }
                
                user_state['smart_exit_config'] = smart_config
                self._continue_bot_creation(chat_id, user_state)

        # XỬ LÝ CÁC BƯỚC TIẾP THEO
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
                # Xử lý đòn bẩy
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
                    
                    # Lấy số dư hiện tại để hiển thị
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
                    
                    # Tính số tiền thực tế sẽ sử dụng
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
                    
                    # TẠO BOT VỚI TẤT CẢ THÔNG TIN
                    strategy = user_state.get('strategy')
                    bot_mode = user_state.get('bot_mode', 'static')
                    leverage = user_state.get('leverage')
                    percent = user_state.get('percent')
                    tp = user_state.get('tp')
                    sl = user_state.get('sl')
                    symbol = user_state.get('symbol')
                    exit_strategy = user_state.get('exit_strategy', 'traditional')
                    smart_exit_config = user_state.get('smart_exit_config', {})
                    
                    # Các tham số đặc biệt
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
                            strategy_type=strategy,
                            smart_exit_config=smart_exit_config
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
                            smart_exit_config=smart_exit_config,
                            threshold=threshold,
                            volatility=volatility,
                            grid_levels=grid_levels
                        )
                    
                    if success:
                        success_msg = (
                            f"✅ <b>ĐÃ TẠO BOT THÀNH CÔNG</b>\n\n"
                            f"🤖 Chiến lược: {strategy}\n"
                            f"🔧 Chế độ: {bot_mode}\n"
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
                        
                        success_msg += f"\n\n🔄 <i>Hệ thống sẽ tự động quản lý và thông báo</i>"
                        if bot_mode == 'dynamic':
                            success_msg += f"\n🎯 <i>Bot sẽ tự động tìm coin mới sau khi đóng lệnh</i>"
                        
                        send_telegram(success_msg, chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    else:
                        send_telegram("❌ Có lỗi khi tạo bot. Vui lòng thử lại.",
                                    chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    
                    # Xóa state
                    self.user_states[chat_id] = {}
                    
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Stop Loss:",
                                chat_id, create_sl_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        # XỬ LÝ CÁC LỆNH CHÍNH
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_mode'}
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(
                f"🎯 <b>CHỌN CHẾ ĐỘ BOT</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"🤖 <b>Bot Tĩnh:</b>\n• Giao dịch coin CỐ ĐỊNH\n• Bạn chọn coin cụ thể\n• Phù hợp chiến lược cá nhân\n\n"
                f"🔄 <b>Bot Động:</b>\n• TỰ ĐỘNG tìm coin tốt nhất\n• Tự tìm coin mới sau khi đóng lệnh\n• Tối ưu hóa tự động",
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
                smart_bots = 0
                dynamic_bots = 0
                for bot_id, bot in self.bots.items():
                    status = "🟢 Mở" if bot.status == "open" else "🟡 Chờ"
                    exit_type = "🔴 Thường" 
                    if hasattr(bot, 'smart_exit') and bot.smart_exit.config['enable_trailing']:
                        exit_type = "🟢 Thông minh"
                        smart_bots += 1
                    
                    mode = "Tĩnh"
                    if hasattr(bot, 'config_key') and bot.config_key:
                        mode = "Động"
                        dynamic_bots += 1
                    
                    message += f"🔹 {bot_id} | {status} | {mode} | {exit_type} | ĐB: {bot.lev}x\n"
                
                message += f"\n📊 Tổng số: {len(self.bots)} bot | 🤖 Thông minh: {smart_bots} | 🔄 Động: {dynamic_bots}"
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
                "🎯 <b>DANH SÁCH CHIẾN LƯỢC HOÀN CHỈNH</b>\n\n"
                
                "🔄 <b>Bot Động Thông Minh</b>\n"
                "• Kết hợp đa chiến lược\n"
                "• Tự động tìm coin tốt nhất\n"
                "• Tự tìm coin mới sau khi đóng lệnh\n"
                "• Smart Exit 4 cơ chế\n"
                "• 🔄 Tự quét toàn Binance\n\n"
                
                "🎯 <b>Reverse 24h</b> - TỰ ĐỘNG\n"
                "• Đảo chiều biến động 24h\n"
                "• Tự tìm coin từ TOÀN BỘ Binance\n"
                "• Tự tìm coin mới sau khi đóng lệnh\n"
                "• Smart Exit bảo vệ lợi nhuận\n\n"
                
                "⚡ <b>Scalping</b> - TỰ ĐỘNG\n"
                "• Giao dịch tốc độ cao\n"
                "• Tự tìm coin biến động\n"
                "• Tự tìm coin mới sau khi đóng lệnh\n"
                "• Smart Exit chốt lời nhanh\n\n"
                
                "🛡️ <b>Safe Grid</b> - TỰ ĐỘNG\n"
                "• Grid an toàn\n"
                "• Tự tìm coin ổn định\n"
                "• Tự tìm coin mới sau khi đóng lệnh\n"
                "• Phân bổ rủi ro thông minh\n\n"
                
                "📈 <b>Trend Following</b> - TỰ ĐỘNG\n"
                "• Theo xu hướng giá\n"
                "• Tự tìm coin trend rõ\n"
                "• Tự tìm coin mới sau khi đóng lệnh\n"
                "• Smart Exit giữ lợi nhuận\n\n"
                
                "🤖 <b>RSI/EMA Recursive</b> - TĨNH\n"
                "• Phân tích RSI + EMA đệ quy\n"
                "• Coin cụ thể do bạn chọn\n\n"
                
                "📊 <b>EMA Crossover</b> - TĨNH\n"
                "• Giao cắt EMA nhanh/chậm\n"
                "• Coin cụ thể do bạn chọn\n\n"
                
                "💡 <b>Smart Exit System</b>\n"
                "• 🔄 Trailing Stop bảo vệ lợi nhuận\n"
                "• ⏰ Time Exit giới hạn rủi ro\n"
                "• 📊 Volume Exit theo momentum\n"
                "• 🎯 Support/Resistance Exit"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            smart_bots_count = sum(1 for bot in self.bots.values() 
                                 if hasattr(bot, 'smart_exit') and bot.smart_exit.config['enable_trailing'])
            
            dynamic_bots_count = sum(1 for bot in self.bots.values() 
                                   if hasattr(bot, 'config_key') and bot.config_key)
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG THÔNG MINH</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🧠 Bot thông minh: {smart_bots_count}\n"
                f"🔄 Bot động: {dynamic_bots_count}\n"
                f"📊 Chiến lược: {len(set(bot.strategy_name for bot in self.bots.values()))}\n"
                f"🔄 Auto scan: {len(self.auto_strategies)} cấu hình\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n"
                f"💡 Smart Exit: {smart_bots_count}/{len(self.bots)} bot\n"
                f"⏰ Cooldown: {self.cooldown_period//60} phút"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        """Tiếp tục quy trình tạo bot sau khi chọn Smart Exit"""
        strategy = user_state.get('strategy')
        bot_mode = user_state.get('bot_mode', 'static')
        
        if bot_mode == 'dynamic' and strategy != "Smart Dynamic":
            # Các chiến lược động khác
            if strategy == "Reverse 24h":
                user_state['step'] = 'waiting_threshold'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"🤖 Bot sẽ tự động tìm coin mới sau khi đóng lệnh\n\n"
                    f"Chọn ngưỡng biến động (%):",
                    chat_id,
                    create_threshold_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "Scalping":
                user_state['step'] = 'waiting_volatility'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"🤖 Bot sẽ tự động tìm coin mới sau khi đóng lệnh\n\n"
                    f"Chọn biến động tối thiểu (%):",
                    chat_id,
                    create_volatility_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "Safe Grid":
                user_state['step'] = 'waiting_grid_levels'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"🤖 Bot sẽ tự động tìm coin mới sau khi đóng lệnh\n\n"
                    f"Chọn số lệnh grid:",
                    chat_id,
                    create_grid_levels_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"🤖 Bot sẽ tự động tìm coin mới sau khi đóng lệnh\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
        else:
            if bot_mode == 'static':
                user_state['step'] = 'waiting_symbol'
                send_telegram(
                    f"🎯 <b>BOT TĨNH: {strategy}</b>\n\n"
                    f"🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n\n"
                    f"Chọn cặp coin:",
                    chat_id,
                    create_symbols_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG THÔNG MINH</b>\n\n"
                    f"🤖 Bot sẽ TỰ ĐỘNG tìm coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📈 Tối ưu hóa tự động\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()
