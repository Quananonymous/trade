# trading_bot_lib.py - HOÀN CHỈNH VỚI BOT ĐỘNG + SMART EXIT NÂNG CẤP + COINMANAGER COOLDOWN
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

# ========== BINANCE API HELPERS ==========
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
            
        opens, highs, lows, closes, volumes = [], [], [], [], []
        for k in data:
            opens.append(float(k[1]))
            highs.append(float(k[2]))
            lows.append(float(k[3]))
            closes.append(float(k[4]))
            volumes.append(float(k[5]))
            
        return (opens, highs, lows, closes, volumes)
    except Exception as e:
        logger.error(f"Lỗi lấy klines {symbol}: {e}")
        return None

def get_positions(symbol=None, api_key=None, api_secret=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
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

# ========== CHỈ BÁO KỸ THUẬT NÂNG CAO ==========
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

# ========== SMART EXIT MANAGER NÂNG CẤP ==========
class SmartExitManager:
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
        self.trailing_active = False
        self.peak_price = self.bot.entry
        self.position_open_time = time.time()
        self.volume_history = []

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

# ========== SCANNER NÂNG CẤP ==========
def _score_symbol_for_strategy(symbol, k5, k15, ticker, strategy_type):
    o5,h5,l5,c5,v5 = k5
    o15,h15,l15,c15,v15 = k15
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
        logger.error(f"❌ Lỗi get_qualified_symbols: {str(e)}")
        return []

# ========== BASE BOT NÂNG CẤP ==========
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
        self.dynamic_mode = dynamic_mode
        
        self.status = "waiting"
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
        
        self.cooldown_period = 300
        self.position_check_interval = 30
        
        self._close_attempted = False
        self._last_close_attempt = 0
        
        self.should_be_removed = False
        
        self.smart_exit = SmartExitManager(self)
        if smart_exit_config:
            self.smart_exit.update_config(**smart_exit_config)
        
        self.coin_manager = CoinManager()
        if symbol:
            success = self.coin_manager.register_coin(self.symbol, f"{strategy_name}_{id(self)}", strategy_name, config_key)
            if not success:
                self.log(f"⚠️ Cảnh báo: {self.symbol} đã được quản lý bởi bot khác")
        
        self._candidate_pool = []
        
        self.check_position_status()
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        mode_text = "ĐỘNG - Tự tìm coin" if dynamic_mode else "TĨNH - Coin cố định"
        self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | Chế độ: {mode_text} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")

    def log(self, msg):
        logger.info(f"[{self.strategy_name}][{self.symbol}] {msg}")
        try:
            send_telegram(f"[{self.strategy_name}][{self.symbol}] {msg}", self.telegram_chat_id, bot_token=self.telegram_bot_token)
        except Exception:
            pass

    def _handle_price_update(self, price):
        if self._stop or not price or price <= 0:
            return
        try:
            self.prices.append(float(price))
            if len(self.prices) > 1000:
                self.prices.pop(0)
            
            if not self.position_open:
                sig = self.get_signal()
                if sig:
                    self.open_position(sig)
            
            self.check_tp_sl()
            
        except Exception as e:
            self.log(f"Lỗi price update: {e}")

    def get_signal(self):
        return None

    def open_position(self, side):
        try:
            price = get_current_price(self.symbol)
            if price <= 0: return False
            
            balance = get_balance(self.api_key, self.api_secret)
            if not balance: return False
            
            usdt = 0
            for b in balance:
                if b.get('asset') == 'USDT':
                    usdt = float(b.get('balance', 0) or b.get('balance', 0))
                    break
                    
            if usdt <= 0: return False
            
            invest = usdt * self.percent
            qty = (invest * self.lev) / price
            step = get_step_size(self.symbol, self.api_key, self.api_secret)
            
            if step and step > 0:
                precision = int(round(-math.log10(step))) if step < 1 else 0
                qty = float(f"{qty:.{precision}f}")
                
            if qty <= 0: return False
            
            res = place_order(self.symbol, 'BUY' if side=='BUY' else 'SELL', qty, self.api_key, self.api_secret)
            if not res: return False
            
            self.position_open = True
            self.side = side
            self.qty = qty if side=='BUY' else -qty
            self.entry = price
            self.smart_exit.position_open_time = time.time()
            self._close_attempted = False
            
            self.log(f"🚀 Mở {side} qty={abs(qty)} @ {price}")
            return True
            
        except Exception as e:
            self.log(f"❌ open_position lỗi: {e}")
            return False

    def partial_close(self, fraction, reason=""):
        try:
            if not self.position_open or fraction <= 0 or fraction >= 1: return False
            qty_to_close = abs(self.qty) * fraction
            if qty_to_close <= 0: return False
            
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

    def close_position(self, reason=""):
        if not self.position_open or self._close_attempted:
            return False
            
        try:
            self._close_attempted = True
            side = "SELL" if self.side == "BUY" else "BUY"
            qty_to_close = abs(self.qty)
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.2)
            
            res = place_order(self.symbol, side, qty_to_close, self.api_key, self.api_secret)
            if res and 'orderId' in res:
                self.log(f"✅ Đóng lệnh: {reason}")
                old_symbol = self.symbol
                self._reset_position()
                self.last_close_time = time.time()
                
                try:
                    self.coin_manager.set_cooldown(old_symbol)
                    self.log(f"⏳ COOLDOWN {old_symbol} ({self.coin_manager.cooldown_left(old_symbol)}s)")
                except Exception:
                    pass
                    
                if self.dynamic_mode:
                    self._find_new_coin_after_exit()
                return True
            else:
                self.log("❌ Đóng lệnh thất bại")
                self._close_attempted = False
                return False
                
        except Exception as e:
            self.log(f"❌ close_position lỗi: {e}")
            self._close_attempted = False
            return False

    def _reset_position(self):
        self.position_open = False
        self.side = None
        self.qty = 0.0
        self.entry = 0.0
        self._close_attempted = False
        self.smart_exit.trailing_active = False
        self.smart_exit._breakeven_active = False
        self.smart_exit._tp_hit.clear()

    def check_tp_sl(self):
        if self.position_open and self.entry > 0:
            current_price = get_current_price(self.symbol)
            if current_price > 0:
                exit_reason = self.smart_exit.check_all_exit_conditions(current_price)
                if exit_reason:
                    self.close_position(exit_reason)
                    return
                    
        if not self.position_open or self.entry <= 0 or self._close_attempted:
            return
            
        current_price = get_current_price(self.symbol)
        if current_price <= 0: return
        
        if self.side == "BUY":
            profit = (current_price - self.entry) * abs(self.qty)
        else:
            profit = (self.entry - current_price) * abs(self.qty)
            
        invested = self.entry * abs(self.qty) / self.lev
        if invested <= 0: return
        
        roi = (profit / invested) * 100

        if self.tp is not None and roi >= self.tp:
            self.close_position(f"✅ Đạt TP {self.tp}% (ROI: {roi:.2f}%)")
        elif self.sl is not None and self.sl > 0 and roi <= -self.sl:
            self.close_position(f"❌ Đạt SL {self.sl}% (ROI: {roi:.2f}%)")

    def _restart_websocket_for_new_coin(self):
        try:
            self.ws_manager.remove_symbol(self.symbol)
            time.sleep(2)
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
            self.log(f"🔗 Restart WS cho {self.symbol}")
        except Exception as e:
            self.log(f"❌ WS restart lỗi: {e}")

    def _find_new_coin_after_exit(self):
        try:
            self.log("🔄 Tìm coin mới (2 ứng viên)...")
            if self._candidate_pool:
                cached = self._candidate_pool.pop(0)
                if not self.coin_manager.is_in_cooldown(cached):
                    old_symbol = self.symbol
                    self.coin_manager.unregister_coin(old_symbol)
                    self.symbol = cached
                    if self.coin_manager.register_coin(cached, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key):
                        self._restart_websocket_for_new_coin()
                        self.log(f"🔁 Dùng ứng viên sẵn có: {old_symbol} → {cached}")
                        return
                        
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
                primary = new_symbols[0]
                backup  = new_symbols[1] if len(new_symbols) > 1 else None
                old_symbol = self.symbol
                self.coin_manager.unregister_coin(old_symbol)
                self.symbol = primary
                
                if self.coin_manager.register_coin(primary, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key):
                    self._restart_websocket_for_new_coin()
                    msg = f"🔄 Chuyển {old_symbol} → {primary}"
                    if backup:
                        self._candidate_pool = [backup]
                        msg += f" | Backup: {backup}"
                    self.log(msg)
                else:
                    self.log(f"❌ Không đăng ký được {primary}")
            else:
                self.log("❌ Không tìm thấy coin mới phù hợp")
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {e}")
            traceback.print_exc()

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

# ========== CÁC CHIẾN LƯỢC GIAO DỊCH NÂNG CẤP ==========
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
            if len(self.prices) < 60: return None
            closes = self.prices[-210:]
            highs = [p*1.0005 for p in closes]
            lows  = [p*0.9995 for p in closes]
            rsi = rsi_wilder_last(closes, self.rsi_period)
            ema_f = ema_last(closes, self.ema_fast)
            ema_s = ema_last(closes, self.ema_slow)
            mid, upper, lower = bbands_last(closes, 20, 2)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            if None in (rsi, ema_f, ema_s, mid, upper, lower, adx): return None
            last = closes[-1]
            if rsi < self.rsi_oversold and ema_f > ema_s and (adx >= 20 or last <= lower):
                return "BUY"
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
            if len(self.prices) < 60: return None
            closes = self.prices[-210:]
            highs = [p*1.0005 for p in closes]
            lows  = [p*0.9995 for p in closes]
            ema_f = ema_last(closes, self.ema_fast)
            ema_s = ema_last(closes, self.ema_slow)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            if None in (ema_f, ema_s, adx): return None
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
            if now - self.last_24h_check < 60: return None
            url = f"{BASE_FAPI}/fapi/v1/ticker/24hr"
            data = binance_api_request(url, {"symbol": self.symbol})
            self.last_24h_check = now
            if not data: return None
            ch = float(data.get('priceChangePercent', 0.0) or 0.0)
            if abs(ch - self.last_reported_change) > 5:
                self.log(f"📊 24h: {ch:.2f}% | Ngưỡng: {self.threshold}%")
                self.last_reported_change = ch
            closes = self.prices[-210:]
            if len(closes) < 40: return None
            highs = [p*1.0006 for p in closes]
            lows  = [p*0.9994 for p in closes]
            mid, up, low = bbands_last(closes, 20, 2)
            rsi = rsi_wilder_last(closes, 14)
            if None in (mid, up, low, rsi): return None
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
            if len(self.prices) < 100: return None
            closes = self.prices[-240:]
            highs = [p*1.0006 for p in closes]
            lows  = [p*0.9994 for p in closes]
            adx, _, _ = adx_last(highs, lows, closes, 14)
            ema_fast = ema_last(closes, 20)
            ema_slow = ema_last(closes, 50)
            if None in (adx, ema_fast, ema_slow): return None
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
            if len(self.prices) < 60: return None
            closes = self.prices[-120:]
            highs = [p*1.0007 for p in closes]
            lows  = [p*0.9993 for p in closes]
            vols  = [1.0]*len(closes)
            rsi = rsi_wilder_last(closes, self.rsi_period)
            mid, up, low = bbands_last(closes, 20, 2)
            mfi = mfi_last(highs, lows, closes, vols, 14)
            if None in (rsi, mid, up, low, mfi): return None
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
            if len(self.prices) < 120: return None
            closes = self.prices[-240:]
            highs = [p*1.0006 for p in closes]
            lows  = [p*0.9994 for p in closes]
            rsi = rsi_wilder_last(closes, 14)
            ema9 = ema_last(closes, 9)
            ema21 = ema_last(closes, 21)
            mid, up, low = bbands_last(closes, 20, 2)
            adx, _, _ = adx_last(highs, lows, closes, 14)
            if None in (rsi, ema9, ema21, mid, up, low, adx): return None
            score_buy = 0; score_sell = 0
            if rsi < 30 and ema9 > ema21: score_buy += 2
            if rsi > 70 and ema9 < ema21: score_sell += 2
            if adx >= 25 and ema9 > ema21: score_buy += 1
            if adx >= 25 and ema9 < ema21: score_sell += 1
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
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES THÔNG MINH</b>\n\n🎯 <b>HỆ THỐNG ĐA CHIẾN LƯỢC + SMART EXIT + BOT ĐỘNG</b>"
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

    def _auto_scan_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                removed_count = 0
                for bot_id in list(self.bots.keys()):
                    bot = self.bots[bot_id]
                    if (hasattr(bot, 'should_be_removed') and bot.should_be_removed and
                        bot.strategy_name in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following", "Smart Dynamic"]):
                        
                        strategy_type = bot.strategy_name
                        config_key = getattr(bot, 'config_key', None)
                        if config_key:
                            self.strategy_cooldowns[strategy_type][config_key] = current_time
                            self.log(f"⏰ Đã thêm cooldown cho {strategy_type} - {config_key}")
                        
                        self.log(f"🔄 Tự động xóa bot {bot_id} (đã đóng lệnh)")
                        self.stop_bot(bot_id)
                        removed_count += 1
                        time.sleep(0.5)
                
                if (removed_count > 0 or 
                    current_time - self.last_auto_scan > self.auto_scan_interval):
                    
                    if removed_count > 0:
                        self.log(f"🗑️ Đã xóa {removed_count} bot, đợi 10s trước khi quét coin mới")
                        time.sleep(10)
                    
                    self._scan_auto_strategies()
                    self.last_auto_scan = current_time
                
                time.sleep(30)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(30)

    def _scan_auto_strategies(self):
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
                dynamic_mode = strategy_config.get('dynamic_mode', False)
                
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
                "Trend Following": Trend_Following_Bot,
                "Smart Dynamic": Smart_Dynamic_Bot
            }.get(strategy_type)
            
            if not bot_class:
                return False
            
            if strategy_type == "Reverse 24h":
                threshold = config.get('threshold', 30)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, threshold, strategy_key, smart_exit_config, dynamic_mode)
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
        try:
            if sl == 0:
                sl = None
                
            if not self.api_key or not self.api_secret:
                self.log("❌ Chưa thiết lập API Key trong BotManager")
                return False
            
            test_balance = get_balance(self.api_key, self.api_secret)
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
            
            elif dynamic_mode and strategy_type in ["Reverse 24h", "Scalping", "Trend Following"]:
                bot_created = self._create_dynamic_bot(
                    strategy_type, lev, percent, tp, sl, 
                    smart_exit_config, threshold, volatility, grid_levels
                )
            
            else:
                bot_created = self._create_static_bot(
                    symbol, strategy_type, lev, percent, tp, sl, smart_exit_config
                )
            
            return bot_created
            
        except Exception as e:
            self.log(f"❌ Lỗi nghiêm trọng trong add_bot: {str(e)}")
            import traceback
            self.log(f"🔍 Chi tiết lỗi: {traceback.format_exc()}")
            return False
    
    def _create_smart_dynamic_bot(self, lev, percent, tp, sl, smart_exit_config, dynamic_mode):
        try:
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
                'strategy_key': strategy_key,
                'smart_exit_config': smart_exit_config,
                'dynamic_mode': True
            }
            
            qualified_symbols = self._find_qualified_symbols(
                "Smart Dynamic", lev, self.auto_strategies[strategy_key], strategy_key
            )
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, "Smart Dynamic", self.auto_strategies[strategy_key])
                    if success:
                        success_count += 1
                        time.sleep(0.5)
            
            if success_count > 0:
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {success_count} BOT ĐỘNG THÔNG MINH</b>\n\n"
                    f"🎯 Chiến lược: Smart Dynamic\n"
                    f"💰 Đòn bẩy: {lev}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl}%\n"
                    f"🤖 Coin: {', '.join(qualified_symbols[:success_count])}\n\n"
                    f"🔑 <b>Config Key:</b> {strategy_key}"
                )
                self.log(success_msg)
                return True
            else:
                self.log("⚠️ Smart Dynamic: chưa tìm thấy coin phù hợp")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tạo Smart Dynamic bot: {str(e)}")
            return False
    
    def _create_dynamic_bot(self, strategy_type, lev, percent, tp, sl, smart_exit_config, threshold, volatility, grid_levels):
        try:
            strategy_key = f"{strategy_type}_{lev}_{percent}_{tp}_{sl}"
            
            if strategy_type == "Reverse 24h":
                strategy_key += f"_th{threshold or 30}"
            elif strategy_type == "Scalping":
                strategy_key += f"_vol{volatility or 3}"
            
            if self._is_in_cooldown(strategy_type, strategy_key):
                self.log(f"⏰ {strategy_type} (Config: {strategy_key}): đang trong cooldown")
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
            
            qualified_symbols = self._find_qualified_symbols(
                strategy_type, lev, config, strategy_key
            )
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, strategy_type, config)
                    if success:
                        success_count += 1
                        time.sleep(0.5)
            
            if success_count > 0:
                success_msg = self._format_success_message(strategy_type, lev, percent, tp, sl, 
                                                         qualified_symbols[:success_count], strategy_key,
                                                         threshold, volatility, grid_levels)
                self.log(success_msg)
                return True
            else:
                self.log(f"⚠️ {strategy_type}: chưa tìm thấy coin phù hợp")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tạo {strategy_type} bot: {str(e)}")
            return False
    
    def _create_static_bot(self, symbol, strategy_type, lev, percent, tp, sl, smart_exit_config):
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
                "Smart Dynamic": Smart_Dynamic_Bot
            }.get(strategy_type)
            
            if not bot_class:
                self.log(f"❌ Chiến lược {strategy_type} không được hỗ trợ")
                return False
            
            if strategy_type == "Reverse 24h":
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, threshold=30, smart_exit_config=smart_exit_config)
            else:
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, smart_exit_config=smart_exit_config)
            
            self.bots[bot_id] = bot
            self.log(f"✅ Đã thêm bot {strategy_type}: {symbol} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot tĩnh {symbol}: {str(e)}")
            return False
    
    def _format_success_message(self, strategy_type, lev, percent, tp, sl, symbols, strategy_key, threshold, volatility, grid_levels):
        message = (
            f"✅ <b>ĐÃ TẠO {len(symbols)} BOT {strategy_type}</b>\n\n"
            f"🎯 Chiến lược: {strategy_type}\n"
            f"💰 Đòn bẩy: {lev}x\n"
            f"📊 % Số dư: {percent}%\n"
            f"🎯 TP: {tp}%\n"
            f"🛡️ SL: {sl}%\n"
        )
        
        if threshold:
            message += f"📈 Ngưỡng: {threshold}%\n"
        if volatility:
            message += f"⚡ Biến động: {volatility}%\n"
        if grid_levels:
            message += f"🛡️ Số lệnh: {grid_levels}\n"
        
        message += f"🤖 Coin: {', '.join(symbols)}\n\n"
        message += f"🔑 <b>Config Key:</b> {strategy_key}\n"
        message += f"🔄 <i>Bot sẽ tự động tìm coin mới sau khi đóng lệnh</i>"
        
        return message

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
        
        if current_step == 'waiting_bot_mode':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
                if text == "🤖 Bot Tĩnh - Coin cụ thể":
                    user_state['dynamic_mode'] = False
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
                    user_state['dynamic_mode'] = True
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT ĐỘNG</b>\n\n"
                        "🤖 Bot sẽ TỰ ĐỘNG tìm coin phù hợp\n"
                        "🔄 Tự chuyển coin sau mỗi lệnh\n"
                        "📈 Luôn giao dịch coin tốt nhất\n\n"
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
                         "📈 Trend Following", "⚡ Scalping", "🔄 Bot Động Thông Minh"]:
                
                strategy_map = {
                    "🤖 RSI/EMA Recursive": "RSI/EMA Recursive",
                    "📊 EMA Crossover": "EMA Crossover", 
                    "🎯 Reverse 24h": "Reverse 24h",
                    "📈 Trend Following": "Trend Following",
                    "⚡ Scalping": "Scalping",
                    "🔄 Bot Động Thông Minh": "Smart Dynamic"
                }
                
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                
                if strategy == "Smart Dynamic":
                    user_state['dynamic_mode'] = True
                
                user_state['step'] = 'waiting_exit_strategy'
                
                strategy_descriptions = {
                    "RSI/EMA Recursive": "Phân tích RSI + EMA đệ quy",
                    "EMA Crossover": "Giao cắt EMA nhanh/chậm", 
                    "Reverse 24h": "Đảo chiều biến động 24h",
                    "Trend Following": "Theo xu hướng giá",
                    "Scalping": "Giao dịch tốc độ cao",
                    "Smart Dynamic": "Bot động thông minh đa chiến lược"
                }
                
                description = strategy_descriptions.get(strategy, "")
                mode_text = "ĐỘNG" if user_state.get('dynamic_mode') else "TĨNH"
                
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {strategy}</b>\n"
                    f"🤖 <b>Chế độ: {mode_text}</b>\n\n"
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
                f"🔄 <b>Bot Động:</b>\n• TỰ ĐỘNG tìm coin tốt nhất\n• Tự chuyển coin sau mỗi lệnh\n• Luôn giao dịch coin tiềm năng",
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
                smart_bots = 0
                for bot_id, bot in self.bots.items():
                    status = "🟢 Mở" if bot.position_open else "🟡 Chờ"
                    mode = "🔄 Động" if getattr(bot, 'dynamic_mode', False) else "🤖 Tĩnh"
                    exit_type = "🔴 Thường" 
                    if hasattr(bot, 'smart_exit') and bot.smart_exit.config['enable_trailing']:
                        exit_type = "🟢 Thông minh"
                        smart_bots += 1
                    if getattr(bot, 'dynamic_mode', False):
                        dynamic_bots += 1
                    message += f"🔹 {bot_id} | {status} | {mode} | {exit_type}\n"
                
                message += f"\n📊 Tổng số: {len(self.bots)} bot | 🔄 Động: {dynamic_bots} | 🤖 Thông minh: {smart_bots}"
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
                "• Tự chuyển coin sau mỗi lệnh\n"
                "• Smart Exit 4 cơ chế\n\n"
                
                "🎯 <b>Reverse 24h</b> - HỖ TRỢ ĐỘNG/TĨNH\n"
                "• Đảo chiều biến động 24h\n"
                "• Tự tìm coin từ TOÀN BỘ Binance\n"
                "• Smart Exit bảo vệ lợi nhuận\n\n"
                
                "⚡ <b>Scalping</b> - HỖ TRỢ ĐỘNG/TĨNH\n"
                "• Giao dịch tốc độ cao\n"
                "• Tự tìm coin biến động\n"
                "• Smart Exit chốt lời nhanh\n\n"
                
                "📈 <b>Trend Following</b> - HỖ TRỢ ĐỘNG/TĨNH\n"
                "• Theo xu hướng giá\n"
                "• Tự tìm coin trend rõ\n"
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
            
            dynamic_bots_count = sum(1 for bot in self.bots.values() if getattr(bot, 'dynamic_mode', False))
            smart_bots_count = sum(1 for bot in self.bots.values() 
                                 if hasattr(bot, 'smart_exit') and bot.smart_exit.config['enable_trailing'])
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG THÔNG MINH</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🔄 Bot động: {dynamic_bots_count}\n"
                f"🧠 Bot thông minh: {smart_bots_count}\n"
                f"📊 Chiến lược: {len(set(bot.strategy_name for bot in self.bots.values()))}\n"
                f"🔄 Auto scan: {len(self.auto_strategies)} cấu hình\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n"
                f"⏰ Cooldown: {self.cooldown_period//60} phút"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        strategy = user_state.get('strategy')
        dynamic_mode = user_state.get('dynamic_mode', False)
        
        if dynamic_mode and strategy != "Smart Dynamic":
            if strategy == "Reverse 24h":
                user_state['step'] = 'waiting_threshold'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"Chọn ngưỡng biến động (%):",
                    chat_id,
                    create_threshold_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "Scalping":
                user_state['step'] = 'waiting_volatility'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"Chọn biến động tối thiểu (%):",
                    chat_id,
                    create_volatility_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
        else:
            if not dynamic_mode:
                user_state['step'] = 'waiting_symbol'
                send_telegram(
                    f"🎯 <b>BOT TĨNH: {strategy}</b>\n\n"
                    f"Chọn cặp coin:",
                    chat_id,
                    create_symbols_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG THÔNG MINH</b>\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()
