# ==============================================================
# trading_bot_lib_v2_full.py — FULL VERSION (LINES 1..1000)
# ==============================================================
# Giữ nguyên cấu trúc bot gốc, và gộp toàn bộ nâng cấp đã thảo luận:
# 1) Hạ tầng: logging, Telegram, Binance REST helpers (ký/signed)
# 2) Chỉ báo nâng cao: RSI (Wilder), EMA, ATR%, ADX, Bollinger, MFI, OBV, Regime
# 3) CoinManager: quản lý coin theo bot + COOLDOWN per-coin
# 4) WebSocketManager: mô phỏng bằng REST polling (giữ API add/remove)
# 5) SmartExit 2.0: trailing adaptive theo ATR%, breakeven, TP ladder (partial close)
# 6) BaseBot: giữ API cũ, thêm partial_close, scanner chuyển coin sau khi đóng lệnh
# 7) Scanner: chọn 2 coin/chiến lược, bỏ coin đang cooldown, tôn trọng leverage/step
# 8) Chiến lược: RSI/EMA, EMA Crossover, Reverse 24h, Trend Following, Scalping, Smart Dynamic
# 9) BotManager: giữ API cũ (start_bot/stop), gắn WS và CoinManager
# --------------------------------------------------------------
# Cách dùng nhanh:
#   - Điền API vào cuối file (ví dụ trong phần __main__) hoặc module riêng.
#   - Khởi tạo BotManager và ít nhất 1 bot (Smart_Dynamic_Bot ...), start_bot.
# ==============================================================

# =============================== IMPORTS ===============================
import json
import hmac
import hashlib
import time
import threading
import urllib.request
import urllib.parse
import numpy as np
import logging
import requests
import math
import traceback
from datetime import datetime
from heapq import nlargest
from typing import List, Tuple, Optional, Dict, Any

# =========================== LOGGING & TELEGRAM =========================

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


def send_telegram(message: str, chat_id: Optional[str] = None, reply_markup=None,
                  bot_token: Optional[str] = None, default_chat_id: Optional[str] = None):
    """Gửi log ra Telegram (tùy chọn)."""
    if not bot_token:
        return
    chat_id = chat_id or default_chat_id
    if not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            logger.error(f"Telegram {r.status_code}: {r.text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ============================= BINANCE HELPERS ==========================

BASE_FAPI = "https://fapi.binance.com"


def signed_request(url_path: str, params: dict, api_key: str, secret_key: str, method: str = 'GET'):
    query = urllib.parse.urlencode(params)
    signature = hmac.new(secret_key.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    url = f"{BASE_FAPI}{url_path}?{query}&signature={signature}"
    req = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read().decode()
        return json.loads(data)


def binance_api_request(url: str, params: dict = None):
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 200:
            return r.json()
        logger.error(f"GET {url} {r.status_code}: {r.text}")
        return None
    except Exception as e:
        logger.error(f"GET error {url}: {e}")
        return None


def get_all_usdt_pairs(limit: int = 300) -> List[str]:
    """Lấy danh sách hợp đồng futures USDT (PERPETUAL/QUARTER)."""
    url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
    data = binance_api_request(url)
    out = []
    if data and 'symbols' in data:
        for sym in data['symbols']:
            if sym.get('contractType') in ("PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER") and sym.get('quoteAsset') == 'USDT':
                out.append(sym['symbol'])
    return out[:limit]


def get_klines(symbol: str, interval: str = "5m", limit: int = 210) -> Optional[Tuple[list, list, list, list, list]]:
    """Trả về (opens, highs, lows, closes, volumes) dạng list[float]."""
    url = f"{BASE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    data = binance_api_request(url, params)
    if not data:
        return None
    o, h, l, c, v = [], [], [], [], []
    for r in data:
        o.append(float(r[1])); h.append(float(r[2])); l.append(float(r[3]))
        c.append(float(r[4])); v.append(float(r[5]))
    return o, h, l, c, v


def get_current_price(symbol: str) -> float:
    try:
        url = f"{BASE_FAPI}/fapi/v1/ticker/price"
        data = binance_api_request(url, {"symbol": symbol.upper()})
        if data and 'price' in data:
            return float(data['price'])
        return 0.0
    except Exception:
        return 0.0


def get_balance(api_key: str, api_secret: str):
    try:
        ts = int(time.time()*1000)
        url_path = "/fapi/v2/balance"
        params = {"timestamp": ts}
        data = signed_request(url_path, params, api_key, api_secret, 'GET')
        return data
    except Exception:
        return None


def set_leverage(symbol: str, leverage: int, api_key: str, api_secret: str) -> bool:
    try:
        ts = int(time.time()*1000)
        url_path = "/fapi/v1/leverage"
        params = {"symbol": symbol.upper(), "leverage": leverage, "timestamp": ts}
        signed_request(url_path, params, api_key, api_secret, 'POST')
        return True
    except Exception as e:
        logger.error(f"set_leverage({symbol}) lỗi: {e}")
        return False


def get_step_size(symbol: str, api_key: str, api_secret: str) -> float:
    try:
        url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return 0
        for s in data.get('symbols', []):
            if s.get('symbol') == symbol:
                for f in s.get('filters', []):
                    if f.get('filterType') == 'LOT_SIZE':
                        return float(f.get('stepSize', '0'))
        return 0
    except Exception:
        return 0


def cancel_all_orders(symbol: str, api_key: str, api_secret: str) -> bool:
    try:
        ts = int(time.time()*1000)
        url_path = "/fapi/v1/allOpenOrders"
        params = {"symbol": symbol.upper(), "timestamp": ts}
        signed_request(url_path, params, api_key, api_secret, 'DELETE')
        return True
    except Exception:
        return False


def place_order(symbol: str, side: str, quantity: float, api_key: str, api_secret: str):
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

# ============================ INDICATORS ==============================


def rsi_wilder_last(prices: list, period: int = 14) -> Optional[float]:
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


def ema_last(values: list, period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2/(period+1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v*k + ema*(1-k)
    return float(ema)


def atr_last(highs: list, lows: list, closes: list, period: int = 14, return_pct: bool = True) -> Optional[float]:
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
    if return_pct and closes[-1] > 0:
        return float(atr / closes[-1] * 100.0)
    return float(atr)


def adx_last(highs: list, lows: list, closes: list, period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
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
    adx = dx  # xấp xỉ cuối
    return float(adx), float(plus_di), float(minus_di)


def bbands_last(closes: list, period: int = 20, std: int = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < period:
        return None, None, None
    w = np.array(closes[-period:])
    mid = w.mean(); dev = w.std(ddof=0)
    return float(mid), float(mid+std*dev), float(mid-std*dev)


def mfi_last(highs: list, lows: list, closes: list, volumes: list, period: int = 14) -> Optional[float]:
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


def obv_last(closes: list, volumes: list) -> float:
    if len(closes) < 2 or len(volumes) < 2:
        return 0.0
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
    return float(obv)


def detect_regime(highs: list, lows: list, closes: list) -> Tuple[str, float, float]:
    adx, _, _ = adx_last(highs, lows, closes, 14)
    atrp = atr_last(highs, lows, closes, 14, True)
    regime = 'unknown'
    if adx is not None:
        if adx >= 25: regime = 'trend'
        elif adx < 20: regime = 'range'
        else: regime = 'transition'
    if atrp is not None and atrp >= 5.0:
        regime += '|hi-vol'
    return regime, (adx or 0.0), (atrp or 0.0)

# ============================== COIN MANAGER ============================

class CoinManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._instance.symbol_to_bot: Dict[str, str] = {}
            cls._instance.symbol_configs: Dict[str, tuple] = {}
            cls._instance.active_configs: Dict[tuple, set] = {}
            # === COOLDOWN per-coin ===
            cls._instance.symbol_cooldowns: Dict[str, float] = {}
            cls._instance.cooldown_seconds: int = 20*60
        return cls._instance

    def register_coin(self, symbol: str, bot_id: str, strategy_name: str, config_key: Optional[str] = None) -> bool:
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

    def unregister_coin(self, symbol: str):
        with self._lock:
            if symbol in self.symbol_to_bot:
                del self.symbol_to_bot[symbol]
            if symbol in self.symbol_configs:
                tup = self.symbol_configs.pop(symbol)
                key = (tup[0], tup[1])
                if key in self.active_configs and symbol in self.active_configs[key]:
                    self.active_configs[key].remove(symbol)

    def has_same_config_bot(self, symbol: str, config_key: Optional[str]) -> bool:
        with self._lock:
            if symbol not in self.symbol_configs:
                return False
            _, cfg = self.symbol_configs.get(symbol, (None, None))
            return cfg == config_key

    # ============== COOLDOWN API ==============
    def set_cooldown(self, symbol: str, seconds: Optional[int] = None):
        ts = time.time() + (seconds or self.cooldown_seconds)
        with self._lock:
            self.symbol_cooldowns[symbol.upper()] = ts

    def is_in_cooldown(self, symbol: str) -> bool:
        with self._lock:
            ts = self.symbol_cooldowns.get(symbol.upper())
            if not ts:
                return False
            if time.time() >= ts:
                # Hết cooldown -> dọn
                del self.symbol_cooldowns[symbol.upper()]
                return False
            return True

    def cooldown_left(self, symbol: str) -> int:
        with self._lock:
            ts = self.symbol_cooldowns.get(symbol.upper())
            return max(0, int(ts - time.time())) if ts else 0

# ============================= WEBSOCKET (POLL) =========================

class WebSocketManager:
    """Giữ API add_symbol/remove_symbol nhưng backend là polling REST."""
    def __init__(self):
        self.symbol_callbacks: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = True

    def add_symbol(self, symbol: str, callback):
        self.symbol_callbacks[symbol] = callback
        t = threading.Thread(target=self._run_symbol, args=(symbol,), daemon=True)
        t.start()
        self.threads[symbol] = t

    def remove_symbol(self, symbol: str):
        if symbol in self.symbol_callbacks:
            del self.symbol_callbacks[symbol]

    def _run_symbol(self, symbol: str):
        last_price = 0.0
        while self.running and symbol in self.symbol_callbacks:
            p = get_current_price(symbol)
            if p and p != last_price:
                last_price = p
                try:
                    self.symbol_callbacks[symbol](p)
                except Exception:
                    pass
            time.sleep(1)

# ============================= SMART EXIT 2.0 ===========================

class SmartExitManager:
    """Quản lý đóng lệnh thông minh: trailing, time, volume, SR + nâng cấp breakeven & TP ladder."""
    def __init__(self, bot_instance: 'BaseBot'):
        self.bot = bot_instance
        self.config = {
            'enable_trailing': True,
            'enable_time_exit': False,
            'enable_volume_exit': False,
            'enable_support_resistance': False,
            'trailing_activation': 30,   # ROI% kích hoạt trailing
            'trailing_distance': 15,     # % khoảng cách trailing mặc định
            'max_hold_time': 6,          # giờ
            'min_profit_for_exit': 10,   # ROI% tối thiểu để cho phép exit có lãi
            # === Nâng cấp ===
            'breakeven_at': 12,          # ROI% kích hoạt dời về hòa vốn
            'trail_adaptive': True,      # trailing thích ứng theo ATR%
            'tp_ladder': [               # TP bậc thang theo ROI%
                {'roi': 15, 'pct': 0.30},
                {'roi': 25, 'pct': 0.30},
                {'roi': 40, 'pct': 0.40},
            ],
        }
        self.trailing_active = False
        self.peak_price = 0.0
        self.position_open_time = 0.0
        self.volume_history: list = []
        self._breakeven_active = False
        self._tp_hit: set = set()

    def update_config(self, **kwargs):
        changed = {}
        for k, v in kwargs.items():
            if k in self.config:
                self.config[k] = v
                changed[k] = v
        if changed:
            self.bot.log(f"⚙️ Cập nhật Smart Exit: {changed}")

    def _calculate_roi(self, current_price: float) -> float:
        if not self.bot.position_open or self.bot.entry <= 0 or abs(self.bot.qty) <= 0:
            return 0.0
        if self.bot.side == "BUY":
            profit = (current_price - self.bot.entry) * abs(self.bot.qty)
        else:
            profit = (self.bot.entry - current_price) * abs(self.bot.qty)
        invested = self.bot.entry * abs(self.bot.qty) / self.bot.lev
        if invested <= 0:
            return 0.0
        return (profit / invested) * 100.0

    def _check_trailing_stop(self, current_price: float) -> Optional[str]:
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

    def check_all_exit_conditions(self, current_price: float, current_volume: Optional[float] = None) -> Optional[str]:
        if not self.bot.position_open:
            return None
        exit_reasons = []
        # trailing
        if self.config['enable_trailing']:
            r = self._check_trailing_stop(current_price)
            if r:
                exit_reasons.append(r)
        # time-based
        if self.config['enable_time_exit'] and self.position_open_time > 0:
            hold_hours = (time.time() - self.position_open_time) / 3600
            if hold_hours >= self.config['max_hold_time']:
                exit_reasons.append(f"⏰ Quá thời gian giữ {self.config['max_hold_time']}h")
        # volume-based (placeholder)
        if self.config['enable_volume_exit'] and current_volume is not None:
            self.volume_history.append(current_volume)
            if len(self.volume_history) > 60:
                self.volume_history.pop(0)
        # support/resistance (placeholder)
        if self.config['enable_support_resistance']:
            pass

        # === Breakeven + TP ladder ===
        current_roi = self._calculate_roi(current_price)
        if (not self._breakeven_active) and (current_roi >= self.config.get('breakeven_at', 12)):
            self._breakeven_active = True
            self.config['min_profit_for_exit'] = max(self.config.get('min_profit_for_exit', 10), 0)
            self.bot.log(f"🟩 Kích hoạt Breakeven tại ROI {current_roi:.1f}%")
        for step in self.config.get('tp_ladder', []):
            roi_lv = step.get('roi', 0)
            pct = step.get('pct', 0)
            key = f"tp_{roi_lv}"
            if current_roi >= roi_lv and key not in self._tp_hit:
                ok = self.bot.partial_close(pct, reason=f"TP ladder {roi_lv}%")
                if ok:
                    self._tp_hit.add(key)

        return exit_reasons[0] if exit_reasons else None

# ================================ BASE BOT ==============================

class BaseBot:
    def __init__(self, symbol: str, lev: int, percent: float, tp: Optional[float], sl: Optional[float],
                 ws_manager: 'WebSocketManager', api_key: str, api_secret: str,
                 telegram_bot_token: Optional[str], telegram_chat_id: Optional[str],
                 strategy_name: str, config_key: Optional[str] = None,
                 smart_exit_config: Optional[dict] = None, dynamic_mode: bool = False):
        self.symbol = symbol
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
        # position state
        self.position_open = False
        self.side: Optional[str] = None
        self.qty: float = 0.0
        self.entry: float = 0.0
        self._close_attempted = False
        self.prices: list = []
        # managers
        self.coin_manager = CoinManager()
        self.smart_exit = SmartExitManager(self)
        if smart_exit_config:
            self.smart_exit.update_config(**smart_exit_config)
        # dynamic candidates pool
        self._candidate_pool: list = []
        self.last_close_time = 0.0

    # -------- logging
    def log(self, msg: str):
        logger.info(f"[{self.strategy_name}][{self.symbol}] {msg}")
        try:
            send_telegram(f"[{self.strategy_name}][{self.symbol}] {msg}", self.telegram_chat_id, bot_token=self.telegram_bot_token)
        except Exception:
            pass

    # -------- price stream callback
    def _handle_price_update(self, price: float):
        try:
            self.prices.append(float(price))
            if len(self.prices) > 1200:
                self.prices.pop(0)
            sig = self.get_signal()
            if sig and not self.position_open:
                self.open_position(sig)
            self.check_tp_sl()
        except Exception as e:
            self.log(f"Lỗi price update: {e}")

    # -------- MUST override
    def get_signal(self) -> Optional[str]:
        return None

    # -------- open/partial/close
    def open_position(self, side: str) -> bool:
        try:
            price = get_current_price(self.symbol)
            if price <= 0:
                return False
            balance = get_balance(self.api_key, self.api_secret)
            if not balance:
                return False
            usdt = 0.0
            for b in balance:
                if b.get('asset') == 'USDT':
                    # futures balance payload có thể dùng 'balance' hoặc 'availableBalance'
                    usdt = float(b.get('availableBalance', b.get('balance', 0)))
                    break
            if usdt <= 0:
                return False
            invest = usdt * self.percent
            qty = (invest * self.lev) / price
            step = get_step_size(self.symbol, self.api_key, self.api_secret)
            if step and step > 0:
                precision = int(round(-math.log10(step))) if step < 1 else 0
                qty = float(f"{qty:.{precision}f}")
            if qty <= 0:
                return False
            res = place_order(self.symbol, 'BUY' if side == 'BUY' else 'SELL', qty, self.api_key, self.api_secret)
            if not res:
                return False
            self.position_open = True
            self.side = side
            self.qty = qty if side == 'BUY' else -qty
            self.entry = price
            self.smart_exit.position_open_time = time.time()
            self._close_attempted = False
            self.log(f"🚀 Mở {side} qty={abs(qty)} @ {price}")
            return True
        except Exception as e:
            self.log(f"❌ open_position lỗi: {e}")
            return False

    def partial_close(self, fraction: float, reason: str = "") -> bool:
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

    def close_position(self, reason: str = "") -> bool:
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
        # 1) SmartExit trước
        if self.position_open and self.entry > 0:
            current_price = get_current_price(self.symbol)
            if current_price > 0:
                exit_reason = self.smart_exit.check_all_exit_conditions(current_price)
                if exit_reason:
                    self.close_position(exit_reason)
                    return
        # 2) TP/SL truyền thống
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

    def _restart_websocket_for_new_coin(self):
        try:
            # Lưu ý: cần remove trước khi đổi self.symbol; ta remove theo old symbol trong _find_new_coin
            time.sleep(1.5)
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
            self.log(f"🔗 Restart WS cho {self.symbol}")
        except Exception as e:
            self.log(f"❌ WS restart lỗi: {e}")

    def _find_new_coin_after_exit(self):
        try:
            self.log("🔄 Tìm coin mới (2 ứng viên)...")
            # Dùng pool sẵn có trước
            if self._candidate_pool:
                cached = self._candidate_pool.pop(0)
                if not self.coin_manager.is_in_cooldown(cached):
                    old_symbol = self.symbol
                    self.ws_manager.remove_symbol(old_symbol)
                    self.coin_manager.unregister_coin(old_symbol)
                    self.symbol = cached
                    if self.coin_manager.register_coin(cached, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key):
                        self._restart_websocket_for_new_coin()
                        self.log(f"🔁 Dùng ứng viên sẵn có: {old_symbol} → {cached}")
                        return
            # Scan mới 2 coin
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
                backup = new_symbols[1] if len(new_symbols) > 1 else None
                old_symbol = self.symbol
                self.ws_manager.remove_symbol(old_symbol)
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

# ================================ SCANNER ===============================

def _score_symbol_for_strategy(symbol: str, k5, k15, ticker: dict, strategy_type: str):
    o5, h5, l5, c5, v5 = k5
    o15, h15, l15, c15, v15 = k15
    atr5 = atr_last(h5, l5, c5, 14, True) or 0.0
    adx15, pdi, mdi = adx_last(h15, l15, c15, 14)
    adx15 = adx15 or 0.0
    ema9_15 = ema_last(c15, 9) or c15[-1]
    ema21_15 = ema_last(c15, 21) or c15[-1]
    trend_bias = 1 if ema9_15 > ema21_15 else -1 if ema9_15 < ema21_15 else 0
    vol_avg = np.mean(v5[-30:]) if len(v5) >= 30 else (sum(v5)/max(len(v5),1))
    vol_surge = (v5[-1]/max(vol_avg, 1e-9))
    abs_change = abs(float(ticker.get('priceChangePercent', 0.0) or 0.0))
    qvol = float(ticker.get('quoteVolume', 0.0) or 0.0)

    base_penalty = 0.0
    if symbol in ('BTCUSDT', 'ETHUSDT'):
        base_penalty = 0.5

    if strategy_type == "Reverse 24h":
        score = (abs_change/10.0) + min(vol_surge, 3.0) - base_penalty
        if atr5 >= 8.0:
            score -= 1.0
    elif strategy_type == "Scalping":
        score = min(vol_surge, 3.0) + min(atr5/2.0, 2.0) - base_penalty
        if atr5 > 10.0:
            score -= 1.0
    elif strategy_type == "Safe Grid":
        mid_vol = 2.0 <= atr5 <= 6.0
        score = (1.5 if mid_vol else 0.5) + min(qvol/5_000_000, 3.0) - base_penalty
    elif strategy_type == "Trend Following":
        score = (adx15/25.0) + (1.0 if trend_bias > 0 else 0.0) + min(vol_surge, 3.0) - base_penalty
    elif strategy_type == "Smart Dynamic":
        base = (min(adx15, 40)/40.0) + min(vol_surge, 3.0)
        if trend_bias != 0 and 3.0 <= atr5 <= 8.0:
            base += 1.0
        score = base - base_penalty
    else:
        score = min(vol_surge, 3.0) + (abs_change/20.0)

    return score, {"atr5%": atr5, "adx15": adx15, "vol_surge": vol_surge, "trend_bias": trend_bias}


def get_qualified_symbols(api_key: str, api_secret: str, strategy_type: str, leverage: int,
                          threshold: Optional[float] = None, volatility: Optional[str] = None, grid_levels: Optional[int] = None,
                          max_candidates: int = 20, final_limit: int = 2, strategy_key: Optional[str] = None) -> List[str]:
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
            if sym not in tickmap:
                continue
            if sym in ('BTCUSDT', 'ETHUSDT'):
                continue
            if strategy_key and coin_manager.has_same_config_bot(sym, strategy_key):
                continue
            if coin_manager.is_in_cooldown(sym):
                continue
            t = tickmap[sym]
            try:
                qvol = float(t.get('quoteVolume', 0.0) or 0.0)
                if qvol < 1_000_000:
                    continue
                k5 = get_klines(sym, "5m", 210)
                k15 = get_klines(sym, "15m", 210)
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
        final_syms: List[str] = []
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
                t = tickmap.get(sym)
                if not t:
                    continue
                try:
                    qv = float(t.get('quoteVolume', 0.0) or 0.0)
                    ch = abs(float(t.get('priceChangePercent', 0.0) or 0.0))
                    if qv > 3_000_000 and 0.5 <= ch <= 10.0 and sym not in ('BTCUSDT', 'ETHUSDT'):
                        backup.append((qv, sym))
                except Exception:
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

# ================================ STRATEGIES ============================
class RSI_EMA_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "RSI/EMA Recursive", smart_exit_config=smart_exit_config)
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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, smart_exit_config=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "EMA Crossover", smart_exit_config=smart_exit_config)
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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, threshold=30, config_key=None, smart_exit_config=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Reverse 24h", config_key, smart_exit_config=smart_exit_config)
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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None, smart_exit_config=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Trend Following", config_key, smart_exit_config=smart_exit_config)
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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None, smart_exit_config=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Scalping", config_key, smart_exit_config=smart_exit_config)
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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, grid_levels=5, config_key=None, smart_exit_config=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "Safe Grid", config_key, smart_exit_config=smart_exit_config)
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

# ========== BOT ĐỘNG THÔNG MINH ==========
class SmartDynamicBot(BaseBot):
    """BOT ĐỘNG THÔNG MINH - KẾT HỢP NHIỀU CHIẾN LƯỢC"""
    
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, smart_exit_config=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Smart Dynamic", config_key, smart_exit_config)
        
        # KÍCH HOẠT SMART EXIT MẶC ĐỊNH
        default_smart_config = {
            'enable_trailing': True,
            'enable_time_exit': True,
            'enable_support_resistance': True,
            'trailing_activation': 30,
            'trailing_distance': 15,
            'max_hold_time': 4,
            'min_profit_for_exit': 15
        }
        self.smart_exit.update_config(**default_smart_config)

    def get_signal(self):
        """KẾT HỢP NHIỀU CHIẾN LƯỢC ĐỂ RA TÍN HIỆU TỐI ƯU"""
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
            
            # VOLATILITY FILTER (tránh market quá biến động)
            if volatility > 8.0:
                score -= 1
            
            # CHỈ VÀO LỆNH KHI SCORE ĐỦ CAO
            if score >= 2:
                self.log(f"🎯 Smart Signal: {signal} | Score: {score}/3 | RSI: {rsi:.1f} | Trend: {trend_strength:.2f}")
                return signal
            
            return None

        except Exception as e:
            self.log(f"❌ Lỗi Smart Dynamic signal: {str(e)}")
            return None

    def _calculate_trend_strength(self):
        """Tính strength của trend"""
        if len(self.prices) < 20:
            return 0
            
        short_trend = (self.prices[-1] - self.prices[-5]) / self.prices[-5]
        medium_trend = (self.prices[-1] - self.prices[-10]) / self.prices[-10]
        long_trend = (self.prices[-1] - self.prices[-20]) / self.prices[-20]
        
        return (short_trend + medium_trend + long_trend) / 3

    def _calculate_volatility(self):
        """Tính độ biến động"""
        if len(self.prices) < 20:
            return 0
            
        returns = []
        for i in range(1, len(self.prices)):
            ret = (self.prices[i] - self.prices[i-1]) / self.prices[i-1]
            returns.append(abs(ret))
            
        return np.mean(returns) * 100

# ========== BOT MANAGER HOÀN CHỈNH VỚI BOT ĐỘNG TỰ TÌM COIN MỚI ==========
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
