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
from datetime import datetime
from heapq import nlargest
from typing import List, Tuple, Optional, Dict, Any

# =========================== LOGGING SETUP =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_complete.log')
    ]
)
logger = logging.getLogger()

# ============================= BINANCE CONFIG ==========================
BASE_FAPI = "https://fapi.binance.com"

# ============================ BINANCE API =====================
def binance_api_request(url: str, params: dict = None, timeout: int = 10):
    """Request tới Binance API với timeout"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        logger.error(f"GET {url} {response.status_code}: {response.text}")
        return None
    except Exception as e:
        logger.error(f"GET error {url}: {e}")
        return None

def signed_request(url_path: str, params: dict, api_key: str, secret_key: str, method: str = 'GET', timeout: int = 10):
    """Signed request với timeout"""
    try:
        query = urllib.parse.urlencode(params)
        signature = hmac.new(secret_key.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {"X-MBX-APIKEY": api_key}
        url = f"{BASE_FAPI}{url_path}?{query}&signature={signature}"
        
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=timeout)
        else:
            response = requests.post(url, headers=headers, timeout=timeout)
            
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Signed request error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Signed request exception: {e}")
        return None

def get_balance(api_key: str, api_secret: str) -> float:
    """Lấy số dư KHẢ DỤNG (availableBalance)"""
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v2/balance"
        params = {"timestamp": ts}
        data = signed_request(url_path, params, api_key, api_secret, 'GET')
        if data:
            for b in data:
                if b.get('asset') == 'USDT':
                    available_balance = float(b.get('availableBalance', 0))
                    total_balance = float(b.get('balance', 0))
                    logger.info(f"💰 Số dư khả dụng: {available_balance:.2f} USDT | Tổng: {total_balance:.2f} USDT")
                    return available_balance
        return 0.0
    except Exception as e:
        logger.error(f"get_balance error: {e}")
        return 0.0

def get_detailed_balance(api_key: str, api_secret: str) -> Dict[str, float]:
    """Lấy chi tiết số dư"""
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v2/balance"
        params = {"timestamp": ts}
        data = signed_request(url_path, params, api_key, api_secret, 'GET')
        if data:
            for b in data:
                if b.get('asset') == 'USDT':
                    return {
                        'available': float(b.get('availableBalance', 0)),
                        'total': float(b.get('balance', 0)),
                        'unrealized_pnl': float(b.get('crossUnPnl', 0)),
                        'margin': float(b.get('crossWalletBalance', 0))
                    }
        return {}
    except Exception as e:
        logger.error(f"get_detailed_balance error: {e}")
        return {}

def get_current_price(symbol: str) -> float:
    """Lấy giá hiện tại"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/ticker/price"
        data = binance_api_request(url, {"symbol": symbol.upper()})
        if data and 'price' in data:
            return float(data['price'])
        return 0.0
    except Exception:
        return 0.0

def get_klines(symbol: str, interval: str = "5m", limit: int = 100) -> Optional[Tuple[list, list, list, list, list]]:
    """Lấy dữ liệu kline"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        data = binance_api_request(url, params)
        
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

def set_leverage(symbol: str, leverage: int, api_key: str, api_secret: str) -> bool:
    """Set đòn bẩy"""
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v1/leverage"
        params = {
            "symbol": symbol.upper(),
            "leverage": leverage,
            "timestamp": ts
        }
        result = signed_request(url_path, params, api_key, api_secret, 'POST')
        return result is not None
    except Exception as e:
        logger.error(f"Lỗi set leverage {symbol}: {e}")
        return False

def place_order(symbol: str, side: str, quantity: float, api_key: str, api_secret: str) -> bool:
    """Đặt lệnh MARKET"""
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
        result = signed_request(url_path, params, api_key, api_secret, 'POST')
        return result is not None
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh {symbol}: {e}")
        return False

def get_positions(api_key: str, api_secret: str):
    """Lấy vị thế"""
    try:
        ts = int(time.time() * 1000)
        url_path = "/fapi/v2/positionRisk"
        params = {"timestamp": ts}
        return signed_request(url_path, params, api_key, api_secret, 'GET')
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {e}")
        return None

def get_all_usdt_pairs(limit: int = 50) -> List[str]:
    """Lấy danh sách cặp USDT"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        symbols = []
        
        if data and 'symbols' in data:
            for symbol in data['symbols']:
                if (symbol.get('contractType') in ["PERPETUAL"] and 
                    symbol.get('quoteAsset') == 'USDT' and
                    symbol.get('status') == 'TRADING'):
                    symbols.append(symbol['symbol'])
                    
        return symbols[:limit]
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách coin: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"]

def get_24h_change(symbol: str) -> Optional[float]:
    """Lấy biến động 24h"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/ticker/24hr"
        data = binance_api_request(url, {"symbol": symbol.upper()})
        if data and 'priceChangePercent' in data:
            return float(data['priceChangePercent'])
        return None
    except Exception as e:
        logger.error(f"Lỗi lấy biến động 24h {symbol}: {e}")
        return None

# ============================ INDICATORS =====================
def calc_rsi(prices: list, period: int = 14) -> Optional[float]:
    """Tính RSI"""
    if len(prices) < period + 1:
        return None
        
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100.0
        
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)

def calc_ema(prices: list, period: int) -> Optional[float]:
    """Tính EMA"""
    if len(prices) < period:
        return None
        
    k = 2.0 / (period + 1.0)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
        
    return float(ema)

# ============================ TELEGRAM MANAGER =======================
def send_telegram(message: str, chat_id: Optional[str] = None, reply_markup=None,
                  bot_token: Optional[str] = None, default_chat_id: Optional[str] = None):
    """Gửi tin nhắn Telegram"""
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
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error(f"Telegram {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")

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

def create_leverage_keyboard():
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

# ============================ COIN MANAGER ==================
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
            cls._instance.symbol_cooldowns: Dict[str, float] = {}
            cls._instance.cooldown_seconds: int = 20*60
            cls._instance.config_coin_count: Dict[str, int] = {}
        return cls._instance

    def register_coin(self, symbol: str, bot_id: str, strategy_name: str, config_key: Optional[str] = None) -> bool:
        with self._lock:
            if symbol in self.symbol_to_bot:
                return False
            
            key = (strategy_name, config_key)
            if key not in self.config_coin_count:
                self.config_coin_count[str(key)] = 0
            
            if self.config_coin_count[str(key)] >= 2:
                return False
                
            self.symbol_to_bot[symbol] = bot_id
            self.symbol_configs[symbol] = (strategy_name, config_key)
            
            if key not in self.active_configs:
                self.active_configs[key] = set()
            self.active_configs[key].add(symbol)
            
            self.config_coin_count[str(key)] += 1
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
                
                if str(key) in self.config_coin_count:
                    self.config_coin_count[str(key)] = max(0, self.config_coin_count[str(key)] - 1)

    def can_add_more_coins(self, strategy_name: str, config_key: Optional[str] = None) -> bool:
        with self._lock:
            key = (strategy_name, config_key)
            current_count = self.config_coin_count.get(str(key), 0)
            return current_count < 2

    def get_coin_count(self, strategy_name: str, config_key: Optional[str] = None) -> int:
        with self._lock:
            key = (strategy_name, config_key)
            return self.config_coin_count.get(str(key), 0)

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
                del self.symbol_cooldowns[symbol.upper()]
                return False
            return True

# ============================ WEBSOCKET MANAGER ==========================
class WebSocketManager:
    def __init__(self):
        self.symbol_callbacks: Dict[str, Any] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = True
        self.connections = set()

    def add_symbol(self, symbol: str, callback):
        self.symbol_callbacks[symbol] = callback
        if symbol not in self.threads:
            t = threading.Thread(target=self._run_symbol, args=(symbol,), daemon=True)
            t.start()
            self.threads[symbol] = t
        self.connections.add(symbol)

    def remove_symbol(self, symbol: str):
        if symbol in self.symbol_callbacks:
            del self.symbol_callbacks[symbol]
        if symbol in self.connections:
            self.connections.remove(symbol)

    def _run_symbol(self, symbol: str):
        last_price = 0.0
        while self.running and symbol in self.symbol_callbacks:
            p = get_current_price(symbol)
            if p and p != last_price:
                last_price = p
                try:
                    self.symbol_callbacks[symbol](p)
                except Exception as e:
                    logger.error(f"Callback error for {symbol}: {e}")
            time.sleep(1)

    def stop(self):
        self.running = False

# ============================ BASE BOT =====================
class BaseBot:
    def __init__(self, symbol: str, lev: int, percent: float, tp: Optional[float], sl: Optional[float],
                 ws_manager: 'WebSocketManager', api_key: str, api_secret: str,
                 telegram_bot_token: Optional[str], telegram_chat_id: Optional[str],
                 strategy_name: str, config_key: Optional[str] = None, dynamic_mode: bool = False):
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
        
        self.position_open = False
        self.side: Optional[str] = None
        self.qty: float = 0.0
        self.entry: float = 0.0
        self._close_attempted = False
        self.prices: list = []
        self.status = "waiting"
        
        self.coin_manager = CoinManager()
        self._candidate_pool: list = []
        self.last_close_time = 0.0
        
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)

    def log(self, msg: str):
        logger.info(f"[{self.strategy_name}][{self.symbol}] {msg}")
        try:
            send_telegram(f"[{self.strategy_name}][{self.symbol}] {msg}", 
                         self.telegram_chat_id, 
                         bot_token=self.telegram_bot_token)
        except Exception:
            pass

    def _handle_price_update(self, price: float):
        try:
            self.prices.append(float(price))
            if len(self.prices) > 1200:
                self.prices.pop(0)
                
            if not self.position_open:
                sig = self.get_signal()
                if sig:
                    self.open_position(sig)
            else:
                self.check_tp_sl()
                
        except Exception as e:
            self.log(f"Lỗi price update: {e}")

    def get_signal(self) -> Optional[str]:
        return None

    def open_position(self, side: str) -> bool:
        try:
            if self.position_open:
                self.log("⚠️ Đang có vị thế, không mở lệnh mới")
                return False
                
            positions = get_positions(self.api_key, self.api_secret)
            if positions:
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0 and pos.get('symbol') == self.symbol:
                        self.log(f"⚠️ Đang có vị thế {self.symbol} trên sàn, không mở lệnh mới")
                        return False
                
            price = get_current_price(self.symbol)
            if price <= 0:
                return False
                
            balance = get_balance(self.api_key, self.api_secret)
            if balance <= 0:
                self.log("❌ Số dư không đủ")
                return False
                
            invest = balance * (self.percent / 100)
            qty = (invest * self.lev) / price
            
            # Đơn giản hóa làm tròn
            qty = round(qty, 4)
            
            if qty <= 0:
                return False
                
            if not set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                self.log("❌ Không thể set đòn bẩy")
                return False
                
            res = place_order(self.symbol, side, qty, self.api_key, self.api_secret)
            if not res:
                return False
                
            self.position_open = True
            self.side = side
            self.qty = qty if side == 'BUY' else -qty
            self.entry = price
            self._close_attempted = False
            self.status = "open"
            
            self.log(f"🚀 Mở {side} qty={abs(qty):.4f} @ {price:.4f}")
            return True
            
        except Exception as e:
            self.log(f"❌ open_position lỗi: {e}")
            return False

    def close_position(self, reason: str = "") -> bool:
        if not self.position_open or self._close_attempted:
            return False
            
        try:
            self._close_attempted = True
            side = "SELL" if self.side == "BUY" else "BUY"
            qty_to_close = abs(self.qty)
            
            res = place_order(self.symbol, side, qty_to_close, self.api_key, self.api_secret)
            if res:
                self.log(f"✅ Đóng lệnh: {reason}")
                old_symbol = self.symbol
                self._reset_position()
                self.last_close_time = time.time()
                
                self.coin_manager.set_cooldown(old_symbol)
                
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
        self.status = "waiting"

    def check_tp_sl(self):
        if not self.position_open or self.entry <= 0:
            return
            
        current_price = get_current_price(self.symbol)
        if current_price <= 0:
            return

        if self._close_attempted:
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

    def _find_new_coin_after_exit(self):
        try:
            current_coin_count = self.coin_manager.get_coin_count(self.strategy_name, self.config_key)
            
            if current_coin_count >= 2:
                self.log(f"⚠️ Config đã có {current_coin_count}/2 coin, không tìm coin mới")
                return
                
            self.log("🔄 Tìm coin mới...")
            
            if self._candidate_pool and current_coin_count < 2:
                cached = self._candidate_pool.pop(0)
                if not self.coin_manager.is_in_cooldown(cached):
                    old_symbol = self.symbol
                    self.ws_manager.remove_symbol(old_symbol)
                    self.coin_manager.unregister_coin(old_symbol)
                    self.symbol = cached
                    
                    if self.coin_manager.can_add_more_coins(self.strategy_name, self.config_key):
                        if self.coin_manager.register_coin(cached, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key):
                            self._restart_websocket_for_new_coin()
                            self.log(f"🔁 Dùng ứng viên sẵn có: {old_symbol} → {cached}")
                            return

            if current_coin_count < 2:
                new_symbols = self._get_qualified_symbols()
                
                if new_symbols:
                    primary = new_symbols[0]
                    old_symbol = self.symbol
                    self.ws_manager.remove_symbol(old_symbol)
                    self.coin_manager.unregister_coin(old_symbol)
                    self.symbol = primary
                    
                    if self.coin_manager.can_add_more_coins(self.strategy_name, self.config_key):
                        if self.coin_manager.register_coin(primary, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key):
                            self._restart_websocket_for_new_coin()
                            msg = f"🔄 Chuyển {old_symbol} → {primary}"
                            
                            if len(new_symbols) > 1 and current_coin_count < 1:
                                self._candidate_pool = [new_symbols[1]]
                                msg += f" | Backup: {new_symbols[1]}"
                            self.log(msg)
                    else:
                        self.log("⚠️ Đã đủ 2 coin, không thêm coin mới")
                else:
                    self.log("❌ Không tìm thấy coin mới phù hợp")
            else:
                self.log(f"✅ Đã đủ {current_coin_count}/2 coin, không tìm coin mới")
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {e}")

    def _get_qualified_symbols(self) -> List[str]:
        """Tìm coin phù hợp cho bot động"""
        try:
            all_symbols = get_all_usdt_pairs(limit=30)
            qualified = []
            
            for symbol in all_symbols:
                if (symbol != self.symbol and 
                    not self.coin_manager.is_in_cooldown(symbol) and
                    self.coin_manager.can_add_more_coins(self.strategy_name, self.config_key)):
                    
                    # Kiểm tra volume
                    klines = get_klines(symbol, "5m", 50)
                    if klines:
                        closes = klines[3]
                        if len(closes) > 20:
                            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
                            if 2 <= volatility <= 10:
                                qualified.append(symbol)
                                
                                if len(qualified) >= 2:
                                    break
            
            return qualified[:2]
            
        except Exception as e:
            self.log(f"❌ Lỗi tìm symbols: {e}")
            return []

    def _restart_websocket_for_new_coin(self):
        try:
            time.sleep(1.5)
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
            self.log(f"🔗 Restart WS cho {self.symbol}")
        except Exception as e:
            self.log(f"❌ WS restart lỗi: {e}")

    def stop(self):
        self.ws_manager.remove_symbol(self.symbol)
        self.coin_manager.unregister_coin(self.symbol)
        self.log("⛔ Bot đã dừng")

# ============================ STRATEGIES ====================
class RSI_EMA_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                        telegram_bot_token, telegram_chat_id, "RSI/EMA Recursive")

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            rsi = calc_rsi(self.prices, 14)
            ema_fast = calc_ema(self.prices, 9)
            ema_slow = calc_ema(self.prices, 21)

            if rsi is None or ema_fast is None or ema_slow is None:
                return None

            if rsi < 30 and ema_fast > ema_slow:
                return "BUY"
            elif rsi > 70 and ema_fast < ema_slow:
                return "SELL"

            return None

        except Exception as e:
            return None

class EMA_Crossover_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                        telegram_bot_token, telegram_chat_id, "EMA Crossover")
        self.prev_ema_fast = None
        self.prev_ema_slow = None

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            ema_fast = calc_ema(self.prices, 9)
            ema_slow = calc_ema(self.prices, 21)

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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, threshold=30, config_key=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                        telegram_bot_token, telegram_chat_id, "Reverse 24h", config_key, dynamic_mode)
        self.threshold = threshold
        self.last_24h_check = 0

    def get_signal(self):
        try:
            current_time = time.time()
            if current_time - self.last_24h_check < 60:
                return None

            change_24h = get_24h_change(self.symbol)
            self.last_24h_check = current_time

            if change_24h is None:
                return None

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
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                        telegram_bot_token, telegram_chat_id, "Trend Following", config_key, dynamic_mode)

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            recent_prices = self.prices[-20:]
            if len(recent_prices) < 2:
                return None
                
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            if price_change > 0.001:
                return "BUY"
            elif price_change < -0.001:
                return "SELL"

            return None

        except Exception as e:
            return None

class Scalping_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                        telegram_bot_token, telegram_chat_id, "Scalping", config_key, dynamic_mode)

    def get_signal(self):
        try:
            if len(self.prices) < 20:
                return None

            rsi = calc_rsi(self.prices, 7)
            if rsi is None:
                return None

            if rsi < 25:
                return "BUY"
            elif rsi > 75:
                return "SELL"

            return None

        except Exception as e:
            return None

class Safe_Grid_Bot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, grid_levels=5, config_key=None, dynamic_mode=False):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                        telegram_bot_token, telegram_chat_id, "Safe Grid", config_key, dynamic_mode)
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

class SmartDynamicBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, dynamic_mode=True):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Smart Dynamic", config_key, dynamic_mode)

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            rsi = calc_rsi(self.prices, 14)
            ema_fast = calc_ema(self.prices, 9)
            ema_slow = calc_ema(self.prices, 21)

            if None in [rsi, ema_fast, ema_slow]:
                return None

            score = 0
            signal = None
            
            if rsi < 30 and ema_fast > ema_slow:
                score += 2
                signal = "BUY"
            elif rsi > 70 and ema_fast < ema_slow:
                score += 2
                signal = "SELL"
            
            if score >= 2:
                self.log(f"🎯 Smart Signal: {signal} | Score: {score}/2 | RSI: {rsi:.1f}")
                return signal
            
            return None

        except Exception as e:
            self.log(f"❌ Lỗi Smart Dynamic signal: {str(e)}")
            return None

# ============================ BOT MANAGER ==================
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
            self.log("🟢 HỆ THỐNG BOT FUTURES ĐÃ KHỞI ĐỘNG")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def _verify_api_connection(self):
        balance = get_balance(self.api_key, self.api_secret)
        if balance == 0:
            self.log("❌ LỖI: Không thể kết nối Binance API.")

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>SYSTEM</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        welcome = "🤖 <b>BOT GIAO DỊCH FUTURES</b>\n\n🎯 <b>HỆ THỐNG ĐA CHIẾN LƯỢC + BOT ĐỘNG TỰ TÌM COIN</b>"
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, **kwargs):
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance == 0:
            self.log("❌ LỖI: Không thể kết nối Binance")
            return False
        
        symbol = symbol.upper()
        bot_id = f"{symbol}_{strategy_type}"
        
        if bot_id in self.bots:
            self.log(f"⚠️ Đã có bot {strategy_type} cho {symbol}")
            return False
            
        try:
            bot_class = {
                "RSI/EMA Recursive": RSI_EMA_Bot,
                "EMA Crossover": EMA_Crossover_Bot,
                "Reverse 24h": Reverse_24h_Bot,
                "Trend Following": Trend_Following_Bot,
                "Scalping": Scalping_Bot,
                "Safe Grid": Safe_Grid_Bot,
                "Smart Dynamic": SmartDynamicBot
            }.get(strategy_type)
            
            if not bot_class:
                self.log(f"❌ Chiến lược {strategy_type} không được hỗ trợ")
                return False
            
            if strategy_type == "Reverse 24h":
                threshold = kwargs.get('threshold', 30)
                dynamic_mode = kwargs.get('dynamic_mode', False)
                config_key = kwargs.get('config_key')
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, threshold, config_key, dynamic_mode)
            elif strategy_type == "Safe Grid":
                grid_levels = kwargs.get('grid_levels', 5)
                dynamic_mode = kwargs.get('dynamic_mode', False)
                config_key = kwargs.get('config_key')
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels, config_key, dynamic_mode)
            elif strategy_type == "Smart Dynamic":
                dynamic_mode = kwargs.get('dynamic_mode', True)
                config_key = kwargs.get('config_key')
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, config_key, dynamic_mode)
            else:
                dynamic_mode = kwargs.get('dynamic_mode', False)
                config_key = kwargs.get('config_key')
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, config_key, dynamic_mode)
            
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
        
        # ========== XỬ LÝ THEO TRẠNG THÁI ==========
        
        # 1. Đang chờ chọn chế độ bot
        if current_step == 'waiting_bot_mode':
            if text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
                user_state['bot_mode'] = 'static' if "Tĩnh" in text else 'dynamic'
                user_state['step'] = 'waiting_strategy'
                self.user_states[chat_id] = user_state
                
                mode_text = "BOT TĨNH - Coin cố định" if user_state['bot_mode'] == 'static' else "BOT ĐỘNG - Tự tìm coin"
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {mode_text}</b>\n\nChọn chiến lược:",
                    chat_id,
                    create_strategy_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                send_telegram("Vui lòng chọn chế độ bot từ bàn phím:", chat_id,
                             create_bot_mode_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            return

        # 2. Đang chờ chọn chiến lược
        elif current_step == 'waiting_strategy':
            strategy_map = {
                "🤖 RSI/EMA Recursive": "RSI/EMA Recursive",
                "📊 EMA Crossover": "EMA Crossover", 
                "🎯 Reverse 24h": "Reverse 24h",
                "📈 Trend Following": "Trend Following",
                "⚡ Scalping": "Scalping",
                "🛡️ Safe Grid": "Safe Grid",
                "🔄 Bot Động Thông Minh": "Smart Dynamic"
            }
            
            if text in strategy_map:
                user_state['strategy'] = strategy_map[text]
                user_state['step'] = 'waiting_leverage'
                self.user_states[chat_id] = user_state
                
                send_telegram(f"🎯 Chiến lược: {user_state['strategy']}\n\nChọn đòn bẩy:", chat_id,
                             create_leverage_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            else:
                send_telegram("Vui lòng chọn chiến lược từ bàn phím:", chat_id,
                             create_strategy_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            return

        # 3. Đang chờ chọn đòn bẩy
        elif current_step == 'waiting_leverage':
            try:
                lev_text = text[:-1] if text.endswith('x') else text
                leverage = int(lev_text)
                if 1 <= leverage <= 100:
                    user_state['leverage'] = leverage
                    user_state['step'] = 'waiting_percent'
                    self.user_states[chat_id] = user_state
                    
                    send_telegram(f"💰 Đòn bẩy: {leverage}x\n\nChọn % số dư:", chat_id,
                                 create_percent_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ Đòn bẩy phải từ 1-100. Chọn lại:", chat_id,
                                 create_leverage_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng chọn đòn bẩy hợp lệ:", chat_id,
                             create_leverage_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            return

        # 4. Đang chờ chọn % số dư
        elif current_step == 'waiting_percent':
            try:
                percent = float(text)
                if 0 < percent <= 100:
                    user_state['percent'] = percent
                    user_state['step'] = 'waiting_tp'
                    self.user_states[chat_id] = user_state
                    
                    send_telegram(f"📊 % Số dư: {percent}%\n\nChọn Take Profit %:", chat_id,
                                 create_tp_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ % số dư phải từ 0.1-100. Chọn lại:", chat_id,
                                 create_percent_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng chọn % hợp lệ:", chat_id,
                             create_percent_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            return

        # 5. Đang chờ chọn Take Profit
        elif current_step == 'waiting_tp':
            try:
                tp = float(text)
                if tp > 0:
                    user_state['tp'] = tp
                    user_state['step'] = 'waiting_sl'
                    self.user_states[chat_id] = user_state
                    
                    send_telegram(f"🎯 Take Profit: {tp}%\n\nChọn Stop Loss %:", chat_id,
                                 create_sl_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ TP phải > 0. Chọn lại:", chat_id,
                                 create_tp_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng chọn TP hợp lệ:", chat_id,
                             create_tp_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            return

        # 6. Đang chờ chọn Stop Loss - BƯỚC CUỐI CÙNG
        elif current_step == 'waiting_sl':
            try:
                sl = float(text)
                if sl >= 0:
                    user_state['sl'] = sl
                    
                    # Lấy tất cả thông tin từ user_state
                    strategy = user_state.get('strategy')
                    bot_mode = user_state.get('bot_mode', 'static')
                    leverage = user_state.get('leverage')
                    percent = user_state.get('percent')
                    tp = user_state.get('tp')
                    sl = user_state.get('sl')
                    
                    # Thêm bot
                    success = self.add_bot(
                        symbol="BTCUSDT",  # Tạm thời dùng BTCUSDT
                        lev=leverage, 
                        percent=percent,
                        tp=tp, 
                        sl=sl, 
                        strategy_type=strategy,
                        bot_mode=bot_mode
                    )
                    
                    if success:
                        success_msg = (
                            f"✅ <b>ĐÃ TẠO BOT THÀNH CÔNG!</b>\n\n"
                            f"🎯 Chiến lược: {strategy}\n"
                            f"🤖 Chế độ: {'TĨNH' if bot_mode == 'static' else 'ĐỘNG'}\n"
                            f"💰 Đòn bẩy: {leverage}x\n"
                            f"📊 % Số dư: {percent}%\n"
                            f"🎯 TP: {tp}%\n"
                            f"🛡️ SL: {sl if sl else 0}%"
                        )
                        send_telegram(success_msg, chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    else:
                        send_telegram("❌ Lỗi tạo bot. Vui lòng thử lại.", chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    
                    # Reset state
                    self.user_states[chat_id] = {}
                    
                else:
                    send_telegram("⚠️ SL phải ≥ 0. Chọn lại:", chat_id,
                                 create_sl_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
            except ValueError:
                send_telegram("⚠️ Vui lòng chọn SL hợp lệ:", chat_id,
                             create_sl_keyboard(),
                             self.telegram_bot_token, self.telegram_chat_id)
            return

        # ========== XỬ LÝ LỆNH KHÔNG CẦN TRẠNG THÁI ==========
        
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_mode'}
            available_balance = get_balance(self.api_key, self.api_secret)
            if available_balance == 0:
                send_telegram("❌ LỖI KẾT NỐI BINANCE\nVui lòng kiểm tra API Key!", chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(f"💰 <b>Số dư khả dụng:</b> {available_balance:.2f} USDT\n\nChọn chế độ bot:", chat_id,
                         create_bot_mode_keyboard(),
                         self.telegram_bot_token, self.telegram_chat_id)
        
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                for bot_id, bot in self.bots.items():
                    status = "🟢 Mở" if bot.position_open else "🟡 Chờ"
                    message += f"🔹 {bot_id} | {status} | ĐB: {bot.lev}x\n"
                send_telegram(message, chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                keyboard = []
                for bot_id in self.bots.keys():
                    keyboard.append([{"text": f"⛔ {bot_id}"}])
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram("⛔ Chọn bot để dừng:", chat_id,
                             {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                             self.telegram_bot_token, self.telegram_chat_id)
        
        elif text.startswith("⛔ "):
            bot_id = text.replace("⛔ ", "").strip()
            if self.stop_bot(bot_id):
                send_telegram(f"⛔ Đã dừng bot {bot_id}", chat_id, create_main_menu(),
                             self.telegram_bot_token, self.telegram_chat_id)
            else:
                send_telegram(f"⚠️ Không tìm thấy bot {bot_id}", chat_id, create_main_menu(),
                             self.telegram_bot_token, self.telegram_chat_id)
        
        elif text == "💰 Số dư":
            detailed_balance = get_detailed_balance(self.api_key, self.api_secret)
            if not detailed_balance:
                send_telegram("❌ Lỗi kết nối Binance", chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = (
                    f"💰 <b>CHI TIẾT SỐ DƯ</b>\n\n"
                    f"🟢 <b>Khả dụng:</b> {detailed_balance['available']:.2f} USDT\n"
                    f"📊 <b>Tổng số dư:</b> {detailed_balance['total']:.2f} USDT\n"
                    f"📈 <b>Lợi nhuận chưa thực hiện:</b> {detailed_balance['unrealized_pnl']:.2f} USDT"
                )
                send_telegram(message, chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "📈 Vị thế":
            positions = get_positions(self.api_key, self.api_secret)
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
                        message += f"🔹 {symbol} | {side} | PnL: {pnl:.2f} USDT\n"
                send_telegram(message, chat_id,
                             bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text in ["🎯 Chiến lược", "⚙️ Cấu hình"]:
            info = (
                "🎯 <b>HỆ THỐNG BOT FUTURES</b>\n\n"
                "🤖 <b>Bot Tĩnh:</b> Coin cố định\n"
                "🔄 <b>Bot Động:</b> Tự tìm coin\n"
                "🎯 <b>Chiến lược:</b> RSI/EMA, Crossover, Reverse 24h, Trend, Scalping, Grid, Smart Dynamic\n"
                f"📊 <b>Đang chạy:</b> {len(self.bots)} bot"
            )
            send_telegram(info, chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        else:
            # Nếu không nhận diện được lệnh, hiển thị menu chính
            self.send_main_menu(chat_id)

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()
