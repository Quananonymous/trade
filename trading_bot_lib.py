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
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================== LOGGING SETUP =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_optimized.log')
    ]
)
logger = logging.getLogger()

# ============================= BINANCE CONFIG ==========================
BASE_FAPI = "https://fapi.binance.com"
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3

# ============================ OPTIMIZED BINANCE API =====================
def binance_request_optimized(url: str, params: dict = None, method: str = 'GET', 
                             api_key: str = None, secret_key: str = None, retry_count: int = 3):
    """Tối ưu hóa request với retry mechanism và timeout"""
    for attempt in range(retry_count):
        try:
            if method == 'GET':
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            else:
                # Xử lý signed request
                query_string = urllib.parse.urlencode(params)
                signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
                full_url = f"{url}?{query_string}&signature={signature}"
                headers = {"X-MBX-APIKEY": api_key}
                response = requests.post(full_url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.warning(f"API {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"Request error (attempt {attempt + 1}): {e}")
        
        if attempt < retry_count - 1:
            time.sleep(1)
    
    return None

def get_balance_fast(api_key: str, api_secret: str) -> float:
    """Lấy số dư nhanh với cache"""
    try:
        url = f"{BASE_FAPI}/fapi/v2/balance"
        params = {"timestamp": int(time.time() * 1000)}
        data = binance_request_optimized(url, params, 'GET', api_key, api_secret)
        
        if data:
            for asset in data:
                if asset.get('asset') == 'USDT':
                    balance = float(asset.get('balance', 0))
                    logger.info(f"💰 Số dư: {balance:.2f} USDT")
                    return balance
        return 0.0
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {e}")
        return 0.0

def get_current_price_fast(symbol: str) -> float:
    """Lấy giá hiện tại nhanh"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/ticker/price"
        data = binance_request_optimized(url, {"symbol": symbol.upper()})
        return float(data['price']) if data and 'price' in data else 0.0
    except Exception:
        return 0.0

def get_klines_fast(symbol: str, interval: str = "5m", limit: int = 100) -> Optional[Tuple[list, list, list, list, list]]:
    """Lấy dữ liệu kline nhanh"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        data = binance_request_optimized(url, params)
        
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

def set_leverage_fast(symbol: str, leverage: int, api_key: str, api_secret: str) -> bool:
    """Set đòn bẩy nhanh"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/leverage"
        params = {
            "symbol": symbol.upper(),
            "leverage": leverage,
            "timestamp": int(time.time() * 1000)
        }
        result = binance_request_optimized(url, params, 'POST', api_key, api_secret)
        return result is not None
    except Exception as e:
        logger.error(f"Lỗi set leverage {symbol}: {e}")
        return False

def place_order_fast(symbol: str, side: str, quantity: float, api_key: str, api_secret: str) -> bool:
    """Đặt lệnh nhanh với xử lý lỗi"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/order"
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "timestamp": int(time.time() * 1000)
        }
        result = binance_request_optimized(url, params, 'POST', api_key, api_secret)
        return result is not None
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh {symbol}: {e}")
        return False

def get_positions_fast(api_key: str, api_secret: str):
    """Lấy vị thế nhanh"""
    try:
        url = f"{BASE_FAPI}/fapi/v2/positionRisk"
        params = {"timestamp": int(time.time() * 1000)}
        return binance_request_optimized(url, params, 'GET', api_key, api_secret)
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {e}")
        return None

def get_all_usdt_pairs_fast(limit: int = 50) -> List[str]:
    """Lấy danh sách cặp USDT nhanh"""
    try:
        url = f"{BASE_FAPI}/fapi/v1/exchangeInfo"
        data = binance_request_optimized(url)
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

# ============================ OPTIMIZED INDICATORS =====================
def calc_rsi_fast(prices: list, period: int = 14) -> Optional[float]:
    """RSI tính nhanh"""
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

def calc_ema_fast(prices: list, period: int) -> Optional[float]:
    """EMA tính nhanh"""
    if len(prices) < period:
        return None
        
    k = 2.0 / (period + 1.0)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
        
    return float(ema)

def calc_atr_fast(highs: list, lows: list, closes: list, period: int = 14) -> Optional[float]:
    """ATR tính nhanh"""
    if len(closes) < period + 1:
        return None
        
    tr_values = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr = max(tr1, tr2, tr3)
        tr_values.append(tr)
        
    atr = np.mean(tr_values[-period:])
    return float(atr / closes[-1] * 100) if closes[-1] > 0 else float(atr)

# ============================ TELEGRAM MANAGER =======================
class TelegramManager:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        
    def send_message(self, message: str, reply_markup=None):
        """Gửi tin nhắn Telegram nhanh"""
        if not self.bot_token or not self.chat_id:
            return
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
            
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass

    def create_keyboard(self, buttons):
        """Tạo keyboard nhanh"""
        return {
            "keyboard": buttons,
            "resize_keyboard": True,
            "one_time_keyboard": False
        }

# ============================ COIN MANAGER OPTIMIZED ==================
class CoinManager:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.symbol_to_bot = {}
            cls._instance.symbol_cooldowns = {}
            cls._instance.config_counts = {}
            cls._instance.lock = threading.Lock()
        return cls._instance

    def register_coin(self, symbol: str, bot_id: str, strategy: str, config_key: str = None) -> bool:
        """Đăng ký coin mới"""
        with self.lock:
            if symbol in self.symbol_to_bot:
                return False
                
            key = f"{strategy}_{config_key}" if config_key else strategy
            current_count = self.config_counts.get(key, 0)
            
            if current_count >= 2:  # Giới hạn 2 coin mỗi config
                return False
                
            self.symbol_to_bot[symbol] = bot_id
            self.config_counts[key] = current_count + 1
            return True

    def unregister_coin(self, symbol: str, strategy: str, config_key: str = None):
        """Hủy đăng ký coin"""
        with self.lock:
            if symbol in self.symbol_to_bot:
                del self.symbol_to_bot[symbol]
                
            key = f"{strategy}_{config_key}" if config_key else strategy
            if key in self.config_counts:
                self.config_counts[key] = max(0, self.config_counts[key] - 1)

    def can_add_coin(self, strategy: str, config_key: str = None) -> bool:
        """Kiểm tra có thể thêm coin mới không"""
        key = f"{strategy}_{config_key}" if config_key else strategy
        return self.config_counts.get(key, 0) < 2

    def set_cooldown(self, symbol: str, seconds: int = 600):
        """Set thời gian chờ cho coin"""
        with self.lock:
            self.symbol_cooldowns[symbol] = time.time() + seconds

    def is_in_cooldown(self, symbol: str) -> bool:
        """Kiểm tra coin có đang trong thời gian chờ không"""
        with self.lock:
            cooldown_time = self.symbol_cooldowns.get(symbol)
            if not cooldown_time:
                return False
                
            if time.time() >= cooldown_time:
                del self.symbol_cooldowns[symbol]
                return False
                
            return True

# ============================ PRICE MANAGER ==========================
class PriceManager:
    def __init__(self):
        self.price_cache = {}
        self.callbacks = {}
        self.running = True
        self.thread = threading.Thread(target=self._update_prices, daemon=True)
        self.thread.start()

    def add_symbol(self, symbol: str, callback):
        """Thêm symbol để theo dõi giá"""
        self.callbacks[symbol] = callback

    def remove_symbol(self, symbol: str):
        """Xóa symbol khỏi theo dõi"""
        if symbol in self.callbacks:
            del self.callbacks[symbol]

    def _update_prices(self):
        """Cập nhật giá liên tục"""
        while self.running:
            try:
                symbols = list(self.callbacks.keys())
                if not symbols:
                    time.sleep(1)
                    continue

                # Lấy giá cho tất cả symbols cùng lúc
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_symbol = {
                        executor.submit(get_current_price_fast, symbol): symbol 
                        for symbol in symbols
                    }
                    
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            price = future.result()
                            if price and price > 0:
                                if symbol in self.callbacks:
                                    self.callbacks[symbol](price)
                        except Exception as e:
                            logger.error(f"Lỗi cập nhật giá {symbol}: {e}")
                
                time.sleep(2)  # Giảm tần suất cập nhật
                
            except Exception as e:
                logger.error(f"Lỗi PriceManager: {e}")
                time.sleep(5)

    def stop(self):
        """Dừng PriceManager"""
        self.running = False

# ============================ BASE BOT OPTIMIZED =====================
class BaseBot:
    def __init__(self, symbol: str, leverage: int, percent: float, 
                 take_profit: float, stop_loss: float, api_key: str, 
                 api_secret: str, telegram_manager: TelegramManager,
                 price_manager: PriceManager, strategy_name: str,
                 dynamic_mode: bool = False, config_key: str = None):
        
        self.symbol = symbol.upper()
        self.leverage = leverage
        self.percent = percent
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram = telegram_manager
        self.price_manager = price_manager
        self.strategy_name = strategy_name
        self.dynamic_mode = dynamic_mode
        self.config_key = config_key
        
        self.position_open = False
        self.position_side = None
        self.quantity = 0.0
        self.entry_price = 0.0
        self.is_closing = False
        self.price_history = []
        
        self.coin_manager = CoinManager()
        self.candidate_pool = []
        
        # Đăng ký theo dõi giá
        self.price_manager.add_symbol(self.symbol, self._on_price_update)
        
        # Đăng ký coin
        self.coin_manager.register_coin(self.symbol, f"{strategy_name}_{id(self)}", 
                                      strategy_name, config_key)
        
        self.log(f"✅ Bot khởi động | ĐB: {leverage}x | Vốn: {percent}% | TP/SL: {take_profit}%/{stop_loss}%")

    def log(self, message: str):
        """Ghi log và gửi Telegram"""
        full_message = f"[{self.strategy_name}][{self.symbol}] {message}"
        logger.info(full_message)
        self.telegram.send_message(full_message)

    def _on_price_update(self, price: float):
        """Xử lý cập nhật giá - được tối ưu để giảm delay"""
        try:
            if len(self.price_history) > 500:  # Giới hạn lịch sử giá
                self.price_history.pop(0)
            self.price_history.append(price)
            
            # Chỉ kiểm tra tín hiệu nếu không có vị thế
            if not self.position_open and not self.is_closing:
                signal = self._get_signal()
                if signal:
                    self._open_position(signal)
            elif self.position_open and not self.is_closing:
                self._check_exit_conditions()
                
        except Exception as e:
            self.log(f"❌ Lỗi xử lý giá: {e}")

    def _get_signal(self) -> Optional[str]:
        """Lấy tín hiệu giao dịch - override bởi strategy"""
        return None

    def _open_position(self, side: str) -> bool:
        """Mở vị thế - được tối ưu để giảm delay"""
        if self.position_open or self.is_closing:
            return False
            
        try:
            # Kiểm tra vị thế thực tế trên sàn
            positions = get_positions_fast(self.api_key, self.api_secret)
            if positions:
                for pos in positions:
                    if (pos.get('symbol') == self.symbol and 
                        float(pos.get('positionAmt', 0)) != 0):
                        self.log("⚠️ Đã có vị thế trên sàn")
                        return False

            current_price = get_current_price_fast(self.symbol)
            if current_price <= 0:
                return False

            balance = get_balance_fast(self.api_key, self.api_secret)
            if balance <= 0:
                self.log("❌ Số dư không đủ")
                return False

            # Tính toán khối lượng
            investment = balance * (self.percent / 100)
            quantity = (investment * self.leverage) / current_price
            
            # Làm tròn khối lượng
            quantity = round(quantity, 4)  # Đơn giản hóa làm tròn
            
            if quantity <= 0:
                return False

            # Set đòn bẩy và đặt lệnh
            if not set_leverage_fast(self.symbol, self.leverage, self.api_key, self.api_secret):
                self.log("❌ Lỗi set đòn bẩy")
                return False

            if place_order_fast(self.symbol, side, quantity, self.api_key, self.api_secret):
                self.position_open = True
                self.position_side = side
                self.quantity = quantity if side == 'BUY' else -quantity
                self.entry_price = current_price
                self.is_closing = False
                
                self.log(f"🚀 Mở {side} | KL: {quantity:.4f} | Giá: {current_price:.4f}")
                return True
                
        except Exception as e:
            self.log(f"❌ Lỗi mở vị thế: {e}")
            
        return False

    def _check_exit_conditions(self):
        """Kiểm tra điều kiện thoát lệnh"""
        if not self.position_open or self.is_closing:
            return
            
        try:
            current_price = get_current_price_fast(self.symbol)
            if current_price <= 0:
                return

            if self.position_side == "BUY":
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100 * self.leverage
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price * 100 * self.leverage

            # Kiểm tra take profit
            if self.take_profit and profit_pct >= self.take_profit:
                self._close_position(f"✅ Đạt TP {self.take_profit}% (Lợi nhuận: {profit_pct:.2f}%)")
            # Kiểm tra stop loss
            elif self.stop_loss and profit_pct <= -self.stop_loss:
                self._close_position(f"❌ Đạt SL {self.stop_loss}% (Lỗ: {profit_pct:.2f}%)")
                
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra exit: {e}")

    def _close_position(self, reason: str = "") -> bool:
        """Đóng vị thế"""
        if self.is_closing or not self.position_open:
            return False
            
        try:
            self.is_closing = True
            close_side = "SELL" if self.position_side == "BUY" else "BUY"
            close_quantity = abs(self.quantity)
            
            if place_order_fast(self.symbol, close_side, close_quantity, self.api_key, self.api_secret):
                self.log(f"Đóng lệnh: {reason}")
                
                # Lưu symbol cũ để chuyển coin
                old_symbol = self.symbol
                
                # Reset trạng thái
                self._reset_position()
                
                # Set cooldown cho coin cũ
                self.coin_manager.set_cooldown(old_symbol)
                
                # Tìm coin mới nếu là bot động
                if self.dynamic_mode:
                    self._find_new_coin()
                    
                return True
                
        except Exception as e:
            self.log(f"❌ Lỗi đóng vị thế: {e}")
            self.is_closing = False
            
        return False

    def _reset_position(self):
        """Reset trạng thái vị thế"""
        self.position_open = False
        self.position_side = None
        self.quantity = 0.0
        self.entry_price = 0.0
        self.is_closing = False

    def _find_new_coin(self):
        """Tìm coin mới cho bot động"""
        try:
            # Kiểm tra số coin hiện tại của config
            if not self.coin_manager.can_add_coin(self.strategy_name, self.config_key):
                self.log("✅ Đã đủ 2 coin, không tìm thêm")
                return
                
            self.log("🔄 Đang tìm coin mới...")
            
            # Tìm coin mới
            new_symbols = self._scan_qualified_symbols()
            
            if new_symbols:
                new_symbol = new_symbols[0]
                old_symbol = self.symbol
                
                # Cập nhật symbol
                self.price_manager.remove_symbol(old_symbol)
                self.coin_manager.unregister_coin(old_symbol, self.strategy_name, self.config_key)
                
                self.symbol = new_symbol
                self.price_manager.add_symbol(self.symbol, self._on_price_update)
                self.coin_manager.register_coin(new_symbol, f"{self.strategy_name}_{id(self)}", 
                                              self.strategy_name, self.config_key)
                
                self.log(f"🔄 Chuyển {old_symbol} → {new_symbol}")
                
                # Lưu candidate backup
                if len(new_symbols) > 1:
                    self.candidate_pool = [new_symbols[1]]
            else:
                self.log("❌ Không tìm thấy coin mới phù hợp")
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {e}")

    def _scan_qualified_symbols(self) -> List[str]:
        """Quét tìm coin phù hợp"""
        try:
            all_symbols = get_all_usdt_pairs_fast(30)
            qualified = []
            
            for symbol in all_symbols:
                if (symbol != self.symbol and 
                    not self.coin_manager.is_in_cooldown(symbol) and
                    self.coin_manager.can_add_coin(self.strategy_name, self.config_key)):
                    
                    # Kiểm tra volume và điều kiện cơ bản
                    klines = get_klines_fast(symbol, "5m", 50)
                    if klines:
                        closes = klines[3]
                        if len(closes) > 20:
                            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) * 100
                            if 2 <= volatility <= 10:  # Điều kiện biến động
                                qualified.append(symbol)
                                
                                if len(qualified) >= 2:  # Chỉ lấy 2 coin
                                    break
            
            return qualified[:2]
            
        except Exception as e:
            self.log(f"❌ Lỗi scan symbols: {e}")
            return []

    def stop(self):
        """Dừng bot"""
        self.price_manager.remove_symbol(self.symbol)
        self.coin_manager.unregister_coin(self.symbol, self.strategy_name, self.config_key)
        self.log("⛔ Bot đã dừng")

# ============================ STRATEGIES OPTIMIZED ====================
class RSI_EMA_Bot(BaseBot):
    def __init__(self, symbol, leverage, percent, tp, sl, api_key, api_secret, 
                 telegram_manager, price_manager, dynamic_mode=False, config_key=None):
        super().__init__(symbol, leverage, percent, tp, sl, api_key, api_secret,
                        telegram_manager, price_manager, "RSI/EMA", dynamic_mode, config_key)

    def _get_signal(self):
        if len(self.price_history) < 30:
            return None
            
        rsi = calc_rsi_fast(self.price_history[-30:], 14)
        ema_fast = calc_ema_fast(self.price_history, 9)
        ema_slow = calc_ema_fast(self.price_history, 21)
        
        if rsi is None or ema_fast is None or ema_slow is None:
            return None
            
        if rsi < 30 and ema_fast > ema_slow:
            return "BUY"
        elif rsi > 70 and ema_fast < ema_slow:
            return "SELL"
            
        return None

class Reverse_24h_Bot(BaseBot):
    def __init__(self, symbol, leverage, percent, tp, sl, api_key, api_secret,
                 telegram_manager, price_manager, threshold=30, dynamic_mode=False, config_key=None):
        super().__init__(symbol, leverage, percent, tp, sl, api_key, api_secret,
                        telegram_manager, price_manager, "Reverse24h", dynamic_mode, config_key)
        self.threshold = threshold
        self.last_check = 0

    def _get_signal(self):
        if time.time() - self.last_check < 300:  # 5 phút check 1 lần
            return None
            
        try:
            url = f"{BASE_FAPI}/fapi/v1/ticker/24hr"
            data = binance_request_optimized(url, {"symbol": self.symbol})
            
            if data and 'priceChangePercent' in data:
                change = float(data['priceChangePercent'])
                self.last_check = time.time()
                
                if abs(change) >= self.threshold:
                    if change > 0:
                        self.log(f"🎯 Tín hiệu SELL - Biến động 24h: +{change:.2f}%")
                        return "SELL"
                    else:
                        self.log(f"🎯 Tín hiệu BUY - Biến động 24h: {change:.2f}%")
                        return "BUY"
                        
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra biến động: {e}")
            
        return None

class Trend_Following_Bot(BaseBot):
    def __init__(self, symbol, leverage, percent, tp, sl, api_key, api_secret,
                 telegram_manager, price_manager, dynamic_mode=False, config_key=None):
        super().__init__(symbol, leverage, percent, tp, sl, api_key, api_secret,
                        telegram_manager, price_manager, "Trend", dynamic_mode, config_key)

    def _get_signal(self):
        if len(self.price_history) < 50:
            return None
            
        # Phân tích trend đơn giản
        short_ma = np.mean(self.price_history[-10:])
        long_ma = np.mean(self.price_history[-30:])
        current_price = self.price_history[-1]
        
        if current_price > short_ma > long_ma:
            return "BUY"
        elif current_price < short_ma < long_ma:
            return "SELL"
            
        return None

class SmartDynamicBot(BaseBot):
    def __init__(self, symbol, leverage, percent, tp, sl, api_key, api_secret,
                 telegram_manager, price_manager, dynamic_mode=True, config_key=None):
        super().__init__(symbol, leverage, percent, tp, sl, api_key, api_secret,
                        telegram_manager, price_manager, "SmartDynamic", dynamic_mode, config_key)

    def _get_signal(self):
        if len(self.price_history) < 50:
            return None
            
        # Kết hợp nhiều indicator
        rsi = calc_rsi_fast(self.price_history, 14)
        ema_fast = calc_ema_fast(self.price_history, 9)
        ema_slow = calc_ema_fast(self.price_history, 21)
        
        if None in [rsi, ema_fast, ema_slow]:
            return None
            
        # Điểm số tín hiệu
        score = 0
        
        if rsi < 35 and ema_fast > ema_slow:
            score += 2
        elif rsi > 65 and ema_fast < ema_slow:
            score += 2
            
        # Thêm các điều kiện khác
        volatility = np.std(self.price_history[-20:]) / np.mean(self.price_history[-20:]) * 100
        if 3 <= volatility <= 8:
            score += 1
            
        if score >= 2:
            if rsi < 35:
                self.log(f"🎯 Smart BUY | RSI: {rsi:.1f} | Vol: {volatility:.1f}%")
                return "BUY"
            else:
                self.log(f"🎯 Smart SELL | RSI: {rsi:.1f} | Vol: {volatility:.1f}%")
                return "SELL"
                
        return None

# ============================ BOT MANAGER OPTIMIZED ==================
class BotManager:
    def __init__(self, api_key: str, api_secret: str, telegram_bot_token: str = None, telegram_chat_id: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Khởi tạo managers
        self.telegram_manager = TelegramManager(telegram_bot_token, telegram_chat_id)
        self.price_manager = PriceManager()
        self.coin_manager = CoinManager()
        
        self.bots = {}
        self.running = True
        
        # Kiểm tra kết nối
        self._verify_connection()
        
        # Khởi chạy background tasks
        self._start_background_tasks()
        
        self.telegram_manager.send_message("🤖 HỆ THỐNG BOT ĐÃ KHỞI ĐỘNG")

    def _verify_connection(self):
        """Kiểm tra kết nối API"""
        balance = get_balance_fast(self.api_key, self.api_secret)
        if balance > 0:
            self.telegram_manager.send_message(f"✅ Kết nối thành công | Số dư: {balance:.2f} USDT")
        else:
            self.telegram_manager.send_message("❌ Lỗi kết nối Binance API")

    def _start_background_tasks(self):
        """Khởi chạy các task nền"""
        def health_check():
            while self.running:
                try:
                    active_bots = len([b for b in self.bots.values() if b.position_open])
                    total_bots = len(self.bots)
                    
                    if total_bots > 0:
                        status_msg = f"📊 Bot Status: {active_bots}/{total_bots} đang mở vị thế"
                        logger.info(status_msg)
                        
                except Exception as e:
                    logger.error(f"Lỗi health check: {e}")
                    
                time.sleep(60)  # Check mỗi phút
                
        threading.Thread(target=health_check, daemon=True).start()

    def add_bot(self, symbol: str, leverage: int, percent: float, 
                take_profit: float, stop_loss: float, strategy: str,
                dynamic_mode: bool = False, **kwargs) -> bool:
        """Thêm bot mới"""
        try:
            if stop_loss == 0:
                stop_loss = None
                
            # Kiểm tra số dư
            balance = get_balance_fast(self.api_key, self.api_secret)
            if balance <= 0:
                self.telegram_manager.send_message("❌ Số dư không đủ")
                return False
                
            # Chọn class bot
            bot_classes = {
                "RSI/EMA": RSI_EMA_Bot,
                "Reverse24h": Reverse_24h_Bot,
                "Trend": Trend_Following_Bot,
                "SmartDynamic": SmartDynamicBot
            }
            
            bot_class = bot_classes.get(strategy)
            if not bot_class:
                self.telegram_manager.send_message(f"❌ Chiến lược {strategy} không được hỗ trợ")
                return False
                
            # Tạo config key cho bot động
            config_key = None
            if dynamic_mode:
                config_key = f"{strategy}_{leverage}_{percent}_{take_profit}_{stop_loss}"
                if not self.coin_manager.can_add_coin(strategy, config_key):
                    self.telegram_manager.send_message(f"✅ Config đã đủ 2 coin, không thêm mới")
                    return True
                    
            # Tạo bot
            if strategy == "Reverse24h":
                threshold = kwargs.get('threshold', 30)
                bot = bot_class(symbol, leverage, percent, take_profit, stop_loss,
                              self.api_key, self.api_secret, self.telegram_manager,
                              self.price_manager, threshold, dynamic_mode, config_key)
            else:
                bot = bot_class(symbol, leverage, percent, take_profit, stop_loss,
                              self.api_key, self.api_secret, self.telegram_manager,
                              self.price_manager, dynamic_mode, config_key)
            
            bot_id = f"{symbol}_{strategy}"
            self.bots[bot_id] = bot
            
            success_msg = (
                f"✅ ĐÃ TẠO BOT THÀNH CÔNG\n\n"
                f"🔗 Coin: {symbol}\n"
                f"🎯 Chiến lược: {strategy}\n"
                f"💰 Đòn bẩy: {leverage}x\n"
                f"📊 % Vốn: {percent}%\n"
                f"🎯 TP: {take_profit}%\n"
                f"🛡️ SL: {stop_loss if stop_loss else 'None'}%\n"
                f"🤖 Chế độ: {'ĐỘNG' if dynamic_mode else 'TĨNH'}"
            )
            
            self.telegram_manager.send_message(success_msg)
            return True
            
        except Exception as e:
            error_msg = f"❌ Lỗi tạo bot: {str(e)}"
            self.telegram_manager.send_message(error_msg)
            return False

    def stop_bot(self, bot_id: str) -> bool:
        """Dừng bot"""
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]
            self.telegram_manager.send_message(f"⛔ Đã dừng bot {bot_id}")
            return True
        return False

    def stop_all(self):
        """Dừng tất cả bot"""
        self.telegram_manager.send_message("⛔ Đang dừng tất cả bot...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.price_manager.stop()
        self.running = False
        self.telegram_manager.send_message("🔴 Hệ thống đã dừng")

    def get_status(self):
        """Lấy trạng thái hệ thống"""
        active_bots = len([b for b in self.bots.values() if b.position_open])
        total_bots = len(self.bots)
        
        status_msg = (
            f"📊 TRẠNG THÁI HỆ THỐNG\n\n"
            f"🤖 Tổng bot: {total_bots}\n"
            f"🟢 Bot active: {active_bots}\n"
            f"💰 Số dư: {get_balance_fast(self.api_key, self.api_secret):.2f} USDT"
        )
        
        self.telegram_manager.send_message(status_msg)

# ============================ USAGE EXAMPLE ==========================
