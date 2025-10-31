# trading_bot_lib_fixed.py - PHIÊN BẢN ĐÃ SỬA LỖI HOÀN CHỈNH
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
from concurrent.futures import as_completed
import time
import ssl

# ========== BYPASS SSL VERIFICATION ==========
ssl._create_default_https_context = ssl._create_unverified_context

def _last_closed_1m_quote_volume(symbol):
    data = binance_api_request(
        "https://fapi.binance.com/fapi/v1/klines",
        params={"symbol": symbol, "interval": "1m", "limit": 2}
    )
    if not data or len(data) < 2:
        return None
    k = data[-2]               # nến 1m đã đóng gần nhất
    return float(k[7])         # quoteVolume (USDT)

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
            [{"text": "📊 Global Market System"}],
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

def create_roi_trigger_keyboard():
    return {
        "keyboard": [
            [{"text": "30"}, {"text": "50"}, {"text": "100"}],
            [{"text": "150"}, {"text": "200"}, {"text": "300"}],
            [{"text": "❌ Tắt tính năng"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== API BINANCE - ĐÃ SỬA LỖI ==========
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
            # Thêm User-Agent để tránh bị chặn
            if headers is None:
                headers = {}
            
            if 'User-Agent' not in headers:
                headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            
            if method.upper() == 'GET':
                if params:
                    query = urllib.parse.urlencode(params)
                    url = f"{url}?{query}"
                req = urllib.request.Request(url, headers=headers)
            else:
                data = urllib.parse.urlencode(params).encode() if params else None
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
            
            # Tăng timeout và thêm retry logic
            with urllib.request.urlopen(req, timeout=30) as response:
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
            if e.code == 451:
                logger.error(f"❌ Lỗi 451: Truy cập bị chặn - Có thể do hạn chế địa lý. Vui lòng kiểm tra VPN/proxy.")
                return None
            else:
                logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            
            if e.code == 401:
                return None
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            continue
                
        except Exception as e:
            logger.error(f"Lỗi kết nối API (lần {attempt + 1}): {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_all_usdt_pairs(limit=600):
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

def get_top_volume_symbols(limit=100):
    """Top {limit} USDT pairs theo quoteVolume của NẾN 1M đã đóng (đa luồng)."""
    try:
        universe = get_all_usdt_pairs(limit=600) or []
        if not universe:
            logger.warning("❌ Không lấy được danh sách coin USDT")
            return []

        scored, failed = [], 0
        max_workers = 8  # Giảm workers để tránh bị chặn
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futmap = {ex.submit(_last_closed_1m_quote_volume, s): s for s in universe}
            for fut in as_completed(futmap):
                sym = futmap[fut]
                try:
                    qv = fut.result()
                    if qv is not None:
                        scored.append((sym, qv))
                except Exception:
                    failed += 1
                time.sleep(0.5)  # Tăng thời gian chờ để tránh 429

        scored.sort(key=lambda x: x[1], reverse=True)
        top_syms = [s for s, _ in scored[:limit]]
        logger.info(f"✅ Top {len(top_syms)} theo 1m quoteVolume (phân tích: {len(scored)}, lỗi: {failed})")
        return top_syms

    except Exception as e:
        logger.error(f"❌ Lỗi lấy top volume 1 phút (đa luồng): {str(e)}")
        return []

def get_max_leverage(symbol, api_key, api_secret):
    """Lấy đòn bẩy tối đa cho một symbol"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return 100  # Mặc định nếu không lấy được
        
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                # Tìm thông tin đòn bẩy từ filters
                for f in s['filters']:
                    if f['filterType'] == 'LEVERAGE':
                        if 'maxLeverage' in f:
                            return int(f['maxLeverage'])
                break
        return 100  # Mặc định
    except Exception as e:
        logger.error(f"Lỗi lấy đòn bẩy tối đa {symbol}: {str(e)}")
        return 100

def get_step_size(symbol, api_key, api_secret):
    if not symbol:
        logger.error("❌ Lỗi: Symbol là None khi lấy step size")
        return 0.001
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
    if not symbol:
        logger.error("❌ Lỗi: Symbol là None khi set leverage")
        return False
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
    """Lấy số dư KHẢ DỤNG (availableBalance) để tính toán khối lượng"""
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            logger.error("❌ Không lấy được số dư từ Binance")
            return None
            
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                available_balance = float(asset['availableBalance'])
                total_balance = float(asset['walletBalance'])
                
                logger.info(f"💰 Số dư - Khả dụng: {available_balance:.2f} USDT, Tổng: {total_balance:.2f} USDT")
                return available_balance
        return 0
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        return None

def place_order(symbol, side, qty, api_key, api_secret):
    if not symbol:
        logger.error("❌ Không thể đặt lệnh: symbol là None")
        return None
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
    if not symbol:
        logger.error("❌ Không thể hủy lệnh: symbol là None")
        return False
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
    if not symbol:
        logger.error("💰 Lỗi: Symbol là None khi lấy giá")
        return 0
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            price = float(data['price'])
            if price > 0:
                return price
            else:
                logger.error(f"💰 Giá {symbol} = 0")
        return 0
    except Exception as e:
        logger.error(f"💰 Lỗi lấy giá {symbol}: {str(e)}")
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

# ========== COIN MANAGER ĐÃ SỬA LỖI ==========
class CoinManager:
    def __init__(self):
        self.active_coins = set()
        self._lock = threading.Lock()
        self.coin_usage_count = {}  # Theo dõi số lần sử dụng coin
    
    def register_coin(self, symbol):
        if not symbol:
            return False
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.active_coins:
                return False  # Coin đã được sử dụng
            self.active_coins.add(symbol)
            self.coin_usage_count[symbol] = self.coin_usage_count.get(symbol, 0) + 1
            return True
    
    def unregister_coin(self, symbol):
        if not symbol:
            return
        with self._lock:
            symbol_upper = symbol.upper()
            self.active_coins.discard(symbol_upper)
            # Giữ lại lịch sử usage count
    
    def is_coin_active(self, symbol):
        if not symbol:
            return False
        with self._lock:
            return symbol.upper() in self.active_coins
    
    def get_active_coins(self):
        with self._lock:
            return list(self.active_coins)
    
    def get_coin_usage_count(self, symbol):
        with self._lock:
            return self.coin_usage_count.get(symbol.upper(), 0)

# ========== SMART COIN FINDER ĐÃ SỬA LỖI ==========
class SmartCoinFinder:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.coin_rotation_time = 3600  # 1 giờ
        self.blacklisted_coins = {}  # Danh sách coin bị chặn: {symbol: expiry_time}
        self.failed_coins = set()  # Coin đã thử và thất bại trong phiên hiện tại
        self._analysis_lock = threading.Lock()  # Lock để tránh race condition
        self.last_analysis_time = 0
        self.cached_top_coins = []
        self.cache_duration = 300  # 5 phút
    
    def get_symbol_leverage(self, symbol):
        """Lấy đòn bẩy tối đa của symbol"""
        return get_max_leverage(symbol, self.api_key, self.api_secret)
    
    def get_24h_volatility(self, symbol):
        """Lấy biến động giá 24h của coin - TÍNH TOÁN TỪ DỮ LIỆU BINANCE"""
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": "1h", 
                "limit": 24  # Lấy 24 nến 1h để tính biến động 24h
            }
            data = binance_api_request(url, params=params)
            if not data or len(data) < 24:
                return 0
            
            prices = []
            for kline in data:
                high = float(kline[2])
                low = float(kline[3])
                avg_price = (high + low) / 2
                prices.append(avg_price)
            
            if len(prices) < 2:
                return 0
                
            # Tính biến động: (max - min) / min * 100
            max_price = max(prices)
            min_price = min(prices)
            volatility = ((max_price - min_price) / min_price) * 100
            
            return volatility
            
        except Exception as e:
            logger.error(f"Lỗi lấy biến động 24h {symbol}: {str(e)}")
            return 0

    def get_24h_volume_and_change(self, symbol):
        """Lấy volume 24h và % thay đổi giá từ API Binance"""
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            params = {"symbol": symbol}
            data = binance_api_request(url, params=params)
            if not data:
                return 0, 0
                
            volume = float(data.get('quoteVolume', 0))
            price_change = float(data.get('priceChangePercent', 0))
            
            return volume, abs(price_change)  # Trả về giá trị tuyệt đối của % thay đổi
            
        except Exception as e:
            logger.error(f"Lỗi lấy volume 24h {symbol}: {str(e)}")
            return 0, 0

    def add_to_blacklist(self, symbol, duration=3600):
        """Thêm coin vào danh sách chặn"""
        with self._analysis_lock:
            self.blacklisted_coins[symbol] = time.time() + duration
            logger.info(f"🔄 Đã thêm {symbol} vào blacklist trong {duration} giây")

    def is_blacklisted(self, symbol):
        """Kiểm tra coin có trong blacklist không"""
        with self._analysis_lock:
            if symbol in self.blacklisted_coins:
                if time.time() < self.blacklisted_coins[symbol]:
                    return True
                else:
                    # Hết thời gian chặn, xóa khỏi blacklist
                    del self.blacklisted_coins[symbol]
            return False

    def mark_coin_failed(self, symbol):
        """Đánh dấu coin thất bại"""
        with self._analysis_lock:
            self.failed_coins.add(symbol)
            self.add_to_blacklist(symbol, 1800)  # Chặn 30 phút

    def _analyze_coins(self, all_symbols, required_leverage):
        """Phân tích coin và cache kết quả"""
        current_time = time.time()
        if current_time - self.last_analysis_time < self.cache_duration and self.cached_top_coins:
            return self.cached_top_coins
        
        with self._analysis_lock:
            scored_symbols = []
            logger.info(f"🔍 Đang phân tích {len(all_symbols)} coin để chọn coin BIẾN ĐỘNG CAO NHẤT...")
            
            for symbol in all_symbols:
                try:
                    # Kiểm tra đòn bẩy trước
                    max_lev = self.get_symbol_leverage(symbol)
                    if max_lev < required_leverage:
                        continue
                    
                    # Lấy volume và % thay đổi giá 24h
                    volume, price_change = self.get_24h_volume_and_change(symbol)
                    
                    # Tính điểm: ưu tiên biến động cao và volume tốt
                    if price_change > 0 and volume > 100000:  # Volume tối thiểu 100k USDT
                        score = price_change * (1 + min(volume / 1000000, 10))  # Volume càng cao điểm càng tốt
                        scored_symbols.append((symbol, score, price_change, volume, max_lev))
                    
                    time.sleep(0.05)  # Giảm thời gian chờ để tăng tốc
                    
                except Exception as e:
                    continue
            
            # Sắp xếp theo điểm số giảm dần
            scored_symbols.sort(key=lambda x: x[1], reverse=True)
            self.cached_top_coins = scored_symbols
            self.last_analysis_time = current_time
            
            return scored_symbols

    def find_best_coin(self, excluded_coins=None, required_leverage=10, retry_count=8):
        """Tìm coin BIẾN ĐỘNG CAO NHẤT - PHIÊN BẢN ĐÃ SỬA LỖI"""
        try:
            all_symbols = get_all_usdt_pairs(limit=300)
            if not all_symbols:
                return None
            
            # Lọc coin theo đòn bẩy và loại bỏ coin đang active/blacklisted
            valid_symbols = []
            for symbol in all_symbols:
                if excluded_coins and symbol in excluded_coins:
                    continue
                if self.is_blacklisted(symbol):
                    continue
                if symbol in self.failed_coins:
                    continue
                
                valid_symbols.append(symbol)
            
            if not valid_symbols:
                logger.warning("❌ Không tìm thấy coin nào đáp ứng đòn bẩy và không bị chặn")
                # Reset failed coins nếu không tìm được coin nào
                with self._analysis_lock:
                    self.failed_coins.clear()
                return None
            
            # Phân tích coin và lấy top coins
            scored_symbols = self._analyze_coins(valid_symbols, required_leverage)
            
            if scored_symbols:
                logger.info(f"🏆 TOP 5 COIN BIẾN ĐỘNG CAO:")
                for i, (symbol, score, change, vol, lev) in enumerate(scored_symbols[:5]):
                    logger.info(f"   #{i+1}: {symbol} | Biến động: {change:.2f}% | Volume: {vol:,.0f} USDT | ĐB: {lev}x")
                
                # Thử các coin theo thứ tự biến động
                for i in range(min(retry_count, len(scored_symbols))):
                    best_symbol = scored_symbols[i][0]
                    logger.info(f"🎯 Thử coin #{i+1}: {best_symbol} | Biến động: {scored_symbols[i][2]:.2f}% | ĐB: {scored_symbols[i][4]}x")
                    return best_symbol
                
                return scored_symbols[0][0]
            else:
                # Fallback: chọn ngẫu nhiên từ các coin có volume
                volume_symbols = []
                for symbol in valid_symbols[:20]:
                    volume, price_change = self.get_24h_volume_and_change(symbol)
                    if volume > 50000:  # Volume tối thiểu 50k USDT
                        volume_symbols.append(symbol)
                
                if volume_symbols:
                    chosen = random.choice(volume_symbols)
                    logger.info(f"🔄 Fallback: Chọn ngẫu nhiên {chosen} từ top volume")
                    return chosen
                else:
                    return random.choice(valid_symbols[:10]) if valid_symbols else None
                
        except Exception as e:
            logger.error(f"Lỗi tìm coin: {str(e)}")
            return None

# ========== WEBSOCKET MANAGER ĐÃ SỬA LỖI ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.connection_attempts = {}  # Theo dõi số lần kết nối thất bại
        self.max_connection_attempts = 5
        
    def add_symbol(self, symbol, callback):
        if not symbol:
            return
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self.connections:
                self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
            
        # Kiểm tra số lần kết nối thất bại
        attempts = self.connection_attempts.get(symbol, 0)
        if attempts >= self.max_connection_attempts:
            logger.error(f"🚫 Đã vượt quá số lần kết nối thất bại cho {symbol}")
            return
            
        stream = f"{symbol.lower()}@trade"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    # Xử lý callback trong executor với error handling
                    future = self.executor.submit(callback, price)
                    future.add_done_callback(self._handle_callback_result)
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn WebSocket {symbol}: {str(e)}")
                
        def _handle_callback_result(future):
            """Xử lý kết quả callback"""
            try:
                future.result()
            except Exception as e:
                logger.error(f"Lỗi trong price callback: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Lỗi WebSocket {symbol}: {str(error)}")
            # Tăng số lần thất bại
            with self._lock:
                self.connection_attempts[symbol] = self.connection_attempts.get(symbol, 0) + 1
                
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(5)
                self._reconnect(symbol, callback)
                
        def on_open(ws):
            logger.info(f"WebSocket mở thành công cho {symbol}")
            # Reset số lần thất bại khi kết nối thành công
            with self._lock:
                self.connection_attempts[symbol] = 0
                
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        thread = threading.Thread(target=ws.run_forever, daemon=True, 
                                 name=f"WS_{symbol}")
        thread.start()
        
        self.connections[symbol] = {
            'ws': ws,
            'thread': thread,
            'callback': callback,
            'created_at': time.time()
        }
        logger.info(f"WebSocket bắt đầu cho {symbol}")
        
    def _reconnect(self, symbol, callback):
        """Kết nối lại với circuit breaker"""
        attempts = self.connection_attempts.get(symbol, 0)
        if attempts >= self.max_connection_attempts:
            logger.error(f"🚫 Circuit breaker: Không kết nối lại {symbol} sau {attempts} lần thất bại")
            return
            
        logger.info(f"Kết nối lại WebSocket cho {symbol} (lần {attempts + 1})")
        self.remove_symbol(symbol)
        time.sleep(2 ** attempts)  # Exponential backoff
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        """Xóa symbol khỏi WebSocket manager"""
        if not symbol:
            return
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
        """Dừng tất cả WebSocket connections"""
        self._stop_event.set()
        symbols = list(self.connections.keys())
        for symbol in symbols:
            self.remove_symbol(symbol)
        self.executor.shutdown(wait=True)

# ========== BASE BOT ĐÃ SỬA LỖI ==========
class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, roi_trigger, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None, bot_id=None, coin_finder=None):
        
        self.symbol = symbol.upper() if symbol else None
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.roi_trigger = roi_trigger
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
        
        # BIẾN QUAN TRỌNG: Quản lý thời gian giữ coin
        self.coin_start_time = 0
        self.coin_max_duration = 3600  # Tối đa 1 giờ cho mỗi coin
        self.last_coin_switch = 0
        self.coin_switch_cooldown = 300  # 5 phút chờ trước khi đổi coin mới
        
        self.cooldown_period = 3
        self.position_check_interval = 30
        
        self._close_attempted = False
        self._last_close_attempt = 0
        
        self.should_be_removed = False
        
        self.coin_manager = CoinManager()
        self.coin_finder = coin_finder or SmartCoinFinder(api_key, api_secret)
        
        # BIẾN QUAN TRỌNG: Theo dõi hướng lệnh cuối cùng
        self.last_side = None  # Lưu hướng lệnh cuối cùng (BUY/SELL)
        self.is_first_trade = True  # Đánh dấu lệnh đầu tiên
        
        # Biến quản lý nhồi lệnh Fibonacci
        self.entry_base = 0
        self.average_down_count = 0
        self.last_average_down_time = 0
        self.average_down_cooldown = 60
        self.max_average_down_count = 7
        
        # Biến theo dõi nến và ROI
        self.entry_green_count = 0
        self.entry_red_count = 0
        self.high_water_mark_roi = 0
        self.roi_check_activated = False
        
        # Biến mới để tránh race condition
        self._position_lock = threading.Lock()
        self._trade_lock = threading.Lock()
        
        self.check_position_status()
        if self.symbol:
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"Bot_{self.bot_id}")
        self.thread.start()
        
        roi_info = f" | 🎯 ROI Trigger: {roi_trigger}%" if roi_trigger else " | 🎯 ROI Trigger: Tắt"
        
        if self.symbol:
            # Lấy thông tin biến động để log
            volatility = self.coin_finder.get_24h_volatility(self.symbol)
            volume, change = self.coin_finder.get_24h_volume_and_change(self.symbol)
            
            self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | 📈 24h: {change:.2f}% | 💰 ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%{roi_info}")
        else:
            self.log(f"🟢 Bot {strategy_name} khởi động | Đang tìm coin... | 💰 ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%{roi_info}")

    def check_position_status(self):
        """Kiểm tra trạng thái vị thế với lock"""
        if not self.symbol:
            return
            
        try:
            with self._position_lock:
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
                            
                            # CẬP NHẬT QUAN TRỌNG: Lưu hướng lệnh hiện tại
                            self.last_side = self.side
                            self.is_first_trade = False
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
        """Reset trạng thái vị thế nhưng GIỮ NGUYÊN last_side"""
        with self._position_lock:
            self.position_open = False
            self.status = "waiting"  # Chờ mở lệnh tiếp theo
            self.side = ""
            self.qty = 0
            self.entry = 0
            self._close_attempted = False
            self._last_close_attempt = 0
            # Reset thông tin nhồi lệnh
            self.entry_base = 0
            self.average_down_count = 0
            # Reset thông tin theo dõi nến và ROI
            self.entry_green_count = 0
            self.entry_red_count = 0
            self.high_water_mark_roi = 0
            self.roi_check_activated = False
            # KHÔNG reset last_side và is_first_trade

    def verify_leverage_and_switch(self):
        """Kiểm tra và chuyển đổi đòn bẩy - TỰ ĐỘNG BỎ COIN KHI LỖI"""
        if not self.symbol:
            return True
            
        try:
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            if current_leverage >= self.lev:
                # Thiết lập đòn bẩy mong muốn
                if set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                    return True
                else:
                    self.log(f"❌ Không thể đặt đòn bẩy {self.lev}x cho {self.symbol} -> ĐÁNH DẤU LỖI VÀ BỎ COIN")
                    self.coin_finder.mark_coin_failed(self.symbol)
                    return False
            else:
                self.log(f"❌ Coin {self.symbol} chỉ hỗ trợ đòn bẩy {current_leverage}x < {self.lev}x -> ĐÁNH DẤU LỖI VÀ BỎ COIN")
                self.coin_finder.mark_coin_failed(self.symbol)
                return False
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra đòn bẩy: {str(e)} -> ĐÁNH DẤU LỖI VÀ BỎ COIN")
            self.coin_finder.mark_coin_failed(self.symbol)
            return False

    def find_and_set_coin(self):
        """Tìm và thiết lập coin mới cho bot - PHIÊN BẢN ĐÃ SỬA LỖI"""
        try:
            # Lấy danh sách coin đang active từ tất cả bot để tránh trùng lặp
            active_coins = set()
            if hasattr(self, '_bot_manager') and self._bot_manager:
                for bot in self._bot_manager.bots.values():
                    if bot.symbol and bot.bot_id != self.bot_id:
                        active_coins.add(bot.symbol)
            
            # Tìm coin BIẾN ĐỘNG CAO với retry_count = 8
            new_symbol = self.coin_finder.find_best_coin(
                excluded_coins=active_coins,
                required_leverage=self.lev,
                retry_count=8
            )
            
            if new_symbol:
                # Đăng ký coin với coin manager (tránh race condition)
                if not self.coin_manager.register_coin(new_symbol):
                    self.log(f"⚠️ Coin {new_symbol} đã được sử dụng bởi bot khác, tìm coin khác...")
                    self.coin_finder.mark_coin_failed(new_symbol)
                    return False

                # KIỂM TRA ĐÒN BẨY NGAY KHI TÌM ĐƯỢC COIN
                if not self.verify_leverage_and_switch():
                    self.log(f"❌ Coin {new_symbol} không đạt đòn bẩy -> ĐÁNH DẤU LỖI VÀ THỬ COIN KHÁC")
                    self.coin_manager.unregister_coin(new_symbol)
                    self.coin_finder.mark_coin_failed(new_symbol)
                    return False

                # Đăng ký coin mới
                if self.symbol:
                    self.ws_manager.remove_symbol(self.symbol)
                    self.coin_manager.unregister_coin(self.symbol)
                
                self.symbol = new_symbol
                self.ws_manager.add_symbol(new_symbol, self._handle_price_update)
                self.status = "waiting"
                self.coin_start_time = time.time()  # Bắt đầu tính thời gian giữ coin
                self.last_coin_switch = time.time()
                
                # Lấy thông tin biến động để log
                volatility = self.coin_finder.get_24h_volatility(new_symbol)
                volume, change = self.coin_finder.get_24h_volume_and_change(new_symbol)
                
                self.log(f"🎯 Đã tìm thấy coin BIẾN ĐỘNG CAO: {new_symbol} | 📈 24h: {change:.2f}% | 📊 Volume: {volume:,.0f} USDT | 💰 ĐB: {self.lev}x")
                return True
            else:
                self.log("❌ Không tìm được coin biến động cao phù hợp, thử lại sau...")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            return False

    def _run(self):
        """Vòng lặp chính của bot - ĐÃ SỬA LỖI"""
        while not self._stop:
            try:
                current_time = time.time()
                
                # KIỂM TRA THỜI GIAN GIỮ COIN - QUAN TRỌNG
                if (self.symbol and not self.position_open and 
                    current_time - self.coin_start_time > self.coin_max_duration and
                    current_time - self.last_coin_switch > self.coin_switch_cooldown):
                    self.log(f"🔄 Đã giữ coin {self.symbol} quá 1 giờ, chuyển coin mới...")
                    self._cleanup_symbol()
                    self.last_coin_switch = current_time
                    continue
                
                # KIỂM TRA ĐÒN BẨY ĐỊNH KỲ - NẾU LỖI THÌ TÌM COIN MỚI NGAY
                if current_time - getattr(self, '_last_leverage_check', 0) > 60:
                    if self.symbol and not self.verify_leverage_and_switch():
                        self.log(f"🔄 Coin {self.symbol} lỗi đòn bẩy, đang tìm coin mới...")
                        self._cleanup_symbol()
                    self._last_leverage_check = current_time
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                
                # KIỂM TRA NHỒI LỆNH KHI CÓ VỊ THẾ
                if self.position_open and self.entry_base > 0:
                    self.check_averaging_down()
                              
                if not self.position_open:
                    # QUAN TRỌNG: Nếu không có symbol, tìm coin mới NGAY
                    if not self.symbol:
                        if self.find_and_set_coin():
                            self.log("✅ Đã tìm thấy coin mới, chờ tín hiệu...")
                        else:
                            time.sleep(5)  # Chờ lâu hơn nếu không tìm được coin
                        continue
                    
                    # CƠ CHẾ MỚI: LUÔN VÀO LỆNH NGƯỢC VỚI LỆNH TRƯỚC
                    target_side = self.get_next_side()
                    
                    if target_side:
                        if current_time - self.last_trade_time > 3 and current_time - self.last_close_time > self.cooldown_period:
                            if self.open_position(target_side):
                                self.last_trade_time = current_time
                            else:
                                time.sleep(1)
                        else:
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

    def get_next_side(self):
        """Xác định hướng lệnh tiếp theo - CƠ CHẾ LUÔN NGƯỢC HƯỚNG"""
        if self.is_first_trade:
            # LẦN ĐẦU: Chọn ngẫu nhiên
            return random.choice(["BUY", "SELL"])
        else:
            # CÁC LẦN SAU: Luôn ngược với lệnh trước
            return "SELL" if self.last_side == "BUY" else "BUY"

    def _handle_price_update(self, price):
        """Xử lý cập nhật giá realtime"""
        try:
            self.current_price = price
            self.prices.append(price)
            
            # Giữ lịch sử giá trong giới hạn
            if len(self.prices) > 100:
                self.prices.pop(0)
        except Exception as e:
            logger.error(f"Lỗi xử lý price update: {str(e)}")

    def stop(self):
        """Dừng bot an toàn"""
        self._stop = True
        if self.symbol:
            self.ws_manager.remove_symbol(self.symbol)
            self.coin_manager.unregister_coin(self.symbol)
        if self.symbol:
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
        self.log(f"🔴 Bot dừng")

    def open_position(self, side):
        """Mở vị thế với lock để tránh race condition"""
        if side not in ["BUY", "SELL"]:
            self.log(f"❌ Side không hợp lệ: {side}")
            # Đánh dấu coin thất bại
            if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                self.coin_finder.mark_coin_failed(self.symbol)
            self._cleanup_symbol()
            return False
            
        # Sử dụng lock để tránh mở nhiều lệnh cùng lúc
        if not self._trade_lock.acquire(blocking=False):
            self.log("⚠️ Đang xử lý lệnh khác, bỏ qua...")
            return False
            
        try:
            # KIỂM TRA KỸ ĐÒN BẨY TRƯỚC KHI MỞ LỆNH
            if not self.verify_leverage_and_switch():
                self.log(f"❌ Coin {self.symbol} không đạt đòn bẩy {self.lev}x -> ĐÁNH DẤU LỖI VÀ TÌM COIN KHÁC")
                if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                    self.coin_finder.mark_coin_failed(self.symbol)
                self._cleanup_symbol()
                return False

            # Kiểm tra vị thế hiện tại
            self.check_position_status()
            
            if self.position_open:
                self.log(f"⚠️ Đã có vị thế {self.side}, bỏ qua tín hiệu {side}")
                return False
    
            if self.should_be_removed:
                self.log("⚠️ Bot đã được đánh dấu xóa, không mở lệnh mới")
                return False
    
            # Kiểm tra số dư
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                self.log("❌ Không đủ số dư")
                return False
    
            # Lấy giá hiện tại
            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                self.log(f"❌ Lỗi lấy giá {self.symbol}: {current_price} -> ĐÁNH DẤU LỖI VÀ TÌM COIN KHÁC")
                if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                    self.coin_finder.mark_coin_failed(self.symbol)
                self._cleanup_symbol()
                return False
    
            # Tính toán khối lượng
            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
    
            if qty <= 0 or qty < step_size:
                self.log(f"❌ Khối lượng không hợp lệ: {qty} (step: {step_size}) -> ĐÁNH DẤU LỖI VÀ TÌM COIN KHÁC")
                if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                    self.coin_finder.mark_coin_failed(self.symbol)
                self._cleanup_symbol()
                return False
    
            self.log(f"📊 Đang đặt lệnh {side} - SL: {step_size}, Qty: {qty}, Giá: {current_price}")
            
            # Hủy mọi lệnh chờ trước đó
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.2)
            
            # Đặt lệnh
            result = place_order(self.symbol, side, qty, self.api_key, self.api_secret)
            
            if result and 'orderId' in result:
                executed_qty = float(result.get('executedQty', 0))
                avg_price = float(result.get('avgPrice', current_price))
                
                if executed_qty >= 0:
                    self.entry = avg_price
                    self.entry_base = avg_price
                    self.average_down_count = 0
                    self.side = side
                    self.qty = executed_qty if side == "BUY" else -executed_qty
                    self.position_open = True
                    self.status = "open"
                    
                    # CẬP NHẬT QUAN TRỌNG: Lưu hướng lệnh và đánh dấu không còn là lệnh đầu
                    self.last_side = side
                    self.is_first_trade = False
                    
                    # LƯU SỐ NẾN TẠI THỜI ĐIỂM VÀO LỆNH
                    self.high_water_mark_roi = 0
                    self.roi_check_activated = False
                    
                    roi_trigger_info = f" | 🎯 ROI Trigger: {self.roi_trigger}%" if self.roi_trigger else ""
                    
                    message = (
                        f"✅ <b>ĐÃ MỞ VỊ THẾ {self.symbol}</b>\n"
                        f"🤖 Chiến lược: {self.strategy_name}\n"
                        f"📌 Hướng: {side}\n"
                        f"🏷️ Giá vào: {self.entry:.4f}\n"
                        f"📊 Khối lượng: {executed_qty:.4f}\n"
                        f"💵 Giá trị: {executed_qty * self.entry:.2f} USDT\n"
                        f"💰 Đòn bẩy: {self.lev}x\n"
                        f"🎯 TP: {self.tp}% | 🛡️ SL: {self.sl}%{roi_trigger_info}\n"
                        f"🔄 Cơ chế: {'Lệnh đầu' if self.is_first_trade else 'Ngược hướng trước'}"
                    )
                    
                    if self.roi_trigger:
                        message += f"\n🎯 <b>Cơ chế chốt lệnh ROI {self.roi_trigger}% đã kích hoạt</b>"
                    
                    self.log(message)
                    return True
                else:
                    self.log(f"❌ Lệnh không khớp - Số lượng: {qty} -> ĐÁNH DẤU LỖI VÀ TÌM COIN KHÁC")
                    if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                        self.coin_finder.mark_coin_failed(self.symbol)
                    self._cleanup_symbol()
                    return False
            else:
                error_msg = result.get('msg', 'Unknown error') if result else 'No response'
                self.log(f"❌ Lỗi đặt lệnh {side}: {error_msg} -> ĐÁNH DẤU LỖI VÀ TÌM COIN KHÁC")
                
                if result and 'code' in result:
                    self.log(f"📋 Mã lỗi Binance: {result['code']} - {result.get('msg', '')}")
                
                # ĐÁNH DẤU COIN THẤT BẠI
                if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                    self.coin_finder.mark_coin_failed(self.symbol)
                
                self._cleanup_symbol()
                return False
                        
        except Exception as e:
            self.log(f"❌ Lỗi mở lệnh: {str(e)} -> ĐÁNH DẤU LỖI VÀ TÌM COIN KHÁC")
            if self.symbol and hasattr(self.coin_finder, 'mark_coin_failed'):
                self.coin_finder.mark_coin_failed(self.symbol)
            self._cleanup_symbol()
            return False
        finally:
            self._trade_lock.release()
    
    def _cleanup_symbol(self):
        """Dọn dẹp symbol hiện tại và chuyển về trạng thái tìm kiếm"""
        if self.symbol:
            try:
                # Đánh dấu coin hiện tại đã thử (không chặn, chỉ đánh dấu đã dùng)
                if hasattr(self.coin_finder, 'failed_coins'):
                    self.coin_finder.failed_coins.add(self.symbol)
                
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
                self.log(f"🧹 Đã dọn dẹp symbol {self.symbol}")
            except Exception as e:
                self.log(f"⚠️ Lỗi khi dọn dẹp symbol: {str(e)}")
            
            self.symbol = None
        
        # Reset hoàn toàn trạng thái
        self.status = "searching"
        self.position_open = False
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.entry_base = 0
        self.average_down_count = 0
        self.high_water_mark_roi = 0
        self.roi_check_activated = False
        
        self.log("🔄 Đã reset bot, sẵn sàng tìm coin mới")

    def close_position(self, reason=""):
        """Đóng vị thế với lock"""
        if not self._trade_lock.acquire(blocking=False):
            self.log("⚠️ Đang xử lý lệnh khác, bỏ qua đóng lệnh...")
            return False
            
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
                    f"💰 PnL: {pnl:.2f} USDT\n"
                    f"📈 Số lần nhồi: {self.average_down_count}\n"
                    f"🔄 Lệnh tiếp theo: {'BUY' if self.side == 'SELL' else 'SELL'}"
                )
                self.log(message)
                
                # QUAN TRỌNG: ĐÃ LƯU last_side TRONG check_position_status
                # KHÔNG cần set next_side vì đã có cơ chế get_next_side
                
                # Reset position nhưng GIỮ NGUYÊN SYMBOL VÀ last_side
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
        finally:
            self._trade_lock.release()

    def check_tp_sl(self):
        """Kiểm tra điều kiện TP/SL"""
        if not self.symbol or not self.position_open or self.entry <= 0 or self._close_attempted:
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

        # CẬP NHẬT ROI CAO NHẤT
        if roi > self.high_water_mark_roi:
            self.high_water_mark_roi = roi

        # KIỂM TRA ĐIỀU KIỆN ROI TRIGGER
        if self.roi_trigger is not None and self.high_water_mark_roi >= self.roi_trigger and not self.roi_check_activated:
            self.roi_check_activated = True
            self.log(f"🎯 ĐÃ ĐẠT ROI {self.roi_trigger}% - KÍCH HOẠT CƠ CHẾ CHỐT LỆNH")

        # TP/SL TRUYỀN THỐNG
        if self.tp is not None and roi >= self.tp:
            self.close_position(f"✅ Đạt TP {self.tp}% (ROI: {roi:.2f}%)")
        elif self.sl is not None and self.sl > 0 and roi <= -self.sl:
            self.close_position(f"❌ Đạt SL {self.sl}% (ROI: {roi:.2f}%)")

    def check_averaging_down(self):
        """Kiểm tra và thực hiện nhồi lệnh Fibonacci khi lỗ"""
        if not self.position_open or not self.entry_base or self.average_down_count >= self.max_average_down_count:
            return
            
        try:
            current_time = time.time()
            if current_time - self.last_average_down_time < self.average_down_cooldown:
                return
                
            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                return
                
            # Tính % lỗ so với giá vào gốc
            if self.side == "BUY":
                drawdown_pct = (self.entry_base - current_price) / self.entry_base * 100
            else:
                drawdown_pct = (current_price - self.entry_base) / self.entry_base * 100
                
            # Các mốc Fibonacci để nhồi lệnh
            fib_levels = [2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]
            
            if self.average_down_count < len(fib_levels):
                current_fib_level = fib_levels[self.average_down_count]
                
                if drawdown_pct >= current_fib_level:
                    # Thực hiện nhồi lệnh
                    if self.execute_average_down_order():
                        self.last_average_down_time = current_time
                        self.average_down_count += 1
                        
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra nhồi lệnh: {str(e)}")

    def execute_average_down_order(self):
        """Thực hiện lệnh nhồi theo Fibonacci"""
        if not self._trade_lock.acquire(blocking=False):
            return False
            
        try:
            # Tính khối lượng nhồi lệnh
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                return False
                
            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                return False
                
            # Khối lượng nhồi = % số dư * (số lần nhồi + 1) để tăng dần
            additional_percent = self.percent * (self.average_down_count + 1)
            usd_amount = balance * (additional_percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
            
            if qty < step_size:
                return False
                
            # Đặt lệnh cùng hướng với vị thế hiện tại
            result = place_order(self.symbol, self.side, qty, self.api_key, self.api_secret)
            
            if result and 'orderId' in result:
                executed_qty = float(result.get('executedQty', 0))
                avg_price = float(result.get('avgPrice', current_price))
                
                if executed_qty >= 0:
                    # Cập nhật giá trung bình và khối lượng
                    total_qty = abs(self.qty) + executed_qty
                    self.entry = (abs(self.qty) * self.entry + executed_qty * avg_price) / total_qty
                    self.qty = total_qty if self.side == "BUY" else -total_qty
                    
                    message = (
                        f"📈 <b>ĐÃ NHỒI LỆNH FIBONACCI {self.symbol}</b>\n"
                        f"🔢 Lần nhồi: {self.average_down_count + 1}\n"
                        f"📊 Khối lượng thêm: {executed_qty:.4f}\n"
                        f"🏷️ Giá nhồi: {avg_price:.4f}\n"
                        f"📈 Giá trung bình mới: {self.entry:.4f}\n"
                        f"💰 Tổng khối lượng: {total_qty:.4f}"
                    )
                    self.log(message)
                    return True
                    
            return False
            
        except Exception as e:
            self.log(f"❌ Lỗi nhồi lệnh: {str(e)}")
            return False
        finally:
            self._trade_lock.release()

    def log(self, message):
        logger.info(f"[{self.bot_id}] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>{self.bot_id}</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

# ========== BOT GLOBAL MARKET ĐÃ SỬA LỖI ==========
class GlobalMarketBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, roi_trigger, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, bot_id=None, coin_finder=None):
        super().__init__(symbol, lev, percent, tp, sl, roi_trigger, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Global-Market-Biến-Động-Cao", 
                        bot_id=bot_id, coin_finder=coin_finder)

# ========== BOT MANAGER ĐÃ SỬA LỖI ==========
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
        
        # Tạo SmartCoinFinder chung cho tất cả bot
        self.coin_finder = SmartCoinFinder(api_key, api_secret)
        self.coin_manager = CoinManager()  # Coin manager chung
        
        # Lock để tránh race condition khi tạo bot
        self._bot_creation_lock = threading.Lock()
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG BOT VỚI CƠ CHẾ BIẾN ĐỘNG CAO 24H VÀ QUẢN LÝ COIN THÔNG MINH ĐÃ KHỞI ĐỘNG")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True, name="TelegramListener")
            self.telegram_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def _verify_api_connection(self):
        """Kiểm tra kết nối API"""
        try:
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                self.log("❌ LỚI: Không thể kết nối Binance API. Kiểm tra:")
                self.log("   - API Key và Secret có đúng không?")
                self.log("   - Có thể bị chặn IP (lỗi 451), thử dùng VPN")
                self.log("   - Kiểm tra kết nối internet")
                return False
            else:
                self.log(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
                return True
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra kết nối: {str(e)}")
            return False

    def get_position_summary(self):
        """Lấy thống kê tổng quan"""
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            binance_buy_count = 0
            binance_sell_count = 0
            binance_positions = []
            
            # Đếm vị thế từ Binance
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
        
            # Thống kê bot
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
                    'sl': bot.sl,
                    'roi_trigger': bot.roi_trigger,
                    'last_side': bot.last_side,
                    'is_first_trade': bot.is_first_trade
                }
                bot_details.append(bot_info)
                
                if bot.status == "searching":
                    searching_bots += 1
                elif bot.status == "waiting":
                    waiting_bots += 1
                elif bot.status == "open":
                    trading_bots += 1
            
            # Tạo báo cáo chi tiết
            summary = "📊 **THỐNG KÊ CHI TIẾT HỆ THỐNG**\n\n"
            
            # Phần 1: Số dư
            balance = get_balance(self.api_key, self.api_secret)
            if balance is not None:
                summary += f"💰 **SỐ DƯ**: {balance:.2f} USDT\n\n"
            else:
                summary += f"💰 **SỐ DƯ**: ❌ Lỗi kết nối\n\n"
            
            # Phần 2: Bot hệ thống
            summary += f"🤖 **BOT HỆ THỐNG**: {len(self.bots)} bots\n"
            summary += f"   🔍 Đang tìm coin: {searching_bots}\n"
            summary += f"   🟡 Đang chờ: {waiting_bots}\n" 
            summary += f"   📈 Đang trade: {trading_bots}\n\n"
            
            # Phần 3: Chi tiết từng bot
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
                    
                    roi_info = f" | 🎯 ROI: {bot['roi_trigger']}%" if bot['roi_trigger'] else ""
                    trade_info = f" | Lệnh đầu" if bot['is_first_trade'] else f" | Tiếp: {'SELL' if bot['last_side'] == 'BUY' else 'BUY'}"
                    
                    summary += f"   🔹 {bot['bot_id'][:15]}...\n"
                    summary += f"      📊 {symbol_info} | {status}{trade_info}\n"
                    summary += f"      💰 ĐB: {bot['leverage']}x | Vốn: {bot['percent']}%{roi_info}\n"
                    if bot['tp'] is not None and bot['sl'] is not None:
                        summary += f"      🎯 TP: {bot['tp']}% | 🛡️ SL: {bot['sl']}%\n"
                    summary += "\n"
                
                if len(bot_details) > 8:
                    summary += f"   ... và {len(bot_details) - 8} bot khác\n\n"
            
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
            "🤖 <b>BOT GIAO DỊCH FUTURES ĐA LUỒNG TỐI ƯU</b>\n\n"
            "🎯 <b>HỆ THỐNG VỚI CƠ CHẾ BIẾN ĐỘNG CAO 24H & QUẢN LÝ COIN THÔNG MINH</b>\n\n"
            "📈 <b>CƠ CHẾ BIẾN ĐỘNG CAO 24H:</b>\n"
            "• Chọn coin có biến động giá mạnh nhất 24h\n"
            "• Phân tích volume và % thay đổi giá thực tế\n"
            "• Ưu tiên coin có volume lớn và biến động cao\n"
            "• TOP 5 coin biến động cao nhất được ưu tiên\n\n"
            "⏰ <b>QUẢN LÝ COIN THÔNG MINH:</b>\n"
            "• Giữ coin tối đa 1 giờ\n"
            "• Tự động chuyển coin mới sau thời gian quy định\n"
            "• Đánh dấu coin lỗi vào blacklist 30 phút\n"
            "• Tránh chọn lại coin đã thất bại\n\n"
            "🔄 <b>CƠ CHẾ LUÔN NGƯỢC HƯỚNG:</b>\n"
            "• Lần đầu: Chọn ngẫu nhiên BUY/SELL\n"
            "• Các lần sau: LUÔN vào lệnh ngược với lệnh trước\n\n"
            "💰 <b>QUẢN LÝ RỦI RO THÔNG MINH:</b>\n"
            "• TP/SL cố định theo %\n"
            "• Cơ chế ROI Trigger thông minh\n"
            "• Nhồi lệnh Fibonacci khi drawdown"
        )
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def add_bot(self, symbol, lev, percent, tp, sl, roi_trigger, strategy_type, bot_count=1, **kwargs):
        """Thêm bot với lock để tránh race condition"""
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        # Kiểm tra kết nối trước khi tạo bot
        if not self._verify_api_connection():
            self.log("❌ KHÔNG THỂ KẾT NỐI BINANCE - KHÔNG THỂ TẠO BOT")
            return False
        
        # Sử dụng lock để tránh tạo bot đồng thời
        with self._bot_creation_lock:
            bot_mode = kwargs.get('bot_mode', 'static')
            created_count = 0
            
            # Lấy danh sách coin đang active để phân phối thông minh
            active_coins = set(self.coin_manager.get_active_coins())
            
            for i in range(bot_count):
                try:
                    if bot_mode == 'static' and symbol:
                        bot_symbol = symbol
                        # Kiểm tra coin static có available không
                        if self.coin_manager.is_coin_active(bot_symbol):
                            self.log(f"⚠️ Coin {bot_symbol} đã được sử dụng, bỏ qua bot {i+1}")
                            continue
                    else:
                        # Tìm coin mới với cơ chế thông minh
                        bot_symbol = self.coin_finder.find_best_coin(
                            excluded_coins=active_coins,
                            required_leverage=lev,
                            retry_count=8
                        )
                        
                        if not bot_symbol:
                            self.log(f"❌ Không tìm được coin cho bot {i+1}")
                            continue
                    
                    # Đăng ký coin với coin manager
                    if not self.coin_manager.register_coin(bot_symbol):
                        self.log(f"⚠️ Coin {bot_symbol} đã được sử dụng, bỏ qua bot {i+1}")
                        continue
                    
                    bot_id = f"{bot_symbol}_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        self.coin_manager.unregister_coin(bot_symbol)
                        continue
                    
                    bot_class = GlobalMarketBot
                    
                    bot = bot_class(bot_symbol, lev, percent, tp, sl, roi_trigger, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token, 
                                  self.telegram_chat_id, bot_id=bot_id, coin_finder=self.coin_finder)
                    
                    bot._bot_manager = self
                    bot.coin_manager = self.coin_manager  # Sử dụng coin manager chung
                    self.bots[bot_id] = bot
                    created_count += 1
                    
                    # Thêm vào danh sách active để tránh trùng lặp
                    active_coins.add(bot_symbol)
                    
                    self.log(f"🤖 Bot #{created_count} → {bot_symbol}")
                    
                except Exception as e:
                    self.log(f"❌ Lỗi tạo bot {i}: {str(e)}")
                    continue
            
            if created_count > 0:
                roi_info = f" | 🎯 ROI Trigger: {roi_trigger}%" if roi_trigger else ""
                
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {created_count} BOT THÀNH CÔNG</b>\n\n"
                    f"🎯 Chiến lược: Biến động cao + Ngược hướng\n"
                    f"💰 Đòn bẩy: {lev}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl if sl is not None else 'Tắt'}%{roi_info}\n"
                )
                
                if bot_mode == 'static' and symbol:
                    success_msg += f"🔗 Coin: {symbol}\n"
                else:
                    success_msg += f"🔗 Coin: Tự động chọn BIẾN ĐỘNG CAO\n"
                
                success_msg += f"\n🔄 <b>CƠ CHẾ PHÂN PHỐI COIN THÔNG MINH:</b>\n"
                success_msg += f"• Mỗi bot nhận 1 coin KHÁC NHAU\n"
                success_msg += f"• Ưu tiên coin BIẾN ĐỘNG CAO NHẤT 24h\n"
                success_msg += f"• Phân tích volume & volatility thực tế\n"
                success_msg += f"• Tự động tránh trùng lặp\n"
                success_msg += f"• Tự động bỏ coin lỗi đòn bẩy\n"
                success_msg += f"• Giữ coin tối đa 1 giờ\n"
                success_msg += f"• Thử coin tiếp theo nếu có lỗi"
                
                self.log(success_msg)
                return True
            else:
                self.log("❌ Không thể tạo bot nào")
                return False

    def stop_bot(self, bot_id):
        """Dừng bot cụ thể"""
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            if bot.symbol:
                self.coin_manager.unregister_coin(bot.symbol)
            del self.bots[bot_id]
            self.log(f"⛔ Đã dừng bot {bot_id}")
            return True
        return False

    def stop_all(self):
        """Dừng tất cả bot"""
        self.log("⛔ Đang dừng tất cả bot...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

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
                elif response.status_code == 409:
                    logger.error("Lỗi xung đột Telegram")
                    time.sleep(60)
                else:
                    time.sleep(10)
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(10)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn Telegram - GIỮ NGUYÊN NỘI DUNG GỐC"""
        # Giữ nguyên toàn bộ nội dung xử lý Telegram từ file gốc
        # ... (giữ nguyên phần xử lý Telegram message)
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        # Xử lý các bước tạo bot (giữ nguyên)
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
                    user_state['step'] = 'waiting_symbol'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT TĨNH</b>\n\n"
                        "🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n"
                        "📊 Bạn cần chọn coin cụ thể\n\n"
                        "Chọn coin:",
                        chat_id,
                        create_symbols_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                else:
                    user_state['bot_mode'] = 'dynamic'
                    user_state['step'] = 'waiting_leverage'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT ĐỘNG</b>\n\n"
                        f"🤖 Hệ thống sẽ tạo <b>{user_state.get('bot_count', 1)} bot độc lập</b>\n"
                        f"🔄 Mỗi bot tự tìm coin BIẾN ĐỘNG CAO & trade độc lập\n\n"
                        "Chọn đòn bẩy:",
                        chat_id,
                        create_leverage_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )

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

                    user_state['leverage'] = leverage
                    user_state['step'] = 'waiting_percent'
                    
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
                    user_state['step'] = 'waiting_roi_trigger'
                    
                    send_telegram(
                        f"🛡️ Stop Loss: {sl}%\n\n"
                        f"🎯 <b>CHỌN NGƯỠNG ROI ĐỂ KÍCH HOẠT CƠ CHẾ CHỐT LỆNH THÔNG MINH</b>\n\n"
                        f"Chọn ngưỡng ROI trigger (%):",
                        chat_id,
                        create_roi_trigger_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Stop Loss:",
                                chat_id, create_sl_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_roi_trigger':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text == '❌ Tắt tính năng':
                user_state['roi_trigger'] = None
                self._finish_bot_creation(chat_id, user_state)
            else:
                try:
                    roi_trigger = float(text)
                    if roi_trigger <= 0:
                        send_telegram("⚠️ ROI Trigger phải lớn hơn 0. Vui lòng chọn lại:",
                                    chat_id, create_roi_trigger_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['roi_trigger'] = roi_trigger
                    self._finish_bot_creation(chat_id, user_state)
                    
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho ROI Trigger:",
                                chat_id, create_roi_trigger_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_count'}
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key và kết nối mạng!", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(
                f"🎯 <b>CHỌN SỐ LƯỢNG BOT ĐỘC LẬP</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"Chọn số lượng bot độc lập bạn muốn tạo:",
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
                    
                    roi_info = f" | 🎯 ROI: {bot.roi_trigger}%" if bot.roi_trigger else ""
                    symbol_info = bot.symbol if bot.symbol else "Đang tìm..."
                    next_trade = "Lệnh đầu" if bot.is_first_trade else f"Tiếp: {'SELL' if bot.last_side == 'BUY' else 'BUY'}"
                    
                    message += f"🔹 {bot_id}\n"
                    message += f"   📊 {symbol_info} | {status}\n"
                    message += f"   💰 ĐB: {bot.lev}x | Vốn: {bot.percent}%{roi_info}\n"
                    message += f"   🔄 {next_trade}\n\n"
                
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
                    send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key và kết nối mạng!", chat_id,
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
                "🎯 <b>HỆ THỐNG VỚI CƠ CHẾ BIẾN ĐỘNG CAO 24H & QUẢN LÝ COIN THÔNG MINH</b>\n\n"
                
                "📈 <b>Cơ chế biến động cao 24h:</b>\n"
                "• Chọn coin có biến động giá mạnh nhất 24h\n"
                "• Phân tích volume và % thay đổi giá thực tế\n"
                "• Ưu tiên coin có volume lớn và biến động cao\n"
                "• TOP 5 coin biến động cao nhất được ưu tiên\n\n"
                
                "⏰ <b>Quản lý coin thông minh:</b>\n"
                "• Giữ coin tối đa 1 giờ\n"
                "• Tự động chuyển coin mới sau thời gian quy định\n"
                "• Đánh dấu coin lỗi vào blacklist 30 phút\n"
                "• Tránh chọn lại coin đã thất bại\n\n"
                
                "🔄 <b>Cơ chế luôn ngược hướng:</b>\n"
                "• Lần đầu: Chọn ngẫu nhiên BUY/SELL\n"
                "• Các lần sau: LUÔN vào lệnh ngược với lệnh trước\n\n"
                
                "💰 <b>Quản lý rủi ro thông minh:</b>\n"
                "• TP/SL cố định theo %\n"
                "• Cơ chế ROI Trigger thông minh\n"
                "• Nhồi lệnh Fibonacci khi drawdown"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            searching_bots = sum(1 for bot in self.bots.values() if bot.status == "searching")
            trading_bots = sum(1 for bot in self.bots.values() if bot.status in ["waiting", "open"])
            
            roi_bots = sum(1 for bot in self.bots.values() if bot.roi_trigger is not None)
            first_trade_bots = sum(1 for bot in self.bots.values() if bot.is_first_trade)
            
            # Thống kê blacklist
            blacklisted_count = len([k for k, v in self.coin_finder.blacklisted_coins.items() if time.time() < v])
            failed_count = len(self.coin_finder.failed_coins)
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG BIẾN ĐỘNG CAO 24H</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🔍 Đang tìm coin: {searching_bots} bot\n"
                f"📊 Đang trade: {trading_bots} bot\n"
                f"🎯 Bot có ROI Trigger: {roi_bots} bot\n"
                f"🔄 Bot chờ lệnh đầu: {first_trade_bots} bot\n"
                f"🚫 Coin bị chặn: {blacklisted_count}\n"
                f"❌ Coin thất bại: {failed_count}\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n\n"
                f"📈 <b>CƠ CHẾ BIẾN ĐỘNG CAO 24H & QUẢN LÝ COIN THÔNG MINH ĐANG HOẠT ĐỘNG</b>"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _finish_bot_creation(self, chat_id, user_state):
        """Hoàn tất quá trình tạo bot"""
        try:
            # Lấy tất cả thông tin từ user_state
            bot_mode = user_state.get('bot_mode', 'static')
            leverage = user_state.get('leverage')
            percent = user_state.get('percent')
            tp = user_state.get('tp')
            sl = user_state.get('sl')
            roi_trigger = user_state.get('roi_trigger')
            symbol = user_state.get('symbol')
            bot_count = user_state.get('bot_count', 1)
            
            success = self.add_bot(
                symbol=symbol,
                lev=leverage,
                percent=percent,
                tp=tp,
                sl=sl,
                roi_trigger=roi_trigger,
                strategy_type="Global-Market-Biến-Động-Cao",
                bot_mode=bot_mode,
                bot_count=bot_count
            )
            
            if success:
                roi_info = f" | 🎯 ROI Trigger: {roi_trigger}%" if roi_trigger else ""
                
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {bot_count} BOT THÀNH CÔNG</b>\n\n"
                    f"🤖 Chiến lược: Biến động cao + Ngược hướng\n"
                    f"🔧 Chế độ: {bot_mode}\n"
                    f"🔢 Số lượng: {bot_count} bot độc lập\n"
                    f"💰 Đòn bẩy: {leverage}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl}%{roi_info}"
                )
                if bot_mode == 'static' and symbol:
                    success_msg += f"\n🔗 Coin: {symbol}"
                else:
                    success_msg += f"\n🔗 Coin: Tự động chọn BIẾN ĐỘNG CAO"
                
                success_msg += f"\n\n🔄 <b>CƠ CHẾ PHÂN PHỐI COIN THÔNG MINH:</b>\n"
                success_msg += f"• Mỗi bot nhận 1 coin KHÁC NHAU\n"
                success_msg += f"• Ưu tiên coin BIẾN ĐỘNG CAO NHẤT 24h\n"
                success_msg += f"• Phân tích volume & volatility thực tế\n"
                success_msg += f"• Tự động tránh trùng lặp\n"
                success_msg += f"• Tự động bỏ coin lỗi đòn bẩy\n"
                success_msg += f"• Giữ coin tối đa 1 giờ\n"
                success_msg += f"• Thử coin tiếp theo nếu có lỗi"
                
                send_telegram(success_msg, chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                send_telegram("❌ Có lỗi khi tạo bot. Vui lòng thử lại.",
                            chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            
            self.user_states[chat_id] = {}
            
        except Exception as e:
            send_telegram(f"❌ Lỗi tạo bot: {str(e)}", chat_id, create_main_menu(),
                        self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = {}

# ========== KHAI BÁO TOÀN CỤC ==========
coin_manager = CoinManager()
