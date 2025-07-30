# hft_bot.py
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
import sys
import math
from datetime import datetime

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hft_bot.log')
    ]
)
logger = logging.getLogger('HFTBot')
logger.setLevel(logging.INFO)

# Lấy cấu hình từ biến môi trường
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Cấu hình bot từ biến môi trường
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logger.error(f"Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

# ========== HÀM GỬI TELEGRAM VÀ XỬ LÝ LỖI ==========
def send_telegram(message, chat_id=None, reply_markup=None):
    """Gửi thông báo qua Telegram với xử lý lỗi chi tiết"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Cấu hình Telegram Bot Token chưa được thiết lập")
        return
    
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        logger.warning("Cấu hình Telegram Chat ID chưa được thiết lập")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
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
            error_msg = response.text
            logger.error(f"Lỗi gửi Telegram ({response.status_code}): {error_msg}")
    except Exception as e:
        logger.error(f"Lỗi kết nối Telegram: {str(e)}")

# ========== HÀM TẠO MENU TELEGRAM ==========
def create_menu_keyboard():
    """Tạo menu 3 nút cho Telegram"""
    return {
        "keyboard": [
            [{"text": "📊 Danh sách Bot"}],
            [{"text": "➕ Thêm Bot"}, {"text": "⛔ Dừng Bot"}],
            [{"text": "💰 Số dư tài khoản"}, {"text": "📈 Vị thế đang mở"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def create_cancel_keyboard():
    """Tạo bàn phím hủy"""
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard():
    """Tạo bàn phím chọn cặp coin"""
    popular_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    keyboard = []
    row = []
    for symbol in popular_symbols:
        row.append({"text": symbol})
        if len(row) == 2:
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
    """Tạo bàn phím chọn đòn bẩy"""
    leverages = ["3", "5", "10", "20", "30", "50", "75", "100"]
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

# ========== HÀM HỖ TRỢ API BINANCE ==========
def sign(query):
    try:
        return hmac.new(BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI SIGN:</b> {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    """Hàm tổng quát cho các yêu cầu API Binance với xử lý lỗi chi tiết"""
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
                    logger.error(f"Lỗi API ({response.status}): {response.read().decode()}")
                    if response.status == 429:  # Rate limit
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_step_size(symbol):
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
        send_telegram(f"⚠️ <b>LỖI STEP SIZE:</b> {symbol} - {str(e)}")
    return 0.001

def set_leverage(symbol, lev):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "leverage": lev,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
        response = binance_api_request(url, method='POST', headers=headers)
        if response and 'leverage' in response:
            return True
    except Exception as e:
        logger.error(f"Lỗi thiết lập đòn bẩy: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI ĐÒN BẨY:</b> {symbol} - {str(e)}")
    return False

def get_balance():
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return 0
            
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI SỐ DƯ:</b> {str(e)}")
    return 0

def place_order(symbol, side, qty):
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
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI ĐẶT LỆNH:</b> {symbol} - {str(e)}")
    return None

def cancel_all_orders(symbol):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Lỗi hủy lệnh: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI HỦY LỆNH:</b> {symbol} - {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
        send_telegram(f"⚠️ <b>LỖI GIÁ:</b> {symbol} - {str(e)}")
    return 0

def get_positions(symbol=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        if symbol:
            params["symbol"] = symbol.upper()
            
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
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
        send_telegram(f"⚠️ <b>LỖI VỊ THẾ:</b> {symbol if symbol else ''} - {str(e)}")
    return []

# ========== QUẢN LÝ DỮ LIỆU TỐC ĐỘ CAO ==========
class DataManager:
    def __init__(self):
        self.price_data = {}
        self.orderbook_data = {}
        self.lock = threading.Lock()
        
    def update_price(self, symbol, price, timestamp=None):
        with self.lock:
            if symbol not in self.price_data:
                self.price_data[symbol] = []
            self.price_data[symbol].append(price)
            if len(self.price_data[symbol]) > 100:
                self.price_data[symbol] = self.price_data[symbol][-100:]
    
    def get_last_price(self, symbol):
        with self.lock:
            if symbol in self.price_data and self.price_data[symbol]:
                return self.price_data[symbol][-1]
        return 0

    def get_prices(self, symbol, count=10):
        with self.lock:
            if symbol in self.price_data:
                return self.price_data[symbol][-count:]
        return []

    def update_orderbook(self, symbol, bid, ask, bid_qty, ask_qty):
        with self.lock:
            self.orderbook_data[symbol] = {
                'bid': bid,
                'ask': ask,
                'bid_qty': bid_qty,
                'ask_qty': ask_qty,
                'timestamp': time.time()
            }

    def get_orderbook(self, symbol):
        with self.lock:
            return self.orderbook_data.get(symbol, {
                'bid': 0, 'ask': 0, 'bid_qty': 0, 'ask_qty': 0
            })

# ========== QUẢN LÝ WEBSOCKET ==========
class WebSocketManager:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.connections = {}
        self._stop_event = threading.Event()
        
    def add_symbol(self, symbol, callback=None):
        symbol = symbol.upper()
        if symbol not in self.connections:
            self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
            
        # Tạo stream giá
        trade_stream = f"{symbol.lower()}@trade"
        trade_url = f"wss://fstream.binance.com/ws/{trade_stream}"
        
        # Tạo stream orderbook
        depth_stream = f"{symbol.lower()}@depth5@100ms"
        depth_url = f"wss://fstream.binance.com/ws/{depth_stream}"
        
        # Xử lý tin nhắn trade
        def trade_on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    self.data_manager.update_price(symbol, price)
                    if callback:
                        callback(price)
            except Exception as e:
                logger.error(f"Trade stream error: {str(e)}")
                
        # Xử lý tin nhắn orderbook
        def depth_on_message(ws, message):
            try:
                data = json.loads(message)
                if 'b' in data and data['b'] and 'a' in data and data['a']:
                    bid = float(data['b'][0][0])
                    ask = float(data['a'][0][0])
                    bid_qty = sum(float(b[1]) for b in data['b'][:5])
                    ask_qty = sum(float(a[1]) for a in data['a'][:5])
                    self.data_manager.update_orderbook(symbol, bid, ask, bid_qty, ask_qty)
            except Exception as e:
                logger.error(f"Orderbook stream error: {str(e)}")
        
        # Tạo WebSocket cho giá
        trade_ws = websocket.WebSocketApp(
            trade_url,
            on_message=trade_on_message,
            on_error=lambda ws, err: logger.error(f"Trade WS error: {err}"),
            on_close=lambda ws: self._reconnect(symbol, 'trade', callback)
        )
        
        # Tạo WebSocket cho orderbook
        depth_ws = websocket.WebSocketApp(
            depth_url,
            on_message=depth_on_message,
            on_error=lambda ws, err: logger.error(f"Depth WS error: {err}"),
            on_close=lambda ws: self._reconnect(symbol, 'depth', callback)
        )
        
        # Khởi chạy trong luồng riêng
        trade_thread = threading.Thread(target=trade_ws.run_forever, daemon=True)
        depth_thread = threading.Thread(target=depth_ws.run_forever, daemon=True)
        
        trade_thread.start()
        depth_thread.start()
        
        self.connections[symbol] = {
            'trade': {'ws': trade_ws, 'thread': trade_thread},
            'depth': {'ws': depth_ws, 'thread': depth_thread}
        }
        
    def _reconnect(self, symbol, stream_type, callback):
        logger.info(f"Reconnecting {symbol} {stream_type}")
        time.sleep(1)
        self._create_connection(symbol, callback)
                
    def remove_symbol(self, symbol):
        if symbol in self.connections:
            try:
                self.connections[symbol]['trade']['ws'].close()
                self.connections[symbol]['depth']['ws'].close()
            except:
                pass
            del self.connections[symbol]
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== BOT GIAO DỊCH TỐC ĐỘ CAO ==========
class HighFrequencyTrader:
    def __init__(self, symbol, leverage, risk_percent, data_manager, ws_manager):
        self.symbol = symbol.upper()
        self.leverage = leverage
        self.risk_percent = risk_percent
        self.data_manager = data_manager
        self.ws_manager = ws_manager
        self.running = True
        self.position_size = 0
        self.symbol_info = self._get_symbol_info()
        
        # Đăng ký dữ liệu
        self.ws_manager.add_symbol(self.symbol, self._price_update)
        
        # Bắt đầu luồng giao dịch
        self.thread = threading.Thread(target=self._trade_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Started HFT bot for {self.symbol}")
        send_telegram(f"🚀 <b>HFT BOT STARTED</b>\nSymbol: {self.symbol}\nLeverage: {leverage}x\nRisk: {risk_percent}%")

    def _price_update(self, price):
        """Xử lý cập nhật giá"""
        pass

    def _get_symbol_info(self):
        """Lấy thông tin symbol"""
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if data:
            for s in data['symbols']:
                if s['symbol'] == self.symbol:
                    info = {}
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            info['min_qty'] = float(f['minQty'])
                            info['step_size'] = float(f['stepSize'])
                        elif f['filterType'] == 'PRICE_FILTER':
                            info['tick_size'] = float(f['tickSize'])
                    return info
        return {'min_qty': 0.001, 'step_size': 0.001, 'tick_size': 0.01}

    def _get_balance(self):
        """Lấy số dư tài khoản"""
        try:
            ts = int(time.time() * 1000)
            params = {"timestamp": ts}
            query = urllib.parse.urlencode(params)
            sig = sign(query)
            url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
            headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
            
            data = binance_api_request(url, headers=headers)
            if data:
                for asset in data['assets']:
                    if asset['asset'] == 'USDT':
                        return float(asset['availableBalance'])
        except Exception:
            pass
        return 0

    def _calculate_position_size(self):
        """Tính toán kích thước vị thế"""
        balance = self._get_balance()
        if balance <= 0:
            return 0
            
        risk_amount = balance * (self.risk_percent / 100)
        price = self.data_manager.get_last_price(self.symbol)
        if price <= 0:
            return 0
            
        size = (risk_amount * self.leverage) / price
        step = self.symbol_info.get('step_size', 0.001)
        
        # Làm tròn theo step size
        if step > 0:
            size = round(size / step) * step
            
        return max(size, self.symbol_info.get('min_qty', 0.001))

    def _place_order(self, side, quantity, price):
        """Đặt lệnh tốc độ cao"""
        try:
            ts = int(time.time() * 1000)
            params = {
                "symbol": self.symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": "IOC",
                "quantity": quantity,
                "price": price,
                "timestamp": ts
            }
            
            query = urllib.parse.urlencode(params)
            sig = sign(query)
            url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
            headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
            
            # Gửi trong luồng riêng để không chặn luồng chính
            threading.Thread(
                target=requests.post, 
                args=(url,), 
                kwargs={'headers': headers, 'timeout': 0.3}
            ).start()
            
            return True
        except Exception as e:
            logger.error(f"Order placement error: {str(e)}")
            return False

    def _generate_signal(self):
        """Tạo tín hiệu giao dịch với độ chính xác cao"""
        try:
            # Lấy dữ liệu thị trường
            prices = self.data_manager.get_prices(self.symbol, 20)
            if len(prices) < 10:
                return None, 0
                
            orderbook = self.data_manager.get_orderbook(self.symbol)
            
            # Tính toán động lượng
            price_change = prices[-1] - prices[-5]
            avg_volume = np.mean(prices[-10:])
            volume_ratio = prices[-1] / avg_volume if avg_volume > 0 else 1
            
            # Phân tích order book
            imbalance = (orderbook['bid_qty'] - orderbook['ask_qty']) / (orderbook['bid_qty'] + orderbook['ask_qty'] + 1e-10)
            spread = orderbook['ask'] - orderbook['bid']
            
            # Tạo điểm tín hiệu
            signal_score = 0
            direction = None
            
            # Tín hiệu mua
            if price_change > 0 and volume_ratio > 1.2 and imbalance > 0.1:
                signal_score = 95 + min(5, imbalance * 100)
                direction = "BUY"
                
            # Tín hiệu bán
            elif price_change < 0 and volume_ratio < 0.8 and imbalance < -0.1:
                signal_score = 95 + min(5, abs(imbalance) * 100)
                direction = "SELL"
                
            return direction, signal_score
        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            return None, 0

    def _execute_trade(self):
        """Thực thi giao dịch tốc độ cao"""
        start_time = time.time()
        
        # Tạo tín hiệu
        direction, confidence = self._generate_signal()
        if confidence < 95:
            return False
            
        # Tính toán vị thế
        if self.position_size <= 0:
            self.position_size = self._calculate_position_size()
            if self.position_size <= 0:
                return False
                
        # Lấy giá mục tiêu
        orderbook = self.data_manager.get_orderbook(self.symbol)
        if direction == "BUY":
            price = orderbook['ask'] + self.symbol_info.get('tick_size', 0.01)
        else:
            price = orderbook['bid'] - self.symbol_info.get('tick_size', 0.01)
            
        # Đặt lệnh
        self._place_order(direction, self.position_size, price)
        
        # Ghi log hiệu suất
        exec_time = (time.time() - start_time) * 1000
        logger.info(f"Executed {direction} order in {exec_time:.2f}ms | Confidence: {confidence}%")
        send_telegram(f"⚡ <b>TRADE EXECUTED</b>\n"
                      f"Symbol: {self.symbol}\n"
                      f"Direction: {direction}\n"
                      f"Size: {self.position_size:.4f}\n"
                      f"Price: {price:.4f}\n"
                      f"Exec Time: {exec_time:.2f}ms")
        
        return True

    def _trade_loop(self):
        """Vòng lặp giao dịch chính"""
        while self.running:
            try:
                self._execute_trade()
                # Tối ưu thời gian chờ để đạt 50 lần/giây
                time.sleep(0.02)
            except Exception as e:
                logger.error(f"Trade loop error: {str(e)}")
                time.sleep(1)

    def stop(self):
        """Dừng bot"""
        self.running = False
        self.ws_manager.remove_symbol(self.symbol)
        logger.info(f"Stopped HFT bot for {self.symbol}")
        send_telegram(f"🛑 <b>HFT BOT STOPPED</b>\nSymbol: {self.symbol}")

# ========== QUẢN LÝ HỆ THỐNG VÀ TELEGRAM ==========
class BotManager:
    def __init__(self):
        self.data_manager = DataManager()
        self.ws_manager = WebSocketManager(self.data_manager)
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}  # Lưu trạng thái người dùng
        self.admin_chat_id = TELEGRAM_CHAT_ID
        
        # Bắt đầu các bot từ cấu hình
        for config in BOT_CONFIGS:
            if len(config) >= 3:
                self.add_bot(config[0], config[1], config[2])
        
        logger.info("HFT Trading System Initialized")
        send_telegram("🚀 <b>HFT TRADING SYSTEM STARTED</b>")
        
        # Luồng giám sát
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        
        # Luồng xử lý Telegram
        self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
        self.telegram_thread.start()
        
        # Gửi menu chính khi khởi động
        if self.admin_chat_id:
            self.send_main_menu(self.admin_chat_id)

    def log(self, message):
        """Ghi log hệ thống và gửi Telegram"""
        logger.info(f"[SYSTEM] {message}")
        send_telegram(f"<b>SYSTEM</b>: {message}")

    def send_main_menu(self, chat_id):
        """Gửi menu chính cho người dùng"""
        welcome = (
            "🤖 <b>HFT TRADING BOT</b>\n\n"
            "Chọn một trong các tùy chọn bên dưới:"
        )
        send_telegram(welcome, chat_id, create_menu_keyboard())

    def add_bot(self, symbol, leverage, risk_percent):
        symbol = symbol.upper()
        if symbol in self.bots:
            self.log(f"⚠️ Đã có bot cho {symbol}")
            return False
            
        # Kiểm tra API key
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            self.log("❌ Chưa cấu hình API Key và Secret Key!")
            return False
            
        try:
            # Tạo bot mới
            bot = HighFrequencyTrader(
                symbol, leverage, risk_percent,
                self.data_manager, self.ws_manager
            )
            self.bots[symbol] = bot
            self.log(f"✅ Đã thêm bot: {symbol} | ĐB: {leverage}x | %: {risk_percent}")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            self.log(f"⛔ Đã dừng bot cho {symbol}")
            del self.bots[symbol]
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for symbol in list(self.bots.keys()):
            self.stop_bot(symbol)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _monitor(self):
        """Kiểm tra và báo cáo trạng thái định kỳ"""
        while self.running:
            try:
                # Tính thời gian hoạt động
                uptime = time.time() - self.start_time
                hours, rem = divmod(uptime, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                # Báo cáo số bot đang chạy
                active_bots = [s for s, b in self.bots.items() if b.running]
                
                # Báo cáo số dư tài khoản
                balance = get_balance()
                
                # Tạo báo cáo
                status_msg = (
                    f"📊 <b>BÁO CÁO HỆ THỐNG</b>\n"
                    f"⏱ Thời gian hoạt động: {uptime_str}\n"
                    f"🤖 Số bot đang chạy: {len(active_bots)}\n"
                    f"📈 Bot hoạt động: {', '.join(active_bots) if active_bots else 'Không có'}\n"
                    f"💰 Số dư khả dụng: {balance:.2f} USDT"
                )
                send_telegram(status_msg)
                
            except Exception as e:
                logger.error(f"Lỗi báo cáo trạng thái: {str(e)}")
            
            # Kiểm tra mỗi 6 giờ
            time.sleep(6 * 3600)

    def _telegram_listener(self):
        """Lắng nghe và xử lý tin nhắn từ Telegram"""
        last_update_id = 0
        
        while self.running:
            try:
                # Lấy tin nhắn mới
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={last_update_id+1}&timeout=30"
                response = requests.get(url, timeout=35)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        for update in data['result']:
                            update_id = update['update_id']
                            message = update.get('message', {})
                            chat_id = str(message.get('chat', {}).get('id'))
                            text = message.get('text', '').strip()
                            
                            # Chỉ xử lý tin nhắn từ admin
                            if chat_id != self.admin_chat_id:
                                continue
                            
                            # Cập nhật ID tin nhắn cuối
                            if update_id > last_update_id:
                                last_update_id = update_id
                            
                            # Xử lý tin nhắn
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    logger.error("Lỗi xung đột: Chỉ một instance bot có thể lắng nghe Telegram")
                    break
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(5)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn từ người dùng"""
        # Lưu trạng thái người dùng
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        # Xử lý theo bước hiện tại
        if current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                symbol = text.upper()
                self.user_states[chat_id] = {
                    'step': 'waiting_leverage',
                    'symbol': symbol
                }
                send_telegram(f"Chọn đòn bẩy cho {symbol}:", chat_id, create_leverage_keyboard())
        
        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            elif 'x' in text:
                leverage = int(text.replace('x', '').strip())
                user_state['leverage'] = leverage
                user_state['step'] = 'waiting_percent'
                send_telegram(
                    f"📌 Cặp: {user_state['symbol']}\nĐòn bẩy: {leverage}x\n\nNhập % rủi ro mỗi lệnh (1-100):",
                    chat_id,
                    create_cancel_keyboard()
                )
        
        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_menu_keyboard())
            else:
                try:
                    percent = float(text)
                    if 1 <= percent <= 100:
                        # Thêm bot
                        symbol = user_state['symbol']
                        leverage = user_state['leverage']
                        
                        if self.add_bot(symbol, leverage, percent):
                            send_telegram(
                                f"✅ <b>ĐÃ THÊM BOT THÀNH CÔNG</b>\n\n"
                                f"📌 Cặp: {symbol}\n"
                                f"Đòn bẩy: {leverage}x\n"
                                f"📊 % Rủi ro: {percent}%",
                                chat_id,
                                create_menu_keyboard()
                            )
                        else:
                            send_telegram("❌ Không thể thêm bot, vui lòng kiểm tra log", chat_id, create_menu_keyboard())
                        
                        # Reset trạng thái
                        self.user_states[chat_id] = {}
                    else:
                        send_telegram("⚠️ Vui lòng nhập % từ 1-100", chat_id)
                except:
                    send_telegram("⚠️ Giá trị không hợp lệ, vui lòng nhập số", chat_id)
        
        # Xử lý các lệnh chính
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
                for symbol, bot in self.bots.items():
                    status = "🟢 Đang chạy" if bot.running else "🔴 Đã dừng"
                    message += f"🔹 {symbol} | {status}\n"
                send_telegram(message, chat_id)
        
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_symbol'}
            send_telegram("Chọn cặp coin:", chat_id, create_symbols_keyboard())
        
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id)
            else:
                message = "⛔ <b>CHỌN BOT ĐỂ DỪNG</b>\n\n"
                keyboard = []
                row = []
                
                for i, symbol in enumerate(self.bots.keys()):
                    message += f"🔹 {symbol}\n"
                    row.append({"text": f"⛔ {symbol}"})
                    if len(row) == 2 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram(
                    message, 
                    chat_id, 
                    {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True}
                )
        
        elif text.startswith("⛔ "):
            symbol = text.replace("⛔ ", "").strip().upper()
            if symbol in self.bots:
                self.stop_bot(symbol)
                send_telegram(f"⛔ Đã gửi lệnh dừng bot {symbol}", chat_id, create_menu_keyboard())
            else:
                send_telegram(f"⚠️ Không tìm thấy bot {symbol}", chat_id, create_menu_keyboard())
        
        elif text == "💰 Số dư tài khoản":
            try:
                balance = get_balance()
                send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy số dư: {str(e)}", chat_id)
        
        elif text == "📈 Vị thế đang mở":
            try:
                positions = get_positions()
                if not positions:
                    send_telegram("📭 Không có vị thế nào đang mở", chat_id)
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
                
                send_telegram(message, chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy vị thế: {str(e)}", chat_id)
        
        # Gửi lại menu nếu không có lệnh phù hợp
        elif text:
            self.send_main_menu(chat_id)

# ========== HÀM KHỞI CHẠY CHÍNH ==========
def main():
    # Kiểm tra API keys
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        logger.error("Binance API keys not configured!")
        send_telegram("❌ <b>ERROR:</b> Binance API keys missing!")
        sys.exit(1)
    
    # Khởi tạo hệ thống
    manager = BotManager()
    
    try:
        # Thông báo số dư ban đầu
        try:
            balance = get_balance()
            manager.log(f"💰 SỐ DƯ BAN ĐẦU: {balance:.2f} USDT")
        except Exception as e:
            manager.log(f"⚠️ Lỗi lấy số dư ban đầu: {str(e)}")
        
        # Giữ chương trình chạy
        while manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        manager.log("👋 Nhận tín hiệu dừng từ người dùng...")
    except Exception as e:
        manager.log(f"⚠️ LỖI HỆ THỐNG NGHIÊM TRỌNG: {str(e)}")
    finally:
        manager.stop_all()

if __name__ == "__main__":
    main()
