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
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Detailed logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_errors.log')
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get configuration from environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Get bot configuration from environment variables (JSON format)
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    logging.error(f"Error parsing BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY

# ========== TELEGRAM FUNCTIONS ==========
def send_telegram(message, chat_id=None, reply_markup=None):
    """Sends a message via Telegram with detailed error handling."""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram Bot Token is not configured.")
        return

    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        logger.warning("Telegram Chat ID is not configured.")
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
            logger.error(f"Telegram send error ({response.status_code}): {error_msg}")
    except Exception as e:
        logger.error(f"Telegram connection error: {str(e)}")

def create_menu_keyboard():
    """Creates a 3-button menu for Telegram."""
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
    """Creates a cancel keyboard."""
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard():
    """Creates a keyboard for selecting coin pairs."""
    popular_symbols = ["SUIUSDT", "DOGEUSDT", "1000PEPEUSDT", "TRUMPUSDT", "XRPUSDT", "ADAUSDT"]
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
    """Creates a keyboard for selecting leverage."""
    leverages = ["3", "8", "10", "20", "30", "50", "75", "100"]
    keyboard = []
    row = []
    for lev in leverages:
        row.append({"text": f" {lev}x"})
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

# ========== BINANCE API HELPER FUNCTIONS ==========
def sign(query):
    try:
        return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Error creating signature: {str(e)}")
        send_telegram(f"⚠️ <b>SIGN ERROR:</b> {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    """General function for Binance API requests with detailed error handling."""
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
                    logger.error(f"API Error ({response.status}): {response.read().decode()}")
                    if response.status == 429:  # Rate limit
                        time.sleep(2 ** attempt)  # Exponential backoff
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error ({e.code}): {e.reason}")
            if e.code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"API connection error: {str(e)}")
            time.sleep(1)

    logger.error(f"Failed to make API request after {max_retries} attempts")
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
        logger.error(f"Error getting step size: {str(e)}")
        send_telegram(f"⚠️ <b>STEP SIZE ERROR:</b> {symbol} - {str(e)}")
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
        headers = {'X-MBX-APIKEY': API_KEY}

        response = binance_api_request(url, method='POST', headers=headers)
        if response and 'leverage' in response:
            return True
    except Exception as e:
        logger.error(f"Error setting leverage: {str(e)}")
        send_telegram(f"⚠️ <b>LEVERAGE ERROR:</b> {symbol} - {str(e)}")
    return False

def get_balance():
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}

        data = binance_api_request(url, headers=headers)
        if not data:
            return 0

        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
    except Exception as e:
        logger.error(f"Error getting balance: {str(e)}")
        send_telegram(f"⚠️ <b>BALANCE ERROR:</b> {str(e)}")
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
        headers = {'X-MBX-APIKEY': API_KEY}

        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Error placing order: {str(e)}")
        send_telegram(f"⚠️ <b>ORDER ERROR:</b> {symbol} - {str(e)}")
    return None

def cancel_all_orders(symbol):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': API_KEY}

        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Error canceling orders: {str(e)}")
        send_telegram(f"⚠️ <b>CANCEL ORDER ERROR:</b> {symbol} - {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Error getting price: {str(e)}")
        send_telegram(f"⚠️ <b>PRICE ERROR:</b> {symbol} - {str(e)}")
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
        headers = {'X-MBX-APIKEY': API_KEY}

        positions = binance_api_request(url, headers=headers)
        if not positions:
            return []

        if symbol:
            for pos in positions:
                if pos['symbol'] == symbol.upper():
                    return [pos]

        return positions
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        send_telegram(f"⚠️ <b>POSITIONS ERROR:</b> {symbol if symbol else ''} - {str(e)}")
    return []

def get_klines(symbol, interval, limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    data = binance_api_request(url)
    if data:
        df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df
    return pd.DataFrame()

# ========== TECHNICAL INDICATORS ==========
def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series([None])

def calc_ema(series, period):
    try:
        return series.ewm(span=period, adjust=False).mean()
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return pd.Series([None])

def calc_atr(df, period=14):
    try:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series([None])

def calc_macd(series, fast_period=12, slow_period=26, signal_period=9):
    try:
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - signal
        return macd, signal, macd_hist
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return pd.Series([None]), pd.Series([None]), pd.Series([None])

def calc_ichimoku(df):
    try:
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_kijun_sen'] = (high_26 + low_26) / 2

        return df['ichimoku_tenkan_sen'], df['ichimoku_kijun_sen']
    except Exception as e:
        logger.error(f"Error calculating Ichimoku: {str(e)}")
        return pd.Series([None]), pd.Series([None])

def calc_adx(df, period=14):
    try:
        # Simplified ADX calculation for demonstration
        df['plus_di'] = df['high'].diff().rolling(period).mean()
        df['minus_di'] = df['low'].diff().rolling(period).mean()
        df['adx'] = (df['plus_di'] + df['minus_di']).abs() / 2
        return df['adx']
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        return pd.Series([None])

def add_technical_indicators(df):
    """Adds all technical indicators to the DataFrame."""
    if df.empty or len(df) < 50:
        return df

    df['RSI'] = calc_rsi(df['close'], 14)
    df['EMA9'] = calc_ema(df['close'], 9)
    df['EMA21'] = calc_ema(df['close'], 21)
    df['ATR'] = calc_atr(df, 14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calc_macd(df['close'])

    # Stochastic Oscillator
    df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Bollinger Bands
    df['bollinger_high'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bollinger_low'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()

    # Ichimoku Cloud
    df['ichimoku_tenkan_sen'], df['ichimoku_kijun_sen'] = calc_ichimoku(df)

    # ADX
    df['ADX'] = calc_adx(df)

    return df

# ========== NEW SIGNAL FUNCTIONS ==========
def get_raw_indicator_signals(df):
    """Calculates raw signals (+1/-1/0) for each indicator."""
    current_signals = {}
    
    # RSI: Tín hiệu mua khi quá bán (< 30), tín hiệu bán khi quá mua (> 70)
    rsi_value = df['RSI'].iloc[-1]
    if rsi_value < 20 or 60 < rsi_value < 80:
        current_signals["RSI"] = 1
    elif rsi_value > 80 or 20 < rsi_value < 40:
        current_signals["RSI"] = -1
    else:
        current_signals["RSI"] = 0

    # MACD: MACD line > signal line là tăng
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        current_signals["MACD"] = 1
    else:
        current_signals["MACD"] = -1

    # EMA Crossover: EMA9 > EMA21 là tăng
    if df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1]:
        current_signals["EMA_Crossover"] = 1
    else:
        current_signals["EMA_Crossover"] = -1

    # Volume Confirmation: Nến tăng + volume cao là tăng
    if df['close'].iloc[-1] > df['open'].iloc[-1] and df['volume'].iloc[-1] > df['volume'].rolling(window=20).mean().iloc[-1] * 1.5:
        current_signals["Volume_Confirmation"] = 1
    elif df['close'].iloc[-1] < df['open'].iloc[-1] and df['volume'].iloc[-1] > df['volume'].rolling(window=20).mean().iloc[-1] * 1.5:
        current_signals["Volume_Confirmation"] = -1
    else:
        current_signals["Volume_Confirmation"] = 0

    # Stochastic Oscillator: K line > D line và cả hai đều dưới 80 là tăng
    stoch_k_value = df['stoch_k'].iloc[-1]
    stoch_d_value = df['stoch_d'].iloc[-1]
    if stoch_k_value > stoch_d_value and stoch_k_value < 80:
        current_signals["Stochastic"] = 1
    else:
        current_signals["Stochastic"] = -1

    # Bollinger Bands: Giá dưới dải băng dưới là tăng, trên dải băng trên là giảm
    close_price = df['close'].iloc[-1]
    if close_price < df['bollinger_low'].iloc[-1]:
        current_signals["BollingerBands"] = 1
    elif close_price > df['bollinger_high'].iloc[-1]:
        current_signals["BollingerBands"] = -1
    else:
        current_signals["BollingerBands"] = 0

    # Ichimoku: Tenkan Sen > Kijun Sen là tín hiệu tăng
    if df['ichimoku_tenkan_sen'].iloc[-1] > df['ichimoku_kijun_sen'].iloc[-1]:
        current_signals["Ichimoku"] = 1
    else:
        current_signals["Ichimoku"] = -1

    # ADX: ADX > 25 và (+DI > -DI) là tín hiệu tăng mạnh
    adx_value = df['ADX'].iloc[-1]
    if adx_value > 25 and df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1]:
        current_signals["ADX"] = 1
    elif adx_value > 25 and df['minus_di'].iloc[-1] > df['plus_di'].iloc[-1]:
        current_signals["ADX"] = -1
    else:
        current_signals["ADX"] = 0
        
    return current_signals

def update_weights_and_stats(current_signals, price_change_percent, indicator_weights, indicator_stats, is_initial_training):
    """
    Dynamically adjusts indicator weights based on their performance on a single candle.
    This function is used for both initial training and real-time learning.
    """
    
    is_price_up = price_change_percent > 0
    is_price_down = price_change_percent < 0
    
    # Giai đoạn huấn luyện ban đầu (điểm số)
    if is_initial_training:
        for indicator, signal in current_signals.items():
            if (signal == 1 and is_price_up) or (signal == -1 and is_price_down):
                indicator_stats[indicator] += 1
            elif (signal == 1 and is_price_down) or (signal == -1 and is_price_up):
                indicator_stats[indicator] -= 1
    
    # Giai đoạn hoạt động thực tế (tỷ lệ phần trăm)
    else:
        adjustment_rate = 0.005 # Điều chỉnh 0.5% mỗi nến
        for indicator, signal in current_signals.items():
            if (signal == 1 and is_price_up) or (signal == -1 and is_price_down):
                indicator_weights[indicator] *= (1 + adjustment_rate)
            elif (signal == 1 and is_price_down) or (signal == -1 and is_price_up):
                indicator_weights[indicator] *= (1 - adjustment_rate)

        # Chuẩn hóa lại các trọng số
        total_weight = sum(indicator_weights.values())
        if total_weight > 0:
            for indicator in indicator_weights:
                indicator_weights[indicator] = (indicator_weights[indicator] / total_weight) * 100
    
    logging.info("--- New weights and stats ---")
    if is_initial_training:
        for indicator, score in indicator_stats.items():
            logging.info(f"📊 {indicator}: Score {score}")
    else:
        for indicator, weight in indicator_weights.items():
            logging.info(f"📊 {indicator}: Weight {weight:.2f}%")
        
    return indicator_weights, indicator_stats

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
                logger.error(f"WebSocket message processing error {symbol}: {str(e)}")

        def on_error(ws, error):
            logger.error(f"WebSocket error {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(300)
                self._reconnect(symbol, callback)

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(300)
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
        logger.info(f"WebSocket started for {symbol}")

    def _reconnect(self, symbol, callback):
        logger.info(f"Reconnecting WebSocket for {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)

    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket removed for {symbol}")

    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== MAIN BOT CLASS ==========
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, initial_weights=None):
        self.symbol = symbol.upper()
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.ws_manager = ws_manager
        
        # ========== KHỞI TẠO TRỌNG SỐ TỪ HUẤN LUYỆN BAN ĐẦU HOẶC THOÁT NẾU KHÔNG CÓ ==========
        if initial_weights and isinstance(initial_weights, dict) and self._are_weights_valid(initial_weights):
            self.indicator_weights = initial_weights
        else:
            self.log("❌ Không tìm thấy trọng số huấn luyện hoặc trọng số không hợp lệ. Bot không thể khởi chạy.")
            self._stop = True
            return
            
        self.indicator_stats = {k: 0 for k in self.indicator_weights.keys()}
        # ========================================================

        self.check_position_status()
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []

        self._stop = False
        self.signal_threshold = 50.0  # Ngưỡng vào lệnh
        self.position_open = False
        self.last_trade_time = 0
        self.position_check_interval = 60
        self.last_position_check = 0
        self.last_error_log_time = 0
        self.last_close_time = 0
        self.cooldown_period = 3
        self.max_position_attempts = 3
        self.position_attempt_count = 0

        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot started for {self.symbol}")

    def _are_weights_valid(self, weights):
        """Kiểm tra tính hợp lệ của trọng số"""
        if not isinstance(weights, dict):
            return False
        if len(weights) == 0:
            return False
        # Kiểm tra xem có ít nhất một trọng số dương
        has_positive_weight = any(weight > 0 for weight in weights.values())
        return has_positive_weight

    def log(self, message):
        logger.info(f"[{self.symbol}] {message}")
        send_telegram(f"<b>{self.symbol}</b>: {message}")

    def _handle_price_update(self, price):
        if self._stop:
            return
        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def get_signal(self, df):
        try:
            current_signals = get_raw_indicator_signals(df)
            total_score = sum(current_signals.get(k, 0) * self.indicator_weights.get(k, 0) for k in current_signals)
            
            if total_score > self.signal_threshold:
                return "BUY", current_signals, total_score
            elif total_score < -self.signal_threshold:
                return "SELL", current_signals, total_score
            return None, current_signals, total_score
        except Exception as e:
            self.log(f"get_signal error: {str(e)}")
            return None, None, None
            
    def _run(self):
        last_candle_close_time = 0
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                    
                df = get_klines(self.symbol, "1m", 300)
                if df.empty or len(df) < 50:
                    time.sleep(1)
                    continue

                df = add_technical_indicators(df)
                
                if df.iloc[-1].isnull().any():
                    time.sleep(1)
                    continue
                    
                if df['close_time'].iloc[-1] != last_candle_close_time:
                    last_candle_close_time = df['close_time'].iloc[-1]
                    
                    price_change_percent = ((df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]) * 100
                    current_signals = get_raw_indicator_signals(df)
                    
                    self.indicator_weights, _ = update_weights_and_stats(
                        current_signals, price_change_percent, self.indicator_weights, self.indicator_stats, False
                    )
                    
                signal, current_signals, total_score = self.get_signal(df)
                
                if self.position_open and self.side == "BUY" and signal == "SELL":
                    self.close_position(f"🔄 Đảo chiều: Tín hiệu SELL mới được tạo.")
                    self.open_position("SELL", current_signals)
                    
                elif self.position_open and self.side == "SELL" and signal == "BUY":
                    self.close_position(f"🔄 Đảo chiều: Tín hiệu BUY mới được tạo.")
                    self.open_position("BUY", current_signals)
                
                elif not self.position_open and self.status == "waiting":
                    if current_time - self.last_close_time < self.cooldown_period:
                        time.sleep(1)
                        continue
                    if signal and current_time - self.last_trade_time > 3:
                        self.open_position(signal, current_signals)
                        self.last_trade_time = current_time
                        
                if self.position_open and self.status == "open":
                    self.check_tp_sl()
                
                time.sleep(1)
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"System error: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        self.ws_manager.remove_symbol(self.symbol)
        try:
            cancel_all_orders(self.symbol)
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Order cancellation error: {str(e)}")
                self.last_error_log_time = time.time()
        self.log(f"🔴 Bot stopped for {self.symbol}")

    def check_position_status(self):
        try:
            positions = get_positions(self.symbol)
            if not positions or len(positions) == 0:
                self.position_open = False
                self.status = "waiting"
                self.side = ""
                self.qty = 0
                self.entry = 0
                return
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    if abs(position_amt) > 0:
                        self.position_open = True
                        self.status = "open"
                        self.side = "BUY" if position_amt > 0 else "SELL"
                        self.qty = position_amt
                        self.entry = float(pos.get('entryPrice', 0))
                        return
            self.position_open = False
            self.status = "waiting"
            self.side = ""
            self.qty = 0
            self.entry = 0
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"Position check error: {str(e)}")
                self.last_error_log_time = time.time()

    def check_tp_sl(self):
        if not self.position_open or not self.entry or not self.qty:
            return
        try:
            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
            if current_price <= 0:
                return
            profit = (current_price - self.entry) * self.qty if self.side == "BUY" else (self.entry - current_price) * abs(self.qty)
            invested = self.entry * abs(self.qty) / self.lev
            if invested <= 0:
                return
            roi = (profit / invested) * 100
            if roi >= self.tp:
                self.close_position(f"✅ TP hit at {self.tp}% (ROI: {roi:.2f}%)")
            elif self.sl > 0 and roi <= -self.sl:
                self.close_position(f"❌ SL hit at -{self.sl}% (ROI: {roi:.2f}%)")
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"TP/SL check error: {str(e)}")
                self.last_error_log_time = time.time()

    def open_position(self, side, current_signals):
        try:
            if self.position_open:
                self.log(f"⚠️ Position already open. Cannot open new position.")
                return
            if self.position_attempt_count >= self.max_position_attempts:
                self.log(f"⚠️ Max position attempts reached. Cooling down.")
                time.sleep(60)
                self.position_attempt_count = 0
                return
            if not set_leverage(self.symbol, self.lev):
                self.log(f"⚠️ Failed to set leverage.")
                return
            balance = get_balance()
            if balance <= 0:
                self.log(f"⚠️ Insufficient balance.")
                return
            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
            if current_price <= 0:
                self.log(f"⚠️ Invalid price.")
                return
            step_size = get_step_size(self.symbol)
            if step_size <= 0:
                step_size = 0.001
            qty = (balance * self.percent / 100) * self.lev / current_price
            qty = math.floor(qty / step_size) * step_size
            if qty <= 0:
                self.log(f"⚠️ Invalid quantity.")
                return
            order = place_order(self.symbol, side, qty)
            if order and 'orderId' in order:
                self.position_open = True
                self.status = "open"
                self.side = side
                self.qty = qty if side == "BUY" else -qty
                self.entry = current_price
                self.position_attempt_count = 0
                self.log(f"🟢 {side} position opened at {current_price} (Qty: {qty})")
                signal_info = " | ".join([f"{k}: {v}" for k, v in current_signals.items()])
                send_telegram(f"📊 <b>{self.symbol}</b> {side} Signals: {signal_info}")
            else:
                self.position_attempt_count += 1
                self.log(f"⚠️ Failed to open {side} position (Attempt {self.position_attempt_count}/{self.max_position_attempts})")
        except Exception as e:
            self.position_attempt_count += 1
            self.log(f"❌ Error opening {side} position: {str(e)}")

    def close_position(self, reason):
        try:
            if not self.position_open or not self.side or not self.qty:
                self.log(f"⚠️ No position to close.")
                return
            side = "SELL" if self.side == "BUY" else "BUY"
            qty = abs(self.qty)
            order = place_order(self.symbol, side, qty)
            if order and 'orderId' in order:
                self.position_open = False
                self.status = "waiting"
                self.side = ""
                self.qty = 0
                self.entry = 0
                self.last_close_time = time.time()
                self.log(f"🔴 Position closed: {reason}")
            else:
                self.log(f"⚠️ Failed to close position.")
        except Exception as e:
            self.log(f"❌ Error closing position: {str(e)}")

# ========== BOT MANAGER ==========
class BotManager:
    def __init__(self):
        self.bots = {}
        self.ws_manager = WebSocketManager()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._lock = threading.Lock()

    def start_bot(self, symbol, lev, percent, tp, sl, initial_weights=None):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.bots:
                send_telegram(f"⚠️ Bot for {symbol} is already running.")
                return False

            # Kiểm tra trọng số trước khi khởi tạo bot
            if not initial_weights or not self._are_weights_valid(initial_weights):
                send_telegram(f"❌ Bot for {symbol} cannot be started due to invalid or missing weights.")
                return False

            try:
                bot = IndicatorBot(symbol, lev, percent, tp, sl, self.ws_manager, initial_weights)
                self.bots[symbol] = bot
                send_telegram(f"🟢 Bot started for {symbol} with leverage {lev}x, {percent}% balance, TP {tp}%, SL {sl}%")
                return True
            except Exception as e:
                send_telegram(f"❌ Error starting bot for {symbol}: {str(e)}")
                return False

    def _are_weights_valid(self, weights):
        """Kiểm tra tính hợp lệ của trọng số"""
        if not isinstance(weights, dict):
            return False
        if len(weights) == 0:
            return False
        # Kiểm tra xem có ít nhất một trọng số dương
        has_positive_weight = any(weight > 0 for weight in weights.values())
        return has_positive_weight

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.bots:
                self.bots[symbol].stop()
                del self.bots[symbol]
                send_telegram(f"🔴 Bot stopped for {symbol}")
                return True
            else:
                send_telegram(f"⚠️ No bot found for {symbol}")
                return False

    def stop_all_bots(self):
        with self._lock:
            for symbol in list(self.bots.keys()):
                self.stop_bot(symbol)
            send_telegram("🔴 All bots stopped")

    def get_bot_status(self):
        with self._lock:
            if not self.bots:
                return "No bots are currently running."
            status_lines = []
            for symbol, bot in self.bots.items():
                status_lines.append(f"• {symbol}: {bot.status} (Leverage: {bot.lev}x, Balance: {bot.percent}%)")
            return "\n".join(status_lines)

# ========== TRAINING FUNCTIONS ==========
def perform_initial_training(symbol, training_period_days=30):
    """
    Performs initial training for a symbol and returns the trained weights.
    This function now returns the weights directly instead of modifying global config.
    """
    try:
        send_telegram(f"🧠 Starting initial training for {symbol} ({training_period_days} days)...")
        
        # Lấy dữ liệu lịch sử
        df = get_klines(symbol, "1h", limit=24 * training_period_days)
        if df.empty or len(df) < 100:
            send_telegram(f"❌ Not enough historical data for {symbol}")
            return None

        # Thêm chỉ báo kỹ thuật
        df = add_technical_indicators(df)
        df = df.dropna()
        
        if df.empty:
            send_telegram(f"❌ No valid data after adding indicators for {symbol}")
            return None

        # Khởi tạo trọng số và thống kê
        indicator_names = ["RSI", "MACD", "EMA_Crossover", "Volume_Confirmation", 
                          "Stochastic", "BollingerBands", "Ichimoku", "ADX"]
        initial_weights = {name: 100.0 / len(indicator_names) for name in indicator_names}
        indicator_stats = {name: 0 for name in indicator_names}

        # Huấn luyện trên dữ liệu lịch sử
        for i in range(50, len(df)-1):
            current_data = df.iloc[:i+1].copy()
            current_signals = get_raw_indicator_signals(current_data)
            
            # Tính phần trăm thay đổi giá cho nến tiếp theo
            price_change_percent = ((df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]) * 100
            
            # Cập nhật trọng số
            initial_weights, indicator_stats = update_weights_and_stats(
                current_signals, price_change_percent, initial_weights, indicator_stats, True
            )

        # Chuyển điểm số thành trọng số phần trăm
        total_score = sum(indicator_stats.values())
        if total_score != 0:
            trained_weights = {k: (v / total_score) * 100 for k, v in indicator_stats.items()}
        else:
            # Nếu tổng điểm bằng 0, sử dụng trọng số đều
            trained_weights = {k: 100.0 / len(indicator_stats) for k in indicator_stats}
        
        send_telegram(f"✅ Training completed for {symbol}")
        
        # Log kết quả huấn luyện
        weight_info = " | ".join([f"{k}: {v:.2f}%" for k, v in trained_weights.items()])
        send_telegram(f"📊 Trained weights for {symbol}: {weight_info}")
        
        return trained_weights
        
    except Exception as e:
        send_telegram(f"❌ Training error for {symbol}: {str(e)}")
        return None

# ========== GLOBAL VARIABLES ==========
bot_manager = BotManager()
user_states = {}

# ========== TELEGRAM BOT HANDLERS ==========
def handle_telegram_message(update):
    try:
        message = update.get('message', {})
        text = message.get('text', '').strip()
        chat_id = message.get('chat', {}).get('id')
        
        if not text or not chat_id:
            return

        user_state = user_states.get(chat_id, {})

        # Xử lý trạng thái nhập thủ công
        if user_state.get('waiting_for_input'):
            del user_states[chat_id]['waiting_for_input']
            handle_manual_input(chat_id, text, user_state)
            return

        # Xử lý lệnh thông thường
        if text == "📊 Danh sách Bot":
            status = bot_manager.get_bot_status()
            send_telegram(f"🤖 <b>Bot Status</b>\n\n{status}", chat_id)

        elif text == "➕ Thêm Bot":
            user_states[chat_id] = {'step': 'select_symbol'}
            send_telegram("🔤 Please enter the trading pair (e.g., BTCUSDT):", chat_id, create_cancel_keyboard())

        elif text == "⛔ Dừng Bot":
            user_states[chat_id] = {'step': 'stop_bot'}
            send_telegram("🔤 Enter the trading pair to stop:", chat_id, create_cancel_keyboard())

        elif text == "💰 Số dư tài khoản":
            balance = get_balance()
            send_telegram(f"💳 <b>Account Balance</b>\n\nAvailable USDT: {balance:.2f}", chat_id)

        elif text == "📈 Vị thế đang mở":
            positions = get_positions()
            open_positions = [pos for pos in positions if float(pos.get('positionAmt', 0)) != 0]
            if open_positions:
                position_info = []
                for pos in open_positions:
                    side = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
                    pnl = float(pos.get('unRealizedProfit', 0))
                    position_info.append(f"• {pos['symbol']} {side} | PnL: {pnl:.2f} USDT")
                send_telegram("📊 <b>Open Positions</b>\n\n" + "\n".join(position_info), chat_id)
            else:
                send_telegram("📊 <b>Open Positions</b>\n\nNo open positions.", chat_id)

        elif text == "❌ Hủy bỏ":
            if chat_id in user_states:
                del user_states[chat_id]
            send_telegram("❌ Operation cancelled.", chat_id, create_menu_keyboard())

        # Xử lý các bước trong quy trình thêm bot
        elif user_state.get('step') == 'select_symbol':
            symbol = text.upper().replace(' ', '')
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            # Thực hiện huấn luyện ban đầu và nhận trọng số
            send_telegram(f"🧠 Training bot for {symbol}... This may take a few minutes.", chat_id)
            trained_weights = perform_initial_training(symbol)
            
            if not trained_weights:
                send_telegram(f"❌ Training failed for {symbol}. Bot cannot be started.", chat_id, create_menu_keyboard())
                del user_states[chat_id]
                return
            
            user_states[chat_id] = {
                'step': 'select_leverage', 
                'symbol': symbol,
                'trained_weights': trained_weights  # Lưu trọng số đã huấn luyện
            }
            send_telegram(f"✅ Training completed for {symbol}. Now select leverage:", chat_id, create_leverage_keyboard())

        elif user_state.get('step') == 'select_leverage':
            if text.replace('x', '').replace(' ', '').isdigit():
                leverage = int(text.replace('x', '').replace(' ', ''))
                user_states[chat_id]['leverage'] = leverage
                user_states[chat_id]['step'] = 'enter_percent'
                send_telegram("💯 Enter the percentage of balance to use per trade (1-100):", chat_id, create_cancel_keyboard())
            else:
                send_telegram("❌ Invalid leverage. Please enter a valid number (e.g., '10' or '10x'):", chat_id, create_leverage_keyboard())

        elif user_state.get('step') == 'enter_percent':
            if text.replace('%', '').replace(' ', '').replace('.', '').isdigit():
                percent = float(text.replace('%', '').replace(' ', ''))
                if 0 < percent <= 100:
                    user_states[chat_id]['percent'] = percent
                    user_states[chat_id]['step'] = 'enter_tp'
                    send_telegram("🎯 Enter Take Profit percentage (e.g., 5 for 5%):", chat_id, create_cancel_keyboard())
                else:
                    send_telegram("❌ Percentage must be between 0.1 and 100. Please enter a valid percentage:", chat_id, create_cancel_keyboard())
            else:
                send_telegram("❌ Invalid percentage. Please enter a valid number (e.g., '10' or '10%'):", chat_id, create_cancel_keyboard())

        elif user_state.get('step') == 'enter_tp':
            if text.replace('%', '').replace(' ', '').replace('.', '').isdigit():
                tp = float(text.replace('%', '').replace(' ', ''))
                user_states[chat_id]['tp'] = tp
                user_states[chat_id]['step'] = 'enter_sl'
                send_telegram("🛑 Enter Stop Loss percentage (e.g., 2 for 2%):", chat_id, create_cancel_keyboard())
            else:
                send_telegram("❌ Invalid TP. Please enter a valid number (e.g., '5' for 5%):", chat_id, create_cancel_keyboard())

        elif user_state.get('step') == 'enter_sl':
            if text.replace('%', '').replace(' ', '').replace('.', '').isdigit():
                sl = float(text.replace('%', '').replace(' ', ''))
                symbol = user_states[chat_id]['symbol']
                leverage = user_states[chat_id]['leverage']
                percent = user_states[chat_id]['percent']
                tp = user_states[chat_id]['tp']
                trained_weights = user_states[chat_id]['trained_weights']  # Lấy trọng số đã huấn luyện
                
                # Khởi chạy bot với trọng số đã huấn luyện
                success = bot_manager.start_bot(symbol, leverage, percent, tp, sl, trained_weights)
                
                if success:
                    send_telegram(f"✅ Bot configuration completed!\n\n"
                                f"Symbol: {symbol}\n"
                                f"Leverage: {leverage}x\n"
                                f"Balance Usage: {percent}%\n"
                                f"Take Profit: {tp}%\n"
                                f"Stop Loss: {sl}%", chat_id, create_menu_keyboard())
                else:
                    send_telegram(f"❌ Failed to start bot for {symbol}. Please try again.", chat_id, create_menu_keyboard())
                
                del user_states[chat_id]
            else:
                send_telegram("❌ Invalid SL. Please enter a valid number (e.g., '2' for 2%):", chat_id, create_cancel_keyboard())

        elif user_state.get('step') == 'stop_bot':
            symbol = text.upper().replace(' ', '')
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            success = bot_manager.stop_bot(symbol)
            if success:
                del user_states[chat_id]
                send_telegram(f"✅ Bot for {symbol} stopped successfully.", chat_id, create_menu_keyboard())
            else:
                send_telegram(f"⚠️ No bot found for {symbol}. Please check the symbol and try again.", chat_id, create_menu_keyboard())

    except Exception as e:
        logger.error(f"Telegram handler error: {str(e)}")
        send_telegram("❌ An error occurred. Please try again.", chat_id, create_menu_keyboard())
        if chat_id in user_states:
            del user_states[chat_id]

def handle_manual_input(chat_id, text, user_state):
    """Xử lý nhập liệu thủ công từ người dùng"""
    try:
        if user_state.get('expecting') == 'symbol_for_training':
            symbol = text.upper().replace(' ', '')
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            send_telegram(f"🧠 Starting training for {symbol}...", chat_id)
            trained_weights = perform_initial_training(symbol)
            
            if trained_weights:
                # Cập nhật config với trọng số đã huấn luyện
                config_updated = False
                for config in BOT_CONFIGS:
                    if config['symbol'] == symbol:
                        config['initial_weights'] = trained_weights
                        config_updated = True
                        break
                
                if config_updated:
                    send_telegram(f"✅ Training completed and config updated for {symbol}", chat_id)
                else:
                    send_telegram(f"✅ Training completed for {symbol}. Add this symbol to config to use the trained weights.", chat_id)
            else:
                send_telegram(f"❌ Training failed for {symbol}", chat_id)
                
        # Xóa trạng thái người dùng
        if chat_id in user_states:
            del user_states[chat_id]
            
    except Exception as e:
        logger.error(f"Manual input handler error: {str(e)}")
        send_telegram("❌ Error processing input. Please try again.", chat_id)

# ========== MAIN EXECUTION ==========
def main():
    """Main function to start the bot system."""
    send_telegram("🤖 <b>Binance Futures Bot Started</b>", reply_markup=create_menu_keyboard())
    
    # Khởi chạy các bot từ config với trọng số đã huấn luyện
    for config in BOT_CONFIGS:
        try:
            symbol = config['symbol']
            lev = config['leverage']
            percent = config['percent']
            tp = config['tp']
            sl = config['sl']
            initial_weights = config.get('initial_weights')
            
            if initial_weights:
                success = bot_manager.start_bot(symbol, lev, percent, tp, sl, initial_weights)
                if success:
                    logger.info(f"Bot started successfully for {symbol}")
                else:
                    logger.error(f"Failed to start bot for {symbol} - invalid weights")
            else:
                logger.warning(f"No initial weights found for {symbol}. Skipping...")
                
        except Exception as e:
            logger.error(f"Error starting bot for {config.get('symbol', 'unknown')}: {str(e)}")

    # Khởi chạy Telegram polling
    from telegram.ext import Updater, MessageHandler, Filters
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text, lambda update, context: handle_telegram_message(update)))
    updater.start_polling()
    logger.info("Telegram bot started polling.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bot_manager.stop_all_bots()
        bot_manager.ws_manager.stop()

if __name__ == "__main__":
    main()
