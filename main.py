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

# ========== NEW GLOBAL VARIABLES FOR THE WEIGHTED SYSTEM ==========
# Initial weights for each indicator (sum to 100).
indicator_weights = {
    "RSI": 10.0,
    "MACD": 10.0,
    "EMA9": 10.0,
    "EMA21": 10.0,
    "ATR": 10.0,
    "volume": 10.0,
    "Stochastic": 10.0,
    "BollingerBands": 10.0,
    "Ichimoku": 10.0,
    "ADX": 10.0,
}

# Counters for correct/incorrect predictions for each indicator
indicator_stats = {
    "RSI": {"correct": 0, "incorrect": 0},
    "MACD": {"correct": 0, "incorrect": 0},
    "EMA9": {"correct": 0, "incorrect": 0},
    "EMA21": {"correct": 0, "incorrect": 0},
    "ATR": {"correct": 0, "incorrect": 0},
    "volume": {"correct": 0, "incorrect": 0},
    "Stochastic": {"correct": 0, "incorrect": 0},
    "BollingerBands": {"correct": 0, "incorrect": 0},
    "Ichimoku": {"correct": 0, "incorrect": 0},
    "ADX": {"correct": 0, "incorrect": 0},
}
# ========== END OF NEW GLOBAL VARIABLES ==========

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
        df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"])
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
    
    # Ichimoku Cloud (simplified)
    # df['ichimoku_tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
    # df['ichimoku_kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
    # df['ichimoku_cloud_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(26)
    # df['ichimoku_cloud_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # ADX (Average Directional Index)
    # The full ADX calculation is complex, here's a placeholder
    # df['ADX'] = ...
    
    return df

# ========== NEW SIGNAL FUNCTIONS ==========
def get_weighted_signal(df):
    """Calculates a trading signal based on a weighted sum of indicator signals."""
    global indicator_weights
    
    current_indicators = {}
    total_score = 0
    
    # RSI: > 50 is bullish (+1), < 50 is bearish (-1)
    if df['RSI'].iloc[-1] > 50:
        current_indicators["RSI"] = 1
        total_score += indicator_weights["RSI"]
    else:
        current_indicators["RSI"] = -1
        total_score -= indicator_weights["RSI"]
        
    # MACD: MACD line > signal line is bullish, otherwise bearish
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
        current_indicators["MACD"] = 1
        total_score += indicator_weights["MACD"]
    else:
        current_indicators["MACD"] = -1
        total_score -= indicator_weights["MACD"]
    
    # EMA Crossover: EMA9 > EMA21 and price > both is bullish
    if df['close'].iloc[-1] > df['EMA9'].iloc[-1] and df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1]:
        current_indicators["EMA9"] = 1
        current_indicators["EMA21"] = 1
        total_score += indicator_weights["EMA9"] + indicator_weights["EMA21"]
    else:
        current_indicators["EMA9"] = -1
        current_indicators["EMA21"] = -1
        total_score -= indicator_weights["EMA9"] + indicator_weights["EMA21"]

    # Volume: High volume (> 1.5x avg) signals strength
    if df['volume'].iloc[-1] > df['volume'].rolling(window=20).mean().iloc[-1] * 1.5:
        current_indicators["volume"] = 1
        total_score += indicator_weights["volume"]
    else:
        current_indicators["volume"] = -1
        total_score -= indicator_weights["volume"]
        
    # Stochastic Oscillator: K line > D line and both are below 80 is bullish
    if df['stoch_k'].iloc[-1] > df['stoch_d'].iloc[-1] and df['stoch_k'].iloc[-1] < 80:
        current_indicators["Stochastic"] = 1
        total_score += indicator_weights["Stochastic"]
    else:
        current_indicators["Stochastic"] = -1
        total_score -= indicator_weights["Stochastic"]

    # Bollinger Bands: Price above the upper band is bearish, below the lower band is bullish
    if df['close'].iloc[-1] < df['bollinger_low'].iloc[-1]:
        current_indicators["BollingerBands"] = 1
        total_score += indicator_weights["BollingerBands"]
    elif df['close'].iloc[-1] > df['bollinger_high'].iloc[-1]:
        current_indicators["BollingerBands"] = -1
        total_score -= indicator_weights["BollingerBands"]
    else:
        current_indicators["BollingerBands"] = 0
    
    # Ichimoku: Placeholder for a more complex indicator
    # For now, let's assume it provides a neutral signal
    current_indicators["Ichimoku"] = 0
    current_indicators["ADX"] = 0
    
    signal = 0
    if total_score > 0:
        signal = 1  # Buy
    elif total_score < 0:
        signal = -1 # Sell
        
    return signal, current_indicators

def update_weights_and_stats(signal, current_indicators, price_change_percent):
    """Dynamically adjusts indicator weights based on their performance."""
    global indicator_weights
    global indicator_stats
    
    # Tốc độ điều chỉnh (ví dụ: 5%)
    adjustment_rate = 0.05

    is_correct_signal = (signal == 1 and price_change_percent > 0) or \
                        (signal == -1 and price_change_percent < 0)

    for indicator, status in current_indicators.items():
        if status == 0: continue # Skip neutral indicators

        # Tăng/giảm trọng số dựa trên sự chính xác của chỉ báo
        if is_correct_signal:
            if status == signal: # Chỉ báo đúng hướng với tín hiệu tổng hợp và kết quả thực tế
                indicator_weights[indicator] *= (1 + adjustment_rate)
                indicator_stats[indicator]["correct"] += 1
            else: # Chỉ báo sai hướng với tín hiệu tổng hợp nhưng kết quả lại đúng (trường hợp này hiếm nhưng vẫn xảy ra)
                indicator_weights[indicator] *= (1 - adjustment_rate)
                indicator_stats[indicator]["incorrect"] += 1
        else:
            if status == signal: # Chỉ báo đúng hướng với tín hiệu tổng hợp nhưng kết quả thực tế lại sai
                indicator_weights[indicator] *= (1 - adjustment_rate)
                indicator_stats[indicator]["incorrect"] += 1
            else: # Chỉ báo sai hướng với tín hiệu tổng hợp nhưng kết quả lại sai, vậy chỉ báo này lại đang "đúng"
                indicator_weights[indicator] *= (1 + adjustment_rate)
                indicator_stats[indicator]["correct"] += 1
    
    # Chuẩn hóa lại các trọng số để tổng bằng 100
    total_weight = sum(indicator_weights.values())
    if total_weight <= 0:
        total_weight = 100
        for indicator in indicator_weights:
            indicator_weights[indicator] = 100 / len(indicator_weights)
    else:
        for indicator in indicator_weights:
            indicator_weights[indicator] = (indicator_weights[indicator] / total_weight) * 100
            
    logging.info("--- New weights and stats ---")
    for indicator, stats in indicator_stats.items():
        total_trades = stats["correct"] + stats["incorrect"]
        if total_trades > 0:
            accuracy = (stats["correct"] / total_trades) * 100
            logging.info(f"📊 {indicator}: Correct {stats['correct']}, Incorrect {stats['incorrect']}. Accuracy: {accuracy:.2f}%, Weight: {indicator_weights[indicator]:.2f}")

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
    def __init__(self, symbol, lev, percent, tp, sl, indicator, ws_manager):
        self.symbol = symbol.upper()
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.indicator = indicator # This is now just a label, e.g., "WEIGHTED_SYSTEM"
        self.ws_manager = ws_manager
        
        self.check_position_status()
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        
        self._stop = False
        self.position_open = False
        self.last_trade_time = 0
        self.position_check_interval = 60
        self.last_position_check = 0
        self.last_error_log_time = 0
        self.last_close_time = 0
        self.cooldown_period = 60
        self.max_position_attempts = 3
        self.position_attempt_count = 0
        
        self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot started for {self.symbol}")

    def log(self, message):
        logger.info(f"[{self.symbol}] {message}")
        send_telegram(f"<b>{self.symbol}</b>: {message}")
    
    def _handle_price_update(self, price):
        if self._stop:
            return
        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def get_signal(self):
        try:
            df = get_klines(self.symbol, "15m", 200)
            if df.empty or len(df) < 50:
                self.log("Not enough data to generate signal.")
                return None, None
            
            df = add_technical_indicators(df)
            
            # Check for NaN values in the last row after adding indicators
            if df.iloc[-1].isnull().any():
                self.log("Data for indicators is incomplete.")
                return None, None
            
            signal, current_indicators = get_weighted_signal(df)
            
            # Convert signal from 1/-1 to "BUY"/"SELL"
            if signal == 1:
                return "BUY", current_indicators
            elif signal == -1:
                return "SELL", current_indicators
            return None, None
        except Exception as e:
            self.log(f"get_signal error: {str(e)}")
            return None, None

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                signal, current_indicators = self.get_signal()
                if not self.position_open and self.status == "waiting":
                    if current_time - self.last_close_time < self.cooldown_period:
                        time.sleep(1)
                        continue
                    if signal and current_time - self.last_trade_time > 60:
                        self.open_position(signal, current_indicators)
                        self.last_trade_time = current_time
                if self.position_open and self.status == "open":
                    self.check_tp_sl()
                    if signal:
                        if (self.side == "BUY" and signal == "SELL") or (self.side == "SELL" and signal == "BUY"):
                            current_price = self.prices[-1] if self.prices else get_current_price(self.symbol)
                            if self.entry > 0 and current_price > 0:
                                profit = (current_price - self.entry) * self.qty if self.side == "BUY" else (self.entry - current_price) * abs(self.qty)
                                invested = self.entry * abs(self.qty) / self.lev
                                roi = (profit / invested) * 100 if invested != 0 else 0
                                if roi >= 20:
                                    self.close_position(f"🔄 ROI {roi:.2f}% exceeded threshold, reversing to {signal}")
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
            elif self.sl is not None and roi <= -self.sl:
                self.close_position(f"❌ SL hit at {self.sl}% (ROI: {roi:.2f}%)")
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"TP/SL check error: {str(e)}")
                self.last_error_log_time = time.time()

    def open_position(self, side, current_indicators=None):
        self.check_position_status()
        try:
            cancel_all_orders(self.symbol)
            if not set_leverage(self.symbol, self.lev):
                self.log(f"Could not set leverage to {self.lev}")
                return
            balance = get_balance()
            if balance <= 0:
                self.log(f"Insufficient USDT balance")
                return
            if self.percent > 100:
                self.percent = 100
            elif self.percent < 1:
                self.percent = 1
            usdt_amount = balance * (self.percent / 100)
            price = get_current_price(self.symbol)
            if price <= 0:
                self.log(f"Error getting price")
                return
            step = get_step_size(self.symbol)
            if step <= 0:
                step = 0.001
            qty = (usdt_amount * self.lev) / price
            if step > 0:
                steps = qty / step
                qty = round(steps) * step
            qty = max(qty, 0)
            qty = round(qty, 8)
            min_qty = step
            if qty < min_qty:
                self.log(f"⚠️ Quantity is too small ({qty}), not placing order")
                return
            self.position_attempt_count += 1
            if self.position_attempt_count > self.max_position_attempts:
                self.log(f"⚠️ Reached max position attempt limit ({self.max_position_attempts})")
                self.position_attempt_count = 0
                return
            res = place_order(self.symbol, side, qty)
            if not res:
                self.log(f"Error placing order")
                return
            executed_qty = float(res.get('executedQty', 0))
            if executed_qty < 0:
                self.log(f"Order not filled, executed quantity: {executed_qty}")
                return
            self.entry = float(res.get('avgPrice', price))
            self.side = side
            self.qty = executed_qty if side == "BUY" else -executed_qty
            self.status = "open"
            self.position_open = True
            self.position_attempt_count = 0
            
            # Lấy thông tin chỉ báo và trọng số
            if current_indicators:
                indicator_info = "Phân tích tín hiệu:\n"
                for indicator, status in current_indicators.items():
                    weight = indicator_weights.get(indicator, 0)
                    sign_text = "🟢 Tăng" if status == 1 else "🔴 Giảm" if status == -1 else "⚪ Trung lập"
                    indicator_info += f"- {indicator}: {weight:.2f}% ({sign_text})\n"
            else:
                indicator_info = "Không đủ dữ liệu chỉ báo."

            message = (f"✅ <b>POSITION OPENED {self.symbol}</b>\n"
                       f"📌 Direction: {side}\n"
                       f"🏷️ Entry Price: {self.entry:.4f}\n"
                       f"📊 Quantity: {executed_qty}\n"
                       f"💵 Value: {executed_qty * self.entry:.2f} USDT\n"
                       f" Leverage: {self.lev}x\n"
                       f"🎯 TP: {self.tp}% | 🛡️ SL: {self.sl}%\n\n"
                       f"{indicator_info}")
            self.log(message)
        except Exception as e:
            self.position_open = False
            self.log(f"❌ Error entering position: {str(e)}")

    def close_position(self, reason=""):
        try:
            cancel_all_orders(self.symbol)
            if abs(self.qty) > 0:
                close_side = "SELL" if self.side == "BUY" else "BUY"
                close_qty = abs(self.qty)
                step = get_step_size(self.symbol)
                if step > 0:
                    steps = close_qty / step
                    close_qty = round(steps) * step
                close_qty = max(close_qty, 0)
                close_qty = round(close_qty, 8)
                res = place_order(self.symbol, close_side, close_qty)
                if res:
                    price = float(res.get('avgPrice', 0))
                    message = (f"⛔ <b>POSITION CLOSED {self.symbol}</b>\n" f"📌 Reason: {reason}\n" f"🏷️ Exit Price: {price:.4f}\n" f"📊 Quantity: {close_qty}\n" f"💵 Value: {close_qty * price:.2f} USDT")
                    self.log(message)
                    self.status = "waiting"
                    self.side = ""
                    self.qty = 0
                    self.entry = 0
                    self.position_open = False
                    self.last_trade_time = time.time()
                    self.last_close_time = time.time()
                else:
                    self.log(f"Error closing position")
        except Exception as e:
            self.log(f"❌ Error closing position: {str(e)}")

# ========== BOT MANAGER ==========
class BotManager:
    def __init__(self):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        self.admin_chat_id = TELEGRAM_CHAT_ID
        self.log("🟢 BOT SYSTEM STARTED")
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()
        self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
        self.telegram_thread.start()
        if self.admin_chat_id:
            self.send_main_menu(self.admin_chat_id)
        
        # Start a thread to update weights
        self.weight_update_thread = threading.Thread(target=self._weight_updater, daemon=True)
        self.weight_update_thread.start()
        
    def _weight_updater(self):
        """Periodically updates indicator weights based on performance."""
        while self.running:
            for symbol, bot in self.bots.items():
                try:
                    df = get_klines(symbol, '5m', 100)
                    if not df.empty and len(df) >= 2:
                        df = add_technical_indicators(df)
                        signal, current_indicators = get_weighted_signal(df)
                        
                        # Get price change between the last two closed candles
                        current_close = df['close'].iloc[-2]
                        next_close = df['close'].iloc[-1]
                        price_change_percent = ((next_close - current_close) / current_close) * 100
                        
                        update_weights_and_stats(signal, current_indicators, price_change_percent)
                except Exception as e:
                    logger.error(f"Error in weight update thread for {symbol}: {str(e)}")
            time.sleep(3600) # Update every hour

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")
        send_telegram(f"<b>SYSTEM</b>: {message}")

    def send_main_menu(self, chat_id):
        welcome = ("🤖 <b>BINANCE FUTURES TRADING BOT</b>\n\nChoose an option below:")
        send_telegram(welcome, chat_id, create_menu_keyboard())

    def add_bot(self, symbol, lev, percent, tp, sl, indicator):
        if sl == 0:
            sl = None
        symbol = symbol.upper()
        if symbol in self.bots:
            self.log(f"⚠️ Bot already exists for {symbol}")
            return False
        if not API_KEY or not API_SECRET:
            self.log("❌ API Key and Secret Key not configured!")
            return False
        try:
            price = get_current_price(symbol)
            if price <= 0:
                self.log(f"❌ Cannot get price for {symbol}")
                return False
            positions = get_positions(symbol)
            if positions and any(float(pos.get('positionAmt', 0)) != 0 for pos in positions):
                self.log(f"⚠️ Open position found for {symbol}")
            bot = IndicatorBot(symbol, lev, percent, tp, sl, "WEIGHTED_SYSTEM", self.ws_manager)
            self.bots[symbol] = bot
            self.log(f"✅ Bot added: {symbol} | Lev: {lev}x | %: {percent} | TP/SL: {tp}%/{sl}%")
            return True
        except Exception as e:
            self.log(f"❌ Error creating bot {symbol}: {str(e)}")
            return False

    def stop_bot(self, symbol):
        symbol = symbol.upper()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            if bot.status == "open":
                bot.close_position("⛔ Manual bot stop")
            self.log(f"⛔ Bot stopped for {symbol}")
            del self.bots[symbol]
            return True
        return False

    def stop_all(self):
        self.log("⛔ Stopping all bots...")
        for symbol in list(self.bots.keys()):
            self.stop_bot(symbol)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 System stopped")

    def _status_monitor(self):
        while self.running:
            try:
                uptime = time.time() - self.start_time
                hours, rem = divmod(uptime, 3600)
                minutes, seconds = divmod(rem, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                active_bots = [s for s, b in self.bots.items() if not b._stop]
                balance = get_balance()
                status_msg = (f"📊 <b>SYSTEM STATUS</b>\n" f"⏱ Uptime: {uptime_str}\n" f"🤖 Active Bots: {len(active_bots)}\n" f"📈 Active Pairs: {', '.join(active_bots) if active_bots else 'None'}\n" f"💰 Available Balance: {balance:.2f} USDT")
                send_telegram(status_msg)
                for symbol, bot in self.bots.items():
                    if bot.status == "open":
                        status_msg = (f"🔹 <b>{symbol}</b>\n" f"📌 Direction: {bot.side}\n" f"🏷️ Entry Price: {bot.entry:.4f}\n" f"📊 Quantity: {abs(bot.qty)}\n" f" Leverage: {bot.lev}x\n" f"🎯 TP: {bot.tp}% | 🛡️ SL: {bot.sl}%")
                        send_telegram(status_msg)
            except Exception as e:
                logger.error(f"Status report error: {str(e)}")
            time.sleep(6 * 3600)

    def _telegram_listener(self):
        last_update_id = 0
        while self.running:
            try:
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
                            if chat_id != self.admin_chat_id:
                                continue
                            if update_id > last_update_id:
                                last_update_id = update_id
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    logger.error("Conflict error: Only one bot instance can listen to Telegram")
                    break
            except Exception as e:
                logger.error(f"Telegram listener error: {str(e)}")
                time.sleep(5)

    def _handle_telegram_message(self, chat_id, text):
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        if current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                symbol = text.upper()
                self.user_states[chat_id] = {'step': 'waiting_leverage', 'symbol': symbol}
                send_telegram(f"Choose leverage for {symbol}:", chat_id, create_leverage_keyboard())
        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            elif 'x' in text:
                leverage = int(text.replace('', '').replace('x', '').strip())
                user_state['leverage'] = leverage
                user_state['step'] = 'waiting_percent'
                send_telegram(f"📌 Pair: {user_state['symbol']}\n Leverage: {leverage}x\n\nEnter % of balance to use (1-100):", chat_id, create_cancel_keyboard())
        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                try:
                    percent = float(text)
                    if 1 <= percent <= 100:
                        user_state['percent'] = percent
                        user_state['step'] = 'waiting_tp'
                        send_telegram(f"📌 Pair: {user_state['symbol']}\n Lev: {user_state['leverage']}x\n📊 %: {percent}%\n\nEnter % Take Profit (e.g., 10):", chat_id, create_cancel_keyboard())
                    else:
                        send_telegram("⚠️ Please enter a % from 1-100", chat_id)
                except Exception:
                    send_telegram("⚠️ Invalid value, please enter a number", chat_id)
        elif current_step == 'waiting_tp':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                try:
                    tp = float(text)
                    if tp > 0:
                        user_state['tp'] = tp
                        user_state['step'] = 'waiting_sl'
                        send_telegram(f"📌 Pair: {user_state['symbol']}\n Lev: {user_state['leverage']}x\n📊 %: {user_state['percent']}%\n🎯 TP: {tp}%\n\nEnter % Stop Loss (e.g., 5):", chat_id, create_cancel_keyboard())
                    else:
                        send_telegram("⚠️ TP must be greater than 0", chat_id)
                except Exception:
                    send_telegram("⚠️ Invalid value, please enter a number", chat_id)
        elif current_step == 'waiting_sl':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Bot addition cancelled", chat_id, create_menu_keyboard())
            else:
                try:
                    sl = float(text)
                    if sl >= 0:
                        symbol = user_state['symbol']
                        leverage = user_state['leverage']
                        percent = user_state['percent']
                        tp = user_state['tp']
                        if self.add_bot(symbol, leverage, percent, tp, sl, "WEIGHTED_SYSTEM"):
                            send_telegram(f"✅ <b>BOT ADDED SUCCESSFULLY</b>\n\n" f"📌 Pair: {symbol}\n" f" Leverage: {leverage}x\n" f"📊 % Balance: {percent}%\n" f"🎯 TP: {tp}%\n" f"🛡️ SL: {sl}%", chat_id, create_menu_keyboard())
                        else:
                            send_telegram("❌ Could not add bot, please check logs", chat_id, create_menu_keyboard())
                        self.user_states[chat_id] = {}
                    else:
                        send_telegram("⚠️ SL must be greater than or equal to 0", chat_id)
                except Exception:
                    send_telegram("⚠️ Invalid value, please enter a number", chat_id)
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 No bots are currently running", chat_id)
            else:
                message = "🤖 <b>LIST OF RUNNING BOTS</b>\n\n"
                for symbol, bot in self.bots.items():
                    status = "🟢 Open" if bot.status == "open" else "🟡 Waiting"
                    message += f"🔹 {symbol} | {status} | {bot.side}\n"
                send_telegram(message, chat_id)
        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_symbol'}
            send_telegram("Choose a coin pair:", chat_id, create_symbols_keyboard())
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 No bots are currently running", chat_id)
            else:
                message = "⛔ <b>CHOOSE BOT TO STOP</b>\n\n"
                keyboard = []
                row = []
                for i, symbol in enumerate(self.bots.keys()):
                    message += f"🔹 {symbol}\n"
                    row.append({"text": f"⛔ {symbol}"})
                    if len(row) == 2 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                send_telegram(message, chat_id, {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True})
        elif text.startswith("⛔ "):
            symbol = text.replace("⛔ ", "").strip().upper()
            if symbol in self.bots:
                self.stop_bot(symbol)
                send_telegram(f"⛔ Stop command sent for bot {symbol}", chat_id, create_menu_keyboard())
            else:
                send_telegram(f"⚠️ Bot not found {symbol}", chat_id, create_menu_keyboard())
        elif text == "💰 Số dư tài khoản":
            try:
                balance = get_balance()
                send_telegram(f"💰 <b>AVAILABLE BALANCE</b>: {balance:.2f} USDT", chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Error getting balance: {str(e)}", chat_id)
        elif text == "📈 Vị thế đang mở":
            try:
                positions = get_positions()
                if not positions:
                    send_telegram("📭 No open positions", chat_id)
                    return
                message = "📈 <b>OPEN POSITIONS</b>\n\n"
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    if position_amt != 0:
                        symbol = pos.get('symbol', 'UNKNOWN')
                        entry = float(pos.get('entryPrice', 0))
                        side = "LONG" if position_amt > 0 else "SHORT"
                        pnl = float(pos.get('unRealizedProfit', 0))
                        message += (f"🔹 {symbol} | {side}\n" f"📊 Quantity: {abs(position_amt):.4f}\n" f"🏷️ Entry Price: {entry:.4f}\n" f"💰 PnL: {pnl:.2f} USDT\n\n")
                send_telegram(message, chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Error getting positions: {str(e)}", chat_id)
        elif text:
            self.send_main_menu(chat_id)

# ========== MAIN FUNCTION ==========
def main():
    manager = BotManager()
    
    if BOT_CONFIGS:
        manager.log("Đang thực hiện huấn luyện ban đầu trên dữ liệu lịch sử...")
        # Lặp qua từng cấu hình bot để huấn luyện
        for config in BOT_CONFIGS:
            symbol, lev, percent, tp, sl, _ = config
            
            # Lấy dữ liệu lịch sử đủ lớn để huấn luyện
            df_history = get_klines(symbol, '15m', 200)
            
            if not df_history.empty:
                manager.log(f"Bắt đầu huấn luyện cho {symbol} với 200 nến 15 phút...")
                # Lặp qua từng nến để tính tín hiệu và cập nhật trọng số
                for i in range(50, len(df_history) - 1): # Bắt đầu từ nến thứ 50 để đảm bảo có đủ dữ liệu cho các chỉ báo
                    df_slice = df_history.iloc[i-50:i+1] # Lấy một lát cắt dữ liệu để mô phỏng
                    df_slice = add_technical_indicators(df_slice)
                    
                    if not df_slice.iloc[-1].isnull().any():
                        signal, current_indicators = get_weighted_signal(df_slice)
                        
                        # Tính toán thay đổi giá của nến tiếp theo để đánh giá tín hiệu
                        price_change_percent = ((df_history['close'].iloc[i+1] - df_history['close'].iloc[i]) / df_history['close'].iloc[i]) * 100
                        
                        if signal:
                            update_weights_and_stats(signal, current_indicators, price_change_percent)
                manager.log(f"Hoàn thành huấn luyện ban đầu cho {symbol}.")
            else:
                manager.log(f"Không đủ dữ liệu lịch sử để huấn luyện bot cho {symbol}.")

    if BOT_CONFIGS:
        for config in BOT_CONFIGS:
            # Sau khi huấn luyện, thêm bot thực tế vào
            symbol, lev, percent, tp, sl, _ = config
            manager.add_bot(symbol, lev, percent, tp, sl, "WEIGHTED_SYSTEM")
    else:
        manager.log("⚠️ Không tìm thấy cấu hình bot nào!")
    try:
        balance = get_balance()
        manager.log(f"💰 SỐ DƯ KHỞI ĐẦU: {balance:.2f} USDT")
    except Exception as e:
        manager.log(f"⚠️ Lỗi khi lấy số dư khởi đầu: {str(e)}")
    try:
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
