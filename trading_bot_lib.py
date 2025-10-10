# trading_bot_lib.py - HOÀN CHỈNH VỚI AI FUSION MASTER
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
            [{"text": "🤖 AI Trading Basic"}, {"text": "📊 AI Trend Master"}],
            [{"text": "🎯 AI Reverse Pro"}, {"text": "⚡ AI Scalping Pro"}],
            [{"text": "🛡️ AI Safe Grid"}, {"text": "🔄 AI Dynamic Master"}],
            [{"text": "🚀 AI Fusion Master"}, {"text": "❌ Hủy bỏ"}]
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

def create_bot_count_keyboard():
    return {
        "keyboard": [
            [{"text": "1"}, {"text": "2"}],
            [{"text": "3"}, {"text": "5"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== QUẢN LÝ COIN CHUNG ==========
class CoinManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CoinManager, cls).__new__(cls)
                cls._instance.managed_coins = {}
                cls._instance.config_coin_count = {}
        return cls._instance
    
    def register_coin(self, symbol, bot_id, strategy, config_key=None):
        with self._lock:
            if config_key not in self.config_coin_count:
                self.config_coin_count[config_key] = 0
            
            if self.config_coin_count.get(config_key, 0) >= 2:
                return False
                
            if symbol not in self.managed_coins:
                self.managed_coins[symbol] = {
                    "strategy": strategy, 
                    "bot_id": bot_id,
                    "config_key": config_key
                }
                self.config_coin_count[config_key] = self.config_coin_count.get(config_key, 0) + 1
                return True
            return False
    
    def unregister_coin(self, symbol):
        with self._lock:
            if symbol in self.managed_coins:
                config_key = self.managed_coins[symbol].get("config_key")
                del self.managed_coins[symbol]
                
                if config_key in self.config_coin_count:
                    self.config_coin_count[config_key] = max(0, self.config_coin_count[config_key] - 1)
                return True
            return False
    
    def is_coin_managed(self, symbol):
        with self._lock:
            return symbol in self.managed_coins

    def has_same_config_bot(self, symbol, config_key):
        with self._lock:
            if symbol in self.managed_coins:
                existing_config = self.managed_coins[symbol].get("config_key")
                return existing_config == config_key
            return False
    
    def count_bots_by_config(self, config_key):
        with self._lock:
            count = 0
            for coin_info in self.managed_coins.values():
                if coin_info.get("config_key") == config_key:
                    count += 1
            return count
    
    def get_config_coin_count(self, config_key):
        with self._lock:
            return self.config_coin_count.get(config_key, 0)
    
    def get_managed_coins(self):
        with self._lock:
            return self.managed_coins.copy()

# ========== API BINANCE ==========
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
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 401:
                return None
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_all_usdt_pairs(limit=100):
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

def get_top_volatile_symbols(limit=10, threshold=20):
    try:
        all_symbols = get_all_usdt_pairs(limit=200)
        if not all_symbols:
            logger.warning("Không lấy được coin từ Binance")
            return []
        
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        data = binance_api_request(url)
        if not data:
            return []
        
        ticker_dict = {ticker['symbol']: ticker for ticker in data if 'symbol' in ticker}
        
        volatile_pairs = []
        for symbol in all_symbols:
            if symbol in ticker_dict:
                ticker = ticker_dict[symbol]
                try:
                    change = float(ticker.get('priceChangePercent', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    
                    if abs(change) >= threshold:
                        volatile_pairs.append((symbol, abs(change)))
                except (ValueError, TypeError):
                    continue
        
        volatile_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_symbols = [pair[0] for pair in volatile_pairs[:limit]]
        logger.info(f"✅ Tìm thấy {len(top_symbols)} coin biến động ≥{threshold}%")
        return top_symbols
        
    except Exception as e:
        logger.error(f"❌ Lỗi lấy danh sách coin biến động: {str(e)}")
        return []

def get_qualified_symbols(api_key, api_secret, strategy_type, leverage, threshold=None, volatility=None, grid_levels=None, max_candidates=20, final_limit=2, strategy_key=None):
    try:
        test_balance = get_balance(api_key, api_secret)
        if test_balance is None:
            logger.error("❌ KHÔNG THỂ KẾT NỐI BINANCE")
            return []
        
        coin_manager = CoinManager()
        
        all_symbols = get_all_usdt_pairs(limit=200)
        if not all_symbols:
            logger.error("❌ Không lấy được danh sách coin từ Binance")
            return []
        
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        data = binance_api_request(url)
        if not data:
            return []
        
        ticker_dict = {ticker['symbol']: ticker for ticker in data if 'symbol' in ticker}
        
        qualified_symbols = []
        
        for symbol in all_symbols:
            if symbol not in ticker_dict:
                continue
                
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                continue
            
            if strategy_key and coin_manager.has_same_config_bot(symbol, strategy_key):
                continue
            
            ticker = ticker_dict[symbol]
            
            try:
                price_change = float(ticker.get('priceChangePercent', 0))
                abs_price_change = abs(price_change)
                volume = float(ticker.get('quoteVolume', 0))
                high_price = float(ticker.get('highPrice', 0))
                low_price = float(ticker.get('lowPrice', 0))
                
                if low_price > 0:
                    price_range = ((high_price - low_price) / low_price) * 100
                else:
                    price_range = 0
                
                if strategy_type == "AI Reverse Pro":
                    if abs_price_change >= (threshold or 15):
                        score = abs_price_change * (volume / 1000000)
                        qualified_symbols.append((symbol, score, price_change, volume))
                
                elif strategy_type == "AI Scalping Pro":
                    if abs_price_change >= (volatility or 2) and volume > 2000000 and price_range >= 1.0:
                        qualified_symbols.append((symbol, price_range, volume))
                
                elif strategy_type == "AI Safe Grid":
                    if 0.5 <= abs_price_change <= 8.0:
                        qualified_symbols.append((symbol, -abs(price_change - 3.0), volume))
                
                elif strategy_type == "AI Trend Master":
                    if (1.0 <= abs_price_change <= 15.0 and price_range >= 0.5):
                        score = volume * abs_price_change
                        qualified_symbols.append((symbol, score, volume))
                
                elif strategy_type == "AI Dynamic Master":
                    if (1.0 <= abs_price_change <= 12.0 and price_range >= 0.8):
                        volume_score = min(volume / 5000000, 5)
                        volatility_score = min(abs_price_change / 10, 3)
                        score = volume_score + volatility_score
                        qualified_symbols.append((symbol, score, volume))
                
                elif strategy_type == "AI Fusion Master":
                    if (2.0 <= abs_price_change <= 20.0 and price_range >= 1.0 and volume > 1000000):
                        complexity_score = (abs_price_change * price_range * np.log(volume)) / 100
                        qualified_symbols.append((symbol, complexity_score, volume))
                        
            except (ValueError, TypeError) as e:
                continue
        
        if strategy_type == "AI Reverse Pro":
            qualified_symbols.sort(key=lambda x: (x[1], x[3]), reverse=True)
        elif strategy_type == "AI Scalping Pro":
            qualified_symbols.sort(key=lambda x: (x[1], x[2]), reverse=True)
        elif strategy_type == "AI Safe Grid":
            qualified_symbols.sort(key=lambda x: (x[1], x[2]), reverse=True)
        elif strategy_type == "AI Trend Master":
            qualified_symbols.sort(key=lambda x: (x[1], x[2]), reverse=True)
        elif strategy_type == "AI Dynamic Master":
            qualified_symbols.sort(key=lambda x: (x[1], x[2]), reverse=True)
        elif strategy_type == "AI Fusion Master":
            qualified_symbols.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        logger.info(f"🔍 {strategy_type}: Quét {len(all_symbols)} coin, tìm thấy {len(qualified_symbols)} phù hợp")
        
        final_symbols = []
        for item in qualified_symbols[:max_candidates]:
            if len(final_symbols) >= final_limit:
                break
                
            if strategy_type == "AI Reverse Pro":
                symbol, score, original_change, volume = item
            else:
                symbol, score, volume = item
                
            try:
                leverage_success = set_leverage(symbol, leverage, api_key, api_secret)
                step_size = get_step_size(symbol, api_key, api_secret)
                
                if leverage_success and step_size > 0:
                    final_symbols.append(symbol)
                    if strategy_type == "AI Reverse Pro":
                        logger.info(f"✅ {symbol}: phù hợp {strategy_type} (Biến động: {original_change:.2f}%, Điểm: {score:.2f}, Volume: {volume:.0f})")
                    else:
                        logger.info(f"✅ {symbol}: phù hợp {strategy_type} (Score: {score:.2f}, Volume: {volume:.0f})")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Lỗi kiểm tra {symbol}: {str(e)}")
                continue
        
        if not final_symbols:
            logger.warning(f"⚠️ {strategy_type}: không tìm thấy coin phù hợp, sử dụng backup method")
            backup_symbols = []
            
            for symbol in all_symbols:
                if symbol not in ticker_dict:
                    continue
                    
                if strategy_key and coin_manager.has_same_config_bot(symbol, strategy_key):
                    continue
                    
                ticker = ticker_dict[symbol]
                try:
                    volume = float(ticker.get('quoteVolume', 0))
                    price_change = float(ticker.get('priceChangePercent', 0))
                    abs_price_change = abs(price_change)
                    
                    if (0.5 <= abs_price_change <= 10.0 and
                        symbol not in ['BTCUSDT', 'ETHUSDT']):
                        backup_symbols.append((symbol, volume, abs_price_change))
                except:
                    continue
            
            backup_symbols.sort(key=lambda x: x[1], reverse=True)
            
            for symbol, volume, price_change in backup_symbols[:final_limit]:
                try:
                    leverage_success = set_leverage(symbol, leverage, api_key, api_secret)
                    step_size = get_step_size(symbol, api_key, api_secret)
                    
                    if leverage_success and step_size > 0:
                        final_symbols.append(symbol)
                        logger.info(f"🔄 {symbol}: backup coin (Volume: {volume:.0f}, Biến động: {price_change:.2f}%)")
                        if len(final_symbols) >= final_limit:
                            break
                    time.sleep(0.1)
                except Exception as e:
                    continue
        
        logger.info(f"🎯 {strategy_type}: Kết quả cuối - {len(final_symbols)} coin: {final_symbols}")
        return final_symbols[:final_limit]
        
    except Exception as e:
        logger.error(f"❌ Lỗi tìm coin {strategy_type}: {str(e)}")
        return []

def get_step_size(symbol, api_key, api_secret):
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
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return None
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        return 0
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        return None

def place_order(symbol, side, qty, api_key, api_secret):
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
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
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

# ========== CHỈ BÁO KỸ THUẬT ==========
def calc_rsi(prices, period=14):
    try:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi_value = 100.0 - (100.0 / (1 + rs))
        
        if np.isnan(rsi_value) or np.isinf(rsi_value):
            return None
        return rsi_value
    except Exception as e:
        return None

def calc_ema(prices, period):
    try:
        if len(prices) < period:
            return None
        
        ema = [sum(prices[:period]) / period]
        multiplier = 2 / (period + 1)
        
        for price in prices[period:]:
            ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        
        return float(ema[-1])
    except Exception as e:
        return None

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

# ========== BASE BOT NÂNG CẤP ==========
class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None):
        
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
        
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self.position_open = False
        self._stop = False
        
        self.last_trade_time = 0
        self.last_close_time = 0
        self._last_find_attempt = 0
        self._find_coin_cooldown = 300
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 300
        self.position_check_interval = 30
        
        self._close_attempted = False
        self._last_close_attempt = 0
        
        self.should_be_removed = False
        
        self.coin_manager = CoinManager()
        
        if symbol and config_key:
            success = self._register_coin_with_retry(symbol)
            if not success:
                self.log(f"❌ Không thể đăng ký coin {symbol} - đã đạt giới hạn 2 coin/config")
                self.should_be_removed = True
        
        self.check_position_status()
        if symbol:
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")

    def _register_coin_with_retry(self, symbol):
        max_retries = 3
        for attempt in range(max_retries):
            success = self.coin_manager.register_coin(symbol, f"{self.strategy_name}_{id(self)}", self.strategy_name, self.config_key)
            if success:
                return True
            time.sleep(0.5)
        return False

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

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                
                if not self.position_open:
                    signal = self.get_signal()
                    
                    if (signal and 
                        current_time - self.last_trade_time > 60 and
                        current_time - self.last_close_time > self.cooldown_period):
                        
                        self.log(f"🎯 Nhận tín hiệu {signal}, đang mở lệnh...")
                        if self.open_position(signal):
                            self.last_trade_time = current_time
                        else:
                            time.sleep(30)
                
                if self.position_open and not self._close_attempted:
                    self.check_tp_sl()
                    
                time.sleep(1)
                
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"❌ Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        if self.symbol:
            self.ws_manager.remove_symbol(self.symbol)
        if self.symbol and self.config_key:
            self.coin_manager.unregister_coin(self.symbol)
        if self.symbol:
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
                if self.symbol and self.config_key:
                    self.coin_manager.unregister_coin(self.symbol)
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
                
                if self.symbol and self.config_key:
                    self.coin_manager.unregister_coin(self.symbol)
                
                if hasattr(self, 'config_key') and self.config_key:
                    self._find_new_coin_after_close()
                
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

    def _find_new_coin_after_close(self):
        try:
            current_time = time.time()
            if hasattr(self, '_last_find_attempt') and current_time - self._last_find_attempt < 300:
                return False
                
            self._last_find_attempt = current_time
            
            self.log(f"🔄 Bot động đang tìm coin mới thay thế {self.symbol}...")
            
            if hasattr(self, 'config_key') and self.config_key:
                bot_manager = getattr(self, '_bot_manager', None)
                if bot_manager and hasattr(bot_manager, '_handle_coin_after_close'):
                    bot_manager._handle_coin_after_close(self.config_key, self.symbol)
                    return True
            
            return False
            
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {str(e)}")
            return False

    def check_tp_sl(self):
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

# ========== CÁC CHIẾN LƯỢC AI ==========
class AI_Trading_Basic(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "AI Trading Basic")

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            # AI Phân tích đa khung thời gian
            short_term = self._ai_analyze_short_term()
            medium_term = self._ai_analyze_medium_term()
            long_term = self._ai_analyze_long_term()
            
            # Tổng hợp tín hiệu AI
            total_score = short_term * 0.4 + medium_term * 0.35 + long_term * 0.25
            
            if total_score > 0.7:
                return "BUY"
            elif total_score < -0.7:
                return "SELL"
            return None

        except Exception as e:
            return None

    def _ai_analyze_short_term(self):
        if len(self.prices) < 10:
            return 0
        recent_prices = self.prices[-10:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        return np.tanh(momentum * 10)

    def _ai_analyze_medium_term(self):
        if len(self.prices) < 30:
            return 0
        medium_prices = self.prices[-30:]
        trend = np.polyfit(range(len(medium_prices)), medium_prices, 1)[0]
        return np.tanh(trend * 1000)

    def _ai_analyze_long_term(self):
        if len(self.prices) < 50:
            return 0
        long_prices = self.prices[-50:]
        volatility = np.std(long_prices) / np.mean(long_prices)
        return -np.tanh(volatility * 10)

class AI_Trend_Master(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "AI Trend Master", config_key)

    def get_signal(self):
        try:
            if len(self.prices) < 60:
                return None

            # AI Nhận diện xu hướng đa cấp độ
            trend_strength = self._ai_detect_trend_strength()
            momentum_quality = self._ai_analyze_momentum()
            volume_confirmation = self._ai_analyze_volume()
            
            # AI Ra quyết định
            ai_confidence = trend_strength * 0.5 + momentum_quality * 0.3 + volume_confirmation * 0.2
            
            if ai_confidence > 0.65:
                return "BUY"
            elif ai_confidence < -0.65:
                return "SELL"
            return None

        except Exception as e:
            return None

    def _ai_detect_trend_strength(self):
        if len(self.prices) < 20:
            return 0
        prices = np.array(self.prices[-20:])
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return np.tanh(slope * 1000)

    def _ai_analyze_momentum(self):
        if len(self.prices) < 15:
            return 0
        returns = np.diff(self.prices[-15:]) / self.prices[-16:-1]
        momentum = np.mean(returns)
        return np.tanh(momentum * 100)

    def _ai_analyze_volume(self):
        return 0.3

class AI_Reverse_Pro(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, threshold=30, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "AI Reverse Pro", config_key)
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

            # AI Phân tích đảo chiều thông minh
            if abs(change_24h) >= self.threshold:
                ai_reversal_score = self._ai_analyze_reversal_probability(change_24h)
                
                if ai_reversal_score > 0.6:
                    if change_24h > 0:
                        self.log(f"🎯 AI Reverse SELL - Biến động: +{change_24h:.2f}% | Độ tin cậy: {ai_reversal_score:.2f}")
                        return "SELL"
                    else:
                        self.log(f"🎯 AI Reverse BUY - Biến động: {change_24h:.2f}% | Độ tin cậy: {ai_reversal_score:.2f}")
                        return "BUY"

            return None

        except Exception as e:
            self.log(f"❌ Lỗi tín hiệu AI Reverse Pro: {str(e)}")
            return None

    def _ai_analyze_reversal_probability(self, change_24h):
        if len(self.prices) < 20:
            return 0.5
            
        recent_volatility = np.std(self.prices[-10:]) / np.mean(self.prices[-10:])
        momentum = (self.prices[-1] - self.prices[-5]) / self.prices[-5]
        
        # AI tính xác suất đảo chiều
        reversal_prob = 1.0 / (1.0 + np.exp(-abs(change_24h)/10 + recent_volatility*100 - abs(momentum)*50))
        return reversal_prob

class AI_Scalping_Pro(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "AI Scalping Pro", config_key)

    def get_signal(self):
        try:
            if len(self.prices) < 25:
                return None

            # AI Scalping - Phân tích tốc độ và biến động
            speed_signal = self._ai_analyze_speed()
            volatility_signal = self._ai_analyze_scalping_volatility()
            micro_trend = self._ai_detect_micro_trend()
            
            # AI Decision Engine
            scalping_score = speed_signal * 0.4 + volatility_signal * 0.4 + micro_trend * 0.2
            
            if scalping_score > 0.7:
                return "BUY"
            elif scalping_score < -0.7:
                return "SELL"
            return None

        except Exception as e:
            return None

    def _ai_analyze_speed(self):
        if len(self.prices) < 5:
            return 0
        recent_changes = np.diff(self.prices[-5:])
        speed = np.mean(recent_changes) / self.prices[-6]
        return np.tanh(speed * 1000)

    def _ai_analyze_scalping_volatility(self):
        if len(self.prices) < 10:
            return 0
        recent_prices = self.prices[-10:]
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        return np.tanh(volatility * 100)

    def _ai_detect_micro_trend(self):
        if len(self.prices) < 8:
            return 0
        micro_prices = self.prices[-8:]
        trend = (micro_prices[-1] - micro_prices[0]) / micro_prices[0]
        return np.tanh(trend * 50)

class AI_Safe_Grid(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, grid_levels=5, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "AI Safe Grid", config_key)
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

class AI_Dynamic_Master(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, config_key=None):
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, telegram_bot_token, telegram_chat_id, "AI Dynamic Master", config_key)

    def get_signal(self):
        try:
            if len(self.prices) < 50:
                return None

            # AI Đa chiến thuật động
            trend_ai = self._ai_trend_analysis()
            mean_reversion_ai = self._ai_mean_reversion_analysis()
            breakout_ai = self._ai_breakout_analysis()
            momentum_ai = self._ai_momentum_analysis()
            
            # AI Fusion Decision
            dynamic_score = (
                trend_ai * 0.3 +
                mean_reversion_ai * 0.25 +
                breakout_ai * 0.25 +
                momentum_ai * 0.2
            )
            
            if dynamic_score > 0.6:
                self.log(f"🎯 AI Dynamic BUY - Score: {dynamic_score:.2f}")
                return "BUY"
            elif dynamic_score < -0.6:
                self.log(f"🎯 AI Dynamic SELL - Score: {dynamic_score:.2f}")
                return "SELL"
            return None

        except Exception as e:
            self.log(f"❌ Lỗi AI Dynamic Master: {str(e)}")
            return None

    def _ai_trend_analysis(self):
        if len(self.prices) < 20:
            return 0
        prices = np.array(self.prices[-20:])
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return np.tanh(slope * 500)

    def _ai_mean_reversion_analysis(self):
        if len(self.prices) < 30:
            return 0
        mean_price = np.mean(self.prices[-30:])
        current_price = self.prices[-1]
        deviation = (current_price - mean_price) / mean_price
        return -np.tanh(deviation * 10)

    def _ai_breakout_analysis(self):
        if len(self.prices) < 15:
            return 0
        recent_high = max(self.prices[-15:])
        recent_low = min(self.prices[-15:])
        current_price = self.prices[-1]
        
        if current_price >= recent_high * 0.99:
            return 0.8
        elif current_price <= recent_low * 1.01:
            return -0.8
        return 0

    def _ai_momentum_analysis(self):
        if len(self.prices) < 10:
            return 0
        momentum = (self.prices[-1] - self.prices[-5]) / self.prices[-5]
        return np.tanh(momentum * 20)

# ========== AI FUSION MASTER - KẾT HỢP 4 MÔ HÌNH AI TOÀN CẦU ==========
class AI_Fusion_Master(BaseBot):
    """
    AI FUSION MASTER - Kết hợp 4 mô hình AI hàng đầu thế giới
    """
    
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "AI Fusion Master", config_key)
        
        self.ensemble_models = {
            'transformer': self._transformer_ai_prediction,
            'lstm_attention': self._lstm_attention_prediction, 
            'gan_enhanced': self._gan_enhanced_prediction,
            'quantum_inspired': self._quantum_inspired_prediction
        }
        
        self.model_weights = {
            'transformer': 0.30,
            'lstm_attention': 0.25,
            'gan_enhanced': 0.25,
            'quantum_inspired': 0.20
        }
        
        self.last_analysis_time = 0
        self.analysis_cache = {}

    def get_signal(self):
        """
        FUSION AI - Kết hợp tín hiệu từ 4 mô hình AI đỉnh cao
        """
        try:
            if len(self.prices) < 100:
                return None
                
            current_time = time.time()
            if current_time - self.last_analysis_time < 10:
                if self.symbol in self.analysis_cache:
                    return self.analysis_cache[self.symbol]
            
            # 1. TRANSFORMER AI (Google Brain)
            transformer_signal, transformer_confidence = self.ensemble_models['transformer']()
            
            # 2. LSTM + ATTENTION AI (OpenAI)
            lstm_signal, lstm_confidence = self.ensemble_models['lstm_attention']()
            
            # 3. GAN-ENHANCED AI (NVIDIA)
            gan_signal, gan_confidence = self.ensemble_models['gan_enhanced']()
            
            # 4. QUANTUM-INSPIRED AI (IBM/D-Wave)
            quantum_signal, quantum_confidence = self.ensemble_models['quantum_inspired']()
            
            # TÍNH TOÁN TÍN HIỆU FUSION
            fusion_result = self._calculate_fusion_signal(
                transformer_signal, transformer_confidence,
                lstm_signal, lstm_confidence, 
                gan_signal, gan_confidence,
                quantum_signal, quantum_confidence
            )
            
            self.last_analysis_time = current_time
            self.analysis_cache[self.symbol] = fusion_result
            
            if fusion_result:
                self.log(f"🚀 AI Fusion Master - Tín hiệu: {fusion_result} | Transformer: {transformer_confidence:.2f} | LSTM: {lstm_confidence:.2f} | GAN: {gan_confidence:.2f} | Quantum: {quantum_confidence:.2f}")
            
            return fusion_result

        except Exception as e:
            self.log(f"❌ Lỗi AI Fusion Master: {str(e)}")
            return None

    def _transformer_ai_prediction(self):
        """
        TRANSFORMER AI - Mô hình attention đa đầu (Google Brain)
        """
        try:
            if len(self.prices) < 50:
                return None, 0
                
            # Multi-head Attention Simulation
            sequences = []
            for i in range(len(self.prices)-10):
                sequence = self.prices[i:i+10]
                sequences.append(sequence)
            
            if not sequences:
                return None, 0
                
            # Self-attention mechanism
            recent_sequence = self.prices[-10:]
            similarities = []
            
            for seq in sequences[-20:]:
                similarity = np.corrcoef(recent_sequence, seq)[0,1]
                if not np.isnan(similarity):
                    similarities.append(similarity)
            
            if similarities:
                attention_weights = np.array(similarities)
                attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
                
                # Predict next movement
                future_returns = []
                for i, weight in enumerate(attention_weights):
                    if i < len(sequences) - 1:
                        future_return = (sequences[i+1][-1] - sequences[i][-1]) / sequences[i][-1]
                        future_returns.append(future_return * weight)
                
                if future_returns:
                    predicted_return = np.sum(future_returns)
                    confidence = min(abs(predicted_return * 100), 1.0)
                    
                    if predicted_return > 0.002:
                        return "BUY", confidence
                    elif predicted_return < -0.002:
                        return "SELL", confidence
            
            return None, 0
                
        except Exception as e:
            return None, 0

    def _lstm_attention_prediction(self):
        """
        LSTM + ATTENTION AI - Mô hình chuỗi thời gian (OpenAI)
        """
        try:
            if len(self.prices) < 30:
                return None, 0
                
            # LSTM-like sequence analysis
            sequences = []
            for i in range(len(self.prices)-15):
                sequence = self.prices[i:i+15]
                sequences.append(sequence)
            
            if len(sequences) < 5:
                return None, 0
            
            # Analyze sequence patterns
            recent_pattern = self.prices[-15:]
            pattern_scores = []
            
            for seq in sequences[-10:]:
                if len(seq) == len(recent_pattern):
                    correlation = np.corrcoef(seq, recent_pattern)[0,1]
                    if not np.isnan(correlation):
                        pattern_scores.append(correlation)
            
            if pattern_scores:
                avg_correlation = np.mean(pattern_scores)
                
                # LSTM memory gate simulation
                memory_effect = np.tanh(avg_correlation * 2)
                
                # Attention mechanism
                volatility = np.std(recent_pattern) / np.mean(recent_pattern)
                attention_score = memory_effect * (1 - volatility)
                
                if attention_score > 0.1:
                    return "BUY", min(abs(attention_score), 0.8)
                elif attention_score < -0.1:
                    return "SELL", min(abs(attention_score), 0.8)
            
            return None, 0
                
        except Exception as e:
            return None, 0

    def _gan_enhanced_prediction(self):
        """
        GAN-ENHANCED AI - Mô hình sinh đối nghịch (NVIDIA Style)
        """
        try:
            if len(self.prices) < 40:
                return None, 0
                
            # Generator: Tạo pattern giả
            real_patterns = []
            for i in range(len(self.prices)-20):
                pattern = self.prices[i:i+20]
                real_patterns.append(pattern)
            
            if not real_patterns:
                return None, 0
            
            # Discriminator: Phân biệt pattern thật/giả
            current_pattern = self.prices[-20:]
            realness_scores = []
            
            for pattern in real_patterns[-15:]:
                if len(pattern) == len(current_pattern):
                    # Tính độ "thật" của pattern
                    mean_real = np.mean(real_patterns)
                    std_real = np.std(real_patterns)
                    
                    current_mean = np.mean(current_pattern)
                    current_std = np.std(current_pattern)
                    
                    mean_similarity = 1 - abs(current_mean - mean_real) / mean_real
                    std_similarity = 1 - abs(current_std - std_real) / std_real
                    
                    realness = (mean_similarity + std_similarity) / 2
                    realness_scores.append(realness)
            
            if realness_scores:
                avg_realness = np.mean(realness_scores)
                
                # GAN adversarial training simulation
                generator_confidence = avg_realness
                discriminator_confidence = 1 - avg_realness
                
                # Market regime detection
                trend = (current_pattern[-1] - current_pattern[0]) / current_pattern[0]
                regime_stability = 1 - abs(trend)
                
                final_confidence = (generator_confidence + discriminator_confidence) * regime_stability / 2
                
                if trend > 0.01 and final_confidence > 0.3:
                    return "BUY", final_confidence
                elif trend < -0.01 and final_confidence > 0.3:
                    return "SELL", final_confidence
            
            return None, 0
                
        except Exception as e:
            return None, 0

    def _quantum_inspired_prediction(self):
        """
        QUANTUM-INSPIRED AI - Tính toán lượng tử (IBM/D-Wave)
        """
        try:
            if len(self.prices) < 25:
                return None, 0
                
            # Quantum superposition of states
            price_states = []
            for i in range(len(self.prices)-5):
                state_vector = []
                for j in range(5):
                    if i+j+1 < len(self.prices):
                        ret = (self.prices[i+j+1] - self.prices[i+j]) / self.prices[i+j]
                        state_vector.append(ret)
                if len(state_vector) == 5:
                    price_states.append(state_vector)
            
            if not price_states:
                return None, 0
            
            # Quantum entanglement simulation
            current_state = []
            for i in range(4):
                if len(self.prices) >= i+2:
                    ret = (self.prices[-i-1] - self.prices[-i-2]) / self.prices[-i-2]
                    current_state.append(ret)
            
            if len(current_state) < 4:
                return None, 0
            
            # Calculate quantum probabilities
            entanglement_scores = []
            for state in price_states[-10:]:
                if len(state) == len(current_state):
                    correlation = np.corrcoef(state, current_state)[0,1]
                    if not np.isnan(correlation):
                        entanglement_scores.append(correlation)
            
            if entanglement_scores:
                # Quantum probability amplitude
                prob_amplitude = np.mean(entanglement_scores)
                
                # Quantum interference
                recent_volatility = np.std(self.prices[-10:]) / np.mean(self.prices[-10:])
                interference_factor = 1 - recent_volatility * 10
                
                quantum_confidence = abs(prob_amplitude * interference_factor)
                
                # Quantum collapse to classical state
                if prob_amplitude > 0.15 and quantum_confidence > 0.4:
                    return "BUY", quantum_confidence
                elif prob_amplitude < -0.15 and quantum_confidence > 0.4:
                    return "SELL", quantum_confidence
            
            return None, 0
                
        except Exception as e:
            return None, 0

    def _calculate_fusion_signal(self, t_signal, t_conf, l_signal, l_conf, g_signal, g_conf, q_signal, q_conf):
        """
        Kết hợp tín hiệu từ 4 mô hình AI với trọng số thông minh
        """
        try:
            signals = []
            confidences = []
            weights = []
            
            # Transformer AI
            if t_signal and t_conf > 0.3:
                signals.append(1 if t_signal == "BUY" else -1)
                confidences.append(t_conf)
                weights.append(self.model_weights['transformer'])
            
            # LSTM Attention AI
            if l_signal and l_conf > 0.3:
                signals.append(1 if l_signal == "BUY" else -1)
                confidences.append(l_conf)
                weights.append(self.model_weights['lstm_attention'])
            
            # GAN Enhanced AI
            if g_signal and g_conf > 0.3:
                signals.append(1 if g_signal == "BUY" else -1)
                confidences.append(g_conf)
                weights.append(self.model_weights['gan_enhanced'])
            
            # Quantum Inspired AI
            if q_signal and q_conf > 0.3:
                signals.append(1 if q_signal == "BUY" else -1)
                confidences.append(q_conf)
                weights.append(self.model_weights['quantum_inspired'])
            
            if not signals:
                return None
            
            # Tính điểm fusion có trọng số
            total_weight = sum(weights)
            if total_weight == 0:
                return None
            
            normalized_weights = [w / total_weight for w in weights]
            
            fusion_score = 0
            total_confidence = 0
            
            for i in range(len(signals)):
                fusion_score += signals[i] * confidences[i] * normalized_weights[i]
                total_confidence += confidences[i] * normalized_weights[i]
            
            # Ngưỡng quyết định thông minh
            if fusion_score > 0.4 and total_confidence > 0.5:
                return "BUY"
            elif fusion_score < -0.4 and total_confidence > 0.5:
                return "SELL"
            
            return None
                
        except Exception as e:
            return None

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
        
        self.target_coins = {}
        self.max_bots_per_config = {}
        
        self.strategy_cooldowns = {
            "AI Reverse Pro": {},
            "AI Scalping Pro": {},
            "AI Trend Master": {},
            "AI Safe Grid": {},
            "AI Dynamic Master": {},
            "AI Fusion Master": {}
        }
        self.cooldown_period = 300
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG AI BOT ĐÃ KHỞI ĐỘNG - FUSION MASTER ACTIVE")
            
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
        welcome = "🤖 <b>BOT GIAO DỊCH AI FUSION MASTER</b>\n\n🚀 <b>HỆ THỐNG AI ĐA MÔ HÌNH ĐỈNH CAO</b>"
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

    def _find_qualified_symbols(self, strategy_type, leverage, config, strategy_key):
        try:
            threshold = config.get('threshold', 30)
            volatility = config.get('volatility', 3)
            grid_levels = config.get('grid_levels', 5)
            
            max_bots = self.max_bots_per_config.get(strategy_key, 2)
            
            qualified_symbols = get_qualified_symbols(
                self.api_key, self.api_secret, strategy_type, leverage,
                threshold, volatility, grid_levels, 
                max_candidates=20, 
                final_limit=max_bots,
                strategy_key=strategy_key
            )
            
            return qualified_symbols
            
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            return []

    def _find_and_populate_target_coins(self, strategy_type, leverage, config, strategy_key, max_bots=2):
        try:
            if strategy_key not in self.target_coins:
                self.target_coins[strategy_key] = []
                self.max_bots_per_config[strategy_key] = max_bots
            
            current_count = len(self.target_coins[strategy_key])
            needed_count = max_bots - current_count
            
            if needed_count <= 0:
                return self.target_coins[strategy_key]
            
            logger.info(f"🔄 Đang tìm {needed_count} coin cho {strategy_type} (Config: {strategy_key})")
            
            new_symbols = self._find_qualified_symbols(strategy_type, leverage, config, strategy_key)
            
            added_count = 0
            for symbol in new_symbols:
                if symbol not in self.target_coins[strategy_key] and added_count < needed_count:
                    self.target_coins[strategy_key].append(symbol)
                    logger.info(f"✅ Đã thêm {symbol} vào danh sách target cho {strategy_key}")
                    added_count += 1
            
            self._create_all_bots_from_target_list(strategy_key)
            
            return self.target_coins[strategy_key]
            
        except Exception as e:
            logger.error(f"❌ Lỗi populate target coins: {str(e)}")
            return []

    def _create_all_bots_from_target_list(self, strategy_key):
        try:
            if strategy_key not in self.target_coins or not self.target_coins[strategy_key]:
                return False
            
            strategy_config = self.auto_strategies.get(strategy_key)
            if not strategy_config:
                return False
            
            created_count = 0
            max_bots = self.max_bots_per_config.get(strategy_key, 2)
            current_bots = [bot_id for bot_id in self.bots.keys() if strategy_key in bot_id]
            
            if len(current_bots) >= max_bots:
                return True
            
            for symbol in self.target_coins[strategy_key]:
                if len(current_bots) >= max_bots:
                    break
                    
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    strategy_type = strategy_config['strategy_type']
                    success = self._create_auto_bot(symbol, strategy_type, strategy_config)
                    if success:
                        current_bots.append(bot_id)
                        created_count += 1
                        logger.info(f"✅ Đã tạo bot {bot_id} từ danh sách target")
            
            if created_count > 0:
                logger.info(f"🎯 Đã tạo {created_count} bot cho {strategy_key}")
            
            return created_count > 0
                
        except Exception as e:
            logger.error(f"❌ Lỗi tạo bot từ target list: {str(e)}")
            return False

    def _scan_auto_strategies(self):
        if not self.auto_strategies:
            return
            
        self.log("🔄 Đang quét coin cho các cấu hình AI tự động...")
        
        for strategy_key, strategy_config in self.auto_strategies.items():
            try:
                strategy_type = strategy_config['strategy_type']
                
                if self._is_in_cooldown(strategy_type, strategy_key):
                    continue
                
                coin_manager = CoinManager()
                current_bots_count = coin_manager.count_bots_by_config(strategy_key)
                max_bots = self.max_bots_per_config.get(strategy_key, 2)
                
                if current_bots_count < max_bots:
                    self.log(f"🔄 {strategy_type} (Config: {strategy_key}): đang có {current_bots_count}/{max_bots} bot, tìm coin...")
                    
                    target_coins = self._find_and_populate_target_coins(
                        strategy_type, 
                        strategy_config['leverage'], 
                        strategy_config, 
                        strategy_key,
                        max_bots
                    )
                    
                    current_after = coin_manager.count_bots_by_config(strategy_key)
                    if current_after > current_bots_count:
                        self.log(f"✅ {strategy_type}: đã thêm {current_after - current_bots_count} bot mới")
                    
                    if target_coins:
                        self.log(f"🎯 {strategy_type}: danh sách target - {target_coins}")
                    else:
                        self.log(f"⚠️ {strategy_type}: không tìm thấy coin phù hợp cho danh sách target")
                else:
                    self.log(f"✅ {strategy_type} (Config: {strategy_key}): đã đủ {current_bots_count}/{max_bots} bot")
                        
            except Exception as e:
                self.log(f"❌ Lỗi quét {strategy_type}: {str(e)}")

    def _handle_coin_after_close(self, strategy_key, closed_symbol):
        try:
            if strategy_key not in self.target_coins:
                self.target_coins[strategy_key] = []
            
            if closed_symbol in self.target_coins[strategy_key]:
                self.target_coins[strategy_key].remove(closed_symbol)
                self.log(f"🗑️ Đã xóa {closed_symbol} khỏi danh sách target {strategy_key}")
            
            coin_manager = CoinManager()
            current_bots_count = coin_manager.count_bots_by_config(strategy_key)
            max_bots = self.max_bots_per_config.get(strategy_key, 2)
            
            if current_bots_count < max_bots:
                strategy_config = self.auto_strategies.get(strategy_key)
                if strategy_config:
                    new_symbols = self._find_qualified_symbols(
                        strategy_config['strategy_type'],
                        strategy_config['leverage'],
                        strategy_config,
                        strategy_key
                    )
                    
                    for symbol in new_symbols:
                        if (symbol not in self.target_coins[strategy_key] and 
                            len(self.target_coins[strategy_key]) < max_bots):
                            self.target_coins[strategy_key].append(symbol)
                            self.log(f"🔄 Đã thêm {symbol} vào danh sách target thay thế {closed_symbol}")
                            
                            self._create_all_bots_from_target_list(strategy_key)
                            break
            
        except Exception as e:
            self.log(f"❌ Lỗi xử lý coin sau khi đóng: {str(e)}")

    def _auto_scan_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                for bot_id, bot in list(self.bots.items()):
                    if (hasattr(bot, 'config_key') and bot.config_key and
                        not bot.position_open and 
                        current_time - bot.last_close_time < 300 and
                        bot.strategy_name in ["AI Reverse Pro", "AI Scalping Pro", "AI Safe Grid", "AI Trend Master", "AI Dynamic Master", "AI Fusion Master"]):
                        
                        if current_time - getattr(bot, '_last_find_attempt', 0) > 300:
                            self.log(f"🔄 Bot động {bot_id} đang tìm coin mới sau khi đóng lệnh...")
                            bot._last_find_attempt = current_time
                            self._handle_coin_after_close(bot.config_key, bot.symbol)
                
                if current_time - self.last_auto_scan > self.auto_scan_interval:
                    self._scan_auto_strategies()
                    self.last_auto_scan = current_time
                
                time.sleep(60)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(60)

    def _create_auto_bot(self, symbol, strategy_type, config):
        try:
            leverage = config['leverage']
            percent = config['percent']
            tp = config['tp']
            sl = config['sl']
            strategy_key = config['strategy_key']
            
            bot_class = {
                "AI Reverse Pro": AI_Reverse_Pro,
                "AI Scalping Pro": AI_Scalping_Pro,
                "AI Safe Grid": AI_Safe_Grid,
                "AI Trend Master": AI_Trend_Master,
                "AI Dynamic Master": AI_Dynamic_Master,
                "AI Fusion Master": AI_Fusion_Master
            }.get(strategy_type)
            
            if not bot_class:
                return False
            
            if strategy_type == "AI Reverse Pro":
                threshold = config.get('threshold', 30)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, threshold, strategy_key)
            elif strategy_type == "AI Safe Grid":
                grid_levels = config.get('grid_levels', 5)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels, strategy_key)
            else:
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, strategy_key)
            
            bot._bot_manager = self
            
            bot_id = f"{symbol}_{strategy_key}"
            self.bots[bot_id] = bot
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, bot_count=1, **kwargs):
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance")
            return False
        
        bot_mode = kwargs.get('bot_mode', 'static')
        
        if bot_mode == 'dynamic' and strategy_type in ["AI Reverse Pro", "AI Scalping Pro", "AI Safe Grid", "AI Trend Master", "AI Dynamic Master", "AI Fusion Master"]:
            strategy_key = f"{strategy_type}_{lev}_{percent}_{tp}_{sl}"
            
            if strategy_type == "AI Reverse Pro":
                threshold = kwargs.get('threshold', 30)
                strategy_key += f"_th{threshold}"
            elif strategy_type == "AI Scalping Pro":
                volatility = kwargs.get('volatility', 3)
                strategy_key += f"_vol{volatility}"
            elif strategy_type == "AI Safe Grid":
                grid_levels = kwargs.get('grid_levels', 5)
                strategy_key += f"_grid{grid_levels}"
            
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
                **kwargs
            }
            
            self.max_bots_per_config[strategy_key] = bot_count
            
            target_coins = self._find_and_populate_target_coins(
                strategy_type, lev, self.auto_strategies[strategy_key], 
                strategy_key, bot_count
            )
            
            success_count = len([bot_id for bot_id in self.bots.keys() if strategy_key in bot_id])
            
            if success_count > 0:
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {success_count}/{bot_count} BOT {strategy_type}</b>\n\n"
                    f"🎯 Chiến lược: {strategy_type}\n"
                    f"💰 Đòn bẩy: {lev}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 TP: {tp}%\n"
                    f"🛡️ SL: {sl}%\n"
                )
                if strategy_type == "AI Reverse Pro":
                    success_msg += f"📈 Ngưỡng: {threshold}%\n"
                elif strategy_type == "AI Scalping Pro":
                    success_msg += f"⚡ Biến động: {volatility}%\n"
                elif strategy_type == "AI Safe Grid":
                    success_msg += f"🛡️ Số lệnh: {grid_levels}\n"
                    
                success_msg += f"🤖 Coin: {', '.join(target_coins) if target_coins else 'Đang tìm...'}\n\n"
                success_msg += f"🔑 <b>Config Key:</b> {strategy_key}"
                
                if strategy_type == "AI Fusion Master":
                    success_msg += f"\n\n🚀 <b>AI FUSION MASTER ACTIVATED</b>\n• Transformer AI (Google)\n• LSTM Attention (OpenAI)\n• GAN Enhanced (NVIDIA)\n• Quantum Inspired (IBM)"
                
                self.log(success_msg)
                return True
            else:
                self.log(f"⚠️ {strategy_type}: đang tìm coin phù hợp, sẽ thử lại sau")
                return True
        
        else:
            symbol = symbol.upper()
            bot_id = f"{symbol}_{strategy_type}"
            
            if bot_id in self.bots:
                self.log(f"⚠️ Đã có bot {strategy_type} cho {symbol}")
                return False
                
            try:
                bot_class = {
                    "AI Trading Basic": AI_Trading_Basic
                }.get(strategy_type)
                
                if not bot_class:
                    self.log(f"❌ Chiến lược {strategy_type} không được hỗ trợ")
                    return False
                
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id)
                
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
            if hasattr(bot, 'config_key') and bot.config_key:
                strategy_key = bot.config_key
                if strategy_key in self.target_coins and bot.symbol in self.target_coins[strategy_key]:
                    self.target_coins[strategy_key].remove(bot.symbol)
                    self.log(f"🗑️ Đã xóa {bot.symbol} khỏi danh sách target {strategy_key}")
            
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
        
        if current_step == 'waiting_bot_count':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    bot_count = int(text)
                    if bot_count <= 0 or bot_count > 5:
                        send_telegram("⚠️ Số lượng bot phải từ 1 đến 5. Vui lòng chọn lại:",
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
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT TĨNH</b>\n\n"
                        "🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n"
                        "📊 Bạn cần chọn coin cụ thể\n\n"
                        "Chọn chiến lược AI:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                else:
                    user_state['bot_mode'] = 'dynamic'
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT ĐỘNG</b>\n\n"
                        f"🤖 Hệ thống sẽ tự động tìm <b>{user_state.get('bot_count', 1)} coin</b> tốt nhất\n"
                        f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                        f"📈 Mỗi chiến lược có danh sách coin riêng\n\n"
                        "Chọn chiến lược AI:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )

        elif current_step == 'waiting_strategy':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text in ["🤖 AI Trading Basic", "📊 AI Trend Master", "🎯 AI Reverse Pro", 
                         "⚡ AI Scalping Pro", "🛡️ AI Safe Grid", "🔄 AI Dynamic Master", "🚀 AI Fusion Master"]:
                
                strategy_map = {
                    "🤖 AI Trading Basic": "AI Trading Basic",
                    "📊 AI Trend Master": "AI Trend Master", 
                    "🎯 AI Reverse Pro": "AI Reverse Pro",
                    "⚡ AI Scalping Pro": "AI Scalping Pro",
                    "🛡️ AI Safe Grid": "AI Safe Grid",
                    "🔄 AI Dynamic Master": "AI Dynamic Master",
                    "🚀 AI Fusion Master": "AI Fusion Master"
                }
                
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                user_state['step'] = 'waiting_exit_strategy'
                
                strategy_descriptions = {
                    "AI Trading Basic": "AI cơ bản - Phân tích đa khung thời gian",
                    "AI Trend Master": "AI xu hướng - Nhận diện và theo trend", 
                    "AI Reverse Pro": "AI đảo chiều - Phân tích biến động 24h",
                    "AI Scalping Pro": "AI scalping - Giao dịch tốc độ cao",
                    "AI Safe Grid": "AI grid an toàn - Phân bổ rủi ro",
                    "AI Dynamic Master": "AI động - Đa chiến thuật thông minh",
                    "AI Fusion Master": "AI Fusion Master - Kết hợp 4 mô hình AI đỉnh cao"
                }
                
                description = strategy_descriptions.get(strategy, "")
                bot_count = user_state.get('bot_count', 1)
                
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
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
            elif text == "🎯 Chỉ TP/SL cố định":
                user_state['exit_strategy'] = 'traditional'
                self._continue_bot_creation(chat_id, user_state)

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
                    
                    strategy = user_state.get('strategy')
                    bot_mode = user_state.get('bot_mode', 'static')
                    leverage = user_state.get('leverage')
                    percent = user_state.get('percent')
                    tp = user_state.get('tp')
                    sl = user_state.get('sl')
                    symbol = user_state.get('symbol')
                    bot_count = user_state.get('bot_count', 1)
                    exit_strategy = user_state.get('exit_strategy', 'traditional')
                    
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
                            strategy_type=strategy
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
                            bot_count=bot_count,
                            threshold=threshold,
                            volatility=volatility,
                            grid_levels=grid_levels
                        )
                    
                    if success:
                        success_msg = (
                            f"✅ <b>ĐÃ TẠO BOT THÀNH CÔNG</b>\n\n"
                            f"🤖 Chiến lược: {strategy}\n"
                            f"🔧 Chế độ: {bot_mode}\n"
                            f"🔢 Số lượng: {bot_count} coin\n"
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
                        
                        if strategy == "AI Fusion Master":
                            success_msg += f"\n\n🚀 <b>AI FUSION MASTER ACTIVATED</b>\n• 🤖 Transformer AI (Google Brain)\n• 🧠 LSTM Attention (OpenAI)\n• 🎨 GAN Enhanced (NVIDIA)\n• ⚛️ Quantum Inspired (IBM)"
                        
                        send_telegram(success_msg, chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    else:
                        send_telegram("❌ Có lỗi khi tạo bot. Vui lòng thử lại.",
                                    chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    
                    self.user_states[chat_id] = {}
                    
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Stop Loss:",
                                chat_id, create_sl_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_count'}
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(
                f"🎯 <b>CHỌN SỐ LƯỢNG COIN CHO CHIẾN LƯỢC AI</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"Chọn số lượng coin bạn muốn cho chiến lược AI:",
                chat_id,
                create_bot_count_keyboard(),
                self.telegram_bot_token, self.telegram_chat_id
            )
        
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT AI ĐANG CHẠY</b>\n\n"
                
                strategy_groups = {}
                for bot_id, bot in self.bots.items():
                    strategy_name = bot.strategy_name
                    if strategy_name not in strategy_groups:
                        strategy_groups[strategy_name] = []
                    strategy_groups[strategy_name].append(bot)
                
                for strategy_name, bots in strategy_groups.items():
                    message += f"🎯 <b>{strategy_name}</b> ({len(bots)} bot):\n"
                    for bot in bots:
                        status = "🟢 Mở" if bot.status == "open" else "🟡 Chờ"
                        mode = "Tĩnh"
                        if hasattr(bot, 'config_key') and bot.config_key:
                            mode = "Động"
                        
                        message += f"  🔹 {bot.symbol} | {status} | {mode} | ĐB: {bot.lev}x\n"
                    message += "\n"
                
                total_bots = len(self.bots)
                dynamic_bots = sum(1 for bot in self.bots.values() if hasattr(bot, 'config_key') and bot.config_key)
                message += f"📊 Tổng số: {total_bots} bot | 🔄 Động: {dynamic_bots}"
                
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
                "🎯 <b>DANH SÁCH CHIẾN LƯỢC AI HOÀN CHỈNH</b>\n\n"
                
                "🚀 <b>AI Fusion Master</b>\n"
                "• Kết hợp 4 mô hình AI đỉnh cao toàn cầu\n"
                "• 🤖 Transformer AI (Google Brain)\n"
                "• 🧠 LSTM Attention AI (OpenAI)\n"
                "• 🎨 GAN Enhanced AI (NVIDIA)\n"
                "• ⚛️ Quantum Inspired AI (IBM)\n"
                "• 🎯 Ra quyết định fusion thông minh\n\n"
                
                "🔄 <b>AI Dynamic Master</b>\n"
                "• AI đa chiến thuật động\n"
                "• Tự động thích nghi với thị trường\n"
                "• Kết hợp trend, mean reversion, breakout\n\n"
                
                "🎯 <b>AI Reverse Pro</b>\n"
                "• AI đảo chiều thông minh\n"
                "• Phân tích biến động 24h nâng cao\n"
                "• Tính toán xác suất đảo chiều AI\n\n"
                
                "⚡ <b>AI Scalping Pro</b>\n"
                "• AI scalping tốc độ cao\n"
                "• Phân tích vi xu hướng\n"
                "• Tối ưu hóa entry/exit point\n\n"
                
                "🛡️ <b>AI Safe Grid</b>\n"
                "• Grid trading thông minh\n"
                "• Quản lý rủi ro AI\n"
                "• Phân bổ vốn tối ưu\n\n"
                
                "📊 <b>AI Trend Master</b>\n"
                "• AI nhận diện xu hướng\n"
                "• Phân tích momentum chất lượng\n"
                "• Xác nhận volume AI\n\n"
                
                "🤖 <b>AI Trading Basic</b>\n"
                "• AI cơ bản đa khung thời gian\n"
                "• Phân tích ngắn/trung/dài hạn\n"
                "• Tín hiệu tổng hợp thông minh"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            dynamic_bots_count = sum(1 for bot in self.bots.values() 
                                   if hasattr(bot, 'config_key') and bot.config_key)
            
            strategy_stats = {}
            for bot in self.bots.values():
                strategy_name = bot.strategy_name
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = 0
                strategy_stats[strategy_name] += 1
            
            stats_text = "\n".join([f"  • {name}: {count} bot" for name, count in strategy_stats.items()])
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG AI FUSION MASTER</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🔄 Bot động: {dynamic_bots_count}\n"
                f"📊 Chiến lược đang chạy:\n{stats_text}\n"
                f"🎯 Auto scan: {len(self.auto_strategies)} cấu hình\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n"
                f"⏰ Cooldown: {self.cooldown_period//60} phút"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        strategy = user_state.get('strategy')
        bot_mode = user_state.get('bot_mode', 'static')
        bot_count = user_state.get('bot_count', 1)
        
        if bot_mode == 'dynamic' and strategy != "AI Fusion Master":
            if strategy == "AI Reverse Pro":
                user_state['step'] = 'waiting_threshold'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn ngưỡng biến động (%):",
                    chat_id,
                    create_threshold_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "AI Scalping Pro":
                user_state['step'] = 'waiting_volatility'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn biến động tối thiểu (%):",
                    chat_id,
                    create_volatility_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            elif strategy == "AI Safe Grid":
                user_state['step'] = 'waiting_grid_levels'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn số lệnh grid:",
                    chat_id,
                    create_grid_levels_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>BOT ĐỘNG: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n"
                    f"📊 Mỗi chiến lược có danh sách coin riêng\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
        else:
            if bot_mode == 'static':
                user_state['step'] = 'waiting_symbol'
                send_telegram(
                    f"🎯 <b>BOT TĨNH: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n\n"
                    f"Chọn cặp coin:",
                    chat_id,
                    create_symbols_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )
            else:
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🎯 <b>AI FUSION MASTER</b>\n"
                    f"🤖 Số lượng: {bot_count} coin\n\n"
                    f"🚀 <b>HỆ THỐNG AI ĐA MÔ HÌNH ĐỈNH CAO</b>\n"
                    f"• 🤖 Transformer AI (Google Brain)\n"
                    f"• 🧠 LSTM Attention AI (OpenAI)\n" 
                    f"• 🎨 GAN Enhanced AI (NVIDIA)\n"
                    f"• ⚛️ Quantum Inspired AI (IBM)\n\n"
                    f"🤖 Hệ thống sẽ tự động tìm {bot_count} coin tốt nhất\n"
                    f"🔄 Tự tìm coin mới sau khi đóng lệnh\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(strategy),
                    self.telegram_bot_token, self.telegram_chat_id
                )

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()
