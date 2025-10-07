# trading_bot_lib.py - HOÀN CHỈNH VỚI BOT ĐỘNG TỰ TÌM COIN MỚI SAU KHI ĐÓNG LỆNH
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

# ========== SMART EXIT MANAGER ==========
class SmartExitManager:
    """QUẢN LÝ THÔNG MINH 4 CƠ CHẾ ĐÓNG LỆNH"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.config = {
            'enable_trailing': False,
            'enable_time_exit': False,
            'enable_volume_exit': False,
            'enable_support_resistance': False,
            'trailing_activation': 30,
            'trailing_distance': 15,
            'max_hold_time': 6,
            'min_profit_for_exit': 10
        }
        
        self.trailing_active = False
        self.peak_price = 0
        self.position_open_time = 0
        self.volume_history = []
        
    def update_config(self, **kwargs):
        """Cập nhật cấu hình từ người dùng"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        self.bot.log(f"⚙️ Cập nhật Smart Exit: {self.config}")
    
    def check_all_exit_conditions(self, current_price, current_volume=None):
        """KIỂM TRA TẤT CẢ ĐIỀU KIỆN ĐÓNG LỆNH"""
        if not self.bot.position_open:
            return None
            
        exit_reasons = []
        
        # 1. TRAILING STOP EXIT
        if self.config['enable_trailing']:
            reason = self._check_trailing_stop(current_price)
            if reason:
                exit_reasons.append(reason)
        
        # 2. TIME-BASED EXIT
        if self.config['enable_time_exit']:
            reason = self._check_time_exit()
            if reason:
                exit_reasons.append(reason)
        
        # 3. VOLUME-BASED EXIT  
        if self.config['enable_volume_exit'] and current_volume:
            reason = self._check_volume_exit(current_volume)
            if reason:
                exit_reasons.append(reason)
        
        # 4. SUPPORT/RESISTANCE EXIT
        if self.config['enable_support_resistance']:
            reason = self._check_support_resistance(current_price)
            if reason:
                exit_reasons.append(reason)
        
        # Chỉ đóng lệnh nếu đang có lãi đạt ngưỡng tối thiểu
        if exit_reasons:
            current_roi = self._calculate_roi(current_price)
            if current_roi >= self.config['min_profit_for_exit']:
                return f"Smart Exit: {' + '.join(exit_reasons)} | Lãi: {current_roi:.1f}%"
        
        return None
    
    def _check_trailing_stop(self, current_price):
        """Trailing Stop - Bảo vệ lợi nhuận"""
        current_roi = self._calculate_roi(current_price)
        
        # Kích hoạt trailing khi đạt ngưỡng
        if current_roi >= self.config['trailing_activation'] and not self.trailing_active:
            self.trailing_active = True
            self.peak_price = current_price
            self.bot.log(f"🟢 Kích hoạt Trailing Stop | Lãi {current_roi:.1f}%")
        
        # Cập nhật đỉnh mới
        if self.trailing_active:
            if (self.bot.side == "BUY" and current_price > self.peak_price) or \
               (self.bot.side == "SELL" and current_price < self.peak_price):
                self.peak_price = current_price
            
            # Tính drawdown từ đỉnh
            if self.bot.side == "BUY":
                drawdown = ((self.peak_price - current_price) / self.peak_price) * 100
            else:
                drawdown = ((current_price - self.peak_price) / self.peak_price) * 100
            
            if drawdown >= self.config['trailing_distance']:
                return f"Trailing(dd:{drawdown:.1f}%)"
        
        return None
    
    def _check_time_exit(self):
        """Time-based Exit - Giới hạn thời gian giữ lệnh"""
        if self.position_open_time == 0:
            return None
            
        holding_hours = (time.time() - self.position_open_time) / 3600
        
        if holding_hours >= self.config['max_hold_time']:
            return f"Time({holding_hours:.1f}h)"
        
        return None
    
    def _check_volume_exit(self, current_volume):
        """Volume-based Exit - Theo dấu hiệu volume"""
        if len(self.volume_history) < 5:
            self.volume_history.append(current_volume)
            return None
        
        avg_volume = sum(self.volume_history[-5:]) / 5
        
        if current_volume < avg_volume * 0.4:
            return "Volume(giảm 60%)"
        
        self.volume_history.append(current_volume)
        if len(self.volume_history) > 10:
            self.volume_history.pop(0)
            
        return None
    
    def _check_support_resistance(self, current_price):
        """Support/Resistance Exit - Theo key levels"""
        if self.bot.side == "BUY":
            target_profit = 5.0
            target_price = self.bot.entry * (1 + target_profit/100)
            
            if current_price >= target_price:
                return f"Resistance(+{target_profit}%)"
        
        return None
    
    def _calculate_roi(self, current_price):
        """Tính ROI hiện tại"""
        if self.bot.side == "BUY":
            return ((current_price - self.bot.entry) / self.bot.entry) * 100
        else:
            return ((self.bot.entry - current_price) / self.bot.entry) * 100
    
    def on_position_opened(self):
        """Khi mở position mới"""
        self.trailing_active = False
        self.peak_price = self.bot.entry
        self.position_open_time = time.time()
        self.volume_history = []

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
    """Bàn phím chọn chiến lược thoát lệnh"""
    return {
        "keyboard": [
            [{"text": "🔄 Thoát lệnh thông minh"}, {"text": "⚡ Thoát lệnh cơ bản"}],
            [{"text": "🎯 Chỉ TP/SL cố định"}, {"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_smart_exit_config_keyboard():
    """Bàn phím cấu hình Smart Exit"""
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
    """Bàn phím chọn chế độ bot"""
    return {
        "keyboard": [
            [{"text": "🤖 Bot Tĩnh - Coin cụ thể"}, {"text": "🔄 Bot Động - Tự tìm coin"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard(strategy=None):
    """Bàn phím chọn coin"""
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
    """Bàn phím chọn đòn bẩy"""
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
    """Bàn phím chọn % số dư"""
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
    """Bàn phím chọn Take Profit"""
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
    """Bàn phím chọn Stop Loss"""
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

# ========== QUẢN LÝ COIN CHUNG ==========
class CoinManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CoinManager, cls).__new__(cls)
                cls._instance.managed_coins = {}
                cls._instance.position_coins = set()
        return cls._instance
    
    def register_coin(self, symbol, bot_id, strategy, config_key=None):
        with self._lock:
            if symbol not in self.managed_coins:
                self.managed_coins[symbol] = {
                    "strategy": strategy, 
                    "bot_id": bot_id,
                    "config_key": config_key
                }
                return True
            return False
    
    def unregister_coin(self, symbol):
        with self._lock:
            if symbol in self.managed_coins:
                del self.managed_coins[symbol]
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
    """Lấy danh sách coin có biến động 24h cao nhất từ toàn bộ Binance"""
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
                    
                    if abs(change) >= threshold and volume > 1000000:
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
    """Tìm coin phù hợp từ TOÀN BỘ Binance - PHÂN BIỆT THEO CẤU HÌNH"""
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
                
            # Loại trừ BTC và ETH để tránh biến động quá cao
            if symbol in ['BTCUSDT', 'ETHUSDT']:
                continue
            
            # Kiểm tra coin đã được quản lý bởi config này chưa
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
                
                # ĐIỀU KIỆN CHO TỪNG CHIẾN LƯỢC - LINH HOẠT HƠN
                if strategy_type == "Reverse 24h":
                    if abs_price_change >= (threshold or 15) and volume > 1000000:
                        score = abs_price_change * (volume / 1000000)
                        qualified_symbols.append((symbol, score, price_change))
                
                elif strategy_type == "Scalping":
                    if abs_price_change >= (volatility or 2) and volume > 2000000 and price_range >= 1.0:
                        qualified_symbols.append((symbol, price_range))
                
                elif strategy_type == "Safe Grid":
                    if 0.5 <= abs_price_change <= 8.0 and volume > 500000:
                        qualified_symbols.append((symbol, -abs(price_change - 3.0)))
                
                elif strategy_type == "Trend Following":
                    # ĐIỀU KIỆN MỞ RỘNG CHO TREND FOLLOWING
                    if (1.0 <= abs_price_change <= 15.0 and 
                        volume > 1000000 and 
                        price_range >= 0.5):
                        score = volume * abs_price_change  # Ưu tiên volume cao + biến động
                        qualified_symbols.append((symbol, score))
                
                elif strategy_type == "Smart Dynamic":
                    # ĐIỀU KIỆN THÔNG MINH LINH HOẠT
                    if (1.0 <= abs_price_change <= 12.0 and
                        volume > 1500000 and
                        price_range >= 0.8):
                        # Tính điểm tổng hợp
                        volume_score = min(volume / 5000000, 5)
                        volatility_score = min(abs_price_change / 10, 3)
                        score = volume_score + volatility_score
                        qualified_symbols.append((symbol, score))
                        
            except (ValueError, TypeError) as e:
                continue
        
        # SẮP XẾP THEO CHIẾN LƯỢC
        if strategy_type == "Reverse 24h":
            qualified_symbols.sort(key=lambda x: x[1], reverse=True)
        elif strategy_type == "Scalping":
            qualified_symbols.sort(key=lambda x: x[1], reverse=True)
        elif strategy_type == "Safe Grid":
            qualified_symbols.sort(key=lambda x: x[1], reverse=True)
        elif strategy_type == "Trend Following":
            qualified_symbols.sort(key=lambda x: x[1], reverse=True)
        elif strategy_type == "Smart Dynamic":
            qualified_symbols.sort(key=lambda x: x[1], reverse=True)
        
        # LOG CHI TIẾT ĐỂ DEBUG
        logger.info(f"🔍 {strategy_type}: Quét {len(all_symbols)} coin, tìm thấy {len(qualified_symbols)} phù hợp")
        
        final_symbols = []
        for item in qualified_symbols[:max_candidates]:
            if len(final_symbols) >= final_limit:
                break
                
            if strategy_type == "Reverse 24h":
                symbol, score, original_change = item
            else:
                symbol, score = item
                
            try:
                leverage_success = set_leverage(symbol, leverage, api_key, api_secret)
                step_size = get_step_size(symbol, api_key, api_secret)
                
                if leverage_success and step_size > 0:
                    final_symbols.append(symbol)
                    if strategy_type == "Reverse 24h":
                        logger.info(f"✅ {symbol}: phù hợp {strategy_type} (Biến động: {original_change:.2f}%, Điểm: {score:.2f})")
                    else:
                        logger.info(f"✅ {symbol}: phù hợp {strategy_type} (Score: {score:.2f})")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Lỗi kiểm tra {symbol}: {str(e)}")
                continue
        
        # BACKUP SYSTEM: Nếu không tìm thấy coin phù hợp, lấy coin có volume cao nhất
        if not final_symbols:
            logger.warning(f"⚠️ {strategy_type}: không tìm thấy coin phù hợp, sử dụng backup method")
            backup_symbols = []
            
            for symbol in all_symbols:
                if symbol not in ticker_dict:
                    continue
                    
                # Kiểm tra coin đã được quản lý bởi config này chưa
                if strategy_key and coin_manager.has_same_config_bot(symbol, strategy_key):
                    continue
                    
                ticker = ticker_dict[symbol]
                try:
                    volume = float(ticker.get('quoteVolume', 0))
                    price_change = float(ticker.get('priceChangePercent', 0))
                    abs_price_change = abs(price_change)
                    
                    # Điều kiện backup: volume cao, biến động vừa phải, không quá mạnh
                    if (volume > 3000000 and 
                        0.5 <= abs_price_change <= 10.0 and
                        symbol not in ['BTCUSDT', 'ETHUSDT']):
                        backup_symbols.append((symbol, volume, abs_price_change))
                except:
                    continue
            
            # Sắp xếp theo volume giảm dần
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
        
        # FINAL CHECK: Nếu vẫn không có coin, thử các coin phổ biến
        if not final_symbols:
            logger.error(f"❌ {strategy_type}: không thể tìm thấy coin nào phù hợp sau backup")
            popular_symbols = ["BNBUSDT", "ADAUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "EOSUSDT"]
            
            for symbol in popular_symbols:
                if len(final_symbols) >= final_limit:
                    break
                    
                try:
                    if symbol in ticker_dict:
                        leverage_success = set_leverage(symbol, leverage, api_key, api_secret)
                        step_size = get_step_size(symbol, api_key, api_secret)
                        
                        if leverage_success and step_size > 0:
                            final_symbols.append(symbol)
                            logger.info(f"🚨 {symbol}: sử dụng coin phổ biến (backup cuối)")
                except:
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
                    'side': 'BUY' if float(pos.get('positionAmt', 0)) > 0 else 'SELL',
                    'size': abs(float(pos.get('positionAmt', 0))),
                    'entry': float(pos.get('entryPrice', 0)),
                    'pnl': float(pos.get('unrealizedProfit', 0))
                })
        return positions
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
    return []

def close_position(symbol, api_key, api_secret):
    try:
        positions = get_positions(symbol, api_key, api_secret)
        if not positions:
            return False
        
        for pos in positions:
            side = 'SELL' if pos['side'] == 'BUY' else 'BUY'
            place_order(symbol, side, pos['size'], api_key, api_secret)
        
        return True
    except Exception as e:
        logger.error(f"Lỗi đóng vị thế: {str(e)}")
    return False

# ========== CHIẾN LƯỢC GIAO DỊCH ==========
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gains = np.mean(gains[:period])
    avg_losses = np.mean(losses[:period])
    if avg_losses == 0:
        return 100
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period):
    if len(prices) < period:
        return np.mean(prices) if prices else 0
    ema = np.mean(prices[:period])
    multiplier = 2 / (period + 1)
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_macd(prices):
    if len(prices) < 26:
        return 0, 0
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema([macd_line], 9)
    return macd_line, signal_line

def calculate_bollinger_bands(prices, period=20):
    if len(prices) < period:
        return 0, 0, 0
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, sma, lower

def calculate_support_resistance(prices, window=50):
    if len(prices) < window:
        return 0, 0
    resistance = max(prices[-window:])
    support = min(prices[-window:])
    return support, resistance

# ========== BOT GIAO DỊCH CHÍNH ==========
class TradingBot:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.symbol = config.get('symbol', 'BTCUSDT')
        self.leverage = config.get('leverage', 10)
        self.quantity_percent = config.get('quantity_percent', 10)
        self.take_profit = config.get('take_profit', 100)
        self.stop_loss = config.get('stop_loss', 50)
        self.strategy_type = config.get('strategy_type', 'RSI/EMA Recursive')
        self.threshold = config.get('threshold', 30)
        self.volatility = config.get('volatility', 3)
        self.grid_levels = config.get('grid_levels', 5)
        self.exit_strategy = config.get('exit_strategy', 'smart')
        self.bot_id = config.get('bot_id', 'default')
        self.telegram_chat_id = config.get('telegram_chat_id')
        self.telegram_bot_token = config.get('telegram_bot_token')
        self.is_running = False
        self.position_open = False
        self.side = None
        self.entry = 0
        self.quantity = 0
        self.position_value = 0
        self.prices = []
        self.volumes = []
        self.last_signal = None
        self.last_update = 0
        self.ws = None
        self.ws_thread = None
        self.analysis_thread = None
        self.coin_manager = CoinManager()
        self.smart_exit = SmartExitManager(self)
        self.dynamic_mode = config.get('dynamic_mode', False)
        self.strategy_key = f"{self.bot_id}_{self.strategy_type}_{self.leverage}"
        
        # Cấu hình Smart Exit mặc định
        smart_exit_config = {
            'enable_trailing': True,
            'enable_time_exit': True, 
            'enable_volume_exit': True,
            'enable_support_resistance': True,
            'trailing_activation': 30,
            'trailing_distance': 15,
            'max_hold_time': 6,
            'min_profit_for_exit': 10
        }
        self.smart_exit.update_config(**smart_exit_config)
        
        self.log(f"🤖 Bot {self.bot_id} khởi tạo: {self.symbol} | {self.strategy_type} | Đòn bẩy {self.leverage}x")

    def log(self, message):
        logger.info(f"[{self.bot_id}] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                send_telegram(f"<code>[{self.bot_id}]</code> {message}", 
                            self.telegram_chat_id, 
                            bot_token=self.telegram_bot_token)
            except:
                pass

    def start(self):
        if self.is_running:
            self.log("⚠️ Bot đang chạy")
            return False
        
        # Đăng ký coin với coin manager
        if not self.dynamic_mode:
            registered = self.coin_manager.register_coin(
                self.symbol, self.bot_id, self.strategy_type, self.strategy_key
            )
            if not registered:
                self.log(f"❌ Coin {self.symbol} đã được quản lý bởi bot khác")
                return False
        
        self.is_running = True
        
        # Thiết lập đòn bẩy
        if not set_leverage(self.symbol, self.leverage, self.api_key, self.api_secret):
            self.log("❌ Không thể thiết lập đòn bẩy")
            self.is_running = False
            return False
        
        # Kiểm tra số dư
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None:
            self.log("❌ Không thể kết nối tới Binance")
            self.is_running = False
            return False
        
        self.log(f"💰 Số dư: {balance:.2f} USDT")
        
        # Kiểm tra vị thế hiện tại
        positions = get_positions(self.symbol, self.api_key, self.api_secret)
        if positions:
            self.position_open = True
            self.side = positions[0]['side']
            self.entry = positions[0]['entry']
            self.quantity = positions[0]['size']
            self.position_value = self.entry * self.quantity
            self.smart_exit.on_position_opened()
            self.log(f"📖 Đã có vị thế: {self.side} {self.quantity} {self.symbol} @ {self.entry}")
        
        # Bắt đầu WebSocket
        self.start_websocket()
        
        self.log(f"🚀 Bot bắt đầu chạy: {self.symbol} | {self.strategy_type}")
        return True

    def stop(self):
        self.is_running = False
        if self.ws:
            self.ws.close()
        if not self.dynamic_mode:
            self.coin_manager.unregister_coin(self.symbol)
        self.log("🛑 Bot dừng")

    def restart(self):
        self.stop()
        time.sleep(2)
        return self.start()

    def start_websocket(self):
        stream_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@kline_1m"
        self.ws = websocket.WebSocketApp(stream_url,
                                        on_open=self.on_open,
                                        on_message=self.on_message,
                                        on_error=self.on_error,
                                        on_close=self.on_close)
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def on_open(self, ws):
        self.log(f"🔗 Kết nối WebSocket: {self.symbol}")

    def on_error(self, ws, error):
        self.log(f"❌ Lỗi WebSocket: {str(error)}")

    def on_close(self, ws, close_status_code, close_msg):
        self.log("🔌 Đóng kết nối WebSocket")
        if self.is_running:
            self.log("🔄 Tự động kết nối lại WebSocket sau 5 giây...")
            time.sleep(5)
            self.start_websocket()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'k' in data:
                kline = data['k']
                is_closed = kline['x']
                close_price = float(kline['c'])
                volume = float(kline['v'])
                
                self.prices.append(close_price)
                if len(self.prices) > 100:
                    self.prices.pop(0)
                
                self.volumes.append(volume)
                if len(self.volumes) > 20:
                    self.volumes.pop(0)
                
                if is_closed and len(self.prices) >= 20:
                    self.analyze_and_trade(close_price, volume)
                    
        except Exception as e:
            self.log(f"❌ Lỗi xử lý message: {str(e)}")

    def analyze_and_trade(self, price, volume):
        if not self.is_running:
            return
        
        # Phân tích và đưa ra tín hiệu
        signal = self.generate_signal()
        
        if signal and signal != self.last_signal:
            self.last_signal = signal
            self.execute_trade(signal, price)
        
        # Kiểm tra điều kiện đóng lệnh
        if self.position_open:
            self.check_exit_conditions(price, volume)

    def generate_signal(self):
        if len(self.prices) < 20:
            return None
        
        current_price = self.prices[-1]
        
        if self.strategy_type == "RSI/EMA Recursive":
            return self.rsi_ema_strategy()
        elif self.strategy_type == "EMA Crossover":
            return self.ema_crossover_strategy()
        elif self.strategy_type == "Reverse 24h":
            return self.reverse_24h_strategy()
        elif self.strategy_type == "Trend Following":
            return self.trend_following_strategy()
        elif self.strategy_type == "Scalping":
            return self.scalping_strategy()
        elif self.strategy_type == "Safe Grid":
            return self.safe_grid_strategy()
        elif self.strategy_type == "Smart Dynamic":
            return self.smart_dynamic_strategy()
        
        return None

    def rsi_ema_strategy(self):
        rsi = calculate_rsi(self.prices, 14)
        ema20 = calculate_ema(self.prices, 20)
        ema50 = calculate_ema(self.prices, 50)
        current_price = self.prices[-1]
        
        if rsi < 30 and current_price > ema20 and ema20 > ema50:
            return "BUY"
        elif rsi > 70 and current_price < ema20 and ema20 < ema50:
            return "SELL"
        return None

    def ema_crossover_strategy(self):
        if len(self.prices) < 50:
            return None
        
        ema9 = calculate_ema(self.prices, 9)
        ema21 = calculate_ema(self.prices, 21)
        prev_ema9 = calculate_ema(self.prices[:-1], 9)
        prev_ema21 = calculate_ema(self.prices[:-1], 21)
        
        if prev_ema9 <= prev_ema21 and ema9 > ema21:
            return "BUY"
        elif prev_ema9 >= prev_ema21 and ema9 < ema21:
            return "SELL"
        return None

    def reverse_24h_strategy(self):
        if len(self.prices) < 50:
            return None
        
        price_change_24h = ((self.prices[-1] - self.prices[0]) / self.prices[0]) * 100
        rsi = calculate_rsi(self.prices, 14)
        
        if price_change_24h <= -self.threshold and rsi < 35:
            return "BUY"
        elif price_change_24h >= self.threshold and rsi > 65:
            return "SELL"
        return None

    def trend_following_strategy(self):
        ema20 = calculate_ema(self.prices, 20)
        ema50 = calculate_ema(self.prices, 50)
        current_price = self.prices[-1]
        
        if current_price > ema20 and ema20 > ema50:
            return "BUY"
        elif current_price < ema20 and ema20 < ema50:
            return "SELL"
        return None

    def scalping_strategy(self):
        rsi = calculate_rsi(self.prices, 14)
        macd_line, signal_line = calculate_macd(self.prices)
        
        if rsi < 25 and macd_line > signal_line:
            return "BUY"
        elif rsi > 75 and macd_line < signal_line:
            return "SELL"
        return None

    def safe_grid_strategy(self):
        current_price = self.prices[-1]
        grid_range = self.volatility / 100
        
        # Tính toán các mức grid
        if not hasattr(self, 'grid_levels_set'):
            self.setup_grid_levels(current_price, grid_range)
        
        # Kiểm tra các mức grid
        for level in self.grid_levels:
            if abs(current_price - level['price']) / level['price'] < 0.002:  # 0.2%
                if level['type'] == 'buy' and not level['triggered']:
                    level['triggered'] = True
                    return "BUY"
                elif level['type'] == 'sell' and not level['triggered']:
                    level['triggered'] = True
                    return "SELL"
        return None

    def setup_grid_levels(self, current_price, grid_range):
        self.grid_levels_set = True
        self.grid_levels = []
        
        for i in range(self.grid_levels):
            buy_price = current_price * (1 - (i + 1) * grid_range)
            sell_price = current_price * (1 + (i + 1) * grid_range)
            
            self.grid_levels.append({'price': buy_price, 'type': 'buy', 'triggered': False})
            self.grid_levels.append({'price': sell_price, 'type': 'sell', 'triggered': False})

    def smart_dynamic_strategy(self):
        # Chiến lược thông minh kết hợp nhiều indicator
        rsi = calculate_rsi(self.prices, 14)
        ema20 = calculate_ema(self.prices, 20)
        ema50 = calculate_ema(self.prices, 50)
        macd_line, signal_line = calculate_macd(self.prices)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(self.prices)
        current_price = self.prices[-1]
        
        score = 0
        
        # RSI
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
        
        # EMA
        if current_price > ema20 and ema20 > ema50:
            score += 1
        elif current_price < ema20 and ema20 < ema50:
            score -= 1
        
        # MACD
        if macd_line > signal_line:
            score += 1
        else:
            score -= 1
        
        # Bollinger Bands
        if current_price <= lower_bb:
            score += 1
        elif current_price >= upper_bb:
            score -= 1
        
        if score >= 3:
            return "BUY"
        elif score <= -3:
            return "SELL"
        
        return None

    def execute_trade(self, signal, price):
        if self.position_open:
            return
        
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None or balance <= 0:
            self.log("❌ Số dư không đủ")
            return
        
        # Tính khối lượng
        self.quantity = (balance * (self.quantity_percent / 100) * self.leverage) / price
        step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
        
        if step_size > 0:
            self.quantity = round(self.quantity / step_size) * step_size
        
        if self.quantity <= 0:
            self.log("❌ Khối lượng quá nhỏ")
            return
        
        # Đặt lệnh
        order_result = place_order(self.symbol, signal, self.quantity, self.api_key, self.api_secret)
        
        if order_result and 'orderId' in order_result:
            self.position_open = True
            self.side = signal
            self.entry = price
            self.position_value = self.entry * self.quantity
            
            # Khởi tạo Smart Exit
            self.smart_exit.on_position_opened()
            
            message = (f"🎯 <b>MỞ LỆNH {signal}</b>\n"
                      f"🏷️ {self.symbol}\n"
                      f"💰 Khối lượng: {self.quantity:.4f}\n" 
                      f"🎚️ Giá vào: ${self.entry:.4f}\n"
                      f"💵 Giá trị: ${self.position_value:.2f}\n"
                      f"🎯 TP: {self.take_profit}% | 🛑 SL: {self.stop_loss}%")
            
            self.log(message)
            
            # Gửi Telegram
            if self.telegram_bot_token and self.telegram_chat_id:
                send_telegram(message, self.telegram_chat_id, bot_token=self.telegram_bot_token)
        else:
            self.log(f"❌ Lỗi đặt lệnh {signal}")

    def check_exit_conditions(self, current_price, volume=None):
        if not self.position_open:
            return
        
        # 1. KIỂM TRA SMART EXIT
        if self.exit_strategy == 'smart':
            exit_reason = self.smart_exit.check_all_exit_conditions(current_price, volume)
            if exit_reason:
                self.close_position_with_reason(exit_reason, current_price)
                return
        
        # 2. KIỂM TRA TP/SL CƠ BẢN
        roi = self.calculate_roi(current_price)
        
        if roi >= self.take_profit:
            self.close_position_with_reason(f"Take Profit {roi:.1f}%", current_price)
            return
        
        if roi <= -self.stop_loss:
            self.close_position_with_reason(f"Stop Loss {roi:.1f}%", current_price)
            return

    def calculate_roi(self, current_price):
        if self.side == "BUY":
            return ((current_price - self.entry) / self.entry) * 100
        else:
            return ((self.entry - current_price) / self.entry) * 100

    def close_position_with_reason(self, reason, current_price):
        try:
            # Đóng vị thế
            success = close_position(self.symbol, self.api_key, self.api_secret)
            
            if success:
                roi = self.calculate_roi(current_price)
                
                message = (f"🏁 <b>ĐÓNG LỆNH</b>\n"
                          f"🏷️ {self.symbol}\n" 
                          f"📊 Lý do: {reason}\n"
                          f"💰 Lãi/lỗ: {roi:.2f}%\n"
                          f"🎚️ Giá vào: ${self.entry:.4f}\n"
                          f"🎚️ Giá ra: ${current_price:.4f}")
                
                self.log(message)
                
                # Gửi Telegram
                if self.telegram_bot_token and self.telegram_chat_id:
                    send_telegram(message, self.telegram_chat_id, bot_token=self.telegram_bot_token)
                
                # Reset trạng thái
                self.position_open = False
                self.side = None
                self.entry = 0
                self.quantity = 0
                self.position_value = 0
                
                # BOT ĐỘNG: Tìm coin mới sau khi đóng lệnh
                if self.dynamic_mode:
                    self.log("🔄 Bot động: Đang tìm coin mới...")
                    self.find_new_coin_after_exit()
                
            else:
                self.log("❌ Lỗi đóng vị thế")
                
        except Exception as e:
            self.log(f"❌ Lỗi khi đóng lệnh: {str(e)}")

    def find_new_coin_after_exit(self):
        """TÌM COIN MỚI CHO BOT ĐỘNG SAU KHI ĐÓNG LỆNH"""
        try:
            self.log("🔄 Bot động đang tìm coin mới...")
            
            # Tìm coin phù hợp
            new_symbols = get_qualified_symbols(
                self.api_key, 
                self.api_secret,
                self.strategy_type,
                self.leverage,
                threshold=self.threshold,
                volatility=self.volatility, 
                grid_levels=self.grid_levels,
                max_candidates=10,
                final_limit=1,
                strategy_key=self.strategy_key
            )
            
            if new_symbols:
                new_symbol = new_symbols[0]
                
                # Hủy đăng ký coin cũ
                self.coin_manager.unregister_coin(self.symbol)
                
                # Cập nhật symbol mới
                old_symbol = self.symbol
                self.symbol = new_symbol
                
                # Đăng ký coin mới
                registered = self.coin_manager.register_coin(
                    new_symbol, self.bot_id, self.strategy_type, self.strategy_key
                )
                
                if registered:
                    # Khởi động lại WebSocket với coin mới
                    self.restart_websocket_for_new_coin()
                    
                    message = f"🔄 Bot động chuyển từ {old_symbol} → {new_symbol}"
                    self.log(message)
                    
                    if self.telegram_bot_token and self.telegram_chat_id:
                        send_telegram(message, self.telegram_chat_id, bot_token=self.telegram_bot_token)
                else:
                    self.log(f"❌ Không thể đăng ký coin mới {new_symbol}")
                    # Quay lại coin cũ nếu không đăng ký được
                    self.symbol = old_symbol
                    self.coin_manager.register_coin(old_symbol, self.bot_id, self.strategy_type, self.strategy_key)
            else:
                self.log("❌ Không tìm thấy coin mới phù hợp, giữ nguyên coin hiện tại")
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin mới: {str(e)}")
            traceback.print_exc()

    def restart_websocket_for_new_coin(self):
        """Khởi động lại WebSocket cho coin mới"""
        try:
            if self.ws:
                self.ws.close()
            
            time.sleep(2)
            self.start_websocket()
            self.log(f"🔗 Khởi động lại WebSocket cho {self.symbol}")
            
        except Exception as e:
            self.log(f"❌ Lỗi khởi động lại WebSocket: {str(e)}")

# ========== QUẢN LÝ BOT ==========
class BotManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BotManager, cls).__new__(cls)
                cls._instance.bots = {}
                cls._instance.user_states = {}
                cls._instance.user_configs = {}
        return cls._instance
    
    def add_bot(self, bot_id, config):
        with self._lock:
            if bot_id in self.bots:
                return False
            
            bot = TradingBot(config)
            if bot.start():
                self.bots[bot_id] = bot
                return True
            return False
    
    def remove_bot(self, bot_id):
        with self._lock:
            if bot_id in self.bots:
                self.bots[bot_id].stop()
                del self.bots[bot_id]
                return True
            return False
    
    def get_bot(self, bot_id):
        with self._lock:
            return self.bots.get(bot_id)
    
    def get_all_bots(self):
        with self._lock:
            return self.bots.copy()
    
    def set_user_state(self, user_id, state, data=None):
        with self._lock:
            self.user_states[user_id] = {'state': state, 'data': data or {}}
    
    def get_user_state(self, user_id):
        with self._lock:
            return self.user_states.get(user_id, {'state': None, 'data': {}})
    
    def clear_user_state(self, user_id):
        with self._lock:
            if user_id in self.user_states:
                del self.user_states[user_id]
    
    def save_user_config(self, user_id, config_key, config):
        with self._lock:
            if user_id not in self.user_configs:
                self.user_configs[user_id] = {}
            self.user_configs[user_id][config_key] = config
    
    def get_user_config(self, user_id, config_key):
        with self._lock:
            if user_id in self.user_configs:
                return self.user_configs[user_id].get(config_key)
            return None

# ========== HÀM XỬ LÝ TELEGRAM ==========
def handle_telegram_message(message_data, bot_token):
    try:
        if 'message' not in message_data:
            return
        
        message = message_data['message']
        user_id = message['from']['id']
        text = message.get('text', '')
        chat_id = message['chat']['id']
        
        bot_manager = BotManager()
        user_state = bot_manager.get_user_state(user_id)
        
        # Xử lý lệnh hủy
        if text == "❌ Hủy bỏ":
            bot_manager.clear_user_state(user_id)
            send_telegram("✅ Đã hủy thao tác", chat_id, create_main_menu(), bot_token)
            return
        
        # Xử lý theo state
        if user_state['state']:
            handle_user_state(user_id, text, chat_id, user_state, bot_token)
        else:
            handle_main_menu(text, user_id, chat_id, bot_token)
            
    except Exception as e:
        logger.error(f"Lỗi xử lý Telegram: {str(e)}")
        traceback.print_exc()

def handle_user_state(user_id, text, chat_id, user_state, bot_token):
    bot_manager = BotManager()
    state = user_state['state']
    data = user_state['data']
    
    if state == 'waiting_strategy':
        handle_strategy_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_bot_mode':
        handle_bot_mode_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_symbol':
        handle_symbol_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_leverage':
        handle_leverage_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_quantity':
        handle_quantity_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_tp':
        handle_tp_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_sl':
        handle_sl_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_exit_strategy':
        handle_exit_strategy_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_smart_exit_config':
        handle_smart_exit_config(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_threshold':
        handle_threshold_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_volatility':
        handle_volatility_selection(user_id, text, chat_id, data, bot_token)
    
    elif state == 'waiting_grid_levels':
        handle_grid_levels_selection(user_id, text, chat_id, data, bot_token)

def handle_main_menu(text, user_id, chat_id, bot_token):
    bot_manager = BotManager()
    
    if text == "📊 Danh sách Bot":
        show_bot_list(chat_id, bot_token)
    
    elif text == "➕ Thêm Bot":
        start_bot_creation(user_id, chat_id, bot_token)
    
    elif text == "⛔ Dừng Bot":
        stop_bot_selection(user_id, chat_id, bot_token)
    
    elif text == "💰 Số dư":
        show_balance(user_id, chat_id, bot_token)
    
    elif text == "📈 Vị thế":
        show_positions(user_id, chat_id, bot_token)
    
    elif text == "⚙️ Cấu hình":
        show_config_menu(chat_id, bot_token)
    
    elif text == "🎯 Chiến lược":
        send_telegram("Chọn chiến lược giao dịch:", chat_id, create_strategy_keyboard(), bot_token)

def start_bot_creation(user_id, chat_id, bot_token):
    bot_manager = BotManager()
    bot_manager.set_user_state(user_id, 'waiting_bot_mode', {})
    
    message = (
        "🤖 <b>CHỌN CHẾ ĐỘ BOT</b>\n\n"
        "• <b>Bot Tĩnh</b>: Giao dịch 1 coin cố định\n"
        "• <b>Bot Động</b>: Tự động tìm coin mới sau mỗi lệnh\n\n"
        "Chọn chế độ:"
    )
    
    send_telegram(message, chat_id, create_bot_mode_keyboard(), bot_token)

def handle_bot_mode_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if "Bot Tĩnh" in text:
        data['dynamic_mode'] = False
        bot_manager.set_user_state(user_id, 'waiting_strategy', data)
        send_telegram("Chọn chiến lược giao dịch:", chat_id, create_strategy_keyboard(), bot_token)
    
    elif "Bot Động" in text:
        data['dynamic_mode'] = True
        bot_manager.set_user_state(user_id, 'waiting_strategy', data)
        
        message = (
            "🔄 <b>BOT ĐỘNG THÔNG MINH</b>\n\n"
            "Bot sẽ tự động:\n"
            "• Tìm coin có biến động tốt\n"
            "• Chuyển coin sau khi đóng lệnh\n"
            "• Tối ưu hóa lợi nhuận\n\n"
            "Chọn chiến lược:"
        )
        
        send_telegram(message, chat_id, create_strategy_keyboard(), bot_token)
    
    else:
        send_telegram("Vui lòng chọn chế độ bot:", chat_id, create_bot_mode_keyboard(), bot_token)

def handle_strategy_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
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
        data['strategy_type'] = strategy_map[text]
        bot_manager.set_user_state(user_id, 'waiting_symbol', data)
        
        if data.get('dynamic_mode', False):
            # Bot động: bỏ qua chọn coin, chuyển thẳng đến đòn bẩy
            handle_symbol_selection(user_id, "AUTO", chat_id, data, bot_token)
        else:
            send_telegram("Chọn coin giao dịch:", chat_id, create_symbols_keyboard(), bot_token)
    
    else:
        send_telegram("Vui lòng chọn chiến lược:", chat_id, create_strategy_keyboard(), bot_token)

def handle_symbol_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if data.get('dynamic_mode', False):
        # Bot động: tự động chọn coin
        data['symbol'] = "AUTO"
        bot_manager.set_user_state(user_id, 'waiting_leverage', data)
        send_telegram("Chọn đòn bẩy:", chat_id, create_leverage_keyboard(), bot_token)
    else:
        if text.endswith('USDT'):
            data['symbol'] = text
            bot_manager.set_user_state(user_id, 'waiting_leverage', data)
            send_telegram("Chọn đòn bẩy:", chat_id, create_leverage_keyboard(), bot_token)
        else:
            send_telegram("Vui lòng chọn coin hợp lệ:", chat_id, create_symbols_keyboard(), bot_token)

def handle_leverage_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if text.endswith('x') and text[:-1].isdigit():
        leverage = int(text[:-1])
        data['leverage'] = leverage
        bot_manager.set_user_state(user_id, 'waiting_quantity', data)
        send_telegram("Chọn % số dư cho mỗi lệnh:", chat_id, create_percent_keyboard(), bot_token)
    else:
        send_telegram("Vui lòng chọn đòn bẩy hợp lệ:", chat_id, create_leverage_keyboard(), bot_token)

def handle_quantity_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if text.isdigit():
        quantity_percent = int(text)
        data['quantity_percent'] = quantity_percent
        bot_manager.set_user_state(user_id, 'waiting_tp', data)
        send_telegram("Chọn Take Profit (%):", chat_id, create_tp_keyboard(), bot_token)
    else:
        send_telegram("Vui lòng chọn % số dư hợp lệ:", chat_id, create_percent_keyboard(), bot_token)

def handle_tp_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if text.isdigit():
        take_profit = int(text)
        data['take_profit'] = take_profit
        bot_manager.set_user_state(user_id, 'waiting_sl', data)
        send_telegram("Chọn Stop Loss (%):", chat_id, create_sl_keyboard(), bot_token)
    else:
        send_telegram("Vui lòng chọn Take Profit hợp lệ:", chat_id, create_tp_keyboard(), bot_token)

def handle_sl_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if text.isdigit():
        stop_loss = int(text)
        data['stop_loss'] = stop_loss
        bot_manager.set_user_state(user_id, 'waiting_exit_strategy', data)
        
        message = (
            "🎯 <b>CHỌN CHIẾN LƯỢC THOÁT LỆNH</b>\n\n"
            "• <b>Thoát lệnh thông minh</b>: Kết hợp 4 cơ chế\n"
            "• <b>Thoát lệnh cơ bản</b>: Chỉ dùng TP/SL\n"
            "• <b>Chỉ TP/SL cố định</b>: Cơ bản nhất\n\n"
            "Chọn chiến lược:"
        )
        
        send_telegram(message, chat_id, create_exit_strategy_keyboard(), bot_token)
    else:
        send_telegram("Vui lòng chọn Stop Loss hợp lệ:", chat_id, create_sl_keyboard(), bot_token)

def handle_exit_strategy_selection(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    if "thông minh" in text.lower():
        data['exit_strategy'] = 'smart'
        bot_manager.set_user_state(user_id, 'waiting_smart_exit_config', data)
        
        message = (
            "🔄 <b>CẤU HÌNH SMART EXIT</b>\n\n"
            "Chọn cấu hình thoát lệnh thông minh:\n\n"
            "• <b>Trailing: 30/15</b>: Kích hoạt 30%, distance 15%\n"
            "• <b>Trailing: 50/20</b>: Kích hoạt 50%, distance 20%\n"
            "• <b>Time Exit: 4h</b>: Giới hạn 4 giờ\n" 
            "• <b>Time Exit: 8h</b>: Giới hạn 8 giờ\n"
            "• <b>Kết hợp Full</b>: Tất cả cơ chế\n"
            "• <b>Cơ bản</b>: Trailing + Time\n"
        )
        
        send_telegram(message, chat_id, create_smart_exit_config_keyboard(), bot_token)
    
    elif "cơ bản" in text.lower():
        data['exit_strategy'] = 'basic'
        complete_bot_creation(user_id, chat_id, data, bot_token)
    
    elif "tpsl" in text.lower():
        data['exit_strategy'] = 'tpsl_only'
        complete_bot_creation(user_id, chat_id, data, bot_token)
    
    else:
        send_telegram("Vui lòng chọn chiến lược thoát lệnh:", chat_id, create_exit_strategy_keyboard(), bot_token)

def handle_smart_exit_config(user_id, text, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    smart_configs = {
        "Trailing: 30/15": {'trailing_activation': 30, 'trailing_distance': 15, 'max_hold_time': 6},
        "Trailing: 50/20": {'trailing_activation': 50, 'trailing_distance': 20, 'max_hold_time': 8},
        "Time Exit: 4h": {'trailing_activation': 25, 'trailing_distance': 10, 'max_hold_time': 4},
        "Time Exit: 8h": {'trailing_activation': 35, 'trailing_distance': 12, 'max_hold_time': 8},
        "Kết hợp Full": {'trailing_activation': 30, 'trailing_distance': 15, 'max_hold_time': 6},
        "Cơ bản": {'trailing_activation': 20, 'trailing_distance': 10, 'max_hold_time': 4}
    }
    
    if text in smart_configs:
        data['smart_exit_config'] = smart_configs[text]
        complete_bot_creation(user_id, chat_id, data, bot_token)
    else:
        send_telegram("Vui lòng chọn cấu hình Smart Exit:", chat_id, create_smart_exit_config_keyboard(), bot_token)

def complete_bot_creation(user_id, chat_id, data, bot_token):
    bot_manager = BotManager()
    
    try:
        # Lấy API keys từ user config
        user_config = bot_manager.get_user_config(user_id, 'api_keys')
        if not user_config:
            send_telegram("❌ Chưa thiết lập API Keys. Vui lòng cấu hình trước.", chat_id, create_main_menu(), bot_token)
            bot_manager.clear_user_state(user_id)
            return
        
        # Tạo config bot hoàn chỉnh
        bot_id = f"bot_{int(time.time())}"
        
        bot_config = {
            'bot_id': bot_id,
            'api_key': user_config['api_key'],
            'api_secret': user_config['api_secret'],
            'symbol': data.get('symbol', 'BTCUSDT'),
            'leverage': data.get('leverage', 10),
            'quantity_percent': data.get('quantity_percent', 10),
            'take_profit': data.get('take_profit', 100),
            'stop_loss': data.get('stop_loss', 50),
            'strategy_type': data.get('strategy_type', 'RSI/EMA Recursive'),
            'exit_strategy': data.get('exit_strategy', 'smart'),
            'telegram_chat_id': chat_id,
            'telegram_bot_token': bot_token,
            'dynamic_mode': data.get('dynamic_mode', False)
        }
        
        # Thêm các tham số chiến lược
        if data['strategy_type'] == "Reverse 24h":
            bot_config['threshold'] = data.get('threshold', 30)
        elif data['strategy_type'] == "Scalping":
            bot_config['volatility'] = data.get('volatility', 3)
        elif data['strategy_type'] == "Safe Grid":
            bot_config['grid_levels'] = data.get('grid_levels', 5)
        
        # Thêm cấu hình Smart Exit
        if data.get('smart_exit_config'):
            bot_config['smart_exit_config'] = data['smart_exit_config']
        
        # Tạo và khởi động bot
        success = bot_manager.add_bot(bot_id, bot_config)
        
        if success:
            # Bot động: tìm coin ngay lập tức
            if data.get('dynamic_mode', False):
                bot = bot_manager.get_bot(bot_id)
                if bot:
                    threading.Thread(target=bot.find_new_coin_after_exit, daemon=True).start()
            
            message = (
                f"✅ <b>BOT KHỞI ĐỘNG THÀNH CÔNG</b>\n\n"
                f"🆔 <b>ID</b>: {bot_id}\n"
                f"🎯 <b>Chiến lược</b>: {data['strategy_type']}\n"
                f"💱 <b>Coin</b>: {bot_config['symbol']}\n"
                f"⚖️ <b>Đòn bẩy</b>: {data['leverage']}x\n"
                f"💰 <b>Khối lượng</b>: {data['quantity_percent']}%\n"
                f"🎯 <b>TP/SL</b>: {data['take_profit']}%/{data['stop_loss']}%\n"
                f"🔄 <b>Thoát lệnh</b>: {data['exit_strategy']}\n"
                f"🤖 <b>Chế độ</b>: {'ĐỘNG' if data.get('dynamic_mode') else 'TĨNH'}\n\n"
                f"<i>Bot đã bắt đầu theo dõi thị trường...</i>"
            )
        else:
            message = "❌ Không thể khởi động bot. Kiểm tra lại cấu hình."
        
        send_telegram(message, chat_id, create_main_menu(), bot_token)
        bot_manager.clear_user_state(user_id)
        
    except Exception as e:
        logger.error(f"Lỗi tạo bot: {str(e)}")
        send_telegram("❌ Lỗi khi tạo bot. Vui lòng thử lại.", chat_id, create_main_menu(), bot_token)
        bot_manager.clear_user_state(user_id)

def show_bot_list(chat_id, bot_token):
    bot_manager = BotManager()
    bots = bot_manager.get_all_bots()
    
    if not bots:
        send_telegram("🤖 Không có bot nào đang chạy", chat_id, create_main_menu(), bot_token)
        return
    
    message = "📊 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
    
    for bot_id, bot in bots.items():
        status = "🟢 Đang chạy" if bot.is_running else "🔴 Dừng"
        position_status = f"📈 {bot.side} {bot.quantity:.4f}" if bot.position_open else "📭 Không có lệnh"
        
        message += (
            f"🆔 <b>{bot_id}</b>\n"
            f"🏷️ {bot.symbol} | ⚖️ {bot.leverage}x\n" 
            f"🎯 {bot.strategy_type}\n"
            f"📊 {position_status}\n"
            f"🔧 {status}\n"
            f"{'-'*20}\n"
        )
    
    send_telegram(message, chat_id, create_main_menu(), bot_token)

def stop_bot_selection(user_id, chat_id, bot_token):
    bot_manager = BotManager()
    bots = bot_manager.get_all_bots()
    
    if not bots:
        send_telegram("🤖 Không có bot nào đang chạy", chat_id, create_main_menu(), bot_token)
        return
    
    keyboard = []
    for bot_id in bots.keys():
        keyboard.append([{"text": f"⛔ Dừng {bot_id}"}])
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    reply_markup = {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True}
    send_telegram("Chọn bot để dừng:", chat_id, reply_markup, bot_token)

def show_balance(user_id, chat_id, bot_token):
    bot_manager = BotManager()
    user_config = bot_manager.get_user_config(user_id, 'api_keys')
    
    if not user_config:
        send_telegram("❌ Chưa thiết lập API Keys", chat_id, create_main_menu(), bot_token)
        return
    
    balance = get_balance(user_config['api_key'], user_config['api_secret'])
    
    if balance is None:
        send_telegram("❌ Không thể kết nối tới Binance", chat_id, create_main_menu(), bot_token)
        return
    
    message = f"💰 <b>SỐ DƯ TÀI KHOẢN</b>\n\n💵 <b>{balance:.2f} USDT</b>"
    send_telegram(message, chat_id, create_main_menu(), bot_token)

def show_positions(user_id, chat_id, bot_token):
    bot_manager = BotManager()
    user_config = bot_manager.get_user_config(user_id, 'api_keys')
    
    if not user_config:
        send_telegram("❌ Chưa thiết lập API Keys", chat_id, create_main_menu(), bot_token)
        return
    
    positions = get_positions(api_key=user_config['api_key'], api_secret=user_config['api_secret'])
    
    if not positions:
        send_telegram("📭 Không có vị thế nào đang mở", chat_id, create_main_menu(), bot_token)
        return
    
    message = "📈 <b>VỊ THẾ ĐANG MỞ</b>\n\n"
    
    for pos in positions:
        pnl_color = "🟢" if pos['pnl'] >= 0 else "🔴"
        message += (
            f"🏷️ <b>{pos['symbol']}</b>\n"
            f"📊 {pos['side']} | Khối lượng: {pos['size']:.4f}\n"
            f"🎯 Giá vào: ${pos['entry']:.4f}\n"
            f"💰 PnL: {pnl_color} ${pos['pnl']:.2f}\n"
            f"{'-'*15}\n"
        )
    
    send_telegram(message, chat_id, create_main_menu(), bot_token)

def show_config_menu(chat_id, bot_token):
    message = (
        "⚙️ <b>CẤU HÌNH HỆ THỐNG</b>\n\n"
        "Các tính năng đang phát triển...\n\n"
        "📊 <b>Bot Manager</b>: Quản lý đa bot\n"
        "🔄 <b>Smart Exit</b>: 4 cơ chế thoát lệnh\n"
        "🎯 <b>Dynamic Coin</b>: Tự động tìm coin\n"
        "📈 <b>Multi Strategy</b>: 7 chiến lược\n\n"
        "<i>Phiên bản hoàn chỉnh</i>"
    )
    
    send_telegram(message, chat_id, create_main_menu(), bot_token)
