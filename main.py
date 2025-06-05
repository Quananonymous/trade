import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
import threading
import time
import json
import os
import numpy as np
from datetime import datetime
import math
import requests
import socket
import ssl

# File config chỉ chứa API keys
try:
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, USE_TESTNET
except ImportError:
    BINANCE_API_KEY = ""
    BINANCE_SECRET_KEY = ""
    USE_TESTNET = True  # Mặc định dùng testnet

class MobileCoin:
    def __init__(self, name, logger):
        self.name = name
        self.price = 0.0
        self.max_leverage = 20
        self.min_order_amount = 10.0
        self.min_qty = 0.001
        self.step_size = 0.001
        self.last_update_time = 0
        self.data_valid = False
        self.logger = logger
        
    def update_data(self):
        try:
            # Giới hạn request: 5 giây giữa các lần cập nhật
            current_time = time.time()
            if current_time - self.last_update_time < 5:
                return self.data_valid
                
            # Tạo client với timeout rõ ràng
            client = Client(
                BINANCE_API_KEY, 
                BINANCE_SECRET_KEY, 
                testnet=USE_TESTNET,
                requests_params={'timeout': 10}
            )
            
            # Cập nhật giá hiện tại
            ticker = client.get_symbol_ticker(symbol=self.name)
            self.price = float(ticker['price'])
            
            # Lấy thông số hợp đồng (chỉ lần đầu)
            if not hasattr(self, 'symbol_info'):
                exchange_info = client.futures_exchange_info()
                for symbol in exchange_info['symbols']:
                    if symbol['symbol'] == self.name:
                        self.symbol_info = symbol
                        self.logger(f"Đã tải thông tin hợp đồng cho {self.name}")
                        break
                
                if hasattr(self, 'symbol_info'):
                    for f in self.symbol_info['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            self.step_size = float(f['stepSize'])
                            self.min_qty = float(f['minQty'])
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            self.min_order_amount = float(f['minNotional'])
            
            self.last_update_time = current_time
            self.data_valid = True
            return True
        except BinanceAPIException as e:
            self.logger(f"Lỗi API Binance: {e.status_code} - {e.message}")
            self.data_valid = False
            return False
        except (requests.exceptions.ConnectionError, socket.gaierror) as e:
            self.logger(f"Lỗi kết nối mạng: {str(e)}")
            self.data_valid = False
            return False
        except (requests.exceptions.Timeout, socket.timeout) as e:
            self.logger(f"Lỗi timeout: Không nhận được phản hồi từ Binance")
            self.data_valid = False
            return False
        except ssl.SSLError as e:
            self.logger(f"Lỗi SSL: {str(e)}")
            self.data_valid = False
            return False
        except Exception as e:
            self.logger(f"Lỗi không xác định khi cập nhật dữ liệu: {str(e)}")
            self.data_valid = False
            return False


class MobileTradingBot:
    def __init__(self, coin, leverage, risk_percent, take_profit, stop_loss, strategy_settings, logger):
        self.coin = coin
        self.leverage = leverage
        self.risk_percent = risk_percent
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.strategy_settings = strategy_settings
        self.active = False
        self.position_open = False
        self.position_side = None
        self.entry_price = 0.0
        self.last_trade_time = 0
        self.cooldown = 60  # 60 giây giữa các lệnh
        self.data_retries = 0  # Đếm số lần thử lại khi mất kết nối
        self.logger = logger
        
    def round_to_step(self, value, step):
        """Làm tròn giá trị theo bước quy định"""
        if step <= 0:
            return value
        return round(round(value / step) * step, 8)
    
    def get_historical_prices(self):
        """Lấy dữ liệu giá lịch sử với xử lý lỗi toàn diện"""
        try:
            client = Client(
                BINANCE_API_KEY, 
                BINANCE_SECRET_KEY, 
                testnet=USE_TESTNET,
                requests_params={'timeout': 15}
            )
            
            # Tính toán giới hạn dữ liệu cần lấy
            period = max(self.strategy_settings['ma_short'], self.strategy_settings['ma_long'])
            limit = min(period * 2, 100)  # Lấy đủ dữ liệu nhưng không quá 100
            
            klines = client.get_klines(
                symbol=self.coin.name,
                interval=self.strategy_settings['timeframe'],
                limit=limit
            )
            return [float(kline[4]) for kline in klines]
        except BinanceAPIException as e:
            self.logger(f"Lỗi API khi lấy dữ liệu lịch sử: {e.status_code} - {e.message}")
            return []
        except (requests.exceptions.ConnectionError, socket.gaierror) as e:
            self.logger(f"Lỗi kết nối mạng khi lấy dữ liệu lịch sử: {str(e)}")
            return []
        except (requests.exceptions.Timeout, socket.timeout) as e:
            self.logger(f"Lỗi timeout khi lấy dữ liệu lịch sử")
            return []
        except ssl.SSLError as e:
            self.logger(f"Lỗi SSL khi lấy dữ liệu lịch sử: {str(e)}")
            return []
        except Exception as e:
            self.logger(f"Lỗi không xác định khi lấy dữ liệu lịch sử: {str(e)}")
            return []
    
    def calculate_signal(self):
        """Tính toán tín hiệu giao dịch với kiểm tra dữ liệu"""
        prices = self.get_historical_prices()
        
        # Kiểm tra xem có đủ dữ liệu không
        if len(prices) < self.strategy_settings['ma_long']:
            self.logger(f"Không đủ dữ liệu: {len(prices)} < {self.strategy_settings['ma_long']}")
            return "neutral"
            
        # Chiến lược MA đơn giản
        try:
            short_ma = np.mean(prices[-self.strategy_settings['ma_short']:])
            long_ma = np.mean(prices[-self.strategy_settings['ma_long']:])
            
            if short_ma > long_ma:
                return "long"
            elif short_ma < long_ma:
                return "short"
        except Exception as e:
            self.logger(f"Lỗi tính toán tín hiệu: {str(e)}")
        
        return "neutral"
    
    def place_trade(self):
        """Đặt lệnh giao dịch với xử lý lỗi toàn diện"""
        if not self.active or self.position_open:
            return None
            
        # Kiểm tra thời gian giữa các lệnh
        current_time = time.time()
        if current_time - self.last_trade_time < self.cooldown:
            return None
            
        try:
            # Cập nhật dữ liệu coin trước khi giao dịch
            if not self.coin.update_data():
                self.data_retries += 1
                if self.data_retries > 3:
                    return "Lỗi: Không thể cập nhật dữ liệu sau 3 lần thử"
                return None
                
            self.data_retries = 0  # Reset bộ đếm sau khi thành công
            
            client = Client(
                BINANCE_API_KEY, 
                BINANCE_SECRET_KEY, 
                testnet=USE_TESTNET,
                requests_params={'timeout': 15}
            )
            
            # Lấy số dư tài khoản
            balance_info = client.futures_account_balance()
            usdt_balance = next((item for item in balance_info if item['asset'] == 'USDT'), None)
            if not usdt_balance:
                return "Lỗi: Không tìm thấy số dư USDT"
                
            balance = float(usdt_balance['balance'])
            
            # Tính toán khối lượng giao dịch
            trade_amount = balance * (self.risk_percent / 100) * self.leverage
            quantity = trade_amount / self.coin.price
            quantity = self.round_to_step(quantity, self.coin.step_size)
            
            # Kiểm tra số lượng tối thiểu
            if quantity < self.coin.min_qty:
                return f"Lỗi: Khối lượng quá nhỏ. Tối thiểu: {self.coin.min_qty}, Tính toán: {quantity}"
            
            # Tính toán tín hiệu
            signal = self.calculate_signal()
            if signal == "neutral":
                return "Không có tín hiệu giao dịch"
            
            # Thực hiện giao dịch
            if signal == "long":
                # Đặt lệnh long
                order = client.futures_create_order(
                    symbol=self.coin.name,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # Đặt take profit và stop loss
                client.futures_create_order(
                    symbol=self.coin.name,
                    side=SIDE_SELL,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=round(self.coin.price * (1 + self.take_profit/100), 2),
                    closePosition=True
                )
                client.futures_create_order(
                    symbol=self.coin.name,
                    side=SIDE_SELL,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=round(self.coin.price * (1 - self.stop_loss/100), 2),
                    closePosition=True
                )
                
                self.position_open = True
                self.position_side = "LONG"
                self.entry_price = self.coin.price
                self.last_trade_time = current_time
                return f"LONG: {quantity} @ {self.coin.price:.2f}"
                
            elif signal == "short":
                # Đặt lệnh short
                order = client.futures_create_order(
                    symbol=self.coin.name,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # Đặt take profit và stop loss
                client.futures_create_order(
                    symbol=self.coin.name,
                    side=SIDE_BUY,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=round(self.coin.price * (1 - self.take_profit/100), 2),
                    closePosition=True
                )
                client.futures_create_order(
                    symbol=self.coin.name,
                    side=SIDE_BUY,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=round(self.coin.price * (1 + self.stop_loss/100), 2),
                    closePosition=True
                )
                
                self.position_open = True
                self.position_side = "SHORT"
                self.entry_price = self.coin.price
                self.last_trade_time = current_time
                return f"SHORT: {quantity} @ {self.coin.price:.2f}"
                
        except BinanceAPIException as e:
            return f"Lỗi API Binance: {e.status_code} - {e.message}"
        except (requests.exceptions.ConnectionError, socket.gaierror) as e:
            return f"Lỗi kết nối mạng: {str(e)}"
        except (requests.exceptions.Timeout, socket.timeout) as e:
            return "Lỗi timeout: Không nhận được phản hồi từ Binance"
        except ssl.SSLError as e:
            return f"Lỗi SSL: {str(e)}"
        except Exception as e:
            return f"Lỗi không xác định khi đặt lệnh: {str(e)}"
        
        return None
    
    def get_tp_sl_info(self):
        """Lấy thông tin TP/SL cho vị thế hiện tại"""
        if not self.position_open:
            return None, None
            
        if self.position_side == "LONG":
            tp_price = round(self.entry_price * (1 + self.take_profit/100), 2)
            sl_price = round(self.entry_price * (1 - self.stop_loss/100), 2)
            return tp_price, sl_price
        elif self.position_side == "SHORT":
            tp_price = round(self.entry_price * (1 - self.take_profit/100), 2)
            sl_price = round(self.entry_price * (1 + self.stop_loss/100), 2)
            return tp_price, sl_price
            
        return None, None
    
    def check_position_status(self):
        """Kiểm tra trạng thái vị thế"""
        if not self.position_open:
            return None
            
        try:
            client = Client(
                BINANCE_API_KEY, 
                BINANCE_SECRET_KEY, 
                testnet=USE_TESTNET,
                requests_params={'timeout': 10}
            )
            positions = client.futures_position_information(symbol=self.coin.name)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position or float(position['positionAmt']) == 0:
                self.position_open = False
                self.position_side = None
                return "Vị thế đã đóng"
                
        except Exception as e:
            return f"Lỗi kiểm tra vị thế: {str(e)}"
            
        return None


class MobileTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot")
        self.root.geometry("400x700")  # Kích thước tối ưu cho điện thoại
        self.root.option_add('*Font', 'Helvetica 14')  # Font lớn dễ đọc
        
        # Biến giao dịch
        self.bot = None
        self.strategy_settings = {
            'ma_short': 10,
            'ma_long': 20,
            'timeframe': "15m"
        }
        
        # Tạo giao diện thân thiện với điện thoại
        self.setup_mobile_ui()
        
        # Tải cài đặt nếu có
        self.load_settings()
        
        # Kiểm tra kết nối ban đầu
        self.check_initial_connection()
    
    def setup_mobile_ui(self):
        # Main frame với khả năng cuộn
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tạo canvas và thanh cuộn
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        # Cấu hình cuộn
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Đóng gói
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Control panel
        control_frame = ttk.LabelFrame(self.scrollable_frame, text="⚙️ CÀI ĐẶT GIAO DỊCH")
        control_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        # Coin selection
        ttk.Label(control_frame, text="Coin:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.coin_entry = ttk.Entry(control_frame, width=10)
        self.coin_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.coin_entry.insert(0, "BTCUSDT")
        
        # Trading parameters
        ttk.Label(control_frame, text="Đòn bẩy:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.leverage_entry = ttk.Entry(control_frame, width=5)
        self.leverage_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.leverage_entry.insert(0, "10")
        
        ttk.Label(control_frame, text="Rủi ro %:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.risk_entry = ttk.Entry(control_frame, width=5)
        self.risk_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.risk_entry.insert(0, "5")
        
        ttk.Label(control_frame, text="Take Profit %:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.tp_entry = ttk.Entry(control_frame, width=5)
        self.tp_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.tp_entry.insert(0, "2.0")
        
        ttk.Label(control_frame, text="Stop Loss %:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.sl_entry = ttk.Entry(control_frame, width=5)
        self.sl_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.sl_entry.insert(0, "1.0")
        
        # Strategy settings
        strategy_frame = ttk.LabelFrame(self.scrollable_frame, text="📊 CHIẾN LƯỢC")
        strategy_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        ttk.Label(strategy_frame, text="Khung thời gian:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.timeframe_combo = ttk.Combobox(strategy_frame, 
                                          values=["5m", "15m", "30m", "1h"], 
                                          width=8)
        self.timeframe_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.timeframe_combo.set("15m")
        
        ttk.Label(strategy_frame, text="MA ngắn:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ma_short_entry = ttk.Entry(strategy_frame, width=5)
        self.ma_short_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.ma_short_entry.insert(0, "10")
        
        ttk.Label(strategy_frame, text="MA dài:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.ma_long_entry = ttk.Entry(strategy_frame, width=5)
        self.ma_long_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.ma_long_entry.insert(0, "20")
        
        # Position info - Hiển thị TP/SL
        self.position_frame = ttk.LabelFrame(self.scrollable_frame, text="📈 VỊ THẾ HIỆN TẠI")
        self.position_frame.pack(fill=tk.X, padx=10, pady=5, ipadx=5, ipady=5)
        
        self.position_info = ttk.Label(self.position_frame, text="Không có vị thế mở")
        self.position_info.pack(padx=5, pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="▶️ BẮT ĐẦU", command=self.start_bot)
        self.start_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ DỪNG", command=self.stop_bot)
        self.stop_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Status panel
        status_frame = ttk.LabelFrame(self.scrollable_frame, text="📝 TRẠNG THÁI")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, ipadx=5, ipady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Helvetica", 16))
        status_label.pack(padx=5, pady=5)
        
        # Log panel
        log_frame = ttk.LabelFrame(self.scrollable_frame, text="📋 NHẬT KÝ")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, ipadx=5, ipady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=("Helvetica", 12))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Cập nhật giao diện cuộn
        self.root.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    
    def log_message(self, message):
        """Ghi log với thời gian và cập nhật trạng thái"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)
        self.status_var.set(message)
    
    def update_position_info(self):
        """Cập nhật thông tin vị thế và TP/SL"""
        if self.bot and self.bot.position_open:
            tp_price, sl_price = self.bot.get_tp_sl_info()
            if tp_price and sl_price:
                position_text = (
                    f"Coin: {self.bot.coin.name}\n"
                    f"Hướng: {self.bot.position_side}\n"
                    f"Giá vào: {self.bot.entry_price:.2f}\n"
                    f"Giá hiện tại: {self.bot.coin.price:.2f}\n"
                    f"Take Profit: {tp_price:.2f}\n"
                    f"Stop Loss: {sl_price:.2f}"
                )
                self.position_info.config(text=position_text)
                return
        
        self.position_info.config(text="Không có vị thế mở")
    
    def check_initial_connection(self):
        """Kiểm tra kết nối ban đầu với Binance"""
        try:
            client = Client(
                BINANCE_API_KEY, 
                BINANCE_SECRET_KEY, 
                testnet=USE_TESTNET,
                requests_params={'timeout': 10}
            )
            client.get_server_time()
            
            # Xác định loại mạng
            net_type = "Testnet" if USE_TESTNET else "Mainnet"
            self.log_message(f"Kết nối Binance {net_type} thành công")
            
            # Kiểm tra API keys có hợp lệ không
            try:
                balance_info = client.futures_account_balance()
                if any(float(item['balance']) > 0 for item in balance_info):
                    self.log_message("API keys hợp lệ")
                else:
                    self.log_message("API keys hợp lệ nhưng không có số dư")
            except BinanceAPIException as e:
                if e.status_code == 401:
                    self.log_message("Lỗi: API keys không hợp lệ hoặc không đủ quyền")
                else:
                    self.log_message(f"Lỗi khi kiểm tra số dư: {e.message}")
        except BinanceAPIException as e:
            net_type = "Testnet" if USE_TESTNET else "Mainnet"
            if e.status_code == 401:
                self.log_message(f"Lỗi xác thực: API keys không hợp lệ cho {net_type}")
            else:
                self.log_message(f"Lỗi API Binance ({net_type}): {e.status_code} - {e.message}")
        except (requests.exceptions.ConnectionError, socket.gaierror) as e:
            self.log_message(f"Lỗi kết nối mạng: {str(e)}")
        except (requests.exceptions.Timeout, socket.timeout) as e:
            self.log_message("Lỗi timeout: Không kết nối được tới Binance")
        except ssl.SSLError as e:
            self.log_message(f"Lỗi SSL: {str(e)}")
        except Exception as e:
            self.log_message(f"Lỗi không xác định khi kiểm tra kết nối: {str(e)}")
    
    def start_bot(self):
        """Khởi động bot giao dịch"""
        coin_name = self.coin_entry.get().strip().upper()
        if not coin_name:
            messagebox.showerror("Lỗi", "Vui lòng nhập tên coin")
            return
            
        try:
            # Kiểm tra coin có hợp lệ không
            client = Client(
                BINANCE_API_KEY, 
                BINANCE_SECRET_KEY, 
                testnet=USE_TESTNET,
                requests_params={'timeout': 10}
            )
            exchange_info = client.futures_exchange_info()
            valid_coins = [s['symbol'] for s in exchange_info['symbols']]
            
            if coin_name not in valid_coins:
                messagebox.showerror("Lỗi", f"Coin {coin_name} không hợp lệ")
                return
            
            leverage = int(self.leverage_entry.get())
            risk_percent = float(self.risk_entry.get())
            take_profit = float(self.tp_entry.get())
            stop_loss = float(self.sl_entry.get())
            
            # Lấy cài đặt chiến lược
            self.strategy_settings['ma_short'] = int(self.ma_short_entry.get())
            self.strategy_settings['ma_long'] = int(self.ma_long_entry.get())
            self.strategy_settings['timeframe'] = self.timeframe_combo.get()
            
            # Tạo coin và bot
            coin = MobileCoin(coin_name, self.log_message)
            if not coin.update_data():
                self.log_message("Không lấy được dữ liệu coin, vui lòng kiểm tra kết nối")
                return
                
            self.bot = MobileTradingBot(
                coin, leverage, risk_percent, 
                take_profit, stop_loss, self.strategy_settings,
                self.log_message
            )
            self.bot.active = True
            
            # Bắt đầu luồng giao dịch
            self.trading_thread = threading.Thread(target=self.run_trading)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.log_message("Bot đã bắt đầu")
            
            # Lưu cài đặt
            self.save_settings()
            
            # Bắt đầu cập nhật giao diện
            self.update_gui()
            
        except Exception as e:
            self.log_message(f"Lỗi khởi động bot: {str(e)}")
    
    def stop_bot(self):
        """Dừng bot giao dịch"""
        if self.bot:
            self.bot.active = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.log_message("Bot đã dừng")
    
    def run_trading(self):
        """Luồng chính thực hiện giao dịch"""
        while self.bot and self.bot.active:
            try:
                # Cập nhật giá
                if not self.bot.coin.update_data():
                    self.log_message("Cập nhật giá thất bại, thử lại sau")
                    time.sleep(10)
                    continue
                
                # Kiểm tra trạng thái vị thế
                status = self.bot.check_position_status()
                if status:
                    self.log_message(status)
                
                # Thực hiện giao dịch nếu có thể
                if not self.bot.position_open:
                    result = self.bot.place_trade()
                    if result:
                        self.log_message(result)
                
                # Chờ giữa các lần kiểm tra
                time.sleep(15)  # Giảm tần suất request
                
            except Exception as e:
                self.log_message(f"Lỗi hệ thống: {str(e)}")
                time.sleep(30)
    
    def update_gui(self):
        """Cập nhật giao diện định kỳ"""
        if self.bot and self.bot.active:
            # Cập nhật thông tin vị thế
            self.update_position_info()
            
            # Cập nhật trạng thái
            self.status_var.set(f"Đang chạy: {self.bot.coin.name}")
            
            # Tiếp tục cập nhật sau 5s
            self.root.after(5000, self.update_gui)
        else:
            self.status_var.set("Sẵn sàng")
            self.update_position_info()
    
    def save_settings(self):
        """Lưu cài đặt vào file"""
        settings = {
            'coin': self.coin_entry.get(),
            'leverage': self.leverage_entry.get(),
            'risk_percent': self.risk_entry.get(),
            'take_profit': self.tp_entry.get(),
            'stop_loss': self.sl_entry.get(),
            'ma_short': self.ma_short_entry.get(),
            'ma_long': self.ma_long_entry.get(),
            'timeframe': self.timeframe_combo.get()
        }
        
        with open("mobile_settings.json", "w") as f:
            json.dump(settings, f, indent=2)
            
        self.log_message("Đã lưu cài đặt")
    
    def load_settings(self):
        """Tải cài đặt từ file"""
        if os.path.exists("mobile_settings.json"):
            try:
                with open("mobile_settings.json", "r") as f:
                    settings = json.load(f)
                
                self.coin_entry.delete(0, tk.END)
                self.coin_entry.insert(0, settings.get('coin', 'BTCUSDT'))
                
                self.leverage_entry.delete(0, tk.END)
                self.leverage_entry.insert(0, settings.get('leverage', '10'))
                
                self.risk_entry.delete(0, tk.END)
                self.risk_entry.insert(0, settings.get('risk_percent', '5'))
                
                self.tp_entry.delete(0, tk.END)
                self.tp_entry.insert(0, settings.get('take_profit', '2.0'))
                
                self.sl_entry.delete(0, tk.END)
                self.sl_entry.insert(0, settings.get('stop_loss', '1.0'))
                
                self.ma_short_entry.delete(0, tk.END)
                self.ma_short_entry.insert(0, settings.get('ma_short', '10'))
                
                self.ma_long_entry.delete(0, tk.END)
                self.ma_long_entry.insert(0, settings.get('ma_long', '20'))
                
                self.timeframe_combo.set(settings.get('timeframe', '15m'))
                
                self.log_message("Đã tải cài đặt")
                
            except Exception as e:
                self.log_message(f"Lỗi tải cài đặt: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MobileTradingApp(root)
    root.mainloop()
