from binance.client import Client
from binance.enums import *
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

try:
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

import numpy as np

# --- Chỉ báo RSI thực tế ---
def get_signal(coin, client, interval="5m", rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    try:
        klines = client.futures_klines(symbol=coin.get_name(), interval=interval, limit=rsi_period+20)
        closes = np.array([float(k[4]) for k in klines])
        if len(closes) < rsi_period + 1:
            return None
        deltas = np.diff(closes)
        seed = deltas[:rsi_period]
        up = seed[seed >= 0].sum() / rsi_period
        down = -seed[seed < 0].sum() / rsi_period
        rs = up / down if down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
        for i in range(rsi_period, len(deltas)):
            delta = deltas[i]
            upval = max(delta, 0)
            downval = -min(delta, 0)
            up = (up * (rsi_period - 1) + upval) / rsi_period
            down = (down * (rsi_period - 1) + downval) / rsi_period
            rs = up / down if down != 0 else 0
            rsi = 100 - 100 / (1 + rs)
        # Quyết định tín hiệu
        if rsi < rsi_oversold:
            return "long"
        elif rsi > rsi_overbought:
            return "short"
        else:
            return None
    except Exception as e:
        return None

# Thư viện Coin
class Coin:
    def __init__(self, name, client):
        self.name = name
        self.client = client
        self.price = 0.0
        self.max_leverage = 20
        self.min_order_amount = 0.0
        self.step_size = 0.001
        self.update_info()

    def update_info(self):
        ticker = self.client.futures_symbol_ticker(symbol=self.name)
        self.price = float(ticker['price'])
        info = self.client.futures_exchange_info()
        for symbol in info['symbols']:
            if symbol['symbol'] == self.name:
                for f in symbol['filters']:
                    if f['filterType'] == 'MARKET_LOT_SIZE':
                        self.step_size = float(f['stepSize'])
                    if f['filterType'] == 'MIN_NOTIONAL':
                        self.min_order_amount = float(f['notional'])
                if 'leverageFilter' in symbol:
                    self.max_leverage = int(symbol['leverageFilter']['maxLeverage'])
                break

    def get_price(self):
        self.update_info()
        return self.price

    def get_max_leverage(self):
        return self.max_leverage

    def get_min_order_amount(self):
        return self.min_order_amount

    def get_step_size(self):
        return self.step_size

    def get_name(self):
        return self.name

# Thư viện Tài Khoản
class TaiKhoan:
    def __init__(self, api_key, secret_key):
        self.BINANCE_API_KEY = api_key
        self.BINANCE_SECRET_KEY = secret_key
        self.client = Client(api_key, secret_key)
        self.balance = self.get_balance()

    def get_balance(self):
        info = self.client.futures_account_balance()
        for item in info:
            if item['asset'] == 'USDT':
                return float(item['balance'])
        return 0.0

    def update_balance(self):
        self.balance = self.get_balance()
        return self.balance

# Thư viện Trading Future Bot
class TradingFutureBot:
    def __init__(self, coin, leverage, percent_balance, tp, sl):
        self.coin = coin
        self.leverage = leverage
        self.percent_balance = percent_balance
        self.tp = tp
        self.sl = sl
        self.status = True  # Đang chạy
        self.thread = None
        self.last_signal = None

    def start_auto_trade(self, account, update_callback=None):
        if self.thread and self.thread.is_alive():
            return
        self.status = True
        self.thread = threading.Thread(target=self._auto_trade_loop, args=(account, update_callback), daemon=True)
        self.thread.start()

    def stop(self):
        self.status = False

    def _auto_trade_loop(self, account, update_callback):
        while self.status:
            try:
                signal = get_signal(self.coin, account.client)
                if signal and signal != self.last_signal:
                    if not self.check_position_open(account):
                        result = self.place_order(account, signal)
                        self.last_signal = signal
                        if update_callback:
                            update_callback(f"{self.coin.get_name()} [{signal.upper()}]: {result}")
                time.sleep(10)
            except Exception as e:
                if update_callback:
                    update_callback(f"Lỗi bot {self.coin.get_name()}: {e}")
                time.sleep(10)

    def check_position_open(self, account):
        positions = account.client.futures_position_information(symbol=self.coin.get_name())
        for pos in positions:
            if float(pos['positionAmt']) != 0:
                return True
        return False

    def place_order(self, account, direction):
        account.update_balance()
        balance = account.balance
        self.coin.update_info()
        min_amount = self.coin.get_min_order_amount()
        price = self.coin.get_price()
        order_amount = balance * self.percent_balance * self.leverage / 100
        qty = round(order_amount / price, 3)
        if qty * price < min_amount or qty <= 0:
            return "Không đủ điều kiện đặt lệnh!"

        # Đặt đòn bẩy
        account.client.futures_change_leverage(symbol=self.coin.get_name(), leverage=self.leverage)

        # Đặt lệnh market
        side = SIDE_BUY if direction == "long" else SIDE_SELL
        order = account.client.futures_create_order(
            symbol=self.coin.get_name(),
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=qty
        )

        # Đặt TP/SL nếu có
        if self.tp > 0:
            tp_price = price * (1 + self.tp / 100) if direction == "long" else price * (1 - self.tp / 100)
            account.client.futures_create_order(
                symbol=self.coin.get_name(),
                side=SIDE_SELL if direction == "long" else SIDE_BUY,
                type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=round(tp_price, 2),
                closePosition=True
            )
        if self.sl > 0:
            sl_price = price * (1 - self.sl / 100) if direction == "long" else price * (1 + self.sl / 100)
            account.client.futures_create_order(
                symbol=self.coin.get_name(),
                side=SIDE_SELL if direction == "long" else SIDE_BUY,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=round(sl_price, 2),
                closePosition=True
            )
        return f"Đã mở vị thế {direction.upper()} {self.coin.get_name()} với {qty} coin!"

# Giao diện GUI với Tkinter, chỉ chọn cặp coin, không chọn hướng
class FutureBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Future Trading Bot")
        self.account = None
        self.bots = {}  # Quản lý nhiều bot theo tên coin

        # Giao diện nhập API
        frame_api = ttk.LabelFrame(root, text="Tài khoản Binance")
        frame_api.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_api, text="API Key:").grid(row=0, column=0, sticky="w")
        ttk.Label(frame_api, text="Secret Key:").grid(row=1, column=0, sticky="w")
        self.api_entry = ttk.Entry(frame_api, width=40)
        self.api_entry.grid(row=0, column=1, padx=5, pady=2)
        self.secret_entry = ttk.Entry(frame_api, width=40, show="*")
        self.secret_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(frame_api, text="Kết nối", command=self.connect_account).grid(row=0, column=2, rowspan=2, padx=5)

        # Nếu dùng config.py thì tự động điền key/secret
        if USE_CONFIG:
            self.api_entry.insert(0, BINANCE_API_KEY)
            self.secret_entry.insert(0, BINANCE_SECRET_KEY)

        # Giao diện đặt lệnh
        self.frame_order = ttk.LabelFrame(root, text="Đặt lệnh Future (chỉ chọn cặp, bot tự quyết định hướng)")
        self.frame_order.pack(fill="x", padx=10, pady=5)
        ttk.Label(self.frame_order, text="Cặp coin:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.frame_order, text="Đòn bẩy:").grid(row=1, column=0, sticky="w")
        ttk.Label(self.frame_order, text="% Số dư:").grid(row=2, column=0, sticky="w")
        ttk.Label(self.frame_order, text="Take Profit (%):").grid(row=3, column=0, sticky="w")
        ttk.Label(self.frame_order, text="Stop Loss (%):").grid(row=4, column=0, sticky="w")

        self.symbol_entry = ttk.Entry(self.frame_order)
        self.symbol_entry.grid(row=0, column=1, padx=5, pady=2)
        self.leverage_entry = ttk.Entry(self.frame_order)
        self.leverage_entry.grid(row=1, column=1, padx=5, pady=2)
        self.percent_entry = ttk.Entry(self.frame_order)
        self.percent_entry.grid(row=2, column=1, padx=5, pady=2)
        self.tp_entry = ttk.Entry(self.frame_order)
        self.tp_entry.grid(row=3, column=1, padx=5, pady=2)
        self.sl_entry = ttk.Entry(self.frame_order)
        self.sl_entry.grid(row=4, column=1, padx=5, pady=2)

        self.btn_order = ttk.Button(self.frame_order, text="Thêm cặp & Tự động lặp", command=self.place_order)
        self.btn_order.grid(row=5, column=0, columnspan=2, pady=5)

        # Ban đầu disable các trường đặt lệnh
        self.set_order_widgets_state("disabled")

        # Danh sách các vị thế đang mở
        self.frame_status = ttk.LabelFrame(root, text="Các cặp đang theo dõi")
        self.frame_status.pack(fill="x", padx=10, pady=5)
        self.status_list = tk.Listbox(self.frame_status, height=6)
        self.status_list.pack(fill="x", padx=5, pady=5)
        ttk.Button(self.frame_status, text="Dừng bot & đóng vị thế", command=self.close_position).pack(pady=5)

        # Trạng thái
        self.status_label = ttk.Label(root, text="Chưa kết nối tài khoản.", foreground="red")
        self.status_label.pack(pady=5)

        # Log
        self.log_text = tk.Text(root, height=6, state="disabled")
        self.log_text.pack(fill="x", padx=10, pady=5)

    def set_order_widgets_state(self, state):
        widgets = [
            self.symbol_entry,
            self.leverage_entry,
            self.percent_entry,
            self.tp_entry,
            self.sl_entry,
            self.btn_order
        ]
        for w in widgets:
            w.config(state=state)

    def connect_account(self):
        api = self.api_entry.get().strip()
        secret = self.secret_entry.get().strip()
        try:
            self.account = TaiKhoan(api, secret)
            self.status_label.config(text="Kết nối thành công!", foreground="green")
            self.set_order_widgets_state("normal")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Kết nối thất bại: {e}")
            self.status_label.config(text="Kết nối thất bại.", foreground="red")
            self.set_order_widgets_state("disabled")

    def place_order(self):
        if not self.account:
            messagebox.showwarning("Chưa kết nối", "Vui lòng kết nối tài khoản Binance trước.")
            return
        try:
            symbol = self.symbol_entry.get().strip().upper()
            if symbol in self.bots:
                messagebox.showwarning("Đã theo dõi", f"Đã có bot cho {symbol}.")
                return
            coin = Coin(symbol, self.account.client)
            leverage = int(self.leverage_entry.get())
            percent = float(self.percent_entry.get())
            tp = float(self.tp_entry.get())
            sl = float(self.sl_entry.get())
            bot = TradingFutureBot(coin, leverage, percent, tp, sl)
            self.bots[symbol] = bot
            self.status_list.insert(tk.END, f"{symbol} | Đòn bẩy: {leverage} | %Số dư: {percent}")
            bot.start_auto_trade(self.account, self.log)
            messagebox.showinfo("Kết quả", f"Đã bắt đầu bot tự động cho {symbol}.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi đặt lệnh: {e}")

    def close_position(self):
        selection = self.status_list.curselection()
        if not selection:
            messagebox.showwarning("Chọn vị thế", "Vui lòng chọn vị thế để dừng.")
            return
        idx = selection[0]
        symbol = self.status_list.get(idx).split('|')[0].strip()
        if symbol in self.bots:
            self.bots[symbol].stop()
            del self.bots[symbol]
            self.status_list.delete(idx)
            self.log(f"Đã dừng bot cho {symbol}.")
            messagebox.showinfo("Đã dừng", f"Đã dừng bot cho {symbol}.\nBạn cần tự đóng vị thế trên Binance nếu muốn.")

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    FutureBotGUI(root)
    root.mainloop()
