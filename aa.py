import json, hmac, hashlib, time, threading
import urllib.request, urllib.parse
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import websocket

try:
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY
except ImportError:
    BINANCE_API_KEY = ""
    BINANCE_SECRET_KEY = ""

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY

# === Indicator Functions ===
def calc_rsi(prices, period=14):
    if len(prices) < period + 1: return None
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period or 1
    rs = up / down
    rsi = 100 - 100 / (1 + rs)
    for i in range(period, len(deltas)):
        delta = deltas[i]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period or 1
        rs = up / down
        rsi = 100 - 100 / (1 + rs)
    return rsi

def calc_ema(prices, period=21):
    if len(prices) < period: return None
    ema = np.mean(prices[:period])
    k = 2 / (period + 1)
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema

def calc_macd(prices):
    if len(prices) < 35: return None, None
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd = ema12 - ema26
    signal = calc_ema(prices[-9:], 9)
    return macd, signal

# === API Helpers ===
def sign(query):
    return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def get_step_size(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        data = json.loads(urllib.request.urlopen(url).read())
        for s in data['symbols']:
            if s['symbol'].lower() == symbol.lower():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except: return 0.001

def set_leverage(symbol, lev):
    ts = int(time.time() * 1000)
    query = f"symbol={symbol.upper()}&leverage={lev}&timestamp={ts}"
    sig = sign(query)
    url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='POST')
    urllib.request.urlopen(req)

def get_balance():
    ts = int(time.time() * 1000)
    query = f"timestamp={ts}"
    sig = sign(query)
    url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
    data = json.loads(urllib.request.urlopen(req).read())
    for asset in data['assets']:
        if asset['asset'] == 'USDT':
            return float(asset['availableBalance'])
    return 0

def place_order(symbol, side, qty):
    ts = int(time.time() * 1000)
    query = f"symbol={symbol.upper()}&side={side}&type=MARKET&quantity={qty}&timestamp={ts}"
    sig = sign(query)
    url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='POST')
    return json.loads(urllib.request.urlopen(req).read())

def get_current_price(symbol):
    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
    return float(json.loads(urllib.request.urlopen(url).read())['price'])

# === Bot Logic ===
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, indicator, log_func, update_func):
        self.symbol = symbol
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.indicator = indicator
        self.log = log_func
        self.update = update_func
        self.status = "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.ws = None
        self.prices = []
        self._stop = False
        self._last_status = None  # Để tránh spam trạng thái
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        if self.ws: self.ws.close()

    def run(self):
        def on_message(ws, msg):
            if self._stop: return
            data = json.loads(msg)
            price = float(data['p'])
            self.prices.append(price)
            if len(self.prices) > 100:
                self.prices = self.prices[-100:]
            if self.status == "waiting":
                signal = self.get_signal()
                if signal:
                    self.open_position(signal)
            elif self.status == "open":
                pnl = (price - self.entry) * self.qty
                roi = pnl * self.lev / abs(self.qty) if self.qty != 0 else 0
                if (roi >= self.tp and self.tp > 0) or (roi <= -self.sl and self.sl > 0):
                    self.close_position()
            # Chỉ cập nhật trạng thái khi vừa mở hoặc vừa đóng lệnh (tránh spam)
            if self.status != self._last_status:
                self.update(self.symbol, self.status, self.side)
                self._last_status = self.status
        def on_error(ws, err): self.log(f"WebSocket lỗi {self.symbol}: {err}")
        def on_close(ws, *_): self.log(f"WebSocket đóng {self.symbol}, reconnect..."); self.start_ws()
        self.start_ws = lambda: threading.Thread(target=lambda: websocket.WebSocketApp(
            f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade",
            on_message=on_message, on_error=on_error, on_close=on_close
        ).run_forever(), daemon=True).start()
        self.start_ws()
        while not self._stop: time.sleep(1)

    def get_signal(self):
        if len(self.prices) < 35: return None
        arr = np.array(self.prices)
        rsi_val = calc_rsi(arr)
        ema_val = calc_ema(arr)
        macd_val, macd_signal = calc_macd(arr)
        if self.indicator == "RSI":
            if rsi_val is not None and rsi_val <= 30: return "BUY"
            if rsi_val is not None and rsi_val >= 70: return "SELL"
        elif self.indicator == "EMA":
            if ema_val is not None and self.prices[-1] > ema_val: return "BUY"
            if ema_val is not None and self.prices[-1] < ema_val: return "SELL"
        elif self.indicator == "MACD":
            if macd_val is not None and macd_signal is not None:
                if macd_val > macd_signal: return "BUY"
                if macd_val < macd_signal: return "SELL"
        elif self.indicator == "Tất cả":
            if (rsi_val is not None and ema_val is not None and macd_val is not None and macd_signal is not None):
                buy = (rsi_val <= 30) and (self.prices[-1] > ema_val) and (macd_val > macd_signal)
                sell = (rsi_val >= 70) and (self.prices[-1] < ema_val) and (macd_val < macd_signal)
                if buy: return "BUY"
                if sell: return "SELL"
        return None

    def open_position(self, side):
        try:
            set_leverage(self.symbol, self.lev)
            balance = get_balance()
            price = get_current_price(self.symbol)
            step = get_step_size(self.symbol)
            qty = balance * (self.percent / 100) * self.lev / price
            qty = round(qty - (qty % step), 6)
            place_order(self.symbol, side, qty)
            self.entry = price
            self.qty = qty if side == "BUY" else -qty
            self.side = side
            self.status = "open"
            self.log(f"Vào lệnh {self.symbol} {side} TP={self.tp}% SL={self.sl}% theo {self.indicator}")
            self.update(self.symbol, self.status, self.side)
        except Exception as e:
            self.log(f"Lỗi vào lệnh {self.symbol}: {e}")

    def close_position(self):
        try:
            place_order(self.symbol, "SELL" if self.side == "BUY" else "BUY", abs(self.qty))
            self.status = "waiting"
            self.log(f"{self.symbol} đã chốt lệnh {self.side}, chờ tín hiệu mới ({self.indicator})")
            self.update(self.symbol, self.status, "")
        except Exception as e:
            self.log(f"Lỗi đóng {self.symbol}: {e}")

# === GUI Setup ===
class FuturesBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Futures Indicator Bot")
        self.root.configure(bg="black")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TLabel", background="black", foreground="lime", font=("Arial", 16))
        style.configure("TButton", background="black", foreground="lime", font=("Arial", 16))

        self.bots = {}

        # Form nhập liệu dọc cho mobile
        form = ttk.Frame(root)
        form.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        row = 0
        ttk.Label(form, text="Cặp coin:").grid(row=row, column=0, sticky="w")
        self.symbol_entry = ttk.Entry(form, width=15, font=("Arial", 16))
        self.symbol_entry.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(form, text="Đòn bẩy:").grid(row=row, column=0, sticky="w")
        self.lev_entry = ttk.Entry(form, width=15, font=("Arial", 16))
        self.lev_entry.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(form, text="% Số dư:").grid(row=row, column=0, sticky="w")
        self.percent_entry = ttk.Entry(form, width=15, font=("Arial", 16))
        self.percent_entry.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(form, text="TP %:").grid(row=row, column=0, sticky="w")
        self.tp_entry = ttk.Entry(form, width=15, font=("Arial", 16))
        self.tp_entry.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(form, text="SL %:").grid(row=row, column=0, sticky="w")
        self.sl_entry = ttk.Entry(form, width=15, font=("Arial", 16))
        self.sl_entry.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(form, text="Chỉ báo:").grid(row=row, column=0, sticky="w")
        self.indicator_var = tk.StringVar(value="RSI")
        self.indicator_menu = ttk.Combobox(form, textvariable=self.indicator_var, values=["RSI", "EMA", "MACD", "Tất cả"], width=13, font=("Arial", 16))
        self.indicator_menu.grid(row=row, column=1, pady=2)
        row += 1
        ttk.Button(form, text="Thêm bot", command=self.menu_add).grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")

        # Danh sách bot dạng Listbox + Scrollbar (rolling)
        ttk.Label(root, text="Các bot đang chạy:").grid(row=1, column=0, sticky="w", padx=10)
        frame_list = ttk.Frame(root)
        frame_list.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.bot_list = tk.Listbox(frame_list, height=8, font=("Arial", 14), bg="black", fg="lime", selectbackground="gray")
        self.bot_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame_list, orient="vertical", command=self.bot_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bot_list.config(yscrollcommand=scrollbar.set)
        ttk.Button(root, text="Dừng bot đã chọn", command=self.stop_selected_bot).grid(row=3, column=0, pady=5, sticky="ew")

        # Log
        ttk.Label(root, text="Log:").grid(row=4, column=0, sticky="w", padx=10)
        self.log_text = tk.Text(root, height=8, width=40, bg="black", fg="lime", font=("Arial", 13))
        self.log_text.grid(row=5, column=0, padx=10, pady=5, sticky="ew")

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def update_tree(self, symbol, status, side):
        # Chỉ cập nhật trạng thái bot trong Listbox khi vừa mở hoặc vừa đóng lệnh (tránh spam)
        for idx in range(self.bot_list.size()):
            if self.bot_list.get(idx).startswith(symbol):
                self.bot_list.delete(idx)
                break
        bot = self.bots.get(symbol)
        if bot:
            self.bot_list.insert(tk.END, f"{symbol} | {status} | {side} | {bot.indicator}")

    def menu_add(self):
        symbol = self.symbol_entry.get().strip().upper()
        try:
            lev = int(self.lev_entry.get())
            percent = float(self.percent_entry.get())
            tp = float(self.tp_entry.get())
            sl = float(self.sl_entry.get())
            indicator = self.indicator_var.get()
        except Exception:
            self.log("Vui lòng nhập đúng thông số!")
            return
        if symbol in self.bots:
            self.log(f"Đã có bot cho {symbol}")
            return
        bot = IndicatorBot(symbol, lev, percent, tp, sl, indicator, self.log, self.update_tree)
        self.bots[symbol] = bot
        self.bot_list.insert(tk.END, f"{symbol} | {bot.status} | {bot.side} | {indicator}")
        self.log(f"Đã thêm bot cho {symbol} với chỉ báo {indicator}")

    def stop_selected_bot(self):
        selected = self.bot_list.curselection()
        if not selected: return
        idx = selected[0]
        line = self.bot_list.get(idx)
        symbol = line.split('|')[0].strip()
        bot = self.bots.get(symbol)
        if bot:
            bot.stop()
            bot.close_position()
            self.log(f"Đã dừng bot cho {symbol}")
            self.bot_list.delete(idx)
            del self.bots[symbol]

if __name__ == "__main__":
    root = tk.Tk()
    FuturesBotGUI(root)
    root.mainloop()
