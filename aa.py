import json, hmac, hashlib, time, threading
import urllib.request, urllib.parse
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import websocket

try:
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

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
    if len(prices) < 35: return None
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd = ema12 - ema26
    signal = calc_ema(prices[-9:], 9)
    return macd, signal

def get_klines(symbol, interval="5m", limit=100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    data = json.loads(urllib.request.urlopen(url).read())
    return np.array([float(k[4]) for k in data])

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

def get_positions():
    ts = int(time.time() * 1000)
    query = f"timestamp={ts}"
    sig = sign(query)
    url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
    return json.loads(urllib.request.urlopen(req).read())

# === Bot Logic ===
class IndicatorBot:
    def __init__(self, symbol, lev, percent, tp, sl, indicator, log_func):
        self.symbol = symbol
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.indicator = indicator
        self.log = log_func
        self.status = "waiting"
        self.side = None
        self.qty = 0
        self.entry = 0
        self.ws = None
        self.prices = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        def on_message(ws, msg):
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
        def on_error(ws, err): self.log(f"WebSocket lỗi {self.symbol}: {err}")
        def on_close(ws, *_): self.log(f"WebSocket đóng {self.symbol}, reconnect..."); self.start_ws()
        self.start_ws = lambda: threading.Thread(target=lambda: websocket.WebSocketApp(
            f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade",
            on_message=on_message, on_error=on_error, on_close=on_close
        ).run_forever(), daemon=True).start()
        self.start_ws()
        while True: time.sleep(1)

    def get_signal(self):
        if len(self.prices) < 35: return None
        rsi_val = calc_rsi(np.array(self.prices))
        ema_val = calc_ema(np.array(self.prices))
        macd_val, macd_signal = calc_macd(np.array(self.prices))
        # Chọn chỉ báo
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
        except Exception as e:
            self.log(f"Lỗi vào lệnh {self.symbol}: {e}")

    def close_position(self):
        try:
            place_order(self.symbol, "SELL" if self.side == "BUY" else "BUY", abs(self.qty))
            self.status = "waiting"
            self.log(f"{self.symbol} đã chốt lệnh {self.side}, chờ tín hiệu mới ({self.indicator})")
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
        style.configure("TLabel", background="black", foreground="lime")
        style.configure("TButton", background="black", foreground="lime")

        self.bots = {}

        self.log_text = tk.Text(root, height=15, width=70, bg="black", fg="lime")
        self.log_text.grid(row=10, columnspan=3)

        menubar = tk.Menu(root)
        menu_main = tk.Menu(menubar, tearoff=0)
        menu_main.add_command(label="Thêm cặp", command=self.menu_add)
        menu_main.add_command(label="Trạng thái", command=self.menu_status)
        menu_main.add_separator()
        menu_main.add_command(label="Thoát", command=root.quit)
        menubar.add_cascade(label="Menu", menu=menu_main)
        root.config(menu=menubar)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def menu_add(self):
        symbol = tk.simpledialog.askstring("Cặp coin", "Nhập cặp (VD: BTCUSDT):")
        if not symbol: return
        symbol = symbol.upper()
        tp = tk.simpledialog.askfloat("TP %", "Nhập TP ROI (VD: 20):") or 0
        sl = tk.simpledialog.askfloat("SL %", "Nhập SL ROI (VD: 10):") or 0
        lev = tk.simpledialog.askinteger("Đòn bẩy", "Nhập đòn bẩy (VD: 20):") or 1
        percent = tk.simpledialog.askfloat("% số dư", "% số dư vào lệnh:") or 100
        indicator = tk.simpledialog.askstring("Chỉ báo", "Chọn chỉ báo: RSI, EMA, MACD, Tất cả").upper()
        if indicator not in ["RSI", "EMA", "MACD", "TẤT CẢ"]:
            self.log("Chỉ báo không hợp lệ! Chọn: RSI, EMA, MACD, Tất cả")
            return
        if indicator == "TẤT CẢ": indicator = "Tất cả"
        bot = IndicatorBot(symbol, lev, percent, tp, sl, indicator, self.log)
        self.bots[symbol] = bot
        self.log(f"Đã thêm bot cho {symbol} với chỉ báo {indicator}")

    def menu_status(self):
        top = tk.Toplevel(self.root)
        top.title("Trạng thái các lệnh")
        for sym, bot in self.bots.items():
            btn = tk.Button(top, text=f"{sym} [{bot.status} - {bot.side}]", command=lambda s=sym: self.manage_order(s))
            btn.pack(pady=2)

    def manage_order(self, sym):
        bot = self.bots[sym]
        action = messagebox.askquestion(sym, f"TP: {bot.tp}%\nSL: {bot.sl}%\nĐóng hoặc chỉnh sửa?", icon='question')
        if action == 'yes':
            bot.close_position()
            self.log(f"Đã đóng {sym} thủ công")

if __name__ == "__main__":
    root = tk.Tk()
    FuturesBotGUI(root)
    root.mainloop()
