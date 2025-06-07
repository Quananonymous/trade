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

def sign(query):
    return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def get_step_size(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        for s in data['symbols']:
            if s['symbol'].lower() == symbol.lower():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        print(f"Error getting step size: {e}")
        return 0.001
    return 0.001

def set_leverage(symbol, lev):
    try:
        ts = int(time.time() * 1000)
        query = f"symbol={symbol.upper()}&leverage={lev}&timestamp={ts}"
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='POST')
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Error setting leverage: {e}")

def get_balance():
    try:
        ts = int(time.time() * 1000)
        query = f"timestamp={ts}"
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read())
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
    except Exception as e:
        print(f"Error getting balance: {e}")
    return 0

def place_order(symbol, side, qty):
    try:
        ts = int(time.time() * 1000)
        query = f"symbol={symbol.upper()}&side={side}&type=MARKET&quantity={qty}&timestamp={ts}"
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='POST')
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def place_stop_market(symbol, side, qty, stop_price):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "STOP_MARKET",
            "quantity": qty,
            "stopPrice": stop_price,
            "reduceOnly": "true", 
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='POST')
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except Exception as e:
        print(f"Error placing stop market: {e}")
        return None

def place_take_profit_market(symbol, side, qty, stop_price):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": qty,
            "stopPrice": stop_price,
            "reduceOnly": "true", 
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='POST')
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except Exception as e:
        print(f"Error placing take profit: {e}")
        return None

def cancel_all_orders(symbol):
    try:
        ts = int(time.time() * 1000)
        query = f"symbol={symbol.upper()}&timestamp={ts}"
        sig = sign(query)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY}, method='DELETE')
        urllib.request.urlopen(req)
    except Exception as e:
        print(f"Error canceling orders: {e}")

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        return float(data['price'])
    except Exception as e:
        print(f"Error getting price: {e}")
        return 0

def calc_rsi(prices, period=14):
    if len(prices) < period:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))

def calc_ema(prices, period=21):
    if len(prices) < period:
        return None
    ema = np.mean(prices[:period])
    k = 2 / (period + 1)
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema

def calc_macd(prices):
    if len(prices) < 35:
        return None, None
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd = ema12 - ema26
    signal = calc_ema(prices[-9:], 9)
    return macd, signal

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
        self.prices = []
        self._stop = False
        self._last_status = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        if hasattr(self, 'ws') and self.ws:
            self.ws.close()
        try:
            cancel_all_orders(self.symbol)
        except Exception as e:
            self.log(f"Lỗi hủy lệnh: {e}")

    def run(self):
        def on_message(ws, msg):
            if self._stop: 
                return
            try:
                data = json.loads(msg)
                price = float(data['p'])
                self.prices.append(price)
                if len(self.prices) > 100:
                    self.prices = self.prices[-100:]
                    
                if self.status == "waiting":
                    signal = self.get_signal()
                    if signal:
                        self.open_position(signal)
                
                if self.status != self._last_status:
                    self.update(self.symbol, self.status, self.side)
                    self._last_status = self.status
            except Exception as e:
                self.log(f"Lỗi xử lý tin nhắn: {e}")

        def on_error(ws, err):
            self.log(f"WebSocket lỗi {self.symbol}: {err}")

        def on_close(ws, *args):
            self.log(f"WebSocket đóng {self.symbol}, kết nối lại...")
            if not self._stop:
                time.sleep(3)
                self.start_ws()

        self.start_ws = lambda: threading.Thread(target=lambda: websocket.WebSocketApp(
            f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade",
            on_message=on_message, 
            on_error=on_error, 
            on_close=on_close
        ).run_forever(), daemon=True).start()
        self.start_ws()
        while not self._stop:
            time.sleep(1)

    def get_signal(self):
        if len(self.prices) < 35:
            return None
            
        arr = np.array(self.prices)
        rsi_val = calc_rsi(arr)
        ema_val = calc_ema(arr)
        macd_val, macd_signal = calc_macd(arr)
        
        if self.indicator == "RSI":
            if rsi_val is not None:
                if rsi_val <= 30: 
                    return "BUY"
                if rsi_val >= 70: 
                    return "SELL"
                    
        elif self.indicator == "EMA":
            if ema_val is not None:
                if self.prices[-1] > ema_val: 
                    return "BUY"
                if self.prices[-1] < ema_val: 
                    return "SELL"
                    
        elif self.indicator == "MACD":
            if macd_val is not None and macd_signal is not None:
                if macd_val > macd_signal: 
                    return "BUY"
                if macd_val < macd_signal: 
                    return "SELL"
                    
        elif self.indicator == "Tất cả":
            if (rsi_val is not None and 
                ema_val is not None and 
                macd_val is not None and 
                macd_signal is not None):
                    
                buy_signals = 0
                sell_signals = 0
                
                if rsi_val <= 30: 
                    buy_signals += 1
                if rsi_val >= 70: 
                    sell_signals += 1
                    
                if self.prices[-1] > ema_val: 
                    buy_signals += 1
                if self.prices[-1] < ema_val: 
                    sell_signals += 1
                    
                if macd_val > macd_signal: 
                    buy_signals += 1
                if macd_val < macd_signal: 
                    sell_signals += 1
                    
                if buy_signals >= 2: 
                    return "BUY"
                if sell_signals >= 2: 
                    return "SELL"
                    
        return None

    def open_position(self, side):
        try:
            # Hủy tất cả lệnh trước khi vào lệnh mới
            cancel_all_orders(self.symbol)
            
            # Đặt đòn bẩy
            set_leverage(self.symbol, self.lev)
            
            # Lấy số dư USDT
            balance = get_balance()
            if balance <= 0:
                self.log(f"Không đủ số dư USDT cho {self.symbol}")
                return
                
            # Lấy giá hiện tại
            price = get_current_price(self.symbol)
            if price <= 0:
                self.log(f"Lỗi lấy giá cho {self.symbol}")
                return
                
            # Tính số lượng dựa trên USDT
            usdt_amount = balance * (self.percent / 100) * self.lev
            step = get_step_size(self.symbol)
            qty = usdt_amount / price
            
            # Làm tròn số lượng theo step size
            if step > 0:
                qty = round(qty - (qty % step), 6)
                
            if qty <= 0:
                self.log(f"Số lượng không hợp lệ cho {self.symbol}: {qty}")
                return
                
            # Đặt lệnh chính
            res = place_order(self.symbol, side, qty)
            if not res:
                self.log(f"Lỗi khi đặt lệnh cho {self.symbol}")
                return
                
            # Lấy thông tin thực tế từ lệnh
            if 'avgPrice' in res and res['avgPrice']:
                self.entry = float(res['avgPrice'])
            else:
                self.entry = price

            executed_qty = float(res['executedQty'])
            if side == "BUY":
                self.qty = executed_qty
            else:
                self.qty = -executed_qty

            self.side = side
            self.status = "open"
            self.log(f"Đã vào lệnh {self.symbol} {side} tại {self.entry:.4f}")
            self.log(f"Số lượng: {executed_qty}, Giá trị: {executed_qty * self.entry:.2f} USDT")
            self.update(self.symbol, self.status, self.side)

            # Tính giá TP và SL
            if side == "BUY":
                tp_price = self.entry * (1 + self.tp / 100) if self.tp > 0 else None
                sl_price = self.entry * (1 - self.sl / 100) if self.sl > 0 else None
            else:  # SELL
                tp_price = self.entry * (1 - self.tp / 100) if self.tp > 0 else None
                sl_price = self.entry * (1 + self.sl / 100) if self.sl > 0 else None

            # Đặt lệnh TP trên Binance
            if tp_price:
                try:
                    tp_side = "SELL" if side == "BUY" else "BUY"
                    tp_res = place_take_profit_market(
                        self.symbol, 
                        tp_side, 
                        abs(self.qty), 
                        round(tp_price, 4)
                    )
                    if tp_res:
                        self.log(f"✅ Đã đặt TP tại {tp_price:.4f} (ID: {tp_res['orderId']})")
                        self.log(f"📊 Giá trị TP: {abs(self.qty) * tp_price:.2f} USDT")
                    else:
                        self.log("❌ Lỗi khi đặt lệnh TP")
                except Exception as e:
                    self.log(f"❌ Lỗi đặt TP: {e}")

            # Đặt lệnh SL trên Binance
            if sl_price:
                try:
                    sl_side = "SELL" if side == "BUY" else "BUY"
                    sl_res = place_stop_market(
                        self.symbol, 
                        sl_side, 
                        abs(self.qty), 
                        round(sl_price, 4)
                    )
                    if sl_res:
                        self.log(f"🛡️ Đã đặt SL tại {sl_price:.4f} (ID: {sl_res['orderId']})")
                        self.log(f"📉 Giá trị SL: {abs(self.qty) * sl_price:.2f} USDT")
                    else:
                        self.log("❌ Lỗi khi đặt lệnh SL")
                except Exception as e:
                    self.log(f"❌ Lỗi đặt SL: {e}")

        except Exception as e:
            self.log(f"❌ Lỗi nghiêm trọng khi vào lệnh {self.symbol}: {e}")

    def close_position(self):
        try:
            cancel_all_orders(self.symbol)
            if self.side == "BUY":
                res = place_order(self.symbol, "SELL", self.qty)
            else:
                res = place_order(self.symbol, "BUY", abs(self.qty))
                
            if res:
                price = float(res['avgPrice']) if 'avgPrice' in res else 0
                self.log(f"Đã đóng lệnh {self.symbol} tại {price:.4f}")
                self.log(f"Giá trị: {abs(self.qty) * price:.2f} USDT")
            else:
                self.log(f"Lỗi khi đóng lệnh {self.symbol}")
                
            self.status = "waiting"
            self.side = ""
            self.qty = 0
            self.entry = 0
            self.update(self.symbol, self.status, "")
        except Exception as e:
            self.log(f"❌ Lỗi khi đóng lệnh {self.symbol}: {e}")

class FuturesBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Futures Indicator Bot")
        self.root.geometry("500x700")
        self.root.configure(bg="#121212")
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#121212", foreground="#00ff00")
        style.configure("TFrame", background="#121212")
        style.configure("TLabel", background="#121212", foreground="#00ff00", font=("Arial", 10))
        style.configure("TButton", background="#333", foreground="#00ff00", font=("Arial", 10))
        style.configure("TEntry", fieldbackground="#222", foreground="#00ff00")
        style.configure("TCombobox", fieldbackground="#222", foreground="#00ff00")
        style.map("TButton", background=[('active', '#555')])

        self.bots = {}

        # Header
        header = ttk.Frame(root)
        header.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(header, text="BOT FUTURES BINANCE", font=("Arial", 14, "bold")).pack()

        # Form
        form = ttk.LabelFrame(root, text="Cài đặt bot", padding=10)
        form.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(form, text="Cặp coin:").grid(row=0, column=0, sticky="w", pady=5)
        self.symbol_entry = ttk.Entry(form, width=15)
        self.symbol_entry.grid(row=0, column=1, pady=5, sticky="ew")
        self.symbol_entry.insert(0, "BTCUSDT")
        
        ttk.Label(form, text="Đòn bẩy:").grid(row=1, column=0, sticky="w", pady=5)
        self.lev_entry = ttk.Entry(form, width=15)
        self.lev_entry.grid(row=1, column=1, pady=5, sticky="ew")
        self.lev_entry.insert(0, "10")
        
        ttk.Label(form, text="% Số dư:").grid(row=2, column=0, sticky="w", pady=5)
        self.percent_entry = ttk.Entry(form, width=15)
        self.percent_entry.grid(row=2, column=1, pady=5, sticky="ew")
        self.percent_entry.insert(0, "10")
        
        ttk.Label(form, text="TP %:").grid(row=3, column=0, sticky="w", pady=5)
        self.tp_entry = ttk.Entry(form, width=15)
        self.tp_entry.grid(row=3, column=1, pady=5, sticky="ew")
        self.tp_entry.insert(0, "1")
        
        ttk.Label(form, text="SL %:").grid(row=4, column=0, sticky="w", pady=5)
        self.sl_entry = ttk.Entry(form, width=15)
        self.sl_entry.grid(row=4, column=1, pady=5, sticky="ew")
        self.sl_entry.insert(0, "1")
        
        ttk.Label(form, text="Chỉ báo:").grid(row=5, column=0, sticky="w", pady=5)
        self.indicator_var = tk.StringVar(value="RSI")
        self.indicator_menu = ttk.Combobox(
            form, 
            textvariable=self.indicator_var, 
            values=["RSI", "EMA", "MACD", "Tất cả"], 
            width=13
        )
        self.indicator_menu.grid(row=5, column=1, pady=5, sticky="ew")
        
        add_btn = ttk.Button(form, text="Thêm bot", command=self.menu_add)
        add_btn.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")

        # Bot list
        bot_frame = ttk.LabelFrame(root, text="Các bot đang chạy", padding=10)
        bot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        list_frame = ttk.Frame(bot_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.bot_list = tk.Listbox(
            list_frame, 
            height=6, 
            bg="#222", 
            fg="#00ff00", 
            selectbackground="#555",
            font=("Arial", 10)
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.bot_list.yview)
        self.bot_list.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bot_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stop_btn = ttk.Button(bot_frame, text="Dừng bot đã chọn", command=self.stop_selected_bot)
        stop_btn.pack(fill=tk.X, pady=5)

        # Log
        log_frame = ttk.LabelFrame(root, text="Nhật ký hoạt động", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(
            log_frame, 
            height=8, 
            bg="#222", 
            fg="#00ff00", 
            font=("Arial", 10)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(root, text="Sẵn sàng...", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start position checker
        self.check_positions()

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.status_bar.config(text=msg)

    def update_tree(self, symbol, status, side):
        for idx in range(self.bot_list.size()):
            if self.bot_list.get(idx).startswith(symbol):
                self.bot_list.delete(idx)
                break
        bot = self.bots.get(symbol)
        if bot:
            self.bot_list.insert(tk.END, f"{symbol} | {status} | {side} | {bot.indicator}")

    def menu_add(self):
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            self.log("❌ Vui lòng nhập cặp coin!")
            return
            
        try:
            lev = int(self.lev_entry.get())
            percent = float(self.percent_entry.get())
            tp = float(self.tp_entry.get())
            sl = float(self.sl_entry.get())
            indicator = self.indicator_var.get()
        except ValueError:
            self.log("❌ Vui lòng nhập đúng thông số!")
            return
            
        if symbol in self.bots:
            self.log(f"⚠️ Đã có bot cho {symbol}")
            return
            
        # Kiểm tra API key
        if not API_KEY or not API_SECRET:
            self.log("❌ Chưa cấu hình API Key và Secret Key!")
            return
            
        bot = IndicatorBot(
            symbol, lev, percent, tp, sl, 
            indicator, self.log, self.update_tree
        )
        self.bots[symbol] = bot
        self.bot_list.insert(tk.END, f"{symbol} | {bot.status} | {bot.side} | {indicator}")
        self.log(f"✅ Đã thêm bot cho {symbol} với chỉ báo {indicator}")

    def stop_selected_bot(self):
        selected = self.bot_list.curselection()
        if not selected: 
            return
            
        idx = selected[0]
        line = self.bot_list.get(idx)
        symbol = line.split('|')[0].strip()
        bot = self.bots.get(symbol)
        
        if bot:
            bot.stop()
            if bot.status == "open":
                bot.close_position()
            self.log(f"⛔ Đã dừng bot cho {symbol}")
            self.bot_list.delete(idx)
            del self.bots[symbol]

    def check_positions(self):
        try:
            for symbol, bot in list(self.bots.items()):
                if bot.status == "open":
                    # Kiểm tra vị thế còn tồn tại không
                    ts = int(time.time() * 1000)
                    query = f"symbol={symbol}&timestamp={ts}"
                    sig = sign(query)
                    url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
                    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
                    response = urllib.request.urlopen(req)
                    positions = json.loads(response.read())
                    
                    position_open = False
                    for pos in positions:
                        if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                            position_open = True
                            break
                            
                    if not position_open:
                        bot.status = "waiting"
                        bot.side = ""
                        bot.qty = 0
                        bot.entry = 0
                        bot.log(f"ℹ️ Vị thế {symbol} đã đóng")
                        bot.update(symbol, bot.status, bot.side)
                        
        except Exception as e:
            self.log(f"⚠️ Lỗi kiểm tra vị thế: {e}")
            
        self.root.after(30000, self.check_positions)

if __name__ == "__main__":
    root = tk.Tk()
    gui = FuturesBotGUI(root)
    root.mainloop()
