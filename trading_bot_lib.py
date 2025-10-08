# ==============================================================
# TRADING BOT MANAGER - PHẦN 2: BOT MANAGER & MAIN SYSTEM
# ==============================================================

import threading
import time
import requests
from trading_bot_core import *

class BotManager:
    def __init__(self, api_key=None, api_secret=None, telegram_bot_token=None, telegram_chat_id=None):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        
        # Cache để tối ưu hiệu năng
        self._balance_cache = {"value": None, "timestamp": 0}
        self._positions_cache = {"value": None, "timestamp": 0}
        self.cache_ttl = 30
        
        # Session cho requests
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=100, max_retries=3))
        
        self.auto_strategies = {}
        self.last_auto_scan = 0
        self.auto_scan_interval = 600
        
        # Cooldown strategies
        self.strategy_cooldowns = {
            "Reverse 24h": {},
            "Scalping": {},
            "Trend Following": {},
            "Safe Grid": {},
            "Smart Dynamic": {}
        }
        self.cooldown_period = 300
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG BOT THÔNG MINH ĐÃ KHỞI ĐỘNG")
            
            # Khởi động Telegram listener
            if self.telegram_bot_token and self.telegram_chat_id:
                self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
                self.telegram_thread.start()
                
                # Gửi menu chào mừng sau 2 giây
                threading.Timer(2.0, lambda: self.send_main_menu(self.telegram_chat_id)).start()
            
            # Auto scan thread
            self.auto_scan_thread = threading.Thread(target=self._auto_scan_loop, daemon=True)
            self.auto_scan_thread.start()
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def _verify_api_connection(self):
        """Xác minh kết nối API Binance"""
        try:
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                self.log("❌ LỖI: Không thể kết nối Binance API.")
            else:
                self.log(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
        except Exception as e:
            self.log(f"❌ Lỗi xác minh API: {str(e)}")

    def log(self, message):
        """Ghi log hệ thống"""
        logger.info(f"[SYSTEM] {message}")
        # Chỉ gửi log quan trọng qua Telegram để tránh spam
        if self.telegram_bot_token and self.telegram_chat_id and any(keyword in message for keyword in ["✅", "❌", "🔄", "⛔", "🎯"]):
            try:
                send_telegram(f"<b>SYSTEM</b>: {message}", 
                             bot_token=self.telegram_bot_token, 
                             default_chat_id=self.telegram_chat_id)
            except:
                pass

    def send_main_menu(self, chat_id):
        """Gửi menu chính"""
        try:
            welcome = (
                "🤖 <b>BOT GIAO DỊCH FUTURES THÔNG MINH</b>\n\n"
                "🎯 <b>HỆ THỐNG ĐA CHIẾN LƯỢC + SMART EXIT + BOT ĐỘNG</b>\n\n"
                "💡 <i>Chọn chức năng từ menu bên dưới:</i>"
            )
            send_telegram(welcome, chat_id, create_main_menu(),
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)
        except Exception as e:
            logger.error(f"Lỗi gửi menu chính: {str(e)}")

    def get_cached_balance(self):
        """Lấy số dư với cache"""
        try:
            current_time = time.time()
            if (self._balance_cache["value"] is None or 
                current_time - self._balance_cache["timestamp"] > self.cache_ttl):
                balance = get_balance(self.api_key, self.api_secret)
                self._balance_cache = {"value": balance, "timestamp": current_time}
            return self._balance_cache["value"]
        except Exception as e:
            logger.error(f"Lỗi lấy số dư: {str(e)}")
            return None

    def get_cached_positions(self):
        """Lấy vị thế với cache"""
        try:
            current_time = time.time()
            if (self._positions_cache["value"] is None or 
                current_time - self._positions_cache["timestamp"] > self.cache_ttl):
                positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
                self._positions_cache = {"value": positions, "timestamp": current_time}
            return self._positions_cache["value"]
        except Exception as e:
            logger.error(f"Lỗi lấy vị thế: {str(e)}")
            return []

    def _is_in_cooldown(self, strategy_type, config_key):
        """Kiểm tra chiến lược có đang trong cooldown không"""
        if strategy_type not in self.strategy_cooldowns:
            return False
            
        last_cooldown_time = self.strategy_cooldowns[strategy_type].get(config_key)
        if last_cooldown_time is None:
            return False
            
        current_time = time.time()
        if current_time - last_cooldown_time < self.cooldown_period:
            return True
            
        # Hết cooldown, xóa khỏi danh sách
        del self.strategy_cooldowns[strategy_type][config_key]
        return False

    def _auto_scan_loop(self):
        """Vòng lặp tự động quét coin"""
        while self.running:
            try:
                current_time = time.time()
                
                # Kiểm tra bot động cần tìm coin mới
                bot_removed = False
                for bot_id, bot in list(self.bots.items()):
                    if hasattr(bot, 'should_be_removed') and bot.should_be_removed:
                        self.stop_bot(bot_id)
                        bot_removed = True
                
                # Quét auto strategies nếu đủ thời gian hoặc có bot bị xóa
                if (current_time - self.last_auto_scan > self.auto_scan_interval or 
                    bot_removed or 
                    any(not bot.position_open and hasattr(bot, 'dynamic_mode') and bot.dynamic_mode for bot in self.bots.values())):
                    
                    self._scan_auto_strategies()
                    self.last_auto_scan = current_time
                
                time.sleep(30)
                
            except Exception as e:
                self.log(f"❌ Lỗi auto scan: {str(e)}")
                time.sleep(60)

    def _scan_auto_strategies(self):
        """Quét và bổ sung coin cho chiến thuật tự động"""
        if not self.auto_strategies:
            return
            
        self.log("🔄 Đang quét coin cho các cấu hình tự động...")
        
        for strategy_key, strategy_config in self.auto_strategies.items():
            try:
                strategy_type = strategy_config['strategy_type']
                
                # Kiểm tra cooldown
                if self._is_in_cooldown(strategy_type, strategy_key):
                    continue
                
                coin_manager = CoinManager()
                current_bots_count = coin_manager.count_bots_by_config(strategy_key)
                
                if current_bots_count < 2:
                    self.log(f"🔄 {strategy_type}: đang có {current_bots_count}/2 bot, tìm thêm coin...")
                    
                    qualified_symbols = self._find_qualified_symbols(strategy_type, strategy_config, strategy_key)
                    
                    added_count = 0
                    for symbol in qualified_symbols:
                        if added_count >= (2 - current_bots_count):
                            break
                            
                        bot_id = f"{symbol}_{strategy_key}"
                        if bot_id not in self.bots:
                            success = self._create_auto_bot(symbol, strategy_type, strategy_config)
                            if success:
                                added_count += 1
                                time.sleep(0.5)
                    
                    if added_count > 0:
                        self.log(f"✅ {strategy_type}: đã thêm {added_count} bot mới")
                        
            except Exception as e:
                self.log(f"❌ Lỗi quét {strategy_type}: {str(e)}")

    def _find_qualified_symbols(self, strategy_type, config, strategy_key):
        """Tìm coin phù hợp cho chiến lược"""
        try:
            leverage = config['leverage']
            threshold = config.get('threshold', 30)
            volatility = config.get('volatility', 3)
            grid_levels = config.get('grid_levels', 5)
            
            qualified_symbols = get_qualified_symbols(
                self.api_key, self.api_secret, strategy_type, leverage,
                threshold, volatility, grid_levels, 
                max_candidates=15,
                final_limit=2,
                strategy_key=strategy_key
            )
            
            return qualified_symbols
            
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            return []

    def _create_auto_bot(self, symbol, strategy_type, config):
        """Tạo bot tự động"""
        try:
            leverage = config['leverage']
            percent = config['percent']
            tp = config['tp']
            sl = config['sl']
            strategy_key = config['strategy_key']
            smart_exit_config = config.get('smart_exit_config', {})
            dynamic_mode = config.get('dynamic_mode', True)
            
            bot_class = {
                "Reverse 24h": Reverse_24h_Bot,
                "Scalping": Scalping_Bot,
                "Safe Grid": Safe_Grid_Bot,
                "Trend Following": Trend_Following_Bot,
                "Smart Dynamic": SmartDynamicBot
            }.get(strategy_type)
            
            if not bot_class:
                return False
            
            # Kiểm tra symbol có tồn tại không
            current_price = get_current_price(symbol)
            if current_price <= 0:
                return False
            
            if strategy_type == "Reverse 24h":
                threshold = config.get('threshold', 30)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token, 
                              self.telegram_chat_id, threshold, strategy_key, smart_exit_config, dynamic_mode)
            elif strategy_type == "Safe Grid":
                grid_levels = config.get('grid_levels', 5)
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels, strategy_key, smart_exit_config, dynamic_mode)
            else:
                bot = bot_class(symbol, leverage, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, strategy_key, smart_exit_config, dynamic_mode)
            
            bot_id = f"{symbol}_{strategy_key}"
            self.bots[bot_id] = bot
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot {symbol}: {str(e)}")
            return False

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, **kwargs):
        """Thêm bot mới - PHIÊN BẢN ĐÃ SỬA LỖI"""
        try:
            # Validation
            if sl == 0:
                sl = None
                
            if not self.api_key or not self.api_secret:
                self.log("❌ Chưa thiết lập API Key")
                return False
            
            # Test connection
            test_balance = self.get_cached_balance()
            if test_balance is None:
                self.log("❌ LỖI: Không thể kết nối Binance")
                return False
            
            smart_exit_config = kwargs.get('smart_exit_config', {})
            dynamic_mode = kwargs.get('dynamic_mode', False)
            threshold = kwargs.get('threshold')
            volatility = kwargs.get('volatility')
            grid_levels = kwargs.get('grid_levels')
            
            # Xử lý theo từng loại bot
            if strategy_type == "Smart Dynamic":
                return self._create_smart_dynamic_bot(lev, percent, tp, sl, smart_exit_config, dynamic_mode)
            elif dynamic_mode and strategy_type in ["Reverse 24h", "Scalping", "Safe Grid", "Trend Following"]:
                return self._create_dynamic_bot(strategy_type, lev, percent, tp, sl, smart_exit_config, threshold, volatility, grid_levels)
            else:
                return self._create_static_bot(symbol, strategy_type, lev, percent, tp, sl, smart_exit_config)
                
        except Exception as e:
            self.log(f"❌ Lỗi nghiêm trọng trong add_bot: {str(e)}")
            return False

    def _create_smart_dynamic_bot(self, lev, percent, tp, sl, smart_exit_config, dynamic_mode):
        """Tạo bot động thông minh"""
        try:
            strategy_key = f"SmartDynamic_{lev}_{percent}_{tp}_{sl}"
            
            if self._is_in_cooldown("Smart Dynamic", strategy_key):
                self.log(f"⏰ Smart Dynamic: đang trong cooldown")
                return False
            
            self.auto_strategies[strategy_key] = {
                'strategy_type': "Smart Dynamic",
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key,
                'smart_exit_config': smart_exit_config,
                'dynamic_mode': True
            }
            
            qualified_symbols = self._find_qualified_symbols(
                "Smart Dynamic", self.auto_strategies[strategy_key], strategy_key
            )
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, "Smart Dynamic", self.auto_strategies[strategy_key])
                    if success:
                        success_count += 1
                        if success_count >= 2:
                            break
            
            if success_count > 0:
                success_msg = f"✅ ĐÃ TẠO {success_count} BOT ĐỘNG THÔNG MINH"
                self.log(success_msg)
                return True
            else:
                self.log("⚠️ Smart Dynamic: chưa tìm thấy coin phù hợp")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tạo Smart Dynamic bot: {str(e)}")
            return False

    def _create_dynamic_bot(self, strategy_type, lev, percent, tp, sl, smart_exit_config, threshold, volatility, grid_levels):
        """Tạo bot động cho các chiến lược"""
        try:
            strategy_key = f"{strategy_type}_{lev}_{percent}_{tp}_{sl}"
            
            if strategy_type == "Reverse 24h":
                strategy_key += f"_th{threshold or 30}"
            elif strategy_type == "Scalping":
                strategy_key += f"_vol{volatility or 3}"
            elif strategy_type == "Safe Grid":
                strategy_key += f"_grid{grid_levels or 5}"
            
            if self._is_in_cooldown(strategy_type, strategy_key):
                self.log(f"⏰ {strategy_type}: đang trong cooldown")
                return False
            
            config = {
                'strategy_type': strategy_type,
                'leverage': lev,
                'percent': percent,
                'tp': tp,
                'sl': sl,
                'strategy_key': strategy_key,
                'smart_exit_config': smart_exit_config,
                'dynamic_mode': True
            }
            
            if threshold: config['threshold'] = threshold
            if volatility: config['volatility'] = volatility
            if grid_levels: config['grid_levels'] = grid_levels
            
            self.auto_strategies[strategy_key] = config
            
            qualified_symbols = self._find_qualified_symbols(strategy_type, config, strategy_key)
            
            success_count = 0
            for symbol in qualified_symbols:
                bot_id = f"{symbol}_{strategy_key}"
                if bot_id not in self.bots:
                    success = self._create_auto_bot(symbol, strategy_type, config)
                    if success:
                        success_count += 1
                        if success_count >= 2:
                            break
            
            if success_count > 0:
                success_msg = f"✅ ĐÃ TẠO {success_count} BOT {strategy_type}"
                self.log(success_msg)
                return True
            else:
                self.log(f"⚠️ {strategy_type}: chưa tìm thấy coin phù hợp")
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tạo {strategy_type} bot: {str(e)}")
            return False

    def _create_static_bot(self, symbol, strategy_type, lev, percent, tp, sl, smart_exit_config):
        """Tạo bot tĩnh"""
        try:
            symbol = symbol.upper() if symbol else "BTCUSDT"
            bot_id = f"{symbol}_{strategy_type}"
            
            if bot_id in self.bots:
                self.log(f"⚠️ Đã có bot {strategy_type} cho {symbol}")
                return False
            
            bot_class = {
                "RSI/EMA Recursive": RSI_EMA_Bot,
                "EMA Crossover": EMA_Crossover_Bot,
                "Reverse 24h": Reverse_24h_Bot,
                "Trend Following": Trend_Following_Bot,
                "Scalping": Scalping_Bot,
                "Safe Grid": Safe_Grid_Bot
            }.get(strategy_type)
            
            if not bot_class:
                self.log(f"❌ Chiến lược {strategy_type} không được hỗ trợ")
                return False
            
            # Validate symbol
            current_price = get_current_price(symbol)
            if current_price <= 0:
                self.log(f"❌ {symbol}: không lấy được giá")
                return False
            
            if strategy_type == "Reverse 24h":
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, threshold=30, smart_exit_config=smart_exit_config)
            elif strategy_type == "Safe Grid":
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, grid_levels=5, smart_exit_config=smart_exit_config)
            else:
                bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                              self.api_key, self.api_secret, self.telegram_bot_token,
                              self.telegram_chat_id, smart_exit_config=smart_exit_config)
            
            self.bots[bot_id] = bot
            self.log(f"✅ Đã thêm bot {strategy_type}: {symbol}")
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot tĩnh {symbol}: {str(e)}")
            return False

    def stop_bot(self, bot_id):
        """Dừng bot cụ thể"""
        bot = self.bots.get(bot_id)
        if bot:
            try:
                bot.stop()
                # Thêm cooldown nếu là bot động
                if hasattr(bot, 'config_key') and bot.config_key and bot.strategy_name in self.strategy_cooldowns:
                    self.strategy_cooldowns[bot.strategy_name][bot.config_key] = time.time()
                
                del self.bots[bot_id]
                # Clear cache
                self._positions_cache = {"value": None, "timestamp": 0}
                self.log(f"⛔ Đã dừng bot {bot_id}")
                return True
            except Exception as e:
                self.log(f"❌ Lỗi khi dừng bot {bot_id}: {str(e)}")
        return False

    def stop_all(self):
        """Dừng toàn bộ hệ thống"""
        self.log("⛔ Đang dừng tất cả bot...")
        self.running = False
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.session.close()
        self.log("🔴 Hệ thống đã dừng hoàn toàn")

    def _telegram_listener(self):
        """Listener Telegram - ĐÃ SỬA LỖI"""
        last_update_id = 0
        error_count = 0
        
        self.log("🔗 Telegram listener đang khởi động...")
        
        while self.running and self.telegram_bot_token and self.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
                params = {
                    "offset": last_update_id + 1,
                    "timeout": 30,
                    "allowed_updates": ["message"]
                }
                
                response = self.session.get(url, params=params, timeout=35)
                
                if response.status_code == 200:
                    data = response.json()
                    error_count = 0
                    
                    if data.get('ok') and data.get('result'):
                        for update in data['result']:
                            update_id = update['update_id']
                            message = update.get('message', {})
                            chat_id = str(message.get('chat', {}).get('id'))
                            text = message.get('text', '').strip()
                            
                            if chat_id == self.telegram_chat_id and text:
                                if update_id > last_update_id:
                                    last_update_id = update_id
                                    self._handle_telegram_message(chat_id, text)
                else:
                    error_count += 1
                    if error_count > 5:
                        self.log("⚠️ Quá nhiều lỗi Telegram, tạm dừng 60s")
                        time.sleep(60)
                        error_count = 0
                    else:
                        time.sleep(10)
                
            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                error_count += 1
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(10)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn Telegram - PHIÊN BẢN ĐẦY ĐỦ"""
        try:
            user_state = self.user_states.get(chat_id, {})
            current_step = user_state.get('step')
            
            # Xử lý hủy bỏ ở mọi bước
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thao tác", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
                return
            
            # Xử lý theo từng bước
            step_handlers = {
                'waiting_bot_mode': self._handle_bot_mode,
                'waiting_strategy': self._handle_strategy,
                'waiting_exit_strategy': self._handle_exit_strategy,
                'waiting_smart_config': self._handle_smart_config,
                'waiting_threshold': self._handle_threshold,
                'waiting_volatility': self._handle_volatility,
                'waiting_grid_levels': self._handle_grid_levels,
                'waiting_symbol': self._handle_symbol,
                'waiting_leverage': self._handle_leverage,
                'waiting_percent': self._handle_percent,
                'waiting_tp': self._handle_tp,
                'waiting_sl': self._handle_sl
            }
            
            if current_step in step_handlers:
                step_handlers[current_step](chat_id, text, user_state)
                return
            
            # Xử lý lệnh chính
            command_handlers = {
                "➕ Thêm Bot": lambda: self._handle_add_bot(chat_id),
                "📊 Danh sách Bot": lambda: self._handle_list_bots(chat_id),
                "⛔ Dừng Bot": lambda: self._handle_stop_bot(chat_id),
                "💰 Số dư": lambda: self._handle_balance(chat_id),
                "📈 Vị thế": lambda: self._handle_positions(chat_id),
                "🎯 Chiến lược": lambda: self._handle_strategies(chat_id),
                "⚙️ Cấu hình": lambda: self._handle_config(chat_id)
            }
            
            if text in command_handlers:
                command_handlers[text]()
            elif text.startswith("⛔ "):
                self._handle_stop_specific_bot(chat_id, text)
            else:
                self.send_main_menu(chat_id)
                
        except Exception as e:
            logger.error(f"Lỗi xử lý Telegram: {str(e)}")
            send_telegram("❌ Có lỗi xảy ra, vui lòng thử lại", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

    # ========== CÁC HANDLER CHO TỪNG BƯỚC ==========

    def _handle_add_bot(self, chat_id):
        """Bắt đầu thêm bot"""
        self.user_states[chat_id] = {'step': 'waiting_bot_mode'}
        balance = self.get_cached_balance()
        
        if balance is None:
            send_telegram("❌ LỖI KẾT NỐI BINANCE", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = {}
            return
        
        send_telegram(f"💰 Số dư: {balance:.2f} USDT\n\nChọn chế độ bot:", chat_id,
                     create_bot_mode_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_bot_mode(self, chat_id, text, user_state):
        if text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
            user_state['dynamic_mode'] = (text == "🔄 Bot Động - Tự tìm coin")
            user_state['step'] = 'waiting_strategy'
            
            mode_text = "ĐỘNG - Tự tìm coin" if user_state['dynamic_mode'] else "TĨNH - Coin cố định"
            send_telegram(f"✅ Đã chọn: {mode_text}\n\nChọn chiến lược:", chat_id,
                         create_strategy_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state

    def _handle_strategy(self, chat_id, text, user_state):
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
            strategy = strategy_map[text]
            user_state['strategy'] = strategy
            
            # Smart Dynamic luôn là bot động
            if strategy == "Smart Dynamic":
                user_state['dynamic_mode'] = True
            
            user_state['step'] = 'waiting_exit_strategy'
            
            mode_text = "ĐỘNG" if user_state.get('dynamic_mode') else "TĨNH"
            send_telegram(f"✅ Chiến lược: {strategy}\nChế độ: {mode_text}\n\nChọn chiến lược thoát lệnh:", chat_id,
                         create_exit_strategy_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state

    def _handle_exit_strategy(self, chat_id, text, user_state):
        exit_configs = {
            "🔄 Thoát lệnh thông minh": {'step': 'waiting_smart_config', 'config': {}},
            "⚡ Thoát lệnh cơ bản": {'step': None, 'config': {
                'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': False
            }},
            "🎯 Chỉ TP/SL cố định": {'step': None, 'config': {
                'enable_trailing': False, 'enable_time_exit': False, 'enable_support_resistance': False
            }}
        }
        
        if text in exit_configs:
            config_info = exit_configs[text]
            user_state['smart_exit_config'] = config_info['config']
            
            if config_info['step']:
                user_state['step'] = config_info['step']
                send_telegram("Chọn cấu hình Smart Exit:", chat_id,
                             create_smart_exit_config_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            else:
                self._continue_bot_creation(chat_id, user_state)
            
            self.user_states[chat_id] = user_state

    def _handle_smart_config(self, chat_id, text, user_state):
        smart_configs = {
            "Trailing: 30/15": {'enable_trailing': True, 'trailing_activation': 30, 'trailing_distance': 15},
            "Trailing: 50/20": {'enable_trailing': True, 'trailing_activation': 50, 'trailing_distance': 20},
            "Time Exit: 4h": {'enable_time_exit': True, 'max_hold_time': 4},
            "Time Exit: 8h": {'enable_time_exit': True, 'max_hold_time': 8},
            "Kết hợp Full": {'enable_trailing': True, 'enable_time_exit': True, 'enable_support_resistance': True},
            "Cơ bản": {'enable_trailing': True, 'enable_time_exit': True}
        }
        
        if text in smart_configs:
            user_state['smart_exit_config'].update(smart_configs[text])
            self._continue_bot_creation(chat_id, user_state)

    def _handle_threshold(self, chat_id, text, user_state):
        try:
            threshold = float(text)
            user_state['threshold'] = threshold
            user_state['step'] = 'waiting_leverage'
            send_telegram(f"✅ Ngưỡng: {threshold}%\n\nChọn đòn bẩy:", chat_id,
                         create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_threshold_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_volatility(self, chat_id, text, user_state):
        try:
            volatility = float(text)
            user_state['volatility'] = volatility
            user_state['step'] = 'waiting_leverage'
            send_telegram(f"✅ Biến động: {volatility}%\n\nChọn đòn bẩy:", chat_id,
                         create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_volatility_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_grid_levels(self, chat_id, text, user_state):
        try:
            grid_levels = int(text)
            user_state['grid_levels'] = grid_levels
            user_state['step'] = 'waiting_leverage'
            send_telegram(f"✅ Số lệnh: {grid_levels}\n\nChọn đòn bẩy:", chat_id,
                         create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = user_state
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_grid_levels_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_symbol(self, chat_id, text, user_state):
        symbol = text.upper()
        user_state['symbol'] = symbol
        user_state['step'] = 'waiting_leverage'
        send_telegram(f"✅ Coin: {symbol}\n\nChọn đòn bẩy:", chat_id,
                     create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
        self.user_states[chat_id] = user_state

    def _handle_leverage(self, chat_id, text, user_state):
        try:
            leverage = int(text.replace('x', '').strip())
            if 1 <= leverage <= 100:
                user_state['leverage'] = leverage
                user_state['step'] = 'waiting_percent'
                send_telegram(f"✅ Đòn bẩy: {leverage}x\n\nChọn % số dư:", chat_id,
                             create_percent_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state
            else:
                send_telegram("⚠️ Đòn bẩy phải từ 1-100x", chat_id,
                             create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_leverage_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_percent(self, chat_id, text, user_state):
        try:
            percent = float(text)
            if 0 < percent <= 100:
                user_state['percent'] = percent
                user_state['step'] = 'waiting_tp'
                send_telegram(f"✅ Số dư: {percent}%\n\nChọn Take Profit (%):", chat_id,
                             create_tp_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state
            else:
                send_telegram("⚠️ % số dư phải từ 0.1-100", chat_id,
                             create_percent_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_percent_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_tp(self, chat_id, text, user_state):
        try:
            tp = float(text)
            if tp >= 0:
                user_state['tp'] = tp
                user_state['step'] = 'waiting_sl'
                send_telegram(f"✅ TP: {tp}%\n\nChọn Stop Loss (%):", chat_id,
                             create_sl_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = user_state
            else:
                send_telegram("⚠️ TP phải >= 0", chat_id,
                             create_tp_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_tp_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _handle_sl(self, chat_id, text, user_state):
        try:
            sl = float(text)
            if sl >= 0:
                user_state['sl'] = sl
                
                # Tạo bot cuối cùng
                success = self._create_bot_from_state(chat_id, user_state)
                
                if success:
                    strategy = user_state.get('strategy')
                    leverage = user_state.get('leverage')
                    percent = user_state.get('percent')
                    dynamic_mode = user_state.get('dynamic_mode', False)
                    
                    success_msg = (
                        f"✅ <b>BOT ĐÃ ĐƯỢC TẠO THÀNH CÔNG!</b>\n\n"
                        f"🤖 {strategy}\n"
                        f"💰 {leverage}x | {percent}% số dư\n"
                        f"🎯 TP: {user_state.get('tp')}% | SL: {sl}%\n"
                        f"🔧 Chế độ: {'ĐỘNG' if dynamic_mode else 'TĨNH'}\n\n"
                        f"📈 Theo dõi trong danh sách bot!"
                    )
                else:
                    success_msg = "❌ <b>LỖI TẠO BOT!</b>\n\nKiểm tra API Key và số dư!"
                
                send_telegram(success_msg, chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
                
                # Reset state
                self.user_states[chat_id] = {}
                
            else:
                send_telegram("⚠️ SL phải >= 0", chat_id,
                             create_sl_keyboard(), self.telegram_bot_token, self.telegram_chat_id)
        except ValueError:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ", chat_id,
                         create_sl_keyboard(), self.telegram_bot_token, self.telegram_chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        """Tiếp tục quy trình tạo bot"""
        strategy = user_state.get('strategy')
        dynamic_mode = user_state.get('dynamic_mode', False)
        
        next_steps = {
            ("Reverse 24h", True): 'waiting_threshold',
            ("Scalping", True): 'waiting_volatility',
            ("Safe Grid", True): 'waiting_grid_levels',
        }
        
        next_step = next_steps.get((strategy, dynamic_mode), 
                                 'waiting_symbol' if not dynamic_mode else 'waiting_leverage')
        
        user_state['step'] = next_step
        
        step_messages = {
            'waiting_threshold': ("Chọn ngưỡng biến động (%):", create_threshold_keyboard()),
            'waiting_volatility': ("Chọn biến động tối thiểu (%):", create_volatility_keyboard()),
            'waiting_grid_levels': ("Chọn số lệnh grid:", create_grid_levels_keyboard()),
            'waiting_symbol': ("Chọn cặp coin:", create_symbols_keyboard(strategy)),
            'waiting_leverage': ("Chọn đòn bẩy:", create_leverage_keyboard(strategy))
        }
        
        if next_step in step_messages:
            message, keyboard = step_messages[next_step]
            send_telegram(message, chat_id, keyboard, self.telegram_bot_token, self.telegram_chat_id)
        
        self.user_states[chat_id] = user_state

    def _create_bot_from_state(self, chat_id, user_state):
        """Tạo bot từ state"""
        try:
            return self.add_bot(
                symbol=user_state.get('symbol'),
                lev=user_state.get('leverage'),
                percent=user_state.get('percent'),
                tp=user_state.get('tp'),
                sl=user_state.get('sl'),
                strategy_type=user_state.get('strategy'),
                dynamic_mode=user_state.get('dynamic_mode', False),
                smart_exit_config=user_state.get('smart_exit_config', {}),
                threshold=user_state.get('threshold'),
                volatility=user_state.get('volatility'),
                grid_levels=user_state.get('grid_levels')
            )
        except Exception as e:
            self.log(f"❌ Lỗi tạo bot từ state: {str(e)}")
            return False

    # ========== CÁC HANDLER LỆNH CHÍNH ==========

    def _handle_list_bots(self, chat_id):
        if not self.bots:
            send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
            message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
            for bot_id, bot in self.bots.items():
                status = "🟢 Mở" if bot.position_open else "🟡 Chờ"
                mode = "🔄 Động" if getattr(bot, 'dynamic_mode', False) else "🤖 Tĩnh"
                message += f"🔹 {bot_id} | {status} | {mode}\n"
            
            message += f"\n📊 Tổng số: {len(self.bots)} bot"
            send_telegram(message, chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

    def _handle_stop_bot(self, chat_id):
        if not self.bots:
            send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
            keyboard = []
            for bot_id in self.bots.keys():
                keyboard.append([{"text": f"⛔ {bot_id}"}])
            keyboard.append([{"text": "❌ Hủy bỏ"}])
            
            send_telegram("⛔ <b>CHỌN BOT ĐỂ DỪNG</b>", chat_id,
                         {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_stop_specific_bot(self, chat_id, text):
        bot_id = text.replace("⛔ ", "").strip()
        if self.stop_bot(bot_id):
            send_telegram(f"⛔ Đã dừng bot {bot_id}", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)
        else:
            send_telegram(f"⚠️ Không tìm thấy bot {bot_id}", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_balance(self, chat_id):
        balance = self.get_cached_balance()
        if balance is None:
            send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
            send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

    def _handle_positions(self, chat_id):
        positions = self.get_cached_positions()
        if not positions:
            send_telegram("📭 Không có vị thế nào đang mở", chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        else:
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

    def _handle_strategies(self, chat_id):
        strategy_info = "🎯 <b>DANH SÁCH CHIẾN LƯỢC</b>\n\n" \
                       "🔄 <b>Bot Động Thông minh</b>\n• Tự động tìm coin\n• Đa chiến lược\n\n" \
                       "🎯 <b>Reverse 24h</b>\n• Đảo chiều biến động\n\n" \
                       "⚡ <b>Scalping</b>\n• Giao dịch tốc độ cao\n\n" \
                       "🛡️ <b>Safe Grid</b>\n• Grid an toàn\n\n" \
                       "📈 <b>Trend Following</b>\n• Theo xu hướng"
        
        send_telegram(strategy_info, chat_id,
                     bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

    def _handle_config(self, chat_id):
        balance = self.get_cached_balance()
        api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
        
        config_info = (
            f"⚙️ <b>CẤU HÌNH HỆ THỐNG</b>\n\n"
            f"🔑 Binance API: {api_status}\n"
            f"🤖 Tổng số bot: {len(self.bots)}\n"
            f"🔄 Bot động: {sum(1 for b in self.bots.values() if getattr(b, 'dynamic_mode', False))}\n"
            f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối"
        )
        send_telegram(config_info, chat_id,
                     bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)

# ==============================================================
