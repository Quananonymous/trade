# main.py
from trading_bot_lib import BotManager  # Import từ file mới
import os
import json
import time
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Lấy cấu hình từ biến môi trường
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Kiểm tra và in thông tin cấu hình (không in secret key)
def check_config():
    logger.info("🔍 Kiểm tra cấu hình...")
    logger.info(f"BINANCE_API_KEY: {'✅ Đã cấu hình' if BINANCE_API_KEY else '❌ Chưa cấu hình'}")
    logger.info(f"BINANCE_SECRET_KEY: {'✅ Đã cấu hình' if BINANCE_SECRET_KEY else '❌ Chưa cấu hình'}")
    logger.info(f"TELEGRAM_BOT_TOKEN: {'✅ Đã cấu hình' if TELEGRAM_BOT_TOKEN else '❌ Chưa cấu hình'}")
    logger.info(f"TELEGRAM_CHAT_ID: {'✅ Đã cấu hình' if TELEGRAM_CHAT_ID else '❌ Chưa cấu hình'}")

# Cấu hình bot từ biến môi trường (dạng JSON)
def load_bot_configs():
    bot_config_json = os.getenv('BOT_CONFIGS', '[]')
    try:
        configs = json.loads(bot_config_json)
        logger.info(f"📋 Đã tải {len(configs)} cấu hình bot")
        return configs
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
        return []

# Cấu hình mặc định nếu không có cấu hình từ biến môi trường
DEFAULT_BOT_CONFIGS = []

def test_binance_connection(api_key: str, api_secret: str) -> bool:
    """Kiểm tra kết nối Binance"""
    try:
        from trading_bot_lib_optimized import get_balance_fast
        balance = get_balance_fast(api_key, api_secret)
        if balance > 0:
            logger.info(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
            return True
        else:
            logger.error("❌ Không thể lấy số dư từ Binance")
            return False
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối Binance: {e}")
        return False

def test_telegram_connection(bot_token: str, chat_id: str) -> bool:
    """Kiểm tra kết nối Telegram"""
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Kết nối Telegram thành công!")
            return True
        else:
            logger.error(f"❌ Lỗi kết nối Telegram: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Lỗi kết nối Telegram: {e}")
        return False

def main():
    # Kiểm tra cấu hình
    check_config()
    
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        logger.error("❌ Chưa cấu hình API Key và Secret Key!")
        logger.info("💡 Vui lòng thiết lập các biến môi trường:")
        logger.info("   - BINANCE_API_KEY")
        logger.info("   - BINANCE_SECRET_KEY")
        return
    
    # Kiểm tra kết nối Binance
    if not test_binance_connection(BINANCE_API_KEY, BINANCE_SECRET_KEY):
        logger.error("❌ Không thể kết nối Binance. Vui lòng kiểm tra API Key và Secret Key.")
        return
    
    # Kiểm tra kết nối Telegram (nếu có cấu hình)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        if not test_telegram_connection(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID):
            logger.warning("⚠️ Không thể kết nối Telegram, bot sẽ chạy không có thông báo")
    else:
        logger.warning("⚠️ Chưa cấu hình Telegram, bot sẽ chạy không có thông báo")
    
    logger.info("🟢 Đang khởi động hệ thống bot...")
    
    try:
        # Khởi tạo hệ thống
        manager = BotManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY,
            telegram_bot_token=TELEGRAM_BOT_TOKEN or None,
            telegram_chat_id=TELEGRAM_CHAT_ID or None
        )
        
        # Tải cấu hình bot
        bot_configs = load_bot_configs()
        if not bot_configs:
            logger.info("⚠️ Không tìm thấy cấu hình từ biến môi trường, sử dụng cấu hình mặc định")
            bot_configs = DEFAULT_BOT_CONFIGS
        
        # Thêm các bot từ cấu hình
        logger.info(f"🟢 Đang khởi động {len(bot_configs)} bot...")
        success_count = 0
        
        for i, config in enumerate(bot_configs):
            try:
                if len(config) >= 6:
                    symbol = config[0]
                    leverage = int(config[1])
                    percent = float(config[2])
                    take_profit = float(config[3])
                    stop_loss = float(config[4])
                    strategy = config[5]
                    dynamic_mode = bool(config[6]) if len(config) > 6 else False
                    threshold = int(config[7]) if len(config) > 7 else None
                    
                    logger.info(f"🤖 Đang tạo bot {i+1}: {strategy} cho {symbol}...")
                    
                    # Thêm bot với các tham số tùy chọn
                    kwargs = {}
                    if strategy == "Reverse24h" and threshold is not None:
                        kwargs['threshold'] = threshold
                    
                    if manager.add_bot(
                        symbol=symbol,
                        leverage=leverage,
                        percent=percent,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        strategy=strategy,
                        dynamic_mode=dynamic_mode,
                        **kwargs
                    ):
                        success_count += 1
                        logger.info(f"✅ Bot {i+1} khởi động thành công")
                    else:
                        logger.error(f"❌ Bot {i+1} khởi động thất bại")
                else:
                    logger.error(f"❌ Cấu hình bot {i+1} không hợp lệ: {config}")
                    
            except Exception as e:
                logger.error(f"❌ Lỗi khi tạo bot {i+1}: {e}")
        
        logger.info(f"🎯 Đã khởi động thành công {success_count}/{len(bot_configs)} bot")
        
        if success_count == 0:
            logger.warning("⚠️ Không có bot nào được khởi động, hệ thống sẽ dừng")
            return
        
        # Hiển thị trạng thái hệ thống
        manager.get_status()
        
        logger.info("🟢 Hệ thống đã sẵn sàng. Đang chạy...")
        logger.info("💡 Nhấn Ctrl+C để dừng hệ thống")
        
        # Giữ chương trình chạy
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Nhận tín hiệu dừng từ người dùng...")
    except Exception as e:
        logger.error(f"❌ LỖI HỆ THỐNG: {e}")
    finally:
        if 'manager' in locals():
            logger.info("🛑 Đang dừng hệ thống...")
            manager.stop_all()
        logger.info("🔴 Hệ thống đã dừng")

if __name__ == "__main__":
    main()

