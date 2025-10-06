# main.py
from trading_bot_lib import BotManager
import os
import json
import time

# Lấy cấu hình từ biến môi trường
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# In ra để kiểm tra (không in secret key)
print(f"BINANCE_API_KEY: {'***' if BINANCE_API_KEY else 'Không có'}")
print(f"BINANCE_SECRET_KEY: {'***' if BINANCE_SECRET_KEY else 'Không có'}")
print(f"TELEGRAM_BOT_TOKEN: {'***' if TELEGRAM_BOT_TOKEN else 'Không có'}")
print(f"TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else 'Không có'}")

# Cấu hình bot từ biến môi trường (dạng JSON)
bot_config_json = os.getenv('BOT_CONFIGS', '[]')
try:
    BOT_CONFIGS = json.loads(bot_config_json)
except Exception as e:
    print(f"Lỗi phân tích cấu hình BOT_CONFIGS: {e}")
    BOT_CONFIGS = []

def main():
    """HÀM KHỞI CHẠY CHÍNH - CHẠY BOT"""
    print("🤖 BOT GIAO DỊCH FUTURES BINANCE - KHỞI ĐỘNG...")
    
    # NHẬP THÔNG TIN TỪ NGƯỜI DÙNG
    api_key = input("Nhập Binance API Key: ").strip()
    api_secret = input("Nhập Binance API Secret: ").strip()
    telegram_bot_token = input("Nhập Telegram Bot Token: ").strip()
    telegram_chat_id = input("Nhập Telegram Chat ID: ").strip()
    
    if not api_key or not api_secret:
        print("❌ LỖI: Cần cung cấp API Key và Secret!")
        return
    
    # KHỞI TẠO BOT MANAGER
    try:
        bot_manager = BotManager(
            api_key=api_key,
            api_secret=api_secret,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id
        )
        
        print("✅ Hệ thống đã khởi động thành công!")
        print("📱 Truy cập Telegram để điều khiển bot...")
        print("⏹️  Nhấn Ctrl+C để dừng hệ thống")
        
        # GIỮ CHƯƠNG TRÌNH CHẠY
        while bot_manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⛔ Đang dừng hệ thống...")
        if 'bot_manager' in locals():
            bot_manager.stop_all()
        print("🔴 Hệ thống đã dừng")
    except Exception as e:
        print(f"❌ Lỗi khởi động: {str(e)}")
        logger.error(f"Lỗi khởi động: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
