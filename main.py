# main.py
from trading_bot_lib import BotManager
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

# In ra để kiểm tra (không in secret key)
print(f"BINANCE_API_KEY: {'***' if BINANCE_API_KEY else 'Không có'}")
print(f"BINANCE_SECRET_KEY: {'***' if BINANCE_SECRET_KEY else 'Không có'}")
print(f"TELEGRAM_BOT_TOKEN: {'***' if TELEGRAM_BOT_TOKEN else 'Không có'}")
print(f"TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else 'Không có'}")

def test_connections():
    """Kiểm tra kết nối API và Telegram"""
    from trading_bot_lib import get_balance, send_telegram
    
    # Test Binance
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        balance = get_balance(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        if balance > 0:
            print(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
        else:
            print("❌ Lỗi kết nối Binance")
            return False
    
    # Test Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram("🤖 Bot khởi động thành công!", 
                         TELEGRAM_CHAT_ID, 
                         bot_token=TELEGRAM_BOT_TOKEN)
            print("✅ Kết nối Telegram thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối Telegram: {e}")
    
    return True

def main():
    # Kiểm tra cấu hình
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        print("❌ Chưa cấu hình API Key và Secret Key!")
        print("💡 Thiết lập biến môi trường:")
        print("   - BINANCE_API_KEY")
        print("   - BINANCE_SECRET_KEY") 
        return
    
    # Kiểm tra kết nối
    if not test_connections():
        return
    
    print("🟢 Đang khởi động hệ thống bot...")
    
    try:
        # Khởi tạo hệ thống
        manager = BotManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY,
            telegram_bot_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID
        )
        
        # Thêm bot mẫu (có thể xóa hoặc thay đổi)
        print("🤖 Đang thêm bot mẫu...")
        
        # Bot RSI/EMA
        manager.add_bot(
            symbol="BTCUSDT",
            lev=10,
            percent=5, 
            tp=50,
            sl=20,
            strategy_type="RSI/EMA Recursive"
        )
        
        # Bot Reverse 24h
        manager.add_bot(
            symbol="ETHUSDT",
            lev=15, 
            percent=3,
            tp=30,
            sl=15,
            strategy_type="Reverse 24h",
            threshold=25
        )
        
        print("🟢 Hệ thống đã sẵn sàng. Đang chạy...")
        
        # Giữ chương trình chạy
        while manager.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Nhận tín hiệu dừng từ người dùng...")
        if 'manager' in locals():
            manager.log("👋 Nhận tín hiệu dừng từ người dùng...")
    except Exception as e:
        print(f"❌ LỖI HỆ THỐNG: {str(e)}")
        if 'manager' in locals():
            manager.log(f"❌ LỖI HỆ THỐNG: {str(e)}")
    finally:
        if 'manager' in locals():
            manager.stop_all()

if __name__ == "__main__":
    main()
