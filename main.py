# main.py - HỆ THỐNG BOT GIAO DỊCH FUTURES TỰ ĐỘNG
import os
import sys
import time
import logging
from trading_bot_lib import create_bot_system, BotManager

# ========== CẤU HÌNH LOGGING ==========
def setup_logging():
    """Thiết lập hệ thống logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_bot.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========== HÀM KIỂM TRA MÔI TRƯỜNG ==========
def check_environment():
    """Kiểm tra môi trường và thư viện"""
    try:
        import numpy as np
        import requests
        import websocket
        import urllib
        import hmac
        import hashlib
        logger.info("✅ Tất cả thư viện đã sẵn sàng")
        return True
    except ImportError as e:
        logger.error(f"❌ Thiếu thư viện: {e}")
        return False

# ========== HÀM ĐỌC CẤU HÌNH TỪ FILE ==========
def load_config_from_file(config_file='config.json'):
    """Đọc cấu hình từ file JSON"""
    try:
        import json
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Đã đọc cấu hình từ {config_file}")
            return config
        else:
            logger.warning(f"⚠️ File cấu hình {config_file} không tồn tại")
            return None
    except Exception as e:
        logger.error(f"❌ Lỗi đọc file cấu hình: {e}")
        return None

# ========== HÀM ĐỌC CẤU HÌNH TỪ BIẾN MÔI TRƯỜNG ==========
def load_config_from_env():
    """Đọc cấu hình từ biến môi trường"""
    config = {
        'api_key': os.getenv('BINANCE_API_KEY'),
        'api_secret': os.getenv('BINANCE_SECRET_KEY'),
        'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
    }
    
    # Kiểm tra các thông tin bắt buộc
    if not config['api_key'] or not config['api_secret']:
        logger.error("❌ Thiếu API Key hoặc API Secret")
        return None
        
    logger.info("✅ Đã đọc cấu hình từ biến môi trường")
    return config

# ========== HÀM NHẬP CẤU HÌNH TỪ NGƯỜI DÙNG ==========
def get_config_from_user():
    """Nhập cấu hình từ người dùng"""
    print("\n" + "="*50)
    print("🤖 THIẾT LẬP HỆ THỐNG BOT GIAO DỊCH")
    print("="*50)
    
    config = {}
    
    # API Binance
    print("\n🔑 THIẾT LẬP BINANCE API:")
    config['api_key'] = input("Nhập API Key: ").strip()
    config['api_secret'] = input("Nhập API Secret: ").strip()
    
    if not config['api_key'] or not config['api_secret']:
        print("❌ API Key và API Secret là bắt buộc!")
        return None
    
    # Telegram (tùy chọn)
    print("\n📱 THIẾT LẬP TELEGRAM (tùy chọn):")
    config['telegram_bot_token'] = input("Nhập Telegram Bot Token (Enter để bỏ qua): ").strip()
    config['telegram_chat_id'] = input("Nhập Telegram Chat ID (Enter để bỏ qua): ").strip()
    
    if not config['telegram_bot_token'] or not config['telegram_chat_id']:
        config['telegram_bot_token'] = None
        config['telegram_chat_id'] = None
        print("⚠️ Chế độ không Telegram - Thông báo sẽ chỉ hiển thị trên console")
    
    return config

# ========== HÀM LƯU CẤU HÌNH ==========
def save_config(config, config_file='config.json'):
    """Lưu cấu hình vào file"""
    try:
        import json
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Đã lưu cấu hình vào {config_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Lỗi lưu cấu hình: {e}")
        return False

# ========== HÀM HIỂN THỊ MENU CHÍNH ==========
def show_main_menu():
    """Hiển thị menu chính"""
    print("\n" + "="*60)
    print("🎯 HỆ THỐNG BOT GIAO DỊCH FUTURES TỰ ĐỘNG")
    print("="*60)
    print("1. 🚀 Khởi động hệ thống bot")
    print("2. ⚙️  Thiết lập cấu hình")
    print("3. 📊 Xem trạng thái hệ thống")
    print("4. 📝 Hướng dẫn sử dụng")
    print("5. 🚪 Thoát")
    print("="*60)

# ========== HÀM HIỂN THỊ HƯỚNG DẪN ==========
def show_instructions():
    """Hiển thị hướng dẫn sử dụng"""
    print("\n" + "🔰 HƯỚNG DẪN SỬ DỤNG")
    print("="*50)
    print("🎯 HỆ THỐNG 5 BƯỚC THÔNG MINH:")
    print("1. Kiểm tra vị thế Binance")
    print("2. Xác định hướng giao dịch") 
    print("3. Tìm coin phù hợp")
    print("4. Kiểm soát lệnh TP/SL")
    print("5. Rotation coin tự động")
    print("\n📱 SỬ DỤNG TELEGRAM:")
    print("- Thêm Bot: Chọn '➕ Thêm Bot'")
    print("- Xem trạng thái: '📊 Danh sách Bot'")
    print("- Thống kê: '📊 Thống kê'")
    print("- Dừng bot: '⛔ Dừng Bot'")
    print("\n⚡ TÍNH NĂNG CHÍNH:")
    print("- Đa bot độc lập")
    "- Tự động tìm coin"
    "- Chỉ báo xu hướng tích hợp"
    "- Quản lý rủi ro thông minh"
    print("- Rotation coin tự động")
    print("="*50)
    input("Nhấn Enter để tiếp tục...")

# ========== HÀM KIỂM TRA KẾT NỐI API ==========
def test_api_connection(api_key, api_secret):
    """Kiểm tra kết nối API Binance"""
    try:
        from trading_bot_lib_final import get_balance
        balance = get_balance(api_key, api_secret)
        if balance is not None:
            print(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
            return True
        else:
            print("❌ Lỗi kết nối Binance API")
            return False
    except Exception as e:
        print(f"❌ Lỗi kiểm tra API: {e}")
        return False

# ========== HÀM CHẠY HỆ THỐNG BOT ==========
def run_bot_system(config):
    """Chạy hệ thống bot chính"""
    try:
        print("\n🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG BOT...")
        
        # Kiểm tra kết nối API
        if not test_api_connection(config['api_key'], config['api_secret']):
            return None
        
        # Khởi tạo hệ thống bot
        bot_manager = create_bot_system(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            telegram_bot_token=config.get('telegram_bot_token'),
            telegram_chat_id=config.get('telegram_chat_id')
        )
        
        print("✅ HỆ THỐNG BOT ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
        print("📱 Sử dụng Telegram để quản lý bot (nếu đã cấu hình)")
        print("💡 Gợi ý: Gửi '/start' hoặc nhấn '➕ Thêm Bot' trên Telegram")
        
        return bot_manager
        
    except Exception as e:
        print(f"❌ Lỗi khởi động hệ thống: {e}")
        return None

# ========== HÀM QUẢN LÝ VÒNG LẶP CHÍNH ==========
def main():
    """Hàm chính của chương trình"""
    
    # Kiểm tra môi trường
    if not check_environment():
        print("❌ Vui lòng cài đặt các thư viện cần thiết!")
        print("👉 Chạy: pip install numpy requests websocket-client")
        return
    
    print("✅ Môi trường đã sẵn sàng!")
    
    bot_manager = None
    current_config = None
    
    while True:
        show_main_menu()
        choice = input("Lựa chọn của bạn (1-5): ").strip()
        
        if choice == '1':
            # Khởi động hệ thống bot
            if current_config is None:
                print("❌ Chưa có cấu hình! Vui lòng thiết lập cấu hình trước.")
                continue
                
            if bot_manager is not None:
                print("⚠️ Hệ thống bot đang chạy. Vui lòng dừng trước khi khởi động lại.")
                continue
                
            bot_manager = run_bot_system(current_config)
            if bot_manager:
                print("\n🎯 HỆ THỐNG ĐANG CHẠY...")
                print("👉 Sử dụng Telegram để thêm bot và quản lý")
                print("👉 Nhấn Ctrl+C để dừng hệ thống")
                
                try:
                    # Giữ chương trình chạy
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\n🛑 ĐANG DỪNG HỆ THỐNG...")
                    if bot_manager:
                        bot_manager.stop_all()
                    print("✅ Đã dừng hệ thống an toàn")
                    bot_manager = None
                    input("Nhấn Enter để tiếp tục...")
                    
        elif choice == '2':
            # Thiết lập cấu hình
            print("\n⚙️  THIẾT LẬP CẤU HÌNH")
            print("1. Nhập thủ công")
            print("2. Đọc từ file config.json")
            print("3. Đọc từ biến môi trường")
            
            config_choice = input("Lựa chọn (1-3): ").strip()
            
            if config_choice == '1':
                current_config = get_config_from_user()
            elif config_choice == '2':
                current_config = load_config_from_file()
            elif config_choice == '3':
                current_config = load_config_from_env()
            else:
                print("❌ Lựa chọn không hợp lệ!")
                continue
                
            if current_config:
                # Hỏi người dùng có muốn lưu cấu hình không
                save_choice = input("💾 Lưu cấu hình vào file config.json? (y/n): ").strip().lower()
                if save_choice == 'y':
                    save_config(current_config)
                
                # Kiểm tra kết nối API
                test_api_connection(current_config['api_key'], current_config['api_secret'])
                
        elif choice == '3':
            # Xem trạng thái hệ thống
            print("\n📊 TRẠNG THÁI HỆ THỐNG")
            print("="*30)
            
            if current_config:
                print("✅ Đã có cấu hình")
                if current_config.get('telegram_bot_token'):
                    print("✅ Đã cấu hình Telegram")
                else:
                    print("⚠️ Chưa cấu hình Telegram")
            else:
                print("❌ Chưa có cấu hình")
                
            if bot_manager:
                print("✅ Hệ thống bot đang chạy")
                print(f"🤖 Số lượng bot: {len(bot_manager.bots)}")
                
                # Thống kê bot
                active_bots = len([b for b in bot_manager.bots.values() if b.position_open])
                searching_bots = len([b for b in bot_manager.bots.values() if b.status == "searching"])
                waiting_bots = len([b for b in bot_manager.bots.values() if b.status == "waiting"])
                
                print(f"   🟢 Đang trade: {active_bots}")
                print(f"   🔍 Đang tìm coin: {searching_bots}")
                print(f"   🟡 Chờ tín hiệu: {waiting_bots}")
            else:
                print("❌ Hệ thống bot chưa chạy")
                
            input("\nNhấn Enter để tiếp tục...")
            
        elif choice == '4':
            # Hiển thị hướng dẫn
            show_instructions()
            
        elif choice == '5':
            # Thoát chương trình
            print("\n👋 ĐANG THOÁT CHƯƠNG TRÌNH...")
            if bot_manager:
                print("🛑 Đang dừng tất cả bot...")
                bot_manager.stop_all()
            print("✅ Đã thoát chương trình an toàn!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 1-5")

# ========== HÀM CHẠY NHANH VỚI CẤU HÌNH MẶC ĐỊNH ==========
def quick_start():
    """Khởi động nhanh với cấu hình từ biến môi trường"""
    print("🚀 KHỞI ĐỘNG NHANH HỆ THỐNG BOT...")
    
    # Đọc cấu hình từ biến môi trường
    config = load_config_from_env()
    if not config:
        print("❌ Không tìm thấy cấu hình trong biến môi trường")
        print("👉 Vui lòng thiết lập các biến môi trường sau:")
        print("   - BINANCE_API_KEY")
        print("   - BINANCE_SECRET_KEY") 
        print("   - TELEGRAM_BOT_TOKEN (tùy chọn)")
        print("   - TELEGRAM_CHAT_ID (tùy chọn)")
        return
    
    # Kiểm tra kết nối API
    if not test_api_connection(config['api_key'], config['api_secret']):
        return
    
    # Khởi động hệ thống
    bot_manager = create_bot_system(
        api_key=config['api_key'],
        api_secret=config['api_secret'],
        telegram_bot_token=config.get('telegram_bot_token'),
        telegram_chat_id=config.get('telegram_chat_id')
    )
    
    if bot_manager:
        print("✅ HỆ THỐNG BOT ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
        print("📱 Sử dụng Telegram để quản lý bot")
        print("💡 Gợi ý: Gửi '/start' hoặc nhấn '➕ Thêm Bot'")
        print("\n🎯 HỆ THỐNG ĐANG CHẠY...")
        print("👉 Nhấn Ctrl+C để dừng hệ thống")
        
        try:
            # Giữ chương trình chạy
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 ĐANG DỪNG HỆ THỐNG...")
            bot_manager.stop_all()
            print("✅ Đã dừng hệ thống an toàn")

# ========== ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH ==========
if __name__ == "__main__":
    print("🤖 HỆ THỐNG BOT GIAO DỊCH FUTURES TỰ ĐỘNG")
    print("⚡ Phiên bản: 2.0 - Hệ thống 5 bước thông minh")
    
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Chế độ khởi động nhanh
        quick_start()
    else:
        # Chế độ menu tương tác
        main()

