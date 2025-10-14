# main.py - FILE CHẠY CHÍNH CHO HỆ THỐNG TRADING BOT
import os
import json
import logging
from trading_bot_lib import start_trading_system, BotManager

# ========== CẤU HÌNH LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ========== HÀM ĐỌC CẤU HÌNH TỪ FILE ==========
def load_config():
    """Đọc cấu hình từ file config.json hoặc biến môi trường"""
    config = {}
    
    # Thử đọc từ file config.json
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info("✅ Đã đọc cấu hình từ config.json")
    except Exception as e:
        logger.warning(f"⚠️ Không thể đọc config.json: {e}")
    
    # Ưu tiên biến môi trường (nếu có)
    env_config = {
        'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
        'BINANCE_SECRET_KEY': os.getenv('BINANCE_SECRET_KEY'),
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }
    
    # Cập nhật config từ biến môi trường (nếu có giá trị)
    for key, value in env_config.items():
        if value:
            clean_key = key.replace('BINANCE_', '').replace('TELEGRAM_', '').lower()
            config[clean_key] = value
    
    return config

# ========== HÀM TẠO FILE CẤU HÌNH MẪU ==========
def create_sample_config():
    """Tạo file config.json mẫu nếu chưa tồn tại"""
    sample_config = {
        "api_key": "your_binance_api_key_here",
        "api_secret": "your_BINANCE_SECRET_KEY_here", 
        "telegram_bot_token": "your_telegram_bot_token_here",
        "telegram_chat_id": "your_telegram_chat_id_here"
    }
    
    if not os.path.exists('config.json'):
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=4, ensure_ascii=False)
        logger.info("📁 Đã tạo file config.json mẫu. Vui lòng điền thông tin của bạn!")
        return False
    return True

# ========== HÀM KIỂM TRA CẤU HÌNH ==========
def validate_config(config):
    """Kiểm tra cấu hình có đầy đủ không"""
    required_fields = ['api_key', 'api_secret']
    
    missing_fields = []
    for field in required_fields:
        if not config.get(field) or config[field].startswith('your_'):
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"❌ Thiếu thông tin cấu hình: {', '.join(missing_fields)}")
        logger.info("💡 Vui lòng cập nhật file config.json hoặc thiết lập biến môi trường")
        return False
    
    # Cảnh báo nếu thiếu Telegram (không bắt buộc)
    if not config.get('telegram_bot_token') or not config.get('telegram_chat_id'):
        logger.warning("⚠️ Chưa cấu hình Telegram - Bot sẽ chạy không có thông báo Telegram")
    
    return True

# ========== HÀM HIỂN THỊ BANNER ==========
def show_banner():
    """Hiển thị banner hệ thống"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   🤖 TRADING BOT SYSTEM 🤖                   ║
    ║                                                              ║
    ║  🎯 Hệ thống giao dịch Futures đa luồng thông minh          ║
    ║  🔄 Tự động tìm coin & Rotation hệ thống 5 bước             ║
    ║  📊 Chỉ báo tích hợp: EMA + RSI + Volume + Support/Resistance║
    ║  📱 Điều khiển qua Telegram Menu đầy đủ                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

# ========== HÀM CHẠY CHÍNH ==========
def main():
    """Hàm chính khởi chạy hệ thống"""
    
    # Hiển thị banner
    show_banner()
    
    # Kiểm tra và tạo config mẫu
    if not create_sample_config():
        print("\n📝 Vui lòng điền thông tin vào file 'config.json' và chạy lại!")
        return
    
    # Đọc cấu hình
    config = load_config()
    
    # Kiểm tra cấu hình
    if not validate_config(config):
        print("\n❌ Cấu hình không hợp lệ. Vui lòng kiểm tra lại!")
        return
    
    # Hiển thị thông tin cấu hình (ẩn key bí mật)
    display_key = config['api_key'][:8] + '...' + config['api_key'][-4:] if len(config['api_key']) > 12 else config['api_key']
    display_secret = config['api_secret'][:8] + '...' + config['api_secret'][-4:] if len(config['api_secret']) > 12 else config['api_secret']
    
    print(f"\n📋 THÔNG TIN CẤU HÌNH:")
    print(f"   🔑 Binance API Key: {display_key}")
    print(f"   🔒 Binance Secret: {display_secret}")
    
    if config.get('telegram_bot_token'):
        display_token = config['telegram_bot_token'][:8] + '...' + config['telegram_bot_token'][-4:] if len(config['telegram_bot_token']) > 12 else config['telegram_bot_token']
        print(f"   🤖 Telegram Bot: {display_token}")
        print(f"   💬 Chat ID: {config.get('telegram_chat_id', 'Chưa cấu hình')}")
    else:
        print(f"   🤖 Telegram: Chưa cấu hình")
    
    # Xác nhận khởi động
    print(f"\n🚀 BẮT ĐẦU KHỞI ĐỘNG HỆ THỐNG...")
    
    try:
        # Khởi động hệ thống
        bot_manager = start_trading_system(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            telegram_bot_token=config.get('telegram_bot_token'),
            telegram_chat_id=config.get('telegram_chat_id')
        )
        
        # Hiển thị hướng dẫn sử dụng
        print(f"\n✅ HỆ THỐNG ĐÃ KHỞI ĐỘNG THÀNH CÔNG!")
        
        if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
            print(f"\n📱 HƯỚNG DẪN SỬ DỤNG TELEGRAM:")
            print(f"   1. Mở Telegram và tìm bot của bạn")
            print(f"   2. Gửi lệnh bất kỳ để hiện menu chính")
            print(f"   3. Chọn '➕ Thêm Bot' để tạo bot giao dịch")
            print(f"   4. Theo dõi '📊 Thống kê' để xem hiệu suất")
        else:
            print(f"\n⚠️  LƯU Ý: Chưa cấu hình Telegram")
            print(f"   Hệ thống vẫn chạy nhưng không có thông báo qua Telegram")
        
        print(f"\n🛑 Để dừng hệ thống, nhấn Ctrl+C")
        
        # Giữ chương trình chạy
        while True:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Nhận tín hiệu dừng từ người dùng...")
        if 'bot_manager' in locals():
            bot_manager.stop_all()
        print(f"🔴 Hệ thống đã dừng an toàn!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động hệ thống: {e}")
        print(f"\n❌ Có lỗi xảy ra: {e}")
        print(f"💡 Vui lòng kiểm tra lại cấu hình và thử lại!")

# ========== HÀM CHẠY TEST ==========
def test_system():
    """Chế độ test - kiểm tra kết nối mà không chạy bot thật"""
    print(f"\n🔍 CHẾ ĐỘ KIỂM TRA KẾT NỐI...")
    
    config = load_config()
    
    if not validate_config(config):
        print("❌ Không thể test do cấu hình không hợp lệ")
        return
    
    try:
        from trading_bot_lib import get_balance, get_all_usdt_pairs
        
        # Test kết nối Binance
        print(f"🔄 Đang kiểm tra kết nối Binance...")
        balance = get_balance(config['api_key'], config['api_secret'])
        
        if balance is not None:
            print(f"✅ Kết nối Binance: THÀNH CÔNG")
            print(f"💰 Số dư khả dụng: {balance:.2f} USDT")
        else:
            print(f"❌ Kết nối Binance: THẤT BẠI")
            return
        
        # Test lấy danh sách coin
        print(f"🔄 Đang lấy danh sách coin...")
        symbols = get_all_usdt_pairs(limit=10)
        
        if symbols:
            print(f"✅ Lấy danh sách coin: THÀNH CÔNG")
            print(f"📊 Số coin: {len(symbols)}")
            print(f"🔗 Ví dụ: {', '.join(symbols[:3])}...")
        else:
            print(f"❌ Lấy danh sách coin: THẤT BẠI")
        
        # Test Telegram (nếu có)
        if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
            print(f"🔄 Đang kiểm tra Telegram...")
            from trading_bot_lib import send_telegram
            try:
                send_telegram(
                    "🤖 **TEST KẾT NỐI**\nHệ thống Trading Bot đã sẵn sàng!",
                    chat_id=config['telegram_chat_id'],
                    bot_token=config['telegram_bot_token']
                )
                print(f"✅ Kết nối Telegram: THÀNH CÔNG")
            except Exception as e:
                print(f"❌ Kết nối Telegram: THẤT BẠI - {e}")
        
        print(f"\n🎉 KIỂM TRA HOÀN TẤT! Hệ thống đã sẵn sàng.")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình test: {e}")

# ========== HÀM HIỂN THỊ TRỢ GIÚP ==========
def show_help():
    """Hiển thị hướng dẫn sử dụng"""
    help_text = """
🤖 HƯỚNG DẪN SỬ DỤNG TRADING BOT SYSTEM

CÁCH CHẠY:
    python main.py                    - Chạy hệ thống chính
    python main.py test              - Kiểm tra kết nối
    python main.py help              - Hiển thị trợ giúp

CẤU HÌNH:
    1. Sửa file config.json hoặc
    2. Thiết lập biến môi trường:
       - BINANCE_API_KEY=your_key
       - BINANCE_SECRET_KEY=your_secret  
       - TELEGRAM_BOT_TOKEN=your_token
       - TELEGRAM_CHAT_ID=your_chat_id

TÍNH NĂNG CHÍNH:
    ✅ Tự động tìm coin theo hệ thống 5 bước
    ✅ Rotation coin thông minh
    ✅ Đa bot độc lập
    ✅ Điều khiển qua Telegram Menu
    ✅ TP/SL linh hoạt
    ✅ Quản lý rủi ro thông minh

TELEGRAM MENU:
    ➕ Thêm Bot    - Tạo bot giao dịch mới
    📊 Danh sách Bot - Xem bot đang chạy  
    📊 Thống kê    - Thống kê hiệu suất
    ⛔ Dừng Bot    - Dừng bot
    💰 Số dư      - Kiểm tra số dư
    📈 Vị thế     - Xem vị thế đang mở
    🎯 Chiến lược - Xem thông tin hệ thống
    ⚙️ Cấu hình   - Xem cấu hình hệ thống
    """
    print(help_text)

# ========== ĐIỂM VÀO CHƯƠNG TRÌNH ==========
if __name__ == "__main__":
    import sys
    
    # Xử lý tham số dòng lệnh
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_system()
        elif command in ['help', '--help', '-h']:
            show_help()
        else:
            print(f"❌ Lệnh không hợp lệ: {command}")
            print(f"💡 Sử dụng: python main.py [test|help]")
    else:
        # Chạy hệ thống chính
        main()
