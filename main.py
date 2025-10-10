# main.py - FILE CHẠY CHÍNH CHO HỆ THỐNG BOT AI THÔNG MINH
import os
import sys
import time
import logging
from trading_bot_lib import BotManager, setup_logging

def load_environment_config():
    """Tải cấu hình từ biến môi trường"""
    config = {
        'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
        'BINANCE_SECRET_KEY': os.getenv('BINANCE_SECRET_KEY'),
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }
    
    # Kiểm tra các biến bắt buộc
    required_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
    missing_vars = [var for var in required_vars if not config[var]]
    
    if missing_vars:
        logging.error(f"❌ Thiếu biến môi trường bắt buộc: {', '.join(missing_vars)}")
        return None
    
    return config

def print_banner():
    """In banner khởi động"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   🤖 BOT TRADING AI THÔNG MINH              ║
    ║                                                              ║
    ║  🎯 Hệ thống giao dịch tự động với AI tìm coin thông minh   ║
    ║  ⚖️  Cân bằng vị thế tự động                                ║
    ║  🔄 Chọn hướng trước - Tìm coin sau                         ║
    ║  📊 Đa chiến lược, đa chỉ báo                              ║
    ║                                                              ║
    ║  Version: 2.0 AI Enhanced                                   ║
    ║  Developed for Futures Trading                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment_config(config):
    """Kiểm tra cấu hình từ biến môi trường"""
    print("🔧 Đang kiểm tra cấu hình từ biến môi trường...")
    
    issues = []
    
    if not config['BINANCE_API_KEY']:
        issues.append("❌ BINANCE_API_KEY chưa được thiết lập")
    
    if not config['BINANCE_SECRET_KEY']:
        issues.append("❌ BINANCE_SECRET_KEY chưa được thiết lập")
    
    if not config['TELEGRAM_BOT_TOKEN']:
        issues.append("⚠️ TELEGRAM_BOT_TOKEN chưa được thiết lập (Telegram notifications sẽ bị tắt)")
    
    if not config['TELEGRAM_CHAT_ID']:
        issues.append("⚠️ TELEGRAM_CHAT_ID chưa được thiết lập (Telegram notifications sẽ bị tắt)")
    
    if issues:
        print("\n".join(issues))
    
    # Chỉ cần Binance API để chạy, Telegram là optional
    if not config['BINANCE_API_KEY'] or not config['BINANCE_SECRET_KEY']:
        print("\n📝 HƯỚNG DẪN CẤU HÌNH TRÊN RAILWAY:")
        print("1. Vào dashboard Railway của bạn")
        print("2. Chọn project → Settings → Variables")
        print("3. Thêm các biến môi trường:")
        print("   - BINANCE_API_KEY")
        print("   - BINANCE_SECRET_KEY") 
        print("   - TELEGRAM_BOT_TOKEN (optional)")
        print("   - TELEGRAM_CHAT_ID (optional)")
        return False
    
    return True

def main():
    """Hàm chính khởi chạy hệ thống"""
    
    # In banner
    print_banner()
    
    # Thiết lập logging
    logger = setup_logging()
    
    # Tải cấu hình từ biến môi trường
    config = load_environment_config()
    if not config:
        print("❌ Không thể tải cấu hình từ biến môi trường")
        return
    
    # Kiểm tra cấu hình
    if not check_environment_config(config):
        return
    
    print("✅ Cấu hình hợp lệ!")
    print(f"🔑 Binance API: ✅")
    if config['TELEGRAM_BOT_TOKEN'] and config['TELEGRAM_CHAT_ID']:
        print(f"🤖 Telegram Bot: ✅")
    else:
        print(f"🤖 Telegram Bot: ⚠️ (Chế độ không Telegram)")
    
    # Khởi tạo BotManager
    try:
        print("\n🚀 Đang khởi động hệ thống Bot AI...")
        
        bot_manager = BotManager(
            api_key=config['BINANCE_API_KEY'],
            api_secret=config['BINANCE_SECRET_KEY'],
            telegram_bot_token=config['TELEGRAM_BOT_TOKEN'],
            telegram_chat_id=config['TELEGRAM_CHAT_ID']
        )
        
        print("✅ Hệ thống Bot AI đã khởi động thành công!")
        print("\n📋 HƯỚNG DẪN SỬ DỤNG:")
        
        if config['TELEGRAM_BOT_TOKEN'] and config['TELEGRAM_CHAT_ID']:
            print("   • Mở Telegram và tìm bot của bạn")
            print("   • Sử dụng menu để thêm bot, xem số dư, quản lý vị thế")
        else:
            print("   • Chạy ở chế độ không Telegram")
            print("   • Sử dụng logs để theo dõi hoạt động")
            
        print("   • Hệ thống sẽ tự động tìm coin và giao dịch")
        print("   • Nhấn Ctrl+C để dừng hệ thống")
        
        # Hiển thị thông tin chiến lược AI
        print("\n🤖 CHIẾN LƯỢC AI ĐÃ KÍCH HOẠT:")
        print("   • AI Market Analyzer - Phân tích thị trường thông minh")
        print("   • Position Balancer - Cân bằng vị thế tự động") 
        print("   • Smart Coin Finder - Tìm coin theo hướng chỉ định")
        
        # Chạy vòng lặp chính
        keep_running = True
        while keep_running and bot_manager.running:
            try:
                time.sleep(1)
                
                # Hiển thị trạng thái mỗi 30 giây
                if int(time.time()) % 30 == 0:
                    active_bots = len(bot_manager.bots)
                    if active_bots > 0:
                        logger.info(f"📊 Hệ thống đang chạy - {active_bots} bot hoạt động")
                        
            except KeyboardInterrupt:
                print("\n\n⏹️  Nhận tín hiệu dừng...")
                keep_running = False
                
    except Exception as e:
        logger.error(f"❌ Lỗi khởi động hệ thống: {str(e)}")
        print(f"❌ Lỗi: {str(e)}")
        return
    
    finally:
        # Dọn dẹp khi dừng
        if 'bot_manager' in locals():
            print("\n🛑 Đang dừng hệ thống...")
            bot_manager.stop_all()
            print("✅ Hệ thống đã dừng an toàn")

def quick_test():
    """Chế độ test nhanh kết nối"""
    print("\n🎯 CHẾ ĐỘ KIỂM TRA NHANH")
    
    config = load_environment_config()
    if not config:
        return
    
    try:
        print("🔗 Đang kết nối Binance...")
        from trading_bot_lib import get_balance
        
        balance = get_balance(config['BINANCE_API_KEY'], config['BINANCE_SECRET_KEY'])
        if balance is not None:
            print(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
        else:
            print("❌ Lỗi kết nối Binance - Kiểm tra API Key/Secret")
            return
            
        if config['TELEGRAM_BOT_TOKEN'] and config['TELEGRAM_CHAT_ID']:
            print("🔗 Đang kiểm tra Telegram...")
            from trading_bot_lib import send_telegram
            try:
                send_telegram(
                    "🤖 Bot AI - Kiểm tra kết nối thành công!\nHệ thống đã sẵn sàng hoạt động.",
                    bot_token=config['TELEGRAM_BOT_TOKEN'],
                    default_chat_id=config['TELEGRAM_CHAT_ID']
                )
                print("✅ Kết nối Telegram thành công!")
            except Exception as e:
                print(f"⚠️ Lỗi Telegram: {e}")
        else:
            print("⚠️ Telegram chưa được cấu hình - Bỏ qua kiểm tra")
        
        print("\n✅ Tất cả kiểm tra đã hoàn tất! Hệ thống sẵn sàng.")
        
    except Exception as e:
        print(f"❌ Lỗi kiểm tra: {str(e)}")

if __name__ == "__main__":
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main()

