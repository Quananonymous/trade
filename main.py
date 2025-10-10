# main.py - FILE CHẠY CHÍNH CHO HỆ THỐNG BOT AI THÔNG MINH
import os
import sys
import time
import logging
from trading_bot_lib import BotManager, setup_logging

# ========== CẤU HÌNH CỦA BẠN ==========
CONFIG = {
    'BINANCE_API_KEY': "your_binance_api_key_here",
    'BINANCE_API_SECRET': "your_binance_secret_key_here", 
    'TELEGRAM_BOT_TOKEN': "your_telegram_bot_token_here",
    'TELEGRAM_CHAT_ID': "your_telegram_chat_id_here"
}

# ========== CẬP NHẬT VỚI THÔNG TIN THỰC TẾ CỦA BẠN ==========
def update_config_with_real_info():
    """Cập nhật config với thông tin thực tế của bạn"""
    # THAY THẾ CÁC GIÁ TRỊ NÀY BẰNG THÔNG TIN THỰC TẾ CỦA BẠN
    CONFIG['BINANCE_API_KEY'] = "*******"  # Thay bằng API key thực
    CONFIG['BINANCE_API_SECRET'] = "*******"  # Thay bằng Secret key thực
    CONFIG['TELEGRAM_BOT_TOKEN'] = "*******"  # Thay bằng Telegram bot token thực
    CONFIG['TELEGRAM_CHAT_ID'] = "*******"  # Thay bằng Chat ID thực

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

def check_config():
    """Kiểm tra cấu hình"""
    print("🔧 Đang kiểm tra cấu hình...")
    
    issues = []
    
    if CONFIG['BINANCE_API_KEY'] in ["your_binance_api_key_here", "*******"]:
        issues.append("❌ BINANCE_API_KEY chưa được cấu hình")
    
    if CONFIG['BINANCE_API_SECRET'] in ["your_binance_secret_key_here", "*******"]:
        issues.append("❌ BINANCE_API_SECRET chưa được cấu hình")
    
    if CONFIG['TELEGRAM_BOT_TOKEN'] in ["your_telegram_bot_token_here", "*******"]:
        issues.append("❌ TELEGRAM_BOT_TOKEN chưa được cấu hình")
    
    if CONFIG['TELEGRAM_CHAT_ID'] in ["your_telegram_chat_id_here", "*******"]:
        issues.append("❌ TELEGRAM_CHAT_ID chưa được cấu hình")
    
    if issues:
        print("\n".join(issues))
        print("\n📝 HƯỚNG DẪN CẤU HÌNH:")
        print("1. Mở file main.py")
        print("2. Tìm hàm update_config_with_real_info()")
        print("3. Thay thế các giá trị '*******' bằng thông tin thực tế của bạn")
        print("4. Lưu file và chạy lại")
        return False
    
    return True

def main():
    """Hàm chính khởi chạy hệ thống"""
    
    # In banner
    print_banner()
    
    # Thiết lập logging
    logger = setup_logging()
    
    # Cập nhật config với thông tin thực tế
    update_config_with_real_info()
    
    # Kiểm tra cấu hình
    if not check_config():
        return
    
    print("✅ Cấu hình hợp lệ!")
    print(f"🔑 Binance API: ✅")
    print(f"🤖 Telegram Bot: ✅")
    
    # Khởi tạo BotManager
    try:
        print("\n🚀 Đang khởi động hệ thống Bot AI...")
        
        bot_manager = BotManager(
            api_key=CONFIG['BINANCE_API_KEY'],
            api_secret=CONFIG['BINANCE_API_SECRET'],
            telegram_bot_token=CONFIG['TELEGRAM_BOT_TOKEN'],
            telegram_chat_id=CONFIG['TELEGRAM_CHAT_ID']
        )
        
        print("✅ Hệ thống Bot AI đã khởi động thành công!")
        print("\n📋 HƯỚNG DẪN SỬ DỤNG:")
        print("   • Mở Telegram và tìm bot của bạn")
        print("   • Sử dụng menu để thêm bot, xem số dư, quản lý vị thế")
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
    """Chế độ test nhanh"""
    print("\n🎯 CHẾ ĐỘ KIỂM TRA NHANH")
    
    # Cập nhật config với thông tin thực tế
    update_config_with_real_info()
    
    if not check_config():
        return
    
    try:
        print("🔗 Đang kết nối Binance...")
        from trading_bot_lib import get_balance
        
        balance = get_balance(CONFIG['BINANCE_API_KEY'], CONFIG['BINANCE_API_SECRET'])
        if balance is not None:
            print(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")
        else:
            print("❌ Lỗi kết nối Binance")
            return
            
        if CONFIG['TELEGRAM_BOT_TOKEN']:
            print("🔗 Đang kiểm tra Telegram...")
            from trading_bot_lib import send_telegram
            try:
                send_telegram(
                    "🤖 Bot AI đã khởi động thành công!\nHệ thống đã sẵn sàng hoạt động.",
                    bot_token=CONFIG['TELEGRAM_BOT_TOKEN'],
                    default_chat_id=CONFIG['TELEGRAM_CHAT_ID']
                )
                print("✅ Kết nối Telegram thành công!")
            except Exception as e:
                print(f"⚠️ Lỗi Telegram: {e}")
        
        print("\n✅ Tất cả kiểm tra đã hoàn tất! Hệ thống sẵn sàng.")
        
    except Exception as e:
        print(f"❌ Lỗi kiểm tra: {str(e)}")

if __name__ == "__main__":
    # Kiểm tra tham số dòng lệnh
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main()
