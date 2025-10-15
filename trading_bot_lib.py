# trading_bot_volume_candle_complete_advanced.py
# HOÀN CHỈNH VỚI HỆ THỐNG XÁC SUẤT ĐA ĐIỂM VÀ RANDOM DIRECTION - PHIÊN BẢN CẢI TIẾN

import json
import hmac
import hashlib
import time
import threading
import urllib.request
import urllib.parse
import numpy as np
import websocket
import logging
import requests
import os
import math
import traceback
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# ========== CẤU HÌNH LOGGING ==========
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot_errors.log')
        ]
    )
    return logging.getLogger()

logger = setup_logging()

# ========== HÀM TELEGRAM ==========
def send_telegram(message, chat_id=None, reply_markup=None, bot_token=None, default_chat_id=None):
    if not bot_token:
        logger.warning("Telegram Bot Token chưa được thiết lập")
        return
    
    chat_id = chat_id or default_chat_id
    if not chat_id:
        logger.warning("Telegram Chat ID chưa được thiết lập")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            logger.error(f"Lỗi Telegram ({response.status_code}): {response.text}")
    except Exception as e:
        logger.error(f"Lỗi kết nối Telegram: {str(e)}")

# ========== MENU TELEGRAM HOÀN CHỈNH ==========
def create_cancel_keyboard():
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "📊 Trend System"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_exit_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "🎯 Chỉ TP/SL cố định"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_bot_mode_keyboard():
    return {
        "keyboard": [
            [{"text": "🤖 Bot Tĩnh - Coin cụ thể"}, {"text": "🔄 Bot Động - Tự tìm coin"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_symbols_keyboard(strategy=None):
    try:
        symbols = get_all_usdt_pairs(limit=12)
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"]
    except:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"]
    
    keyboard = []
    row = []
    for symbol in symbols:
        row.append({"text": symbol})
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_main_menu():
    return {
        "keyboard": [
            [{"text": "📊 Danh sách Bot"}, {"text": "📊 Thống kê"}],
            [{"text": "➕ Thêm Bot"}, {"text": "⛔ Dừng Bot"}],
            [{"text": "💰 Số dư"}, {"text": "📈 Vị thế"}],
            [{"text": "⚙️ Cấu hình"}, {"text": "🎯 Chiến lược"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def create_leverage_keyboard(strategy=None):
    leverages = ["3", "5", "10", "15", "20", "25", "50", "75", "100"]
    
    keyboard = []
    row = []
    for lev in leverages:
        row.append({"text": f"{lev}x"})
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([{"text": "❌ Hủy bỏ"}])
    
    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_percent_keyboard():
    return {
        "keyboard": [
            [{"text": "1"}, {"text": "3"}, {"text": "5"}, {"text": "10"}],
            [{"text": "15"}, {"text": "20"}, {"text": "25"}, {"text": "50"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_tp_keyboard():
    return {
        "keyboard": [
            [{"text": "50"}, {"text": "100"}, {"text": "200"}],
            [{"text": "300"}, {"text": "500"}, {"text": "1000"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_sl_keyboard():
    return {
        "keyboard": [
            [{"text": "0"}, {"text": "50"}, {"text": "100"}],
            [{"text": "150"}, {"text": "200"}, {"text": "500"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_bot_count_keyboard():
    return {
        "keyboard": [
            [{"text": "1"}, {"text": "2"}, {"text": "3"}],
            [{"text": "5"}, {"text": "10"}],
            [{"text": "❌ Hủy bỏ"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

# ========== HÀM KIỂM TRA ĐÒN BẨY TỐI ĐA ==========
def get_max_leverage(symbol, api_key, api_secret):
    """Lấy đòn bẩy tối đa cho một symbol"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return 100
        
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LEVERAGE':
                        if 'maxLeverage' in f:
                            return int(f['maxLeverage'])
                break
        return 100
    except Exception as e:
        logger.error(f"Lỗi lấy đòn bẩy tối đa {symbol}: {str(e)}")
        return 100

# ========== HỆ THỐNG THỐNG KÊ XÁC SUẤT ĐA ĐIỂM ==========
class ProbabilityAnalyzer:
    """PHÂN TÍCH XÁC SUẤT THẮNG TẠI NHIỀU ĐIỂM TRÊN CÁC CHỈ BÁO"""
    
    def __init__(self, lookback=200, evaluation_period=20):
        self.lookback = lookback
        self.evaluation_period = evaluation_period
        self.history_data = {}
        
        # CẤU TRÚC DỮ LIỆU CHI TIẾT CHO TỪNG CHỈ BÁO - NHIỀU ĐIỂM
        self.probability_stats = {
            'rsi_multiple_points': {
                'rsi_levels': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
                'rsi_zones': ['oversold', 'neutral_low', 'neutral', 'neutral_high', 'overbought'],
                'correct_predictions': {level: 0 for level in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]},
                'total_predictions': {level: 0 for level in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]},
                'zone_correct': {zone: 0 for zone in ['oversold', 'neutral_low', 'neutral', 'neutral_high', 'overbought']},
                'zone_total': {zone: 0 for zone in ['oversold', 'neutral_low', 'neutral', 'neutral_high', 'overbought']},
                'expectations': {level: 0.0 for level in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]},
                'variances': {level: 0.0 for level in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]}
            },
            'ema_multiple_conditions': {
                'conditions': [
                    'price_above_all_ema', 'price_below_all_ema',
                    'ema_fast_above_slow', 'ema_fast_below_slow', 
                    'ema_slow_above_trend', 'ema_slow_below_trend',
                    'golden_cross_recent', 'death_cross_recent',
                    'ema_alignment_bullish', 'ema_alignment_bearish'
                ],
                'correct_predictions': {cond: 0 for cond in [
                    'price_above_all_ema', 'price_below_all_ema',
                    'ema_fast_above_slow', 'ema_fast_below_slow', 
                    'ema_slow_above_trend', 'ema_slow_below_trend',
                    'golden_cross_recent', 'death_cross_recent',
                    'ema_alignment_bullish', 'ema_alignment_bearish'
                ]},
                'total_predictions': {cond: 0 for cond in [
                    'price_above_all_ema', 'price_below_all_ema',
                    'ema_fast_above_slow', 'ema_fast_below_slow', 
                    'ema_slow_above_trend', 'ema_slow_below_trend',
                    'golden_cross_recent', 'death_cross_recent',
                    'ema_alignment_bullish', 'ema_alignment_bearish'
                ]},
                'expectations': {cond: 0.0 for cond in [
                    'price_above_all_ema', 'price_below_all_ema',
                    'ema_fast_above_slow', 'ema_fast_below_slow', 
                    'ema_slow_above_trend', 'ema_slow_below_trend',
                    'golden_cross_recent', 'death_cross_recent',
                    'ema_alignment_bullish', 'ema_alignment_bearish'
                ]}
            },
            'volume_multiple_levels': {
                'volume_ratios': [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
                'volume_zones': ['very_low', 'low', 'normal', 'high', 'very_high', 'extreme'],
                'correct_predictions': {ratio: 0 for ratio in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]},
                'total_predictions': {ratio: 0 for ratio in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]},
                'zone_correct': {zone: 0 for zone in ['very_low', 'low', 'normal', 'high', 'very_high', 'extreme']},
                'zone_total': {zone: 0 for zone in ['very_low', 'low', 'normal', 'high', 'very_high', 'extreme']},
                'expectations': {ratio: 0.0 for ratio in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]}
            },
            'combined_signals': {
                'signal_types': ['bullish', 'bearish', 'neutral'],
                'correct_predictions': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'total_predictions': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'expectations': {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0},
                'variances': {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}
            },
            'signal_strength': {
                'strength_levels': ['weak', 'medium', 'strong'],
                'correct_predictions': {'weak': 0, 'medium': 0, 'strong': 0},
                'total_predictions': {'weak': 0, 'medium': 0, 'strong': 0},
                'expectations': {'weak': 0.0, 'medium': 0.0, 'strong': 0.0},
                'variances': {'weak': 0.0, 'medium': 0.0, 'strong': 0.0}
            }
        }
        self.last_update_time = 0
        self.update_interval = 1800  # 30 phút
        self.min_data_points = 5
    
    def get_rsi_zone(self, rsi_value):
        """XÁC ĐỊNH VÙNG RSI"""
        if rsi_value < 30:
            return 'oversold'
        elif rsi_value < 45:
            return 'neutral_low'
        elif rsi_value < 55:
            return 'neutral'
        elif rsi_value < 70:
            return 'neutral_high'
        else:
            return 'overbought'
    
    def get_volume_zone(self, volume_ratio):
        """XÁC ĐỊNH VÙNG VOLUME"""
        if volume_ratio < 0.7:
            return 'very_low'
        elif volume_ratio < 0.9:
            return 'low'
        elif volume_ratio < 1.1:
            return 'normal'
        elif volume_ratio < 1.5:
            return 'high'
        elif volume_ratio < 2.0:
            return 'very_high'
        else:
            return 'extreme'
    
    def get_closest_rsi_level(self, rsi_value):
        """TÌM ĐIỂM RSI GẦN NHẤT TRONG DANH SÁCH"""
        levels = self.probability_stats['rsi_multiple_points']['rsi_levels']
        return min(levels, key=lambda x: abs(x - rsi_value))
    
    def get_closest_volume_level(self, volume_ratio):
        """TÌM ĐIỂM VOLUME GẦN NHẤT TRONG DANH SÁCH"""
        ratios = self.probability_stats['volume_multiple_levels']['volume_ratios']
        return min(ratios, key=lambda x: abs(x - volume_ratio))
    
    def analyze_combined_signal(self, signals_data):
        """PHÂN TÍCH VÀ TÍCH HỢP TẤT CẢ TÍN HIỆU CHỈ BÁO - PHIÊN BẢN CẢI TIẾN"""
        try:
            bullish_score = 0
            bearish_score = 0
            total_strength = 0
            signal_count = 0
            
            # 1. PHÂN TÍCH TÍN HIỆU EMA - TRỌNG SỐ CAO
            ema_signal = signals_data.get('ema_signal', 'NEUTRAL')
            ema_strength = signals_data.get('ema_strength', 0)
            
            if ema_signal == "BUY":
                bullish_score += 4.0 * ema_strength
                total_strength += ema_strength
                signal_count += 1
            elif ema_signal == "SELL":
                bearish_score += 4.0 * ema_strength
                total_strength += ema_strength
                signal_count += 1
            
            # 2. PHÂN TÍCH TÍN HIỆU RSI - TRỌNG SỐ CAO
            rsi_signal = signals_data.get('rsi_signal', 'NEUTRAL')
            rsi_strength = signals_data.get('rsi_strength', 0)
            
            if rsi_signal == "BUY":
                bullish_score += 3.5 * rsi_strength
                total_strength += rsi_strength
                signal_count += 1
            elif rsi_signal == "SELL":
                bearish_score += 3.5 * rsi_strength
                total_strength += rsi_strength
                signal_count += 1
            
            # 3. PHÂN TÍCH VOLUME - TRỌNG SỐ TRUNG BÌNH
            volume_ratio = signals_data.get('volume_ratio', 1.0)
            price_vs_ema = signals_data.get('price_vs_ema', 0)
            
            if volume_ratio > 1.5:
                volume_strength = min((volume_ratio - 1.0) / 2.0, 1.0)
                if price_vs_ema > 0:
                    bullish_score += 2.0 * volume_strength
                    total_strength += volume_strength
                else:
                    bearish_score += 2.0 * volume_strength
                    total_strength += volume_strength
                signal_count += 1
            
            # 4. PHÂN TÍCH SUPPORT/RESISTANCE - TRỌNG SỐ TRUNG BÌNH
            sr_signal = signals_data.get('sr_signal', 'NEUTRAL')
            sr_strength = signals_data.get('sr_strength', 0)
            
            if sr_signal == "BUY":
                bullish_score += 2.5 * sr_strength
                total_strength += sr_strength
                signal_count += 1
            elif sr_signal == "SELL":
                bearish_score += 2.5 * sr_strength
                total_strength += sr_strength
                signal_count += 1
            
            # 5. PHÂN TÍCH MARKET STRUCTURE - TRỌNG SỐ THẤP
            structure_signal = signals_data.get('structure_signal', 'NEUTRAL')
            if structure_signal == "BUY":
                bullish_score += 1.5
                total_strength += 0.5
                signal_count += 1
            elif structure_signal == "SELL":
                bearish_score += 1.5
                total_strength += 0.5
                signal_count += 1
            
            # TÍNH TOÁN TÍN HIỆU TỔNG HỢP - NGƯỠNG ĐIỀU CHỈNH
            score_difference = bullish_score - bearish_score
            avg_strength = total_strength / max(signal_count, 1)
            
            # XÁC ĐỊNH HƯỚNG CHÍNH VÀ ĐỘ MẠNH - ĐIỀU CHỈNH NGƯỠNG
            if score_difference > 1.5 and avg_strength > 0.5:
                main_signal = "BUY"
                strength_level = "strong"
                confidence = min(avg_strength * (abs(score_difference) / 5), 0.95)
            elif score_difference > 0.8 and avg_strength > 0.4:
                main_signal = "BUY" 
                strength_level = "medium"
                confidence = min(avg_strength * (abs(score_difference) / 4), 0.85)
            elif score_difference > 0.3 and avg_strength > 0.3:
                main_signal = "BUY"
                strength_level = "weak"
                confidence = min(avg_strength * (abs(score_difference) / 3), 0.7)
            elif score_difference < -1.5 and avg_strength > 0.5:
                main_signal = "SELL"
                strength_level = "strong"
                confidence = min(avg_strength * (abs(score_difference) / 5), 0.95)
            elif score_difference < -0.8 and avg_strength > 0.4:
                main_signal = "SELL"
                strength_level = "medium" 
                confidence = min(avg_strength * (abs(score_difference) / 4), 0.85)
            elif score_difference < -0.3 and avg_strength > 0.3:
                main_signal = "SELL"
                strength_level = "weak"
                confidence = min(avg_strength * (abs(score_difference) / 3), 0.7)
            else:
                main_signal = "NEUTRAL"
                strength_level = "weak"
                confidence = 0.1
            
            return {
                'signal': main_signal,
                'strength': strength_level,
                'confidence': confidence,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'score_difference': score_difference,
                'avg_strength': avg_strength,
                'signal_count': signal_count
            }
            
        except Exception as e:
            logger.error(f"Lỗi phân tích tín hiệu tổng hợp: {str(e)}")
            return {'signal': 'NEUTRAL', 'strength': 'weak', 'confidence': 0.1}
    
    def get_final_signal_with_probability(self, symbol, signals_data):
        """LẤY TÍN HIỆU CUỐI CÙNG VỚI PHÂN TÍCH XÁC SUẤT VÀ KỲ VỌNG - LOGIC MỚI: XEM XÉT HƯỚNG NGƯỢC LẠI"""
        try:
            # PHÂN TÍCH TÍN HIỆU TỔNG HỢP
            combined_analysis = self.analyze_combined_signal(signals_data)
            main_signal = combined_analysis['signal']
            strength_level = combined_analysis['strength']
            base_confidence = combined_analysis['confidence']
    
            if main_signal == "NEUTRAL":
                return "NEUTRAL", 0, 0, 0
    
            # PHÂN TÍCH XÁC SUẤT LỊCH SỬ
            stats = self.analyze_historical_performance(symbol)
    
            # LẤY THỐNG KÊ CHO LOẠI TÍN HIỆU
            signal_type = "bullish" if main_signal == "BUY" else "bearish"
            opposite_signal_type = "bearish" if main_signal == "BUY" else "bullish"
    
            total_predictions = stats['combined_signals']['total_predictions'].get(signal_type, 0)
            total_predictions_opposite = stats['combined_signals']['total_predictions'].get(opposite_signal_type, 0)
    
            # NẾU KHÔNG CÓ DỮ LIỆU LỊCH SỬ, SỬ DỤNG GIÁ TRỊ MẶC ĐỊNH HỢP LÝ
            if total_predictions < self.min_data_points:
                probability = 0.55
                expectation = 0.5
                variance = 0.12
            else:
                correct_predictions = stats['combined_signals']['correct_predictions'].get(signal_type, 0)
                probability = correct_predictions / total_predictions
                expectation = stats['combined_signals']['expectations'].get(signal_type, 0.0)
                variance = stats['combined_signals']['variances'].get(signal_type, 0.1)
    
            # TÍNH ĐỘ TIN CẬY CUỐI CÙNG CHO HƯỚNG CHÍNH
            final_confidence = base_confidence * probability
    
            # ĐIỀU CHỈNH TĂNG ĐỘ TIN CẬY CHO TÍN HIỆU MẠNH
            if strength_level == "strong":
                final_confidence *= 1.2
            elif strength_level == "medium":
                final_confidence *= 1.1
    
            final_confidence = min(final_confidence, 0.95)  # Giới hạn tối đa
    
            # LOGIC MỚI: NẾU HƯỚNG CHÍNH CÓ CONFIDENCE THẤP, KIỂM TRA HƯỚNG NGƯỢC LẠI
            if final_confidence < 0.6:
                # TÍNH CONFIDENCE CHO HƯỚNG NGƯỢC LẠI
                if total_predictions_opposite >= self.min_data_points:
                    correct_predictions_opposite = stats['combined_signals']['correct_predictions'].get(opposite_signal_type, 0)
                    probability_opposite = correct_predictions_opposite / total_predictions_opposite
                    expectation_opposite = stats['combined_signals']['expectations'].get(opposite_signal_type, 0.0)
                    variance_opposite = stats['combined_signals']['variances'].get(opposite_signal_type, 0.1)
    
                    # TÍNH CONFIDENCE CHO HƯỚNG NGƯỢC LẠI (đảo ngược base_confidence)
                    opposite_base_confidence = 1.0 - base_confidence
                    final_confidence_opposite = opposite_base_confidence * probability_opposite
    
                    # ĐIỀU CHỈNH CHO TÍN HIỆU MẠNH
                    if strength_level == "strong":
                        final_confidence_opposite *= 1.2
                    elif strength_level == "medium":
                        final_confidence_opposite *= 1.1
    
                    final_confidence_opposite = min(final_confidence_opposite, 0.95)
    
                    # NẾU HƯỚNG NGƯỢC LẠI CÓ CONFIDENCE CAO HƠN 0.6, THÌ CHỌN HƯỚNG NGƯỢC LẠI
                    if final_confidence_opposite >= 0.55 and expectation_opposite > -2:
                        opposite_signal = "SELL" if main_signal == "BUY" else "BUY"
                        logger.info(f"🎯 {symbol} - ĐẢO CHIỀU: {opposite_signal}({strength_level}) | "
                                   f"Conf: {final_confidence_opposite:.2f} | "
                                   f"Prob: {probability_opposite:.2f} | "
                                   f"Exp: {expectation_opposite:.2f}% | "
                                   f"Var: {variance_opposite:.3f}")
                        return opposite_signal, final_confidence_opposite, expectation_opposite, variance_opposite
    
                # Nếu không đạt ngưỡng cho cả hai hướng, trả về NEUTRAL
                logger.info(f"⚪ {symbol} - KHÔNG GIAO DỊCH: Confidence chính {final_confidence:.2f} < 0.55 và ngược lại cũng không đủ")
                return "NEUTRAL", 0, 0, 0
    
            # Nếu hướng chính đạt ngưỡng, trả về hướng chính
            if final_confidence >= 0.6 and expectation > -2:
                logger.info(f"✅ {symbol} - QUYẾT ĐỊNH: {main_signal}({strength_level}) | "
                           f"Conf: {final_confidence:.2f} | "
                           f"Prob: {probability:.2f} | "
                           f"Exp: {expectation:.2f}% | "
                           f"Var: {variance:.3f}")
                return main_signal, final_confidence, expectation, variance
            else:
                logger.info(f"⚪ {symbol} - KHÔNG GIAO DỊCH: Confidence {final_confidence:.2f} < 0.6 hoặc Expectation {expectation:.2f}% quá thấp")
                return "NEUTRAL", 0, 0, 0
    
        except Exception as e:
            logger.error(f"Lỗi lấy tín hiệu cuối cùng: {str(e)}")
            return "NEUTRAL", 0, 0, 0
    def analyze_historical_performance(self, symbol):
        """PHÂN TÍCH HIỆU SUẤT LỊCH SỬ CHI TIẾT VỚI ĐA ĐIỂM - TỐI ƯU HIỆU SUẤT"""
        try:
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return self.probability_stats
            
            klines = self.get_historical_klines(symbol, '15m', self.lookback + self.evaluation_period)
            if not klines or len(klines) < 50:  # Giảm yêu cầu dữ liệu tối thiểu
                logger.warning(f"⚠️ Không đủ dữ liệu lịch sử cho {symbol}")
                return self._get_reasonable_default_stats()
            
            self._reset_stats()
            analyzer = TrendIndicatorSystem()
            
            processed_count = 0
            # Giới hạn số lượng điểm dữ liệu để xử lý
            max_data_points = min(len(klines) - self.evaluation_period, 80)
            
            for i in range(self.evaluation_period, max_data_points):
                try:
                    current_data = klines[i]
                    current_close = float(current_data[4])
                    future_data = klines[i + self.evaluation_period]
                    future_close = float(future_data[4])
                    
                    price_change = (future_close - current_close) / current_close * 100
                    is_price_up = price_change > 0
                    
                    historical_klines = klines[:i+1]
                    closes = [float(candle[4]) for candle in historical_klines]
                    
                    if len(closes) < 30:  # Giảm yêu cầu dữ liệu
                        continue
                    
                    # LẤY TÍN HIỆU VÀ CHỈ BÁO TẠI THỜI ĐIỂM LỊCH SỬ
                    signals_data = self._get_historical_signals(historical_klines, closes, analyzer)
                    combined_analysis = self.analyze_combined_signal(signals_data)
                    
                    # CẬP NHẬT THỐNG KÊ CHI TIẾT CHO TỪNG CHỈ BÁO
                    self._update_detailed_stats(signals_data, combined_analysis, is_price_up, price_change)
                    
                    processed_count += 1
                            
                except Exception as e:
                    continue
            
            if processed_count < self.min_data_points:
                logger.warning(f"⚠️ Không đủ điểm dữ liệu cho {symbol}: {processed_count}")
                return self._get_reasonable_default_stats()
            
            self._calculate_final_stats()
            self.last_update_time = current_time
            
            logger.info(f"📊 Đã cập nhật thống kê đa điểm cho {symbol} với {processed_count} điểm dữ liệu")
            return self.probability_stats
            
        except Exception as e:
            logger.error(f"Lỗi phân tích hiệu suất lịch sử: {str(e)}")
            return self._get_reasonable_default_stats()

    def _get_reasonable_default_stats(self):
        """TRẢ VỀ THỐNG KÊ MẶC ĐỊNH HỢP LÝ KHI KHÔNG CÓ ĐỦ DỮ LIỆU"""
        stats = self.probability_stats.copy()
        
        # THIẾT LẬP GIÁ TRỊ MẶC ĐỊNH HỢP LÝ
        for level in stats['rsi_multiple_points']['rsi_levels']:
            if 30 <= level <= 70:
                stats['rsi_multiple_points']['correct_predictions'][level] = 6
                stats['rsi_multiple_points']['total_predictions'][level] = 10
                stats['rsi_multiple_points']['expectations'][level] = 0.5
            else:
                stats['rsi_multiple_points']['correct_predictions'][level] = 7
                stats['rsi_multiple_points']['total_predictions'][level] = 10
                stats['rsi_multiple_points']['expectations'][level] = 1.0
        
        # TÍN HIỆU TỔNG HỢP
        stats['combined_signals']['correct_predictions']['bullish'] = 6
        stats['combined_signals']['total_predictions']['bullish'] = 10
        stats['combined_signals']['expectations']['bullish'] = 0.8
        stats['combined_signals']['correct_predictions']['bearish'] = 6
        stats['combined_signals']['total_predictions']['bearish'] = 10
        stats['combined_signals']['expectations']['bearish'] = 0.8
        
        # ĐỘ MẠNH TÍN HIỆU
        stats['signal_strength']['correct_predictions']['strong'] = 7
        stats['signal_strength']['total_predictions']['strong'] = 10
        stats['signal_strength']['expectations']['strong'] = 1.2
        
        return stats

    def _get_historical_signals(self, klines, closes, analyzer):
        """LẤY TẤT CẢ TÍN HIỆU CHỈ BÁO TẠI MỘT THỜI ĐIỂM LỊCH SỬ"""
        current_price = closes[-1] if closes else 0
        
        # TÍNH TOÁN CÁC CHỈ BÁO CƠ BẢN
        ema_fast = analyzer.calculate_ema(closes, analyzer.ema_fast)
        ema_slow = analyzer.calculate_ema(closes, analyzer.ema_slow)
        ema_trend = analyzer.calculate_ema(closes, analyzer.ema_trend)
        rsi = analyzer.calculate_rsi(closes, analyzer.rsi_period)
        
        # TÍN HIỆU EMA CHI TIẾT
        ema_conditions = self._get_ema_conditions(current_price, ema_fast, ema_slow, ema_trend, closes)
        ema_signal = "NEUTRAL"
        ema_strength = 0
        
        if ema_conditions['price_above_all_ema']:
            ema_signal = "BUY"
            ema_strength = 1.0
        elif ema_conditions['price_below_all_ema']:
            ema_signal = "SELL"
            ema_strength = 1.0
        elif ema_conditions['ema_alignment_bullish']:
            ema_signal = "BUY"
            ema_strength = 0.8
        elif ema_conditions['ema_alignment_bearish']:
            ema_signal = "SELL"
            ema_strength = 0.8
        elif ema_conditions['golden_cross_recent']:
            ema_signal = "BUY"
            ema_strength = 0.7
        elif ema_conditions['death_cross_recent']:
            ema_signal = "SELL"
            ema_strength = 0.7
        
        # TÍN HIỆU RSI
        rsi_signal = "NEUTRAL"
        rsi_strength = 0
        if rsi < 35:  # Điều chỉnh ngưỡng
            rsi_signal = "BUY"
            rsi_strength = min((35 - rsi) / 35, 1.0)
        elif rsi > 65:  # Điều chỉnh ngưỡng
            rsi_signal = "SELL"
            rsi_strength = min((rsi - 65) / 35, 1.0)
        
        # VOLUME
        volumes = [float(candle[5]) for candle in klines]
        current_volume = volumes[-1] if volumes else 0
        avg_volume = np.mean(volumes[-15:-1]) if len(volumes) >= 15 else 1.0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # SUPPORT/RESISTANCE
        support, resistance = analyzer.get_support_resistance_from_klines(klines)
        sr_signal = "NEUTRAL"
        sr_strength = 0
        if support > 0 and resistance > 0:
            distance_to_resistance = abs(resistance - current_price) / current_price
            distance_to_support = abs(current_price - support) / current_price
            
            if current_price > resistance and volume_ratio > 1.2:
                sr_signal = "BUY"
                sr_strength = min(volume_ratio * 0.3, 1.0)
            elif current_price < support and volume_ratio > 1.2:
                sr_signal = "SELL"
                sr_strength = min(volume_ratio * 0.3, 1.0)
            elif distance_to_resistance < 0.008:  # Nới lỏng ngưỡng
                sr_signal = "SELL"
                sr_strength = 0.6
            elif distance_to_support < 0.008:  # Nới lỏng ngưỡng
                sr_signal = "BUY"
                sr_strength = 0.6
        
        # MARKET STRUCTURE
        structure_signal = analyzer.analyze_market_structure(closes)
        
        return {
            'ema_signal': ema_signal,
            'ema_strength': ema_strength,
            'ema_conditions': ema_conditions,
            'rsi_signal': rsi_signal,
            'rsi_strength': rsi_strength,
            'rsi_value': rsi,
            'volume_ratio': volume_ratio,
            'price_vs_ema': current_price - ema_fast,
            'sr_signal': sr_signal,
            'sr_strength': sr_strength,
            'structure_signal': structure_signal
        }

    def _get_ema_conditions(self, current_price, ema_fast, ema_slow, ema_trend, closes):
        """LẤY CÁC ĐIỀU KIỆN EMA CHI TIẾT"""
        conditions = {}
        
        # ĐIỀU KIỆN CƠ BẢN
        conditions['price_above_all_ema'] = current_price > ema_fast > ema_slow > ema_trend
        conditions['price_below_all_ema'] = current_price < ema_fast < ema_slow < ema_trend
        conditions['ema_fast_above_slow'] = ema_fast > ema_slow
        conditions['ema_fast_below_slow'] = ema_fast < ema_slow
        conditions['ema_slow_above_trend'] = ema_slow > ema_trend
        conditions['ema_slow_below_trend'] = ema_slow < ema_trend
        
        # KIỂM TRA GOLDEN/DEATH CROSS
        conditions['golden_cross_recent'] = False
        conditions['death_cross_recent'] = False
        if len(closes) >= 8:
            prev_ema_fast = self.calculate_ema(closes[:-3], 9)  # EMA fast trước 3 nến
            prev_ema_slow = self.calculate_ema(closes[:-3], 21)  # EMA slow trước 3 nến
            conditions['golden_cross_recent'] = ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow
            conditions['death_cross_recent'] = ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow
        
        # CĂN CHỈNH EMA
        conditions['ema_alignment_bullish'] = ema_fast > ema_slow > ema_trend
        conditions['ema_alignment_bearish'] = ema_fast < ema_slow < ema_trend
        
        return conditions

    def calculate_ema(self, prices, period):
        """TÍNH EMA CHO PHÂN TÍCH LỊCH SỬ"""
        if len(prices) < period:
            return prices[-1] if prices else 0
            
        ema = [prices[0]]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            ema.append(ema_value)
            
        return ema[-1]

    def _update_detailed_stats(self, signals_data, combined_analysis, is_correct, price_change):
        """CẬP NHẬT THỐNG KÊ CHI TIẾT CHO TẤT CẢ CHỈ BÁO"""
        
        # CẬP NHẬT RSI - NHIỀU ĐIỂM
        rsi_value = signals_data.get('rsi_value', 50)
        closest_rsi_level = self.get_closest_rsi_level(rsi_value)
        rsi_zone = self.get_rsi_zone(rsi_value)
        
        self.probability_stats['rsi_multiple_points']['total_predictions'][closest_rsi_level] += 1
        self.probability_stats['rsi_multiple_points']['zone_total'][rsi_zone] += 1
        
        if is_correct:
            self.probability_stats['rsi_multiple_points']['correct_predictions'][closest_rsi_level] += 1
            self.probability_stats['rsi_multiple_points']['zone_correct'][rsi_zone] += 1
        
        self.probability_stats['rsi_multiple_points']['expectations'][closest_rsi_level] += price_change
        
        # CẬP NHẬT EMA - NHIỀU ĐIỀU KIỆN
        ema_conditions = signals_data.get('ema_conditions', {})
        for condition, is_true in ema_conditions.items():
            if is_true:
                self.probability_stats['ema_multiple_conditions']['total_predictions'][condition] += 1
                
                # XÁC ĐỊNH XEM ĐIỀU KIỆN CÓ DỰ ĐOÁN ĐÚNG KHÔNG
                if condition in ['price_above_all_ema', 'ema_fast_above_slow', 'ema_slow_above_trend', 
                               'golden_cross_recent', 'ema_alignment_bullish']:
                    condition_correct = is_correct
                else:
                    condition_correct = not is_correct
                
                if condition_correct:
                    self.probability_stats['ema_multiple_conditions']['correct_predictions'][condition] += 1
                
                self.probability_stats['ema_multiple_conditions']['expectations'][condition] += price_change
        
        # CẬP NHẬT VOLUME - NHIỀU MỨC
        volume_ratio = signals_data.get('volume_ratio', 1.0)
        closest_volume_level = self.get_closest_volume_level(volume_ratio)
        volume_zone = self.get_volume_zone(volume_ratio)
        
        self.probability_stats['volume_multiple_levels']['total_predictions'][closest_volume_level] += 1
        self.probability_stats['volume_multiple_levels']['zone_total'][volume_zone] += 1
        
        if is_correct:
            self.probability_stats['volume_multiple_levels']['correct_predictions'][closest_volume_level] += 1
            self.probability_stats['volume_multiple_levels']['zone_correct'][volume_zone] += 1
        
        self.probability_stats['volume_multiple_levels']['expectations'][closest_volume_level] += price_change
        
        # CẬP NHẬT TÍN HIỆU TỔNG HỢP
        if combined_analysis['signal'] != "NEUTRAL":
            signal_type = "bullish" if combined_analysis['signal'] == "BUY" else "bearish"
            self.probability_stats['combined_signals']['total_predictions'][signal_type] += 1
            
            signal_correct = (combined_analysis['signal'] == "BUY" and is_correct) or \
                           (combined_analysis['signal'] == "SELL" and not is_correct)
            
            if signal_correct:
                self.probability_stats['combined_signals']['correct_predictions'][signal_type] += 1
            
            self.probability_stats['combined_signals']['expectations'][signal_type] += price_change
            
            # CẬP NHẬT ĐỘ MẠNH TÍN HIỆU
            strength_level = combined_analysis['strength']
            self.probability_stats['signal_strength']['total_predictions'][strength_level] += 1
            
            if signal_correct:
                self.probability_stats['signal_strength']['correct_predictions'][strength_level] += 1
            
            self.probability_stats['signal_strength']['expectations'][strength_level] += price_change

    def _calculate_final_stats(self):
        """TÍNH TOÁN GIÁ TRỊ CUỐI CÙNG CHO TẤT CẢ THỐNG KÊ"""
        
        # TÍNH CHO RSI
        for level in self.probability_stats['rsi_multiple_points']['rsi_levels']:
            total = self.probability_stats['rsi_multiple_points']['total_predictions'][level]
            if total > 0:
                self.probability_stats['rsi_multiple_points']['expectations'][level] /= total
                base_variance = abs(self.probability_stats['rsi_multiple_points']['expectations'][level]) * 0.15
                self.probability_stats['rsi_multiple_points']['variances'][level] = max(base_variance, 0.05)
        
        # TÍNH CHO EMA
        for condition in self.probability_stats['ema_multiple_conditions']['conditions']:
            total = self.probability_stats['ema_multiple_conditions']['total_predictions'][condition]
            if total > 0:
                self.probability_stats['ema_multiple_conditions']['expectations'][condition] /= total
        
        # TÍNH CHO VOLUME
        for ratio in self.probability_stats['volume_multiple_levels']['volume_ratios']:
            total = self.probability_stats['volume_multiple_levels']['total_predictions'][ratio]
            if total > 0:
                self.probability_stats['volume_multiple_levels']['expectations'][ratio] /= total
        
        # TÍNH CHO TÍN HIỆU TỔNG HỢP
        for signal_type in ['bullish', 'bearish', 'neutral']:
            total = self.probability_stats['combined_signals']['total_predictions'][signal_type]
            if total > 0:
                self.probability_stats['combined_signals']['expectations'][signal_type] /= total
                base_variance = abs(self.probability_stats['combined_signals']['expectations'][signal_type]) * 0.15
                self.probability_stats['combined_signals']['variances'][signal_type] = max(base_variance, 0.05)
        
        # TÍNH CHO ĐỘ MẠNH TÍN HIỆU
        for strength in ['weak', 'medium', 'strong']:
            total = self.probability_stats['signal_strength']['total_predictions'][strength]
            if total > 0:
                self.probability_stats['signal_strength']['expectations'][strength] /= total
                base_variance = abs(self.probability_stats['signal_strength']['expectations'][strength]) * 0.1
                self.probability_stats['signal_strength']['variances'][strength] = max(base_variance, 0.03)

    def _reset_stats(self):
        """RESET LẠI TẤT CẢ THỐNG KÊ"""
        # Reset RSI
        for level in self.probability_stats['rsi_multiple_points']['rsi_levels']:
            self.probability_stats['rsi_multiple_points']['correct_predictions'][level] = 0
            self.probability_stats['rsi_multiple_points']['total_predictions'][level] = 0
            self.probability_stats['rsi_multiple_points']['expectations'][level] = 0.0
            self.probability_stats['rsi_multiple_points']['variances'][level] = 0.0
        
        for zone in self.probability_stats['rsi_multiple_points']['rsi_zones']:
            self.probability_stats['rsi_multiple_points']['zone_correct'][zone] = 0
            self.probability_stats['rsi_multiple_points']['zone_total'][zone] = 0
        
        # Reset EMA
        for condition in self.probability_stats['ema_multiple_conditions']['conditions']:
            self.probability_stats['ema_multiple_conditions']['correct_predictions'][condition] = 0
            self.probability_stats['ema_multiple_conditions']['total_predictions'][condition] = 0
            self.probability_stats['ema_multiple_conditions']['expectations'][condition] = 0.0
        
        # Reset Volume
        for ratio in self.probability_stats['volume_multiple_levels']['volume_ratios']:
            self.probability_stats['volume_multiple_levels']['correct_predictions'][ratio] = 0
            self.probability_stats['volume_multiple_levels']['total_predictions'][ratio] = 0
            self.probability_stats['volume_multiple_levels']['expectations'][ratio] = 0.0
        
        for zone in self.probability_stats['volume_multiple_levels']['volume_zones']:
            self.probability_stats['volume_multiple_levels']['zone_correct'][zone] = 0
            self.probability_stats['volume_multiple_levels']['zone_total'][zone] = 0
        
        # Reset Combined
        for signal_type in ['bullish', 'bearish', 'neutral']:
            self.probability_stats['combined_signals']['correct_predictions'][signal_type] = 0
            self.probability_stats['combined_signals']['total_predictions'][signal_type] = 0
            self.probability_stats['combined_signals']['expectations'][signal_type] = 0.0
            self.probability_stats['combined_signals']['variances'][signal_type] = 0.0
        
        # Reset Strength
        for strength in ['weak', 'medium', 'strong']:
            self.probability_stats['signal_strength']['correct_predictions'][strength] = 0
            self.probability_stats['signal_strength']['total_predictions'][strength] = 0
            self.probability_stats['signal_strength']['expectations'][strength] = 0.0
            self.probability_stats['signal_strength']['variances'][strength] = 0.0

    def get_historical_klines(self, symbol, interval, limit):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            return binance_api_request(url, params=params)
        except Exception as e:
            logger.error(f"Lỗi lấy nến lịch sử {symbol}: {str(e)}")
            return None

    def get_detailed_probability_report(self, symbol):
        """BÁO CÁO XÁC SUẤT CHI TIẾT VỚI ĐA ĐIỂM"""
        try:
            stats = self.analyze_historical_performance(symbol)
            if not stats:
                return "❌ Không có dữ liệu thống kê"
            
            report = f"📊 <b>BÁO CÁO XÁC SUẤT ĐA ĐIỂM - {symbol}</b>\n\n"
            
            report += "📈 <b>RSI - NHIỀU ĐIỂM:</b>\n"
            for level in [20, 30, 40, 50, 60, 70, 80]:
                total = stats['rsi_multiple_points']['total_predictions'][level]
                if total > 5:  # Giảm ngưỡng
                    correct = stats['rsi_multiple_points']['correct_predictions'][level]
                    prob = correct / total
                    exp = stats['rsi_multiple_points']['expectations'][level]
                    report += f"   RSI {level}: {prob:.1%} (E:{exp:.2f}%)\n"
            
            report += "\n📉 <b>EMA - NHIỀU ĐIỀU KIỆN:</b>\n"
            for condition in ['price_above_all_ema', 'ema_alignment_bullish', 'golden_cross_recent']:
                total = stats['ema_multiple_conditions']['total_predictions'][condition]
                if total > 3:  # Giảm ngưỡng
                    correct = stats['ema_multiple_conditions']['correct_predictions'][condition]
                    prob = correct / total
                    exp = stats['ema_multiple_conditions']['expectations'][condition]
                    report += f"   {condition}: {prob:.1%} (E:{exp:.2f}%)\n"
            
            report += "\n📊 <b>VOLUME - NHIỀU MỨC:</b>\n"
            for ratio in [0.8, 1.0, 1.2, 1.5, 2.0]:
                total = stats['volume_multiple_levels']['total_predictions'][ratio]
                if total > 3:  # Giảm ngưỡng
                    correct = stats['volume_multiple_levels']['correct_predictions'][ratio]
                    prob = correct / total
                    exp = stats['volume_multiple_levels']['expectations'][ratio]
                    report += f"   Vol {ratio}x: {prob:.1%} (E:{exp:.2f}%)\n"
            
            report += "\n🎯 <b>TÍN HIỆU TỔNG HỢP:</b>\n"
            for signal_type in ['bullish', 'bearish']:
                total = stats['combined_signals']['total_predictions'][signal_type]
                if total > 0:
                    correct = stats['combined_signals']['correct_predictions'][signal_type]
                    prob = correct / total
                    exp = stats['combined_signals']['expectations'][signal_type]
                    var = stats['combined_signals']['variances'][signal_type]
                    report += f"   {signal_type.upper()}: {prob:.1%} (E:{exp:.2f}%, V:{var:.3f})\n"
            
            return report
            
        except Exception as e:
            return f"❌ Lỗi báo cáo: {str(e)}"

# ========== HỆ THỐNG CHỈ BÁO XU HƯỚNG TÍCH HỢP ==========
class TrendIndicatorSystem:
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.ema_trend = 50
        self.rsi_period = 14
        self.lookback = 80  # Giảm lookback
        self.probability_analyzer = ProbabilityAnalyzer()
    
    def calculate_ema(self, prices, period):
        if len(prices) < period:
            return prices[-1] if prices else 0
            
        ema = [prices[0]]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            ema.append(ema_value)
            
        return ema[-1]
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
            
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
            
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_volume_data(self, symbol):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': '15m',
                'limit': 15  # Giảm số nến
            }
            data = binance_api_request(url, params=params)
            if not data:
                return 1.0
                
            volumes = [float(candle[5]) for candle in data]
            if len(volumes) < 2:
                return 1.0
                
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])
            
            return current_volume / avg_volume if avg_volume > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Lỗi lấy volume {symbol}: {str(e)}")
            return 1.0
    
    def get_support_resistance(self, symbol):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': '15m',
                'limit': 25  # Giảm số nến
            }
            data = binance_api_request(url, params=params)
            if not data or len(data) < 15:
                return 0, 0
                
            highs = [float(candle[2]) for candle in data]
            lows = [float(candle[3]) for candle in data]
            
            resistance = max(highs[-15:])
            support = min(lows[-15:])
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Lỗi lấy S/R {symbol}: {str(e)}")
            return 0, 0

    def get_support_resistance_from_klines(self, klines):
        if not klines or len(klines) < 15:
            return 0, 0
            
        highs = [float(candle[2]) for candle in klines]
        lows = [float(candle[3]) for candle in klines]
        
        resistance = max(highs[-15:])
        support = min(lows[-15:])
        
        return support, resistance
    
    def analyze_market_structure(self, prices):
        if len(prices) < 8:  # Giảm yêu cầu
            return "NEUTRAL"
            
        recent_highs = prices[-4:]
        recent_lows = prices[-4:]
        prev_highs = prices[-8:-4] 
        prev_lows = prices[-8:-4]
        
        if (max(recent_highs) > max(prev_highs) and 
            min(recent_lows) > min(prev_lows)):
            return "BUY"
        elif (max(recent_highs) < max(prev_highs) and 
              min(recent_lows) < min(prev_lows)):
            return "SELL"
        return "NEUTRAL"
    
    def analyze_symbol(self, symbol):
        try:
            klines = self.get_klines(symbol, '15m', self.lookback)
            if not klines or len(klines) < 30:  # Giảm yêu cầu
                return "NEUTRAL"
            
            closes = [float(candle[4]) for candle in klines]
            
            signals_data = self._calculate_all_indicators(closes, symbol)
            
            final_signal, confidence, expectation, variance = \
                self.probability_analyzer.get_final_signal_with_probability(symbol, signals_data)
            
            # GIỮ NGUYÊN NGƯỠNG CHẤT LƯỢNG
            if confidence >= 0.6 and expectation > -2:
                logger.info(f"✅ {symbol} - QUYẾT ĐỊNH: {final_signal} "
                           f"(Conf: {confidence:.2f}, Exp: {expectation:.2f}%)")
                return final_signal
            
            logger.info(f"⚪ {symbol} - KHÔNG GIAO DỊCH: "
                       f"Confidence {confidence:.2f} < 0.6 hoặc Expectation {expectation:.2f}% quá thấp")
            return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"❌ Lỗi phân tích {symbol}: {str(e)}")
            return "NEUTRAL"
    
    def _calculate_all_indicators(self, closes, symbol):
        current_price = closes[-1]
        
        ema_fast = self.calculate_ema(closes, self.ema_fast)
        ema_slow = self.calculate_ema(closes, self.ema_slow)
        ema_trend = self.calculate_ema(closes, self.ema_trend)
        
        ema_signal = "NEUTRAL"
        ema_strength = 0
        if current_price > ema_fast > ema_slow > ema_trend:
            ema_signal = "BUY"
            ema_strength = 1.0
        elif current_price < ema_fast < ema_slow < ema_trend:
            ema_signal = "SELL" 
            ema_strength = 1.0
        elif current_price > ema_fast > ema_slow:
            ema_signal = "BUY"
            ema_strength = 0.8
        elif current_price < ema_fast < ema_slow:
            ema_signal = "SELL"
            ema_strength = 0.8
        elif ema_fast > ema_slow > ema_trend:
            ema_signal = "BUY"
            ema_strength = 0.6
        elif ema_fast < ema_slow < ema_trend:
            ema_signal = "SELL"
            ema_strength = 0.6
        
        rsi = self.calculate_rsi(closes, self.rsi_period)
        
        rsi_signal = "NEUTRAL"
        rsi_strength = 0
        if rsi < 35:  # Điều chỉnh ngưỡng
            rsi_signal = "BUY"
            rsi_strength = min((35 - rsi) / 35, 1.0)
        elif rsi > 65:  # Điều chỉnh ngưỡng
            rsi_signal = "SELL" 
            rsi_strength = min((rsi - 65) / 35, 1.0)
        
        volume_ratio = self.get_volume_data(symbol)
        
        support, resistance = self.get_support_resistance(symbol)
        sr_signal = "NEUTRAL"
        sr_strength = 0
        
        if support > 0 and resistance > 0:
            distance_to_resistance = abs(resistance - current_price) / current_price
            distance_to_support = abs(current_price - support) / current_price
            
            if current_price > resistance and volume_ratio > 1.2:
                sr_signal = "BUY"
                sr_strength = min(volume_ratio * 0.3, 1.0)
            elif current_price < support and volume_ratio > 1.2:
                sr_signal = "SELL"
                sr_strength = min(volume_ratio * 0.3, 1.0)
            elif distance_to_resistance < 0.008:
                sr_signal = "SELL"
                sr_strength = 0.6
            elif distance_to_support < 0.008:
                sr_signal = "BUY"
                sr_strength = 0.6
        
        structure_signal = self.analyze_market_structure(closes)
        
        return {
            'ema_signal': ema_signal,
            'ema_strength': ema_strength,
            'rsi_signal': rsi_signal,
            'rsi_strength': rsi_strength,
            'volume_ratio': volume_ratio,
            'price_vs_ema': current_price - ema_fast,
            'sr_signal': sr_signal,
            'sr_strength': sr_strength,
            'structure_signal': structure_signal
        }

    def get_klines(self, symbol, interval, limit):
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            return binance_api_request(url, params=params)
        except Exception as e:
            logger.error(f"Lỗi lấy nến {symbol} {interval}: {str(e)}")
            return None

    def get_probability_report(self, symbol):
        return self.probability_analyzer.get_detailed_probability_report(symbol)

# ========== SMART COIN FINDER NÂNG CẤP ==========
class SmartCoinFinder:
    """TÌM COIN THÔNG MINH DỰA TRÊN HỆ THỐNG XU HƯỚNG VÀ ĐÒN BẨY"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.analyzer = TrendIndicatorSystem()
        self.leverage_cache = {}
        self.qualified_symbols_cache = {}
        self.cache_timeout = 300
        self.last_cache_update = 0
        
    def clear_cache(self):
        self.leverage_cache.clear()
        self.qualified_symbols_cache.clear()
        self.last_cache_update = 0
        logger.info("🧹 Đã xóa cache tìm kiếm coin")
        
    def get_pre_filtered_symbols(self, target_leverage):
        try:
            current_time = time.time()
            
            if (target_leverage in self.qualified_symbols_cache and 
                self.qualified_symbols_cache[target_leverage] and
                current_time - self.last_cache_update < self.cache_timeout):
                return self.qualified_symbols_cache[target_leverage]
            
            logger.info(f"🔍 Đang lọc coin hỗ trợ đòn bẩy ≥ {target_leverage}x...")
            all_symbols = get_all_usdt_pairs(limit=400)  # Giảm số lượng coin
            if not all_symbols:
                if target_leverage in self.qualified_symbols_cache:
                    return self.qualified_symbols_cache[target_leverage]
                return []
            
            qualified_symbols = []
            
            def check_symbol_leverage(symbol):
                try:
                    max_leverage = self.get_symbol_leverage(symbol)
                    return symbol if max_leverage >= target_leverage else None
                except:
                    return None
            
            with ThreadPoolExecutor(max_workers=8) as executor:  # Giảm workers
                results = list(executor.map(check_symbol_leverage, all_symbols))
            
            qualified_symbols = [symbol for symbol in results if symbol is not None]
            
            self.qualified_symbols_cache[target_leverage] = qualified_symbols
            self.last_cache_update = current_time
            
            logger.info(f"✅ Đã lọc được {len(qualified_symbols)} coin hỗ trợ đòn bẩy ≥ {target_leverage}x")
            return qualified_symbols
            
        except Exception as e:
            logger.error(f"❌ Lỗi lọc coin theo đòn bẩy: {str(e)}")
            if target_leverage in self.qualified_symbols_cache:
                return self.qualified_symbols_cache[target_leverage]
            return []
        
    def get_symbol_leverage(self, symbol):
        if symbol in self.leverage_cache:
            return self.leverage_cache[symbol]
        
        max_leverage = get_max_leverage(symbol, self.api_key, self.api_secret)
        self.leverage_cache[symbol] = max_leverage
        return max_leverage
    
    def find_coin_by_direction(self, target_direction, target_leverage, excluded_symbols=None):
        try:
            if excluded_symbols is None:
                excluded_symbols = set()
            
            logger.info(f"🔍 Bot đang tìm coin {target_direction} với đòn bẩy {target_leverage}x...")
            
            qualified_symbols = self.get_pre_filtered_symbols(target_leverage)
            if not qualified_symbols:
                logger.error(f"❌ Không tìm thấy coin nào hỗ trợ đòn bẩy {target_leverage}x")
                return None
            
            available_symbols = [s for s in qualified_symbols if s not in excluded_symbols]
            
            if not available_symbols:
                logger.warning(f"⚠️ Tất cả coin đủ đòn bẩy đều đang được trade: {excluded_symbols}")
                return None
            
            random.shuffle(available_symbols)
            symbols_to_check = available_symbols[:30]  # Giảm số lượng coin cần kiểm tra
            
            logger.info(f"🔍 Sẽ kiểm tra {len(symbols_to_check)} coin đủ đòn bẩy...")
            
            checked_count = 0
            signal_passed = 0
            
            for symbol in symbols_to_check:
                try:
                    checked_count += 1
                    
                    current_max_leverage = self.get_symbol_leverage(symbol)
                    if current_max_leverage < target_leverage:
                        continue
                    
                    signal = self.analyzer.analyze_symbol(symbol)
                    
                    if signal == target_direction:
                        signal_passed += 1
                        max_leverage = current_max_leverage
                        
                        logger.info(f"✅ Bot đã tìm thấy coin: {symbol} - {target_direction} - Đòn bẩy: {max_leverage}x")
                        return {
                            'symbol': symbol,
                            'direction': target_direction,
                            'max_leverage': max_leverage,
                            'score': 0.8,
                            'qualified': True
                        }
                    else:
                        logger.debug(f"⚪ {symbol} - Tín hiệu {signal} không khớp {target_direction}")
                        
                except Exception as e:
                    logger.debug(f"❌ Lỗi phân tích {symbol}: {str(e)}")
                    continue
            
            logger.warning(f"⚠️ Không tìm thấy coin {target_direction} phù hợp. "
                          f"Đã kiểm tra: {checked_count} coin, "
                          f"Tín hiệu đạt: {signal_passed}")
            return None
                
        except Exception as e:
            logger.error(f"❌ Lỗi tìm coin: {str(e)}")
            return None

# ========== QUẢN LÝ COIN CHUNG ==========
class CoinManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CoinManager, cls).__new__(cls)
                cls._instance.managed_coins = {}
                cls._instance.config_coin_count = {}
        return cls._instance
    
    def register_coin(self, symbol, bot_id, strategy, config_key=None):
        with self._lock:
            if symbol not in self.managed_coins:
                self.managed_coins[symbol] = {
                    "strategy": strategy, 
                    "bot_id": bot_id,
                    "config_key": config_key
                }
                return True
            return False
    
    def unregister_coin(self, symbol):
        with self._lock:
            if symbol in self.managed_coins:
                del self.managed_coins[symbol]
                return True
            return False
    
    def is_coin_managed(self, symbol):
        with self._lock:
            return symbol in self.managed_coins
    
    def get_managed_coins(self):
        with self._lock:
            return self.managed_coins.copy()

# ========== API BINANCE ==========
def sign(query, api_secret):
    try:
        return hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"Lỗi tạo chữ ký: {str(e)}")
        return ""

def binance_api_request(url, method='GET', params=None, headers=None):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if method.upper() == 'GET':
                if params:
                    query = urllib.parse.urlencode(params)
                    url = f"{url}?{query}"
                req = urllib.request.Request(url, headers=headers or {})
            else:
                data = urllib.parse.urlencode(params).encode() if params else None
                req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    return json.loads(response.read().decode())
                else:
                    error_content = response.read().decode()
                    logger.error(f"Lỗi API ({response.status}): {error_content}")
                    if response.status == 401:
                        return None
                    if response.status == 429:
                        time.sleep(2 ** attempt)
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            logger.error(f"Lỗi HTTP ({e.code}): {e.reason}")
            if e.code == 401:
                return None
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Lỗi kết nối API: {str(e)}")
            time.sleep(1)
    
    logger.error(f"Không thể thực hiện yêu cầu API sau {max_retries} lần thử")
    return None

def get_all_usdt_pairs(limit=400):
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            logger.warning("Không lấy được dữ liệu từ Binance, trả về danh sách rỗng")
            return []
        
        usdt_pairs = []
        for symbol_info in data.get('symbols', []):
            symbol = symbol_info.get('symbol', '')
            if symbol.endswith('USDT') and symbol_info.get('status') == 'TRADING':
                usdt_pairs.append(symbol)
        
        logger.info(f"✅ Lấy được {len(usdt_pairs)} coin USDT từ Binance")
        return usdt_pairs[:limit] if limit else usdt_pairs
        
    except Exception as e:
        logger.error(f"❌ Lỗi lấy danh sách coin từ Binance: {str(e)}")
        return []

def get_step_size(symbol, api_key, api_secret):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        data = binance_api_request(url)
        if not data:
            return 0.001
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        return float(f['stepSize'])
    except Exception as e:
        logger.error(f"Lỗi lấy step size: {str(e)}")
    return 0.001

def set_leverage(symbol, lev, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "leverage": lev,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/leverage?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        response = binance_api_request(url, method='POST', headers=headers)
        if response is None:
            return False
        if response and 'leverage' in response:
            return True
        return False
    except Exception as e:
        logger.error(f"Lỗi thiết lập đòn bẩy: {str(e)}")
        return False

def get_balance(api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/account?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        data = binance_api_request(url, headers=headers)
        if not data:
            return None
        for asset in data['assets']:
            if asset['asset'] == 'USDT':
                return float(asset['availableBalance'])
        return 0
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {str(e)}")
        return None

def place_order(symbol, side, qty, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "timestamp": ts
        }
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/order?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        return binance_api_request(url, method='POST', headers=headers)
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh: {str(e)}")
    return None

def cancel_all_orders(symbol, api_key, api_secret):
    try:
        ts = int(time.time() * 1000)
        params = {"symbol": symbol.upper(), "timestamp": ts}
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v1/allOpenOrders?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        binance_api_request(url, method='DELETE', headers=headers)
        return True
    except Exception as e:
        logger.error(f"Lỗi hủy lệnh: {str(e)}")
    return False

def get_current_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'price' in data:
            return float(data['price'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá: {str(e)}")
    return 0

def get_positions(symbol=None, api_key=None, api_secret=None):
    try:
        ts = int(time.time() * 1000)
        params = {"timestamp": ts}
        if symbol:
            params["symbol"] = symbol.upper()
        query = urllib.parse.urlencode(params)
        sig = sign(query, api_secret)
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{query}&signature={sig}"
        headers = {'X-MBX-APIKEY': api_key}
        
        positions = binance_api_request(url, headers=headers)
        if not positions:
            return []
        if symbol:
            for pos in positions:
                if pos['symbol'] == symbol.upper():
                    return [pos]
        return positions
    except Exception as e:
        logger.error(f"Lỗi lấy vị thế: {str(e)}")
    return []

# ========== WEBSOCKET MANAGER ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=8)  # Giảm workers
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def add_symbol(self, symbol, callback):
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self.connections:
                self._create_connection(symbol, callback)
                
    def _create_connection(self, symbol, callback):
        if self._stop_event.is_set():
            return
        stream = f"{symbol.lower()}@trade"
        url = f"wss://fstream.binance.com/ws/{stream}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'p' in data:
                    price = float(data['p'])
                    self.executor.submit(callback, price)
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn WebSocket {symbol}: {str(e)}")
                
        def on_error(ws, error):
            logger.error(f"Lỗi WebSocket {symbol}: {str(error)}")
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket đóng {symbol}: {close_status_code} - {close_msg}")
            if not self._stop_event.is_set() and symbol in self.connections:
                time.sleep(5)
                self._reconnect(symbol, callback)
                
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        
        self.connections[symbol] = {
            'ws': ws,
            'thread': thread,
            'callback': callback
        }
        logger.info(f"WebSocket bắt đầu cho {symbol}")
        
    def _reconnect(self, symbol, callback):
        logger.info(f"Kết nối lại WebSocket cho {symbol}")
        self.remove_symbol(symbol)
        self._create_connection(symbol, callback)
        
    def remove_symbol(self, symbol):
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.connections:
                try:
                    self.connections[symbol]['ws'].close()
                except Exception as e:
                    logger.error(f"Lỗi đóng WebSocket {symbol}: {str(e)}")
                del self.connections[symbol]
                logger.info(f"WebSocket đã xóa cho {symbol}")
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== BASE BOT NÂNG CẤP ==========
class BaseBot:
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, strategy_name, config_key=None, bot_id=None):
        
        self.symbol = symbol.upper() if symbol else None
        self.lev = lev
        self.percent = percent
        self.tp = tp
        self.sl = sl
        self.ws_manager = ws_manager
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.strategy_name = strategy_name
        self.config_key = config_key
        self.bot_id = bot_id or f"{strategy_name}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.status = "searching"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self.prices = []
        self.position_open = False
        self._stop = False
        
        self.last_trade_time = 0
        self.last_close_time = 0
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 3
        self.position_check_interval = 30
        
        self._close_attempted = False
        self._last_close_attempt = 0
        
        self.should_be_removed = False
        
        self.position_balance_check = 0
        self.balance_check_interval = 60
        
        self.coin_manager = CoinManager()
        self.coin_finder = SmartCoinFinder(api_key, api_secret)
        
        self.current_target_direction = None
        self.last_find_time = 0
        self.find_interval = 45  # Giảm thời gian tìm kiếm
        
        self.check_position_status()
        if self.symbol:
            self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        if self.symbol:
            self.log(f"🟢 Bot {strategy_name} khởi động | {self.symbol} | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")
        else:
            self.log(f"🟢 Bot {strategy_name} khởi động | Đang tìm coin... | ĐB: {lev}x | Vốn: {percent}% | TP/SL: {tp}%/{sl}%")

    def _register_coin_with_retry(self, symbol):
        max_retries = 3
        for attempt in range(max_retries):
            success = self.coin_manager.register_coin(symbol, self.bot_id, self.strategy_name, self.config_key)
            if success:
                return True
            time.sleep(0.5)
        return False

    def log(self, message):
        bot_info = f"[Bot {self.bot_id}]" if hasattr(self, 'bot_id') else ""
        logger.info(f"{bot_info} [{self.symbol or 'NO_COIN'} - {self.strategy_name}] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            symbol_info = f"<b>{self.symbol}</b>" if self.symbol else "<i>Đang tìm coin...</i>"
            send_telegram(f"{symbol_info} ({self.strategy_name} - Bot {self.bot_id}): {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def clear_finder_cache(self):
        try:
            self.coin_finder.clear_cache()
            self.log("🧹 Đã xóa cache tìm kiếm coin")
        except Exception as e:
            self.log(f"⚠️ Lỗi khi xóa cache: {str(e)}")

    def _handle_price_update(self, price):
        if self._stop or not price or price <= 0:
            return
        try:
            self.prices.append(float(price))
            if len(self.prices) > 50:  # Giảm kích thước buffer
                self.prices = self.prices[-50:]
        except Exception as e:
            self.log(f"❌ Lỗi xử lý giá: {str(e)}")

    def get_signal(self):
        raise NotImplementedError("Phương thức get_signal cần được triển khai")

    def get_target_direction(self):
        """
        XÁC ĐỊNH HƯỚNG GIAO DỊCH - RANDOM HOÀN TOÀN KHÔNG ÉP HƯỚNG
        """
        try:
            # RANDOM HOÀN TOÀN - 50% BUY, 50% SELL
            direction = "BUY" if random.random() > 0.5 else "SELL"
            
            self.log(f"🎲 QUYẾT ĐỊNH HƯỚNG: RANDOM {direction}")
            return direction
            
        except Exception as e:
            self.log(f"❌ Lỗi random direction: {str(e)}")
            return "BUY" if random.random() > 0.5 else "SELL"

    def verify_leverage_and_switch(self):
        if not self.symbol or not self.position_open:
            return True
            
        try:
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            
            if current_leverage < self.lev:
                self.log(f"⚠️ Coin {self.symbol} chỉ hỗ trợ đòn bẩy {current_leverage}x < {self.lev}x -> TÌM COIN MỚI")
                
                if self.position_open:
                    self.close_position(f"Đòn bẩy không đủ ({current_leverage}x < {self.lev}x)")
                
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
                self.symbol = None
                self.status = "searching"
                return False
                
            return True
            
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra đòn bẩy: {str(e)}")
            return True

    def find_and_set_coin(self):
        try:
            current_time = time.time()
            if current_time - self.last_find_time < self.find_interval:
                return False
                
            self.current_target_direction = self.get_target_direction()
            
            self.log(f"🎯 Đang tìm coin {self.current_target_direction} với đòn bẩy {self.lev}x...")
            
            managed_coins = self.coin_manager.get_managed_coins()
            excluded_symbols = set(managed_coins.keys())
            
            if excluded_symbols:
                self.log(f"🚫 Tránh các coin đang trade: {', '.join(list(excluded_symbols)[:5])}...")
            
            coin_data = self.coin_finder.find_coin_by_direction(
                self.current_target_direction, 
                self.lev,
                excluded_symbols
            )
        
            if coin_data is None:
                self.log(f"⚠️ Không tìm thấy coin {self.current_target_direction} với đòn bẩy {self.lev}x phù hợp")
                self.last_find_time = current_time
                return False
                
            if not coin_data.get('qualified', False):
                self.log(f"⚠️ Coin {coin_data.get('symbol', 'UNKNOWN')} không đủ tiêu chuẩn, tìm coin khác")
                self.last_find_time = current_time
                return False
            
            new_symbol = coin_data['symbol']
            max_leverage = coin_data.get('max_leverage', 100)
            
            if max_leverage < self.lev:
                self.log(f"❌ Coin {new_symbol} chỉ hỗ trợ {max_leverage}x < {self.lev}x -> BỎ QUA VÀ TÌM COIN KHÁC")
                self.last_find_time = current_time
                return False
            
            if self._register_coin_with_retry(new_symbol):
                if self.symbol:
                    self.ws_manager.remove_symbol(self.symbol)
                    self.coin_manager.unregister_coin(self.symbol)
                
                self.symbol = new_symbol
                self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
                
                self.log(f"✅ Đã tìm thấy và đăng ký coin {new_symbol} - {self.current_target_direction} - Đòn bẩy: {self.lev}x")
                
                self.status = "waiting"
                self.last_find_time = current_time
                return True
            else:
                self.log(f"❌ Không thể đăng ký coin {new_symbol} - có thể đã có bot khác trade, tìm coin khác")
                self.last_find_time = current_time
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            self.last_find_time = time.time()
            return False

    def check_position_status(self):
        if not self.symbol:
            return
            
        try:
            positions = get_positions(self.symbol, self.api_key, self.api_secret)
            if not positions:
                self._reset_position()
                return
            
            position_found = False
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    position_amt = float(pos.get('positionAmt', 0))
                    if abs(position_amt) > 0:
                        position_found = True
                        self.position_open = True
                        self.status = "open"
                        self.side = "BUY" if position_amt > 0 else "SELL"
                        self.qty = position_amt
                        self.entry = float(pos.get('entryPrice', 0))
                        break
                    else:
                        position_found = True
                        self._reset_position()
                        break
            
            if not position_found:
                self._reset_position()
                
        except Exception as e:
            if time.time() - self.last_error_log_time > 10:
                self.log(f"❌ Lỗi kiểm tra vị thế: {str(e)}")
                self.last_error_log_time = time.time()

    def _reset_position(self):
        self.position_open = False
        self.status = "searching" if not self.symbol else "waiting"
        self.side = ""
        self.qty = 0
        self.entry = 0
        self._close_attempted = False
        self._last_close_attempt = 0

    def _run(self):
        while not self._stop:
            try:
                current_time = time.time()
                
                if current_time - getattr(self, '_last_leverage_check', 0) > 60:
                    if not self.verify_leverage_and_switch():
                        if self.symbol:
                            self.ws_manager.remove_symbol(self.symbol)
                            self.coin_manager.unregister_coin(self.symbol)
                            self.symbol = None
                        time.sleep(1)
                        continue
                    self._last_leverage_check = current_time
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                              
                if not self.position_open:
                    if not self.symbol:
                        self.find_and_set_coin()
                        time.sleep(1)
                        continue
                    
                    signal = self.get_signal()
                    
                    if signal and signal != "NEUTRAL":
                        if current_time - self.last_trade_time > 3 and current_time - self.last_close_time > self.cooldown_period:
                            if self.open_position(signal):
                                self.last_trade_time = current_time
                            else:
                                if self.symbol:
                                    self.ws_manager.remove_symbol(self.symbol)
                                    self.coin_manager.unregister_coin(self.symbol)
                                    self.symbol = None
                                time.sleep(1)
                    else:
                        time.sleep(1)
                
                if self.position_open and not self._close_attempted:
                    self.check_tp_sl()
                    
                time.sleep(1)
            
            except Exception as e:
                if time.time() - self.last_error_log_time > 10:
                    self.log(f"❌ Lỗi hệ thống: {str(e)}")
                    self.last_error_log_time = time.time()
                time.sleep(1)

    def stop(self):
        self._stop = True
        if self.symbol:
            self.ws_manager.remove_symbol(self.symbol)
        if self.symbol:
            self.coin_manager.unregister_coin(self.symbol)
        if self.symbol:
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
        self.log(f"🔴 Bot dừng")

    def open_position(self, side):
        if side not in ["BUY", "SELL"]:
            self.log(f"❌ Side không hợp lệ: {side}")
            return False
            
        try:
            self.check_position_status()
            if self.position_open:
                self.log(f"⚠️ Đã có vị thế {self.side}, bỏ qua tín hiệu {side}")
                return False
    
            if self.should_be_removed:
                self.log("⚠️ Bot đã được đánh dấu xóa, không mở lệnh mới")
                return False
    
            current_leverage = self.coin_finder.get_symbol_leverage(self.symbol)
            if current_leverage < self.lev:
                self.log(f"❌ Coin {self.symbol} chỉ hỗ trợ đòn bẩy {current_leverage}x < {self.lev}x -> TÌM COIN KHÁC")
                self._cleanup_symbol()
                return False
    
            if not set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                self.log(f"❌ Không thể đặt đòn bẩy {self.lev}x -> TÌM COIN KHÁC")
                self._cleanup_symbol()
                return False
    
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                self.log("❌ Không đủ số dư")
                return False
    
            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                self.log("❌ Lỗi lấy giá -> TÌM COIN KHÁC")
                self._cleanup_symbol()
                return False
    
            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
    
            if qty < step_size:
                self.log(f"❌ Số lượng quá nhỏ: {qty} < {step_size}")
                return False
    
            self.log(f"📊 Đang đặt lệnh {side} - SL: {step_size}, Qty: {qty}, Giá: {current_price}")
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.2)
            
            result = place_order(self.symbol, side, qty, self.api_key, self.api_secret)
            
            if result and 'orderId' in result:
                executed_qty = float(result.get('executedQty', 0))
                avg_price = float(result.get('avgPrice', current_price))
                
                if executed_qty >= 0:
                    self.entry = avg_price
                    self.side = side
                    self.qty = executed_qty if side == "BUY" else -executed_qty
                    self.position_open = True
                    self.status = "open"
                    
                    message = (
                        f"✅ <b>ĐÃ MỞ VỊ THẾ {self.symbol}</b>\n"
                        f"🤖 Chiến lược: {self.strategy_name}\n"
                        f"📌 Hướng: {side}\n"
                        f"🏷️ Giá vào: {self.entry:.4f}\n"
                        f"📊 Khối lượng: {executed_qty:.4f}\n"
                        f"💵 Giá trị: {executed_qty * self.entry:.2f} USDT\n"
                        f"💰 Đòn bẩy: {self.lev}x\n"
                        f"🎯 TP: {self.tp}% | 🛡️ SL: {self.sl}%"
                    )
                    self.log(message)
                    return True
                else:
                    self.log(f"❌ Lệnh không khớp - Số lượng: {qty} -> TÌM COIN KHÁC")
                    return False
            else:
                error_msg = result.get('msg', 'Unknown error') if result else 'No response'
                self.log(f"❌ Lỗi đặt lệnh {side}: {error_msg} -> TÌM COIN KHÁC")
                
                if result and 'code' in result:
                    self.log(f"📋 Mã lỗi Binance: {result['code']} - {result.get('msg', '')}")
                
                self._cleanup_symbol()
                return False
                    
        except Exception as e:
            self.log(f"❌ Lỗi mở lệnh: {str(e)} -> TÌM COIN KHÁC")
            self._cleanup_symbol()
            return False
    
    def _cleanup_symbol(self):
        if self.symbol:
            try:
                self.ws_manager.remove_symbol(self.symbol)
                self.coin_manager.unregister_coin(self.symbol)
                self.log(f"🧹 Đã dọn dẹp symbol {self.symbol} và tìm coin mới")
            except Exception as e:
                self.log(f"⚠️ Lỗi khi dọn dẹp symbol: {str(e)}")
            
            self.symbol = None
        self.status = "searching"
        self.position_open = False
        self.side = ""
        self.qty = 0
        self.entry = 0

    def close_position(self, reason=""):
        try:
            self.check_position_status()
            
            if not self.position_open or abs(self.qty) <= 0:
                self.log(f"⚠️ Không có vị thế để đóng: {reason}")
                if self.symbol:
                    self.coin_manager.unregister_coin(self.symbol)
                return False

            current_time = time.time()
            if self._close_attempted and current_time - self._last_close_attempt < 30:
                self.log(f"⚠️ Đang thử đóng lệnh lần trước, chờ...")
                return False
            
            self._close_attempted = True
            self._last_close_attempt = current_time

            close_side = "SELL" if self.side == "BUY" else "BUY"
            close_qty = abs(self.qty)
            
            cancel_all_orders(self.symbol, self.api_key, self.api_secret)
            time.sleep(0.5)
            
            result = place_order(self.symbol, close_side, close_qty, self.api_key, self.api_secret)
            if result and 'orderId' in result:
                current_price = get_current_price(self.symbol)
                pnl = 0
                if self.entry > 0:
                    if self.side == "BUY":
                        pnl = (current_price - self.entry) * abs(self.qty)
                    else:
                        pnl = (self.entry - current_price) * abs(self.qty)
                
                message = (
                    f"⛔ <b>ĐÃ ĐÓNG VỊ THẾ {self.symbol}</b>\n"
                    f"🤖 Chiến lược: {self.strategy_name}\n"
                    f"📌 Lý do: {reason}\n"
                    f"🏷️ Giá ra: {current_price:.4f}\n"
                    f"📊 Khối lượng: {close_qty:.4f}\n"
                    f"💰 PnL: {pnl:.2f} USDT"
                )
                self.log(message)
                
                if self.symbol:
                    self.coin_manager.unregister_coin(self.symbol)
                    self.ws_manager.remove_symbol(self.symbol)
                
                self._reset_position()
                self.last_close_time = time.time()
                self.symbol = None
                self.status = "searching"
                
                time.sleep(2)
                self.check_position_status()
                
                return True
            else:
                error_msg = result.get('msg', 'Unknown error') if result else 'No response'
                self.log(f"❌ Lỗi đóng lệnh: {error_msg}")
                self._close_attempted = False
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi đóng lệnh: {str(e)}")
            self._close_attempted = False
            return False

    def check_tp_sl(self):
        if not self.position_open or self.entry <= 0 or self._close_attempted:
            return

        current_price = get_current_price(self.symbol)
        if current_price <= 0:
            return

        if self.side == "BUY":
            profit = (current_price - self.entry) * abs(self.qty)
        else:
            profit = (self.entry - current_price) * abs(self.qty)
            
        invested = self.entry * abs(self.qty) / self.lev
        if invested <= 0:
            return
            
        roi = (profit / invested) * 100

        if self.tp is not None and roi >= self.tp:
            self.close_position(f"✅ Đạt TP {self.tp}% (ROI: {roi:.2f}%)")
        elif self.sl is not None and self.sl > 0 and roi <= -self.sl:
            self.close_position(f"❌ Đạt SL {self.sl}% (ROI: {roi:.2f}%)")

# ========== BOT XU HƯỚNG TÍCH HỢP ==========
class TrendBot(BaseBot):
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, bot_id=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Trend System", 
                        config_key, bot_id)
        
        self.analyzer = TrendIndicatorSystem()
        self.last_analysis_time = 0
        self.analysis_interval = 120  # Giảm thời gian phân tích
        
    def get_signal(self):
        if not self.symbol:
            return None
            
        try:
            current_time = time.time()
            if current_time - self.last_analysis_time < self.analysis_interval:
                return None
            
            self.last_analysis_time = current_time
            
            signal = self.analyzer.analyze_symbol(self.symbol)
            
            if signal != "NEUTRAL":
                self.log(f"🎯 Nhận tín hiệu {signal} từ hệ thống xu hướng")
            
            return signal
            
        except Exception as e:
            self.log(f"❌ Lỗi phân tích xu hướng: {str(e)}")
            return None

# ========== BOT MANAGER HOÀN CHỈNH ==========
class BotManager:
    def __init__(self, api_key=None, api_secret=None, telegram_bot_token=None, telegram_chat_id=None):
        self.ws_manager = WebSocketManager()
        self.bots = {}
        self.running = True
        self.start_time = time.time()
        self.user_states = {}
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        if api_key and api_secret:
            self._verify_api_connection()
            self.log("🟢 HỆ THỐNG BOT XU HƯỚNG TÍCH HỢP ĐÃ KHỞI ĐỘNG - PHIÊN BẢN CẢI TIẾN")
            self.log("🎯 Sử dụng hệ thống chỉ báo: EMA + RSI + Volume + Support/Resistance")
            self.log("📊 Hệ thống thống kê xác suất đa điểm: Phân tích 200 nến lịch sử")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def _verify_api_connection(self):
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance API.")
        else:
            self.log(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")

    def get_position_summary(self):
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            binance_buy_count = 0
            binance_sell_count = 0
            binance_positions = []
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:
                    symbol = pos.get('symbol', 'UNKNOWN')
                    entry_price = float(pos.get('entryPrice', 0))
                    leverage = float(pos.get('leverage', 1))
                    position_value = abs(position_amt) * entry_price / leverage
                    
                    if position_amt > 0:
                        binance_buy_count += 1
                        binance_positions.append({
                            'symbol': symbol,
                            'side': 'LONG',
                            'leverage': leverage,
                            'size': abs(position_amt),
                            'entry': entry_price,
                            'value': position_value
                        })
                    else:
                        binance_sell_count += 1
                        binance_positions.append({
                            'symbol': symbol, 
                            'side': 'SHORT',
                            'leverage': leverage,
                            'size': abs(position_amt),
                            'entry': entry_price,
                            'value': position_value
                        })
        
            bot_details = []
            searching_bots = 0
            waiting_bots = 0
            trading_bots = 0
            
            for bot_id, bot in self.bots.items():
                bot_info = {
                    'bot_id': bot_id,
                    'symbol': bot.symbol or 'Đang tìm...',
                    'status': bot.status,
                    'side': bot.side,
                    'leverage': bot.lev,
                    'percent': bot.percent,
                    'tp': bot.tp,
                    'sl': bot.sl
                }
                bot_details.append(bot_info)
                
                if bot.status == "searching":
                    searching_bots += 1
                elif bot.status == "waiting":
                    waiting_bots += 1
                elif bot.status == "open":
                    trading_bots += 1
            
            summary = "📊 **THỐNG KÊ CHI TIẾT HỆ THỐNG**\n\n"
            
            balance = get_balance(self.api_key, self.api_secret)
            summary += f"💰 **SỐ DƯ**: {balance:.2f} USDT\n\n"
            
            summary += f"🤖 **BOT HỆ THỐNG**: {len(self.bots)} bots\n"
            summary += f"   🔍 Đang tìm coin: {searching_bots}\n"
            summary += f"   🟡 Đang chờ: {waiting_bots}\n" 
            summary += f"   📈 Đang trade: {trading_bots}\n\n"
            
            if bot_details:
                summary += "📋 **CHI TIẾT TỪNG BOT**:\n"
                for bot in bot_details[:8]:
                    symbol_info = bot['symbol'] if bot['symbol'] != 'Đang tìm...' else '🔍 Đang tìm'
                    status_map = {
                        "searching": "🔍 Tìm coin",
                        "waiting": "🟡 Chờ tín hiệu", 
                        "open": "🟢 Đang trade"
                    }
                    status = status_map.get(bot['status'], bot['status'])
                    
                    summary += f"   🔹 {bot['bot_id'][:15]}...\n"
                    summary += f"      📊 {symbol_info} | {status}\n"
                    summary += f"      💰 ĐB: {bot['leverage']}x | Vốn: {bot['percent']}%\n"
                    if bot['tp'] is not None and bot['sl'] is not None:
                        summary += f"      🎯 TP: {bot['tp']}% | 🛡️ SL: {bot['sl']}%\n"
                    summary += "\n"
                
                if len(bot_details) > 8:
                    summary += f"   ... và {len(bot_details) - 8} bot khác\n\n"
            
            total_binance = binance_buy_count + binance_sell_count
            if total_binance > 0:
                summary += f"💰 **TẤT CẢ VỊ THẾ BINANCE**: {total_binance} vị thế\n"
                summary += f"   🟢 LONG: {binance_buy_count}\n"
                summary += f"   🔴 SHORT: {binance_sell_count}\n\n"
                
                summary += "📈 **CHI TIẾT VỊ THẾ**:\n"
                for pos in binance_positions[:5]:
                    summary += f"   🔹 {pos['symbol']} | {pos['side']}\n"
                    summary += f"      📊 KL: {pos['size']:.4f} | Giá: {pos['entry']:.4f}\n"
                    summary += f"      💰 ĐB: {pos['leverage']}x | GT: ${pos['value']:.0f}\n\n"
                
                if len(binance_positions) > 5:
                    summary += f"   ... và {len(binance_positions) - 5} vị thế khác\n"
                        
            else:
                summary += f"💰 **TẤT CẢ VỊ THẾ BINANCE**: Không có vị thế nào\n"
                    
            return summary
                    
        except Exception as e:
            return f"❌ Lỗi thống kê: {str(e)}"

    def log(self, message):
        logger.info(f"[SYSTEM] {message}")
        if self.telegram_bot_token and self.telegram_chat_id:
            send_telegram(f"<b>SYSTEM</b>: {message}", 
                         bot_token=self.telegram_bot_token, 
                         default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        welcome = (
            "🤖 <b>BOT GIAO DỊCH FUTURES ĐA LUỒNG - PHIÊN BẢN CẢI TIẾN</b>\n\n"
            "🎯 <b>HỆ THỐNG XU HƯỚNG TÍCH HỢP NÂNG CẤP</b>\n"
            "📊 EMA + RSI + Volume + Support/Resistance\n"
            "📈 Phân tích xác suất đa điểm 200 nến lịch sử\n"
            "🎲 Random direction - Không ép hướng\n\n"
            "⚡ <b>TỐI ƯU HIỆU SUẤT & CHẤT LƯỢNG TÍN HIỆU</b>"
        )
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, bot_count=1, **kwargs):
        if sl == 0:
            sl = None
            
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance")
            return False
        
        bot_mode = kwargs.get('bot_mode', 'static')
        created_count = 0
        
        for i in range(bot_count):
            try:
                if bot_mode == 'static' and symbol:
                    bot_id = f"{symbol}_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = TrendBot
                    
                    if not bot_class:
                        continue
                    
                    bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token, 
                                  self.telegram_chat_id, bot_id=bot_id)
                    
                else:
                    bot_id = f"DYNAMIC_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = TrendBot
                    
                    if not bot_class:
                        continue
                    
                    bot = bot_class(None, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token,
                                  self.telegram_chat_id, bot_id=bot_id)
                
                bot._bot_manager = self
                self.bots[bot_id] = bot
                created_count += 1
                
            except Exception as e:
                self.log(f"❌ Lỗi tạo bot {i}: {str(e)}")
                continue
        
        if created_count > 0:
            success_msg = (
                f"✅ <b>ĐÃ TẠO {created_count}/{bot_count} BOT XU HƯỚNG</b>\n\n"
                f"🎯 Hệ thống: Trend Indicator System\n"
                f"💰 Đòn bẩy: {lev}x\n"
                f"📈 % Số dư: {percent}%\n"
                f"🎯 TP: {tp}%\n"
                f"🛡️ SL: {sl if sl is not None else 'Tắt'}%\n"
                f"🔧 Chế độ: {bot_mode}\n"
            )
            
            if bot_mode == 'static' and symbol:
                success_msg += f"🔗 Coin: {symbol}\n"
            else:
                success_msg += f"🔗 Coin: Tự động tìm kiếm\n"
            
            success_msg += f"\n🎯 <b>Mỗi bot là 1 vòng lặp độc lập</b>"
            
            self.log(success_msg)
            return True
        else:
            self.log("❌ Không thể tạo bot nào")
            return False

    def stop_bot(self, bot_id):
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]
            self.log(f"⛔ Đã dừng bot {bot_id}")
            return True
        return False

    def stop_all(self):
        self.log("⛔ Đang dừng tất cả bot...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _telegram_listener(self):
        last_update_id = 0
        
        while self.running and self.telegram_bot_token:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates?offset={last_update_id+1}&timeout=30"
                response = requests.get(url, timeout=35)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        for update in data['result']:
                            update_id = update['update_id']
                            message = update.get('message', {})
                            chat_id = str(message.get('chat', {}).get('id'))
                            text = message.get('text', '').strip()
                            
                            if chat_id != self.telegram_chat_id:
                                continue
                            
                            if update_id > last_update_id:
                                last_update_id = update_id
                            
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    logger.error("Lỗi xung đột Telegram")
                    time.sleep(60)
                else:
                    time.sleep(10)
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(10)

    def _handle_telegram_message(self, chat_id, text):
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        if current_step == 'waiting_bot_count':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    bot_count = int(text)
                    if bot_count <= 0 or bot_count > 10:
                        send_telegram("⚠️ Số lượng bot phải từ 1 đến 10. Vui lòng chọn lại:",
                                    chat_id, create_bot_count_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['bot_count'] = bot_count
                    user_state['step'] = 'waiting_bot_mode'
                    
                    send_telegram(
                        f"🤖 Số lượng bot: {bot_count}\n\n"
                        f"Chọn chế độ bot:",
                        chat_id,
                        create_bot_mode_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho số lượng bot:",
                                chat_id, create_bot_count_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_bot_mode':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
                if text == "🤖 Bot Tĩnh - Coin cụ thể":
                    user_state['bot_mode'] = 'static'
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT TĨNH</b>\n\n"
                        "🤖 Bot sẽ giao dịch coin CỐ ĐỊNH\n"
                        "📊 Bạn cần chọn coin cụ thể\n\n"
                        "Chọn chiến lược:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                else:
                    user_state['bot_mode'] = 'dynamic'
                    user_state['step'] = 'waiting_strategy'
                    send_telegram(
                        "🎯 <b>ĐÃ CHỌN: BOT ĐỘNG</b>\n\n"
                        f"🤖 Hệ thống sẽ tạo <b>{user_state.get('bot_count', 1)} bot độc lập</b>\n"
                        f"🔄 Mỗi bot tự tìm coin & trade độc lập\n"
                        f"🎯 Tự reset hoàn toàn sau mỗi lệnh\n"
                        f"📊 Mỗi bot là 1 vòng lặp hoàn chỉnh\n\n"
                        "Chọn chiến lược:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )

        elif current_step == 'waiting_strategy':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text in ["📊 Trend System"]:
                
                strategy_map = {
                    "📊 Trend System": "Trend-System"
                }
                
                strategy = strategy_map[text]
                user_state['strategy'] = strategy
                user_state['step'] = 'waiting_exit_strategy'
                
                strategy_descriptions = {
                    "Trend-System": "Hệ thống xu hướng tích hợp: EMA + RSI + Volume + Support/Resistance"
                }
                
                description = strategy_descriptions.get(strategy, "")
                bot_count = user_state.get('bot_count', 1)
                
                send_telegram(
                    f"🎯 <b>ĐÃ CHỌN: {strategy}</b>\n"
                    f"🤖 Số lượng: {bot_count} bot độc lập\n\n"
                    f"{description}\n\n"
                    f"Chọn chiến lược thoát lệnh:",
                    chat_id,
                    create_exit_strategy_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )

        elif current_step == 'waiting_exit_strategy':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif text == "🎯 Chỉ TP/SL cố định":
                user_state['exit_strategy'] = 'traditional'
                self._continue_bot_creation(chat_id, user_state)

        elif current_step == 'waiting_symbol':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                user_state['symbol'] = text
                user_state['step'] = 'waiting_leverage'
                send_telegram(
                    f"🔗 Coin: {text}\n\n"
                    f"Chọn đòn bẩy:",
                    chat_id,
                    create_leverage_keyboard(),
                    self.telegram_bot_token, self.telegram_chat_id
                )

        elif current_step == 'waiting_leverage':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                if text.endswith('x'):
                    lev_text = text[:-1]
                else:
                    lev_text = text

                try:
                    leverage = int(lev_text)
                    if leverage <= 0 or leverage > 100:
                        send_telegram("⚠️ Đòn bẩy phải từ 1 đến 100. Vui lòng chọn lại:",
                                    chat_id, create_leverage_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    warning_msg = ""
                    if leverage > 50:
                        warning_msg = f"\n\n⚠️ <b>CẢNH BÁO RỦI RO CAO</b>\nĐòn bẩy {leverage}x rất nguy hiểm!"
                    elif leverage > 20:
                        warning_msg = f"\n\n⚠️ <b>CẢNH BÁO RỦI RO</b>\nĐòn bẩy {leverage}x có rủi ro cao!"

                    user_state['leverage'] = leverage
                    user_state['step'] = 'waiting_percent'
                    
                    balance = get_balance(self.api_key, self.api_secret)
                    balance_info = f"\n💰 Số dư hiện có: {balance:.2f} USDT" if balance else ""
                    
                    send_telegram(
                        f"💰 Đòn bẩy: {leverage}x{balance_info}{warning_msg}\n\n"
                        f"Chọn % số dư cho mỗi lệnh:",
                        chat_id,
                        create_percent_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho đòn bẩy:",
                                chat_id, create_leverage_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_percent':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    percent = float(text)
                    if percent <= 0 or percent > 100:
                        send_telegram("⚠️ % số dư phải từ 0.1 đến 100. Vui lòng chọn lại:",
                                    chat_id, create_percent_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['percent'] = percent
                    user_state['step'] = 'waiting_tp'
                    
                    balance = get_balance(self.api_key, self.api_secret)
                    actual_amount = balance * (percent / 100) if balance else 0
                    
                    send_telegram(
                        f"📊 % Số dư: {percent}%\n"
                        f"💵 Số tiền mỗi lệnh: ~{actual_amount:.2f} USDT\n\n"
                        f"Chọn Take Profit (%):",
                        chat_id,
                        create_tp_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho % số dư:",
                                chat_id, create_percent_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_tp':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    tp = float(text)
                    if tp <= 0:
                        send_telegram("⚠️ Take Profit phải lớn hơn 0. Vui lòng chọn lại:",
                                    chat_id, create_tp_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['tp'] = tp
                    user_state['step'] = 'waiting_sl'
                    
                    send_telegram(
                        f"🎯 Take Profit: {tp}%\n\n"
                        f"Chọn Stop Loss (%):",
                        chat_id,
                        create_sl_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Take Profit:",
                                chat_id, create_tp_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif current_step == 'waiting_sl':
            if text == '❌ Hủy bỏ':
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy thêm bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                try:
                    sl = float(text)
                    if sl < 0:
                        send_telegram("⚠️ Stop Loss phải lớn hơn hoặc bằng 0. Vui lòng chọn lại:",
                                    chat_id, create_sl_keyboard(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                        return

                    user_state['sl'] = sl
                    
                    strategy = user_state.get('strategy')
                    bot_mode = user_state.get('bot_mode', 'static')
                    leverage = user_state.get('leverage')
                    percent = user_state.get('percent')
                    tp = user_state.get('tp')
                    sl = user_state.get('sl')
                    symbol = user_state.get('symbol')
                    bot_count = user_state.get('bot_count', 1)
                    
                    success = self.add_bot(
                        symbol=symbol,
                        lev=leverage,
                        percent=percent,
                        tp=tp,
                        sl=sl,
                        strategy_type=strategy,
                        bot_mode=bot_mode,
                        bot_count=bot_count
                    )
                    
                    if success:
                        success_msg = (
                            f"✅ <b>ĐÃ TẠO {bot_count} BOT THÀNH CÔNG</b>\n\n"
                            f"🤖 Chiến lược: {strategy}\n"
                            f"🔧 Chế độ: {bot_mode}\n"
                            f"🔢 Số lượng: {bot_count} bot độc lập\n"
                            f"💰 Đòn bẩy: {leverage}x\n"
                            f"📊 % Số dư: {percent}%\n"
                            f"🎯 TP: {tp}%\n"
                            f"🛡️ SL: {sl}%"
                        )
                        if bot_mode == 'static' and symbol:
                            success_msg += f"\n🔗 Coin: {symbol}"
                        
                        success_msg += f"\n\n🎯 <b>Mỗi bot là 1 vòng lặp độc lập</b>\n"
                        success_msg += f"🔄 <b>Tự reset hoàn toàn sau mỗi lệnh</b>\n"
                        success_msg += f"📊 <b>Tự tìm coin & trade độc lập</b>"
                        
                        send_telegram(success_msg, chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    else:
                        send_telegram("❌ Có lỗi khi tạo bot. Vui lòng thử lại.",
                                    chat_id, create_main_menu(),
                                    self.telegram_bot_token, self.telegram_chat_id)
                    
                    self.user_states[chat_id] = {}
                    
                except ValueError:
                    send_telegram("⚠️ Vui lòng nhập số hợp lệ cho Stop Loss:",
                                chat_id, create_sl_keyboard(),
                                self.telegram_bot_token, self.telegram_chat_id)

        elif text == "➕ Thêm Bot":
            self.user_states[chat_id] = {'step': 'waiting_bot_count'}
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                return
            
            send_telegram(
                f"🎯 <b>CHỌN SỐ LƯỢNG BOT ĐỘC LẬP</b>\n\n"
                f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
                f"Chọn số lượng bot độc lập bạn muốn tạo:\n"
                f"<i>Mỗi bot sẽ tự tìm coin & trade độc lập</i>",
                chat_id,
                create_bot_count_keyboard(),
                self.telegram_bot_token, self.telegram_chat_id
            )
        
        elif text == "📊 Danh sách Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "🤖 <b>DANH SÁCH BOT ĐỘC LẬP ĐANG CHẠY</b>\n\n"
                
                active_bots = 0
                searching_bots = 0
                trading_bots = 0
                
                for bot_id, bot in self.bots.items():
                    if bot.status == "searching":
                        status = "🔍 Đang tìm coin"
                        searching_bots += 1
                    elif bot.status == "waiting":
                        status = "🟡 Chờ tín hiệu"
                        trading_bots += 1
                    elif bot.status == "open":
                        status = "🟢 Đang trade"
                        trading_bots += 1
                    else:
                        status = "⚪ Unknown"
                    
                    symbol_info = bot.symbol if bot.symbol else "Đang tìm..."
                    message += f"🔹 {bot_id}\n"
                    message += f"   📊 {symbol_info} | {status}\n"
                    message += f"   💰 ĐB: {bot.lev}x | Vốn: {bot.percent}%\n\n"
                
                message += f"📈 Tổng số: {len(self.bots)} bot\n"
                message += f"🔍 Đang tìm coin: {searching_bots} bot\n"
                message += f"📊 Đang trade: {trading_bots} bot"
                
                send_telegram(message, chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "📊 Thống kê":
            summary = self.get_position_summary()
            send_telegram(summary, chat_id,
                         bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⛔ Dừng Bot":
            if not self.bots:
                send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            else:
                message = "⛔ <b>CHỌN BOT ĐỂ DỪNG</b>\n\n"
                keyboard = []
                row = []
                
                for i, bot_id in enumerate(self.bots.keys()):
                    bot = self.bots[bot_id]
                    symbol_info = bot.symbol if bot.symbol else "No Coin"
                    message += f"🔹 {bot_id} - {symbol_info}\n"
                    row.append({"text": f"⛔ {bot_id}"})
                    if len(row) == 1 or i == len(self.bots) - 1:
                        keyboard.append(row)
                        row = []
                
                keyboard.append([{"text": "⛔ DỪNG TẤT CẢ"}])
                keyboard.append([{"text": "❌ Hủy bỏ"}])
                
                send_telegram(
                    message, 
                    chat_id, 
                    {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                    self.telegram_bot_token, self.telegram_chat_id
                )
        
        elif text.startswith("⛔ "):
            bot_id = text.replace("⛔ ", "").strip()
            if bot_id == "DỪNG TẤT CẢ":
                self.stop_all()
                send_telegram("⛔ Đã dừng tất cả bot", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            elif self.stop_bot(bot_id):
                send_telegram(f"⛔ Đã dừng bot {bot_id}", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
            else:
                send_telegram(f"⚠️ Không tìm thấy bot {bot_id}", chat_id, create_main_menu(),
                            self.telegram_bot_token, self.telegram_chat_id)
        
        elif text == "💰 Số dư":
            try:
                balance = get_balance(self.api_key, self.api_secret)
                if balance is None:
                    send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                                bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                else:
                    send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id,
                                bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy số dư: {str(e)}", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "📈 Vị thế":
            try:
                positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
                if not positions:
                    send_telegram("📭 Không có vị thế nào đang mở", chat_id,
                                bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
                    return
                
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
            except Exception as e:
                send_telegram(f"⚠️ Lỗi lấy vị thế: {str(e)}", chat_id,
                            bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "🎯 Chiến lược":
            strategy_info = (
                "🎯 <b>HỆ THỐNG XU HƯỚNG TÍCH HỢP NÂNG CẤP - PHIÊN BẢN CẢI TIẾN</b>\n\n"
                
                "📊 <b>Chỉ báo sử dụng:</b>\n"
                "• EMA (9, 21, 50) - Trọng số 40%\n"
                "• RSI (14) + Volume - Trọng số 35%\n"  
                "• Support/Resistance - Trọng số 15%\n"
                "• Market Structure - Trọng số 10%\n\n"
                
                "📈 <b>Hệ thống thống kê xác suất đa điểm:</b>\n"
                "• Phân tích 200 nến lịch sử (tối ưu)\n"
                "• 13 điểm RSI, 10 điều kiện EMA, 9 mức Volume\n"
                "• Tính xác suất thắng cho từng điểm chỉ báo\n"
                "• Tính kỳ vọng & phương sai\n"
                "• Đề xuất hướng tối ưu\n\n"
                
                "🎲 <b>Random Direction:</b>\n"
                "• Mỗi bot chọn hướng random 50/50\n"
                "• Không ép hướng ngược chiều\n"
                "• Đảm bảo đa dạng hóa tự nhiên\n\n"
                
                "⚡ <b>Tối ưu hiệu suất:</b>\n"
                "• Giảm yêu cầu dữ liệu tối thiểu\n"
                "• Tăng tốc độ phân tích\n"
                "• Cải thiện chất lượng tín hiệu\n"
                "• Giữ nguyên ngưỡng chất lượng 0.6"
            )
            send_telegram(strategy_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text == "⚙️ Cấu hình":
            balance = get_balance(self.api_key, self.api_secret)
            api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
            
            searching_bots = sum(1 for bot in self.bots.values() if bot.status == "searching")
            trading_bots = sum(1 for bot in self.bots.values() if bot.status in ["waiting", "open"])
            
            config_info = (
                "⚙️ <b>CẤU HÌNH HỆ THỐNG ĐA LUỒNG NÂNG CẤP</b>\n\n"
                f"🔑 Binance API: {api_status}\n"
                f"🤖 Tổng số bot: {len(self.bots)}\n"
                f"🔍 Đang tìm coin: {searching_bots} bot\n"
                f"📊 Đang trade: {trading_bots} bot\n"
                f"🌐 WebSocket: {len(self.ws_manager.connections)} kết nối\n\n"
                f"🎯 <b>Hệ thống xác suất đa điểm đã kích hoạt</b>\n"
                f"⚡ <b>Phiên bản tối ưu hiệu suất</b>"
            )
            send_telegram(config_info, chat_id,
                        bot_token=self.telegram_bot_token, default_chat_id=self.telegram_chat_id)
        
        elif text:
            self.send_main_menu(chat_id)

    def _continue_bot_creation(self, chat_id, user_state):
        strategy = user_state.get('strategy')
        bot_mode = user_state.get('bot_mode', 'static')
        bot_count = user_state.get('bot_count', 1)
        
        if bot_mode == 'static':
            user_state['step'] = 'waiting_symbol'
            send_telegram(
                f"🎯 <b>BOT TĨNH: {strategy}</b>\n"
                f"🤖 Số lượng: {bot_count} bot độc lập\n\n"
                f"🤖 Mỗi bot sẽ trade coin CỐ ĐỊNH\n\n"
                f"Chọn cặp coin:",
                chat_id,
                create_symbols_keyboard(strategy),
                self.telegram_bot_token, self.telegram_chat_id
            )
        else:
            user_state['step'] = 'waiting_leverage'
            send_telegram(
                f"🎯 <b>BOT ĐỘNG ĐA LUỒNG</b>\n"
                f"🤖 Số lượng: {bot_count} bot độc lập\n\n"
                f"🤖 Mỗi bot sẽ tự tìm coin & trade độc lập\n"
                f"🔄 Tự reset hoàn toàn sau mỗi lệnh\n"
                f"📊 Mỗi bot là 1 vòng lặp hoàn chỉnh\n"
                f"⚖️ Tự cân bằng với các bot khác\n\n"
                f"Chọn đòn bẩy:",
                chat_id,
                create_leverage_keyboard(strategy),
                self.telegram_bot_token, self.telegram_chat_id
            )

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()

