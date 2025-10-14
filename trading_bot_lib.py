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

# ========== CẤU HÌNH LOGGING TỐI ƯU ==========
def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,  # CHỈ HIỂN THỊ LỖI VÀ CẢNH BÁO
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
        return
    
    chat_id = chat_id or default_chat_id
    if not chat_id:
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

# ========== MENU TELEGRAM ==========
def create_cancel_keyboard():
    return {
        "keyboard": [[{"text": "❌ Hủy bỏ"}]],
        "resize_keyboard": True,
        "one_time_keyboard": True
    }

def create_strategy_keyboard():
    return {
        "keyboard": [
            [{"text": "⏰ Trend System"}],
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

# ========== HỆ THỐNG THỐNG KÊ XÁC SUẤT & KỲ VỌNG ==========
class ProbabilityAnalyzer:
    """PHÂN TÍCH XÁC SUẤT THẮNG CỦA CÁC CHỈ BÁO"""
    
    def __init__(self, lookback=200, evaluation_period=20):
        self.lookback = lookback
        self.evaluation_period = evaluation_period
        self.history_data = {}
        self.probability_stats = {
            'rsi': {
                'ranges': [(0, 20), (20, 45), (45, 55), (55, 80), (80, 100)],
                'correct_predictions': {i: 0 for i in range(5)},
                'total_predictions': {i: 0 for i in range(5)},
                'expectations': {i: 0.0 for i in range(5)},
                'variances': {i: 0.0 for i in range(5)}
            },
            'ema': {
                'conditions': ['above_fast', 'below_fast', 'above_slow', 'below_slow', 
                             'above_trend', 'below_trend', 'golden_cross', 'death_cross'],
                'correct_predictions': {cond: 0 for cond in ['above_fast', 'below_fast', 'above_slow', 'below_slow', 
                                                           'above_trend', 'below_trend', 'golden_cross', 'death_cross']},
                'total_predictions': {cond: 0 for cond in ['above_fast', 'below_fast', 'above_slow', 'below_slow', 
                                                         'above_trend', 'below_trend', 'golden_cross', 'death_cross']},
                'expectations': {cond: 0.0 for cond in ['above_fast', 'below_fast', 'above_slow', 'below_slow', 
                                                      'above_trend', 'below_trend', 'golden_cross', 'death_cross']},
                'variances': {cond: 0.0 for cond in ['above_fast', 'below_fast', 'above_slow', 'below_slow', 
                                                   'above_trend', 'below_trend', 'golden_cross', 'death_cross']}
            },
            'volume': {
                'ratios': [1.2, 1.5, 1.8, 2.0],
                'correct_predictions': {i: 0 for i in range(4)},
                'total_predictions': {i: 0 for i in range(4)},
                'expectations': {i: 0.0 for i in range(4)},
                'variances': {i: 0.0 for i in range(4)}
            }
        }
        self.last_update_time = 0
        self.update_interval = 3600
    
    def get_rsi_range_index(self, rsi_value):
        for i, (low, high) in enumerate(self.probability_stats['rsi']['ranges']):
            if low <= rsi_value < high:
                return i
        return 2
    
    def get_volume_ratio_index(self, volume_ratio):
        for i, ratio in enumerate(self.probability_stats['volume']['ratios']):
            if volume_ratio >= ratio:
                return i
        return 0
    
    def analyze_historical_performance(self, symbol):
        try:
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return self.probability_stats
            
            klines = self.get_historical_klines(symbol, '15m', self.lookback + self.evaluation_period)
            if not klines or len(klines) < self.lookback + 10:
                return self.probability_stats
            
            self._reset_stats()
            analyzer = TrendIndicatorSystem()
            
            for i in range(self.evaluation_period, len(klines) - self.evaluation_period):
                try:
                    current_data = klines[i]
                    current_close = float(current_data[4])
                    future_data = klines[i + self.evaluation_period]
                    future_close = float(future_data[4])
                    
                    price_change = (future_close - current_close) / current_close * 100
                    is_price_up = price_change > 0
                    
                    historical_klines = klines[:i+1]
                    closes = [float(candle[4]) for candle in historical_klines]
                    
                    if len(closes) < 50:
                        continue
                    
                    rsi = analyzer.calculate_rsi(closes, analyzer.rsi_period)
                    rsi_index = self.get_rsi_range_index(rsi)
                    
                    rsi_prediction = self._get_rsi_prediction(rsi)
                    if rsi_prediction is not None:
                        is_correct = (rsi_prediction == "BUY" and is_price_up) or (rsi_prediction == "SELL" and not is_price_up)
                        self._update_rsi_stats(rsi_index, is_correct, price_change)
                    
                    ema_fast = analyzer.calculate_ema(closes, analyzer.ema_fast)
                    ema_slow = analyzer.calculate_ema(closes, analyzer.ema_slow)
                    ema_trend = analyzer.calculate_ema(closes, analyzer.ema_trend)
                    
                    self._update_ema_stats(closes, ema_fast, ema_slow, ema_trend, is_price_up, price_change, analyzer)
                    
                    if len(historical_klines) >= 20:
                        volumes = [float(candle[5]) for candle in historical_klines]
                        current_volume = volumes[-1] if volumes else 0
                        avg_volume = np.mean(volumes[-20:-1]) if len(volumes) >= 20 else 1.0
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                        
                        volume_index = self.get_volume_ratio_index(volume_ratio)
                        volume_prediction = self._get_volume_prediction(volume_ratio, current_close, ema_fast)
                        
                        if volume_prediction is not None:
                            is_correct = (volume_prediction == "BUY" and is_price_up) or (volume_prediction == "SELL" and not is_price_up)
                            self._update_volume_stats(volume_index, is_correct, price_change)
                            
                except Exception:
                    continue
            
            self._calculate_final_stats()
            self.last_update_time = current_time
            
            return self.probability_stats
            
        except Exception as e:
            logger.error(f"Lỗi phân tích hiệu suất lịch sử: {str(e)}")
            return self.probability_stats
    
    def _get_rsi_prediction(self, rsi):
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        return None
    
    def _get_volume_prediction(self, volume_ratio, price, ema_fast):
        if volume_ratio > 1.2:
            if price > ema_fast:
                return "BUY"
            else:
                return "SELL"
        return None
    
    def _update_rsi_stats(self, index, is_correct, price_change):
        self.probability_stats['rsi']['total_predictions'][index] += 1
        if is_correct:
            self.probability_stats['rsi']['correct_predictions'][index] += 1
        
        current_expectation = self.probability_stats['rsi']['expectations'][index]
        self.probability_stats['rsi']['expectations'][index] = current_expectation + price_change
    
    def _update_ema_stats(self, closes, ema_fast, ema_slow, ema_trend, is_price_up, price_change, analyzer):
        current_price = closes[-1] if closes else 0
        
        conditions = {
            'above_fast': current_price > ema_fast,
            'below_fast': current_price < ema_fast,
            'above_slow': current_price > ema_slow,
            'below_slow': current_price < ema_slow,
            'above_trend': current_price > ema_trend,
            'below_trend': current_price < ema_trend,
        }
        
        if len(closes) >= 2:
            prev_ema_fast = analyzer.calculate_ema(closes[:-1], analyzer.ema_fast)
            prev_ema_slow = analyzer.calculate_ema(closes[:-1], analyzer.ema_slow)
            conditions['golden_cross'] = ema_fast > ema_slow and prev_ema_fast <= prev_ema_slow
            conditions['death_cross'] = ema_fast < ema_slow and prev_ema_fast >= prev_ema_slow
        
        for condition, is_true in conditions.items():
            if is_true:
                self.probability_stats['ema']['total_predictions'][condition] += 1
                
                if condition in ['above_fast', 'above_slow', 'above_trend', 'golden_cross']:
                    is_correct = is_price_up
                else:
                    is_correct = not is_price_up
                
                if is_correct:
                    self.probability_stats['ema']['correct_predictions'][condition] += 1
                
                current_expectation = self.probability_stats['ema']['expectations'][condition]
                self.probability_stats['ema']['expectations'][condition] = current_expectation + price_change
    
    def _update_volume_stats(self, index, is_correct, price_change):
        self.probability_stats['volume']['total_predictions'][index] += 1
        if is_correct:
            self.probability_stats['volume']['correct_predictions'][index] += 1
        
        current_expectation = self.probability_stats['volume']['expectations'][index]
        self.probability_stats['volume']['expectations'][index] = current_expectation + price_change
    
    def _calculate_final_stats(self):
        for i in range(5):
            total = self.probability_stats['rsi']['total_predictions'][i]
            if total > 0:
                self.probability_stats['rsi']['expectations'][i] /= total
                self.probability_stats['rsi']['variances'][i] = abs(self.probability_stats['rsi']['expectations'][i]) * 0.1
        
        for condition in self.probability_stats['ema']['conditions']:
            total = self.probability_stats['ema']['total_predictions'][condition]
            if total > 0:
                self.probability_stats['ema']['expectations'][condition] /= total
                self.probability_stats['ema']['variances'][condition] = abs(self.probability_stats['ema']['expectations'][condition]) * 0.1
        
        for i in range(4):
            total = self.probability_stats['volume']['total_predictions'][i]
            if total > 0:
                self.probability_stats['volume']['expectations'][i] /= total
                self.probability_stats['volume']['variances'][i] = abs(self.probability_stats['volume']['expectations'][i]) * 0.1
    
    def _reset_stats(self):
        for indicator in self.probability_stats:
            if 'correct_predictions' in self.probability_stats[indicator]:
                for key in self.probability_stats[indicator]['correct_predictions']:
                    self.probability_stats[indicator]['correct_predictions'][key] = 0
                    self.probability_stats[indicator]['total_predictions'][key] = 0
                    self.probability_stats[indicator]['expectations'][key] = 0.0
                    self.probability_stats[indicator]['variances'][key] = 0.0
    
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
    
    def get_recommended_direction(self, symbol, current_analysis):
        try:
            stats = self.analyze_historical_performance(symbol)
            if not stats:
                return "NEUTRAL"
            
            current_rsi = current_analysis.get('rsi', 50)
            current_volume_ratio = current_analysis.get('volume_ratio', 1.0)
            current_ema_condition = current_analysis.get('ema_condition', 'neutral')
            
            buy_score = 0
            sell_score = 0
            
            rsi_index = self.get_rsi_range_index(current_rsi)
            rsi_prob = self._get_probability(stats['rsi'], rsi_index)
            rsi_expectation = stats['rsi']['expectations'][rsi_index]
            rsi_variance = stats['rsi']['variances'][rsi_index]
            
            if current_rsi < 45:
                buy_score += rsi_expectation * rsi_prob * (1 - rsi_variance)
            elif current_rsi > 55:
                sell_score += abs(rsi_expectation) * rsi_prob * (1 - rsi_variance)
            
            ema_conditions = []
            if current_ema_condition == 'bullish':
                ema_conditions = ['above_fast', 'above_slow', 'above_trend', 'golden_cross']
            elif current_ema_condition == 'bearish':
                ema_conditions = ['below_fast', 'below_slow', 'below_trend', 'death_cross']
            
            for condition in ema_conditions:
                ema_prob = self._get_probability(stats['ema'], condition)
                ema_expectation = stats['ema']['expectations'][condition]
                ema_variance = stats['ema']['variances'][condition]
                
                if 'above' in condition or 'golden' in condition:
                    buy_score += ema_expectation * ema_prob * (1 - ema_variance)
                else:
                    sell_score += abs(ema_expectation) * ema_prob * (1 - ema_variance)
            
            volume_index = self.get_volume_ratio_index(current_volume_ratio)
            volume_prob = self._get_probability(stats['volume'], volume_index)
            volume_expectation = stats['volume']['expectations'][volume_index]
            volume_variance = stats['volume']['variances'][volume_index]
            
            if current_volume_ratio > 1.2:
                if buy_score > sell_score:
                    buy_score += volume_expectation * volume_prob * (1 - volume_variance)
                else:
                    sell_score += abs(volume_expectation) * volume_prob * (1 - volume_variance)
            
            if buy_score > sell_score and buy_score > 0.1:
                return "BUY"
            elif sell_score > buy_score and sell_score > 0.1:
                return "SELL"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Lỗi đề xuất hướng: {str(e)}")
            return "NEUTRAL"
    
    def _get_probability(self, indicator_stats, key):
        total = indicator_stats['total_predictions'].get(key, 0)
        if total == 0:
            return 0.5
        correct = indicator_stats['correct_predictions'].get(key, 0)
        return correct / total

    def get_probability_report(self, symbol):
        try:
            stats = self.analyze_historical_performance(symbol)
            if not stats:
                return "❌ Không có dữ liệu thống kê"
            
            report = f"📊 <b>BÁO CÁO XÁC SUẤT - {symbol}</b>\n\n"
            
            report += "📈 <b>RSI PROBABILITIES:</b>\n"
            for i, (low, high) in enumerate(stats['rsi']['ranges']):
                prob = self._get_probability(stats['rsi'], i)
                exp = stats['rsi']['expectations'][i]
                var = stats['rsi']['variances'][i]
                report += f"   {low}-{high}: {prob:.1%} (E:{exp:.2f}%, V:{var:.3f})\n"
            
            report += "\n📉 <b>EMA PROBABILITIES:</b>\n"
            for condition in stats['ema']['conditions']:
                prob = self._get_probability(stats['ema'], condition)
                exp = stats['ema']['expectations'][condition]
                var = stats['ema']['variances'][condition]
                report += f"   {condition}: {prob:.1%} (E:{exp:.2f}%, V:{var:.3f})\n"
            
            report += "\n📊 <b>VOLUME PROBABILITIES:</b>\n"
            for i, ratio in enumerate(stats['volume']['ratios']):
                prob = self._get_probability(stats['volume'], i)
                exp = stats['volume']['expectations'][i]
                var = stats['volume']['variances'][i]
                report += f"   >{ratio}x: {prob:.1%} (E:{exp:.2f}%, V:{var:.3f})\n"
            
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
        self.lookback = 100
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
                'limit': 20
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
                'limit': 30
            }
            data = binance_api_request(url, params=params)
            if not data or len(data) < 20:
                return 0, 0
                
            highs = [float(candle[2]) for candle in data]
            lows = [float(candle[3]) for candle in data]
            
            resistance = max(highs[-20:])
            support = min(lows[-20:])
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Lỗi lấy S/R {symbol}: {str(e)}")
            return 0, 0
    
    def analyze_symbol(self, symbol):
        try:
            klines = self.get_klines(symbol, '15m', self.lookback)
            if not klines or len(klines) < 50:
                return "NEUTRAL"
            
            closes = [float(candle[4]) for candle in klines]
            current_price = closes[-1]
            
            ema_fast = self.calculate_ema(closes, self.ema_fast)
            ema_slow = self.calculate_ema(closes, self.ema_slow)
            ema_trend = self.calculate_ema(closes, self.ema_trend)
            
            ema_signal = "NEUTRAL"
            if current_price > ema_fast > ema_slow > ema_trend:
                ema_signal = "BUY"
                ema_strength = 1.0
            elif current_price < ema_fast < ema_slow < ema_trend:
                ema_signal = "SELL" 
                ema_strength = 1.0
            elif current_price > ema_fast > ema_slow:
                ema_signal = "BUY"
                ema_strength = 0.7
            elif current_price < ema_fast < ema_slow:
                ema_signal = "SELL"
                ema_strength = 0.7
            else:
                ema_strength = 0.3
            
            rsi = self.calculate_rsi(closes, self.rsi_period)
            volume_ratio = self.get_volume_data(symbol)
            
            rsi_signal = "NEUTRAL"
            rsi_strength = 0
            
            if rsi < 30 and volume_ratio > 1.2:
                rsi_signal = "BUY"
                rsi_strength = min((30 - rsi) / 30 * volume_ratio, 1.0)
            elif rsi > 70 and volume_ratio > 1.2:
                rsi_signal = "SELL" 
                rsi_strength = min((rsi - 70) / 30 * volume_ratio, 1.0)
            elif 40 < rsi < 60:
                rsi_strength = 0.2
            
            support, resistance = self.get_support_resistance(symbol)
            sr_signal = "NEUTRAL"
            sr_strength = 0
            
            if support > 0 and resistance > 0:
                distance_to_resistance = (resistance - current_price) / current_price
                distance_to_support = (current_price - support) / current_price
                
                if current_price > resistance and volume_ratio > 1.3:
                    sr_signal = "BUY"
                    sr_strength = min(volume_ratio * 0.8, 1.0)
                elif current_price < support and volume_ratio > 1.3:
                    sr_signal = "SELL"
                    sr_strength = min(volume_ratio * 0.8, 1.0)
                elif distance_to_resistance < 0.01:
                    sr_signal = "SELL"
                    sr_strength = 0.6
                elif distance_to_support < 0.01:
                    sr_signal = "BUY" 
                    sr_strength = 0.6
            
            structure_signal = self.analyze_market_structure(closes)
            structure_strength = 0.5 if structure_signal != "NEUTRAL" else 0.2
            
            current_analysis = {
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'ema_condition': 'bullish' if ema_signal == "BUY" else 'bearish' if ema_signal == "SELL" else 'neutral'
            }
            
            probability_signal = self.probability_analyzer.get_recommended_direction(symbol, current_analysis)
            probability_strength = 0.8 if probability_signal != "NEUTRAL" else 0.2
            
            signals = {
                "BUY": 0,
                "SELL": 0, 
                "NEUTRAL": 0
            }
            
            weights = {
                "EMA": 0.3,
                "RSI_VOLUME": 0.25, 
                "SR": 0.15,
                "STRUCTURE": 0.1,
                "PROBABILITY": 0.2
            }
            
            if ema_signal != "NEUTRAL":
                signals[ema_signal] += weights["EMA"] * ema_strength
                
            if rsi_signal != "NEUTRAL": 
                signals[rsi_signal] += weights["RSI_VOLUME"] * rsi_strength
                
            if sr_signal != "NEUTRAL":
                signals[sr_signal] += weights["SR"] * sr_strength
                
            if structure_signal != "NEUTRAL":
                signals[structure_signal] += weights["STRUCTURE"] * structure_strength
                
            if probability_signal != "NEUTRAL":
                signals[probability_signal] += weights["PROBABILITY"] * probability_strength
            
            max_signal = max(signals, key=signals.get)
            confidence = signals[max_signal]
            
            if confidence >= 0.4:
                return max_signal
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Lỗi phân tích {symbol}: {str(e)}")
            return "NEUTRAL"
    
    def analyze_market_structure(self, prices):
        if len(prices) < 10:
            return "NEUTRAL"
            
        recent_highs = prices[-5:]
        recent_lows = prices[-5:]
        prev_highs = prices[-10:-5] 
        prev_lows = prices[-10:-5]
        
        if (max(recent_highs) > max(prev_highs) and 
            min(recent_lows) > min(prev_lows)):
            return "BUY"
        elif (max(recent_highs) < max(prev_highs) and 
              min(recent_lows) < min(prev_lows)):
            return "SELL"
        return "NEUTRAL"
    
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
        return self.probability_analyzer.get_probability_report(symbol)

# ========== SMART COIN FINDER - ĐÃ TỐI ƯU ==========
class SmartCoinFinder:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.analyzer = TrendIndicatorSystem()
        self.leverage_cache = {}
        self.cache_timeout = 300
        self.batch_size = 20  # Tìm theo lô 20 coin
    
    def get_symbol_leverage_info(self, symbol):
        try:
            current_time = time.time()
            if symbol in self.leverage_cache:
                cached_data, timestamp = self.leverage_cache[symbol]
                if current_time - timestamp < self.cache_timeout:
                    return cached_data
            
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            data = binance_api_request(url)
            if not data:
                return None
                
            for s in data['symbols']:
                if s['symbol'] == symbol.upper():
                    for f in s['filters']:
                        if f['filterType'] == 'LEVERAGE':
                            max_leverage = int(f.get('maxLeverage', 20))
                            self.leverage_cache[symbol] = (max_leverage, current_time)
                            return max_leverage
                    
                    self.leverage_cache[symbol] = (20, current_time)
                    return 20
                    
            return None
            
        except Exception as e:
            logger.error(f"Lỗi lấy đòn bẩy {symbol}: {str(e)}")
            return None
    
    def is_leverage_supported(self, symbol, target_leverage):
        try:
            max_leverage = self.get_symbol_leverage_info(symbol)
            if max_leverage is None:
                return False
                
            return max_leverage >= target_leverage
            
        except Exception as e:
            logger.error(f"Lỗi kiểm tra đòn bẩy {symbol}: {str(e)}")
            return False
    
    def filter_symbols_by_leverage(self, symbols, target_leverage):
        if not symbols:
            return []
            
        valid_symbols = []
        
        for symbol in symbols:
            if self.is_leverage_supported(symbol, target_leverage):
                valid_symbols.append(symbol)
        
        return valid_symbols

    def get_market_position_balance(self):
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            buy_count = 0
            sell_count = 0
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt > 0:
                    buy_count += 1
                elif position_amt < 0:
                    sell_count += 1
            
            if buy_count > sell_count:
                return "SELL"
            elif sell_count > buy_count:
                return "BUY"
            else:
                return "BUY" if random.random() > 0.5 else "SELL"
                    
        except Exception as e:
            logger.error(f"Lỗi kiểm tra vị thế: {str(e)}")
            return "BUY" if random.random() > 0.5 else "SELL"
    
    def find_coin_by_direction(self, target_direction, excluded_symbols=None, target_leverage=20):
        try:
            if excluded_symbols is None:
                excluded_symbols = set()
            
            all_symbols = get_all_usdt_pairs(limit=200)  # Giảm xuống 200 coin để tối ưu
            if not all_symbols:
                return None
            
            valid_symbols = self.filter_symbols_by_leverage(all_symbols, target_leverage)
            if not valid_symbols:
                return None
            
            random.shuffle(valid_symbols)
            
            coin_manager = CoinManager()
            managed_coins = coin_manager.get_managed_coins()
            excluded_symbols.update(managed_coins.keys())
            
            # TÌM THEO LÔ 20 COIN - NẾU KHÔNG CÓ THÌ CHUYỂN LÔ TIẾP THEO
            batch_count = 0
            total_batches = (len(valid_symbols) + self.batch_size - 1) // self.batch_size
            
            for batch_start in range(0, len(valid_symbols), self.batch_size):
                batch_count += 1
                batch_symbols = valid_symbols[batch_start:batch_start + self.batch_size]
                
                for symbol in batch_symbols:
                    try:
                        if symbol in excluded_symbols or symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
                            continue
                        
                        signal = self.analyzer.analyze_symbol(symbol)
                        
                        if signal == target_direction:
                            return {
                                'symbol': symbol,
                                'direction': target_direction,
                                'signal_strength': 'high',
                                'batch_found': batch_count,
                                'total_batches': total_batches,
                                'qualified': True,
                                'leverage_supported': True
                            }
                            
                    except Exception:
                        continue
            
            return self._find_fallback_coin(target_direction, excluded_symbols, target_leverage)
            
        except Exception as e:
            logger.error(f"Lỗi tìm coin: {str(e)}")
            return None
    
    def _find_fallback_coin(self, target_direction, excluded_symbols, target_leverage):
        all_symbols = get_all_usdt_pairs(limit=100)
        if not all_symbols:
            return None
            
        valid_symbols = self.filter_symbols_by_leverage(all_symbols, target_leverage)
        if not valid_symbols:
            return None
            
        random.shuffle(valid_symbols)
        
        for symbol in valid_symbols:
            if symbol in ['BTCUSDT', 'ETHUSDT'] or symbol in excluded_symbols:
                continue
                
            try:
                change_24h = get_24h_change(symbol)
                if change_24h is None:
                    continue
                
                score = 0
                if target_direction == "BUY" and change_24h < -5:
                    score = abs(change_24h) / 15
                elif target_direction == "SELL" and change_24h > 5:
                    score = abs(change_24h) / 15
                
                if score > 0.2:
                    return {
                        'symbol': symbol,
                        'direction': target_direction,
                        'score': score,
                        'fallback': True,
                        'qualified': True,
                        'leverage_supported': True
                    }
                        
            except Exception:
                continue
        
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
                    if response.status == 401:
                        return None
                    if response.status == 429:
                        time.sleep(2 ** attempt)
                    elif response.status >= 500:
                        time.sleep(1)
                    continue
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return None
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            continue
        except Exception as e:
            time.sleep(1)
    
    return None

def get_all_usdt_pairs(limit=200):
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = binance_api_request(url)
        if not data:
            return []
        
        usdt_pairs = []
        for symbol_info in data.get('symbols', []):
            symbol = symbol_info.get('symbol', '')
            if symbol.endswith('USDT') and symbol_info.get('status') == 'TRADING':
                usdt_pairs.append(symbol)
        
        return usdt_pairs[:limit] if limit else usdt_pairs
        
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách coin từ Binance: {str(e)}")
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

def get_24h_change(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol.upper()}"
        data = binance_api_request(url)
        if data and 'priceChangePercent' in data:
            change = data['priceChangePercent']
            if change is None:
                return 0.0
            return float(change) if change is not None else 0.0
        return 0.0
    except Exception as e:
        logger.error(f"Lỗi lấy biến động 24h cho {symbol}: {str(e)}")
    return 0.0

# ========== WEBSOCKET MANAGER ==========
class WebSocketManager:
    def __init__(self):
        self.connections = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
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
            if not self._stop_event.is_set():
                time.sleep(5)
                self._reconnect(symbol, callback)
            
        def on_close(ws, close_status_code, close_msg):
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
        
    def _reconnect(self, symbol, callback):
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
                
    def stop(self):
        self._stop_event.set()
        for symbol in list(self.connections.keys()):
            self.remove_symbol(symbol)

# ========== KHỞI TẠO GLOBAL INSTANCES ==========
coin_manager = CoinManager()

# ========== HÀM KHỞI ĐỘNG HỆ THỐNG ==========
def start_trading_system(api_key, api_secret, telegram_bot_token, telegram_chat_id):
    bot_manager = BotManager(
        api_key=api_key,
        api_secret=api_secret,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id
    )
    
    return bot_manager

# ========== BASE BOT NÂNG CẤP - ĐÃ TỐI ƯU ==========
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
        self.prices = deque(maxlen=100)
        self.position_open = False
        self._stop = False
        
        self.last_trade_time = 0
        self.last_close_time = 0
        self.last_position_check = 0
        self.last_error_log_time = 0
        
        self.cooldown_period = 300
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
        self.find_interval = 60
        
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
        # CHỈ LOG LỖI VÀ MỞ/ĐÓNG VỊ THẾ
        if "LỖI" in message.upper() or "❌" in message or "✅" in message or "🟢" in message or "🔴" in message or "⛔" in message:
            logger.info(f"[{self.symbol or 'NO_COIN'} - {self.strategy_name}] {message}")
            if self.telegram_bot_token and self.telegram_chat_id:
                symbol_info = f"<b>{self.symbol}</b>" if self.symbol else "<i>Đang tìm coin...</i>"
                send_telegram(f"{symbol_info} ({self.strategy_name}): {message}", 
                             bot_token=self.telegram_bot_token, 
                             default_chat_id=self.telegram_chat_id)

    def _handle_price_update(self, price):
        if self._stop or not price or price <= 0:
            return
        try:
            self.prices.append(float(price))
        except Exception as e:
            self.log(f"❌ Lỗi xử lý giá: {str(e)}")

    def get_signal(self):
        raise NotImplementedError("Phương thức get_signal cần được triển khai")

    def get_target_direction(self):
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            buy_count = 0
            sell_count = 0
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt > 0:
                    buy_count += 1
                elif position_amt < 0:
                    sell_count += 1
            
            if buy_count > sell_count:
                return "SELL"
            elif sell_count > buy_count:
                return "BUY"
            else:
                return "BUY" if random.random() > 0.5 else "SELL"
                    
        except Exception as e:
            self.log(f"❌ Lỗi kiểm tra vị thế Binance: {str(e)}")
            return "BUY" if random.random() > 0.5 else "SELL"

    def find_and_set_coin(self):
        try:
            current_time = time.time()
            if current_time - self.last_find_time < self.find_interval:
                return False
            
            self.last_find_time = current_time
            
            self.current_target_direction = self.get_target_direction()
            
            managed_coins = self.coin_manager.get_managed_coins()
            excluded_symbols = set(managed_coins.keys())
            
            coin_data = self.coin_finder.find_coin_by_direction(
                self.current_target_direction, 
                excluded_symbols,
                self.lev
            )
        
            if coin_data is None:
                return False
                
            if not coin_data.get('qualified', False):
                return False
            
            new_symbol = coin_data['symbol']
            
            if self._register_coin_with_retry(new_symbol):
                if self.symbol:
                    self.ws_manager.remove_symbol(self.symbol)
                    self.coin_manager.unregister_coin(self.symbol)
                
                self.symbol = new_symbol
                self.ws_manager.add_symbol(self.symbol, self._handle_price_update)
                
                self.log(f"🎯 Đã tìm thấy coin {new_symbol} - {self.current_target_direction}")
                
                self.status = "waiting"
                return True
            else:
                return False
                
        except Exception as e:
            self.log(f"❌ Lỗi tìm coin: {str(e)}")
            return False

    def get_signal_with_balance(self, original_signal):
        try:
            current_time = time.time()
            if current_time - self.position_balance_check < self.balance_check_interval:
                return original_signal
            
            self.position_balance_check = current_time
            
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            if not all_positions:
                return original_signal
            
            buy_count = 0
            sell_count = 0
            total_value = 0
            buy_value = 0
            sell_value = 0
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                entry_price = float(pos.get('entryPrice', 0))
                leverage = float(pos.get('leverage', 1))
                
                if position_amt != 0:
                    position_value = abs(position_amt) * entry_price / leverage
                    total_value += position_value
                    
                    if position_amt > 0:
                        buy_count += 1
                        buy_value += position_value
                    else:
                        sell_count += 1
                        sell_value += position_value
            
            total = buy_count + sell_count
            buy_ratio = buy_value / total_value if total_value > 0 else 0.5
            sell_ratio = sell_value / total_value if total_value > 0 else 0.5
            
            if original_signal == "BUY" and buy_ratio >= 0.6:
                return "SELL"
            elif original_signal == "SELL" and sell_ratio >= 0.6:
                return "BUY"
            elif original_signal == "BUY" and buy_ratio > sell_ratio + 0.2:
                return "SELL"
            elif original_signal == "SELL" and sell_ratio > buy_ratio + 0.2:
                return "BUY"
            else:
                return original_signal
                
        except Exception as e:
            self.log(f"❌ Lỗi cân bằng với vị thế Binance: {str(e)}")
            return original_signal

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
                
                if current_time - self.last_position_check > self.position_check_interval:
                    self.check_position_status()
                    self.last_position_check = current_time
                              
                if not self.position_open:
                    signal = self.get_signal()
                    
                    if signal and signal != "NEUTRAL":
                        balanced_signal = self.get_signal_with_balance(signal)
                        
                        if (balanced_signal and balanced_signal != "NEUTRAL" and
                            current_time - self.last_trade_time > 60 and
                            current_time - self.last_close_time > self.cooldown_period):
                            
                            if self.open_position(balanced_signal):
                                self.last_trade_time = current_time
                    else:
                        time.sleep(5)
                
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
            return False
            
        try:
            if not self.coin_finder.is_leverage_supported(self.symbol, self.lev):
                self.close_position("Coin không hỗ trợ đòn bẩy")
                self.symbol = None
                self.status = "searching"
                return False

            self.check_position_status()
            if self.position_open:
                return False
    
            if self.should_be_removed:
                return False
    
            if not set_leverage(self.symbol, self.lev, self.api_key, self.api_secret):
                return False
    
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None or balance <= 0:
                return False
    
            current_price = get_current_price(self.symbol)
            if current_price <= 0:
                return False
    
            step_size = get_step_size(self.symbol, self.api_key, self.api_secret)
            usd_amount = balance * (self.percent / 100)
            qty = (usd_amount * self.lev) / current_price
            
            if step_size > 0:
                qty = math.floor(qty / step_size) * step_size
                qty = round(qty, 8)
    
            if qty < step_size:
                return False
    
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
                    return False
            else:
                return False
                    
        except Exception as e:
            self.log(f"❌ Lỗi mở lệnh: {str(e)}")
            return False

    def close_position(self, reason=""):
        try:
            self.check_position_status()
            
            if not self.position_open or abs(self.qty) <= 0:
                if self.symbol:
                    self.coin_manager.unregister_coin(self.symbol)
                return False

            current_time = time.time()
            if self._close_attempted and current_time - self._last_close_attempt < 30:
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


# ========== BOT XU HƯỚNG ĐỘNG ==========
class DynamicTrendBot(BaseBot):
    """Bot động sử dụng hệ thống chỉ báo xu hướng tích hợp"""
    
    def __init__(self, symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret, 
                 telegram_bot_token, telegram_chat_id, config_key=None, bot_id=None):
        
        super().__init__(symbol, lev, percent, tp, sl, ws_manager, api_key, api_secret,
                        telegram_bot_token, telegram_chat_id, "Dynamic Trend System", 
                        config_key, bot_id)
        
        self.analyzer = TrendIndicatorSystem()
        self.last_analysis_time = 0
        self.analysis_interval = 180
        
    def get_signal(self):
        """Lấy tín hiệu từ hệ thống chỉ báo tích hợp"""
        if not self.symbol:
            if self.find_and_set_coin():
                return None
            else:
                return None
            
        try:
            current_time = time.time()
            if current_time - self.last_analysis_time < self.analysis_interval:
                return None
            
            self.last_analysis_time = current_time
            
            signal = self.analyzer.analyze_symbol(self.symbol)
            
            if signal != "NEUTRAL":
                self.log(f"🎯 Nhận tín hiệu {signal} từ hệ thống chỉ báo")
            
            return signal
            
        except Exception as e:
            self.log(f"❌ Lỗi phân tích chỉ báo: {str(e)}")
            return None

# ========== BOT MANAGER HOÀN CHỈNH - HỖ TRỢ SL=0 ==========
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
            self.log("🟢 HỆ THỐNG BOT XU HƯỚNG ĐÃ KHỞI ĐỘNG")
            self.log("🎯 Sử dụng hệ thống chỉ báo tích hợp: EMA + RSI + Volume + Support/Resistance")
            self.log("📊 Hệ thống thống kê xác suất: Phân tích 200 nến lịch sử")
            self.log("🔄 Hệ thống Rotation Coin: Tự động tìm coin mới sau khi đóng lệnh")
            
            self.telegram_thread = threading.Thread(target=self._telegram_listener, daemon=True)
            self.telegram_thread.start()
            
            if self.telegram_chat_id:
                self.send_main_menu(self.telegram_chat_id)
        else:
            self.log("⚡ BotManager khởi động ở chế độ không config")

    def _verify_api_connection(self):
        """Kiểm tra kết nối API Binance"""
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance API.")
        else:
            self.log(f"✅ Kết nối Binance thành công! Số dư: {balance:.2f} USDT")

    def get_detailed_statistics(self):
        """Thống kê chi tiết hệ thống"""
        try:
            all_positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            
            system_positions = []
            external_positions = []
            total_pnl = 0
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:
                    symbol = pos.get('symbol', 'UNKNOWN')
                    entry_price = float(pos.get('entryPrice', 0))
                    leverage = float(pos.get('leverage', 1))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    total_pnl += unrealized_pnl
                    
                    position_info = {
                        'symbol': symbol,
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'leverage': leverage,
                        'entry_price': entry_price,
                        'size': abs(position_amt),
                        'pnl': unrealized_pnl
                    }
                    
                    coin_manager = CoinManager()
                    if coin_manager.is_coin_managed(symbol):
                        system_positions.append(position_info)
                    else:
                        external_positions.append(position_info)
            
            report = "📊 **THỐNG KÊ CHI TIẾT HỆ THỐNG**\n\n"
            
            balance = get_balance(self.api_key, self.api_secret)
            report += f"💰 **Số dư khả dụng**: {balance:.2f} USDT\n"
            report += f"📈 **Tổng PnL chưa thực hiện**: {total_pnl:.2f} USDT\n\n"
            
            report += f"🤖 **VỊ THẾ HỆ THỐNG** ({len(system_positions)}):\n"
            for pos in system_positions:
                report += (
                    f"🔹 {pos['symbol']} | {pos['side']} | {pos['leverage']}x\n"
                    f"   📊 Size: {pos['size']:.4f} | Entry: {pos['entry_price']:.4f}\n"
                    f"   💰 PnL: {pos['pnl']:.2f} USDT\n\n"
                )
            
            if not system_positions:
                report += "   📭 Không có vị thế hệ thống\n\n"
            
            if external_positions:
                report += f"🌐 **VỊ THẾ NGOÀI HỆ THỐNG** ({len(external_positions)}):\n"
                for pos in external_positions:
                    report += f"🔸 {pos['symbol']} | {pos['side']} | {pos['leverage']}x\n"
                report += "\n"
            
            active_bots = len([b for b in self.bots.values() if b.position_open])
            searching_bots = len([b for b in self.bots.values() if b.status == "searching"])
            waiting_bots = len([b for b in self.bots.values() if b.status == "waiting"])
            
            report += f"🤖 **TRẠNG THÁI BOT**:\n"
            report += f"   🟢 Đang trade: {active_bots}\n"
            report += f"   🔍 Đang tìm coin: {searching_bots}\n"
            report += f"   🟡 Chờ tín hiệu: {waiting_bots}\n"
            report += f"   📈 Tổng số bot: {len(self.bots)}\n\n"
            
            coin_manager = CoinManager()
            managed_coins = coin_manager.get_managed_coins()
            if managed_coins:
                report += f"🔗 **COIN ĐANG QUẢN LÝ** ({len(managed_coins)}):\n"
                for symbol, info in list(managed_coins.items())[:10]:
                    report += f"   {symbol} ({info['strategy']})\n"
                if len(managed_coins) > 10:
                    report += f"   ... và {len(managed_coins) - 10} coin khác"
            
            return report
            
        except Exception as e:
            return f"❌ Lỗi thống kê: {str(e)}"

    def get_probability_report(self, symbol):
        """Lấy báo cáo xác suất cho symbol"""
        analyzer = TrendIndicatorSystem()
        return analyzer.get_probability_report(symbol)

    def log(self, message):
        """Log thông điệp hệ thống"""
        if "❌" in message or "✅" in message or "🟢" in message or "🔴" in message or "⛔" in message:
            logger.info(f"[SYSTEM] {message}")
            if self.telegram_bot_token and self.telegram_chat_id:
                send_telegram(f"<b>SYSTEM</b>: {message}", 
                             bot_token=self.telegram_bot_token, 
                             default_chat_id=self.telegram_chat_id)

    def send_main_menu(self, chat_id):
        """Gửi menu chính Telegram"""
        welcome = (
            "🤖 <b>BOT GIAO DỊCH FUTURES ĐA LUỒNG</b>\n\n"
            "🎯 <b>HỆ THỐNG CHỈ BÁO XU HƯỚNG TÍCH HỢP</b>\n"
            "📊 <b>THỐNG KÊ XÁC SUẤT 200 NẾN</b>\n"
            "🔄 <b>ROTATION COIN TỰ ĐỘNG</b>\n\n"
            "👉 <b>Chọn chức năng bên dưới:</b>"
        )
        send_telegram(welcome, chat_id, create_main_menu(),
                     bot_token=self.telegram_bot_token, 
                     default_chat_id=self.telegram_chat_id)

    def add_bot(self, symbol, lev, percent, tp, sl, strategy_type, bot_count=1, **kwargs):
        """Thêm bot với đầy đủ thông số - HỖ TRỢ SL=0"""
        bot_mode = kwargs.get('bot_mode', 'static')
        
        # 🎯 QUAN TRỌNG: Cho phép SL = 0 (không có stop loss)
        # Không cần convert sl = 0 thành None, giữ nguyên giá trị
        
        if not self.api_key or not self.api_secret:
            self.log("❌ Chưa thiết lập API Key trong BotManager")
            return False
        
        test_balance = get_balance(self.api_key, self.api_secret)
        if test_balance is None:
            self.log("❌ LỖI: Không thể kết nối Binance")
            return False
        
        created_count = 0
        
        for i in range(bot_count):
            try:
                if bot_mode == 'static' and symbol:
                    bot_id = f"{symbol}_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = DynamicTrendBot
                    
                    bot = bot_class(symbol, lev, percent, tp, sl, self.ws_manager,
                                  self.api_key, self.api_secret, self.telegram_bot_token, 
                                  self.telegram_chat_id, bot_id=bot_id)
                    
                else:
                    bot_id = f"DYNAMIC_{strategy_type}_{i}_{int(time.time())}"
                    
                    if bot_id in self.bots:
                        continue
                    
                    bot_class = DynamicTrendBot
                    
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
            # Hiển thị thông tin SL
            sl_display = "KHÔNG" if sl == 0 else f"{sl}%"
            
            success_msg = (
                f"✅ <b>ĐÃ TẠO {created_count}/{bot_count} BOT THÀNH CÔNG</b>\n\n"
                f"🎯 Hệ thống: Trend Indicator System\n"
                f"📊 Chỉ báo: EMA + RSI + Volume + Support/Resistance\n"
                f"📈 Thống kê: Phân tích xác suất 200 nến\n"
                f"💰 Đòn bẩy: {lev}x\n"
                f"📈 % Số dư: {percent}%\n"
                f"🎯 TP: {tp}%\n"
                f"🛡️ SL: {sl_display}\n"
                f"🔧 Chế độ: {bot_mode}\n"
            )
            
            if bot_mode == 'static' and symbol:
                success_msg += f"🔗 Coin: {symbol}\n"
            else:
                success_msg += f"🔗 Coin: Tự động tìm kiếm\n"
            
            success_msg += (
                f"\n📊 <b>Hệ thống thống kê xác suất:</b>\n"
                f"• Phân tích 200 nến lịch sử\n"
                f"• Tính kỳ vọng & phương sai\n"
                f"• Đề xuất hướng tối ưu"
            )
            
            self.log(success_msg)
            return True
        else:
            self.log("❌ Không thể tạo bot nào")
            return False

    def stop_bot(self, bot_id):
        """Dừng bot cụ thể"""
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]
            self.log(f"⛔ Đã dừng bot {bot_id}")
            return True
        return False

    def stop_all(self):
        """Dừng tất cả bot"""
        self.log("⛔ Đang dừng tất cả bot...")
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        self.ws_manager.stop()
        self.running = False
        self.log("🔴 Hệ thống đã dừng")

    def _telegram_listener(self):
        """Lắng nghe tin nhắn Telegram"""
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
                            
                            if update_id > last_update_id:
                                last_update_id = update_id
                            
                            self._handle_telegram_message(chat_id, text)
                elif response.status_code == 409:
                    time.sleep(60)
                else:
                    time.sleep(10)
                
            except Exception as e:
                logger.error(f"Lỗi Telegram listener: {str(e)}")
                time.sleep(10)

    def _handle_telegram_message(self, chat_id, text):
        """Xử lý tin nhắn Telegram - ĐÃ SỬA LỖI XÁC NHẬN"""
        user_state = self.user_states.get(chat_id, {})
        current_step = user_state.get('step')
        
        # 🎯 QUAN TRỌNG: Xử lý xác nhận tạo bot TRƯỚC
        if current_step == 'waiting_confirmation':
            if text == "✅ Xác nhận tạo bot":
                self._confirm_create_bot(chat_id, user_state)
                return
            elif text == "❌ Hủy bỏ":
                self.user_states[chat_id] = {}
                send_telegram("❌ Đã hủy tạo bot", chat_id, create_main_menu(),
                             self.telegram_bot_token, self.telegram_chat_id)
                return
        
        # Xử lý các lệnh menu chính
        if text == "➕ Thêm Bot":
            self._handle_add_bot(chat_id)
        elif text == "📊 Danh sách Bot":
            self._handle_bot_list(chat_id)
        elif text == "📊 Thống kê":
            self._handle_statistics(chat_id)
        elif text == "⛔ Dừng Bot":
            self._handle_stop_bot(chat_id)
        elif text == "💰 Số dư":
            self._handle_balance(chat_id)
        elif text == "📈 Vị thế":
            self._handle_positions(chat_id)
        elif text == "🎯 Chiến lược":
            self._handle_strategy(chat_id)
        elif text == "⚙️ Cấu hình":
            self._handle_config(chat_id)
        elif text.startswith("⛔ "):
            self._handle_stop_specific_bot(chat_id, text)
        elif text.startswith("📊 "):
            symbol = text.replace("📊 ", "").strip()
            report = self.get_probability_report(symbol)
            send_telegram(report, chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
        elif current_step:
            self._handle_bot_creation_steps(chat_id, text, user_state, current_step)
        elif text.strip():
            self.send_main_menu(chat_id)

    def _handle_add_bot(self, chat_id):
        """Xử lý thêm bot - GIAO DIỆN ĐẦY ĐỦ"""
        self.user_states[chat_id] = {'step': 'waiting_bot_count'}
        balance = get_balance(self.api_key, self.api_secret)
        if balance is None:
            send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
            return
        
        send_telegram(
            f"🎯 <b>TẠO BOT GIAO DỊCH MỚI</b>\n\n"
            f"💰 Số dư hiện có: <b>{balance:.2f} USDT</b>\n\n"
            f"<b>Bước 1:</b> Chọn số lượng bot độc lập bạn muốn tạo:",
            chat_id,
            create_bot_count_keyboard(),
            self.telegram_bot_token, self.telegram_chat_id
        )

    def _handle_bot_creation_steps(self, chat_id, text, user_state, current_step):
        """Xử lý các bước tạo bot - GIAO DIỆN ĐẦY ĐỦ"""
        if text == '❌ Hủy bỏ':
            self.user_states[chat_id] = {}
            send_telegram("❌ Đã hủy quá trình tạo bot", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)
            return

        try:
            if current_step == 'waiting_bot_count':
                if text.isdigit():
                    bot_count = int(text)
                    if 1 <= bot_count <= 10:
                        user_state['bot_count'] = bot_count
                        user_state['step'] = 'waiting_bot_mode'
                        
                        send_telegram(
                            f"✅ <b>Bước 1:</b> Số lượng bot: {bot_count}\n\n"
                            f"<b>Bước 2:</b> Chọn chế độ bot:",
                            chat_id,
                            create_bot_mode_keyboard(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    else:
                        send_telegram("⚠️ Số lượng bot phải từ 1 đến 10. Vui lòng chọn lại:",
                                     chat_id, create_bot_count_keyboard(),
                                     self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ Vui lòng chọn số lượng bot từ bàn phím:",
                                 chat_id, create_bot_count_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)

            elif current_step == 'waiting_bot_mode':
                if text in ["🤖 Bot Tĩnh - Coin cụ thể", "🔄 Bot Động - Tự tìm coin"]:
                    user_state['bot_mode'] = 'static' if text == "🤖 Bot Tĩnh - Coin cụ thể" else 'dynamic'
                    user_state['step'] = 'waiting_strategy'
                    
                    mode_text = "BOT TĨNH - Coin cố định" if user_state['bot_mode'] == 'static' else "BOT ĐỘNG - Tự tìm coin"
                    
                    send_telegram(
                        f"✅ <b>Bước 2:</b> Chế độ: {mode_text}\n\n"
                        f"<b>Bước 3:</b> Chọn chiến lược giao dịch:",
                        chat_id,
                        create_strategy_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                else:
                    send_telegram("⚠️ Vui lòng chọn chế độ bot từ bàn phím:",
                                 chat_id, create_bot_mode_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)

            elif current_step == 'waiting_strategy':
                if text in ["⏰ Trend System"]:
                    user_state['strategy'] = "Trend System"
                    user_state['step'] = 'waiting_leverage'
                    
                    send_telegram(
                        f"✅ <b>Bước 3:</b> Chiến lược: Trend System\n\n"
                        f"<b>Bước 4:</b> Chọn đòn bẩy:",
                        chat_id,
                        create_leverage_keyboard(),
                        self.telegram_bot_token, self.telegram_chat_id
                    )
                else:
                    send_telegram("⚠️ Vui lòng chọn chiến lược từ bàn phím:",
                                 chat_id, create_strategy_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)

            elif current_step == 'waiting_leverage':
                lev_text = text[:-1] if text.endswith('x') else text
                if lev_text.isdigit():
                    leverage = int(lev_text)
                    if 1 <= leverage <= 100:
                        user_state['leverage'] = leverage
                        user_state['step'] = 'waiting_percent'
                        
                        send_telegram(
                            f"✅ <b>Bước 4:</b> Đòn bẩy: {leverage}x\n\n"
                            f"<b>Bước 5:</b> Chọn % số dư cho mỗi lệnh:",
                            chat_id,
                            create_percent_keyboard(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    else:
                        send_telegram("⚠️ Đòn bẩy phải từ 1 đến 100. Vui lòng chọn lại:",
                                     chat_id, create_leverage_keyboard(),
                                     self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ Vui lòng chọn đòn bẩy từ bàn phím:",
                                 chat_id, create_leverage_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)

            elif current_step == 'waiting_percent':
                if text.replace('.', '').isdigit():
                    percent = float(text)
                    if 0.1 <= percent <= 100:
                        user_state['percent'] = percent
                        user_state['step'] = 'waiting_tp'
                        
                        send_telegram(
                            f"✅ <b>Bước 5:</b> % Số dư: {percent}%\n\n"
                            f"<b>Bước 6:</b> Chọn Take Profit (%):",
                            chat_id,
                            create_tp_keyboard(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    else:
                        send_telegram("⚠️ % số dư phải từ 0.1 đến 100. Vui lòng chọn lại:",
                                     chat_id, create_percent_keyboard(),
                                     self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ Vui lòng chọn % số dư từ bàn phím:",
                                 chat_id, create_percent_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)

            elif current_step == 'waiting_tp':
                if text.isdigit():
                    tp = float(text)
                    if tp > 0:
                        user_state['tp'] = tp
                        user_state['step'] = 'waiting_sl'
                        
                        send_telegram(
                            f"✅ <b>Bước 6:</b> Take Profit: {tp}%\n\n"
                            f"<b>Bước 7:</b> Chọn Stop Loss (%):\n\n"
                            f"💡 <i>Chọn '0' để KHÔNG sử dụng Stop Loss</i>",
                            chat_id,
                            create_sl_keyboard(),
                            self.telegram_bot_token, self.telegram_chat_id
                        )
                    else:
                        send_telegram("⚠️ Take Profit phải lớn hơn 0. Vui lòng chọn lại:",
                                     chat_id, create_tp_keyboard(),
                                     self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ Vui lòng chọn Take Profit từ bàn phím:",
                                 chat_id, create_tp_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)

            elif current_step == 'waiting_sl':
                if text.isdigit():
                    sl = float(text)
                    if sl >= 0:  # 🎯 CHO PHÉP SL = 0
                        user_state['sl'] = sl
                        
                        # Chuyển sang bước xác nhận
                        self._create_bot_from_user_state(chat_id, user_state)
                        
                    else:
                        send_telegram("⚠️ Stop Loss phải lớn hơn hoặc bằng 0. Vui lòng chọn lại:",
                                     chat_id, create_sl_keyboard(),
                                     self.telegram_bot_token, self.telegram_chat_id)
                else:
                    send_telegram("⚠️ Vui lòng chọn Stop Loss từ bàn phím:",
                                 chat_id, create_sl_keyboard(),
                                 self.telegram_bot_token, self.telegram_chat_id)
                
        except ValueError as e:
            send_telegram("⚠️ Vui lòng nhập số hợp lệ:", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
        except Exception as e:
            send_telegram(f"❌ Lỗi xử lý: {str(e)}", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = {}

    def _create_bot_from_user_state(self, chat_id, user_state):
        """Tạo bot từ thông tin người dùng - ĐÃ SỬA"""
        try:
            strategy = user_state.get('strategy', 'Trend System')
            bot_mode = user_state.get('bot_mode', 'dynamic')
            leverage = user_state.get('leverage', 20)
            percent = user_state.get('percent', 5)
            tp = user_state.get('tp', 100)
            sl = user_state.get('sl', 50)  # SL có thể = 0
            symbol = user_state.get('symbol')
            bot_count = user_state.get('bot_count', 1)
            
            # 🎯 HIỂN THỊ SL=0 RÕ RÀNG
            if sl == 0:
                sl_display = "KHÔNG (SL=0)"
            else:
                sl_display = f"{sl}%"
            
            # Hiển thị thông tin tổng hợp
            summary_msg = (
                f"📋 <b>TỔNG HỢP THÔNG TIN BOT</b>\n\n"
                f"🤖 Số lượng: {bot_count} bot\n"
                f"🔧 Chế độ: {'Tĩnh - Coin cố định' if bot_mode == 'static' else 'Động - Tự tìm coin'}\n"
                f"🎯 Chiến lược: {strategy}\n"
                f"💰 Đòn bẩy: {leverage}x\n"
                f"📊 % Số dư: {percent}%\n"
                f"🎯 Take Profit: {tp}%\n"
                f"🛡️ Stop Loss: {sl_display}\n"
            )
            
            if bot_mode == 'static' and symbol:
                summary_msg += f"🔗 Coin: {symbol}\n"
            
            summary_msg += f"\n⚠️ <b>Xác nhận tạo {bot_count} bot với thông tin trên?</b>"
            
            # Gửi bàn phím xác nhận
            confirm_keyboard = {
                "keyboard": [
                    [{"text": "✅ Xác nhận tạo bot"}, {"text": "❌ Hủy bỏ"}]
                ],
                "resize_keyboard": True,
                "one_time_keyboard": True
            }
            
            send_telegram(summary_msg, chat_id, confirm_keyboard,
                         self.telegram_bot_token, self.telegram_chat_id)
            
            # 🎯 QUAN TRỌNG: Chuyển sang trạng thái chờ xác nhận
            user_state['step'] = 'waiting_confirmation'
            user_state['pending_creation'] = {
                'strategy': strategy,
                'bot_mode': bot_mode,
                'leverage': leverage,
                'percent': percent,
                'tp': tp,
                'sl': sl,  # SL có thể = 0
                'symbol': symbol,
                'bot_count': bot_count
            }
            
        except Exception as e:
            send_telegram(f"❌ Lỗi tạo bot: {str(e)}", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = {}

    def _confirm_create_bot(self, chat_id, user_state):
        """Xác nhận và tạo bot - ĐÃ SỬA LỖI"""
        try:
            pending_creation = user_state.get('pending_creation', {})
            
            if not pending_creation:
                send_telegram("❌ Lỗi: Không tìm thấy thông tin tạo bot", chat_id, create_main_menu(),
                             self.telegram_bot_token, self.telegram_chat_id)
                self.user_states[chat_id] = {}
                return
            
            # Lấy thông tin từ pending_creation
            strategy = pending_creation.get('strategy', 'Trend System')
            bot_mode = pending_creation.get('bot_mode', 'dynamic')
            leverage = pending_creation.get('leverage', 20)
            percent = pending_creation.get('percent', 5)
            tp = pending_creation.get('tp', 100)
            sl = pending_creation.get('sl', 50)  # SL có thể = 0
            symbol = pending_creation.get('symbol')
            bot_count = pending_creation.get('bot_count', 1)
            
            # Tạo bot
            success = self.add_bot(
                symbol=symbol,
                lev=leverage,
                percent=percent,
                tp=tp,
                sl=sl,  # SL có thể = 0
                strategy_type=strategy,
                bot_mode=bot_mode,
                bot_count=bot_count
            )
            
            if success:
                # Hiển thị thông tin SL
                sl_display = "KHÔNG" if sl == 0 else f"{sl}%"
                
                success_msg = (
                    f"✅ <b>ĐÃ TẠO {bot_count} BOT THÀNH CÔNG!</b>\n\n"
                    f"🤖 Số lượng: {bot_count} bot\n"
                    f"🔧 Chế độ: {'Tĩnh - Coin cố định' if bot_mode == 'static' else 'Động - Tự tìm coin'}\n"
                    f"🎯 Chiến lược: {strategy}\n"
                    f"💰 Đòn bẩy: {leverage}x\n"
                    f"📊 % Số dư: {percent}%\n"
                    f"🎯 Take Profit: {tp}%\n"
                    f"🛡️ Stop Loss: {sl_display}\n"
                )
                
                if bot_mode == 'static' and symbol:
                    success_msg += f"🔗 Coin: {symbol}\n"
                
                success_msg += f"\n🚀 <b>Bot đã bắt đầu hoạt động!</b>"
                
            else:
                success_msg = "❌ <b>Lỗi khi tạo bot. Vui lòng thử lại.</b>"
            
            send_telegram(success_msg, chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)
            
            # Xóa state sau khi tạo bot
            self.user_states[chat_id] = {}
            
        except Exception as e:
            send_telegram(f"❌ Lỗi xác nhận tạo bot: {str(e)}", chat_id, create_main_menu(),
                         self.telegram_bot_token, self.telegram_chat_id)
            self.user_states[chat_id] = {}

    def _handle_bot_list(self, chat_id):
        """Hiển thị danh sách bot"""
        if not self.bots:
            send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
        else:
            message = "🤖 <b>DANH SÁCH BOT ĐANG CHẠY</b>\n\n"
            
            for bot_id, bot in self.bots.items():
                if bot.status == "searching":
                    status = "🔍 Đang tìm coin"
                elif bot.status == "waiting":
                    status = "🟡 Chờ tín hiệu"
                elif bot.status == "open":
                    status = "🟢 Đang trade"
                else:
                    status = "⚪ Unknown"
                
                symbol_info = bot.symbol if bot.symbol else "Đang tìm..."
                sl_display = "KHÔNG" if bot.sl == 0 else f"{bot.sl}%"
                
                message += (
                    f"🔹 <b>{bot_id}</b>\n"
                    f"   📊 {symbol_info} | {status}\n"
                    f"   💰 ĐB: {bot.lev}x | Vốn: {bot.percent}%\n"
                    f"   🎯 TP: {bot.tp}% | 🛡️ SL: {sl_display}\n\n"
                )
            
            message += f"📈 Tổng số: {len(self.bots)} bot"
            
            send_telegram(message, chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_statistics(self, chat_id):
        """Hiển thị thống kê"""
        summary = self.get_detailed_statistics()
        send_telegram(summary, chat_id,
                     self.telegram_bot_token, self.telegram_chat_id)

    def _handle_stop_bot(self, chat_id):
        """Xử lý dừng bot"""
        if not self.bots:
            send_telegram("🤖 Không có bot nào đang chạy", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)
        else:
            message = "⛔ <b>CHỌN BOT ĐỂ DỪNG</b>\n\n"
            keyboard = []
            
            for bot_id in self.bots.keys():
                keyboard.append([{"text": f"⛔ {bot_id}"}])
            
            keyboard.append([{"text": "⛔ DỪNG TẤT CẢ"}])
            keyboard.append([{"text": "❌ Hủy bỏ"}])
            
            send_telegram(
                message, 
                chat_id, 
                {"keyboard": keyboard, "resize_keyboard": True, "one_time_keyboard": True},
                self.telegram_bot_token, self.telegram_chat_id
            )

    def _handle_stop_specific_bot(self, chat_id, text):
        """Xử lý dừng bot cụ thể"""
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

    def _handle_balance(self, chat_id):
        """Hiển thị số dư"""
        try:
            balance = get_balance(self.api_key, self.api_secret)
            if balance is None:
                send_telegram("❌ <b>LỖI KẾT NỐI BINANCE</b>\nVui lòng kiểm tra API Key!", chat_id,
                             self.telegram_bot_token, self.telegram_chat_id)
            else:
                send_telegram(f"💰 <b>SỐ DƯ KHẢ DỤNG</b>: {balance:.2f} USDT", chat_id,
                             self.telegram_bot_token, self.telegram_chat_id)
        except Exception as e:
            send_telegram(f"⚠️ Lỗi lấy số dư: {str(e)}", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_positions(self, chat_id):
        """Hiển thị vị thế"""
        try:
            positions = get_positions(api_key=self.api_key, api_secret=self.api_secret)
            if not positions:
                send_telegram("📭 Không có vị thế nào đang mở", chat_id,
                             self.telegram_bot_token, self.telegram_chat_id)
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
                         self.telegram_bot_token, self.telegram_chat_id)
        except Exception as e:
            send_telegram(f"⚠️ Lỗi lấy vị thế: {str(e)}", chat_id,
                         self.telegram_bot_token, self.telegram_chat_id)

    def _handle_strategy(self, chat_id):
        """Hiển thị thông tin chiến lược"""
        strategy_info = (
            "🎯 <b>HỆ THỐNG BOT XU HƯỚNG TÍCH HỢP</b>\n\n"
            "📊 <b>Hệ Thống Thống Kê Xác Suất</b>\n"
            "• Phân tích 200 nến lịch sử\n"
            "• Tính xác suất thắng cho từng chỉ báo\n"
            "• Tính kỳ vọng & phương sai\n"
            "• Đề xuất hướng tối ưu\n\n"
            "🤖 <b>Chỉ Báo Sử Dụng</b>\n"
            "• EMA (9, 21, 50) - Trọng số 30%\n"
            "• RSI + Volume - Trọng số 25%\n"  
            "• Support/Resistance - Trọng số 15%\n"
            "• Market Structure - Trọng số 10%\n"
            "• Probability Analysis - Trọng số 20%\n\n"
            "🔄 <b>Rotation Coin Tự Động</b>\n"
            "• Tự động tìm coin mới sau khi đóng lệnh\n"
            "• Phân tích đa khung thời gian\n"
            "• Quản lý rủi ro thông minh"
        )
        send_telegram(strategy_info, chat_id,
                     self.telegram_bot_token, self.telegram_chat_id)

    def _handle_config(self, chat_id):
        """Hiển thị cấu hình hệ thống"""
        balance = get_balance(self.api_key, self.api_secret)
        api_status = "✅ Đã kết nối" if balance is not None else "❌ Lỗi kết nối"
        
        searching_bots = sum(1 for bot in self.bots.values() if bot.status == "searching")
        trading_bots = sum(1 for bot in self.bots.values() if bot.status in ["waiting", "open"])
        total_bots = len(self.bots)
        
        config_info = (
            "⚙️ <b>CẤU HÌNH HỆ THỐNG ĐA LUỒNG</b>\n\n"
            f"🔑 Binance API: {api_status}\n"
            f"💰 Số dư: {balance:.2f} USDT\n"
            f"🤖 Tổng số bot: {total_bots}\n"
            f"🔍 Đang tìm coin: {searching_bots} bot\n"
            f"📊 Đang trade: {trading_bots} bot\n"
            f"📈 Thống kê xác suất: ✅ Đã kích hoạt\n"
            f"🔄 Rotation Coin: ✅ Đã kích hoạt\n"
            f"🎯 Hệ thống chỉ báo: ✅ Đã kích hoạt\n"
            f"🛡️ Hỗ trợ SL=0: ✅ Đã kích hoạt"
        )
        send_telegram(config_info, chat_id,
                     self.telegram_bot_token, self.telegram_chat_id)
