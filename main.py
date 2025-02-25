import unittest
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime, timedelta
import openai
import os
import json
import logging
import matplotlib.pyplot as plt
import requests
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

# Load environment variables
load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("CHAT_API")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# -------------------------------
# Main Trading Bot Code
# -------------------------------

# Initialize exchange (Binance futures in this case)
exchange = ccxt.binance({
    'enableRateLimit': True,  # Important to avoid API rate limits
    'apiKey': os.getenv("BINANCE_API_KEY"),  # From .env file
    'secret': os.getenv("BINANCE_SECRET_KEY"),  # From .env file
    'options': {
        'defaultType': 'future',  # For trading futures
    }
})

# Configuration parameters
SYMBOL = 'BTC/USDT'  # Use the proper symbol format for Binance futures
TIMEFRAME = '4h'
HISTORY_LIMIT = 500  # Number of candles to fetch
POSITION_SIZE = 0.01  # Size of each trade in BTC (smaller for safety)
STOP_LOSS_PCT = 2.0  # Stop loss percentage
TAKE_PROFIT_PCT = 3.5  # Take profit percentage
MAX_TRADES = 3  # Maximum concurrent trades
DRY_RUN = True  # Set to False to enable real trading
ENABLE_TELEGRAM = True  # Set to True to enable Telegram notifications

# Trade tracking
active_trades = {}
trade_history = []

# Create directories for saving data and charts
os.makedirs('data', exist_ok=True)
os.makedirs('charts', exist_ok=True)

class Trade:
    """Class to track individual trades"""
    def __init__(self, trade_id, symbol, side, entry_price, quantity, 
                 stop_loss, take_profit, timestamp, indicators=None):
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = timestamp
        self.exit_time = None
        self.exit_price = None
        self.pnl = None
        self.pnl_percent = None
        self.status = "OPEN"
        self.exit_reason = None
        self.indicators = indicators or {}
    
    def close_trade(self, exit_price, exit_time, exit_reason):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.status = "CLOSED"
        
        # Calculate PnL
        if self.side == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_percent = ((exit_price / self.entry_price) - 1) * 100
        else:  # sell
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_percent = ((self.entry_price / exit_price) - 1) * 100
        
        return self.pnl
    
    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else self.entry_time,
            'exit_time': self.exit_time.isoformat() if isinstance(self.exit_time, datetime) and self.exit_time else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'status': self.status,
            'exit_reason': self.exit_reason,
            'indicators': self.indicators
        }

def send_telegram_message(message):
    """Send message via Telegram"""
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram notifications not enabled or configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Telegram API error: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

def send_telegram_chart(chart_path, caption):
    """Send chart image via Telegram"""
    if not ENABLE_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram notifications not enabled or configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(chart_path, 'rb') as chart:
            files = {'photo': chart}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            response = requests.post(url, data=data, files=files)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram API error: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Failed to send Telegram chart: {e}")
        return False

def fetch_data(symbol, timeframe, limit=HISTORY_LIMIT):
    """Fetch market data from the exchange"""
    try:
        logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save raw data for debugging
        df.to_csv(f'data/{symbol.replace("/", "_")}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def apply_indicators(df):
    """Apply technical indicators to the dataframe"""
    try:
        # Trend indicators
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        
        # Volume indicator: VWAP
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        
        # Volatility indicators: Bollinger Bands and ATR
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bollinger_upper'] = bollinger.bollinger_hband()
        df['bollinger_lower'] = bollinger.bollinger_lband()
        df['bollinger_mid'] = bollinger.bollinger_mavg()
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Trend direction and strength: MACD and ADX
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        # Previous period values for determining crosses
        df['prev_ema_20'] = df['ema_20'].shift(1)
        df['prev_ema_50'] = df['ema_50'].shift(1)
        df['prev_macd'] = df['macd'].shift(1)
        df['prev_macd_signal'] = df['macd_signal'].shift(1)
        df['prev_stoch_k'] = df['stoch_k'].shift(1)
        df['prev_stoch_d'] = df['stoch_d'].shift(1)
        
        # Market regime and volatility state
        df['high_volatility'] = df['atr'] > df['atr'].rolling(30).mean() * 1.5
        
        return df
    except Exception as e:
        logger.error(f"Error applying indicators: {e}")
        raise
def check_market_hours():
    """Check if the gold market is likely to be open and active.
    Gold typically follows Forex hours with reduced weekend activity."""
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday (5) and Sunday (6)
        return False
    return True

def generate_signals(df):
    """Generate trading signals from technical indicators"""
    # Initialize signal columns
    df['buy_signal'] = False
    df['sell_signal'] = False
    df['signal_strength'] = 0

    # Trend condition - EMA alignment
    df['uptrend'] = (df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
    df['downtrend'] = (df['ema_20'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])
    
    # MACD crosses with an adaptive threshold
    macd_threshold = 0.02 * df['close'].mean() / 10000
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) &
        (df['prev_macd'] <= df['prev_macd_signal']) &
        ((df['macd'] - df['macd_signal']).abs() > macd_threshold)
    )
    df['macd_cross_down'] = (
        (df['macd'] < df['macd_signal']) &
        (df['prev_macd'] >= df['prev_macd_signal']) &
        ((df['macd'] - df['macd_signal']).abs() > macd_threshold)
    )
    
    # Stochastic crosses
    df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (df['prev_stoch_k'] <= df['prev_stoch_d'])
    df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['prev_stoch_k'] >= df['prev_stoch_d'])
    
    # RSI thresholds
    df['rsi_strong_oversold'] = df['rsi'] < 20    # more extreme for buys
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_strong_overbought'] = df['rsi'] > 80    # more extreme for sells
    df['rsi_overbought'] = df['rsi'] > 70

    # Bollinger Bands filter: check if price is near the extreme bands
    df['near_lower_band'] = df['close'] <= df['bollinger_lower'] * 1.01
    df['near_upper_band'] = df['close'] >= df['bollinger_upper'] * 0.99

    # Require a strong trend using ADX > 30
    df['strong_trend'] = df['adx'] > 30

    # VWAP filter for additional confirmation
    df['above_vwap'] = df['close'] > df['vwap']
    df['below_vwap'] = df['close'] < df['vwap']
    
    # Volume spike detection
    df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
    
    # Enhanced Buy Signals:
    strong_buy = (
        df['uptrend'] &
        df['macd_cross_up'] &
        (df['rsi_strong_oversold'] | (df['rsi_oversold'] & df['stoch_cross_up'])) &
        df['strong_trend'] &
        df['near_lower_band'] &
        df['volume_spike']
    )
    moderate_buy = (
        df['uptrend'] &
        df['macd_cross_up'] &
        df['rsi_oversold'] &
        df['above_vwap'] &
        df['stoch_cross_up']
    )
    
    # Enhanced Sell Signals:
    strong_sell = (
        df['downtrend'] &
        df['macd_cross_down'] &
        (df['rsi_strong_overbought'] | (df['rsi_overbought'] & df['stoch_cross_down'])) &
        df['strong_trend'] &
        df['near_upper_band'] &
        df['volume_spike']
    )
    moderate_sell = (
        df['downtrend'] &
        df['macd_cross_down'] &
        df['rsi_overbought'] &
        df['below_vwap'] &
        df['stoch_cross_down']
    )
    
    # Assign signals and strength
    df.loc[strong_buy, 'buy_signal'] = True
    df.loc[strong_buy, 'signal_strength'] = 3
    df.loc[moderate_buy & ~strong_buy, 'buy_signal'] = True
    df.loc[moderate_buy & ~strong_buy, 'signal_strength'] = 2
    
    df.loc[strong_sell, 'sell_signal'] = True
    df.loc[strong_sell, 'signal_strength'] = 3
    df.loc[moderate_sell & ~strong_sell, 'sell_signal'] = True
    df.loc[moderate_sell & ~strong_sell, 'signal_strength'] = 2
    
    # Market regime filter - reduce signals during extremely high volatility
    df.loc[df['high_volatility'] & (df['signal_strength'] < 3), 'buy_signal'] = False
    df.loc[df['high_volatility'] & (df['signal_strength'] < 3), 'sell_signal'] = False
    
    return df

def calculate_risk_reward(df, stop_loss_pct, take_profit_pct):
    """Calculate stop loss and take profit levels for trades using both percentage and ATR-based methods"""
    atr_multiplier = 1.5
    
    # Standard percentage-based stops
    df['stop_loss_buy_pct'] = df['close'] * (1 - stop_loss_pct/100)
    df['take_profit_buy_pct'] = df['close'] * (1 + take_profit_pct/100)
    df['stop_loss_sell_pct'] = df['close'] * (1 + stop_loss_pct/100)
    df['take_profit_sell_pct'] = df['close'] * (1 - take_profit_pct/100)
    
    # ATR-based stops
    df['stop_loss_buy_atr'] = df['close'] - (df['atr'] * atr_multiplier)
    df['take_profit_buy_atr'] = df['close'] + (df['atr'] * atr_multiplier * (take_profit_pct/stop_loss_pct))
    df['stop_loss_sell_atr'] = df['close'] + (df['atr'] * atr_multiplier)
    df['take_profit_sell_atr'] = df['close'] - (df['atr'] * atr_multiplier * (take_profit_pct/stop_loss_pct))
    
    # Choose the more conservative stop loss (for buys: the higher price, for sells: the lower price)
    df['stop_loss_buy'] = df[['stop_loss_buy_pct', 'stop_loss_buy_atr']].max(axis=1)
    df['take_profit_buy'] = df[['take_profit_buy_pct', 'take_profit_buy_atr']].min(axis=1)
    df['stop_loss_sell'] = df[['stop_loss_sell_pct', 'stop_loss_sell_atr']].min(axis=1)
    df['take_profit_sell'] = df[['take_profit_sell_pct', 'take_profit_sell_atr']].max(axis=1)
    
    return df

def generate_trade_chart(df, trade, filename):
    """Generate a chart for the trade with indicators"""
    try:
        # Select recent data around the trade
        if isinstance(trade.entry_time, str):
            entry_time = datetime.fromisoformat(trade.entry_time)
        else:
            entry_time = trade.entry_time
            
        start_time = entry_time - timedelta(minutes=20 * int(TIMEFRAME.replace('m', '')))
        chart_df = df.loc[start_time:].copy() if start_time in df.index else df.iloc[-40:].copy()
        
        plt.figure(figsize=(12, 10))
        
        # Price chart with EMAs and Bollinger Bands
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(chart_df.index, chart_df['close'], label='Close Price')
        ax1.plot(chart_df.index, chart_df['ema_20'], label='EMA 20', linestyle='--')
        ax1.plot(chart_df.index, chart_df['ema_50'], label='EMA 50', linestyle='--')
        ax1.plot(chart_df.index, chart_df['bollinger_upper'], label='BB Upper', linestyle=':')
        ax1.plot(chart_df.index, chart_df['bollinger_lower'], label='BB Lower', linestyle=':')
        
        # Mark trade entry/exit
        if trade.side == 'buy':
            ax1.axhline(y=trade.entry_price, color='g', linestyle='-', alpha=0.5)
            ax1.axhline(y=trade.stop_loss, color='r', linestyle='-', alpha=0.5)
            ax1.axhline(y=trade.take_profit, color='b', linestyle='-', alpha=0.5)
            ax1.scatter([entry_time], [trade.entry_price], marker='^', color='g', s=100)
            if trade.exit_price:
                exit_time = datetime.fromisoformat(trade.exit_time) if isinstance(trade.exit_time, str) else trade.exit_time
                ax1.scatter([exit_time], [trade.exit_price], marker='v', color='r' if trade.pnl < 0 else 'g', s=100)
        else:
            ax1.axhline(y=trade.entry_price, color='r', linestyle='-', alpha=0.5)
            ax1.axhline(y=trade.stop_loss, color='r', linestyle='-', alpha=0.5)
            ax1.axhline(y=trade.take_profit, color='b', linestyle='-', alpha=0.5)
            ax1.scatter([entry_time], [trade.entry_price], marker='v', color='r', s=100)
            if trade.exit_price:
                exit_time = datetime.fromisoformat(trade.exit_time) if isinstance(trade.exit_time, str) else trade.exit_time
                ax1.scatter([exit_time], [trade.exit_price], marker='^', color='r' if trade.pnl < 0 else 'g', s=100)
        
        ax1.set_title(f"{trade.symbol} - {trade.side.upper()} Trade")
        ax1.legend()
        ax1.grid(True)
        
        # RSI Chart
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(chart_df.index, chart_df['rsi'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.legend()
        ax2.grid(True)
        
        # MACD Chart
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(chart_df.index, chart_df['macd'], label='MACD')
        ax3.plot(chart_df.index, chart_df['macd_signal'], label='Signal')
        ax3.bar(chart_df.index, chart_df['macd_hist'], label='Histogram', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return filename
    except Exception as e:
        logger.error(f"Error generating trade chart: {e}")
        return None

def save_trade_data():
    """Save trade history and active trades to JSON files"""
    try:
        with open('data/trade_history.json', 'w') as f:
            json.dump([trade.to_dict() for trade in trade_history], f, indent=2)
        with open('data/active_trades.json', 'w') as f:
            json.dump({k: v.to_dict() for k, v in active_trades.items()}, f, indent=2)
        logger.info("Trade data saved successfully")
    except Exception as e:
        logger.error(f"Error saving trade data: {e}")

def load_trade_data():
    """Load trade history and active trades from JSON files"""
    global trade_history, active_trades
    try:
        if os.path.exists('data/trade_history.json'):
            with open('data/trade_history.json', 'r') as f:
                trade_data = json.load(f)
                for trade_dict in trade_data:
                    trade = Trade(
                        trade_id=trade_dict['trade_id'],
                        symbol=trade_dict['symbol'],
                        side=trade_dict['side'],
                        entry_price=trade_dict['entry_price'],
                        quantity=trade_dict['quantity'],
                        stop_loss=trade_dict['stop_loss'],
                        take_profit=trade_dict['take_profit'],
                        timestamp=trade_dict['entry_time'],
                        indicators=trade_dict.get('indicators', {})
                    )
                    if trade_dict['exit_price']:
                        trade.close_trade(
                            exit_price=trade_dict['exit_price'],
                            exit_time=trade_dict['exit_time'],
                            exit_reason=trade_dict['exit_reason']
                        )
                    trade_history.append(trade)
        if os.path.exists('data/active_trades.json'):
            with open('data/active_trades.json', 'r') as f:
                active_trades_data = json.load(f)
                for trade_id, trade_dict in active_trades_data.items():
                    trade = Trade(
                        trade_id=trade_dict['trade_id'],
                        symbol=trade_dict['symbol'],
                        side=trade_dict['side'],
                        entry_price=trade_dict['entry_price'],
                        quantity=trade_dict['quantity'],
                        stop_loss=trade_dict['stop_loss'],
                        take_profit=trade_dict['take_profit'],
                        timestamp=trade_dict['entry_time'],
                        indicators=trade_dict.get('indicators', {})
                    )
                    active_trades[trade_id] = trade
        logger.info(f"Loaded {len(trade_history)} historical trades and {len(active_trades)} active trades")
    except Exception as e:
        logger.error(f"Error loading trade data: {e}")

def chatgpt_analyze_trade(trade_details):
    """
    Call the ChatGPT API to analyze the trade details.
    Expects a one-word response: "GO" to proceed or "HOLD" to skip.
    """
    prompt = f"""
You are a trading analysis assistant. A trade signal has been generated with the following details:
Trade type: {trade_details['trade_type']}
Symbol: {trade_details['symbol']}
Current Price: {trade_details['current_price']}
Stop Loss: {trade_details['stop_loss']}
Take Profit: {trade_details['take_profit']}
RSI: {trade_details['rsi']}
ADX: {trade_details['adx']}
MACD: {trade_details['macd']}
EMA 20: {trade_details['ema_20']}
EMA 50: {trade_details['ema_50']}
EMA 200: {trade_details['ema_200']}

Based on these details, do you recommend executing this trade? 
Respond with a one-word answer: "GO" to execute, or "HOLD" to skip.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a trading analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().upper()
        logger.info("ChatGPT analysis response: " + answer)
        return answer == "GO"
    except Exception as e:
        logger.error("Error calling ChatGPT API: " + str(e))
        return False

def execute_trade_order(signal, symbol, price, order_type='market', amount=POSITION_SIZE, 
                          stop_loss=None, take_profit=None, indicators=None):
    """Execute a trade and track it"""
    global active_trades
    try:
        if len(active_trades) >= MAX_TRADES:
            logger.warning(f"Maximum concurrent trades ({MAX_TRADES}) reached. Skipping trade.")
            return False

        trade_id = f"{symbol}_{signal}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"[{datetime.now()}] Placing {signal.upper()} order for {symbol} at {price}")

        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=signal,
            entry_price=price,
            quantity=POSITION_SIZE,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now(),
            indicators=indicators
        )

        if not DRY_RUN:
            if signal == 'buy':
                order = exchange.create_market_buy_order(symbol, POSITION_SIZE)
                if stop_loss:
                    exchange.create_order(symbol, 'stop', 'sell', POSITION_SIZE, None, {'stopPrice': stop_loss, 'reduceOnly': True})
                if take_profit:
                    exchange.create_order(symbol, 'limit', 'sell', POSITION_SIZE, take_profit, {'reduceOnly': True})
            elif signal == 'sell':
                order = exchange.create_market_sell_order(symbol, POSITION_SIZE)
                if stop_loss:
                    exchange.create_order(symbol, 'stop', 'buy', POSITION_SIZE, None, {'stopPrice': stop_loss, 'reduceOnly': True})
                if take_profit:
                    exchange.create_order(symbol, 'limit', 'buy', POSITION_SIZE, take_profit, {'reduceOnly': True})
            logger.info(f"Order executed: {order}")
        else:
            logger.info(f"DRY RUN: Would execute {signal.upper()} order for {symbol} at {price}")

        active_trades[trade_id] = trade

        # Prepare notification message
        message = f"*New {signal.upper()} Trade*\nSymbol: {symbol}\nEntry Price: {price}\nStop Loss: {stop_loss}\nTake Profit: {take_profit}\nPosition Size: {POSITION_SIZE}\nTime: {datetime.now()}\n"
        if indicators:
            message += "\n*Key Indicators:*\n"
            for key, value in indicators.items():
                message += f"{key}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n"
        if ENABLE_TELEGRAM:
            threading.Thread(target=send_telegram_message, args=(message,)).start()

        save_trade_data()
        return True
    except Exception as e:
        logger.error("Error executing trade: " + str(e))
        return False

def update_trades(current_price, symbol):
    """Update status of all active trades based on current price"""
    global active_trades, trade_history
    trades_to_close = []
    for trade_id, trade in active_trades.items():
        if trade.symbol != symbol:
            continue
        if trade.side == 'buy':
            if current_price <= trade.stop_loss:
                trades_to_close.append((trade_id, current_price, 'stop_loss'))
            elif current_price >= trade.take_profit:
                trades_to_close.append((trade_id, current_price, 'take_profit'))
        else:  # sell
            if current_price >= trade.stop_loss:
                trades_to_close.append((trade_id, current_price, 'stop_loss'))
            elif current_price <= trade.take_profit:
                trades_to_close.append((trade_id, current_price, 'take_profit'))
    for trade_id, exit_price, exit_reason in trades_to_close:
        close_trade(trade_id, exit_price, exit_reason)

def close_trade(trade_id, exit_price, exit_reason):
    """Close a specific trade"""
    global active_trades, trade_history
    if trade_id not in active_trades:
        logger.warning(f"Trade {trade_id} not found in active trades")
        return False
    trade = active_trades[trade_id]
    pnl = trade.close_trade(exit_price, datetime.now(), exit_reason)
    logger.info(f"Closing trade {trade_id} at {exit_price}, PnL: {pnl}")
    if not DRY_RUN:
        try:
            if trade.side == 'buy':
                exchange.create_market_sell_order(trade.symbol, trade.quantity)
            else:
                exchange.create_market_buy_order(trade.symbol, trade.quantity)
        except Exception as e:
            logger.error(f"Error executing close order: {e}")
    trade_history.append(trade)
    del active_trades[trade_id]
    save_trade_data()
    return True

def run_trading_bot(backtest=False):
    """Main trading bot function with optional backtesting mode"""
    logger.info(f"Starting trading bot for {SYMBOL} on {TIMEFRAME} timeframe")
    df = fetch_data(SYMBOL, TIMEFRAME)
    if df is None or len(df) < 200:
        logger.warning("Insufficient data to generate reliable signals")
        return
    df = apply_indicators(df)
    df = generate_signals(df)
    df = calculate_risk_reward(df, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
    
    if backtest:
        results = run_backtest(df)
        logger.info("Backtest results:")
        logger.info(results)
        return results
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_price = latest['close']
    
    if not check_market_hours():
        logger.info("Market is likely closed. Skipping this iteration.")
        return
    
    # Check for a strong signal (signal_strength==3) in both the current and previous candles
    if latest['buy_signal'] and prev['buy_signal'] and latest['signal_strength'] == 3 and prev['signal_strength'] == 3:
        logger.info(f"BUY signal detected with strength {latest['signal_strength']}/3")
        trade_details = {
            "trade_type": "buy",
            "symbol": SYMBOL,
            "current_price": current_price,
            "stop_loss": latest['stop_loss_buy'],
            "take_profit": latest['take_profit_buy'],
            "rsi": latest['rsi'],
            "adx": latest['adx'],
            "macd": latest['macd'],
            "ema_20": latest['ema_20'],
            "ema_50": latest['ema_50'],
            "ema_200": latest['ema_200']
        }
        if chatgpt_analyze_trade(trade_details):
            execute_trade_order('buy', SYMBOL, current_price, stop_loss=latest['stop_loss_buy'], take_profit=latest['take_profit_buy'], indicators=trade_details)
        else:
            logger.info("ChatGPT analysis did not recommend executing the BUY trade.")
    elif latest['sell_signal'] and prev['sell_signal'] and latest['signal_strength'] == 3 and prev['signal_strength'] == 3:
        logger.info(f"SELL signal detected with strength {latest['signal_strength']}/3")
        trade_details = {
            "trade_type": "sell",
            "symbol": SYMBOL,
            "current_price": current_price,
            "stop_loss": latest['stop_loss_sell'],
            "take_profit": latest['take_profit_sell'],
            "rsi": latest['rsi'],
            "adx": latest['adx'],
            "macd": latest['macd'],
            "ema_20": latest['ema_20'],
            "ema_50": latest['ema_50'],
            "ema_200": latest['ema_200']
        }
        if chatgpt_analyze_trade(trade_details):
            execute_trade_order('sell', SYMBOL, current_price, stop_loss=latest['stop_loss_sell'], take_profit=latest['take_profit_sell'], indicators=trade_details)
        else:
            logger.info("ChatGPT analysis did not recommend executing the SELL trade.")
    else:
        logger.info(f"No trade signal at {datetime.now()}")
        logger.info(f"Current price: {current_price}")
        logger.info(f"RSI: {latest['rsi']:.2f}, ADX: {latest['adx']:.2f}, MACD: {latest['macd']:.6f}")

def run_backtest(df):
    """Simple backtest functionality to evaluate strategy performance"""
    test_df = df.copy()
    test_df['position'] = 0  
    test_df['entry_price'] = 0.0
    test_df['exit_price'] = 0.0
    test_df['pnl'] = 0.0
    test_df['cumulative_pnl'] = 0.0
    
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(1, len(test_df)):
        prev_row = test_df.iloc[i-1]
        curr_row = test_df.iloc[i]
        
        if position == 0:
            if prev_row['buy_signal']:
                position = 1
                entry_price = curr_row['open']
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('entry_price')] = entry_price
                trades.append({'type': 'buy', 'entry': entry_price, 'time': str(curr_row.name)})
            elif prev_row['sell_signal']:
                position = -1
                entry_price = curr_row['open']
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('entry_price')] = entry_price
                trades.append({'type': 'sell', 'entry': entry_price, 'time': str(curr_row.name)})
        elif position == 1:
            stop_price = prev_row['stop_loss_buy']
            target_price = prev_row['take_profit_buy']
            if curr_row['low'] <= stop_price:
                position = 0
                exit_price = stop_price
                pnl = (exit_price - entry_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': str(curr_row.name), 'result': 'stop_loss'})
            elif curr_row['high'] >= target_price:
                position = 0
                exit_price = target_price
                pnl = (exit_price - entry_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': str(curr_row.name), 'result': 'take_profit'})
            elif prev_row['sell_signal']:
                position = 0
                exit_price = curr_row['open']
                pnl = (exit_price - entry_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': str(curr_row.name), 'result': 'signal_exit'})
        elif position == -1:
            stop_price = prev_row['stop_loss_sell']
            target_price = prev_row['take_profit_sell']
            if curr_row['high'] >= stop_price:
                position = 0
                exit_price = stop_price
                pnl = (entry_price - exit_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': str(curr_row.name), 'result': 'stop_loss'})
            elif curr_row['low'] <= target_price:
                position = 0
                exit_price = target_price
                pnl = (entry_price - exit_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': str(curr_row.name), 'result': 'take_profit'})
            elif prev_row['buy_signal']:
                position = 0
                exit_price = curr_row['open']
                pnl = (entry_price - exit_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': str(curr_row.name), 'result': 'signal_exit'})
    
    test_df['cumulative_pnl'] = test_df['pnl'].cumsum()
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) <= 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = np.mean([trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([trade['pnl'] for trade in trades if trade.get('pnl', 0) <= 0]) if losing_trades > 0 else 0
    profit_factor = (sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0) /
                     abs(sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) <= 0))
                     if losing_trades > 0 else float('inf'))
    cumulative = np.array(test_df['cumulative_pnl'])
    max_dd = 0
    peak = cumulative[0]
    for value in cumulative:
        if value > peak:
            peak = value
        dd = (peak - value)
        if dd > max_dd:
            max_dd = dd
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'total_return': test_df['cumulative_pnl'].iloc[-1] if len(test_df) > 0 else 0,
        'trades': trades
    }
# -------------------------------
# Main block
# -------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        logger.info("Running backtest to validate strategy...")
        backtest_results = run_trading_bot(backtest=True)
        logger.info("Backtest completed. Starting continuous live trading...")
        while True:
            try:
                run_trading_bot()
                sleep_seconds = 3600 if TIMEFRAME == '4h' else 60
                logger.info(f"Waiting {sleep_seconds} seconds for next update...")
                time.sleep(sleep_seconds)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(20)
