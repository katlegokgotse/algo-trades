import unittest
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime
import openai
import os

load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("CHAT_API")  # Replace with your actual API key
# -------------------------------
# Main Trading Bot Code
# -------------------------------

# Initialize exchange (Binance in this case)
# Consider using a testnet first for paper trading
exchange = ccxt.binance({
    'enableRateLimit': True,  # Important to avoid API rate limits
    # 'apiKey': 'YOUR_API_KEY',  # Uncomment when ready to trade with real account
    # 'secret': 'YOUR_SECRET_KEY',
})

# Configuration parameters
SYMBOL = 'BTC/USD'
TIMEFRAME = '1h'
HISTORY_LIMIT = 500  # Increased for better indicator calculation
POSITION_SIZE = 0.1  # Size of each trade
STOP_LOSS_PCT = 2.0  # Stop loss percentage
TAKE_PROFIT_PCT = 3.5  # Take profit percentage

def fetch_data(symbol, timeframe, limit=HISTORY_LIMIT):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def apply_indicators(df):
    # Trend indicators
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # Volume indicator: VWAP
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    
    # Volatility indicators: Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    df['bollinger_mid'] = bollinger.bollinger_mavg()
    
    # Trend direction and strength: MACD and ADX
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # Previous period values for determining crosses
    df['prev_ema_20'] = df['ema_20'].shift(1)
    df['prev_ema_50'] = df['ema_50'].shift(1)
    df['prev_macd'] = df['macd'].shift(1)
    df['prev_macd_signal'] = df['macd_signal'].shift(1)
    
    return df

def generate_signals(df):
    # Initialize signal columns
    df['buy_signal'] = False
    df['sell_signal'] = False
    df['signal_strength'] = 0

    # Trend condition - EMA alignment
    df['uptrend'] = (df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
    df['downtrend'] = (df['ema_20'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])
    
    # MACD crosses with a minimum threshold to filter weak signals
    macd_threshold = 0.02
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) &
        (df['prev_macd'] <= df['prev_macd_signal']) &
        ((df['macd'] - df['macd_signal']) > macd_threshold)
    )
    df['macd_cross_down'] = (
        (df['macd'] < df['macd_signal']) &
        (df['prev_macd'] >= df['prev_macd_signal']) &
        ((df['macd_signal'] - df['macd']) > macd_threshold)
    )
    
    # Adjusted RSI thresholds for stronger signals:
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
    
    # Enhanced Buy Signals:
    strong_buy = (
        df['uptrend'] &
        df['macd_cross_up'] &
        df['rsi_strong_oversold'] &
        df['strong_trend'] &
        df['near_lower_band']
    )
    moderate_buy = (
        df['uptrend'] &
        df['macd_cross_up'] &
        df['rsi_oversold'] &
        df['above_vwap']
    )
    
    # Enhanced Sell Signals:
    strong_sell = (
        df['downtrend'] &
        df['macd_cross_down'] &
        df['rsi_strong_overbought'] &
        df['strong_trend'] &
        df['near_upper_band']
    )
    moderate_sell = (
        df['downtrend'] &
        df['macd_cross_down'] &
        df['rsi_overbought'] &
        df['below_vwap']
    )
    
    df.loc[strong_buy, 'buy_signal'] = True
    df.loc[strong_buy, 'signal_strength'] = 3
    df.loc[moderate_buy & ~strong_buy, 'buy_signal'] = True
    df.loc[moderate_buy & ~strong_buy, 'signal_strength'] = 2
    
    df.loc[strong_sell, 'sell_signal'] = True
    df.loc[strong_sell, 'signal_strength'] = 3
    df.loc[moderate_sell & ~strong_sell, 'sell_signal'] = True
    df.loc[moderate_sell & ~strong_sell, 'signal_strength'] = 2
    
    return df

def calculate_risk_reward(df, stop_loss_pct, take_profit_pct):
    """Calculate stop loss and take profit levels for trades"""
    df['stop_loss_buy'] = df['close'] * (1 - stop_loss_pct/100)
    df['take_profit_buy'] = df['close'] * (1 + take_profit_pct/100)
    df['stop_loss_sell'] = df['close'] * (1 + stop_loss_pct/100)
    df['take_profit_sell'] = df['close'] * (1 - take_profit_pct/100)
    return df

def execute_trade(signal, symbol, price, order_type='market', amount=POSITION_SIZE, 
                  stop_loss=None, take_profit=None):
    """Execute a trade with proper risk management"""
    try:
        if signal == 'buy':
            print(f"[{datetime.now()}] Placing BUY order for {symbol} at {price}")
            # Uncomment the following line for live trading:
            # order = exchange.create_market_buy_order(symbol, amount)
            if stop_loss:
                print(f"Setting stop loss at {stop_loss}")
                # exchange.create_order(symbol, 'stop_loss', 'sell', amount, None, {'stopPrice': stop_loss})
            if take_profit:
                print(f"Setting take profit at {take_profit}")
                # exchange.create_order(symbol, 'take_profit', 'sell', amount, take_profit)
        elif signal == 'sell':
            print(f"[{datetime.now()}] Placing SELL order for {symbol} at {price}")
            # Uncomment the following line for live trading:
            # order = exchange.create_market_sell_order(symbol, amount)
            if stop_loss:
                print(f"Setting stop loss at {stop_loss}")
                # exchange.create_order(symbol, 'stop_loss', 'buy', amount, None, {'stopPrice': stop_loss})
            if take_profit:
                print(f"Setting take profit at {take_profit}")
                # exchange.create_order(symbol, 'take_profit', 'buy', amount, take_profit)
        return True
    except Exception as e:
        print(f"Error executing trade: {e}")
        return False

def check_market_hours():
    """Check if the gold market is likely to be open and active.
    Gold typically follows Forex hours with reduced weekend activity."""
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday (5) and Sunday (6)
        return False
    return True

def chatgpt_analyze_trade(trade_details):
    """
    Calls the ChatGPT API to analyze the trade details.
    Expects ChatGPT to respond with "GO" if trade is recommended or "HOLD" if not.
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

Considering these details, do you recommend executing this trade? 
Respond with a one-word answer: "GO" if you recommend executing the trade, or "HOLD" if you do not.
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
        print("ChatGPT analysis response:", answer)
        return answer == "GO"
    except Exception as e:
        print("Error calling ChatGPT API:", e)
        return False

def run_trading_bot(backtest=False):
    """Main trading bot function with optional backtesting mode"""
    print(f"Starting trading bot for {SYMBOL} on {TIMEFRAME} timeframe")
    
    df = fetch_data(SYMBOL, TIMEFRAME)
    if df is None or len(df) < 200:
        print("Insufficient data to generate reliable signals")
        return
    
    df = apply_indicators(df)
    df = generate_signals(df)
    df = calculate_risk_reward(df, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
    
    if backtest:
        backtest_results = run_backtest(df)
        print("Backtest results:")
        print(backtest_results)
        return backtest_results
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]  # Previous candle for confirmation
    current_price = latest['close']
    
    # Check if market is open
    if not check_market_hours():
        print("Market is likely closed. Skipping this iteration.")
        return
    
    # Only execute trade if both the current and previous candles show a strong (signal_strength==3) signal.
    if latest['buy_signal'] and prev['buy_signal'] and latest['signal_strength'] == 3 and prev['signal_strength'] == 3:
        print(f"BUY signal detected with strength {latest['signal_strength']}/3")
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
            execute_trade('buy', SYMBOL, current_price, 
                          stop_loss=latest['stop_loss_buy'], 
                          take_profit=latest['take_profit_buy'])
        else:
            print("ChatGPT analysis did not recommend executing the BUY trade.")
        
    elif latest['sell_signal'] and prev['sell_signal'] and latest['signal_strength'] == 3 and prev['signal_strength'] == 3:
        print(f"SELL signal detected with strength {latest['signal_strength']}/3")
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
            execute_trade('sell', SYMBOL, current_price, 
                          stop_loss=latest['stop_loss_sell'], 
                          take_profit=latest['take_profit_sell'])
        else:
            print("ChatGPT analysis did not recommend executing the SELL trade.")
    else:
        print(f"No trade signal at {datetime.now()}")
        print(f"Current price: {current_price}")
        print(f"RSI: {latest['rsi']:.2f}, ADX: {latest['adx']:.2f}, MACD: {latest['macd']:.6f}")

def run_backtest(df):
    """Simple backtest functionality to evaluate strategy performance"""
    test_df = df.copy()
    test_df['position'] = 0  # 1 for long, -1 for short, 0 for no position
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
                trades.append({'type': 'buy', 'entry': entry_price, 'time': curr_row.name})
            elif prev_row['sell_signal']:
                position = -1
                entry_price = curr_row['open']
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('entry_price')] = entry_price
                trades.append({'type': 'sell', 'entry': entry_price, 'time': curr_row.name})
        
        elif position == 1:  # Long position
            stop_price = prev_row['stop_loss_buy']
            target_price = prev_row['take_profit_buy']
            if curr_row['low'] <= stop_price:
                position = 0
                exit_price = stop_price
                pnl = (exit_price - entry_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': curr_row.name, 'result': 'stop_loss'})
            elif curr_row['high'] >= target_price:
                position = 0
                exit_price = target_price
                pnl = (exit_price - entry_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': curr_row.name, 'result': 'take_profit'})
            elif prev_row['sell_signal']:
                position = 0
                exit_price = curr_row['open']
                pnl = (exit_price - entry_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': curr_row.name, 'result': 'signal_exit'})
        
        elif position == -1:  # Short position
            stop_price = prev_row['stop_loss_sell']
            target_price = prev_row['take_profit_sell']
            if curr_row['high'] >= stop_price:
                position = 0
                exit_price = stop_price
                pnl = (entry_price - exit_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': curr_row.name, 'result': 'stop_loss'})
            elif curr_row['low'] <= target_price:
                position = 0
                exit_price = target_price
                pnl = (entry_price - exit_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': curr_row.name, 'result': 'take_profit'})
            elif prev_row['buy_signal']:
                position = 0
                exit_price = curr_row['open']
                pnl = (entry_price - exit_price) / entry_price * 100
                test_df.iloc[i, test_df.columns.get_loc('position')] = position
                test_df.iloc[i, test_df.columns.get_loc('exit_price')] = exit_price
                test_df.iloc[i, test_df.columns.get_loc('pnl')] = pnl
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_time': curr_row.name, 'result': 'signal_exit'})
    
    test_df['cumulative_pnl'] = test_df['pnl'].cumsum()
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if 'pnl' in trade and trade['pnl'] > 0)
    losing_trades = sum(1 for trade in trades if 'pnl' in trade and trade['pnl'] <= 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = np.mean([trade['pnl'] for trade in trades if 'pnl' in trade and trade['pnl'] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([trade['pnl'] for trade in trades if 'pnl' in trade and trade['pnl'] <= 0]) if losing_trades > 0 else 0
    profit_factor = (sum(trade['pnl'] for trade in trades if 'pnl' in trade and trade['pnl'] > 0) /
                     abs(sum(trade['pnl'] for trade in trades if 'pnl' in trade and trade['pnl'] <= 0))
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
    # If 'test' is passed as an argument, run the tests.
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        # Run backtest first to validate strategy before going live.
        print("Running backtest to validate strategy...")
        backtest_results = run_trading_bot(backtest=True)
        
        # Regardless of performance, start continuous live trading.
        print("Backtest completed. Starting continuous live trading...")
        while True:
            try:
                run_trading_bot()
                sleep_seconds = 20 if TIMEFRAME == '1h' else 60
                print(f"Waiting {sleep_seconds} seconds for next update...")
                time.sleep(sleep_seconds)
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(20)  # Wait on error
