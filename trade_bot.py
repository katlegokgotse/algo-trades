from datetime import datetime, timedelta, timezone
import time
import logging
from config import logger
from chatgpt_analyser import chatgpt_analyze_trade
from data_fetcher import DataFetcher
from indicator_calculator import IndicatorCalculator
from notifier import Notifier
from persistence_manager import PersistenceManager
from risk_manager import RiskManager
from signal_generator import SignalGenerator
from trade_executor import TradeExecutor
from trade_manager import TradeManager
import pandas as pd
import sys
import os
import csv
import json

class TradingBot:
    def __init__(self, 
                 exchange, 
                 symbol: str, 
                 timeframe: str, 
                 position_size: float, 
                 stop_loss_pct: float, 
                 take_profit_pct: float, 
                 max_trades: int, 
                 dry_run: bool, 
                 enable_telegram: bool, 
                 telegram_bot_token: str, 
                 telegram_chat_id: str):
        """
        Initialize the TradingBot with necessary components.
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades = max_trades
        self.dry_run = dry_run
        self.active_trades = {}
        self.trade_history = []
        
        # Create results directory if it doesn't exist
        self.results_dir = "backtest_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Instantiate modular components
        try:
            self.data_fetcher = DataFetcher(exchange)
            self.indicator_calculator = IndicatorCalculator()
            self.signal_generator = SignalGenerator()
            self.risk_manager = RiskManager()
            self.trade_executor = TradeExecutor(exchange, dry_run, max_trades)
            self.trade_manager = TradeManager(exchange=exchange)
            self.notifier = Notifier(enable_telegram, telegram_bot_token, telegram_chat_id)
            self.persistence_manager = PersistenceManager()
            self.persistence_manager.load_trade_data(self.active_trades, self.trade_history)
            logger.info(f"Successfully initialized all components for {symbol} trading bot")
        except Exception as e:
            logger.critical(f"Failed to initialize trading bot components: {str(e)}")
            raise
        
        # Flag for graceful shutdown
        self._shutdown = False

    def run(self) -> None:
        """
        Execute one iteration of the trading loop.
        """
        try:
            # Fetch market data
            data_frame = self.data_fetcher.fetch_data(self.symbol, self.timeframe)
            if data_frame is None or len(data_frame) < 50:
                logger.warning(f"Insufficient data for {self.symbol}, minimum 50 candles required")
                return
            
            # Apply indicators and generate signals
            data_frame = self.indicator_calculator.apply_indicators(data_frame)
            data_frame = self.signal_generator.generate_signals(data_frame)
            data_frame = self.risk_manager.calculate_risk_reward(data_frame, self.stop_loss_pct, self.take_profit_pct)
            
            latest = data_frame.iloc[-1]
            prev = data_frame.iloc[-2]
            current_price = latest['close']
            logger.info(f"Latest candle close: {current_price} at {data_frame.index[-1]}")
            
            if not self.check_market_hours():
                logger.info("Market is likely closed")
                return
            
            # Check for sufficient balance before processing signals
            if not self.dry_run and not self.check_sufficient_balance():
                logger.warning("Insufficient balance to execute trades")
                return
            
            # Update active trades and save state
            updated_trades = self.trade_manager.update_trades(current_price, self.symbol, 
                                                            self.active_trades, self.trade_history, self.dry_run)
            
            # If trades were updated (closed), save the updated state
            if updated_trades:
                self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
            
            # Define signal strength for logging consistency
            buy_signal_strength = 0
            if latest['buy_signal']: buy_signal_strength += 1
            if prev['buy_signal']: buy_signal_strength += 1
            if latest['signal_strength'] >= 2: buy_signal_strength += 1
            
            sell_signal_strength = 0
            if latest['sell_signal']: sell_signal_strength += 1
            if prev['sell_signal']: sell_signal_strength += 1
            if latest['signal_strength'] >= 2: sell_signal_strength += 1
            
            # Check for BUY signals
            if (latest['buy_signal'] and prev['buy_signal'] and 
                latest['signal_strength'] >= 2 and prev['signal_strength'] >= 2):
                logger.info(f"BUY signal detected with strength {buy_signal_strength}/3")
                trade_details = {
                    "trade_type": "buy", 
                    "symbol": self.symbol, 
                    "current_price": current_price,
                    "stop_loss": latest['stop_loss_buy'], 
                    "take_profit": latest['take_profit_buy'],
                    "rsi": latest['rsi'], 
                    "adx": latest['adx'], 
                    "macd": latest['macd'],
                    "ema_20": latest['ema_20'], 
                    "ema_50": latest['ema_50'], 
                    "ema_200": latest['ema_200'],
                    "fib_trend": latest.get('fib_trend', 'N/A'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Try ChatGPT analysis with fallback
                try:
                    chatgpt_approval = chatgpt_analyze_trade(trade_details)
                except Exception as e:
                    logger.warning(f"ChatGPT analysis failed: {str(e)}. Using fallback approval.")
                    chatgpt_approval = self.fallback_trade_analysis(trade_details)
                
                if chatgpt_approval:
                    if self.trade_executor.execute_trade_order('buy', self.symbol, current_price, 
                                                              latest['stop_loss_buy'], latest['take_profit_buy'], 
                                                              trade_details, self.active_trades, self.position_size):
                        last_trade = self.active_trades[list(self.active_trades.keys())[-1]]
                        self.notifier.notify_trade(last_trade, trade_details)
                        self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
                else:
                    logger.info("Trade analysis did not recommend executing the BUY trade.")
            
            # Check for SELL signals
            elif (latest['sell_signal'] and prev['sell_signal'] and 
                  latest['signal_strength'] >= 2 and prev['signal_strength'] >= 2):
                logger.info(f"SELL signal detected with strength {sell_signal_strength}/3")
                trade_details = {
                    "trade_type": "sell", 
                    "symbol": self.symbol, 
                    "current_price": current_price,
                    "stop_loss": latest['stop_loss_sell'], 
                    "take_profit": latest['take_profit_sell'],
                    "rsi": latest['rsi'], 
                    "adx": latest['adx'], 
                    "macd": latest['macd'],
                    "ema_20": latest['ema_20'], 
                    "ema_50": latest['ema_50'], 
                    "ema_200": latest['ema_200'],
                    "fib_trend": latest.get('fib_trend', 'N/A'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Try ChatGPT analysis with fallback
                try:
                    chatgpt_approval = chatgpt_analyze_trade(trade_details)
                except Exception as e:
                    logger.warning(f"ChatGPT analysis failed: {str(e)}. Using fallback approval.")
                    chatgpt_approval = self.fallback_trade_analysis(trade_details)
                
                if chatgpt_approval:
                    if self.trade_executor.execute_trade_order('sell', self.symbol, current_price, 
                                                              latest['stop_loss_sell'], latest['take_profit_sell'], 
                                                              trade_details, self.active_trades, self.position_size):
                        last_trade = self.active_trades[list(self.active_trades.keys())[-1]]
                        self.notifier.notify_trade(last_trade, trade_details)
                        self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
                else:
                    logger.info("Trade analysis did not recommend executing the SELL trade.")
            else:
                logger.info(f"Update: ${current_price}")
                logger.info(f"Market Conditions: No valid trade signal at {datetime.now(timezone.utc)}")
                logger.info(f"Trend: {latest.get('fib_trend', 'N/A')}")
                logger.info(f"Symbol: {self.symbol}")
                
        except Exception as e:
            logger.error(f"Error during trading cycle: {str(e)}")
            # Save trade data even on error to prevent data loss
            self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)

    def fallback_trade_analysis(self, trade_details):
        """
        Fallback analysis when ChatGPT is unavailable.
        Simple rule-based filter as a backup.
        
        Args:
            trade_details: Dictionary with trade details
            
        Returns:
            Boolean indicating whether to proceed with the trade
        """
        try:
            # Basic fallback rules
            if trade_details['trade_type'] == 'buy':
                # For buy signals
                if (trade_details['rsi'] < 70 and  # Not overbought
                    trade_details['adx'] > 25 and  # Strong trend
                    trade_details['ema_20'] > trade_details['ema_50']):  # Uptrend
                    return True
            else:  # sell
                # For sell signals
                if (trade_details['rsi'] > 30 and  # Not oversold
                    trade_details['adx'] > 25 and  # Strong trend
                    trade_details['ema_20'] < trade_details['ema_50']):  # Downtrend
                    return True
            return False
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return False  # If even the fallback fails, don't trade
    def check_sufficient_balance(self) -> bool:
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
        
            # Handle different symbol formats
            if '/' in self.symbol:
                # Format like "BTC/USD"
                symbol_parts = self.symbol.split('/')
                base_currency = symbol_parts[0]
                quote_currency = symbol_parts[1]
            else:
                # Format like "XBTUSDT" - need to extract base and quote currencies
                # Common quote currencies to check for at the end of the symbol
                common_quote_currencies = ["USDT", "USD", "BTC", "ETH", "EUR", "GBP", "JPY", "AUD", "ZAR"]
            
                # Try to find the quote currency in the symbol
                found_quote = False
                for quote in common_quote_currencies:
                    if self.symbol.endswith(quote):
                        quote_currency = quote
                        base_currency = self.symbol[:-len(quote)]
                        found_quote = True
                        break
            
                if not found_quote:
                    logger.error(f"Could not parse symbol format: {self.symbol}")
                    return False
            
                logger.info(f"Parsed symbol {self.symbol} as {base_currency}/{quote_currency}")
        
        # Check if we have enough balance
            if self.position_size > 0:
                # For buy orders, calculate cost in quote currency
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                cost = self.position_size * current_price
            
            # Add a check to ensure balance doesn't go below $1
                if quote_currency in balance and balance[quote_currency]['free'] >= cost and (balance[quote_currency]['free'] - cost) >= 1:
                    return True
                else:
                    logger.warning(f"Insufficient {quote_currency} balance for trade or would reduce balance below $1")
                    return False
            return True  # If position size is 0 (e.g., in dry run)
        except Exception as e:
            logger.error(f"Error checking balance: {str(e)}")
            return False  # If we can't check balance, don't trade     
    def backtest(self):
        """
        Run a backtest using historical data and return results.
        """
        try:
            data_frame = self.data_fetcher.fetch_data(self.symbol, self.timeframe, limit=500)
            if data_frame is None or len(data_frame) < 200:
                logger.warning("Insufficient data for backtest")
                return None
            data_frame = self.indicator_calculator.apply_indicators(data_frame)
            data_frame = self.signal_generator.generate_signals(data_frame)
            data_frame = self.risk_manager.calculate_risk_reward(data_frame, self.stop_loss_pct, self.take_profit_pct)
            
            results = self.run_backtest(data_frame)
            
            # Save backtest results to files
            if results:
                self.save_backtest_results(results)
                
            return results
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            return None

    def run_backtest(self, data_frame: pd.DataFrame):
        """
        Backtest logic to simulate trades based on signals, including total loss.

        Args:
            data_frame: DataFrame with price data, indicators, and signals.

        Returns:
            Dictionary with backtest results (total trades, win rate, loss rate, total loss, etc.).
        """
        total_trades = 0
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0
        open_trades = 0
        
        # Track all trades for reporting
        trade_log = []

        for i in range(1, len(data_frame)):
            prev = data_frame.iloc[i - 1]
            current = data_frame.iloc[i]
            current_price = current['close']

            # Simulate BUY trade
            if prev['buy_signal'] and prev['signal_strength'] >= 2:
                total_trades += 1
                entry_price = current_price
                stop_loss = prev['stop_loss_buy']
                take_profit = prev['take_profit_buy']
                trade_date = str(data_frame.index[i])
                
                trade_record = {
                    "id": total_trades,
                    "type": "buy",
                    "entry_date": trade_date,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }

                # Check future prices to determine trade outcome
                trade_closed = False
                for j in range(i + 1, min(i + 50, len(data_frame))):  # Limit look-ahead to 50 bars
                    future_price = data_frame.iloc[j]['close']
                    future_date = str(data_frame.index[j])
                    
                    if future_price <= stop_loss:
                        loss = (entry_price - stop_loss) * self.position_size
                        total_loss += loss
                        losses += 1
                        trade_closed = True
                        trade_record.update({
                            "exit_date": future_date,
                            "exit_price": stop_loss,
                            "pnl": -loss,
                            "outcome": "loss",
                            "reason": "stop_loss_hit"
                        })
                        break
                    elif future_price >= take_profit:
                        profit = (take_profit - entry_price) * self.position_size
                        total_profit += profit
                        wins += 1
                        trade_closed = True
                        trade_record.update({
                            "exit_date": future_date,
                            "exit_price": take_profit,
                            "pnl": profit,
                            "outcome": "win",
                            "reason": "take_profit_hit"
                        })
                        break
                
                if not trade_closed:
                    # For unclosed trades, use the last price in our data
                    last_price = data_frame.iloc[-1]['close']
                    last_date = str(data_frame.index[-1])
                    
                    # Calculate PnL based on current price
                    if last_price > entry_price:
                        unrealized_profit = (last_price - entry_price) * self.position_size
                        total_profit += unrealized_profit
                        wins += 1
                        outcome = "win"
                    else:
                        unrealized_loss = (entry_price - last_price) * self.position_size
                        total_loss += unrealized_loss
                        losses += 1
                        outcome = "loss"
                    
                    trade_record.update({
                        "exit_date": last_date,
                        "exit_price": last_price,
                        "pnl": unrealized_profit if outcome == "win" else -unrealized_loss,
                        "outcome": outcome,
                        "reason": "unclosed_position"
                    })
                    open_trades += 1
                
                trade_log.append(trade_record)

            # Simulate SELL trade
            elif prev['sell_signal'] and prev['signal_strength'] >= 2:
                total_trades += 1
                entry_price = current_price
                stop_loss = prev['stop_loss_sell']
                take_profit = prev['take_profit_sell']
                trade_date = str(data_frame.index[i])
                
                trade_record = {
                    "id": total_trades,
                    "type": "sell",
                    "entry_date": trade_date,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }

                # Check future prices to determine trade outcome
                trade_closed = False
                for j in range(i + 1, min(i + 50, len(data_frame))):  # Limit look-ahead to 50 bars
                    future_price = data_frame.iloc[j]['close']
                    future_date = str(data_frame.index[j])
                    
                    if future_price >= stop_loss:
                        loss = (stop_loss - entry_price) * self.position_size
                        total_loss += loss
                        losses += 1
                        trade_closed = True
                        trade_record.update({
                            "exit_date": future_date,
                            "exit_price": stop_loss,
                            "pnl": -loss,
                            "outcome": "loss",
                            "reason": "stop_loss_hit"
                        })
                        break
                    elif future_price <= take_profit:
                        profit = (entry_price - take_profit) * self.position_size
                        total_profit += profit
                        wins += 1
                        trade_closed = True
                        trade_record.update({
                            "exit_date": future_date,
                            "exit_price": take_profit,
                            "pnl": profit,
                            "outcome": "win",
                            "reason": "take_profit_hit"
                        })
                        break
                
                if not trade_closed:
                    # For unclosed trades, use the last price in our data
                    last_price = data_frame.iloc[-1]['close']
                    last_date = str(data_frame.index[-1])
                    
                    # Calculate PnL based on current price
                    if last_price < entry_price:
                        unrealized_profit = (entry_price - last_price) * self.position_size
                        total_profit += unrealized_profit
                        wins += 1
                        outcome = "win"
                    else:
                        unrealized_loss = (last_price - entry_price) * self.position_size
                        total_loss += unrealized_loss
                        losses += 1
                        outcome = "loss"
                    
                    trade_record.update({
                        "exit_date": last_date,
                        "exit_price": last_price,
                        "pnl": unrealized_profit if outcome == "win" else -unrealized_loss,
                        "outcome": outcome,
                        "reason": "unclosed_position"
                    })
                    open_trades += 1
                
                trade_log.append(trade_record)

        # Calculate metrics
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf') if total_profit > 0 else 0
        total_pnl = total_profit - total_loss
        avg_profit_per_win = (total_profit / wins) if wins > 0 else 0
        avg_loss_per_loss = (total_loss / losses) if losses > 0 else 0

        return {
            "total_trades": int(total_trades),
            "wins": int(wins),
            "losses": int(losses),
            "open_trades": int(open_trades),
            "win_rate": round(float(win_rate), 2),
            "loss_rate": round(float(loss_rate), 2),
            "profit_factor": round(float(profit_factor), 2),
            "total_profit_gross": round(float(total_profit), 2),  # Gross profit from wins
            "total_loss": round(float(total_loss), 2),            # Gross loss from losses
            "total_profit_net": round(float(total_pnl), 2),       # Net profit (profit - loss)
            "avg_profit_per_win": round(float(avg_profit_per_win), 2),
            "avg_loss_per_loss": round(float(avg_loss_per_loss), 2),
            "trade_log": trade_log  # Detailed log of all trades for analysis
        }
        
    def save_backtest_results(self, results):
        """
        Save backtest results to both text and CSV files.
        
        Args:
            results: Dictionary with backtest results
        """
        try:
            # Generate timestamp for file names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create file names
            summary_file = os.path.join(self.results_dir, f"{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}_summary.txt")
            csv_file = os.path.join(self.results_dir, f"{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}_trades.csv")
            json_file = os.path.join(self.results_dir, f"{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}_results.json")
            
            # Copy trade log for separate handling
            trade_log = results.get('trade_log', [])
            results_copy = results.copy()
            if 'trade_log' in results_copy:
                del results_copy['trade_log']
            
            # Write summary to text file
            with open(summary_file, 'w') as f:
                f.write(f"===== BACKTEST RESULTS FOR {self.symbol} {self.timeframe} =====\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Strategy Parameters:\n")
                f.write(f"  Position Size: {self.position_size}\n")
                f.write(f"  Stop Loss: {self.stop_loss_pct}%\n")
                f.write(f"  Take Profit: {self.take_profit_pct}%\n\n")
                
                f.write(f"Performance Metrics:\n")
                for key, value in results_copy.items():
                    # Format the key to be more readable
                    formatted_key = key.replace('_', ' ').title()
                    f.write(f"  {formatted_key}: {value}\n")
                
                # Add risk-adjusted metrics
                if results_copy['total_trades'] > 0:
                    f.write(f"\nRisk Metrics:\n")
                    # Calculate Sharpe ratio-like metric (assuming average trade duration is similar)
                    avg_return = results_copy['total_profit_net'] / results_copy['total_trades']
                    f.write(f"  Average Return Per Trade: {avg_return:.2f}\n")
                    
                    # Max drawdown approximation (if we had the equity curve)
                    f.write(f"  Risk-Reward Ratio: {results_copy['avg_profit_per_win'] / results_copy['avg_loss_per_loss']:.2f} (Higher is better)\n")
                    
                    # Expectancy
                    expectancy = (results_copy['win_rate'] / 100 * results_copy['avg_profit_per_win']) - \
                                 (results_copy['loss_rate'] / 100 * results_copy['avg_loss_per_loss'])
                    f.write(f"  Expectancy (Expected Return Per Trade): {expectancy:.2f}\n")
                
                # Summary of trade types
                if trade_log:
                    buy_trades = sum(1 for trade in trade_log if trade['type'] == 'buy')
                    sell_trades = sum(1 for trade in trade_log if trade['type'] == 'sell')
                    buy_wins = sum(1 for trade in trade_log if trade['type'] == 'buy' and trade['outcome'] == 'win')
                    sell_wins = sum(1 for trade in trade_log if trade['type'] == 'sell' and trade['outcome'] == 'win')
                    
                    f.write(f"\nTrade Type Analysis:\n")
                    f.write(f"  Buy Trades: {buy_trades} (Win Rate: {buy_wins / buy_trades * 100 if buy_trades > 0 else 0:.2f}%)\n")
                    f.write(f"  Sell Trades: {sell_trades} (Win Rate: {sell_wins / sell_trades * 100 if sell_trades > 0 else 0:.2f}%)\n")
            
            # Write trade log to CSV
            if trade_log:
                with open(csv_file, 'w', newline='') as csvfile:
                    fieldnames = [
                        'id', 'type', 'entry_date', 'exit_date', 'entry_price', 
                        'exit_price', 'stop_loss', 'take_profit', 'pnl', 'outcome', 'reason'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for trade in trade_log:
                        # Extract only the fields we want
                        row = {field: trade.get(field, '') for field in fieldnames}
                        writer.writerow(row)
            
            # Save full results to JSON for programmatic access
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Backtest results saved to {summary_file} and {csv_file}")
            return summary_file, csv_file, json_file
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            return None

    def start(self) -> None:
        """
        Start the trading bot loop with a continuous countdown on a single line 
        until the next candle. Runs the trading logic when the next candle is due or overdue.
        """
        logger.info("Starting trading bot loop...")
        try:
            while not self._shutdown:
                try:
                    # Calculate time until next candle
                    sleep_seconds = self.calculate_sleep_time()
                    self.run()
                    if sleep_seconds <= 0:
                        # Next candle is due or overdue, run immediately
                        logger.info("Next candle is due or overdue, running trading logic now...")
                        self.run()
                    else:
                        # Countdown until the next candle on a single line
                        remaining_seconds = sleep_seconds
                        while remaining_seconds > 0 and not self._shutdown:
                            # Use \r to overwrite the line with proper flushing for console display
                            sys.stdout.write(f"\rWaiting for next candle: {remaining_seconds:.2f} seconds remaining")
                            sys.stdout.flush()
                            time.sleep(min(1, remaining_seconds))  # Sleep for 1 second or less
                            remaining_seconds -= 1  # Decrement seconds
                            
                            # Periodically recalculate for better accuracy
                            if remaining_seconds % 30 == 0:
                                remaining_seconds = self.calculate_sleep_time()
                            
                    if not self._shutdown:
                        sys.stdout.write("\r" + " " * 80 + "\r")  # Clear the line
                        sys.stdout.flush()
                        logger.info("Countdown complete, running trading logic...")
                        self.run()

                except Exception as e:
                    sys.stdout.write("\r" + " " * 80 + "\r")  # Clear the line on error
                    sys.stdout.flush()
                    logger.error(f"Error in main loop: {str(e)}")
                    # Save trade data on error
                    self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
                    time.sleep(60)  # Brief pause on error before retrying
        finally:
            # Graceful shutdown procedure
            self.graceful_shutdown()
                
    def calculate_sleep_time(self) -> float:
        """
        Calculate the time to sleep until the next candle close plus a buffer.
        
        Returns:
            Number of seconds to sleep.
        """
        try:
            current_time = datetime.now(timezone.utc)
            timeframe_seconds = self.exchange.parse_timeframe(self.timeframe)
            seconds_since_epoch = current_time.timestamp()
            next_candle_seconds = ((seconds_since_epoch // timeframe_seconds) + 1) * timeframe_seconds
            next_candle_seconds = ((seconds_since_epoch // timeframe_seconds) + 1) * timeframe_seconds
            next_candle_time = datetime.fromtimestamp(next_candle_seconds, timezone.utc)
            # Add a 1-minute buffer for processing delay
            sleep_until = next_candle_time + timedelta(minutes=1)
            sleep_time = (sleep_until - current_time).total_seconds()
            logger.debug(f"Current time (UTC): {current_time}, Next candle time: {next_candle_time}, Sleep time: {sleep_time}")
            return max(sleep_time, 0)
        except Exception as e:
            logger.error(f"Error calculating sleep time: {str(e)}")
            return 60  # Default to 60 seconds if calculation fails

    @staticmethod
    def check_market_hours() -> bool:
        """
        Check if the market is open (crypto markets are 24/7, so always True for Luno).
        For traditional markets, this would check trading hours.
        
        Returns:
            True for crypto markets.
        """
        # This could be expanded to check maintenance windows or other exchange-specific downtime
        return True

    def shutdown(self) -> None:
        """
        Signal the bot to shutdown gracefully.
        """
        logger.info("Shutdown signal received. Stopping trading bot.")
        self._shutdown = True
    def graceful_shutdown(self) -> None:
        """
        Perform cleanup operations on shutdown.
        Save all trade data and notify about open positions.
        """
        logger.info("Performing graceful shutdown...")
        
        # Save all trade data
        self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
        
        # Notify about open positions
        if self.active_trades:
            open_trades_info = [f"{trade_id}: {trade.get('symbol')} {trade.get('side')} at {trade.get('entry_price')}" 
                               for trade_id, trade in self.active_trades.items()]
            open_trades_str = "\n".join(open_trades_info)
            shutdown_message = f"Trading bot shutting down with {len(self.active_trades)} open positions:\n{open_trades_str}"
            logger.warning(shutdown_message)
            self.notifier.send_message(shutdown_message)
        else:
            self.notifier.send_message("Trading bot shutting down with no open positions.")
            logger.info("Trading bot shutting down with no open positions.")
            
        logger.info("Shutdown complete.")