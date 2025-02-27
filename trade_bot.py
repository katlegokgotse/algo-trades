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
        
        # Instantiate modular components
        self.data_fetcher = DataFetcher(exchange)
        self.indicator_calculator = IndicatorCalculator()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.trade_executor = TradeExecutor(exchange, dry_run, max_trades)
        self.trade_manager = TradeManager(exchange=exchange)
        self.notifier = Notifier(enable_telegram, telegram_bot_token, telegram_chat_id)
        self.persistence_manager = PersistenceManager()
        self.persistence_manager.load_trade_data(self.active_trades, self.trade_history)
        
        # Flag for graceful shutdown
        self._shutdown = False

    def run(self) -> None:
        """
        Execute one iteration of the trading loop.
        """
        data_frame = self.data_fetcher.fetch_data(self.symbol, self.timeframe)
        if data_frame is None or len(data_frame) < 200:
            logger.warning("Insufficient data")
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
        
        # Update active trades before new signals are processed
        self.trade_manager.update_trades(current_price, self.symbol, self.active_trades, self.trade_history, self.dry_run)
        
        # Check for BUY signals
        if (latest['buy_signal'] and prev['buy_signal'] and 
            latest['signal_strength'] >= 2 and prev['signal_strength'] >= 2):
            logger.info("BUY signal detected with strength 3/3")
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
                "fib_trend": latest.get('fib_trend', 'N/A')
            }
            if chatgpt_analyze_trade(trade_details):
                if self.trade_executor.execute_trade_order('buy', self.symbol, current_price, 
                                                          latest['stop_loss_buy'], latest['take_profit_buy'], 
                                                          trade_details, self.active_trades, self.position_size):
                    last_trade = self.active_trades[list(self.active_trades.keys())[-1]]
                    self.notifier.notify_trade(last_trade, trade_details)
                    self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
            else:
                logger.info("ChatGPT analysis did not recommend executing the BUY trade.")
        
        # Check for SELL signals
        elif (latest['sell_signal'] and prev['sell_signal'] and 
              latest['signal_strength'] >= 2 and prev['signal_strength'] >= 2):
            logger.info("SELL signal detected with strength 3/3")
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
                "fib_trend": latest.get('fib_trend', 'N/A')
            }
            if chatgpt_analyze_trade(trade_details):
                if self.trade_executor.execute_trade_order('sell', self.symbol, current_price, 
                                                          latest['stop_loss_sell'], latest['take_profit_sell'], 
                                                          trade_details, self.active_trades, self.position_size):
                    last_trade = self.active_trades[list(self.active_trades.keys())[-1]]
                    self.notifier.notify_trade(last_trade, trade_details)
                    self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
            else:
                logger.info("ChatGPT analysis did not recommend executing the SELL trade.")
        else:
            logger.info(f"Update: ${current_price}")
            logger.info(f"Uncertain Market Conditions: No trade signal at {datetime.now()}")
            logger.info(f"{latest.get('fib_trend', 'N/A')}")
            logger.info(f"{self.symbol}")

    def backtest(self):
        """
        Run a backtest using historical data and return results.
        """
        data_frame = self.data_fetcher.fetch_data(self.symbol, self.timeframe, limit=500)
        if data_frame is None or len(data_frame) < 200:
            logger.warning("Insufficient data for backtest")
            return None
        data_frame = self.indicator_calculator.apply_indicators(data_frame)
        data_frame = self.signal_generator.generate_signals(data_frame)
        data_frame = self.risk_manager.calculate_risk_reward(data_frame, self.stop_loss_pct, self.take_profit_pct)
        return self.run_backtest(data_frame)

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

                # Check future prices to determine trade outcome
                trade_closed = False
                for j in range(i + 1, len(data_frame)):
                    future_price = data_frame.iloc[j]['close']
                    if future_price <= stop_loss:
                        loss = (entry_price - stop_loss) * self.position_size
                        total_loss += loss
                        losses += 1
                        trade_closed = True
                        break
                    elif future_price >= take_profit:
                        profit = (take_profit - entry_price) * self.position_size
                        total_profit += profit
                        wins += 1
                        trade_closed = True
                        break
                if not trade_closed:
                    logger.debug(f"Buy trade at index {i} did not reach TP or SL within data.")

            # Simulate SELL trade
            elif prev['sell_signal'] and prev['signal_strength'] >= 2:
                total_trades += 1
                entry_price = current_price
                stop_loss = prev['stop_loss_sell']
                take_profit = prev['take_profit_sell']

                # Check future prices to determine trade outcome
                trade_closed = False
                for j in range(i + 1, len(data_frame)):
                    future_price = data_frame.iloc[j]['close']
                    if future_price >= stop_loss:
                        loss = (stop_loss - entry_price) * self.position_size
                        total_loss += loss
                        losses += 1
                        trade_closed = True
                        break
                    elif future_price <= take_profit:
                        profit = (entry_price - take_profit) * self.position_size
                        total_profit += profit
                        wins += 1
                        trade_closed = True
                        break
                if not trade_closed:
                    logger.debug(f"Sell trade at index {i} did not reach TP or SL within data.")

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
            "win_rate": round(float(win_rate), 2),
            "loss_rate": round(float(loss_rate), 2),
            "profit_factor": round(float(profit_factor), 2),
            "total_profit_gross": round(float(total_profit), 2),  # Gross profit from wins
            "total_loss": round(float(total_loss), 2),            # Gross loss from losses (added explicitly)
            "total_profit_net": round(float(total_pnl), 2),       # Net profit (profit - loss)
            "avg_profit_per_win": round(float(avg_profit_per_win), 2),
            "avg_loss_per_loss": round(float(avg_loss_per_loss), 2))
        }

    def start(self) -> None:
        """
        Start the trading bot loop until a shutdown signal is received.
        """
        logger.info("Starting trading bot loop...")
        while not self._shutdown:
            try:
                self.run()
                sleep_seconds = self.calculate_sleep_time()
                if sleep_seconds > 0:
                    logger.info(f"Sleeping for {sleep_seconds:.2f} seconds until next candle")
                    time.sleep(sleep_seconds)
                else:
                    logger.info("Next candle close is in the past, running immediately")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

    def calculate_sleep_time(self) -> float:
        """
        Calculate the time to sleep until the next candle close plus a buffer.
        
        Returns:
            Number of seconds to sleep.
        """
        current_time = datetime.now(timezone.utc)
        timeframe_seconds = self.exchange.parse_timeframe(self.timeframe)
        seconds_since_epoch = current_time.timestamp()
        next_candle_seconds = ((seconds_since_epoch // timeframe_seconds) + 1) * timeframe_seconds
        next_candle_time = datetime.fromtimestamp(next_candle_seconds, timezone.utc)
        # Add a 1-minute buffer for processing delay
        sleep_until = next_candle_time + timedelta(minutes=1)
        sleep_time = (sleep_until - current_time).total_seconds()
        logger.debug(f"Current time: {current_time}, Next candle time: {next_candle_time}, Sleep time: {sleep_time}")
        return max(sleep_time, 0)

    @staticmethod
    def check_market_hours() -> bool:
        """
        Check if the market is open (crypto markets are 24/7, so always True for Luno).
        
        Returns:
            True for crypto markets.
        """
        return True

    def shutdown(self) -> None:
        """
        Signal the bot to shutdown gracefully.
        """
        logger.info("Shutdown signal received. Stopping trading bot.")
        self._shutdown = True