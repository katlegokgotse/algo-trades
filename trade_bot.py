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
        self.trade_manager = TradeManager()
        self.notifier = Notifier(enable_telegram, telegram_bot_token, telegram_chat_id)
        self.persistence_manager = PersistenceManager()
        self.persistence_manager.load_trade_data(self.active_trades, self.trade_history)
        
        # Flag for graceful shutdown
        self._shutdown = False

    def run(self) -> None:
        """
        Execute one iteration of the trading loop.
        """
        df = self.data_fetcher.fetch_data(self.symbol, self.timeframe)
        if df is None or len(df) < 200:
            logger.warning("Insufficient data")
            return
        
        # Apply indicators and generate signals
        df = self.indicator_calculator.apply_indicators(df)
        df = self.signal_generator.generate_signals(df)
        df = self.risk_manager.calculate_risk_reward(df, self.stop_loss_pct, self.take_profit_pct)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = latest['close']
        logger.info(f"Latest candle close: {current_price} at {df.index[-1]}")
        
        if not self.check_market_hours():
            logger.info("Market is likely closed")
            return
        
        # Update active trades before new signals are processed
        self.trade_manager.update_trades(current_price, self.symbol, self.active_trades, self.trade_history, self.dry_run)
        
        # Check for BUY signals
        if (latest['buy_signal'] and prev['buy_signal'] and 
            latest['signal_strength'] == 3 and prev['signal_strength'] == 3):
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
                    # Notify and persist trade
                    last_trade = self.active_trades[list(self.active_trades.keys())[-1]]
                    self.notifier.notify_trade(last_trade, trade_details)
                    self.persistence_manager.save_trade_data(self.active_trades, self.trade_history)
            else:
                logger.info("ChatGPT analysis did not recommend executing the BUY trade.")
        
        # Check for SELL signals
        elif (latest['sell_signal'] and prev['sell_signal'] and 
              latest['signal_strength'] == 3 and prev['signal_strength'] == 3):
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
            logger.info(f"No trade signal at {datetime.now()}")

    def backtest(self):
        """
        Run a backtest using historical data and return results.
        """
        df = self.data_fetcher.fetch_data(self.symbol, self.timeframe, limit=500)
        if df is None or len(df) < 200:
            logger.warning("Insufficient data for backtest")
            return None
        df = self.indicator_calculator.apply_indicators(df)
        df = self.signal_generator.generate_signals(df)
        df = self.risk_manager.calculate_risk_reward(df, self.stop_loss_pct, self.take_profit_pct)
        return self.run_backtest(df)

    def run_backtest(self, df: pd.DataFrame):
        """
        Backtest logic (to be implemented). Returns a dictionary with backtest results.
        """
        # (Insert your original backtesting logic here.)
        # For now, we'll return a dummy result.
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0}

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
        Check if the market is open (weekdays).
        
        Returns:
            True if market is open, False otherwise.
        """
        now = datetime.now()
        return now.weekday() < 5

    def shutdown(self) -> None:
        """
        Signal the bot to shutdown gracefully.
        """
        logger.info("Shutdown signal received. Stopping trading bot.")
        self._shutdown = True
