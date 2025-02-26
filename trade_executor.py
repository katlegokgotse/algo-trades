from datetime import datetime
from typing import Dict, Any, Literal, Optional, Tuple
from trade import Trade
from config import logger
from concurrent.futures import ThreadPoolExecutor
import time

class TradeExecutor:
    """
    Executes trade orders via a given exchange and tracks trades.
    Optimized for performance, reliability, and error handling.
    """
    def __init__(self, exchange: Any, dry_run: bool, max_trades: int, retry_attempts: int = 3) -> None:
        """
        Initialize the TradeExecutor.
        
        Args:
            exchange: A ccxt exchange instance.
            dry_run: If True, orders will not be actually executed.
            max_trades: Maximum number of concurrent trades allowed.
            retry_attempts: Number of times to retry failed orders before giving up.
        """
        self.exchange = exchange
        self.dry_run = dry_run
        self.max_trades = max_trades
        self.retry_attempts = retry_attempts
        self.thread_pool = ThreadPoolExecutor(max_workers=3)  # For asynchronous order verification
        self._order_cache: Dict[str, float] = {}  # Cache to prevent duplicate order submission

    def execute_trade_order(self, 
                            signal: Literal['buy', 'sell'], 
                            symbol: str, 
                            price: float, 
                            stop_loss: Optional[float], 
                            take_profit: Optional[float], 
                            indicators: Dict[str, Any],
                            active_trades: Dict[str, Trade],
                            position_size: float = 0.01) -> bool:
        """
        Execute a trade order if the maximum trade limit has not been reached.
        
        Args:
            signal: 'buy' or 'sell'.
            symbol: Trading symbol (e.g., "BTC/USDT").
            price: Current price.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            indicators: Dictionary containing indicator values.
            active_trades: Dictionary tracking active trades.
            position_size: Size of the position to trade.
        
        Returns:
            True if the trade order is (or would be) executed successfully; False otherwise.
        """
        signal = signal.lower()
        if not self._validate_inputs(signal, symbol, price, stop_loss, take_profit, position_size):
            return False

        if len(active_trades) >= self.max_trades:
            logger.warning(f"Maximum concurrent trades ({self.max_trades}) reached.")
            return False

        timestamp = datetime.now()
        trade_id = f"{symbol}_{signal}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        cache_key = f"{symbol}_{signal}_{price:.2f}"
        if cache_key in self._order_cache and time.time() - self._order_cache[cache_key] < 60:
            logger.warning(f"Duplicate order detected for {symbol} {signal.upper()} at {price:.2f}")
            return False

        logger.info(f"Placing {signal.upper()} order for {symbol} at {price:.2f}")
        trade = Trade(trade_id, symbol, signal, price, position_size, stop_loss, take_profit, timestamp, indicators.copy())

        if self.dry_run:
            logger.info(f"DRY RUN: Would execute {signal.upper()} order for {symbol} at {price:.2f}")
            active_trades[trade_id] = trade
            self._order_cache[cache_key] = time.time()
            return True

        success = self._execute_order_with_retry(signal, symbol, price, stop_loss, take_profit, position_size)
        if success:
            active_trades[trade_id] = trade
            self._order_cache[cache_key] = time.time()
            logger.info(f"Order executed for trade {trade_id}")
            # Asynchronously verify the order status
            self.thread_pool.submit(self._verify_order_execution, trade_id, symbol, signal)
            return True
        else:
            logger.error(f"Failed to execute {signal.upper()} order for {symbol} after {self.retry_attempts} attempts")
            return False

    def _validate_inputs(self, signal: str, symbol: str, price: float, 
                         stop_loss: Optional[float], take_profit: Optional[float], 
                         position_size: float) -> bool:
        """Validate trade inputs."""
        if signal not in ['buy', 'sell']:
            logger.error(f"Invalid signal: {signal}. Must be 'buy' or 'sell'.")
            return False

        if price <= 0 or position_size <= 0:
            logger.error(f"Invalid price ({price}) or position size ({position_size}).")
            return False

        if stop_loss is not None:
            if (signal == 'buy' and stop_loss >= price) or (signal == 'sell' and stop_loss <= price):
                logger.error(f"Invalid stop loss ({stop_loss}) for {signal.upper()} at price {price}.")
                return False

        if take_profit is not None:
            if (signal == 'buy' and take_profit <= price) or (signal == 'sell' and take_profit >= price):
                logger.error(f"Invalid take profit ({take_profit}) for {signal.upper()} at price {price}.")
                return False

        return True

    def _execute_order_with_retry(self, signal: str, symbol: str, price: float, 
                                  stop_loss: Optional[float], take_profit: Optional[float], 
                                  position_size: float) -> bool:
        """Execute the order with retry logic."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._place_orders(signal, symbol, position_size, stop_loss, take_profit)
                return True
            except Exception as e:
                if attempt < self.retry_attempts:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Order failed after {attempt} attempts: {e}")
        return False

    def _place_orders(self, signal: str, symbol: str, position_size: float, 
                      stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        """Place the main market order and conditional orders."""
        if signal == 'buy':
            self.exchange.create_market_buy_order(symbol, position_size)
            if stop_loss:
                self.exchange.create_order(symbol, 'stop', 'sell', position_size, None, 
                                             {'stopPrice': stop_loss, 'reduceOnly': True})
            if take_profit:
                self.exchange.create_order(symbol, 'limit', 'sell', position_size, take_profit, 
                                             {'reduceOnly': True})
        elif signal == 'sell':
            self.exchange.create_market_sell_order(symbol, position_size)
            if stop_loss:
                self.exchange.create_order(symbol, 'stop', 'buy', position_size, None, 
                                             {'stopPrice': stop_loss, 'reduceOnly': True})
            if take_profit:
                self.exchange.create_order(symbol, 'limit', 'buy', position_size, take_profit, 
                                             {'reduceOnly': True})

    def _verify_order_execution(self, trade_id: str, symbol: str, signal: str) -> None:
        """Asynchronously verify that the order was executed correctly."""
        try:
            time.sleep(5)  # Allow time for order processing
            position = self.exchange.fetch_position(symbol)
            if position:
                expected_side = 'long' if signal == 'buy' else 'short'
                if position.get('side') != expected_side:
                    logger.warning(f"Trade {trade_id} position mismatch: expected {expected_side}, got {position.get('side')}")
            else:
                logger.warning(f"No position found for trade {trade_id} after execution.")
        except Exception as e:
            logger.error(f"Error verifying order execution for trade {trade_id}: {e}")

    def cleanup(self) -> None:
        """Clean up resources (shutdown thread pool)."""
        self.thread_pool.shutdown(wait=False)

    def get_market_order_fee(self, symbol: str) -> float:
        """Retrieve the market order fee for a symbol."""
        try:
            market = self.exchange.market(symbol)
            return market.get('taker', 0.001)
        except Exception as e:
            logger.warning(f"Error retrieving fee for {symbol}: {e}. Using default 0.1% fee.")
            return 0.001

    def calculate_order_size(self, symbol: str, account_percentage: float) -> Tuple[float, float]:
        """
        Calculate the order size based on a percentage of the available account balance.
        
        Args:
            symbol: Trading symbol.
            account_percentage: Percentage of the account to use (0-100).
            
        Returns:
            A tuple (position_size, usd_value).
        """
        try:
            balance = self.exchange.fetch_balance()
            available = balance['free'].get('USDT', 0)
            ticker = self.exchange.fetch_ticker(symbol)
            usd_value = available * (account_percentage / 100)
            position_size = usd_value / ticker['last']
            market = self.exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if position_size < min_amount:
                logger.warning(f"Calculated position size {position_size} is below minimum {min_amount}")
                return (0, 0)
            return (position_size, usd_value)
        except Exception as e:
            logger.error(f"Error calculating order size for {symbol}: {e}")
            return (0, 0)
