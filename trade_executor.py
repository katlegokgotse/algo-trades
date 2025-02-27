from datetime import datetime
import random
from typing import Dict, Any, Literal, Optional, Tuple
from trade import Trade
from config import logger
from concurrent.futures import ThreadPoolExecutor
import time
import openai
from cachetools import TTLCache

class TradeExecutor:
    """
    Executes trade orders via a given exchange and tracks trades.
    Optimized for performance, reliability, and error handling.
    Integrates ChatGPT for selective post-trade analysis.
    """
    def __init__(self, exchange: Any, dry_run: bool, max_trades: int, retry_attempts: int = 3, cache_ttl: int = 60) -> None:
        """
        Initialize the TradeExecutor.

        Args:
            exchange: A ccxt exchange instance.
            dry_run: If True, orders are simulated.
            max_trades: Max concurrent trades allowed.
            retry_attempts: Number of retries for failed orders.
            cache_ttl: Time-to-live (seconds) for order cache.
        """
        self.exchange = exchange
        self.dry_run = dry_run
        self.max_trades = max_trades
        self.retry_attempts = retry_attempts
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self._order_cache = TTLCache(maxsize=100, ttl=cache_ttl)  # Configurable TTL
        self._fee_cache = TTLCache(maxsize=100, ttl=3600)  # Cache fees for 1 hour

    def execute_trade_order(self, 
                            signal: Literal['buy', 'sell'], 
                            symbol: str, 
                            price: float, 
                            stop_loss: Optional[float], 
                            take_profit: Optional[float], 
                            indicators: Dict[str, Any],
                            active_trades: Dict[str, Trade],
                            position_size: float = 0.01) -> bool:
        """Execute a trade order with enhanced checks."""
        signal = signal.lower()
        if not self._validate_inputs(signal, symbol, price, stop_loss, take_profit, position_size):
            return False

        if len(active_trades) >= self.max_trades:
            logger.warning(f"Max concurrent trades ({self.max_trades}) reached.")
            return False

        timestamp = datetime.now()
        trade_id = f"{symbol}_{signal}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        cache_key = f"{symbol}_{signal}_{price:.2f}"
        if cache_key in self._order_cache:
            logger.warning(f"Duplicate order detected for {symbol} {signal.upper()} at {price:.2f}")
            return False

        logger.info(f"Placing {signal.upper()} order for {symbol} at {price:.2f}")
        trade = Trade(trade_id, symbol, signal, price, position_size, stop_loss, take_profit, timestamp, indicators.copy())

        if self.dry_run:
            logger.info(f"DRY RUN: Simulated {signal.upper()} order for {symbol} at {price:.2f}")
            active_trades[trade_id] = trade
            self._order_cache[cache_key] = time.time()
            return True

        success = self._execute_order_with_retry(signal, symbol, price, stop_loss, take_profit, position_size)
        if success:
            active_trades[trade_id] = trade
            self._order_cache[cache_key] = time.time()
            logger.info(f"Order executed for trade {trade_id}")
            self.thread_pool.submit(self._verify_order_execution, trade_id, symbol, signal)
            return True
        logger.error(f"Failed to execute {signal.upper()} order for {symbol} after {self.retry_attempts} attempts")
        return False

    def _validate_inputs(self, signal: str, symbol: str, price: float, 
                         stop_loss: Optional[float], take_profit: Optional[float], 
                         position_size: float) -> bool:
        """Validate trade inputs with symbol check."""
        if signal not in ['buy', 'sell']:
            logger.error(f"Invalid signal: {signal}. Must be 'buy' or 'sell'.")
            return False
        if symbol not in self.exchange.markets:
            logger.error(f"Invalid symbol: {symbol}")
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
                                  position_size: float, retry_attempts: int = 3) -> bool:
        """Execute order with capped retry delays."""
        for attempt in range(1, retry_attempts + 1):
            try:
                self._place_orders(signal, symbol, price, position_size, stop_loss, take_profit)
                logger.info(f"Order executed successfully for {symbol}")
                return True
            except Exception as e:
                if attempt < self.retry_attempts:
                    base_wait = min(2 ** attempt, 60)  # Cap at 60s
                    wait_time = base_wait + random.uniform(0, base_wait * 0.5) 
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Order failed after {attempt} attempts: {e}")
                    return False

    def _place_orders(self, signal: str, symbol: str, position_size: float, 
                      stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        """Place orders, using OCO if supported."""
        oco_supported = getattr(self.exchange, 'has', {}).get('createOcoOrder', False)
        if oco_supported and stop_loss and take_profit:
            side = 'buy' if signal == 'buy' else 'sell'
            opposite_side = 'sell' if signal == 'buy' else 'buy'
            self.exchange.create_market_order(symbol, side, position_size)
            self.exchange.create_oco_order(symbol, opposite_side, position_size, take_profit, stop_loss, {'reduceOnly': True})
        else:
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
        """Verify order execution with polling."""
        max_wait = 30  # Configurable timeout
        start_time = time.time()
        expected_side = 'long' if signal == 'buy' else 'short'
        while time.time() - start_time < max_wait:
            try:
                position = self.exchange.fetch_position(symbol)
                if position and position.get('side') == expected_side:
                    return
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error verifying trade {trade_id}: {e}")
        logger.warning(f"Trade {trade_id} not confirmed after {max_wait}s")

    def analyze_trade_outcome(self, trade: Trade) -> None:
        """Analyze significant trade outcomes using ChatGPT."""
        if trade.pnl_percent is None or abs(trade.pnl_percent) < 1:  # Example threshold
            return
        prompt = f"""
You are a trading performance evaluator.
Trade ID: {trade.trade_id}
Symbol: {trade.symbol}
Side: {trade.side.upper()}
Entry Price: {trade.entry_price}
Exit Price: {trade.exit_price}
PnL: {trade.pnl}
PnL Percentage: {trade.pnl_percent:.2f}%
Indicators: {trade.indicators}

Respond with "REWARD" (good outcome) or "PUNISH" (bad outcome).
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a trading performance evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip().upper()
            if answer in ["REWARD", "PUNISH"]:
                logger.info(f"ChatGPT analysis for {trade.trade_id}: {answer}")
            else:
                logger.warning(f"Unexpected ChatGPT response: {answer}")
        except Exception as e:
            logger.error(f"Error analyzing trade {trade.trade_id}: {e}")

    def cleanup(self) -> None:
        """Clean up resources properly."""
        self.thread_pool.shutdown(wait=True)

    def get_market_order_fee(self, symbol: str) -> float:
        """Retrieve cached market order fee."""
        if symbol in self._fee_cache:
            return self._fee_cache[symbol]
        try:
            market = self.exchange.market(symbol)
            fee = market.get('taker', 0.001)
            self._fee_cache[symbol] = fee
            return fee
        except Exception as e:
            logger.warning(f"Error retrieving fee for {symbol}: {e}. Defaulting to 0.1%.")
            return 0.001

    def calculate_order_size(self, symbol: str, account_percentage: float) -> Tuple[float, float]:
        """Calculate precise order size."""
        try:
            balance = self.exchange.fetch_balance()
            market = self.exchange.market(symbol)
            quote = market['quote']
            available = balance['free'].get(quote, 0)
            ticker = self.exchange.fetch_ticker(symbol)
            usd_value = available * (account_percentage / 100)
            position_size = usd_value / ticker['last']
            position_size = self.exchange.amount_to_precision(symbol, position_size)
            min_amount = market['limits']['amount']['min']
            min_cost = market['limits'].get('cost', {}).get('min', 0)
            if position_size < min_amount or (usd_value < min_cost and min_cost > 0):
                logger.warning(f"Position size {position_size} or cost {usd_value} below minimum.")
                return (0, 0)
            return (float(position_size), usd_value)
        except Exception as e:
            logger.error(f"Error calculating order size for {symbol}: {e}")
            return (0, 0)