from datetime import datetime
from trade import Trade
from config import logger
class TradeExecutor:
    def __init__(self, exchange, dry_run, max_trades):
        self.exchange = exchange
        self.dry_run = dry_run
        self.max_trades = max_trades

    def execute_trade_order(self, signal, symbol, price, stop_loss, take_profit, indicators, active_trades, position_size=0.01):
        if len(active_trades) >= self.max_trades:
            logger.warning(f"Maximum concurrent trades ({self.max_trades}) reached.")
            return False
        trade_id = f"{symbol}_{signal}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Placing {signal.upper()} order for {symbol} at {price}")
        trade = Trade(trade_id, symbol, signal, price, position_size, stop_loss, take_profit, datetime.now(), indicators)

        if not self.dry_run:
            if signal == 'buy':
                self.exchange.create_market_buy_order(symbol, position_size)
                if stop_loss:
                    self.exchange.create_order(symbol, 'stop', 'sell', position_size, None, {'stopPrice': stop_loss, 'reduceOnly': True})
                if take_profit:
                    self.exchange.create_order(symbol, 'limit', 'sell', position_size, take_profit, {'reduceOnly': True})
            elif signal == 'sell':
                self.exchange.create_market_sell_order(symbol, position_size)
                if stop_loss:
                    self.exchange.create_order(symbol, 'stop', 'buy', position_size, None, {'stopPrice': stop_loss, 'reduceOnly': True})
                if take_profit:
                    self.exchange.create_order(symbol, 'limit', 'buy', position_size, take_profit, {'reduceOnly': True})
        else:
            logger.info(f"DRY RUN: Would execute {signal.upper()} order for {symbol} at {price}")

        active_trades[trade_id] = trade
        return True