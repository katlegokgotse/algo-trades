from datetime import datetime


class TradeManager:
    def update_trades(self, current_price, symbol, active_trades, trade_history, dry_run):
        trades_to_close = []
        for trade_id, trade in active_trades.items():
            if trade.symbol != symbol:
                continue
            if trade.side == 'buy':
                if current_price <= trade.stop_loss:
                    trades_to_close.append((trade_id, current_price, 'stop_loss'))
                elif current_price >= trade.take_profit:
                    trades_to_close.append((trade_id, current_price, 'take_profit'))
            else:
                if current_price >= trade.stop_loss:
                    trades_to_close.append((trade_id, current_price, 'stop_loss'))
                elif current_price <= trade.take_profit:
                    trades_to_close.append((trade_id, current_price, 'take_profit'))
        for trade_id, exit_price, exit_reason in trades_to_close:
            self.close_trade(trade_id, exit_price, exit_reason, active_trades, trade_history, dry_run)

    def close_trade(self, trade_id, exit_price, exit_reason, active_trades, trade_history, dry_run):
        if trade_id not in active_trades:
            logger.warning(f"Trade {trade_id} not found")
            return False
        trade = active_trades[trade_id]
        trade.close_trade(exit_price, datetime.now(), exit_reason)
        logger.info(f"Closing trade {trade_id} at {exit_price}, PnL: {trade.pnl}")
        if not dry_run:
            if trade.side == 'buy':
                exchange.create_market_sell_order(trade.symbol, trade.quantity)
            else:
                exchange.create_market_buy_order(trade.symbol, trade.quantity)
        trade_history.append(trade)
        del active_trades[trade_id]
        return True