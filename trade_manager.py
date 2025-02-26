from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from config import logger

class TradeManager:
    """
    Manages active trades, monitoring for stop loss and take profit conditions.
    Handles trade lifecycle including closing trades and updating trade history.
    """
    def __init__(self, exchange: Any, trade_executor: Optional[Any] = None):
        """
        Initialize the TradeManager.
        
        Args:
            exchange: The exchange instance for executing orders
            trade_executor: Optional TradeExecutor instance for order management and analysis
        """
        self.exchange = exchange
        self.trade_executor = trade_executor
    
    def update_trades(self, current_price: float, symbol: str, 
                      active_trades: Dict[str, Any], 
                      trade_history: List[Any], 
                      dry_run: bool) -> List[str]:
        """
        Check active trades and close any that have hit stop loss or take profit levels.
        
        Args:
            current_price: Current market price
            symbol: Trading symbol
            active_trades: Dictionary of active trades
            trade_history: List to store closed trades
            dry_run: If True, simulate orders instead of executing them
            
        Returns:
            List of trade IDs that were closed
        """
        trades_to_close = []
        
        # Identify trades that need to be closed
        for trade_id, trade in list(active_trades.items()):
            # Skip trades for other symbols
            if trade.symbol != symbol:
                continue
                
            if trade.side == 'buy':
                if current_price <= trade.stop_loss:
                    trades_to_close.append((trade_id, current_price, 'stop_loss'))
                elif current_price >= trade.take_profit:
                    trades_to_close.append((trade_id, current_price, 'take_profit'))
            else:  # sell side
                if current_price >= trade.stop_loss:
                    trades_to_close.append((trade_id, current_price, 'stop_loss'))
                elif current_price <= trade.take_profit:
                    trades_to_close.append((trade_id, current_price, 'take_profit'))
        
        # Close identified trades
        closed_trade_ids = []
        for trade_id, exit_price, exit_reason in trades_to_close:
            if self.close_trade(trade_id, exit_price, exit_reason, active_trades, trade_history, dry_run):
                closed_trade_ids.append(trade_id)
                # Analyze significant trade outcomes if available
                if self.trade_executor and trade_id in trade_history:
                    self.trade_executor.analyze_trade_outcome(trade_history[-1])
        
        return closed_trade_ids
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str, 
                    active_trades: Dict[str, Any], trade_history: List[Any], 
                    dry_run: bool) -> bool:
        """
        Close a specific trade.
        
        Args:
            trade_id: ID of the trade to close
            exit_price: Price at which to close the trade
            exit_reason: Reason for closing (stop_loss/take_profit)
            active_trades: Dictionary of active trades
            trade_history: List to store closed trades
            dry_run: If True, simulate orders instead of executing them
            
        Returns:
            True if trade was successfully closed, False otherwise
        """
        if trade_id not in active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades")
            return False
            
        trade = active_trades[trade_id]
        trade.close_trade(exit_price, datetime.now(), exit_reason)
        
        # Format PnL with percentage if available
        pnl_str = f"{trade.pnl:.2f} ({trade.pnl_percent:.2f}%)" if hasattr(trade, 'pnl_percent') else str(trade.pnl)
        logger.info(f"Closing trade {trade_id} at {exit_price}, PnL: {pnl_str}, Reason: {exit_reason}")
        
        if not dry_run:
            try:
                # Execute the closing order
                if trade.side == 'buy':
                    self.exchange.create_market_sell_order(trade.symbol, trade.quantity)
                else:
                    self.exchange.create_market_buy_order(trade.symbol, trade.quantity)
                
                # Cancel any remaining open orders
                self._cancel_pending_orders(trade.symbol)
                
            except Exception as e:
                logger.error(f"Failed to execute market order for trade {trade_id}: {e}")
                return False
                
        # Record trade history and remove from active trades
        trade_history.append(trade)
        del active_trades[trade_id]
        return True
    
    def _cancel_pending_orders(self, symbol: str) -> None:
        """
        Cancel all pending orders for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(order['id'], symbol)
                logger.info(f"Canceled pending order {order['id']} for {symbol}")
        except Exception as e:
            logger.error(f"Error canceling pending orders for {symbol}: {e}")