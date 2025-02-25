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
