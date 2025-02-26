import json
import os
from trade import Trade
from config import logger
class PersistenceManager:
    def save_trade_data(self, active_trades, trade_history):
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/trade_history.json', 'w') as f:
                json.dump([trade.to_dict() for trade in trade_history], f, indent=2)
            with open('data/active_trades.json', 'w') as f:
                json.dump({k: v.to_dict() for k, v in active_trades.items()}, f, indent=2)
            logger.info("Trade data saved")
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")

    def load_trade_data(self, active_trades, trade_history):
        try:
            if os.path.exists('data/trade_history.json'):
                with open('data/trade_history.json', 'r') as f:
                    trade_data = json.load(f)
                    for trade_dict in trade_data:
                        trade = Trade(**{k: v for k, v in trade_dict.items() if k != 'exit_price'})
                        if trade_dict['exit_price']:
                            trade.close_trade(trade_dict['exit_price'], trade_dict['exit_time'], trade_dict['exit_reason'])
                        trade_history.append(trade)
            if os.path.exists('data/active_trades.json'):
                with open('data/active_trades.json', 'r') as f:
                    active_trades.update({
                        trade_id: Trade(**trade_dict)
                        for trade_id, trade_dict in json.load(f).items()
                    })
            logger.info(f"Loaded {len(trade_history)} historical trades, {len(active_trades)} active trades")
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")