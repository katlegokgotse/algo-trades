from datetime import datetime
import pandas as pd
from config import logger
class DataFetcher:
    def __init__(self, exchange):
        self.exchange = exchange

    def fetch_data(self, symbol, timeframe, limit=500):
        try:
            logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.to_csv(f'data/{symbol.replace("/", "_")}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None