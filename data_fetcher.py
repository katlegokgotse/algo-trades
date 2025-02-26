from datetime import datetime
import os
import pandas as pd
from config import logger

class DataFetcher:
    """
    DataFetcher is responsible for retrieving historical OHLCV data from a given exchange,
    converting timestamps, saving the data to a CSV file, and returning a pandas DataFrame.
    """
    def __init__(self, exchange) -> None:
        """
        Initialize the DataFetcher with a given ccxt exchange instance.
        
        :param exchange: A ccxt exchange instance.
        """
        self.exchange = exchange
        # Ensure the 'data' directory exists
        os.makedirs('data', exist_ok=True)

    def fetch_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data from the exchange, save it to a CSV file, and return it as a DataFrame.
        
        :param symbol: Trading symbol (e.g., "BTC/USDT").
        :param timeframe: Candle interval (e.g., "15m", "1h").
        :param limit: Number of candles to fetch.
        :return: A pandas DataFrame with the OHLCV data or None if an error occurs.
        """
        try:
            logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            data_frame = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'], unit='ms')
            data_frame.set_index('timestamp', inplace=True)

            # Create a filename using symbol, timeframe, and current timestamp
            filename = os.path.join('data', f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            data_frame.to_csv(filename)
            logger.info(f"Data saved to {filename}")
            return data_frame
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} on timeframe {timeframe}: {e}")
            return None
