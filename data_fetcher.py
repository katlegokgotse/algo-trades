from datetime import datetime
import os
import pandas as pd
import ccxt
from config import logger

class DataFetcher:
    """
    DataFetcher is responsible for retrieving historical OHLCV data from a given exchange,
    converting timestamps, saving the data to a CSV file (optional), and returning a pandas DataFrame.
    """
    def __init__(self, exchange, timestamp_unit: str = 'ms') -> None:
        """
        Initialize the DataFetcher with a given ccxt exchange instance and timestamp unit.
        
        :param exchange: A ccxt exchange instance.
        :param timestamp_unit: Unit of the timestamp in fetched data (default 'ms' for milliseconds).
        """
        self.exchange = exchange
        self.timestamp_unit = timestamp_unit
        # Ensure the 'data' directory exists
        os.makedirs('data', exist_ok=True)

    def fetch_data(self, symbol: str, timeframe: str, limit: int = 500, save_to_file: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data from the exchange, save it to a CSV file (if specified), and return it as a DataFrame.
        
        :param symbol: Trading symbol (e.g., "BTC/USDT").
        :param timeframe: Candle interval (e.g., "15m", "1h").
        :param limit: Number of candles to fetch.
        :param save_to_file: Whether to save the data to a CSV file (default True).
        :return: A pandas DataFrame with the OHLCV data or None if an error occurs.
        """
        try:
            logger.info(f"Fetching up to {limit} {timeframe} candles for {symbol}")
            all_ohlcv = []
            since = None  # Start from the earliest available if None
            while len(all_ohlcv) < limit:
                remaining = limit - len(all_ohlcv)
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(remaining, 500))
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                if len(ohlcv) < min(remaining, 500):
                    break  # No more data available
                since = ohlcv[-1][0] + 1  # Next timestamp after the last fetched

            if not all_ohlcv:
                logger.warning(f"No data returned for {symbol} ({timeframe})")
                return None

            data_frame = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'], unit=self.timestamp_unit)
            data_frame.set_index('timestamp', inplace=True)

            # Log the date range of fetched data
            if not data_frame.empty:
                logger.info(f"Fetched {len(data_frame)} candles from {data_frame.index[0]} to {data_frame.index[-1]}")
            else:
                logger.warning(f"No data fetched for {symbol} ({timeframe})")

            # Save to CSV if requested
            if save_to_file:
                filename = os.path.join('data', f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                data_frame.to_csv(filename)
                logger.info(f"Data saved to {filename}")

            return data_frame

        except ccxt.NetworkError as e:
            logger.error(f"Network issue fetching {symbol} ({timeframe}): {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error for {symbol} ({timeframe}): {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol} ({timeframe}): {e}")
            return None