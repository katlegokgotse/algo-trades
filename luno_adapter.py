from time import time
from config import logger
class LunoExchangeAdapter:
    def __init__(self, client):
        self.client = client
        self.timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }

    def get_ticker(self, pair):
        return self.client.get_ticker(pair=pair)

    def get_all_tickers(self):
        """Fetch all available tickers from Luno."""
        return self.client.get_tickers()

    def parse_timeframe(self, timeframe):
        """Convert timeframe string to seconds."""
        return self.timeframe_map.get(timeframe, 3600)  # Default to 1 hour if not found

    def fetch_ohlcv(self, pair, timeframe, limit=500):
        """Fetch OHLCV data (placeholder)."""
        logger.warning("Luno OHLCV fetching not fully implemented. Using ticker data as placeholder.")
        ticker = self.get_ticker(pair)
        current_time = int(time.time() * 1000)
        return [[current_time - i * 1000, float(ticker['last_trade']), float(ticker['last_trade']), 
                 float(ticker['last_trade']), float(ticker['last_trade']), 0] for i in range(limit)]
