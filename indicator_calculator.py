from config import FIB_LEVELS
import ta
from config import logger
class IndicatorCalculator:
    def __init__(self):
        pass
    def apply_indicators(self, df):
        try:
            # Trend indicators
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)

            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)

            # Volume indicator
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])

            # Volatility indicators
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bollinger_upper'] = bollinger.bollinger_hband()
            df['bollinger_lower'] = bollinger.bollinger_lband()
            df['bollinger_mid'] = bollinger.bollinger_mavg()
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

            # Trend direction and strength
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
            df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)

            # Previous values for crosses
            df['prev_ema_20'] = df['ema_20'].shift(1)
            df['prev_ema_50'] = df['ema_50'].shift(1)
            df['prev_macd'] = df['macd'].shift(1)
            df['prev_macd_signal'] = df['macd_signal'].shift(1)
            df['prev_stoch_k'] = df['stoch_k'].shift(1)
            df['prev_stoch_d'] = df['stoch_d'].shift(1)

            # Fibonacci levels
            self.calc_fibonacci_levels(df)

            # Market regime
            df['high_volatility'] = df['atr'] > df['atr'].rolling(30).mean() * 1.5
            return df
        except Exception as e:
            logger.error(f"Error applying indicators: {e}")
            raise

    def calc_fibonacci_levels(self, df, lookback=60):
        df['fib_trend'] = 'neutral'
        for i in range(lookback, len(df)):
            section = df.iloc[i - lookback:i]
            highest_high = section['high'].max()
            lowest_low = section['low'].min()
            highest_idx = section['high'].idxmax()
            lowest_idx = section['low'].idxmin()
            if highest_idx > lowest_idx:
                diff = highest_high - lowest_low
                df.loc[df.index[i], 'fib_trend'] = 'uptrend'
                for level in FIB_LEVELS:
                    df.loc[df.index[i], f'fib_{str(level).replace(".", "_")}'] = highest_high - (diff * level)
            else:
                diff = highest_high - lowest_low
                df.loc[df.index[i], 'fib_trend'] = 'downtrend'
                for level in FIB_LEVELS:
                    df.loc[df.index[i], f'fib_{str(level).replace(".", "_")}'] = lowest_low + (diff * level)
        return df