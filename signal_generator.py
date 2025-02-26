import pandas as pd

from config import FIB_LEVELS

class SignalGenerator:
    def generate_signals(self, df):
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_strength'] = 0
        buffer = 0.005

        # Trend conditions
        df['uptrend'] = (df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
        df['downtrend'] = (df['ema_20'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])

        # MACD crosses
        macd_threshold = 0.02 * df['close'].mean() / 10000
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['prev_macd'] <= df['prev_macd_signal']) & 
                               ((df['macd'] - df['macd_signal']).abs() > macd_threshold))
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['prev_macd'] >= df['prev_macd_signal']) & 
                                 ((df['macd'] - df['macd_signal']).abs() > macd_threshold))

        # Stochastic crosses
        df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (df['prev_stoch_k'] <= df['prev_stoch_d'])
        df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['prev_stoch_k'] >= df['prev_stoch_d'])

        # RSI thresholds
        df['rsi_strong_oversold'] = df['rsi'] < 20
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_strong_overbought'] = df['rsi'] > 80
        df['rsi_overbought'] = df['rsi'] > 70

        # Bollinger Bands
        df['near_lower_band'] = df['close'] <= df['bollinger_lower'] * 1.01
        df['near_upper_band'] = df['close'] >= df['bollinger_upper'] * 0.99

        # Trend strength
        df['strong_trend'] = df['adx'] > 30

        # VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']

        # Volume spike
        df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5

        # Fibonacci support/resistance
        df['at_fib_support'] = False
        df['at_fib_resistance'] = False
        for i in range(len(df)):
            row = df.iloc[i]
            if 'fib_0' not in row or pd.isna(row['fib_0']):
                continue
            current_price = row['close']
            for level in FIB_LEVELS:
                level_str = f'fib_{str(level).replace(".", "_")}'
                if level_str in row and not pd.isna(row[level_str]):
                    fib_level = row[level_str]
                    if (row['fib_trend'] == 'uptrend' and 
                        current_price >= fib_level * (1 - buffer) and 
                        current_price <= fib_level * (1 + buffer)):
                        df.loc[df.index[i], 'at_fib_support'] = True
                    elif (row['fib_trend'] == 'downtrend' and 
                          current_price >= fib_level * (1 - buffer) and 
                          current_price <= fib_level * (1 + buffer)):
                        df.loc[df.index[i], 'at_fib_resistance'] = True

        # Buy signals
        strong_buy = (df['uptrend'] & df['macd_cross_up'] & 
                      (df['rsi_strong_oversold'] | (df['rsi_oversold'] & df['stoch_cross_up'])) & 
                      df['strong_trend'] & df['near_lower_band'] & df['volume_spike'] & 
                      df['at_fib_support'])
        moderate_buy = (df['uptrend'] & df['macd_cross_up'] & df['rsi_oversold'] & 
                        df['above_vwap'] & df['stoch_cross_up'])

        # Sell signals (fixed: 'at_fib_resistance' for strong_sell)
        strong_sell = (df['downtrend'] & df['macd_cross_down'] & 
                       (df['rsi_strong_overbought'] | (df['rsi_overbought'] & df['stoch_cross_down'])) & 
                       df['strong_trend'] & df['near_upper_band'] & df['volume_spike'] & 
                       df['at_fib_resistance'])
        moderate_sell = (df['downtrend'] & df['macd_cross_down'] & df['rsi_overbought'] & 
                         df['below_vwap'] & df['stoch_cross_down'])

        # Assign signals
        df.loc[strong_buy, 'buy_signal'] = True
        df.loc[strong_buy, 'signal_strength'] = 3
        df.loc[moderate_buy & ~strong_buy, 'buy_signal'] = True
        df.loc[moderate_buy & ~strong_buy, 'signal_strength'] = 2

        df.loc[strong_sell, 'sell_signal'] = True
        df.loc[strong_sell, 'signal_strength'] = 3
        df.loc[moderate_sell & ~strong_sell, 'sell_signal'] = True
        df.loc[moderate_sell & ~strong_sell, 'signal_strength'] = 2

        # Volatility filter
        df.loc[df['high_volatility'] & (df['signal_strength'] < 3), 'buy_signal'] = False
        df.loc[df['high_volatility'] & (df['signal_strength'] < 3), 'sell_signal'] = False

        return df