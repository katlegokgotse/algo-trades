import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class SignalGenerator:
    """Generates trading signals based on technical indicators."""

    DEFAULT_FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    DEFAULT_RSI_THRESHOLDS = {'strong_oversold': 20, 'oversold': 30, 'overbought': 70, 'strong_overbought': 80}

    def __init__(
        self,
        fib_levels: Optional[List[float]] = None,
        macd_threshold_factor: float = 0.02,
        volume_spike_factor: float = 1.5,
        fib_buffer: float = 0.005,
        rsi_thresholds: Optional[Dict[str, int]] = None,
        adx_threshold: int = 30,
    ):
        """Initialize with configurable parameters."""
        self.fib_levels = fib_levels or self.DEFAULT_FIB_LEVELS
        self.macd_threshold_factor = macd_threshold_factor
        self.volume_spike_factor = volume_spike_factor
        self.fib_buffer = fib_buffer
        self.rsi_thresholds = rsi_thresholds or self.DEFAULT_RSI_THRESHOLDS
        self.adx_threshold = adx_threshold

    def _validate_columns(self, data_frame: pd.DataFrame, required_columns: List[str]) -> None:
        """Raise error if required columns are missing."""
        missing = [col for col in required_columns if col not in data_frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

    def _identify_trends(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Identify trends using EMA relationships."""
        self._validate_columns(data_frame, ['ema_20', 'ema_50', 'ema_200'])
        data_frame['uptrend'] = (data_frame['ema_20'] > data_frame['ema_50']) & (data_frame['ema_50'] > data_frame['ema_200'])
        data_frame['downtrend'] = (data_frame['ema_20'] < data_frame['ema_50']) & (data_frame['ema_50'] < data_frame['ema_200'])
        data_frame['trend_strength'] = np.abs((data_frame['ema_20'] - data_frame['ema_50']) / data_frame['ema_50'] * 100)
        return data_frame

    def _identify_macd_signals(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Identify MACD crossover signals."""
        self._validate_columns(data_frame, ['macd', 'macd_signal', 'prev_macd', 'prev_macd_signal', 'close'])
        threshold = self.macd_threshold_factor * data_frame['close'].mean() / 10000
        data_frame['macd_cross_up'] = (
            (data_frame['macd'] > data_frame['macd_signal']) &
            (data_frame['prev_macd'] <= data_frame['prev_macd_signal']) &
            (np.abs(data_frame['macd'] - data_frame['macd_signal']) > threshold)
        )
        data_frame['macd_cross_down'] = (
            (data_frame['macd'] < data_frame['macd_signal']) &
            (data_frame['prev_macd'] >= data_frame['prev_macd_signal']) &
            (np.abs(data_frame['macd'] - data_frame['macd_signal']) > threshold)
        )
        data_frame['macd_histogram'] = data_frame['macd'] - data_frame['macd_signal']
        data_frame['macd_histogram_increasing'] = data_frame['macd_histogram'] > data_frame['macd_histogram'].shift(1)
        return data_frame

    def _identify_oscillator_signals(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Identify RSI and Stochastic signals."""
        has_stoch = all(col in data_frame.columns for col in ['stoch_k', 'stoch_d', 'prev_stoch_k', 'prev_stoch_d'])
        has_rsi = 'rsi' in data_frame.columns

        if has_stoch:
            data_frame['stoch_cross_up'] = (data_frame['stoch_k'] > data_frame['stoch_d']) & (data_frame['prev_stoch_k'] <= data_frame['prev_stoch_d'])
            data_frame['stoch_cross_down'] = (data_frame['stoch_k'] < data_frame['stoch_d']) & (data_frame['prev_stoch_k'] >= data_frame['prev_stoch_d'])
            data_frame['stoch_oversold'] = data_frame['stoch_k'] < 20
            data_frame['stoch_overbought'] = data_frame['stoch_k'] > 80

        if has_rsi:
            data_frame['rsi_strong_oversold'] = data_frame['rsi'] < self.rsi_thresholds['strong_oversold']
            data_frame['rsi_oversold'] = data_frame['rsi'] < self.rsi_thresholds['oversold']
            data_frame['rsi_overbought'] = data_frame['rsi'] > self.rsi_thresholds['overbought']
            data_frame['rsi_strong_overbought'] = data_frame['rsi'] > self.rsi_thresholds['strong_overbought']
            data_frame['rsi_rising'] = data_frame['rsi'] > data_frame['rsi'].shift(1)
            data_frame['rsi_falling'] = data_frame['rsi'] < data_frame['rsi'].shift(1)

        return data_frame

    def _identify_price_patterns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Identify price patterns (Bollinger, VWAP, volume, ADX)."""
        if all(col in data_frame.columns for col in ['close', 'bollinger_lower', 'bollinger_upper']):
            data_frame['near_lower_band'] = data_frame['close'] <= data_frame['bollinger_lower'] * 1.01
            data_frame['near_upper_band'] = data_frame['close'] >= data_frame['bollinger_upper'] * 0.99
            data_frame['bollinger_squeeze'] = (data_frame['bollinger_upper'] - data_frame['bollinger_lower']) / data_frame['close'] < 0.03

        if 'vwap' in data_frame.columns and 'close' in data_frame.columns:
            data_frame['above_vwap'] = data_frame['close'] > data_frame['vwap']
            data_frame['below_vwap'] = data_frame['close'] < data_frame['vwap']

        if 'volume' in data_frame.columns:
            data_frame['volume_spike'] = data_frame['volume'] > data_frame['volume'].rolling(20).mean() * self.volume_spike_factor
            data_frame['volume_increasing'] = data_frame['volume'] > data_frame['volume'].shift(1)

        if 'adx' in data_frame.columns:
            data_frame['strong_trend'] = data_frame['adx'] > self.adx_threshold

        return data_frame

    def _process_fibonacci_levels(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Identify Fibonacci support/resistance levels."""
        data_frame['at_fib_support'] = False
        data_frame['at_fib_resistance'] = False

        fib_columns = [f'fib_{str(level).replace(".", "_")}' for level in self.fib_levels]
        if not ('fib_trend' in data_frame.columns and any(col in data_frame.columns for col in fib_columns)):
            return data_frame

        for i, row in data_frame.iterrows():
            if pd.isna(row.get('fib_0')):
                continue
            price = row['close']
            for level in self.fib_levels:
                level_key = f'fib_{str(level).replace(".", "_")}'
                if level_key in row and not pd.isna(row[level_key]):
                    fib_level = row[level_key]
                    if fib_level * (1 - self.fib_buffer) <= price <= fib_level * (1 + self.fib_buffer):
                        if row['fib_trend'] == 'uptrend':
                            data_frame.at[i, 'at_fib_support'] = True
                        elif row['fib_trend'] == 'downtrend':
                            data_frame.at[i, 'at_fib_resistance'] = True
        return data_frame

    def _calculate_signal_strength(self, row: pd.Series, signal_type: str) -> float:
        """Calculate signal strength (0-5) based on confirming factors."""
        strength = 0.0
        if signal_type == 'buy':
            strength += 1 if row.get('uptrend', False) and row.get('strong_trend', False) else 0
            strength += 1 if row.get('rsi_strong_oversold', False) else (0.5 if row.get('rsi_oversold', False) else 0)
            strength += 1 if row.get('macd_cross_up', False) else 0
            strength += 1 if row.get('stoch_cross_up', False) and row.get('stoch_oversold', False) else 0
            strength += 0.5 if row.get('near_lower_band', False) else 0
            strength += 0.5 if row.get('volume_spike', False) else 0
            strength += 1 if row.get('at_fib_support', False) else 0
        elif signal_type == 'sell':
            strength += 1 if row.get('downtrend', False) and row.get('strong_trend', False) else 0
            strength += 1 if row.get('rsi_strong_overbought', False) else (0.5 if row.get('rsi_overbought', False) else 0)
            strength += 1 if row.get('macd_cross_down', False) else 0
            strength += 1 if row.get('stoch_cross_down', False) and row.get('stoch_overbought', False) else 0
            strength += 0.5 if row.get('near_upper_band', False) else 0
            strength += 0.5 if row.get('volume_spike', False) else 0
            strength += 1 if row.get('at_fib_resistance', False) else 0
        return min(round(strength * 2) / 2, 5)

    def generate_signals(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from technical indicators."""
        data_frame = data_frame.copy()
        data_frame['buy_signal'] = False
        data_frame['sell_signal'] = False
        data_frame['signal_strength'] = 0
        data_frame['signal_type'] = None

        data_frame = self._identify_trends(data_frame)
        data_frame = self._identify_macd_signals(data_frame)
        data_frame = self._identify_oscillator_signals(data_frame)
        data_frame = self._identify_price_patterns(data_frame)
        data_frame = self._process_fibonacci_levels(data_frame)

        print("Debug Info:")
        print(f"Trend: Up={data_frame['uptrend'].iloc[-1]}, Down={data_frame['downtrend'].iloc[-1]}")
        print(f"MACD: {data_frame['macd_cross_up'].iloc[-1]}, {data_frame['macd_cross_down'].iloc[-1]}")
        print(f"RSI: {data_frame['rsi'].iloc[-1]}, ADX: {data_frame['adx'].iloc[-1]}")
        print(f"Fib: Support={data_frame['at_fib_support'].iloc[-1]}, Resistance={data_frame['at_fib_resistance'].iloc[-1]}")

        for i, row in data_frame.iterrows():
            buy_strength = self._calculate_signal_strength(row, 'buy')
            data_frame.at[i, 'signal_strength'] = int(buy_strength)
            if buy_strength >= 1:
                data_frame.at[i, 'buy_signal'] = True
                data_frame.at[i, 'signal_type'] = 'buy'
            else:
                sell_strength = self._calculate_signal_strength(row, 'sell')
                data_frame.at[i, 'signal_strength'] = int(sell_strength)
                if sell_strength >= 1:
                    data_frame.at[i, 'sell_signal'] = True
                    data_frame.at[i, 'signal_type'] = 'sell'

        print(f"Final: Buy={data_frame['buy_signal'].iloc[-1]}, Sell={data_frame['sell_signal'].iloc[-1]}, Strength={data_frame['signal_strength'].iloc[-1]}")

        if 'high_volatility' in data_frame.columns:
            mask = data_frame['high_volatility'] & (data_frame['signal_strength'] < 2.5)
            data_frame.loc[mask, ['buy_signal', 'sell_signal', 'signal_type']] = [False, False, None]

        return data_frame

    def analyze_performance(self, data_frame: pd.DataFrame, lookback_periods: int = 50) -> tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """Analyze signal performance over a lookback period."""
        self._validate_columns(data_frame, ['buy_signal', 'sell_signal', 'close'])

        def calculate_returns(signals: pd.Series, is_buy: bool) -> List[Dict]:
            indices = data_frame.index[signals].tolist()
            returns = []
            for idx in indices:
                pos = data_frame.index.get_loc(idx)
                if pos + lookback_periods >= len(data_frame):
                    continue
                entry_price = data_frame.iloc[pos]['close']
                exit_pos = next(
                    (i for i in range(pos + 1, min(pos + lookback_periods, len(data_frame)))
                     if data_frame['sell_signal' if is_buy else 'buy_signal'].iloc[i]),
                    min(pos + lookback_periods, len(data_frame) - 1)
                )
                exit_price = data_frame.iloc[exit_pos]['close']
                pct_return = (exit_price - entry_price) / entry_price * 100 if is_buy else (entry_price - exit_price) / entry_price * 100
                returns.append({
                    'entry_date': data_frame.index[pos],
                    'exit_date': data_frame.index[exit_pos],
                    'holding_periods': exit_pos - pos,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pct_return': pct_return,
                    'signal_strength': data_frame.iloc[pos]['signal_strength']
                })
            return returns

        buy_returns = calculate_returns(data_frame['buy_signal'], True)
        sell_returns = calculate_returns(data_frame['sell_signal'], False)

        buy_data_frame = pd.DataFrame(buy_returns)
        sell_data_frame = pd.DataFrame(sell_returns)

        results = {
            'buy_signals': {
                'count': len(buy_returns),
                'avg_return': buy_data_frame['pct_return'].mean() if not buy_data_frame.empty else 0,
                'win_rate': (buy_data_frame['pct_return'] > 0).mean() if not buy_data_frame.empty else 0,
                'avg_holding': buy_data_frame['holding_periods'].mean() if not buy_data_frame.empty else 0,
                'strength_correlation': buy_data_frame[['signal_strength', 'pct_return']].corr().iloc[0, 1] if not buy_data_frame.empty else 0
            },
            'sell_signals': {
                'count': len(sell_returns),
                'avg_return': sell_data_frame['pct_return'].mean() if not sell_data_frame.empty else 0,
                'win_rate': (sell_data_frame['pct_return'] > 0).mean() if not sell_data_frame.empty else 0,
                'avg_holding': sell_data_frame['holding_periods'].mean() if not sell_data_frame.empty else 0,
                'strength_correlation': sell_data_frame[['signal_strength', 'pct_return']].corr().iloc[0, 1] if not sell_data_frame.empty else 0
            }
        }

        return results, buy_data_frame, sell_data_frame