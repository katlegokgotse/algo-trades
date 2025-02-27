from datetime import datetime
import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Optional
import logging

class IndicatorCalculator:
    """
    Calculate technical indicators for financial market data.
    """

    def __init__(self,
                 fib_levels: Optional[List[float]] = None,
                 lookback_periods: Optional[Dict[str, int]] = None,
                 volatility_threshold: float = 1.5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the IndicatorCalculator with configurable parameters.

        Args:
            fib_levels: List of Fibonacci retracement levels.
            lookback_periods: Dictionary of lookback periods for various indicators.
            volatility_threshold: Threshold multiplier for high volatility detection.
            logger: Logger instance for error logging.
        """
        self.fib_levels = fib_levels or [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        # Default lookback periods for indicators
        self.lookback_periods = lookback_periods or {
            'ema_short': 20,
            'ema_medium': 50,
            'ema_long': 200,
            'rsi': 14,
            'stoch': 14,
            'stoch_smooth': 3,
            'bollinger': 20,
            'bollinger_dev': 2,
            'atr': 14,
            'adx': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'fib_lookback': 60,
            'volume_ma': 20,
            'volatility_ma': 30
        }

        self.volatility_threshold = volatility_threshold
        self.logger = logger or logging.getLogger(__name__)

    def _validate_data(self, data_frame: pd.DataFrame) -> None:
        """
        Validate that the dataframe contains the required columns.

        Args:
            data_frame: DataFrame with OHLCV data.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data_frame.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _calculate_trend_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """[Your existing trend indicator method remains unchanged]"""
        try:
            data_frame['ema_20'] = ta.trend.ema_indicator(data_frame['close'], window=self.lookback_periods['ema_short'])
            data_frame['ema_50'] = ta.trend.ema_indicator(data_frame['close'], window=self.lookback_periods['ema_medium'])
            data_frame['ema_200'] = ta.trend.ema_indicator(data_frame['close'], window=self.lookback_periods['ema_long'])

            macd = ta.trend.MACD(
                data_frame['close'],
                window_fast=self.lookback_periods['macd_fast'],
                window_slow=self.lookback_periods['macd_slow'],
                window_sign=self.lookback_periods['macd_signal']
            )
            data_frame['macd'] = macd.macd()
            data_frame['macd_signal'] = macd.macd_signal()
            data_frame['macd_hist'] = macd.macd_diff()

            data_frame['adx'] = ta.trend.adx(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=self.lookback_periods['adx']
            )
            data_frame['adx_pos'] = ta.trend.adx_pos(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=self.lookback_periods['adx']
            )
            data_frame['adx_neg'] = ta.trend.adx_neg(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=self.lookback_periods['adx']
            )

            data_frame['prev_ema_20'] = data_frame['ema_20'].shift(1)
            data_frame['prev_ema_50'] = data_frame['ema_50'].shift(1)
            data_frame['prev_macd'] = data_frame['macd'].shift(1)
            data_frame['prev_macd_signal'] = data_frame['macd_signal'].shift(1)

            data_frame['ema_cross_20_50'] = (data_frame['ema_20'] > data_frame['ema_50']) & (data_frame['prev_ema_20'] <= data_frame['prev_ema_50'])
            data_frame['ema_cross_50_20'] = (data_frame['ema_20'] < data_frame['ema_50']) & (data_frame['prev_ema_20'] >= data_frame['prev_ema_50'])

            data_frame['trending_up'] = (data_frame['adx'] > 20) & (data_frame['adx_pos'] > data_frame['adx_neg'])
            data_frame['trending_down'] = (data_frame['adx'] > 20) & (data_frame['adx_pos'] < data_frame['adx_neg'])

            data_frame['trend_strength'] = np.abs((data_frame['ema_20'] - data_frame['ema_50']) / data_frame['ema_50'] * 100)

            return data_frame

        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {str(e)}")
            raise

    def _calculate_momentum_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """[Your existing momentum indicator method remains unchanged]"""
        try:
            data_frame['rsi'] = ta.momentum.rsi(
                data_frame['close'],
                window=self.lookback_periods['rsi']
            )
            data_frame['rsi_prev'] = data_frame['rsi'].shift(1)
            data_frame['rsi_change'] = data_frame['rsi'] - data_frame['rsi_prev']
            data_frame['rsi_rising'] = data_frame['rsi_change'] > 0
            data_frame['rsi_falling'] = data_frame['rsi_change'] < 0

            data_frame['stoch_k'] = ta.momentum.stoch(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=self.lookback_periods['stoch'],
                smooth_window=self.lookback_periods['stoch_smooth']
            )
            data_frame['stoch_d'] = ta.momentum.stoch_signal(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=self.lookback_periods['stoch'],
                smooth_window=self.lookback_periods['stoch_smooth']
            )

            data_frame['prev_stoch_k'] = data_frame['stoch_k'].shift(1)
            data_frame['prev_stoch_d'] = data_frame['stoch_d'].shift(1)

            data_frame['stoch_above_80'] = data_frame['stoch_k'] > 80
            data_frame['stoch_below_20'] = data_frame['stoch_k'] < 20

            data_frame['mfi'] = ta.volume.money_flow_index(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                data_frame['volume'],
                window=14
            )

            data_frame['cmf'] = ta.volume.chaikin_money_flow(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                data_frame['volume'],
                window=20
            )

            return data_frame

        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            raise

    def _calculate_volatility_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """[Your existing volatility indicator method remains unchanged]"""
        try:
            bollinger = ta.volatility.BollingerBands(
                data_frame['close'],
                window=self.lookback_periods['bollinger'],
                window_dev=self.lookback_periods['bollinger_dev']
            )
            data_frame['bollinger_upper'] = bollinger.bollinger_hband()
            data_frame['bollinger_lower'] = bollinger.bollinger_lband()
            data_frame['bollinger_mid'] = bollinger.bollinger_mavg()
            data_frame['bollinger_pct_b'] = bollinger.bollinger_pband()
            data_frame['bollinger_width'] = (data_frame['bollinger_upper'] - data_frame['bollinger_lower']) / data_frame['bollinger_mid']
            data_frame['bollinger_squeeze'] = data_frame['bollinger_width'] < data_frame['bollinger_width'].rolling(20).mean() * 0.8

            data_frame['atr'] = ta.volatility.average_true_range(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=self.lookback_periods['atr']
            )
            data_frame['atr_pct'] = data_frame['atr'] / data_frame['close'] * 100

            volatility_ma = data_frame['atr'].rolling(self.lookback_periods['volatility_ma']).mean()
            data_frame['high_volatility'] = data_frame['atr'] > volatility_ma * self.volatility_threshold

            keltner = ta.volatility.KeltnerChannel(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                window=20,
                window_atr=10
            )
            data_frame['keltner_upper'] = keltner.keltner_channel_hband()
            data_frame['keltner_lower'] = keltner.keltner_channel_lband()
            data_frame['keltner_mid'] = keltner.keltner_channel_mband()

            return data_frame

        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {str(e)}")
            raise

    def _calculate_volume_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """[Your existing volume indicator method remains unchanged]"""
        try:
            data_frame['vwap'] = ta.volume.volume_weighted_average_price(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                data_frame['volume']
            )
            data_frame['obv'] = ta.volume.on_balance_volume(data_frame['close'], data_frame['volume'])
            data_frame['obv_change'] = data_frame['obv'].diff()
            data_frame['obv_slope'] = data_frame['obv'].diff(5) / 5

            data_frame['volume_sma'] = data_frame['volume'].rolling(self.lookback_periods['volume_ma']).mean()
            data_frame['volume_ratio'] = data_frame['volume'] / data_frame['volume_sma']
            data_frame['volume_spike'] = data_frame['volume'] > data_frame['volume_sma'] * 1.5

            data_frame['adl'] = ta.volume.acc_dist_index(
                data_frame['high'],
                data_frame['low'],
                data_frame['close'],
                data_frame['volume']
            )
            data_frame['eom'] = ta.volume.ease_of_movement(
                data_frame['high'],
                data_frame['low'],
                data_frame['volume'],
                window=14
            )
            return data_frame
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            raise

    def calc_fibonacci_levels(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """[Your existing Fibonacci method remains unchanged]"""
        try:
            lookback = self.lookback_periods['fib_lookback']
            data_frame['fib_trend'] = 'neutral'
            data_frame['fib_swing_high'] = np.nan
            data_frame['fib_swing_low'] = np.nan
            data_frame['fib_diff'] = np.nan

            for level in self.fib_levels:
                level_str = f'fib_{str(level).replace(".", "_")}'
                data_frame[level_str] = np.nan

            for i in range(lookback, len(data_frame)):
                section = data_frame.iloc[i - lookback:i]
                highest_high = section['high'].max()
                lowest_low = section['low'].min()
                highest_idx = section['high'].idxmax()
                lowest_idx = section['low'].idxmin()
                diff = highest_high - lowest_low

                data_frame.loc[data_frame.index[i], 'fib_swing_high'] = highest_high
                data_frame.loc[data_frame.index[i], 'fib_swing_low'] = lowest_low
                data_frame.loc[data_frame.index[i], 'fib_diff'] = diff

                if highest_idx > lowest_idx:
                    data_frame.loc[data_frame.index[i], 'fib_trend'] = 'uptrend'
                    for level in self.fib_levels:
                        level_str = f'fib_{str(level).replace(".", "_")}'
                        data_frame.loc[data_frame.index[i], level_str] = highest_high - (diff * level)
                else:
                    data_frame.loc[data_frame.index[i], 'fib_trend'] = 'downtrend'
                    for level in self.fib_levels:
                        level_str = f'fib_{str(level).replace(".", "_")}'
                        data_frame.loc[data_frame.index[i], level_str] = lowest_low + (diff * level)

            return data_frame
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            raise

    def _calculate_custom_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom or composite indicators with enhanced implementations.

        Args:
            data_frame: DataFrame with OHLCV data.

        Returns:
            DataFrame with added custom indicators.
        """
        try:
            # Price Rate of Change
            data_frame['price_roc'] = data_frame['close'].pct_change(periods=14) * 100

            # Hull Moving Average
            ema_short = self.lookback_periods['ema_short']
            half_length = int(ema_short / 2)
            sqrt_length = int(np.sqrt(ema_short))

            data_frame['wma_half'] = data_frame['close'].rolling(half_length).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)),
                raw=True
            )
            data_frame['wma_full'] = data_frame['close'].rolling(ema_short).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)),
                raw=True
            )
            data_frame['hull_raw'] = 2 * data_frame['wma_half'] - data_frame['wma_full']
            data_frame['hull_ma'] = data_frame['hull_raw'].rolling(sqrt_length).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)),
                raw=True
            )

            # Ichimoku Cloud components
            data_frame['tenkan_sen'] = (
                data_frame['high'].rolling(9).max() +
                data_frame['low'].rolling(9).min()
            ) / 2
            data_frame['kijun_sen'] = (
                data_frame['high'].rolling(26).max() +
                data_frame['low'].rolling(26).min()
            ) / 2
            data_frame['senkou_span_a'] = (
                (data_frame['tenkan_sen'] + data_frame['kijun_sen']) / 2
            ).shift(26)
            data_frame['senkou_span_b'] = (
                (data_frame['high'].rolling(52).max() + data_frame['low'].rolling(52).min()) / 2
            ).shift(26)
            data_frame['chikou_span'] = data_frame['close'].shift(-26)

            # Composite indicators from your version
            data_frame['trend_confirmation'] = (
                (data_frame['adx'] > 25) &
                ((data_frame['ema_20'] > data_frame['ema_50']) == (data_frame['macd'] > 0))
            )
            data_frame['oversold_consensus'] = (
                (data_frame['rsi'] < 30) &
                (data_frame['stoch_k'] < 20) &
                (data_frame['close'] < data_frame['bollinger_lower'])
            )
            data_frame['overbought_consensus'] = (
                (data_frame['rsi'] > 70) &
                (data_frame['stoch_k'] > 80) &
                (data_frame['close'] > data_frame['bollinger_upper'])
            )
            data_frame['volume_confirmed_up'] = (
                (data_frame['close'] > data_frame['close'].shift(1)) &
                (data_frame['volume'] > data_frame['volume_sma'])
            )
            data_frame['volume_confirmed_down'] = (
                (data_frame['close'] < data_frame['close'].shift(1)) &
                (data_frame['volume'] > data_frame['volume_sma'])
            )

            # Dynamic support/resistance
            data_frame['support_level'] = data_frame.apply(
                lambda x: max(
                    x['bollinger_lower'],
                    x['ema_200'],
                    x.get('fib_0_618', float('-inf'))
                ) if x['fib_trend'] == 'uptrend' else float('-inf'),
                axis=1
            )
            data_frame['resistance_level'] = data_frame.apply(
                lambda x: min(
                    x['bollinger_upper'],
                    x['ema_50'],
                    x.get('fib_0_618', float('inf'))
                ) if x['fib_trend'] == 'downtrend' else float('inf'),
                axis=1
            )

            return data_frame

        except Exception as e:
            self.logger.error(f"Error calculating custom indicators: {str(e)}")
            raise

    def apply_indicators(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """[Your existing apply_indicators method remains unchanged]"""
        try:
            self._validate_data(data_frame)
            data_frame = data_frame.copy()
            data_frame = self._calculate_trend_indicators(data_frame)
            data_frame = self._calculate_momentum_indicators(data_frame)
            data_frame = self._calculate_volatility_indicators(data_frame)
            data_frame = self._calculate_volume_indicators(data_frame)
            data_frame = self.calc_fibonacci_levels(data_frame)
            data_frame = self._calculate_custom_indicators(data_frame)
            data_frame = data_frame.bfill().ffill()
            return data_frame
        except Exception as e:
            self.logger.error(f"Error applying indicators: {str(e)}")
            raise

    def remove_redundant_indicators(self, data_frame: pd.DataFrame,
                                    corr_threshold: float = 0.85) -> pd.DataFrame:
        """[Your existing remove_redundant_indicators method remains unchanged]"""
        try:
            numeric_data_frame = data_frame.select_dtypes(include=[np.number])
            corr_matrix = numeric_data_frame.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
            self.logger.info(f"Removing {len(to_drop)} redundant indicators: {to_drop}")
            data_frame_reduced = data_frame.drop(to_drop, axis=1)
            return data_frame_reduced
        except Exception as e:
            self.logger.error(f"Error removing redundant indicators: {str(e)}")
            return data_frame

    def diagnostic_report(self, data_frame: pd.DataFrame) -> Dict:
        """[Your existing diagnostic_report method remains unchanged]"""
        try:
            nan_counts = data_frame.isna().sum()
            indicator_variance = data_frame.select_dtypes(include=[np.number]).var()
            rsi_price_divergence = ((data_frame['close'] > data_frame['close'].shift(5)) & (data_frame['rsi'] < data_frame['rsi'].shift(5))).sum()
            macd_price_divergence = ((data_frame['close'] > data_frame['close'].shift(5)) & (data_frame['macd'] < data_frame['macd'].shift(5))).sum()
            rsi_extremes = ((data_frame['rsi'] < 10) | (data_frame['rsi'] > 90)).sum()
            regime_changes = (data_frame['high_volatility'] != data_frame['high_volatility'].shift(1)).sum()
            report = {
                'nan_count': {col: int(count) for col, count in nan_counts.items() if count > 0},
                'indicator_stability': {col: float(var) for col, var in indicator_variance.items()},
                'divergences': {
                    'rsi_price': int(rsi_price_divergence),
                    'macd_price': int(macd_price_divergence)
                },
                'extreme_values': {
                    'rsi': int(rsi_extremes)
                },
                'regime_changes': int(regime_changes),
                'data_quality': {
                    'rows': len(data_frame),
                    'missing_data_pct': data_frame.isna().sum().sum() / (data_frame.shape[0] * data_frame.shape[1]) * 100
                }
            }
            return report
        except Exception as e:
            self.logger.error(f"Error generating diagnostic report: {str(e)}")
            return {'error': str(e)}