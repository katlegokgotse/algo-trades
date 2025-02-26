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
        
        # Default lookback periods for indicators.
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
        
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that the dataframe contains the required columns.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators (EMAs, MACD, ADX).
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with added trend indicators.
        """
        try:
            # EMA calculations
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=self.lookback_periods['ema_short'])
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=self.lookback_periods['ema_medium'])
            df['ema_200'] = ta.trend.ema_indicator(df['close'], window=self.lookback_periods['ema_long'])
            
            # MACD calculation
            macd = ta.trend.MACD(
                df['close'], 
                window_fast=self.lookback_periods['macd_fast'],
                window_slow=self.lookback_periods['macd_slow'], 
                window_sign=self.lookback_periods['macd_signal']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # ADX calculation
            df['adx'] = ta.trend.adx(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.lookback_periods['adx']
            )
            df['adx_pos'] = ta.trend.adx_pos(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.lookback_periods['adx']
            )
            df['adx_neg'] = ta.trend.adx_neg(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.lookback_periods['adx']
            )
            
            # Previous values for crosses
            df['prev_ema_20'] = df['ema_20'].shift(1)
            df['prev_ema_50'] = df['ema_50'].shift(1)
            df['prev_macd'] = df['macd'].shift(1)
            df['prev_macd_signal'] = df['macd_signal'].shift(1)
            
            # Additional trend metrics
            df['ema_cross_20_50'] = (df['ema_20'] > df['ema_50']) & (df['prev_ema_20'] <= df['prev_ema_50'])
            df['ema_cross_50_20'] = (df['ema_20'] < df['ema_50']) & (df['prev_ema_20'] >= df['prev_ema_50'])
            
            # Trend direction
            df['trending_up'] = (df['adx'] > 20) & (df['adx_pos'] > df['adx_neg'])
            df['trending_down'] = (df['adx'] > 20) & (df['adx_pos'] < df['adx_neg'])
            
            # Trend strength metrics
            df['trend_strength'] = np.abs((df['ema_20'] - df['ema_50']) / df['ema_50'] * 100)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {str(e)}")
            raise
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators (RSI, Stochastic).
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with added momentum indicators.
        """
        try:
            # RSI calculation
            df['rsi'] = ta.momentum.rsi(
                df['close'], 
                window=self.lookback_periods['rsi']
            )
            
            # RSI direction and rate of change
            df['rsi_prev'] = df['rsi'].shift(1)
            df['rsi_change'] = df['rsi'] - df['rsi_prev']
            df['rsi_rising'] = df['rsi_change'] > 0
            df['rsi_falling'] = df['rsi_change'] < 0
            
            # Stochastic oscillator
            df['stoch_k'] = ta.momentum.stoch(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.lookback_periods['stoch'],
                smooth_window=self.lookback_periods['stoch_smooth']
            )
            
            df['stoch_d'] = ta.momentum.stoch_signal(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.lookback_periods['stoch'],
                smooth_window=self.lookback_periods['stoch_smooth']
            )
            
            # Previous stochastic values for crosses
            df['prev_stoch_k'] = df['stoch_k'].shift(1)
            df['prev_stoch_d'] = df['stoch_d'].shift(1)
            
            # Additional momentum metrics
            df['stoch_above_80'] = df['stoch_k'] > 80
            df['stoch_below_20'] = df['stoch_k'] < 20
            
            # Money Flow Index
            df['mfi'] = ta.volume.money_flow_index(
                df['high'],
                df['low'],
                df['close'],
                df['volume'],
                window=14
            )
            
            # Chaikin Money Flow
            df['cmf'] = ta.volume.chaikin_money_flow(
                df['high'],
                df['low'],
                df['close'],
                df['volume'],
                window=20
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            raise
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators (Bollinger Bands, ATR, Keltner Channels).
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with added volatility indicators.
        """
        try:
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                df['close'], 
                window=self.lookback_periods['bollinger'],
                window_dev=self.lookback_periods['bollinger_dev']
            )
            
            df['bollinger_upper'] = bollinger.bollinger_hband()
            df['bollinger_lower'] = bollinger.bollinger_lband()
            df['bollinger_mid'] = bollinger.bollinger_mavg()
            df['bollinger_pct_b'] = bollinger.bollinger_pband()  # Position within bands (0-1)
            df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_mid']
            
            # Bollinger band squeezes (narrowing bands)
            df['bollinger_squeeze'] = df['bollinger_width'] < df['bollinger_width'].rolling(20).mean() * 0.8
            
            # Average True Range
            df['atr'] = ta.volatility.average_true_range(
                df['high'], 
                df['low'], 
                df['close'], 
                window=self.lookback_periods['atr']
            )
            
            # ATR percentage of price
            df['atr_pct'] = df['atr'] / df['close'] * 100
            
            # High volatility flag
            volatility_ma = df['atr'].rolling(self.lookback_periods['volatility_ma']).mean()
            df['high_volatility'] = df['atr'] > volatility_ma * self.volatility_threshold
            
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(
                df['high'],
                df['low'],
                df['close'],
                window=20,
                window_atr=10
            )
            df['keltner_upper'] = keltner.keltner_channel_hband()
            df['keltner_lower'] = keltner.keltner_channel_lband()
            df['keltner_mid'] = keltner.keltner_channel_mband()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {str(e)}")
            raise
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators (VWAP, OBV, etc.).

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with added volume indicators.
        """
        try:
            # VWAP calculation
            df['vwap'] = ta.volume.volume_weighted_average_price(
                df['high'], 
                df['low'], 
                df['close'], 
                df['volume']
            )
        
            # On-Balance Volume
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['obv_change'] = df['obv'].diff()
            df['obv_slope'] = df['obv'].diff(5) / 5  # 5-period OBV slope
        
            # Volume relative to average
            df['volume_sma'] = df['volume'].rolling(self.lookback_periods['volume_ma']).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_spike'] = df['volume'] > df['volume_sma'] * 1.5
        
            # Accumulation/Distribution Line
            df['adl'] = ta.volume.acc_dist_index(
                df['high'],
                df['low'],
                df['close'],
                df['volume']
            )
        
            # Ease of Movement
            df['eom'] = ta.volume.ease_of_movement(
                df['high'],
                df['low'],
                df['volume'],
                window=14
            )
            return df
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            raise

    def calc_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels based on price swings.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with added Fibonacci level columns.
        """
        try:
            lookback = self.lookback_periods['fib_lookback']
            
            # Initialize columns
            df['fib_trend'] = 'neutral'
            df['fib_swing_high'] = np.nan
            df['fib_swing_low'] = np.nan
            df['fib_diff'] = np.nan
            
            # Create columns for each Fibonacci level
            for level in self.fib_levels:
                level_str = f'fib_{str(level).replace(".", "_")}'
                df[level_str] = np.nan
            
            # Loop over each row after the lookback period
            for i in range(lookback, len(df)):
                section = df.iloc[i - lookback:i]
                highest_high = section['high'].max()
                lowest_low = section['low'].min()
                highest_idx = section['high'].idxmax()
                lowest_idx = section['low'].idxmin()
                diff = highest_high - lowest_low
                
                # Store swing high/low and difference
                df.loc[df.index[i], 'fib_swing_high'] = highest_high
                df.loc[df.index[i], 'fib_swing_low'] = lowest_low
                df.loc[df.index[i], 'fib_diff'] = diff
                
                # Determine trend direction and calculate Fibonacci levels accordingly
                if highest_idx > lowest_idx:
                    df.loc[df.index[i], 'fib_trend'] = 'uptrend'
                    for level in self.fib_levels:
                        level_str = f'fib_{str(level).replace(".", "_")}'
                        df.loc[df.index[i], level_str] = highest_high - (diff * level)
                else:
                    df.loc[df.index[i], 'fib_trend'] = 'downtrend'
                    for level in self.fib_levels:
                        level_str = f'fib_{str(level).replace(".", "_")}'
                        df.loc[df.index[i], level_str] = lowest_low + (diff * level)
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            raise
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom or composite indicators.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with added custom indicators.
        """
        try:
            # Price Rate of Change
            df['price_roc'] = df['close'].pct_change(periods=14) * 100
            
            # Hull Moving Average
            wma_half_length = int(np.sqrt(self.lookback_periods['ema_short']))
            half_length = int(self.lookback_periods['ema_short'] / 2)
            df['wma_half'] = df['close'].rolling(half_length).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)),
                raw=True
            )
            df['wma_full'] = df['close'].rolling(self.lookback_periods['ema_short']).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)),
                raw=True
            )
            df['hull_raw'] = 2 * df['wma_half'] - df['wma_full']
            df['hull_ma'] = df['hull_raw'].rolling(wma_half_length).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)),
                raw=True
            )
            
            # Composite indicators
            df['trend_confirmation'] = (
                (df['adx'] > 25) & 
                ((df['ema_20'] > df['ema_50']) == (df['macd'] > 0))
            )
            
            df['oversold_consensus'] = (
                (df['rsi'] < 30) & 
                (df['stoch_k'] < 20) & 
                (df['close'] < df['bollinger_lower'])
            )
            
            df['overbought_consensus'] = (
                (df['rsi'] > 70) & 
                (df['stoch_k'] > 80) & 
                (df['close'] > df['bollinger_upper'])
            )
            
            df['volume_confirmed_up'] = (
                (df['close'] > df['close'].shift(1)) & 
                (df['volume'] > df['volume_sma'])
            )
            df['volume_confirmed_down'] = (
                (df['close'] < df['close'].shift(1)) & 
                (df['volume'] > df['volume_sma'])
            )
            
            # Dynamic support/resistance
            df['support_level'] = df.apply(
                lambda x: max(
                    x['bollinger_lower'],
                    x['ema_200'],
                    x.get('fib_0_618', float('-inf'))
                ) if x['fib_trend'] == 'uptrend' else float('-inf'),
                axis=1
            )
            
            df['resistance_level'] = df.apply(
                lambda x: min(
                    x['bollinger_upper'],
                    x['ema_50'],
                    x.get('fib_0_618', float('inf'))
                ) if x['fib_trend'] == 'downtrend' else float('inf'),
                axis=1
            )
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculating custom indicators: {str(e)}")
            raise
    
    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            DataFrame with all indicators applied.
        """
        try:
            # Validate input data
            self._validate_data(df)
            
            # Work on a copy of the dataframe
            df = df.copy()
            
            # Calculate all indicator categories
            df = self._calculate_trend_indicators(df)
            df = self._calculate_momentum_indicators(df)
            df = self._calculate_volatility_indicators(df)
            df = self._calculate_volume_indicators(df)
            df = self.calc_fibonacci_levels(df)
            df = self._calculate_custom_indicators(df)
            
            # Fill missing values
            df = df.bfill().ffill()
            
            return df
        except Exception as e:
            self.logger.error(f"Error applying indicators: {str(e)}")
            raise
    
    def remove_redundant_indicators(self, df: pd.DataFrame, 
                                    corr_threshold: float = 0.85) -> pd.DataFrame:
        """
        Remove highly correlated indicators to reduce redundancy.
        
        Args:
            df: DataFrame with indicators.
            corr_threshold: Correlation threshold above which indicators are considered redundant.
            
        Returns:
            DataFrame with a reduced set of indicators.
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
            self.logger.info(f"Removing {len(to_drop)} redundant indicators: {to_drop}")
            df_reduced = df.drop(to_drop, axis=1)
            return df_reduced
        except Exception as e:
            self.logger.error(f"Error removing redundant indicators: {str(e)}")
            return df
    
    def diagnostic_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a diagnostic report on indicators.
        
        Args:
            df: DataFrame with indicators.
            
        Returns:
            Dictionary with diagnostic information.
        """
        try:
            nan_counts = df.isna().sum()
            indicator_variance = df.select_dtypes(include=[np.number]).var()
            rsi_price_divergence = ((df['close'] > df['close'].shift(5)) & (df['rsi'] < df['rsi'].shift(5))).sum()
            macd_price_divergence = ((df['close'] > df['close'].shift(5)) & (df['macd'] < df['macd'].shift(5))).sum()
            rsi_extremes = ((df['rsi'] < 10) | (df['rsi'] > 90)).sum()
            regime_changes = (df['high_volatility'] != df['high_volatility'].shift(1)).sum()
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
                    'rows': len(df),
                    'missing_data_pct': df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                }
            }
            return report
        except Exception as e:
            self.logger.error(f"Error generating diagnostic report: {str(e)}")
            return {'error': str(e)}
