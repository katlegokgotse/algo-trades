import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

class SignalGenerator:
    def __init__(self, 
                 fib_levels: List[float] = None,
                 macd_threshold_factor: float = 0.02,
                 volume_spike_factor: float = 1.5,
                 fib_buffer: float = 0.005,
                 rsi_thresholds: Dict[str, int] = None,
                 adx_threshold: int = 30):
        """
        Initialize the SignalGenerator with configurable parameters.
        
        Args:
            fib_levels: List of Fibonacci levels to check for support/resistance
            macd_threshold_factor: Factor to calculate MACD threshold (relative to price)
            volume_spike_factor: Factor above average to identify volume spikes
            fib_buffer: Buffer percentage for Fibonacci level checks
            rsi_thresholds: Dictionary of RSI thresholds for overbought/oversold
            adx_threshold: Threshold to identify strong trends
        """
        self.fib_levels = fib_levels or [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.macd_threshold_factor = macd_threshold_factor
        self.volume_spike_factor = volume_spike_factor
        self.fib_buffer = fib_buffer
        self.rsi_thresholds = rsi_thresholds or {
            'strong_oversold': 20,
            'oversold': 30,
            'overbought': 70,
            'strong_overbought': 80
        }
        self.adx_threshold = adx_threshold
        
    def _identify_trends(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market trends based on EMA relationships.
        """
        # Check required columns
        required_columns = ['ema_20', 'ema_50', 'ema_200']
        self._validate_columns(data_frame, required_columns)
        
        # Trend conditions
        data_frame['uptrend'] = (data_frame['ema_20'] > data_frame['ema_50']) & (data_frame['ema_50'] > data_frame['ema_200'])
        data_frame['downtrend'] = (data_frame['ema_20'] < data_frame['ema_50']) & (data_frame['ema_50'] < data_frame['ema_200'])
        # Add trend strength based on distance between EMAs
        data_frame['trend_strength'] = np.abs((data_frame['ema_20'] - data_frame['ema_50']) / data_frame['ema_50'] * 100)
        
        return data_frame
    
    def _identify_macd_signals(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Identify MACD crossover signals.
        """
        required_columns = ['macd', 'macd_signal', 'prev_macd', 'prev_macd_signal', 'close']
        self._validate_columns(data_frame, required_columns)
        
        # Calculate dynamic threshold based on price
        macd_threshold = self.macd_threshold_factor * data_frame['close'].mean() / 10000
        
        # MACD crosses
        data_frame['macd_cross_up'] = ((data_frame['macd'] > data_frame['macd_signal']) & 
                               (data_frame['prev_macd'] <= data_frame['prev_macd_signal']) & 
                               ((data_frame['macd'] - data_frame['macd_signal']).abs() > macd_threshold))
        
        data_frame['macd_cross_down'] = ((data_frame['macd'] < data_frame['macd_signal']) & 
                                 (data_frame['prev_macd'] >= data_frame['prev_macd_signal']) & 
                                 ((data_frame['macd'] - data_frame['macd_signal']).abs() > macd_threshold))
        
        # Add MACD histogram direction
        data_frame['macd_histogram'] = data_frame['macd'] - data_frame['macd_signal']
        data_frame['macd_histogram_increasing'] = data_frame['macd_histogram'] > data_frame['macd_histogram'].shift(1)
        
        return data_frame
    
    def _identify_oscillator_signals(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Identify oscillator-based signals (RSI, Stochastic).
        """
        # Check for stochastic columns
        stoch_columns = ['stoch_k', 'stoch_d', 'prev_stoch_k', 'prev_stoch_d']
        has_stoch = all(col in data_frame.columns for col in stoch_columns)
        
        # Check for RSI column
        has_rsi = 'rsi' in data_frame.columns
        
        if has_stoch:
            # Stochastic crosses
            data_frame['stoch_cross_up'] = (data_frame['stoch_k'] > data_frame['stoch_d']) & (data_frame['prev_stoch_k'] <= data_frame['prev_stoch_d'])
            data_frame['stoch_cross_down'] = (data_frame['stoch_k'] < data_frame['stoch_d']) & (data_frame['prev_stoch_k'] >= data_frame['prev_stoch_d'])
            
            # Stochastic levels
            data_frame['stoch_oversold'] = data_frame['stoch_k'] < 20
            data_frame['stoch_overbought'] = data_frame['stoch_k'] > 80
        
        if has_rsi:
            # RSI thresholds
            data_frame['rsi_strong_oversold'] = data_frame['rsi'] < self.rsi_thresholds['strong_oversold']
            data_frame['rsi_oversold'] = data_frame['rsi'] < self.rsi_thresholds['oversold']
            data_frame['rsi_overbought'] = data_frame['rsi'] > self.rsi_thresholds['overbought']
            data_frame['rsi_strong_overbought'] = data_frame['rsi'] > self.rsi_thresholds['strong_overbought']
            
            # RSI direction
            data_frame['rsi_rising'] = data_frame['rsi'] > data_frame['rsi'].shift(1)
            data_frame['rsi_falling'] = data_frame['rsi'] < data_frame['rsi'].shift(1)
        
        return data_frame
    
    def _identify_price_patterns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Identify price patterns and relationships to indicators.
        """
        # Bollinger Bands check
        if all(col in data_frame.columns for col in ['close', 'bollinger_lower', 'bollinger_upper']):
            data_frame['near_lower_band'] = data_frame['close'] <= data_frame['bollinger_lower'] * 1.01
            data_frame['near_upper_band'] = data_frame['close'] >= data_frame['bollinger_upper'] * 0.99
            data_frame['bollinger_squeeze'] = (data_frame['bollinger_upper'] - data_frame['bollinger_lower']) / data_frame['close'] < 0.03
        
        # VWAP check
        if 'vwap' in data_frame.columns and 'close' in data_frame.columns:
            data_frame['above_vwap'] = data_frame['close'] > data_frame['vwap']
            data_frame['below_vwap'] = data_frame['close'] < data_frame['vwap']
        
        # Volume analysis
        if 'volume' in data_frame.columns:
            data_frame['volume_spike'] = data_frame['volume'] > data_frame['volume'].rolling(20).mean() * self.volume_spike_factor
            data_frame['volume_increasing'] = data_frame['volume'] > data_frame['volume'].shift(1)
        
        # ADX for trend strength
        if 'adx' in data_frame.columns:
            data_frame['strong_trend'] = data_frame['adx'] > self.adx_threshold
        
        return data_frame
    
    def _process_fibonacci_levels(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Process Fibonacci retracement/extension levels.
        """
        data_frame['at_fib_support'] = False
        data_frame['at_fib_resistance'] = False
        
        # Check if we have fibonacci columns
        fib_columns = [f'fib_{str(level).replace(".", "_")}' for level in self.fib_levels]
        has_fib = all(col in data_frame.columns for col in fib_columns[:1])
        
        if not has_fib or 'fib_trend' not in data_frame.columns:
            return data_frame
        
        buffer = self.fib_buffer
        
        for i in range(len(data_frame)):
            row = data_frame.iloc[i]
            if pd.isna(row.get('fib_0')):
                continue
                
            current_price = row['close']
            
            for level in self.fib_levels:
                level_str = f'fib_{str(level).replace(".", "_")}'
                if level_str in row and not pd.isna(row[level_str]):
                    fib_level = row[level_str]
                    
                    # Check if price is near a Fibonacci level
                    is_near_level = (current_price >= fib_level * (1 - buffer) and 
                                     current_price <= fib_level * (1 + buffer))
                    
                    if is_near_level:
                        if row['fib_trend'] == 'uptrend':
                            data_frame.loc[data_frame.index[i], 'at_fib_support'] = True
                        elif row['fib_trend'] == 'downtrend':
                            data_frame.loc[data_frame.index[i], 'at_fib_resistance'] = True
        
        return data_frame
    
    def _calculate_signal_strength(self, 
                                  row: pd.Series, 
                                  signal_type: str) -> int:
        """
        Calculate signal strength based on confirming factors.
        
        Args:
            row: DataFrame row
            signal_type: 'buy' or 'sell'
            
        Returns:
            int: Signal strength (0-5)
        """
        strength = 0
        
        # Base conditions depending on signal type
        if signal_type == 'buy':
            # Strong conditions
            if row.get('uptrend', False) and row.get('strong_trend', False):
                strength += 1
            
            # Oscillator conditions
            if row.get('rsi_strong_oversold', False):
                strength += 1
            elif row.get('rsi_oversold', False):
                strength += 0.5
                
            # MACD and Stochastic
            if row.get('macd_cross_up', False):
                strength += 1
            if row.get('stoch_cross_up', False) and row.get('stoch_oversold', False):
                strength += 1
                
            # Additional confirmations
            if row.get('near_lower_band', False):
                strength += 0.5
            if row.get('volume_spike', False):
                strength += 0.5
            if row.get('at_fib_support', False):
                strength += 1
                
        elif signal_type == 'sell':
            # Strong conditions
            if row.get('downtrend', False) and row.get('strong_trend', False):
                strength += 1
            
            # Oscillator conditions
            if row.get('rsi_strong_overbought', False):
                strength += 1
            elif row.get('rsi_overbought', False):
                strength += 0.5
                
            # MACD and Stochastic
            if row.get('macd_cross_down', False):
                strength += 1
            if row.get('stoch_cross_down', False) and row.get('stoch_overbought', False):
                strength += 1
                
            # Additional confirmations
            if row.get('near_upper_band', False):
                strength += 0.5
            if row.get('volume_spike', False):
                strength += 0.5
            if row.get('at_fib_resistance', False):
                strength += 1
        
        # Round to nearest 0.5 and cap at 5
        strength = min(round(strength * 2) / 2, 5)
        return strength
    
    def _validate_columns(self, data_frame: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Validate that required columns exist in the dataframe.
        """
        missing_columns = [col for col in required_columns if col not in data_frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    def generate_signals(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data_frame: DataFrame with technical indicators
            
        Returns:
            DataFrame with added signal columns
        """
        # Make a copy to avoid modifying the original
        data_frame = data_frame.copy()
        
        # Initialize signal columns
        data_frame['buy_signal'] = False
        data_frame['sell_signal'] = False
        data_frame['signal_strength'] = 0
        data_frame['signal_type'] = None
        
        # Process indicators
        data_frame = self._identify_trends(data_frame)
        data_frame = self._identify_macd_signals(data_frame)
        data_frame = self._identify_oscillator_signals(data_frame)
        data_frame = self._identify_price_patterns(data_frame)
        data_frame = self._process_fibonacci_levels(data_frame)
        
        # Calculate signal conditions row by row for better flexibility
        for i in range(len(data_frame)):
            row = data_frame.iloc[i]
            
            # Determine buy signal
            buy_strength = self._calculate_signal_strength(row, 'buy')
            if buy_strength >= 2:
                data_frame.loc[data_frame.index[i], 'buy_signal'] = True
                data_frame.loc[data_frame.index[i], 'signal_strength'] = int(buy_strength)
                data_frame.loc[data_frame.index[i], 'signal_type'] = 'buy'
            
            # Determine sell signal - only if no buy signal
            if not data_frame.loc[data_frame.index[i], 'buy_signal']:
                sell_strength = self._calculate_signal_strength(row, 'sell')
                if sell_strength >= 2:
                    data_frame.loc[data_frame.index[i], 'sell_signal'] = True
                    data_frame.loc[data_frame.index[i], 'signal_strength'] =  int(sell_strength)
                    data_frame.loc[data_frame.index[i], 'signal_type'] = 'sell'
        
        # Volatility filter
        if 'high_volatility' in data_frame.columns:
            volatility_threshold = 2.5  # Threshold for ignoring weaker signals in high volatility
            data_frame.loc[data_frame['high_volatility'] & (data_frame['signal_strength'] < volatility_threshold), 'buy_signal'] = False
            data_frame.loc[data_frame['high_volatility'] & (data_frame['signal_strength'] < volatility_threshold), 'sell_signal'] = False
            data_frame.loc[data_frame['high_volatility'] & (data_frame['signal_strength'] < volatility_threshold), 'signal_type'] = None
        
        return data_frame
    
    def analyze_performance(self, data_frame: pd.DataFrame, 
                           lookback_periods: int = 50) -> pd.DataFrame:
        """
        Analyze the performance of generated signals.
        
        Args:
            data_frame: DataFrame with signals already generated
            lookback_periods: Number of periods to look back for analysis
            
        Returns:
            DataFrame with performance metrics
        """
        # Validate signals exist
        required_columns = ['buy_signal', 'sell_signal', 'close']
        self._validate_columns(data_frame, required_columns)
        
        results = {}
        
        # Get buy and sell signal indices
        buy_indices = data_frame.index[data_frame['buy_signal']].tolist()
        sell_indices = data_frame.index[data_frame['sell_signal']].tolist()
        
        # Calculate forward returns for buy signals
        buy_returns = []
        for idx in buy_indices:
            pos = data_frame.index.get_loc(idx)
            
            # Skip if near the end of the dataframe
            if pos + lookback_periods >= len(data_frame):
                continue
                
            entry_price = data_frame.iloc[pos]['close']
            
            # Find next exit (either sell signal or end of lookback)
            exit_pos = None
            for i in range(pos + 1, min(pos + lookback_periods, len(data_frame))):
                if data_frame.index[i] in sell_indices:
                    exit_pos = i
                    break
            
            if exit_pos is None:
                exit_pos = min(pos + lookback_periods, len(data_frame) - 1)
                
            exit_price = data_frame.iloc[exit_pos]['close']
            pct_return = (exit_price - entry_price) / entry_price * 100
            
            buy_returns.append({
                'entry_date': data_frame.index[pos],
                'exit_date': data_frame.index[exit_pos],
                'holding_periods': exit_pos - pos,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pct_return': pct_return,
                'signal_strength': data_frame.iloc[pos]['signal_strength']
            })
        
        # Similar analysis for sell signals (assuming short selling)
        sell_returns = []
        for idx in sell_indices:
            pos = data_frame.index.get_loc(idx)
            
            # Skip if near the end of the dataframe
            if pos + lookback_periods >= len(data_frame):
                continue
                
            entry_price = data_frame.iloc[pos]['close']
            
            # Find next exit (either buy signal or end of lookback)
            exit_pos = None
            for i in range(pos + 1, min(pos + lookback_periods, len(data_frame))):
                if data_frame.index[i] in buy_indices:
                    exit_pos = i
                    break
            
            if exit_pos is None:
                exit_pos = min(pos + lookback_periods, len(data_frame) - 1)
                
            exit_price = data_frame.iloc[exit_pos]['close']
            pct_return = (entry_price - exit_price) / entry_price * 100  # Reversed for short selling
            
            sell_returns.append({
                'entry_date': data_frame.index[pos],
                'exit_date': data_frame.index[exit_pos],
                'holding_periods': exit_pos - pos,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pct_return': pct_return,
                'signal_strength': data_frame.iloc[pos]['signal_strength']
            })
        
        # Create DataFrames from results
        buy_data_frame = pd.DataFrame(buy_returns) if buy_returns else pd.DataFrame()
        sell_data_frame = pd.DataFrame(sell_returns) if sell_returns else pd.DataFrame()
        
        # Calculate summary statistics
        results = {
            'buy_signals': {
                'count': len(buy_returns),
                'avg_return': buy_data_frame['pct_return'].mean() if not buy_data_frame.empty else 0,
                'win_rate': (buy_data_frame['pct_return'] > 0).mean() if not buy_data_frame.empty else 0,
                'avg_holding': buy_data_frame['holding_periods'].mean() if not buy_data_frame.empty else 0
            },
            'sell_signals': {
                'count': len(sell_returns),
                'avg_return': sell_data_frame['pct_return'].mean() if not sell_data_frame.empty else 0,
                'win_rate': (sell_data_frame['pct_return'] > 0).mean() if not sell_data_frame.empty else 0,
                'avg_holding': sell_data_frame['holding_periods'].mean() if not sell_data_frame.empty else 0
            }
        }
        
        # Correlation between signal strength and returns
        if not buy_data_frame.empty:
            results['buy_signals']['strength_correlation'] = buy_data_frame[['signal_strength', 'pct_return']].corr().iloc[0, 1]
        
        if not sell_data_frame.empty:
            results['sell_signals']['strength_correlation'] = sell_data_frame[['signal_strength', 'pct_return']].corr().iloc[0, 1]
        
        return results, buy_data_frame, sell_data_frame