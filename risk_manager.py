import pandas as pd
class RiskManager:
    def calculate_risk_reward(self, data_frame, stop_loss_pct, take_profit_pct):
        atr_multiplier = 1.5

        # Percentage-based stops
        data_frame['stop_loss_buy_pct'] = data_frame['close'] * (1 - stop_loss_pct / 100)
        data_frame['take_profit_buy_pct'] = data_frame['close'] * (1 + take_profit_pct / 100)
        data_frame['stop_loss_sell_pct'] = data_frame['close'] * (1 + stop_loss_pct / 100)
        data_frame['take_profit_sell_pct'] = data_frame['close'] * (1 - take_profit_pct / 100)

        # ATR-based stops
        data_frame['stop_loss_buy_atr'] = data_frame['close'] - (data_frame['atr'] * atr_multiplier)
        data_frame['take_profit_buy_atr'] = data_frame['close'] + (data_frame['atr'] * atr_multiplier * (take_profit_pct / stop_loss_pct))
        data_frame['stop_loss_sell_atr'] = data_frame['close'] + (data_frame['atr'] * atr_multiplier)
        data_frame['take_profit_sell_atr'] = data_frame['close'] - (data_frame['atr'] * atr_multiplier * (take_profit_pct / stop_loss_pct))

        # Conservative stops
        data_frame['stop_loss_buy'] = data_frame[['stop_loss_buy_pct', 'stop_loss_buy_atr']].max(axis=1)
        data_frame['take_profit_buy'] = data_frame[['take_profit_buy_pct', 'take_profit_buy_atr']].min(axis=1)
        data_frame['stop_loss_sell'] = data_frame[['stop_loss_sell_pct', 'stop_loss_sell_atr']].min(axis=1)
        data_frame['take_profit_sell'] = data_frame[['take_profit_sell_pct', 'take_profit_sell_atr']].max(axis=1)

        # Adjust with Fibonacci levels
        for i in range(len(data_frame)):
            row = data_frame.iloc[i]
            if 'fib_0' not in row or pd.isna(row['fib_0']):
                continue
            if row['buy_signal'] and row['fib_trend'] == 'uptrend':
                if (not pd.isna(row['fib_0_236']) and row['fib_0_236'] < row['close'] and 
                    row['fib_0_236'] > row['close'] * (1 - (stop_loss_pct * 1.5) / 100)):
                    data_frame.loc[data_frame.index[i], 'stop_loss_buy'] = row['fib_0_236']
                if (not pd.isna(row['fib_0_618']) and row['fib_0_618'] > row['close'] and 
                    row['fib_0_618'] < row['close'] * (1 + (take_profit_pct * 1.5) / 100)):
                    data_frame.loc[data_frame.index[i], 'take_profit_buy'] = row['fib_0_618']
                elif (not pd.isna(row['fib_0_786']) and row['fib_0_786'] > row['close'] and 
                      row['fib_0_786'] < row['close'] * (1 + (take_profit_pct * 2) / 100)):
                    data_frame.loc[data_frame.index[i], 'take_profit_buy'] = row['fib_0_786']
            elif row['sell_signal'] and row['fib_trend'] == 'downtrend':
                if (not pd.isna(row['fib_0_236']) and row['fib_0_236'] > row['close'] and 
                    row['fib_0_236'] < row['close'] * (1 + (stop_loss_pct * 1.5) / 100)):
                    data_frame.loc[data_frame.index[i], 'stop_loss_sell'] = row['fib_0_236']
                if (not pd.isna(row['fib_0_618']) and row['fib_0_618'] < row['close'] and 
                    row['fib_0_618'] > row['close'] * (1 - (take_profit_pct * 1.5) / 100)):
                    data_frame.loc[data_frame.index[i], 'take_profit_sell'] = row['fib_0_618']
                elif (not pd.isna(row['fib_0_786']) and row['fib_0_786'] < row['close'] and 
                      row['fib_0_786'] > row['close'] * (1 - (take_profit_pct * 2) / 100)):
                    data_frame.loc[data_frame.index[i], 'take_profit_sell'] = row['fib_0_786']
        return data_frame