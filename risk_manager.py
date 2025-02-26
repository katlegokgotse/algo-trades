import pandas as pd
class RiskManager:
    def calculate_risk_reward(self, df, stop_loss_pct, take_profit_pct):
        atr_multiplier = 1.5

        # Percentage-based stops
        df['stop_loss_buy_pct'] = df['close'] * (1 - stop_loss_pct / 100)
        df['take_profit_buy_pct'] = df['close'] * (1 + take_profit_pct / 100)
        df['stop_loss_sell_pct'] = df['close'] * (1 + stop_loss_pct / 100)
        df['take_profit_sell_pct'] = df['close'] * (1 - take_profit_pct / 100)

        # ATR-based stops
        df['stop_loss_buy_atr'] = df['close'] - (df['atr'] * atr_multiplier)
        df['take_profit_buy_atr'] = df['close'] + (df['atr'] * atr_multiplier * (take_profit_pct / stop_loss_pct))
        df['stop_loss_sell_atr'] = df['close'] + (df['atr'] * atr_multiplier)
        df['take_profit_sell_atr'] = df['close'] - (df['atr'] * atr_multiplier * (take_profit_pct / stop_loss_pct))

        # Conservative stops
        df['stop_loss_buy'] = df[['stop_loss_buy_pct', 'stop_loss_buy_atr']].max(axis=1)
        df['take_profit_buy'] = df[['take_profit_buy_pct', 'take_profit_buy_atr']].min(axis=1)
        df['stop_loss_sell'] = df[['stop_loss_sell_pct', 'stop_loss_sell_atr']].min(axis=1)
        df['take_profit_sell'] = df[['take_profit_sell_pct', 'take_profit_sell_atr']].max(axis=1)

        # Adjust with Fibonacci levels
        for i in range(len(df)):
            row = df.iloc[i]
            if 'fib_0' not in row or pd.isna(row['fib_0']):
                continue
            if row['buy_signal'] and row['fib_trend'] == 'uptrend':
                if (not pd.isna(row['fib_0_236']) and row['fib_0_236'] < row['close'] and 
                    row['fib_0_236'] > row['close'] * (1 - (stop_loss_pct * 1.5) / 100)):
                    df.loc[df.index[i], 'stop_loss_buy'] = row['fib_0_236']
                if (not pd.isna(row['fib_0_618']) and row['fib_0_618'] > row['close'] and 
                    row['fib_0_618'] < row['close'] * (1 + (take_profit_pct * 1.5) / 100)):
                    df.loc[df.index[i], 'take_profit_buy'] = row['fib_0_618']
                elif (not pd.isna(row['fib_0_786']) and row['fib_0_786'] > row['close'] and 
                      row['fib_0_786'] < row['close'] * (1 + (take_profit_pct * 2) / 100)):
                    df.loc[df.index[i], 'take_profit_buy'] = row['fib_0_786']
            elif row['sell_signal'] and row['fib_trend'] == 'downtrend':
                if (not pd.isna(row['fib_0_236']) and row['fib_0_236'] > row['close'] and 
                    row['fib_0_236'] < row['close'] * (1 + (stop_loss_pct * 1.5) / 100)):
                    df.loc[df.index[i], 'stop_loss_sell'] = row['fib_0_236']
                if (not pd.isna(row['fib_0_618']) and row['fib_0_618'] < row['close'] and 
                    row['fib_0_618'] > row['close'] * (1 - (take_profit_pct * 1.5) / 100)):
                    df.loc[df.index[i], 'take_profit_sell'] = row['fib_0_618']
                elif (not pd.isna(row['fib_0_786']) and row['fib_0_786'] < row['close'] and 
                      row['fib_0_786'] > row['close'] * (1 - (take_profit_pct * 2) / 100)):
                    df.loc[df.index[i], 'take_profit_sell'] = row['fib_0_786']
        return df