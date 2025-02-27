import openai
from config import logger
def chatgpt_analyze_trade(trade_details):
    prompt = (f"You are a trading analysis assistant. A trade signal has been generated with the following details:\n"
              f"Trade type: {trade_details['trade_type']}\nSymbol: {trade_details['symbol']}\n"
              f"Current Price: {trade_details['current_price']}\nStop Loss: {trade_details['stop_loss']}\n"
              f"Take Profit: {trade_details['take_profit']}\nRSI: {trade_details['rsi']}\n"
              f"ADX: {trade_details['adx']}\nMACD: {trade_details['macd']}\n"
              f"EMA 20: {trade_details['ema_20']}\nEMA 50: {trade_details['ema_50']}\n"
              f"EMA 200: {trade_details['ema_200']}\nFibonacci Trend: {trade_details.get('fib_trend', 'N/A')}\n"
              "Based on these details, do you recommend executing this trade? Respond with 'GO' or 'HOLD'.")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-40-mini",
            messages=[{"role": "system", "content": "You are a trading analysis assistant."}, 
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().upper()
        logger.info(f"ChatGPT response: {answer}")
        return answer == "GO"
    except Exception as e:
        logger.error(f"Error calling ChatGPT API: {e}")
        return False