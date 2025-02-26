import unittest
from dotenv import load_dotenv
import ccxt
import openai
import os
import matplotlib.pyplot as plt
from trade_bot import TradingBot
from config import logger
# Load environment variables
load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("CHAT_API")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

# -------------------------------
# Main block
# -------------------------------
if __name__ == '__main__':
    import sys

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': os.getenv("BINANCE_API_KEY"),
        'secret': os.getenv("BINANCE_SECRET_KEY"),
        'options': {'defaultType': 'future'}
    })

    bot = TradingBot(
        exchange=exchange,
        symbol='BTCUSD_PERP',
        timeframe='4h',
        position_size=0.01,
        stop_loss_pct=2.0,
        take_profit_pct=3.5,
        max_trades=3,
        dry_run=True, #Set to false for real time trading
        enable_telegram=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        logger.info("Running backtest...")
        backtest_results = bot.backtest()
        logger.info("Backtest completed. Starting live trading...")
        bot.start()