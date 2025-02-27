import unittest
from dotenv import load_dotenv
import ccxt
import openai
import os
from trade_bot import TradingBot
from config import logger
import http.server
import threading

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("CHAT_API")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Simple HTTP server to keep the service alive
def start_http_server():
    port = int(os.environ.get("PORT", 8080))
    
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Trading bot is running")
            
    httpd = http.server.HTTPServer(('0.0.0.0', port), Handler)
    logger.info(f"Starting HTTP server on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    import sys
    
    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # Initialize ccxt Luno exchange
    exchange = ccxt.luno({
        'apiKey': os.getenv("LUNO_ID"),
        'secret': os.getenv("LUNO_SECRET"),
        'enableRateLimit': True,  # Respect rate limits
    })

    # Fetch and log available markets
    try:
        markets = exchange.load_markets()
        available_pairs = list(markets.keys())
        #logger.info(f"Available trading pairs on Luno via ccxt: {available_pairs}")
    except Exception as e:
        logger.error(f"Failed to load markets from Luno via ccxt: {e}")
        sys.exit(1)

    # Set the trading pair (replace with a valid pair from the list above)
    trading_pair = "BTC/USDT"  # ccxt uses '/' instead of Luno's concatenated format
    if trading_pair not in available_pairs:
        logger.error(f"Selected pair {trading_pair} is not available. Choose from: {available_pairs}")
        sys.exit(1)

    # Fetch ticker data to verify connection
    try:
        ticker = exchange.fetch_ticker(trading_pair)
        #logger.info(f"Ticker data for {trading_pair}: {ticker}")
    except Exception as e:
        logger.error(f"Failed to fetch ticker from Luno via ccxt for {trading_pair}: {e}")
        sys.exit(1)

    # Initialize TradingBot with the ccxt exchange object
    bot = TradingBot(
        exchange=exchange,  # Pass the ccxt exchange object directly
        symbol='XBTUSDT',
        timeframe='4h',
        position_size=0.001,
        stop_loss_pct=2.0,
        take_profit_pct=3.5,
        max_trades=3,
        dry_run=False,  # Set to False for real-time trading
        enable_telegram=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        logger.info("Running backtest...")
        backtest_results = bot.backtest()
        #logger.info(f"Backtest results: {backtest_results}")
        logger.info("Backtest completed. Starting live trading...")
        bot.start()