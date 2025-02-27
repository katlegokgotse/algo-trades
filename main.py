import unittest
from dotenv import load_dotenv
import ccxt
import openai
import os
from trade_bot import TradingBot
from config import logger
# Add these imports for HTTP server
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
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': os.getenv("BINANCE_API_KEY"),
        'secret': os.getenv("BINANCE_SECRET_KEY"),
        'options': {'defaultType': 'future'}
    })
    
    bot = TradingBot(exchange=exchange, symbol='BTCUSD_PERP', timeframe='4h', position_size=0.01,
        stop_loss_pct=2.0, take_profit_pct=3.5, max_trades=3, dry_run=True, #Set to false for real time trading
        enable_telegram=True, telegram_bot_token=TELEGRAM_BOT_TOKEN, telegram_chat_id=TELEGRAM_CHAT_ID
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        logger.info("Running backtest...")
        backtest_results = bot.backtest()
        logger.info("Backtest completed. Starting live trading...")
        bot.start()