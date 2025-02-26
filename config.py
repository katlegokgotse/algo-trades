# config.py
import logging


FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")
