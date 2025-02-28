# config.py
import logging

from dotenv import load_dotenv
from openai import OpenAI
import os 

FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

client = OpenAI(
    api_key=os.getenv("CHAT_API"),  # This is the default and can be omitted
)

