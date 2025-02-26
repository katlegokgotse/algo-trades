import threading
import requests

class Notifier:
    def __init__(self, enabled, bot_token, chat_id):
        self.enabled = enabled
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message):
        if not self.enabled or not self.bot_token or not self.chat_id:
            logger.info("Telegram notifications not enabled")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
            response = requests.post(url, data=payload)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def notify_trade(self, trade, indicators):
        message = (f"*New {trade.side.upper()} Trade*\n"
                   f"Symbol: {trade.symbol}\nEntry Price: {trade.entry_price}\n"
                   f"Stop Loss: {trade.stop_loss}\nTake Profit: {trade.take_profit}\n"
                   f"Position Size: {trade.quantity}\nTime: {trade.entry_time}\n")
        if indicators:
            message += "\n*Key Indicators:*\n" + "\n".join(
                f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in indicators.items()
            )
        threading.Thread(target=self.send_message, args=(message,)).start()