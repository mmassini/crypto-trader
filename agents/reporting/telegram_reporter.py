import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class TelegramReporter:
    def __init__(self):
        self._base = f"https://api.telegram.org/bot{settings.telegram_bot_token}"

    async def send(self, message: str):
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            logger.debug(f"Telegram not configured, skipping: {message}")
            return
        logger.info(f"Telegram token prefix: {settings.telegram_bot_token[:10]}... chat_id: {settings.telegram_chat_id}")
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self._base}/sendMessage",
                    json={
                        "chat_id": settings.telegram_chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.error(f"Telegram error: {e}")
