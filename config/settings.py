from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Binance
    binance_api_key: str = Field("", env="BINANCE_API_KEY")
    binance_secret_key: str = Field("", env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(True, env="BINANCE_TESTNET")

    # Telegram
    telegram_bot_token: str = Field("", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field("", env="TELEGRAM_CHAT_ID")

    # Risk
    risk_per_trade: float = 0.01       # 1% por operación
    max_daily_drawdown: float = 0.05   # 5% halt diario
    max_concurrent_positions: int = 3

    # ML
    min_confidence: float = 0.50
    lookback_days: int = 180

    # Trading
    decision_interval_seconds: int = 60   # cada 1 minuto (1m bars)
    report_interval_hours: int = 2

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
