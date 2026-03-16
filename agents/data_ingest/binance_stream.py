import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from binance.enums import KLINE_INTERVAL_1MINUTE

from config.settings import settings

logger = logging.getLogger(__name__)

# Almacena las últimas N velas por símbolo
CANDLE_BUFFER = 300  # suficiente para calcular todos los indicadores


class BinanceStream:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.client: AsyncClient | None = None
        self.bm: BinanceSocketManager | None = None
        self._candles: dict[str, deque] = defaultdict(lambda: deque(maxlen=CANDLE_BUFFER))
        self._running = False

    async def start(self):
        self.client = await AsyncClient.create(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet,
        )
        self.bm = BinanceSocketManager(self.client)
        await self._backfill()
        self._running = True
        asyncio.create_task(self._stream_loop())
        logger.info(f"BinanceStream started for {self.symbols}")

    async def stop(self):
        self._running = False
        if self.client:
            await self.client.close_connection()

    async def _backfill(self):
        """Carga las últimas CANDLE_BUFFER velas históricas para cada símbolo."""
        for symbol in self.symbols:
            try:
                klines = await self.client.get_klines(
                    symbol=symbol,
                    interval=KLINE_INTERVAL_1MINUTE,
                    limit=CANDLE_BUFFER,
                )
                for k in klines:
                    self._candles[symbol].append(_parse_kline(k))
                logger.info(f"Backfilled {len(klines)} candles for {symbol}")
            except Exception as e:
                logger.error(f"Backfill error {symbol}: {e}")

    async def _stream_loop(self):
        streams = [f"{s.lower()}@kline_1m" for s in self.symbols]
        async with self.bm.multiplex_socket(streams) as ms:
            while self._running:
                try:
                    msg = await ms.recv()
                    if msg and msg.get("data", {}).get("e") == "kline":
                        kline = msg["data"]["k"]
                        if kline["x"]:  # vela cerrada
                            symbol = kline["s"]
                            self._candles[symbol].append({
                                "open_time": datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc),
                                "open": float(kline["o"]),
                                "high": float(kline["h"]),
                                "low": float(kline["l"]),
                                "close": float(kline["c"]),
                                "volume": float(kline["v"]),
                                "quote_volume": float(kline["q"]),
                                "trades": int(kline["n"]),
                                "taker_buy_volume": float(kline["V"]),
                            })
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    await asyncio.sleep(1)

    def get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        candles = list(self._candles[symbol])
        if len(candles) < 50:
            return None
        df = pd.DataFrame(candles)
        df.set_index("open_time", inplace=True)
        return df

    def is_ready(self, symbol: str) -> bool:
        return len(self._candles[symbol]) >= 50


def _parse_kline(k: list) -> dict:
    return {
        "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
        "quote_volume": float(k[7]),
        "trades": int(k[8]),
        "taker_buy_volume": float(k[9]),
    }
