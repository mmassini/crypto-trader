import asyncio
import logging
from datetime import datetime, timezone

import yaml

from agents.data_ingest.binance_stream import BinanceStream
from agents.execution.binance_executor import BinanceExecutor
from agents.ml_analysis.feature_engineer import build_features
from agents.ml_analysis.xgboost_model import CryptoModel
from agents.reporting.telegram_reporter import TelegramReporter
from agents.risk.position_sizer import RiskAgent
from binance import AsyncClient
from config.settings import settings

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        with open("config/instruments.yaml") as f:
            cfg = yaml.safe_load(f)
        self.symbols: list[str] = cfg["symbols"]
        self.atr_sl_mult: float = cfg["atr_multiplier_sl"]
        self.rr_ratio: float = cfg["reward_risk_ratio"]

        self.models: dict[str, CryptoModel] = {s: CryptoModel(s) for s in self.symbols}
        self.risk = RiskAgent()
        self.reporter = TelegramReporter()
        self.stream: BinanceStream | None = None
        self.executor: BinanceExecutor | None = None
        self.client: AsyncClient | None = None
        self._starting_balance: float = 0.0
        self._running = False

    async def start(self):
        logger.info("Starting Orchestrator")
        self.client = await AsyncClient.create(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_secret_key,
            testnet=settings.binance_testnet,
        )
        self.executor = BinanceExecutor(self.client)
        self.stream = BinanceStream(self.symbols)
        await self.stream.start()

        self._starting_balance = await self.executor.get_balance()
        self.risk.record_equity(self._starting_balance)
        self._running = True

        await self.reporter.send(
            f"Crypto Trader iniciado\n"
            f"Símbolos: {', '.join(self.symbols)}\n"
            f"Balance: ${self._starting_balance:,.2f}\n"
            f"Modo: {'TESTNET' if settings.binance_testnet else 'REAL'}"
        )

        await self._loop()

    async def _loop(self):
        while self._running:
            try:
                await self._decision_cycle()
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
            await asyncio.sleep(settings.decision_interval_seconds)

    async def _decision_cycle(self):
        balance = await self.executor.get_balance()
        if self.risk.check_daily_halt(balance, self._starting_balance):
            logger.warning("Daily halt active, skipping cycle")
            return

        for symbol in self.symbols:
            if not self.stream.is_ready(symbol):
                continue

            df = self.stream.get_dataframe(symbol)
            if df is None:
                continue

            try:
                features = build_features(df)
            except Exception as e:
                logger.error(f"Feature error {symbol}: {e}")
                continue

            model = self.models[symbol]
            if not model.is_trained():
                continue

            signal, confidence = model.predict(features)

            if signal == "FLAT" or confidence < settings.min_confidence:
                continue

            if not self.risk.can_open_position():
                continue

            current_price = float(df["close"].iloc[-1])
            atr = features["atr_14"].iloc[-1]
            stop_loss = (
                current_price - self.atr_sl_mult * atr
                if signal == "LONG"
                else current_price + self.atr_sl_mult * atr
            )
            take_profit = (
                current_price + self.rr_ratio * self.atr_sl_mult * atr
                if signal == "LONG"
                else current_price - self.rr_ratio * self.atr_sl_mult * atr
            )

            qty = self.risk.calculate_position_size(balance, current_price, stop_loss)
            if qty <= 0:
                continue

            trade = await self.executor.open_position(
                symbol=symbol,
                direction=signal,
                quantity=qty,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
            )

            if trade:
                await self.reporter.send(
                    f"Trade abierto\n"
                    f"{signal} {symbol}\n"
                    f"Cantidad: {qty}\n"
                    f"Entrada: {current_price:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}\n"
                    f"Confianza: {confidence:.1%}"
                )

    async def stop(self):
        self._running = False
        logger.info("Stopping orchestrator, closing all positions...")
        if self.executor:
            await self.executor.close_all_positions()
        if self.stream:
            await self.stream.stop()
        if self.client:
            await self.client.close_connection()
        balance = await self.executor.get_balance() if self.executor else 0.0
        pnl = balance - self._starting_balance
        await self.reporter.send(
            f"Sesión terminada\n"
            f"Balance final: ${balance:,.2f}\n"
            f"PnL sesión: ${pnl:+,.2f}"
        )
