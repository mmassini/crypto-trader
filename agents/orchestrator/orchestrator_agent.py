import asyncio
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import yaml

from agents.data_ingest.binance_stream import BinanceStream
from agents.execution.binance_executor import BinanceExecutor
from agents.ml_analysis.feature_engineer import build_features
from agents.ml_analysis.xgboost_model import CryptoModel
from agents.reporting.telegram_reporter import TelegramReporter
from agents.risk.position_sizer import RiskAgent
from binance import AsyncClient
from config.settings import settings
from storage.schema import Session, Trade

logger = logging.getLogger(__name__)

UY = ZoneInfo("America/Montevideo")

# Horarios de reporte en hora Uruguay
# (hora, minuto, tipo)
REPORT_SCHEDULE = [
    (7, 30, "morning"),    # Resumen madrugada
    (10, 30, "periodic"),
    (13, 30, "periodic"),
    (16, 30, "periodic"),
    (19, 30, "periodic"),
    (22, 0,  "close"),     # Cierre del día
]


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
        self._sent_reports: set = set()  # evita duplicados

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

        self._starting_balance = await self.executor.get_wallet_balance()
        self.risk.record_equity(self._starting_balance)
        self._running = True

        await self.reporter.send(
            f"🤖 <b>[Crypto Trader] Iniciado</b>\n"
            f"Símbolos: {', '.join(self.symbols)}\n"
            f"Balance: ${self._starting_balance:,.2f}\n"
            f"Modo: {'TESTNET' if settings.binance_testnet else 'REAL'}"
        )

        await asyncio.gather(
            self._loop(),
            self._report_loop(),
        )

    # ── Trading loop ──────────────────────────────────────────────────────────

    async def _loop(self):
        while self._running:
            try:
                await self._decision_cycle()
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
            await asyncio.sleep(settings.decision_interval_seconds)

    async def _decision_cycle(self):
        wallet_balance = await self.executor.get_wallet_balance()
        if self.risk.check_daily_halt(wallet_balance, self._starting_balance):
            logger.warning("Daily halt active, skipping cycle")
            return

        balance = await self.executor.get_balance()  # available margin for sizing

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
            logger.info(f"{symbol} → {signal} ({confidence:.1%})")

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
                    f"📊 <b>[Crypto Trader] Trade abierto</b>\n"
                    f"{'🟢' if signal == 'LONG' else '🔴'} {signal} {symbol}\n"
                    f"Cantidad: {qty}\n"
                    f"Entrada: {current_price:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}\n"
                    f"Confianza: {confidence:.1%}"
                )

    # ── Report loop ───────────────────────────────────────────────────────────

    async def _report_loop(self):
        """Chequea cada minuto si corresponde enviar un reporte programado."""
        while self._running:
            now = datetime.now(UY)
            for hour, minute, report_type in REPORT_SCHEDULE:
                key = (now.date(), hour, minute)
                if now.hour == hour and now.minute == minute and key not in self._sent_reports:
                    self._sent_reports.add(key)
                    try:
                        await self._send_scheduled_report(report_type, now)
                    except Exception as e:
                        logger.error(f"Report error ({report_type}): {e}")
            await asyncio.sleep(60)

    async def _send_scheduled_report(self, report_type: str, now: datetime):
        balance = await self.executor.get_balance() if self.executor else 0.0
        pnl_session = balance - self._starting_balance

        # Determinar ventana de tiempo para el reporte
        if report_type == "morning":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
            title = "🌅 <b>[Crypto Trader] Resumen madrugada</b>"
        elif report_type == "close":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
            title = "🌙 <b>[Crypto Trader] Cierre del día</b>"
        else:
            since = now - timedelta(hours=3)
            title = "📈 <b>[Crypto Trader] Reporte periódico</b>"

        since_utc = since.astimezone(timezone.utc).replace(tzinfo=None)

        # Consultar trades en la ventana
        with Session() as session:
            trades = session.query(Trade).filter(
                Trade.opened_at >= since_utc,
                Trade.is_open == False,
            ).all()

            open_trades = session.query(Trade).filter(Trade.is_open == True).all()

        total_trades = len(trades)
        winning = [t for t in trades if (t.pnl or 0) > 0]
        losing  = [t for t in trades if (t.pnl or 0) < 0]
        pnl_period = sum(t.pnl or 0 for t in trades)
        win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0

        open_info = ""
        if open_trades:
            open_info = f"\n📂 Posiciones abiertas: {len(open_trades)}"

        msg = (
            f"{title}\n"
            f"🕐 {now.strftime('%H:%M')} UY\n\n"
            f"💰 Balance: ${balance:,.2f}\n"
            f"📊 PnL sesión: ${pnl_session:+,.2f}\n\n"
            f"🔢 Trades cerrados: {total_trades}\n"
            f"✅ Ganadores: {len(winning)} | ❌ Perdedores: {len(losing)}\n"
            f"🎯 Win rate: {win_rate:.1f}%\n"
            f"💵 PnL período: ${pnl_period:+,.2f}"
            f"{open_info}"
        )

        await self.reporter.send(msg)

    # ── Shutdown ──────────────────────────────────────────────────────────────

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
            f"🛑 <b>[Crypto Trader] Sesión terminada</b>\n"
            f"Balance final: ${balance:,.2f}\n"
            f"PnL sesión: ${pnl:+,.2f}"
        )
