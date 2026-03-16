import logging
from datetime import date

from config.settings import settings
from storage.schema import Session, EquitySnapshot, Trade

logger = logging.getLogger(__name__)


class RiskAgent:
    def __init__(self):
        self._daily_pnl: float = 0.0
        self._last_reset: date | None = None
        self._halted: bool = False

    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        """
        Calcula cantidad a operar basada en riesgo por trade.
        Riesgo = balance × risk_per_trade
        Qty = riesgo / |entry - stop_loss|
        """
        risk_amount = balance * settings.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0.0
        qty = risk_amount / risk_per_unit
        return round(qty, 6)

    def check_daily_halt(self, current_balance: float, starting_balance: float) -> bool:
        """Retorna True si se superó el drawdown diario y hay que parar."""
        drawdown = (starting_balance - current_balance) / starting_balance
        if drawdown >= settings.max_daily_drawdown:
            logger.warning(f"Daily drawdown halt triggered: {drawdown:.1%}")
            self._halted = True
        return self._halted

    def is_halted(self) -> bool:
        return self._halted

    def reset_daily(self):
        self._daily_pnl = 0.0
        self._halted = False
        logger.info("Risk agent daily reset")

    def count_open_positions(self) -> int:
        with Session() as session:
            return session.query(Trade).filter(Trade.is_open == True).count()

    def can_open_position(self) -> bool:
        if self._halted:
            return False
        open_pos = self.count_open_positions()
        return open_pos < settings.max_concurrent_positions

    def record_equity(self, balance: float):
        with Session() as session:
            snap = EquitySnapshot(balance=balance)
            session.add(snap)
            session.commit()
