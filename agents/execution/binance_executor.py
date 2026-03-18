import logging
from datetime import datetime, timezone, timedelta

from binance import AsyncClient
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, FUTURE_ORDER_TYPE_STOP_MARKET, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET

from config.settings import settings
from storage.schema import Session, Trade

logger = logging.getLogger(__name__)


_MARGIN_COOLDOWN_SECONDS = 600  # 10 min after insufficient margin error


class BinanceExecutor:
    def __init__(self, client: AsyncClient):
        self.client = client
        self._qty_precision: dict[str, int] = {}
        self._margin_cooldown_until: datetime | None = None

    async def _get_qty_precision(self, symbol: str) -> int:
        if symbol not in self._qty_precision:
            try:
                info = await self.client.futures_exchange_info()
                for s in info["symbols"]:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = f["stepSize"]
                            precision = len(step.rstrip("0").split(".")[-1]) if "." in step else 0
                            self._qty_precision[s["symbol"]] = precision
            except Exception as e:
                logger.error(f"Error fetching exchange info: {e}")
                self._qty_precision[symbol] = 0
        return self._qty_precision.get(symbol, 3)

    async def open_position(
        self,
        symbol: str,
        direction: str,   # LONG / SHORT
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
    ) -> Trade | None:
        side = SIDE_BUY if direction == "LONG" else SIDE_SELL
        close_side = SIDE_SELL if direction == "LONG" else SIDE_BUY

        # Cooldown after insufficient margin error
        if self._margin_cooldown_until:
            now = datetime.now(timezone.utc)
            if now < self._margin_cooldown_until:
                remaining = int((self._margin_cooldown_until - now).total_seconds())
                logger.warning(f"Skipping {symbol} — margin cooldown active ({remaining}s remaining)")
                return None
            self._margin_cooldown_until = None

        try:
            precision = await self._get_qty_precision(symbol)
            quantity = round(quantity, precision)
            if quantity <= 0:
                logger.warning(f"Quantity too small after rounding for {symbol}, skipping")
                return None

            # Establecer apalancamiento 10x
            try:
                await self.client.futures_change_leverage(symbol=symbol, leverage=10)
            except Exception:
                pass

            # Orden de entrada (market)
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            order_id = str(order["orderId"])

            # Stop loss + take profit — if these fail, close entry immediately
            try:
                await self.client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=round(stop_loss, 2),
                    quantity=quantity,
                    reduceOnly=True,
                )
                await self.client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=round(take_profit, 2),
                    quantity=quantity,
                    reduceOnly=True,
                )
            except Exception as sl_exc:
                logger.error(f"SL/TP order failed for {symbol}, closing entry position: {sl_exc}")
                try:
                    await self.client.futures_create_order(
                        symbol=symbol,
                        side=close_side,
                        type=ORDER_TYPE_MARKET,
                        quantity=quantity,
                        reduceOnly=True,
                    )
                except Exception as close_exc:
                    logger.error(f"Failed to close orphan position {symbol}: {close_exc}")
                return None

            trade = Trade(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                order_id=order_id,
                is_open=True,
            )
            with Session() as session:
                session.add(trade)
                session.commit()
                session.refresh(trade)

            logger.info(f"Opened {direction} {symbol} qty={quantity} entry={entry_price} SL={stop_loss} TP={take_profit}")
            return trade

        except Exception as e:
            if "-2019" in str(e):
                self._margin_cooldown_until = datetime.now(timezone.utc) + timedelta(
                    seconds=_MARGIN_COOLDOWN_SECONDS
                )
                logger.error(f"Margin insufficient for {symbol} — cooldown {_MARGIN_COOLDOWN_SECONDS}s: {e}")
            else:
                logger.error(f"Error opening position {symbol}: {e}")
            return None

    async def close_position(self, trade: Trade, exit_price: float) -> float:
        """Cierra posición y retorna PnL."""
        side = SIDE_SELL if trade.direction == "LONG" else SIDE_BUY
        pnl = 0.0

        try:
            await self.client.futures_create_order(
                symbol=trade.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=trade.quantity,
                reduceOnly=True,
            )

            if trade.direction == "LONG":
                pnl = (exit_price - trade.entry_price) * trade.quantity
            else:
                pnl = (trade.entry_price - exit_price) * trade.quantity

            with Session() as session:
                db_trade = session.get(Trade, trade.id)
                db_trade.exit_price = exit_price
                db_trade.pnl = pnl
                db_trade.closed_at = datetime.now(timezone.utc)
                db_trade.is_open = False
                session.commit()

            logger.info(f"Closed {trade.symbol} PnL={pnl:.2f}")

        except Exception as e:
            logger.error(f"Error closing position {trade.symbol}: {e}")

        return pnl

    async def close_all_positions(self) -> float:
        """Cierra todas las posiciones abiertas. Retorna PnL total."""
        total_pnl = 0.0
        try:
            positions = await self.client.futures_position_information()
            for pos in positions:
                amt = float(pos["positionAmt"])
                if amt == 0:
                    continue
                symbol = pos["symbol"]
                side = SIDE_SELL if amt > 0 else SIDE_BUY
                await self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=abs(amt),
                    reduceOnly=True,
                )
                logger.info(f"Force-closed {symbol} amt={amt}")
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
        return total_pnl

    async def get_balance(self) -> float:
        try:
            account = await self.client.futures_account()
            return float(account["availableBalance"])
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
