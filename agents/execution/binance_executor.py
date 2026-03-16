import logging
from datetime import datetime, timezone

from binance import AsyncClient
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, FUTURE_ORDER_TYPE_STOP_MARKET, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET

from config.settings import settings
from storage.schema import Session, Trade

logger = logging.getLogger(__name__)


class BinanceExecutor:
    def __init__(self, client: AsyncClient):
        self.client = client

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

        try:
            # Orden de entrada (market)
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            order_id = str(order["orderId"])

            # Stop loss
            await self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=round(stop_loss, 2),
                closePosition=True,
            )

            # Take profit
            await self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=round(take_profit, 2),
                closePosition=True,
            )

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
            return float(account["totalWalletBalance"])
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
