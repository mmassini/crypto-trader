import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from binance import AsyncClient
from binance.enums import KLINE_INTERVAL_1MINUTE

from agents.ml_analysis.feature_engineer import build_features
from agents.ml_analysis.xgboost_model import CryptoModel
from config.settings import settings
from storage.schema import Session, ModelVersion, Trade

logger = logging.getLogger(__name__)


def _label_signal(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0003) -> pd.Series:
    """
    Genera etiquetas: 2=LONG, 1=FLAT, 0=SHORT.
    Basado en retorno futuro a `horizon` velas.
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    labels = pd.Series(1, index=df.index)  # FLAT por defecto
    labels[future_return > threshold] = 2   # LONG
    labels[future_return < -threshold] = 0  # SHORT
    return labels


async def fetch_historical(client: AsyncClient, symbol: str, days: int = 180) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    klines = await client.get_historical_klines(
        symbol=symbol,
        interval=KLINE_INTERVAL_1MINUTE,
        start_str=str(int(start.timestamp() * 1000)),
        end_str=str(int(end.timestamp() * 1000)),
    )
    rows = []
    for k in klines:
        rows.append({
            "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]), "quote_volume": float(k[7]),
            "trades": int(k[8]), "taker_buy_volume": float(k[9]),
        })
    df = pd.DataFrame(rows)
    df.set_index("open_time", inplace=True)
    logger.info(f"Fetched {len(df)} rows for {symbol}")
    return df


def compute_sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252 * 24 * 60))


async def run_retrain(symbols: list[str]):
    # Datos históricos desde Binance real (endpoint público, sin auth)
    # El testnet no tiene histórico suficiente para entrenar
    client = await AsyncClient.create()
    try:
        for symbol in symbols:
            logger.info(f"Retraining {symbol}...")
            df = await fetch_historical(client, symbol, days=settings.lookback_days)
            features = build_features(df)
            labels = _label_signal(df).reindex(features.index).dropna()
            features = features.reindex(labels.index)

            # Walk-forward: 80% train, 20% val
            split = int(len(features) * 0.8)
            X_train, y_train = features.iloc[:split], labels.iloc[:split]
            X_val, y_val = features.iloc[split:], labels.iloc[split:]

            model = CryptoModel(symbol)
            model.train(X_train, y_train)

            # Evaluar en validación
            preds = [model.predict(X_val.iloc[[i]])[0] for i in range(len(X_val))]
            pred_returns = pd.Series([
                0.001 if p == "LONG" else (-0.001 if p == "SHORT" else 0.0)
                for p in preds
            ])
            sharpe = compute_sharpe(pred_returns)
            accuracy = float((pd.Series(preds).map({"LONG": 2, "FLAT": 1, "SHORT": 0}).values == y_val.values).mean())

            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

            # Champion challenge: nuevo modelo debe superar sharpe del campeón en 5%
            promote = False
            with Session() as session:
                champion = session.query(ModelVersion).filter(
                    ModelVersion.symbol == symbol,
                    ModelVersion.is_champion == True,
                ).order_by(ModelVersion.trained_at.desc()).first()

                if champion is None or sharpe > (champion.sharpe or 0) * 1.05:
                    promote = True
                    if champion:
                        champion.is_champion = False
                        session.commit()

                mv = ModelVersion(symbol=symbol, version=version, sharpe=sharpe, accuracy=accuracy, is_champion=promote)
                session.add(mv)
                session.commit()

            model.save(version, is_champion=promote)
            logger.info(f"{symbol} retrain done: sharpe={sharpe:.3f} acc={accuracy:.1%} champion={promote}")
    finally:
        await client.close_connection()
