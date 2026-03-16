import numpy as np
import pandas as pd
import ta


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera features técnicas + crypto-específicas a partir de velas 1m.
    Retorna DataFrame con columnas de features, sin NaN.
    """
    f = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    taker_buy = df["taker_buy_volume"]

    # --- Retornos ---
    f["return_1"] = close.pct_change(1)
    f["return_3"] = close.pct_change(3)
    f["return_5"] = close.pct_change(5)
    f["return_10"] = close.pct_change(10)

    # --- RSI ---
    f["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    f["rsi_7"] = ta.momentum.RSIIndicator(close, window=7).rsi()

    # --- MACD ---
    macd = ta.trend.MACD(close)
    f["macd"] = macd.macd()
    f["macd_signal"] = macd.macd_signal()
    f["macd_diff"] = macd.macd_diff()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close, window=20)
    f["bb_upper"] = bb.bollinger_hband()
    f["bb_lower"] = bb.bollinger_lband()
    f["bb_pct"] = bb.bollinger_pband()
    f["bb_width"] = bb.bollinger_wband()

    # --- ATR ---
    f["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    f["atr_pct"] = f["atr_14"] / close  # ATR normalizado

    # --- ADX ---
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    f["adx"] = adx.adx()
    f["adx_pos"] = adx.adx_pos()
    f["adx_neg"] = adx.adx_neg()

    # --- EMA crossovers ---
    ema_9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema_21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    f["ema_9_21_cross"] = (ema_9 - ema_21) / close
    f["ema_21_50_cross"] = (ema_21 - ema_50) / close
    f["price_vs_ema21"] = (close - ema_21) / ema_21

    # --- Stochastic ---
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    f["stoch_k"] = stoch.stoch()
    f["stoch_d"] = stoch.stoch_signal()

    # --- CCI ---
    f["cci"] = ta.trend.CCIIndicator(high, low, close).cci()

    # --- OBV ---
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    f["obv_slope"] = obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-8)

    # --- VWAP (aproximado con 1m bars) ---
    typical_price = (high + low + close) / 3
    f["vwap_dist"] = (close - (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()) / close

    # --- Volume delta (presión compradora vs vendedora) ---
    seller_volume = volume - taker_buy
    f["volume_delta"] = (taker_buy - seller_volume) / (volume + 1e-8)
    f["volume_delta_ma"] = f["volume_delta"].rolling(5).mean()

    # --- Volatilidad rolling ---
    f["volatility_10"] = close.pct_change().rolling(10).std()
    f["volatility_30"] = close.pct_change().rolling(30).std()

    # --- Features de tiempo ---
    f["hour"] = df.index.hour
    f["minute"] = df.index.minute
    f["day_of_week"] = df.index.dayofweek

    # --- Lags ---
    for lag in [1, 2, 3, 5]:
        f[f"return_lag_{lag}"] = f["return_1"].shift(lag)
        f[f"rsi_lag_{lag}"] = f["rsi_14"].shift(lag)

    f.dropna(inplace=True)
    return f


FEATURE_COLUMNS = None  # se setea al entrenar el primer modelo


def get_feature_names() -> list[str]:
    dummy = pd.DataFrame({
        "open": [1.0] * 200, "high": [1.1] * 200, "low": [0.9] * 200,
        "close": [1.0] * 200, "volume": [1000.0] * 200,
        "quote_volume": [1000.0] * 200, "trades": [100] * 200,
        "taker_buy_volume": [500.0] * 200,
    })
    dummy.index = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
    return list(build_features(dummy).columns)
