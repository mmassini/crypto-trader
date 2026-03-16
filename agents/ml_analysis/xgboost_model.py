import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Clases: 0=SHORT, 1=FLAT, 2=LONG
CLASS_MAP = {0: "SHORT", 1: "FLAT", 2: "LONG"}


class CryptoModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model: XGBClassifier | None = None
        self.feature_names: list[str] = []
        self._load_champion()

    def predict(self, features: pd.DataFrame) -> tuple[str, float]:
        """Retorna (signal, confidence). signal = LONG / SHORT / FLAT."""
        if self.model is None:
            return "FLAT", 0.0
        X = features[self.feature_names].iloc[[-1]]
        proba = self.model.predict_proba(X)[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        return CLASS_MAP[class_idx], confidence

    def is_trained(self) -> bool:
        return self.model is not None

    def train(self, X: pd.DataFrame, y: pd.Series, params: dict | None = None):
        if params is None:
            params = {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
                "random_state": 42,
            }
        self.model = XGBClassifier(**params)
        self.model.fit(X, y)
        self.feature_names = list(X.columns)

    def save(self, version: str, is_champion: bool = False):
        path = MODELS_DIR / f"{self.symbol}_{version}.json"
        self.model.save_model(str(path))
        meta = {"feature_names": self.feature_names, "version": version}
        meta_path = MODELS_DIR / f"{self.symbol}_{version}_meta.json"
        meta_path.write_text(json.dumps(meta))
        if is_champion:
            champion_path = MODELS_DIR / f"{self.symbol}_champion.json"
            champion_meta_path = MODELS_DIR / f"{self.symbol}_champion_meta.json"
            import shutil
            shutil.copy(path, champion_path)
            shutil.copy(meta_path, champion_meta_path)
        logger.info(f"Model saved: {path} (champion={is_champion})")

    def _load_champion(self):
        path = MODELS_DIR / f"{self.symbol}_champion.json"
        meta_path = MODELS_DIR / f"{self.symbol}_champion_meta.json"
        if path.exists() and meta_path.exists():
            self.model = XGBClassifier()
            self.model.load_model(str(path))
            meta = json.loads(meta_path.read_text())
            self.feature_names = meta["feature_names"]
            logger.info(f"Champion model loaded for {self.symbol}")
