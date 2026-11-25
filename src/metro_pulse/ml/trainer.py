"""LightGBM model training and XAI feature importance extraction."""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from metro_pulse.core.config import settings
from metro_pulse.ml.loader import MetroDataLoader

FEATURE_COLS = [
    "station_id",
    "line_code",
    "time_idx",
    "base_pax",
    "scenario_code",
    "weather_coeff",
    "event_coeff",
]
TARGET_COL = "predicted_pax"


class LightGBMTrainer:
    def __init__(self, artifact_path: Path = settings.model_artifact_path):
        self.artifact_path = artifact_path
        self.model: LGBMRegressor = None
        self.feature_importance_pct: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}

    def train(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """LightGBM 모델 학습 및 평가, 중요도 추출 후 저장."""
        if df is None:
            loader = MetroDataLoader.get_instance()
            df = loader.generate_training_dataset(num_samples_per_station=12)

        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        # 모델 정의 및 학습
        self.model = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self.model.fit(X_train, y_train)

        # 성능 검증
        preds = self.model.predict(X_test)
        r2 = float(r2_score(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))

        # XAI: Feature Importance 추출 및 백분율 환산
        raw_importances = self.model.feature_importances_
        total_importance = sum(raw_importances) if sum(raw_importances) > 0 else 1.0

        importance_dict = {}
        for col, imp in zip(FEATURE_COLS, raw_importances):
            pct = round((imp / total_importance) * 100.0, 1)
            importance_dict[col] = pct

        self.feature_importance_pct = importance_dict
        self.metrics = {"r2_score": round(r2, 4), "mae": round(mae, 2)}

        # 저장
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_importance_pct": self.feature_importance_pct,
            "metrics": self.metrics,
            "feature_cols": FEATURE_COLS,
        }
        joblib.dump(payload, self.artifact_path)

        return payload

    def load_or_train(self) -> Tuple[LGBMRegressor, Dict[str, float]]:
        """이미 저장된 아티팩트가 있으면 로드, 없으면 새로 학습."""
        if self.artifact_path.exists():
            try:
                data = joblib.load(self.artifact_path)
                self.model = data["model"]
                self.feature_importance_pct = data.get("feature_importance_pct", {})
                self.metrics = data.get("metrics", {})
                return self.model, self.feature_importance_pct
            except Exception as e:
                print(f">> [Warning] 기존 모델 아티팩트 로드 실패 ({e}), 재학습을 진행합니다.")

        res = self.train()
        return res["model"], res["feature_importance_pct"]


def get_trained_model():
    trainer = LightGBMTrainer()
    return trainer.load_or_train()
