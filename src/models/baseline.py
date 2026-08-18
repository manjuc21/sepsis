"""Tabular baseline models: class-weighted Logistic Regression and XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config


class LogRegModel:
    """Class-weighted logistic regression with a scaler fit on train only."""

    def __init__(self, random_state: int = config.RANDOM_SEED):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogRegModel":
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]


class XGBoostModel:
    """Gradient-boosted trees with scale_pos_weight tuned to class imbalance."""

    def __init__(self, random_state: int = config.RANDOM_SEED):
        self.clf: xgb.XGBClassifier | None = None
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        scale_pos_weight = max(1.0, n_neg / max(1, n_pos))

        self.clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.clf is not None, "call fit() before predict_proba()"
        return self.clf.predict_proba(X)[:, 1]
