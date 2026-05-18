"""
classification/lgbm_classifier.py
-----------------------------------
LightGBM-based region classifier.

Responsibilities:
  - Training on labelled (features, labels) datasets
  - Saving / loading the fitted model
  - Inference with confidence filtering
  - Basic cross-validation reporting
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLASSIFIER, ClassifierConfig,
    REGION_LABELS,
    LGBM_MODEL_PATH, LABEL_ENCODER_PATH,
)


class LayoutClassifier:
    """
    Wraps LightGBM for document region classification.

    Usage — Training
    ----------------
    clf = LayoutClassifier()
    clf.train(X_train, y_train)   # y_train = string labels
    clf.save()

    Usage — Inference
    -----------------
    clf = LayoutClassifier.load()
    labels, confs = clf.predict(X)  # X: (N, D) feature matrix
    """

    def __init__(self, cfg: ClassifierConfig = CLASSIFIER):
        self.cfg  = cfg
        self.model: Optional[lgb.LGBMClassifier] = None
        self.le:    Optional[LabelEncoder]        = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: List[str],
        eval_split: bool = True,
    ) -> "LayoutClassifier":
        """
        Fit the LightGBM classifier.

        Parameters
        ----------
        X          : (N, D) float32 feature matrix
        y          : list of string labels (e.g. "text", "figure")
        eval_split : if True, print a 5-fold CV accuracy estimate
        """
        logger.info(f"Training on {len(y):,} samples  |  {X.shape[1]} features")

        # Encode string labels → integers
        self.le = LabelEncoder()
        y_enc   = self.le.fit_transform(y)

        self.model = lgb.LGBMClassifier(
            n_estimators    = self.cfg.n_estimators,
            num_leaves      = self.cfg.num_leaves,
            learning_rate   = self.cfg.learning_rate,
            max_depth       = self.cfg.max_depth,
            min_child_samples = self.cfg.min_child_samples,
            subsample       = self.cfg.subsample,
            colsample_bytree= self.cfg.colsample_bytree,
            n_jobs          = self.cfg.n_jobs,
            random_state    = self.cfg.random_state,
            class_weight    = self.cfg.class_weight,
            verbose         = -1,
        )

        if eval_split and len(np.unique(y_enc)) > 1:
            kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                self.model, X, y_enc, cv=kf, scoring="accuracy", n_jobs=-1
            )
            logger.info(
                f"5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}"
            )

        self.model.fit(X, y_enc)

        # Full-set report (informational)
        y_pred = self.model.predict(X)
        report = classification_report(
            y_enc, y_pred, target_names=self.le.classes_, zero_division=0
        )
        logger.info(f"Training classification report:\n{report}")

        return self

    def train_incremental(
        self,
        X: np.ndarray,
        y: List[str],
        init_model: Optional[str] = None,
    ) -> "LayoutClassifier":
        """
        Continue training from an existing model checkpoint.
        Useful for large corpora processed in chunks.
        """
        if self.le is None:
            self.le = LabelEncoder().fit(REGION_LABELS)

        y_enc = self.le.transform(y)

        if self.model is None:
            return self.train(X, y)

        # LightGBM native incremental: refit with init_model
        dataset = lgb.Dataset(X, label=y_enc)
        params  = self.model.get_params()
        params.pop("n_estimators", None)

        booster = lgb.train(
            params={**params, "num_leaves": self.cfg.num_leaves,
                    "learning_rate": self.cfg.learning_rate},
            train_set=dataset,
            num_boost_round=100,
            init_model=self.model.booster_ if self.model else None,
        )
        self.model.booster_ = booster
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Predict labels and confidence scores.

        Returns
        -------
        labels : list of string label names (length N)
        confs  : float32 array of max-class probabilities (length N)
        """
        self._check_fitted()

        proba  = self.model.predict_proba(X)            # (N, C)
        ids    = np.argmax(proba, axis=1)
        confs  = proba[np.arange(len(ids)), ids].astype(np.float32)
        labels = self.le.inverse_transform(ids).tolist()
        return labels, confs

    def predict_filtered(
        self,
        X: np.ndarray,
        boxes: list,
    ) -> Tuple[List[str], np.ndarray, list]:
        """
        Predict and drop regions below the confidence threshold.

        Returns
        -------
        labels, confs, kept_boxes  — all filtered to high-confidence only
        """
        labels, confs = self.predict(X)
        mask   = confs >= self.cfg.confidence_threshold
        labels = [l for l, m in zip(labels, mask) if m]
        confs  = confs[mask]
        kept   = [b for b, m in zip(boxes, mask) if m]
        logger.debug(
            f"Kept {len(kept)}/{len(boxes)} regions "
            f"above conf={self.cfg.confidence_threshold}"
        )
        return labels, confs, kept

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(
        self,
        model_path: Path = LGBM_MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
    ) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(encoder_path, "wb") as f:
            pickle.dump(self.le, f)
        logger.info(f"Model saved → {model_path}")
        logger.info(f"Encoder saved → {encoder_path}")

    @classmethod
    def load(
        cls,
        model_path: Path = LGBM_MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
        cfg: ClassifierConfig = CLASSIFIER,
    ) -> "LayoutClassifier":
        if not model_path.exists():
            raise FileNotFoundError(f"No model at {model_path}. Train first.")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        obj       = cls(cfg)
        obj.model = model
        obj.le    = le
        logger.info(f"Model loaded from {model_path}")
        return obj

    def is_fitted(self) -> bool:
        return self.model is not None and self.le is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError("Model not fitted. Call .train() or .load() first.")

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, top_k: int = 20) -> np.ndarray:
        """Return indices of top_k most important features."""
        self._check_fitted()
        imp = self.model.feature_importances_
        return np.argsort(-imp)[:top_k]


# ─── Synthetic data demo ──────────────────────────────────────────────────────

def _make_synthetic_data(n: int = 2000, d: int = 400) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = rng.choice(REGION_LABELS, size=n).tolist()
    return X, y


if __name__ == "__main__":
    X, y = _make_synthetic_data()
    clf  = LayoutClassifier()
    clf.train(X, y)

    labels, confs = clf.predict(X[:10])
    for lbl, conf in zip(labels, confs):
        print(f"  {lbl:12s}  conf={conf:.3f}")
