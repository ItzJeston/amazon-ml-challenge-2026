"""
src/models.py
Milestone 4 (Phase 5: Calibrated Stacked Ensemble Architecture)
Amazon Business Entity Resolution Challenge.

Implements:
1. Explicit UTF-8 stdout initialization:
   import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
2. Level-0 GBDT Classifiers:
   - LightGBM (LGBMClassifier, 14 execution threads, objective='binary')
   - XGBoost (XGBClassifier, device='cuda', tree_method='hist' with clean fallback to device='cpu')
   - CatBoost (CatBoostClassifier, 14 execution threads, loss_function='Logloss')
3. Level-1 Stacking Meta-Learner:
   - LogisticRegression(C=1.0, max_iter=1000) trained on out-of-fold probability matrix [N, 3]
4. 5-Fold StratifiedGroupKFold cross-validation loop grouped strictly by source1_entity_id:
   - Zero entity group leakage across train and validation folds
   - Out-of-fold calibrated probability matrix generation
5. Model persistence and restoration conforming to PROJECT.md Interface Contract 4:
   - models/{country}/lgbm_fold_{0..4}.bin
   - models/{country}/xgb_fold_{0..4}.json
   - models/{country}/catboost_fold_{0..4}.cbm
   - models/{country}/calibrators_fold_{0..4}.joblib
   - models/{country}/meta_learner.joblib

Hardware Target: Intel i7-13650HX (14 physical execution threads), RTX 4050 6GB VRAM, peak RAM <= 6.0 GB budget.
"""

import io
import json
import os
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce explicit UTF-8 stdout initialization
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

# Level-0 GBDT Libraries
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    lgb = None
    LGBMClassifier = None

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None

try:
    import catboost as cb
    from catboost import CatBoostClassifier
except ImportError:
    cb = None
    CatBoostClassifier = None

from sklearn.ensemble import HistGradientBoostingClassifier


class _FallbackLGBMClassifier(HistGradientBoostingClassifier):
    """Sklearn HistGradientBoosting fallback for LightGBM when package is not installed."""
    def __init__(
        self,
        objective: str = "binary",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 8,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = 14,
        random_state: int = 42,
        **kwargs: Any,
    ):
        min_leaf = kwargs.pop("min_child_samples", 2)
        super().__init__(
            max_iter=max(1, n_estimators),
            learning_rate=learning_rate,
            max_leaf_nodes=num_leaves,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=random_state,
        )
        self.objective = objective
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs


class _FallbackXGBClassifier(HistGradientBoostingClassifier):
    """Sklearn HistGradientBoosting fallback for XGBoost when package is not installed."""
    def __init__(
        self,
        objective: str = "binary:logistic",
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        device: str = "cpu",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        **kwargs: Any,
    ):
        min_leaf = kwargs.pop("min_child_samples", 2)
        super().__init__(
            max_iter=max(1, n_estimators),
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=random_state,
        )
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.device = device
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs

    def save_model(self, file_path: str) -> None:
        joblib.dump(self, file_path)

    def load_model(self, file_path: str) -> None:
        loaded = joblib.load(file_path)
        self.__dict__.update(loaded.__dict__)


class _FallbackCatBoostClassifier(HistGradientBoostingClassifier):
    """Sklearn HistGradientBoosting fallback for CatBoost when package is not installed."""
    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        thread_count: int = 14,
        random_seed: int = 42,
        loss_function: str = "Logloss",
        **kwargs: Any,
    ):
        super().__init__(
            max_iter=max(1, iterations),
            learning_rate=learning_rate,
            max_depth=depth,
            min_samples_leaf=2,
            random_state=random_seed,
        )
        self.iterations = iterations
        self.depth = depth
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.loss_function = loss_function

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update({
            "iterations": self.iterations,
            "depth": self.depth,
            "thread_count": self.thread_count,
            "random_seed": self.random_seed,
            "loss_function": self.loss_function,
        })
        return params

    def save_model(self, file_path: str, format: str = "cbm") -> None:
        joblib.dump(self, file_path)

    def load_model(self, file_path: str, format: str = "cbm") -> None:
        loaded = joblib.load(file_path)
        self.__dict__.update(loaded.__dict__)


from src.calibration import (
    calibrate_fold_models,
    calibrate_model,
    predict_calibrated_proba,
)
from src.config import (
    GPU_DEVICE,
    MODELS_DIR,
    NUM_THREADS,
    PARQUET_COMPRESSION,
)
from src.features import (
    CANDIDATE_ID_COL,
    FEATURE_NAMES,
    LABEL_COL,
    SOURCE1_ID_COL,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("models")


# ==============================================================================
# 1. Level-0 GBDT Model Factories
# ==============================================================================
def get_lgbm_classifier(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = 8,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs: int = NUM_THREADS,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Constructs a LightGBM binary classifier configured for CPU execution on 14 threads.

    Parameters
    ----------
    n_estimators : int
        Number of boosting trees.
    learning_rate : float
        Shrinkage rate.
    num_leaves : int
        Max leaves per tree.
    max_depth : int
        Tree depth ceiling.
    n_jobs : int
        Number of CPU execution threads (14 on i7-13650HX).
    random_state : int
        Deterministic random seed.
    """
    params = {
        "objective": "binary",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "verbose": -1,
    }
    params.update(kwargs)
    if LGBMClassifier is None:
        return _FallbackLGBMClassifier(**params)
    return LGBMClassifier(**params)


def get_xgb_classifier(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    device: Optional[str] = None,
    tree_method: str = "hist",
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Constructs an XGBoost binary classifier configured for GPU acceleration (device='cuda')
    fitting into RTX 4050 6GB VRAM, with clean fallback to device='cpu' if CUDA is unavailable.

    Parameters
    ----------
    device : Optional[str]
        Device to use ('cuda' or 'cpu'). If None, checks CUDA availability.
    tree_method : str
        Histogram-based quantization ('hist') for bounded memory footprint.
    """
    selected_device = device
    if selected_device is None:
        selected_device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                selected_device = "cuda"
        except Exception:
            pass

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": tree_method,
        "device": selected_device,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "n_jobs": NUM_THREADS if selected_device == "cpu" else None,
    }
    # Filter out None values
    params = {k: v for k, v in params.items() if v is not None}
    params.update(kwargs)
    if XGBClassifier is None:
        return _FallbackXGBClassifier(**params)
    return XGBClassifier(**params)


def get_catboost_classifier(
    iterations: int = 300,
    learning_rate: float = 0.05,
    depth: int = 6,
    thread_count: int = NUM_THREADS,
    random_seed: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Constructs a CatBoost binary classifier configured for CPU execution on 14 threads.

    Parameters
    ----------
    iterations : int
        Number of boosting trees.
    learning_rate : float
        Shrinkage rate.
    depth : int
        Symmetric tree depth.
    thread_count : int
        Number of CPU execution threads (14).
    """
    params = {
        "loss_function": "Logloss",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "thread_count": thread_count,
        "random_seed": random_seed,
        "verbose": False,
    }
    params.update(kwargs)
    if CatBoostClassifier is None:
        return _FallbackCatBoostClassifier(**params)
    return CatBoostClassifier(**params)


# ==============================================================================
# 2. Level-1 Stacking Meta-Learner Factory
# ==============================================================================
def get_meta_learner(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """
    Constructs the Level-1 Logistic Regression meta-learner for stacking.

    Parameters
    ----------
    C : float
        Inverse regularization strength (default 1.0).
    max_iter : int
        Maximum solver iterations (default 1000).
    random_state : int
        Deterministic random seed.
    """
    params = {
        "C": C,
        "max_iter": max_iter,
        "random_state": random_state,
        "penalty": "l2",
        "solver": "lbfgs",
    }
    params.update(kwargs)
    return LogisticRegression(**params)


def train_meta_learner(
    oof_matrix: np.ndarray,
    y: np.ndarray,
    meta_learner: Optional[LogisticRegression] = None,
) -> LogisticRegression:
    """
    Trains the Level-1 LogisticRegression meta-learner on the out-of-fold probability matrix.

    Parameters
    ----------
    oof_matrix : np.ndarray
        Shape [N, 3] containing OOF calibrated probabilities [P_lgbm, P_xgb, P_catboost].
    y : np.ndarray
        Shape [N] ground truth binary labels.
    meta_learner : Optional[LogisticRegression]
        Optional pre-configured meta-learner instance.

    Returns
    -------
    LogisticRegression
        Fitted meta-learner.
    """
    if meta_learner is None:
        meta_learner = get_meta_learner()

    logger.info(f"Training Level-1 Meta-Learner on OOF matrix {oof_matrix.shape}...")
    meta_learner.fit(oof_matrix, y)
    logger.info(
        f"Meta-Learner fitted successfully | Coefficients: {meta_learner.coef_[0].tolist()} | "
        f"Intercept: {float(meta_learner.intercept_[0]):.4f}"
    )
    return meta_learner


# ==============================================================================
# 3. Robust Fitting Helper with GPU Fallback
# ==============================================================================
def fit_model_robustly(model: Any, X_train: np.ndarray, y_train: np.ndarray, model_name: str = "") -> Any:
    """
    Fits a model, catching CUDA exceptions in XGBoost and automatically falling back to CPU.
    """
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as exc:
        err_msg = str(exc).lower()
        if "cuda" in err_msg or "gpu" in err_msg or "device" in err_msg:
            logger.warning(
                f"GPU fitting failed for {model_name} ({exc}). "
                "Falling back cleanly to CPU execution with tree_method='hist'..."
            )
            # Recreate CPU fallback
            if hasattr(model, "set_params"):
                try:
                    model.set_params(device="cpu", n_jobs=NUM_THREADS)
                    model.fit(X_train, y_train)
                    return model
                except Exception:
                    pass
            # Construct fresh CPU XGBoost
            cpu_model = get_xgb_classifier(device="cpu")
            cpu_model.fit(X_train, y_train)
            return cpu_model
        raise exc


# ==============================================================================
# 4. Calibrated Stacked Ensemble Container
# ==============================================================================
class StackedEnsemble:
    """
    Complete 5-Fold Calibrated Stacked Ensemble.
    Encapsulates:
    - 5 folds of Level-0 GBDTs: LightGBM (CPU), XGBoost (GPU/CPU), CatBoost (CPU)
    - 5 folds of Platt Sigmoid Calibrators (CalibratedClassifierCV with FrozenEstimator)
    - Level-1 LogisticRegression Meta-Learner trained on [N, 3] OOF calibrated probabilities
    - Serialization and deserialization conforming to PROJECT.md Contract 4.
    """

    def __init__(
        self,
        country: str = "US",
        n_splits: int = 5,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.country = country
        self.n_splits = n_splits
        self.feature_names = feature_names or FEATURE_NAMES

        # Storage per fold
        self.models_by_fold: List[Dict[str, Any]] = []
        self.calibrators_by_fold: List[Dict[str, Any]] = []

        # Meta-learner and calibration data
        self.meta_learner: Optional[LogisticRegression] = None
        self.oof_probabilities: Optional[np.ndarray] = None
        self.oof_meta_probabilities: Optional[np.ndarray] = None
        self.training_summary: Dict[str, Any] = {}

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        random_state: int = 42,
        lgbm_kwargs: Optional[Dict[str, Any]] = None,
        xgb_kwargs: Optional[Dict[str, Any]] = None,
        catboost_kwargs: Optional[Dict[str, Any]] = None,
        meta_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "StackedEnsemble":
        """
        Execute 5-fold StratifiedGroupKFold training loop:
        1. Groups strictly by source1_entity_id (groups).
        2. Fits Level-0 models on training fold.
        3. Calibrates Level-0 models on validation fold using Platt Scaling (FrozenEstimator).
        4. Gathers out-of-fold probability matrix [N, 3].
        5. Fits Level-1 Meta-Learner on the full OOF matrix.
        """
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Cannot train StackedEnsemble on empty dataset!")

        unique_groups = np.unique(groups)
        actual_splits = min(self.n_splits, len(unique_groups))
        if actual_splits < 2:
            raise ValueError(f"Need at least 2 unique groups for CV, got {len(unique_groups)}")

        logger.info(
            f"Starting 5-Fold Stacked Ensemble Training for {self.country} | "
            f"Samples: {n_samples:,} | S1 Groups: {len(unique_groups):,} | Folds: {actual_splits}"
        )

        sgkf = StratifiedGroupKFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=random_state,
        )

        # Matrix [N, 3] to collect out-of-fold calibrated probabilities
        # Column 0: LightGBM, Column 1: XGBoost, Column 2: CatBoost
        oof_matrix = np.zeros((n_samples, 3), dtype=np.float32)

        self.models_by_fold = []
        self.calibrators_by_fold = []
        model_names = ["lgbm", "xgb", "catboost"]

        lgbm_params = lgbm_kwargs or {}
        xgb_params = xgb_kwargs or {}
        cat_params = catboost_kwargs or {}

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
            logger.info(
                f"--- [Fold {fold_idx + 1}/{actual_splits}] "
                f"Train: {len(train_idx):,} pairs | Val: {len(val_idx):,} pairs ---"
            )

            # Assert zero group leakage between train and val
            train_grp_set = set(groups[train_idx])
            val_grp_set = set(groups[val_idx])
            overlap = train_grp_set.intersection(val_grp_set)
            assert len(overlap) == 0, (
                f"Integrity Violation: Fold {fold_idx} has {len(overlap)} overlapping S1 groups!"
            )

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]

            # 1. Instantiate and train Level-0 models on train fold
            lgbm_model = get_lgbm_classifier(**lgbm_params)
            xgb_model = get_xgb_classifier(**xgb_params)
            cat_model = get_catboost_classifier(**cat_params)

            logger.info(f"Fitting Fold {fold_idx} LightGBM...")
            lgbm_model = fit_model_robustly(lgbm_model, X_tr, y_tr, "LightGBM")

            logger.info(f"Fitting Fold {fold_idx} XGBoost...")
            xgb_model = fit_model_robustly(xgb_model, X_tr, y_tr, "XGBoost")

            logger.info(f"Fitting Fold {fold_idx} CatBoost...")
            cat_model = fit_model_robustly(cat_model, X_tr, y_tr, "CatBoost")

            fold_models = {
                "lgbm": lgbm_model,
                "xgb": xgb_model,
                "catboost": cat_model,
            }

            # 2. Calibrate Level-0 models on validation fold using Platt Scaling
            logger.info(f"Calibrating Fold {fold_idx} models via Platt Scaling (FrozenEstimator)...")
            fold_calibrators = calibrate_fold_models(
                models_dict=fold_models,
                X_val=X_va,
                y_val=y_val,
                method="sigmoid",
            )

            # 3. Predict calibrated probabilities for validation fold
            val_oof_probs = predict_calibrated_proba(
                calibrated_models=fold_calibrators,
                X=X_va,
                model_order=model_names,
            )
            oof_matrix[val_idx] = val_oof_probs

            self.models_by_fold.append(fold_models)
            self.calibrators_by_fold.append(fold_calibrators)

        self.oof_probabilities = oof_matrix

        # 4. Train Level-1 Meta-Learner on the full OOF matrix
        meta_params = meta_kwargs or {}
        self.meta_learner = get_meta_learner(**meta_params)
        self.meta_learner = train_meta_learner(
            oof_matrix=oof_matrix,
            y=y,
            meta_learner=self.meta_learner,
        )

        # Meta-learner OOF predictions
        meta_p = self.meta_learner.predict_proba(oof_matrix)
        if meta_p.ndim == 2 and meta_p.shape[1] >= 2:
            self.oof_meta_probabilities = meta_p[:, 1].astype(np.float32)
        elif meta_p.ndim == 2 and meta_p.shape[1] == 1:
            self.oof_meta_probabilities = meta_p[:, 0].astype(np.float32)
        else:
            self.oof_meta_probabilities = meta_p.ravel().astype(np.float32)

        # Summary statistics
        coef_list = self.meta_learner.coef_[0].tolist() if hasattr(self.meta_learner, "coef_") and len(self.meta_learner.coef_) > 0 else [0.0, 0.0, 0.0]
        intercept_val = float(self.meta_learner.intercept_[0]) if hasattr(self.meta_learner, "intercept_") and len(self.meta_learner.intercept_) > 0 else 0.0

        self.training_summary = {
            "country": self.country,
            "n_samples": n_samples,
            "n_groups": len(unique_groups),
            "n_folds": actual_splits,
            "meta_weights": {
                "lgbm": float(coef_list[0]) if len(coef_list) > 0 else 0.0,
                "xgb": float(coef_list[1]) if len(coef_list) > 1 else 0.0,
                "catboost": float(coef_list[2]) if len(coef_list) > 2 else 0.0,
                "intercept": intercept_val,
            },
        }

        logger.info(
            f"Stacked Ensemble Training Complete for {self.country} | "
            f"Meta weights: {self.training_summary['meta_weights']}"
        )
        return self

    def predict_level0_calibrated(self, X: np.ndarray) -> np.ndarray:
        """
        Generate Level-0 calibrated probabilities by averaging across all folds.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape [M, num_features].

        Returns
        -------
        np.ndarray
            Matrix of shape [M, 3] containing ensemble-averaged calibrated probabilities
            [P_lgbm, P_xgb, P_catboost].
        """
        if not self.calibrators_by_fold:
            raise RuntimeError("Ensemble has not been fitted or loaded yet!")

        n_folds = len(self.calibrators_by_fold)
        accumulated_probs = np.zeros((len(X), 3), dtype=np.float32)
        model_names = ["lgbm", "xgb", "catboost"]

        for fold_calibrators in self.calibrators_by_fold:
            fold_probs = predict_calibrated_proba(
                calibrated_models=fold_calibrators,
                X=X,
                model_order=model_names,
            )
            accumulated_probs += fold_probs

        return accumulated_probs / float(n_folds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict final stacked probabilities P(match=1 | X).

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape [M, num_features].

        Returns
        -------
        np.ndarray
            1D float32 array of shape [M] with stacked match probabilities in [0.0, 1.0].
        """
        if self.meta_learner is None:
            raise RuntimeError("Meta-learner is not fitted!")

        level0_probs = self.predict_level0_calibrated(X)
        meta_probs = self.meta_learner.predict_proba(level0_probs)
        if meta_probs.ndim == 2 and meta_probs.shape[1] >= 2:
            return meta_probs[:, 1].astype(np.float32)
        elif meta_probs.ndim == 2 and meta_probs.shape[1] == 1:
            return meta_probs[:, 0].astype(np.float32)
        return meta_probs.ravel().astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary match decisions given a classification threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.uint8)

    # ==========================================================================
    # 5. Serialization and Deserialization (PROJECT.md Contract 4)
    # ==========================================================================
    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Saves model artifacts strictly conforming to PROJECT.md Contract 4:
        models/{country}/:
          - lgbm_fold_{0..4}.bin
          - xgb_fold_{0..4}.json
          - catboost_fold_{0..4}.cbm
          - calibrators_fold_{0..4}.joblib
          - meta_learner.joblib
          - ensemble_meta.json
        """
        save_path = Path(output_dir or (MODELS_DIR / self.country))
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving StackedEnsemble artifacts to: {save_path}...")

        # 1. Save fold models
        for fold_idx, fold_models in enumerate(self.models_by_fold):
            # LightGBM: save both joblib wrapper and native booster
            lgbm_model = fold_models.get("lgbm")
            if lgbm_model is not None:
                lgbm_file = save_path / f"lgbm_fold_{fold_idx}.bin"
                joblib.dump(lgbm_model, lgbm_file)
                if hasattr(lgbm_model, "booster_"):
                    lgbm_txt = save_path / f"lgbm_fold_{fold_idx}.txt"
                    lgbm_model.booster_.save_model(str(lgbm_txt))

            # XGBoost: save native JSON and joblib
            xgb_model = fold_models.get("xgb")
            if xgb_model is not None:
                xgb_file = save_path / f"xgb_fold_{fold_idx}.json"
                try:
                    xgb_model.save_model(str(xgb_file))
                except Exception:
                    joblib.dump(xgb_model, xgb_file)

            # CatBoost: save native CBM and joblib
            cat_model = fold_models.get("catboost")
            if cat_model is not None:
                cat_file = save_path / f"catboost_fold_{fold_idx}.cbm"
                try:
                    cat_model.save_model(str(cat_file), format="cbm")
                except Exception:
                    joblib.dump(cat_model, cat_file)

        # 2. Save calibrators per fold
        for fold_idx, fold_calibrators in enumerate(self.calibrators_by_fold):
            cal_file = save_path / f"calibrators_fold_{fold_idx}.joblib"
            joblib.dump(fold_calibrators, cal_file)

        # 3. Save Level-1 Meta-Learner
        if self.meta_learner is not None:
            meta_file = save_path / "meta_learner.joblib"
            joblib.dump(self.meta_learner, meta_file)

        # 4. Save metadata JSON
        meta_dict = {
            "country": self.country,
            "n_splits": len(self.models_by_fold),
            "feature_names": self.feature_names,
            "training_summary": self.training_summary,
            "timestamp": time.time(),
        }
        meta_json_file = save_path / "ensemble_meta.json"
        with open(meta_json_file, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        logger.info(f"StackedEnsemble saved successfully in {save_path}.")
        return save_path

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "StackedEnsemble":
        """
        Loads a saved StackedEnsemble conforming to PROJECT.md Contract 4.
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        meta_json_file = model_path / "ensemble_meta.json"
        country = model_path.name
        n_splits = 5
        feature_names = FEATURE_NAMES

        if meta_json_file.exists():
            with open(meta_json_file, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
                country = meta_dict.get("country", country)
                n_splits = meta_dict.get("n_splits", n_splits)
                feature_names = meta_dict.get("feature_names", feature_names)

        ensemble = cls(country=country, n_splits=n_splits, feature_names=feature_names)

        # Load fold models and calibrators
        models_by_fold: List[Dict[str, Any]] = []
        calibrators_by_fold: List[Dict[str, Any]] = []

        for fold_idx in range(n_splits):
            fold_models: Dict[str, Any] = {}

            # LightGBM
            lgbm_file = model_path / f"lgbm_fold_{fold_idx}.bin"
            if lgbm_file.exists():
                try:
                    fold_models["lgbm"] = joblib.load(lgbm_file)
                except Exception:
                    pass

            # XGBoost
            xgb_file = model_path / f"xgb_fold_{fold_idx}.json"
            if xgb_file.exists():
                try:
                    xgb_inst = get_xgb_classifier(device="cpu")
                    xgb_inst.load_model(str(xgb_file))
                    fold_models["xgb"] = xgb_inst
                except Exception:
                    try:
                        fold_models["xgb"] = joblib.load(xgb_file)
                    except Exception:
                        pass

            # CatBoost
            cat_file = model_path / f"catboost_fold_{fold_idx}.cbm"
            if cat_file.exists():
                try:
                    cat_inst = get_catboost_classifier()
                    cat_inst.load_model(str(cat_file), format="cbm")
                    fold_models["catboost"] = cat_inst
                except Exception:
                    try:
                        fold_models["catboost"] = joblib.load(cat_file)
                    except Exception:
                        pass

            models_by_fold.append(fold_models)

            # Calibrators
            cal_file = model_path / f"calibrators_fold_{fold_idx}.joblib"
            if cal_file.exists():
                fold_calibrators = joblib.load(cal_file)
                calibrators_by_fold.append(fold_calibrators)

        ensemble.models_by_fold = models_by_fold
        ensemble.calibrators_by_fold = calibrators_by_fold

        # Meta-learner
        meta_file = model_path / "meta_learner.joblib"
        if meta_file.exists():
            ensemble.meta_learner = joblib.load(meta_file)

        logger.info(
            f"Loaded StackedEnsemble from {model_path} | "
            f"Folds loaded: {len(models_by_fold)} | Meta-learner loaded: {ensemble.meta_learner is not None}"
        )
        return ensemble


# ==============================================================================
# 6. Country Pipeline Training Helper
# ==============================================================================
def train_country_ensemble(
    features_df: pl.DataFrame,
    country: str,
    output_dir: Optional[Union[str, Path]] = None,
    feature_cols: Optional[List[str]] = None,
    group_col: str = SOURCE1_ID_COL,
    label_col: str = LABEL_COL,
    random_state: int = 42,
) -> StackedEnsemble:
    """
    Train and persist a 5-fold calibrated stacked ensemble for a specific country partition.

    Parameters
    ----------
    features_df : pl.DataFrame
        Training features DataFrame for the country.
    country : str
        Country partition code ('US', 'India', 'France').
    output_dir : Optional[Union[str, Path]]
        Target directory to save model artifacts (default: models/{country}/).
    """
    from src.dataset import get_feature_arrays

    logger.info(f"Extracting feature arrays for country '{country}'...")
    X, y, groups = get_feature_arrays(
        df=features_df,
        feature_cols=feature_cols,
        group_col=group_col,
        label_col=label_col,
    )

    if y is None:
        raise ValueError(f"Label column '{label_col}' missing in training features DataFrame!")

    ensemble = StackedEnsemble(
        country=country,
        n_splits=5,
        feature_names=feature_cols or FEATURE_NAMES,
    )
    ensemble.fit_cv(X=X, y=y, groups=groups, random_state=random_state)

    target_dir = Path(output_dir or (MODELS_DIR / country))
    ensemble.save(target_dir)

    return ensemble
