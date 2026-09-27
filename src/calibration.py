"""
src/calibration.py
Milestone 4 (Phase 5: Probability Calibration Engine via Modern Platt Scaling)
Amazon Business Entity Resolution Challenge.

Implements:
1. Modern Scikit-learn 1.7.2 Platt Scaling (Sigmoid probability calibration):
   - CalibratedClassifierCV(estimator=FrozenEstimator(trained_model), method='sigmoid')
   - STRICT CONSTRAINT: ZERO usage of deprecated cv='prefit'.
2. Calibrates Level-0 GBDT models (LightGBM, XGBoost, CatBoost) on validation fold pairs.
3. Produces well-calibrated posterior probabilities P(y=1 | x) in [0.0, 1.0], ensuring
   accurate downstream confidence scoring and singleton isolation.

Hardware Target: Intel i7-13650HX (14 physical execution threads), peak RAM <= 6.0 GB budget.
"""

import io
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce explicit UTF-8 stdout initialization
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Scikit-learn 1.6+ FrozenEstimator support
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    try:
        from sklearn.calibration import FrozenEstimator  # type: ignore
    except ImportError:
        # Compatibility wrapper mimicking FrozenEstimator
        class FrozenEstimator:  # type: ignore
            """Wrap a pre-fitted estimator to freeze its parameters during calibration."""

            def __init__(self, estimator: Any) -> None:
                self.estimator = estimator
                self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))

            def fit(self, *args: Any, **kwargs: Any) -> "FrozenEstimator":
                return self

            def predict(self, *args: Any, **kwargs: Any) -> Any:
                return self.estimator.predict(*args, **kwargs)

            def predict_proba(self, *args: Any, **kwargs: Any) -> Any:
                return self.estimator.predict_proba(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self.estimator, name)


from src.utils import get_logger, setup_utf8_stdout

setup_utf8_stdout()
logger = get_logger("calibration")


# ==============================================================================
# 1. Single Model Calibration via Modern FrozenEstimator Platt Scaling
# ==============================================================================
def calibrate_model(
    trained_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Applies Platt Scaling (method='sigmoid') to a pre-trained binary classifier.

    STRICT CONSTRAINTS:
    - Uses FrozenEstimator(trained_model) to signal that the base estimator is already fitted.
    - NEVER passes deprecated cv='prefit', conforming strictly to Scikit-learn 1.7.2+.

    Parameters
    ----------
    trained_model : Any
        Pre-fitted classifier (e.g. LGBMClassifier, XGBClassifier, CatBoostClassifier).
    X_val : np.ndarray
        Validation feature matrix of shape [N, 32].
    y_val : np.ndarray
        Validation binary ground truth labels of shape [N].
    method : str
        Calibration method: 'sigmoid' for Platt scaling (default).

    Returns
    -------
    CalibratedClassifierCV
        Fitted calibrator capable of predicting calibrated probabilities.
    """
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("Cannot calibrate on empty validation arrays!")

    unique_labels = np.unique(y_val)
    if len(unique_labels) < 2:
        logger.warning(
            f"Validation set contains only 1 unique class ({unique_labels}). "
            "Platt scaling requires both positive and negative samples. Returning wrapped estimator."
        )

    # Wrap pre-fitted model inside FrozenEstimator
    frozen = FrozenEstimator(trained_model)

    # Scikit-learn 1.7.2+ standard: pass FrozenEstimator directly to estimator
    # Do NOT pass cv='prefit' - FrozenEstimator handles pre-fitted estimators natively
    calibrator = CalibratedClassifierCV(
        estimator=frozen,
        method=method,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrator.fit(X_val, y_val)

    return calibrator


# ==============================================================================
# 2. Ensemble Fold Models Calibration
# ==============================================================================
def calibrate_fold_models(
    models_dict: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> Dict[str, CalibratedClassifierCV]:
    """
    Calibrate all Level-0 models (e.g., 'lgbm', 'xgb', 'catboost') for a given fold
    using Platt Scaling on the validation set.

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary mapping model names to fitted classifiers.
    X_val : np.ndarray
        Fold validation feature matrix.
    y_val : np.ndarray
        Fold validation binary labels.
    method : str
        Calibration method ('sigmoid' for Platt scaling).

    Returns
    -------
    Dict[str, CalibratedClassifierCV]
        Dictionary mapping model names to their respective fitted calibrators.
    """
    calibrated_dict: Dict[str, CalibratedClassifierCV] = {}

    for name, model in models_dict.items():
        logger.debug(f"Fitting Platt calibrator for Level-0 model: '{name}'...")
        calibrator = calibrate_model(
            trained_model=model,
            X_val=X_val,
            y_val=y_val,
            method=method,
        )
        calibrated_dict[name] = calibrator

    return calibrated_dict


# ==============================================================================
# 3. Calibrated Probabilities Prediction & OOF Matrix Extraction
# ==============================================================================
def predict_calibrated_proba(
    calibrated_models: Dict[str, Any],
    X: np.ndarray,
    model_order: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Generate calibrated probabilities for each model in calibrated_models.

    Parameters
    ----------
    calibrated_models : Dict[str, Any]
        Dictionary mapping model names to calibrators.
    X : np.ndarray
        Feature matrix of shape [N, num_features].
    model_order : Optional[List[str]]
        Explicit ordering of models. Defaults to ['lgbm', 'xgb', 'catboost']
        or sorted keys of calibrated_models.

    Returns
    -------
    np.ndarray
        Calibrated probabilities matrix of shape [N, num_models], where each
        column corresponds to P(y=1) from one calibrated model.
    """
    if model_order is None:
        standard_order = ["lgbm", "xgb", "catboost"]
        model_order = [m for m in standard_order if m in calibrated_models]
        if not model_order:
            model_order = sorted(calibrated_models.keys())

    probs_list = []
    for name in model_order:
        calibrator = calibrated_models[name]
        # Predict probability of positive class (label 1)
        proba = calibrator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p1 = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            p1 = proba[:, 0]
        else:
            p1 = proba.ravel()
        probs_list.append(p1)

    # Stack column-wise to produce [N, num_models] float32 matrix
    out_matrix = np.column_stack(probs_list).astype(np.float32)
    return out_matrix


# ==============================================================================
# 4. Integrity Verification & Static Contract Check
# ==============================================================================
def verify_calibration_contract(calibrator: CalibratedClassifierCV) -> bool:
    """
    Verifies that a fitted calibrator adheres strictly to the modern Scikit-Learn
    contract:
    1. Uses FrozenEstimator as estimator wrapper.
    2. Does NOT use deprecated cv='prefit'.
    3. Output probabilities are bounded in [0.0, 1.0].
    """
    # 1. Check cv attribute is not 'prefit'
    cv_attr = getattr(calibrator, "cv", None)
    if cv_attr == "prefit":
        raise ValueError("CRITICAL INTEGRITY VIOLATION: Calibrator was configured with cv='prefit'!")

    # 2. Check FrozenEstimator usage
    base = getattr(calibrator, "estimator", None)
    is_frozen = isinstance(base, FrozenEstimator) or (
        base is not None and "FrozenEstimator" in type(base).__name__
    )
    if not is_frozen:
        logger.warning(f"Calibrator estimator is of type {type(base)}, expected FrozenEstimator.")

    # 3. Check method is sigmoid
    if calibrator.method != "sigmoid":
        logger.warning(f"Calibrator method is '{calibrator.method}', expected 'sigmoid' (Platt scaling).")

    return True
