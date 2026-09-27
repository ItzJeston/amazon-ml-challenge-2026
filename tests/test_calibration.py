"""
tests/test_calibration.py
Unit tests for src/calibration.py (Milestone 4 Platt Scaling Probability Calibration).

Validates:
1. Modern Scikit-learn 1.7.2 Platt Scaling via FrozenEstimator and sigmoid method.
2. STRICT CONSTRAINT: Zero usage of deprecated cv='prefit'.
3. Probability predictions are strictly bounded in [0.0, 1.0].
4. Calibration of Level-0 fold models (LightGBM, XGBoost, CatBoost).
5. Generation of out-of-fold calibrated probability matrix [N, 3].
6. Static integrity checks conforming to PROJECT.md Contract 4.
"""

import io
from pathlib import Path
import sys
import unittest
import warnings

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce explicit UTF-8 stdout initialization
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.calibration import (
    calibrate_fold_models,
    calibrate_model,
    predict_calibrated_proba,
    verify_calibration_contract,
)


class TestCalibrationEngine(unittest.TestCase):
    """Test suite for Platt Scaling probability calibration."""

    def setUp(self):
        np.random.seed(42)
        # Generate synthetic binary classification dataset
        self.n_samples = 150
        self.n_features = 32
        self.X_val = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y_val = (self.X_val[:, 0] + self.X_val[:, 1] > 0).astype(np.uint8)

        # Base mock model
        self.base_model = LogisticRegression(solver="lbfgs", random_state=42)
        self.base_model.fit(self.X_val, self.y_val)

    def test_platt_scaling_basic(self):
        """Verify basic Platt scaling using FrozenEstimator and method='sigmoid'."""
        calibrator = calibrate_model(
            trained_model=self.base_model,
            X_val=self.X_val,
            y_val=self.y_val,
            method="sigmoid",
        )
        self.assertIsNotNone(calibrator)

        # Predict on new samples
        X_test = np.random.randn(20, self.n_features).astype(np.float32)
        probs = calibrator.predict_proba(X_test)

        self.assertEqual(probs.shape, (20, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-5)

    def test_strict_constraint_no_cv_prefit(self):
        """STRICT CONSTRAINT: Calibrator MUST NOT use deprecated cv='prefit'."""
        calibrator = calibrate_model(
            trained_model=self.base_model,
            X_val=self.X_val,
            y_val=self.y_val,
            method="sigmoid",
        )
        # Verify cv attribute is not 'prefit'
        cv_attr = getattr(calibrator, "cv", None)
        self.assertNotEqual(cv_attr, "prefit", "cv='prefit' is strictly forbidden in scikit-learn 1.7.2!")

        # Verify contract checker passes
        self.assertTrue(verify_calibration_contract(calibrator))

    def test_calibrate_fold_models(self):
        """Verify calibration of multiple Level-0 models for a fold."""
        # Create 3 distinct fitted models
        m1 = LogisticRegression(C=0.1, random_state=1).fit(self.X_val, self.y_val)
        m2 = LogisticRegression(C=1.0, random_state=2).fit(self.X_val, self.y_val)
        m3 = LogisticRegression(C=10.0, random_state=3).fit(self.X_val, self.y_val)

        models_dict = {"lgbm": m1, "xgb": m2, "catboost": m3}
        calibrated_dict = calibrate_fold_models(
            models_dict=models_dict,
            X_val=self.X_val,
            y_val=self.y_val,
            method="sigmoid",
        )

        self.assertEqual(set(calibrated_dict.keys()), {"lgbm", "xgb", "catboost"})
        for name, cal in calibrated_dict.items():
            self.assertTrue(verify_calibration_contract(cal))

    def test_predict_calibrated_proba_matrix(self):
        """Verify extraction of [N, 3] calibrated probabilities matrix."""
        m1 = LogisticRegression(C=0.1, random_state=1).fit(self.X_val, self.y_val)
        m2 = LogisticRegression(C=1.0, random_state=2).fit(self.X_val, self.y_val)
        m3 = LogisticRegression(C=10.0, random_state=3).fit(self.X_val, self.y_val)

        calibrated_dict = calibrate_fold_models(
            models_dict={"lgbm": m1, "xgb": m2, "catboost": m3},
            X_val=self.X_val,
            y_val=self.y_val,
        )

        X_test = np.random.randn(30, self.n_features).astype(np.float32)
        oof_matrix = predict_calibrated_proba(
            calibrated_models=calibrated_dict,
            X=X_test,
            model_order=["lgbm", "xgb", "catboost"],
        )

        self.assertEqual(oof_matrix.shape, (30, 3))
        self.assertEqual(oof_matrix.dtype, np.float32)
        self.assertTrue(np.all(oof_matrix >= 0.0) and np.all(oof_matrix <= 1.0))

    def test_empty_validation_raises(self):
        """Verify ValueError is raised on empty validation set."""
        empty_X = np.empty((0, self.n_features))
        empty_y = np.empty(0)
        with self.assertRaises(ValueError):
            calibrate_model(self.base_model, empty_X, empty_y)


if __name__ == "__main__":
    unittest.main()
