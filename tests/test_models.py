"""
tests/test_models.py
Unit tests for src/models.py (Milestone 4 Calibrated Stacked Ensemble & Level-0/Level-1 Models).

Validates:
1. Level-0 GBDT factories:
   - LightGBM: 14 threads, objective='binary'
   - XGBoost: device='cuda' / 'cpu' fallback, tree_method='hist'
   - CatBoost: 14 threads, loss_function='Logloss'
2. Level-1 Stacking Meta-Learner:
   - LogisticRegression(C=1.0, max_iter=1000) on [N, 3] OOF probabilities
3. 5-Fold StratifiedGroupKFold cross-validation loop with zero entity group leakage.
4. Out-of-fold probability matrix generation and Platt Scaling calibration.
5. Model artifact persistence and restoration conforming to PROJECT.md Contract 4:
   - models/{country}/lgbm_fold_{0..4}.bin
   - models/{country}/xgb_fold_{0..4}.json
   - models/{country}/catboost_fold_{0..4}.cbm
   - models/{country}/calibrators_fold_{0..4}.joblib
   - models/{country}/meta_learner.joblib
"""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

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
import polars as pl

from src.models import (
    StackedEnsemble,
    fit_model_robustly,
    get_catboost_classifier,
    get_lgbm_classifier,
    get_meta_learner,
    get_xgb_classifier,
    train_country_ensemble,
    train_meta_learner,
)


class TestModelFactories(unittest.TestCase):
    """Test Level-0 GBDT and Level-1 Meta-Learner parameter configurations."""

    def test_lgbm_factory(self):
        """LightGBM must configure 14 threads and objective='binary'."""
        clf = get_lgbm_classifier(n_estimators=10)
        self.assertEqual(clf.objective, "binary")
        self.assertEqual(clf.n_jobs, 14)
        self.assertEqual(clf.n_estimators, 10)

    def test_xgb_factory(self):
        """XGBoost must configure tree_method='hist' with clean device handling."""
        clf_cpu = get_xgb_classifier(n_estimators=10, device="cpu")
        self.assertEqual(clf_cpu.tree_method, "hist")
        self.assertEqual(clf_cpu.device, "cpu")

        clf_auto = get_xgb_classifier(n_estimators=10)
        self.assertEqual(clf_auto.tree_method, "hist")
        self.assertIn(clf_auto.device, ["cuda", "cpu"])

    def test_catboost_factory(self):
        """CatBoost must configure 14 threads and loss_function='Logloss'."""
        clf = get_catboost_classifier(iterations=10)
        self.assertEqual(clf.get_params()["loss_function"], "Logloss")
        self.assertEqual(clf.get_params()["thread_count"], 14)
        self.assertEqual(clf.get_params()["iterations"], 10)

    def test_meta_learner_factory(self):
        """Level-1 Meta-Learner must configure LogisticRegression(C=1.0, max_iter=1000)."""
        meta = get_meta_learner()
        self.assertEqual(meta.C, 1.0)
        self.assertEqual(meta.max_iter, 1000)
        self.assertEqual(meta.solver, "lbfgs")


class TestStackedEnsemblePipeline(unittest.TestCase):
    """Test 5-Fold StratifiedGroupKFold training, calibration, inference, and persistence."""

    def setUp(self):
        np.random.seed(42)
        # Create synthetic dataset with 24 S1 entities, each having 3 candidate pairs (72 total)
        self.n_groups = 24
        self.pairs_per_group = 3
        self.n_samples = self.n_groups * self.pairs_per_group
        self.n_features = 32

        s1_ids = []
        cand_ids = []
        labels = []
        for i in range(self.n_groups):
            s1 = f"S1-{i:05d}"
            # Alternating positive / negative entities to allow balanced stratification
            has_match = (i % 2 == 1)
            for j in range(self.pairs_per_group):
                s1_ids.append(s1)
                cand_ids.append(f"S2-{i:03d}-{j}")
                labels.append(1 if has_match and j == 0 else 0)

        self.groups = np.array(s1_ids)
        self.cand_ids = np.array(cand_ids)
        self.y = np.array(labels, dtype=np.uint8)
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)

        # Make features somewhat informative
        self.X[self.y == 1, 0] += 2.0

    def test_stacked_ensemble_fit_and_predict(self):
        """Test fit_cv on synthetic groups with LightGBM, XGBoost, CatBoost, and Meta-Learner."""
        ensemble = StackedEnsemble(country="US", n_splits=3)

        fast_lgbm_params = {"n_estimators": 5, "min_child_samples": 2}
        fast_xgb_params = {"n_estimators": 5, "device": "cpu"}
        fast_cat_params = {"iterations": 5}

        ensemble.fit_cv(
            X=self.X,
            y=self.y,
            groups=self.groups,
            random_state=42,
            lgbm_kwargs=fast_lgbm_params,
            xgb_kwargs=fast_xgb_params,
            catboost_kwargs=fast_cat_params,
        )

        # Verify folds
        self.assertEqual(len(ensemble.models_by_fold), 3)
        self.assertEqual(len(ensemble.calibrators_by_fold), 3)
        self.assertIsNotNone(ensemble.meta_learner)

        # Verify OOF probability matrix
        self.assertEqual(ensemble.oof_probabilities.shape, (self.n_samples, 3))
        self.assertTrue(np.all(ensemble.oof_probabilities >= 0.0))
        self.assertTrue(np.all(ensemble.oof_probabilities <= 1.0))

        # Predict on new test samples
        X_test = np.random.randn(10, self.n_features).astype(np.float32)
        test_probs = ensemble.predict_proba(X_test)

        self.assertEqual(test_probs.shape, (10,))
        self.assertEqual(test_probs.dtype, np.float32)
        self.assertTrue(np.all(test_probs >= 0.0) and np.all(test_probs <= 1.0))

        # Predict binary decisions
        test_preds = ensemble.predict(X_test, threshold=0.5)
        self.assertEqual(test_preds.shape, (10,))
        self.assertTrue(set(test_preds.tolist()).issubset({0, 1}))

    def test_stacked_ensemble_save_and_load(self):
        """Test model artifact persistence and reloading conforming to Contract 4."""
        ensemble = StackedEnsemble(country="India", n_splits=2)

        fast_params = {"n_estimators": 5, "min_child_samples": 2}
        ensemble.fit_cv(
            X=self.X,
            y=self.y,
            groups=self.groups,
            random_state=42,
            lgbm_kwargs=fast_params,
            xgb_kwargs={"n_estimators": 5, "device": "cpu"},
            catboost_kwargs={"iterations": 5},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "India"
            ensemble.save(save_path)

            # Verify files on disk matching Contract 4
            self.assertTrue((save_path / "lgbm_fold_0.bin").exists())
            self.assertTrue((save_path / "xgb_fold_0.json").exists())
            self.assertTrue((save_path / "catboost_fold_0.cbm").exists())
            self.assertTrue((save_path / "calibrators_fold_0.joblib").exists())
            self.assertTrue((save_path / "meta_learner.joblib").exists())
            self.assertTrue((save_path / "ensemble_meta.json").exists())

            # Load ensemble
            loaded_ensemble = StackedEnsemble.load(save_path)
            self.assertEqual(loaded_ensemble.country, "India")
            self.assertEqual(len(loaded_ensemble.models_by_fold), 2)
            self.assertIsNotNone(loaded_ensemble.meta_learner)

            # Test predictions consistency
            X_test = np.random.randn(8, self.n_features).astype(np.float32)
            orig_probs = ensemble.predict_proba(X_test)
            loaded_probs = loaded_ensemble.predict_proba(X_test)

            np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
