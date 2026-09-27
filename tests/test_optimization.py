"""
tests/test_optimization.py
Unit tests for src/optimization.py (Milestone 4 Entity-Level Macro F0.5 & 3D Bayesian Optuna Threshold Optimization).

Validates:
1. Entity-level Macro F0.5 metric implementation:
   - beta = 0.5 (weighting precision twice as heavily as recall: F0.5 = 1.25 * P * R / (0.25 * P + R)).
   - Singleton isolation: entities with 0 ground truth matches score 1.0 if prediction is empty,
     and 0.0 if any false merge is predicted.
   - Non-singletons: score 0.0 if prediction is empty or TP == 0; exact F0.5 otherwise.
2. Fast vectorized holdout evaluation across country partitions.
3. 3D Bayesian Optuna search mapping (tau_US, tau_India, tau_France) in [0.40, 0.95].
4. Threshold persistence to and loading from models/thresholds.json.
"""

import io
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

from src.optimization import (
    DEFAULT_THRESHOLD,
    HoldoutEvaluationDataset,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    compute_entity_macro_f05,
    compute_single_entity_f05,
    load_ground_truth_dict,
    load_thresholds,
    optimize_3d_thresholds,
    save_thresholds,
)


class TestEntityMacroF05Metric(unittest.TestCase):
    """Test suite for competition singleton handling and Macro F0.5 metric."""

    def test_singleton_handling(self):
        """Singletons (0 true matches) must score 1.0 if empty, 0.0 if false merge."""
        true_singleton = set()

        # Correctly isolated singleton -> 1.0
        score_clean = compute_single_entity_f05(true_singleton, set())
        self.assertEqual(score_clean, 1.0, "Correctly isolated singleton must score 1.0!")

        # False merge on singleton -> 0.0
        score_false_merge = compute_single_entity_f05(true_singleton, {"S2-00001"})
        self.assertEqual(score_false_merge, 0.0, "False merge on singleton must score 0.0!")

        # Multiple false merges on singleton -> 0.0
        score_multi_merge = compute_single_entity_f05(true_singleton, {"S2-00001", "S3-00002"})
        self.assertEqual(score_multi_merge, 0.0)

    def test_non_singleton_exact_match(self):
        """Non-singleton with exact prediction must score 1.0."""
        true_matches = {"S2-00001", "S3-00002"}
        pred_matches = {"S2-00001", "S3-00002"}

        score = compute_single_entity_f05(true_matches, pred_matches)
        self.assertEqual(score, 1.0)

    def test_non_singleton_missed_link(self):
        """Non-singleton with empty prediction (missed link) must score 0.0."""
        true_matches = {"S2-00001"}
        score = compute_single_entity_f05(true_matches, set())
        self.assertEqual(score, 0.0)

    def test_non_singleton_zero_tp(self):
        """Non-singleton where predicted candidates do not overlap true candidates must score 0.0."""
        true_matches = {"S2-00001"}
        pred_matches = {"S2-99999"}
        score = compute_single_entity_f05(true_matches, pred_matches)
        self.assertEqual(score, 0.0)

    def test_f05_mathematical_precision_weighting(self):
        """
        Verify exact F0.5 formula: (1.25 * P * R) / (0.25 * P + R).
        Example 1: True = {A, B}, Pred = {A}.
        TP = 1, FP = 0, FN = 1.
        P = 1.0, R = 0.5.
        F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.83333...
        """
        true_matches = {"S2-A", "S3-B"}
        pred_matches = {"S2-A"}
        score = compute_single_entity_f05(true_matches, pred_matches)
        expected = 5.0 / 6.0
        self.assertAlmostEqual(score, expected, places=5)

    def test_f05_mathematical_fp_penalty(self):
        """
        Example 2: True = {A}, Pred = {A, X}.
        TP = 1, FP = 1, FN = 0.
        P = 0.5, R = 1.0.
        F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.55555...
        Notice: Because beta=0.5 weights precision twice as heavily as recall,
        having low precision (5/9 = 0.555) is penalized more heavily than low recall (5/6 = 0.833)!
        """
        true_matches = {"S2-A"}
        pred_matches = {"S2-A", "S2-X"}
        score = compute_single_entity_f05(true_matches, pred_matches)
        expected = 5.0 / 9.0
        self.assertAlmostEqual(score, expected, places=5)

    def test_macro_f05_across_entities(self):
        """Test macro average across a diverse set of entities."""
        ground_truth = {
            "S1-01": set(),  # singleton
            "S1-02": {"S2-A"},  # perfect match
            "S1-03": {"S2-B"},  # missed link
            "S1-04": set(),  # false merge singleton
        }
        predictions = {
            "S1-01": set(),  # clean singleton -> 1.0
            "S1-02": {"S2-A"},  # perfect match -> 1.0
            "S1-03": set(),  # missed link -> 0.0
            "S1-04": {"S2-Z"},  # false merge on singleton -> 0.0
        }
        macro_score = compute_entity_macro_f05(ground_truth, predictions)
        # Expected: (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.50
        self.assertAlmostEqual(macro_score, 0.50, places=5)


class TestOptuna3DOptimization(unittest.TestCase):
    """Test suite for 3D Bayesian Optuna threshold search on holdout."""

    def setUp(self):
        np.random.seed(42)
        # Construct synthetic holdout dataset for US and India
        self.ground_truth = {
            "S1-US-01": {"S2-001"},
            "S1-US-02": set(),  # singleton
            "S1-IN-01": {"S3-101"},
            "S1-IN-02": set(),  # singleton
        }

        # US holdout pairs: candidate probabilities
        us_df = pl.DataFrame({
            "source1_entity_id": ["S1-US-01", "S1-US-01", "S1-US-02"],
            "candidate_entity_id": ["S2-001", "S2-999", "S2-888"],
        })
        us_probs = np.array([0.90, 0.45, 0.55], dtype=np.float32)

        # India holdout pairs
        in_df = pl.DataFrame({
            "source1_entity_id": ["S1-IN-01", "S1-IN-01", "S1-IN-02"],
            "candidate_entity_id": ["S3-101", "S3-999", "S3-888"],
        })
        in_probs = np.array([0.85, 0.40, 0.50], dtype=np.float32)

        self.holdout_dataset = HoldoutEvaluationDataset.from_holdout_pairs(
            holdout_pairs_by_country={"US": us_df, "India": in_df},
            probabilities_by_country={"US": us_probs, "India": in_probs},
            ground_truth=self.ground_truth,
        )

    def test_holdout_evaluation_threshold_sweep(self):
        """High thresholds (e.g. 0.88) should filter low false positive candidates and boost F0.5."""
        # Low threshold (0.42) lets false candidates through
        low_score = self.holdout_dataset.evaluate_thresholds({"US": 0.42, "India": 0.42})

        # High threshold (0.80) cuts off noise and singletons are cleanly isolated
        high_score = self.holdout_dataset.evaluate_thresholds({"US": 0.80, "India": 0.80})

        self.assertGreater(high_score, low_score)

    def test_3d_optuna_search_and_persistence(self):
        """Execute 3D Optuna optimization and verify threshold file persistence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            thresholds_json = Path(tmp_dir) / "thresholds.json"

            best_thresholds = optimize_3d_thresholds(
                holdout_eval_data=self.holdout_dataset,
                n_trials=15,
                seed=42,
                output_path=thresholds_json,
            )

            # Assert thresholds in valid range [0.40, 0.95]
            for c in ["US", "India", "France"]:
                self.assertIn(c, best_thresholds)
                val = best_thresholds[c]
                self.assertGreaterEqual(val, MIN_THRESHOLD)
                self.assertLessEqual(val, MAX_THRESHOLD)

            # Assert file exists and load works
            self.assertTrue(thresholds_json.exists())
            loaded = load_thresholds(thresholds_json)
            self.assertEqual(loaded["US"], best_thresholds["US"])
            self.assertEqual(loaded["India"], best_thresholds["India"])
            self.assertEqual(loaded["France"], best_thresholds["France"])


if __name__ == "__main__":
    unittest.main()
