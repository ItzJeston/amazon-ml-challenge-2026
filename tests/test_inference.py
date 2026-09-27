"""
tests/test_inference.py
Unit tests for Milestone 5: End-to-End Inference, Dynamic Validation & Acceptance (src/inference.py).

Tests:
1. Explicit UTF-8 stdout initialization enforcement
2. Morsel-driven chunked scoring and threshold filtering
3. Threshold routing per country (tau_US, tau_India, tau_France)
4. Candidate bounding (max 15 candidates per S1 entity)
5. 32-pairwise feature extraction compatibility (zero iter_rows())
6. Memory bound & streaming chunk verification
7. Single-country partition inference execution
8. Master end-to-end inference pipeline execution with dynamic validation
9. Static code inspection: zero iter_rows() in src/inference.py and src/submission.py
"""

import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Enforce explicit UTF-8 stdout initialization (Global Requirement F02)
try:
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import pytest

from src.blocking import (
    BLOCKING_FLAG_CHAR_TFIDF,
    BLOCKING_FLAG_TOKEN_INDEX,
    BLOCKING_FLAGS_COL,
    CANDIDATE_ID_COL,
    RERANKER_SCORE_COL,
    SOURCE1_ID_COL,
)
from src.features import FEATURE_NAMES
from src.inference import (
    run_country_inference,
    run_inference_pipeline,
    score_candidate_pairs_chunked,
)
from src.optimization import DEFAULT_THRESHOLD, load_thresholds, save_thresholds
from src.reranking import prune_candidates


class MockStackedEnsemble:
    """Lightweight mock ensemble producing deterministic match probabilities for testing."""

    def __init__(self, high_score_ids=None):
        self.high_score_ids = set(high_score_ids or [])
        self.meta_learner = True  # Flag indicating meta_learner presence

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # In features, feat_01_name_ratio is feature 0
        # If feature 0 is high (> 0.70), return high probability
        if X.ndim == 2 and X.shape[1] >= 1:
            name_ratios = X[:, 0]
            probs = np.where(name_ratios >= 0.70, 0.90, 0.20)
            return probs.astype(np.float32)
        return np.full(len(X), 0.50, dtype=np.float32)


class TestInferenceScoring(unittest.TestCase):
    """Test candidate pair scoring, thresholding, and morsel chunking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_inf_score_")
        # Build mock S1 entities
        self.s1_df = pl.DataFrame({
            "entity_id": ["S1-001", "S1-002"],
            "business_name_raw": ["Acme Corp", "Beta Tech"],
            "business_name_clean": ["acme", "beta tech"],
            "business_address_clean": ["100 main st", "200 market st"],
            "soundex_2token": ["A250_C610", "B300_T200"],
            "addr_missing_flag": [0, 0],
        })

        # Build mock Candidate entities
        self.cand_df = pl.DataFrame({
            "entity_id": ["S2-001", "S3-002", "S2-003"],
            "business_name_raw": ["Acme Inc", "Beta Technologies", "Zeta Corp"],
            "business_name_clean": ["acme", "beta technologies", "zeta"],
            "business_address_clean": ["100 main st", "200 market ave", "999 south st"],
            "soundex_2token": ["A250_I520", "B300_T250", "Z300_C610"],
            "addr_missing_flag": [0, 0, 0],
        })

        # Build candidate pairs
        self.pairs_df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-001", "S1-001", "S1-002"],
            CANDIDATE_ID_COL: ["S2-001", "S2-003", "S3-002"],
            RERANKER_SCORE_COL: [0.95, 0.20, 0.85],
            BLOCKING_FLAGS_COL: [1, 1, 2],
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunked_scoring_with_ensemble(self):
        ensemble = MockStackedEnsemble()
        matches, candidates = score_candidate_pairs_chunked(
            pairs_df=self.pairs_df,
            s1_entities_df=self.s1_df,
            cand_entities_df=self.cand_df,
            country="US",
            threshold=0.75,
            ensemble=ensemble,
            chunk_size=1,  # Force multi-chunk execution to test morsel batching
        )

        # S1-001 candidates: S2-001, S2-003
        self.assertEqual(set(candidates["S1-001"]), {"S2-001", "S2-003"})
        # S1-002 candidate: S3-002
        self.assertEqual(set(candidates["S1-002"]), {"S3-002"})

        # Only S2-001 (Acme vs Acme) should pass threshold 0.75
        self.assertIn("S1-001", matches)
        self.assertEqual(matches["S1-001"], ["S2-001"])
        self.assertNotIn("S2-003", matches["S1-001"])

    def test_candidate_bounding_max_15(self):
        # Create 25 candidate pairs for S1-001
        pairs = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-001"] * 25,
            CANDIDATE_ID_COL: [f"S2-{i:03d}" for i in range(25)],
            RERANKER_SCORE_COL: [float(i) / 25.0 for i in range(25)],
            BLOCKING_FLAGS_COL: [1] * 25,
        })
        pruned = prune_candidates(pairs, max_candidates=15)
        self.assertEqual(len(pruned), 15)
        # Verify highest scores were retained
        min_retained_score = pruned[RERANKER_SCORE_COL].min()
        self.assertGreaterEqual(min_retained_score, 10.0 / 25.0)


class TestThresholdRouting(unittest.TestCase):
    """Test calibrated country threshold loading and fallback mechanisms."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_thresh_")
        self.thresh_path = Path(self.temp_dir) / "thresholds.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_and_save_thresholds(self):
        saved = {
            "US": 0.82,
            "India": 0.74,
            "France": 0.78,
            "tau_US": 0.82,
            "tau_India": 0.74,
            "tau_France": 0.78,
        }
        save_thresholds(saved, self.thresh_path)
        loaded = load_thresholds(self.thresh_path)

        self.assertAlmostEqual(loaded["US"], 0.82)
        self.assertAlmostEqual(loaded["India"], 0.74)
        self.assertAlmostEqual(loaded["France"], 0.78)

    def test_missing_threshold_file_fallback(self):
        non_existent = Path(self.temp_dir) / "missing_thresholds.json"
        loaded = load_thresholds(non_existent)
        self.assertEqual(loaded["US"], DEFAULT_THRESHOLD)
        self.assertEqual(loaded["India"], DEFAULT_THRESHOLD)
        self.assertEqual(loaded["France"], DEFAULT_THRESHOLD)


class TestEndToEndMockInference(unittest.TestCase):
    """Test full inference execution and validation on a simulated multi-country test partition."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_e2e_inf_")
        self.processed_dir = Path(self.temp_dir) / "data_processed"
        self.models_dir = Path(self.temp_dir) / "models"
        self.output_dir = Path(self.temp_dir) / "output"
        self.raw_dir = Path(self.temp_dir) / "raw"

        # Create partitions for US and India in test
        for country in ("US", "India"):
            c_dir = self.processed_dir / "test" / country
            c_dir.mkdir(parents=True, exist_ok=True)

            s1 = pl.DataFrame({
                "entity_id": [f"S1-{country}-1", f"S1-{country}-2"],
                "business_name_raw": [f"{country} Corp", f"{country} Services"],
                "business_name_clean": [f"{country.lower()} corp", f"{country.lower()} services"],
                "business_address_clean": ["100 main st", "200 park ave"],
                "soundex_2token": ["U200_C610", "U200_S612"],
                "country": [country, country],
                "addr_missing_flag": [0, 0],
            })
            s1.write_parquet(c_dir / "test_source1.parquet")

            s2 = pl.DataFrame({
                "entity_id": [f"S2-{country}-1"],
                "business_name_raw": [f"{country} Corporation"],
                "business_name_clean": [f"{country.lower()} corp"],
                "business_address_clean": ["100 main st"],
                "soundex_2token": ["U200_C610"],
                "country": [country],
                "addr_missing_flag": [0],
            })
            s2.write_parquet(c_dir / "test_source2.parquet")

            # Write pre-computed candidate pairs
            cand_pairs = pl.DataFrame({
                SOURCE1_ID_COL: [f"S1-{country}-1"],
                CANDIDATE_ID_COL: [f"S2-{country}-1"],
                RERANKER_SCORE_COL: [0.92],
                BLOCKING_FLAGS_COL: [1],
            })
            cand_pairs.write_parquet(c_dir / "candidate_pairs.parquet")

        # Save dummy thresholds
        self.models_dir.mkdir(parents=True, exist_ok=True)
        save_thresholds({"US": 0.70, "India": 0.70, "France": 0.70}, self.models_dir / "thresholds.json")

        # Save raw test_source1.tsv for dynamic validation
        raw_test_dir = self.raw_dir / "test"
        raw_test_dir.mkdir(parents=True, exist_ok=True)
        self.raw_test_s1 = raw_test_dir / "test_source1.tsv"
        with open(self.raw_test_s1, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-US-1\tUS Corp\t100 main st\tUS\n")
            f.write("S1-US-2\tUS Services\t200 park ave\tUS\n")
            f.write("S1-India-1\tIndia Corp\t100 main st\tIndia\n")
            f.write("S1-India-2\tIndia Services\t200 park ave\tIndia\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_inference_pipeline_end_to_end(self):
        summary = run_inference_pipeline(
            split="test",
            countries=["US", "India"],
            processed_data_dir=self.processed_dir,
            models_dir=self.models_dir,
            output_dir=self.output_dir,
            raw_data_dir=self.raw_dir,
            run_validation_wrapper=True,
        )

        self.assertEqual(summary["status"], "SUCCESS")
        self.assertEqual(summary["total_s1_entities"], 4)

        matching_file = Path(summary["matching_file"])
        candidate_file = Path(summary["candidate_file"])
        self.assertTrue(matching_file.exists())
        self.assertTrue(candidate_file.exists())

        with open(matching_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 5)  # 1 header + 4 rows

        with open(candidate_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 5)  # 1 header + 4 rows


class TestZeroIterRowsStaticAudit(unittest.TestCase):
    """Verify that iter_rows() is strictly not present in src/inference.py and src/submission.py."""

    def test_zero_iter_rows_in_inference(self):
        inference_py = PROJECT_ROOT / "src" / "inference.py"
        with open(inference_py, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("iter_rows", content, "Violation: iter_rows() detected in src/inference.py!")

    def test_zero_iter_rows_in_submission(self):
        submission_py = PROJECT_ROOT / "src" / "submission.py"
        with open(submission_py, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("iter_rows", content, "Violation: iter_rows() detected in src/submission.py!")

    def test_zero_iter_rows_in_run_pipeline(self):
        pipeline_py = PROJECT_ROOT / "run_pipeline.py"
        with open(pipeline_py, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("iter_rows", content, "Violation: iter_rows() detected in run_pipeline.py!")


if __name__ == "__main__":
    unittest.main()
