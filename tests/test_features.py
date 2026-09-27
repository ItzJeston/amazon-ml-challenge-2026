"""
tests/test_features.py
Unit tests for src/features.py (Milestone 3 / Phase 4 Feature Engineering Engine).

Validates:
1. Feature calculation correctness across all 32 pairwise metrics.
2. Zero NaN or Inf values in output feature vectors.
3. Clean handling of missing, null, or empty addresses (defaulting to 0.0 with addr_missing_flag = 1.0).
4. RapidFuzz processor=rfuzz_utils.default_process enforcement (case, whitespace, punctuation).
5. Strict prohibition of iter_rows() in src/features.py and src/dataset.py via AST static scan.
6. Interface Contract 3 Parquet compliance (Float32 numerical types, required columns).
"""

import ast
import io
import os
from pathlib import Path
import re
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
from rapidfuzz import utils as rfuzz_utils

from src.features import (
    CANDIDATE_ID_COL,
    FEATURE_NAMES,
    LABEL_COL,
    POLARS_FEATURE_STRUCT_TYPE,
    SOURCE1_ID_COL,
    TIER1_FEATURE_NAMES,
    batch_extract_32_features,
    compute_pairwise_features,
    compute_qgram_similarity,
    export_features_to_parquet,
    prepare_joined_candidate_pairs,
)


class TestFeatureNamesAndSchema(unittest.TestCase):
    """Test feature names, counts, and structural definitions."""

    def test_feature_count_is_exact_32(self):
        self.assertEqual(len(FEATURE_NAMES), 32, "Feature engine must define exactly 32 canonical features")
        self.assertEqual(len(TIER1_FEATURE_NAMES), 32, "Tier 1 feature names list must contain exactly 32 features")

    def test_canonical_feature_names_start_with_feat_prefix(self):
        for idx, feat_name in enumerate(FEATURE_NAMES, 1):
            expected_prefix = f"feat_{idx:02d}_"
            self.assertTrue(
                feat_name.startswith(expected_prefix),
                f"Feature #{idx} '{feat_name}' must start with '{expected_prefix}'",
            )

    def test_first_and_last_feature_names(self):
        self.assertEqual(FEATURE_NAMES[0], "feat_01_name_ratio")
        self.assertEqual(FEATURE_NAMES[-1], "feat_32_blocking_rank")


class TestFeatureExtractionCorrectness(unittest.TestCase):
    """Test mathematical and string comparison logic for the 32 features."""

    def setUp(self):
        self.sample_df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-001", "S1-002", "S1-003", "S1-004"],
            CANDIDATE_ID_COL: ["S2-001", "S3-002", "S2-003", "S3-004"],
            "s1_name": [
                "Amazon Web Services Inc",
                "Tata Consultancy Services Ltd",
                "Apple Computer Inc",
                "Microsoft Corporation",
            ],
            "c_name": [
                "Amazon AWS Technologies LLC",
                "Tata Consultancy Services",
                "Apple Computer Inc",
                "Google Alphabet Inc",
            ],
            "s1_addr": [
                "410 Terry Ave N Seattle WA 98109",
                "TCS House Mumbai 400001",
                "1 Infinite Loop Cupertino CA 95014",
                "1 Microsoft Way Redmond WA 98052",
            ],
            "c_addr": [
                "410 Terry Ave North Seattle WA 98109",
                "TCS Campus Mumbai 400001",
                "1 Infinite Loop Cupertino CA 95014",
                None,  # Null address test
            ],
            "s1_soundex": ["A525_W126", "T300_C524", "A140_C513", "M262_C616"],
            "c_soundex": ["A525_A232", "T300_C524", "A140_C513", "G240_A411"],
            "reranker_cosine_score": [0.85, 0.95, 1.00, 0.20],
            "blocking_source_flags": [3, 7, 15, 1],
            "blocking_rank": [1, 1, 1, 4],
            "c_addr_missing_flag": [0, 0, 0, 1],
        })

    def test_identical_records_produce_expected_max_similarities(self):
        # Row 2 (Apple Computer Inc) is an exact match on name and address
        res = compute_pairwise_features(self.sample_df)
        row2 = res.filter(pl.col(SOURCE1_ID_COL) == "S1-003")

        # Name metrics should be 1.0 (or close for fuzzy)
        self.assertAlmostEqual(row2["feat_01_name_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_02_name_partial_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_03_name_token_sort_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_04_name_token_set_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_05_name_wratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_06_name_jaro_winkler"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_07_name_qgram"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_08_name_token_jaccard"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_09_name_token_overlap"][0], 1.0, places=3)
        self.assertEqual(row2["feat_10_name_first_token_match"][0], 1.0)
        self.assertEqual(row2["feat_11_name_last_token_match"][0], 1.0)

        # Soundex metrics
        self.assertEqual(row2["feat_12_soundex_2token_match"][0], 1.0)
        self.assertEqual(row2["feat_13_soundex_first_token_match"][0], 1.0)

        # Address metrics
        self.assertAlmostEqual(row2["feat_14_addr_levenshtein"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_15_addr_token_set_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_16_addr_token_sort_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(row2["feat_17_addr_token_jaccard"][0], 1.0, places=3)
        self.assertEqual(row2["feat_19_addr_numeric_exact_match"][0], 1.0)
        self.assertEqual(row2["feat_20_addr_numeric_jaccard"][0], 1.0)
        self.assertEqual(row2["feat_21_addr_missing_flag"][0], 0.0)

        # Length differences should be 0.0, ratios should be 1.0
        self.assertEqual(row2["feat_24_name_len_diff"][0], 0.0)
        self.assertAlmostEqual(row2["feat_25_name_len_ratio"][0], 1.0, places=3)
        self.assertEqual(row2["feat_26_name_token_count_diff"][0], 0.0)
        self.assertEqual(row2["feat_27_addr_len_diff"][0], 0.0)
        self.assertAlmostEqual(row2["feat_28_addr_len_ratio"][0], 1.0, places=3)

    def test_disjoint_records_produce_zero_similarities(self):
        # Row 3 (Microsoft vs Google)
        res = compute_pairwise_features(self.sample_df)
        row3 = res.filter(pl.col(SOURCE1_ID_COL) == "S1-004")

        # Name tokens are disjoint
        self.assertEqual(row3["feat_08_name_token_jaccard"][0], 0.0)
        self.assertEqual(row3["feat_09_name_token_overlap"][0], 0.0)
        self.assertEqual(row3["feat_10_name_first_token_match"][0], 0.0)
        self.assertEqual(row3["feat_12_soundex_2token_match"][0], 0.0)

    def test_source_indicator_encoding(self):
        res = compute_pairwise_features(self.sample_df)
        # S2-001 -> 0.0, S3-002 -> 1.0, S2-003 -> 0.0, S3-004 -> 1.0
        src_indicators = res["feat_29_source_indicator"].to_list()
        self.assertEqual(src_indicators, [0.0, 1.0, 0.0, 1.0])

    def test_zero_nan_or_inf_in_computed_features(self):
        res = compute_pairwise_features(self.sample_df)
        for col in FEATURE_NAMES:
            vals = res[col].to_numpy()
            self.assertFalse(np.isnan(vals).any(), f"NaN detected in feature '{col}'!")
            self.assertFalse(np.isinf(vals).any(), f"Inf detected in feature '{col}'!")

    def test_all_features_are_float32(self):
        res = compute_pairwise_features(self.sample_df)
        for col in FEATURE_NAMES:
            self.assertEqual(
                res[col].dtype,
                pl.Float32,
                f"Feature '{col}' must be Float32, got {res[col].dtype}!",
            )


class TestMissingAndNullAddressHandling(unittest.TestCase):
    """Test robust handling of null or missing addresses."""

    def test_missing_address_defaults_cleanly_to_zero(self):
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-010"],
            CANDIDATE_ID_COL: ["S2-020"],
            "s1_name": ["Test Business Inc"],
            "c_name": ["Test Business LLC"],
            "s1_addr": ["100 Main St Boston MA"],
            "c_addr": [None],  # Null candidate address (3.3% dataset condition)
            "s1_soundex": ["T231_B252"],
            "c_soundex": ["T231_B252"],
            "c_addr_missing_flag": [1],
        })

        res = compute_pairwise_features(df)
        # All address similarities must cleanly evaluate to 0.0
        self.assertEqual(res["feat_14_addr_levenshtein"][0], 0.0)
        self.assertEqual(res["feat_15_addr_token_set_ratio"][0], 0.0)
        self.assertEqual(res["feat_16_addr_token_sort_ratio"][0], 0.0)
        self.assertEqual(res["feat_17_addr_token_jaccard"][0], 0.0)
        self.assertEqual(res["feat_18_addr_token_overlap"][0], 0.0)
        self.assertEqual(res["feat_19_addr_numeric_exact_match"][0], 0.0)
        self.assertEqual(res["feat_20_addr_numeric_jaccard"][0], 0.0)
        # Missing address flag must evaluate to 1.0
        self.assertEqual(res["feat_21_addr_missing_flag"][0], 1.0)
        # Interaction address metrics must default to 0.0
        self.assertEqual(res["feat_22_name_addr_token_set"][0], 0.0)
        self.assertEqual(res["feat_23_name_addr_jaccard"][0], 0.0)
        self.assertEqual(res["feat_27_addr_len_diff"][0], 0.0)
        self.assertEqual(res["feat_28_addr_len_ratio"][0], 0.0)

    def test_both_addresses_missing_safe(self):
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-011"],
            CANDIDATE_ID_COL: ["S2-021"],
            "s1_name": ["Alpha Corp"],
            "c_name": ["Alpha Corp"],
            "s1_addr": [""],
            "c_addr": [""],
            "c_addr_missing_flag": [1],
        })
        res = compute_pairwise_features(df)
        self.assertEqual(res["feat_14_addr_levenshtein"][0], 0.0)
        self.assertEqual(res["feat_21_addr_missing_flag"][0], 1.0)


class TestRapidFuzzProcessorEnforcement(unittest.TestCase):
    """Test that processor=rfuzz_utils.default_process is strictly applied."""

    def test_casing_insensitivity_and_whitespace_collapse(self):
        # Ensure that uppercase, excess whitespace, and punctuation are normalized
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-100", "S1-101"],
            CANDIDATE_ID_COL: ["S2-100", "S2-101"],
            "s1_name": ["  STARBUCKS COFFEE LLC  ", "  STARBUCKS   COFFEE,  LLC  "],
            "c_name": ["starbucks coffee llc", "starbucks coffee llc"],
            "s1_addr": ["100 Broadway St", "100 Broadway St."],
            "c_addr": ["100 broadway st", "100 broadway st"],
        })
        res = compute_pairwise_features(df)
        # S1-100: exact match after case and leading/trailing whitespace strip
        self.assertAlmostEqual(res["feat_01_name_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(res["feat_03_name_token_sort_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(res["feat_04_name_token_set_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(res["feat_06_name_jaro_winkler"][0], 1.0, places=3)
        self.assertAlmostEqual(res["feat_14_addr_levenshtein"][0], 1.0, places=3)

        # S1-101: token sort and token set normalize punctuation and multiple spaces
        self.assertAlmostEqual(res["feat_03_name_token_sort_ratio"][1], 1.0, places=3)
        self.assertAlmostEqual(res["feat_04_name_token_set_ratio"][1], 1.0, places=3)
        self.assertAlmostEqual(res["feat_08_name_token_jaccard"][1], 1.0, places=3)


class TestZeroIterRowsStaticCheck(unittest.TestCase):
    """Static AST check verifying ZERO .iter_rows() in src/features.py and src/dataset.py."""

    def test_no_iter_rows_in_features_file(self):
        features_path = PROJECT_ROOT / "src" / "features.py"
        self.assertTrue(features_path.exists(), "src/features.py must exist")

        with open(features_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertNotIn(
            ".iter_rows(",
            code,
            "FORBIDDEN: iter_rows() detected in src/features.py! Pure native Polars map_batches required.",
        )

        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "iter_rows":
                self.fail(f"AST detected iter_rows call at line {node.lineno} in src/features.py!")

    def test_no_iter_rows_in_dataset_file(self):
        dataset_path = PROJECT_ROOT / "src" / "dataset.py"
        self.assertTrue(dataset_path.exists(), "src/dataset.py must exist")

        with open(dataset_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertNotIn(
            ".iter_rows(",
            code,
            "FORBIDDEN: iter_rows() detected in src/dataset.py! Pure Polars slicing required.",
        )

        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "iter_rows":
                self.fail(f"AST detected iter_rows call at line {node.lineno} in src/dataset.py!")


class TestInterfaceContract3Compliance(unittest.TestCase):
    """Test Parquet export and schema adherence to Interface Contract 3."""

    def test_parquet_export_schema_and_types(self):
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-001", "S1-002"],
            CANDIDATE_ID_COL: ["S2-001", "S3-002"],
            LABEL_COL: [1, 0],
            "s1_name": ["Name A", "Name B"],
            "c_name": ["Name A", "Name C"],
            "s1_addr": ["Addr A", "Addr B"],
            "c_addr": ["Addr A", "Addr C"],
        })
        features_df = compute_pairwise_features(df)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_parquet = Path(tmp_dir) / "features.parquet"
            export_features_to_parquet(features_df, tmp_parquet)

            self.assertTrue(tmp_parquet.exists())
            read_back = pl.read_parquet(tmp_parquet)

            self.assertEqual(len(read_back), 2)
            self.assertIn(SOURCE1_ID_COL, read_back.columns)
            self.assertIn(CANDIDATE_ID_COL, read_back.columns)
            self.assertIn(LABEL_COL, read_back.columns)

            for col in FEATURE_NAMES:
                self.assertIn(col, read_back.columns)
                self.assertEqual(read_back[col].dtype, pl.Float32)


if __name__ == "__main__":
    unittest.main()
