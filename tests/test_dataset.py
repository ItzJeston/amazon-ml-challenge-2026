"""
tests/test_dataset.py
Unit tests for src/dataset.py (Milestone 3 / Phase 4 Disjoint Splits & CV Engine).

Validates:
1. Disjoint 80/20 train/holdout split at source1_entity_id level via GroupShuffleSplit.
2. Hard assertion ensuring ZERO overlap of S1 entities between train and holdout splits.
3. Candidate pair clustering: all candidate pairs for any given S1 entity reside in the same split.
4. Candidate-side leakage diagnostic (bipartite candidate sharing).
5. 5-Fold StratifiedGroupKFold cross-validation splits with zero group leakage across all 5 folds.
6. Memory-safe streaming batch iterators and feature array extraction.
7. Zero iter_rows() usage in dataset logic.
"""

import ast
import io
import os
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

from src.dataset import (
    create_disjoint_train_holdout_split,
    diagnose_candidate_leakage,
    get_cv_indices,
    get_feature_arrays,
    get_stratified_group_kfold_splits,
    iter_feature_batches,
    load_country_features,
    save_split_datasets,
)
from src.features import (
    CANDIDATE_ID_COL,
    FEATURE_NAMES,
    LABEL_COL,
    SOURCE1_ID_COL,
)


class TestDisjointTrainHoldoutSplit(unittest.TestCase):
    """Test 80/20 GroupShuffleSplit strictly grouped by source1_entity_id."""

    def setUp(self):
        np.random.seed(42)
        # Create 100 unique S1 entities, each having 2 to 6 candidate pairs
        s1_ids = []
        cand_ids = []
        labels = []

        cand_counter = 0
        for i in range(100):
            s1_id = f"S1-{i:05d}"
            n_pairs = np.random.randint(2, 7)
            for _ in range(n_pairs):
                s1_ids.append(s1_id)
                cand_ids.append(f"S2-{cand_counter:06d}")
                labels.append(np.random.choice([0, 1], p=[0.75, 0.25]))
                cand_counter += 1

        self.df = pl.DataFrame({
            SOURCE1_ID_COL: s1_ids,
            CANDIDATE_ID_COL: cand_ids,
            LABEL_COL: labels,
        })
        # Add dummy 32 feature columns
        for feat in FEATURE_NAMES:
            self.df = self.df.with_columns(
                pl.lit(0.5, dtype=pl.Float32).alias(feat)
            )

    def test_split_ratio_approximately_20_percent(self):
        train_df, holdout_df = create_disjoint_train_holdout_split(
            self.df, test_size=0.20, random_state=42
        )
        unique_s1_total = len(self.df[SOURCE1_ID_COL].unique())
        unique_s1_holdout = len(holdout_df[SOURCE1_ID_COL].unique())
        holdout_ratio = unique_s1_holdout / unique_s1_total

        # Should be exactly 20 out of 100 entities (20%)
        self.assertEqual(unique_s1_holdout, 20)
        self.assertAlmostEqual(holdout_ratio, 0.20, delta=0.02)

    def test_zero_s1_entity_leakage_assertion(self):
        train_df, holdout_df = create_disjoint_train_holdout_split(
            self.df, test_size=0.20, random_state=42
        )
        train_s1 = set(train_df[SOURCE1_ID_COL].to_list())
        holdout_s1 = set(holdout_df[SOURCE1_ID_COL].to_list())

        # Crucial integrity mandate: Zero overlap
        overlap = train_s1.intersection(holdout_s1)
        self.assertEqual(
            len(overlap),
            0,
            f"Data leakage detected! S1 entities overlap between train and holdout: {overlap}",
        )

    def test_pair_clustering_integrity(self):
        # All candidate pairs for an S1 entity must be exclusively in train OR holdout
        train_df, holdout_df = create_disjoint_train_holdout_split(
            self.df, test_size=0.20, random_state=42
        )
        train_s1_set = set(train_df[SOURCE1_ID_COL].to_list())
        holdout_s1_set = set(holdout_df[SOURCE1_ID_COL].to_list())

        # Verify that total rows match
        self.assertEqual(len(train_df) + len(holdout_df), len(self.df))

        # Check that no row in train belongs to holdout S1 entities
        for s1 in train_df[SOURCE1_ID_COL].to_list():
            self.assertNotIn(s1, holdout_s1_set)

        # Check that no row in holdout belongs to train S1 entities
        for s1 in holdout_df[SOURCE1_ID_COL].to_list():
            self.assertNotIn(s1, train_s1_set)

    def test_reproducibility_with_random_state(self):
        tr1, ho1 = create_disjoint_train_holdout_split(self.df, test_size=0.20, random_state=123)
        tr2, ho2 = create_disjoint_train_holdout_split(self.df, test_size=0.20, random_state=123)

        self.assertEqual(tr1[SOURCE1_ID_COL].to_list(), tr2[SOURCE1_ID_COL].to_list())
        self.assertEqual(ho1[SOURCE1_ID_COL].to_list(), ho2[SOURCE1_ID_COL].to_list())

    def test_empty_dataframe_split(self):
        empty_df = pl.DataFrame(schema=self.df.schema)
        tr, ho = create_disjoint_train_holdout_split(empty_df, test_size=0.20)
        self.assertEqual(len(tr), 0)
        self.assertEqual(len(ho), 0)


class TestCandidateLeakageDiagnostic(unittest.TestCase):
    """Test candidate-side bipartite leakage detection."""

    def test_zero_candidate_leakage(self):
        pairs_df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1", "S1-2", "S1-3"],
            CANDIDATE_ID_COL: ["S2-A", "S2-B", "S2-C"],
        })
        diag = diagnose_candidate_leakage(pairs_df)
        self.assertEqual(diag["total_unique_candidates"], 3)
        self.assertEqual(diag["multi_s1_candidates"], 0)
        self.assertEqual(diag["leakage_fraction"], 0.0)

    def test_detected_candidate_leakage(self):
        # Candidate 'S2-SHARED' spans across S1-1 and S1-2
        pairs_df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1", "S1-2", "S1-3"],
            CANDIDATE_ID_COL: ["S2-SHARED", "S2-SHARED", "S2-SOLO"],
        })
        diag = diagnose_candidate_leakage(pairs_df)
        self.assertEqual(diag["total_unique_candidates"], 2)
        self.assertEqual(diag["multi_s1_candidates"], 1)
        self.assertAlmostEqual(diag["leakage_fraction"], 0.50, delta=0.01)


class TestStratifiedGroupKFoldValidation(unittest.TestCase):
    """Test 5-fold cross-validation scheme grouped by S1 entity."""

    def setUp(self):
        np.random.seed(99)
        s1_ids = np.repeat([f"S1-{i:04d}" for i in range(50)], 4)
        cand_ids = [f"S2-{j:05d}" for j in range(len(s1_ids))]
        labels = np.random.choice([0, 1], size=len(s1_ids), p=[0.70, 0.30])

        self.df = pl.DataFrame({
            SOURCE1_ID_COL: s1_ids,
            CANDIDATE_ID_COL: cand_ids,
            LABEL_COL: labels,
        })
        for feat in FEATURE_NAMES:
            self.df = self.df.with_columns(
                pl.lit(0.75, dtype=pl.Float32).alias(feat)
            )

    def test_five_folds_generation(self):
        splits = get_cv_indices(self.df, n_splits=5, random_state=42)
        self.assertEqual(len(splits), 5, "Must generate exactly 5 CV folds")

    def test_zero_group_leakage_across_all_folds(self):
        splits = get_cv_indices(self.df, n_splits=5, random_state=42)
        groups = self.df[SOURCE1_ID_COL].to_numpy()

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups.intersection(val_groups)
            self.assertEqual(
                len(overlap),
                0,
                f"CV Fold {fold_idx} has {len(overlap)} overlapping S1 entities!",
            )

    def test_complete_validation_coverage(self):
        splits = get_cv_indices(self.df, n_splits=5, random_state=42)
        all_val_indices = []
        for _, val_idx in splits:
            all_val_indices.extend(val_idx)

        # Every single row in the dataset must appear in validation exactly once
        self.assertEqual(len(all_val_indices), len(self.df))
        self.assertEqual(set(all_val_indices), set(range(len(self.df))))

    def test_stratified_group_generator_yields_valid_dfs(self):
        gen = get_stratified_group_kfold_splits(self.df, n_splits=5, random_state=42)
        fold_count = 0
        for train_fold, val_fold in gen:
            fold_count += 1
            self.assertGreater(len(train_fold), 0)
            self.assertGreater(len(val_fold), 0)
            self.assertEqual(len(train_fold) + len(val_fold), len(self.df))
        self.assertEqual(fold_count, 5)


class TestBatchIteratorsAndFeatureArrays(unittest.TestCase):
    """Test streaming batch iteration and numpy feature matrix extraction."""

    def test_iter_feature_batches_chunking(self):
        # 125 rows sliced with batch_size 50 -> chunks of 50, 50, 25
        df = pl.DataFrame({
            SOURCE1_ID_COL: [f"S1-{i}" for i in range(125)],
            CANDIDATE_ID_COL: [f"S2-{i}" for i in range(125)],
        })
        chunks = list(iter_feature_batches(df, batch_size=50))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 50)
        self.assertEqual(len(chunks[1]), 50)
        self.assertEqual(len(chunks[2]), 25)
        # Verify no data loss
        reconstructed = pl.concat(chunks)
        self.assertEqual(len(reconstructed), 125)

    def test_get_feature_arrays_extraction(self):
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1", "S1-2"],
            CANDIDATE_ID_COL: ["S2-1", "S2-2"],
            LABEL_COL: [1, 0],
        })
        for feat in FEATURE_NAMES:
            df = df.with_columns(pl.lit(0.42, dtype=pl.Float32).alias(feat))

        X, y, groups = get_feature_arrays(df)
        self.assertEqual(X.shape, (2, 32))
        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.allclose(X, 0.42))
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(groups.tolist(), ["S1-1", "S1-2"])

    def test_save_and_load_split_datasets(self):
        train_df = pl.DataFrame({SOURCE1_ID_COL: ["S1-1"], CANDIDATE_ID_COL: ["S2-1"]})
        holdout_df = pl.DataFrame({SOURCE1_ID_COL: ["S1-2"], CANDIDATE_ID_COL: ["S2-2"]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            tr_path, ho_path = save_split_datasets(
                train_df, holdout_df, country="US", data_dir=tmp_dir
            )
            self.assertTrue(tr_path.exists())
            self.assertTrue(ho_path.exists())

            read_tr = pl.read_parquet(tr_path)
            read_ho = pl.read_parquet(ho_path)
            self.assertEqual(len(read_tr), 1)
            self.assertEqual(len(read_ho), 1)


class TestZeroIterRowsInDataset(unittest.TestCase):
    """Verify zero iter_rows() calls in src/dataset.py."""

    def test_no_iter_rows_in_dataset_py(self):
        dataset_path = PROJECT_ROOT / "src" / "dataset.py"
        with open(dataset_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertNotIn(".iter_rows(", code, "Forbidden iter_rows() found in src/dataset.py!")

        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "iter_rows":
                self.fail(f"AST detected iter_rows at line {node.lineno} in src/dataset.py!")


if __name__ == "__main__":
    unittest.main()
