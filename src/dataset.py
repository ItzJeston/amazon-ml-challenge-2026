"""
src/dataset.py
Milestone 3 (Phase 4: Disjoint Entity Splits & Cross-Validation Engine)
Amazon Business Entity Resolution Challenge.

Implements:
1. Strict 80/20 train/holdout split at source1_entity_id level via GroupShuffleSplit:
   - Eliminates entity token memorization leakage between Phase 5 (HPO/GBDT training)
     and Phase 6 (3D Bayesian threshold tuning).
   - Strict runtime assertion guaranteeing zero overlap between train and holdout S1 entities:
     assert len(train_s1 & holdout_s1) == 0.
2. Candidate-side leakage diagnostic (bipartite overlap check):
   - Computes total_unique_candidates, multi_s1_candidates, and leakage_fraction.
3. 5-Fold StratifiedGroupKFold splits for Phase 5 GBDT ensembling & Platt calibration:
   - Groups strictly by source1_entity_id.
   - Stratifies binary label distribution across all 5 validation folds.
   - Guarantees zero group leakage across CV folds.
4. Memory-bounded batch iterators (iter_feature_batches):
   - Streaming morsels via Polars slicing.
   - ZERO iter_rows() usage!

Hardware Target: Intel i7-13650HX (14 physical cores), peak memory <= 6.0 GB budget.
"""

import gc
import io
import os
from pathlib import Path
import sys
import time
from typing import Dict, Generator, List, Optional, Tuple, Union

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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.config import (
    ALL_COUNTRIES,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PROCESSED_DATA_DIR,
    STREAMING_BATCH_SIZE,
)
from src.features import (
    CANDIDATE_ID_COL,
    FEATURE_NAMES,
    LABEL_COL,
    SOURCE1_ID_COL,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("dataset")


# ==============================================================================
# 1. Disjoint 80/20 Train / Holdout Split (Strict Zero Leakage Assertion)
# ==============================================================================
def create_disjoint_train_holdout_split(
    features_df: pl.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    group_col: str = SOURCE1_ID_COL,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Partition feature pairs strictly at the source1_entity_id level into:
    - 80% Phase 5 Training Set (for GBDT 5-fold CV, Platt Scaling, and Meta-Learner)
    - 20% Phase 6 Holdout Set (strictly reserved for 3D Optuna Threshold Optimization)

    Integrity Mandate:
    Enforces a hard assertion that the intersection of S1 entities between
    train and holdout is strictly empty (zero leakage).

    Parameters
    ----------
    features_df : pl.DataFrame
        Complete training candidate pairs feature DataFrame.
    test_size : float
        Holdout fraction (default 0.20 for 80/20 split).
    random_state : int
        Deterministic random seed.
    group_col : str
        Entity ID column name used for grouping.

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        (train_df, holdout_df) with zero entity overlap.
    """
    if len(features_df) == 0:
        return features_df, features_df

    if group_col not in features_df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in DataFrame columns!")

    groups = features_df[group_col].to_numpy()
    unique_groups = np.unique(groups)
    n_unique_groups = len(unique_groups)

    if n_unique_groups < 2:
        logger.warning(f"Only {n_unique_groups} unique group(s) present, cannot perform disjoint split.")
        return features_df, features_df.clear()

    # Execute GroupShuffleSplit strictly on S1 entity groups
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, holdout_indices = next(gss.split(features_df, groups=groups))

    # Slice DataFrames using native Polars index filtering
    train_df = features_df[train_indices]
    holdout_df = features_df[holdout_indices]

    # ==========================================================================
    # ZERO-LEAKAGE INTEGRITY ASSERTIONS
    # ==========================================================================
    train_s1 = set(train_df[group_col].to_list())
    holdout_s1 = set(holdout_df[group_col].to_list())
    overlap = train_s1.intersection(holdout_s1)

    assert len(overlap) == 0, (
        f"CRITICAL LEAKAGE DETECTED: {len(overlap)} S1 entities appear in both "
        f"train and holdout sets! Overlapping samples: {list(overlap)[:5]}"
    )

    actual_holdout_ratio = len(holdout_s1) / n_unique_groups
    logger.info(
        f"Disjoint 80/20 Split Completed | Total S1 Entities: {n_unique_groups:,} | "
        f"Train S1: {len(train_s1):,} ({len(train_df):,} pairs) | "
        f"Holdout S1: {len(holdout_s1):,} ({len(holdout_df):,} pairs) | "
        f"Holdout Entity Ratio: {actual_holdout_ratio:.4f} (target: {test_size:.2f}) | "
        f"Zero Overlap Verified: YES"
    )

    return train_df, holdout_df


# ==============================================================================
# 2. Candidate-Side Bipartite Leakage Diagnostic
# ==============================================================================
def diagnose_candidate_leakage(
    pairs_df: Union[pl.DataFrame, pl.LazyFrame],
    s1_col: str = SOURCE1_ID_COL,
    cand_col: str = CANDIDATE_ID_COL,
) -> Dict[str, Union[int, float]]:
    """
    Quantify bipartite graph candidate sharing across Source 1 entity boundaries.
    Detects if individual candidate records (from S2 or S3) are paired with multiple S1 entities.

    Returns
    -------
    dict with:
    - total_unique_candidates: int
    - multi_s1_candidates: int (number of candidates spanning >1 S1 entity)
    - leakage_fraction: float (fraction of multi-span candidates)
    """
    stats_df = (
        pairs_df.lazy()
        .group_by(cand_col)
        .agg(pl.col(s1_col).n_unique().alias("s1_span_count"))
        .select(
            [
                pl.len().alias("total_unique_candidates"),
                (pl.col("s1_span_count") > 1).sum().alias("multi_s1_candidates"),
                (
                    (pl.col("s1_span_count") > 1).sum().cast(pl.Float64)
                    / pl.len().cast(pl.Float64)
                ).alias("leakage_fraction"),
            ]
        )
        .collect()
    )

    res = stats_df.to_dicts()[0]
    total = int(res["total_unique_candidates"] or 0)
    multi = int(res["multi_s1_candidates"] or 0)
    frac = float(res["leakage_fraction"] or 0.0)

    logger.info(
        f"Candidate Leakage Diagnostic: {multi:,} / {total:,} unique candidates "
        f"({frac * 100:.2f}%) span across multiple S1 entities."
    )
    return {
        "total_unique_candidates": total,
        "multi_s1_candidates": multi,
        "leakage_fraction": frac,
    }


# ==============================================================================
# 3. 5-Fold StratifiedGroupKFold Validation Iterators
# ==============================================================================
def get_cv_indices(
    df: pl.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    group_col: str = SOURCE1_ID_COL,
    label_col: str = LABEL_COL,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate 5-fold cross-validation train and validation index tuples.
    Groups strictly by source1_entity_id, stratifying on binary match label.

    Integrity Mandate:
    Asserts zero entity group leakage across train and val folds for every split.
    """
    if len(df) == 0:
        return []

    groups = df[group_col].to_numpy()
    if label_col in df.columns:
        y = df[label_col].to_numpy()
    else:
        # Fallback to pseudo labels if unlabeled
        y = np.zeros(len(df), dtype=np.int32)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups=groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)

        assert len(overlap) == 0, (
            f"CV FOLD {fold_idx} INTEGRITY VIOLATION: {len(overlap)} S1 groups "
            f"leak between training and validation folds!"
        )
        splits.append((train_idx, val_idx))

    return splits


def get_stratified_group_kfold_splits(
    df: pl.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    group_col: str = SOURCE1_ID_COL,
    label_col: str = LABEL_COL,
) -> Generator[Tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """
    Generator yielding (train_fold_df, val_fold_df) for 5-fold StratifiedGroupKFold cross-validation.
    Guarantees pure zero group leakage.
    """
    splits = get_cv_indices(
        df=df,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        group_col=group_col,
        label_col=label_col,
    )

    for train_idx, val_idx in splits:
        yield df[train_idx], df[val_idx]


# ==============================================================================
# 4. Memory-Safe Streaming Batch Iterator (Zero iter_rows)
# ==============================================================================
def iter_feature_batches(
    df: pl.DataFrame,
    batch_size: int = STREAMING_BATCH_SIZE,
) -> Generator[pl.DataFrame, None, None]:
    """
    Yields contiguous DataFrame slices of batch_size rows.
    Avoids in-memory replication, strictly preserving peak RAM <= 6.0 GB budget.
    STRICT CONSTRAINT: Pure Polars slicing. ZERO iter_rows() used!
    """
    n = len(df)
    for offset in range(0, n, batch_size):
        yield df.slice(offset, min(batch_size, n - offset))


# ==============================================================================
# 5. Feature Array Extraction Utilities for Model Training
# ==============================================================================
def get_feature_arrays(
    df: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    group_col: str = SOURCE1_ID_COL,
    label_col: str = LABEL_COL,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Extract (X, y, groups) NumPy arrays from Polars feature DataFrame.

    Returns
    -------
    X : np.ndarray (float32, shape [N, 32])
    y : Optional[np.ndarray] (uint8/float32, shape [N]) if label_col present, else None
    groups : np.ndarray (string, shape [N]) containing S1 entity IDs
    """
    cols = feature_cols or [c for c in FEATURE_NAMES if c in df.columns]
    if len(cols) == 0:
        raise ValueError("No feature columns found in DataFrame to extract!")

    # Zero-copy conversion where possible into float32 contiguous buffer
    X = df.select(cols).to_numpy().astype(np.float32)

    y = None
    if label_col in df.columns:
        y = df[label_col].to_numpy().astype(np.uint8)

    groups = df[group_col].to_numpy() if group_col in df.columns else np.empty(len(df), dtype=object)

    return X, y, groups


# ==============================================================================
# 6. Parquet Ingestion and Partition Saving
# ==============================================================================
def load_country_features(
    split: str,
    country: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """
    Load features.parquet for a specific split and country partition.
    """
    base = Path(data_dir or PROCESSED_DATA_DIR)
    path = base / split / country / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found at: {path}")

    logger.info(f"Loading features from {path}...")
    return pl.read_parquet(path)


def save_split_datasets(
    train_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    country: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """
    Persist disjoint train and holdout splits to zstd-compressed Parquet files:
    - data_processed/train/{country}/features_train.parquet
    - data_processed/train/{country}/features_holdout.parquet
    """
    base = Path(data_dir or PROCESSED_DATA_DIR) / "train" / country
    base.mkdir(parents=True, exist_ok=True)

    train_path = base / "features_train.parquet"
    holdout_path = base / "features_holdout.parquet"

    train_df.write_parquet(train_path, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    holdout_df.write_parquet(holdout_path, compression=PARQUET_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)

    logger.info(
        f"Saved disjoint splits for {country}: "
        f"Train -> {train_path} ({len(train_df):,} pairs) | "
        f"Holdout -> {holdout_path} ({len(holdout_df):,} pairs)"
    )
    return train_path, holdout_path


if __name__ == "__main__":
    logger.info("Executing src/dataset.py standalone smoke test...")
    # Quick sanity check on synthetic entities
    mock_s1 = [f"S1-{i:04d}" for i in range(50)]
    mock_df = pl.DataFrame({
        SOURCE1_ID_COL: np.repeat(mock_s1, 3),
        CANDIDATE_ID_COL: [f"S2-{j:05d}" for j in range(150)],
        LABEL_COL: np.random.choice([0, 1], size=150, p=[0.7, 0.3]),
    })
    tr, ho = create_disjoint_train_holdout_split(mock_df, test_size=0.20, random_state=42)
    diag = diagnose_candidate_leakage(mock_df)
    splits = get_cv_indices(tr, n_splits=5)
    print(f"Smoke test passed: Train={len(tr)}, Holdout={len(ho)}, CV Folds={len(splits)}")
    logger.info("src/dataset.py initialized successfully.")
