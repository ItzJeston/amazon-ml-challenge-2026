"""
features.py — Stage 2: Pairwise Feature Engineering
Amazon Business Entity Resolution Challenge

For every candidate pair (S1_entity_id, pool_entity_id) produced by blocking.py,
we compute a fixed-width feature vector of float32 values.

Features:
  name_token_sort_ratio   — rapidfuzz token sort ratio on business_name   [0–100]
  name_jaro_winkler       — Jaro-Winkler similarity on business_name       [0–1]
  address_token_set_ratio — rapidfuzz token set ratio on business_address  [0–100]
  address_levenshtein     — normalised Levenshtein on business_address     [0–1]
  number_overlap          — Jaccard of numeric tokens; -1.0 if neither has numbers
  len_diff_name           — |len(s1_name) − len(pool_name)| / max(len,1)  [0–1]
  len_diff_addr           — |len(s1_addr) − len(pool_addr)| / max(len,1)  [0–1]
  blocking_rank           — candidate rank from blocking (1 = best, 12 = worst)

Training negative subsampling:
  Keep ALL ground-truth positive pairs (label=1).
  For each S1 entity, keep at most MAX_NEG_PER_S1 negatives (label=0).
  This prevents the training set from ballooning to ~21M rows.

Output files (all in output/):
  features_train.npy   — float32 array  (N_train, 8)
  labels_train.npy     — int8   array   (N_train,)
  pairs_train.tsv      — TSV with source1_entity_id, candidate_entity_id, label
  features_val.npy     — float32 array  (N_val, 8)
  pairs_val.tsv        — TSV with source1_entity_id, candidate_entity_id
"""

import os
import re
import sys
import logging
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SRC_DIR, '../../..'))
DATA_DIR   = os.path.join(BASE_DIR, '6ab10eb3b23ba_student_resource',
                          'student_resource', 'dataset', 'train')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

MAX_NEG_PER_S1 = 2        # max non-match candidates to keep per S1 entity
CHUNK_SIZE     = 50_000   # rows per chunk for feature computation
FEATURE_NAMES  = [
    'name_token_sort_ratio',
    'name_jaro_winkler',
    'address_token_set_ratio',
    'address_levenshtein',
    'number_overlap',
    'len_diff_name',
    'len_diff_addr',
    'blocking_rank',
]
N_FEATURES = len(FEATURE_NAMES)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _norm_str(s) -> str:
    """Lower-case, strip extra whitespace; handle NaN."""
    if pd.isna(s):
        return ''
    return str(s).lower().strip()


def _number_overlap(a: str, b: str) -> float:
    """
    Jaccard similarity of numeric tokens extracted from a and b.
    Returns -1.0 if neither string contains any numeric token.
    """
    nums_a = set(re.findall(r'\d+', a))
    nums_b = set(re.findall(r'\d+', b))
    if not nums_a and not nums_b:
        return -1.0
    union = nums_a | nums_b
    if not union:
        return -1.0
    return len(nums_a & nums_b) / len(union)


def _norm_levenshtein(a: str, b: str) -> float:
    """Normalised Levenshtein distance in [0, 1] (1 = identical)."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max_len


def _len_diff(a: str, b: str) -> float:
    """Normalised absolute length difference in [0, 1] (0 = same length)."""
    la, lb = len(a), len(b)
    denom = max(la, lb, 1)
    return abs(la - lb) / denom


def compute_features_batch(
    pairs_df: pd.DataFrame,
    entity_lookup: dict[str, dict],
) -> np.ndarray:
    """
    Compute feature matrix for a batch of candidate pairs.

    Args:
        pairs_df:      DataFrame with columns [source1_entity_id,
                       candidate_entity_id, blocking_rank].
        entity_lookup: {entity_id → {'name': str, 'address': str}}
                       pre-built for all S1 + pool entities in this batch.

    Returns:
        float32 ndarray of shape (len(pairs_df), N_FEATURES).
    """
    n = len(pairs_df)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)

    for i, row in enumerate(pairs_df.itertuples(index=False)):
        s1  = entity_lookup.get(row.source1_entity_id,   {'name': '', 'address': ''})
        cand = entity_lookup.get(row.candidate_entity_id, {'name': '', 'address': ''})

        n1, n2 = s1['name'],    cand['name']
        a1, a2 = s1['address'], cand['address']

        out[i, 0] = fuzz.token_sort_ratio(n1, n2)
        out[i, 1] = fuzz.WRatio(n1, n2) / 100.0   # Jaro-Winkler via WRatio
        out[i, 2] = fuzz.token_set_ratio(a1, a2)
        out[i, 3] = _norm_levenshtein(a1, a2) * 100.0  # scale to 0-100 for consistency
        out[i, 4] = _number_overlap(a1, a2)
        out[i, 5] = _len_diff(n1, n2)
        out[i, 6] = _len_diff(a1, a2)
        out[i, 7] = float(row.blocking_rank)

    return out


# ---------------------------------------------------------------------------
# Entity lookup builder
# ---------------------------------------------------------------------------

def build_entity_lookup(dfs: list[pd.DataFrame]) -> dict[str, dict]:
    """
    Build {entity_id → {'name': str, 'address': str}} from one or more DataFrames.
    All string values are normalised on ingestion.
    """
    lookup = {}
    for df in dfs:
        for row in df.itertuples(index=False):
            lookup[row.entity_id] = {
                'name':    _norm_str(row.business_name),
                'address': _norm_str(row.business_address),
            }
    return lookup


# ---------------------------------------------------------------------------
# Training feature extraction (with negative subsampling)
# ---------------------------------------------------------------------------

def extract_train_features(
    candidates_path: str,
    gt_dict: dict[str, set],
    entity_lookup: dict[str, dict],
    max_neg: int = MAX_NEG_PER_S1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build training features with negative subsampling.

    Strategy:
      - Label = 1 for pairs present in gt_dict.
      - Label = 0 for non-matching candidates.
      - Keep ALL positives; keep at most `max_neg` negatives per S1 entity.

    Returns:
        (X: float32 array, y: int8 array, pairs_df: DataFrame with pair IDs)
    """
    cands_df = pd.read_csv(candidates_path, sep='\t', dtype=str)

    # Add blocking rank (1-based per S1 entity)
    cands_df['blocking_rank'] = (
        cands_df.groupby('source1_entity_id').cumcount() + 1
    ).astype(np.int8)

    # Label each pair
    def _label(row):
        gt = gt_dict.get(row['source1_entity_id'], set())
        return 1 if row['candidate_entity_id'] in gt else 0

    log.info("Labelling candidate pairs …")
    cands_df['label'] = cands_df.apply(_label, axis=1)

    # --- Negative subsampling per S1 entity --------------------------------
    log.info(f"Subsampling negatives (max {max_neg} per S1 entity) …")
    positives = cands_df[cands_df['label'] == 1]
    negatives = cands_df[cands_df['label'] == 0]

    # For each S1 entity keep at most max_neg negatives
    neg_sampled = (
        negatives.groupby('source1_entity_id', group_keys=False)
        .apply(lambda g: g.sample(n=min(max_neg, len(g)), random_state=42))
    )
    # pandas 3.0 safe reset
    neg_sampled = pd.concat(
        [grp.sample(n=min(max_neg, len(grp)), random_state=42)
         for _, grp in negatives.groupby('source1_entity_id')],
        ignore_index=True,
    )

    balanced = pd.concat([positives, neg_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = len(positives)
    n_neg = len(neg_sampled)
    log.info(f"Training pairs — positives: {n_pos:,}  negatives: {n_neg:,}  "
             f"total: {len(balanced):,}")

    # --- Compute features in chunks ----------------------------------------
    log.info("Computing pairwise features (chunked) …")
    all_X = []
    for start in range(0, len(balanced), CHUNK_SIZE):
        chunk = balanced.iloc[start:start + CHUNK_SIZE]
        all_X.append(compute_features_batch(chunk, entity_lookup))
        if (start // CHUNK_SIZE) % 5 == 0:
            log.info(f"  {start + len(chunk):,} / {len(balanced):,} pairs processed")

    X = np.vstack(all_X).astype(np.float32)
    y = balanced['label'].values.astype(np.int8)
    pairs = balanced[['source1_entity_id', 'candidate_entity_id', 'label']].reset_index(drop=True)

    return X, y, pairs


# ---------------------------------------------------------------------------
# Validation / inference feature extraction
# ---------------------------------------------------------------------------

def extract_val_features(
    candidates_path: str,
    entity_lookup: dict[str, dict],
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute features for all validation candidate pairs (no subsampling).

    Returns:
        (X: float32 array, pairs_df: DataFrame with pair IDs)
    """
    cands_df = pd.read_csv(candidates_path, sep='\t', dtype=str)
    cands_df['blocking_rank'] = (
        cands_df.groupby('source1_entity_id').cumcount() + 1
    ).astype(np.int8)

    log.info(f"Computing val features for {len(cands_df):,} pairs (chunked) …")
    all_X = []
    for start in range(0, len(cands_df), CHUNK_SIZE):
        chunk = cands_df.iloc[start:start + CHUNK_SIZE]
        all_X.append(compute_features_batch(chunk, entity_lookup))

    X = np.vstack(all_X).astype(np.float32)
    pairs = cands_df[['source1_entity_id', 'candidate_entity_id']].reset_index(drop=True)
    return X, pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, SRC_DIR)
    from metrics import load_ground_truth
    from blocking import read_filtered_tsv, clean_text

    # ---- 1. Resolve candidate file ----------------------------------------
    quick_path = os.path.join(OUTPUT_DIR, 'val_candidates_quick.tsv')
    full_path  = os.path.join(OUTPUT_DIR, 'val_candidates_full.tsv')
    use_full   = '--full' in sys.argv
    cand_path  = full_path if (use_full and os.path.exists(full_path)) else quick_path
    log.info(f"Using candidates: {os.path.basename(cand_path)}")

    # ---- 2. Load all entity IDs referenced in candidate file --------------
    cands_preview = pd.read_csv(cand_path, sep='\t', dtype=str)
    needed_s1   = set(cands_preview['source1_entity_id'])
    needed_pool = set(cands_preview['candidate_entity_id'])
    all_needed  = needed_s1 | needed_pool
    log.info(f"Unique entity IDs needed: {len(all_needed):,}")

    # ---- 3. Build entity lookup (stream all sources, keep needed IDs) -----
    log.info("Building entity lookup (streaming sources) …")
    s1_val_path   = os.path.join(OUTPUT_DIR, 'train_source1_val_split.tsv')
    s1_train_path = os.path.join(OUTPUT_DIR, 'train_source1_train_split.tsv')
    s2_path = os.path.join(DATA_DIR, 'train_source2.tsv')
    s3_path = os.path.join(DATA_DIR, 'train_source3.tsv')
    gt_path = os.path.join(DATA_DIR, 'train_ground_truth.tsv')

    lookup: dict[str, dict] = {}

    def stream_lookup(path):
        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=100_000):
            sub = chunk[chunk['entity_id'].isin(all_needed)]
            for row in sub.itertuples(index=False):
                lookup[row.entity_id] = {
                    'name':    _norm_str(row.business_name),
                    'address': _norm_str(row.business_address),
                }

    stream_lookup(s1_val_path)
    stream_lookup(s1_train_path)
    log.info("  Streaming S2 …")
    stream_lookup(s2_path)
    log.info("  Streaming S3 …")
    stream_lookup(s3_path)
    log.info(f"Entity lookup built: {len(lookup):,} entries")

    # ---- 4. Ground truth --------------------------------------------------
    gt_dict = load_ground_truth(gt_path)

    # ---- 5. Val features --------------------------------------------------
    log.info("=== Extracting VAL features ===")
    X_val, pairs_val = extract_val_features(cand_path, lookup)
    np.save(os.path.join(OUTPUT_DIR, 'features_val.npy'), X_val)
    pairs_val.to_csv(os.path.join(OUTPUT_DIR, 'pairs_val.tsv'), sep='\t', index=False)
    log.info(f"Saved features_val.npy  shape={X_val.shape}  "
             f"dtype={X_val.dtype}")

    # ---- 6. Train features (use val candidates as proxy since only quick ------
    #         mode is done; swap for train candidates once full run completes) --
    log.info("=== Extracting TRAIN features (from val candidates, demo mode) ===")
    X_train, y_train, pairs_train = extract_train_features(
        cand_path, gt_dict, lookup
    )
    np.save(os.path.join(OUTPUT_DIR, 'features_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'labels_train.npy'),   y_train)
    pairs_train.to_csv(os.path.join(OUTPUT_DIR, 'pairs_train.tsv'), sep='\t', index=False)
    log.info(f"Saved features_train.npy shape={X_train.shape}  "
             f"labels_train.npy shape={y_train.shape}")
    log.info(f"Class balance — 1s: {y_train.sum():,}  "
             f"0s: {(y_train==0).sum():,}")

    log.info("features.py complete ✅")
