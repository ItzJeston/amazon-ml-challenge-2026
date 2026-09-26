"""
blocking.py - Stage 1 Candidate Generator for the Amazon Business Entity Resolution Challenge.

Strategy:
  1. Load the 20% validation split (S1) using chunked, low-RAM reader.
  2. Stream S2 and S3, keeping only rows matching S1 countries.
  3. Clean text: lowercase, remove punctuation, normalize legal suffixes.
  4. Partition records by country (dynamic — no hardcoded country values).
  5. For each country, build a TOKEN INVERTED INDEX over S2+S3 combined text.
     (This replaces TF-IDF + cosine similarity, which required a ~5 GB dense
      matrix and repeatedly triggered the Linux OOM killer.)
  6. For each S1 entity, look up its tokens in the index, score candidates by
     weighted token overlap (IDF-like: rarer tokens score higher), return top-12.
  7. Log candidate recall and average candidates per S1 entity on the val set.

Memory profile (quick mode):  ~300-500 MB peak   (vs ~5-8 GB with TF-IDF)
Runtime (quick mode, 10K S1): ~2-4 minutes        (vs OOM-killed every time)
"""

import os
import re
import sys
import time
import math
import logging
import collections
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SRC_DIR, '../../..'))
DATA_DIR   = os.path.join(BASE_DIR, '6ab10eb3b23ba_student_resource', 'student_resource', 'dataset', 'train')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SPLITS_DIR = OUTPUT_DIR

TOP_K          = 12
CHUNK_SIZE     = 100_000   # rows per chunk when streaming S2/S3
MAX_POSTINGS   = 20_000    # skip tokens appearing in > this many pool docs
                            # (they're de-facto stopwords with no discriminative
                            #  power but cause O(N) inner loops per query)

# Legal suffix normalization (expand → canonical short form)
LEGAL_SUFFIXES = [
    (r'\bprivate\b',     'pvt'),
    (r'\blimited\b',     'ltd'),
    (r'\bincorporated\b','inc'),
    (r'\bcorporation\b', 'corp'),
    (r'\bcompany\b',     'co'),
]

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip punctuation, normalize legal suffixes."""
    s = series.fillna('').str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    for pattern, replacement in LEGAL_SUFFIXES:
        s = s.str.replace(pattern, replacement, regex=True)
    return s


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """business_name + ' ' + business_address, cleaned."""
    return clean_text(df['business_name']) + ' ' + clean_text(df['business_address'])


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens (min length 2)."""
    return [t for t in text.split() if len(t) >= 2]


# ---------------------------------------------------------------------------
# Chunked / memory-safe loader
# ---------------------------------------------------------------------------

def read_filtered_tsv(path: str, countries: set, chunksize: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    Stream a TSV in chunks, keeping only rows whose 'country' is in `countries`.
    Peak RAM = one chunk + accumulated filtered rows — never loads full file.
    """
    parts = []
    for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=chunksize):
        filtered = chunk[chunk['country'].isin(countries)]
        if not filtered.empty:
            parts.append(filtered)
    if not parts:
        return pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Inverted index candidate generator
# ---------------------------------------------------------------------------

def build_inverted_index(pool_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build {token → [entity_id, ...]} and {token → idf_weight} for pool_df.

    Tokens appearing in > MAX_POSTINGS documents are discarded — they are
    de-facto stopwords and cause O(N) inner loops per query without adding
    discriminative power.

    IDF weight = log(N / df) + 1  (smoothed, always ≥ 1).
    """
    N = len(pool_df)
    pool_df = pool_df.copy()
    pool_df['comb_text'] = build_combined_text(pool_df)

    index: dict[str, list[str]] = collections.defaultdict(list)

    for row in pool_df.itertuples(index=False):
        eid    = row.entity_id
        tokens = set(tokenize(row.comb_text))
        for tok in tokens:
            index[tok].append(eid)

    # Prune tokens that are too frequent (stopword behaviour)
    before = len(index)
    index  = {tok: eids for tok, eids in index.items()
               if len(eids) <= MAX_POSTINGS}
    log.debug(f"  Index pruned {before - len(index):,} high-freq tokens "
              f"(>{MAX_POSTINGS} docs); kept {len(index):,}")

    # IDF weights (computed on pruned index so common tokens are already gone)
    idf: dict[str, float] = {
        tok: math.log((N + 1) / (len(eids) + 1)) + 1.0
        for tok, eids in index.items()
    }
    return dict(index), idf


def query_index(
    s1_text: str,
    index: dict[str, list[str]],
    idf: dict[str, float],
    top_k: int,
) -> list[str]:
    """
    Score every candidate that shares ≥1 token with s1_text.
    Score = sum of IDF weights of shared tokens.
    """
    tokens  = set(tokenize(s1_text))
    scores: dict[str, float] = collections.defaultdict(float)
    for tok in tokens:
        if tok in index:
            w = idf.get(tok, 1.0)
            for eid in index[tok]:
                scores[eid] += w

    if not scores:
        return []

    # Return top-k by score (descending)
    top = sorted(scores, key=scores.__getitem__, reverse=True)
    return top[:top_k]


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = TOP_K,
) -> dict[str, list[str]]:
    """
    For each S1 entity, retrieve top-k candidates from S2 ∪ S3 via
    IDF-weighted token inverted index (no dense matrices).
    """
    pool_df = pd.concat([s2_df, s3_df], ignore_index=True)

    countries = s1_df['country'].dropna().unique()
    log.info(f"Processing {len(countries)} countries: {sorted(countries)}")

    s1_df = s1_df.copy()
    s1_df['comb_text'] = build_combined_text(s1_df)

    all_candidates: dict[str, list[str]] = {}

    for country in countries:
        s1_c    = s1_df[s1_df['country'] == country]
        pool_c  = pool_df[pool_df['country'] == country]

        if s1_c.empty or pool_c.empty:
            for eid in s1_c['entity_id']:
                all_candidates[eid] = []
            continue

        log.info(f"  [{country}] S1={len(s1_c):,}  pool(S2+S3)={len(pool_c):,}")

        index, idf = build_inverted_index(pool_c)
        log.info(f"  [{country}] Index built — {len(index):,} unique tokens")

        for row in s1_c.itertuples(index=False):
            all_candidates[row.entity_id] = query_index(
                row.comb_text, index, idf, top_k
            )

    return all_candidates


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------

def compute_candidate_recall(
    candidates: dict[str, list[str]],
    gt_dict: dict[str, set],
) -> tuple[float, float]:
    """
    Candidate recall = TP / total GT positives (singletons excluded from denom).
    Also returns average candidate count per S1 entity.
    """
    total_tp = total_gt = total_cand = n = 0

    for s1_id, cand_list in candidates.items():
        cand_set = set(cand_list)
        gt_set   = gt_dict.get(s1_id, set())
        n           += 1
        total_cand  += len(cand_set)
        if gt_set:
            total_tp += len(cand_set & gt_set)
            total_gt += len(gt_set)

    recall         = total_tp / total_gt if total_gt > 0 else 0.0
    avg_candidates = total_cand / n      if n        > 0 else 0.0
    return recall, avg_candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# CLI flags:
#   (default)   QUICK mode — 10K S1 sample, 300K pool/country  (~3 min, ~400 MB)
#   --full      FULL  mode — all 441K val entities              (use /goal overnight)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, SRC_DIR)
    from metrics import load_ground_truth

    QUICK       = '--full' not in sys.argv
    SAMPLE_S1   = 10_000
    SAMPLE_POOL = 300_000

    t0 = time.time()

    # 1. Load S1 val split
    s1_val_path = os.path.join(SPLITS_DIR, 'train_source1_val_split.tsv')
    s2_path     = os.path.join(DATA_DIR,   'train_source2.tsv')
    s3_path     = os.path.join(DATA_DIR,   'train_source3.tsv')
    gt_path     = os.path.join(DATA_DIR,   'train_ground_truth.tsv')

    if not os.path.exists(s1_val_path):
        log.warning("Val split not found — regenerating …")
        from data_loader import load_data, create_validation_split
        s1_full, _, _, _ = load_data(DATA_DIR)
        create_validation_split(s1_full, SPLITS_DIR)

    log.info(f"Mode: {'QUICK (10K sample)' if QUICK else 'FULL (441K val)'}")
    s1_val = pd.read_csv(s1_val_path, sep='\t', dtype=str)

    if QUICK:
        s1_val = s1_val.sample(n=min(SAMPLE_S1, len(s1_val)), random_state=42)

    val_countries = set(s1_val['country'].dropna().unique())
    log.info(f"S1_val={len(s1_val):,}  countries={sorted(val_countries)}")

    # 2. Stream S2 / S3 — chunked, filter to relevant countries only
    log.info("Streaming S2 (chunked) …")
    s2 = read_filtered_tsv(s2_path, val_countries)
    log.info("Streaming S3 (chunked) …")
    s3 = read_filtered_tsv(s3_path, val_countries)

    if QUICK:
        s2 = pd.concat(
            [g.sample(n=min(SAMPLE_POOL, len(g)), random_state=42)
             for _, g in s2.groupby('country')],
            ignore_index=True,
        )
        s3 = pd.concat(
            [g.sample(n=min(SAMPLE_POOL, len(g)), random_state=42)
             for _, g in s3.groupby('country')],
            ignore_index=True,
        )

    log.info(f"Pool: S2={len(s2):,}  S3={len(s3):,}")

    # 3. Ground truth (filtered to val batch)
    gt_dict = load_ground_truth(gt_path)
    val_ids = set(s1_val['entity_id'].values)
    gt_val  = {k: v for k, v in gt_dict.items() if k in val_ids}
    log.info(f"GT entries for this batch: {len(gt_val):,}")

    # 4. Generate candidates via inverted index
    log.info(f"Building inverted index and retrieving top-{TOP_K} candidates …")
    candidates = generate_candidates(s1_val, s2, s3, top_k=TOP_K)

    t1 = time.time()

    # 5. Evaluate recall
    recall, avg_cands = compute_candidate_recall(candidates, gt_val)

    log.info("=" * 55)
    log.info(f"Candidate recall ({'quick' if QUICK else 'full'}): "
             f"{recall:.4f}  ({recall * 100:.2f}%)")
    log.info(f"Avg candidates per S1 entity:   {avg_cands:.2f}")
    log.info(f"Total time elapsed:             {t1 - t0:.1f}s")
    log.info("=" * 55)

    # 6. Save
    suffix   = 'quick' if QUICK else 'full'
    rows     = [{'source1_entity_id': s1_id, 'candidate_entity_id': cid}
                for s1_id, cands in candidates.items() for cid in cands]
    out_df   = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f'val_candidates_{suffix}.tsv')
    out_df.to_csv(out_path, sep='\t', index=False)
    log.info(f"Saved {len(out_df):,} candidate pairs → {out_path}")
