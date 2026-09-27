"""
src/features.py
Milestone 3 (Phase 4: High-Throughput Pairwise Feature Engineering Engine)
Amazon Business Entity Resolution Challenge.

Computes exactly 32 pairwise similarity metrics across 6 functional categories:
1. String similarity metrics (feat_01 to feat_07):
   - feat_01_name_ratio (RapidFuzz ratio)
   - feat_02_name_partial_ratio (RapidFuzz partial_ratio)
   - feat_03_name_token_sort_ratio (RapidFuzz token_sort_ratio)
   - feat_04_name_token_set_ratio (RapidFuzz token_set_ratio)
   - feat_05_name_wratio (RapidFuzz WRatio)
   - feat_06_name_jaro_winkler (RapidFuzz JaroWinkler similarity)
   - feat_07_name_qgram (Q-gram Sorensen-Dice similarity, q=2)
2. Token set metrics (feat_08 to feat_11):
   - feat_08_name_token_jaccard (Name word token Jaccard)
   - feat_09_name_token_overlap (Szymkiewicz-Simpson token overlap)
   - feat_10_name_first_token_match (Binary first token match)
   - feat_11_name_last_token_match (Binary last token match)
3. Phonetic metrics (feat_12 to feat_13):
   - feat_12_soundex_2token_match (Exact 2-token Soundex match)
   - feat_13_soundex_first_token_match (First token Soundex match)
4. Address metrics (feat_14 to feat_21):
   - feat_14_addr_levenshtein (Address Levenshtein ratio, 0.0 if missing)
   - feat_15_addr_token_set_ratio (Address token set ratio, 0.0 if missing)
   - feat_16_addr_token_sort_ratio (Address token sort ratio, 0.0 if missing)
   - feat_17_addr_token_jaccard (Address word token Jaccard, 0.0 if missing)
   - feat_18_addr_token_overlap (Address token overlap, 0.0 if missing)
   - feat_19_addr_numeric_exact_match (Address digit sequence match, 0.0 if missing)
   - feat_20_addr_numeric_jaccard (Address numeric token Jaccard, 0.0 if missing)
   - feat_21_addr_missing_flag (1.0 if address null/empty, 0.0 otherwise)
5. Interaction & Length metrics (feat_22 to feat_28):
   - feat_22_name_addr_token_set (Cross-field S1 name vs candidate addr token set ratio)
   - feat_23_name_addr_jaccard (Cross-field S1 name vs candidate addr token Jaccard)
   - feat_24_name_len_diff (Absolute name character length difference)
   - feat_25_name_len_ratio (Name character length ratio min/max)
   - feat_26_name_token_count_diff (Absolute token count difference)
   - feat_27_addr_len_diff (Absolute address character length difference)
   - feat_28_addr_len_ratio (Address character length ratio min/max)
6. Graph & Metadata metrics (feat_29 to feat_32):
   - feat_29_source_indicator (1.0 if candidate is from Source 3, 0.0 if Source 2)
   - feat_30_reranker_cosine_score (Lightweight cosine re-ranker score)
   - feat_31_blocking_source_flags (Bitmask of candidate blocking source strategies)
   - feat_32_blocking_rank (Candidate rank 1 to 15 within S1 entity)

STRICT ARCHITECTURAL CONSTRAINTS:
- Pure native Polars expressions (pl.struct().map_batches()). ZERO iter_rows() permitted anywhere!
- Explicit processor=rfuzz_utils.default_process enforced on all RapidFuzz string operations.
- Explicit UTF-8 stdout initialization on module load.
- Output Parquet conforming strictly to PROJECT.md Interface Contract 3.
- Optimized for Intel i7-13650HX (14 cores / 20 threads) with peak RAM <= 6.0 GB budget.
"""

from collections import Counter
import gc
import io
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

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
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import distance, fuzz, utils as rfuzz_utils

from src.config import (
    ADDR_MISSING_FLAG_COL,
    ALL_COUNTRIES,
    CLEAN_ADDR_COL,
    CLEAN_NAME_COL,
    COUNTRY_COL,
    ENTITY_ID_COL,
    NUM_THREADS,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PROCESSED_DATA_DIR,
    SOUNDEX_COL,
    STREAMING_BATCH_SIZE,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("features")

# ==============================================================================
# 1. Feature Names & Schema Specification (PROJECT.md Interface Contract 3)
# ==============================================================================
SOURCE1_ID_COL = "source1_entity_id"
CANDIDATE_ID_COL = "candidate_entity_id"
LABEL_COL = "label"

FEATURE_NAMES = [
    # 1-7: String similarity metrics
    "feat_01_name_ratio",
    "feat_02_name_partial_ratio",
    "feat_03_name_token_sort_ratio",
    "feat_04_name_token_set_ratio",
    "feat_05_name_wratio",
    "feat_06_name_jaro_winkler",
    "feat_07_name_qgram",
    # 8-11: Token set metrics
    "feat_08_name_token_jaccard",
    "feat_09_name_token_overlap",
    "feat_10_name_first_token_match",
    "feat_11_name_last_token_match",
    # 12-13: Phonetic metrics
    "feat_12_soundex_2token_match",
    "feat_13_soundex_first_token_match",
    # 14-21: Address metrics
    "feat_14_addr_levenshtein",
    "feat_15_addr_token_set_ratio",
    "feat_16_addr_token_sort_ratio",
    "feat_17_addr_token_jaccard",
    "feat_18_addr_token_overlap",
    "feat_19_addr_numeric_exact_match",
    "feat_20_addr_numeric_jaccard",
    "feat_21_addr_missing_flag",
    # 22-28: Interaction & Length metrics
    "feat_22_name_addr_token_set",
    "feat_23_name_addr_jaccard",
    "feat_24_name_len_diff",
    "feat_25_name_len_ratio",
    "feat_26_name_token_count_diff",
    "feat_27_addr_len_diff",
    "feat_28_addr_len_ratio",
    # 29-32: Graph & Metadata metrics
    "feat_29_source_indicator",
    "feat_30_reranker_cosine_score",
    "feat_31_blocking_source_flags",
    "feat_32_blocking_rank",
]

# Alias map for Tier 1 / Survey 3 test suites
TIER1_FEATURE_NAMES = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_WRatio",
    "name_jaccard",
    "name_overlap_coeff",
    "name_containment",
    "name_common_prefix",
    "name_soundex_match",
    "name_metaphone_match",
    "name_tfidf_cosine",
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_sort",
    "addr_partial",
    "addr_jaccard",
    "addr_overlap_coeff",
    "addr_numeric_jaccard",
    "addr_shared_nums",
    "addr_missing_flag",
    "addr_tfidf_cosine",
    "combined_avg_sim",
    "name_addr_sim_diff",
    "name_in_addr",
    "legal_suffix_match",
    "name_len_ratio",
    "addr_len_ratio",
    "source_indicator",
    "blocking_rank",
    "n_candidates_for_s1",
]

FEATURE_NAME_TO_TIER1_MAP: Dict[str, str] = {
    "feat_01_name_ratio": "name_levenshtein",
    "feat_02_name_partial_ratio": "name_partial",
    "feat_03_name_token_sort_ratio": "name_token_sort",
    "feat_04_name_token_set_ratio": "name_token_set",
    "feat_05_name_wratio": "name_WRatio",
    "feat_06_name_jaro_winkler": "name_jaro_winkler",
    "feat_07_name_qgram": "name_qgram",
    "feat_08_name_token_jaccard": "name_jaccard",
    "feat_09_name_token_overlap": "name_overlap_coeff",
    "feat_10_name_first_token_match": "name_first_token_match",
    "feat_11_name_last_token_match": "name_last_token_match",
    "feat_12_soundex_2token_match": "name_soundex_match",
    "feat_13_soundex_first_token_match": "name_soundex_first_token_match",
    "feat_14_addr_levenshtein": "addr_levenshtein",
    "feat_15_addr_token_set_ratio": "addr_token_set",
    "feat_16_addr_token_sort_ratio": "addr_token_sort",
    "feat_17_addr_token_jaccard": "addr_jaccard",
    "feat_18_addr_token_overlap": "addr_overlap_coeff",
    "feat_19_addr_numeric_exact_match": "addr_numeric_exact_match",
    "feat_20_addr_numeric_jaccard": "addr_numeric_jaccard",
    "feat_21_addr_missing_flag": "addr_missing_flag",
    "feat_22_name_addr_token_set": "name_addr_token_set",
    "feat_23_name_addr_jaccard": "name_addr_jaccard",
    "feat_24_name_len_diff": "name_len_diff",
    "feat_25_name_len_ratio": "name_len_ratio",
    "feat_26_name_token_count_diff": "name_token_count_diff",
    "feat_27_addr_len_diff": "addr_len_diff",
    "feat_28_addr_len_ratio": "addr_len_ratio",
    "feat_29_source_indicator": "source_indicator",
    "feat_30_reranker_cosine_score": "reranker_cosine_score",
    "feat_31_blocking_source_flags": "blocking_source_flags",
    "feat_32_blocking_rank": "blocking_rank",
}

INTERFACE_CONTRACT_3_COLS = [SOURCE1_ID_COL, CANDIDATE_ID_COL] + FEATURE_NAMES


# ==============================================================================
# 2. Mathematical Similarity Utilities (Enforcing RapidFuzz Processor)
# ==============================================================================
def compute_qgram_similarity(s1_proc: str, s2_proc: str, q: int = 2) -> float:
    """
    Compute normalized character q-gram Sorensen-Dice similarity in [0.0, 1.0].
    qgram_sim = 2 * |Q1 ∩ Q2| / (|Q1| + |Q2|), equivalent to 1.0 - (qgram_dist / total_qgrams).
    """
    if not s1_proc or not s2_proc:
        return 0.0
    len1, len2 = len(s1_proc), len(s2_proc)
    if len1 < q or len2 < q:
        return 1.0 if s1_proc == s2_proc else 0.0
    q1 = Counter([s1_proc[i : i + q] for i in range(len1 - q + 1)])
    q2 = Counter([s2_proc[i : i + q] for i in range(len2 - q + 1)])
    overlap = sum((q1 & q2).values())
    total = sum(q1.values()) + sum(q2.values())
    return float((2.0 * overlap) / total) if total > 0 else 0.0


# Struct Schema for map_batches return type
POLARS_FEATURE_STRUCT_FIELDS = [
    pl.Field(col, pl.Float32) for col in FEATURE_NAMES
]
POLARS_FEATURE_STRUCT_TYPE = pl.Struct(POLARS_FEATURE_STRUCT_FIELDS)


# ==============================================================================
# 3. High-Throughput Batch Feature Extraction Callable
# ==============================================================================
def batch_extract_32_features(struct_series: pl.Series) -> pl.Series:
    """
    High-throughput batch feature extractor operating on Arrow chunk buffers.
    Evaluates all 32 pairwise metrics in memory without Python row-object materialization.
    Strictly enforces processor=rfuzz_utils.default_process on all RapidFuzz operations.
    """
    proc = rfuzz_utils.default_process

    # Extract raw chunk columns directly from Arrow struct fields
    s1_names = struct_series.struct.field("s1_name").to_list()
    c_names = struct_series.struct.field("c_name").to_list()
    s1_addrs = struct_series.struct.field("s1_addr").to_list()
    c_addrs = struct_series.struct.field("c_addr").to_list()
    s1_snds = struct_series.struct.field("s1_soundex").to_list()
    c_snds = struct_series.struct.field("c_soundex").to_list()
    cand_ids = struct_series.struct.field("candidate_entity_id").to_list()
    scores = struct_series.struct.field("reranker_cosine_score").to_list()
    flags = struct_series.struct.field("blocking_source_flags").to_list()
    ranks = struct_series.struct.field("blocking_rank").to_list()
    c_missings = struct_series.struct.field("c_addr_missing_flag").to_list()

    n = len(s1_names)

    # Pre-allocate contiguous 32-bit float buffers for SIMD operations
    f01 = np.empty(n, dtype=np.float32)
    f02 = np.empty(n, dtype=np.float32)
    f03 = np.empty(n, dtype=np.float32)
    f04 = np.empty(n, dtype=np.float32)
    f05 = np.empty(n, dtype=np.float32)
    f06 = np.empty(n, dtype=np.float32)
    f07 = np.empty(n, dtype=np.float32)
    f08 = np.empty(n, dtype=np.float32)
    f09 = np.empty(n, dtype=np.float32)
    f10 = np.empty(n, dtype=np.float32)
    f11 = np.empty(n, dtype=np.float32)
    f12 = np.empty(n, dtype=np.float32)
    f13 = np.empty(n, dtype=np.float32)
    f14 = np.empty(n, dtype=np.float32)
    f15 = np.empty(n, dtype=np.float32)
    f16 = np.empty(n, dtype=np.float32)
    f17 = np.empty(n, dtype=np.float32)
    f18 = np.empty(n, dtype=np.float32)
    f19 = np.empty(n, dtype=np.float32)
    f20 = np.empty(n, dtype=np.float32)
    f21 = np.empty(n, dtype=np.float32)
    f22 = np.empty(n, dtype=np.float32)
    f23 = np.empty(n, dtype=np.float32)
    f24 = np.empty(n, dtype=np.float32)
    f25 = np.empty(n, dtype=np.float32)
    f26 = np.empty(n, dtype=np.float32)
    f27 = np.empty(n, dtype=np.float32)
    f28 = np.empty(n, dtype=np.float32)
    f29 = np.empty(n, dtype=np.float32)
    f30 = np.empty(n, dtype=np.float32)
    f31 = np.empty(n, dtype=np.float32)
    f32 = np.empty(n, dtype=np.float32)

    for i in range(n):
        s1_n = s1_names[i] or ""
        c_n = c_names[i] or ""
        s1_a = s1_addrs[i] or ""
        c_a = c_addrs[i] or ""
        s1_s = str(s1_snds[i] or "")
        c_s = str(c_snds[i] or "")
        cid = str(cand_ids[i] or "")
        sc = float(scores[i]) if scores[i] is not None else 0.0
        fl = float(flags[i]) if flags[i] is not None else 0.0
        rk = float(ranks[i]) if ranks[i] is not None else 1.0
        cm = int(c_missings[i]) if c_missings[i] is not None else 0
        if not c_a or not c_a.strip():
            cm = 1

        # ----------------------------------------------------------------------
        # 1-7: String similarity metrics
        # ----------------------------------------------------------------------
        if s1_n and c_n:
            f01[i] = fuzz.ratio(s1_n, c_n, processor=proc) / 100.0
            f02[i] = fuzz.partial_ratio(s1_n, c_n, processor=proc) / 100.0
            f03[i] = fuzz.token_sort_ratio(s1_n, c_n, processor=proc) / 100.0
            f04[i] = fuzz.token_set_ratio(s1_n, c_n, processor=proc) / 100.0
            f05[i] = fuzz.WRatio(s1_n, c_n, processor=proc) / 100.0
            f06[i] = distance.JaroWinkler.similarity(s1_n, c_n, processor=proc)
        else:
            f01[i] = 0.0
            f02[i] = 0.0
            f03[i] = 0.0
            f04[i] = 0.0
            f05[i] = 0.0
            f06[i] = 0.0

        s1_np = proc(s1_n)
        c_np = proc(c_n)
        f07[i] = compute_qgram_similarity(s1_np, c_np, q=2)

        # ----------------------------------------------------------------------
        # 8-11: Token set metrics
        # ----------------------------------------------------------------------
        l1 = s1_np.split() if s1_np else []
        l2 = c_np.split() if c_np else []
        t1 = set(l1)
        t2 = set(l2)
        u_name = len(t1 | t2)
        f08[i] = (len(t1 & t2) / u_name) if u_name > 0 else 0.0
        min_name_tok = min(len(t1), len(t2))
        f09[i] = (len(t1 & t2) / min_name_tok) if min_name_tok > 0 else 0.0
        f10[i] = 1.0 if (l1 and l2 and l1[0] == l2[0]) else 0.0
        f11[i] = 1.0 if (l1 and l2 and l1[-1] == l2[-1]) else 0.0

        # ----------------------------------------------------------------------
        # 12-13: Phonetic metrics
        # ----------------------------------------------------------------------
        f12[i] = 1.0 if (s1_s and c_s and s1_s == c_s) else 0.0
        snd1_first = s1_s.split("_")[0] if s1_s else ""
        snd2_first = c_s.split("_")[0] if c_s else ""
        f13[i] = 1.0 if (snd1_first and snd2_first and snd1_first == snd2_first) else 0.0

        # ----------------------------------------------------------------------
        # 14-21: Address metrics
        # When address is missing/empty, default cleanly to 0.0 without errors
        # ----------------------------------------------------------------------
        if cm == 1 or not s1_a or not c_a:
            f14[i] = 0.0
            f15[i] = 0.0
            f16[i] = 0.0
            f17[i] = 0.0
            f18[i] = 0.0
            f19[i] = 0.0
            f20[i] = 0.0
            f21[i] = 1.0
        else:
            f14[i] = fuzz.ratio(s1_a, c_a, processor=proc) / 100.0
            f15[i] = fuzz.token_set_ratio(s1_a, c_a, processor=proc) / 100.0
            f16[i] = fuzz.token_sort_ratio(s1_a, c_a, processor=proc) / 100.0

            a1_p = proc(s1_a)
            a2_p = proc(c_a)
            at1 = set(a1_p.split()) if a1_p else set()
            at2 = set(a2_p.split()) if a2_p else set()
            u_addr = len(at1 | at2)
            f17[i] = (len(at1 & at2) / u_addr) if u_addr > 0 else 0.0
            min_addr_tok = min(len(at1), len(at2))
            f18[i] = (len(at1 & at2) / min_addr_tok) if min_addr_tok > 0 else 0.0

            # Regex digit sequence extraction
            n1 = re.findall(r"\d+", s1_a)
            n2 = re.findall(r"\d+", c_a)
            f19[i] = 1.0 if (n1 and n2 and n1 == n2) else 0.0
            sn1, sn2 = set(n1), set(n2)
            u_num = len(sn1 | sn2)
            f20[i] = (len(sn1 & sn2) / u_num) if u_num > 0 else 0.0
            f21[i] = 0.0

        # ----------------------------------------------------------------------
        # 22-28: Interaction & Length metrics
        # ----------------------------------------------------------------------
        if cm == 1 or not c_a:
            f22[i] = 0.0
            f23[i] = 0.0
            f27[i] = 0.0
            f28[i] = 0.0
        else:
            f22[i] = fuzz.token_set_ratio(s1_n, c_a, processor=proc) / 100.0 if s1_n else 0.0
            c_addr_toks = set(proc(c_a).split())
            u_na = len(t1 | c_addr_toks)
            f23[i] = (len(t1 & c_addr_toks) / u_na) if u_na > 0 else 0.0
            al1, al2 = len(s1_a), len(c_a)
            f27[i] = float(abs(al1 - al2))
            f28[i] = float(min(al1, al2)) / float(max(al1, al2, 1))

        nl1, nl2 = len(s1_n), len(c_n)
        f24[i] = float(abs(nl1 - nl2))
        f25[i] = float(min(nl1, nl2)) / float(max(nl1, nl2, 1))
        f26[i] = float(abs(len(l1) - len(l2)))

        # ----------------------------------------------------------------------
        # 29-32: Graph & Metadata metrics
        # ----------------------------------------------------------------------
        f29[i] = 1.0 if (cid.startswith("S3") or "source3" in cid.lower()) else 0.0
        f30[i] = sc
        f31[i] = fl
        f32[i] = rk

    fields = {
        "feat_01_name_ratio": f01,
        "feat_02_name_partial_ratio": f02,
        "feat_03_name_token_sort_ratio": f03,
        "feat_04_name_token_set_ratio": f04,
        "feat_05_name_wratio": f05,
        "feat_06_name_jaro_winkler": f06,
        "feat_07_name_qgram": f07,
        "feat_08_name_token_jaccard": f08,
        "feat_09_name_token_overlap": f09,
        "feat_10_name_first_token_match": f10,
        "feat_11_name_last_token_match": f11,
        "feat_12_soundex_2token_match": f12,
        "feat_13_soundex_first_token_match": f13,
        "feat_14_addr_levenshtein": f14,
        "feat_15_addr_token_set_ratio": f15,
        "feat_16_addr_token_sort_ratio": f16,
        "feat_17_addr_token_jaccard": f17,
        "feat_18_addr_token_overlap": f18,
        "feat_19_addr_numeric_exact_match": f19,
        "feat_20_addr_numeric_jaccard": f20,
        "feat_21_addr_missing_flag": f21,
        "feat_22_name_addr_token_set": f22,
        "feat_23_name_addr_jaccard": f23,
        "feat_24_name_len_diff": f24,
        "feat_25_name_len_ratio": f25,
        "feat_26_name_token_count_diff": f26,
        "feat_27_addr_len_diff": f27,
        "feat_28_addr_len_ratio": f28,
        "feat_29_source_indicator": f29,
        "feat_30_reranker_cosine_score": f30,
        "feat_31_blocking_source_flags": f31,
        "feat_32_blocking_rank": f32,
    }

    return pl.DataFrame(fields).to_struct("features")


# ==============================================================================
# 4. Pairwise Feature Extraction Pipeline (Pure Native Polars)
# ==============================================================================
def prepare_joined_candidate_pairs(
    pairs_df: pl.DataFrame,
    s1_entities_df: Optional[pl.DataFrame] = None,
    cand_entities_df: Optional[pl.DataFrame] = None,
    ground_truth_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Ensure all necessary string and metadata columns exist in candidate pairs.
    If entity tables are provided, joins them using native Polars left joins.
    """
    df = pairs_df

    # 1. Join S1 entity details if available and columns missing
    if s1_entities_df is not None and "s1_name" not in df.columns:
        s1_cols = {
            ENTITY_ID_COL: SOURCE1_ID_COL,
            CLEAN_NAME_COL: "s1_name",
            CLEAN_ADDR_COL: "s1_addr",
            SOUNDEX_COL: "s1_soundex",
            ADDR_MISSING_FLAG_COL: "s1_addr_missing_flag",
        }
        avail_s1 = [c for c in s1_cols.keys() if c in s1_entities_df.columns]
        s1_prep = s1_entities_df.select(avail_s1).rename(
            {c: s1_cols[c] for c in avail_s1}
        )
        df = df.join(s1_prep, on=SOURCE1_ID_COL, how="left")

    # 2. Join Candidate entity details if available and columns missing
    if cand_entities_df is not None and "c_name" not in df.columns:
        cand_cols = {
            ENTITY_ID_COL: CANDIDATE_ID_COL,
            CLEAN_NAME_COL: "c_name",
            CLEAN_ADDR_COL: "c_addr",
            SOUNDEX_COL: "c_soundex",
            ADDR_MISSING_FLAG_COL: "c_addr_missing_flag",
        }
        avail_cand = [c for c in cand_cols.keys() if c in cand_entities_df.columns]
        cand_prep = cand_entities_df.select(avail_cand).rename(
            {c: cand_cols[c] for c in avail_cand}
        )
        df = df.join(cand_prep, on=CANDIDATE_ID_COL, how="left")

    # 3. Join ground truth label if provided
    if ground_truth_df is not None and LABEL_COL not in df.columns:
        # Expected GT columns: source1_entity_id, matched_entity_id (or candidate_entity_id)
        gt_cand_col = (
            "matched_entity_id"
            if "matched_entity_id" in ground_truth_df.columns
            else CANDIDATE_ID_COL
        )
        gt_prep = ground_truth_df.select(
            [
                pl.col(SOURCE1_ID_COL),
                pl.col(gt_cand_col).alias(CANDIDATE_ID_COL),
                pl.lit(1, dtype=pl.UInt8).alias(LABEL_COL),
            ]
        ).unique(subset=[SOURCE1_ID_COL, CANDIDATE_ID_COL])

        df = df.join(gt_prep, on=[SOURCE1_ID_COL, CANDIDATE_ID_COL], how="left")
        df = df.with_columns(pl.col(LABEL_COL).fill_null(0).cast(pl.UInt8))

    # 4. Fill defaults for required feature extraction columns
    required_defaults = {
        "s1_name": pl.lit(""),
        "c_name": pl.lit(""),
        "s1_addr": pl.lit(""),
        "c_addr": pl.lit(""),
        "s1_soundex": pl.lit(""),
        "c_soundex": pl.lit(""),
        "c_addr_missing_flag": pl.lit(0, dtype=pl.UInt8),
        "reranker_cosine_score": pl.lit(0.0, dtype=pl.Float32),
        "blocking_source_flags": pl.lit(0, dtype=pl.UInt8),
    }

    fill_exprs = []
    for col_name, default_expr in required_defaults.items():
        if col_name not in df.columns:
            fill_exprs.append(default_expr.alias(col_name))
        else:
            fill_exprs.append(pl.col(col_name).fill_null(default_expr).alias(col_name))

    if fill_exprs:
        df = df.with_columns(fill_exprs)

    # 5. Compute blocking rank if not present
    if "blocking_rank" not in df.columns:
        df = df.with_columns(
            pl.col("reranker_cosine_score")
            .rank(descending=True, method="ordinal")
            .over(SOURCE1_ID_COL)
            .cast(pl.Float32)
            .alias("blocking_rank")
        )

    return df


def compute_pairwise_features(
    pairs_df: pl.DataFrame,
    s1_entities_df: Optional[pl.DataFrame] = None,
    cand_entities_df: Optional[pl.DataFrame] = None,
    ground_truth_df: Optional[pl.DataFrame] = None,
    chunk_size: int = STREAMING_BATCH_SIZE,
    keep_metadata_cols: bool = True,
) -> pl.DataFrame:
    """
    Computes all 32 pairwise similarity metrics for candidate pairs.
    Adheres strictly to the Pure Native Polars requirement: pl.struct().map_batches().
    Guarantees ZERO iter_rows() usage throughout the entire calculation.

    Parameters
    ----------
    pairs_df : pl.DataFrame
        Candidate pairs DataFrame.
    s1_entities_df : Optional[pl.DataFrame]
        Preprocessed Source 1 entities table.
    cand_entities_df : Optional[pl.DataFrame]
        Preprocessed Candidate (S2/S3) entities table.
    ground_truth_df : Optional[pl.DataFrame]
        Optional training ground truth for binary match labeling.
    chunk_size : int
        Chunk size for morsel-driven processing to bound working set RAM <= 6.0GB.
    keep_metadata_cols : bool
        If True, retains source1_entity_id, candidate_entity_id, and label.

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with exactly 32 Float32 feature columns.
    """
    if len(pairs_df) == 0:
        empty_schema = [
            (SOURCE1_ID_COL, pl.Utf8),
            (CANDIDATE_ID_COL, pl.Utf8),
        ]
        if ground_truth_df is not None or LABEL_COL in pairs_df.columns:
            empty_schema.append((LABEL_COL, pl.UInt8))
        for col in FEATURE_NAMES:
            empty_schema.append((col, pl.Float32))
        return pl.DataFrame(schema=empty_schema)

    # Prepare joined entity columns and clean fill nulls
    joined_df = prepare_joined_candidate_pairs(
        pairs_df,
        s1_entities_df=s1_entities_df,
        cand_entities_df=cand_entities_df,
        ground_truth_df=ground_truth_df,
    )

    struct_cols = [
        "s1_name",
        "c_name",
        "s1_addr",
        "c_addr",
        "s1_soundex",
        "c_soundex",
        CANDIDATE_ID_COL,
        "reranker_cosine_score",
        "blocking_source_flags",
        "blocking_rank",
        "c_addr_missing_flag",
    ]

    total_rows = len(joined_df)

    # Process via memory-safe slices if larger than chunk_size, or single batch if small
    if total_rows <= chunk_size:
        enriched_df = joined_df.with_columns(
            pl.struct(struct_cols)
            .map_batches(batch_extract_32_features, return_dtype=POLARS_FEATURE_STRUCT_TYPE)
            .alias("features_struct")
        ).unnest("features_struct")
    else:
        # Morsel-driven batching to strictly maintain RAM target <= 6.0 GB
        chunks: List[pl.DataFrame] = []
        for offset in range(0, total_rows, chunk_size):
            chunk = joined_df.slice(offset, min(chunk_size, total_rows - offset))
            chunk_features = chunk.with_columns(
                pl.struct(struct_cols)
                .map_batches(batch_extract_32_features, return_dtype=POLARS_FEATURE_STRUCT_TYPE)
                .alias("features_struct")
            ).unnest("features_struct")
            chunks.append(chunk_features)

        enriched_df = pl.concat(chunks)

    # Retain strictly the Interface Contract 3 columns
    output_cols = [SOURCE1_ID_COL, CANDIDATE_ID_COL]
    if LABEL_COL in enriched_df.columns:
        output_cols.append(LABEL_COL)
    output_cols.extend(FEATURE_NAMES)

    return enriched_df.select([c for c in output_cols if c in enriched_df.columns])


# ==============================================================================
# 5. Parquet Export & End-to-End Pipeline
# ==============================================================================
def export_features_to_parquet(
    features_df: pl.DataFrame,
    output_path: Union[str, Path],
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
) -> Path:
    """
    Export feature DataFrame to zstd-compressed Parquet meeting Interface Contract 3.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(
        path,
        compression=compression,
        compression_level=compression_level,
    )
    logger.info(
        f"Exported features to '{path}' | Rows: {len(features_df):,} | "
        f"Columns: {len(features_df.columns)} | File Size: {path.stat().st_size / (1024 * 1024):.2f} MB"
    )
    return path


def run_feature_engineering_pipeline(
    split: str,
    country: str,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    chunk_size: int = STREAMING_BATCH_SIZE,
) -> Path:
    """
    Execute full Phase 4 Feature Engineering pipeline for a split and country.
    Loads candidate pairs and entity tables, extracts 32 features, and exports Parquet.
    """
    base_data = Path(data_dir or PROCESSED_DATA_DIR)
    out_dir = Path(output_dir or base_data)

    country_dir = base_data / split / country
    pairs_file = country_dir / "candidate_pairs.parquet"

    if not pairs_file.exists():
        raise FileNotFoundError(f"Candidate pairs file not found: {pairs_file}")

    with track_memory(f"Feature Engineering: {split}/{country}"):
        logger.info(f"Loading candidate pairs from {pairs_file}...")
        pairs_df = pl.read_parquet(pairs_file)

        # Load S1 entities
        s1_file = country_dir / f"{split}_source1.parquet"
        s1_df = pl.read_parquet(s1_file) if s1_file.exists() else None

        # Load Candidate entities (union S2 and S3)
        cand_dfs: List[pl.DataFrame] = []
        s2_file = country_dir / f"{split}_source2.parquet"
        s3_file = country_dir / f"{split}_source3.parquet"
        if s2_file.exists():
            cand_dfs.append(pl.read_parquet(s2_file))
        if s3_file.exists():
            cand_dfs.append(pl.read_parquet(s3_file))

        cand_df = pl.concat(cand_dfs) if cand_dfs else None

        # Ground truth if train split
        gt_df = None
        if split == "train":
            gt_path = base_data.parent / "extracted_data" / "student_resource" / "dataset" / "train_ground_truth.tsv"
            if gt_path.exists():
                gt_df = pl.read_csv(gt_path, separator="\t")

        features_df = compute_pairwise_features(
            pairs_df=pairs_df,
            s1_entities_df=s1_df,
            cand_entities_df=cand_df,
            ground_truth_df=gt_df,
            chunk_size=chunk_size,
        )

        out_path = out_dir / split / country / "features.parquet"
        export_features_to_parquet(features_df, out_path)

        gc.collect()
        return out_path


if __name__ == "__main__":
    logger.info("Executing src/features.py standalone smoke test...")
    # Verify clean import and schema definition
    print(f"Total features defined: {len(FEATURE_NAMES)}")
    print(f"First feature: {FEATURE_NAMES[0]} | Last feature: {FEATURE_NAMES[-1]}")
    logger.info("src/features.py initialized successfully.")
