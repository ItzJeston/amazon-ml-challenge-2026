"""
src/blocking.py
Milestone 2 (Phase 3: Cascaded Blocking Engine)
Amazon Business Entity Resolution Pipeline.

Implements 4 complementary cascaded blocking strategies unioned with bitmask flags:
- Strategy A: Character 3-gram TF-IDF via SciPy CSR & sparse_dot_topn.sp_matmul_topn
- Strategy B: Polars exact token inverted index (tokens >= 3 chars, max_df <= 1500 cap)
- Strategy C: Word-level address TF-IDF (word 1-2 grams, threshold=0.20, non-empty addrs)
- Strategy D: Exact phonetic 2-token Soundex matching (pre-transliterated names)

Adheres strictly to hardware limits (Intel i7-13650HX 14 threads, peak RAM <= 5.2GB).
Enforces zero row iteration in all Polars logic.
"""

import gc
import io
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# Enforce explicit UTF-8 stdout initialization
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    ADDR_MISSING_FLAG_COL,
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
logger = get_logger("blocking")

# ==============================================================================
# 1. Blocking Strategy Bitmask Constants & Column Names
# ==============================================================================
BLOCKING_FLAG_CHAR_TFIDF = 1   # Bit 0 (1 << 0): Strategy A (Char N-gram TF-IDF)
BLOCKING_FLAG_TOKEN_INDEX = 2  # Bit 1 (1 << 1): Strategy B (Token Inverted Index)
BLOCKING_FLAG_ADDR_TFIDF = 4   # Bit 2 (1 << 2): Strategy C (Address TF-IDF)
BLOCKING_FLAG_SOUNDEX = 8      # Bit 3 (1 << 3): Strategy D (Phonetic Soundex)

SOURCE1_ID_COL = "source1_entity_id"
CANDIDATE_ID_COL = "candidate_entity_id"
RERANKER_SCORE_COL = "reranker_cosine_score"
BLOCKING_FLAGS_COL = "blocking_source_flags"

# Safe import of sparse_dot_topn
try:
    from sparse_dot_topn import sp_matmul_topn
    SPARSE_DOT_TOPN_AVAILABLE = True
except ImportError:
    SPARSE_DOT_TOPN_AVAILABLE = False


# ==============================================================================
# 2. Sparse Matrix Dot Product Utilities & Fallbacks
# ==============================================================================
def sp_matmul_topn_fallback(
    A: sp.csr_matrix,
    B_T: sp.csr_matrix,
    top_n: int = 50,
    threshold: float = 0.05,
) -> sp.csr_matrix:
    """
    Zero-dependency pure SciPy/NumPy chunked fallback for sparse_dot_topn.
    Evaluates sub-chunks of 5,000 rows to bound intermediate memory usage.
    Employs np.argpartition for O(K) top-N extraction.
    """
    M = A.shape[0]
    all_data: List[float] = []
    all_indices: List[int] = []
    indptr: List[int] = [0]

    sub_chunk_size = 5000
    for chunk_start in range(0, M, sub_chunk_size):
        chunk_end = min(chunk_start + sub_chunk_size, M)
        A_sub = A[chunk_start:chunk_end]
        dot = A_sub.dot(B_T)

        for i in range(dot.shape[0]):
            s, e = dot.indptr[i], dot.indptr[i + 1]
            row_nnz = e - s
            if row_nnz == 0:
                indptr.append(len(all_indices))
                continue

            cols = dot.indices[s:e]
            vals = dot.data[s:e]

            mask = vals >= threshold
            cols = cols[mask]
            vals = vals[mask]

            if len(vals) > top_n:
                part = np.argpartition(-vals, top_n)[:top_n]
                sort_idx = part[np.argsort(-vals[part])]
                cols = cols[sort_idx]
                vals = vals[sort_idx]
            elif len(vals) > 0:
                sort_idx = np.argsort(-vals)
                cols = cols[sort_idx]
                vals = vals[sort_idx]

            all_data.extend(vals)
            all_indices.extend(cols)
            indptr.append(len(all_indices))

    return sp.csr_matrix(
        (
            np.array(all_data, dtype=np.float32),
            np.array(all_indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
        ),
        shape=(M, B_T.shape[1]),
    )


def run_sp_matmul_topn(
    A: sp.csr_matrix,
    B_T: sp.csr_matrix,
    top_n: int = 50,
    threshold: float = 0.05,
    n_threads: int = NUM_THREADS,
) -> sp.csr_matrix:
    """
    Execute top-N matrix multiplication:
    Uses fused C++ OpenMP sparse_dot_topn if available; falls back to SciPy argpartition.
    Guarantees sorted output by descending cosine similarity.
    """
    if SPARSE_DOT_TOPN_AVAILABLE:
        try:
            return sp_matmul_topn(
                A,
                B_T,
                top_n=top_n,
                threshold=threshold,
                sort=True,
                n_threads=n_threads,
            )
        except Exception as e:
            logger.warning(f"sparse_dot_topn failed with: {e}. Using pure SciPy fallback.")
            return sp_matmul_topn_fallback(A, B_T, top_n=top_n, threshold=threshold)
    else:
        return sp_matmul_topn_fallback(A, B_T, top_n=top_n, threshold=threshold)


def _csr_results_to_polars(
    sim_csr: sp.csr_matrix,
    s1_ids: np.ndarray,
    s23_ids: np.ndarray,
    score_col_name: str,
    flag_col_name: str,
    flag_val: int,
) -> pl.DataFrame:
    """
    Vectorized extraction of non-zero entries from a CSR matrix into a Polars DataFrame.
    Zero Python row loops or row-by-row iteration.
    """
    if sim_csr.nnz == 0:
        return pl.DataFrame(
            schema={
                SOURCE1_ID_COL: pl.String,
                CANDIDATE_ID_COL: pl.String,
                score_col_name: pl.Float32,
                flag_col_name: pl.UInt8,
            }
        )

    nnz_per_row = np.diff(sim_csr.indptr)
    row_indices = np.repeat(np.arange(sim_csr.shape[0]), nnz_per_row)
    col_indices = sim_csr.indices
    scores = sim_csr.data

    matched_s1 = s1_ids[row_indices]
    matched_cand = s23_ids[col_indices]

    df = pl.DataFrame(
        {
            SOURCE1_ID_COL: matched_s1,
            CANDIDATE_ID_COL: matched_cand,
            score_col_name: scores.astype(np.float32),
            flag_col_name: np.full(len(scores), flag_val, dtype=np.uint8),
        }
    )
    # Guard against accidental S1 self-matching
    return df.filter(pl.col(CANDIDATE_ID_COL).str.starts_with("S1-").not_())


# ==============================================================================
# 3. Strategy A: Character N-Gram TF-IDF via CSR & sp_matmul_topn
# ==============================================================================
def blocking_strategy_a_char_tfidf(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    top_n: int = 50,
    threshold: float = 0.05,
    n_threads: int = NUM_THREADS,
    vectorizer: Optional[TfidfVectorizer] = None,
    s23_csr_T: Optional[sp.csr_matrix] = None,
) -> Tuple[pl.DataFrame, TfidfVectorizer, sp.csr_matrix]:
    """
    Strategy A: Character 3-5 gram TF-IDF via SciPy CSR and sp_matmul_topn.
    Parameters:
    - analyzer='char_wb', ngram_range=(3, 5)
    - sublinear_tf=True (logarithmic term frequency damping)
    - dtype=np.float32 (reduces RAM by 50%)
    - max_features=300,000
    - Bounded top-N max-heap retrieval with sort=True
    """
    if len(s1_df) == 0 or len(s23_df) == 0:
        empty_df = pl.DataFrame(
            schema={
                SOURCE1_ID_COL: pl.String,
                CANDIDATE_ID_COL: pl.String,
                "name_tfidf_cosine": pl.Float32,
                "flag_a": pl.UInt8,
            }
        )
        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                dtype=np.float32,
                max_features=300_000,
            )
        if s23_csr_T is None:
            s23_csr_T = sp.csr_matrix((0, 0), dtype=np.float32)
        return empty_df, vectorizer, s23_csr_T

    s1_names = s1_df[CLEAN_NAME_COL].fill_null("").to_list()
    s23_names = s23_df[CLEAN_NAME_COL].fill_null("").to_list()
    s1_ids = np.array(s1_df[ENTITY_ID_COL].to_list())
    s23_ids = np.array(s23_df[ENTITY_ID_COL].to_list())

    # Fit vectorizer if not provided
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            dtype=np.float32,
            max_features=300_000,
        )
        vectorizer.fit(s1_names + s23_names)

    # Compute S23 CSR transpose if not provided
    if s23_csr_T is None:
        s23_csr = vectorizer.transform(s23_names)
        s23_csr_T = s23_csr.T.tocsr()
        del s23_csr
        gc.collect()

    # Transform S1 batch
    s1_csr = vectorizer.transform(s1_names)

    # Execute fused top-N matrix multiplication
    sim_csr = run_sp_matmul_topn(
        s1_csr,
        s23_csr_T,
        top_n=top_n,
        threshold=threshold,
        n_threads=n_threads,
    )
    del s1_csr
    gc.collect()

    candidates_df = _csr_results_to_polars(
        sim_csr=sim_csr,
        s1_ids=s1_ids,
        s23_ids=s23_ids,
        score_col_name="name_tfidf_cosine",
        flag_col_name="flag_a",
        flag_val=BLOCKING_FLAG_CHAR_TFIDF,
    )
    del sim_csr
    gc.collect()

    return candidates_df, vectorizer, s23_csr_T


# ==============================================================================
# 4. Strategy B: Polars Exact Token Inverted Index
# ==============================================================================
def blocking_strategy_b_token_index(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    min_token_len: int = 3,
    max_df: int = 1500,
    top_k: int = 15,
) -> pl.DataFrame:
    """
    Strategy B: Polars exact token inverted index.
    - Tokenizes business_name_clean on whitespace.
    - Filters tokens >= min_token_len characters (>= 3 chars).
    - Caps document frequency at max_df <= 1500 to prevent Cartesian explosion on stopwords.
    - Joins on rare tokens, groups by (S1, Candidate), counts shared tokens, and truncates to top_k.
    - Zero row-by-row iteration used.
    """
    empty_df = pl.DataFrame(
        schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "flag_b": pl.UInt8,
        }
    )

    if len(s1_df) == 0 or len(s23_df) == 0:
        return empty_df

    # 1. Explode tokens for S1
    s1_tokens = (
        s1_df.select([pl.col(ENTITY_ID_COL).alias(SOURCE1_ID_COL), CLEAN_NAME_COL])
        .with_columns(pl.col(CLEAN_NAME_COL).fill_null("").str.split(" ").alias("token"))
        .explode("token")
        .filter(pl.col("token").str.len_chars() >= min_token_len)
        .select([SOURCE1_ID_COL, "token"])
    )

    # 2. Explode tokens for S23
    s23_tokens = (
        s23_df.select([pl.col(ENTITY_ID_COL).alias(CANDIDATE_ID_COL), CLEAN_NAME_COL])
        .with_columns(pl.col(CLEAN_NAME_COL).fill_null("").str.split(" ").alias("token"))
        .explode("token")
        .filter(pl.col("token").str.len_chars() >= min_token_len)
        .select([CANDIDATE_ID_COL, "token"])
    )

    if len(s1_tokens) == 0 or len(s23_tokens) == 0:
        return empty_df

    # 3. Document frequency capping: drop common tokens occurring > max_df times
    freq_s23 = (
        s23_tokens.group_by("token")
        .len()
        .filter(pl.col("len") <= max_df)
        .select("token")
    )

    s1_rare = s1_tokens.join(freq_s23, on="token", how="inner")
    s23_rare = s23_tokens.join(freq_s23, on="token", how="inner")

    if len(s1_rare) == 0 or len(s23_rare) == 0:
        return empty_df

    # 4. Relational join on rare tokens and count shared tokens
    matches = (
        s1_rare.join(s23_rare, on="token", how="inner")
        .filter(pl.col(CANDIDATE_ID_COL).str.starts_with("S1-").not_())
        .group_by([SOURCE1_ID_COL, CANDIDATE_ID_COL])
        .len()
        .rename({"len": "shared_tokens"})
    )

    if len(matches) == 0:
        return empty_df

    # 5. Rank by shared tokens and prune to top_k per S1
    top_matches = (
        matches.sort([SOURCE1_ID_COL, "shared_tokens"], descending=[False, True])
        .group_by(SOURCE1_ID_COL, maintain_order=True)
        .head(top_k)
        .select([
            SOURCE1_ID_COL,
            CANDIDATE_ID_COL,
            pl.lit(BLOCKING_FLAG_TOKEN_INDEX, dtype=pl.UInt8).alias("flag_b"),
        ])
    )

    return top_matches


# ==============================================================================
# 5. Strategy C: Word-Level Address TF-IDF
# ==============================================================================
def blocking_strategy_c_addr_tfidf(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    top_n: int = 20,
    threshold: float = 0.20,
    n_threads: int = NUM_THREADS,
    vectorizer: Optional[TfidfVectorizer] = None,
    s23_addr_csr_T: Optional[sp.csr_matrix] = None,
    s23_addr_ids: Optional[np.ndarray] = None,
) -> Tuple[pl.DataFrame, Optional[TfidfVectorizer], Optional[sp.csr_matrix], Optional[np.ndarray]]:
    """
    Strategy C: Word-level address TF-IDF.
    - Evaluated ONLY on non-empty addresses (handling S2/S3 3.3% missing address).
    - Parameters: analyzer='word', ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32
    - Threshold=0.20 (higher threshold to avoid generic city false associations).
    - Top-N=20 per non-empty S1 entity.
    """
    empty_df = pl.DataFrame(
        schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "addr_tfidf_cosine": pl.Float32,
            "flag_c": pl.UInt8,
        }
    )

    # Filter non-empty addresses
    s1_valid = s1_df.filter(
        pl.col(CLEAN_ADDR_COL).fill_null("").str.strip_chars().str.len_chars() > 0
    )
    s23_valid = s23_df.filter(
        pl.col(CLEAN_ADDR_COL).fill_null("").str.strip_chars().str.len_chars() > 0
    )

    if len(s1_valid) == 0 or len(s23_valid) == 0:
        return empty_df, vectorizer, s23_addr_csr_T, s23_addr_ids

    s1_addrs = s1_valid[CLEAN_ADDR_COL].to_list()
    s23_addrs = s23_valid[CLEAN_ADDR_COL].to_list()
    s1_ids = np.array(s1_valid[ENTITY_ID_COL].to_list())

    if s23_addr_ids is None:
        s23_addr_ids = np.array(s23_valid[ENTITY_ID_COL].to_list())

    # Fit vectorizer if not provided
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
            max_features=200_000,
        )
        vectorizer.fit(s1_addrs + s23_addrs)

    # Pre-compute S23 transpose CSR if not provided
    if s23_addr_csr_T is None:
        s23_addr_csr = vectorizer.transform(s23_addrs)
        s23_addr_csr_T = s23_addr_csr.T.tocsr()
        del s23_addr_csr
        gc.collect()

    # Transform S1 valid addresses
    s1_addr_csr = vectorizer.transform(s1_addrs)

    sim_csr = run_sp_matmul_topn(
        s1_addr_csr,
        s23_addr_csr_T,
        top_n=top_n,
        threshold=threshold,
        n_threads=n_threads,
    )
    del s1_addr_csr
    gc.collect()

    candidates_df = _csr_results_to_polars(
        sim_csr=sim_csr,
        s1_ids=s1_ids,
        s23_ids=s23_addr_ids,
        score_col_name="addr_tfidf_cosine",
        flag_col_name="flag_c",
        flag_val=BLOCKING_FLAG_ADDR_TFIDF,
    )
    del sim_csr
    gc.collect()

    return candidates_df, vectorizer, s23_addr_csr_T, s23_addr_ids


# ==============================================================================
# 6. Strategy D: Exact Phonetic 2-Token Soundex Matching
# ==============================================================================
def blocking_strategy_d_soundex(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    max_block_size: int = 2000,
    top_k: int = 15,
) -> pl.DataFrame:
    """
    Strategy D: Exact phonetic 2-token Soundex matching on pre-transliterated names.
    - Evaluated on non-empty soundex_2token values.
    - Caps block membership in S23 at max_block_size=2000 to prevent Cartesian explosions.
    - Prunes to top_k per S1 entity.
    - Zero row-by-row iteration.
    """
    empty_df = pl.DataFrame(
        schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "flag_d": pl.UInt8,
        }
    )

    if len(s1_df) == 0 or len(s23_df) == 0:
        return empty_df

    # Filter non-empty Soundex
    s1_valid = (
        s1_df.filter(pl.col(SOUNDEX_COL).fill_null("").str.strip_chars().str.len_chars() > 0)
        .select([pl.col(ENTITY_ID_COL).alias(SOURCE1_ID_COL), SOUNDEX_COL])
    )
    s23_valid = (
        s23_df.filter(pl.col(SOUNDEX_COL).fill_null("").str.strip_chars().str.len_chars() > 0)
        .select([pl.col(ENTITY_ID_COL).alias(CANDIDATE_ID_COL), SOUNDEX_COL])
    )

    if len(s1_valid) == 0 or len(s23_valid) == 0:
        return empty_df

    # Block safety: truncate huge blocks in S23 to at most max_block_size
    s23_capped = (
        s23_valid.group_by(SOUNDEX_COL, maintain_order=True)
        .head(max_block_size)
    )

    # Inner join on soundex_2token
    matches = (
        s1_valid.join(s23_capped, on=SOUNDEX_COL, how="inner")
        .filter(pl.col(CANDIDATE_ID_COL).str.starts_with("S1-").not_())
        .select([SOURCE1_ID_COL, CANDIDATE_ID_COL])
        .group_by(SOURCE1_ID_COL, maintain_order=True)
        .head(top_k)
        .with_columns(pl.lit(BLOCKING_FLAG_SOUNDEX, dtype=pl.UInt8).alias("flag_d"))
    )

    return matches


# ==============================================================================
# 7. Cascaded Union & Bitmask Deduplication
# ==============================================================================
def union_candidate_sources(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    df_c: pl.DataFrame,
    df_d: pl.DataFrame,
) -> pl.DataFrame:
    """
    Union candidates across all 4 strategies:
    - Deduplicates candidate pairs (source1_entity_id, candidate_entity_id).
    - Computes bitmask blocking_source_flags (1=A, 2=B, 4=C, 8=D).
    - Preserves name_tfidf_cosine and addr_tfidf_cosine scores (filling 0.0 if not computed).
    - Pure Polars relational logic, zero row-by-row iteration.
    """
    empty_result = pl.DataFrame(
        schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "name_tfidf_cosine": pl.Float32,
            "addr_tfidf_cosine": pl.Float32,
            BLOCKING_FLAGS_COL: pl.UInt8,
        }
    )

    dfs = [df for df in [df_a, df_b, df_c, df_d] if len(df) > 0]
    if not dfs:
        return empty_result

    # Diagonal concatenation handles varying columns across strategies
    all_candidates = pl.concat(dfs, how="diagonal")

    # Ensure required columns exist
    if "name_tfidf_cosine" not in all_candidates.columns:
        all_candidates = all_candidates.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("name_tfidf_cosine"))
    if "addr_tfidf_cosine" not in all_candidates.columns:
        all_candidates = all_candidates.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("addr_tfidf_cosine"))
    for flag_col in ["flag_a", "flag_b", "flag_c", "flag_d"]:
        if flag_col not in all_candidates.columns:
            all_candidates = all_candidates.with_columns(pl.lit(0, dtype=pl.UInt8).alias(flag_col))

    # Aggregate by candidate pair
    unioned = (
        all_candidates
        .group_by([SOURCE1_ID_COL, CANDIDATE_ID_COL])
        .agg([
            pl.col("name_tfidf_cosine").max().fill_null(0.0).cast(pl.Float32),
            pl.col("addr_tfidf_cosine").max().fill_null(0.0).cast(pl.Float32),
            (
                pl.col("flag_a").fill_null(0).max()
                | pl.col("flag_b").fill_null(0).max()
                | pl.col("flag_c").fill_null(0).max()
                | pl.col("flag_d").fill_null(0).max()
            ).cast(pl.UInt8).alias(BLOCKING_FLAGS_COL),
        ])
    )

    return unioned


# ==============================================================================
# 8. High-Level Orchestrator: run_cascaded_blocking
# ==============================================================================
def run_cascaded_blocking(
    s1_df: pl.DataFrame,
    s23_df: pl.DataFrame,
    top_n_a: int = 50,
    threshold_a: float = 0.05,
    top_k_b: int = 15,
    max_df_b: int = 1500,
    top_n_c: int = 20,
    threshold_c: float = 0.20,
    top_k_d: int = 15,
    max_block_size_d: int = 2000,
    n_threads: int = NUM_THREADS,
) -> pl.DataFrame:
    """
    Run the end-to-end 4-strategy cascaded blocking on S1 vs S23 DataFrames.
    Returns unioned candidate pairs with TF-IDF cosine scores and bitmask flags.
    """
    logger.info(f"Running Cascaded Blocking: {len(s1_df):,} S1 vs {len(s23_df):,} S2/S3 entities...")

    # Strategy A: Char TF-IDF
    df_a, _, _ = blocking_strategy_a_char_tfidf(
        s1_df, s23_df, top_n=top_n_a, threshold=threshold_a, n_threads=n_threads
    )
    logger.info(f"Strategy A (Char TF-IDF) retrieved {len(df_a):,} pairs.")

    # Strategy B: Token Inverted Index
    df_b = blocking_strategy_b_token_index(
        s1_df, s23_df, max_df=max_df_b, top_k=top_k_b
    )
    logger.info(f"Strategy B (Token Inverted Index) retrieved {len(df_b):,} pairs.")

    # Strategy C: Word Address TF-IDF
    df_c, _, _, _ = blocking_strategy_c_addr_tfidf(
        s1_df, s23_df, top_n=top_n_c, threshold=threshold_c, n_threads=n_threads
    )
    logger.info(f"Strategy C (Address TF-IDF) retrieved {len(df_c):,} pairs.")

    # Strategy D: Phonetic Soundex
    df_d = blocking_strategy_d_soundex(
        s1_df, s23_df, max_block_size=max_block_size_d, top_k=top_k_d
    )
    logger.info(f"Strategy D (Phonetic Soundex) retrieved {len(df_d):,} pairs.")

    # Union all 4 candidate sets
    unioned = union_candidate_sources(df_a, df_b, df_c, df_d)
    logger.info(f"Cascaded Blocking complete: {len(unioned):,} unique candidate pairs unioned.")

    return unioned
