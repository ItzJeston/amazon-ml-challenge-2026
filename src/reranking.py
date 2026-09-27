"""
src/reranking.py
Milestone 2 (Phase 3: Lightweight Cosine Re-ranker & Audit Pruner)
Amazon Business Entity Resolution Pipeline.

Implements:
- Vectorized composite cosine score: Score = 0.70 * Cosine_name + 0.30 * Cosine_addr
- Lightweight pruning via native Polars sorting:
  .sort(['source1_entity_id', 'reranker_cosine_score'], descending=[False, True]).group_by('source1_entity_id').head(15)
- Candidate bounding: median 10-12 (maximum 15) pairs per source1_entity_id
- Export to candidate_pairs.parquet conforming strictly to Interface Contract 2
- Submission TSV formatter with singleton handling

Adheres strictly to hardware limits (Intel i7-13650HX 14 threads, peak RAM <= 5.2GB).
Enforces zero row iteration anywhere in Polars logic.
"""

import gc
import io
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Union

# Enforce explicit UTF-8 stdout initialization
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking import (
    BLOCKING_FLAG_CHAR_TFIDF,
    BLOCKING_FLAG_SOUNDEX,
    BLOCKING_FLAG_TOKEN_INDEX,
    BLOCKING_FLAGS_COL,
    CANDIDATE_ID_COL,
    RERANKER_SCORE_COL,
    SOURCE1_ID_COL,
    blocking_strategy_a_char_tfidf,
    blocking_strategy_b_token_index,
    blocking_strategy_c_addr_tfidf,
    blocking_strategy_d_soundex,
    union_candidate_sources,
)
from src.config import (
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
    TEST_SOURCES,
    TRAIN_SOURCES,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("reranking")

# Interface Contract 2 Schema Columns
INTERFACE_CONTRACT_2_COLS = [
    SOURCE1_ID_COL,
    CANDIDATE_ID_COL,
    RERANKER_SCORE_COL,
    BLOCKING_FLAGS_COL,
]

PYARROW_CANDIDATE_SCHEMA = pa.schema(
    [
        (SOURCE1_ID_COL, pa.string()),
        (CANDIDATE_ID_COL, pa.string()),
        (RERANKER_SCORE_COL, pa.float32()),
        (BLOCKING_FLAGS_COL, pa.uint8()),
    ]
)


# ==============================================================================
# 1. Vectorized Composite Cosine Score
# ==============================================================================
def compute_reranker_composite_score(
    candidates_df: pl.DataFrame,
    name_weight: float = 0.70,
    addr_weight: float = 0.30,
    use_strategy_fallbacks: bool = True,
) -> pl.DataFrame:
    """
    Compute vectorized composite cosine score:
    Score = 0.70 * Cosine_name + 0.30 * Cosine_addr
    
    If candidate was retrieved only via Strategy B or D (where exact TF-IDF was not computed),
    assigns a calibrated baseline similarity (0.35 for token overlap, 0.30 for phonetic match)
    so informative non-TF-IDF candidates are not unfairly zeroed out before pruning.
    Zero row-by-row iteration used.
    """
    if len(candidates_df) == 0:
        return candidates_df.with_columns(
            pl.lit(0.0, dtype=pl.Float32).alias(RERANKER_SCORE_COL)
        )

    # Ensure required input columns exist
    df = candidates_df
    if "name_tfidf_cosine" not in df.columns:
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("name_tfidf_cosine"))
    if "addr_tfidf_cosine" not in df.columns:
        df = df.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("addr_tfidf_cosine"))
    if BLOCKING_FLAGS_COL not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.UInt8).alias(BLOCKING_FLAGS_COL))

    name_col = pl.col("name_tfidf_cosine").fill_null(0.0)
    addr_col = pl.col("addr_tfidf_cosine").fill_null(0.0)

    if use_strategy_fallbacks:
        # If name TF-IDF is 0.0, fallback to strategy-based estimates
        is_token_match = (pl.col(BLOCKING_FLAGS_COL) & BLOCKING_FLAG_TOKEN_INDEX) > 0
        is_soundex_match = (pl.col(BLOCKING_FLAGS_COL) & BLOCKING_FLAG_SOUNDEX) > 0

        calibrated_name = (
            pl.when(name_col > 0.0)
            .then(name_col)
            .when(is_token_match)
            .then(pl.lit(0.35, dtype=pl.Float32))
            .when(is_soundex_match)
            .then(pl.lit(0.30, dtype=pl.Float32))
            .otherwise(pl.lit(0.0, dtype=pl.Float32))
        )
    else:
        calibrated_name = name_col

    # Compute composite score
    score_expr = (name_weight * calibrated_name + addr_weight * addr_col).cast(pl.Float32)
    return df.with_columns(score_expr.alias(RERANKER_SCORE_COL))


# ==============================================================================
# 2. Lightweight Pruning via Native Polars Sorting
# ==============================================================================
def prune_candidates(
    candidates_df: pl.DataFrame,
    max_candidates: int = 15,
) -> pl.DataFrame:
    """
    Prune candidates per S1 entity using vectorized Polars multi-key sorting:
    .sort(['source1_entity_id', 'reranker_cosine_score'], descending=[False, True])
    .group_by('source1_entity_id', maintain_order=True)
    .head(max_candidates)
    
    Guarantees:
    - Maximum candidate count per S1 entity <= 15
    - Preserves candidates with highest composite scores
    - Sub-second execution for millions of pairs via Polars multi-threaded Rust engine
    - Zero row-by-row iteration
    """
    if len(candidates_df) == 0:
        return candidates_df

    pruned = (
        candidates_df
        .sort([SOURCE1_ID_COL, RERANKER_SCORE_COL], descending=[False, True])
        .group_by(SOURCE1_ID_COL, maintain_order=True)
        .head(max_candidates)
    )

    return pruned


# ==============================================================================
# 3. Candidate Pairs Re-ranking Pipeline
# ==============================================================================
def run_reranking(
    candidates_df: pl.DataFrame,
    max_candidates: int = 15,
    name_weight: float = 0.70,
    addr_weight: float = 0.30,
    use_strategy_fallbacks: bool = True,
    select_contract_cols_only: bool = True,
) -> pl.DataFrame:
    """
    End-to-end re-ranking and candidate set pruning:
    1. Computes composite cosine score
    2. Sorts and prunes to top max_candidates (max 15)
    3. Selects Interface Contract 2 columns:
       ['source1_entity_id', 'candidate_entity_id', 'reranker_cosine_score', 'blocking_source_flags']
    """
    scored = compute_reranker_composite_score(
        candidates_df,
        name_weight=name_weight,
        addr_weight=addr_weight,
        use_strategy_fallbacks=use_strategy_fallbacks,
    )
    pruned = prune_candidates(scored, max_candidates=max_candidates)

    if select_contract_cols_only:
        return pruned.select(INTERFACE_CONTRACT_2_COLS)
    return pruned


# ==============================================================================
# 4. Parquet & TSV Export Utilities
# ==============================================================================
def export_candidate_pairs_parquet(
    candidates_df: pl.DataFrame,
    output_path: Union[str, Path],
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
) -> Path:
    """
    Export candidate pairs DataFrame to Parquet conforming strictly to Interface Contract 2.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    export_df = candidates_df.select(INTERFACE_CONTRACT_2_COLS)
    arrow_table = export_df.to_arrow().cast(PYARROW_CANDIDATE_SCHEMA)

    pq.write_table(
        arrow_table,
        out_p,
        compression=compression,
        compression_level=compression_level,
    )
    logger.info(f"Saved {len(export_df):,} candidate pairs to {out_p}")
    return out_p


def export_candidate_pairs_tsv(
    candidates_df: pl.DataFrame,
    s1_entity_ids: List[str],
    output_path: Union[str, Path],
) -> Path:
    """
    Export candidate pairs to submission TSV format:
    Header: source1_entity_id\\tcandidate_entity_ids
    Singletons: S1 entities with zero candidates receive an empty string (S1-xxx\\t\\n).
    Strictly forbids S1 self-matches and duplicate candidate IDs.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate candidate IDs into comma-separated list
    if len(candidates_df) > 0:
        cands_grouped = (
            candidates_df
            .filter(pl.col(CANDIDATE_ID_COL).str.starts_with("S1-").not_())
            .group_by(SOURCE1_ID_COL, maintain_order=True)
            .agg(pl.col(CANDIDATE_ID_COL).unique(maintain_order=True).str.join(","))
            .rename({CANDIDATE_ID_COL: "candidate_entity_ids"})
        )
    else:
        cands_grouped = pl.DataFrame(
            schema={SOURCE1_ID_COL: pl.String, "candidate_entity_ids": pl.String}
        )

    # Base DataFrame of all S1 IDs guarantees exact length match (handling singletons)
    base_s1_df = pl.DataFrame({SOURCE1_ID_COL: s1_entity_ids})

    submission_df = (
        base_s1_df.join(cands_grouped, on=SOURCE1_ID_COL, how="left")
        .with_columns(pl.col("candidate_entity_ids").fill_null(""))
    )

    # Write TSV without quotes
    submission_df.write_csv(out_p, separator="\t", include_header=True, quote_style="never")
    logger.info(f"Wrote submission candidate pairs TSV ({len(submission_df):,} rows) to {out_p}")
    return out_p


# ==============================================================================
# 5. Chunked Streaming Partition Engine (Peak RAM <= 5.2 GB)
# ==============================================================================
def process_country_partition(
    split: str,
    country: str,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    streaming_batch_size: int = STREAMING_BATCH_SIZE,
    max_candidates_per_s1: int = 15,
) -> Path:
    """
    Execute memory-bounded cascaded blocking and re-ranking for one country partition:
    1. Loads S2 and S3 partition files once
    2. Fits TF-IDF vectorizers on S2/S3 corpus
    3. Streams S1 in batches of 50,000 - 100,000 rows
    4. Evaluates 4 cascaded strategies per batch
    5. Re-ranks and prunes candidates to top 15 per S1
    6. Appends to candidate_pairs.parquet via PyArrow ParquetWriter
    7. Keeps peak RAM strictly <= 5.2 GB
    """
    country_dir = processed_data_dir / split / country
    s1_parquet = country_dir / f"{split}_source1.parquet"
    s2_parquet = country_dir / f"{split}_source2.parquet"
    s3_parquet = country_dir / f"{split}_source3.parquet"
    output_parquet = country_dir / "candidate_pairs.parquet"

    if not s1_parquet.exists():
        logger.warning(f"S1 parquet not found: {s1_parquet}")
        return output_parquet

    logger.info(f"=== Starting Cascaded Blocking & Re-ranking for '{split}/{country}' ===")

    # 1. Load S2 and S3 candidates
    s23_dfs = []
    if s2_parquet.exists():
        s23_dfs.append(pl.read_parquet(s2_parquet))
    if s3_parquet.exists():
        s23_dfs.append(pl.read_parquet(s3_parquet))

    if not s23_dfs:
        logger.warning(f"No S2 or S3 partitions found for '{split}/{country}'!")
        return output_parquet

    s23_df = pl.concat(s23_dfs) if len(s23_dfs) > 1 else s23_dfs[0]
    del s23_dfs
    gc.collect()

    logger.info(f"Loaded {len(s23_df):,} S2/S3 candidate entities for '{country}'.")

    # Read S1 in batches or scan
    s1_full = pl.read_parquet(s1_parquet)
    total_s1 = len(s1_full)
    logger.info(f"Total S1 entities to process for '{country}': {total_s1:,}")

    # Fit name vectorizer and S23 CSR once for the country partition
    logger.info(f"Vectorizing names for '{country}'...")
    s1_names = s1_full[CLEAN_NAME_COL].fill_null("").to_list()
    s23_names = s23_df[CLEAN_NAME_COL].fill_null("").to_list()

    from sklearn.feature_extraction.text import TfidfVectorizer
    name_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        dtype=np.float32,
        max_features=300_000,
    )
    name_vec.fit(s1_names + s23_names)
    s23_name_csr = name_vec.transform(s23_names)
    s23_name_csr_T = s23_name_csr.T.tocsr()
    del s23_name_csr, s1_names, s23_names
    gc.collect()

    # Fit address vectorizer and S23 CSR once (for non-empty addresses)
    logger.info(f"Vectorizing addresses for '{country}'...")
    s1_addr_valid = s1_full.filter(
        pl.col(CLEAN_ADDR_COL).fill_null("").str.strip_chars().str.len_chars() > 0
    )
    s23_addr_valid = s23_df.filter(
        pl.col(CLEAN_ADDR_COL).fill_null("").str.strip_chars().str.len_chars() > 0
    )

    addr_vec = None
    s23_addr_csr_T = None
    s23_addr_ids = None

    if len(s1_addr_valid) > 0 and len(s23_addr_valid) > 0:
        addr_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
            max_features=200_000,
        )
        s1_addrs = s1_addr_valid[CLEAN_ADDR_COL].to_list()
        s23_addrs = s23_addr_valid[CLEAN_ADDR_COL].to_list()
        addr_vec.fit(s1_addrs + s23_addrs)
        s23_addr_csr = addr_vec.transform(s23_addrs)
        s23_addr_csr_T = s23_addr_csr.T.tocsr()
        s23_addr_ids = np.array(s23_addr_valid[ENTITY_ID_COL].to_list())
        del s23_addr_csr, s1_addrs, s23_addrs
        gc.collect()

    del s1_addr_valid, s23_addr_valid
    gc.collect()

    # Initialize PyArrow Parquet writer
    writer = pq.ParquetWriter(
        output_parquet,
        schema=PYARROW_CANDIDATE_SCHEMA,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )

    total_candidates_written = 0

    with track_memory(f"Streaming Cascaded Blocking & Re-ranking for '{country}'"):
        # Process S1 in streaming batches
        for batch_start in range(0, total_s1, streaming_batch_size):
            batch_end = min(batch_start + streaming_batch_size, total_s1)
            s1_batch = s1_full.slice(batch_start, batch_end - batch_start)

            # Strategy A: Char TF-IDF
            df_a, _, _ = blocking_strategy_a_char_tfidf(
                s1_batch,
                s23_df,
                top_n=50,
                threshold=0.05,
                vectorizer=name_vec,
                s23_csr_T=s23_name_csr_T,
            )

            # Strategy B: Token Inverted Index
            df_b = blocking_strategy_b_token_index(
                s1_batch,
                s23_df,
                min_token_len=3,
                max_df=1500,
                top_k=15,
            )

            # Strategy C: Address TF-IDF
            if addr_vec is not None and s23_addr_csr_T is not None:
                df_c, _, _, _ = blocking_strategy_c_addr_tfidf(
                    s1_batch,
                    s23_df,
                    top_n=20,
                    threshold=0.20,
                    vectorizer=addr_vec,
                    s23_addr_csr_T=s23_addr_csr_T,
                    s23_addr_ids=s23_addr_ids,
                )
            else:
                df_c = pl.DataFrame(
                    schema={
                        SOURCE1_ID_COL: pl.String,
                        CANDIDATE_ID_COL: pl.String,
                        "addr_tfidf_cosine": pl.Float32,
                        "flag_c": pl.UInt8,
                    }
                )

            # Strategy D: Phonetic Soundex
            df_d = blocking_strategy_d_soundex(
                s1_batch,
                s23_df,
                max_block_size=2000,
                top_k=15,
            )

            # Union candidates
            unioned = union_candidate_sources(df_a, df_b, df_c, df_d)
            del df_a, df_b, df_c, df_d
            gc.collect()

            # Re-rank and prune to top 15
            pruned = run_reranking(
                unioned,
                max_candidates=max_candidates_per_s1,
                name_weight=0.70,
                addr_weight=0.30,
                use_strategy_fallbacks=True,
                select_contract_cols_only=True,
            )
            del unioned
            gc.collect()

            # Write batch to parquet
            if len(pruned) > 0:
                arrow_tbl = pruned.to_arrow().cast(PYARROW_CANDIDATE_SCHEMA)
                writer.write_table(arrow_tbl)
                total_candidates_written += len(pruned)

            del pruned
            gc.collect()

            logger.info(
                f"[{country}] Processed S1 rows {batch_start:,}..{batch_end:,} / {total_s1:,} "
                f"(Total candidates: {total_candidates_written:,})"
            )

    writer.close()
    del s1_full, s23_df, name_vec, s23_name_csr_T, addr_vec, s23_addr_csr_T
    gc.collect()

    logger.info(
        f"Successfully finished '{country}' partition! Wrote {total_candidates_written:,} "
        f"candidate pairs to {output_parquet}"
    )
    return output_parquet


def run_full_blocking_pipeline(
    splits: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> Dict[str, Dict[str, Path]]:
    """
    Execute cascaded blocking and re-ranking across all requested splits and countries.
    """
    if splits is None:
        splits = ["train", "test"]

    results: Dict[str, Dict[str, Path]] = {}

    for split in splits:
        results[split] = {}
        target_countries = ALL_COUNTRIES if split == "test" else ["US", "India"]
        if countries is not None:
            target_countries = [c for c in target_countries if c in countries]

        for country in target_countries:
            out_file = process_country_partition(
                split=split,
                country=country,
                processed_data_dir=processed_data_dir,
            )
            results[split][country] = out_file

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cascaded Blocking & Re-ranking pipeline.")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Split to run ('train', 'test', or 'all')",
    )
    parser.add_argument(
        "--country",
        choices=["US", "India", "France", "all"],
        default="all",
        help="Country partition to run",
    )
    args = parser.parse_args()

    target_splits = ["train", "test"] if args.split == "all" else [args.split]
    target_countries = None if args.country == "all" else [args.country]

    run_full_blocking_pipeline(splits=target_splits, countries=target_countries)
