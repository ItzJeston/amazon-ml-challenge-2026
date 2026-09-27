"""
src/inference.py
Milestone 5 (Phase 7: End-to-End Inference, Dynamic Validation & Final Integration)
Amazon Business Entity Resolution Challenge.

Responsibilities:
1. Explicit UTF-8 stdout initialization:
   import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
2. Complete end-to-end inference pipeline:
   - Ingests test partitions: data_processed/test/{country}/
   - Generates candidate pairs via src.blocking.run_cascaded_blocking
   - Prunes via src.reranking.prune_candidate_pairs (median 10-12, max 15 per S1)
   - Extracts 32 pairwise features via src.features.compute_pairwise_features (ZERO row iteration)
   - Scores candidate pairs with models/{country}/ StackedEnsemble
   - Applies calibrated country thresholds (tau_US, tau_India, tau_France) from models/thresholds.json
   - Aggregates matched candidate IDs per S1 entity
   - Enforces memory limit <= 6.0GB peak RAM via streaming chunks
3. Generates submission files via src.submission:
   - output/matching_results.tsv
   - output/candidate_pairs.tsv
4. Hard runtime dynamic assertions and official validation:
   - verify_submission_integrity
   - validate_submission_files
"""

import gc
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# Enforce explicit UTF-8 stdout initialization (Global Architecture Requirement)
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.blocking import (
    BLOCKING_FLAGS_COL,
    CANDIDATE_ID_COL,
    RERANKER_SCORE_COL,
    SOURCE1_ID_COL,
    run_cascaded_blocking,
)
from src.config import (
    ALL_COUNTRIES,
    CLEAN_ADDR_COL,
    CLEAN_NAME_COL,
    COUNTRY_COL,
    ENTITY_ID_COL,
    MODELS_DIR,
    NUM_THREADS,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STREAMING_BATCH_SIZE,
    TEST_COUNTRIES,
)
from src.features import (
    FEATURE_NAMES,
    compute_pairwise_features,
)
from src.models import StackedEnsemble
from src.optimization import DEFAULT_THRESHOLD, load_thresholds
from src.reranking import (
    export_candidate_pairs_parquet,
    prune_candidates,
    run_reranking,
)
from src.submission import (
    export_candidate_pairs_tsv,
    export_matching_results_tsv,
    read_s1_entity_ids,
    validate_submission_files,
    verify_submission_integrity,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

# Alias prune_candidate_pairs to prune_candidates for exact spec naming
prune_candidate_pairs = prune_candidates

setup_utf8_stdout()
logger = get_logger("inference")


# ==============================================================================
# 1. Candidate Scoring Engine with Streaming Morsel Batches (Peak RAM <= 6.0GB)
# ==============================================================================
def score_candidate_pairs_chunked(
    pairs_df: pl.DataFrame,
    s1_entities_df: pl.DataFrame,
    cand_entities_df: pl.DataFrame,
    country: str,
    threshold: float,
    ensemble: Optional[StackedEnsemble] = None,
    chunk_size: int = STREAMING_BATCH_SIZE,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Score candidate pairs in streaming morsel batches:
    1. Extracts 32 pairwise features via pure native Polars (zero row iteration).
    2. Generates calibrated stacked ensemble match probabilities P(y=1 | X).
    3. Filters matches based on country-calibrated threshold tau_country.
    4. Accumulates candidates and matches per S1 entity ID.
    5. Bounds working memory strictly <= 6.0 GB via morsel-driven processing.

    Parameters
    ----------
    pairs_df : pl.DataFrame
        Candidate pairs DataFrame conforming to Interface Contract 2.
    s1_entities_df : pl.DataFrame
        Preprocessed Source 1 entities for the country partition.
    cand_entities_df : pl.DataFrame
        Preprocessed Candidate entities (S2/S3) for the country partition.
    country : str
        Country partition name ('US', 'India', 'France').
    threshold : float
        Decision threshold tau_country in [0.40, 0.95].
    ensemble : Optional[StackedEnsemble]
        Fitted or loaded StackedEnsemble model. If None, falls back to reranker cosine score.
    chunk_size : int
        Chunk size for morsel-driven feature extraction.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        (matches_by_s1, candidates_by_s1)
    """
    matches_by_s1: Dict[str, List[str]] = {}
    candidates_by_s1: Dict[str, List[str]] = {}

    total_pairs = len(pairs_df)
    if total_pairs == 0:
        return matches_by_s1, candidates_by_s1

    logger.info(
        f"[{country}] Scoring {total_pairs:,} candidate pairs with threshold={threshold:.4f} "
        f"(Chunk size: {chunk_size:,})..."
    )

    # Pre-index entity tables for fast lookups if necessary
    for batch_start in range(0, total_pairs, chunk_size):
        batch_end = min(batch_start + chunk_size, total_pairs)
        pair_chunk = pairs_df.slice(batch_start, batch_end - batch_start)

        # 1. Compute 32 pairwise similarity metrics (Pure Native Polars, ZERO row iteration)
        features_chunk = compute_pairwise_features(
            pairs_df=pair_chunk,
            s1_entities_df=s1_entities_df,
            cand_entities_df=cand_entities_df,
            ground_truth_df=None,
            chunk_size=chunk_size,
            keep_metadata_cols=True,
        )

        # 2. Extract feature matrix X of shape [N_chunk, 32]
        X_chunk = features_chunk[FEATURE_NAMES].to_numpy()

        # 3. Model scoring: Stacked Ensemble or Cosine Fallback
        if ensemble is not None and getattr(ensemble, "meta_learner", None) is not None:
            try:
                probs = ensemble.predict_proba(X_chunk)
            except Exception as exc:
                logger.warning(
                    f"[{country}] Ensemble predict_proba failed ({exc}); falling back to reranker cosine score."
                )
                probs = pair_chunk[RERANKER_SCORE_COL].fill_null(0.0).to_numpy()
        else:
            probs = pair_chunk[RERANKER_SCORE_COL].fill_null(0.0).to_numpy()

        # 4. Filter matches based on country threshold
        s1_ids = pair_chunk[SOURCE1_ID_COL].to_list()
        c_ids = pair_chunk[CANDIDATE_ID_COL].to_list()

        for s1, c, p in zip(s1_ids, c_ids, probs):
            # Accumulate candidate ID
            if s1 not in candidates_by_s1:
                candidates_by_s1[s1] = [c]
            else:
                candidates_by_s1[s1].append(c)

            # Accumulate match if above calibrated threshold
            if p >= threshold:
                if s1 not in matches_by_s1:
                    matches_by_s1[s1] = [c]
                else:
                    matches_by_s1[s1].append(c)

        del pair_chunk, features_chunk, X_chunk, probs, s1_ids, c_ids
        gc.collect()

    matched_s1_count = len(matches_by_s1)
    logger.info(
        f"[{country}] Scoring completed: {total_pairs:,} pairs evaluated | "
        f"S1 with matches: {matched_s1_count:,} / {len(candidates_by_s1):,}"
    )

    return matches_by_s1, candidates_by_s1


# ==============================================================================
# 2. Single Country Partition Inference Engine
# ==============================================================================
def run_country_inference(
    country: str,
    split: str = "test",
    processed_data_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    models_dir: Union[str, Path] = MODELS_DIR,
    threshold: Optional[float] = None,
    ensemble: Optional[StackedEnsemble] = None,
    max_candidates_per_s1: int = 15,
    streaming_batch_size: int = STREAMING_BATCH_SIZE,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """
    Execute full inference pipeline for a single country partition.

    Steps:
    1. Ingest test partition files: data_processed/test/{country}/
    2. Check for or generate candidate pairs via run_cascaded_blocking
    3. Prune candidate pairs to top 15 per S1 (prune_candidate_pairs)
    4. Load StackedEnsemble from models/{country}/
    5. Resolve threshold tau_country from models/thresholds.json
    6. Score candidate pairs in streaming chunks <= 6.0GB peak RAM
    7. Return matches, candidates, and ordered S1 IDs.
    """
    country_dir = Path(processed_data_dir) / split / country
    s1_parquet = country_dir / f"{split}_source1.parquet"
    s2_parquet = country_dir / f"{split}_source2.parquet"
    s3_parquet = country_dir / f"{split}_source3.parquet"
    cand_parquet = country_dir / "candidate_pairs.parquet"

    if not s1_parquet.exists():
        logger.warning(f"S1 partition not found: {s1_parquet}. Skipping {country}...")
        return {}, {}, []

    logger.info(f"=== Starting Inference Pipeline for Partition '{split}/{country}' ===")

    # 1. Ingest S1 and S23 entities
    s1_df = pl.read_parquet(s1_parquet)
    country_s1_ids = s1_df[ENTITY_ID_COL].to_list()
    logger.info(f"[{country}] Ingested {len(country_s1_ids):,} Source-1 entities.")

    s23_dfs = []
    if s2_parquet.exists():
        s23_dfs.append(pl.read_parquet(s2_parquet))
    if s3_parquet.exists():
        s23_dfs.append(pl.read_parquet(s3_parquet))

    s23_df = pl.concat(s23_dfs) if len(s23_dfs) > 1 else (s23_dfs[0] if s23_dfs else pl.DataFrame())
    del s23_dfs
    gc.collect()

    logger.info(f"[{country}] Ingested {len(s23_df):,} Source-2/3 candidate entities.")

    # 2. Ingest or Generate Candidate Pairs
    if cand_parquet.exists():
        logger.info(f"[{country}] Loading existing candidate pairs from {cand_parquet}...")
        cand_pairs_df = pl.read_parquet(cand_parquet)
    else:
        logger.info(f"[{country}] candidate_pairs.parquet not found. Running cascaded blocking...")
        if len(s23_df) > 0:
            unioned_pairs = run_cascaded_blocking(
                s1_df=s1_df,
                s23_df=s23_df,
                n_threads=NUM_THREADS,
            )
            # Prune via prune_candidate_pairs (median 10-12, max 15 per S1)
            cand_pairs_df = run_reranking(
                candidates_df=unioned_pairs,
                max_candidates=max_candidates_per_s1,
                name_weight=0.70,
                addr_weight=0.30,
                use_strategy_fallbacks=True,
                select_contract_cols_only=True,
            )
            # Save for future reproducibility
            export_candidate_pairs_parquet(cand_pairs_df, cand_parquet)
            del unioned_pairs
            gc.collect()
        else:
            cand_pairs_df = pl.DataFrame(
                schema={
                    SOURCE1_ID_COL: pl.String,
                    CANDIDATE_ID_COL: pl.String,
                    RERANKER_SCORE_COL: pl.Float32,
                    BLOCKING_FLAGS_COL: pl.UInt8,
                }
            )

    logger.info(
        f"[{country}] Candidate pairs ready: {len(cand_pairs_df):,} pairs "
        f"(Max candidates per S1: {max_candidates_per_s1})."
    )

    # 3. Resolve Stacked Ensemble Model
    if ensemble is None:
        country_model_dir = Path(models_dir) / country
        if country_model_dir.exists() and (country_model_dir / "ensemble_meta.json").exists():
            logger.info(f"[{country}] Loading StackedEnsemble from {country_model_dir}...")
            try:
                ensemble = StackedEnsemble.load(country_model_dir)
            except Exception as e:
                logger.warning(f"[{country}] Failed to load ensemble from {country_model_dir}: {e}")
                ensemble = None
        else:
            # Fallback model search: If France model is absent, try US or India transfer ensemble
            if country == "France":
                for fallback_country in ("US", "India"):
                    fallback_dir = Path(models_dir) / fallback_country
                    if fallback_dir.exists() and (fallback_dir / "ensemble_meta.json").exists():
                        logger.info(
                            f"[{country}] Using transfer StackedEnsemble from {fallback_country} ({fallback_dir})..."
                        )
                        try:
                            ensemble = StackedEnsemble.load(fallback_dir)
                            break
                        except Exception:
                            pass

    # 4. Resolve Calibrated Threshold
    if threshold is None:
        thresholds_dict = load_thresholds(Path(models_dir) / "thresholds.json")
        threshold = thresholds_dict.get(
            country,
            thresholds_dict.get(f"tau_{country}", DEFAULT_THRESHOLD),
        )

    logger.info(f"[{country}] Applying calibrated threshold tau_{country} = {threshold:.4f}")

    # 5. Score Candidates in Morsel Batches
    with track_memory(f"Candidate Scoring for '{country}'"):
        matches, candidates = score_candidate_pairs_chunked(
            pairs_df=cand_pairs_df,
            s1_entities_df=s1_df,
            cand_entities_df=s23_df,
            country=country,
            threshold=threshold,
            ensemble=ensemble,
            chunk_size=streaming_batch_size,
        )

    del s1_df, s23_df, cand_pairs_df
    gc.collect()

    return matches, candidates, country_s1_ids


# ==============================================================================
# 3. Master End-to-End Inference Pipeline
# ==============================================================================
def run_inference_pipeline(
    split: str = "test",
    countries: Optional[List[str]] = None,
    processed_data_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    models_dir: Union[str, Path] = MODELS_DIR,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    raw_data_dir: Union[str, Path] = RAW_DATA_DIR,
    streaming_batch_size: int = STREAMING_BATCH_SIZE,
    max_candidates_per_s1: int = 15,
    custom_thresholds: Optional[Dict[str, float]] = None,
    run_validation_wrapper: bool = True,
) -> Dict[str, Any]:
    """
    Master End-to-End Inference Pipeline:
    1. Iterates over test partitions (US, India, France).
    2. Runs cascaded blocking & pruning (median 10-12, max 15 per S1).
    3. Extracts 32 pairwise features in pure Polars (zero row iteration).
    4. Predicts match probabilities using calibrated StackedEnsemble.
    5. Applies calibrated thresholds (tau_US, tau_India, tau_France).
    6. Exports output/matching_results.tsv and output/candidate_pairs.tsv.
    7. Runs hard runtime dynamic assertions against test_source1.tsv.
    8. Validates submission via validate_submission.py wrapper.
    9. Guarantees memory consumption <= 6.0GB peak RAM.

    Returns
    -------
    Dict[str, Any]
        Inference execution summary and telemetry.
    """
    start_time = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_countries = countries or (TEST_COUNTRIES if split == "test" else ["US", "India"])
    logger.info(
        f"Initiating Master Inference Pipeline across countries: {target_countries} "
        f"for split: '{split}'..."
    )

    all_matches_by_s1: Dict[str, List[str]] = {}
    all_candidates_by_s1: Dict[str, List[str]] = {}
    all_s1_ids: List[str] = []

    # Process each country partition independently to enforce memory boundary <= 6.0GB
    with track_memory("Master Test Inference Pipeline"):
        for country in target_countries:
            tau = custom_thresholds.get(country) if custom_thresholds else None

            c_matches, c_candidates, c_s1_ids = run_country_inference(
                country=country,
                split=split,
                processed_data_dir=processed_data_dir,
                models_dir=models_dir,
                threshold=tau,
                max_candidates_per_s1=max_candidates_per_s1,
                streaming_batch_size=streaming_batch_size,
            )

            all_matches_by_s1.update(c_matches)
            all_candidates_by_s1.update(c_candidates)
            all_s1_ids.extend(c_s1_ids)

            gc.collect()

    logger.info(
        f"Country aggregation complete! Total S1 entities collected: {len(all_s1_ids):,} | "
        f"Total entities with matches: {len(all_matches_by_s1):,}"
    )

    # If raw test_source1.tsv is available, ensure all test entities are represented
    test_s1_tsv = Path(raw_data_dir) / split / f"{split}_source1.tsv"
    if test_s1_tsv.exists():
        raw_s1_ids = read_s1_entity_ids(test_s1_tsv)
        if countries is None or set(target_countries) >= set(TEST_COUNTRIES if split == "test" else TRAIN_COUNTRIES):
            logger.info(
                f"Using reference S1 ID order from {test_s1_tsv} ({len(raw_s1_ids):,} entities)."
            )
            all_s1_ids = raw_s1_ids
        elif len(raw_s1_ids) > len(all_s1_ids):
            logger.info(
                f"Syncing S1 list with raw reference ({len(raw_s1_ids):,} entities from {test_s1_tsv})."
            )
            all_s1_ids = raw_s1_ids

    # 4. Export submission files adhering strictly to formatting rules
    matching_tsv_path = out_dir / "matching_results.tsv"
    candidate_tsv_path = out_dir / "candidate_pairs.tsv"

    export_matching_results_tsv(
        matches=all_matches_by_s1,
        s1_entity_ids=all_s1_ids,
        output_path=matching_tsv_path,
    )

    export_candidate_pairs_tsv(
        candidates=all_candidates_by_s1,
        s1_entity_ids=all_s1_ids,
        output_path=candidate_tsv_path,
    )

    # 5. Hard runtime dynamic assertions (assert len(matching) == len(test_source1), etc.)
    integrity_metrics = {}
    if test_s1_tsv.exists():
        integrity_metrics = verify_submission_integrity(
            matching_source=matching_tsv_path,
            candidate_source=candidate_tsv_path,
            test_source1=test_s1_tsv,
        )
    else:
        integrity_metrics = verify_submission_integrity(
            matching_source=matching_tsv_path,
            candidate_source=candidate_tsv_path,
            test_source1=all_s1_ids,
        )

    # 6. Official validation wrapper
    validator_errors, validator_warnings = [], []
    if run_validation_wrapper and test_s1_tsv.exists():
        validator_errors, validator_warnings = validate_submission_files(
            matching_path=matching_tsv_path,
            candidate_path=candidate_tsv_path,
            test_dir=test_s1_tsv.parent,
            check_ids=False,
            raise_on_error=True,
        )

    elapsed_time = time.time() - start_time
    summary = {
        "status": "SUCCESS",
        "elapsed_seconds": round(elapsed_time, 2),
        "matching_file": str(matching_tsv_path),
        "candidate_file": str(candidate_tsv_path),
        "total_s1_entities": len(all_s1_ids),
        "integrity_metrics": integrity_metrics,
        "validator_errors": len(validator_errors),
        "validator_warnings": len(validator_warnings),
    }

    logger.info(
        f"Master Inference Pipeline Completed Successfully in {elapsed_time:.2f}s! "
        f"Outputs: {matching_tsv_path} & {candidate_tsv_path}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run End-to-End Inference Pipeline for Business Entity Resolution."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train"],
        help="Split to infer on ('test' or 'train')",
    )
    parser.add_argument(
        "--country",
        default="all",
        choices=["all", "US", "India", "France"],
        help="Country partition to run",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for TSV submission files",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help="Directory containing trained model artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=STREAMING_BATCH_SIZE,
        help="Morsel batch chunk size for feature computation",
    )
    args = parser.parse_args()

    target_countries = None if args.country == "all" else [args.country]

    run_inference_pipeline(
        split=args.split,
        countries=target_countries,
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        streaming_batch_size=args.batch_size,
    )
