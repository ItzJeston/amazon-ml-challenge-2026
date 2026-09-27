#!/usr/bin/env python3
"""
run_pipeline.py
Amazon Business Entity Resolution Challenge — Master CLI Entry Point.

Unified pipeline orchestrator supporting:
  --stage [all, preprocess, block, features, train, infer, validate]

Stages:
1. preprocess:
   - Ingests raw TSV files from extracted_data/student_resource/dataset/
   - Normalizes addresses, strips legal suffixes, transliterates non-ASCII, computes Soundex
   - Partitions into zstd-compressed Parquet by country ('US', 'India', 'France')
2. block:
   - Cascaded blocking (4 strategies: Char TF-IDF, Token Index, Address TF-IDF, Soundex)
   - Lightweight cosine re-ranking & candidate bounding (median 10-12, max 15 per S1)
   - Exports candidate_pairs.parquet conforming to Interface Contract 2
3. features:
   - Computes 32 pairwise similarity metrics via pure Polars (ZERO row iteration)
   - Exports features.parquet conforming to Interface Contract 3
   - Creates disjoint 80/20 train/holdout split at S1 entity level
4. train:
   - 5-Fold StratifiedGroupKFold on S1 entity ID
   - Level-0 GBDT models: LightGBM (CPU), XGBoost (GPU), CatBoost (CPU)
   - Platt Scaling via CalibratedClassifierCV(FrozenEstimator)
   - Level-1 LogisticRegression Meta-Learner
   - 3D Bayesian Optuna search for (tau_US, tau_India, tau_France)
5. infer:
   - End-to-end inference on test partitions (streaming chunks <= 6.0GB peak RAM)
   - Exports output/matching_results.tsv and output/candidate_pairs.tsv
   - Hard runtime dynamic assertions (assert len(matching) == len(test_source1))
6. validate:
   - Runs official submission validation wrapper (validate_submission.py)
   - Verifies TSV formatting, prefix integrity, singleton handling, and subset rules
7. all:
   - Sequentially executes stages 1 through 6
"""

import argparse
import gc
import io
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Enforce explicit UTF-8 stdout initialization (Global Architecture Requirement F02)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.config import (
    ALL_COUNTRIES,
    MODELS_DIR,
    NUM_THREADS,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STREAMING_BATCH_SIZE,
    TEST_COUNTRIES,
    TRAIN_COUNTRIES,
)
from src.dataset import create_disjoint_train_holdout_split
from src.features import (
    FEATURE_NAMES,
    compute_pairwise_features,
    export_features_to_parquet,
    run_feature_engineering_pipeline,
)
from src.inference import run_inference_pipeline
from src.models import (
    StackedEnsemble,
    train_country_ensemble,
)
from src.optimization import (
    HoldoutEvaluationDataset,
    load_country_holdout_features,
    optimize_3d_thresholds,
)
from src.preprocessing import run_preprocessing_pipeline
from src.reranking import (
    process_country_partition,
    run_full_blocking_pipeline,
)
from src.submission import (
    validate_submission_files,
    verify_submission_integrity,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("run_pipeline")


# ==============================================================================
# 1. Stage Handlers
# ==============================================================================
def stage_preprocess(
    splits: List[str],
    raw_data_dir: Path,
    processed_data_dir: Path,
) -> Dict[str, Any]:
    """Execute Phase 1-2: Preprocessing, Normalization & Partitioning."""
    logger.info(">>> Executing Stage: PREPROCESS")
    with track_memory("Stage Preprocess"):
        results = run_preprocessing_pipeline(
            splits=splits,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
        )
    logger.info(">>> Stage PREPROCESS completed successfully.")
    return results


def stage_block(
    splits: List[str],
    countries: Optional[List[str]],
    processed_data_dir: Path,
) -> Dict[str, Any]:
    """Execute Phase 3: Cascaded Blocking & Re-ranking."""
    logger.info(">>> Executing Stage: BLOCK")
    with track_memory("Stage Block"):
        results = run_full_blocking_pipeline(
            splits=splits,
            countries=countries,
            processed_data_dir=processed_data_dir,
        )
    logger.info(">>> Stage BLOCK completed successfully.")
    return results


def stage_features(
    splits: List[str],
    countries: Optional[List[str]],
    processed_data_dir: Path,
    chunk_size: int = STREAMING_BATCH_SIZE,
) -> Dict[str, Any]:
    """Execute Phase 4: Feature Engineering & Disjoint 80/20 Splits."""
    logger.info(">>> Executing Stage: FEATURES")
    results = {}
    with track_memory("Stage Features"):
        for split in splits:
            results[split] = {}
            target_countries = (
                countries
                if countries
                else (ALL_COUNTRIES if split == "test" else list(TRAIN_COUNTRIES))
            )
            for country in target_countries:
                feat_path = run_feature_engineering_pipeline(
                    split=split,
                    country=country,
                    data_dir=processed_data_dir,
                    output_dir=processed_data_dir,
                    chunk_size=chunk_size,
                )
                results[split][country] = str(feat_path)

                # For training partitions, generate disjoint 80/20 train/holdout split
                if split == "train":
                    logger.info(f"Creating disjoint 80/20 train/holdout split for train/{country}...")
                    features_df = pl.read_parquet(feat_path)
                    train_df, holdout_df = create_disjoint_train_holdout_split(
                        features_df=features_df,
                        test_size=0.20,
                        random_state=42,
                    )
                    country_dir = processed_data_dir / split / country
                    export_features_to_parquet(train_df, country_dir / "features_train.parquet")
                    export_features_to_parquet(holdout_df, country_dir / "features_holdout.parquet")
                    del features_df, train_df, holdout_df
                    gc.collect()

    logger.info(">>> Stage FEATURES completed successfully.")
    return results


def stage_train(
    countries: Optional[List[str]],
    processed_data_dir: Path,
    models_dir: Path,
    n_splits: int = 5,
    n_trials_optuna: int = 50,
) -> Dict[str, Any]:
    """Execute Phase 5-6: Stacked Ensemble Training, Platt Calibration & 3D Optuna Search."""
    logger.info(">>> Executing Stage: TRAIN")
    results = {}
    train_countries = [c for c in (countries or TRAIN_COUNTRIES) if c in TRAIN_COUNTRIES]

    holdout_dict: Dict[str, pl.DataFrame] = {}

    with track_memory("Stage Train"):
        # 1. Train Stacked Ensemble per Country
        for country in train_countries:
            train_features_path = (
                processed_data_dir / "train" / country / "features_train.parquet"
            )
            if not train_features_path.exists():
                train_features_path = (
                    processed_data_dir / "train" / country / "features.parquet"
                )

            if not train_features_path.exists():
                logger.warning(
                    f"Train features file not found for {country}: {train_features_path}. Skipping..."
                )
                continue

            logger.info(f"Training StackedEnsemble for country '{country}'...")
            feat_df = pl.read_parquet(train_features_path)
            country_model_dir = models_dir / country

            ensemble = train_country_ensemble(
                features_df=feat_df,
                country=country,
                output_dir=country_model_dir,
                n_splits=n_splits,
            )
            results[country] = str(country_model_dir)

            del feat_df
            gc.collect()

            # Load holdout for threshold optimization
            holdout_path = (
                processed_data_dir / "train" / country / "features_holdout.parquet"
            )
            if holdout_path.exists():
                logger.info(f"Precomputing holdout predictions for '{country}'...")
                holdout_df = pl.read_parquet(holdout_path)
                holdout_dict[country] = holdout_df

        # 2. 3D Bayesian Optuna Search mapping (tau_US, tau_India, tau_France)
        if holdout_dict:
            logger.info("Starting 3D Bayesian Optuna Threshold Optimization on holdout data...")
            # Precompute holdout evaluation dataset
            ensemble_dict = {}
            for c in holdout_dict.keys():
                c_dir = models_dir / c
                if c_dir.exists() and (c_dir / "ensemble_meta.json").exists():
                    ensemble_dict[c] = StackedEnsemble.load(c_dir)

            holdout_eval_data = HoldoutEvaluationDataset.from_holdout_features(
                holdout_features=holdout_dict,
                ensembles=ensemble_dict,
            )

            thresholds = optimize_3d_thresholds(
                holdout_eval_data=holdout_eval_data,
                n_trials=n_trials_optuna,
                output_path=models_dir / "thresholds.json",
            )
            results["thresholds"] = thresholds
        else:
            logger.warning("No holdout data available for threshold optimization.")

    logger.info(">>> Stage TRAIN completed successfully.")
    return results


def stage_infer(
    countries: Optional[List[str]],
    processed_data_dir: Path,
    models_dir: Path,
    output_dir: Path,
    raw_data_dir: Path,
    streaming_batch_size: int = STREAMING_BATCH_SIZE,
) -> Dict[str, Any]:
    """Execute Phase 7: Final Inference & Submission File Generation."""
    logger.info(">>> Executing Stage: INFER")
    with track_memory("Stage Infer"):
        results = run_inference_pipeline(
            split="test",
            countries=countries,
            processed_data_dir=processed_data_dir,
            models_dir=models_dir,
            output_dir=output_dir,
            raw_data_dir=raw_data_dir,
            streaming_batch_size=streaming_batch_size,
            run_validation_wrapper=True,
        )
    logger.info(">>> Stage INFER completed successfully.")
    return results


def stage_validate(
    output_dir: Path,
    raw_data_dir: Path,
    check_ids: bool = False,
) -> Dict[str, Any]:
    """Execute Submission File Validation."""
    logger.info(">>> Executing Stage: VALIDATE")
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"
    test_dir = raw_data_dir / "test"

    if not matching_tsv.exists():
        raise FileNotFoundError(f"Matching results file not found: {matching_tsv}")

    with track_memory("Stage Validate"):
        errors, warnings = validate_submission_files(
            matching_path=matching_tsv,
            candidate_path=candidate_tsv if candidate_tsv.exists() else None,
            test_dir=test_dir,
            check_ids=check_ids,
            raise_on_error=True,
        )

        # Dynamic integrity assertions
        integrity_metrics = {}
        test_s1_tsv = test_dir / "test_source1.tsv"
        if test_s1_tsv.exists():
            integrity_metrics = verify_submission_integrity(
                matching_source=matching_tsv,
                candidate_source=candidate_tsv if candidate_tsv.exists() else matching_tsv,
                test_source1=test_s1_tsv,
            )

    logger.info(">>> Stage VALIDATE completed successfully.")
    return {
        "status": "VALID",
        "errors": errors,
        "warnings": warnings,
        "integrity_metrics": integrity_metrics,
    }


# ==============================================================================
# 2. Master CLI Entry Point
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master Execution Pipeline — Amazon Business Entity Resolution Challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "block", "features", "train", "infer", "validate"],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "test"],
        help="Dataset split to operate on",
    )
    parser.add_argument(
        "--country",
        default="all",
        choices=["all", "US", "India", "France"],
        help="Country partition to operate on",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DATA_DIR),
        help="Path to raw dataset folder",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Path to processed parquet data directory",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help="Path to models directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Path to output submission directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=STREAMING_BATCH_SIZE,
        help="Morsel chunk size for streaming execution",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=NUM_THREADS,
        help="Number of CPU execution threads",
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Enable deep S2/S3 ID-existence check during validation",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    target_splits = ["train", "test"] if args.split == "all" else [args.split]
    target_countries = None if args.country == "all" else [args.country]

    start_total_time = time.time()
    logger.info("==============================================================================")
    logger.info("AMAZON BUSINESS ENTITY RESOLUTION PIPELINE")
    logger.info(f"Stage: {args.stage} | Splits: {target_splits} | Countries: {target_countries or 'ALL'}")
    logger.info(f"Threads: {args.num_threads} | Morsel Batch Size: {args.batch_size:,}")
    logger.info("==============================================================================")

    try:
        if args.stage in ("all", "preprocess"):
            stage_preprocess(
                splits=target_splits,
                raw_data_dir=raw_dir,
                processed_data_dir=processed_dir,
            )

        if args.stage in ("all", "block"):
            stage_block(
                splits=target_splits,
                countries=target_countries,
                processed_data_dir=processed_dir,
            )

        if args.stage in ("all", "features"):
            stage_features(
                splits=target_splits,
                countries=target_countries,
                processed_data_dir=processed_dir,
                chunk_size=args.batch_size,
            )

        if args.stage in ("all", "train"):
            stage_train(
                countries=target_countries,
                processed_data_dir=processed_dir,
                models_dir=models_dir,
            )

        if args.stage in ("all", "infer"):
            stage_infer(
                countries=target_countries,
                processed_data_dir=processed_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                raw_data_dir=raw_dir,
                streaming_batch_size=args.batch_size,
            )

        if args.stage in ("all", "validate"):
            stage_validate(
                output_dir=output_dir,
                raw_data_dir=raw_dir,
                check_ids=args.check_ids,
            )

        total_elapsed = time.time() - start_total_time
        logger.info("==============================================================================")
        logger.info(f"PIPELINE RUN COMPLETED SUCCESSFULLY in {total_elapsed:.2f}s!")
        logger.info("==============================================================================")
        return 0

    except Exception as exc:
        logger.exception(f"Pipeline execution encountered fatal error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
