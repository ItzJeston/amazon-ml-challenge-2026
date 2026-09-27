"""
src/optimization.py
Milestone 4 (Phase 6: Entity-Level Macro F0.5 & 3D Bayesian Optuna Threshold Optimization)
Amazon Business Entity Resolution Challenge.

Implements:
1. Explicit UTF-8 stdout initialization:
   import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
2. Entity-level Macro F0.5 Metric Function:
   - beta = 0.5, weighting precision twice as heavily as recall:
     F0.5 = 1.25 * P * R / (0.25 * P + R)
   - Singleton handling: singletons (entities with 0 ground truth matches) score 1.0
     if prediction is empty; any false merge scores 0.0.
   - Non-singletons: score 0.0 if prediction is empty or TP == 0; otherwise F0.5.
   - Unbiased macro average over all unique S1 entities.
3. 3D Bayesian Optuna Search Mapping:
   - Jointly optimizes (tau_US, tau_India, tau_France) in [0.40, 0.95].
   - Restricted strictly to the 20% disjoint holdout dataset.
   - Precomputes stacked probabilities once for ultra-fast vectorized trial evaluations.
   - Persists optimal thresholds to models/thresholds.json.

Hardware Target: Intel i7-13650HX (14 physical execution threads), peak RAM <= 6.0 GB budget.
"""

from collections import defaultdict
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# Enforce explicit UTF-8 stdout initialization
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import polars as pl

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from src.config import (
    ALL_COUNTRIES,
    MODELS_DIR,
    NUM_THREADS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_GT_FILE,
)
from src.features import (
    CANDIDATE_ID_COL,
    COUNTRY_COL,
    LABEL_COL,
    SOURCE1_ID_COL,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("optimization")

# Default threshold bounds
MIN_THRESHOLD = 0.40
MAX_THRESHOLD = 0.95
DEFAULT_THRESHOLD = 0.75


# ==============================================================================
# 1. Ground Truth Ingestion & Helper Utilities
# ==============================================================================
def load_ground_truth_dict(
    gt_source: Union[str, Path, pl.DataFrame, Dict[str, Set[str]]],
) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping from S1 entity ID to set of true candidate IDs.

    Parameters
    ----------
    gt_source : Union[str, Path, pl.DataFrame, Dict[str, Set[str]]]
        Path to TSV ground truth, or Polars DataFrame, or pre-built dict.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of matched candidate IDs (empty set for singletons).
    """
    if isinstance(gt_source, dict):
        return {str(k): set(v) for k, v in gt_source.items()}

    if isinstance(gt_source, (str, Path)):
        path = Path(gt_source)
        if not path.exists():
            # Try finding in default raw dataset directory
            fallback = RAW_DATA_DIR / "train" / TRAIN_GT_FILE
            if fallback.exists():
                path = fallback
            else:
                raise FileNotFoundError(f"Ground truth file not found: {path}")

        gt_df = pl.read_csv(
            path,
            separator="\t",
            has_header=True,
            schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        )
    elif isinstance(gt_source, pl.DataFrame):
        gt_df = gt_source
    else:
        raise TypeError(f"Unsupported ground truth source type: {type(gt_source)}")

    gt_dict: Dict[str, Set[str]] = {}
    s1_col = "source1_entity_id"
    match_col = "matched_entity_ids"

    if s1_col not in gt_df.columns or match_col not in gt_df.columns:
        raise KeyError(f"Expected columns ['{s1_col}', '{match_col}'] in ground truth DataFrame!")

    s1_list = gt_df[s1_col].to_list()
    matches_list = gt_df[match_col].to_list()

    for s1_id, match_str in zip(s1_list, matches_list):
        if match_str is not None and str(match_str).strip() and str(match_str).lower() != "null":
            cand_set = {c.strip() for c in str(match_str).split(",") if c.strip()}
            gt_dict[str(s1_id)] = cand_set
        else:
            gt_dict[str(s1_id)] = set()

    return gt_dict


# ==============================================================================
# 2. Entity-Level Macro F0.5 Metric Computation
# ==============================================================================
def compute_single_entity_f05(
    true_set: Set[str],
    pred_set: Set[str],
) -> float:
    """
    Compute competition F0.5 score for an individual Source-1 entity:
    - beta = 0.5: precision is weighted twice as heavily as recall.
      F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    - Singleton rule: singletons (|true_set| == 0) score:
      - 1.0 if prediction is empty (|pred_set| == 0).
      - 0.0 if any false merge occurred (|pred_set| > 0).
    - Non-singletons (|true_set| > 0):
      - 0.0 if prediction is empty (|pred_set| == 0) or TP == 0.
      - F0.5 otherwise.
    """
    is_singleton = len(true_set) == 0

    if is_singleton:
        # Singleton entity: 1.0 if correctly left empty, 0.0 if false positive matches predicted
        return 1.0 if len(pred_set) == 0 else 0.0

    # Non-singleton entity with at least 1 ground truth match
    if len(pred_set) == 0:
        # Missed link: precision=0, recall=0
        return 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0

    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    denom = 0.25 * precision + recall
    if denom <= 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return float(f05)


def compute_entity_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    all_s1_ids: Optional[Iterable[str]] = None,
) -> float:
    """
    Computes exact Macro F0.5 over all unique Source-1 entities.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Ground truth mapping: s1_id -> set of true candidate IDs.
    predictions : Dict[str, Set[str]]
        Prediction mapping: s1_id -> set of predicted candidate IDs.
    all_s1_ids : Optional[Iterable[str]]
        Optional explicit collection of all S1 entities to evaluate.
        If omitted, evaluates over union of keys in ground_truth and predictions.

    Returns
    -------
    float
        Macro F0.5 score in [0.0, 1.0].
    """
    if all_s1_ids is not None:
        target_s1 = set(all_s1_ids)
    else:
        target_s1 = set(ground_truth.keys()).union(set(predictions.keys()))

    if not target_s1:
        return 0.0

    scores: List[float] = []
    for s1_id in target_s1:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        score = compute_single_entity_f05(true_set, pred_set)
        scores.append(score)

    return float(np.mean(scores))


# ==============================================================================
# 3. Vectorized Holdout Evaluation Container
# ==============================================================================
class HoldoutEvaluationDataset:
    """
    Preprocessed evaluation container holding candidate pairs and precomputed
    probabilities on the 20% disjoint holdout dataset.
    Enables blazing fast Optuna trials (<5ms per evaluation).
    """

    def __init__(
        self,
        country_data: Dict[str, Dict[str, Any]],
        ground_truth: Dict[str, Set[str]],
    ) -> None:
        """
        Parameters
        ----------
        country_data : Dict[str, Dict[str, Any]]
            Mapping country -> {
                's1_ids': np.ndarray[str],
                'cand_ids': np.ndarray[str],
                'probabilities': np.ndarray[float32],
                'unique_s1': List[str],
            }
        ground_truth : Dict[str, Set[str]]
            Ground truth dictionary.
        """
        self.country_data = country_data
        self.ground_truth = ground_truth

        # Total unique S1 entities across all countries evaluated
        all_unique = set()
        for c, data in country_data.items():
            all_unique.update(data["unique_s1"])
        self.all_unique_s1 = all_unique
        self.total_unique_s1 = len(all_unique)

    @classmethod
    def from_holdout_pairs(
        cls,
        holdout_pairs_by_country: Dict[str, pl.DataFrame],
        probabilities_by_country: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        s1_col: str = SOURCE1_ID_COL,
        cand_col: str = CANDIDATE_ID_COL,
    ) -> "HoldoutEvaluationDataset":
        """
        Construct evaluation container from holdout DataFrames and precomputed probabilities.
        """
        country_data = {}

        for country, df in holdout_pairs_by_country.items():
            if len(df) == 0:
                continue

            s1_arr = df[s1_col].to_numpy()
            cand_arr = df[cand_col].to_numpy()
            probs = probabilities_by_country[country].astype(np.float32)

            unique_s1 = np.unique(s1_arr).tolist()
            country_data[country] = {
                "s1_ids": s1_arr,
                "cand_ids": cand_arr,
                "probabilities": probs,
                "unique_s1": unique_s1,
            }

        return cls(country_data=country_data, ground_truth=ground_truth)

    def evaluate_thresholds(self, thresholds: Dict[str, float]) -> float:
        """
        Evaluate Macro F0.5 across all countries using country-specific thresholds.
        """
        all_scores: List[float] = []

        for country, data in self.country_data.items():
            tau = thresholds.get(country, DEFAULT_THRESHOLD)
            s1_arr = data["s1_ids"]
            cand_arr = data["cand_ids"]
            probs = data["probabilities"]
            unique_s1 = data["unique_s1"]

            # Filter candidate pairs that meet or exceed threshold tau
            mask = probs >= tau
            filtered_s1 = s1_arr[mask]
            filtered_cands = cand_arr[mask]

            # Build prediction dictionary for this country
            pred_dict: Dict[str, Set[str]] = defaultdict(set)
            for s1, c in zip(filtered_s1, filtered_cands):
                pred_dict[s1].add(c)

            # Score each unique S1 entity in this country
            for s1_id in unique_s1:
                true_set = self.ground_truth.get(s1_id, set())
                pred_set = pred_dict.get(s1_id, set())
                score = compute_single_entity_f05(true_set, pred_set)
                all_scores.append(score)

        if not all_scores:
            return 0.0

        return float(np.mean(all_scores))


# ==============================================================================
# 4. 3D Bayesian Optuna Threshold Optimization
# ==============================================================================
def optimize_3d_thresholds(
    holdout_eval_data: HoldoutEvaluationDataset,
    n_trials: int = 100,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Executes 3D Bayesian search (tau_US, tau_India, tau_France) via Optuna TPE sampler,
    restricted strictly to the 20% disjoint holdout dataset.

    Parameters
    ----------
    holdout_eval_data : HoldoutEvaluationDataset
        Container with precomputed probabilities and ground truth on holdout.
    n_trials : int
        Number of Bayesian optimization trials (default 100).
    seed : int
        Deterministic random seed.
    output_path : Optional[Union[str, Path]]
        Destination path to save models/thresholds.json.

    Returns
    -------
    Dict[str, float]
        Dictionary with optimal thresholds: {'US': tau_us, 'India': tau_india, 'France': tau_france}.
    """
    logger.info(
        f"Starting 3D Threshold Optimization ({n_trials} trials, seed={seed}) | "
        f"Search space: [{MIN_THRESHOLD}, {MAX_THRESHOLD}]^3 | "
        f"Total holdout entities: {holdout_eval_data.total_unique_s1:,}"
    )

    if optuna is None:
        logger.info(
            f"Optuna is not installed. Executing deterministic threshold optimization fallback "
            f"({n_trials} evaluations, seed={seed})..."
        )
        rng = np.random.RandomState(seed)
        best_score = -1.0
        best_params = {
            "tau_us": DEFAULT_THRESHOLD,
            "tau_india": DEFAULT_THRESHOLD,
            "tau_france": DEFAULT_THRESHOLD,
        }
        for _ in range(n_trials):
            t_us = float(rng.uniform(MIN_THRESHOLD, MAX_THRESHOLD))
            t_in = float(rng.uniform(MIN_THRESHOLD, MAX_THRESHOLD))
            t_fr = float(rng.uniform(MIN_THRESHOLD, MAX_THRESHOLD))
            trial_thresholds = {"US": t_us, "India": t_in, "France": t_fr}
            macro_f05 = holdout_eval_data.evaluate_thresholds(trial_thresholds)
            if macro_f05 > best_score:
                best_score = macro_f05
                best_params = {"tau_us": t_us, "tau_india": t_in, "tau_france": t_fr}
        best_value = best_score
    else:
        def objective(trial: optuna.Trial) -> float:
            # Suggest thresholds in [0.40, 0.95] for US, India, and France
            tau_us = trial.suggest_float("tau_us", MIN_THRESHOLD, MAX_THRESHOLD)
            tau_india = trial.suggest_float("tau_india", MIN_THRESHOLD, MAX_THRESHOLD)
            tau_france = trial.suggest_float("tau_france", MIN_THRESHOLD, MAX_THRESHOLD)

            trial_thresholds = {
                "US": tau_us,
                "India": tau_india,
                "France": tau_france,
            }

            # Fast vectorized holdout evaluation
            macro_f05 = holdout_eval_data.evaluate_thresholds(trial_thresholds)
            return macro_f05

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = float(best_trial.value)

    tau_us = float(best_params.get("tau_us", DEFAULT_THRESHOLD))
    tau_india = float(best_params.get("tau_india", DEFAULT_THRESHOLD))
    tau_france = float(best_params.get("tau_france", DEFAULT_THRESHOLD))

    # If France was not represented in holdout data, calibrate France conservatively
    # based on US and India best parameters
    if "France" not in holdout_eval_data.country_data:
        tau_france = float(np.clip((tau_us + tau_india) / 2.0, MIN_THRESHOLD, MAX_THRESHOLD))
        logger.info(
            f"France data not present in holdout set. "
            f"Setting open-set tau_France to mean of US and India: {tau_france:.4f}"
        )

    best_thresholds = {
        "US": round(tau_us, 4),
        "India": round(tau_india, 4),
        "France": round(tau_france, 4),
        # Include prefixed keys for compatibility with various callers
        "tau_US": round(tau_us, 4),
        "tau_India": round(tau_india, 4),
        "tau_France": round(tau_france, 4),
        "best_holdout_macro_f05": round(float(best_value), 5),
    }

    logger.info(
        f"3D Threshold Optimization Complete! Best Holdout Macro F0.5: {best_value:.5f} | "
        f"Optimal Thresholds: US={best_thresholds['US']}, "
        f"India={best_thresholds['India']}, France={best_thresholds['France']}"
    )

    # Persist thresholds to JSON
    target_path = Path(output_path or (MODELS_DIR / "thresholds.json"))
    save_thresholds(best_thresholds, target_path)

    return best_thresholds


# ==============================================================================
# 5. Threshold Persistence and Loading
# ==============================================================================
def save_thresholds(
    thresholds: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Saves optimal threshold dictionary to models/thresholds.json.
    """
    target_path = Path(output_path or (MODELS_DIR / "thresholds.json"))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    logger.info(f"Thresholds saved successfully to {target_path}.")
    return target_path


def load_thresholds(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Loads threshold dictionary from models/thresholds.json.
    Falls back to safe defaults if file is missing.
    """
    target_path = Path(path or (MODELS_DIR / "thresholds.json"))
    defaults = {
        "US": DEFAULT_THRESHOLD,
        "India": DEFAULT_THRESHOLD,
        "France": DEFAULT_THRESHOLD,
        "tau_US": DEFAULT_THRESHOLD,
        "tau_India": DEFAULT_THRESHOLD,
        "tau_France": DEFAULT_THRESHOLD,
    }

    if not target_path.exists():
        logger.warning(f"Threshold file not found at {target_path}. Returning defaults: {defaults}")
        return defaults

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize keys
    result = dict(defaults)
    for k, v in data.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)

    return result


def load_country_holdout_features(
    country: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """
    Convenience function to load holdout features Parquet for a specific country:
    data_processed/train/{country}/features_holdout.parquet.
    """
    base = Path(data_dir or PROCESSED_DATA_DIR) / "train" / country / "features_holdout.parquet"
    if not base.exists():
        raise FileNotFoundError(f"Holdout features Parquet file not found at: {base}")
    return pl.read_parquet(base)

