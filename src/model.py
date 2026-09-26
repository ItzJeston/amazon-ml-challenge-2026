"""
Model module for business entity resolution.
Implements pairwise binary classification using LightGBM,
threshold calibration on realistic imbalanced candidate distributions,
1-to-1 cardinality assignment simulation, and model persistence.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
import joblib

from src.preprocess import clean_name, clean_address, extract_digits, load_partition
from src.blocking import CountryInvertedIndex
from src.features import compute_pair_features, check_house_conflict, FEATURE_NAMES


def compute_macro_f05(
    ground_truth_map: Dict[str, Set[str]],
    predictions_map: Dict[str, Set[str]],
) -> float:
    """Compute official macro-average F_0.5 score across all S1 entities.
    
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Rules:
        - Singletons (no true matches):
            - 1.0 if predicted empty set
            - 0.0 if any match predicted
        - Non-singletons:
            - Standard F_0.5 with precision weighted 2x over recall
    """
    scores = []
    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())

        if not true_set:
            # Singleton entity
            scores.append(1.0 if not pred_set else 0.0)
        else:
            if not pred_set:
                scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                if tp == 0:
                    scores.append(0.0)
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    denom = 0.25 * prec + rec
                    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
                    scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


def train_pairwise_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    random_state: int = 42,
) -> LGBMClassifier:
    """Train LightGBM binary classifier for pairwise entity matching."""
    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def optimize_threshold_imbalanced(
    model: LGBMClassifier,
    val_s1_dict: Dict[str, Tuple[str, str, List[str], str]],
    val_gt_map: Dict[str, Set[str]],
    indices: Dict[str, CountryInvertedIndex],
    s23_clean: Dict[str, Tuple[str, str, List[str]]],
    thresh_range: np.ndarray = np.arange(0.70, 0.94, 0.02),
) -> Tuple[float, float]:
    """Find the optimal probability threshold on realistic imbalanced candidate sets (1:10 ratio)
    incorporating house number conflict penalties and global 1-to-1 cardinality assignment.
    
    Returns:
        (best_threshold, best_f05_score)
    """
    print("\nBuilding realistic imbalanced validation set from blocking candidates...")
    val_pairs = []
    
    for s1_id, (cn1, ca1, cd1, country) in val_s1_dict.items():
        idx = indices.get(country)
        cands = idx.query(cn1, ca1, cd1) if idx else []
        for cid in cands:
            if cid in s23_clean:
                cn2, ca2, cd2 = s23_clean[cid]
                feats = compute_pair_features(cn1, ca1, cd1, cid, cn2, ca2, cd2)
                has_conflict = check_house_conflict(cd1, cd2)
                val_pairs.append((s1_id, cid, feats, has_conflict))

    print(f"Realistic validation pairs: {len(val_pairs)} across {len(val_s1_dict)} entities.")
    if not val_pairs:
        return 0.85, 0.0

    X_val = np.array([p[2] for p in val_pairs], dtype=np.float32)
    raw_probs = model.predict_proba(X_val)[:, 1]

    # Apply house number conflict penalty (50% probability discount)
    adjusted_probs = np.copy(raw_probs)
    for i, (_, _, _, has_conflict) in enumerate(val_pairs):
        if has_conflict:
            adjusted_probs[i] *= 0.5

    best_thresh = 0.85
    best_score = -1.0

    print("\nTuning threshold with 1-to-1 cardinality post-processing:")
    for thresh in thresh_range:
        # Collect passing candidate links
        passing = []
        for (s1_id, cid, _, _), prob in zip(val_pairs, adjusted_probs):
            if prob >= thresh:
                passing.append((prob, s1_id, cid))

        # Greedy 1-to-1 bipartite assignment by probability score
        passing.sort(key=lambda x: x[0], reverse=True)
        assigned_cands = set()
        pred_map = {s1: set() for s1 in val_gt_map}
        for prob, s1_id, cid in passing:
            if cid not in assigned_cands:
                assigned_cands.add(cid)
                pred_map[s1_id].add(cid)

        score = compute_macro_f05(val_gt_map, pred_map)
        print(f"  Threshold: {thresh:.2f} -> Macro F_0.5: {score:.4f} (Matched links: {len(assigned_cands)})")

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    # If top scores are tied or very close, prefer higher threshold (e.g. 0.84+) for precision
    print(f"\nOptimal Calibrated Threshold: {best_thresh:.2f} (Macro F_0.5 = {best_score:.4f})")
    return best_thresh, best_score


def save_model(model: LGBMClassifier, threshold: float, save_dir: str = "models"):
    """Persist trained model and calibrated threshold to disk."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model": model,
        "threshold": threshold,
        "feature_names": FEATURE_NAMES,
    }
    model_path = os.path.join(save_dir, "matcher_lgbm.joblib")
    joblib.dump(payload, model_path)
    print(f"Model and threshold saved to {model_path}")


def load_model(save_dir: str = "models") -> Tuple[LGBMClassifier, float]:
    """Load persisted model and calibrated threshold from disk."""
    model_path = os.path.join(save_dir, "matcher_lgbm.joblib")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    payload = joblib.load(model_path)
    return payload["model"], payload["threshold"]


def train_and_evaluate(
    train_dir: str = "dataset/train",
    save_dir: str = "models",
    sample_size: int = 50000,
) -> Tuple[LGBMClassifier, float, float]:
    """Build dataset from train ground truth, train LightGBM, and calibrate threshold."""
    print("=" * 60)
    print("STARTING ENHANCED MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Read ground truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print(f"Reading ground truth from {gt_path}...")
    gt_df = pl.read_csv(gt_path, separator="\t", n_rows=sample_size)
    
    gt_map: Dict[str, Set[str]] = {}
    all_matched_ids: Set[str] = set()
    singletons_count = 0
    
    for row in gt_df.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if matches_str:
            m_set = set(matches_str.split(','))
            gt_map[s1_id] = m_set
            all_matched_ids.update(m_set)
        else:
            gt_map[s1_id] = set()
            singletons_count += 1

    print(f"Sampled {len(gt_map)} S1 entities ({singletons_count} singletons, {len(all_matched_ids)} total matched IDs).")

    # 2. Load S1, S2, S3 records
    s1_ids = list(gt_map.keys())
    s1_df = (
        pl.scan_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
        .filter(pl.col("entity_id").is_in(s1_ids))
        .collect()
    )

    s2_matched_ids = [m for m in all_matched_ids if m.startswith("S2-")]
    s3_matched_ids = [m for m in all_matched_ids if m.startswith("S3-")]

    s2_pos = (
        pl.scan_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")
        .filter(pl.col("entity_id").is_in(s2_matched_ids))
        .collect()
    )
    s3_pos = (
        pl.scan_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t")
        .filter(pl.col("entity_id").is_in(s3_matched_ids))
        .collect()
    )

    # Distractors
    s2_dist = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=60000)
    s3_dist = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=60000)

    s2_all = pl.concat([s2_pos, s2_dist]).unique("entity_id")
    s3_all = pl.concat([s3_pos, s3_dist]).unique("entity_id")
    print(f"Corpus size: S1={len(s1_df)}, S2={len(s2_all)}, S3={len(s3_all)}")

    # 3. Clean records and build country inverted indices
    s1_clean: Dict[str, Tuple[str, str, List[str], str]] = {}
    for r in s1_df.iter_rows(named=True):
        cn = clean_name(r['business_name'])
        ca = clean_address(r['business_address'])
        cd = extract_digits(ca)
        s1_clean[r['entity_id']] = (cn, ca, cd, r['country'])

    s23_clean: Dict[str, Tuple[str, str, List[str]]] = {}
    for df in [s2_all, s3_all]:
        for r in df.iter_rows(named=True):
            cn = clean_name(r['business_name'])
            ca = clean_address(r['business_address'])
            cd = extract_digits(ca)
            s23_clean[r['entity_id']] = (cn, ca, cd)

    indices: Dict[str, CountryInvertedIndex] = {}
    for country in ["US", "India"]:
        s2_c = s2_all.filter(pl.col("country") == country)
        s3_c = s3_all.filter(pl.col("country") == country)
        idx = CountryInvertedIndex(max_bucket_size=300, max_candidates=12)
        idx.build(s2_c, s3_c)
        indices[country] = idx

    # 4. Generate pairs for training
    print("Generating training candidate pairs and computing features...")
    X_list = []
    y_list = []
    pairs_meta = []

    for s1_id, (cn1, ca1, cd1, country) in s1_clean.items():
        true_set = gt_map[s1_id]
        idx = indices.get(country)
        cands = idx.query(cn1, ca1, cd1) if idx else []
        all_candidates = set(cands) | true_set

        for cid in all_candidates:
            if cid not in s23_clean:
                continue
            cn2, ca2, cd2 = s23_clean[cid]
            feats = compute_pair_features(cn1, ca1, cd1, cid, cn2, ca2, cd2)
            label = 1 if cid in true_set else 0

            X_list.append(feats)
            y_list.append(label)
            pairs_meta.append((s1_id, cid))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Total pairs generated: {len(X)} | Positives: {int(np.sum(y))} | Negatives: {int(len(y) - np.sum(y))}")

    # 5. Split train / validation sets by entity (75% train, 25% val)
    unique_s1 = list(s1_clean.keys())
    random.seed(42)
    random.shuffle(unique_s1)
    split_pt = int(0.75 * len(unique_s1))
    train_s1_set = set(unique_s1[:split_pt])
    val_s1_set = set(unique_s1[split_pt:])

    train_mask = np.array([p[0] in train_s1_set for p in pairs_meta])
    X_train, y_train = X[train_mask], y[train_mask]

    print(f"Train pairs: {len(X_train)} | Val entities: {len(val_s1_set)}")

    # 6. Fit LightGBM Classifier
    print("Fitting LightGBM classifier...")
    model = train_pairwise_model(X_train, y_train)

    # 7. Evaluate and optimize threshold on realistic imbalanced candidates
    val_s1_dict = {s1: s1_clean[s1] for s1 in val_s1_set}
    val_gt_map = {s1: gt_map[s1] for s1 in val_s1_set}
    best_thresh, best_score = optimize_threshold_imbalanced(
        model=model,
        val_s1_dict=val_s1_dict,
        val_gt_map=val_gt_map,
        indices=indices,
        s23_clean=s23_clean,
        thresh_range=np.arange(0.74, 0.94, 0.02),
    )

    # 8. Save model and threshold
    save_model(model, best_thresh, save_dir)
    return model, best_thresh, best_score


if __name__ == "__main__":
    train_and_evaluate()
