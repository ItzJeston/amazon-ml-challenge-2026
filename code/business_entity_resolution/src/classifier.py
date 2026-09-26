"""
classifier.py — Stage 3: LightGBM Classifier
Amazon Business Entity Resolution Challenge

Loads pre-extracted train/val features, trains LightGBMClassifier, runs
threshold search (0.50 → 0.88, step 0.02) and reports the best Macro F0.5.

Usage:
    python classifier.py           # uses val_candidates_quick.tsv features
    python classifier.py --full    # uses val_candidates_full.tsv features
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SRC_DIR, '../../..'))
DATA_DIR   = os.path.join(BASE_DIR, '6ab10eb3b23ba_student_resource',
                          'student_resource', 'dataset', 'train')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(prefix: str) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """
    Load pre-saved feature array, optional labels, and pair IDs.
    prefix is either 'train' or 'val'.
    """
    X_path     = os.path.join(OUTPUT_DIR, f'features_{prefix}.npy')
    y_path     = os.path.join(OUTPUT_DIR, f'labels_{prefix}.npy')
    pairs_path = os.path.join(OUTPUT_DIR, f'pairs_{prefix}.tsv')

    X = np.load(X_path)
    y = np.load(y_path) if os.path.exists(y_path) else None
    pairs = pd.read_csv(pairs_path, sep='\t', dtype=str)
    log.info(f"Loaded {prefix} — X={X.shape}  y={'None' if y is None else y.shape}")
    return X, y, pairs


def build_preds_dict(
    pairs_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, set]:
    """
    Convert per-pair probabilities → {s1_id → set of predicted matched IDs}.
    Only pairs with proba >= threshold are included in the prediction set.
    """
    preds: dict[str, set] = {}
    for s1_id in pairs_df['source1_entity_id'].unique():
        preds[s1_id] = set()

    mask = proba >= threshold
    matched = pairs_df[mask]
    for row in matched.itertuples(index=False):
        preds[row.source1_entity_id].add(row.candidate_entity_id)

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, SRC_DIR)
    from metrics import load_ground_truth, compute_macro_f05

    gt_path = os.path.join(DATA_DIR, 'train_ground_truth.tsv')

    # ---- 1. Check if features are already extracted -----------------------
    train_feat_path = os.path.join(OUTPUT_DIR, 'features_train.npy')
    val_feat_path   = os.path.join(OUTPUT_DIR, 'features_val.npy')

    if not os.path.exists(train_feat_path) or not os.path.exists(val_feat_path):
        log.info("Feature files not found — running features.py first …")
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, 'features.py')] +
            (['--full'] if '--full' in sys.argv else []),
            check=True
        )

    # ---- 2. Load features --------------------------------------------------
    log.info("=== Loading features ===")
    X_train, y_train, pairs_train = load_features('train')
    X_val,   _,       pairs_val   = load_features('val')

    log.info(f"Train class balance — positives: {y_train.sum():,}  "
             f"negatives: {(y_train == 0).sum():,}")

    # ---- 3. Train LightGBM -------------------------------------------------
    log.info("=== Training LGBMClassifier ===")
    clf = LGBMClassifier(
        n_estimators  = 300,
        learning_rate = 0.05,
        num_leaves    = 31,
        random_state  = 42,
        n_jobs        = -1,
        verbose       = -1,       # suppress LightGBM internal logs
    )
    clf.fit(X_train, y_train)
    log.info("Training complete ✅")

    # Feature importance
    feat_names = [
        'name_token_sort_ratio', 'name_jaro_winkler',
        'address_token_set_ratio', 'address_levenshtein',
        'number_overlap', 'len_diff_name', 'len_diff_addr',
        'blocking_rank',
    ]
    importances = sorted(
        zip(feat_names, clf.feature_importances_),
        key=lambda x: -x[1]
    )
    log.info("Feature importances:")
    for name, imp in importances:
        log.info(f"  {name:<30s} {imp:>6.0f}")

    # ---- 4. Inference on validation set ------------------------------------
    log.info("=== Running inference on validation set ===")
    proba = clf.predict_proba(X_val)[:, 1].astype(np.float32)

    # ---- 5. Load ground truth (filtered to val entities) ------------------
    log.info("Loading ground truth …")
    gt_dict  = load_ground_truth(gt_path)
    val_ids  = set(pairs_val['source1_entity_id'].unique())
    gt_val   = {k: v for k, v in gt_dict.items() if k in val_ids}
    log.info(f"GT entries for val set: {len(gt_val):,}")

    # ---- 6. Threshold search -----------------------------------------------
    log.info("=== Threshold search (0.50 → 0.88, step 0.02) ===")
    thresholds = np.arange(0.50, 0.90, 0.02)

    best_thresh = 0.50
    best_f05    = -1.0
    results     = []

    for thresh in thresholds:
        preds_dict = build_preds_dict(pairs_val, proba, float(thresh))
        f05 = compute_macro_f05(preds_dict, gt_val)
        results.append((thresh, f05))
        log.info(f"  threshold={thresh:.2f}  Macro F0.5={f05:.4f}")
        if f05 > best_f05:
            best_f05    = f05
            best_thresh = thresh

    # ---- 7. Report ----------------------------------------------------------
    log.info("=" * 55)
    log.info(f"Best threshold:     {best_thresh:.2f}")
    log.info(f"Best Macro F0.5:    {best_f05:.4f}")
    log.info("=" * 55)

    # Save threshold search results
    results_df = pd.DataFrame(results, columns=['threshold', 'macro_f05'])
    results_path = os.path.join(OUTPUT_DIR, 'threshold_search.tsv')
    results_df.to_csv(results_path, sep='\t', index=False, float_format='%.4f')
    log.info(f"Threshold search results saved → {results_path}")

    # Save best predictions
    best_preds = build_preds_dict(pairs_val, proba, best_thresh)
    pred_rows = [
        {'source1_entity_id': s1_id,
         'matched_entity_ids': ','.join(sorted(ids))}
        for s1_id, ids in best_preds.items()
    ]
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(OUTPUT_DIR, 'val_predictions.tsv')
    pred_df.to_csv(pred_path, sep='\t', index=False)
    log.info(f"Validation predictions saved → {pred_path}")
    log.info("classifier.py complete ✅")
