"""
metrics.py - Evaluation utilities for the Amazon Business Entity Resolution Challenge.

Competition metric: Macro-averaged F0.5 score.
  - F0.5 weights precision twice as heavily as recall.
  - Singletons (ground truth = empty set):
      - Predict empty  → score 1.0
      - Predict any ID → score 0.0
"""

from typing import Dict, Set


def _f05_single(pred_set: Set[str], gt_set: Set[str]) -> float:
    """
    Compute the F0.5 score for a single Source-1 entity.

    Args:
        pred_set: Set of predicted matched entity IDs (from S2 / S3).
        gt_set:   Set of ground-truth matched entity IDs.

    Returns:
        F0.5 score in [0.0, 1.0].
    """
    # --- Singleton handling -------------------------------------------
    # Ground truth has no matches for this entity.
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # --- Standard F0.5 --------------------------------------------------
    if len(pred_set) == 0:
        # Nothing predicted → precision undefined (treat as 0), recall = 0
        return 0.0

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)

    if precision == 0 and recall == 0:
        return 0.0

    beta = 0.5
    beta_sq = beta ** 2
    f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f05


def compute_macro_f05(
    preds_dict: Dict[str, Set[str]],
    ground_truth_dict: Dict[str, Set[str]],
) -> float:
    """
    Compute the macro-averaged F0.5 score across all Source-1 entities.

    Every entity present in ground_truth_dict is scored. Entities in
    preds_dict that are absent from ground_truth_dict are ignored (they
    would be false positives with no ground-truth entry to compare against).

    Args:
        preds_dict:        {source1_entity_id -> set of predicted entity IDs}
        ground_truth_dict: {source1_entity_id -> set of ground-truth matched IDs}
                           An *empty* set means the entity is a singleton.

    Returns:
        Macro-averaged F0.5 score in [0.0, 1.0].
    """
    scores = []
    for s1_id, gt_set in ground_truth_dict.items():
        pred_set = preds_dict.get(s1_id, set())
        scores.append(_f05_single(pred_set, gt_set))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def load_ground_truth(ground_truth_path: str) -> Dict[str, Set[str]]:
    """
    Parse ground truth TSV into a dict for use with compute_macro_f05.

    Expected columns: source1_entity_id, matched_entity_ids
    The matched_entity_ids column is a comma-separated list of S2/S3 IDs,
    or empty for singletons.

    Args:
        ground_truth_path: Path to train_ground_truth.tsv.

    Returns:
        {source1_entity_id -> set of matched entity IDs}
    """
    import pandas as pd

    gt_df = pd.read_csv(ground_truth_path, sep='\t', dtype=str)
    gt_dict: Dict[str, Set[str]] = {}

    for _, row in gt_df.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        raw   = str(row['matched_entity_ids']).strip()
        if raw == '' or raw.lower() == 'nan':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(x.strip() for x in raw.split(',') if x.strip())

    return gt_dict


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Basic functional tests ---
    # 1. Singleton, correct (empty pred)
    assert _f05_single(set(), set()) == 1.0, "Singleton empty pred should be 1.0"

    # 2. Singleton, wrong (non-empty pred)
    assert _f05_single({'S2-1'}, set()) == 0.0, "Singleton with pred should be 0.0"

    # 3. Perfect match
    assert _f05_single({'S2-1', 'S3-2'}, {'S2-1', 'S3-2'}) == 1.0, "Perfect match"

    # 4. Partial match
    score = _f05_single({'S2-1'}, {'S2-1', 'S2-2'})
    assert 0 < score < 1.0, f"Partial match score should be between 0 and 1, got {score}"

    # 5. No overlap
    assert _f05_single({'S2-99'}, {'S2-1'}) == 0.0, "No-overlap prediction"

    # 6. Macro average
    preds = {'e1': {'S2-1'}, 'e2': set(), 'e3': {'S2-5', 'S3-9'}}
    gt    = {'e1': {'S2-1'}, 'e2': set(), 'e3': {'S2-5'}}
    macro = compute_macro_f05(preds, gt)
    print(f"Self-test macro F0.5: {macro:.4f}  (expected ≈ 0.9167)")

    print("All self-tests passed.")
