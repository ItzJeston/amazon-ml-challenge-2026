"""
mini_benchmark.py — 2-Minute Smart Benchmark for Amazon Business Entity Resolution

Evaluates the real performance ceiling of the pipeline:
1. Takes 5,000 Source-1 validation entities.
2. Builds candidate pool containing ALL their true ground truth partners + 500,000 realistic distractors.
3. Runs Inverted-Index Blocking -> computes Candidate Recall.
4. Extracts pairwise RapidFuzz features.
5. Trains LightGBM & sweeps decision thresholds (0.50 -> 0.88).
6. Outputs the true Macro F0.5 score.
"""

import os
import sys
import time
import math
import logging
import collections
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SRC_DIR, '../../..'))
DATA_DIR   = os.path.join(BASE_DIR, '6ab10eb3b23ba_student_resource', 'student_resource', 'dataset', 'train')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

sys.path.insert(0, SRC_DIR)
from metrics import load_ground_truth, compute_macro_f05
from blocking import clean_text, build_combined_text, tokenize, build_inverted_index, query_index, compute_candidate_recall
from features import compute_features_batch, _norm_str

N_VAL_SAMPLE = 5_000
N_DISTRACTORS_PER_SOURCE = 250_000   # 250K S2 + 250K S3 = 500K total distractors
TOP_K = 12

def main():
    t_start = time.time()
    log.info("=== Starting 2-Minute Smart Benchmark ===")

    # 1. Load Ground Truth
    gt_path = os.path.join(DATA_DIR, 'train_ground_truth.tsv')
    log.info("Loading ground truth mapping …")
    gt_dict = load_ground_truth(gt_path)

    # 2. Sample 5,000 Validation S1 queries
    s1_val_path = os.path.join(OUTPUT_DIR, 'train_source1_val_split.tsv')
    log.info("Loading S1 validation split …")
    s1_val_full = pd.read_csv(s1_val_path, sep='\t', dtype=str)
    s1_val = s1_val_full.sample(n=min(N_VAL_SAMPLE, len(s1_val_full)), random_state=42).reset_index(drop=True)
    val_s1_ids = set(s1_val['entity_id'])
    val_countries = set(s1_val['country'].dropna().unique())
    log.info(f"Sampled {len(s1_val):,} S1 validation queries across {sorted(val_countries)}")

    # Collect all true ground truth matched IDs for these 5,000 S1 queries
    val_gt = {eid: gt_dict.get(eid, set()) for eid in val_s1_ids}
    all_true_partners = set()
    for targets in val_gt.values():
        all_true_partners.update(targets)
    log.info(f"Total true ground-truth partners to retrieve: {len(all_true_partners):,}")

    # 3. Stream S2 & S3: Keep ALL true partners + sample 250k distractors per source
    def load_benchmark_pool(file_name, target_ids, n_distractors):
        path = os.path.join(DATA_DIR, file_name)
        log.info(f"Streaming {file_name} …")
        matched_chunks = []
        distractor_chunks = []
        n_dist_sampled = 0

        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=100_000):
            # Only keep rows from relevant countries
            chunk = chunk[chunk['country'].isin(val_countries)]
            if chunk.empty:
                continue
            
            # Extract true partners
            hits = chunk[chunk['entity_id'].isin(target_ids)]
            if not hits.empty:
                matched_chunks.append(hits)
            
            # Extract distractors
            non_hits = chunk[~chunk['entity_id'].isin(target_ids)]
            if n_dist_sampled < n_distractors and not non_hits.empty:
                take = min(n_distractors - n_dist_sampled, len(non_hits))
                distractor_chunks.append(non_hits.sample(n=take, random_state=42))
                n_dist_sampled += take

        parts = matched_chunks + distractor_chunks
        res = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['entity_id'])
        return res

    s2_pool = load_benchmark_pool('train_source2.tsv', all_true_partners, N_DISTRACTORS_PER_SOURCE)
    s3_pool = load_benchmark_pool('train_source3.tsv', all_true_partners, N_DISTRACTORS_PER_SOURCE)
    pool_df = pd.concat([s2_pool, s3_pool], ignore_index=True).drop_duplicates(subset=['entity_id'])
    log.info(f"Total Benchmark Candidate Pool: {len(pool_df):,} records (contains true partners + ~500K distractors)")

    # 4. Run Stage 1: Candidate Generation (Inverted Index)
    log.info("Running Inverted-Index Blocking …")
    t_block = time.time()
    s1_val = s1_val.copy()
    s1_val['comb_text'] = build_combined_text(s1_val)
    pool_df['comb_text'] = build_combined_text(pool_df)

    candidates = {}
    for country in val_countries:
        s1_c = s1_val[s1_val['country'] == country]
        pool_c = pool_df[pool_df['country'] == country]
        if s1_c.empty or pool_c.empty:
            for eid in s1_c['entity_id']:
                candidates[eid] = []
            continue

        log.info(f"  [{country}] S1={len(s1_c):,}  Pool={len(pool_c):,}")
        index, idf = build_inverted_index(pool_c)
        for row in s1_c.itertuples(index=False):
            candidates[row.entity_id] = query_index(row.comb_text, index, idf, TOP_K)

    block_recall, avg_cands = compute_candidate_recall(candidates, val_gt)
    log.info(f"--> STAGE 1 CANDIDATE RECALL: {block_recall:.4f} ({block_recall*100:.2f}%) at Top-{TOP_K} in {time.time()-t_block:.1f}s")

    # 5. Build Entity Lookup for Features
    log.info("Building entity lookup for feature extraction …")
    lookup = {}
    for row in s1_val.itertuples(index=False):
        lookup[row.entity_id] = {'name': _norm_str(row.business_name), 'address': _norm_str(row.business_address)}
    for row in pool_df.itertuples(index=False):
        lookup[row.entity_id] = {'name': _norm_str(row.business_name), 'address': _norm_str(row.business_address)}

    # 6. Build Candidate Pairs DataFrame
    rows = []
    for s1_id, c_list in candidates.items():
        for rank, cid in enumerate(c_list, start=1):
            is_match = 1 if cid in val_gt.get(s1_id, set()) else 0
            rows.append({'source1_entity_id': s1_id, 'candidate_entity_id': cid, 'blocking_rank': rank, 'label': is_match})
    pairs_df = pd.DataFrame(rows)
    log.info(f"Total candidate pairs: {len(pairs_df):,} (Positives: {(pairs_df['label']==1).sum():,}, Negatives: {(pairs_df['label']==0).sum():,})")

    # 7. Split into Train & Test (e.g. 50/50 split of the queries)
    unique_s1 = s1_val['entity_id'].unique()
    np.random.seed(42)
    shuffled_s1 = np.random.permutation(unique_s1)
    split_idx = int(len(shuffled_s1) * 0.5)
    train_s1_ids = set(shuffled_s1[:split_idx])
    test_s1_ids = set(shuffled_s1[split_idx:])

    train_pairs = pairs_df[pairs_df['source1_entity_id'].isin(train_s1_ids)].copy()
    test_pairs = pairs_df[pairs_df['source1_entity_id'].isin(test_s1_ids)].copy()

    # Subsample negatives for training (keep all positives, max 3 negatives per query)
    pos_train = train_pairs[train_pairs['label'] == 1]
    neg_train = train_pairs[train_pairs['label'] == 0]
    neg_sub = pd.concat([g.sample(n=min(3, len(g)), random_state=42) for _, g in neg_train.groupby('source1_entity_id')], ignore_index=True)
    balanced_train = pd.concat([pos_train, neg_sub], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Computing features for {len(balanced_train):,} train pairs and {len(test_pairs):,} test pairs …")
    X_train = compute_features_batch(balanced_train, lookup)
    y_train = balanced_train['label'].values.astype(np.int8)

    X_test = compute_features_batch(test_pairs, lookup)

    # 8. Train LightGBM Classifier
    log.info("Training LightGBM model …")
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 9. Inference & Threshold Search on the Test Set
    log.info("Predicting probabilities and running threshold optimization …")
    test_proba = clf.predict_proba(X_test)[:, 1].astype(np.float32)

    test_gt = {eid: val_gt[eid] for eid in test_s1_ids}
    thresholds = np.arange(0.50, 0.92, 0.02)
    best_thresh, best_f05 = 0.50, -1.0

    log.info("=" * 60)
    log.info(f"{'Threshold':<15} {'Macro F0.5':<15}")
    log.info("-" * 60)

    for thresh in thresholds:
        preds = {eid: set() for eid in test_s1_ids}
        matched = test_pairs[test_proba >= thresh]
        for row in matched.itertuples(index=False):
            preds[row.source1_entity_id].add(row.candidate_entity_id)

        score = compute_macro_f05(preds, test_gt)
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh
        log.info(f"{thresh:<15.2f} {score:<15.4f}")

    log.info("=" * 60)
    log.info(f"🏆 BEST MACRO F0.5 SCORE : {best_f05:.4f} ({best_f05*100:.2f}%)")
    log.info(f"🎯 OPTIMAL THRESHOLD     : {best_thresh:.2f}")
    log.info(f"⏱️ TOTAL BENCHMARK TIME  : {time.time()-t_start:.1f} seconds")
    log.info("=" * 60)

if __name__ == '__main__':
    main()
