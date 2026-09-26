"""
test_inference.py — Test Set Inference & Submission Packaging
Amazon Business Entity Resolution Challenge

End-to-End Test Workflow:
1. Stage 1 (Blocking): Inverted index search on test_source1 vs test_source2 & test_source3.
   Generates `output/candidate_pairs.tsv`.
2. Stage 2 (Features): Extract 8 RapidFuzz similarity features on test candidate pairs.
3. Stage 3 (Inference): Load pre-trained LightGBM model, apply optimal decision threshold (0.88).
   Generates `output/matching_results.tsv`.
4. Stage 4 (Validation): Runs official `utils/validate_submission.py`.
5. Stage 5 (Zip): Creates `output/submission.zip`.
"""

import os
import sys
import time
import zipfile
import subprocess
import logging
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SRC_DIR, '../../..'))
TEST_DIR   = os.path.join(BASE_DIR, '6ab10eb3b23ba_student_resource', 'student_resource', 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

sys.path.insert(0, SRC_DIR)
from blocking import clean_text, build_combined_text, tokenize, build_inverted_index, query_index, read_filtered_tsv
from features import compute_features_batch, _norm_str

TOP_K = 12
OPTIMAL_THRESHOLD = 0.88

def main():
    t_start = time.time()
    log.info("🚀 STARTING OFFICIAL TEST SET INFERENCE PIPELINE")

    s1_test_path = os.path.join(TEST_DIR, 'test_source1.tsv')
    s2_test_path = os.path.join(TEST_DIR, 'test_source2.tsv')
    s3_test_path = os.path.join(TEST_DIR, 'test_source3.tsv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    candidate_tsv_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    matching_tsv_path  = os.path.join(OUTPUT_DIR, 'matching_results.tsv')

    # =========================================================================
    # STAGE 1: TEST CANDIDATE BLOCKING
    # =========================================================================
    log.info("=== STAGE 1: TEST CANDIDATE BLOCKING ===")
    log.info(f"Loading Test Source 1 from {s1_test_path} …")
    df_s1 = pd.read_csv(s1_test_path, sep='\t', dtype=str)
    all_s1_ids = list(df_s1['entity_id'].values)
    log.info(f"Total Test Source 1 Queries: {len(df_s1):,}")

    df_s1['comb_text'] = build_combined_text(df_s1)
    test_countries = sorted(df_s1['country'].dropna().unique())
    log.info(f"Test Countries: {test_countries}")

    candidates_dict = {eid: [] for eid in all_s1_ids}
    flat_candidate_pairs = []

    for country in test_countries:
        sub_s1 = df_s1[df_s1['country'] == country]
        if sub_s1.empty:
            continue
        log.info(f"--- Processing Country [{country}] (Queries: {len(sub_s1):,}) ---")
        
        log.info(f"  Streaming Test S2 & S3 pool for [{country}] …")
        s2_c = read_filtered_tsv(s2_test_path, {country})
        s3_c = read_filtered_tsv(s3_test_path, {country})
        pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c

        log.info(f"  [{country}] S1={len(sub_s1):,}  Pool={len(pool_c):,}")
        index, idf = build_inverted_index(pool_c)
        log.info(f"  [{country}] Inverted index ready — {len(index):,} tokens. Querying …")
        del pool_c

        for row in sub_s1.itertuples(index=False):
            c_list = query_index(row.comb_text, index, idf, TOP_K)
            candidates_dict[row.entity_id] = c_list
            for rank, cid in enumerate(c_list, start=1):
                flat_candidate_pairs.append({
                    'source1_entity_id': row.entity_id,
                    'candidate_entity_id': cid,
                    'blocking_rank': rank
                })
        del index, idf

    # Format and save candidate_pairs.tsv: source1_entity_id \t candidate_entity_ids
    log.info(f"Writing official {candidate_tsv_path} …")
    cand_rows = [
        {'source1_entity_id': s1_id, 'candidate_entity_ids': ','.join(candidates_dict[s1_id])}
        for s1_id in all_s1_ids
    ]
    pd.DataFrame(cand_rows).to_csv(candidate_tsv_path, sep='\t', index=False)
    log.info(f"Saved {len(cand_rows):,} rows to {candidate_tsv_path}")

    # =========================================================================
    # STAGE 2: TEST FEATURE EXTRACTION
    # =========================================================================
    log.info("=== STAGE 2: TEST FEATURE EXTRACTION ===")
    pairs_df = pd.DataFrame(flat_candidate_pairs)
    log.info(f"Total candidate pairs to evaluate: {len(pairs_df):,}")

    needed_ids = set(pairs_df['source1_entity_id']).union(set(pairs_df['candidate_entity_id']))
    log.info(f"Streaming text lookups for {len(needed_ids):,} unique entities …")

    lookup = {}
    for path in (s1_test_path, s2_test_path, s3_test_path):
        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=100_000):
            sub = chunk[chunk['entity_id'].isin(needed_ids)]
            for r in sub.itertuples(index=False):
                lookup[r.entity_id] = {'name': _norm_str(r.business_name), 'address': _norm_str(r.business_address)}

    log.info("Computing RapidFuzz similarity features in chunks …")
    CHUNK_SIZE = 100_000
    all_X = []
    for start in range(0, len(pairs_df), CHUNK_SIZE):
        chunk = pairs_df.iloc[start:start+CHUNK_SIZE]
        all_X.append(compute_features_batch(chunk, lookup))
    X_test = np.vstack(all_X).astype(np.float32)
    del all_X, lookup

    # =========================================================================
    # STAGE 3: CLASSIFIER INFERENCE & THRESHOLDING
    # =========================================================================
    log.info("=== STAGE 3: CLASSIFIER INFERENCE ===")
    log.info("Loading training matrices features_train.npy & labels_train.npy …")
    X_train = np.load(os.path.join(OUTPUT_DIR, 'features_train.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'labels_train.npy'))

    log.info(f"Training LightGBM on {len(X_train):,} training pairs …")
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(X_train, y_train)

    log.info(f"Predicting probabilities on {len(X_test):,} test candidate pairs …")
    proba = clf.predict_proba(X_test)[:, 1].astype(np.float32)

    log.info(f"Applying optimal threshold {OPTIMAL_THRESHOLD:.2f} …")
    predictions = {eid: [] for eid in all_s1_ids}
    mask = proba >= OPTIMAL_THRESHOLD
    matched_df = pairs_df[mask]

    for row in matched_df.itertuples(index=False):
        predictions[row.source1_entity_id].append(row.candidate_entity_id)

    # Format and save matching_results.tsv: source1_entity_id \t matched_entity_ids
    log.info(f"Writing official {matching_tsv_path} …")
    match_rows = [
        {'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(predictions[s1_id])}
        for s1_id in all_s1_ids
    ]
    pd.DataFrame(match_rows).to_csv(matching_tsv_path, sep='\t', index=False)
    log.info(f"Saved {len(match_rows):,} rows to {matching_tsv_path}")

    # =========================================================================
    # STAGE 4: RUN OFFICIAL VALIDATION SCRIPT
    # =========================================================================
    log.info("=== STAGE 4: VALIDATING SUBMISSION FORMAT ===")
    validator_script = os.path.join(BASE_DIR, 'utils', 'validate_submission.py')
    val_cmd = [
        sys.executable, validator_script,
        '--matching', matching_tsv_path,
        '--candidate', candidate_tsv_path,
        '--test-dir', TEST_DIR
    ]
    val_res = subprocess.run(val_cmd)
    if val_res.returncode != 0:
        log.error("Submission validation failed! Please check errors.")
        sys.exit(val_res.returncode)

    # =========================================================================
    # STAGE 5: PACKAGE SUBMISSION ZIP
    # =========================================================================
    zip_path = os.path.join(OUTPUT_DIR, 'submission.zip')
    log.info(f"Packaging {zip_path} …")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(matching_tsv_path, arcname='matching_results.tsv')
        zipf.write(candidate_tsv_path, arcname='candidate_pairs.tsv')

    log.info("=" * 60)
    log.info(f"🎉 TEST PIPELINE COMPLETED IN {(time.time() - t_start)/60:.1f} MINUTES!")
    log.info(f"📦 FINAL SUBMISSION FILE READY: {zip_path}")
    log.info("=" * 60)

if __name__ == '__main__':
    main()
