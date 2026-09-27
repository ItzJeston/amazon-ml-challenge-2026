"""
test_inference.py — Official Test Set Inference & Submission Packaging
Amazon Business Entity Resolution Challenge

End-to-End Test Workflow (Country-by-Country Memory-Optimized):
1. Trains LightGBM on existing full training matrices (features_train.npy & labels_train.npy).
2. For each country (France, India, US):
   a. Builds Inverted Index on Test S2 + S3.
   b. Queries Top-20 candidates for Test S1 entities.
   c. Builds text lookup for entities in this country.
   d. Extracts 13 RapidFuzz features in 100K-pair chunks.
   e. Predicts match probabilities with LightGBM.
   f. Applies veto (zero proba for vetoed pairs) and writes above-threshold predictions.
   g. Frees all country-specific memory before moving to the next country.
3. Runs post_process.py for pool-side dedup (each S2/S3 maps to at most one S1).
4. Writes formatted output/candidate_pairs.tsv and output/matching_results.tsv.
5. Executes utils/validate_submission.py to guarantee submission compliance.
6. Packages output/submission.zip ready for leaderboard upload.
"""

import os
import sys
import gc
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
from blocking import clean_text, build_combined_text, tokenize, build_inverted_index, query_index, read_filtered_tsv, TOP_K
from features import compute_features_batch, _norm_str, N_FEATURES

# Use the threshold found by classifier.py's search; default to 0.88
# This will be overridden if threshold_search.tsv exists
OPTIMAL_THRESHOLD = 0.88
CHUNK_SIZE = 100_000

def load_best_threshold() -> float:
    """Load the best threshold from threshold_search.tsv if available."""
    thresh_path = os.path.join(OUTPUT_DIR, 'threshold_search.tsv')
    if os.path.exists(thresh_path):
        df = pd.read_csv(thresh_path, sep='\t')
        best_idx = df['macro_f05'].idxmax()
        best_thresh = df.loc[best_idx, 'threshold']
        best_f05 = df.loc[best_idx, 'macro_f05']
        log.info(f"Loaded best threshold from search: {best_thresh:.2f} (F0.5={best_f05:.4f})")
        return float(best_thresh)
    return OPTIMAL_THRESHOLD


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
    # STEP 0: TRAIN LIGHTGBM CLASSIFIER
    # =========================================================================
    log.info("=== STEP 0: TRAINING LIGHTGBM CLASSIFIER ===")
    feat_train_path = os.path.join(OUTPUT_DIR, 'features_train.npy')
    lbl_train_path  = os.path.join(OUTPUT_DIR, 'labels_train.npy')

    if not os.path.exists(feat_train_path) or not os.path.exists(lbl_train_path):
        log.error(f"Training features missing in {OUTPUT_DIR}! Run run_full_benchmark.py first.")
        sys.exit(1)

    log.info(f"Loading training data from {feat_train_path} …")
    X_train = np.load(feat_train_path)
    y_train = np.load(lbl_train_path)
    log.info(f"Loaded X_train shape={X_train.shape}, y_train shape={y_train.shape} (positives: {(y_train==1).sum():,})")

    # Use tuned hyperparameters matching classifier.py
    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1
    )
    t_train = time.time()
    clf.fit(X_train, y_train)
    log.info(f"LightGBM trained successfully in {time.time() - t_train:.1f}s ✅ ({clf.n_features_in_} features)")
    del X_train, y_train
    gc.collect()

    # Load the best threshold from threshold search
    threshold = load_best_threshold()
    log.info(f"Using decision threshold: {threshold:.2f}")

    # Predictions file for post-processing
    predictions_path = os.path.join(OUTPUT_DIR, 'raw_predictions.tsv')
    if os.path.exists(predictions_path):
        os.remove(predictions_path)
    with open(predictions_path, 'w') as f:
        f.write("source1_entity_id\tcandidate_entity_id\tprobability\n")

    # =========================================================================
    # STEP 1: LOAD S1 QUERIES
    # =========================================================================
    log.info("=== STEP 1: LOADING TEST SOURCE 1 QUERIES ===")
    df_s1 = pd.read_csv(s1_test_path, sep='\t', dtype=str)
    all_s1_ids = list(df_s1['entity_id'].values)
    log.info(f"Total Test Source 1 Queries: {len(df_s1):,}")

    df_s1['comb_text'] = build_combined_text(df_s1)
    test_countries = sorted(df_s1['country'].dropna().unique())
    log.info(f"Discovered Test Countries: {test_countries}")

    candidates_dict: dict[str, list[str]] = {}

    # =========================================================================
    # STEP 2: PROCESS COUNTRY-BY-COUNTRY (BLOCKING + FEATURES + INFERENCE)
    # =========================================================================
    for country in test_countries:
        t_country = time.time()
        sub_s1 = df_s1[df_s1['country'] == country]
        if sub_s1.empty:
            continue

        log.info(f"\n=======================================================")
        log.info(f"🌍 PROCESSING COUNTRY: [{country}] ({len(sub_s1):,} S1 Queries)")
        log.info(f"=======================================================")

        # 2a. Stream Test S2 & S3 Pool for Country
        log.info(f"[{country}] Streaming Test S2 & S3 pool …")
        s2_c = read_filtered_tsv(s2_test_path, {country})
        s3_c = read_filtered_tsv(s3_test_path, {country})
        pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()

        log.info(f"[{country}] Candidate Pool: {len(pool_c):,} records")
        index, idf = build_inverted_index(pool_c)
        log.info(f"[{country}] Inverted index ready ({len(index):,} tokens). Querying S1 …")

        del pool_c
        gc.collect()

        # 2b. Candidate Retrieval (uses TOP_K from blocking.py = 20)
        cands_country: list[tuple[str, str, int]] = []
        for row in sub_s1.itertuples(index=False):
            c_list = query_index(row.comb_text, index, idf, TOP_K)
            candidates_dict[row.entity_id] = c_list
            for rank, cid in enumerate(c_list, start=1):
                cands_country.append((row.entity_id, cid, rank))

        del index, idf
        gc.collect()

        log.info(f"[{country}] Generated {len(cands_country):,} candidate pairs")

        # 2c. Build Entity Text Lookup for this country only
        needed_country_ids = set()
        for s1_id, cid, _ in cands_country:
            needed_country_ids.add(s1_id)
            needed_country_ids.add(cid)
        log.info(f"[{country}] Unique entities to lookup: {len(needed_country_ids):,}")

        lookup_country = {}
        # S1 entities
        for r in sub_s1.itertuples(index=False):
            if r.entity_id in needed_country_ids:
                lookup_country[r.entity_id] = {'name': _norm_str(r.business_name), 'address': _norm_str(r.business_address)}

        # S2 & S3 entities
        for path in (s2_test_path, s3_test_path):
            for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=100_000):
                sub = chunk[chunk['country'] == country]
                sub = sub[sub['entity_id'].isin(needed_country_ids)]
                for r in sub.itertuples(index=False):
                    lookup_country[r.entity_id] = {'name': _norm_str(r.business_name), 'address': _norm_str(r.business_address)}

        log.info(f"[{country}] Entity lookup built: {len(lookup_country):,} records")

        # 2d. Feature Extraction & Inference in Chunks
        log.info(f"[{country}] Extracting features & running LightGBM inference …")
        n_pairs = len(cands_country)
        n_matched_country = 0
        for start in range(0, n_pairs, CHUNK_SIZE):
            chunk_slice = cands_country[start:start + CHUNK_SIZE]
            chunk_df = pd.DataFrame(chunk_slice, columns=['source1_entity_id', 'candidate_entity_id', 'blocking_rank'])
            
            X_chunk = compute_features_batch(chunk_df, lookup_country)
            
            # Handle model trained with different feature counts
            n_model_feats = clf.n_features_in_
            if n_model_feats < X_chunk.shape[1]:
                proba_chunk = clf.predict_proba(X_chunk[:, :n_model_feats])[:, 1].astype(np.float32)
            else:
                proba_chunk = clf.predict_proba(X_chunk)[:, 1].astype(np.float32)

            # Apply veto: if veto_flag == 1, force probability to 0
            # veto_flag is at index 12 in the 13-feature layout
            if X_chunk.shape[1] > 12:
                veto_mask = X_chunk[:, 12] == 1.0
                proba_chunk[veto_mask] = 0.0

            # Save predictions above threshold for post-processing
            match_mask = proba_chunk >= threshold
            n_matched_country += match_mask.sum()
            valid_chunk = chunk_df[match_mask].copy()
            valid_chunk['probability'] = proba_chunk[match_mask]
            
            # Append to file
            valid_chunk[['source1_entity_id', 'candidate_entity_id', 'probability']].to_csv(
                predictions_path, sep='\t', index=False, mode='a', header=False
            )

            if (start // CHUNK_SIZE) % 5 == 0 or (start + CHUNK_SIZE >= n_pairs):
                log.info(f"  [{country}] {min(start + CHUNK_SIZE, n_pairs):,} / {n_pairs:,} pairs evaluated")

        del lookup_country, cands_country
        gc.collect()

        elapsed_c = time.time() - t_country
        log.info(f"[{country}] Completed in {elapsed_c / 60:.1f} minutes ({n_matched_country:,} matches) ✅")

    # =========================================================================
    # STEP 3: WRITE FINAL OUTPUT TSV FILES
    # =========================================================================
    log.info("\n=== STEP 3: WRITING SUBMISSION TSV FILES ===")

    # 3a. candidate_pairs.tsv
    log.info(f"Writing {candidate_tsv_path} …")
    with open(candidate_tsv_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            c_str = ','.join(candidates_dict.get(s1_id, []))
            f.write(f"{s1_id}\t{c_str}\n")
    log.info(f"Candidate pairs saved ({os.path.getsize(candidate_tsv_path) / (1024**2):.1f} MB) ✅")

    # 3b. matching_results.tsv via post_process.py (pool-side dedup)
    log.info("Running post_process.py for pool-side dedup …")
    
    s1_ids_path = os.path.join(OUTPUT_DIR, 's1_ids.txt')
    with open(s1_ids_path, 'w') as f:
        for s1_id in all_s1_ids:
            f.write(f"{s1_id}\n")
            
    post_process_script = os.path.join(SRC_DIR, 'post_process.py')
    pp_cmd = [
        sys.executable, post_process_script,
        predictions_path, matching_tsv_path, s1_ids_path
    ]
    log.info(f"Command: {' '.join(pp_cmd)}")
    pp_res = subprocess.run(pp_cmd)
    if pp_res.returncode != 0:
        log.error("❌ post_process.py failed!")
        sys.exit(pp_res.returncode)

    # =========================================================================
    # STEP 4: RUN OFFICIAL VALIDATION SCRIPT
    # =========================================================================
    log.info("\n=== STEP 4: RUNNING OFFICIAL SUBMISSION VALIDATOR ===")
    validator_script = os.path.join(BASE_DIR, 'utils', 'validate_submission.py')
    val_cmd = [
        sys.executable, validator_script,
        '--matching', matching_tsv_path,
        '--candidate', candidate_tsv_path,
        '--test-dir', TEST_DIR
    ]
    log.info(f"Running validator: {' '.join(val_cmd)}")
    val_res = subprocess.run(val_cmd)
    if val_res.returncode != 0:
        log.error(f"❌ Submission validation failed with return code {val_res.returncode}!")
        sys.exit(val_res.returncode)
    log.info("✅ Official Submission Validator passed with 0 errors!")

    # =========================================================================
    # STEP 5: PACKAGE SUBMISSION ZIP
    # =========================================================================
    log.info("\n=== STEP 5: PACKAGING SUBMISSION ZIP ===")
    zip_path = os.path.join(OUTPUT_DIR, 'submission.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(matching_tsv_path, arcname='matching_results.tsv')
        zipf.write(candidate_tsv_path, arcname='candidate_pairs.tsv')

    zip_size_mb = os.path.getsize(zip_path) / (1024**2)
    total_min = (time.time() - t_start) / 60
    log.info("=" * 65)
    log.info(f"🎉 FINAL TEST SUBMISSION PIPELINE COMPLETED IN {total_min:.1f} MINUTES!")
    log.info(f"📦 SUBMISSION ZIP READY: {zip_path} ({zip_size_mb:.1f} MB)")
    log.info("=" * 65)

if __name__ == '__main__':
    main()
