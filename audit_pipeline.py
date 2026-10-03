"""
Diagnostic audit script for Business Entity Resolution pipeline.
Evaluates on a random 5,000-sample of dataset/train/ against the actual multi-million record background corpus:
  1. Blocking Recall Ceiling
  2. Classifier Retention
  3. Metric Breakdown (Precision, Recall, Macro F0.5, Singleton Accuracy)
"""

import sys
import os
import time
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
from tqdm import tqdm

from src.preprocess import clean_name, clean_address, extract_digits, load_partition
from src.blocking import CountryInvertedIndex
from src.features import extract_batch_features, check_house_conflict
from src.model import load_model, compute_macro_f05


def run_diagnostic_audit(
    train_dir: str = "dataset/train",
    model_dir: str = "models",
    sample_size: int = 5000,
    seed: int = 42,
    threshold: float = None,
):
    print("=" * 70)
    print("DIAGNOSTIC AUDIT: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)
    start_total_time = time.time()

    # Load model and threshold
    model, default_thresh = load_model(model_dir)
    tau = threshold if threshold is not None else default_thresh
    print(f"Loaded LightGBM model from: {model_dir}")
    print(f"Operating Decision Threshold tau: {tau:.2f}")

    # 1. Sample 5,000 random entities from train_ground_truth.tsv
    print(f"\nSampling {sample_size} random S1 entities from {train_dir}/train_ground_truth.tsv (seed={seed})...")
    gt_df = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")
    
    random.seed(seed)
    sampled_indices = set(random.sample(range(len(gt_df)), sample_size))
    sample_gt_df = gt_df.filter(pl.Series(range(len(gt_df))).is_in(sampled_indices))

    # Build ground truth mapping
    gt_map: Dict[str, Set[str]] = {}
    for row in sample_gt_df.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        gt_map[s1_id] = set(matches_str.split(',')) if matches_str else set()

    sampled_s1_ids = list(gt_map.keys())

    # 2. Retrieve S1 metadata (name, address, country) for the sampled entities
    print("Retrieving S1 records from train_source1.tsv...")
    s1_df = (
        pl.scan_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
        .filter(pl.col("entity_id").is_in(sampled_s1_ids))
        .collect()
    )

    s1_records: Dict[str, Tuple[str, str, List[str], str]] = {}
    for r in s1_df.iter_rows(named=True):
        s1_id = r['entity_id']
        cn = clean_name(r['business_name'])
        ca = clean_address(r['business_address'])
        cd = extract_digits(ca)
        s1_records[s1_id] = (cn, ca, cd, r['country'])

    countries = sorted(list(set(r[3] for r in s1_records.values())))
    print(f"Sample breakdown: {len(s1_records)} S1 entities across countries: {countries}")
    for c in countries:
        cnt = sum(1 for r in s1_records.values() if r[3] == c)
        print(f"  - {c}: {cnt} entities")

    # Tracking metrics across all countries
    all_s1_candidates: Dict[str, List[str]] = {}
    all_s1_predictions: Dict[str, List[str]] = {}

    total_true_matches_corpus = sum(len(m) for m in gt_map.values())
    total_singletons_corpus = sum(1 for m in gt_map.values() if len(m) == 0)

    # 3. Process each country partition
    for country in countries:
        print(f"\n{'-'*30} [{country}] {'-'*30}")
        country_s1 = {eid: rec for eid, rec in s1_records.items() if rec[3] == country}
        print(f"[{country}] S1 sample entities: {len(country_s1)}")

        print(f"[{country}] Loading full background corpus from train_source2 and train_source3...")
        t0 = time.time()
        s2_df = load_partition(train_dir, "source2", country)
        s3_df = load_partition(train_dir, "source3", country)
        print(f"[{country}] Loaded S2={len(s2_df)}, S3={len(s3_df)} in {time.time()-t0:.2f}s")

        # Build S2/S3 lookup
        t0 = time.time()
        s23_clean: Dict[str, Tuple[str, str, List[str]]] = {}
        for df in [s2_df, s3_df]:
            eids = df['entity_id'].to_list()
            names = df['business_name'].to_list()
            addrs = df['business_address'].to_list()
            for eid, name, addr in zip(eids, names, addrs):
                cn = clean_name(name)
                ca = clean_address(addr)
                cd = extract_digits(ca)
                s23_clean[eid] = (cn, ca, cd)

        # Build Inverted Index from src/blocking.py
        indexer = CountryInvertedIndex(max_bucket_size=300, max_candidates=12)
        indexer.build(s2_df, s3_df)
        print(f"[{country}] Inverted index built with {len(indexer.index)} keys in {time.time()-t0:.2f}s")

        del s2_df, s3_df

        # Blocking & Candidate Generation
        print(f"[{country}] Querying candidates and scoring candidate pairs...")
        country_candidate_links = []
        
        for s1_id, (cn1, ca1, cd1, _) in country_s1.items():
            cands = indexer.query(cn1, ca1, cd1)
            all_s1_candidates[s1_id] = cands

            # Prepare pairs to score
            pairs_to_score = []
            cand_meta = []
            for cid in cands:
                if cid in s23_clean:
                    cn2, ca2, cd2 = s23_clean[cid]
                    pairs_to_score.append((cn1, ca1, cd1, cid, cn2, ca2, cd2))
                    has_conflict = check_house_conflict(cd1, cd2)
                    cand_meta.append((cid, has_conflict))

            if pairs_to_score:
                X_batch = extract_batch_features(pairs_to_score)
                probs = model.predict_proba(X_batch)[:, 1]

                for i, prob in enumerate(probs):
                    cid, has_conflict = cand_meta[i]
                    if has_conflict:
                        prob *= 0.5  # 50% discount penalty
                    if prob >= tau:
                        country_candidate_links.append((float(prob), s1_id, cid))

        print(f"[{country}] Passing candidate links above tau={tau:.2f}: {len(country_candidate_links)}")

        # Global 1-to-1 Competition Cardinality Post-Processing
        country_candidate_links.sort(key=lambda x: x[0], reverse=True)
        assigned_cands: Set[str] = set()
        s1_matches: Dict[str, List[str]] = defaultdict(list)
        discarded_claims = 0

        for prob, s1_id, cid in country_candidate_links:
            if cid not in assigned_cands:
                assigned_cands.add(cid)
                s1_matches[s1_id].append(cid)
            else:
                discarded_claims += 1

        print(f"[{country}] 1-to-1 cardinality resolved: {len(assigned_cands)} unique matches assigned ({discarded_claims} multi-claims eliminated).")

        for s1_id in country_s1:
            all_s1_predictions[s1_id] = s1_matches.get(s1_id, [])

    # 4. Compute Exact Diagnostic Audit Metrics
    print("\n" + "=" * 70)
    print("DIAGNOSTIC AUDIT RESULTS (5,000 Sample of dataset/train/)")
    print("=" * 70)

    # 1. Blocking Recall Ceiling
    total_true_matches = 0
    true_matches_in_candidates = 0
    total_candidates_generated = sum(len(c) for c in all_s1_candidates.values())

    for s1_id, true_set in gt_map.items():
        total_true_matches += len(true_set)
        cand_set = set(all_s1_candidates.get(s1_id, []))
        true_matches_in_candidates += len(true_set & cand_set)

    blocking_recall_ceiling = (true_matches_in_candidates / total_true_matches * 100.0) if total_true_matches > 0 else 0.0

    # 2. Classifier Retention
    true_matches_predicted = 0
    total_predicted_matches = sum(len(p) for p in all_s1_predictions.values())

    for s1_id, true_set in gt_map.items():
        pred_set = set(all_s1_predictions.get(s1_id, []))
        true_matches_predicted += len(true_set & pred_set)

    classifier_retention = (true_matches_predicted / true_matches_in_candidates * 100.0) if true_matches_in_candidates > 0 else 0.0

    # 3. Metric Breakdown
    tp = true_matches_predicted
    fp = total_predicted_matches - tp
    fn = total_true_matches - tp

    micro_precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    micro_recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0

    # Entity-level macro averages
    per_entity_prec = []
    per_entity_rec = []
    singletons_correct = 0
    singletons_total = 0

    for s1_id, true_set in gt_map.items():
        pred_set = set(all_s1_predictions.get(s1_id, []))
        if len(true_set) == 0:
            singletons_total += 1
            if len(pred_set) == 0:
                singletons_correct += 1
        else:
            if len(pred_set) > 0:
                p_i = len(true_set & pred_set) / len(pred_set)
                per_entity_prec.append(p_i)
            r_i = len(true_set & pred_set) / len(true_set)
            per_entity_rec.append(r_i)

    macro_precision = (np.mean(per_entity_prec) * 100.0) if per_entity_prec else 0.0
    macro_recall = (np.mean(per_entity_rec) * 100.0) if per_entity_rec else 0.0
    singleton_accuracy = (singletons_correct / singletons_total * 100.0) if singletons_total > 0 else 100.0

    # Official Macro F0.5
    gt_map_for_metric = {s1: set(gt_map[s1]) for s1 in gt_map}
    pred_map_for_metric = {s1: set(all_s1_predictions[s1]) for s1 in all_s1_predictions}
    official_macro_f05 = compute_macro_f05(gt_map_for_metric, pred_map_for_metric)

    # Print exact required percentages to console
    print(f"\n1. BLOCKING RECALL CEILING:")
    print(f"   Blocking Recall Ceiling:    {blocking_recall_ceiling:.2f}% ({true_matches_in_candidates:,} / {total_true_matches:,} true matches)")
    print(f"   Total Candidates Generated: {total_candidates_generated:,} (avg {total_candidates_generated/sample_size:.2f} candidates/entity)")
    print(f"   True Matches Missed by Index: {total_true_matches - true_matches_in_candidates:,}")

    print(f"\n2. CLASSIFIER RETENTION:")
    print(f"   Classifier Retention Rate:  {classifier_retention:.2f}% ({true_matches_predicted:,} / {true_matches_in_candidates:,} retrieved matches)")
    print(f"   Matches Filtered by Tau/1to1: {true_matches_in_candidates - true_matches_predicted:,}")

    print(f"\n3. METRIC BREAKDOWN:")
    print(f"   Precision (Micro):          {micro_precision:.2f}% ({tp:,} / {tp + fp:,} predictions)")
    print(f"   Recall (Micro):             {micro_recall:.2f}% ({tp:,} / {total_true_matches:,} true matches)")
    print(f"   Precision (Macro):          {macro_precision:.2f}% (mean across entities with predictions)")
    print(f"   Recall (Macro):             {macro_recall:.2f}% (mean across non-singleton entities)")
    print(f"   Singleton Accuracy:         {singleton_accuracy:.2f}% ({singletons_correct:,} / {singletons_total:,} empty predictions)")
    print(f"   --------------------------------------------------")
    print(f"   OFFICIAL MACRO F_0.5 SCORE: {official_macro_f05:.4f}")
    print(f"   --------------------------------------------------")

    elapsed_total = time.time() - start_total_time
    print(f"\nDiagnostic audit completed in {elapsed_total:.1f}s ({elapsed_total/60:.2f} mins).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run diagnostic audit on training data")
    parser.add_argument("--sample-size", type=int, default=5000, help="Sample size of S1 entities")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold tau")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_diagnostic_audit(sample_size=args.sample_size, threshold=args.threshold, seed=args.seed)
