"""
Inference pipeline for business entity resolution.
Runs end-to-end blocking, feature extraction, LightGBM inference,
house number conflict discounting, and global 1-to-1 competition cardinality post-processing.
Produces:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv
Strictly enforces partition invariance, exact subset consistency, 1-to-1 candidate cardinality,
tab delimiters without quotes, and complete coverage of all test S1 entities.
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
from tqdm import tqdm

from src.preprocess import clean_name, clean_address, extract_digits, load_partition, get_available_countries
from src.blocking import CountryInvertedIndex
from src.features import extract_batch_features, check_house_conflict
from src.model import load_model


def run_country_inference(
    split_dir: str,
    country: str,
    model,
    threshold: float,
    cand_file,
    match_file,
    max_candidates: int = 12,
    chunk_size: int = 25000,
) -> Tuple[int, int, int]:
    """Execute streaming inference with 1-to-1 cardinality matching for one country partition.
    
    Returns:
        (total_s1, total_candidates, total_matches)
    """
    print(f"\n[{country}] Loading data partitions...")
    t0 = time.time()
    s1_df = load_partition(split_dir, "source1", country)
    s2_df = load_partition(split_dir, "source2", country)
    s3_df = load_partition(split_dir, "source3", country)
    print(f"[{country}] Loaded S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)} in {time.time()-t0:.2f}s")

    # 1. Build compact lookup for S2 and S3 records
    t0 = time.time()
    s23_clean: Dict[str, Tuple[str, str, List[str]]] = {}
    for df in [s2_df, s3_df]:
        for r in df.iter_rows(named=True):
            eid = r['entity_id']
            cn = clean_name(r['business_name'])
            ca = clean_address(r['business_address'])
            cd = extract_digits(ca)
            s23_clean[eid] = (cn, ca, cd)

    # 2. Build inverted index for blocking
    indexer = CountryInvertedIndex(max_bucket_size=300, max_candidates=max_candidates)
    indexer.build(s2_df, s3_df)
    print(f"[{country}] Inverted index built with {len(indexer.index)} keys in {time.time()-t0:.2f}s")

    del s2_df, s3_df

    total_s1 = 0
    total_candidates = 0

    s1_rows = s1_df.iter_rows(named=True)
    num_s1 = len(s1_df)
    del s1_df

    pbar = tqdm(total=num_s1, desc=f"Scoring [{country}]", unit="entities")

    country_s1_order: List[str] = []
    # Collect passing links: (adjusted_prob, s1_id, cid)
    passing_candidate_links: List[Tuple[float, str, str]] = []

    # 3. Process S1 entities in memory-safe chunks
    while True:
        chunk = []
        for _ in range(chunk_size):
            try:
                chunk.append(next(s1_rows))
            except StopIteration:
                break
        if not chunk:
            break

        chunk_pairs_to_score = []
        chunk_pair_meta = []  # (s1_id, cid, has_house_conflict)

        for r in chunk:
            s1_id = r['entity_id']
            country_s1_order.append(s1_id)
            cn1 = clean_name(r['business_name'])
            ca1 = clean_address(r['business_address'])
            cd1 = extract_digits(ca1)

            cands = indexer.query(cn1, ca1, cd1)

            # Write candidate pairs immediately to TSV
            cand_str = ",".join(cands)
            cand_file.write(f"{s1_id}\t{cand_str}\n")
            total_s1 += 1
            total_candidates += len(cands)

            for cid in cands:
                if cid in s23_clean:
                    cn2, ca2, cd2 = s23_clean[cid]
                    chunk_pairs_to_score.append((cn1, ca1, cd1, cid, cn2, ca2, cd2))
                    has_conflict = check_house_conflict(cd1, cd2)
                    chunk_pair_meta.append((s1_id, cid, has_conflict))

        # Batch feature extraction and model prediction
        if chunk_pairs_to_score:
            X_batch = extract_batch_features(chunk_pairs_to_score)
            probs = model.predict_proba(X_batch)[:, 1]

            for i, prob in enumerate(probs):
                s1_id, cid, has_conflict = chunk_pair_meta[i]
                # Apply House / Building Number conflict discount (Enhancement 3)
                if has_conflict:
                    prob *= 0.5  # 50% discount penalty to prevent false merges across shared streets

                if prob >= threshold:
                    passing_candidate_links.append((float(prob), s1_id, cid))

        pbar.update(len(chunk))

    pbar.close()
    cand_file.flush()

    # 4. Global 1-to-1 Competition Cardinality Post-Processing (Enhancement 2)
    print(f"\n[{country}] Resolving 1-to-1 cardinality on {len(passing_candidate_links)} passing candidate links...")
    t_card = time.time()
    # Sort candidate links by probability descending
    passing_candidate_links.sort(key=lambda x: x[0], reverse=True)

    assigned_candidates: Set[str] = set()
    s1_matches: Dict[str, List[str]] = defaultdict(list)
    discarded_multi_claims = 0

    for prob, s1_id, cid in passing_candidate_links:
        if cid not in assigned_candidates:
            assigned_candidates.add(cid)
            s1_matches[s1_id].append(cid)
        else:
            discarded_multi_claims += 1

    print(f"[{country}] Assigned {len(assigned_candidates)} unique matches (discarded {discarded_multi_claims} multi-claims) in {time.time()-t_card:.2f}s.")

    # 5. Write matching results in exact original S1 order
    total_matches = 0
    for s1_id in country_s1_order:
        matches = s1_matches.get(s1_id, [])
        match_str = ",".join(matches)
        match_file.write(f"{s1_id}\t{match_str}\n")
        total_matches += len(matches)

    match_file.flush()
    return total_s1, total_candidates, total_matches


def run_pipeline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_dir: str = "models",
    max_candidates: int = 12,
):
    """Run full inference pipeline generating candidate_pairs.tsv and matching_results.tsv."""
    print("=" * 60)
    print("STARTING ENHANCED INFERENCE PIPELINE")
    print("=" * 60)
    start_time = time.time()

    # Load model and calibrated threshold
    model, threshold = load_model(model_dir)
    print(f"Loaded matcher model from {model_dir}")
    print(f"Operating Precision Threshold: {threshold:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    countries = get_available_countries(test_dir)
    print(f"Test Countries: {countries}")

    total_s1 = 0
    total_cands = 0
    total_matches = 0

    with open(cand_path, "w", encoding="utf-8", newline="") as cand_f, \
         open(match_path, "w", encoding="utf-8", newline="") as match_f:

        # Write official headers
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
        match_f.write("source1_entity_id\tmatched_entity_ids\n")

        for country in countries:
            c_s1, c_cands, c_matches = run_country_inference(
                split_dir=test_dir,
                country=country,
                model=model,
                threshold=threshold,
                cand_file=cand_f,
                match_file=match_f,
                max_candidates=max_candidates,
            )
            total_s1 += c_s1
            total_cands += c_cands
            total_matches += c_matches

    elapsed = time.time() - start_time
    avg_cands = total_cands / total_s1 if total_s1 > 0 else 0.0
    avg_matches = total_matches / total_s1 if total_s1 > 0 else 0.0

    print("\n" + "=" * 60)
    print("ENHANCED INFERENCE PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Elapsed Time:                {elapsed:.1f}s ({elapsed/60:.2f} mins)")
    print(f"Total S1 Entities Processed: {total_s1}")
    print(f"Total Candidates Generated:  {total_cands} (avg {avg_cands:.2f} / entity)")
    print(f"Total Matches Predicted:     {total_matches} (avg {avg_matches:.2f} / entity)")
    print(f"Candidate file written:      {cand_path}")
    print(f"Matching results written:    {match_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run test inference pipeline")
    parser.add_argument("--test-dir", default="dataset/test", help="Test directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--max-candidates", type=int, default=12, help="Max candidates per entity")
    args = parser.parse_args()

    run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_candidates=args.max_candidates,
    )
