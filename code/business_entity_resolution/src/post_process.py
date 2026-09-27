"""
post_process.py — Post-processing for the Amazon Business Entity Resolution Challenge

Enforces the dataset's constraint: each S2/S3 entity can match at most ONE S1 entity.
(But each S1 entity CAN match MULTIPLE S2/S3 entities — this is 1-to-many on S1 side.)

Strategy:
  1. Load all predictions (s1_id, candidate_id, probability) above threshold.
  2. Sort by probability descending (highest confidence first).
  3. Iterate through and approve a match ONLY if the candidate_id has not
     already been assigned to a DIFFERENT S1 entity.
  4. Build the final matching_results.tsv with all approved matches per S1.
"""

import os
import sys
import time
import logging
import pandas as pd
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 4:
        print("Usage: python post_process.py <predictions.tsv> <matching_results.tsv> <s1_ids.txt>")
        sys.exit(1)
        
    predictions_path = sys.argv[1]
    matching_tsv_path = sys.argv[2]
    s1_ids_path = sys.argv[3]
    
    t0 = time.time()
    
    log.info(f"Loading predictions from {predictions_path} ...")
    df = pd.read_csv(predictions_path, sep='\t')
    log.info(f"Loaded {len(df):,} predictions above threshold.")
    
    log.info("Sorting by probability descending ...")
    df = df.sort_values(by='probability', ascending=False)
    
    log.info("Applying pool-side dedup constraint (each S2/S3 ID → at most 1 S1) ...")
    
    # Each S1 entity CAN have multiple matches (1-to-many on S1 side)
    # But each pool entity (S2/S3) can only be matched by ONE S1 entity
    matches_dict: dict[str, list[str]] = defaultdict(list)
    seen_candidates: set[str] = set()
    n_approved = 0
    n_rejected = 0
    
    for row in df.itertuples(index=False):
        s1 = row.source1_entity_id
        cand = row.candidate_entity_id
        
        # Pool-side constraint only: each candidate can be claimed by at most one S1
        if cand in seen_candidates:
            n_rejected += 1
            continue
            
        # Approve match
        matches_dict[s1].append(cand)
        seen_candidates.add(cand)
        n_approved += 1
        
    log.info(f"Approved {n_approved:,} matches, rejected {n_rejected:,} duplicates "
             f"({len(matches_dict):,} S1 entities with matches).")
    
    log.info(f"Loading all Source 1 IDs from {s1_ids_path} ...")
    with open(s1_ids_path, 'r') as f:
        all_s1_ids = [line.strip() for line in f if line.strip()]
        
    log.info(f"Writing {matching_tsv_path} ...")
    n_singletons = 0
    n_matched = 0
    with open(matching_tsv_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            m_list = matches_dict.get(s1_id, [])
            if m_list:
                m_str = ','.join(m_list)
                n_matched += 1
            else:
                m_str = ''
                n_singletons += 1
            f.write(f"{s1_id}\t{m_str}\n")
            
    log.info(f"Predictions summary: {n_matched:,} entities matched, {n_singletons:,} singletons (blank)")
    log.info(f"post_process.py complete in {time.time() - t0:.1f}s ✅")

if __name__ == '__main__':
    main()
