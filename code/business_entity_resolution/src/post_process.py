import os
import sys
import time
import logging
import pandas as pd

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
    
    log.info("Applying bipartite 1-to-1 matching constraint ...")
    
    # We will populate matches_dict and enforce 1-to-1 global mapping
    matches_dict = {}
    seen_s1 = set()
    seen_candidates = set()
    
    for row in df.itertuples(index=False):
        s1 = row.source1_entity_id
        cand = row.candidate_entity_id
        
        # 1-to-1 constraint: S1 can only have one candidate?
        # Wait, the challenge requires S1 to have 0 or 1 matches?
        # Actually, the user says "Iterate and approve a match ONLY if the candidate ID has not already been assigned to another Source 1 entity globally."
        # And what if the Source 1 entity has already been assigned a match? It should only have 1 match! 
        # But wait, bipartite matching means 1-to-1. So each S1 gets max 1 match, and each candidate gets max 1 match.
        if s1 in seen_s1:
            continue
        if cand in seen_candidates:
            continue
            
        # Approve match
        matches_dict[s1] = cand
        seen_s1.add(s1)
        seen_candidates.add(cand)
        
    log.info(f"Approved {len(matches_dict):,} 1-to-1 matches out of {len(df):,} candidates.")
    
    log.info(f"Loading all Source 1 IDs from {s1_ids_path} ...")
    with open(s1_ids_path, 'r') as f:
        all_s1_ids = [line.strip() for line in f if line.strip()]
        
    log.info(f"Writing {matching_tsv_path} ...")
    n_singletons = 0
    n_matched = 0
    with open(matching_tsv_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            cand = matches_dict.get(s1_id)
            if cand:
                m_str = cand
                n_matched += 1
            else:
                m_str = ''
                n_singletons += 1
            f.write(f"{s1_id}\t{m_str}\n")
            
    log.info(f"Predictions summary: {n_matched:,} entities matched, {n_singletons:,} singletons (blank)")
    log.info(f"post_process.py complete in {time.time() - t0:.1f}s ✅")

if __name__ == '__main__':
    main()
