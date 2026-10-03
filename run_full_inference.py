import os
import gc
import time
import joblib
import numpy as np
import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import clean_name, clean_address, load_partition, get_available_countries

def run_full_pipeline(
    split_dir: str = "dataset/test", 
    output_dir: str = "output", 
    max_candidates: int = 12,
    threshold: float = 0.76
):
    os.makedirs(output_dir, exist_ok=True)
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    
    countries = get_available_countries(split_dir)
    print(f"Starting full pipeline inference for countries: {countries}")

    total_s1 = 0
    total_candidates = 0

    with open(candidate_path, "w", encoding="utf-8", newline="") as cand_f, \
         open(matching_path, "w", encoding="utf-8", newline="") as match_f:
        
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")
        match_f.write("source1_entity_id\tmatched_entity_id\n")

        for country in countries:
            print(f"\n--- Processing Country: {country} ---")
            t0 = time.time()
            
            s1_df = load_partition(split_dir, "source1", country)
            s2_df = load_partition(split_dir, "source2", country)
            s3_df = load_partition(split_dir, "source3", country)
            
            print(f"[{country}] Loaded S1: {len(s1_df)}, S2: {len(s2_df)}, S3: {len(s3_df)} in {time.time()-t0:.2f}s")

            # 1. Build background index (S2 + S3)
            cand_ids = []
            cand_docs = []
            
            for df in [s2_df, s3_df]:
                for row in df.iter_rows(named=True):
                    cand_ids.append(row['entity_id'])
                    cand_docs.append(f"{clean_name(row['business_name'])} {clean_address(row['business_address'])}".strip())
            
            print(f"[{country}] Vectorizing background corpus of {len(cand_docs)} items...")
            vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
                dtype=np.float32
            )
            cand_vectors = vectorizer.fit_transform(cand_docs)
            cand_ids_arr = np.array(cand_ids)
            del cand_docs, s2_df, s3_df
            gc.collect()

            # 2. Extract S1 documents
            s1_ids = []
            s1_docs = []
            for row in s1_df.iter_rows(named=True):
                s1_ids.append(row['entity_id'])
                s1_docs.append(f"{clean_name(row['business_name'])} {clean_address(row['business_address'])}".strip())

            print(f"[{country}] Vectorizing {len(s1_docs)} query entities...")
            s1_vectors = vectorizer.transform(s1_docs)
            del s1_docs, s1_df
            gc.collect()

            batch_size = 1000
            k = min(max_candidates, len(cand_ids))
            
            print(f"[{country}] Running ultra-fast sparse candidate scoring & matching...")
            for i in tqdm(range(0, s1_vectors.shape[0], batch_size), desc=f"Scoring {country}"):
                batch = s1_vectors[i:i + batch_size]
                sim_matrix = batch.dot(cand_vectors.T) # CSR matrix
                
                # Extract C-level memory arrays once per batch
                indptr = sim_matrix.indptr
                indices = sim_matrix.indices
                data = sim_matrix.data
                
                for idx_in_batch in range(sim_matrix.shape[0]):
                    actual_s1_id = s1_ids[i + idx_in_batch]
                    
                    # Direct memory slice (O(1) operation, no object allocation)
                    start_idx = indptr[idx_in_batch]
                    end_idx = indptr[idx_in_batch + 1]
                    
                    row_data = data[start_idx:end_idx]
                    row_indices = indices[start_idx:end_idx]
                    
                    n_val = len(row_data)
                    if n_val > 0:
                        if n_val > k:
                            top_k_local = np.argpartition(-row_data, kth=k-1)[:k]
                            sort_idx = top_k_local[np.argsort(-row_data[top_k_local])]
                        else:
                            sort_idx = np.argsort(-row_data)
                        
                        valid_cands = [cand_ids_arr[row_indices[idx]] for idx in sort_idx]
                        best_score = row_data[sort_idx[0]]
                        best_match_id = cand_ids_arr[row_indices[sort_idx[0]]] if best_score >= threshold else ""
                    else:
                        valid_cands = []
                        best_match_id = ""

                    cand_f.write(f"{actual_s1_id}\t{','.join(valid_cands)}\n")
                    match_f.write(f"{actual_s1_id}\t{best_match_id}\n")

            del s1_vectors, cand_vectors, vectorizer
            gc.collect()

    print(f"\nPipeline Complete!")
    print(f" - Candidate pairs saved to: {candidate_path}")
    print(f" - Matching results saved to: {matching_path}")

if __name__ == "__main__":
    run_full_pipeline()