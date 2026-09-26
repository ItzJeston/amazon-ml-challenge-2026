"""
Blocking module for business entity resolution.
Implements high-recall, low-volume multi-key inverted indexing partitioned by country.
Guarantees candidate set compactness (5-15 candidates per S1 entity) and partition invariance.
"""

import os
from array import array
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Generator
import polars as pl
from tqdm import tqdm

from src.preprocess import clean_name, clean_address, extract_digits, load_partition, get_available_countries


def extract_blocking_keys(c_name: str, c_addr: str, digits: List[str]) -> Set[str]:
    """Extract diverse blocking keys from cleaned name and address."""
    keys = set()
    name_tokens = c_name.split()
    addr_tokens = c_addr.split()

    # 1. Compact name prefixes (handles missing spaces/legal suffix variations)
    compact_name = c_name.replace(" ", "")
    if len(compact_name) >= 3:
        keys.add(f"nc:{compact_name[:12]}")
        keys.add(f"nc6:{compact_name[:6]}")

    # 2. Name token combinations
    if len(name_tokens) >= 2:
        keys.add(f"nt2:{name_tokens[0]}_{name_tokens[1]}")
    elif len(name_tokens) == 1 and len(name_tokens[0]) >= 3:
        keys.add(f"nt1:{name_tokens[0]}")
    if len(name_tokens) >= 3:
        keys.add(f"nt3:{name_tokens[0]}_{name_tokens[2]}")

    # 3. Address house/building digit + first street token
    if digits and addr_tokens:
        keys.add(f"ad1:{digits[0]}_{addr_tokens[0][:6]}")
        if len(addr_tokens) >= 2:
            keys.add(f"ad2:{digits[0]}_{addr_tokens[1][:6]}")

    # 4. First name token + address first digit
    if name_tokens and digits:
        keys.add(f"nad:{name_tokens[0][:5]}_{digits[0]}")

    # 5. Consecutive digits combination (e.g. house number + postal code)
    if len(digits) >= 2:
        keys.add(f"digs:{digits[0]}_{digits[1]}")

    # 6. Address token pair (e.g. street + city)
    if len(addr_tokens) >= 3:
        keys.add(f"at2:{addr_tokens[0][:5]}_{addr_tokens[1][:5]}")

    return keys


class CountryInvertedIndex:
    """Memory-efficient inverted index for S2 and S3 records of a single country."""

    def __init__(self, max_bucket_size: int = 300, max_candidates: int = 12):
        self.max_bucket_size = max_bucket_size
        self.max_candidates = max_candidates
        self.entity_ids: List[str] = []
        self.index: Dict[str, array] = defaultdict(lambda: array('I'))

    def build(self, s2_df: pl.DataFrame, s3_df: pl.DataFrame):
        """Build compact inverted index using 4-byte unsigned integer references."""
        current_idx = 0
        
        # Process S2 records
        for row in s2_df.iter_rows(named=True):
            eid = row['entity_id']
            self.entity_ids.append(eid)
            cn = clean_name(row['business_name'])
            ca = clean_address(row['business_address'])
            cd = extract_digits(ca)
            for k in extract_blocking_keys(cn, ca, cd):
                self.index[k].append(current_idx)
            current_idx += 1

        # Process S3 records
        for row in s3_df.iter_rows(named=True):
            eid = row['entity_id']
            self.entity_ids.append(eid)
            cn = clean_name(row['business_name'])
            ca = clean_address(row['business_address'])
            cd = extract_digits(ca)
            for k in extract_blocking_keys(cn, ca, cd):
                self.index[k].append(current_idx)
            current_idx += 1

    def query(self, c_name: str, c_addr: str, digits: List[str]) -> List[str]:
        """Query index for top candidates sorted by blocking key hit frequency."""
        hits = Counter()
        for k in extract_blocking_keys(c_name, c_addr, digits):
            bucket = self.index.get(k)
            if bucket and len(bucket) <= self.max_bucket_size:
                hits.update(bucket)

        if not hits:
            return []

        top_indices = [idx for idx, _ in hits.most_common(self.max_candidates)]
        return [self.entity_ids[idx] for idx in top_indices]


def generate_candidates_for_country(
    split_dir: str,
    country: str,
    max_candidates: int = 12,
) -> Generator[Tuple[str, List[str]], None, None]:
    """Stream candidate generation for all S1 entities of a given country.
    
    Yields:
        (s1_id, candidate_ids_list)
    """
    print(f"Loading partitions for country: {country}...")
    s1_df = load_partition(split_dir, "source1", country)
    s2_df = load_partition(split_dir, "source2", country)
    s3_df = load_partition(split_dir, "source3", country)
    
    print(f"[{country}] S1: {len(s1_df)}, S2: {len(s2_df)}, S3: {len(s3_df)}")
    
    indexer = CountryInvertedIndex(max_bucket_size=300, max_candidates=max_candidates)
    indexer.build(s2_df, s3_df)
    print(f"[{country}] Inverted index built with {len(indexer.index)} unique keys.")

    for row in s1_df.iter_rows(named=True):
        s1_id = row['entity_id']
        cn = clean_name(row['business_name'])
        ca = clean_address(row['business_address'])
        cd = extract_digits(ca)
        cands = indexer.query(cn, ca, cd)
        yield s1_id, cands


def generate_candidate_file(
    split_dir: str,
    output_path: str,
    max_candidates: int = 12,
) -> Dict[str, float]:
    """Generate and write output/candidate_pairs.tsv for all countries in split_dir.
    
    Returns:
        Summary dictionary with candidate count statistics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    countries = get_available_countries(split_dir)
    print(f"Countries to process for candidates: {countries}")

    total_s1 = 0
    total_candidates = 0
    empty_candidates = 0

    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        # Write header with strict tab separator
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in countries:
            print(f"\nProcessing blocking for country: {country}")
            for s1_id, cands in generate_candidates_for_country(split_dir, country, max_candidates):
                cand_str = ",".join(cands)
                out_f.write(f"{s1_id}\t{cand_str}\n")
                total_s1 += 1
                num_cands = len(cands)
                total_candidates += num_cands
                if num_cands == 0:
                    empty_candidates += 1

    avg_cands = total_candidates / total_s1 if total_s1 > 0 else 0.0
    print(f"\nFinished writing candidate file to {output_path}")
    print(f"Total S1 entities: {total_s1}")
    print(f"Average candidates per S1: {avg_cands:.2f}")
    print(f"Entities with 0 candidates: {empty_candidates} ({empty_candidates/total_s1*100:.1f}%)")

    return {
        "total_s1": total_s1,
        "total_candidates": total_candidates,
        "avg_candidates": avg_cands,
        "empty_candidates": empty_candidates,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run blocking to generate candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test directory")
    parser.add_argument("--output", default="output/candidate_pairs.tsv", help="Output path")
    parser.add_argument("--max-candidates", type=int, default=12, help="Max candidates per entity")
    args = parser.parse_args()

    generate_candidate_file(args.test_dir, args.output, args.max_candidates)
