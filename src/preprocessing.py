"""
Data Ingestion, Preprocessing & Partitioning Module.
Handles:
- 2-token Soundex lookup via jellyfish (with pure-Python fallback).
- Missing address indicator (addr_missing_flag = 1) and .fill_null("").
- Address normalization and legal suffix stripping.
- Grouping data strictly by country ('US', 'India', 'France').
- Memory-safe streaming export to zstd-compressed Parquet partitions.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    ADDR_MISSING_FLAG_COL,
    ALL_COUNTRIES,
    CLEAN_ADDR_COL,
    CLEAN_NAME_COL,
    COUNTRY_COL,
    ENTITY_ID_COL,
    MAX_SOUNDEX_TOKENS,
    MIN_SOUNDEX_TOKEN_LEN,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PROCESSED_DATA_DIR,
    PROCESSED_SCHEMA_COLS,
    RAW_ADDR_COL,
    RAW_DATA_DIR,
    RAW_NAME_COL,
    SOUNDEX_COL,
    STREAMING_BATCH_SIZE,
    TEST_SOURCES,
    TRAIN_SOURCES,
)
from src.normalization import (
    normalize_business_address,
    strip_legal_suffixes,
)
from src.utils import get_logger, setup_utf8_stdout, track_memory

setup_utf8_stdout()
logger = get_logger("preprocessing")


# ==============================================================================
# 1. Phonetic 2-Token Soundex (jellyfish with Pure-Python Fallback)
# ==============================================================================
def compute_soundex_token(token: str) -> str:
    """
    Standard American Soundex for a single alphabetic token.
    Uses jellyfish if available, otherwise pure-Python fallback.
    """
    if not token:
        return ""

    # Try jellyfish first
    try:
        import jellyfish
        code = jellyfish.soundex(token)
        if code and code[0].isalpha() and len(code) == 4:
            return code
    except Exception:
        pass

    # Pure-Python fallback
    alpha_chars = [c for c in token if c.isalpha()]
    if not alpha_chars:
        return ""

    first_letter = alpha_chars[0].upper()
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    digits = [first_letter]
    prev_code = mapping.get(first_letter, "")

    for ch in alpha_chars[1:]:
        ch_upper = ch.upper()
        if ch_upper in mapping:
            code = mapping[ch_upper]
            if code != prev_code:
                digits.append(code)
                prev_code = code
        elif ch_upper in "AEIOUY":
            prev_code = ""  # vowels reset duplicate check
        # 'H' and 'W' are ignored without resetting prev_code

    code_str = digits[0] + "".join(d for d in digits[1:] if d.isdigit())
    return code_str[:4].ljust(4, "0")


def compute_2token_soundex(cleaned_name: Optional[str]) -> str:
    """
    Compute 2-token Soundex lookup key:
    Extracts first 2 tokens (length >= 3, containing letters) and joins with '_'.
    """
    if not cleaned_name:
        return ""

    tokens = [
        t for t in str(cleaned_name).split()
        if len(t) >= MIN_SOUNDEX_TOKEN_LEN and any(c.isalpha() for c in t)
    ]
    if not tokens:
        return ""

    codes = [compute_soundex_token(t) for t in tokens[:MAX_SOUNDEX_TOKENS]]
    valid_codes = [c for c in codes if c]
    return "_".join(valid_codes)


# ==============================================================================
# 2. Batch Processing & Partitioning Pipeline
# ==============================================================================
def process_dataframe_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process a single batch/chunk of raw entity data:
    1. Fill null addresses with "" and record addr_missing_flag.
    2. Strip legal suffixes and transliterate names -> business_name_clean.
    3. Normalize addresses via ADDR_ABBREVS -> business_address_clean.
    4. Compute 2-token Soundex -> soundex_2token.
    5. Return DataFrame matching PROCESSED_SCHEMA_COLS.
    """
    # Check null or empty address
    is_addr_null = (
        pl.col(RAW_ADDR_COL).is_null()
        | (pl.col(RAW_ADDR_COL).fill_null("").str.strip_chars() == "")
    ).cast(pl.UInt8)

    # Clean address and names without iter_rows()
    df_clean = df.with_columns(
        [
            pl.col(RAW_NAME_COL).alias("business_name_raw"),
            pl.col(RAW_ADDR_COL).fill_null(""),
            is_addr_null.alias(ADDR_MISSING_FLAG_COL),
        ]
    )

    df_clean = df_clean.with_columns(
        [
            pl.col("business_name_raw")
            .map_elements(strip_legal_suffixes, return_dtype=pl.String)
            .alias(CLEAN_NAME_COL),
            pl.col(RAW_ADDR_COL)
            .map_elements(normalize_business_address, return_dtype=pl.String)
            .alias(CLEAN_ADDR_COL),
        ]
    )

    df_clean = df_clean.with_columns(
        [
            pl.col(CLEAN_NAME_COL)
            .map_elements(compute_2token_soundex, return_dtype=pl.String)
            .alias(SOUNDEX_COL)
        ]
    )

    return df_clean.select(PROCESSED_SCHEMA_COLS)


def process_source_file(
    split: str,
    source_name: str,
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    batch_size: int = STREAMING_BATCH_SIZE,
) -> Dict[str, int]:
    """
    Stream a single raw TSV file in batches, process normalization and Soundex,
    and export into country-partitioned zstd Parquet files.
    Returns dictionary of row counts per country.
    """
    tsv_path = raw_data_dir / split / f"{source_name}.tsv"
    if not tsv_path.exists():
        logger.warning(f"File not found: {tsv_path}")
        return {}

    logger.info(f"Processing '{split}/{source_name}.tsv' (Batch size: {batch_size:,})...")

    # Target parquet writers by country
    writers: Dict[str, pq.ParquetWriter] = {}
    country_counts: Dict[str, int] = {}
    pyarrow_schema = pa.schema(
        [
            (ENTITY_ID_COL, pa.string()),
            ("business_name_raw", pa.string()),
            (CLEAN_NAME_COL, pa.string()),
            (CLEAN_ADDR_COL, pa.string()),
            (SOUNDEX_COL, pa.string()),
            (COUNTRY_COL, pa.string()),
            (ADDR_MISSING_FLAG_COL, pa.uint8()),
        ]
    )

    reader = pl.read_csv_batched(
        tsv_path,
        separator="\t",
        has_header=True,
        batch_size=batch_size,
        infer_schema_length=10_000,
        truncate_ragged_lines=True,
    )

    total_processed = 0
    with track_memory(f"Ingest & Partition {split}/{source_name}"):
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            raw_batch = batches[0]
            if raw_batch.is_empty():
                continue

            cleaned_batch = process_dataframe_chunk(raw_batch)
            total_processed += len(cleaned_batch)

            # Partition batch by country
            for country in cleaned_batch[COUNTRY_COL].unique():
                c_str = str(country)
                country_df = cleaned_batch.filter(pl.col(COUNTRY_COL) == c_str)
                count = len(country_df)
                if count == 0:
                    continue

                country_counts[c_str] = country_counts.get(c_str, 0) + count

                # Initialize writer if not present
                if c_str not in writers:
                    out_dir = processed_data_dir / split / c_str
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{source_name}.parquet"
                    writers[c_str] = pq.ParquetWriter(
                        out_file,
                        schema=pyarrow_schema,
                        compression=PARQUET_COMPRESSION,
                        compression_level=PARQUET_COMPRESSION_LEVEL,
                    )

                # Write arrow table
                arrow_table = country_df.to_arrow().cast(pyarrow_schema)
                writers[c_str].write_table(arrow_table)

        # Close all parquet writers
        for c_str, writer in writers.items():
            writer.close()

    logger.info(
        f"Completed '{split}/{source_name}': {total_processed:,} total rows. "
        f"Partition counts: {country_counts}"
    )
    return country_counts


def run_preprocessing_pipeline(
    splits: Optional[List[str]] = None,
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Run the end-to-end preprocessing and partitioning pipeline for specified splits.
    Defaults to both 'train' and 'test'.
    """
    if splits is None:
        splits = ["train", "test"]

    results: Dict[str, Dict[str, Dict[str, int]]] = {}

    for split in splits:
        results[split] = {}
        sources = TRAIN_SOURCES if split == "train" else TEST_SOURCES
        for source in sources:
            counts = process_source_file(
                split=split,
                source_name=source,
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
            )
            results[split][source] = counts

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess and partition entity datasets.")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Split to preprocess ('train', 'test', or 'all')",
    )
    args = parser.parse_args()

    target_splits = ["train", "test"] if args.split == "all" else [args.split]
    run_preprocessing_pipeline(splits=target_splits)
