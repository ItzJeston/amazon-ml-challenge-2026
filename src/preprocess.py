"""
Preprocess module for business entity resolution.
Handles text cleaning, abbreviation standardization, legal suffix normalization,
digit extraction, and country-partitioned data ingestion.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
import polars as pl

# Multilingual legal entity suffixes
# Removing these is critical for TF-IDF, otherwise "ltd" or "inc" generates 
# high-weight n-grams that match entirely unrelated companies.
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b('
    r'incorporated|inc|corporation|corp|limited\s+liability\s+company|llc|'
    r'limited\s+liability\s+partnership|llp|limited\s+partnership|lp|limited|ltd|'
    r'company|co|professional\s+corporation|pc|pllc|'
    r'private\s+limited|pvt\s+ltd|pvt|p\s+ltd|pvt\.\s*ltd\.|'
    r'societe\s+anonyme|sarl|sasu|sas|eurl|snc|gie|sca|scs|sa|'
    r'holdings?|enterprises?|services?|solutions?|group|consulting'
    r')\b',
    re.IGNORECASE,
)

# Address abbreviations standardization
# Flipped to EXPAND abbreviations. For character n-grams, "street" and "street" 
# generate more overlapping tokens than "st" and "st".
ADDRESS_ABBR_MAP = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bln\b': 'lane',
    r'\bdr\b': 'drive',
    r'\bhwy\b': 'highway',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bbldg\b': 'building',
}
ADDRESS_ABBR_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBR_MAP.items()
]

PUNCT_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
SPACE_REGEX = re.compile(r'\s+')
DIGITS_REGEX = re.compile(r'\d+')

def strip_accents(text: str) -> str:
    """Normalize unicode and strip diacritical marks (e.g., é -> e)."""
    if not text or not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def clean_name(name: Optional[str]) -> str:
    """Clean and normalize business name."""
    if not name or not isinstance(name, str):
        return ""
    text = strip_accents(name).lower()
    text = LEGAL_SUFFIXES_PATTERN.sub(' ', text)
    text = PUNCT_REGEX.sub(' ', text)
    text = SPACE_REGEX.sub(' ', text).strip()
    return text

def clean_address(address: Optional[str]) -> str:
    """Clean and standardize business address."""
    if not address or not isinstance(address, str):
        return ""
    text = strip_accents(address).lower()
    # Strip punctuation first so things like "st." become "st" and get caught by the map
    text = PUNCT_REGEX.sub(' ', text)
    for pattern, repl in ADDRESS_ABBR_COMPILED:
        text = pattern.sub(repl, text)
    text = SPACE_REGEX.sub(' ', text).strip()
    return text

def extract_digits(text: Optional[str]) -> List[str]:
    """Extract list of digit sequences from text."""
    if not text or not isinstance(text, str):
        return []
    return DIGITS_REGEX.findall(text)

def load_partition(
    split_dir: str,
    source_name: str,
    country: str,
) -> pl.DataFrame:
    """Load a specific country partition for a source file safely."""
    prefix = "test" if "test" in split_dir else "train"
    path = f"{split_dir}/{prefix}_{source_name}.tsv"
    
    # ADDED infer_schema_length=0: Forces Polars to treat everything as strings immediately.
    # This prevents OOM (Out of Memory) spikes when Pandas/Polars tries to guess data types on large files.
    df = (
        pl.scan_csv(path, separator="\t", infer_schema_length=0)
        .filter(pl.col("country") == country)
        .collect()
    )
    return df

def get_available_countries(split_dir: str) -> List[str]:
    """Get list of unique countries present in test_source1 or train_source1."""
    prefix = "test" if "test" in split_dir else "train"
    path = f"{split_dir}/{prefix}_source1.tsv"
    countries = (
        pl.scan_csv(path, separator="\t", infer_schema_length=0)
        .select("country")
        .collect()["country"]
        .unique()
        .to_list()
    )
    return sorted([c for c in countries if c])