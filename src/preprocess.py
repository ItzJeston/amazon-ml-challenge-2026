"""
Preprocess module for business entity resolution.
Handles text cleaning, abbreviation standardization, legal suffix normalization,
digit extraction, and country-partitioned data ingestion.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
import polars as pl

# Multilingual legal entity suffixes (US, India, France)
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b('
    # US / UK
    r'incorporated|inc|corporation|corp|limited\s+liability\s+company|llc|'
    r'limited\s+liability\s+partnership|llp|limited\s+partnership|lp|limited|ltd|'
    r'company|co|professional\s+corporation|pc|pllc|'
    # India
    r'private\s+limited|pvt\s+ltd|pvt|p\s+ltd|pvt\.\s*ltd\.|'
    # France
    r'societe\s+anonyme|sarl|sasu|sas|eurl|snc|gie|sca|scs|sa|'
    # Common generic business descriptors
    r'holdings?|enterprises?|services?|solutions?|group|consulting'
    r')\b',
    re.IGNORECASE,
)

# Address abbreviations standardization
ADDRESS_ABBR_MAP = {
    r'\bstreet\b': 'st',
    r'\broad\b': 'rd',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd',
    r'\blane\b': 'ln',
    r'\bdrive\b': 'dr',
    r'\bhighway\b': 'hwy',
    r'\bapartment\b': 'apt',
    r'\bsuite\b': 'ste',
    r'\bfloor\b': 'fl',
    r'\bbuilding\b': 'bldg',
    # French abbreviations
    r'\brue\b': 'r',
    r'\bav\b': 'ave',
    r'\bbd\b': 'blvd',
}
ADDRESS_ABBR_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBR_MAP.items()
]

PUNCT_REGEX = re.compile(r'[^a-zA-Z0-9\s]')
SPACE_REGEX = re.compile(r'\s+')
DIGITS_REGEX = re.compile(r'\b\d+\b')


def strip_accents(text: str) -> str:
    """Normalize unicode and strip diacritical marks (e.g., é -> e)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_name(name: Optional[str]) -> str:
    """Clean and normalize business name.
    
    Strips accents, legal suffixes, punctuation, and extra whitespace.
    """
    if not name:
        return ""
    text = strip_accents(name).lower()
    text = LEGAL_SUFFIXES_PATTERN.sub(' ', text)
    text = PUNCT_REGEX.sub(' ', text)
    text = SPACE_REGEX.sub(' ', text).strip()
    return text


def clean_address(address: Optional[str]) -> str:
    """Clean and standardize business address.
    
    Strips accents, standardizes street abbreviations, removes punctuation.
    """
    if not address:
        return ""
    text = strip_accents(address).lower()
    for pattern, repl in ADDRESS_ABBR_COMPILED:
        text = pattern.sub(repl, text)
    text = PUNCT_REGEX.sub(' ', text)
    text = SPACE_REGEX.sub(' ', text).strip()
    return text


def extract_digits(text: Optional[str]) -> List[str]:
    """Extract list of digit sequences from text."""
    if not text:
        return []
    return DIGITS_REGEX.findall(text)


def load_partition(
    split_dir: str,
    source_name: str,
    country: str,
) -> pl.DataFrame:
    """Load a specific country partition for a source file without OOM.
    
    Args:
        split_dir: Directory containing TSVs (e.g. 'dataset/test' or 'dataset/train')
        source_name: Name of source, e.g. 'source1', 'source2', 'source3'
        country: Country string, e.g. 'France', 'India', 'US'
    Returns:
        Filtered Polars DataFrame for that country.
    """
    # File naming format: {split_dir}/{split_prefix}_{source_name}.tsv
    # For example dataset/test/test_source1.tsv
    prefix = "test" if "test" in split_dir else "train"
    path = f"{split_dir}/{prefix}_{source_name}.tsv"
    
    df = (
        pl.scan_csv(path, separator="\t")
        .filter(pl.col("country") == country)
        .collect()
    )
    return df


def get_available_countries(split_dir: str) -> List[str]:
    """Get list of unique countries present in test_source1 or train_source1."""
    prefix = "test" if "test" in split_dir else "train"
    path = f"{split_dir}/{prefix}_source1.tsv"
    countries = (
        pl.scan_csv(path, separator="\t")
        .select("country")
        .collect()["country"]
        .unique()
        .to_list()
    )
    return sorted(countries)
