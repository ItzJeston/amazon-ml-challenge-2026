"""
tests/e2e/helpers.py
Shared utilities, specification reference oracles, and test fixtures for E2E tests.
Amazon Business Entity Resolution Challenge
"""

import os
import re
import sys
import io
import math
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

# Ensure stdout uses UTF-8 encoding
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Hardware & Configuration Constants
# ---------------------------------------------------------------------------
TARGET_CPU_THREADS = 14
TARGET_PEAK_RAM_GB = 6.0
MAX_RAM_LIMIT_GB = 10.0
TARGET_GPU_DEVICE = "cuda"

COUNTRIES = ["US", "India", "France"]

# ---------------------------------------------------------------------------
# Reference Address Abbreviations & Regex Patterns
# ---------------------------------------------------------------------------
COMPOUND_ADDR_MAP = [
    (re.compile(r'\bb/h\b', re.IGNORECASE), 'behind'),
    (re.compile(r'\bopp\.?', re.IGNORECASE), 'opposite'),
    (re.compile(r'\bh\.?\s*no\.?', re.IGNORECASE), 'house number'),
    (re.compile(r'\bflat\s*no\.?', re.IGNORECASE), 'flat number'),
    (re.compile(r'\bplot\s*no\.?', re.IGNORECASE), 'plot number'),
    (re.compile(r'\bsh(?:o)?p\s*no\.?', re.IGNORECASE), 'shop number'),
    (re.compile(r'\bkh\s*no\.?', re.IGNORECASE), 'khasra number'),
    (re.compile(r'\bp\.?\s*o\.?', re.IGNORECASE), 'post office'),
    (re.compile(r'\bc/o\b', re.IGNORECASE), 'care of'),
    (re.compile(r'\bs/o\b', re.IGNORECASE), 'son of'),
    (re.compile(r'\bd/o\b', re.IGNORECASE), 'daughter of'),
    (re.compile(r'\bw/o\b', re.IGNORECASE), 'wife of'),
]

ADDR_ABBREVS = {
    # US & Common Street Suffixes
    'rd': 'road',
    'st': 'street',
    'str': 'street',
    'ave': 'avenue',
    'av': 'avenue',
    'blvd': 'boulevard',
    'bd': 'boulevard',
    'dr': 'drive',
    'ln': 'lane',
    'hwy': 'highway',
    'ct': 'court',
    'pl': 'place',
    'pkwy': 'parkway',
    'cir': 'circle',
    'ter': 'terrace',
    'terr': 'terrace',
    'trl': 'trail',
    'sq': 'square',
    'ste': 'suite',
    'apt': 'apartment',
    'bldg': 'building',
    'fl': 'floor',
    'rm': 'room',
    'dept': 'department',
    'rt': 'route',
    'rte': 'route',
    'expy': 'expressway',
    'tpke': 'turnpike',
    # India Informal Address & Landmark Markers
    'nr': 'near',
    'opp': 'opposite',
    'bh': 'behind',
    'ext': 'extension',
    'col': 'colony',
    'mkt': 'market',
    'bzd': 'bazaar',
    'soc': 'society',
    'chs': 'cooperative housing society',
    'indl': 'industrial',
    'area': 'area',
    'nagar': 'nagar',
    'marg': 'marg',
    'rly': 'railway',
    'stn': 'station',
    'gt': 'grand trunk',
    # France Address Markers
    'bd': 'boulevard',
    'av': 'avenue',
    'r': 'rue',
    'rte': 'route',
    'pl': 'place',
    'all': 'allee',
    'imp': 'impasse',
    'che': 'chemin',
    'sq': 'square',
    'crs': 'cours',
    'fg': 'faubourg',
    'res': 'residence',
    'bat': 'batiment',
    'etg': 'etage',
    'porte': 'porte',
    'esc': 'escalier',
    'zi': 'zone industrielle',
    'za': 'zone activite',
    'zac': 'zone amenagement concerte',
}

DEVANAGARI_LEGAL_MAP = {
    "प्राइवेट लिमिटेड": " private limited",
    "प्रा. लि.": " pvt ltd",
    "प्रा० लि०": " pvt ltd",
    "लिमिटेड": " limited",
    "एलएलपी": " llp",
    "कंपनी": " company",
}

LEGAL_SUFFIX_PATTERNS = [
    # French commercial entities
    r'\b(societe a responsabilite limitee|societe anonyme|societe par actions simplifiee|'
    r'sarl|sas|sasu|sa|eurl|sci|snc|gie|eirl|sep)\b',
    # Indian corporate entities
    r'\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|llp|plc)\b',
    # US corporate entities
    r'\b(incorporated|inc\.?|corporation|corp\.?|limited liability company|llc|pllc|pc|co\.?|company|lp)\b',
    # Global / European forms
    r'\b(gmbh|ag|bv|nv|spa|srl)\b',
    # Transliterated Indic legal forms
    r'\b(praivet limited|praiveta limited|praivet|pvt\.?|ltd\.?)\b',
]
COMPILED_LEGAL_SUFFIX_REGEX = re.compile(
    r'(?:' + '|'.join(LEGAL_SUFFIX_PATTERNS) + r')(?:\.|\b)', 
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Reference Implementation Oracles (Pure Python, Zero Dependency)
# ---------------------------------------------------------------------------

def reference_normalize_address(addr: Optional[str]) -> str:
    """Normalize address using compound patterns and ADDR_ABBREVS dictionary."""
    if not addr or not isinstance(addr, str) or not addr.strip():
        return ""
    text = addr.lower()
    # 1. Apply compound mappings
    for pattern, replacement in COMPOUND_ADDR_MAP:
        text = pattern.sub(f" {replacement} ", text)
    # 2. Clean punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    # 3. Expand abbreviations
    expanded = [ADDR_ABBREVS.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


def reference_strip_legal_suffix(name: str) -> Tuple[str, str]:
    """Pre-map Devanagari, strip legal suffixes, return (clean_name, extracted_suffix)."""
    if not name or not isinstance(name, str):
        return ("", "")
    text = name.strip()
    # Pre-map Devanagari
    for dev_suffix, eng_suffix in DEVANAGARI_LEGAL_MAP.items():
        if dev_suffix in text:
            text = text.replace(dev_suffix, eng_suffix)
    # Find match
    match = COMPILED_LEGAL_SUFFIX_REGEX.search(text)
    suffix = ""
    if match:
        suffix = match.group(0).strip(" .").lower()
        text = COMPILED_LEGAL_SUFFIX_REGEX.sub(" ", text)
    clean_name = re.sub(r'\s+', ' ', text).strip()
    return (clean_name, suffix)


def reference_transliterate(text: str) -> str:
    """Transliterate Unicode text to Latin ASCII (oracle implementation)."""
    if not text:
        return ""
    # Try anyascii if installed
    try:
        import anyascii
        return anyascii.anyascii(text)
    except ImportError:
        pass
    # Basic NFKD accent stripping for Latin characters
    nfkd = "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    # Basic mapping for known Devanagari tokens
    devanagari_sample = {
        "राम": "Ram", "मार्केटिंग": "Marketing", "प्राइवेट": "Private",
        "लिमिटेड": "Limited", "कंपनी": "Company", "स्टोर्स": "Stores",
        "इंटरप्राइजेज": "Enterprises", "ओम": "Om", "ट्रेडर्स": "Traders"
    }
    for dev, eng in devanagari_sample.items():
        nfkd = nfkd.replace(dev, eng)
    return nfkd


def reference_single_soundex(word: str) -> str:
    """Standard American Soundex algorithm (pure Python oracle)."""
    word = re.sub(r'[^a-zA-Z]', '', word).upper()
    if not word:
        return "0000"
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first_char = word[0]
    tail = word[1:]
    encoded = []
    prev_code = mapping.get(first_char, '0')
    for char in tail:
        code = mapping.get(char, '0')
        if code != '0' and code != prev_code:
            encoded.append(code)
        prev_code = code
    soundex_digits = "".join(encoded)
    soundex_code = (first_char + soundex_digits + "000")[:4]
    return soundex_code


def reference_2token_soundex(name: str) -> str:
    """Generate 2-token Soundex in 'X000_Y000' format."""
    clean = reference_transliterate(name)
    tokens = re.findall(r'[a-zA-Z]+', clean)
    if not tokens:
        return "0000_0000"
    tok1 = tokens[0]
    tok2 = tokens[1] if len(tokens) > 1 else None
    c1 = reference_single_soundex(tok1)
    c2 = reference_single_soundex(tok2) if tok2 else "0000"
    return f"{c1}_{c2}"


def compute_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]],
    all_s1_ids: Set[str]
) -> float:
    """
    Computes exact Macro F0.5 per S1 entity with competition singleton rules:
    - Singletons (true matches = 0):
        * predicted empty -> 1.0
        * predicted >= 1 match -> 0.0 (false merge)
    - Non-singletons:
        * predicted empty -> 0.0 (missed link)
        * precision = TP / (TP + FP)
        * recall = TP / (TP + FN)
        * F0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Macro average across all S1 entities.
    """
    if not all_s1_ids:
        return 0.0
    scores = []
    for s1_id in all_s1_ids:
        true_set = ground_truth_dict.get(s1_id, set())
        pred_set = predictions_dict.get(s1_id, set())

        if len(true_set) == 0:
            # Singleton evaluation
            if len(pred_set) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            # Non-singleton evaluation
            if len(pred_set) == 0:
                scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                denom = 0.25 * prec + rec
                f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
                scores.append(f05)
    return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Synthetic Dataset Generators for Realistic Testing
# ---------------------------------------------------------------------------

SAMPLE_RAW_DATA = {
    "train_source1": [
        ("S1-10001", "Apple Store Manhattan Inc", "767 5th Ave, New York, NY 10153", "US"),
        ("S1-10002", "Starbucks Coffee Co", "1912 Pike Pl, Seattle, WA 98101", "US"),
        ("S1-10003", "Ram Marketing Pvt Ltd", "Plot No 42, Near Railway Station, Pune", "India"),
        ("S1-10004", "Tata Consultancy Services Ltd", "B/H TCS Banyan Park, Andheri East, Mumbai", "India"),
        ("S1-10005", "Acme Desert Singleton Corp", "101 Lone Highway, Phoenix, AZ", "US"),  # Singleton!
    ],
    "train_source2": [
        ("S2-20001", "Apple Store", "767 5th Avenue, New York", "US"),
        ("S2-20002", "Starbucks", "1912 Pike Place, Seattle", "US"),
        ("S2-20003", "राम मार्केटिंग प्राइवेट लिमिटेड", "Plot 42, Nr Rly Stn, Pune", "India"),
        ("S2-20004", "Tata Consultancy Services", "Behind TCS Banyan Park, Andheri, Mumbai", "India"),
        ("S2-20005", "Independent Pizza", "500 Main St, Chicago, IL", "US"),
    ],
    "train_source3": [
        ("S3-30001", "Apple Retail LLC", "767 Fifth Ave, NYC", "US"),
        ("S3-30002", "Starbucks Cafe", "Pike Pl Market, Seattle", "US"),
        ("S3-30003", "Ram Marketing", "Plot No. 42 Pune", "India"),
        ("S3-30004", "TCS Limited", "Andheri East Mumbai", "India"),
        ("S3-30005", "Bengaluru Tech Hub", "Electronic City Phase 1", "India"),
    ],
    "train_ground_truth": [
        ("S1-10001", "S2-20001,S3-30001"),
        ("S1-10002", "S2-20002,S3-30002"),
        ("S1-10003", "S2-20003,S3-30003"),
        ("S1-10004", "S2-20004,S3-30004"),
        ("S1-10005", ""),  # Singleton
    ],
    "test_source1": [
        ("S1-90001", "Microsoft Store Bellevue", "Bellevue Square, Bellevue, WA", "US"),
        ("S1-90002", "Infosys Limited", "Electronics City, Hosur Road, Bangalore", "India"),
        ("S1-90003", "Elephant Centre EURL", "30 Rue Lachassaigne, Bordeaux", "France"),
        ("S1-90004", "Solo French Baker SAS", "12 Rue de la Paix, Paris", "France"),  # Singleton
    ],
    "test_source2": [
        ("S2-80001", "Microsoft Retail Store", "Bellevue Sq, Bellevue", "US"),
        ("S2-80002", "इंफोसिस लिमिटेड", "Hosur Rd, Electronic City, Bengaluru", "India"),
        ("S2-80003", "Elephant Centre", "30 R Lachassaigne, Bordeaux", "France"),
        ("S2-80004", "Random Paris Bistro", "5 Avenue Victor Hugo, Paris", "France"),
    ],
    "test_source3": [
        ("S3-70001", "Microsoft", "Bellevue Square Mall", "US"),
        ("S3-70002", "Infosys Technologies Ltd", "Bangalore Electronics City", "India"),
        ("S3-70003", "Centre Elephant", "30 Rue Lachassaigne", "France"),
        ("S3-70004", "Bordeaux Wine Merchant", "1 Quai des Chartrons, Bordeaux", "France"),
    ],
}
