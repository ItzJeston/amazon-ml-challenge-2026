"""
Feature engineering module for pairwise business entity resolution.
Computes multi-dimensional string, token, digit, and structural similarities
between S1 reference entities and S2/S3 candidate entities using RapidFuzz.
Includes house / building number exact conflict detection.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "addr_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_is_null_s1",
    "addr_is_null_cand",
    "digits_jaccard",
    "first_digit_match",
    "name_len_diff",
    "name_len_ratio",
    "first_char_match",
    "is_source2",
    "house_conflict",
]


def check_house_conflict(s1_digits: List[str], cand_digits: List[str]) -> float:
    """Detect if both addresses contain explicit street/building numbers that completely conflict.
    
    Returns 1.0 if both have numbers, their primary numbers differ, and neither contains the other's number.
    Returns 0.0 otherwise.
    """
    if s1_digits and cand_digits:
        h1 = s1_digits[0]
        h2 = cand_digits[0]
        if h1 != h2 and (h1 not in cand_digits) and (h2 not in s1_digits):
            return 1.0
    return 0.0


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_digits: List[str],
    cand_id: str,
    cand_name: str,
    cand_addr: str,
    cand_digits: List[str],
) -> List[float]:
    """Extract 17 high-signal similarity features for a single candidate pair."""
    # 1. Name similarities
    n_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    n_part = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    n_tsort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    n_tset = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    n_jw = distance.JaroWinkler.similarity(s1_name, cand_name)

    # 2. Address similarities
    has_a1 = bool(s1_addr)
    has_a2 = bool(cand_addr)
    a1_null = 0.0 if has_a1 else 1.0
    a2_null = 0.0 if has_a2 else 1.0

    if has_a1 and has_a2:
        a_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
        a_tset = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        a_part = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
    else:
        a_ratio = 0.0
        a_tset = 0.0
        a_part = 0.0

    # 3. Digit / House number / Postal code overlaps
    d1_set = set(s1_digits)
    d2_set = set(cand_digits)
    if d1_set and d2_set:
        inter = len(d1_set & d2_set)
        union = len(d1_set | d2_set)
        dig_jaccard = inter / union if union > 0 else 0.0
        first_dig_match = 1.0 if s1_digits[0] == cand_digits[0] else 0.0
    elif not d1_set and not d2_set:
        dig_jaccard = 1.0
        first_dig_match = 0.0
    else:
        dig_jaccard = 0.0
        first_dig_match = 0.0

    # 4. Structural differences
    len1 = len(s1_name)
    len2 = len(cand_name)
    len_diff = float(abs(len1 - len2))
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0
    
    first_char_match = 1.0 if (s1_name and cand_name and s1_name[0] == cand_name[0]) else 0.0
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    # 5. House / building number complete conflict
    house_conflict = check_house_conflict(s1_digits, cand_digits)

    return [
        n_ratio,
        n_part,
        n_tsort,
        n_tset,
        n_jw,
        a_ratio,
        a_tset,
        a_part,
        a1_null,
        a2_null,
        dig_jaccard,
        first_dig_match,
        len_diff,
        len_ratio,
        first_char_match,
        is_s2,
        house_conflict,
    ]


def extract_batch_features(
    batch_pairs: List[Tuple[str, str, List[str], str, str, str, List[str]]]
) -> np.ndarray:
    """Extract feature matrix for a batch of candidate pairs.
    
    Args:
        batch_pairs: List of (s1_name, s1_addr, s1_digits, cand_id, cand_name, cand_addr, cand_digits)
    Returns:
        2D numpy array of shape (N, 17)
    """
    n = len(batch_pairs)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i, pair in enumerate(batch_pairs):
        features[i] = compute_pair_features(*pair)
    return features
