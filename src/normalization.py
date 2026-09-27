"""
Text and Address Normalization Module for Business Entity Resolution Pipeline.
Includes:
- Full ADDR_ABBREVS regex normalization mapping for US, India, and France.
- anyascii transliteration before Soundex and TF-IDF blocking (handling Indic scripts).
- Devanagari legal suffix pre-mapping + corp-names==0.3.3 + regex fallback.
"""

import re
import sys
import unicodedata
from typing import Optional

from src.utils import setup_utf8_stdout

setup_utf8_stdout()

# ==============================================================================
# 1. Indic Devanagari Legal Suffix Pre-Mapping
# ==============================================================================
DEVANAGARI_LEGAL_MAP = {
    "प्राइवेट लिमिटेड": " private limited",
    "प्रा. लि.": " pvt ltd",
    "प्रा० लि०": " pvt ltd",
    "लिमिटेड": " limited",
    "एलएलपी": " llp",
    "कंपनी": " company",
}

# Precompile Devanagari regexes
DEVANAGARI_PATTERNS = [
    (re.compile(re.escape(k), re.IGNORECASE), v)
    for k, v in DEVANAGARI_LEGAL_MAP.items()
]

# ==============================================================================
# 2. Comprehensive Fallback Regex Engine for Legal Suffixes
# ==============================================================================
FALLBACK_SUFFIX_PATTERNS = [
    # French commercial entity types
    r"\b(societe a responsabilite limitee|societe anonyme|societe par actions simplifiee|"
    r"societe civile immobiliere|groupement d interet economique|"
    r"sarl|sasu|sas|eurl|sci|snc|gie|eirl|sep|sa)\b",
    # Indian corporate entity types
    r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|llp|plc|opc)\b",
    # US corporate entity types
    r"\b(incorporated|inc\.?|corporation|corp\.?|limited liability company|"
    r"limited partnership|llc|pllc|pc|co\.?|company|lp|llp)\b",
    # Global / European corporate forms
    r"\b(gmbh|ag|bv|nv|spa|srl)\b",
    # Transliterated Indic legal forms (post-anyascii romanization)
    r"\b(praivet limited|praiveta limited|praivet|pvt\.?|ltd\.?|kampani)\b",
]

COMPILED_SUFFIX_REGEX = re.compile(
    r"(?:" + "|".join(FALLBACK_SUFFIX_PATTERNS) + r")(?:\.|\b)",
    re.IGNORECASE,
)

# ==============================================================================
# 3. Address Normalization: Compound Map & ADDR_ABBREVS
# ==============================================================================
COMPOUND_ADDR_MAP = [
    (re.compile(r"\bb/h\b", re.I), "behind"),
    (re.compile(r"\bopp\.?", re.I), "opposite"),
    (re.compile(r"\bh\.?\s*no\.?", re.I), "house number"),
    (re.compile(r"\bflat\s*no\.?", re.I), "flat number"),
    (re.compile(r"\bplot\s*no\.?", re.I), "plot number"),
    (re.compile(r"\bsh(?:o)?p\s*no\.?", re.I), "shop number"),
    (re.compile(r"\bkh\s*no\.?", re.I), "khasra number"),
    (re.compile(r"\bp\.?\s*o\.?", re.I), "post office"),
    (re.compile(r"\bc/o\b", re.I), "care of"),
    (re.compile(r"\bs/o\b", re.I), "son of"),
    (re.compile(r"\bd/o\b", re.I), "daughter of"),
    (re.compile(r"\bw/o\b", re.I), "wife of"),
]

ADDR_ABBREVS = {
    # --- US & Common English Street Suffixes ---
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "terr": "terrace",
    "trl": "trail",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "rm": "room",
    "dept": "department",
    "rt": "route",
    "rte": "route",
    "expy": "expressway",
    "tpke": "turnpike",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    # --- India Informal Address & Landmark Markers ---
    "nr": "near",
    "opp": "opposite",
    "bh": "behind",
    "adj": "adjacent",
    "col": "colony",
    "sec": "sector",
    "sect": "sector",
    "ph": "phase",
    "vill": "village",
    "dist": "district",
    "po": "post office",
    "mrkt": "market",
    "mkt": "market",
    "ind": "industrial",
    "indl": "industrial",
    "ext": "extension",
    "extn": "extension",
    "cplx": "complex",
    "cmplx": "complex",
    "twr": "tower",
    "twrs": "towers",
    "hsg": "housing",
    "soc": "society",
    "chbrs": "chambers",
    "chmbr": "chambers",
    "bg": "bungalow",
    "bglw": "bungalow",
    "no": "number",
    "nagar": "nagar",
    "rasta": "road",
    "marg": "road",
    # --- France Thoroughfare & Spatial Abbreviations ---
    "r": "rue",
    "all": "allee",
    "chem": "chemin",
    "ch": "chemin",
    "imp": "impasse",
    "crs": "cours",
    "pass": "passage",
    "qai": "quai",
    "bat": "batiment",
    "res": "residence",
    "appt": "appartement",
    "zi": "zone industrielle",
    "za": "zone d activite",
    "zac": "zone d amenagement concerte",
    "cedex": "cedex",
}


# ==============================================================================
# 4. Transliteration via anyascii (with NFKD Fallback)
# ==============================================================================
def transliterate_text(text: str) -> str:
    """
    Transliterate non-ASCII text (e.g. Indic Devanagari/Tamil or accented Latin)
    into clean ASCII representations.
    Uses `anyascii` if available, with NFKD accent-stripping fallback.
    """
    if not text:
        return ""

    try:
        import anyascii
        return anyascii.anyascii(text)
    except Exception:
        # Fallback for accented Latin
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))


# ==============================================================================
# 5. Business Name Normalization & Legal Suffix Stripping
# ==============================================================================
def strip_legal_suffixes(name: str) -> str:
    """
    Robust legal suffix stripping using:
    1. Indic Devanagari legal suffix pre-mapping
    2. anyascii transliteration
    3. corp-names==0.3.3 (when available)
    4. Comprehensive regex fallback for US, India, France, and transliterated forms
    """
    if not name or not str(name).strip():
        return ""

    text = str(name).strip()

    # Step 1: Devanagari legal suffix pre-mapping
    for pattern, replacement in DEVANAGARI_PATTERNS:
        text = pattern.sub(replacement, text)

    # Step 2: Transliterate to ASCII
    text = transliterate_text(text)

    # Step 3: Attempt corp-names if available
    cleaned_by_corp = None
    try:
        import corp_names
        # corp-names 0.3.3 provides normalize_company_name or CompanyNormalizer
        if hasattr(corp_names, "normalize_company_name"):
            res = corp_names.normalize_company_name(text)
            if hasattr(res, "normalized") and res.normalized:
                cleaned_by_corp = res.normalized
            elif hasattr(res, "original") and res.original:
                cleaned_by_corp = res.original
        elif hasattr(corp_names, "CompanyNormalizer"):
            normalizer = corp_names.CompanyNormalizer()
            res = normalizer.normalize(text)
            if hasattr(res, "normalized") and res.normalized:
                cleaned_by_corp = res.normalized
    except Exception:
        pass

    target_text = cleaned_by_corp if cleaned_by_corp else text

    # Step 4: Apply compiled regex fallback
    stripped = COMPILED_SUFFIX_REGEX.sub(" ", target_text)

    # Step 5: Clean punctuation and normalize whitespace
    stripped = re.sub(r"[^\w\s]", " ", stripped)
    cleaned = " ".join(stripped.split()).lower()

    # Safety check: If suffix stripping removed the entire name (e.g. "LLC"), retain original
    if not cleaned:
        fallback = re.sub(r"[^\w\s]", " ", text)
        return " ".join(fallback.split()).lower()

    return cleaned


# ==============================================================================
# 6. Business Address Normalization
# ==============================================================================
def normalize_business_address(address: Optional[str]) -> str:
    """
    Offline address abbreviation normalizer:
    1. Null / NaN check -> returns ""
    2. NFKD accent stripping for French / European tokens
    3. Compound abbreviation expansion (e.g. b/h, opp., h.no)
    4. Punctuation stripping
    5. Token-level dictionary lookup against ADDR_ABBREVS
    6. Non-alphanumeric cleanup & whitespace collapsing
    """
    if address is None:
        return ""

    s_addr = str(address).strip()
    if not s_addr or s_addr.lower() in ("nan", "none", "null"):
        return ""

    # NFKD strip accents for French / European tokens
    text = "".join(
        c for c in unicodedata.normalize("NFKD", s_addr) if not unicodedata.combining(c)
    )
    text = text.lower()

    # Step 1: Replace compound slash/dot abbreviations
    for pattern, replacement in COMPOUND_ADDR_MAP:
        text = pattern.sub(replacement, text)

    # Step 2: Convert punctuation to spaces
    text = re.sub(r"[\/\\,\.\-\(\)#:;]", " ", text)

    # Step 3: Replace word tokens via ADDR_ABBREVS
    tokens = text.split()
    normalized_tokens = [ADDR_ABBREVS.get(t, t) for t in tokens]
    text = " ".join(normalized_tokens)

    # Step 4: Clean extra non-alphanumeric characters & whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())
