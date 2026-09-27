"""
Unit tests for text and address normalization.
Tests ADDR_ABBREVS, anyascii transliteration, Devanagari legal suffix pre-mapping,
corp-names, and regex fallback across US, India, and France.
"""

import pytest

from src.normalization import (
    normalize_business_address,
    strip_legal_suffixes,
    transliterate_text,
)


# ==============================================================================
# 1. Address Normalization Tests
# ==============================================================================
class TestAddressNormalization:
    def test_us_address_abbreviations(self):
        raw = "123 Main St., Apt. 4B, New York, NY"
        cleaned = normalize_business_address(raw)
        assert "street" in cleaned
        assert "apartment" in cleaned
        assert "." not in cleaned
        assert "," not in cleaned

    def test_us_street_suffixes_and_directionals(self):
        raw = "456 Oak Rd, Suite 100, Blvd, Hwy 101, N Ave"
        cleaned = normalize_business_address(raw)
        assert "road" in cleaned
        assert "suite" in cleaned
        assert "boulevard" in cleaned
        assert "highway" in cleaned
        assert "north" in cleaned
        assert "avenue" in cleaned

    def test_india_compound_patterns(self):
        raw = "B/H City Mall, Opp. Metro Station, H. No. 42, Sec 15, Vill Rampur"
        cleaned = normalize_business_address(raw)
        assert "behind" in cleaned
        assert "opposite" in cleaned
        assert "house number" in cleaned
        assert "sector" in cleaned
        assert "village" in cleaned
        assert "/" not in cleaned

    def test_india_informal_markers(self):
        raw = "Plot No. 12, Indl Area, Extn 2, Twr B, Soc Complex, Dist Thane"
        cleaned = normalize_business_address(raw)
        assert "plot number" in cleaned
        assert "industrial" in cleaned
        assert "extension" in cleaned
        assert "tower" in cleaned
        assert "society" in cleaned
        assert "district" in cleaned

    def test_india_care_of_and_post_office(self):
        raw = "C/O Ramesh Sharma, Flat No. 3, Nr Bus Stand, P.O. Box 45"
        cleaned = normalize_business_address(raw)
        assert "care of" in cleaned
        assert "flat number" in cleaned
        assert "near" in cleaned
        assert "post office" in cleaned

    def test_france_thoroughfare_abbreviations(self):
        raw = "15 R de la Paix, Bat A, Res Les Pins, Cedex 06"
        cleaned = normalize_business_address(raw)
        assert "rue" in cleaned
        assert "batiment" in cleaned
        assert "residence" in cleaned
        assert "cedex" in cleaned

    def test_france_spatial_abbreviations(self):
        raw = "Chem des Fleurs, Impasse du Moulin, ZI Nord"
        cleaned = normalize_business_address(raw)
        assert "chemin" in cleaned
        assert "impasse" in cleaned
        assert "zone industrielle" in cleaned

    def test_france_accents_stripped(self):
        raw = "30 Rue Lachassaigne, Éléphant Centre, Résidence Méditerranée"
        cleaned = normalize_business_address(raw)
        assert "rue" in cleaned
        assert "elephant" in cleaned
        assert "residence" in cleaned
        assert "mediterranee" in cleaned
        # Must be pure ASCII alphanumeric with spaces
        assert all(ord(c) < 128 for c in cleaned)

    def test_null_empty_nan_address(self):
        assert normalize_business_address(None) == ""
        assert normalize_business_address("") == ""
        assert normalize_business_address("   ") == ""
        assert normalize_business_address("NaN") == ""
        assert normalize_business_address("nan") == ""
        assert normalize_business_address("None") == ""
        assert normalize_business_address("null") == ""


# ==============================================================================
# 2. Transliteration Tests
# ==============================================================================
class TestTransliteration:
    def test_devanagari_transliteration(self):
        hindi_text = "राम मार्केटिंग"
        trans = transliterate_text(hindi_text)
        assert all(ord(c) < 128 for c in trans)
        assert len(trans.strip()) > 0
        assert "ram" in trans.lower() or "marketing" in trans.lower()

    def test_french_accents_transliteration(self):
        french_text = "Éléphant Centre EURL"
        trans = transliterate_text(french_text)
        assert "Elephant Centre" in trans
        assert all(ord(c) < 128 for c in trans)

    def test_empty_transliteration(self):
        assert transliterate_text("") == ""
        assert transliterate_text(None) == ""


# ==============================================================================
# 3. Legal Suffix Stripping Tests
# ==============================================================================
class TestLegalSuffixStripping:
    def test_devanagari_legal_suffix_pre_mapping(self):
        name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        cleaned = strip_legal_suffixes(name)
        assert "limited" not in cleaned
        assert "private" not in cleaned
        assert "pvt" not in cleaned
        assert "ltd" not in cleaned
        assert len(cleaned) > 0
        assert all(ord(c) < 128 for c in cleaned)

    def test_devanagari_limited_suffix(self):
        name = "अग्रवाल स्टील लिमिटेड"
        cleaned = strip_legal_suffixes(name)
        assert "limited" not in cleaned
        assert len(cleaned) > 0

    def test_devanagari_llp_suffix(self):
        name = "विकास कंसल्टेंसी एलएलपी"
        cleaned = strip_legal_suffixes(name)
        assert "llp" not in cleaned
        assert len(cleaned) > 0

    def test_us_legal_suffixes(self):
        cases = [
            ("Acme Corporation", "acme"),
            ("Global Solutions, Inc.", "global solutions"),
            ("Alpha Beta LLC", "alpha beta"),
            ("Omega Limited Liability Company", "omega"),
            ("Starlight Technologies Incorporated", "starlight technologies"),
            ("Apex Holdings Co.", "apex holdings"),
            ("Metro Logistics Corp.", "metro logistics"),
        ]
        for raw, expected in cases:
            cleaned = strip_legal_suffixes(raw)
            assert cleaned == expected or expected in cleaned

    def test_india_legal_suffixes(self):
        cases = [
            ("Tata Steel Limited", "tata steel"),
            ("Reliance Retail Private Limited", "reliance retail"),
            ("Infosys Pvt Ltd", "infosys"),
            ("Adani Power Ltd.", "adani power"),
            ("Bharat Heavy Electricals Ltd", "bharat heavy electricals"),
        ]
        for raw, expected in cases:
            cleaned = strip_legal_suffixes(raw)
            assert cleaned == expected or expected in cleaned

    def test_france_legal_suffixes(self):
        cases = [
            ("Elephant Centre EURL", "elephant centre"),
            ("Societe Generale SA", "societe generale"),
            ("Dassault Aviation SASU", "dassault aviation"),
            ("Leroy Merlin SARL", "leroy merlin"),
            ("Airbus SAS", "airbus"),
        ]
        for raw, expected in cases:
            cleaned = strip_legal_suffixes(raw)
            assert cleaned == expected or expected in cleaned

    def test_transliterated_indic_suffixes(self):
        raw = "Shree Ram Trading Praivet Limited"
        cleaned = strip_legal_suffixes(raw)
        assert "praivet" not in cleaned
        assert "limited" not in cleaned
        assert "shree ram trading" in cleaned

    def test_punctuation_and_noisy_inputs(self):
        assert strip_legal_suffixes("<< Team Ecole >>") == "team ecole"
        assert "b retail" in strip_legal_suffixes("B+ Retail Inc.")

    def test_standalone_suffix_does_not_vanish(self):
        # When name is literally only the suffix, fallback must retain non-empty string
        assert strip_legal_suffixes("LLC") == "llc"
        assert strip_legal_suffixes("Inc") == "inc"

    def test_empty_and_whitespace(self):
        assert strip_legal_suffixes("") == ""
        assert strip_legal_suffixes("   ") == ""
        assert strip_legal_suffixes(None) == ""
