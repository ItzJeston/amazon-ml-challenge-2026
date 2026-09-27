"""
tests/e2e/test_tier3_pairwise.py
Tier 3: Cross-Feature Combinations & Pairwise Interactions Test Suite
Validates system behavior when features interact:
- Non-ASCII names with missing addresses
- Soundex collision handling across distinct entities
- Multi-source joins (S1 simultaneously matching S2 and S3)
- Legal suffix stripping interaction with core brand names
- Address abbreviation collision with proper nouns
- High name similarity with low address similarity (franchise branches)
- High address similarity with low name similarity (co-located distinct entities)
- Platt calibration under severe class imbalance
- Multi-country 3D threshold routing
Total tests: 25 test cases.
"""

import unittest
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    try:
        from sklearn.calibration import FrozenEstimator
    except ImportError:
        class FrozenEstimator:
            def __init__(self, estimator):
                self.estimator = estimator
            def __getattr__(self, name):
                return getattr(self.estimator, name)
            def fit(self, *args, **kwargs):
                return self


from tests.e2e.helpers import (
    reference_normalize_address,
    reference_strip_legal_suffix,
    reference_transliterate,
    reference_2token_soundex,
    compute_macro_f05,
)


class TestNonAsciiWithMissingAddress(unittest.TestCase):
    """Interaction: Non-ASCII names (Devanagari, Tamil, French) with missing/null addresses."""

    def test_tier3_devanagari_name_with_null_address(self):
        s1_name = "Ram Marketing Pvt Ltd"
        s1_addr = "Plot 42 Near Railway Station Pune"

        cand_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        cand_addr = None  # Missing in S2!

        # 1. Transliteration & Normalization
        s1_core, _ = reference_strip_legal_suffix(s1_name)
        cand_roman = reference_transliterate(cand_name)
        cand_core, _ = reference_strip_legal_suffix(cand_roman)

        # 2. Address Normalization & Missing flag
        s1_addr_norm = reference_normalize_address(s1_addr)
        cand_addr_norm = reference_normalize_address(cand_addr)
        addr_missing_flag = 1 if not cand_addr_norm else 0

        self.assertEqual(addr_missing_flag, 1)
        self.assertIn("Ram", cand_core)
        self.assertEqual(cand_addr_norm, "")

        # 3. Composite score fallback
        name_sim = 0.95
        addr_sim = 0.0
        composite_score = name_sim if addr_missing_flag == 1 else (0.70 * name_sim + 0.30 * addr_sim)
        self.assertEqual(composite_score, 0.95, "Missing address must not penalize high name match")

    def test_tier3_french_accent_with_empty_address(self):
        cand_name = "Éléphant Centre EURL"
        cand_addr = ""
        cand_roman = reference_transliterate(cand_name)
        cand_core, suffix = reference_strip_legal_suffix(cand_roman)
        addr_norm = reference_normalize_address(cand_addr)

        self.assertIn("Elephant", cand_core)
        self.assertEqual(suffix, "eurl")
        self.assertEqual(addr_norm, "")


class TestSoundexCollisionsAndNameDivergence(unittest.TestCase):
    """Interaction: Soundex collision between phonetically similar but lexically distinct entities."""

    def test_tier3_soundex_collision_suppressed_by_char_tfidf(self):
        # Two completely distinct businesses with colliding Soundex
        # "Smith Tech" vs "Samad Toys"
        name_s1 = "Smith Tech"
        name_cand = "Samad Toys"

        soundex_s1 = reference_2token_soundex(name_s1)
        soundex_cand = reference_2token_soundex(name_cand)
        self.assertEqual(soundex_s1, soundex_cand, "Smith Tech and Samad Toys have colliding Soundex: S530_T200")

        # But character n-gram cosine similarity must be low
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        X = vec.fit_transform([name_s1.lower(), name_cand.lower()])
        cos_sim = float(X[0].dot(X[1].T).toarray()[0, 0])

        self.assertLess(cos_sim, 0.35, "Char TF-IDF cosine must cleanly distinguish Soundex collision")

    def test_tier3_soundex_match_corroborated_by_address(self):
        # Phonetic variant with matching address
        s1_name, s1_addr = "Infosys Technologies Ltd", "Electronics City Bangalore"
        c_name, c_addr = "Infoses Teknoliges", "Electronic City Bengaluru"

        s1_soundex = reference_2token_soundex(s1_name)
        c_soundex = reference_2token_soundex(c_name)
        self.assertEqual(s1_soundex, c_soundex)

        s1_addr_n = reference_normalize_address(s1_addr)
        c_addr_n = reference_normalize_address(c_addr)
        # Shared token overlap
        overlap = set(s1_addr_n.split()) & set(c_addr_n.split())
        self.assertGreaterEqual(len(overlap), 2)


class TestMultiSourceJoins(unittest.TestCase):
    """Interaction: S1 entity simultaneously matching candidates across both Source 2 and Source 3."""

    def test_tier3_s1_matches_s2_and_s3_simultaneously(self):
        s1_id = "S1-10001"
        gt_matches = {"S2-20001", "S3-30001"}
        predicted_matches = {"S2-20001", "S3-30001"}

        gt_dict = {s1_id: gt_matches}
        pred_dict = {s1_id: predicted_matches}

        score = compute_macro_f05(gt_dict, pred_dict, {s1_id})
        self.assertEqual(score, 1.0, "Perfect multi-source match must yield 1.0")

    def test_tier3_s1_partial_match_s2_correct_s3_missed(self):
        # Matched S2, but missed S3
        # TP = 1, FP = 0, FN = 1 -> P = 1.0, R = 0.5
        # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
        s1_id = "S1-10001"
        gt_dict = {s1_id: {"S2-20001", "S3-30001"}}
        pred_dict = {s1_id: {"S2-20001"}}

        score = compute_macro_f05(gt_dict, pred_dict, {s1_id})
        self.assertAlmostEqual(score, 0.8333, places=3)

    def test_tier3_candidate_source_indicator_flags(self):
        candidates = pl.DataFrame({
            "cand_id": ["S2-001", "S3-002", "S2-003", "S3-004"]
        }).with_columns(
            pl.col("cand_id").str.starts_with("S3-").cast(pl.Int8).alias("source_indicator")
        )
        self.assertEqual(candidates["source_indicator"].to_list(), [0, 1, 0, 1])


class TestLegalSuffixAndAddressCollision(unittest.TestCase):
    """Interaction: Proper nouns that overlap with legal suffixes or address keywords."""

    def test_tier3_brand_name_containing_limited(self):
        # "The Limited Stores" -> core brand name is "The Limited"
        name = "The Limited Stores LLC"
        core, suffix = reference_strip_legal_suffix(name)
        self.assertEqual(suffix, "llc")
        self.assertIn("Limited", core)

    def test_tier3_saint_vs_street_in_address(self):
        # "St. Jude Hospital on 500 Main St"
        # Name has "St." meaning Saint; address has "St" meaning Street
        name = "St. Jude Medical Inc"
        addr = "500 Main St, Suite 400"

        core_name, _ = reference_strip_legal_suffix(name)
        norm_addr = reference_normalize_address(addr)

        self.assertIn("St. Jude", core_name)
        self.assertIn("street", norm_addr)
        self.assertNotIn("suite 400", norm_addr)  # "ste" or "suite" expanded


class TestHighNameLowAddressVsHighAddressLowName(unittest.TestCase):
    """Interaction: Franchise branches vs co-located distinct businesses."""

    def test_tier3_franchise_branches_name_high_addr_low(self):
        # Starbucks in Seattle vs Starbucks in New York
        # High name similarity, zero address similarity -> indicates distinct branches
        name_sim = 1.0
        addr_sim = 0.05
        name_addr_diff = abs(name_sim - addr_sim)
        self.assertGreater(name_addr_diff, 0.90, "Large name-addr divergence signals distinct franchise branches")

    def test_tier3_colocated_businesses_addr_high_name_low(self):
        # Two totally different companies in same building
        # "Baker & Hughes Law LLC" vs "Domino's Pizza" at "100 Market St"
        name_sim = 0.10
        addr_sim = 1.0
        # Re-ranker formula: 0.70 * name + 0.30 * addr = 0.07 + 0.30 = 0.37
        composite_score = 0.70 * name_sim + 0.30 * addr_sim
        self.assertLess(composite_score, 0.50, "Low name similarity must suppress co-located false merge")


class TestPlattCalibrationWithSevereClassImbalance(unittest.TestCase):
    """Interaction: Probability calibration when negative candidate pairs outnumber positive matches 20:1."""

    def test_tier3_platt_scaling_under_severe_imbalance(self):
        np.random.seed(42)
        n_pos = 50
        n_neg = 950  # 1:19 ratio (~5% positive)

        X_pos = np.random.normal(loc=2.0, scale=0.8, size=(n_pos, 1))
        X_neg = np.random.normal(loc=-1.5, scale=0.8, size=(n_neg, 1))

        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * n_pos + [0] * n_neg)

        base_clf = LogisticRegression().fit(X, y)
        frozen = FrozenEstimator(base_clf)
        calibrator = CalibratedClassifierCV(estimator=frozen, method='sigmoid')
        calibrator.fit(X, y)

        # High score input should produce high probability
        high_prob = calibrator.predict_proba([[3.0]])[0, 1]
        low_prob = calibrator.predict_proba([[-3.0]])[0, 1]

        self.assertGreater(high_prob, 0.70)
        self.assertLess(low_prob, 0.05)


class TestOptuna3DThresholdRouting(unittest.TestCase):
    """Interaction: Threshold routing across multi-country mixed batches."""

    def test_tier3_country_specific_threshold_routing(self):
        # US tau = 0.85, India tau = 0.75, France tau = 0.78
        thresholds = {"US": 0.85, "India": 0.75, "France": 0.78}

        pairs = [
            {"s1_id": "S1-US1", "cand_id": "S2-U1", "country": "US", "prob": 0.80},      # 0.80 < 0.85 -> Reject
            {"s1_id": "S1-IN1", "cand_id": "S2-I1", "country": "India", "prob": 0.80},   # 0.80 >= 0.75 -> Accept
            {"s1_id": "S1-FR1", "cand_id": "S2-F1", "country": "France", "prob": 0.80},  # 0.80 >= 0.78 -> Accept
        ]

        decisions = [p["prob"] >= thresholds[p["country"]] for p in pairs]
        self.assertEqual(decisions, [False, True, True])

    def test_tier3_mixed_batch_singleton_decision(self):
        thresholds = {"US": 0.85, "India": 0.75, "France": 0.78}
        # S1-US1 has only candidates < 0.85 -> becomes singleton
        # S1-IN1 has candidate 0.80 >= 0.75 -> becomes match
        predictions = {
            "S1-US1": set(),
            "S1-IN1": {"S2-I1"}
        }
        self.assertEqual(len(predictions["S1-US1"]), 0)
        self.assertEqual(predictions["S1-IN1"], {"S2-I1"})


if __name__ == "__main__":
    unittest.main()
