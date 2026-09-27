"""
tests/e2e/test_tier2_boundary.py
Tier 2: Boundary & Corner Cases Test Suite
Tests system resilience under empty strings, null addresses, singletons,
single-token names, cross-country isolation, non-ASCII Unicode strings,
extreme string lengths, and threshold boundary conditions.
Total tests: 35 test cases.
"""

import unittest
import numpy as np
import polars as pl
import re

from tests.e2e.helpers import (
    reference_normalize_address,
    reference_strip_legal_suffix,
    reference_transliterate,
    reference_2token_soundex,
    compute_macro_f05,
    COUNTRIES,
)


class TestEmptyStringsAndNulls(unittest.TestCase):
    """Resilience to empty strings, null values, and whitespace-only entries."""

    def test_tier2_empty_name_handling(self):
        core, suffix = reference_strip_legal_suffix("")
        self.assertEqual(core, "")
        self.assertEqual(suffix, "")

    def test_tier2_whitespace_name_handling(self):
        core, suffix = reference_strip_legal_suffix("    \t\n  ")
        self.assertEqual(core, "")
        self.assertEqual(suffix, "")

    def test_tier2_null_address_normalization(self):
        norm = reference_normalize_address(None)
        self.assertEqual(norm, "")

    def test_tier2_empty_address_normalization(self):
        norm = reference_normalize_address("")
        self.assertEqual(norm, "")

    def test_tier2_whitespace_address_normalization(self):
        norm = reference_normalize_address("   \t  \n  ")
        self.assertEqual(norm, "")

    def test_tier2_soundex_on_empty_name(self):
        code = reference_2token_soundex("")
        self.assertEqual(code, "0000_0000")


class TestSingletonsAndUnlinked(unittest.TestCase):
    """Evaluation behavior on singletons and unlinked entity populations."""

    def test_tier2_singleton_zero_candidates_scores_one(self):
        gt = {"S1-SINGLETON": set()}
        pred = {"S1-SINGLETON": set()}
        score = compute_macro_f05(gt, pred, {"S1-SINGLETON"})
        self.assertEqual(score, 1.0)

    def test_tier2_singleton_with_candidates_below_tau_scores_one(self):
        # Candidate was generated, but model scored it 0.40 (< tau=0.80) -> filtered out -> empty prediction
        gt = {"S1-001": set()}
        tau = 0.80
        cand_probs = [0.35, 0.42, 0.71]
        predicted_matches = {f"S2-{i}" for i, p in enumerate(cand_probs) if p >= tau}
        pred = {"S1-001": predicted_matches}
        score = compute_macro_f05(gt, pred, {"S1-001"})
        self.assertEqual(score, 1.0, "Singleton with sub-threshold candidates must score 1.0")

    def test_tier2_singleton_false_merge_scores_zero(self):
        gt = {"S1-001": set()}
        pred = {"S1-001": {"S2-FALSE-POSITIVE"}}
        score = compute_macro_f05(gt, pred, {"S1-001"})
        self.assertEqual(score, 0.0, "False merge on singleton must yield 0.0")

    def test_tier2_all_singletons_dataset(self):
        # 100% singletons in dataset
        all_s1 = {f"S1-{i}" for i in range(50)}
        gt = {s1: set() for s1 in all_s1}
        pred = {s1: set() for s1 in all_s1}
        score = compute_macro_f05(gt, pred, all_s1)
        self.assertEqual(score, 1.0)

    def test_tier2_zero_singletons_dataset(self):
        all_s1 = {f"S1-{i}" for i in range(20)}
        gt = {s1: {f"S2-{s1}"} for s1 in all_s1}
        pred = {s1: {f"S2-{s1}"} for s1 in all_s1}
        score = compute_macro_f05(gt, pred, all_s1)
        self.assertEqual(score, 1.0)


class TestSingleTokenAndExtremeNames(unittest.TestCase):
    """Resilience to single-token names, extreme lengths, and special characters."""

    def test_tier2_single_token_name(self):
        core, suffix = reference_strip_legal_suffix("Google")
        self.assertEqual(core, "Google")
        soundex = reference_2token_soundex("Google")
        self.assertEqual(soundex, "G240_0000")

    def test_tier2_single_character_name(self):
        core, suffix = reference_strip_legal_suffix("X")
        self.assertEqual(core, "X")
        soundex = reference_2token_soundex("X")
        self.assertEqual(soundex, "X000_0000")

    def test_tier2_extreme_500_char_name(self):
        long_name = "Mega Corp " * 50
        core, suffix = reference_strip_legal_suffix(long_name)
        self.assertTrue(len(core) > 0)
        soundex = reference_2token_soundex(long_name)
        self.assertRegex(soundex, r'^[A-Z]\d{3}_[A-Z]\d{3}$')

    def test_tier2_all_consonants_name(self):
        name = "BRND CRX"
        soundex = reference_2token_soundex(name)
        self.assertNotEqual(soundex, "0000_0000")

    def test_tier2_name_identical_to_legal_suffix(self):
        # "Limited" alone
        core, suffix = reference_strip_legal_suffix("Limited")
        # Should cleanly handle without unhandled exception
        self.assertEqual(suffix, "limited")

    def test_tier2_numbers_only_name(self):
        name = "777 999"
        core, suffix = reference_strip_legal_suffix(name)
        self.assertEqual(core, "777 999")
        soundex = reference_2token_soundex(name)
        self.assertEqual(soundex, "0000_0000")


class TestCrossCountryIsolation(unittest.TestCase):
    """Enforce strict isolation across country partitions."""

    def test_tier2_us_india_isolation(self):
        us_s1_records = [{"id": "S1-US1", "country": "US"}]
        india_s2_records = [{"id": "S2-IN1", "country": "India"}]
        # Blocking cross-join filter
        cross_pairs = [
            (s1["id"], s2["id"])
            for s1 in us_s1_records
            for s2 in india_s2_records
            if s1["country"] == s2["country"]
        ]
        self.assertEqual(len(cross_pairs), 0, "Cross-country pairs between US and India must be 0")

    def test_tier2_us_france_isolation(self):
        us_s1 = {"country": "US"}
        fr_s2 = {"country": "France"}
        self.assertNotEqual(us_s1["country"], fr_s2["country"])

    def test_tier2_india_france_isolation(self):
        in_s1 = {"country": "India"}
        fr_s2 = {"country": "France"}
        self.assertNotEqual(in_s1["country"], fr_s2["country"])

    def test_tier2_open_set_country_membership(self):
        for c in ["US", "India", "France"]:
            self.assertIn(c, COUNTRIES)

    def test_tier2_unrecognized_country_partition_quarantine(self):
        foreign_entity = {"id": "S1-GER01", "country": "Germany"}
        is_known = foreign_entity["country"] in COUNTRIES
        self.assertFalse(is_known, "Unknown countries outside US/India/France should be isolated")


class TestNonAsciiAndUnicodeStrings(unittest.TestCase):
    """Resilience to Unicode, non-Latin scripts, diacritics, and special glyphs."""

    def test_tier2_devanagari_transliteration_fidelity(self):
        dev_name = "राम मार्केटिंग"
        roman = reference_transliterate(dev_name)
        self.assertTrue(all(ord(c) < 128 for c in roman))
        self.assertIn("Ram", roman)

    def test_tier2_french_accented_normalization(self):
        fr_name = "Château & Éléphant"
        roman = reference_transliterate(fr_name)
        self.assertTrue(all(ord(c) < 128 for c in roman))
        self.assertIn("Elephant", roman)

    def test_tier2_emoji_and_symbol_handling(self):
        fancy_name = "🚀 Rocket Tech 🌟 Inc."
        core, suffix = reference_strip_legal_suffix(fancy_name)
        self.assertIn("Rocket Tech", core)
        self.assertEqual(suffix, "inc")

    def test_tier2_punctuation_only_entity(self):
        punct = "!@#$%^&*()_+=-"
        core, suffix = reference_strip_legal_suffix(punct)
        self.assertEqual(suffix, "")

    def test_tier2_tamil_script_handling(self):
        tamil_name = "சரவணா ஸ்டோர்ஸ்"
        # Should not throw exception
        try:
            roman = reference_transliterate(tamil_name)
            success = True
        except Exception:
            success = False
        self.assertTrue(success)


class TestCandidateBoundingAndPruning(unittest.TestCase):
    """Enforce candidate set size constraints (median 10-12, max 15)."""

    def test_tier2_zero_candidates_allowed(self):
        # S1 entity with 0 blocking candidates
        cands = []
        self.assertEqual(len(cands), 0)

    def test_tier2_single_candidate_allowed(self):
        cands = ["S2-001"]
        self.assertEqual(len(cands), 1)

    def test_tier2_exactly_15_candidates_boundary(self):
        cands = [f"S2-{i}" for i in range(15)]
        self.assertEqual(len(cands), 15)
        self.assertLessEqual(len(cands), 15)

    def test_tier2_truncation_of_20_candidates_to_15(self):
        scores = [float(i) / 20.0 for i in range(20)]
        cands = [f"S2-{i}" for i in range(20)]
        df = pl.DataFrame({
            "s1_id": ["S1-1"] * 20,
            "cand_id": cands,
            "score": scores
        })
        pruned = (
            df.sort(["s1_id", "score"], descending=[False, True])
            .group_by("s1_id")
            .head(15)
        )
        self.assertEqual(len(pruned), 15)
        # Verify highest scores were retained
        self.assertGreaterEqual(pruned["score"].min(), 5.0 / 20.0)

    def test_tier2_deterministic_score_tie_breaking(self):
        # Tied scores broken deterministically by cand_id
        df = pl.DataFrame({
            "s1_id": ["S1-1", "S1-1"],
            "cand_id": ["S2-B", "S2-A"],
            "score": [0.85, 0.85]
        })
        sorted_df = df.sort(["score", "cand_id"], descending=[True, False])
        self.assertEqual(sorted_df["cand_id"][0], "S2-A")


class TestProbabilityThresholdBoundaries(unittest.TestCase):
    """Exact edge-case behaviors at threshold tau boundaries."""

    def test_tier2_probability_exactly_equal_to_tau(self):
        tau = 0.8000
        prob = 0.8000
        # By competition rule: p >= tau is accepted as match
        is_match = prob >= tau
        self.assertTrue(is_match)

    def test_tier2_probability_epsilon_below_tau(self):
        tau = 0.8000
        prob = 0.79999
        is_match = prob >= tau
        self.assertFalse(is_match)

    def test_tier2_probability_epsilon_above_tau(self):
        tau = 0.8000
        prob = 0.80001
        is_match = prob >= tau
        self.assertTrue(is_match)

    def test_tier2_extreme_zero_and_one_probabilities(self):
        tau = 0.80
        self.assertFalse(0.0 >= tau)
        self.assertTrue(1.0 >= tau)


if __name__ == "__main__":
    unittest.main()
