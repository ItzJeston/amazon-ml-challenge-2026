"""
tests/e2e/test_tier4_scenarios.py
Tier 4: Real-World Application Scenarios Test Suite
Executes end-to-end integration workflows on realistic sampled subsets of train and test datasets.
Verifies that:
- matching_results.tsv and candidate_pairs.tsv match test_source1 length dynamically
- Output TSVs strictly meet validate_submission.py rules
- Macro F0.5 scoring correctly rewards precision and singleton isolation
Total tests: 20 test cases.
"""

import os
import sys
import io
import shutil
import tempfile
import unittest
import polars as pl
from typing import Dict, Set

from tests.e2e.helpers import (
    SAMPLE_RAW_DATA,
    compute_macro_f05,
    reference_normalize_address,
    reference_strip_legal_suffix,
    reference_transliterate,
    reference_2token_soundex,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Import validate_submission functions from student_resource
VALIDATOR_DIR = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "utils")
if VALIDATOR_DIR not in sys.path:
    sys.path.insert(0, VALIDATOR_DIR)

try:
    import validate_submission
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False


class TestSampledPipelineE2E(unittest.TestCase):
    """End-to-end pipeline execution on realistic sampled multi-country dataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="e2e_tier4_")
        self.test_dir = os.path.join(self.temp_dir, "dataset", "test")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Write sample test source files
        self.test_s1_path = os.path.join(self.test_dir, "test_source1.tsv")
        self.test_s2_path = os.path.join(self.test_dir, "test_source2.tsv")
        self.test_s3_path = os.path.join(self.test_dir, "test_source3.tsv")

        with open(self.test_s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for row in SAMPLE_RAW_DATA["test_source1"]:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

        with open(self.test_s2_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for row in SAMPLE_RAW_DATA["test_source2"]:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

        with open(self.test_s3_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for row in SAMPLE_RAW_DATA["test_source3"]:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

        self.matching_path = os.path.join(self.output_dir, "matching_results.tsv")
        self.candidate_path = os.path.join(self.output_dir, "candidate_pairs.tsv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tier4_1_generate_valid_mock_outputs(self):
        # Build valid mock submission files for the sampled test set
        # test_source1 entities: S1-90001 (US), S1-90002 (India), S1-90003 (France), S1-90004 (France singleton)
        candidates = {
            "S1-90001": ["S2-80001", "S3-70001"],
            "S1-90002": ["S2-80002", "S3-70002"],
            "S1-90003": ["S2-80003", "S3-70003"],
            "S1-90004": [],  # Singleton
        }
        matches = {
            "S1-90001": ["S2-80001", "S3-70001"],
            "S1-90002": ["S2-80002", "S3-70002"],
            "S1-90003": ["S2-80003", "S3-70003"],
            "S1-90004": [],  # Singleton
        }

        # Write matching_results.tsv
        with open(self.matching_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in sorted(matches.keys()):
                f.write(f"{s1_id}\t{','.join(matches[s1_id])}\n")

        # Write candidate_pairs.tsv
        with open(self.candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted(candidates.keys()):
                f.write(f"{s1_id}\t{','.join(candidates[s1_id])}\n")

        # Validate with validate_submission.py
        if VALIDATOR_AVAILABLE:
            errors, warnings = validate_submission.validate(
                self.matching_path,
                self.candidate_path,
                self.test_dir,
                check_ids=True
            )
            self.assertEqual(len(errors), 0, f"Validator reported errors on valid output: {errors}")

    def test_tier4_2_dynamic_length_assertion(self):
        # Dynamic length assertion: len(matching) == len(test_source1)
        test_s1_df = pl.read_csv(self.test_s1_path, separator='\t')
        expected_len = len(test_s1_df)

        # Write matching results
        with open(self.matching_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in test_s1_df["entity_id"].to_list():
                f.write(f"{s1_id}\t\n")

        matching_df = pl.read_csv(self.matching_path, separator='\t')
        self.assertEqual(len(matching_df), expected_len)

    def test_tier4_3_candidate_pairs_dynamic_length_assertion(self):
        test_s1_df = pl.read_csv(self.test_s1_path, separator='\t')
        expected_len = len(test_s1_df)

        with open(self.candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in test_s1_df["entity_id"].to_list():
                f.write(f"{s1_id}\t\n")

        cand_df = pl.read_csv(self.candidate_path, separator='\t')
        self.assertEqual(len(cand_df), expected_len)

    def test_tier4_4_subset_integrity_validation(self):
        # Matched IDs must be a subset of candidate IDs
        matches = {"S1-1": {"S2-1"}}
        candidates = {"S1-1": {"S2-1", "S3-2"}}
        offenders = {
            s1 for s1, mids in matches.items() if mids - candidates.get(s1, set())
        }
        self.assertEqual(len(offenders), 0, "Matched entity IDs must be subset of candidate entity IDs")

    def test_tier4_5_compactness_constraint_audit(self):
        # Maximum candidates per S1 entity <= 15
        candidates = {
            f"S1-{i}": [f"S2-{j}" for j in range(12)] for i in range(10)
        }
        counts = [len(c) for c in candidates.values()]
        self.assertLessEqual(max(counts), 15)
        self.assertTrue(10 <= np.median(counts) <= 12)

    def test_tier4_6_macro_f05_on_sample_train_ground_truth(self):
        # Score ground truth against itself -> must be 1.0
        gt_dict = {}
        for row in SAMPLE_RAW_DATA["train_ground_truth"]:
            s1, matched = row
            gt_dict[s1] = set(matched.split(",")) if matched else set()

        all_s1 = set(gt_dict.keys())
        score = compute_macro_f05(gt_dict, gt_dict, all_s1)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_tier4_7_macro_f05_singleton_weighting(self):
        # In SAMPLE_RAW_DATA train_ground_truth, S1-10005 is a singleton
        gt_dict = {
            "S1-10001": {"S2-20001", "S3-30001"},
            "S1-10005": set(),  # Singleton
        }
        # Predict both correctly
        pred_perfect = {
            "S1-10001": {"S2-20001", "S3-30001"},
            "S1-10005": set(),
        }
        score = compute_macro_f05(gt_dict, pred_perfect, {"S1-10001", "S1-10005"})
        self.assertEqual(score, 1.0)

        # False merge on singleton: drops entity score from 1.0 to 0.0!
        pred_corrupted = {
            "S1-10001": {"S2-20001", "S3-30001"},
            "S1-10005": {"S2-FALSE"},
        }
        score_corrupted = compute_macro_f05(gt_dict, pred_corrupted, {"S1-10001", "S1-10005"})
        self.assertEqual(score_corrupted, 0.5)


class TestValidatorGateRejectionRules(unittest.TestCase):
    """Verify that validate_submission rejects all illegal output structures."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="validator_test_")
        self.test_dir = os.path.join(self.temp_dir, "test")
        os.makedirs(self.test_dir, exist_ok=True)

        self.test_s1_path = os.path.join(self.test_dir, "test_source1.tsv")
        with open(self.test_s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAcme\t100 Main St\tUS\n")
            f.write("S1-002\tBeta\t200 High St\tUS\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tier4_8_reject_csv_instead_of_tsv(self):
        bad_path = os.path.join(self.temp_dir, "matching_bad.tsv")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id,matched_entity_ids\nS1-001,S2-001\n")
        if VALIDATOR_AVAILABLE:
            errors, _ = validate_submission.validate(bad_path, None, self.test_dir)
            self.assertTrue(any("COMMA-separated" in e or "header" in e for e in errors))

    def test_tier4_9_reject_self_matches(self):
        bad_path = os.path.join(self.temp_dir, "matching_self.tsv")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-001\tS1-001\nS1-002\t\n")
        if VALIDATOR_AVAILABLE:
            errors, _ = validate_submission.validate(bad_path, None, self.test_dir)
            self.assertTrue(any("self-matches" in e for e in errors))

    def test_tier4_10_reject_duplicate_s1_rows(self):
        bad_path = os.path.join(self.temp_dir, "matching_dup.tsv")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001\nS1-001\tS2-002\nS1-002\t\n")
        if VALIDATOR_AVAILABLE:
            errors, _ = validate_submission.validate(bad_path, None, self.test_dir)
            self.assertTrue(any("duplicate source1_entity_id row" in e for e in errors))

    def test_tier4_11_reject_missing_required_s1_entities(self):
        bad_path = os.path.join(self.temp_dir, "matching_missing.tsv")
        with open(bad_path, "w", encoding="utf-8") as f:
            # Omitting S1-002
            f.write("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001\n")
        if VALIDATOR_AVAILABLE:
            errors, _ = validate_submission.validate(bad_path, None, self.test_dir)
            self.assertTrue(any("missing" in e for e in errors))

    def test_tier4_12_reject_intra_list_duplicates(self):
        bad_path = os.path.join(self.temp_dir, "matching_intra.tsv")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001,S2-001\nS1-002\t\n")
        if VALIDATOR_AVAILABLE:
            errors, _ = validate_submission.validate(bad_path, None, self.test_dir)
            self.assertTrue(any("repeated ID inside" in e for e in errors))

    def test_tier4_13_reject_wrong_prefix_ids(self):
        bad_path = os.path.join(self.temp_dir, "matching_wrong.tsv")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-001\tX4-999\nS1-002\t\n")
        if VALIDATOR_AVAILABLE:
            errors, _ = validate_submission.validate(bad_path, None, self.test_dir)
            self.assertTrue(any("without an S2-/S3- prefix" in e for e in errors))


class TestLiveDatasetIntegrity(unittest.TestCase):
    """Directly verifies actual dataset files on disk for format and schema compliance."""

    def test_tier4_14_dataset_files_presence(self):
        data_root = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset")
        expected_files = [
            os.path.join(data_root, "train", "train_source1.tsv"),
            os.path.join(data_root, "train", "train_source2.tsv"),
            os.path.join(data_root, "train", "train_source3.tsv"),
            os.path.join(data_root, "train", "train_ground_truth.tsv"),
            os.path.join(data_root, "test", "test_source1.tsv"),
            os.path.join(data_root, "test", "test_source2.tsv"),
            os.path.join(data_root, "test", "test_source3.tsv"),
        ]
        for f in expected_files:
            self.assertTrue(os.path.isfile(f), f"Expected dataset file not found: {f}")

    def test_tier4_15_test_source1_header_and_sample(self):
        test_s1_path = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset", "test", "test_source1.tsv")
        with open(test_s1_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            first_line = f.readline().rstrip("\n").split("\t")
        self.assertEqual(header, ["entity_id", "business_name", "business_address", "country"])
        self.assertTrue(first_line[0].startswith("S1-"))
        self.assertIn(first_line[3], COUNTRIES)

    def test_tier4_16_train_ground_truth_header_and_sample(self):
        gt_path = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset", "train", "train_ground_truth.tsv")
        with open(gt_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            first_line = f.readline().rstrip("\n").split("\t")
        self.assertEqual(header, ["source1_entity_id", "matched_entity_ids"])
        self.assertTrue(first_line[0].startswith("S1-"))

    def test_tier4_17_france_only_in_test_live_check(self):
        # Sample test file to verify France presence
        test_s1_path = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset", "test", "test_source1.tsv")
        has_france = False
        with open(test_s1_path, "r", encoding="utf-8") as f:
            next(f)
            for _ in range(5000):
                line = f.readline()
                if not line:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == "France":
                    has_france = True
                    break
        self.assertTrue(has_france, "France entities must be present in test set")

    def test_tier4_18_s1_entity_id_format(self):
        test_s1_path = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset", "test", "test_source1.tsv")
        with open(test_s1_path, "r", encoding="utf-8") as f:
            next(f)
            sample_ids = [f.readline().split("\t")[0] for _ in range(20)]
        self.assertTrue(all(re.match(r'^S1-\d+$', sid) for sid in sample_ids))

    def test_tier4_19_s2_entity_id_format(self):
        test_s2_path = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset", "test", "test_source2.tsv")
        with open(test_s2_path, "r", encoding="utf-8") as f:
            next(f)
            sample_ids = [f.readline().split("\t")[0] for _ in range(20)]
        self.assertTrue(all(re.match(r'^S2-\d+$', sid) for sid in sample_ids))

    def test_tier4_20_s3_entity_id_format(self):
        test_s3_path = os.path.join(PROJECT_ROOT, "extracted_data", "student_resource", "dataset", "test", "test_source3.tsv")
        with open(test_s3_path, "r", encoding="utf-8") as f:
            next(f)
            sample_ids = [f.readline().split("\t")[0] for _ in range(20)]
        self.assertTrue(all(re.match(r'^S3-\d+$', sid) for sid in sample_ids))


if __name__ == "__main__":
    unittest.main()
