"""
tests/test_submission.py
Unit tests for Milestone 5: Submission Generation, Dynamic Validation & Integrity Assertions (src/submission.py).

Tests:
1. Header exactness (source1_entity_id\tmatched_entity_ids and source1_entity_id\tcandidate_entity_ids)
2. Strict tab separation and quote-free TSV formatting (quote_style="never")
3. Singleton formatting (S1-xxxxx\t\n with empty target string)
4. Deduplication and sorted deterministic target ID list formatting
5. Self-match rejection (forbidding S1- IDs in candidate or match sets)
6. Hard runtime dynamic assertions:
   - assert len(matching_results) == len(test_source1)
   - assert len(candidate_pairs) == len(test_source1)
   - assert set(matching_results['source1_entity_id']) == set(test_source1['entity_id'])
   - assert every matched ID is a subset of candidate IDs for that S1 entity
7. Official validation wrapper against validate_submission.py
8. Native Polars DataFrame export support without iter_rows()
"""

import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Enforce explicit UTF-8 stdout initialization (Global Requirement F02)
try:
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import pytest

from src.submission import (
    CANDIDATE_HEADER,
    MATCHING_HEADER,
    clean_target_ids,
    export_candidate_pairs_tsv,
    export_matching_results_tsv,
    format_id_list_str,
    parse_submission_tsv,
    read_s1_entity_ids,
    validate_submission_files,
    verify_submission_integrity,
)


class TestSubmissionFormatting(unittest.TestCase):
    """Test TSV output formatting rules, headers, delimiters, and singletons."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_submission_")
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_target_ids_deduplication_and_sorting(self):
        raw_ids = ["S3-0005", "S2-0002", "S2-0001", "S3-0005"]
        cleaned = clean_target_ids(raw_ids)
        self.assertEqual(cleaned, ["S2-0001", "S2-0002", "S3-0005"])
        # String formatting
        formatted = format_id_list_str(raw_ids)
        self.assertEqual(formatted, "S2-0001,S2-0002,S3-0005")

    def test_clean_target_ids_rejects_s1_self_match(self):
        raw_ids = ["S2-0001", "S1-0002"]
        with self.assertRaises(ValueError) as ctx:
            clean_target_ids(raw_ids, s1_id="S1-0001")
        self.assertIn("Self-match detected", str(ctx.exception))

    def test_clean_target_ids_rejects_invalid_prefix(self):
        raw_ids = ["X4-0001"]
        with self.assertRaises(ValueError) as ctx:
            clean_target_ids(raw_ids)
        self.assertIn("Invalid target ID prefix", str(ctx.exception))

    def test_matching_header_and_singleton_formatting(self):
        s1_ids = ["S1-001", "S1-002", "S1-003"]
        matches = {
            "S1-001": ["S2-101", "S3-201"],
            "S1-002": [],  # Singleton
            "S1-003": ["S2-103"],
        }
        out_file = self.output_dir / "matching_results.tsv"
        export_matching_results_tsv(matches, s1_ids, out_file)

        self.assertTrue(out_file.exists())
        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 4)  # Header + 3 entities
        # Header check
        self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids\n")
        # Line checks
        self.assertEqual(lines[1], "S1-001\tS2-101,S3-201\n")
        # Singleton must have empty string after tab: S1-002\t\n
        self.assertEqual(lines[2], "S1-002\t\n")
        self.assertEqual(lines[3], "S1-003\tS2-103\n")

        # Zero quote characters in file
        full_content = "".join(lines)
        self.assertNotIn('"', full_content)

    def test_candidate_header_and_formatting(self):
        s1_ids = ["S1-001", "S1-002"]
        candidates = {
            "S1-001": ["S2-101", "S3-201", "S2-102"],
            "S1-002": [],
        }
        out_file = self.output_dir / "candidate_pairs.tsv"
        export_candidate_pairs_tsv(candidates, s1_ids, out_file)

        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids\n")
        self.assertEqual(lines[1], "S1-001\tS2-101,S2-102,S3-201\n")
        self.assertEqual(lines[2], "S1-002\t\n")

    def test_export_from_polars_dataframe(self):
        s1_ids = ["S1-1", "S1-2", "S1-3"]
        df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-3"],
            "matched_entity_id": ["S2-11", "S3-22", "S2-33"],
        })
        out_file = self.output_dir / "matching_df.tsv"
        export_matching_results_tsv(df, s1_ids, out_file)

        mapping, ordered = parse_submission_tsv(out_file, MATCHING_HEADER)
        self.assertEqual(ordered, ["S1-1", "S1-2", "S1-3"])
        self.assertEqual(mapping["S1-1"], {"S2-11", "S3-22"})
        self.assertEqual(mapping["S1-2"], set())  # S1-2 was singleton in df
        self.assertEqual(mapping["S1-3"], {"S2-33"})


class TestDynamicAssertions(unittest.TestCase):
    """Test hard runtime dynamic assertions in verify_submission_integrity."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_assertions_")
        self.test_s1 = ["S1-01", "S1-02", "S1-03", "S1-04"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integrity_success(self):
        candidates = {
            "S1-01": {"S2-1", "S3-1"},
            "S1-02": {"S2-2"},
            "S1-03": {"S3-3"},
            "S1-04": set(),
        }
        matches = {
            "S1-01": {"S2-1"},
            "S1-02": {"S2-2"},
            "S1-03": set(),
            "S1-04": set(),
        }
        metrics = verify_submission_integrity(matches, candidates, self.test_s1)
        self.assertEqual(metrics["status"], "PASS")
        self.assertEqual(metrics["total_s1_entities"], 4)
        self.assertEqual(metrics["singletons_count"], 2)

    def test_assertion_row_count_mismatch(self):
        # Matching has only 3 rows instead of 4
        matches = {"S1-01": {"S2-1"}, "S1-02": {"S2-2"}, "S1-03": set()}
        candidates = {s: {"S2-1"} for s in self.test_s1}

        with self.assertRaises(AssertionError) as ctx:
            verify_submission_integrity(matches, candidates, self.test_s1)
        self.assertIn("Dynamic assertion failed: matching_results row count", str(ctx.exception))

    def test_assertion_candidate_row_count_mismatch(self):
        matches = {s: set() for s in self.test_s1}
        candidates = {"S1-01": {"S2-1"}}  # only 1 row

        with self.assertRaises(AssertionError) as ctx:
            verify_submission_integrity(matches, candidates, self.test_s1)
        self.assertIn("Dynamic assertion failed: candidate_pairs row count", str(ctx.exception))

    def test_assertion_missing_s1_entity(self):
        # 4 rows, but S1-99 instead of S1-04
        wrong_s1 = ["S1-01", "S1-02", "S1-03", "S1-99"]
        matches = {s: set() for s in wrong_s1}
        candidates = {s: set() for s in wrong_s1}

        with self.assertRaises(AssertionError) as ctx:
            verify_submission_integrity(matches, candidates, self.test_s1)
        self.assertIn("required S1 entities missing from matching_results", str(ctx.exception))

    def test_assertion_matched_not_subset_of_candidates(self):
        candidates = {
            "S1-01": {"S2-1"},
            "S1-02": {"S2-2"},
            "S1-03": set(),
            "S1-04": set(),
        }
        matches = {
            "S1-01": {"S2-1", "S3-ROGUE"},  # S3-ROGUE not in candidates!
            "S1-02": {"S2-2"},
            "S1-03": set(),
            "S1-04": set(),
        }

        with self.assertRaises(AssertionError) as ctx:
            verify_submission_integrity(matches, candidates, self.test_s1)
        self.assertIn("not present in candidate_pairs.tsv", str(ctx.exception))


class TestValidationWrapper(unittest.TestCase):
    """Test validation wrapper against student_resource validate_submission.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_validator_")
        self.test_dir = Path(self.temp_dir) / "test"
        self.output_dir = Path(self.temp_dir) / "output"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test_source1.tsv
        self.test_s1_path = self.test_dir / "test_source1.tsv"
        with open(self.test_s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAlpha Corp\t123 Main St\tUS\n")
            f.write("S1-002\tBeta LLC\t456 Market St\tIndia\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validation_wrapper_valid_files(self):
        m_file = self.output_dir / "matching_results.tsv"
        c_file = self.output_dir / "candidate_pairs.tsv"

        with open(m_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-101\n")
            f.write("S1-002\t\n")

        with open(c_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-001\tS2-101,S3-201\n")
            f.write("S1-002\t\n")

        errors, warnings = validate_submission_files(
            matching_path=m_file,
            candidate_path=c_file,
            test_dir=self.test_dir,
            raise_on_error=True,
        )
        self.assertEqual(len(errors), 0)

    def test_validation_wrapper_detects_comma_csv(self):
        bad_file = self.output_dir / "matching_bad.tsv"
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id,matched_entity_ids\nS1-001,S2-101\nS1-002,\n")

        with self.assertRaises(AssertionError) as ctx:
            validate_submission_files(
                matching_path=bad_file,
                test_dir=self.test_dir,
                raise_on_error=True,
            )
        self.assertIn("ERROR(S)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
