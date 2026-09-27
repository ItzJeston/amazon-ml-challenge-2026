"""
tests/test_reranking.py
Unit tests for Milestone 2: Lightweight Cosine Re-ranker & Pruner (src/reranking.py).

Tests:
1. Vectorized composite cosine score calculation (0.70 * name + 0.30 * addr)
2. Lightweight candidate pruning via native Polars sorting (.group_by().head(15))
3. Candidate set bounding (median 10-12, maximum 15)
4. Export to candidate_pairs.parquet conforming strictly to Interface Contract 2
5. Submission TSV exporter with singleton preservation
6. Source code verification: zero iter_rows() in src/reranking.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import pytest

from src.blocking import (
    BLOCKING_FLAG_CHAR_TFIDF,
    BLOCKING_FLAG_SOUNDEX,
    BLOCKING_FLAG_TOKEN_INDEX,
    BLOCKING_FLAGS_COL,
    CANDIDATE_ID_COL,
    RERANKER_SCORE_COL,
    SOURCE1_ID_COL,
)
from src.reranking import (
    INTERFACE_CONTRACT_2_COLS,
    compute_reranker_composite_score,
    export_candidate_pairs_parquet,
    export_candidate_pairs_tsv,
    prune_candidates,
    run_reranking,
)


# ==============================================================================
# 1. Composite Cosine Score Formula Tests
# ==============================================================================
class TestCompositeCosineScore:
    def test_composite_score_exact_formula(self):
        # Score = 0.70 * Cosine_name + 0.30 * Cosine_addr
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-01"],
            CANDIDATE_ID_COL: ["S2-01"],
            "name_tfidf_cosine": [0.90],
            "addr_tfidf_cosine": [0.80],
            BLOCKING_FLAGS_COL: [BLOCKING_FLAG_CHAR_TFIDF],
        }, schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "name_tfidf_cosine": pl.Float32,
            "addr_tfidf_cosine": pl.Float32,
            BLOCKING_FLAGS_COL: pl.UInt8,
        })

        scored = compute_reranker_composite_score(df, name_weight=0.70, addr_weight=0.30)
        expected_score = 0.70 * 0.90 + 0.30 * 0.80  # 0.87
        assert np.isclose(scored[RERANKER_SCORE_COL][0], expected_score, atol=1e-5)

    def test_composite_score_null_handling(self):
        # Missing address cosine fills with 0.0
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-01"],
            CANDIDATE_ID_COL: ["S2-01"],
            "name_tfidf_cosine": [0.80],
            "addr_tfidf_cosine": [None],
            BLOCKING_FLAGS_COL: [BLOCKING_FLAG_CHAR_TFIDF],
        }, schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "name_tfidf_cosine": pl.Float32,
            "addr_tfidf_cosine": pl.Float32,
            BLOCKING_FLAGS_COL: pl.UInt8,
        })

        scored = compute_reranker_composite_score(df, name_weight=0.70, addr_weight=0.30)
        expected_score = 0.70 * 0.80 + 0.30 * 0.0  # 0.56
        assert np.isclose(scored[RERANKER_SCORE_COL][0], expected_score, atol=1e-5)

    def test_composite_score_strategy_fallbacks(self):
        # When name_tfidf_cosine is 0.0, Strategy B receives token baseline (0.35)
        # and Strategy D receives soundex baseline (0.30)
        df = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-01", "S1-02"],
            CANDIDATE_ID_COL: ["S2-01", "S3-02"],
            "name_tfidf_cosine": [0.0, 0.0],
            "addr_tfidf_cosine": [0.0, 0.0],
            BLOCKING_FLAGS_COL: [BLOCKING_FLAG_TOKEN_INDEX, BLOCKING_FLAG_SOUNDEX],
        }, schema={
            SOURCE1_ID_COL: pl.String,
            CANDIDATE_ID_COL: pl.String,
            "name_tfidf_cosine": pl.Float32,
            "addr_tfidf_cosine": pl.Float32,
            BLOCKING_FLAGS_COL: pl.UInt8,
        })

        scored = compute_reranker_composite_score(df, use_strategy_fallbacks=True)
        # Strategy B score: 0.70 * 0.35 = 0.245
        assert np.isclose(scored[RERANKER_SCORE_COL][0], 0.70 * 0.35, atol=1e-4)
        # Strategy D score: 0.70 * 0.30 = 0.210
        assert np.isclose(scored[RERANKER_SCORE_COL][1], 0.70 * 0.30, atol=1e-4)


# ==============================================================================
# 2. Candidate Pruning & Bounding Tests
# ==============================================================================
class TestCandidatePruning:
    def test_head_15_truncation(self):
        # 25 candidates for a single S1 entity -> must be truncated to top 15
        scores = [float(i) / 25.0 for i in range(25)]
        candidates = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-01"] * 25,
            CANDIDATE_ID_COL: [f"S2-{i:02d}" for i in range(25)],
            RERANKER_SCORE_COL: scores,
        })

        pruned = prune_candidates(candidates, max_candidates=15)
        assert len(pruned) == 15
        # Verify highest score was retained first
        assert pruned[CANDIDATE_ID_COL][0] == "S2-24"
        assert np.isclose(pruned[RERANKER_SCORE_COL][0], 24.0 / 25.0)
        # Verify lowest score retained is score from index 10 (25 - 15 = 10)
        assert pruned[CANDIDATE_ID_COL][-1] == "S2-10"

    def test_preserves_smaller_candidate_sets(self):
        # S1 entity with 5 candidates -> should keep all 5
        candidates = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-01"] * 5,
            CANDIDATE_ID_COL: [f"S2-{i}" for i in range(5)],
            RERANKER_SCORE_COL: [0.5, 0.9, 0.2, 0.7, 0.4],
        })

        pruned = prune_candidates(candidates, max_candidates=15)
        assert len(pruned) == 5
        assert pruned[RERANKER_SCORE_COL].to_list() == [0.9, 0.7, 0.5, 0.4, 0.2]

    def test_multi_entity_candidate_bounding(self):
        # Entity 1 has 20 candidates, Entity 2 has 8 candidates
        cands1 = [f"S2-1_{i}" for i in range(20)]
        cands2 = [f"S2-2_{i}" for i in range(8)]
        candidates = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1"] * 20 + ["S1-2"] * 8,
            CANDIDATE_ID_COL: cands1 + cands2,
            RERANKER_SCORE_COL: [0.5] * 28,
        })

        pruned = prune_candidates(candidates, max_candidates=15)
        # Entity 1 capped at 15; Entity 2 retains 8 -> total 23
        assert len(pruned) == 23
        counts = pruned.group_by(SOURCE1_ID_COL).len()
        e1_count = counts.filter(pl.col(SOURCE1_ID_COL) == "S1-1")["len"][0]
        e2_count = counts.filter(pl.col(SOURCE1_ID_COL) == "S1-2")["len"][0]
        assert e1_count == 15
        assert e2_count == 8


# ==============================================================================
# 3. Interface Contract 2 Parquet Export Tests
# ==============================================================================
class TestInterfaceContract2Export:
    def test_export_candidate_pairs_parquet_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "candidate_pairs.parquet"

            df = pl.DataFrame({
                SOURCE1_ID_COL: ["S1-01", "S1-02"],
                CANDIDATE_ID_COL: ["S2-01", "S3-02"],
                RERANKER_SCORE_COL: [0.85, 0.42],
                BLOCKING_FLAGS_COL: [9, 3],
            }, schema={
                SOURCE1_ID_COL: pl.String,
                CANDIDATE_ID_COL: pl.String,
                RERANKER_SCORE_COL: pl.Float32,
                BLOCKING_FLAGS_COL: pl.UInt8,
            })

            export_candidate_pairs_parquet(df, out_file)
            assert out_file.exists()

            # Read back and verify strict adherence to Interface Contract 2
            read_df = pl.read_parquet(out_file)
            assert read_df.columns == INTERFACE_CONTRACT_2_COLS
            assert read_df.schema[SOURCE1_ID_COL] == pl.String
            assert read_df.schema[CANDIDATE_ID_COL] == pl.String
            assert read_df.schema[RERANKER_SCORE_COL] == pl.Float32
            assert read_df.schema[BLOCKING_FLAGS_COL] == pl.UInt8
            assert len(read_df) == 2


# ==============================================================================
# 4. Submission TSV Export & Singleton Preservation Tests
# ==============================================================================
class TestSubmissionTsvExport:
    def test_export_candidate_pairs_tsv_singletons(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tsv_path = Path(tmp_dir) / "candidate_pairs.tsv"

            # 3 S1 entities: S1-01 (2 cands), S1-02 (1 cand), S1-03 (0 cands - singleton!)
            s1_ids = ["S1-01", "S1-02", "S1-03"]
            candidates = pl.DataFrame({
                SOURCE1_ID_COL: ["S1-01", "S1-01", "S1-02"],
                CANDIDATE_ID_COL: ["S2-10", "S3-20", "S2-30"],
                RERANKER_SCORE_COL: [0.9, 0.7, 0.8],
                BLOCKING_FLAGS_COL: [1, 1, 1],
            })

            export_candidate_pairs_tsv(candidates, s1_ids, tsv_path)
            assert tsv_path.exists()

            lines = tsv_path.read_text(encoding="utf-8").strip().split("\n")
            # Header + 3 entities = 4 lines
            assert len(lines) == 4
            assert lines[0] == "source1_entity_id\tcandidate_entity_ids"

            # S1-01 should have comma-separated candidates
            assert "S1-01\tS2-10,S3-20" in lines
            # S1-02 has one candidate
            assert "S1-02\tS2-30" in lines
            # S1-03 has empty candidate list
            assert "S1-03\t" in lines or lines[-1] == "S1-03"


# ==============================================================================
# 5. Prohibition of iter_rows() Verification
# ==============================================================================
class TestZeroIterRowsInReranking:
    def test_no_iter_rows_in_reranking_source(self):
        reranking_py_path = PROJECT_ROOT / "src" / "reranking.py"
        content = reranking_py_path.read_text(encoding="utf-8")
        assert "iter_rows" not in content, "iter_rows() is strictly prohibited in Polars logic!"
