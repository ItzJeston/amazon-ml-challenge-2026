"""
tests/test_blocking.py
Unit tests for Milestone 2: Cascaded Blocking Engine (src/blocking.py).

Tests:
1. UTF-8 stdout setup & imports
2. Sparse matrix top-N multiplication and pure SciPy fallback
3. Strategy A: Character N-Gram TF-IDF via CSR and sp_matmul_topn
4. Strategy B: Polars exact token inverted index (token length, frequency capping)
5. Strategy C: Word-level address TF-IDF (missing address safety)
6. Strategy D: Exact phonetic 2-token Soundex matching
7. 4-strategy union and bitmask aggregation
8. Zero iter_rows() verification in source code
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking import (
    BLOCKING_FLAG_ADDR_TFIDF,
    BLOCKING_FLAG_CHAR_TFIDF,
    BLOCKING_FLAG_SOUNDEX,
    BLOCKING_FLAG_TOKEN_INDEX,
    BLOCKING_FLAGS_COL,
    CANDIDATE_ID_COL,
    SOURCE1_ID_COL,
    SPARSE_DOT_TOPN_AVAILABLE,
    _csr_results_to_polars,
    blocking_strategy_a_char_tfidf,
    blocking_strategy_b_token_index,
    blocking_strategy_c_addr_tfidf,
    blocking_strategy_d_soundex,
    run_cascaded_blocking,
    run_sp_matmul_topn,
    sp_matmul_topn_fallback,
    union_candidate_sources,
)
from src.config import (
    ADDR_MISSING_FLAG_COL,
    CLEAN_ADDR_COL,
    CLEAN_NAME_COL,
    COUNTRY_COL,
    ENTITY_ID_COL,
    SOUNDEX_COL,
)


# ==============================================================================
# 1. Sparse Matrix Multiplier & Fallback Tests
# ==============================================================================
class TestSparseMatrixTopN:
    def test_sp_matmul_topn_fallback_correctness(self):
        # A: 2 rows, B: 3 rows
        A = sp.csr_matrix(
            np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], dtype=np.float32)
        )
        B = sp.csr_matrix(
            np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        )
        B_T = B.T.tocsr()

        C = sp_matmul_topn_fallback(A, B_T, top_n=2, threshold=0.1)
        assert C.shape == (2, 3)
        assert C.nnz == 4

        # Expected dot products for Row 0:
        # dot(A[0], B[0]) = 1.0 + 0.25 = 1.25 (col 0)
        # dot(A[0], B[1]) = 0.5 + 0.5 = 1.00 (col 1)
        # dot(A[0], B[2]) = 0.00 (< threshold)
        row0_cols = C.indices[C.indptr[0]:C.indptr[1]]
        row0_data = C.data[C.indptr[0]:C.indptr[1]]
        assert set(row0_cols) == {0, 1}
        assert np.isclose(row0_data[0], 1.25)
        assert np.isclose(row0_data[1], 1.0)

    def test_run_sp_matmul_topn_matches_threshold_and_topn(self):
        A = sp.csr_matrix(np.eye(5, dtype=np.float32))
        B_T = sp.csr_matrix(np.eye(5, dtype=np.float32))

        # Query with threshold 0.5 -> each row should match only its diagonal
        res = run_sp_matmul_topn(A, B_T, top_n=3, threshold=0.5)
        assert res.nnz == 5
        for i in range(5):
            assert res.indices[i] == i
            assert np.isclose(res.data[i], 1.0)

    def test_csr_results_to_polars_vectorized(self):
        A = sp.csr_matrix(np.array([[0.9, 0.0], [0.0, 0.8]], dtype=np.float32))
        s1_ids = np.array(["S1-1", "S1-2"])
        s23_ids = np.array(["S2-A", "S3-B"])

        df = _csr_results_to_polars(
            A, s1_ids, s23_ids,
            score_col_name="test_score",
            flag_col_name="test_flag",
            flag_val=1,
        )
        assert len(df) == 2
        assert df[SOURCE1_ID_COL].to_list() == ["S1-1", "S1-2"]
        assert df[CANDIDATE_ID_COL].to_list() == ["S2-A", "S3-B"]
        assert np.allclose(df["test_score"].to_list(), [0.9, 0.8])
        assert df["test_flag"].to_list() == [1, 1]


# ==============================================================================
# 2. Strategy A: Character N-Gram TF-IDF Tests
# ==============================================================================
class TestStrategyA_CharTfidf:
    def test_strategy_a_similar_names_retrieved(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-001", "S1-002"],
            CLEAN_NAME_COL: ["microsoft software", "starbucks coffee"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-101", "S3-202", "S2-303"],
            CLEAN_NAME_COL: ["microsoft corporation", "starbucks retail cafe", "random unaligned"],
        })

        df_a, vec, s23_csr_T = blocking_strategy_a_char_tfidf(
            s1_df, s23_df, top_n=5, threshold=0.1
        )

        assert len(df_a) >= 2
        assert df_a[SOURCE1_ID_COL].dtype == pl.String
        assert df_a[CANDIDATE_ID_COL].dtype == pl.String
        assert df_a["name_tfidf_cosine"].dtype == pl.Float32
        assert df_a["flag_a"].dtype == pl.UInt8

        # S1-001 should match S2-101 (Microsoft)
        ms_matches = df_a.filter(
            (pl.col(SOURCE1_ID_COL) == "S1-001") & (pl.col(CANDIDATE_ID_COL) == "S2-101")
        )
        assert len(ms_matches) == 1
        assert ms_matches["name_tfidf_cosine"][0] > 0.4
        assert ms_matches["flag_a"][0] == BLOCKING_FLAG_CHAR_TFIDF

        # S1-002 should match S3-202 (Starbucks)
        sb_matches = df_a.filter(
            (pl.col(SOURCE1_ID_COL) == "S1-002") & (pl.col(CANDIDATE_ID_COL) == "S3-202")
        )
        assert len(sb_matches) == 1
        assert sb_matches["name_tfidf_cosine"][0] > 0.3

    def test_strategy_a_empty_input_safe(self):
        empty_s1 = pl.DataFrame(schema={ENTITY_ID_COL: pl.String, CLEAN_NAME_COL: pl.String})
        empty_s23 = pl.DataFrame(schema={ENTITY_ID_COL: pl.String, CLEAN_NAME_COL: pl.String})
        df_a, _, _ = blocking_strategy_a_char_tfidf(empty_s1, empty_s23)
        assert len(df_a) == 0

    def test_strategy_a_precomputed_vectorizer_reuse(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            CLEAN_NAME_COL: ["tesla motors"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01"],
            CLEAN_NAME_COL: ["tesla energy"],
        })
        # Fit first
        df1, vec, s23_csr_T = blocking_strategy_a_char_tfidf(s1_df, s23_df)
        # Reuse pre-computed vectorizer and S23 transpose CSR
        df2, _, _ = blocking_strategy_a_char_tfidf(
            s1_df, s23_df, vectorizer=vec, s23_csr_T=s23_csr_T
        )
        assert len(df2) == len(df1)
        assert df2[CANDIDATE_ID_COL][0] == "S2-01"


# ==============================================================================
# 3. Strategy B: Polars Token Inverted Index Tests
# ==============================================================================
class TestStrategyB_TokenIndex:
    def test_strategy_b_token_overlap(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01", "S1-02"],
            CLEAN_NAME_COL: ["apex bio tech labs", "quantum dynamics systems"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01", "S3-02", "S2-03"],
            CLEAN_NAME_COL: ["apex bio innovations", "quantum systems global", "unrelated entity"],
        })

        df_b = blocking_strategy_b_token_index(s1_df, s23_df, min_token_len=3, max_df=1500, top_k=5)
        assert len(df_b) == 2

        # Check S1-01 matched S2-01
        m1 = df_b.filter(pl.col(SOURCE1_ID_COL) == "S1-01")
        assert m1[CANDIDATE_ID_COL][0] == "S2-01"
        assert m1["flag_b"][0] == BLOCKING_FLAG_TOKEN_INDEX

        # Check S1-02 matched S3-02
        m2 = df_b.filter(pl.col(SOURCE1_ID_COL) == "S1-02")
        assert m2[CANDIDATE_ID_COL][0] == "S3-02"

    def test_strategy_b_token_length_filter(self):
        # Tokens shorter than 3 chars (e.g. 'ai', 'co') should be ignored
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            CLEAN_NAME_COL: ["ai co"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01"],
            CLEAN_NAME_COL: ["ai co"],
        })
        df_b = blocking_strategy_b_token_index(s1_df, s23_df, min_token_len=3)
        assert len(df_b) == 0

    def test_strategy_b_frequency_capping(self):
        # Stopword 'solutions' appears 5 times in S23, max_df=3 -> should be filtered out
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            CLEAN_NAME_COL: ["global solutions"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: [f"S2-{i}" for i in range(5)],
            CLEAN_NAME_COL: ["solutions"] * 5,
        })
        df_b = blocking_strategy_b_token_index(s1_df, s23_df, min_token_len=3, max_df=3)
        # Because 'solutions' has frequency 5 > max_df 3, it should be dropped!
        assert len(df_b) == 0


# ==============================================================================
# 4. Strategy C: Word-Level Address TF-IDF Tests
# ==============================================================================
class TestStrategyC_AddressTfidf:
    def test_strategy_c_matching_addresses(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            CLEAN_ADDR_COL: ["1600 amphitheatre pkwy mountain view ca"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01", "S3-02"],
            CLEAN_ADDR_COL: ["1600 amphitheatre parkway mountain view", "1 infinite loop cupertino ca"],
        })

        df_c, _, _, _ = blocking_strategy_c_addr_tfidf(
            s1_df, s23_df, top_n=5, threshold=0.20
        )
        assert len(df_c) >= 1
        assert df_c[CANDIDATE_ID_COL][0] == "S2-01"
        assert df_c["addr_tfidf_cosine"][0] >= 0.20
        assert df_c["flag_c"][0] == BLOCKING_FLAG_ADDR_TFIDF

    def test_strategy_c_missing_address_safe(self):
        # Empty and null addresses must not crash and produce empty results
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01", "S1-02"],
            CLEAN_ADDR_COL: ["", None],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01"],
            CLEAN_ADDR_COL: ["123 main street"],
        })
        df_c, _, _, _ = blocking_strategy_c_addr_tfidf(s1_df, s23_df)
        assert len(df_c) == 0


# ==============================================================================
# 5. Strategy D: Exact Phonetic 2-Token Soundex Matching Tests
# ==============================================================================
class TestStrategyD_Soundex:
    def test_strategy_d_exact_soundex_match(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            SOUNDEX_COL: ["M262_T254"],  # Microsoft Technologies
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01", "S3-02"],
            SOUNDEX_COL: ["M262_T254", "A140_0000"],
        })

        df_d = blocking_strategy_d_soundex(s1_df, s23_df, max_block_size=500, top_k=5)
        assert len(df_d) == 1
        assert df_d[SOURCE1_ID_COL][0] == "S1-01"
        assert df_d[CANDIDATE_ID_COL][0] == "S2-01"
        assert df_d["flag_d"][0] == BLOCKING_FLAG_SOUNDEX

    def test_strategy_d_block_safety_capping(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            SOUNDEX_COL: ["A100_0000"],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: [f"S2-{i}" for i in range(100)],
            SOUNDEX_COL: ["A100_0000"] * 100,
        })
        # max_block_size=10, top_k=5
        df_d = blocking_strategy_d_soundex(s1_df, s23_df, max_block_size=10, top_k=5)
        assert len(df_d) <= 5

    def test_strategy_d_empty_soundex_safe(self):
        s1_df = pl.DataFrame({
            ENTITY_ID_COL: ["S1-01"],
            SOUNDEX_COL: [""],
        })
        s23_df = pl.DataFrame({
            ENTITY_ID_COL: ["S2-01"],
            SOUNDEX_COL: [""],
        })
        df_d = blocking_strategy_d_soundex(s1_df, s23_df)
        assert len(df_d) == 0


# ==============================================================================
# 6. Cascaded Union & Deduplication Tests
# ==============================================================================
class TestCascadedUnion:
    def test_union_candidate_sources_bitmask_and_deduplication(self):
        # Candidate pair (S1-1, S2-A) retrieved by A and D -> flag should be 1 | 8 = 9
        df_a = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1"],
            CANDIDATE_ID_COL: ["S2-A"],
            "name_tfidf_cosine": [0.85],
            "flag_a": [1],
        }, schema={SOURCE1_ID_COL: pl.String, CANDIDATE_ID_COL: pl.String, "name_tfidf_cosine": pl.Float32, "flag_a": pl.UInt8})

        # Candidate pair (S1-1, S2-A) also retrieved by B -> flag should be 9 | 2 = 11
        df_b = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1"],
            CANDIDATE_ID_COL: ["S2-A"],
            "flag_b": [2],
        }, schema={SOURCE1_ID_COL: pl.String, CANDIDATE_ID_COL: pl.String, "flag_b": pl.UInt8})

        # Candidate pair (S1-1, S3-B) retrieved by C
        df_c = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1"],
            CANDIDATE_ID_COL: ["S3-B"],
            "addr_tfidf_cosine": [0.90],
            "flag_c": [4],
        }, schema={SOURCE1_ID_COL: pl.String, CANDIDATE_ID_COL: pl.String, "addr_tfidf_cosine": pl.Float32, "flag_c": pl.UInt8})

        df_d = pl.DataFrame({
            SOURCE1_ID_COL: ["S1-1"],
            CANDIDATE_ID_COL: ["S2-A"],
            "flag_d": [8],
        }, schema={SOURCE1_ID_COL: pl.String, CANDIDATE_ID_COL: pl.String, "flag_d": pl.UInt8})

        unioned = union_candidate_sources(df_a, df_b, df_c, df_d)
        assert len(unioned) == 2

        # S2-A: Flags 1 | 2 | 8 = 11
        s2_row = unioned.filter(pl.col(CANDIDATE_ID_COL) == "S2-A")
        assert s2_row[BLOCKING_FLAGS_COL][0] == 11
        assert np.isclose(s2_row["name_tfidf_cosine"][0], 0.85)
        assert np.isclose(s2_row["addr_tfidf_cosine"][0], 0.0)

        # S3-B: Flag 4
        s3_row = unioned.filter(pl.col(CANDIDATE_ID_COL) == "S3-B")
        assert s3_row[BLOCKING_FLAGS_COL][0] == 4
        assert np.isclose(s3_row["name_tfidf_cosine"][0], 0.0)
        assert np.isclose(s3_row["addr_tfidf_cosine"][0], 0.90)

    def test_run_cascaded_blocking_end_to_end(self):
        s1 = pl.DataFrame({
            ENTITY_ID_COL: ["S1-10", "S1-20"],
            CLEAN_NAME_COL: ["tata consultancy services", "dassault aviation"],
            CLEAN_ADDR_COL: ["bkc bandra mumbai", "15 rue de la paix paris"],
            SOUNDEX_COL: ["T300_C524", "D243_A135"],
        })
        s23 = pl.DataFrame({
            ENTITY_ID_COL: ["S2-10", "S3-20", "S2-99"],
            CLEAN_NAME_COL: ["tata consultancy ltd", "dassault sasu", "unrelated entity"],
            CLEAN_ADDR_COL: ["bkc mumbai india", "15 r de la paix", "unrelated street"],
            SOUNDEX_COL: ["T300_C524", "D243_0000", "U564_0000"],
        })

        candidates = run_cascaded_blocking(s1, s23)
        assert len(candidates) >= 2
        assert set(candidates[SOURCE1_ID_COL].unique().to_list()) == {"S1-10", "S1-20"}


# ==============================================================================
# 7. Prohibition of iter_rows() Verification
# ==============================================================================
class TestZeroIterRowsInBlocking:
    def test_no_iter_rows_in_blocking_source(self):
        blocking_py_path = PROJECT_ROOT / "src" / "blocking.py"
        content = blocking_py_path.read_text(encoding="utf-8")
        assert "iter_rows" not in content, "iter_rows() is strictly prohibited in Polars logic!"
