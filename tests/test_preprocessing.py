"""
Unit tests for data ingestion, preprocessing, Soundex, and country partitioning.
"""

from pathlib import Path
import tempfile
import polars as pl
import pytest

from src.config import (
    ADDR_MISSING_FLAG_COL,
    CLEAN_ADDR_COL,
    CLEAN_NAME_COL,
    COUNTRY_COL,
    ENTITY_ID_COL,
    PROCESSED_SCHEMA_COLS,
    SOUNDEX_COL,
)
from src.preprocessing import (
    compute_2token_soundex,
    compute_soundex_token,
    process_dataframe_chunk,
    process_source_file,
)


# ==============================================================================
# 1. Phonetic Soundex Tests
# ==============================================================================
class TestSoundex:
    def test_single_token_soundex(self):
        # Known American Soundex standard codes
        assert compute_soundex_token("Google") == "G240"
        assert compute_soundex_token("Amazon") == "A525"
        assert compute_soundex_token("Robert") == "R163"
        assert compute_soundex_token("Rupert") == "R163"

    def test_soundex_edge_cases(self):
        assert compute_soundex_token("") == ""
        assert compute_soundex_token(None) == ""
        assert compute_soundex_token("12345") == ""
        assert compute_soundex_token("$%^") == ""

    def test_2token_soundex_two_words(self):
        code = compute_2token_soundex("Amazon Retail")
        assert code == "A525_R340"

    def test_2token_soundex_three_words_takes_first_two(self):
        code = compute_2token_soundex("Amazon Web Services")
        assert code == "A525_W100"

    def test_2token_soundex_skips_short_tokens(self):
        # Tokens with < 3 characters (e.g. 'A', 'B') should be skipped
        code = compute_2token_soundex("A B Google Technologies")
        assert code == "G240_T254"

    def test_2token_soundex_single_word(self):
        code = compute_2token_soundex("Google")
        assert code == "G240"

    def test_2token_soundex_empty(self):
        assert compute_2token_soundex("") == ""
        assert compute_2token_soundex("   ") == ""
        assert compute_2token_soundex(None) == ""


# ==============================================================================
# 2. DataFrame Chunk Processing & Missing Address Tests
# ==============================================================================
class TestDataFrameChunkProcessing:
    def test_missing_address_flag_and_null_fill(self):
        data = {
            "entity_id": ["S1-001", "S2-002", "S3-003", "S1-004"],
            "business_name": [
                "Acme Corp.",
                "राम मार्केटिंग प्राइवेट लिमिटेड",
                "Dassault Aviation SASU",
                "Tata Steel Limited",
            ],
            "business_address": [
                "123 Main St, Apt 4B, New York, NY",
                None,  # Missing address in S2
                "15 R de la Paix, Cedex 06",
                None,  # Missing address in S1
            ],
            "country": ["US", "India", "France", "India"],
        }
        df_raw = pl.DataFrame(data)
        df_processed = process_dataframe_chunk(df_raw)

        # 1. Verify schema columns
        assert df_processed.columns == PROCESSED_SCHEMA_COLS

        # 2. Verify addr_missing_flag
        flags = df_processed[ADDR_MISSING_FLAG_COL].to_list()
        assert flags == [0, 1, 0, 1]

        # 3. Verify clean address handles null as empty string
        addrs = df_processed[CLEAN_ADDR_COL].to_list()
        assert addrs[1] == ""
        assert addrs[3] == ""
        assert "street" in addrs[0]
        assert "rue" in addrs[2]

        # 4. Verify clean name strips legal suffixes
        names = df_processed[CLEAN_NAME_COL].to_list()
        assert names[0] == "acme"
        assert "limited" not in names[1]
        assert names[2] == "dassault aviation"
        assert names[3] == "tata steel"

        # 5. Verify Soundex is generated
        soundexes = df_processed[SOUNDEX_COL].to_list()
        for sx in soundexes:
            assert len(sx) > 0
            assert "_" in sx or len(sx) == 4


# ==============================================================================
# 3. Partitioning & Parquet Export Tests
# ==============================================================================
class TestPartitioningAndExport:
    def test_partitioning_by_country_to_zstd_parquet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            proc_dir = tmp_path / "processed"

            split = "train"
            source_name = "test_mock_source"
            (raw_dir / split).mkdir(parents=True, exist_ok=True)

            mock_tsv_path = raw_dir / split / f"{source_name}.tsv"
            mock_data = (
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAcme Corp\t123 Main St\tUS\n"
                "S1-2\tGoogle Inc\t1600 Amphitheatre Pkwy\tUS\n"
                "S1-3\tTata Steel Ltd\tB/H City Mall, H. No. 5\tIndia\n"
                "S1-4\tInfosys Pvt Ltd\t\tIndia\n"
                "S1-5\tAirbus SAS\t15 R de la Paix\tFrance\n"
            )
            mock_tsv_path.write_text(mock_data, encoding="utf-8")

            # Run processing
            counts = process_source_file(
                split=split,
                source_name=source_name,
                raw_data_dir=raw_dir,
                processed_data_dir=proc_dir,
                batch_size=2,  # Exercise batching
            )

            assert counts["US"] == 2
            assert counts["India"] == 2
            assert counts["France"] == 1

            # Verify files exist in expected partition paths
            us_parquet = proc_dir / split / "US" / f"{source_name}.parquet"
            india_parquet = proc_dir / split / "India" / f"{source_name}.parquet"
            france_parquet = proc_dir / split / "France" / f"{source_name}.parquet"

            assert us_parquet.exists()
            assert india_parquet.exists()
            assert france_parquet.exists()

            # Read back parquet with Polars and verify contents
            df_us = pl.read_parquet(us_parquet)
            assert len(df_us) == 2
            assert set(df_us[COUNTRY_COL].to_list()) == {"US"}
            assert df_us.columns == PROCESSED_SCHEMA_COLS

            df_india = pl.read_parquet(india_parquet)
            assert len(df_india) == 2
            assert set(df_india[COUNTRY_COL].to_list()) == {"India"}
            # Record S1-4 had empty address -> missing flag should be 1
            s1_4_row = df_india.filter(pl.col(ENTITY_ID_COL) == "S1-4")
            assert s1_4_row[ADDR_MISSING_FLAG_COL][0] == 1
            assert s1_4_row[CLEAN_ADDR_COL][0] == ""
