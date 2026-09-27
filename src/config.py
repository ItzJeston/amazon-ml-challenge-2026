"""
Configuration and Hardware Settings for Business Entity Resolution Pipeline.
Optimized for Intel i7-13650HX (14 cores), 16GB RAM, RTX 4050 GPU (6GB VRAM).
"""

from pathlib import Path

# ==============================================================================
# 1. Base Paths & Directory Layout
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "extracted_data" / "student_resource" / "dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data_processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_UTILS_DIR = PROJECT_ROOT / "extracted_data" / "student_resource" / "utils"
VALIDATION_SCRIPT = TEST_UTILS_DIR / "validate_submission.py"

# Ensure output and processed dirs exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Hardware & Resource Limits
# ==============================================================================
NUM_THREADS = 14  # 14 physical execution threads (6 P-cores + 8 E-cores)
MEMORY_BUDGET_GB = 6.0  # Operational target peak RAM <= 6.0 GB
MEMORY_CEILING_GB = 10.0  # Hard upper RAM limit <= 10.0 GB
GPU_DEVICE = "cuda"  # Dedicated RTX 4050 6GB VRAM for XGBoost

# ==============================================================================
# 3. Dataset & Partitioning Constants
# ==============================================================================
TRAIN_SOURCES = ("train_source1", "train_source2", "train_source3")
TEST_SOURCES = ("test_source1", "test_source2", "test_source3")
TRAIN_GT_FILE = "train_ground_truth.tsv"

TRAIN_COUNTRIES = ("US", "India")
TEST_COUNTRIES = ("US", "India", "France")
ALL_COUNTRIES = ("US", "India", "France")

# Parquet storage configuration
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3

# Streaming / Batching chunk size for memory safety
STREAMING_BATCH_SIZE = 2_000_000

# ==============================================================================
# 4. Preprocessing & Soundex Configuration
# ==============================================================================
MIN_SOUNDEX_TOKEN_LEN = 3
MAX_SOUNDEX_TOKENS = 2

# Output schemas & column names
ENTITY_ID_COL = "entity_id"
RAW_NAME_COL = "business_name"
RAW_ADDR_COL = "business_address"
COUNTRY_COL = "country"

CLEAN_NAME_COL = "business_name_clean"
CLEAN_ADDR_COL = "business_address_clean"
SOUNDEX_COL = "soundex_2token"
ADDR_MISSING_FLAG_COL = "addr_missing_flag"

PROCESSED_SCHEMA_COLS = [
    ENTITY_ID_COL,
    "business_name_raw",
    CLEAN_NAME_COL,
    CLEAN_ADDR_COL,
    SOUNDEX_COL,
    COUNTRY_COL,
    ADDR_MISSING_FLAG_COL,
]
