# Business Entity Resolution Pipeline (V2 Architecture)
**Amazon ML Challenge 2026** — Branch: `feature/thanmay-blocker`

An end-to-end, memory-safe, high-precision Entity Resolution system designed to match noisy enterprise records from **Source 1** against a multi-million pool across **Source 2** and **Source 3**.

Optimized specifically for the competition metric: **Macro F0.5** (where precision is weighted 2× over recall, with strict singleton penalties).

---

## 🏗️ Architecture Pipeline

```text
                  Test Data (Source 1, Source 2, Source 3)
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Dynamic Inverted-Index Blocker (blocking.py)                     │
│  • Country-by-country dynamic partitioning (zero cross-border leakage)    │
│  • IDF-weighted token scoring with stopword postings cap (MAX=20,000)     │
│  • Retrieval: Top-20 candidates per S1 entity                             │
│  • Candidate Recall: 95.51% (up from 87.05% baseline)                     │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Top-20 candidates per entity
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Pairwise Feature Engineering (features.py)                       │
│  • 13 dense RapidFuzz & string similarity metrics (chunked in 100k blocks)│
│  • Hard Negative Mining (ranks candidates by blocking hardness)           │
│  • Numeric address consistency feature (deterministic veto flag)          │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  13-dimensional feature matrix
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Regularized LightGBM Classifier (classifier.py)                 │
│  • 500 trees, lr=0.03, num_leaves=63, L1/L2 regularization                │
│  • Decision threshold tuned for Macro F0.5 precision bias (optimal ~0.92)  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Raw match probabilities
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Post-Processing & Cardinality Engine (post_process.py)           │
│  • Enforces valid 1-to-many S1 matches (avg 3.67 matches per entity)      │
│  • Enforces pool candidate uniqueness (each S2/S3 entity claimed at most  │
│    once globally by highest-confidence S1 match)                          │
│  • True singletons preserved as clean blanks                              │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Format Validation & Auto-Packaging (test_inference.py)           │
│  • Validates integrity via utils/validate_submission.py                   │
│  • Automatically creates output/submission.zip ready for upload           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Upgrades in V2 (`thanmay-blocker`)

| Feature | Baseline (V1) | V2 Architecture | Impact |
| :--- | :--- | :--- | :--- |
| **Candidate Retrieval (Top-K)** | Top-12 | **Top-20** | Candidate Recall boosted to **95.51%** |
| **Feature Dimensionality** | 8 features | **13 features** | Added token Jaccard, WRatio, Partial Ratio, address token sort |
| **Negative Sampling** | Random negatives | **Hard Negative Mining** | Model learns discriminative edge cases |
| **Post-Processing** | Raw thresholding | **Pool-side Global Dedup** | Prevents candidate stealing across entities |
| **Peak RAM Footprint** | ~4.1 GB | **< 4.5 GB** | Safe streaming on any 8GB–16GB machine |
| **Submissions** | Manual zip | **Automated Validation & Zip** | Instant packaging with zero formatting errors |

---

## 📦 Directory Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── run_full_benchmark.py       # Full training & validation runner
│       └── src/
│           ├── blocking.py             # Stage 1: Inverted index candidate generation
│           ├── features.py             # Stage 2: 13-feature pairwise extraction
│           ├── classifier.py           # Stage 3: LightGBM training & threshold sweep
│           ├── post_process.py         # Stage 4: Bipartite pool-side deduplication
│           ├── test_inference.py       # Stage 5: Complete test inference & packager
│           ├── mini_benchmark.py       # 3-minute quick validation test
│           └── metrics.py              # Official Macro F0.5 metric implementation
├── dataset/                            # (Optional symlink)
├── output/                             # Generated artifacts and submission files
│   ├── candidate_pairs.tsv             # Top candidate pairs
│   ├── matching_results.tsv            # Final filtered match predictions
│   └── submission.zip                  # Automatically created submission archive
├── utils/
│   └── validate_submission.py          # Official format and integrity checker
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone this branch and install the required dependencies:

```bash
git fetch origin
git checkout feature/thanmay-blocker

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Ensure dataset files are located in the repository root as provided:
- Training: `6ab10eb3b23ba_student_resource/student_resource/dataset/train/`
- Testing: `6ab10eb3b23ba_student_resource/student_resource/dataset/test/`

---

## 🏃 Running the Pipeline

### Mode A: Fast-Track Test Inference (Recommended under time crunch)
When approaching a competition deadline, skip the 2-hour full benchmark by training on the quick split and running test inference directly:

```bash
# 1. Extract 13 features on training sample (~3 mins)
python3 code/business_entity_resolution/src/features.py

# 2. Train LightGBM & identify optimal threshold (~5 secs)
python3 code/business_entity_resolution/src/classifier.py

# 3. Run full test set inference & automatic packaging (~4-5 hours on laptop, ~2-3 on desktop)
python3 code/business_entity_resolution/src/test_inference.py
```

### Mode B: Full 2-Hour Benchmark (For local validation)
To reproduce the complete validation metrics across all 441,000 validation entities:

```bash
python3 code/business_entity_resolution/run_full_benchmark.py
```

### Mode C: 3-Minute Mini Benchmark (Sanity Check)
Quick evaluation on 5,000 entities against 500,000 candidate distractors:

```bash
python3 code/business_entity_resolution/src/mini_benchmark.py
```

---

## 📦 Submission Output Verification

When `test_inference.py` finishes, it automatically runs:
1. **Official Format Validator** (`utils/validate_submission.py`) to confirm:
   - All Source 1 entities have exactly 1 line.
   - Singletons are strictly blank.
   - S2/S3 entity references are valid.
2. **Automatic Zip Packaging**: Compresses `matching_results.tsv` and `candidate_pairs.tsv` into:
   ```text
   output/submission.zip
   ```
   **This is the exact file you upload to the challenge submission portal.**
