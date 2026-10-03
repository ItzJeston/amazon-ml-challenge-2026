# Amazon ML Challenge 2026: Business Entity Resolution

Comprehensive documentation and onboarding guide for the **Business Entity Resolution** repository.

---

## 1. Problem Overview & Objective

### What is Business Entity Resolution (ER)?
Business Entity Resolution (or Record Linkage) is the task of identifying records from disparate databases that refer to the same real-world business entity. In real-world data, company names, addresses, and details often contain:
- Spelling errors, abbreviations (e.g., `Pvt Ltd` vs `Private Limited`, `St` vs `Street`).
- Missing or inconsistent fields.
- Different formatting and language variations across regions (e.g., US, India, France).

### The Challenge Goal
Given records from **Source 1** (`source1`), link each entity to its matching records in **Source 2** (`source2`) and **Source 3** (`source3`).
- **Input**:
  - `source1`: The query entities to resolve (~1.73M in test set).
  - `source2` & `source3`: The candidate pool containing potential matches (~10M records).
- **Outputs**:
  1. `output/candidate_pairs.tsv`: Output of candidate generation / blocking (candidate pool per Source 1 entity).
  2. `output/matching_results.tsv`: Final predicted positive matches evaluated on the leaderboard.
- **Evaluation Metric**: **Macro F0.5 Score** (places 2× weight on **precision** than recall — false positives penalize score more severely than false negatives, with strict singleton rules).

---

## 2. Directory Structure

```text
amazon-ml-challenge/
├── dataset/
│   ├── train/                  # Train sources (source1, source2, source3, ground_truth)
│   └── test/                   # Test sources (source1, source2, source3)
│
├── code/
│   └── business_entity_resolution/
│       ├── README.md           # Detailed documentation for the V2 high-performance pipeline
│       ├── run_full_benchmark.py # Complete 3-stage validation runner
│       └── src/
│           ├── blocking.py     # Stage 1: Dynamic Inverted-Index Blocker (Top-20, 95.5% recall)
│           ├── features.py     # Stage 2: 13-feature pairwise extractor with Hard Negative Mining
│           ├── classifier.py   # Stage 3: Tuned LightGBM Classifier with threshold search
│           ├── post_process.py # Stage 4: Global pool-side deduplication engine
│           ├── test_inference.py # Stage 5: Full test inference & auto-packager
│           ├── mini_benchmark.py # 3-minute quick validation runner
│           └── metrics.py      # Macro F0.5 metric implementation
│
├── src/                        # Baseline modular pipeline (TF-IDF & baseline matcher)
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── predict.py
│   └── preprocess.py
│
├── output/                     # Generated submission files
│   ├── candidate_pairs.tsv     # Top-K candidate pairs for each S1 entity
│   ├── matching_results.tsv    # Final scored matches for leaderboard submission
│   └── submission.zip          # Packaged zip ready for leaderboard portal
│
├── utils/
│   └── validate_submission.py  # Official submission format and integrity validator
├── requirements.txt            # Python dependencies
└── README.md                   # This onboarding guide
```

---

## 3. High-Performance V2 Pipeline (`code/business_entity_resolution/`)

An optimized, memory-safe, two-stage architecture designed to process all 1.73M queries against the 10M candidate pool comfortably within <4.5 GB RAM:

```text
                  Test Data (Source 1, Source 2, Source 3)
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Dynamic Inverted-Index Blocker (src/blocking.py)                 │
│  • Country-by-country dynamic partitioning (zero cross-border leakage)    │
│  • IDF-weighted token scoring with stopword postings cap (MAX=20,000)     │
│  • Retrieval: Top-20 candidates per S1 entity                             │
│  • Candidate Recall: 95.51% (up from 87.05% baseline)                     │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Top-20 candidates per entity
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Pairwise Feature Engineering (src/features.py)                   │
│  • 13 dense RapidFuzz & string similarity metrics (chunked in 100k blocks)│
│  • Hard Negative Mining (ranks candidates by blocking hardness)           │
│  • Numeric address consistency feature (deterministic veto flag)          │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  13-dimensional feature matrix
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Regularized LightGBM Classifier (src/classifier.py)              │
│  • 500 trees, lr=0.03, num_leaves=63, L1/L2 regularization                │
│  • Decision threshold tuned for Macro F0.5 precision bias                 │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │  Raw match probabilities
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Post-Processing & Cardinality Engine (src/post_process.py)       │
│  • Enforces valid 1-to-many S1 matches (avg 3.67 matches per entity)      │
│  • Enforces pool candidate uniqueness (each S2/S3 entity claimed at most  │
│    once globally by highest-confidence S1 match)                          │
│  • True singletons preserved as clean blanks                              │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Format Validation & Auto-Packaging (src/test_inference.py)       │
│  • Validates integrity via utils/validate_submission.py                   │
│  • Automatically creates output/submission.zip ready for upload           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. How to Run

### Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option A: Run Full Test Inference & Generate `submission.zip`
```bash
python3 code/business_entity_resolution/src/test_inference.py
```
This streams the 1.73M test records country-by-country, validates the output with `utils/validate_submission.py`, and packages `output/submission.zip`.

### Option B: Run 3-Minute Mini Benchmark
```bash
python3 code/business_entity_resolution/src/mini_benchmark.py
```

### Option C: Validate Any Generated Submission Files
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
