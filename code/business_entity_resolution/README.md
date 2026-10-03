# Business Entity Resolution Pipeline (V2)
**Amazon ML Challenge 2026** — Branch: `feature/thanmay-blocker`

This folder contains the complete source code for the V2 Entity Resolution pipeline.

---

## 🏛️ Pipeline Overview

1. **Stage 1: Inverted Index Blocker (`src/blocking.py`)**
   - Dynamic country-by-country token indexing.
   - Stopword pruning (`MAX_POSTINGS = 20,000`).
   - Retrieves **Top-20** candidates per query entity.
   - Candidate recall: **95.51%**.

2. **Stage 2: Pairwise Feature Extractor (`src/features.py`)**
   - 13 pairwise string distance features (RapidFuzz WRatio, Token Set/Sort Ratio, Partial Ratio, Levenshtein, Token Overlaps, Number Overlap).
   - Hard Negative Mining (top negatives prioritized by blocking difficulty).
   - Numeric consistency indicator (`veto_flag`).

3. **Stage 3: LightGBM Classifier (`src/classifier.py`)**
   - Tuned parameters: 500 trees, lr=0.03, num_leaves=63, L1/L2 regularization.
   - Dynamic threshold search for Macro F0.5 optimization.

4. **Stage 4: Post-Processing & Cardinality Engine (`src/post_process.py`)**
   - Preserves 1-to-many matches per S1 entity.
   - Enforces unique assignment for S2/S3 candidate records.

5. **Stage 5: Test Inference & Packaging (`src/test_inference.py`)**
   - Country-by-country streaming across all 1.73M test queries.
   - Executes `utils/validate_submission.py`.
   - Packages `output/submission.zip` automatically.

---

## 🏃 Quick Start Commands

```bash
# Fast-Track Test Inference
python3 src/features.py
python3 src/classifier.py
python3 src/test_inference.py
```
*(All artifacts will be saved in `../../output/`)*
