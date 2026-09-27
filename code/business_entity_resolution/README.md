# Business Entity Resolution Pipeline
**Amazon ML Challenge 2026**

This repository contains the end-to-end, high-performance Entity Resolution pipeline for resolving noisy business entities across 3 disparate data sources.

---

## Architecture Overview

```
Test Data (S1, S2, S3)
       │
       ▼
[Stage 1: Inverted Index Blocker] (Country-by-Country IDF posting list query)
       │  Top-12 candidates per entity
       ▼
   output/candidate_pairs.tsv
       │
       ▼
[Stage 2: Pairwise Feature Extractor] (8 RapidFuzz similarity features in chunks)
       │
       ▼
[Stage 3: LightGBM Classifier] (Trained on 2.21M pairs, Decision Threshold = 0.88)
       │  High precision matching (Macro F0.5 = 0.8723)
       ▼
   output/matching_results.tsv
       │
       ▼
[Stage 4: Format Validator & Packaging] (utils/validate_submission.py -> submission.zip)
```

### Key Highlights:
1. **Dynamic Country Partitioning**: Automatically handles open-world countries (India, US, and test-introduced France) in isolated candidate pools, eliminating cross-country false positives.
2. **Memory-Safe Inverted Index**: Prunes postings lists (`MAX_POSTINGS = 20,000`) and avoids dense similarity matrices. Runs comfortably within 4.5 GB RAM on 10 million entities.
3. **Macro F0.5 Metric Alignment**: Decision threshold tuned to **0.88** to strongly prioritize precision (2× weight over recall) and singletons.

---

## Requirements & Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies:
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.2.0`
- `lightgbm>=4.0.0`
- `rapidfuzz>=3.0.0`

---

## Reproducing Results

### 1. Run Official Test Set Inference
To generate both `matching_results.tsv` and `candidate_pairs.tsv` and produce `submission.zip`:

```bash
python src/test_inference.py
```

### 2. Validate Submission Format
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
