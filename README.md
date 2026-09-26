# Amazon ML Challenge 2026: Business Entity Resolution

Comprehensive documentation and onboarding guide for the **Business Entity Resolution** pipeline.

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
  - `source1`: The query entities to resolve.
  - `source2` & `source3`: The candidate pool containing potential matches.
- **Outputs**:
  1. `candidate_pairs.tsv`: Output of candidate generation / blocking (candidate pool per Source 1 entity).
  2. `matching_results.tsv`: Final predicted positive matches evaluated on the leaderboard.
- **Evaluation Metric**: **F0.5 Score** (places higher weight on **precision** than recall — false positives penalize score more severely than false negatives).

---

## 2. Directory Structure

```text
amazon-ml-challenge/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv       # Query entities with IDs, business names, addresses, country
│   │   ├── train_source2.tsv       # Candidate entity pool 1
│   │   ├── train_source3.tsv       # Candidate entity pool 2
│   │   └── train_ground_truth.tsv  # Actual entity mappings (source1_entity_id -> matched_entity_ids)
│   └── test/
│       ├── test_source1.tsv        # Test query entities (~175 MB)
│       ├── test_source2.tsv        # Test candidate pool 1 (~509 MB)
│       └── test_source3.tsv        # Test candidate pool 2 (~506 MB)
│
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt        # Python package dependencies
│       ├── run_pipeline.py         # Main execution pipeline script
│       └── src/
│           ├── cleaner.py          # Text normalization & standardization module
│           ├── blocker.py          # TF-IDF candidate generation module
│           └── matcher.py          # Feature extraction & LightGBM classification module
│
├── output/                         # Generated submission files
│   ├── candidate_pairs.tsv         # Top-K candidate pairs for each S1 entity
│   └── matching_results.tsv        # Final scored matches for leaderboard submission
│
├── utils/
│   └── validate_submission.py      # Official submission validator
└── README.md                       # This onboarding guide
```

---

## 3. What Has Been Built So Far

We built a modular, end-to-end Machine Learning pipeline following standard Entity Resolution best practices:

### A. Data Preprocessing & Cleaning (`src/cleaner.py`)
Raw company and address strings are messy. The cleaner standardizes text:
- **Case Normalization & Punctuation**: Converts strings to lowercase and strips special characters (`[^a-z0-9\s]`).
- **Legal & Address Suffix Expansion**: Normalizes abbreviations so they match across records:
  - `pvt` → `private`
  - `ltd` → `limited`
  - `inc` → `incorporated`
  - `st` / `rd` / `ave` → `street` / `road` / `avenue`
- **Full Text Construction**: Combines cleaned business name and address into `clean_full`.
- **Country Handling**: Normalizes country labels and fills missing entries with `UNKNOWN`.

### B. Candidate Generation / Blocking (`src/blocker.py`)
Comparing all entities in Source 1 against millions of entities in Sources 2 & 3 ($O(N \times M)$) is computationally impossible. We use **Blocking** to prune the search space down to high-probability candidates:
- **Country-based Partitioning**: Entities are partitioned by country to only search within the same geographical domain.
- **Character N-gram TF-IDF (`char_wb`, n-grams 3 to 5)**: Robust against minor typos and character permutations.
- **Nearest Neighbors Search**: Fast cosine similarity retrieval returning the top $K=20$ candidates per entity.

### C. Pairwise Feature Engineering & Classification (`src/matcher.py`)
For each candidate pair $(S_1, S_{cand})$, we extract a 10-dimensional similarity feature vector:
1. `fuzz.ratio(name1, name2)`
2. `fuzz.token_sort_ratio(name1, name2)`
3. `fuzz.token_set_ratio(name1, name2)`
4. `JaroWinkler.similarity(name1, name2)`
5. `fuzz.ratio(addr1, addr2)`
6. `fuzz.token_sort_ratio(addr1, addr2)`
7. `JaroWinkler.similarity(addr1, addr2)`
8. `fuzz.ratio(full_text1, full_text2)`
9. Name length ratio difference: `|len1 - len2| / max(len1, len2)`
10. Address length ratio difference: `|len1 - len2| / max(len1, len2)`

- **Model**: **LightGBM Classifier** (`LGBMClassifier`) trained on labeled candidate pairs from `train_ground_truth.tsv`.

### D. Pipeline Orchestrator (`run_pipeline.py`)
Connects the entire workflow:
1. Loads training data and parses ground truth matches.
2. Runs blocking to produce training pairs (positives from ground truth, negatives from non-matching candidates).
3. Trains the LightGBM matcher model.
4. Loads test datasets (`test_source1`, `test_source2`, `test_source3`).
5. Generates candidate blocks for test data and outputs `output/candidate_pairs.tsv`.
6. Performs pairwise feature extraction on test candidates and predicts match probabilities.
7. Applies a precision-oriented decision threshold (`THRESHOLD = 0.65` for F0.5 optimization) to output final matches in `output/matching_results.tsv`.

### E. Environment & Dependency Setup
Configured and verified dependencies in `code/business_entity_resolution/requirements.txt`:
- `rapidfuzz>=3.0.0`
- `lightgbm>=4.0.0`
- `scikit-learn>=1.2.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `tqdm>=4.65.0`

---

## 4. How to Run the Project

### 1. Install Dependencies
Make sure you are in the project folder and run:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
cd code/business_entity_resolution
python run_pipeline.py
```
This will train the model, run test inference, and write the output files to `output/`.

### 3. Validate Submission Files
Before submitting to the competition portal, run the official validation script:
```bash
# Run from repository root
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
A return code of `0` indicates the submission files strictly satisfy all format and constraint requirements.

---

## 5. Next Steps & Ideas for Improvement

If you want to push for a higher leaderboard score, here are high-impact enhancements:
1. **Memory & Batching Optimization**: The test dataset contains ~1.7M records across sources (~1.2 GB). Chunking or batching TF-IDF search prevents out-of-memory errors on modest hardware.
2. **Dense Semantic Embeddings**: Combine character n-gram TF-IDF with lightweight sentence embeddings (e.g., `all-MiniLM-L6-v2`) for semantic matching.
3. **Phonetic & Metaphone Matching**: Double Metaphone or Soundex features to capture phonetic misspellings.
4. **Threshold Tuning**: Run cross-validation to search for the optimal classification threshold that maximizes the F0.5 score directly.
