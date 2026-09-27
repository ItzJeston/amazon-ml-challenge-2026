# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed a scalable, memory-efficient, multi-stage Entity Resolution pipeline designed to resolve noisy business records across three disparate data sources. The solution combines dynamic country-partitioned inverted index blocking with rapid IDF candidate retrieval, a rich 8-dimensional pairwise similarity feature extraction engine using Levenshtein, Jaro-Winkler, and numeric token overlap, and a LightGBM classifier. By systematically calibrating our decision threshold to **0.88**, our solution heavily prioritizes precision and singleton preservation, achieving a validated **0.8723 Macro $F_{0.5}$** on the full 441,365 validation entities and scaling effortlessly across all 1.73M test queries (including France) without exceeding 4.5 GB of RAM.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights from exploratory data analysis:
- **Noise Patterns**: Business names exhibit extensive abbreviation differences (e.g., "Corp" vs. "Corporation", "Pvt Ltd" vs. "Private Limited"), word-order inversions, and typos.
- **Address Irregularities**: Addresses vary drastically in granularity (missing postal codes, landmarks like "Near SBI ATM", component re-ordering).
- **Metric Dynamics ($F_{0.5}$)**: The competition metric weights precision twice as heavily as recall ($F_{0.5}$). Crucially, singletons (entities with zero matches) award a full 1.0 score when correctly predicted as empty, but 0.0 on false merges. A high-precision conservative matching strategy is mathematically optimal.
- **Open-World Country Set**: The test set introduces a third country, `France`, that was absent in training. Partitioning logic must dynamically adapt at runtime rather than relying on hardcoded labels.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Stage Pipeline (Dynamic Country Partitioning $\rightarrow$ Inverted Index Blocker $\rightarrow$ RapidFuzz Feature Engineering $\rightarrow$ Threshold-Tuned LightGBM Classifier).
- **Core Innovation**: A postings-pruned inverted index blocking mechanism that avoids dense matrix computation entirely, combined with chunked country-by-country streaming to achieve 87.05% candidate recall across 10+ million entities under strict memory limits.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 10\text{M}$ comparison space:
- **Dynamic Country Partitioning**: Queries and candidate pools are partitioned by country (France, India, US). Entities from one country are strictly prohibited from matching candidates from another, eliminating cross-country false positives.
- **Blocking Keys & Inverted Index**: Text is normalized and tokenized into character n-grams and word tokens. An inverted index maps tokens to document postings with inverse document frequency (IDF) weights.
- **Postings List Pruning**: Tokens appearing in $> 20,000$ pool records (high-frequency stop-words) are pruned to prevent query latency blow-ups.
- **Candidate Retrieval**: For each Source 1 query, top candidate documents are accumulated using a min-heap (`heapq.nlargest`) based on aggregated IDF weights to retrieve the Top-12 candidate pool.
- **Recall Preservation**: Evaluated on the full 441,365 validation entities against 10.3M pool records, this blocking strategy captured **87.05%** of all true matches.

---

## 4. Matching Model

### Features Used (8 Pairwise Dimensions):
1. **`name_token_sort_ratio`**: Token sort ratio on `business_name` via RapidFuzz (handles word transpositions).
2. **`name_jaro_winkler`**: Prefix-weighted Jaro-Winkler distance on `business_name`.
3. **`address_token_set_ratio`**: Token set ratio on `business_address` (robust to subset/landmark additions).
4. **`address_levenshtein`**: Normalized Levenshtein similarity on `business_address`.
5. **`number_overlap`**: Jaccard similarity of extracted numeric tokens (`\d+`) from addresses (captures building numbers/PIN codes; defaults to -1.0 when neither has numbers).
6. **`len_diff_name`**: Normalized relative length difference of business names.
7. **`len_diff_addr`**: Normalized relative length difference of business addresses.
8. **`blocking_rank`**: Retrieval rank (1–12) from the blocking stage as a confidence proxy.

### Model Architecture & Training:
- **Classifier**: LightGBM (`LGBMClassifier`, 300 estimators, learning rate 0.05, 31 leaves).
- **Negative Subsampling**: Ground truth matches were retained (label 1), while non-matching candidate pairs were subsampled to at most 2 negatives per S1 entity during training to maintain balance and avoid combinatorial dataset explosion.
- **Threshold Selection**: Calibrated via grid search on the held-out validation split. A threshold of **`0.88`** maximized Macro $F_{0.5}$ by ruthlessly eliminating marginal matches.

---

## 5. Results & Error Analysis

- **Full Validation Macro $F_{0.5}$ Score**: **`0.8723` (87.23%)** at optimal decision threshold `0.88`.
- **Mini-Benchmark Score**: **`0.9456`** on 5,000 S1 entities against 500K distractors.
- **Common False Positives (Avoided)**: Shared chain names (e.g., branches of national franchises or retail outlets) at different street addresses were effectively rejected by the high threshold and address Levenshtein/number overlap features.
- **Singletons**: Accurately left 149,094 entities as empty matches in the test set, maximizing the singleton score bonus.

---

## 6. Conclusion

Our solution achieves state-of-the-art Entity Resolution performance by combining memory-safe inverted index candidate retrieval with precision-weighted gradient boosted decision trees. The system effortlessly scales to 10+ million entities across multiple countries in ~5.5 hours on standard hardware while strictly respecting all competition constraints.

---

## Appendix

### A. Code Artefacts
- **Entry Points**:
  - `src/test_inference.py`: Runs full test set inference and outputs `candidate_pairs.tsv` and `matching_results.tsv`.
  - `run_full_benchmark.py`: Reproduces the validation benchmark.
  - `src/blocking.py`: Inverted index blocking implementation.
  - `src/features.py`: Feature computation and subsampling logic.
  - `src/classifier.py`: LightGBM training and threshold search.
  - `src/metrics.py`: Official Macro $F_{0.5}$ calculation with singleton handling.
