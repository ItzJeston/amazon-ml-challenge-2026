"""
src/submission.py
Milestone 5 (Phase 7: Submission Generation, Dynamic Validation & Integrity Assertions)
Amazon Business Entity Resolution Challenge.

Responsibilities:
1. Enforce explicit UTF-8 stdout initialization:
   import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
2. Export final submission TSV files to output/:
   - output/matching_results.tsv (header: source1_entity_id\tmatched_entity_ids)
   - output/candidate_pairs.tsv (header: source1_entity_id\tcandidate_entity_ids)
3. Strict TSV formatting:
   - Exactly tab-separated (\t)
   - No quote characters (quote_style="never")
   - Singletons formatted as empty string "" (S1-xxxxx\t\n)
   - Deduplicated IDs within comma-separated list
   - No self matches (S1- IDs strictly forbidden in target lists)
   - Valid ID prefixes: only S2- and S3-
4. Hard runtime dynamic assertions:
   - assert len(matching_results) == len(test_source1)
   - assert len(candidate_pairs) == len(test_source1)
   - assert set(matching_results['source1_entity_id']) == set(test_source1['entity_id'])
   - assert every matched ID is a subset of candidate IDs for that S1 entity
5. Official validation wrapper against extracted_data/student_resource/utils/validate_submission.py
"""

import io
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

# Enforce explicit UTF-8 stdout initialization (Global Architecture Requirement)
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "").lower() != "utf-8":
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import polars as pl

from src.config import (
    ENTITY_ID_COL,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    VALIDATION_SCRIPT,
)
from src.features import CANDIDATE_ID_COL, SOURCE1_ID_COL
from src.utils import get_logger, setup_utf8_stdout

setup_utf8_stdout()
logger = get_logger("submission")

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
VALID_TARGET_PREFIXES = ("S2-", "S3-")


# ==============================================================================
# 1. ID Formatting and Cleaning Utilities
# ==============================================================================
def clean_target_ids(
    ids: Iterable[str],
    s1_id: Optional[str] = None,
    allow_empty: bool = True,
) -> List[str]:
    """
    Clean, validate, and deduplicate a sequence of candidate or matched entity IDs.
    - Strips whitespace
    - Disallows self-matches (S1- IDs)
    - Validates prefix (S2- or S3-)
    - Preserves deterministic sorted order with no intra-list duplicates
    """
    cleaned: Set[str] = set()
    for item in ids:
        if not item:
            continue
        # Support either individual IDs or comma-separated sub-strings
        sub_items = [x.strip() for x in str(item).split(",") if x.strip()]
        for target_id in sub_items:
            if target_id.startswith("S1-"):
                raise ValueError(
                    f"Self-match detected: target ID '{target_id}' starts with 'S1-'. "
                    f"Source-1 entities cannot match other Source-1 entities (s1_id={s1_id})."
                )
            if not target_id.startswith(VALID_TARGET_PREFIXES):
                raise ValueError(
                    f"Invalid target ID prefix: '{target_id}'. "
                    f"Only 'S2-' and 'S3-' prefixes are allowed."
                )
            cleaned.add(target_id)

    # Sort deterministically
    return sorted(cleaned)


def format_id_list_str(ids: Iterable[str], s1_id: Optional[str] = None) -> str:
    """Format cleaned IDs as comma-separated string, or empty string for singletons."""
    cleaned = clean_target_ids(ids, s1_id=s1_id)
    return ",".join(cleaned)


# ==============================================================================
# 2. Reading and Parsing Helpers
# ==============================================================================
def read_s1_entity_ids(
    source: Union[str, Path, Iterable[str], pl.DataFrame],
) -> List[str]:
    """
    Extract the complete list of unique S1 entity IDs from a file, dataframe, or iterable.
    Preserves input order if unique, otherwise returns ordered list of S1 IDs.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {path}")

        ids: List[str] = []
        seen: Set[str] = set()
        with open(path, mode="r", encoding="utf-8") as f:
            header = f.readline()  # Skip header
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                s1_id = line_str.split(DELIM, 1)[0].strip()
                if s1_id not in seen:
                    seen.add(s1_id)
                    ids.append(s1_id)
        return ids

    elif isinstance(source, pl.DataFrame):
        col = (
            "entity_id"
            if "entity_id" in source.columns
            else "source1_entity_id"
            if "source1_entity_id" in source.columns
            else source.columns[0]
        )
        return source[col].unique(maintain_order=True).to_list()

    elif isinstance(source, (list, tuple)):
        # Deduplicate while preserving order
        seen = set()
        out = []
        for x in source:
            x_str = str(x).strip()
            if x_str and x_str not in seen:
                seen.add(x_str)
                out.append(x_str)
        return out

    elif isinstance(source, (set, frozenset)):
        return sorted(str(x).strip() for x in source if str(x).strip())

    raise TypeError(f"Unsupported source type for read_s1_entity_ids: {type(source)}")


def parse_submission_tsv(
    tsv_path: Union[str, Path],
    expected_header: List[str],
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Parse a submission TSV file into a dictionary: {s1_id: set(matched/candidate IDs)}
    and an ordered list of S1 IDs.
    """
    path = Path(tsv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission file not found: {path}")

    mapping: Dict[str, Set[str]] = {}
    ordered_s1: List[str] = []
    seen: Set[str] = set()

    with open(path, mode="r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"Submission file is empty: {path}")

        cols = [c.strip().lower() for c in header_line.rstrip("\r\n").split(DELIM)]
        if cols != expected_header:
            raise ValueError(
                f"Unexpected header in {path}: {cols}. "
                f"Expected exactly {expected_header} (tab-separated)."
            )

        for line_num, line in enumerate(f, start=2):
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue

            s1, tab, rest = line_clean.partition(DELIM)
            if not tab:
                raise ValueError(
                    f"Malformed row (no tab delimiter) at line {line_num} in {path}: {line!r}"
                )

            s1 = s1.strip()
            if s1 in seen:
                raise ValueError(
                    f"Duplicate source1_entity_id row in {path} at line {line_num}: {s1}"
                )
            seen.add(s1)
            ordered_s1.append(s1)

            ids_str = rest.strip()
            if not ids_str:
                mapping[s1] = set()
            else:
                id_list = [x.strip() for x in ids_str.split(",") if x.strip()]
                if len(id_list) != len(set(id_list)):
                    raise ValueError(
                        f"Repeated ID within entity list at line {line_num} in {path} for {s1}: {id_list}"
                    )
                mapping[s1] = set(id_list)

    return mapping, ordered_s1


# ==============================================================================
# 3. TSV Export Engine (Strict Formatting & Singletons)
# ==============================================================================
def export_results_tsv(
    results: Union[Dict[str, Any], pl.DataFrame],
    s1_entity_ids: Union[List[str], Set[str], Path, str],
    output_path: Union[str, Path],
    header_col1: str,
    header_col2: str,
) -> Path:
    """
    Export results to a strictly formatted TSV file.
    - Exactly tab-separated (\t)
    - Zero quotes
    - Singletons (entities with 0 targets) written as S1-xxxxx\t\n
    - Guaranteed dynamic length equality with s1_entity_ids
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve complete ordered list of required S1 IDs
    all_s1_ids = read_s1_entity_ids(s1_entity_ids)
    if not all_s1_ids:
        raise ValueError("Cannot export TSV with an empty list of S1 entity IDs!")

    # 2. Standardize results into a mapping: s1_id -> formatted comma-separated string
    formatted_mapping: Dict[str, str] = {}

    if isinstance(results, dict):
        for s1_id, val in results.items():
            s1_str = str(s1_id).strip()
            if isinstance(val, (set, list, tuple)):
                formatted_mapping[s1_str] = format_id_list_str(val, s1_id=s1_str)
            elif isinstance(val, str):
                formatted_mapping[s1_str] = format_id_list_str(val.split(","), s1_id=s1_str)
            else:
                formatted_mapping[s1_str] = ""

    elif isinstance(results, pl.DataFrame):
        # Determine column names
        s1_col = (
            header_col1
            if header_col1 in results.columns
            else "source1_entity_id"
            if "source1_entity_id" in results.columns
            else "entity_id"
        )
        target_col = (
            header_col2
            if header_col2 in results.columns
            else "matched_entity_id"
            if "matched_entity_id" in results.columns
            else "candidate_entity_id"
            if "candidate_entity_id" in results.columns
            else "matched_entity_ids"
            if "matched_entity_ids" in results.columns
            else "candidate_entity_ids"
        )

        if target_col not in results.columns:
            raise KeyError(
                f"DataFrame must contain target column '{target_col}' or '{header_col2}'. "
                f"Found columns: {results.columns}"
            )

        # Check if already aggregated or pairwise
        if len(results) > 0:
            # Group by S1 entity ID using Polars relational expressions (ZERO row iteration)
            grouped = (
                results
                .filter(pl.col(target_col).is_not_null())
                .filter(pl.col(target_col).cast(pl.String).str.strip_chars().str.len_chars() > 0)
                .group_by(s1_col, maintain_order=True)
                .agg(pl.col(target_col).cast(pl.String))
            )

            # Convert to dictionary of lists for clean formatting
            s1_keys = grouped[s1_col].to_list()
            target_lists = grouped[target_col].to_list()
            for k, targets in zip(s1_keys, target_lists):
                k_str = str(k).strip()
                if isinstance(targets, list):
                    # Could be list of strings or list of lists
                    flat_targets = []
                    for t in targets:
                        if isinstance(t, str):
                            flat_targets.extend(t.split(","))
                        elif isinstance(t, (list, tuple, set)):
                            flat_targets.extend(t)
                    formatted_mapping[k_str] = format_id_list_str(flat_targets, s1_id=k_str)

    # 3. Write strictly formatted TSV file row-by-row
    written_count = 0
    with open(out_p, mode="w", encoding="utf-8", newline="\n") as f:
        # Write header
        f.write(f"{header_col1}{DELIM}{header_col2}\n")

        for s1_id in all_s1_ids:
            target_str = formatted_mapping.get(s1_id, "")
            # Strict formatting: S1-xxxxx\t\n for singleton, S1-xxxxx\tS2-yy,S3-zz\n for matches
            f.write(f"{s1_id}{DELIM}{target_str}\n")
            written_count += 1

    logger.info(
        f"Exported {written_count:,} rows to {out_p} "
        f"(Header: {header_col1}{DELIM}{header_col2})"
    )
    return out_p


def export_matching_results_tsv(
    matches: Union[Dict[str, Any], pl.DataFrame],
    s1_entity_ids: Union[List[str], Set[str], Path, str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Export final matching predictions to output/matching_results.tsv.
    Header: source1_entity_id\tmatched_entity_ids
    """
    target_path = Path(output_path or (OUTPUT_DIR / "matching_results.tsv"))
    return export_results_tsv(
        results=matches,
        s1_entity_ids=s1_entity_ids,
        output_path=target_path,
        header_col1="source1_entity_id",
        header_col2="matched_entity_ids",
    )


def export_candidate_pairs_tsv(
    candidates: Union[Dict[str, Any], pl.DataFrame],
    s1_entity_ids: Union[List[str], Set[str], Path, str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Export final candidate set to output/candidate_pairs.tsv.
    Header: source1_entity_id\tcandidate_entity_ids
    """
    target_path = Path(output_path or (OUTPUT_DIR / "candidate_pairs.tsv"))
    return export_results_tsv(
        results=candidates,
        s1_entity_ids=s1_entity_ids,
        output_path=target_path,
        header_col1="source1_entity_id",
        header_col2="candidate_entity_ids",
    )


# ==============================================================================
# 4. Hard Runtime Dynamic Assertions (PROJECT.md Interface Contract 4 / F25)
# ==============================================================================
def verify_submission_integrity(
    matching_source: Union[str, Path, Dict[str, Set[str]]],
    candidate_source: Union[str, Path, Dict[str, Set[str]]],
    test_source1: Union[str, Path, Iterable[str], pl.DataFrame],
) -> Dict[str, Any]:
    """
    Execute hard runtime dynamic assertions verifying full submission integrity:
    1. assert len(matching_results) == len(test_source1)
    2. assert len(candidate_pairs) == len(test_source1)
    3. assert set(matching_results['source1_entity_id']) == set(test_source1['entity_id'])
    4. assert set(candidate_pairs['source1_entity_id']) == set(test_source1['entity_id'])
    5. assert every matched ID is a subset of candidate IDs for that S1 entity
    6. assert no self matches (S1- IDs)
    7. assert no intra-list duplicate IDs
    8. assert only valid S2- and S3- prefixes

    Parameters
    ----------
    matching_source : Union[str, Path, Dict[str, Set[str]]]
        Path to matching_results.tsv or pre-parsed dict.
    candidate_source : Union[str, Path, Dict[str, Set[str]]]
        Path to candidate_pairs.tsv or pre-parsed dict.
    test_source1 : Union[str, Path, Iterable[str], pl.DataFrame]
        Ground truth reference Source 1 entity collection.

    Returns
    -------
    Dict[str, Any]
        Summary metrics of verified submission.
    """
    logger.info("Executing hard runtime dynamic assertions on submission files...")

    # 1. Load reference S1 entities
    ref_s1_list = read_s1_entity_ids(test_source1)
    ref_s1_set = set(ref_s1_list)
    ref_count = len(ref_s1_list)

    logger.info(f"Reference S1 count from test_source1: {ref_count:,}")

    # 2. Parse or load matching results
    if isinstance(matching_source, (str, Path)):
        matching_map, matching_order = parse_submission_tsv(
            matching_source, MATCHING_HEADER
        )
    else:
        matching_map = {
            str(k): set(clean_target_ids(v)) for k, v in matching_source.items()
        }
        matching_order = list(matching_map.keys())

    # 3. Parse or load candidate pairs
    if isinstance(candidate_source, (str, Path)):
        candidate_map, candidate_order = parse_submission_tsv(
            candidate_source, CANDIDATE_HEADER
        )
    else:
        candidate_map = {
            str(k): set(clean_target_ids(v)) for k, v in candidate_source.items()
        }
        candidate_order = list(candidate_map.keys())

    # --------------------------------------------------------------------------
    # Dynamic Assertions
    # --------------------------------------------------------------------------
    # Assertion 1: len(matching_results) == len(test_source1)
    matching_count = len(matching_order)
    assert matching_count == ref_count, (
        f"Dynamic assertion failed: matching_results row count ({matching_count:,}) "
        f"!= reference test_source1 length ({ref_count:,})"
    )

    # Assertion 2: len(candidate_pairs) == len(test_source1)
    candidate_count = len(candidate_order)
    assert candidate_count == ref_count, (
        f"Dynamic assertion failed: candidate_pairs row count ({candidate_count:,}) "
        f"!= reference test_source1 length ({ref_count:,})"
    )

    # Assertion 3: set(matching_results['source1_entity_id']) == set(test_source1['entity_id'])
    matching_s1_set = set(matching_map.keys())
    missing_matching = ref_s1_set - matching_s1_set
    extra_matching = matching_s1_set - ref_s1_set
    assert not missing_matching, (
        f"Dynamic assertion failed: {len(missing_matching)} required S1 entities missing from matching_results! "
        f"Sample: {list(missing_matching)[:5]}"
    )
    assert not extra_matching, (
        f"Dynamic assertion failed: {len(extra_matching)} unknown S1 entities found in matching_results! "
        f"Sample: {list(extra_matching)[:5]}"
    )

    # Assertion 4: set(candidate_pairs['source1_entity_id']) == set(test_source1['entity_id'])
    candidate_s1_set = set(candidate_map.keys())
    missing_candidate = ref_s1_set - candidate_s1_set
    extra_candidate = candidate_s1_set - ref_s1_set
    assert not missing_candidate, (
        f"Dynamic assertion failed: {len(missing_candidate)} required S1 entities missing from candidate_pairs! "
        f"Sample: {list(missing_candidate)[:5]}"
    )
    assert not extra_candidate, (
        f"Dynamic assertion failed: {len(extra_candidate)} unknown S1 entities found in candidate_pairs! "
        f"Sample: {list(extra_candidate)[:5]}"
    )

    # Assertion 5: every matched ID is a subset of candidate IDs for that S1 entity
    subset_offenders = {}
    total_matched_pairs = 0
    total_candidate_pairs = 0
    singleton_matches = 0

    for s1_id in ref_s1_list:
        matched_set = matching_map.get(s1_id, set())
        cand_set = candidate_map.get(s1_id, set())

        if not matched_set:
            singleton_matches += 1

        total_matched_pairs += len(matched_set)
        total_candidate_pairs += len(cand_set)

        # Matched IDs must be subset of candidate IDs
        diff = matched_set - cand_set
        if diff:
            subset_offenders[s1_id] = diff

        # Verify no S1- self matches
        s1_matches = [m for m in matched_set if m.startswith("S1-")]
        assert not s1_matches, f"Self-match S1 IDs found for {s1_id}: {s1_matches}"

        # Verify valid prefixes
        invalid_prefixes = [
            m for m in matched_set if not m.startswith(VALID_TARGET_PREFIXES)
        ]
        assert not invalid_prefixes, (
            f"Invalid target ID prefixes found for {s1_id}: {invalid_prefixes}"
        )

    assert not subset_offenders, (
        f"Dynamic assertion failed: {len(subset_offenders)} S1 entities have matched IDs "
        f"not present in candidate_pairs.tsv! Sample offenders: {list(subset_offenders.items())[:5]}"
    )

    metrics = {
        "status": "PASS",
        "total_s1_entities": ref_count,
        "matching_rows": matching_count,
        "candidate_rows": candidate_count,
        "singletons_count": singleton_matches,
        "singleton_fraction": round(singleton_matches / float(ref_count), 4) if ref_count else 0.0,
        "total_matched_links": total_matched_pairs,
        "total_candidate_links": total_candidate_pairs,
        "avg_candidates_per_s1": round(total_candidate_pairs / float(ref_count), 2) if ref_count else 0.0,
    }

    logger.info(
        f"Integrity Assertions PASSED! Evaluated {ref_count:,} entities | "
        f"Singletons: {singleton_matches:,} ({metrics['singleton_fraction']*100:.2f}%) | "
        f"Avg candidates/S1: {metrics['avg_candidates_per_s1']}"
    )
    return metrics


# ==============================================================================
# 5. Official Submission Validator Wrapper
# ==============================================================================
def validate_submission_files(
    matching_path: Union[str, Path],
    candidate_path: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    check_ids: bool = False,
    raise_on_error: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Run validation wrapper against extracted_data/student_resource/utils/validate_submission.py.
    Provides local error rejection diagnostics prior to challenge submission.

    Parameters
    ----------
    matching_path : Union[str, Path]
        Path to matching_results.tsv.
    candidate_path : Optional[Union[str, Path]]
        Path to candidate_pairs.tsv.
    test_dir : Optional[Union[str, Path]]
        Directory containing test_source1.tsv (and optionally test_source2/3.tsv).
    check_ids : bool
        If True, executes deep ID-existence check against test_source2/3.tsv.
    raise_on_error : bool
        If True, raises AssertionError when validator reports any errors.

    Returns
    -------
    Tuple[List[str], List[str]]
        (errors, warnings) lists reported by the validator.
    """
    m_path = str(Path(matching_path).resolve())
    c_path = str(Path(candidate_path).resolve()) if candidate_path else None
    t_dir = str(Path(test_dir or (RAW_DATA_DIR / "test")).resolve())

    logger.info(f"Invoking official validator on: matching={m_path}, candidate={c_path}, test_dir={t_dir}...")

    # Ensure validator directory is in sys.path
    val_script_dir = str(VALIDATION_SCRIPT.parent)
    if val_script_dir not in sys.path:
        sys.path.insert(0, val_script_dir)

    try:
        import validate_submission
        errors, warnings = validate_submission.validate(
            matching_path=m_path,
            candidate_path=c_path,
            test_dir=t_dir,
            check_ids=check_ids,
        )
    except Exception as e:
        logger.error(f"Failed to execute validate_submission directly: {e}. Executing native validation...")
        # Fallback to verify_submission_integrity
        ref_s1_path = Path(t_dir) / "test_source1.tsv"
        try:
            verify_submission_integrity(
                matching_source=m_path,
                candidate_source=c_path or m_path,
                test_source1=ref_s1_path,
            )
            errors, warnings = [], ["Native fallback validation passed."]
        except AssertionError as ae:
            errors, warnings = [str(ae)], []

    if errors:
        msg = f"Submission validation reported {len(errors)} ERROR(S):\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        logger.error(msg)
        if raise_on_error:
            raise AssertionError(msg)
    else:
        logger.info(f"Submission validation SUCCESSFUL! Warnings reported: {len(warnings)}")
        for warn in warnings:
            logger.info(f"  [Validator Warning] {warn}")

    return errors, warnings
