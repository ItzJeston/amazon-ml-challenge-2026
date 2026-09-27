#!/usr/bin/env python3
"""
tests/e2e/runner.py
Master E2E Test Suite Harness and Runner
Amazon Business Entity Resolution Challenge

Usage:
    python tests/e2e/runner.py --all
    python tests/e2e/runner.py --tier 1
    python tests/e2e/runner.py --tier 2
    python tests/e2e/runner.py --tier 3
    python tests/e2e/runner.py --tier 4
    python tests/e2e/runner.py --all --verbose
    python tests/e2e/runner.py --all --report-json output/e2e_report.json
"""

import sys
import io
import os
import time
import json
import argparse
import unittest
from typing import Dict, List, Any

# Ensure stdout uses UTF-8 encoding (Requirement F02)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class DetailedTestResult(unittest.TextTestResult):
    """Custom TestResult collecting granular status and execution telemetry."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_records = []
        self._start_time = {}

    def startTest(self, test):
        self._start_time[test.id()] = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self._start_time.get(test.id(), time.time())
        self.test_records.append({
            "test_id": test.id(),
            "name": test._testMethodName,
            "status": "PASS",
            "duration": elapsed,
            "error": None
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        elapsed = time.time() - self._start_time.get(test.id(), time.time())
        self.test_records.append({
            "test_id": test.id(),
            "name": test._testMethodName,
            "status": "FAIL",
            "duration": elapsed,
            "error": self._exc_info_to_string(err, test)
        })

    def addError(self, test, err):
        super().addError(test, err)
        elapsed = time.time() - self._start_time.get(test.id(), time.time())
        self.test_records.append({
            "test_id": test.id(),
            "name": test._testMethodName,
            "status": "ERROR",
            "duration": elapsed,
            "error": self._exc_info_to_string(err, test)
        })

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_records.append({
            "test_id": test.id(),
            "name": test._testMethodName,
            "status": "SKIP",
            "duration": 0.0,
            "error": reason
        })


def get_tier_modules(tier: int) -> List[str]:
    """Map tier integer to module names."""
    tier_map = {
        1: ["tests.e2e.test_tier1_features"],
        2: ["tests.e2e.test_tier2_boundary"],
        3: ["tests.e2e.test_tier3_pairwise"],
        4: ["tests.e2e.test_tier4_scenarios"],
    }
    return tier_map.get(tier, [])


def run_e2e_suite(
    tiers: List[int],
    verbose: bool = False,
    report_json_path: str = None
) -> int:
    """Execute E2E test suite for specified tiers."""
    print("=" * 80)
    print("  AMAZON BUSINESS ENTITY RESOLUTION — E2E TEST SUITE")
    print("=" * 80)
    print(f"  Active Tiers: {tiers}")
    print(f"  Verbosity: {'Verbose' if verbose else 'Standard'}")
    print(f"  Python Runtime: {sys.version.split()[0]} ({sys.platform})")
    print(f"  Target Hardware: 14 Cores / <=6GB RAM Target (<=10GB Limit) / RTX 4050 GPU")
    print("=" * 80)
    print()

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tiers:
        modules = get_tier_modules(t)
        for mod_name in modules:
            try:
                mod = __import__(mod_name, fromlist=["*"])
                sub_suite = loader.loadTestsFromModule(mod)
                suite.addTests(sub_suite)
                print(f"  [DISCOVERED] Tier {t}: {mod_name} ({sub_suite.countTestCases()} test cases)")
            except Exception as e:
                print(f"  [ERROR] Failed to import {mod_name}: {e}")
                return 1

    total_tests = suite.countTestCases()
    print(f"\n  Total Discovered Tests: {total_tests}")
    print("-" * 80)

    stream = io.StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2 if verbose else 1,
        resultclass=DetailedTestResult
    )

    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    # Print summary output
    print(stream.getvalue())
    print("-" * 80)
    print("  E2E TEST EXECUTION SUMMARY")
    print("-" * 80)
    print(f"  Total Run:    {result.testsRun}")
    print(f"  Passed:       {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failed:       {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Skipped:      {len(result.skipped)}")
    print(f"  Duration:     {elapsed:.2f} seconds")

    if result.wasSuccessful():
        print("\n  STATUS: >>> ALL TESTS PASSED (100% SUCCESS) <<<")
    else:
        print("\n  STATUS: >>> TEST SUITE FAILED <<<")
        if result.failures:
            print("\n  Failures:")
            for test, msg in result.failures:
                print(f"    - {test.id()}: {msg.splitlines()[-1] if msg else 'AssertionFailed'}")
        if result.errors:
            print("\n  Errors:")
            for test, msg in result.errors:
                print(f"    - {test.id()}: {msg.splitlines()[-1] if msg else 'Error'}")

    print("=" * 80)

    # Export JSON report if requested
    if report_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_json_path)), exist_ok=True)
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_run": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "elapsed_seconds": elapsed,
            "success": result.wasSuccessful(),
            "records": getattr(result, "test_records", [])
        }
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"  [REPORT SAVED] {report_json_path}")

    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(
        description="E2E Test Runner for Amazon Business Entity Resolution"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific test tier (1, 2, 3, or 4)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 4 tiers of tests (default)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose test output"
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to output machine-readable JSON report"
    )

    args = parser.parse_args()

    if args.tier:
        tiers = [args.tier]
    else:
        tiers = [1, 2, 3, 4]

    exit_code = run_e2e_suite(
        tiers=tiers,
        verbose=args.verbose,
        report_json_path=args.report_json
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
