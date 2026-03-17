#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pack Integration Test Runner
==============================

CI-friendly runner for Solution Pack integration tests. Runs each pack's
integration tests in a separate pytest process to avoid conftest collisions
between packs that share the same `tests/conftest.py` module name.

Usage:
    python tests/run_pack_integration.py           # Run all pack integration tests
    python tests/run_pack_integration.py --verbose  # Verbose output
    python tests/run_pack_integration.py --pack 001 # Run only PACK-001 tests

Exit codes:
    0: All tests passed (or skipped)
    1: Some tests failed
    2: Error in test collection
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EU_COMPLIANCE = PROJECT_ROOT / "packs" / "eu-compliance"

PACK_DIRS = {
    "001": EU_COMPLIANCE / "PACK-001-csrd-starter",
    "002": EU_COMPLIANCE / "PACK-002-csrd-professional",
    "003": EU_COMPLIANCE / "PACK-003-csrd-enterprise",
    "004": EU_COMPLIANCE / "PACK-004-cbam-readiness",
    "005": EU_COMPLIANCE / "PACK-005-cbam-complete",
    "006": EU_COMPLIANCE / "PACK-006-eudr-starter",
    "007": EU_COMPLIANCE / "PACK-007-eudr-professional",
    "008": EU_COMPLIANCE / "PACK-008-eu-taxonomy-alignment",
    "009": EU_COMPLIANCE / "PACK-009-eu-climate-compliance-bundle",
    "010": EU_COMPLIANCE / "PACK-010-sfdr-article-8",
}

CROSS_PACK_TEST = PROJECT_ROOT / "tests" / "pack_integration" / "test_cross_pack_integration.py"


def run_single_test(test_file, verbose=False):
    """Run a single test file via pytest in its own process."""
    cmd = [
        sys.executable, "-m", "pytest",
        "-m", "integration",
        "--tb=short",
        "--no-header",
        f"--rootdir={PROJECT_ROOT}",
        "--override-ini=addopts=",
        "--timeout=120",
        str(test_file),
    ]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=not verbose)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run Solution Pack integration tests"
    )
    parser.add_argument(
        "--pack", type=str, default=None,
        help="Run tests for a specific pack (e.g., 001, 004)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose test output",
    )
    parser.add_argument(
        "--cross-pack-only", action="store_true",
        help="Only run the cross-pack integration suite",
    )
    args = parser.parse_args()

    # Determine which packs to test
    if args.cross_pack_only:
        test_files = [(CROSS_PACK_TEST, "Cross-Pack")]
    elif args.pack:
        pack_dir = PACK_DIRS.get(args.pack)
        if pack_dir is None:
            print(f"ERROR: Unknown pack ID '{args.pack}'")
            sys.exit(2)
        test_file = pack_dir / "tests" / "test_agent_integration.py"
        test_files = [(test_file, f"PACK-{args.pack}")]
    else:
        test_files = []
        for pack_id, pack_dir in PACK_DIRS.items():
            test_file = pack_dir / "tests" / "test_agent_integration.py"
            if test_file.exists():
                test_files.append((test_file, f"PACK-{pack_id}"))
        if CROSS_PACK_TEST.exists():
            test_files.append((CROSS_PACK_TEST, "Cross-Pack"))

    # Run each test file separately to avoid conftest collisions
    total = len(test_files)
    passed = 0
    failed = 0
    skipped = 0

    print(f"Running {total} integration test suite(s)...\n")

    for test_file, label in test_files:
        if not test_file.exists():
            print(f"  SKIP  {label}: {test_file} not found")
            skipped += 1
            continue

        print(f"  RUN   {label}: {test_file.name}")
        rc = run_single_test(test_file, verbose=args.verbose)
        if rc == 0:
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label} (exit code {rc})")
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped / {total} total")
    print(f"{'='*60}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
