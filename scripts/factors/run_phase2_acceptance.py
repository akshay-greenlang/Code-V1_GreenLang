#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GreenLang Factors Phase 2 - CTO §2.7 acceptance runner.

Executes the 10 minimum required test suites enumerated in
``docs/factors/PHASE_2_PLAN.md`` Section 2.7, prints a colour-coded
matrix of results, and exits with a non-zero code if any CTO row fails.

Usage::

    python scripts/factors/run_phase2_acceptance.py
    python scripts/factors/run_phase2_acceptance.py --no-color
    python scripts/factors/run_phase2_acceptance.py --json-out report.json

CI wiring: see ``.github/workflows/factors_ci.yml`` job
``phase2-acceptance``.

The runner takes no test framework dependencies beyond pytest-on-PATH;
each suite is launched via subprocess so a single broken suite does
not poison the others.

Performance budget: < 5 minutes wall-clock on a clean SQLite checkout
(per CTO Phase 2 acceptance constraints).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# CTO §2.7 minimum required suites - keep in lockstep with
# docs/factors/PHASE_2_TEST_COVERAGE.md.
# ---------------------------------------------------------------------------


@dataclass
class CTORow:
    """One row of the CTO §2.7 acceptance matrix."""

    cto_id: str
    title: str
    suite_paths: List[str]
    workstream: str
    in_flight: bool = False  # True if owned by a workstream still in flight

    # Filled in at run-time:
    status: str = "PENDING"
    duration_s: float = 0.0
    pytest_exit: int = -1
    output_tail: str = ""


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests" / "factors" / "v0_1_alpha" / "phase2"


def _suite(rel: str) -> str:
    return str(TESTS_ROOT / rel)


CTO_2_7_ROWS: List[CTORow] = [
    CTORow(
        cto_id="2.7-1",
        title="test_schema_validates_alpha_catalog",
        suite_paths=[_suite("test_schema_validates_alpha_catalog.py")],
        workstream="WS1",
    ),
    CTORow(
        cto_id="2.7-2",
        title="test_urn_parse_build_roundtrip",
        suite_paths=[_suite("test_urn_property_roundtrip.py")],
        workstream="WS2",
    ),
    CTORow(
        cto_id="2.7-3",
        title="test_urn_uniqueness_db",
        suite_paths=[_suite("test_urn_uniqueness_db.py")],
        workstream="WS2",
    ),
    CTORow(
        cto_id="2.7-4",
        title="test_ontology_fk_enforcement",
        suite_paths=[_suite("test_ontology_fk_enforcement.py")],
        workstream="WS10",
    ),
    CTORow(
        cto_id="2.7-5",
        title="test_alembic_up_down",
        suite_paths=[_suite("test_alembic_up_down.py")],
        workstream="WS7",
    ),
    CTORow(
        cto_id="2.7-6",
        title="test_seed_load (+ activity)",
        suite_paths=[
            _suite("test_seed_load.py"),
            _suite("test_activity_seed_load.py"),
        ],
        workstream="WS3+4+5+6",
    ),
    CTORow(
        cto_id="2.7-7",
        title="test_publish_rejection_matrix",
        suite_paths=[
            _suite("test_publish_rejection_matrix.py"),
            _suite("test_publish_pipeline_e2e.py"),
        ],
        workstream="WS8",
        in_flight=False,  # WS8 shipped 2026-04-27.
    ),
    CTORow(
        cto_id="2.7-8",
        title="test_api_query_factor_by_urn",
        suite_paths=[_suite("test_api_query_factor_by_urn.py")],
        workstream="WS10",
    ),
    CTORow(
        cto_id="2.7-9",
        title="test_sdk_fetch_by_urn",
        suite_paths=[_suite("test_sdk_fetch_by_urn.py")],
        workstream="WS10",
    ),
    CTORow(
        cto_id="2.7-10",
        title="test_provenance_checksum",
        suite_paths=[_suite("test_provenance_checksum.py")],
        workstream="WS10",
    ),
]


# ---------------------------------------------------------------------------
# Status formatting
# ---------------------------------------------------------------------------


_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "magenta": "\x1b[35m",
}


def _colour(text: str, colour: str, enable: bool) -> str:
    if not enable:
        return text
    return f"{_ANSI.get(colour, '')}{text}{_ANSI['reset']}"


def _status_label(row: CTORow) -> str:
    return row.status


def _status_colour(row: CTORow) -> str:
    if row.status == "PASS":
        return "green"
    if row.status == "FAIL":
        return "red"
    if row.status == "MISSING":
        return "red"
    if row.status == "IN-FLIGHT":
        return "yellow"
    if row.status == "SKIPPED":
        return "cyan"
    return "dim"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_single_suite(suite_path: str, *, env: dict) -> tuple[int, str, float]:
    """Run a single pytest invocation; return (exit_code, output_tail, duration_s)."""
    if not Path(suite_path).is_file():
        return 5, f"MISSING: {suite_path}\n", 0.0
    cmd = [
        sys.executable, "-m", "pytest",
        suite_path,
        "-q",
        "--tb=line",
        "--no-header",
        "-p", "no:cacheprovider",
    ]
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=600,  # per-suite cap; aggregate budget < 5 min
    )
    elapsed = time.monotonic() - started
    tail_lines = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    return proc.returncode, "\n".join(tail_lines), elapsed


def run_acceptance(
    rows: List[CTORow],
    *,
    skip_in_flight: bool,
    env: Optional[dict] = None,
) -> List[CTORow]:
    """Run every CTO row's suite(s); fill in status / duration / output_tail."""
    if env is None:
        env = os.environ.copy()
        # Hermetic defaults for CI - safe to override per row.
        env.setdefault("GL_FACTORS_RELEASE_PROFILE", "alpha-v0.1")
        env.setdefault("GL_ENV", "test")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    for row in rows:
        if row.in_flight and skip_in_flight:
            row.status = "IN-FLIGHT"
            row.pytest_exit = -1
            row.output_tail = (
                f"Skipped because workstream {row.workstream} is still "
                "authoring this suite."
            )
            continue

        # Run every path under this row sequentially; first non-zero stops it.
        total_dur = 0.0
        last_tail = ""
        last_exit = 0
        for path in row.suite_paths:
            exit_code, tail, dur = _run_single_suite(path, env=env)
            total_dur += dur
            last_tail = tail
            last_exit = exit_code
            if exit_code != 0:
                break

        row.duration_s = total_dur
        row.pytest_exit = last_exit
        row.output_tail = last_tail
        if last_exit == 0:
            row.status = "PASS"
        elif last_exit == 5:
            # pytest exit-5 = "no tests collected" / file missing
            row.status = "MISSING"
        else:
            row.status = "FAIL"

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_table(rows: List[CTORow], use_colour: bool) -> str:
    header = (
        f"{'#':<7} {'Title':<42} {'WS':<10} {'Status':<10} {'Time(s)':<8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        title = row.title[:42]
        status = _colour(row.status, _status_colour(row), use_colour)
        # Pad before colour-wrapping so widths still align.
        status_padded = f"{row.status:<10}"
        coloured = _colour(status_padded, _status_colour(row), use_colour)
        lines.append(
            f"{row.cto_id:<7} {title:<42} {row.workstream:<10} "
            f"{coloured} {row.duration_s:>6.2f}"
        )
    return "\n".join(lines)


def _fmt_summary(rows: List[CTORow], use_colour: bool, total_runtime: float) -> str:
    counts = {"PASS": 0, "FAIL": 0, "MISSING": 0, "IN-FLIGHT": 0, "SKIPPED": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    parts = []
    for label, key, colour in (
        ("PASS", "PASS", "green"),
        ("FAIL", "FAIL", "red"),
        ("MISSING", "MISSING", "red"),
        ("IN-FLIGHT", "IN-FLIGHT", "yellow"),
        ("SKIPPED", "SKIPPED", "cyan"),
    ):
        n = counts.get(key, 0)
        if n:
            parts.append(_colour(f"{label}={n}", colour, use_colour))
    summary = " ".join(parts) if parts else "(no rows)"
    return (
        f"\n{summary}  total_runtime={total_runtime:.2f}s  "
        f"({len(rows)} CTO §2.7 rows)"
    )


def _emit_failure_tails(rows: List[CTORow], use_colour: bool) -> str:
    failed = [r for r in rows if r.status in ("FAIL", "MISSING")]
    if not failed:
        return ""
    out = ["", _colour("Failure tails:", "bold", use_colour)]
    for r in failed:
        out.append(_colour(f"--- {r.cto_id} {r.title} ---", "red", use_colour))
        out.append(r.output_tail or "(no output)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the CTO §2.7 Phase 2 acceptance suite."
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour in the matrix output.",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Write the structured matrix as JSON to this path.",
    )
    p.add_argument(
        "--include-in-flight",
        action="store_true",
        help=(
            "Run the IN-FLIGHT row(s) too. Default: skipped because the "
            "owning workstream is still authoring them."
        ),
    )
    p.add_argument(
        "--require-all-pass",
        action="store_true",
        default=True,
        help=(
            "Exit non-zero unless EVERY row is PASS. Default behaviour. "
            "Pass --allow-in-flight to relax this."
        ),
    )
    p.add_argument(
        "--allow-in-flight",
        action="store_true",
        help=(
            "Treat IN-FLIGHT rows as non-blocking (exit 0 even when an "
            "IN-FLIGHT row is present). Default: blocks the build."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    use_colour = not args.no_color and sys.stdout.isatty()

    started = time.monotonic()
    rows = run_acceptance(
        CTO_2_7_ROWS,
        skip_in_flight=not args.include_in_flight,
    )
    total_runtime = time.monotonic() - started

    print(_colour("Phase 2 - CTO §2.7 Acceptance Matrix", "bold", use_colour))
    print(_fmt_table(rows, use_colour))
    print(_fmt_summary(rows, use_colour, total_runtime))
    failure_block = _emit_failure_tails(rows, use_colour)
    if failure_block:
        print(failure_block)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "rows": [asdict(r) for r in rows],
                    "total_runtime_s": total_runtime,
                },
                fh,
                indent=2,
                sort_keys=True,
            )

    # Decide exit code.
    bad_statuses = {"FAIL", "MISSING"}
    if not args.allow_in_flight:
        bad_statuses.add("IN-FLIGHT")
    has_bad = any(r.status in bad_statuses for r in rows)
    return 1 if has_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
