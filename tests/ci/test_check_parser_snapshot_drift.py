# -*- coding: utf-8 -*-
"""Tests for scripts/ci/check_parser_snapshot_drift.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_parser_snapshot_drift.py"


def _run(args, cwd=None):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return proc


def _write_lines(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "label,head_lines,expected_exit",
    [
        ("no parser change", ["docs/factors/PHASE_3_PLAN.md"], 0),
        (
            "parser change without snapshot",
            ["greenlang/factors/ingestion/parsers/some_new_parser.py"],
            1,
        ),
        (
            "parser change WITH snapshot",
            [
                "greenlang/factors/ingestion/parsers/some_new_parser.py",
                "tests/factors/v0_1_alpha/phase3/parser_snapshots/some_new__0.1.0.golden.json",
            ],
            0,
        ),
        ("snapshot only", ["tests/factors/v0_1_alpha/phase3/parser_snapshots/x.json"], 0),
    ],
)
def test_drift_gate_parametrized(tmp_path, label, head_lines, expected_exit):
    head_file = _write_lines(tmp_path, "head.txt", head_lines)
    base_file = _write_lines(tmp_path, "base.txt", [])
    proc = _run([
        "--head-content", str(head_file),
        "--base-content", str(base_file),
        "--repo-root", str(REPO_ROOT),
    ])
    assert proc.returncode == expected_exit, (
        "case=%s\nstdout=%s\nstderr=%s" % (label, proc.stdout, proc.stderr)
    )


def test_drift_gate_override_marker_allows(tmp_path):
    """A parser file carrying the override marker line is allowed even
    without a snapshot change."""
    parser_dir = tmp_path / "greenlang" / "factors" / "ingestion" / "parsers"
    parser_dir.mkdir(parents=True)
    parser_path = parser_dir / "demo_override.py"
    parser_path.write_text(
        "# parser-snapshot-drift: intentional, regenerated via UPDATE_PARSER_SNAPSHOT=1\n"
        "def parse():\n    return []\n",
        encoding="utf-8",
    )
    head_file = _write_lines(
        tmp_path,
        "head.txt",
        ["greenlang/factors/ingestion/parsers/demo_override.py"],
    )
    proc = _run([
        "--head-content", str(head_file),
        "--repo-root", str(tmp_path),
    ])
    assert proc.returncode == 0, "stdout=%s\nstderr=%s" % (proc.stdout, proc.stderr)
    assert "override marker" in proc.stdout
