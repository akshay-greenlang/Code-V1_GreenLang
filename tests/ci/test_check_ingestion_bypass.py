# -*- coding: utf-8 -*-
"""Tests for scripts/ci/check_ingestion_bypass.py.

We build a tiny fake repo tree under tmp_path with two sub-roots:
    greenlang/<file>.py
    scripts/<file>.py

and invoke the gate with --root tmp_path so it scans only the test tree.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_ingestion_bypass.py"


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _seed(root: Path, files: dict):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_clean_tree_passes(tmp_path):
    _seed(tmp_path, {
        "greenlang/clean.py": "def f():\n    return 1\n",
        "scripts/clean.py": "print('ok')\n",
    })
    proc = _run(["--root", str(tmp_path)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_factors_v01_factor_insert_fails(tmp_path):
    _seed(tmp_path, {
        "greenlang/bad.py": (
            "SQL = \"INSERT INTO factors_v0_1.factor (urn, value) VALUES (?, ?)\"\n"
        ),
    })
    proc = _run(["--root", str(tmp_path)])
    assert proc.returncode == 1, proc.stdout
    assert "factors_v0_1.factor" in proc.stdout


def test_alpha_factors_insert_fails(tmp_path):
    _seed(tmp_path, {
        "greenlang/bad.py": (
            "SQL = \"INSERT INTO alpha_factors_v0_1 (urn) VALUES (?)\"\n"
        ),
    })
    proc = _run(["--root", str(tmp_path)])
    assert proc.returncode == 1, proc.stdout
    assert "alpha_factors_v0_1" in proc.stdout


def test_executemany_factor_fails(tmp_path):
    _seed(tmp_path, {
        "greenlang/bad.py": (
            "conn.executemany('INSERT INTO factors_v0_1.factor VALUES (?)', rows)\n"
        ),
    })
    proc = _run(["--root", str(tmp_path)])
    assert proc.returncode == 1, proc.stdout


def test_factor_aliases_does_not_match(tmp_path):
    """factor_aliases is a sibling table — must NOT trigger the gate."""
    _seed(tmp_path, {
        "greenlang/ok.py": (
            "SQL = 'INSERT INTO factors_v0_1.factor_aliases (legacy_id) VALUES (?)'\n"
        ),
    })
    proc = _run(["--root", str(tmp_path)])
    assert proc.returncode == 0, proc.stdout


def test_whitelist_extra_skips(tmp_path):
    _seed(tmp_path, {
        "greenlang/bad.py": (
            "SQL = 'INSERT INTO factors_v0_1.factor (urn) VALUES (?)'\n"
        ),
    })
    proc = _run([
        "--root", str(tmp_path),
        "--whitelist-extra", "greenlang/bad.py",
    ])
    assert proc.returncode == 0, proc.stdout


def test_real_tree_passes():
    """Acceptance criterion: the actual repository must be clean today."""
    proc = _run([])
    assert proc.returncode == 0, (
        "current tree should have NO factor-table writes outside the canonical repository\n"
        + proc.stdout + proc.stderr
    )
