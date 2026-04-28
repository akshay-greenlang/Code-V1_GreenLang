# -*- coding: utf-8 -*-
"""Tests for scripts/ci/check_raw_artifact_metadata.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_raw_artifact_metadata.py"


CATALOG_REL = "greenlang/factors/data/catalog_seed/demo_source/v0.1.0.json"
SHA = "a" * 64


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _write_catalog(root: Path, factors: list, *, name: str = CATALOG_REL):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "source_id": "demo_source",
                "source_version": "v0.1.0",
                "factors": factors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return p


def _record(urn: str, *, with_extraction: bool):
    rec = {
        "urn": urn,
        "factor_id": urn,
        "value": 1.0,
        "unit": "kgco2e/kwh",
    }
    if with_extraction:
        rec["extraction"] = {
            "raw_artifact_uri": "s3://bucket/raw/demo.xlsx",
            "raw_artifact_sha256": SHA,
        }
    return rec


def test_no_catalog_changes_passes(tmp_path):
    head_list = tmp_path / "head.txt"
    head_list.write_text("docs/factors/PHASE_3_PLAN.md\n", encoding="utf-8")
    proc = _run([
        "--head-content", str(head_list),
        "--head-dir", str(tmp_path),
        "--backfill-out", str(tmp_path / "BACKFILL.md"),
    ])
    assert proc.returncode == 0
    assert "no catalog seed JSON" in proc.stdout


def test_new_record_with_extraction_passes(tmp_path):
    base_dir = tmp_path / "base"
    head_dir = tmp_path / "head"
    _write_catalog(base_dir, [_record("urn:gl:factor:demo:1", with_extraction=True)])
    _write_catalog(
        head_dir,
        [
            _record("urn:gl:factor:demo:1", with_extraction=True),
            _record("urn:gl:factor:demo:2", with_extraction=True),
        ],
    )
    head_list = tmp_path / "head.txt"
    head_list.write_text(CATALOG_REL + "\n", encoding="utf-8")
    proc = _run([
        "--head-content", str(head_list),
        "--base-dir", str(base_dir),
        "--head-dir", str(head_dir),
        "--backfill-out", str(tmp_path / "BACKFILL.md"),
    ])
    assert proc.returncode == 0, "stdout=%s\nstderr=%s" % (proc.stdout, proc.stderr)
    assert "PASS" in proc.stdout


def test_new_record_without_extraction_fails(tmp_path):
    base_dir = tmp_path / "base"
    head_dir = tmp_path / "head"
    _write_catalog(base_dir, [_record("urn:gl:factor:demo:1", with_extraction=True)])
    _write_catalog(
        head_dir,
        [
            _record("urn:gl:factor:demo:1", with_extraction=True),
            _record("urn:gl:factor:demo:2", with_extraction=False),
        ],
    )
    head_list = tmp_path / "head.txt"
    head_list.write_text(CATALOG_REL + "\n", encoding="utf-8")
    proc = _run([
        "--head-content", str(head_list),
        "--base-dir", str(base_dir),
        "--head-dir", str(head_dir),
        "--backfill-out", str(tmp_path / "BACKFILL.md"),
    ])
    assert proc.returncode == 1, "stdout=%s" % proc.stdout
    assert "demo:2" in proc.stdout


def test_existing_record_without_extraction_writes_backfill(tmp_path):
    base_dir = tmp_path / "base"
    head_dir = tmp_path / "head"
    _write_catalog(base_dir, [_record("urn:gl:factor:demo:1", with_extraction=False)])
    # Same record (pre-existing, missing extraction) carried forward; head
    # has only this one record, so no NEW additions.
    _write_catalog(head_dir, [_record("urn:gl:factor:demo:1", with_extraction=False)])
    head_list = tmp_path / "head.txt"
    head_list.write_text(CATALOG_REL + "\n", encoding="utf-8")
    backfill_out = tmp_path / "BACKFILL.md"
    proc = _run([
        "--head-content", str(head_list),
        "--base-dir", str(base_dir),
        "--head-dir", str(head_dir),
        "--backfill-out", str(backfill_out),
    ])
    assert proc.returncode == 0, "stdout=%s" % proc.stdout
    assert backfill_out.is_file()
    body = backfill_out.read_text(encoding="utf-8")
    assert "demo:1" in body


def test_invalid_sha256_fails(tmp_path):
    base_dir = tmp_path / "base"
    head_dir = tmp_path / "head"
    _write_catalog(base_dir, [])
    bad = _record("urn:gl:factor:demo:bad", with_extraction=True)
    bad["extraction"]["raw_artifact_sha256"] = "ABC"  # not 64-hex lowercase
    _write_catalog(head_dir, [bad])
    head_list = tmp_path / "head.txt"
    head_list.write_text(CATALOG_REL + "\n", encoding="utf-8")
    proc = _run([
        "--head-content", str(head_list),
        "--base-dir", str(base_dir),
        "--head-dir", str(head_dir),
        "--backfill-out", str(tmp_path / "BACKFILL.md"),
    ])
    assert proc.returncode == 1
    assert "sha256" in proc.stdout.lower()
