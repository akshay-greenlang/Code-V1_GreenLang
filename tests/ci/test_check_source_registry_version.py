# -*- coding: utf-8 -*-
"""Tests for scripts/ci/check_source_registry_version.py."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_source_registry_version.py"


def _run(args):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return proc


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


BASE_YAML = textwrap.dedent("""
    sources:
      - source_id: epa_hub
        parser_version: 0.1.0
      - source_id: egrid
        parser_version: 0.1.0
""").lstrip()

# Bumped epa_hub from 0.1.0 -> 0.2.0
HEAD_YAML_BUMP = textwrap.dedent("""
    sources:
      - source_id: epa_hub
        parser_version: 0.2.0
      - source_id: egrid
        parser_version: 0.1.0
""").lstrip()

CHANGELOG_OK = textwrap.dedent("""
    # Source Registry CHANGELOG

    ## epa_hub 0.2.0 - 2026-04-28

    - Reason: schema bump for new column.
""").lstrip()

CHANGELOG_MISSING = textwrap.dedent("""
    # Source Registry CHANGELOG

    (no entries yet)
""").lstrip()


def test_no_bumps_passes(tmp_path):
    base = _write(tmp_path, "base.yaml", BASE_YAML)
    head = _write(tmp_path, "head.yaml", BASE_YAML)  # identical
    cl = _write(tmp_path, "changelog.md", CHANGELOG_MISSING)
    proc = _run([
        "--base-content", str(base),
        "--head-content", str(head),
        "--changelog-content", str(cl),
    ])
    assert proc.returncode == 0
    assert "no parser_version bumps" in proc.stdout


def test_bump_with_changelog_passes(tmp_path):
    base = _write(tmp_path, "base.yaml", BASE_YAML)
    head = _write(tmp_path, "head.yaml", HEAD_YAML_BUMP)
    cl = _write(tmp_path, "changelog.md", CHANGELOG_OK)
    proc = _run([
        "--base-content", str(base),
        "--head-content", str(head),
        "--changelog-content", str(cl),
    ])
    assert proc.returncode == 0, "stdout=%s\nstderr=%s" % (proc.stdout, proc.stderr)
    assert "PASS" in proc.stdout


def test_bump_without_changelog_fails(tmp_path):
    base = _write(tmp_path, "base.yaml", BASE_YAML)
    head = _write(tmp_path, "head.yaml", HEAD_YAML_BUMP)
    cl = _write(tmp_path, "changelog.md", CHANGELOG_MISSING)
    proc = _run([
        "--base-content", str(base),
        "--head-content", str(head),
        "--changelog-content", str(cl),
    ])
    assert proc.returncode == 1, "stdout=%s" % proc.stdout
    assert "FAIL" in proc.stdout
    assert "epa_hub" in proc.stdout


def test_changelog_header_case_insensitive(tmp_path):
    base = _write(tmp_path, "base.yaml", BASE_YAML)
    head = _write(tmp_path, "head.yaml", HEAD_YAML_BUMP)
    # Mixed-case header — should still match.
    cl_text = "## EPA_HUB 0.2.0 - 2026-04-28\n"
    cl = _write(tmp_path, "changelog.md", cl_text)
    proc = _run([
        "--base-content", str(base),
        "--head-content", str(head),
        "--changelog-content", str(cl),
    ])
    assert proc.returncode == 0, "stdout=%s" % proc.stdout
