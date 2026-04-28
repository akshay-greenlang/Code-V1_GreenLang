# -*- coding: utf-8 -*-
"""Tests for scripts/ci/check_pending_legal_blocked.py."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_pending_legal_blocked.py"


REGISTRY_OK = textwrap.dedent("""
    sources:
      - source_id: epa_hub
        status: alpha_v0_1
        release_milestone: v0.1
      - source_id: pending_src
        status: pending_legal_review
        release_milestone: v0.1
      - source_id: future_src
        status: alpha_v0_1
        release_milestone: v0.5
""").lstrip()


def _run(args):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_no_invocations_passes(tmp_path):
    reg = _write(tmp_path, "reg.yaml", REGISTRY_OK)
    workflows = _write(
        tmp_path, "wf.yml",
        "name: test\non: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hello\n",
    )
    proc = _run([
        "--registry-content", str(reg),
        "--workflows-content", str(workflows),
        "--workflows-dir", str(tmp_path / "_no_dir"),
        "--scripts-dir", str(tmp_path / "_no_dir"),
    ])
    assert proc.returncode == 0
    assert "no `gl factors ingest`" in proc.stdout


def test_dev_invocation_against_pending_passes(tmp_path):
    """Dev / staging runs are not blocked even against pending sources."""
    reg = _write(tmp_path, "reg.yaml", REGISTRY_OK)
    workflows = _write(
        tmp_path, "wf.yml",
        "run: gl factors ingest --env dev --source pending_src\n",
    )
    proc = _run([
        "--registry-content", str(reg),
        "--workflows-content", str(workflows),
        "--workflows-dir", str(tmp_path / "_no_dir"),
        "--scripts-dir", str(tmp_path / "_no_dir"),
    ])
    assert proc.returncode == 0, "stdout=%s" % proc.stdout


def test_prod_invocation_against_pending_fails(tmp_path):
    reg = _write(tmp_path, "reg.yaml", REGISTRY_OK)
    workflows = _write(
        tmp_path, "wf.yml",
        "run: gl factors ingest --env production --source pending_src\n",
    )
    proc = _run([
        "--registry-content", str(reg),
        "--workflows-content", str(workflows),
        "--workflows-dir", str(tmp_path / "_no_dir"),
        "--scripts-dir", str(tmp_path / "_no_dir"),
    ])
    assert proc.returncode == 1, "stdout=%s" % proc.stdout
    assert "pending_src" in proc.stdout


def test_prod_invocation_against_future_milestone_fails(tmp_path):
    reg = _write(tmp_path, "reg.yaml", REGISTRY_OK)
    workflows = _write(
        tmp_path, "wf.yml",
        "run: gl factors ingest --env production --source future_src\n",
    )
    proc = _run([
        "--registry-content", str(reg),
        "--workflows-content", str(workflows),
        "--workflows-dir", str(tmp_path / "_no_dir"),
        "--scripts-dir", str(tmp_path / "_no_dir"),
    ])
    assert proc.returncode == 1
    assert "future_src" in proc.stdout
    assert "v0.5" in proc.stdout or "release_milestone" in proc.stdout


def test_prod_invocation_against_approved_passes(tmp_path):
    reg = _write(tmp_path, "reg.yaml", REGISTRY_OK)
    workflows = _write(
        tmp_path, "wf.yml",
        "run: gl factors ingest --env production --source epa_hub\n",
    )
    proc = _run([
        "--registry-content", str(reg),
        "--workflows-content", str(workflows),
        "--workflows-dir", str(tmp_path / "_no_dir"),
        "--scripts-dir", str(tmp_path / "_no_dir"),
    ])
    assert proc.returncode == 0, "stdout=%s" % proc.stdout


def test_envvar_form_also_caught(tmp_path):
    reg = _write(tmp_path, "reg.yaml", REGISTRY_OK)
    workflows = _write(
        tmp_path, "wf.yml",
        "env:\n  GL_FACTORS_ENV: production\nrun: gl factors ingest --source pending_src\n",
    )
    proc = _run([
        "--registry-content", str(reg),
        "--workflows-content", str(workflows),
        "--workflows-dir", str(tmp_path / "_no_dir"),
        "--scripts-dir", str(tmp_path / "_no_dir"),
    ])
    assert proc.returncode == 1, "stdout=%s" % proc.stdout
    assert "pending_src" in proc.stdout
