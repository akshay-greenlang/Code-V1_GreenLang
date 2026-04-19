# -*- coding: utf-8 -*-
"""Determinism and audit standards for GreenLang v2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


REQUIRED_AUDIT_ARTIFACTS = [
    "audit/run_manifest.json",
    "audit/checksums.json",
]


@dataclass
class DeterminismResult:
    same_fileset: bool
    diff_count: int
    diffs: list[str]


def compare_artifact_hashes(run_a: Path, run_b: Path) -> DeterminismResult:
    files_a = sorted([p.relative_to(run_a).as_posix() for p in run_a.rglob("*") if p.is_file()])
    files_b = sorted([p.relative_to(run_b).as_posix() for p in run_b.rglob("*") if p.is_file()])
    same_fileset = files_a == files_b

    diffs: list[str] = []
    for rel_path in set(files_a).intersection(files_b):
        hash_a = hashlib.sha256((run_a / rel_path).read_bytes()).hexdigest()
        hash_b = hashlib.sha256((run_b / rel_path).read_bytes()).hexdigest()
        if hash_a != hash_b:
            diffs.append(rel_path)
    return DeterminismResult(
        same_fileset=same_fileset,
        diff_count=len(diffs),
        diffs=sorted(diffs),
    )

