# -*- coding: utf-8 -*-
"""Determinism, auditability, and observability standards for GreenLang v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_AUDIT_ARTIFACTS = [
    "audit/run_manifest.json",
    "audit/checksums.json",
]

REQUIRED_OBSERVABILITY_FIELDS = [
    "app_id",
    "pipeline_id",
    "run_id",
    "status",
    "duration_ms",
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


def write_observability_event(path: Path, event: dict[str, Any]) -> None:
    payload = {field: event.get(field) for field in REQUIRED_OBSERVABILITY_FIELDS}
    payload["extra"] = {
        key: value for key, value in event.items() if key not in REQUIRED_OBSERVABILITY_FIELDS
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

