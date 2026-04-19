# -*- coding: utf-8 -*-
"""Runtime helpers for GreenLang v1 profile smoke execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .profiles import V1AppProfile
from .standards import REQUIRED_OBSERVABILITY_FIELDS, write_observability_event


def _load_profile_contract(profile: V1AppProfile) -> dict[str, Any]:
    with open(profile.v1_dir / "gl.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _stable_run_id(profile_key: str, input_path: Path) -> str:
    seed = f"{profile_key}:{input_path.as_posix()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def generate_profile_smoke_artifacts(
    profile: V1AppProfile,
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    """
    Generate deterministic smoke artifacts for a v1 app profile.

    This is intentionally deterministic and lightweight so CI lanes can
    validate runtime contracts without requiring full app backends.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_profile_contract(profile)
    runtime_conventions = contract.get("runtime_conventions", {}) or {}
    artifacts: list[str] = runtime_conventions.get("artifact_contract", [])

    run_id = _stable_run_id(profile.key, input_path)
    for artifact in artifacts:
        artifact_path = output_dir / artifact
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "app_id": profile.app_id,
            "profile": profile.key,
            "artifact": artifact,
            "run_id": run_id,
            "input": input_path.name,
            "status": "ok",
        }
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)

    write_observability_event(
        output_dir / "audit" / "observability_event.json",
        {
            "app_id": profile.app_id,
            "pipeline_id": contract.get("pipeline_id", "unknown"),
            "run_id": run_id,
            "status": "ok",
            "duration_ms": 1,
            "input_file": input_path.name,
        },
    )

    # Explicitly ensure required observability keys always exist.
    obs_path = output_dir / "audit" / "observability_event.json"
    with open(obs_path, "r", encoding="utf-8") as handle:
        obs_payload = json.load(handle)
    for field in REQUIRED_OBSERVABILITY_FIELDS:
        obs_payload.setdefault(field, None)
    with open(obs_path, "w", encoding="utf-8") as handle:
        json.dump(obs_payload, handle, sort_keys=True, indent=2)

    return artifacts

