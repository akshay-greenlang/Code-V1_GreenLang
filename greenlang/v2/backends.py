# -*- coding: utf-8 -*-
"""Executable backend adapters for v2 app profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from greenlang.v1.backends import (
    BackendRunResult,
    run_cbam_backend,
    run_csrd_backend,
    run_vcci_backend,
)
from .profiles import V2_APP_PROFILES

V2_BLOCKED_EXIT_CODE = 4
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audit_bundle(
    output_dir: Path,
    app_id: str,
    pipeline_id: str,
    artifacts: list[str],
    status: str,
    warnings: list[str] | None = None,
) -> None:
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for artifact in artifacts:
        target = output_dir / artifact
        if target.exists():
            checksums[artifact] = _sha256(target)

    manifest = {
        "app_id": app_id,
        "pipeline_id": pipeline_id,
        "status": status,
        "execution_mode": "native",
        "artifacts": artifacts,
        "warnings": warnings or [],
    }
    (audit_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (audit_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _required_artifacts(profile_key: str) -> list[str]:
    profile = V2_APP_PROFILES[profile_key]
    gl_yaml = (REPO_ROOT / profile.v2_dir / "gl.yaml").resolve()
    payload = yaml.safe_load(gl_yaml.read_text(encoding="utf-8")) or {}
    runtime = payload.get("runtime_conventions", {}) or {}
    contract = runtime.get("artifact_contract", [])
    return [str(item) for item in contract if isinstance(item, str) and item.strip()]


def _validate_artifacts_exist(output_dir: Path, artifacts: list[str]) -> list[str]:
    return [artifact for artifact in artifacts if not (output_dir / artifact).exists()]


def _load_json_input(input_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid JSON input: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("input JSON root must be an object")
    return payload


def _run_eudr_backend(input_path: Path, output_dir: Path) -> BackendRunResult:
    payload = _load_json_input(input_path)
    suppliers = payload.get("suppliers", [])
    records_processed = len(suppliers) if isinstance(suppliers, list) else 0
    status = "blocked" if bool(payload.get("policy_block", False)) else "ok"
    statement = {
        "app_id": "GL-EUDR-APP",
        "pipeline_id": "eudr-due-diligence-v2",
        "status": status,
        "records_processed": records_processed,
        "high_risk_suppliers": len(
            [item for item in suppliers if isinstance(item, dict) and item.get("risk") == "high"]
        )
        if isinstance(suppliers, list)
        else 0,
    }
    (output_dir / "due_diligence_statement.json").write_text(
        json.dumps(statement, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = _required_artifacts("eudr")
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-EUDR-APP",
        pipeline_id="eudr-due-diligence-v2",
        artifacts=artifacts,
        status=status,
    )
    if status == "blocked":
        return BackendRunResult(
            success=True,
            exit_code=V2_BLOCKED_EXIT_CODE,
            artifacts=artifacts,
            errors=[],
            warnings=["policy gate blocked EUDR export"],
            native_backend_used=True,
            fallback_used=False,
        )
    return BackendRunResult(
        success=True,
        exit_code=0,
        artifacts=artifacts,
        errors=[],
        warnings=[],
        native_backend_used=True,
        fallback_used=False,
    )


def _run_ghg_backend(input_path: Path, output_dir: Path) -> BackendRunResult:
    payload = _load_json_input(input_path)
    activities = payload.get("activities", [])
    total_emissions = 0.0
    records = 0
    if isinstance(activities, list):
        for row in activities:
            if not isinstance(row, dict):
                continue
            records += 1
            quantity = float(row.get("quantity", 0) or 0)
            factor = float(row.get("emission_factor", 0) or 0)
            total_emissions += quantity * factor
    status = "blocked" if bool(payload.get("policy_block", False)) else "ok"
    report = {
        "app_id": "GL-GHG-APP",
        "pipeline_id": "ghg-inventory-v2",
        "status": status,
        "records_processed": records,
        "total_emissions_kgco2e": round(total_emissions, 6),
    }
    (output_dir / "ghg_inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = _required_artifacts("ghg")
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-GHG-APP",
        pipeline_id="ghg-inventory-v2",
        artifacts=artifacts,
        status=status,
    )
    if status == "blocked":
        return BackendRunResult(
            success=True,
            exit_code=V2_BLOCKED_EXIT_CODE,
            artifacts=artifacts,
            errors=[],
            warnings=["policy gate blocked GHG export"],
            native_backend_used=True,
            fallback_used=False,
        )
    return BackendRunResult(
        success=True,
        exit_code=0,
        artifacts=artifacts,
        errors=[],
        warnings=[],
        native_backend_used=True,
        fallback_used=False,
    )


def _run_iso14064_backend(input_path: Path, output_dir: Path) -> BackendRunResult:
    payload = _load_json_input(input_path)
    controls = payload.get("controls", [])
    control_count = len(controls) if isinstance(controls, list) else 0
    passed_count = len(
        [item for item in controls if isinstance(item, dict) and bool(item.get("passed", False))]
    ) if isinstance(controls, list) else 0
    status = "blocked" if bool(payload.get("policy_block", False)) else "ok"
    report = {
        "app_id": "GL-ISO14064-APP",
        "pipeline_id": "iso14064-verification-v2",
        "status": status,
        "controls_checked": control_count,
        "controls_passed": passed_count,
        "conformance_percent": round((passed_count / control_count) * 100.0, 2) if control_count else 0.0,
    }
    (output_dir / "iso14064_verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = _required_artifacts("iso14064")
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-ISO14064-APP",
        pipeline_id="iso14064-verification-v2",
        artifacts=artifacts,
        status=status,
    )
    if status == "blocked":
        return BackendRunResult(
            success=True,
            exit_code=V2_BLOCKED_EXIT_CODE,
            artifacts=artifacts,
            errors=[],
            warnings=["policy gate blocked ISO14064 export"],
            native_backend_used=True,
            fallback_used=False,
        )
    return BackendRunResult(
        success=True,
        exit_code=0,
        artifacts=artifacts,
        errors=[],
        warnings=[],
        native_backend_used=True,
        fallback_used=False,
    )


def run_v2_profile_backend(
    profile_key: str,
    input_path: Path,
    output_dir: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> BackendRunResult:
    del strict, allow_fallback  # V2 adapters are deterministic and local for now.
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        return BackendRunResult(
            success=False,
            exit_code=2,
            artifacts=[],
            errors=[f"input not found: {input_path}"],
            warnings=[],
            native_backend_used=False,
            fallback_used=False,
        )

    key = profile_key.lower()
    if key == "cbam":
        result = run_cbam_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=False)
    elif key == "csrd":
        result = run_csrd_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=False)
    elif key == "vcci":
        result = run_vcci_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=False)
    elif key == "eudr":
        try:
            result = _run_eudr_backend(input_path=input_path, output_dir=output_dir)
        except ValueError as exc:
            return BackendRunResult(
                success=False,
                exit_code=2,
                artifacts=[],
                errors=[str(exc)],
                warnings=[],
                native_backend_used=False,
                fallback_used=False,
            )
    elif key == "ghg":
        try:
            result = _run_ghg_backend(input_path=input_path, output_dir=output_dir)
        except ValueError as exc:
            return BackendRunResult(
                success=False,
                exit_code=2,
                artifacts=[],
                errors=[str(exc)],
                warnings=[],
                native_backend_used=False,
                fallback_used=False,
            )
    elif key == "iso14064":
        try:
            result = _run_iso14064_backend(input_path=input_path, output_dir=output_dir)
        except ValueError as exc:
            return BackendRunResult(
                success=False,
                exit_code=2,
                artifacts=[],
                errors=[str(exc)],
                warnings=[],
                native_backend_used=False,
                fallback_used=False,
            )
    else:
        return BackendRunResult(
            success=False,
            exit_code=2,
            artifacts=[],
            errors=[f"unsupported v2 profile key: {profile_key}"],
            warnings=[],
            native_backend_used=False,
            fallback_used=False,
        )

    if result.success and result.exit_code in {0, V2_BLOCKED_EXIT_CODE}:
        expected = _required_artifacts(key)
        missing = _validate_artifacts_exist(output_dir, expected)
        if missing:
            return BackendRunResult(
                success=False,
                exit_code=1,
                artifacts=result.artifacts,
                errors=[f"missing required artifacts for {key}: {missing}"],
                warnings=result.warnings,
                native_backend_used=False,
                fallback_used=False,
            )
    return result
