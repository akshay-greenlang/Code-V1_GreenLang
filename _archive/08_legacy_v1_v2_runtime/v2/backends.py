# -*- coding: utf-8 -*-
"""Executable backend adapters for v2 app profiles."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
NATIVE_BACKEND_TIMEOUT_SECONDS = 240


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


def _run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=NATIVE_BACKEND_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timeout_msg = (
            f"backend command timed out after {NATIVE_BACKEND_TIMEOUT_SECONDS}s"
        )
        return 124, f"{stdout}{stderr}\n{timeout_msg}".strip()


def _run_eudr_backend_local(input_path: Path, output_dir: Path) -> BackendRunResult:
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
        fallback_used=True,
    )


def _run_ghg_backend_local(input_path: Path, output_dir: Path) -> BackendRunResult:
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
        fallback_used=True,
    )


def _run_sb253_backend_local(input_path: Path, output_dir: Path) -> BackendRunResult:
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
        "app_id": "GL-SB253-APP",
        "pipeline_id": "sb253-disclosure-v2",
        "status": status,
        "records_processed": records,
        "total_emissions_kgco2e": round(total_emissions, 6),
    }
    (output_dir / "sb253_disclosure.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = _required_artifacts("sb253")
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-SB253-APP",
        pipeline_id="sb253-disclosure-v2",
        artifacts=artifacts,
        status=status,
    )
    if status == "blocked":
        return BackendRunResult(
            success=True,
            exit_code=V2_BLOCKED_EXIT_CODE,
            artifacts=artifacts,
            errors=[],
            warnings=["policy gate blocked SB253 export"],
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
        fallback_used=True,
    )


def _run_taxonomy_backend_local(input_path: Path, output_dir: Path) -> BackendRunResult:
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
        "app_id": "GL-Taxonomy-APP",
        "pipeline_id": "eu-taxonomy-alignment-v2",
        "status": status,
        "records_processed": records,
        "total_emissions_kgco2e": round(total_emissions, 6),
    }
    (output_dir / "taxonomy_alignment.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = _required_artifacts("taxonomy")
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-Taxonomy-APP",
        pipeline_id="eu-taxonomy-alignment-v2",
        artifacts=artifacts,
        status=status,
    )
    if status == "blocked":
        return BackendRunResult(
            success=True,
            exit_code=V2_BLOCKED_EXIT_CODE,
            artifacts=artifacts,
            errors=[],
            warnings=["policy gate blocked taxonomy export"],
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
        fallback_used=True,
    )


def _run_iso14064_backend_local(input_path: Path, output_dir: Path) -> BackendRunResult:
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
        fallback_used=True,
    )


def _run_native_v2_backend(
    app_dir: Path,
    input_path: Path,
    output_dir: Path,
) -> tuple[int, str]:
    entrypoint = app_dir / "v2" / "runtime_backend.py"
    command = [
        sys.executable,
        str(entrypoint.resolve()),
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_dir.resolve()),
    ]
    return _run_subprocess(command, cwd=app_dir)


def run_v2_profile_backend(
    profile_key: str,
    input_path: Path,
    output_dir: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> BackendRunResult:
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
    native_failures: list[str] = []
    if key == "cbam":
        try:
            app_dir = REPO_ROOT / "applications" / "GL-CBAM-APP"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_subprocess(
                    [sys.executable, str(entrypoint.resolve()), "--input", str(input_path.resolve()), "--output", str(output_dir.resolve())],
                    cwd=app_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("cbam")
                    _write_audit_bundle(
                        output_dir=output_dir, app_id="GL-CBAM-APP",
                        pipeline_id="cbam-quarterly-v2", artifacts=artifacts, status="ok",
                    )
                    result = BackendRunResult(
                        success=True, exit_code=0, artifacts=artifacts, errors=[], warnings=[],
                        native_backend_used=True, fallback_used=False,
                    )
                else:
                    native_failures.append(f"cbam v2 native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(success=False, exit_code=code or 1, artifacts=[], errors=native_failures, warnings=[], native_backend_used=False, fallback_used=False)
                    result = run_cbam_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=True)
                    result.warnings = result.warnings + ["cbam v2 native failed; v1 fallback engaged"]
            else:
                result = run_cbam_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=allow_fallback)
                result.warnings = result.warnings + ["cbam v2 entrypoint missing; v1 backend used"]
        except Exception as exc:
            result = run_cbam_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=True)
            result.warnings = result.warnings + [f"cbam v2 error: {exc}; v1 fallback engaged"]
    elif key == "csrd":
        try:
            app_dir = REPO_ROOT / "applications" / "GL-CSRD-APP" / "CSRD-Reporting-Platform"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_subprocess(
                    [sys.executable, str(entrypoint.resolve()), "--input", str(input_path.resolve()), "--output", str(output_dir.resolve())],
                    cwd=app_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("csrd")
                    _write_audit_bundle(
                        output_dir=output_dir, app_id="GL-CSRD-APP",
                        pipeline_id="csrd-esrs-v2", artifacts=artifacts, status="ok",
                    )
                    result = BackendRunResult(
                        success=True, exit_code=0, artifacts=artifacts, errors=[], warnings=[],
                        native_backend_used=True, fallback_used=False,
                    )
                else:
                    native_failures.append(f"csrd v2 native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(success=False, exit_code=code or 1, artifacts=[], errors=native_failures, warnings=[], native_backend_used=False, fallback_used=False)
                    result = run_csrd_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=True)
                    result.warnings = result.warnings + ["csrd v2 native failed; v1 fallback engaged"]
            else:
                result = run_csrd_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=allow_fallback)
                result.warnings = result.warnings + ["csrd v2 entrypoint missing; v1 backend used"]
        except Exception as exc:
            result = run_csrd_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=True)
            result.warnings = result.warnings + [f"csrd v2 error: {exc}; v1 fallback engaged"]
    elif key == "vcci":
        try:
            app_dir = REPO_ROOT / "applications" / "GL-VCCI-Carbon-APP" / "VCCI-Scope3-Platform"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_subprocess(
                    [sys.executable, str(entrypoint.resolve()), "--input", str(input_path.resolve()), "--output", str(output_dir.resolve())],
                    cwd=app_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("vcci")
                    _write_audit_bundle(
                        output_dir=output_dir, app_id="GL-VCCI-Carbon-APP",
                        pipeline_id="vcci-scope3-v2", artifacts=artifacts, status="ok",
                    )
                    result = BackendRunResult(
                        success=True, exit_code=0, artifacts=artifacts, errors=[], warnings=[],
                        native_backend_used=True, fallback_used=False,
                    )
                else:
                    native_failures.append(f"vcci v2 native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(success=False, exit_code=code or 1, artifacts=[], errors=native_failures, warnings=[], native_backend_used=False, fallback_used=False)
                    result = run_vcci_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=True)
                    result.warnings = result.warnings + ["vcci v2 native failed; v1 fallback engaged"]
            else:
                result = run_vcci_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=allow_fallback)
                result.warnings = result.warnings + ["vcci v2 entrypoint missing; v1 backend used"]
        except Exception as exc:
            result = run_vcci_backend(input_path=input_path, output_dir=output_dir, strict=True, allow_fallback=True)
            result.warnings = result.warnings + [f"vcci v2 error: {exc}; v1 fallback engaged"]
    elif key == "eudr":
        try:
            app_dir = REPO_ROOT / "applications" / "GL-EUDR-APP"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_native_v2_backend(
                    app_dir=app_dir,
                    input_path=input_path,
                    output_dir=output_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("eudr")
                    payload = json.loads(
                        (output_dir / "due_diligence_statement.json").read_text(encoding="utf-8")
                    )
                    blocked = payload.get("status") == "blocked"
                    _write_audit_bundle(
                        output_dir=output_dir,
                        app_id="GL-EUDR-APP",
                        pipeline_id="eudr-due-diligence-v2",
                        artifacts=artifacts,
                        status="blocked" if blocked else "ok",
                    )
                    result = BackendRunResult(
                        success=True,
                        exit_code=V2_BLOCKED_EXIT_CODE if blocked else 0,
                        artifacts=artifacts,
                        errors=[],
                        warnings=["policy gate blocked EUDR export"] if blocked else [],
                        native_backend_used=True,
                        fallback_used=False,
                    )
                else:
                    native_failures.append(f"eudr native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(
                            success=False,
                            exit_code=code or 1,
                            artifacts=[],
                            errors=native_failures,
                            warnings=[],
                            native_backend_used=False,
                            fallback_used=False,
                        )
                    result = _run_eudr_backend_local(input_path=input_path, output_dir=output_dir)
                    result.errors = native_failures + result.errors
                    result.warnings = result.warnings + ["eudr fallback adapter engaged"]
            else:
                if strict and not allow_fallback:
                    return BackendRunResult(
                        success=False,
                        exit_code=1,
                        artifacts=[],
                        errors=["eudr native backend missing: applications/GL-EUDR-APP/v2/runtime_backend.py"],
                        warnings=[],
                        native_backend_used=False,
                        fallback_used=False,
                    )
                result = _run_eudr_backend_local(input_path=input_path, output_dir=output_dir)
                result.warnings = result.warnings + ["eudr fallback adapter engaged"]
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
            app_dir = REPO_ROOT / "applications" / "GL-GHG-APP"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_native_v2_backend(
                    app_dir=app_dir,
                    input_path=input_path,
                    output_dir=output_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("ghg")
                    payload = json.loads(
                        (output_dir / "ghg_inventory.json").read_text(encoding="utf-8")
                    )
                    blocked = payload.get("status") == "blocked"
                    _write_audit_bundle(
                        output_dir=output_dir,
                        app_id="GL-GHG-APP",
                        pipeline_id="ghg-inventory-v2",
                        artifacts=artifacts,
                        status="blocked" if blocked else "ok",
                    )
                    result = BackendRunResult(
                        success=True,
                        exit_code=V2_BLOCKED_EXIT_CODE if blocked else 0,
                        artifacts=artifacts,
                        errors=[],
                        warnings=["policy gate blocked GHG export"] if blocked else [],
                        native_backend_used=True,
                        fallback_used=False,
                    )
                else:
                    native_failures.append(f"ghg native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(
                            success=False,
                            exit_code=code or 1,
                            artifacts=[],
                            errors=native_failures,
                            warnings=[],
                            native_backend_used=False,
                            fallback_used=False,
                        )
                    result = _run_ghg_backend_local(input_path=input_path, output_dir=output_dir)
                    result.errors = native_failures + result.errors
                    result.warnings = result.warnings + ["ghg fallback adapter engaged"]
            else:
                if strict and not allow_fallback:
                    return BackendRunResult(
                        success=False,
                        exit_code=1,
                        artifacts=[],
                        errors=["ghg native backend missing: applications/GL-GHG-APP/v2/runtime_backend.py"],
                        warnings=[],
                        native_backend_used=False,
                        fallback_used=False,
                    )
                result = _run_ghg_backend_local(input_path=input_path, output_dir=output_dir)
                result.warnings = result.warnings + ["ghg fallback adapter engaged"]
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
            app_dir = REPO_ROOT / "applications" / "GL-ISO14064-APP"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_native_v2_backend(
                    app_dir=app_dir,
                    input_path=input_path,
                    output_dir=output_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("iso14064")
                    payload = json.loads(
                        (output_dir / "iso14064_verification_report.json").read_text(encoding="utf-8")
                    )
                    blocked = payload.get("status") == "blocked"
                    _write_audit_bundle(
                        output_dir=output_dir,
                        app_id="GL-ISO14064-APP",
                        pipeline_id="iso14064-verification-v2",
                        artifacts=artifacts,
                        status="blocked" if blocked else "ok",
                    )
                    result = BackendRunResult(
                        success=True,
                        exit_code=V2_BLOCKED_EXIT_CODE if blocked else 0,
                        artifacts=artifacts,
                        errors=[],
                        warnings=["policy gate blocked ISO14064 export"] if blocked else [],
                        native_backend_used=True,
                        fallback_used=False,
                    )
                else:
                    native_failures.append(f"iso14064 native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(
                            success=False,
                            exit_code=code or 1,
                            artifacts=[],
                            errors=native_failures,
                            warnings=[],
                            native_backend_used=False,
                            fallback_used=False,
                        )
                    result = _run_iso14064_backend_local(input_path=input_path, output_dir=output_dir)
                    result.errors = native_failures + result.errors
                    result.warnings = result.warnings + ["iso14064 fallback adapter engaged"]
            else:
                if strict and not allow_fallback:
                    return BackendRunResult(
                        success=False,
                        exit_code=1,
                        artifacts=[],
                        errors=["iso14064 native backend missing: applications/GL-ISO14064-APP/v2/runtime_backend.py"],
                        warnings=[],
                        native_backend_used=False,
                        fallback_used=False,
                    )
                result = _run_iso14064_backend_local(input_path=input_path, output_dir=output_dir)
                result.warnings = result.warnings + ["iso14064 fallback adapter engaged"]
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
    elif key == "sb253":
        try:
            app_dir = REPO_ROOT / "applications" / "GL-SB253-APP"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_native_v2_backend(
                    app_dir=app_dir,
                    input_path=input_path,
                    output_dir=output_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("sb253")
                    payload = json.loads(
                        (output_dir / "sb253_disclosure.json").read_text(encoding="utf-8")
                    )
                    blocked = payload.get("status") == "blocked"
                    _write_audit_bundle(
                        output_dir=output_dir,
                        app_id="GL-SB253-APP",
                        pipeline_id="sb253-disclosure-v2",
                        artifacts=artifacts,
                        status="blocked" if blocked else "ok",
                    )
                    result = BackendRunResult(
                        success=True,
                        exit_code=V2_BLOCKED_EXIT_CODE if blocked else 0,
                        artifacts=artifacts,
                        errors=[],
                        warnings=["policy gate blocked SB253 export"] if blocked else [],
                        native_backend_used=True,
                        fallback_used=False,
                    )
                else:
                    native_failures.append(f"sb253 native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(
                            success=False,
                            exit_code=code or 1,
                            artifacts=[],
                            errors=native_failures,
                            warnings=[],
                            native_backend_used=False,
                            fallback_used=False,
                        )
                    result = _run_sb253_backend_local(input_path=input_path, output_dir=output_dir)
                    result.errors = native_failures + result.errors
                    result.warnings = result.warnings + ["sb253 fallback adapter engaged"]
            else:
                if strict and not allow_fallback:
                    return BackendRunResult(
                        success=False,
                        exit_code=1,
                        artifacts=[],
                        errors=["sb253 native backend missing: applications/GL-SB253-APP/v2/runtime_backend.py"],
                        warnings=[],
                        native_backend_used=False,
                        fallback_used=False,
                    )
                result = _run_sb253_backend_local(input_path=input_path, output_dir=output_dir)
                result.warnings = result.warnings + ["sb253 fallback adapter engaged"]
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
    elif key == "taxonomy":
        try:
            app_dir = REPO_ROOT / "applications" / "GL-Taxonomy-APP"
            entrypoint = app_dir / "v2" / "runtime_backend.py"
            if entrypoint.exists():
                code, output = _run_native_v2_backend(
                    app_dir=app_dir,
                    input_path=input_path,
                    output_dir=output_dir,
                )
                if code == 0:
                    artifacts = _required_artifacts("taxonomy")
                    payload = json.loads(
                        (output_dir / "taxonomy_alignment.json").read_text(encoding="utf-8")
                    )
                    blocked = payload.get("status") == "blocked"
                    _write_audit_bundle(
                        output_dir=output_dir,
                        app_id="GL-Taxonomy-APP",
                        pipeline_id="eu-taxonomy-alignment-v2",
                        artifacts=artifacts,
                        status="blocked" if blocked else "ok",
                    )
                    result = BackendRunResult(
                        success=True,
                        exit_code=V2_BLOCKED_EXIT_CODE if blocked else 0,
                        artifacts=artifacts,
                        errors=[],
                        warnings=["policy gate blocked taxonomy export"] if blocked else [],
                        native_backend_used=True,
                        fallback_used=False,
                    )
                else:
                    native_failures.append(f"taxonomy native backend failed: {code}")
                    native_failures.append(output[-800:])
                    if strict and not allow_fallback:
                        return BackendRunResult(
                            success=False,
                            exit_code=code or 1,
                            artifacts=[],
                            errors=native_failures,
                            warnings=[],
                            native_backend_used=False,
                            fallback_used=False,
                        )
                    result = _run_taxonomy_backend_local(input_path=input_path, output_dir=output_dir)
                    result.errors = native_failures + result.errors
                    result.warnings = result.warnings + ["taxonomy fallback adapter engaged"]
            else:
                if strict and not allow_fallback:
                    return BackendRunResult(
                        success=False,
                        exit_code=1,
                        artifacts=[],
                        errors=["taxonomy native backend missing: applications/GL-Taxonomy-APP/v2/runtime_backend.py"],
                        warnings=[],
                        native_backend_used=False,
                        fallback_used=False,
                    )
                result = _run_taxonomy_backend_local(input_path=input_path, output_dir=output_dir)
                result.warnings = result.warnings + ["taxonomy fallback adapter engaged"]
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
