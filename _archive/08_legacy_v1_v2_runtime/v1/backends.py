# -*- coding: utf-8 -*-
"""Executable backend adapters for v1 app profiles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profiles import V1_APP_PROFILES, V1AppProfile
from .standards import write_observability_event

NATIVE_BACKEND_TIMEOUT_SECONDS = 240
REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


@dataclass
class BackendRunResult:
    success: bool
    exit_code: int
    artifacts: list[str]
    errors: list[str]
    warnings: list[str]
    native_backend_used: bool
    fallback_used: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audit_bundle(
    output_dir: Path,
    app_id: str,
    pipeline_id: str,
    artifacts: list[str],
    execution_mode: str,
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
        "execution_mode": execution_mode,
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
    write_observability_event(
        audit_dir / "observability_event.json",
        {
            "app_id": app_id,
            "pipeline_id": pipeline_id,
            "run_id": hashlib.sha256(f"{app_id}:{pipeline_id}".encode("utf-8")).hexdigest()[:12],
            "status": "ok",
            "duration_ms": 1,
        },
    )


def _run_subprocess(command: list[str], cwd: Path, env_overrides: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
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


def _build_csrd_report(input_path: Path, output_dir: Path) -> None:
    # Lightweight deterministic report from CSV/JSON inputs.
    rows = 0
    if input_path.suffix.lower() == ".csv":
        with open(input_path, "r", encoding="utf-8") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
    else:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = len(payload)
        elif isinstance(payload, dict):
            rows = len(payload.get("records", [])) if isinstance(payload.get("records"), list) else 1
        else:
            rows = 0

    report = {
        "app_id": "GL-CSRD-APP",
        "report_type": "esrs",
        "records_processed": rows,
        "status": "generated",
    }
    (output_dir / "esrs_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_vcci_inventory(input_path: Path, output_dir: Path) -> None:
    total_emissions = 0.0
    rows = 0
    if input_path.suffix.lower() == ".csv":
        with open(input_path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                supplier_pcf = float(row.get("supplier_pcf", 0) or 0)
                quantity = float(row.get("quantity", 0) or 0)
                total_emissions += supplier_pcf * quantity
    else:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        rows = 1
        supplier_pcf = float(payload.get("supplier_pcf", 0) or 0)
        quantity = float(payload.get("quantity", 0) or 0)
        total_emissions = supplier_pcf * quantity

    inventory = {
        "app_id": "GL-VCCI-Carbon-APP",
        "inventory_type": "scope3",
        "records_processed": rows,
        "total_emissions_kgco2e": round(total_emissions, 6),
    }
    (output_dir / "scope3_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _materialize_csrd_contract_artifact(output_dir: Path) -> bool:
    """
    Build `esrs_report.json` from native CSRD outputs when possible.

    Returns True if the contract artifact exists after normalization.
    """
    artifact = output_dir / "esrs_report.json"

    pipeline_result = output_dir / "pipeline_result.json"
    if not pipeline_result.exists():
        # If native output already produced a contract artifact, keep it.
        return artifact.exists()

    payload = json.loads(pipeline_result.read_text(encoding="utf-8"))
    report_payload = payload.get("csrd_report", {}) if isinstance(payload, dict) else {}
    report_metadata = report_payload.get("metadata", {}) if isinstance(report_payload, dict) else {}
    report_outputs = report_payload.get("outputs", {}) if isinstance(report_payload, dict) else {}
    compliance_audit = payload.get("compliance_audit", {}) if isinstance(payload, dict) else {}
    compliance_report = compliance_audit.get("compliance_report", {}) if isinstance(compliance_audit, dict) else {}
    normalized = {
        "app_id": "GL-CSRD-APP",
        "pipeline_id": "csrd-esrs-core",
        "status": payload.get("status", "unknown"),
        "compliance_status": payload.get("compliance_status", "unknown"),
        "records_processed": payload.get("total_data_points_processed", 0),
        "report_metadata": {
            "validation_status": report_metadata.get("validation_status"),
            "validation_errors": report_metadata.get("validation_errors"),
            "validation_warnings": report_metadata.get("validation_warnings"),
            "total_xbrl_facts": report_metadata.get("total_xbrl_facts"),
            "narratives_generated": report_metadata.get("narratives_generated"),
            "locale": report_metadata.get("locale"),
        },
        "esef_package": {
            "file_count": len((report_outputs.get("esef_package", {}) or {}).get("files", [])),
        },
        "compliance_audit": {
            "rules_checked": compliance_report.get("total_rules_checked"),
            "rules_passed": compliance_report.get("rules_passed"),
            "rules_failed": compliance_report.get("rules_failed"),
            "critical_failures": compliance_report.get("critical_failures"),
        },
    }
    artifact.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return True


def _materialize_vcci_contract_artifact(output_dir: Path) -> bool:
    """
    Build `scope3_inventory.json` from native VCCI outputs when possible.

    Returns True if the contract artifact exists after normalization.
    """
    artifact = output_dir / "scope3_inventory.json"
    if artifact.exists():
        return True

    reports = sorted(output_dir.glob("scope3_report_*.json"))
    if not reports:
        return False

    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    calculation = payload.get("calculation_results", {})
    normalized = {
        "app_id": "GL-VCCI-Carbon-APP",
        "inventory_type": "scope3",
        "records_processed": len(calculation.get("categories", {})),
        "total_emissions_kgco2e": round(float(calculation.get("total_emissions", 0.0)), 6),
    }
    artifact.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return True


def run_csrd_backend(
    input_path: Path,
    output_dir: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> BackendRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    app_root = _repo_path("applications", "GL-CSRD-APP", "CSRD-Reporting-Platform")
    command = [
        sys.executable,
        "csrd_pipeline.py",
        "--esg-data",
        str(input_path.resolve()),
        "--company-profile",
        str((app_root / "examples/demo_company_profile.json").resolve()),
        "--config",
        str((app_root / "config/csrd_config.yaml").resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    code, output = _run_subprocess(command, cwd=app_root)
    errors: list[str] = []
    warnings: list[str] = []
    if code == 0 and _materialize_csrd_contract_artifact(output_dir):
        artifacts = ["esrs_report.json", "audit/run_manifest.json", "audit/checksums.json"]
        _write_audit_bundle(
            output_dir=output_dir,
            app_id="GL-CSRD-APP",
            pipeline_id="csrd-esrs-core",
            artifacts=artifacts,
            execution_mode="native",
            status="ok",
            warnings=warnings,
        )
        return BackendRunResult(
            success=True,
            exit_code=0,
            artifacts=artifacts,
            errors=[],
            warnings=warnings,
            native_backend_used=True,
            fallback_used=False,
        )

    if code != 0:
        errors.append(f"csrd backend command failed: {code}")
    else:
        errors.append("csrd backend completed without required contract artifact: esrs_report.json")
    errors.append(output[-800:])

    if strict and not allow_fallback:
        return BackendRunResult(
            success=False,
            exit_code=code or 1,
            artifacts=[],
            errors=errors,
            warnings=warnings,
            native_backend_used=False,
            fallback_used=False,
        )

    warnings.append("csrd native backend unavailable; using deterministic fallback adapter")
    _build_csrd_report(input_path, output_dir)
    artifacts = ["esrs_report.json", "audit/run_manifest.json", "audit/checksums.json"]
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-CSRD-APP",
        pipeline_id="csrd-esrs-core",
        artifacts=artifacts,
        execution_mode="fallback",
        status="degraded",
        warnings=warnings,
    )
    return BackendRunResult(
        success=True,
        exit_code=0,
        artifacts=artifacts,
        errors=errors,
        warnings=warnings,
        native_backend_used=False,
        fallback_used=True,
    )


def _default_cbam_imports_path() -> Path:
    return _repo_path("cbam-pack-mvp", "examples", "sample_imports.csv")


def run_cbam_backend(
    input_path: Path,
    output_dir: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> BackendRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    imports_path = _default_cbam_imports_path()
    command = [
        sys.executable,
        "-m",
        "greenlang.cli.main",
        "run",
        "cbam",
        str(input_path.resolve()),
        str(imports_path.resolve()),
        str(output_dir.resolve()),
    ]
    code, output = _run_subprocess(command, cwd=REPO_ROOT)
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = [
        "cbam_report.xml",
        "report_summary.xlsx",
        "audit/run_manifest.json",
        "audit/checksums.json",
    ]
    missing = [artifact for artifact in artifacts if not (output_dir / artifact).exists()]
    if code == 0 and not missing:
        write_observability_event(
            output_dir / "audit" / "observability_event.json",
            {
                "app_id": "GL-CBAM-APP",
                "pipeline_id": "cbam-quarterly-core",
                "run_id": hashlib.sha256("GL-CBAM-APP:cbam-quarterly-core".encode("utf-8")).hexdigest()[:12],
                "status": "ok",
                "duration_ms": 1,
            },
        )
        return BackendRunResult(
            success=True,
            exit_code=0,
            artifacts=artifacts,
            errors=[],
            warnings=warnings,
            native_backend_used=True,
            fallback_used=False,
        )

    if code != 0:
        errors.append(f"cbam backend command failed: {code}")
    if missing:
        errors.append(f"cbam backend missing artifacts: {missing}")
    errors.append(output[-800:])
    if allow_fallback:
        warnings.append("cbam fallback adapter is not supported; strict native path required")
    return BackendRunResult(
        success=False,
        exit_code=code or 1,
        artifacts=[],
        errors=errors,
        warnings=warnings,
        native_backend_used=False,
        fallback_used=False,
    )


def run_vcci_backend(
    input_path: Path,
    output_dir: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> BackendRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    app_root = _repo_path("applications", "GL-VCCI-Carbon-APP", "VCCI-Scope3-Platform")
    command = [
        sys.executable,
        "v1_backend_entrypoint.py",
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_dir.resolve()),
        "--categories",
        "1",
        "--report-format",
        "ghg-protocol",
    ]
    env_overrides = {
        "GL_VCCI_ALLOW_OPTIONAL_FALLBACK": "1" if allow_fallback else "0",
    }
    code, output = _run_subprocess(command, cwd=app_root, env_overrides=env_overrides)
    errors: list[str] = []
    warnings: list[str] = []
    if code == 0 and _materialize_vcci_contract_artifact(output_dir):
        artifacts = ["scope3_inventory.json", "audit/run_manifest.json", "audit/checksums.json"]
        _write_audit_bundle(
            output_dir=output_dir,
            app_id="GL-VCCI-Carbon-APP",
            pipeline_id="vcci-scope3-core",
            artifacts=artifacts,
            execution_mode="native",
            status="ok",
            warnings=warnings,
        )
        return BackendRunResult(
            success=True,
            exit_code=0,
            artifacts=artifacts,
            errors=[],
            warnings=warnings,
            native_backend_used=True,
            fallback_used=False,
        )

    if code != 0:
        errors.append(f"vcci backend command failed: {code}")
    else:
        errors.append("vcci backend completed without required contract artifact: scope3_inventory.json")
    errors.append(output[-800:])

    if strict and not allow_fallback:
        return BackendRunResult(
            success=False,
            exit_code=code or 1,
            artifacts=[],
            errors=errors,
            warnings=warnings,
            native_backend_used=False,
            fallback_used=False,
        )

    warnings.append("vcci native backend unavailable; using deterministic fallback adapter")
    _build_vcci_inventory(input_path, output_dir)
    artifacts = ["scope3_inventory.json", "audit/run_manifest.json", "audit/checksums.json"]
    _write_audit_bundle(
        output_dir=output_dir,
        app_id="GL-VCCI-Carbon-APP",
        pipeline_id="vcci-scope3-core",
        artifacts=artifacts,
        execution_mode="fallback",
        status="degraded",
        warnings=warnings,
    )
    return BackendRunResult(
        success=True,
        exit_code=0,
        artifacts=artifacts,
        errors=errors,
        warnings=warnings,
        native_backend_used=False,
        fallback_used=True,
    )


def run_profile_backend(
    profile: V1AppProfile,
    input_path: Path,
    output_dir: Path,
    strict: bool = True,
    allow_fallback: bool = False,
) -> BackendRunResult:
    if profile.key == "cbam":
        return run_cbam_backend(
            input_path=input_path,
            output_dir=output_dir,
            strict=strict,
            allow_fallback=allow_fallback,
        )
    if profile.key == "csrd":
        return run_csrd_backend(
            input_path=input_path,
            output_dir=output_dir,
            strict=strict,
            allow_fallback=allow_fallback,
        )
    if profile.key == "vcci":
        return run_vcci_backend(
            input_path=input_path,
            output_dir=output_dir,
            strict=strict,
            allow_fallback=allow_fallback,
        )
    return BackendRunResult(
        success=False,
        exit_code=2,
        artifacts=[],
        errors=[f"profile '{profile.key}' backend adapter not implemented"],
        warnings=[],
        native_backend_used=False,
        fallback_used=False,
    )


def get_default_backend_input(profile_key: str) -> Path:
    if profile_key == "csrd":
        return _repo_path(
            "applications",
            "GL-CSRD-APP",
            "CSRD-Reporting-Platform",
            "examples",
            "demo_esg_data.csv",
        )
    if profile_key == "vcci":
        return _repo_path(
            "applications",
            "GL-VCCI-Carbon-APP",
            "VCCI-Scope3-Platform",
            "examples",
            "sample_category1_batch.csv",
        )
    if profile_key == "cbam":
        return _repo_path("cbam-pack-mvp", "examples", "sample_config.yaml")
    return (REPO_ROOT / V1_APP_PROFILES[profile_key].v1_dir / "smoke_input.json").resolve()

