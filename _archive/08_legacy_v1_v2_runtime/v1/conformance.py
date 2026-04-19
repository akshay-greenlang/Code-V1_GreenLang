# -*- coding: utf-8 -*-
"""Conformance and release gate checks for GreenLang v1."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from greenlang.utilities.provenance.signing import verify_pack_signature

from .contracts import validate_v1_pack, validate_v1_pipeline
from .profiles import V1_APP_PROFILES
from .backends import get_default_backend_input, run_profile_backend
from .runtime import generate_profile_smoke_artifacts
from .standards import REQUIRED_OBSERVABILITY_FIELDS, compare_artifact_hashes


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]


V1_APP_PROFILE_DIRS = [
    Path("applications/GL-CBAM-APP/v1"),
    Path("applications/GL-CSRD-APP/CSRD-Reporting-Platform/v1"),
    Path("applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v1"),
]


def contract_checks(app_dirs: Iterable[Path] = V1_APP_PROFILE_DIRS) -> list[CheckResult]:
    results: list[CheckResult] = []
    for app_dir in app_dirs:
        pack_result = validate_v1_pack(app_dir / "pack.yaml")
        pipe_result = validate_v1_pipeline(app_dir / "gl.yaml")
        results.append(
            CheckResult(
                name=f"{app_dir.as_posix()} pack contract",
                ok=pack_result.ok,
                details=pack_result.errors,
            )
        )
        results.append(
            CheckResult(
                name=f"{app_dir.as_posix()} pipeline contract",
                ok=pipe_result.ok,
                details=pipe_result.errors,
            )
        )
    return results


def signed_pack_enforcement_checks(app_dirs: Iterable[Path] = V1_APP_PROFILE_DIRS) -> list[CheckResult]:
    results: list[CheckResult] = []
    for app_dir in app_dirs:
        pack_file = app_dir / "pack.yaml"
        result = validate_v1_pack(pack_file)
        findings: list[str] = []
        ok = result.ok
        if not ok:
            findings.extend(result.errors)
        else:
            is_verified, signature_info = verify_pack_signature(app_dir, app_dir / "pack.sig")
            if not is_verified:
                ok = False
                findings.append(f"signature verification failed: {signature_info}")
            else:
                findings.append("signed pack cryptographically verified")
        results.append(
            CheckResult(
                name=f"{app_dir.as_posix()} signed-pack policy",
                ok=ok,
                details=findings,
            )
        )
    return results


def release_gate_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.extend(contract_checks())
    checks.extend(signed_pack_enforcement_checks())
    checks.extend(runtime_convention_checks())
    checks.extend(profile_full_backend_checks())
    checks.extend(docs_contract_checks())
    return checks


def runtime_convention_checks(
    app_dirs: Iterable[Path] = V1_APP_PROFILE_DIRS,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    expected_commands = {
        profile.app_id: profile.command_template for profile in V1_APP_PROFILES.values()
    }
    for app_dir in app_dirs:
        pipeline_result = validate_v1_pipeline(app_dir / "gl.yaml")
        details: list[str] = []
        ok = pipeline_result.ok
        if not ok:
            details.extend(pipeline_result.errors)
        else:
            # reload file to inspect command string cheaply
            import yaml

            with open(app_dir / "gl.yaml", "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            app_id = data.get("app_id", "")
            got_command = (data.get("runtime_conventions", {}) or {}).get("command", "")
            expected = expected_commands.get(app_id, "")
            if expected and got_command != expected:
                ok = False
                details.append(f"runtime command mismatch: expected '{expected}', got '{got_command}'")
        checks.append(
            CheckResult(
                name=f"{app_dir.as_posix()} runtime conventions",
                ok=ok,
                details=details,
            )
        )
    return checks


def docs_contract_checks() -> list[CheckResult]:
    required_docs = [
        Path("docs/v1/CONTRACTS.md"),
        Path("docs/v1/MIGRATION_GUIDE.md"),
        Path("docs/v1/DOCS_CONTRACT.md"),
        Path("docs/v1/QUICKSTART.md"),
        Path("docs/v1/RUNBOOK_TEMPLATE.md"),
        Path("docs/v1/STANDARDS.md"),
        Path("docs/v1/SECURITY_POLICY_BASELINE.md"),
        Path("docs/v1/RELEASE_CHECKLIST.md"),
        Path("docs/v1/PACK_LIFECYCLE_PLAYBOOK.md"),
        Path("docs/v1/RELEASE_CANDIDATE_PROCESS.md"),
        Path("docs/v1/RELEASE_NOTES_v1.0.md"),
        Path("docs/v1/ROADMAP_v1_1.md"),
        Path("docs/v1/UAT_RESULTS.md"),
        Path("docs/v1/FULL_BACKEND_ACCEPTANCE.md"),
        Path("docs/v1/AUDIT_STATUS_MATRIX.md"),
        Path("docs/v1/RC_SOAK_LOG.md"),
        Path("docs/v1/GO_NO_GO_RECORD.md"),
        Path("docs/v1/MILESTONE_CALENDAR.md"),
        Path("docs/v1/DEPENDENCY_GRAPH.md"),
        Path("docs/v1/apps/GL-CBAM-APP_RUNBOOK.md"),
        Path("docs/v1/apps/GL-CSRD-APP_RUNBOOK.md"),
        Path("docs/v1/apps/GL-VCCI-Carbon-APP_RUNBOOK.md"),
    ]
    checks: list[CheckResult] = []
    for doc in required_docs:
        checks.append(
            CheckResult(
                name=f"docs contract: {doc.as_posix()}",
                ok=doc.exists(),
                details=[] if doc.exists() else ["missing required v1 doc"],
            )
        )
    return checks


def profile_smoke_checks(output_root: Path | None = None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if output_root is None:
        output_root = Path(tempfile.mkdtemp(prefix="greenlang_v1_smoke_"))
    output_root.mkdir(parents=True, exist_ok=True)

    for profile in V1_APP_PROFILES.values():
        with open(profile.v1_dir / "gl.yaml", "r", encoding="utf-8") as handle:
            pipeline_contract = yaml.safe_load(handle) or {}
        required_artifacts: list[str] = (
            (pipeline_contract.get("runtime_conventions", {}) or {}).get("artifact_contract", [])
        )

        input_path = profile.v1_dir / "smoke_input.json"
        run1_dir = output_root / profile.key / "run1"
        run2_dir = output_root / profile.key / "run2"
        generate_profile_smoke_artifacts(profile, input_path, run1_dir)
        generate_profile_smoke_artifacts(profile, input_path, run2_dir)

        missing: list[str] = []
        for artifact in required_artifacts:
            if not (run1_dir / artifact).exists():
                missing.append(f"run1 missing artifact: {artifact}")
            if not (run2_dir / artifact).exists():
                missing.append(f"run2 missing artifact: {artifact}")

        determinism = compare_artifact_hashes(run1_dir, run2_dir)
        if not determinism.same_fileset:
            missing.append("determinism failed: run filesets differ")
        if determinism.diff_count != 0:
            missing.append(f"determinism failed: differing artifacts {determinism.diffs}")

        obs_file = run1_dir / "audit" / "observability_event.json"
        if not obs_file.exists():
            missing.append("missing observability_event.json")
        else:
            with open(obs_file, "r", encoding="utf-8") as handle:
                event = json.load(handle)
            for field in REQUIRED_OBSERVABILITY_FIELDS:
                if field not in event:
                    missing.append(f"observability field missing: {field}")
                elif event[field] in (None, ""):
                    missing.append(f"observability field empty: {field}")

        checks.append(
            CheckResult(
                name=f"{profile.v1_dir.as_posix()} profile smoke",
                ok=not missing,
                details=missing,
            )
        )
    return checks


def profile_full_backend_checks(output_root: Path | None = None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if output_root is None:
        output_root = Path(tempfile.mkdtemp(prefix="greenlang_v1_full_backend_"))
    output_root.mkdir(parents=True, exist_ok=True)

    for profile in V1_APP_PROFILES.values():
        with open(profile.v1_dir / "gl.yaml", "r", encoding="utf-8") as handle:
            pipeline_contract = yaml.safe_load(handle) or {}
        required_artifacts: list[str] = (
            (pipeline_contract.get("runtime_conventions", {}) or {}).get("artifact_contract", [])
        )
        input_path = get_default_backend_input(profile.key)

        run1_dir = output_root / profile.key / "run1"
        run2_dir = output_root / profile.key / "run2"
        result_1 = run_profile_backend(
            profile=profile,
            input_path=input_path,
            output_dir=run1_dir,
            strict=True,
            allow_fallback=False,
        )
        result_2 = run_profile_backend(
            profile=profile,
            input_path=input_path,
            output_dir=run2_dir,
            strict=True,
            allow_fallback=False,
        )
        details: list[str] = []
        ok = result_1.success and result_2.success
        if not result_1.success:
            details.extend(result_1.errors)
        if not result_2.success:
            details.extend(result_2.errors)
        if result_1.fallback_used or result_2.fallback_used:
            ok = False
            details.append("full backend lane must not use fallback adapter")
        if not result_1.native_backend_used or not result_2.native_backend_used:
            ok = False
            details.append("full backend lane requires native backend execution")

        for artifact in required_artifacts:
            if not (run1_dir / artifact).exists():
                ok = False
                details.append(f"run1 missing artifact: {artifact}")
            if not (run2_dir / artifact).exists():
                ok = False
                details.append(f"run2 missing artifact: {artifact}")

        compare_root_1 = output_root / profile.key / "compare_run1"
        compare_root_2 = output_root / profile.key / "compare_run2"
        for root in (compare_root_1, compare_root_2):
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        compare_artifacts = list(dict.fromkeys(required_artifacts + ["audit/observability_event.json"]))
        for artifact in compare_artifacts:
            source_1 = run1_dir / artifact
            source_2 = run2_dir / artifact
            if source_1.exists():
                target_1 = compare_root_1 / artifact
                target_1.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_1, target_1)
            if source_2.exists():
                target_2 = compare_root_2 / artifact
                target_2.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_2, target_2)

        determinism = compare_artifact_hashes(compare_root_1, compare_root_2)
        if not determinism.same_fileset:
            ok = False
            details.append("determinism failed: filesets differ")
        if determinism.diff_count != 0:
            ok = False
            details.append(f"determinism failed: artifact hash diffs {determinism.diffs}")

        for run_dir, run_name in ((run1_dir, "run1"), (run2_dir, "run2")):
            obs_file = run_dir / "audit" / "observability_event.json"
            if not obs_file.exists():
                ok = False
                details.append(f"{run_name} missing observability_event.json")
                continue
            with open(obs_file, "r", encoding="utf-8") as handle:
                event = json.load(handle)
            for field in REQUIRED_OBSERVABILITY_FIELDS:
                if field not in event:
                    ok = False
                    details.append(f"{run_name} observability field missing: {field}")
                elif event[field] in (None, ""):
                    ok = False
                    details.append(f"{run_name} observability field empty: {field}")

        checks.append(
            CheckResult(
                name=f"{profile.v1_dir.as_posix()} full backend",
                ok=ok,
                details=details,
            )
        )
    return checks

