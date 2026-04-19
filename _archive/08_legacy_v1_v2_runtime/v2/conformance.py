# -*- coding: utf-8 -*-
"""Conformance checks for GreenLang v2 portfolio scale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import shutil
import tempfile

from .agent_lifecycle import validate_agent_registry
from .backends import run_v2_profile_backend
from .contracts import validate_v2_pack, validate_v2_pipeline
from .pack_tiers import validate_tier_registry
from .profiles import V2_APP_PROFILES
from .reliability_runtime import validate_connector_registry
from .standards import REQUIRED_AUDIT_ARTIFACTS, compare_artifact_hashes


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]


V2_APP_PROFILE_DIRS = [profile.v2_dir for profile in V2_APP_PROFILES.values()]


def contract_checks(app_dirs: Iterable[Path] = V2_APP_PROFILE_DIRS) -> list[CheckResult]:
    results: list[CheckResult] = []
    for app_dir in app_dirs:
        pack_result = validate_v2_pack(app_dir / "pack.yaml")
        pipe_result = validate_v2_pipeline(app_dir / "gl.yaml")
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


def runtime_convention_checks(app_dirs: Iterable[Path] = V2_APP_PROFILE_DIRS) -> list[CheckResult]:
    checks: list[CheckResult] = []
    expected_commands = {
        profile.app_id: profile.command_template for profile in V2_APP_PROFILES.values()
    }
    for app_dir in app_dirs:
        finding = validate_v2_pipeline(app_dir / "gl.yaml")
        ok = finding.ok
        details: list[str] = []
        if not ok:
            details.extend(finding.errors)
        else:
            import yaml

            with open(app_dir / "gl.yaml", "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            app_id = data.get("app_id", "")
            got = (data.get("runtime_conventions", {}) or {}).get("command", "")
            expected = expected_commands.get(app_id, "")
            if expected and got != expected:
                ok = False
                details.append(f"runtime command mismatch: expected '{expected}', got '{got}'")
        checks.append(
            CheckResult(
                name=f"{app_dir.as_posix()} runtime conventions",
                ok=ok,
                details=details,
            )
        )
    return checks


def runtime_execution_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    regulated_profiles = ["eudr", "ghg", "iso14064", "sb253", "taxonomy"]
    for key in regulated_profiles:
        profile = V2_APP_PROFILES[key]
        smoke_input = profile.v2_dir / "smoke_input.json"
        details: list[str] = []
        ok = True
        if not smoke_input.exists():
            ok = False
            details.append(f"missing smoke input: {smoke_input.as_posix()}")
        else:
            output_root = Path(tempfile.mkdtemp(prefix=f"greenlang_v2_exec_{key}_"))
            result = run_v2_profile_backend(
                profile_key=key,
                input_path=smoke_input,
                output_dir=output_root,
                strict=True,
                allow_fallback=False,
            )
            if not result.success:
                ok = False
                details.extend(result.errors)
            if result.exit_code != 0:
                ok = False
                details.append(f"expected exit_code=0, got {result.exit_code}")
            if result.fallback_used:
                ok = False
                details.append("release profile cannot use fallback backend")
            for artifact in result.artifacts:
                if not (output_root / artifact).exists():
                    ok = False
                    details.append(f"missing artifact: {artifact}")
        checks.append(
            CheckResult(
                name=f"runtime execution: {profile.app_id}",
                ok=ok,
                details=details,
            )
        )
    return checks


def docs_contract_checks() -> list[CheckResult]:
    required_docs = [
        Path("docs/v2/PHASE3_CHARTER.md"),
        Path("docs/v2/PHASE0_CHARTER.md"),
        Path("docs/v2/V2_SCOPE_AND_DEFERRED.md"),
        Path("docs/v2/RFC_PROCESS.md"),
        Path("docs/v2/OWNERSHIP_MATRIX.md"),
        Path("docs/v2/COMPATIBILITY_MATRIX.md"),
        Path("docs/v2/PACK_TIERING_POLICY.md"),
        Path("docs/v2/AGENT_LIFECYCLE_POLICY.md"),
        Path("docs/v2/RELEASE_TRAINS.md"),
        Path("docs/v2/ONCALL_AND_SLOS.md"),
        Path("docs/v2/PRIORITIZED_CONNECTORS.md"),
        Path("docs/v2/PLATFORM_HANDBOOK.md"),
        Path("docs/v2/MIGRATION_PLAYBOOKS.md"),
        Path("docs/v2/GO_NO_GO_RECORD.md"),
        Path("docs/v2/UAT_RESULTS.md"),
        Path("docs/v2/RC_SOAK_LOG.md"),
        Path("docs/v2/RELEASE_TRAIN_CYCLE_LOG.md"),
        Path("docs/v2/IMMUTABLE_EVIDENCE_MANIFEST.json"),
        Path("docs/v2/PHASE1_GATE_STATUS.json"),
        Path("docs/v2/PHASE4_GATE_STATUS.json"),
        Path("docs/v2/PHASE5_GATE_STATUS.json"),
        Path("docs/v2/PHASE6_GATE_STATUS.json"),
        Path("docs/runbooks/V2_CONNECTOR_ALERT_MATRIX.md"),
    ]
    checks: list[CheckResult] = []
    for doc in required_docs:
        checks.append(
            CheckResult(
                name=f"docs contract: {doc.as_posix()}",
                ok=doc.exists(),
                details=[] if doc.exists() else ["missing required v2 doc"],
            )
        )
    return checks


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def immutable_evidence_checks() -> list[CheckResult]:
    manifest_path = Path("docs/v2/IMMUTABLE_EVIDENCE_MANIFEST.json")
    if not manifest_path.exists():
        return [
            CheckResult(
                name="immutable evidence manifest",
                ok=False,
                details=["missing docs/v2/IMMUTABLE_EVIDENCE_MANIFEST.json"],
            )
        ]

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            CheckResult(
                name="immutable evidence manifest",
                ok=False,
                details=[f"manifest parse error: {exc}"],
            )
        ]

    hashes = payload.get("artifact_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        return [
            CheckResult(
                name="immutable evidence manifest",
                ok=False,
                details=["artifact_hashes must be a non-empty object"],
            )
        ]

    ok = True
    details: list[str] = []
    for rel_path, expected_hash in hashes.items():
        artifact = Path(rel_path)
        if not artifact.exists():
            ok = False
            details.append(f"missing artifact: {rel_path}")
            continue
        actual_hash = _file_sha256(artifact)
        if actual_hash != expected_hash:
            ok = False
            details.append(f"hash mismatch: {rel_path}")
    return [CheckResult(name="immutable evidence manifest", ok=ok, details=details)]


def agent_lifecycle_checks() -> list[CheckResult]:
    result = validate_agent_registry(Path("greenlang/agents/v2_agent_registry.yaml"))
    return [
        CheckResult(
            name="agent lifecycle registry",
            ok=result.ok,
            details=result.errors,
        )
    ]


def connector_reliability_checks() -> list[CheckResult]:
    result = validate_connector_registry(Path("applications/connectors/v2_connector_registry.yaml"))
    return [
        CheckResult(
            name="connector reliability registry",
            ok=result.ok,
            details=result.errors,
        )
    ]


def tier_registry_checks() -> list[CheckResult]:
    registry_path = Path("greenlang/ecosystem/packs/v2_tier_registry.yaml")
    if not registry_path.exists():
        return [
            CheckResult(
                name="pack tier registry",
                ok=False,
                details=["missing greenlang/ecosystem/packs/v2_tier_registry.yaml"],
            )
        ]
    errors = validate_tier_registry(registry_path)
    return [
        CheckResult(
            name="pack tier registry",
            ok=not errors,
            details=errors,
        )
    ]


def policy_bundle_checks() -> list[CheckResult]:
    required_policy_bundles = [
        Path("greenlang/governance/policy/bundles/v2_authz.rego"),
        Path("greenlang/governance/policy/bundles/v2_pack_trust.rego"),
        Path("greenlang/governance/policy/bundles/v2_egress_controls.rego"),
        Path("greenlang/governance/policy/bundles/v2_data_controls.rego"),
        Path("greenlang/governance/policy/bundles/v2_pack_tier_policy.rego"),
    ]
    checks: list[CheckResult] = []
    for bundle in required_policy_bundles:
        checks.append(
            CheckResult(
                name=f"policy bundle: {bundle.as_posix()}",
                ok=bundle.exists(),
                details=[] if bundle.exists() else ["missing required policy bundle"],
            )
        )
    return checks


def _determinism_input_for_profile(profile_key: str, target: Path) -> None:
    payloads = {
        "eudr": {"suppliers": [{"id": "S1", "risk": "low"}]},
        "ghg": {"activities": [{"quantity": 10, "emission_factor": 1.5}]},
        "iso14064": {"controls": [{"id": "C1", "passed": True}]},
        "sb253": {"activities": [{"quantity": 10, "emission_factor": 1.5}]},
        "taxonomy": {"activities": [{"quantity": 10, "emission_factor": 1.5}]},
    }
    target.write_text(json.dumps(payloads[profile_key], indent=2), encoding="utf-8")


def determinism_contract_checks(output_root: Path | None = None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if output_root is None:
        output_root = Path(tempfile.mkdtemp(prefix="greenlang_v2_determinism_"))
    output_root.mkdir(parents=True, exist_ok=True)

    regulated_profiles = ["eudr", "ghg", "iso14064", "sb253", "taxonomy"]
    for profile_key in regulated_profiles:
        profile = V2_APP_PROFILES[profile_key]
        run1 = output_root / profile_key / "run1"
        run2 = output_root / profile_key / "run2"
        input_path = output_root / profile_key / "input.json"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        _determinism_input_for_profile(profile_key, input_path)

        result_1 = run_v2_profile_backend(
            profile_key=profile_key,
            input_path=input_path,
            output_dir=run1,
            strict=True,
            allow_fallback=False,
        )
        result_2 = run_v2_profile_backend(
            profile_key=profile_key,
            input_path=input_path,
            output_dir=run2,
            strict=True,
            allow_fallback=False,
        )
        ok = result_1.success and result_2.success and result_1.exit_code == 0 and result_2.exit_code == 0
        details: list[str] = []
        if not result_1.success:
            details.extend(result_1.errors)
        if not result_2.success:
            details.extend(result_2.errors)
        if result_1.exit_code != 0:
            details.append(f"run1 exit_code expected 0, got {result_1.exit_code}")
        if result_2.exit_code != 0:
            details.append(f"run2 exit_code expected 0, got {result_2.exit_code}")

        gl_data = validate_v2_pipeline(profile.v2_dir / "gl.yaml")
        if not gl_data.ok:
            ok = False
            details.extend(gl_data.errors)
            artifact_contract: list[str] = []
        else:
            import yaml
            loaded = yaml.safe_load((profile.v2_dir / "gl.yaml").read_text(encoding="utf-8")) or {}
            artifact_contract = (loaded.get("runtime_conventions", {}) or {}).get("artifact_contract", [])

        compare_root_1 = output_root / profile_key / "compare_run1"
        compare_root_2 = output_root / profile_key / "compare_run2"
        for root in (compare_root_1, compare_root_2):
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        compare_artifacts = list(dict.fromkeys(REQUIRED_AUDIT_ARTIFACTS + artifact_contract))
        for artifact in compare_artifacts:
            source_1 = run1 / artifact
            source_2 = run2 / artifact
            if not source_1.exists():
                ok = False
                details.append(f"run1 missing artifact: {artifact}")
            if not source_2.exists():
                ok = False
                details.append(f"run2 missing artifact: {artifact}")
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
            details.append(f"determinism failed: hash diffs {determinism.diffs}")
        checks.append(
            CheckResult(
                name=f"determinism contract: {profile.app_id}",
                ok=ok,
                details=details,
            )
        )
    return checks


def release_gate_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.extend(contract_checks())
    checks.extend(runtime_convention_checks())
    checks.extend(runtime_execution_checks())
    checks.extend(agent_lifecycle_checks())
    checks.extend(connector_reliability_checks())
    checks.extend(tier_registry_checks())
    checks.extend(policy_bundle_checks())
    checks.extend(determinism_contract_checks())
    checks.extend(docs_contract_checks())
    checks.extend(immutable_evidence_checks())
    return checks

