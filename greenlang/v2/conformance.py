# -*- coding: utf-8 -*-
"""Conformance checks for GreenLang v2 portfolio scale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .agent_lifecycle import validate_agent_registry
from .contracts import validate_v2_pack, validate_v2_pipeline
from .profiles import V2_APP_PROFILES
from .reliability_runtime import validate_connector_registry


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
        Path("docs/v2/GO_NO_GO_RECORD.md"),
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


def release_gate_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.extend(contract_checks())
    checks.extend(runtime_convention_checks())
    checks.extend(agent_lifecycle_checks())
    checks.extend(connector_reliability_checks())
    checks.extend(docs_contract_checks())
    return checks

