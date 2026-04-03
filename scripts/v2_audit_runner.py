#!/usr/bin/env python3
"""Reproducible audit evidence runner for V2.2."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "docs" / "v2" / "AUDIT_BASELINE_EVIDENCE.json"
REPORT_PATH = REPO_ROOT / "docs" / "v2" / "audit_delta_report.md"


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    ok: bool
    stdout_tail: str
    stderr_tail: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(name: str, command: list[str]) -> CommandResult:
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandResult(
        name=name,
        command=command,
        returncode=proc.returncode,
        ok=proc.returncode == 0,
        stdout_tail=(proc.stdout or "")[-4000:],
        stderr_tail=(proc.stderr or "")[-2000:],
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {"_error": f"missing: {path.as_posix()}"}
    return json.loads(path.read_text(encoding="utf-8"))


def _render_report(evidence: dict) -> str:
    failed = [c for c in evidence["command_results"] if not c["ok"]]
    lines = [
        "# GreenLang V2.2 Audit Delta Report",
        "",
        f"- Generated at: `{evidence['generated_at_utc']}`",
        "",
        f"- Commands passed: `{len(evidence['command_results']) - len(failed)}`",
        f"- Commands failed: `{len(failed)}`",
        "",
        "## Command Summary",
    ]
    for c in evidence["command_results"]:
        lines.append(f"- `{c['name']}`: {'OK' if c['ok'] else 'FAIL'} (rc={c['returncode']})")
    return "\n".join(lines) + "\n"


def main() -> int:
    commands = [
        ("v2_validate_contracts", [sys.executable, "-m", "greenlang.cli.main", "v2", "validate-contracts"]),
        ("v2_runtime_checks", [sys.executable, "-m", "greenlang.cli.main", "v2", "runtime-checks"]),
        ("v2_docs_check", [sys.executable, "-m", "greenlang.cli.main", "v2", "docs-check"]),
        ("v2_agent_checks", [sys.executable, "-m", "greenlang.cli.main", "v2", "agent-checks"]),
        ("v2_connector_checks", [sys.executable, "-m", "greenlang.cli.main", "v2", "connector-checks"]),
        ("v2_gate", [sys.executable, "-m", "greenlang.cli.main", "v2", "gate"]),
    ]
    results = [_run(name, cmd) for name, cmd in commands]
    evidence = {
        "generated_at_utc": _now(),
        "command_results": [asdict(item) for item in results],
        "json_files": {
            "mvp_release_gate.json": _read_json(REPO_ROOT / "mvp_release_gate.json"),
            "mvp_v1_v1_1_closure_report.json": _read_json(REPO_ROOT / "mvp_v1_v1_1_closure_report.json"),
            "v2_closure_report.json": _read_json(REPO_ROOT / "v2_closure_report.json"),
        },
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(_render_report(evidence), encoding="utf-8")
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
