#!/usr/bin/env python3
"""Execute two-cycle local V2.2 release-train proof."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "v2" / "RELEASE_TRAIN_LOCAL_EVIDENCE.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(command: list[str], cwd: Path) -> dict:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=240,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout_tail": (proc.stdout or "")[-3000:],
            "stderr_tail": (proc.stderr or "")[-1500:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "ok": False,
            "stdout_tail": ((exc.stdout or "")[-3000:] if isinstance(exc.stdout, str) else ""),
            "stderr_tail": ((exc.stderr or "")[-1500:] if isinstance(exc.stderr, str) else ""),
            "timeout": True,
        }


def _cycle(name: str) -> dict:
    commands = [
        ([sys.executable, "-m", "greenlang.cli.main", "v2", "validate-contracts"], REPO_ROOT),
        ([sys.executable, "-m", "greenlang.cli.main", "v2", "runtime-checks"], REPO_ROOT),
        ([sys.executable, "-m", "greenlang.cli.main", "v2", "docs-check"], REPO_ROOT),
        ([sys.executable, "-m", "greenlang.cli.main", "v2", "agent-checks"], REPO_ROOT),
        ([sys.executable, "-m", "greenlang.cli.main", "v2", "connector-checks"], REPO_ROOT),
        ([sys.executable, "-m", "greenlang.cli.main", "v2", "gate"], REPO_ROOT),
        (["npm.cmd", "run", "lint"], REPO_ROOT / "frontend"),
        (["npm.cmd", "run", "test"], REPO_ROOT / "frontend"),
        (["npm.cmd", "run", "build"], REPO_ROOT / "frontend"),
    ]
    results = [_run(cmd, cwd) for cmd, cwd in commands]
    return {
        "cycle": name,
        "executed_at_utc": _now(),
        "all_passed": all(item["ok"] for item in results),
        "results": results,
    }


def main() -> int:
    evidence = {
        "generated_at_utc": _now(),
        "cycles": [_cycle("cycle-1"), _cycle("cycle-2")],
    }
    OUT_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if all(cycle["all_passed"] for cycle in evidence["cycles"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
