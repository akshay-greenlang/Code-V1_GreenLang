#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL-ISO14064-APP v2 runtime backend")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output directory")
    return parser.parse_args()


def main() -> int:
    args = _parse()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON root must be an object")

    controls = payload.get("controls", [])
    if controls is None:
        controls = []
    if not isinstance(controls, list):
        raise ValueError("controls must be a list")

    control_count = len(controls)
    passed_count = len(
        [item for item in controls if isinstance(item, dict) and bool(item.get("passed", False))]
    )
    failed_count = max(0, control_count - passed_count)
    status = "blocked" if bool(payload.get("policy_block", False)) else "ok"
    report = {
        "app_id": "GL-ISO14064-APP",
        "pipeline_id": "iso14064-verification-v2",
        "status": status,
        "controls_checked": control_count,
        "controls_passed": passed_count,
        "controls_failed": failed_count,
        "conformance_percent": round((passed_count / control_count) * 100.0, 2) if control_count else 0.0,
    }
    (output_dir / "iso14064_verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
