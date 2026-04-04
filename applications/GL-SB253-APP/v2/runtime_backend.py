#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL-SB253-APP v2 runtime backend")
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

    activities = payload.get("activities", [])
    if activities is None:
        activities = []
    if not isinstance(activities, list):
        raise ValueError("activities must be a list")

    total_emissions = 0.0
    records = 0
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
        "jurisdiction": "US-CA",
    }
    (output_dir / "sb253_disclosure.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
