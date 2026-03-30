#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL-EUDR-APP v2 runtime backend")
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
    suppliers = payload.get("suppliers", [])
    if suppliers is None:
        suppliers = []
    if not isinstance(suppliers, list):
        raise ValueError("suppliers must be a list")

    records_processed = len(suppliers)
    high_risk_suppliers = len(
        [item for item in suppliers if isinstance(item, dict) and item.get("risk") == "high"]
    )
    status = "blocked" if bool(payload.get("policy_block", False)) else "ok"

    statement = {
        "app_id": "GL-EUDR-APP",
        "pipeline_id": "eudr-due-diligence-v2",
        "status": status,
        "records_processed": records_processed,
        "high_risk_suppliers": high_risk_suppliers,
        "risk_distribution": {
            "high": high_risk_suppliers,
            "non_high": max(0, records_processed - high_risk_suppliers),
        },
    }
    (output_dir / "due_diligence_statement.json").write_text(
        json.dumps(statement, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
