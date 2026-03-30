#!/usr/bin/env python3
"""GL-VCCI-Carbon-APP v2 runtime backend.

Wraps the VCCI pipeline with V2 contract compliance, producing
scope3_inventory.json and audit bundle artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL-VCCI-Carbon-APP v2 runtime backend")
    parser.add_argument("--input", required=True, help="Path to input CSV/JSON")
    parser.add_argument("--output", required=True, help="Path to output directory")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _parse()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]

    try:
        sys.path.insert(0, str(repo_root))
        from greenlang.v1.backends import run_vcci_backend
    except ImportError:
        _build_fallback_inventory(input_path, output_dir)
        _write_v2_audit_bundle(output_dir, ["scope3_inventory.json"], "degraded", ["v1 backend import failed; fallback used"])
        return 0

    result = run_vcci_backend(
        input_path=input_path,
        output_dir=output_dir,
        strict=True,
        allow_fallback=True,
    )

    if not (output_dir / "scope3_inventory.json").exists():
        _build_fallback_inventory(input_path, output_dir)

    _write_v2_audit_bundle(
        output_dir,
        result.artifacts if result.artifacts else ["scope3_inventory.json"],
        "ok" if result.success else "failed",
        result.warnings,
    )
    return 0 if result.success else 1


def _build_fallback_inventory(input_path: Path, output_dir: Path) -> None:
    import csv
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
        "pipeline_id": "vcci-scope3-v2",
        "inventory_type": "scope3",
        "records_processed": rows,
        "total_emissions_kgco2e": round(total_emissions, 6),
    }
    (output_dir / "scope3_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8",
    )


def _write_v2_audit_bundle(
    output_dir: Path,
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
        "app_id": "GL-VCCI-Carbon-APP",
        "pipeline_id": "vcci-scope3-v2",
        "status": status,
        "execution_mode": "native-v2",
        "artifacts": artifacts,
        "warnings": warnings or [],
    }
    (audit_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    (audit_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
