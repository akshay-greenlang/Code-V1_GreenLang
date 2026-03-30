#!/usr/bin/env python3
"""GL-CSRD-APP v2 runtime backend.

Wraps the CSRD pipeline with V2 contract compliance, producing
esrs_report.json and audit bundle artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL-CSRD-APP v2 runtime backend")
    parser.add_argument("--input", required=True, help="Path to ESG data CSV/JSON")
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
        from greenlang.v1.backends import run_csrd_backend
    except ImportError:
        _build_fallback_report(input_path, output_dir)
        _write_v2_audit_bundle(output_dir, ["esrs_report.json"], "degraded", ["v1 backend import failed; fallback used"])
        return 0

    result = run_csrd_backend(
        input_path=input_path,
        output_dir=output_dir,
        strict=True,
        allow_fallback=True,
    )

    if not (output_dir / "esrs_report.json").exists():
        _build_fallback_report(input_path, output_dir)

    _write_v2_audit_bundle(
        output_dir,
        result.artifacts if result.artifacts else ["esrs_report.json"],
        "ok" if result.success else "failed",
        result.warnings,
    )
    return 0 if result.success else 1


def _build_fallback_report(input_path: Path, output_dir: Path) -> None:
    import csv
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

    report = {
        "app_id": "GL-CSRD-APP",
        "pipeline_id": "csrd-esrs-v2",
        "report_type": "esrs",
        "records_processed": rows,
        "status": "generated",
    }
    (output_dir / "esrs_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
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
        "app_id": "GL-CSRD-APP",
        "pipeline_id": "csrd-esrs-v2",
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
