#!/usr/bin/env python3
"""GL-CBAM-APP v2 runtime backend.

Wraps the CBAM pipeline with V2 contract compliance, producing
cbam_report.xml, report_summary.xlsx, and audit bundle artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL-CBAM-APP v2 runtime backend")
    parser.add_argument("--input", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output directory")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _parse()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]

    try:
        sys.path.insert(0, str(repo_root))
        from greenlang.v1.backends import run_cbam_backend
    except ImportError:
        sys.path.insert(0, str(repo_root / "cbam-pack-mvp" / "src"))
        try:
            from cbam_pack.pipeline import CBAMPipeline
            imports_path = repo_root / "cbam-pack-mvp" / "examples" / "sample_imports.csv"
            pipeline = CBAMPipeline(
                config_path=input_path,
                imports_path=imports_path,
                output_dir=output_dir,
                verbose=False,
                dry_run=False,
            )
            result = pipeline.run()
            _write_v2_audit_bundle(output_dir, result.artifacts, "ok" if result.success else "failed")
            return 0 if result.success else 1
        except Exception as exc:
            _write_minimal_output(output_dir, str(exc))
            return 1

    result = run_cbam_backend(
        input_path=input_path,
        output_dir=output_dir,
        strict=True,
        allow_fallback=True,
    )

    _write_v2_audit_bundle(
        output_dir,
        result.artifacts,
        "ok" if result.success else "failed",
        result.warnings,
    )
    return 0 if result.success else 1


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
        "app_id": "GL-CBAM-APP",
        "pipeline_id": "cbam-quarterly-v2",
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


def _write_minimal_output(output_dir: Path, error: str) -> None:
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "app_id": "GL-CBAM-APP",
        "pipeline_id": "cbam-quarterly-v2",
        "status": "failed",
        "execution_mode": "native-v2",
        "artifacts": [],
        "errors": [error],
    }
    (audit_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
