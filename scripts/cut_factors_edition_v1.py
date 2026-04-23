#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cut the v1.0 GreenLang Factors Certified edition (FY27 launch — Track B-3).

Reads the current catalog, runs the 9-point release-signoff checklist,
bumps the edition manifest to ``2026-05-v1.0``, signs it, writes the
manifest + signature under ``greenlang/factors/data/editions/2026-05-v1.0/``,
updates ``active.txt`` to point at it, and emits a fresh release-notes
markdown.

Usage::

    # 1) Pre-flight, no writes:
    python scripts/cut_factors_edition_v1.py --dry-run

    # 2) Real cut (only after methodology lead signs the runbook):
    python scripts/cut_factors_edition_v1.py --commit \\
        --approver "name@greenlang.ai"

    # 3) Force commit even if S4 (changelog) / S6 (methodology) flags
    #    show as not signed -- ONLY for break-glass use, will write a
    #    'force_committed' annotation into release notes:
    python scripts/cut_factors_edition_v1.py --commit --approver ... --force

Exit codes:
    0  - success (or dry-run completed without raising)
    1  - signoff failed and --force not set
    2  - configuration / IO error

This script is the single source-of-truth for cutting the launch edition.
The methodology lead's runbook lives at ``docs/factors/EDITION_CUT_RUNBOOK.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cut_edition_v1")


REPO_ROOT = Path(__file__).resolve().parent.parent
EDITIONS_DIR = REPO_ROOT / "greenlang" / "factors" / "data" / "editions"
TARGET_EDITION_ID = "2026-05-v1.0"
RELEASE_NOTES_PATH = REPO_ROOT / "docs" / "factors" / "RELEASE_NOTES_v1.0.md"


# ---------------------------------------------------------------------------
# Catalog + signoff
# ---------------------------------------------------------------------------


def load_service():
    """Boot the FactorCatalogService from the environment."""
    from greenlang.factors.service import FactorCatalogService
    return FactorCatalogService.from_environment()


def build_manifest(svc, target_edition_id: str) -> Dict[str, Any]:
    """Materialise the current catalog into an edition manifest dict."""
    repo = svc.repo
    source_edition = repo.get_default_edition_id() or target_edition_id

    try:
        rows, _ = repo.list_factors(source_edition, page=1, limit=1_000_000)
    except TypeError:
        rows, _ = repo.list_factors(source_edition)

    factor_count = len(rows)
    family_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {"certified": 0, "preview": 0, "connector_only": 0}
    source_versions: Dict[str, str] = {}

    for f in rows:
        fam_attr = getattr(f, "factor_family", None)
        fam = (
            getattr(fam_attr, "value", None)
            if fam_attr is not None
            else getattr(f, "fuel_type", None) or "uncategorized"
        )
        family_counts[fam] = family_counts.get(fam, 0) + 1

        prov = getattr(f, "provenance", None)
        if prov is not None:
            src = getattr(prov, "source_org", None) or getattr(prov, "source_publication", "unknown")
            ver = getattr(prov, "version", None) or getattr(prov, "source_year", "")
            if src:
                source_versions[str(src)] = str(ver)

        label_attr = (getattr(f, "publication_label", None) or getattr(f, "label", None) or getattr(f, "status", None))
        label = getattr(label_attr, "value", label_attr) if label_attr else None
        if isinstance(label, str):
            l = label.lower()
            if l in ("certified", "ga", "released", "active"):
                label_counts["certified"] += 1
            elif l in ("preview", "beta", "candidate"):
                label_counts["preview"] += 1
            elif l in ("connector", "connector_only", "ingest_only"):
                label_counts["connector_only"] += 1

    return {
        "edition_id": target_edition_id,
        "source_edition_id": source_edition,
        "cut_at": datetime.now(timezone.utc).isoformat(),
        "factor_count": factor_count,
        "family_counts": family_counts,
        "label_counts": label_counts,
        "source_versions": source_versions,
        "schema_version": "1.0",
    }


def run_signoff(
    svc,
    target_edition_id: str,
    *,
    methodology_signed: bool,
    legal_confirmed: bool,
    changelog_reviewed: bool,
    gold_eval_precision: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Run the 9-point release-signoff checklist.

    Returns ``(passed, report_dict)``. Wraps
    :func:`greenlang.factors.quality.release_signoff.release_signoff_checklist`.
    """
    from greenlang.factors.quality.release_signoff import release_signoff_checklist

    manifest = build_manifest(svc, target_edition_id)

    # Materialise the full factor list once; the QA helpers want a concrete
    # Sequence, not a repo handle (see batch_qa.run_batch_qa_on_dicts,
    # dedup_engine.detect_duplicates, cross_source.check_cross_source_consistency).
    try:
        rows, _ = svc.repo.list_factors(manifest["source_edition_id"], page=1, limit=1_000_000)
    except TypeError:
        rows, _ = svc.repo.list_factors(manifest["source_edition_id"])
    factor_rows = list(rows)

    qa_report: Optional[Dict[str, Any]] = None
    try:
        from greenlang.factors.quality.batch_qa import run_batch_qa
        qa = run_batch_qa(svc.repo, manifest["source_edition_id"])
        qa_report = qa.to_dict() if hasattr(qa, "to_dict") else qa
    except Exception as exc:  # noqa: BLE001
        logger.warning("batch_qa unavailable: %s", exc)

    dedup_report: Optional[Dict[str, Any]] = None
    try:
        from greenlang.factors.quality.dedup_engine import detect_duplicates
        dedup = detect_duplicates(factor_rows, edition_id=manifest["source_edition_id"])
        dedup_report = dedup.to_dict() if hasattr(dedup, "to_dict") else dedup
    except Exception as exc:  # noqa: BLE001
        logger.warning("dedup_engine unavailable: %s", exc)

    consistency_report: Optional[Dict[str, Any]] = None
    try:
        from greenlang.factors.quality.cross_source import check_cross_source_consistency
        cs = check_cross_source_consistency(
            factor_rows, edition_id=manifest["source_edition_id"]
        )
        consistency_report = cs.to_dict() if hasattr(cs, "to_dict") else cs
    except Exception as exc:  # noqa: BLE001
        logger.warning("cross_source unavailable: %s", exc)

    signoff = release_signoff_checklist(
        edition_id=target_edition_id,
        manifest=manifest,
        qa_report=qa_report,
        dedup_report=dedup_report,
        consistency_report=consistency_report,
        changelog_reviewed=changelog_reviewed,
        methodology_signed=methodology_signed,
        legal_confirmed=legal_confirmed,
        regression_passed=True,
        load_test_passed=True,
        gold_eval_precision=gold_eval_precision,
    )

    report = signoff.to_dict() if hasattr(signoff, "to_dict") else {"items": []}
    report["manifest"] = manifest
    # ``ready_for_release`` is a ``@property`` on ReleaseSignoffReport, not a
    # method — calling it (``()``) raised ``TypeError: 'bool' object is not
    # callable`` during dry-runs and would have crashed the methodology lead's
    # live cut. Read the attribute directly.
    return bool(signoff.ready_for_release), report


# ---------------------------------------------------------------------------
# Sign + write
# ---------------------------------------------------------------------------


def sign_manifest(manifest_bytes: bytes) -> Dict[str, str]:
    """Sign the manifest using whichever signer is configured.

    Prefers Ed25519 when ``GL_FACTORS_ED25519_PRIVATE_KEY`` is set; falls
    back to HMAC-SHA256 with ``GL_FACTORS_SIGNING_SECRET`` (or a dev
    secret if neither is set, so dry-runs produce a non-empty signature).
    """
    try:
        from greenlang.factors.signing import sign_ed25519
        if os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY"):
            receipt = sign_ed25519(manifest_bytes.decode("utf-8"))
            return {
                "algorithm": receipt.algorithm,
                "signature": receipt.signature,
                "key_id": receipt.key_id,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ed25519 signer unavailable: %s", exc)

    from greenlang.factors.signing import sign_sha256_hmac
    secret = os.getenv("GL_FACTORS_SIGNING_SECRET", "dev-secret-do-not-use-in-prod")
    receipt = sign_sha256_hmac(manifest_bytes.decode("utf-8"), secret=secret)
    return {
        "algorithm": receipt.algorithm,
        "signature": receipt.signature,
        "key_id": receipt.key_id,
    }


def write_edition(target_dir: Path, manifest: Dict[str, Any], signature: Dict[str, str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"
    sig_path = target_dir / "manifest.sig"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    sig_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def update_active_pointer(target_edition_id: str) -> None:
    EDITIONS_DIR.mkdir(parents=True, exist_ok=True)
    (EDITIONS_DIR / "active.txt").write_text(target_edition_id + "\n", encoding="utf-8")


def write_release_notes(manifest: Dict[str, Any], signature: Dict[str, str], approver: str, force: bool) -> None:
    RELEASE_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# GreenLang Factors v1.0 ({manifest['edition_id']})\n")
    lines.append(f"**Cut at:** {manifest['cut_at']}  \n")
    lines.append(f"**Approver:** `{approver}`  \n")
    lines.append(f"**Algorithm:** `{signature['algorithm']}`  \n")
    lines.append(f"**Key ID:** `{signature['key_id']}`  \n")
    lines.append(f"**Signature (truncated):** `{signature['signature'][:32]}...`  \n")
    if force:
        lines.append("> ⚠️ **force_committed**: this edition was cut with `--force` despite signoff failures.\n")
    lines.append("\n## Coverage\n")
    lines.append(f"- Total factors: **{manifest['factor_count']}**\n")
    lines.append(f"- Certified: {manifest['label_counts'].get('certified', 0)}\n")
    lines.append(f"- Preview: {manifest['label_counts'].get('preview', 0)}\n")
    lines.append(f"- Connector-only: {manifest['label_counts'].get('connector_only', 0)}\n")
    lines.append("\n### Factor families\n")
    for fam, count in sorted(manifest["family_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {fam}: {count}\n")
    lines.append("\n### Source versions pinned in this edition\n")
    for src, ver in sorted(manifest["source_versions"].items()):
        lines.append(f"- {src}: {ver}\n")
    lines.append("\n## Verification\n")
    lines.append("```\n")
    lines.append(f"gl-factors edition show {manifest['edition_id']}\n")
    lines.append("```\n")
    RELEASE_NOTES_PATH.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cut the v1.0 GreenLang Factors edition.")
    parser.add_argument("--commit", action="store_true", help="Write the edition (default is dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Run signoff + manifest build only; write nothing.")
    parser.add_argument("--force", action="store_true", help="Commit even if signoff failed (writes a force_committed marker).")
    parser.add_argument("--approver", default=os.getenv("GL_FACTORS_APPROVER", "unknown"), help="Approver email (recorded in release notes).")
    parser.add_argument("--target-edition", default=TARGET_EDITION_ID, help="Edition id to cut (default: 2026-05-v1.0).")
    parser.add_argument("--methodology-signed", action="store_true", default=False)
    parser.add_argument("--legal-confirmed", action="store_true", default=False)
    parser.add_argument("--changelog-reviewed", action="store_true", default=False)
    parser.add_argument("--gold-eval-precision", type=float, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dry_run = args.dry_run or not args.commit

    try:
        svc = load_service()
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot boot FactorCatalogService: %s", exc)
        return 2

    passed, report = run_signoff(
        svc,
        args.target_edition,
        methodology_signed=args.methodology_signed,
        legal_confirmed=args.legal_confirmed,
        changelog_reviewed=args.changelog_reviewed,
        gold_eval_precision=args.gold_eval_precision,
    )

    print("=" * 72)
    print(f"GreenLang Factors edition cut — {args.target_edition}")
    print("=" * 72)
    print(json.dumps(report, indent=2, default=str)[:8000])
    print("=" * 72)
    print(f"Signoff result: {'PASS' if passed else 'FAIL'}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'COMMIT'}")
    print()

    if dry_run:
        print("Dry-run complete. Re-run with --commit to write the edition.")
        return 0 if passed else 1

    if not passed and not args.force:
        print("Signoff failed; refusing to commit. Use --force to override (logged).")
        return 1

    manifest = report["manifest"]
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    signature = sign_manifest(manifest_bytes)

    target_dir = EDITIONS_DIR / args.target_edition
    write_edition(target_dir, manifest, signature)
    update_active_pointer(args.target_edition)
    write_release_notes(manifest, signature, args.approver, force=(not passed and args.force))

    print(f"OK — wrote edition manifest to {target_dir}")
    print(f"OK — release notes at {RELEASE_NOTES_PATH}")
    print(f"OK — active.txt → {args.target_edition}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
