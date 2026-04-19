# -*- coding: utf-8 -*-
"""CLI entry: python -m greenlang.factors.cli"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _cmd_inventory(args: argparse.Namespace) -> int:
    from greenlang.factors.inventory import write_coverage_matrix

    out = Path(args.out)
    data = write_coverage_matrix(out)
    print(json.dumps({"written": str(out), "totals": data.get("totals")}, indent=2))
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.edition_manifest import build_manifest_for_factors

    db = EmissionFactorDatabase(enable_cache=False)
    m = build_manifest_for_factors(
        args.edition_id,
        args.status,
        list(db.factors.values()),
        changelog=[args.message or f"Manifest for {args.edition_id}"],
    )
    out = Path(args.out)
    out.write_text(m.to_json(), encoding="utf-8")
    print(json.dumps({"fingerprint": m.manifest_fingerprint(), "out": str(out)}, indent=2))
    return 0


def _cmd_ingest_builtin(args: argparse.Namespace) -> int:
    from greenlang.factors.etl.ingest import ingest_builtin_database

    n = ingest_builtin_database(Path(args.sqlite), args.edition_id, label=args.label)
    print(json.dumps({"ingested": n, "sqlite": str(args.sqlite)}, indent=2))
    return 0


def _cmd_watch_dry_run(args: argparse.Namespace) -> int:
    from greenlang.factors.watch.source_watch import dry_run_registry_urls

    rows = dry_run_registry_urls()
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_validate_registry(args: argparse.Namespace) -> int:
    from greenlang.factors.source_registry import load_source_registry, validate_registry

    issues = validate_registry()
    payload = {"ok": not issues, "issues": issues, "count": len(load_source_registry())}
    print(json.dumps(payload, indent=2))
    return 0 if not issues else 1


def _cmd_ingest_paths(args: argparse.Namespace) -> int:
    from greenlang.factors.etl.ingest import ingest_from_paths

    paths = [Path(x) for x in args.paths]
    n = ingest_from_paths(
        Path(args.sqlite),
        args.edition_id,
        paths,
        label=args.label,
        status=args.status,
    )
    print(json.dumps({"ingested": n, "sqlite": str(args.sqlite)}, indent=2))
    return 0


def _cmd_watch_run(args: argparse.Namespace) -> int:
    from greenlang.factors.notifications.webhook_notifier import build_watch_notify_callback
    from greenlang.factors.watch.scheduler import run_watch, watch_summary

    db_path = Path(args.db) if args.db else None
    notify = build_watch_notify_callback() if args.notify else None
    results = run_watch(db_path=db_path, store=bool(db_path), notify=notify)
    summary = watch_summary(results)
    print(json.dumps(summary, indent=2))
    return 1 if summary["errors"] > 0 else 0


def _cmd_release_prepare(args: argparse.Namespace) -> int:
    from greenlang.factors.service import FactorCatalogService
    from greenlang.factors.watch.release_orchestrator import prepare_release

    svc = FactorCatalogService.from_environment()
    report = prepare_release(
        svc.repo,
        args.edition_id,
        previous_edition_id=args.previous,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print(json.dumps({"written": args.out, "status": report.status, "ready": report.is_ready()}, indent=2))
    else:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.is_ready() else 1


def _cmd_connector_list(args: argparse.Namespace) -> int:
    from greenlang.factors.connectors import register_default_connectors

    registry = register_default_connectors()
    data = []
    for sid in registry.list_source_ids():
        c = registry.get(sid)
        if c:
            cap = c.capabilities
            data.append({
                "source_id": sid,
                "requires_license": cap.requires_license,
                "typical_factor_count": cap.typical_factor_count,
                "supports_real_time": cap.supports_real_time,
                "supports_batch": cap.supports_batch_fetch,
            })
    print(json.dumps(data, indent=2))
    return 0


def _cmd_connector_health(args: argparse.Namespace) -> int:
    from greenlang.factors.connectors import register_default_connectors

    registry = register_default_connectors()
    c = registry.get(args.connector_id, license_key=args.license_key)
    if not c:
        print(json.dumps({"error": f"Unknown connector: {args.connector_id}"}, indent=2))
        return 1
    result = c.health_check()
    print(json.dumps({
        "connector_id": args.connector_id,
        "status": result.status.value,
        "latency_ms": result.latency_ms,
        "message": result.message,
        "checked_at": result.checked_at,
    }, indent=2))
    return 0 if result.status.value == "healthy" else 1


def _cmd_connector_fetch_metadata(args: argparse.Namespace) -> int:
    from greenlang.factors.connectors import register_default_connectors

    registry = register_default_connectors()
    c = registry.get(args.connector_id, license_key=args.license_key)
    if not c:
        print(json.dumps({"error": f"Unknown connector: {args.connector_id}"}, indent=2))
        return 1
    meta = c.fetch_metadata()
    if args.out:
        Path(args.out).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"written": args.out, "count": len(meta)}, indent=2))
    else:
        print(json.dumps({"count": len(meta), "sample": meta[:5]}, indent=2))
    return 0


def _cmd_release_publish(args: argparse.Namespace) -> int:
    from greenlang.factors.service import FactorCatalogService
    from greenlang.factors.watch.release_orchestrator import publish_release

    svc = FactorCatalogService.from_environment()
    result = publish_release(svc.repo, args.edition_id, args.approved_by)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_bulk_ingest(args: argparse.Namespace) -> int:
    """Run the full ingestion pipeline via scripts/run_ingestion.py logic."""
    import time

    mode = args.mode
    sqlite_path = Path(args.sqlite)
    edition_id = args.edition_id
    dry_run = args.dry_run

    logger.info(
        "bulk-ingest: mode=%s sqlite=%s edition=%s dry_run=%s",
        mode, sqlite_path, edition_id, dry_run,
    )

    if mode == "builtin":
        from greenlang.factors.etl.ingest import ingest_builtin_database

        if dry_run:
            from greenlang.data.emission_factor_database import EmissionFactorDatabase

            db = EmissionFactorDatabase(enable_cache=False)
            n = len(db.factors)
            print(json.dumps({"dry_run": True, "mode": mode, "would_ingest": n}, indent=2))
            return 0

        n = ingest_builtin_database(sqlite_path, edition_id, label=args.label)
        print(json.dumps({"ingested": n, "mode": mode, "sqlite": str(sqlite_path)}, indent=2))
        return 0

    elif mode == "synthetic":
        from greenlang.factors.ingestion.synthetic_data import generate_and_validate
        from greenlang.factors.etl.normalize import dict_to_emission_factor_record
        from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
        from greenlang.factors.edition_manifest import build_manifest_for_factors

        count = args.count
        seed = args.seed
        years = [int(y.strip()) for y in args.years.split(",")]

        t0 = time.monotonic()
        valid_dicts, total_gen, total_rej = generate_and_validate(
            count=count, seed=seed, years=years,
        )

        if dry_run:
            elapsed = time.monotonic() - t0
            print(json.dumps({
                "dry_run": True,
                "mode": mode,
                "generated": total_gen,
                "valid": len(valid_dicts),
                "rejected": total_rej,
                "elapsed_s": round(elapsed, 2),
                "sample_ids": [fd["factor_id"] for fd in valid_dicts[:5]],
            }, indent=2))
            return 0

        records = []
        for fd in valid_dicts:
            try:
                records.append(dict_to_emission_factor_record(fd))
            except Exception as exc:
                logger.warning("Conversion error: %s", exc)

        if not records:
            print(json.dumps({"error": "No records survived conversion"}, indent=2))
            return 1

        repo = SqliteFactorCatalogRepository(sqlite_path)
        manifest = build_manifest_for_factors(
            edition_id, "stable", records,
            changelog=[f"Synthetic bulk-ingest: {len(records)} factors (seed={seed})"],
        )
        repo.upsert_edition(edition_id, "stable", args.label, manifest.to_dict(), manifest.changelog)
        repo.insert_factors(edition_id, records)

        elapsed = time.monotonic() - t0
        print(json.dumps({
            "mode": mode,
            "ingested": len(records),
            "generated": total_gen,
            "rejected": total_rej,
            "elapsed_s": round(elapsed, 2),
            "sqlite": str(sqlite_path),
            "edition_id": edition_id,
        }, indent=2))
        return 0

    elif mode == "full":
        from greenlang.factors.etl.ingest import ingest_builtin_database
        from greenlang.factors.ingestion.synthetic_data import generate_and_validate
        from greenlang.factors.etl.normalize import dict_to_emission_factor_record
        from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
        from greenlang.factors.edition_manifest import build_manifest_for_factors

        t0 = time.monotonic()

        # Phase 1: builtin
        if not dry_run:
            n_builtin = ingest_builtin_database(sqlite_path, edition_id, label=args.label)
        else:
            from greenlang.data.emission_factor_database import EmissionFactorDatabase

            n_builtin = len(EmissionFactorDatabase(enable_cache=False).factors)

        # Phase 2: synthetic
        count = args.count
        seed = args.seed
        years = [int(y.strip()) for y in args.years.split(",")]
        valid_dicts, total_gen, total_rej = generate_and_validate(
            count=count, seed=seed, years=years,
        )

        if dry_run:
            elapsed = time.monotonic() - t0
            print(json.dumps({
                "dry_run": True,
                "mode": mode,
                "builtin_count": n_builtin,
                "synthetic_generated": total_gen,
                "synthetic_valid": len(valid_dicts),
                "synthetic_rejected": total_rej,
                "elapsed_s": round(elapsed, 2),
            }, indent=2))
            return 0

        records = []
        for fd in valid_dicts:
            try:
                records.append(dict_to_emission_factor_record(fd))
            except Exception:
                pass

        if records:
            repo = SqliteFactorCatalogRepository(sqlite_path)
            repo.insert_factors(edition_id, records)

        elapsed = time.monotonic() - t0
        print(json.dumps({
            "mode": mode,
            "builtin_ingested": n_builtin,
            "synthetic_ingested": len(records),
            "total_ingested": n_builtin + len(records),
            "elapsed_s": round(elapsed, 2),
            "sqlite": str(sqlite_path),
            "edition_id": edition_id,
        }, indent=2))
        return 0

    else:
        print(json.dumps({"error": f"Unknown mode: {mode}"}, indent=2))
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GreenLang Factors catalog tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inventory", help="Write coverage matrix JSON")
    pi.add_argument("--out", default="greenlang/factors/artifacts/coverage_matrix.json")
    pi.set_defaults(func=_cmd_inventory)

    pm = sub.add_parser("manifest", help="Write EditionManifest JSON from built-in DB")
    pm.add_argument("--edition-id", dest="edition_id", required=True)
    pm.add_argument("--status", default="stable")
    pm.add_argument("--message", default="")
    pm.add_argument("--out", required=True)
    pm.set_defaults(func=_cmd_manifest)

    ib = sub.add_parser("ingest-builtin", help="Load EmissionFactorDatabase into SQLite")
    ib.add_argument("--sqlite", required=True)
    ib.add_argument("--edition-id", dest="edition_id", required=True)
    ib.add_argument("--label", default="Built-in v2 EmissionFactorDatabase")
    ib.set_defaults(func=_cmd_ingest_builtin)

    wd = sub.add_parser("watch-dry-run", help="Probe registry watch URLs (no writes)")
    wd.set_defaults(func=_cmd_watch_dry_run)

    vr = sub.add_parser("validate-registry", help="Validate source_registry.yaml (G1–G6)")
    vr.set_defaults(func=_cmd_validate_registry)

    ip = sub.add_parser("ingest-paths", help="Normalize JSON paths and load SQLite")
    ip.add_argument("--sqlite", required=True)
    ip.add_argument("--edition-id", dest="edition_id", required=True)
    ip.add_argument("--paths", nargs="+", required=True)
    ip.add_argument("--label", default="ETL bundle")
    ip.add_argument("--status", default="stable")
    ip.set_defaults(func=_cmd_ingest_paths)

    wr = sub.add_parser("watch-run", help="Run automated source watch (F050)")
    wr.add_argument("--db", help="SQLite path to store watch results")
    wr.add_argument("--notify", action="store_true", help="Enable Slack/email notifications")
    wr.set_defaults(func=_cmd_watch_run)

    rp = sub.add_parser("release-prepare", help="Prepare release report (F053)")
    rp.add_argument("--edition-id", dest="edition_id", required=True)
    rp.add_argument("--previous", help="Previous edition ID for changelog comparison")
    rp.add_argument("--out", help="Output JSON path for release report")
    rp.set_defaults(func=_cmd_release_prepare)

    rpub = sub.add_parser("release-publish", help="Promote edition to stable (F053)")
    rpub.add_argument("--edition-id", dest="edition_id", required=True)
    rpub.add_argument("--approved-by", dest="approved_by", required=True, help="Approver email/username")
    rpub.set_defaults(func=_cmd_release_publish)

    cl = sub.add_parser("connector-list", help="List registered connectors (F060)")
    cl.set_defaults(func=_cmd_connector_list)

    ch = sub.add_parser("connector-health", help="Check connector health (F060)")
    ch.add_argument("--connector-id", dest="connector_id", required=True)
    ch.add_argument("--license-key", dest="license_key", help="License key (or use env var)")
    ch.set_defaults(func=_cmd_connector_health)

    cm = sub.add_parser("connector-metadata", help="Fetch connector metadata (F060)")
    cm.add_argument("--connector-id", dest="connector_id", required=True)
    cm.add_argument("--license-key", dest="license_key", help="License key (or use env var)")
    cm.add_argument("--out", help="Output JSON path")
    cm.set_defaults(func=_cmd_connector_fetch_metadata)

    bi = sub.add_parser(
        "bulk-ingest",
        help="Bulk ingestion pipeline: builtin, synthetic, or full (F019)",
    )
    bi.add_argument(
        "--mode",
        choices=["builtin", "synthetic", "full"],
        default="synthetic",
        help="Ingestion mode (default: synthetic)",
    )
    bi.add_argument("--sqlite", required=True, help="SQLite database path")
    bi.add_argument("--edition-id", dest="edition_id", required=True, help="Edition identifier")
    bi.add_argument("--label", default="Bulk ingestion", help="Edition label")
    bi.add_argument(
        "--count",
        type=int,
        default=25000,
        help="Number of synthetic factors (default: 25000)",
    )
    bi.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    bi.add_argument(
        "--years",
        default="2022,2023,2024,2025",
        help="Comma-separated years (default: 2022,2023,2024,2025)",
    )
    bi.add_argument("--dry-run", dest="dry_run", action="store_true", help="Validate without writing")
    bi.set_defaults(func=_cmd_bulk_ingest)

    args = p.parse_args(argv)
    logger.info("CLI command=%s", args.cmd)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
