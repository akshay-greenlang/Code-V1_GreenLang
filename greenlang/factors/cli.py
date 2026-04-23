# -*- coding: utf-8 -*-
"""CLI entry: python -m greenlang.factors.cli"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


# ============================================================================
# resolve / explain — CTO-critical commands (mirror /v1/resolve & /v1/explain)
# ============================================================================


def _print_payload(payload: Dict[str, Any], *, json_mode: bool, compact: bool) -> None:
    """Print a payload either as JSON (compact or indented) or as a short
    human-readable summary suitable for a terminal."""
    if json_mode:
        print(json.dumps(payload, default=str, separators=(",", ":") if compact else None,
                         indent=None if compact else 2))
        return

    # Human-friendly view (compact summary). The CLI defaults to JSON
    # because that mirrors the API; the text view is a convenience.
    chosen = payload.get("explain", {}).get("chosen") or {
        "factor_id": payload.get("chosen_factor_id"),
        "source": payload.get("source_id"),
        "source_version": payload.get("source_version"),
    }
    derivation = payload.get("explain", {}).get("derivation") or {
        "fallback_rank": payload.get("fallback_rank"),
        "step_label": payload.get("step_label"),
        "why_chosen": payload.get("why_chosen"),
    }
    quality = payload.get("explain", {}).get("quality") or {
        "score": payload.get("quality_score"),
    }
    emissions = payload.get("explain", {}).get("emissions") or payload.get("gas_breakdown", {}) or {}
    print(f"Factor:           {chosen.get('factor_id')}")
    print(f"Source:           {chosen.get('source')} (version {chosen.get('source_version')})")
    print(f"Method profile:   {chosen.get('method_profile') or payload.get('method_profile')}")
    print(f"Fallback rank:    {derivation.get('fallback_rank')} ({derivation.get('step_label')})")
    print(f"Quality score:    {quality.get('score')}")
    print(f"CO2e (kg):        {emissions.get('co2e_total_kg')}  (basis: {emissions.get('gwp_basis')})")
    why = derivation.get("why_chosen")
    if why:
        print(f"Why chosen:       {why}")
    alts = payload.get("alternates") or []
    if alts:
        print(f"Alternates:       {len(alts)}")
        for a in alts[:3]:
            print(f"  - {a.get('factor_id')}  score={a.get('tie_break_score')}")


def _build_resolve_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the /v1/resolve payload using the same engine the API uses.

    The CLI deliberately reuses :func:`build_resolution_explain` from
    ``api_endpoints`` so resolve/explain output is byte-identical to the
    HTTP surface.
    """
    from greenlang.data.canonical_v2 import MethodProfile
    from greenlang.factors.api_endpoints import build_resolution_explain
    from greenlang.factors.service import FactorCatalogService

    svc = FactorCatalogService.from_environment()
    edition_id = svc.repo.resolve_edition(args.edition)

    # Validate method_profile early so the user sees a friendly error
    # instead of a Pydantic stack trace.
    try:
        MethodProfile(args.method_profile)
    except ValueError as exc:
        valid = ", ".join(p.value for p in MethodProfile)
        raise SystemExit(
            f"Unknown method-profile: {args.method_profile!r}. "
            f"Valid options: {valid}."
        ) from exc

    request_dict: Dict[str, Any] = {
        "activity": args.activity,
        "method_profile": args.method_profile,
    }
    if args.country:
        request_dict["jurisdiction"] = args.country
    if args.target_unit:
        request_dict["target_unit"] = args.target_unit
    if args.reporting_date:
        request_dict["reporting_date"] = args.reporting_date
    if args.tenant:
        request_dict["tenant_id"] = args.tenant
    if args.supplier:
        request_dict["supplier_id"] = args.supplier
    if args.facility:
        request_dict["facility_id"] = args.facility
    if args.utility:
        request_dict["utility_or_grid_region"] = args.utility
    if args.include_preview:
        request_dict["include_preview"] = True

    extras: Dict[str, Any] = {}
    if args.quantity is not None:
        extras["quantity"] = float(args.quantity)
    if args.unit:
        extras["activity_unit"] = args.unit
    if extras:
        request_dict["extras"] = extras

    payload = build_resolution_explain(
        svc.repo,
        edition_id,
        request_dict,
        include_preview=args.include_preview,
        include_connector=False,
    )
    payload["edition_id"] = edition_id

    # Surface co2e directly when the caller passed --quantity, mirroring
    # the API quickstart shape.
    if args.quantity is not None:
        gas = payload.get("gas_breakdown") or {}
        co2e_per_unit = gas.get("co2e_total_kg")
        if co2e_per_unit is not None:
            try:
                payload["co2e"] = float(co2e_per_unit) * float(args.quantity)
                payload["co2e_unit"] = "kg"
            except (TypeError, ValueError):
                pass
    return payload


def _cmd_resolve(args: argparse.Namespace) -> int:
    """gl-factors resolve — POST /v1/resolve via CLI."""
    try:
        payload = _build_resolve_payload(args)
    except SystemExit:
        raise
    except Exception as exc:
        err = {"error": "resolve_failed", "message": str(exc)}
        print(json.dumps(err, indent=2 if not args.compact else None))
        return 2

    _print_payload(payload, json_mode=args.json or args.compact, compact=args.compact)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """gl-factors explain <factor_id> — GET /v1/factors/{id}/explain via CLI."""
    from greenlang.factors.api_endpoints import build_factor_explain
    from greenlang.factors.service import FactorCatalogService

    svc = FactorCatalogService.from_environment()
    try:
        edition_id = svc.repo.resolve_edition(args.edition)
    except Exception as exc:
        print(json.dumps({"error": "unknown_edition", "message": str(exc)}, indent=2))
        return 2

    payload = build_factor_explain(
        svc.repo,
        edition_id,
        args.factor_id,
        method_profile=args.method_profile,
    )
    if payload is None:
        err = {
            "error": "factor_not_found",
            "factor_id": args.factor_id,
            "edition_id": edition_id,
        }
        print(json.dumps(err, indent=2))
        return 1

    payload["edition_id"] = edition_id
    _print_payload(payload, json_mode=args.json or args.compact, compact=args.compact)
    return 0


# ============================================================================
# Argument parser
# ============================================================================


def _add_resolve_parser(sub: argparse._SubParsersAction) -> None:
    rs = sub.add_parser(
        "resolve",
        help="Resolve a factor through the 7-step cascade (mirrors POST /v1/resolve)",
    )
    rs.add_argument("--activity", required=True, help="Free-text activity, e.g. 'purchased electricity'")
    rs.add_argument("--method-profile", dest="method_profile", required=True,
                    help="Method profile (e.g. corporate_scope2_location_based)")
    rs.add_argument("--quantity", type=float, default=None,
                    help="Activity quantity. When supplied, payload includes computed co2e.")
    rs.add_argument("--unit", default=None,
                    help="Activity unit (kWh, MMBtu, kg, etc.)")
    rs.add_argument("--country", default=None,
                    help="ISO country / region code (e.g. IN, US-CA, EU)")
    rs.add_argument("--edition", default=None,
                    help="Pin a specific catalog edition (defaults to stable)")
    rs.add_argument("--target-unit", dest="target_unit", default=None,
                    help="Convert factor to this denominator unit (uses unit graph)")
    rs.add_argument("--reporting-date", dest="reporting_date", default=None,
                    help="ISO-8601 reporting date (defaults to today)")
    rs.add_argument("--tenant", default=None, help="Tenant id for the customer-override step")
    rs.add_argument("--supplier", default=None, help="Supplier id for step 2")
    rs.add_argument("--facility", default=None, help="Facility id for step 3")
    rs.add_argument("--utility", default=None,
                    help="Utility tariff or grid sub-region for step 4 (e.g. eGRID-SERC)")
    rs.add_argument("--include-preview", dest="include_preview", action="store_true",
                    help="Allow preview-status factors when no certified match exists")
    rs.add_argument("--json", action="store_true", help="Emit JSON (default)")
    rs.add_argument("--compact", action="store_true", help="Single-line JSON output")
    rs.set_defaults(func=_cmd_resolve)


def _add_explain_parser(sub: argparse._SubParsersAction) -> None:
    ex = sub.add_parser(
        "explain",
        help="Show the explain payload for a specific factor (mirrors GET /v1/factors/{id}/explain)",
    )
    ex.add_argument("factor_id", help="Factor id to explain")
    ex.add_argument("--edition", default=None,
                    help="Pin a specific catalog edition (defaults to stable)")
    ex.add_argument("--method-profile", dest="method_profile", default=None,
                    help="Method profile (defaults to derived from factor scope)")
    ex.add_argument("--json", action="store_true", help="Emit JSON (default)")
    ex.add_argument("--compact", action="store_true", help="Single-line JSON output")
    ex.set_defaults(func=_cmd_explain)


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

    # CTO-critical resolve / explain commands
    _add_resolve_parser(sub)
    _add_explain_parser(sub)

    args = p.parse_args(argv)
    logger.info("CLI command=%s", args.cmd)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
