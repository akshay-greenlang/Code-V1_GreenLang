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

    args = p.parse_args(argv)
    logger.info("CLI command=%s", args.cmd)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
