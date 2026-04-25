#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha — staging/production publish CLI.

Wave E / TaskCreate #23 / WS9-T1. Ships the `gl factors-alpha` operator
surface that satisfies CTO doc §19.1:

    "Runbook for a manual 'publish to production namespace' step:
     climate-methodology lead reviews staging diffs and flips visibility."

Subcommands
-----------
    staging  --source <source_id>          publishes seeds -> staging
    diff                                    prints staging vs production
    flip     --urn <urn> --approved-by <e>  promote one URN
    flip     --all-staging --approved-by <e> bulk promote
    rollback --batch-id <id> --approved-by <e>  demote a batch
    list     --namespace {staging,production}

DSN resolution: ``--dsn`` flag wins, else ``GL_FACTORS_ALPHA_DSN`` env
var, else ``sqlite:///./alpha_factors_v0_1.db`` (a file in CWD so a
local operator gets persistence between subcommand invocations).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Allow running the script directly without pip-installing the repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from greenlang.factors.release.alpha_publisher import (  # noqa: E402
    AlphaPublisher,
    AlphaPublisherError,
    StagingDiff,
)
from greenlang.factors.repositories import AlphaFactorRepository  # noqa: E402
from greenlang.factors.quality.alpha_provenance_gate import (  # noqa: E402
    AlphaProvenanceGateError,
)
from greenlang.factors.repositories.alpha_v0_1_repository import (  # noqa: E402
    FactorURNAlreadyExistsError,
)


logger = logging.getLogger("factors_alpha_publish")

_DEFAULT_DSN = "sqlite:///./alpha_factors_v0_1.db"
_SEED_DIR = (
    _REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v0_1"
)
_DIFF_OUT_DIR = _REPO_ROOT / "out" / "factors" / "v0_1_alpha"


# ---------------------------------------------------------------------------
# Boot helpers
# ---------------------------------------------------------------------------


def _resolve_dsn(args: argparse.Namespace) -> str:
    if getattr(args, "dsn", None):
        return args.dsn
    env_dsn = os.environ.get("GL_FACTORS_ALPHA_DSN")
    if env_dsn:
        return env_dsn
    return _DEFAULT_DSN


def _make_publisher(args: argparse.Namespace) -> AlphaPublisher:
    repo = AlphaFactorRepository(dsn=_resolve_dsn(args))
    return AlphaPublisher(repo)


def _load_seed(source_id: str) -> List[Dict[str, Any]]:
    path = _SEED_DIR / source_id / "v1.json"
    if not path.exists():
        raise FileNotFoundError(
            f"seed file not found for source_id={source_id!r}: {path}"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(
            f"seed file {path} does not contain a 'records' list"
        )
    return records


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_staging(args: argparse.Namespace) -> int:
    pub = _make_publisher(args)
    try:
        records = _load_seed(args.source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ok = 0
    skipped = 0
    failed = 0
    for rec in records:
        try:
            pub.publish_to_staging(rec)
            ok += 1
        except FactorURNAlreadyExistsError:
            skipped += 1
        except AlphaProvenanceGateError as exc:
            failed += 1
            print(
                f"GATE REJECT urn={rec.get('urn')!r}: {exc}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(
                f"PUBLISH ERROR urn={rec.get('urn')!r}: {exc}",
                file=sys.stderr,
            )

    print(
        f"staging: source={args.source} published={ok} "
        f"already_staged={skipped} failed={failed}"
    )
    return 0 if failed == 0 else 1


def cmd_diff(args: argparse.Namespace) -> int:
    pub = _make_publisher(args)
    diff = pub.diff_staging_vs_production()
    print(diff.summary())

    md = _render_diff_markdown(diff)
    if args.write or args.write_path:
        out_path = Path(args.write_path) if args.write_path else None
        if out_path is None:
            _DIFF_OUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = _DIFF_OUT_DIR / f"staging-diff-{ts}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"diff report written to {out_path}")
    elif args.format == "markdown":
        print(md)
    elif args.format == "json":
        print(json.dumps(_diff_to_json(diff), indent=2))
    return 0


def cmd_flip(args: argparse.Namespace) -> int:
    pub = _make_publisher(args)
    if args.all_staging:
        urns = [r.get("urn") for r in pub.list_staging() if r.get("urn")]
    elif args.urn:
        urns = list(args.urn)
    else:
        print("ERROR: provide either --urn or --all-staging", file=sys.stderr)
        return 2
    if not urns:
        print("flip: nothing to promote (staging is empty)")
        return 0
    try:
        promoted = pub.flip_to_production(urns=urns, approved_by=args.approved_by)
    except AlphaPublisherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"flip: promoted={promoted} of {len(urns)} requested "
        f"approved_by={args.approved_by}"
    )
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    pub = _make_publisher(args)
    try:
        demoted = pub.rollback(
            batch_id=args.batch_id, approved_by=args.approved_by
        )
    except AlphaPublisherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"rollback: batch={args.batch_id} demoted={demoted} "
        f"approved_by={args.approved_by}"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    pub = _make_publisher(args)
    if args.namespace == "staging":
        rows = pub.list_staging()
    else:
        rows = pub.list_production()
    if args.format == "json":
        print(json.dumps([{"urn": r.get("urn")} for r in rows], indent=2))
    else:
        for r in rows:
            print(r.get("urn"))
        print(f"# total={len(rows)} namespace={args.namespace}")
    return 0


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------


def _diff_to_json(diff: StagingDiff) -> Dict[str, Any]:
    return {
        "additions": [{"urn": r.get("urn")} for r in diff.additions],
        "removals": list(diff.removals),
        "changes": [
            {"old_urn": old, "new_urn": new} for old, new in diff.changes
        ],
        "unchanged": diff.unchanged,
    }


def _render_diff_markdown(diff: StagingDiff) -> str:
    lines: List[str] = []
    ts = datetime.now(timezone.utc).isoformat()
    lines.append("# Factors v0.1 Alpha — Staging vs Production Diff")
    lines.append("")
    lines.append(f"_Generated: {ts}_")
    lines.append("")
    lines.append(f"**Summary:** {diff.summary()}")
    lines.append("")
    lines.append("## Additions (in staging, not in production)")
    if not diff.additions:
        lines.append("- _none_")
    else:
        for rec in diff.additions:
            lines.append(
                f"- `{rec.get('urn')}` — source=`{rec.get('source_urn')}`"
                f" pack=`{rec.get('factor_pack_urn')}`"
            )
    lines.append("")
    lines.append("## Removals (in production, missing from staging)")
    if not diff.removals:
        lines.append("- _none_")
    else:
        for urn in diff.removals:
            lines.append(f"- `{urn}`")
    lines.append("")
    lines.append("## Changes (supersede pairs)")
    if not diff.changes:
        lines.append("- _none_")
    else:
        for old, new in diff.changes:
            lines.append(f"- `{old}` -> `{new}`")
    lines.append("")
    lines.append(f"## Unchanged: {diff.unchanged}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gl factors-alpha",
        description=(
            "GreenLang Factors v0.1 alpha — staging/production publish "
            "operator surface."
        ),
    )
    p.add_argument(
        "--dsn",
        default=None,
        help=(
            "Repository DSN (sqlite:///path or postgresql://...). "
            "Defaults to $GL_FACTORS_ALPHA_DSN or sqlite:///./alpha_factors_v0_1.db"
        ),
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    staging = sub.add_parser(
        "staging",
        help="publish v0.1 seeds for a source to the staging namespace",
    )
    staging.add_argument(
        "--source",
        required=True,
        help="source_id whose seed file under catalog_seed_v0_1 to publish",
    )
    staging.set_defaults(func=cmd_staging)

    diff = sub.add_parser(
        "diff", help="compute and print staging vs production diff"
    )
    diff.add_argument(
        "--format", choices=["markdown", "json"], default="markdown"
    )
    diff.add_argument(
        "--write",
        action="store_true",
        help="write the report to out/factors/v0_1_alpha/staging-diff-{ts}.md",
    )
    diff.add_argument(
        "--write-path",
        default=None,
        help="write report to a specific path (overrides --write default)",
    )
    diff.set_defaults(func=cmd_diff)

    flip = sub.add_parser(
        "flip",
        help="promote staging URNs to production (requires methodology lead)",
    )
    flip.add_argument(
        "--urn", action="append", help="URN to promote (repeatable)"
    )
    flip.add_argument(
        "--all-staging",
        action="store_true",
        help="promote every URN currently in staging",
    )
    flip.add_argument(
        "--approved-by",
        required=True,
        help="methodology lead identifier (must start with 'human:')",
    )
    flip.set_defaults(func=cmd_flip)

    rollback = sub.add_parser(
        "rollback",
        help="demote a batch of promoted URNs back to staging",
    )
    rollback.add_argument("--batch-id", required=True)
    rollback.add_argument(
        "--approved-by",
        required=True,
        help="methodology lead identifier (must start with 'human:')",
    )
    rollback.set_defaults(func=cmd_rollback)

    listc = sub.add_parser("list", help="list URNs in a namespace")
    listc.add_argument(
        "--namespace",
        choices=["staging", "production"],
        default="production",
    )
    listc.add_argument(
        "--format", choices=["text", "json"], default="text"
    )
    listc.set_defaults(func=cmd_list)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
