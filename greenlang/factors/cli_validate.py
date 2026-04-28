# -*- coding: utf-8 -*-
"""``gl factors validate`` — Phase 2 publish-time gate dry-run CLI (WS8).

Loads a JSON v0.1 factor record from disk, runs all seven publish gates
via :class:`PublishGateOrchestrator`, and prints a per-gate result table.
The CLI is wired into ``greenlang.factors.cli`` as the ``validate``
subcommand and surfaced via ``python -m greenlang.factors validate``.

Exit codes:
    0 — all seven gates PASS.
    1 — at least one gate FAIL (or the input file is malformed).
    2 — invocation error (file not found / unreadable / not JSON).

Design choices:
    * The CLI never RAISES on gate failure — it always prints the full
      seven-row table so a failing CI build can show which gate(s) caught
      the record. ``--strict`` (no longer needed) was rejected because
      ``dry_run`` is the canonical "show me everything" view; the exit
      code is the strict signal.
    * When ``--dsn`` is omitted, the CLI builds an in-memory sqlite
      repository and seeds the ontology tables from the canonical YAML
      seeds. This makes the CLI useful for parsers running without a
      live Postgres but still wanting FK enforcement.
    * Colour output is opt-OUT via ``--no-color`` and auto-disabled when
      stdout is not a tty.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

logger = logging.getLogger(__name__)


__all__ = [
    "build_validate_parser",
    "cmd_validate",
    "main",
]


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------


_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _supports_colour(stream: TextIO, force_no_colour: bool) -> bool:
    if force_no_colour:
        return False
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


def _paint(text: str, code: str, *, enable: bool) -> str:
    if not enable:
        return text
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Repository bootstrap
# ---------------------------------------------------------------------------


def _seed_ontology_tables(repo: Any) -> None:
    """Create + seed geography / unit / methodology / activity tables on sqlite.

    Postgres mode skips this — the tables are owned by V500/V502 migrations.
    """
    if getattr(repo, "_is_postgres", False):
        return
    conn = repo._connect()  # type: ignore[attr-defined]
    if not isinstance(conn, sqlite3.Connection):
        return

    # Lazy imports so the CLI starts fast in negative-path tests.
    from greenlang.factors.data.ontology.loaders.geography_loader import (
        create_sqlite_geography_table,
        load_geography,
    )
    from greenlang.factors.data.ontology.loaders.unit_loader import (
        create_sqlite_unit_table,
        load_units,
    )
    from greenlang.factors.data.ontology.loaders.methodology_loader import (
        create_sqlite_methodology_table,
        load_methodologies,
    )
    from greenlang.factors.data.ontology.loaders.activity_loader import (
        create_sqlite_activity_table,
        load_into_sqlite as load_activities_sqlite,
    )

    create_sqlite_geography_table(conn)
    create_sqlite_unit_table(conn)
    create_sqlite_methodology_table(conn)
    create_sqlite_activity_table(conn)
    try:
        load_geography(conn)
        load_units(conn)
        load_methodologies(conn)
        load_activities_sqlite(conn)
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.warning("cli_validate: ontology seed failed: %s", exc)


def _build_repo(dsn: Optional[str], *, seed_ontology: bool) -> Any:
    """Construct an :class:`AlphaFactorRepository` for the CLI."""
    from greenlang.factors.repositories import AlphaFactorRepository

    repo = AlphaFactorRepository(
        dsn=dsn or "sqlite:///:memory:",
        publish_orchestrator=False,
    )
    if seed_ontology:
        _seed_ontology_tables(repo)
    return repo


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _format_results(
    results: List[Any],
    *,
    use_colour: bool,
) -> str:
    """Pretty-print a list of GateResult instances.

    Each row carries the gate id, outcome, and (on FAIL) the reason. The
    table is plain-text so it survives log scraping and CI annotation.
    """
    lines: List[str] = []
    lines.append(
        f"{'Gate':32}  {'Outcome':9}  {'Detail'}"
    )
    lines.append("-" * 78)
    for r in results:
        if r.outcome == "PASS":
            colour = _GREEN
        elif r.outcome == "FAIL":
            colour = _RED
        else:
            colour = _DIM
        outcome_painted = _paint(r.outcome, colour, enable=use_colour)
        # ``r.reason`` is empty on PASS / NOT_RUN; on FAIL it carries
        # the gate's failure message.
        detail = r.reason or "-"
        lines.append(f"{r.gate_id:32}  {outcome_painted:18}  {detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Entry-point for the ``validate`` subcommand.

    Args:
        args: parsed namespace with ``record_path``, ``env``, ``dry_run``,
            ``dsn``, ``no_color``.

    Returns:
        0 if every gate passes; 1 if any gate fails or the input is bad.
    """
    record_path = Path(args.record_path)
    if not record_path.exists():
        sys.stderr.write(f"error: file not found: {record_path}\n")
        return 2
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"error: {record_path} is not valid JSON: {exc}\n")
        return 2

    use_colour = _supports_colour(sys.stdout, force_no_colour=args.no_color)

    # Build the orchestrator with a CLI-local repo so we never hit the
    # production database. Ontology seed is on by default; opt-out via
    # --skip-seed when the caller supplies a real DSN already seeded.
    repo = _build_repo(args.dsn, seed_ontology=not args.skip_seed)

    from greenlang.factors.quality.publish_gates import PublishGateOrchestrator

    orchestrator = PublishGateOrchestrator(repo, env=args.env)

    if args.dry_run:
        results = orchestrator.dry_run(record)
        sys.stdout.write(_format_results(results, use_colour=use_colour) + "\n")
        # Aggregate: any FAIL -> non-zero exit code.
        any_fail = any(r.outcome == "FAIL" for r in results)
        # Cleanup the in-memory repo eagerly (matters in long-running shells).
        try:
            repo.close()
        except Exception:  # noqa: BLE001
            pass
        return 1 if any_fail else 0

    # Non dry-run: assert mode. Raise on first failure but render a
    # one-row failure summary so callers don't have to parse the
    # exception class.
    try:
        orchestrator.assert_publishable(record)
    except Exception as exc:  # noqa: BLE001
        gate_id = getattr(exc, "gate_id", "publish_gate")
        reason = getattr(exc, "reason", str(exc))
        urn = getattr(exc, "urn", record.get("urn") if isinstance(record, dict) else None)
        sys.stderr.write(
            f"FAIL [{gate_id}] urn={urn} reason={reason}\n"
        )
        try:
            repo.close()
        except Exception:  # noqa: BLE001
            pass
        return 1

    sys.stdout.write(
        _paint("PASS — all 7 gates accepted the record", _GREEN, enable=use_colour)
        + "\n"
    )
    try:
        repo.close()
    except Exception:  # noqa: BLE001
        pass
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_validate_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Attach the ``validate`` subcommand to a parent ``add_subparsers``.

    Hook called from :func:`greenlang.factors.cli.main` to keep all
    factor commands under a single argparse tree.
    """
    p = sub.add_parser(
        "validate",
        help="Run the seven publish-time validation gates against a v0.1 factor JSON",
        description=(
            "Loads <record-path> as JSON and runs the seven publish "
            "gates (schema, URN uniqueness, ontology FK, source registry, "
            "licence match, provenance completeness, lifecycle status). "
            "Exits 0 on full PASS, 1 on any FAIL, 2 on invocation error."
        ),
    )
    p.add_argument(
        "record_path",
        help="Path to a v0.1 factor record JSON file",
    )
    p.add_argument(
        "--env",
        choices=("production", "staging", "dev"),
        default="production",
        help="Publish environment (affects gate 4 and gate 7 strictness)",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run every gate (no early exit) and print a per-gate table",
    )
    p.add_argument(
        "--dsn",
        default=None,
        help=(
            "Optional database DSN (sqlite:///path or postgresql://...). "
            "Default is an in-memory SQLite seeded with the canonical "
            "ontology tables."
        ),
    )
    p.add_argument(
        "--skip-seed",
        dest="skip_seed",
        action="store_true",
        help="Skip the in-memory ontology seed (for live-DB callers).",
    )
    p.add_argument(
        "--no-color",
        dest="no_color",
        action="store_true",
        help="Disable ANSI colour output",
    )
    p.set_defaults(func=cmd_validate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Stand-alone entry-point: ``python -m greenlang.factors.cli_validate``.

    For the unified ``gl factors validate`` UX, the subparser is wired
    into :mod:`greenlang.factors.cli` via :func:`build_validate_parser`.
    """
    parser = argparse.ArgumentParser(
        prog="gl-factors-validate",
        description="GreenLang Factors v0.1 publish-gate validator (WS8)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_validate_parser(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
