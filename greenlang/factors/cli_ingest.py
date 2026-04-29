# -*- coding: utf-8 -*-
"""``gl factors ingest`` — Phase 3 raw-ingestion Click subcommand group.

This module exposes the eight Phase 3 ingestion subcommands as a Click
group. It deliberately leaves the legacy argparse subcommands in
:mod:`greenlang.factors.cli` (``ingest-builtin``, ``ingest-paths``,
``bulk-ingest``) untouched — those may be migrated in a later wave.

Click was chosen (rather than extending the existing argparse tree)
because the ingestion CLI must compose into the top-level ``gl`` Typer
application via ``typer.main.get_command``/``add_typer``, and Click
groups travel cleanly across that boundary.

Design contract (see ``docs/factors/PHASE_3_PLAN.md``):

* Every command supports a global ``--json`` flag. In JSON mode each
  command emits **exactly one line** of JSON to stdout on success and
  **exactly one line** of JSON to stderr on failure. In non-JSON mode
  the same data is rendered as a short human-readable summary.
* Every command supports ``--dsn`` (override the run-repository DSN —
  sqlite or postgres) and ``--verbose``/``-v`` (bump log level to DEBUG).
* All imports of :mod:`greenlang.factors.ingestion.runner` and
  :mod:`greenlang.factors.ingestion.run_repository` happen **inside**
  the command callbacks. This keeps the CLI module importable even
  when the sibling runner agent hasn't shipped yet, and surfaces a
  clean error message instead of a crash on import.
* ``publish`` and ``rollback`` enforce that the approver string matches
  ``^human:[^@]+@.+\\..+$``. Bot operators are rejected with a non-zero
  exit code.
* ``run`` orchestrates stages 1–6 only. Stage 7 (publish) is **always**
  gated behind the explicit ``publish`` subcommand with a human
  approver — even when ``--auto-stage`` is given.

Exit codes:

* 0 — success / non-failing terminal state (``published``, ``staged``,
  ``review_required``, ``created`` in-progress).
* 1 — run rejected (validation gates failed).
* 2 — run failed (pipeline error / fetcher / parser exception).
* 3 — run rolled back.
* 4 — unknown ``run_id`` / ``batch_id``.
* 5 — invocation error (missing args, bad approver, runner unavailable).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click

logger = logging.getLogger(__name__)

__all__ = [
    "ingest_group",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_HUMAN_APPROVER_RE = re.compile(r"^human:[^@\s]+@[^@\s]+\.[^@\s]+$")

# Map run-status enum values to CLI exit codes. Successes (incl. in-progress
# states the caller can resume) map to 0; explicit terminal failures get
# distinct non-zero codes so CI can differentiate.
_STATUS_EXIT_CODES: Dict[str, int] = {
    "created": 0,
    "fetched": 0,
    "parsed": 0,
    "normalized": 0,
    "validated": 0,
    "deduped": 0,
    "staged": 0,
    "review_required": 0,
    "published": 0,
    "rejected": 1,
    "failed": 2,
    "rolled_back": 3,
}

_PUBLISHABLE_STATUSES = {"staged", "review_required"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    """Configure stdlib logging for the CLI process.

    ``verbose=True`` raises the root level to DEBUG; otherwise the CLI
    runs at INFO. Idempotent — calling twice in the same process is safe.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        )
    else:
        root.setLevel(level)


def _emit_success(payload: Dict[str, Any], *, json_mode: bool, human_lines: Iterable[str]) -> None:
    """Emit a success payload. Always sets ``ok=True`` in JSON mode."""
    if json_mode:
        body = {"ok": True, **payload}
        click.echo(json.dumps(body, default=str, separators=(",", ":")), nl=True)
        return
    for line in human_lines:
        click.echo(line)


def _emit_error(
    *,
    json_mode: bool,
    error: str,
    stage: Optional[str] = None,
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured error payload to stderr."""
    if json_mode:
        body: Dict[str, Any] = {"ok": False, "error": error}
        if stage is not None:
            body["stage"] = stage
        if run_id is not None:
            body["run_id"] = run_id
        if extra:
            body.update(extra)
        click.echo(json.dumps(body, default=str, separators=(",", ":")), err=True, nl=True)
        return
    prefix = f"error [{stage}] " if stage else "error: "
    click.echo(f"{prefix}{error}", err=True)
    if run_id:
        click.echo(f"  run_id: {run_id}", err=True)


def _exit_for_status(status: Optional[str]) -> int:
    """Return the CLI exit code for a given run-status string."""
    if status is None:
        return 4
    return _STATUS_EXIT_CODES.get(status, 5)


def _approver_ok(approver: str) -> bool:
    """Return ``True`` iff the approver string is a well-formed human email."""
    return bool(_HUMAN_APPROVER_RE.match(approver or ""))


def _load_runner(dsn: Optional[str]) -> Tuple[Any, Any]:
    """Lazily import the ingestion runner + repository.

    Returns ``(runner, repo)``. Raises ``click.ClickException`` when the
    sibling agent hasn't shipped the runner yet — the CLI module itself
    stays importable.
    """
    try:
        from greenlang.factors.ingestion.runner import IngestionPipelineRunner
        from greenlang.factors.ingestion.run_repository import IngestionRunRepository
    except ImportError as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"ingestion runner not available yet: {exc}. "
            "Phase 3 task #26 (runner) ships before this CLI can execute."
        ) from exc

    repo = IngestionRunRepository(dsn=dsn) if dsn else IngestionRunRepository()
    runner = IngestionPipelineRunner(repository=repo)
    return runner, repo


def _run_to_dict(run: Any) -> Dict[str, Any]:
    """Coerce an :class:`IngestionRun` (or shape-equivalent) to a dict.

    The runner module is expected to expose either a Pydantic-like
    ``.model_dump()`` / ``.dict()`` method or a plain dataclass.
    """
    if run is None:
        return {}
    for attr in ("model_dump", "dict", "to_dict"):
        m = getattr(run, attr, None)
        if callable(m):
            try:
                return dict(m())
            except Exception:  # noqa: BLE001 — fall through to other shapes
                continue
    if hasattr(run, "__dict__"):
        return {k: v for k, v in vars(run).items() if not k.startswith("_")}
    if isinstance(run, dict):
        return dict(run)
    return {"value": str(run)}


# ---------------------------------------------------------------------------
# Click group + global option decorator
# ---------------------------------------------------------------------------


def _global_options(func: Any) -> Any:
    """Decorator: attach the three global flags (``--json``, ``--dsn``, ``-v``)."""
    func = click.option(
        "--json",
        "json_mode",
        is_flag=True,
        default=False,
        help="Emit a single-line JSON payload instead of human text.",
    )(func)
    func = click.option(
        "--dsn",
        default=None,
        help="Override the run-repository DSN (sqlite:///… or postgresql://…).",
    )(func)
    func = click.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        help="Bump log level to DEBUG.",
    )(func)
    return func


@click.group(name="ingest", help="Phase 3 raw-ingestion pipeline (8 subcommands).")
def ingest_group() -> None:
    """Click group for ``gl factors ingest``."""
    # No-op: subcommands carry the work. This callback exists so Click
    # treats ``ingest`` as a grouping prefix.
    return None


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


@ingest_group.command("fetch", help="Stage 1 — fetch raw artifact for a source.")
@click.option("--source", "source", required=True, help="Source id (e.g. defra-2025).")
@click.option("--version", "version", required=True, help="Source version label.")
@click.option(
    "--operator",
    default="bot:cli-fetch",
    show_default=True,
    help="Operator string ('bot:<id>' for automation, 'human:<email>' for manual).",
)
@_global_options
def fetch_cmd(
    source: str,
    version: str,
    operator: str,
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Run pipeline stage 1 (fetch) and persist the raw artifact."""
    _configure_logging(verbose)
    try:
        runner, _ = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    try:
        result = runner.fetch(source_id=source, source_version=version, operator=operator)
    except Exception as exc:  # noqa: BLE001 — surface any pipeline error
        logger.exception("fetch failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="fetch")
        sys.exit(2)

    payload = {
        "run_id": getattr(result, "run_id", None),
        "artifact_id": getattr(result, "artifact_id", None),
        "sha256": getattr(result, "sha256", None),
        "bytes_size": getattr(result, "bytes_size", None),
        "source_url": getattr(result, "source_url", None),
        "fetched_at": getattr(result, "fetched_at", None),
        "status": getattr(result, "status", None),
    }
    human = [
        f"fetched: source={source}@{version}",
        f"  run_id     : {payload['run_id']}",
        f"  artifact_id: {payload['artifact_id']}",
        f"  sha256     : {payload['sha256']}",
        f"  bytes_size : {payload['bytes_size']}",
        f"  status     : {payload['status']}",
    ]
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(payload.get("status")))


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


@ingest_group.command("parse", help="Stage 2 — parse a previously fetched artifact.")
@click.option("--artifact", "artifact_id", required=True, help="Artifact id (from fetch).")
@click.option(
    "--operator",
    default="bot:cli-parse",
    show_default=True,
    help="Operator string ('bot:<id>' or 'human:<email>').",
)
@_global_options
def parse_cmd(
    artifact_id: str,
    operator: str,
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Run pipeline stage 2 (parse) over a stored artifact."""
    _configure_logging(verbose)
    try:
        runner, _ = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    try:
        result = runner.parse(artifact_id=artifact_id, operator=operator)
    except Exception as exc:  # noqa: BLE001
        logger.exception("parse failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="parse")
        sys.exit(2)

    payload = {
        "run_id": getattr(result, "run_id", None),
        "artifact_id": artifact_id,
        "parsed_row_count": getattr(result, "parsed_row_count", None),
        "sheets": getattr(result, "sheets", None),
        "parser_version": getattr(result, "parser_version", None),
        "parser_commit": getattr(result, "parser_commit", None),
        "status": getattr(result, "status", None),
    }
    human = [
        f"parsed: artifact={artifact_id}",
        f"  run_id          : {payload['run_id']}",
        f"  parsed_row_count: {payload['parsed_row_count']}",
        f"  parser_version  : {payload['parser_version']}",
        f"  parser_commit   : {payload['parser_commit']}",
        f"  status          : {payload['status']}",
    ]
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(payload.get("status")))


# ---------------------------------------------------------------------------
# run (stages 1-6, never 7)
# ---------------------------------------------------------------------------


@ingest_group.command("run", help="Stages 1-6 (fetch through stage+diff). Never publishes.")
@click.option("--source", "source", required=True, help="Source id.")
@click.option("--version", "version", required=True, help="Source version.")
@click.option(
    "--auto-stage",
    is_flag=True,
    default=False,
    help="Continue all the way through stage 6 even if a previous run is staged.",
)
@click.option(
    "--operator",
    default="bot:cli-run",
    show_default=True,
    help="Operator string.",
)
@click.option(
    "--from-stage",
    "from_stage",
    type=click.Choice(
        ["parse", "normalize", "validate", "dedupe", "stage"],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Resume mode: rerun an existing failed/rejected run starting "
        "from this stage. Requires --run-id. The stage's predecessor "
        "must match the failed run's current_stage (Phase 3 audit gap B)."
    ),
)
@click.option(
    "--run-id",
    "run_id",
    default=None,
    help="Existing run id to resume (only used with --from-stage).",
)
@_global_options
def run_cmd(
    source: str,
    version: str,
    auto_stage: bool,
    operator: str,
    from_stage: Optional[str],
    run_id: Optional[str],
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Orchestrate stages 1-6. Stage 7 ALWAYS requires explicit publish.

    Resume mode (Phase 3 audit gap B): pass ``--from-stage <name>`` plus
    ``--run-id <existing>`` to re-execute a previously-failed run from
    the named stage onward. The runner re-validates the stage-precondition
    matrix and refuses if the run isn't in ``failed`` / ``rejected`` or
    if the requested stage's predecessor doesn't match the failed
    ``current_stage``.
    """
    _configure_logging(verbose)
    try:
        runner, _ = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    # --- Resume mode (Phase 3 audit gap B) ---
    if from_stage is not None:
        if not run_id:
            _emit_error(
                json_mode=json_mode,
                error="--from-stage requires --run-id <existing-run-id>",
                stage="run",
            )
            sys.exit(5)
        try:
            run = runner.resume(
                run_id, from_stage=from_stage, source_id=source
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("resume failed")
            _emit_error(
                json_mode=json_mode, error=str(exc), stage="resume", run_id=run_id
            )
            sys.exit(2)
        run_dict = _run_to_dict(run)
        diff_uri = run_dict.get("diff_uri") or run_dict.get("diff_artifact_uri")
        payload = {**run_dict, "diff_uri": diff_uri, "resumed_from": from_stage}
        status = payload.get("status")
        human = [
            f"resume: run_id={run_id} from_stage={from_stage}",
            f"  status   : {status}",
            f"  diff_uri : {diff_uri}",
        ]
        _emit_success(payload, json_mode=json_mode, human_lines=human)
        sys.exit(_exit_for_status(status))

    try:
        run = runner.run(
            source_id=source,
            source_version=version,
            auto_stage=auto_stage,
            operator=operator,
            stop_after_stage=6,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="run")
        sys.exit(2)

    run_dict = _run_to_dict(run)
    diff_uri = run_dict.get("diff_uri") or run_dict.get("diff_artifact_uri")
    payload = {**run_dict, "diff_uri": diff_uri}
    status = payload.get("status")
    human = [
        f"run: source={source}@{version} auto_stage={auto_stage}",
        f"  run_id   : {payload.get('run_id')}",
        f"  status   : {status}",
        f"  diff_uri : {diff_uri}",
    ]
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(status))


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@ingest_group.command("diff", help="Stage 6 replay — emit MD or JSON diff for a run.")
@click.option("--run-id", "run_id", required=True, help="Ingestion run id.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["md", "json"], case_sensitive=False),
    default="md",
    show_default=True,
    help="Diff output format.",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Write to this file path. Default: stdout.",
)
@_global_options
def diff_cmd(
    run_id: str,
    fmt: str,
    out_path: Optional[str],
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Replay stage-6 diff output (Markdown by default, deterministic JSON optional)."""
    _configure_logging(verbose)
    try:
        runner, _ = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    try:
        diff_payload = runner.diff(run_id=run_id, fmt=fmt.lower())
    except LookupError:
        _emit_error(json_mode=json_mode, error=f"run not found: {run_id}", stage="diff", run_id=run_id)
        sys.exit(4)
    except Exception as exc:  # noqa: BLE001
        logger.exception("diff failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="diff", run_id=run_id)
        sys.exit(2)

    # The runner returns either a string (md) or a JSON-able object (json).
    if fmt.lower() == "json":
        rendered = (
            diff_payload
            if isinstance(diff_payload, str)
            else json.dumps(diff_payload, default=str, sort_keys=True, indent=2)
        )
    else:
        rendered = diff_payload if isinstance(diff_payload, str) else str(diff_payload)

    if out_path:
        from pathlib import Path  # local import keeps top-of-file lean
        Path(out_path).write_text(rendered, encoding="utf-8")
        payload = {"run_id": run_id, "format": fmt.lower(), "out": out_path, "bytes": len(rendered)}
        _emit_success(
            payload,
            json_mode=json_mode,
            human_lines=[f"diff written: {out_path} ({len(rendered)} bytes, format={fmt.lower()})"],
        )
        sys.exit(0)

    if json_mode:
        # In JSON mode we emit a single envelope; the diff body is nested.
        envelope = {"ok": True, "run_id": run_id, "format": fmt.lower(), "diff": rendered}
        click.echo(json.dumps(envelope, default=str, separators=(",", ":")))
    else:
        click.echo(rendered)
    sys.exit(0)


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


@ingest_group.command("stage", help="Stage 6 explicit — promote validated/deduped run to staged.")
@click.option("--run-id", "run_id", required=True, help="Ingestion run id.")
@_global_options
def stage_cmd(
    run_id: str,
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Run stage 6 over an existing validated/deduped run."""
    _configure_logging(verbose)
    try:
        runner, _ = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    try:
        run = runner.stage(run_id=run_id)
    except LookupError:
        _emit_error(json_mode=json_mode, error=f"run not found: {run_id}", stage="stage", run_id=run_id)
        sys.exit(4)
    except Exception as exc:  # noqa: BLE001
        logger.exception("stage failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="stage", run_id=run_id)
        sys.exit(2)

    run_dict = _run_to_dict(run)
    status = run_dict.get("status")
    payload = {"run_id": run_id, **run_dict}
    human = [
        f"staged: run_id={run_id}",
        f"  status  : {status}",
        f"  diff_uri: {run_dict.get('diff_uri') or run_dict.get('diff_artifact_uri')}",
    ]
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(status))


# ---------------------------------------------------------------------------
# publish (stage 7)
# ---------------------------------------------------------------------------


@ingest_group.command("publish", help="Stage 7 — atomic publish a staged run (human approval required).")
@click.option("--run-id", "run_id", required=True, help="Ingestion run id.")
@click.option(
    "--approved-by",
    "approved_by",
    required=True,
    help="Approver — must match 'human:<email>' (bots are rejected).",
)
@_global_options
def publish_cmd(
    run_id: str,
    approved_by: str,
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Stage 7 publish. Refuses bots; refuses runs that aren't staged/review_required."""
    _configure_logging(verbose)
    if not _approver_ok(approved_by):
        _emit_error(
            json_mode=json_mode,
            error=(
                "approver must match 'human:<email>'; bots cannot publish. "
                f"got: {approved_by!r}"
            ),
            stage="publish",
            run_id=run_id,
        )
        sys.exit(5)

    try:
        runner, repo = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    # Pre-flight: refuse if the run is not in a publishable state.
    try:
        current = repo.get_run(run_id)
    except LookupError:
        _emit_error(json_mode=json_mode, error=f"run not found: {run_id}", stage="publish", run_id=run_id)
        sys.exit(4)
    except Exception as exc:  # noqa: BLE001
        logger.exception("publish preflight failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="publish", run_id=run_id)
        sys.exit(2)

    current_status = getattr(current, "status", None) or _run_to_dict(current).get("status")
    if current_status not in _PUBLISHABLE_STATUSES:
        _emit_error(
            json_mode=json_mode,
            error=(
                f"refuse to publish run in status {current_status!r}; "
                f"must be one of {sorted(_PUBLISHABLE_STATUSES)}"
            ),
            stage="publish",
            run_id=run_id,
        )
        sys.exit(1)

    try:
        published = runner.publish(run_id=run_id, approver=approved_by)
    except Exception as exc:  # noqa: BLE001
        logger.exception("publish failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="publish", run_id=run_id)
        sys.exit(2)

    pub_dict = _run_to_dict(published)
    status = pub_dict.get("status") or "published"
    payload = {"run_id": run_id, "approved_by": approved_by, **pub_dict}
    human = [
        f"published: run_id={run_id}",
        f"  approved_by: {approved_by}",
        f"  batch_id   : {pub_dict.get('batch_id')}",
        f"  status     : {status}",
    ]
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(status))


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


@ingest_group.command("rollback", help="Demote a published batch back to staged (metadata flip).")
@click.option("--batch-id", "batch_id", required=True, help="Publish batch id.")
@click.option(
    "--approved-by",
    "approved_by",
    required=True,
    help="Approver — must match 'human:<email>'.",
)
@_global_options
def rollback_cmd(
    batch_id: str,
    approved_by: str,
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Demote a published batch. Phase 2 publish gates are NOT re-run."""
    _configure_logging(verbose)
    if not _approver_ok(approved_by):
        _emit_error(
            json_mode=json_mode,
            error=(
                "approver must match 'human:<email>'; bots cannot rollback. "
                f"got: {approved_by!r}"
            ),
            stage="rollback",
        )
        sys.exit(5)

    try:
        runner, repo = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    # Pre-flight: only published batches can be rolled back.
    try:
        batch = repo.get_batch(batch_id)
    except LookupError:
        _emit_error(json_mode=json_mode, error=f"batch not found: {batch_id}", stage="rollback")
        sys.exit(4)
    except Exception as exc:  # noqa: BLE001
        logger.exception("rollback preflight failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="rollback")
        sys.exit(2)

    batch_status = getattr(batch, "status", None) or _run_to_dict(batch).get("status")
    if batch_status != "published":
        _emit_error(
            json_mode=json_mode,
            error=(
                f"refuse to rollback batch in status {batch_status!r}; "
                "must be 'published'"
            ),
            stage="rollback",
        )
        sys.exit(1)

    try:
        result = runner.rollback(batch_id=batch_id, approver=approved_by)
    except Exception as exc:  # noqa: BLE001
        logger.exception("rollback failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="rollback")
        sys.exit(2)

    res_dict = _run_to_dict(result)
    status = res_dict.get("status") or "rolled_back"
    payload = {"batch_id": batch_id, "approved_by": approved_by, **res_dict}
    human = [
        f"rolled back: batch_id={batch_id}",
        f"  approved_by: {approved_by}",
        f"  status     : {status}",
    ]
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(status))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@ingest_group.command("status", help="Show current state machine state + last error for a run.")
@click.option("--run-id", "run_id", required=True, help="Ingestion run id.")
@_global_options
def status_cmd(
    run_id: str,
    json_mode: bool,
    dsn: Optional[str],
    verbose: bool,
) -> None:
    """Read the current run row and print state-machine progress."""
    _configure_logging(verbose)
    try:
        _, repo = _load_runner(dsn)
    except click.ClickException as exc:
        _emit_error(json_mode=json_mode, error=exc.message, stage="bootstrap")
        sys.exit(5)

    try:
        run = repo.get_run(run_id)
    except LookupError:
        _emit_error(json_mode=json_mode, error=f"run not found: {run_id}", stage="status", run_id=run_id)
        sys.exit(4)
    except Exception as exc:  # noqa: BLE001
        logger.exception("status read failed")
        _emit_error(json_mode=json_mode, error=str(exc), stage="status", run_id=run_id)
        sys.exit(2)

    run_dict = _run_to_dict(run)
    status = run_dict.get("status")
    payload = {"run_id": run_id, **run_dict}

    transitions: List[str] = list(run_dict.get("transitions") or [])
    last_error: Optional[str] = run_dict.get("last_error") or run_dict.get("error")

    human = [
        f"status: run_id={run_id}",
        f"  current   : {status}",
        f"  source    : {run_dict.get('source_id')}@{run_dict.get('source_version')}",
        f"  artifact  : {run_dict.get('artifact_id')}",
        f"  updated_at: {run_dict.get('updated_at')}",
        f"  last_error: {last_error or '-'}",
    ]
    if transitions:
        human.append("  history   :")
        for t in transitions[-10:]:
            human.append(f"    - {t}")
    _emit_success(payload, json_mode=json_mode, human_lines=human)
    sys.exit(_exit_for_status(status))


# ---------------------------------------------------------------------------
# Stand-alone entry-point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Stand-alone entry-point for ``python -m greenlang.factors.cli_ingest``.

    The Click group raises ``SystemExit`` itself, so we trap it and
    return the exit code so callers can use :func:`sys.exit` cleanly.
    """
    try:
        ingest_group.main(args=argv, standalone_mode=False)
        return 0
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
