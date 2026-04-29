# -*- coding: utf-8 -*-
"""Phase 3 Block 6 — branch-coverage tests for ``greenlang/factors/cli_ingest.py``.

Targets the residual ~54pp gap left by the existing smoke tests in
``test_cli_subcommands.py``. Each subcommand is parametrized across:

  * success path — stub runner returns a result-shaped object;
  * failure path — stub runner raises (mapped to exit 2);
  * ``LookupError`` path — exits 4 (run not found / batch not found);
  * ``--json`` mode — emits a single line of JSON to stdout / stderr;
  * approver-format check on ``publish`` / ``rollback`` (exit 5);
  * publishable-status pre-flight on ``publish`` (exit 1);
  * batch-status pre-flight on ``rollback`` (exit 1).

The module-level helper functions ``_emit_success``, ``_emit_error``,
``_exit_for_status``, ``_approver_ok``, ``_run_to_dict`` and
``_configure_logging`` are exercised directly so the helper coverage
does not depend on subcommand wiring.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import click
import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Module-level helpers — direct unit tests.
# ---------------------------------------------------------------------------


def test_approver_ok_accepts_well_formed_human():
    from greenlang.factors.cli_ingest import _approver_ok

    assert _approver_ok("human:lead@greenlang.io") is True
    assert _approver_ok("human:a.b@example.co.uk") is True


def test_approver_ok_rejects_bots_and_garbage():
    from greenlang.factors.cli_ingest import _approver_ok

    assert _approver_ok("bot:auto") is False
    assert _approver_ok("plain-string") is False
    assert _approver_ok("") is False
    assert _approver_ok("human:no-domain") is False
    assert _approver_ok("human:two@@x.com") is False


def test_exit_for_status_terminal_states():
    from greenlang.factors.cli_ingest import _exit_for_status

    assert _exit_for_status("published") == 0
    assert _exit_for_status("staged") == 0
    assert _exit_for_status("review_required") == 0
    assert _exit_for_status("rejected") == 1
    assert _exit_for_status("failed") == 2
    assert _exit_for_status("rolled_back") == 3
    assert _exit_for_status(None) == 4
    assert _exit_for_status("not-a-status") == 5


def test_exit_for_status_in_progress_states():
    from greenlang.factors.cli_ingest import _exit_for_status

    assert _exit_for_status("created") == 0
    assert _exit_for_status("fetched") == 0
    assert _exit_for_status("parsed") == 0
    assert _exit_for_status("normalized") == 0


def test_emit_success_json_mode():
    from greenlang.factors.cli_ingest import _emit_success

    runner = CliRunner()

    @click.command()
    def _c():
        _emit_success({"x": 1}, json_mode=True, human_lines=["ignored"])

    result = runner.invoke(_c)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {"ok": True, "x": 1}


def test_emit_success_human_mode_emits_lines():
    from greenlang.factors.cli_ingest import _emit_success

    runner = CliRunner()

    @click.command()
    def _c():
        _emit_success({"y": 2}, json_mode=False, human_lines=["line A", "line B"])

    result = runner.invoke(_c)
    assert result.exit_code == 0
    assert "line A" in result.output
    assert "line B" in result.output


def test_emit_error_json_mode_with_extras():
    from greenlang.factors.cli_ingest import _emit_error

    runner = CliRunner()

    @click.command()
    def _c():
        _emit_error(
            json_mode=True,
            error="boom",
            stage="fetch",
            run_id="r1",
            extra={"hint": "retry"},
        )

    result = runner.invoke(_c)
    # Click 8.3 directs stderr to result.stderr; combined output may
    # also surface the line. Look in either.
    body = result.stderr if result.stderr else result.output
    payload = json.loads(body.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert payload["error"] == "boom"
    assert payload["stage"] == "fetch"
    assert payload["run_id"] == "r1"
    assert payload["hint"] == "retry"


def test_emit_error_human_mode_with_run_id():
    from greenlang.factors.cli_ingest import _emit_error

    runner = CliRunner()

    @click.command()
    def _c():
        _emit_error(
            json_mode=False, error="oops", stage="parse", run_id="r99"
        )

    result = runner.invoke(_c)
    body = result.stderr if result.stderr else result.output
    assert "error [parse]" in body
    assert "r99" in body


def test_emit_error_human_mode_no_stage():
    from greenlang.factors.cli_ingest import _emit_error

    runner = CliRunner()

    @click.command()
    def _c():
        _emit_error(json_mode=False, error="standalone")

    result = runner.invoke(_c)
    body = result.stderr if result.stderr else result.output
    assert "error: standalone" in body


def test_run_to_dict_handles_pydantic_like_model_dump():
    from greenlang.factors.cli_ingest import _run_to_dict

    class _Model:
        def model_dump(self):
            return {"a": 1, "b": "two"}

    assert _run_to_dict(_Model()) == {"a": 1, "b": "two"}


def test_run_to_dict_handles_dict_method():
    from greenlang.factors.cli_ingest import _run_to_dict

    class _Old:
        def dict(self):
            return {"x": 9}

    assert _run_to_dict(_Old()) == {"x": 9}


def test_run_to_dict_falls_through_when_dump_raises():
    from greenlang.factors.cli_ingest import _run_to_dict

    class _Crashy:
        def model_dump(self):
            raise RuntimeError("nope")

        def to_dict(self):
            return {"recovered": True}

    assert _run_to_dict(_Crashy()) == {"recovered": True}


def test_run_to_dict_uses_vars_for_plain_object():
    from greenlang.factors.cli_ingest import _run_to_dict

    class _Plain:
        def __init__(self):
            self.run_id = "r1"
            self.status = "ok"
            self._private = "skip-me"

    out = _run_to_dict(_Plain())
    assert out == {"run_id": "r1", "status": "ok"}


def test_run_to_dict_handles_dict_input():
    from greenlang.factors.cli_ingest import _run_to_dict

    assert _run_to_dict({"k": "v"}) == {"k": "v"}


def test_run_to_dict_handles_none():
    from greenlang.factors.cli_ingest import _run_to_dict

    assert _run_to_dict(None) == {}


def test_run_to_dict_handles_primitive_value():
    from greenlang.factors.cli_ingest import _run_to_dict

    out = _run_to_dict(42)
    assert out == {"value": "42"}


def test_configure_logging_idempotent_at_info():
    from greenlang.factors.cli_ingest import _configure_logging

    _configure_logging(False)
    _configure_logging(False)  # second call is a no-op (handlers already present)
    assert logging.getLogger().level <= logging.INFO


def test_configure_logging_verbose_raises_to_debug():
    from greenlang.factors.cli_ingest import _configure_logging

    _configure_logging(True)
    assert logging.getLogger().level == logging.DEBUG
    # Reset for other tests.
    logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Subcommand tests — stubbed runner / repo via monkeypatching ``_load_runner``.
# ---------------------------------------------------------------------------


class _StubResult:
    """Generic shape — every CLI ``getattr(result, name, None)`` is a string."""

    def __init__(self, **kw: Any) -> None:
        self.run_id = kw.get("run_id", "stub-run-1")
        self.artifact_id = kw.get("artifact_id", "art-1")
        self.sha256 = kw.get("sha256", "0" * 64)
        self.bytes_size = kw.get("bytes_size", 100)
        self.source_url = kw.get("source_url", "file:///x")
        self.fetched_at = kw.get("fetched_at", "2026-04-28T12:00:00Z")
        self.status = kw.get("status", "fetched")
        self.parsed_row_count = kw.get("parsed_row_count", 5)
        self.sheets = kw.get("sheets", ["Sheet1"])
        self.parser_version = kw.get("parser_version", "1.0")
        self.parser_commit = kw.get("parser_commit", "abc")
        self.batch_id = kw.get("batch_id", "batch-1")

    def model_dump(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_")
        }


class _StubRunner:
    def __init__(self, *, raises: Optional[Exception] = None) -> None:
        self._raises = raises
        self.calls: List[str] = []

    def fetch(self, *, source_id, source_version, operator):
        self.calls.append("fetch")
        if self._raises:
            raise self._raises
        return _StubResult(status="fetched")

    def parse(self, *, artifact_id, operator):
        self.calls.append("parse")
        if self._raises:
            raise self._raises
        return _StubResult(status="parsed")

    def run(self, *, source_id, source_version, auto_stage, operator, stop_after_stage):
        self.calls.append("run")
        if self._raises:
            raise self._raises
        return _StubResult(status="staged")

    def diff(self, *, run_id, fmt):
        self.calls.append("diff")
        if self._raises:
            raise self._raises
        if fmt == "json":
            return {"summary": {"added": 1}}
        return "# diff markdown\n"

    def stage(self, *, run_id):
        self.calls.append("stage")
        if self._raises:
            raise self._raises
        return _StubResult(status="staged")

    def publish(self, *, run_id, approver):
        self.calls.append("publish")
        if self._raises:
            raise self._raises
        return _StubResult(status="published")

    def rollback(self, *, batch_id, approver):
        self.calls.append("rollback")
        if self._raises:
            raise self._raises
        return _StubResult(status="rolled_back")


class _StubRepo:
    def __init__(self, *, run_status: str = "staged", batch_status: str = "published") -> None:
        self._run_status = run_status
        self._batch_status = batch_status
        self._raise_get_run: Optional[Exception] = None
        self._raise_get_batch: Optional[Exception] = None

    def get_run(self, run_id: str) -> Any:
        if self._raise_get_run:
            raise self._raise_get_run
        return _StubResult(run_id=run_id, status=self._run_status)

    def get_batch(self, batch_id: str) -> Any:
        if self._raise_get_batch:
            raise self._raise_get_batch
        return _StubResult(batch_id=batch_id, status=self._batch_status)


@pytest.fixture()
def cli_runner():
    return CliRunner()


@pytest.fixture()
def stub_loader(monkeypatch):
    """Replace ``_load_runner`` with a factory returning configurable doubles."""
    state: Dict[str, Any] = {
        "runner": _StubRunner(),
        "repo": _StubRepo(),
    }

    def _factory(dsn=None):
        return state["runner"], state["repo"]

    monkeypatch.setattr(
        "greenlang.factors.cli_ingest._load_runner", _factory, raising=True,
    )
    return state


# -- fetch ------------------------------------------------------------------


def test_cli_fetch_success_json(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group,
        ["fetch", "--source", "src", "--version", "1", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["status"] == "fetched"


def test_cli_fetch_success_human(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["fetch", "--source", "src", "--version", "1"],
    )
    assert result.exit_code == 0
    assert "fetched: source=src@1" in result.output


def test_cli_fetch_runner_raises_exit_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=RuntimeError("network down"))
    result = cli_runner.invoke(
        ingest_group, ["fetch", "--source", "src", "--version", "1", "--json"],
    )
    assert result.exit_code == 2


def test_cli_fetch_missing_required_args_exits_with_2(cli_runner, stub_loader):
    """Click's missing-required exits with usage error code 2."""
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(ingest_group, ["fetch"])
    assert result.exit_code == 2


# -- parse ------------------------------------------------------------------


def test_cli_parse_success_json(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["parse", "--artifact", "art-1", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["status"] == "parsed"


def test_cli_parse_runner_raises(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=RuntimeError("parse boom"))
    result = cli_runner.invoke(
        ingest_group, ["parse", "--artifact", "art-1", "--json"],
    )
    assert result.exit_code == 2


def test_cli_parse_human_output(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["parse", "--artifact", "art-1"],
    )
    assert result.exit_code == 0
    assert "parsed: artifact=art-1" in result.output


# -- run --------------------------------------------------------------------


def test_cli_run_success_json(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group,
        ["run", "--source", "src", "--version", "1", "--auto-stage", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["status"] == "staged"


def test_cli_run_human_output(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["run", "--source", "src", "--version", "1"],
    )
    assert result.exit_code == 0
    assert "run: source=src@1" in result.output


def test_cli_run_runner_raises(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=RuntimeError("kaboom"))
    result = cli_runner.invoke(
        ingest_group, ["run", "--source", "src", "--version", "1"],
    )
    assert result.exit_code == 2


# -- diff -------------------------------------------------------------------


def test_cli_diff_md_to_stdout(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["diff", "--run-id", "r1", "--format", "md"],
    )
    assert result.exit_code == 0
    assert "diff markdown" in result.output


def test_cli_diff_json_to_stdout(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["diff", "--run-id", "r1", "--format", "json"],
    )
    assert result.exit_code == 0
    # Output starts with valid JSON.
    body = result.output.strip()
    assert body.startswith("{") or body.startswith("[")


def test_cli_diff_json_envelope_under_json_mode(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group,
        ["diff", "--run-id", "r1", "--format", "json", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["run_id"] == "r1"
    assert "diff" in payload


def test_cli_diff_writes_to_out_file(cli_runner, stub_loader, tmp_path):
    from greenlang.factors.cli_ingest import ingest_group

    out = tmp_path / "diff.md"
    result = cli_runner.invoke(
        ingest_group,
        ["diff", "--run-id", "r1", "--format", "md", "--out", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert "diff markdown" in out.read_text()


def test_cli_diff_lookup_error_exits_4(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=LookupError("nope"))
    result = cli_runner.invoke(
        ingest_group,
        ["diff", "--run-id", "r-missing", "--json"],
    )
    assert result.exit_code == 4


def test_cli_diff_runner_raises_exits_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=RuntimeError("disk down"))
    result = cli_runner.invoke(
        ingest_group,
        ["diff", "--run-id", "r1", "--json"],
    )
    assert result.exit_code == 2


# -- stage ------------------------------------------------------------------


def test_cli_stage_success(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["stage", "--run-id", "r1", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["ok"] is True


def test_cli_stage_lookup_error_exits_4(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=LookupError("missing"))
    result = cli_runner.invoke(
        ingest_group, ["stage", "--run-id", "r-missing"],
    )
    assert result.exit_code == 4


def test_cli_stage_runner_raises_exits_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=RuntimeError("flop"))
    result = cli_runner.invoke(
        ingest_group, ["stage", "--run-id", "r1"],
    )
    assert result.exit_code == 2


# -- publish ----------------------------------------------------------------


def test_cli_publish_rejects_bot_approver(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group,
        ["publish", "--run-id", "r1", "--approved-by", "bot:auto", "--json"],
    )
    assert result.exit_code == 5


def test_cli_publish_run_not_in_publishable_status(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["repo"] = _StubRepo(run_status="created")
    result = cli_runner.invoke(
        ingest_group,
        [
            "publish",
            "--run-id", "r1",
            "--approved-by", "human:lead@x.com",
            "--json",
        ],
    )
    assert result.exit_code == 1


def test_cli_publish_run_not_found(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    bad_repo = _StubRepo()
    bad_repo._raise_get_run = LookupError("missing")
    stub_loader["repo"] = bad_repo
    result = cli_runner.invoke(
        ingest_group,
        ["publish", "--run-id", "r-x", "--approved-by", "human:l@x.io", "--json"],
    )
    assert result.exit_code == 4


def test_cli_publish_preflight_repo_failure_exits_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    bad_repo = _StubRepo()
    bad_repo._raise_get_run = RuntimeError("db down")
    stub_loader["repo"] = bad_repo
    result = cli_runner.invoke(
        ingest_group,
        ["publish", "--run-id", "r1", "--approved-by", "human:l@x.io"],
    )
    assert result.exit_code == 2


def test_cli_publish_runner_failure_exits_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["repo"] = _StubRepo(run_status="staged")
    stub_loader["runner"] = _StubRunner(raises=RuntimeError("flip failed"))
    result = cli_runner.invoke(
        ingest_group,
        ["publish", "--run-id", "r1", "--approved-by", "human:l@x.io"],
    )
    assert result.exit_code == 2


def test_cli_publish_success_json(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["repo"] = _StubRepo(run_status="staged")
    result = cli_runner.invoke(
        ingest_group,
        [
            "publish",
            "--run-id", "r1",
            "--approved-by", "human:l@x.io",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["status"] == "published"


# -- rollback ---------------------------------------------------------------


def test_cli_rollback_rejects_bot(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group,
        [
            "rollback",
            "--batch-id", "b1",
            "--approved-by", "bot:auto",
            "--json",
        ],
    )
    assert result.exit_code == 5


def test_cli_rollback_batch_not_found(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    bad_repo = _StubRepo()
    bad_repo._raise_get_batch = LookupError("missing")
    stub_loader["repo"] = bad_repo
    result = cli_runner.invoke(
        ingest_group,
        [
            "rollback",
            "--batch-id", "bx",
            "--approved-by", "human:l@x.io",
            "--json",
        ],
    )
    assert result.exit_code == 4


def test_cli_rollback_batch_not_published(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["repo"] = _StubRepo(batch_status="staged")
    result = cli_runner.invoke(
        ingest_group,
        [
            "rollback",
            "--batch-id", "b1",
            "--approved-by", "human:l@x.io",
            "--json",
        ],
    )
    assert result.exit_code == 1


def test_cli_rollback_preflight_repo_failure(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    bad_repo = _StubRepo()
    bad_repo._raise_get_batch = RuntimeError("oops")
    stub_loader["repo"] = bad_repo
    result = cli_runner.invoke(
        ingest_group,
        [
            "rollback",
            "--batch-id", "b1",
            "--approved-by", "human:l@x.io",
        ],
    )
    assert result.exit_code == 2


def test_cli_rollback_runner_failure_exits_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    stub_loader["runner"] = _StubRunner(raises=RuntimeError("flop"))
    result = cli_runner.invoke(
        ingest_group,
        [
            "rollback",
            "--batch-id", "b1",
            "--approved-by", "human:l@x.io",
        ],
    )
    assert result.exit_code == 2


def test_cli_rollback_success_json(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group,
        [
            "rollback",
            "--batch-id", "b1",
            "--approved-by", "human:l@x.io",
            "--json",
        ],
    )
    assert result.exit_code == 3  # rolled_back exit code
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["status"] == "rolled_back"


# -- status -----------------------------------------------------------------


def test_cli_status_success(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(
        ingest_group, ["status", "--run-id", "r1", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["run_id"] == "r1"


def test_cli_status_lookup_error_exits_4(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    bad_repo = _StubRepo()
    bad_repo._raise_get_run = LookupError("missing")
    stub_loader["repo"] = bad_repo
    result = cli_runner.invoke(
        ingest_group,
        ["status", "--run-id", "r-x", "--json"],
    )
    assert result.exit_code == 4


def test_cli_status_repo_failure_exits_2(cli_runner, stub_loader):
    from greenlang.factors.cli_ingest import ingest_group

    bad_repo = _StubRepo()
    bad_repo._raise_get_run = RuntimeError("oops")
    stub_loader["repo"] = bad_repo
    result = cli_runner.invoke(
        ingest_group, ["status", "--run-id", "r1"],
    )
    assert result.exit_code == 2


def test_cli_status_renders_transitions_history():
    """Status should render the last 10 transitions if present."""
    from greenlang.factors.cli_ingest import ingest_group, _load_runner  # noqa: F401

    runner = CliRunner()

    class _RepoWithHistory:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "status": "staged",
                "source_id": "src",
                "source_version": "1",
                "transitions": [f"t-{i}" for i in range(15)],
                "last_error": None,
            }

    def _factory(dsn=None):
        return _StubRunner(), _RepoWithHistory()

    import greenlang.factors.cli_ingest as cli_ingest_mod
    original = cli_ingest_mod._load_runner
    cli_ingest_mod._load_runner = _factory
    try:
        result = runner.invoke(
            ingest_group, ["status", "--run-id", "r1"],
        )
    finally:
        cli_ingest_mod._load_runner = original
    assert result.exit_code == 0
    # The last 10 transitions are emitted; t-5..t-14 are present.
    assert "t-14" in result.output
    assert "t-5" in result.output


# -- main() entry-point + group help ---------------------------------------


def test_main_returns_zero_on_help():
    from greenlang.factors.cli_ingest import main

    rc = main(["--help"])
    assert rc == 0


def test_main_returns_nonzero_on_unknown_subcommand():
    from greenlang.factors.cli_ingest import main

    rc = main(["this-subcommand-does-not-exist"])
    assert rc != 0


def test_group_help_lists_subcommands(cli_runner):
    from greenlang.factors.cli_ingest import ingest_group

    result = cli_runner.invoke(ingest_group, ["--help"])
    assert result.exit_code == 0
    for s in ("fetch", "parse", "run", "diff", "stage", "publish", "rollback", "status"):
        assert s in result.output


# -- bootstrap failure (load_runner raises ClickException) ------------------


def test_cli_bootstrap_failure_propagates_exit_5(monkeypatch, cli_runner):
    """When ``_load_runner`` raises ``ClickException`` every command exits 5."""
    from greenlang.factors.cli_ingest import ingest_group

    def _bad_loader(dsn=None):
        raise click.ClickException("runner not built yet")

    monkeypatch.setattr(
        "greenlang.factors.cli_ingest._load_runner", _bad_loader, raising=True,
    )
    result = cli_runner.invoke(
        ingest_group, ["fetch", "--source", "s", "--version", "1", "--json"],
    )
    assert result.exit_code == 5


def test_cli_bootstrap_failure_for_each_subcommand(monkeypatch, cli_runner):
    """Smoke: every subcommand must surface bootstrap failure as exit 5."""
    from greenlang.factors.cli_ingest import ingest_group

    def _bad_loader(dsn=None):
        raise click.ClickException("kaboom")

    monkeypatch.setattr(
        "greenlang.factors.cli_ingest._load_runner", _bad_loader, raising=True,
    )
    invocations = (
        ["parse", "--artifact", "a", "--json"],
        ["run", "--source", "s", "--version", "1", "--json"],
        ["diff", "--run-id", "r", "--json"],
        ["stage", "--run-id", "r", "--json"],
        ["publish", "--run-id", "r", "--approved-by", "human:a@b.c", "--json"],
        ["rollback", "--batch-id", "b", "--approved-by", "human:a@b.c", "--json"],
        ["status", "--run-id", "r", "--json"],
    )
    for argv in invocations:
        r = cli_runner.invoke(ingest_group, argv)
        assert r.exit_code == 5, f"{argv!r} → exit {r.exit_code}"
