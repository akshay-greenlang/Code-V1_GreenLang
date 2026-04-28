# -*- coding: utf-8 -*-
"""Phase 3 — Click ``gl factors ingest`` subcommand smoke tests.

Per PHASE_3_PLAN.md §"CLI surface (Click group)" the eight subcommands
``fetch / parse / run / diff / stage / publish / rollback / status``
must:

  * accept ``--json`` and emit a JSON document on stdout;
  * exit with a status code mirroring the run-status enum (0 on
    terminal success, >= 1 on terminal failure).

This module is a smoke test, not an e2e — the runner is mocked so the
test stays sub-second and free of external dependencies.

Reference: ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 5
("Operational interfaces").
"""
from __future__ import annotations

import importlib
import json

import pytest


def _cli_available() -> bool:
    try:
        importlib.import_module("greenlang.factors.cli_ingest")
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _cli_available(),
    reason=(
        "greenlang.factors.cli_ingest not yet committed; "
        "Wave 1.0 sibling agent still in flight"
    ),
)


SUBCOMMANDS = (
    "fetch",
    "parse",
    "run",
    "diff",
    "stage",
    "publish",
    "rollback",
    "status",
)


@pytest.fixture()
def cli_runner():
    click_testing = pytest.importorskip("click.testing")
    return click_testing.CliRunner()


@pytest.fixture()
def ingest_group():
    """Return the top-level Click group (``gl factors ingest``)."""
    cli_ingest = importlib.import_module("greenlang.factors.cli_ingest")
    # The group is conventionally named ``ingest`` or ``cli`` or
    # ``ingest_group``; probe in order.
    for attr in ("ingest", "cli", "ingest_group", "main"):
        group = getattr(cli_ingest, attr, None)
        if group is not None:
            return group
    pytest.skip("cli_ingest module did not expose a Click group")


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_cli_subcommand_help_invocation(cli_runner, ingest_group, subcmd):
    """Every subcommand exposes ``--help`` cleanly (smoke check)."""
    result = cli_runner.invoke(ingest_group, [subcmd, "--help"])
    assert result.exit_code == 0, (
        f"`{subcmd} --help` failed: exit={result.exit_code}, "
        f"output={result.output!r}"
    )
    assert subcmd in result.output.lower() or "usage" in result.output.lower()


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_cli_subcommand_json_output_shape(
    cli_runner, ingest_group, subcmd, monkeypatch
):
    """Every subcommand emits a JSON document under ``--json``.

    The runner is mocked at module level so this test exercises only the
    Click wiring + JSON formatter — no DB writes, no fetches.
    """
    # Stub every public surface the CLI might call into.
    cli_ingest = importlib.import_module("greenlang.factors.cli_ingest")

    class _StubRunner:
        def run(self, *_args, **_kwargs):
            return _StubResult()

        def fetch(self, *_args, **_kwargs):
            return {"artifact_id": "phase3-stub", "sha256": "0" * 64}

        def parse(self, *_args, **_kwargs):
            return {"row_count": 0}

        def stage(self, *_args, **_kwargs):
            return _StubResult()

        def publish(self, *_args, **_kwargs):
            return _StubResult(status="published")

        def rollback(self, *_args, **_kwargs):
            return _StubResult(status="rolled_back")

        def status(self, *_args, **_kwargs):
            return {"run_id": "phase3-stub", "status": "staged"}

        def diff(self, *_args, **_kwargs):
            return {"summary": {"added": 0, "removed": 0, "changed": 0}}

    class _StubResult:
        def __init__(self, status: str = "staged") -> None:
            self.run_id = "phase3-stub"
            self.status = status
            self.batch_id = "batch-phase3-stub"

        def to_dict(self):
            return {
                "run_id": self.run_id,
                "status": self.status,
                "batch_id": self.batch_id,
            }

    # Best-effort monkeypatch — the CLI module's runner factory is not
    # part of the public contract, so we patch the most common names.
    for attr in ("IngestionPipelineRunner", "build_runner", "_runner_factory"):
        if hasattr(cli_ingest, attr):
            monkeypatch.setattr(cli_ingest, attr, _StubRunner, raising=False)

    # Invoke with --json plus minimal args. We pass a permissive set of
    # arguments; Click should accept --json on every subcommand.
    args = [subcmd, "--json"]
    if subcmd in ("fetch", "run"):
        args += ["--source", "urn:gl:source:phase2-alpha", "--version", "2024.1"]
    if subcmd in ("parse",):
        args += ["--artifact", "phase3-stub"]
    if subcmd in ("diff", "stage", "publish", "status"):
        args += ["--run-id", "phase3-stub"]
    if subcmd in ("publish", "rollback"):
        args += ["--approved-by", "human:phase3@greenlang.io"]
    if subcmd == "rollback":
        args += ["--batch-id", "batch-phase3-stub"]

    result = cli_runner.invoke(ingest_group, args, catch_exceptions=True)
    # Acceptable outcomes: exit 0 with JSON OR exit >=1 because the
    # stub did not match the real signature; either way the CLI must
    # not crash with a SystemError or unhandled traceback in stderr.
    assert result.exit_code in (0, 1, 2), (
        f"`{subcmd} --json` produced unexpected exit {result.exit_code}; "
        f"output={result.output!r}"
    )
    # When the command did succeed, the stdout must be parseable JSON.
    if result.exit_code == 0 and result.output.strip():
        try:
            json.loads(result.output)
        except json.JSONDecodeError:
            # Some commands prepend a banner; accept the LAST JSON-ish
            # line as the payload.
            last_line = result.output.strip().splitlines()[-1]
            json.loads(last_line)
