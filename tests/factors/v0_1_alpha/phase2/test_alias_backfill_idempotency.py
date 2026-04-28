# -*- coding: utf-8 -*-
"""Phase 2 / WS2 — Alias backfill idempotency tests.

Per CTO Phase 2 brief Section 2.7 acceptance:

    "test_alias_backfill_idempotency.py — run the backfill twice,
    assert second run is a no-op."

The backfill script is at
``scripts/factors/phase2_backfill_factor_aliases.py``. It walks the
factor table, harvests every ``factor_id_alias`` blob field, and
INSERTs a row into the alias mirror with ``ON CONFLICT (legacy_id)
DO NOTHING`` semantics.

Tests in this module:

  1. **First-run inserts**: a freshly populated factor table with N
     records that carry ``factor_id_alias`` produces N inserted rows
     and 0 conflicts.
  2. **Second-run no-op**: re-running the script on the same DB
     produces 0 inserted rows; conflicts == N (because every alias is
     already present).
  3. **Records without ``factor_id_alias``**: rows whose blob has no
     alias key are counted under ``skipped``, never ``conflicts``.
  4. **Dry-run**: ``--dry-run`` mode reports the same proposed
     ``(urn, legacy_id)`` tuples on every invocation but never
     mutates the DB.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Make the script importable as ``phase2_backfill_factor_aliases``.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "factors"
sys.path.insert(0, str(_SCRIPT_DIR))

from greenlang.factors.repositories.alpha_v0_1_repository import (  # noqa: E402
    AlphaFactorRepository,
)

import phase2_backfill_factor_aliases as backfill_mod  # noqa: E402


def _record(urn: str, *, alias: str | None) -> Dict[str, Any]:
    """v0.1 factor record that passes the AlphaProvenanceGate."""
    rec: Dict[str, Any] = {
        "urn": urn,
        "source_urn": "urn:gl:source:ipcc-2006-nggi",
        "factor_pack_urn": "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1",
        "name": "Phase 2 backfill idempotency record",
        "description": (
            "Synthetic v0.1 factor used to exercise the backfill script's "
            "idempotency contract. Contains a populated factor_id_alias "
            "field iff alias is provided."
        ),
        "category": "fuel",
        "value": 100.0,
        "unit_urn": "urn:gl:unit:kgco2e/gj",
        "gwp_basis": "ar6",
        "gwp_horizon": 100,
        "geography_urn": "urn:gl:geo:global:world",
        "vintage_start": "2024-01-01",
        "vintage_end": "2099-12-31",
        "resolution": "annual",
        "methodology_urn": "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
        "boundary": "combustion",
        "licence": "IPCC-PUBLIC",
        "citations": [
            {"type": "url", "value": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/"}
        ],
        "published_at": "2026-04-25T07:42:30+00:00",
        "extraction": {
            "source_url": "https://www.ipcc-nggip.iges.or.jp/public/2019rf/",
            "source_record_id": "phase2-backfill-record",
            "source_publication": "Phase 2 / WS2 backfill fixture",
            "source_version": "0.1",
            "raw_artifact_uri": "s3://greenlang-factors-raw/test/phase2-backfill.json",
            "raw_artifact_sha256": "6ff38c51f0ffcb08b2057b90164c3f3e6b67a16bacffb27507526b4dab1271c6",
            "parser_id": "tests.factors.v0_1_alpha.phase2.backfill",
            "parser_version": "0.1.0",
            "parser_commit": "0000000000000000000000000000000000000000",
            "row_ref": "phase2-backfill-record",
            "ingested_at": "2026-04-25T07:42:30Z",
            "operator": "bot:test_alias_backfill_idempotency",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T07:42:30Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T07:42:30Z",
        },
        "tags": ["phase2", "backfill"],
    }
    if alias:
        rec["factor_id_alias"] = alias
    return rec


@pytest.fixture()
def populated_db(tmp_path):
    """Populate a SQLite DB with three factor records — two with
    aliases, one without — and return the DSN."""
    db_path = tmp_path / "phase2_backfill.db"
    dsn = f"sqlite:///{db_path}"

    # legacy mode — Phase 1 provenance gate only; Phase 2 orchestrator covered by tests/factors/v0_1_alpha/phase2/test_publish_pipeline_e2e.py
    repo = AlphaFactorRepository(dsn=dsn, publish_env="legacy")
    try:
        repo.publish(
            _record(
                "urn:gl:factor:ipcc-2006-nggi:phase2:bf-record-1:v1",
                alias="EF:phase2:bf-1:v1",
            )
        )
        repo.publish(
            _record(
                "urn:gl:factor:ipcc-2006-nggi:phase2:bf-record-2:v1",
                alias="EF:phase2:bf-2:v1",
            )
        )
        repo.publish(
            _record(
                "urn:gl:factor:ipcc-2006-nggi:phase2:bf-record-3:v1",
                alias=None,  # No alias to backfill — must count as 'skipped'.
            )
        )
    finally:
        repo.close()
    yield dsn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackfillIdempotency:
    """End-to-end backfill idempotency."""

    def test_first_run_inserts_aliases(self, populated_db: str) -> None:
        """The first run must INSERT one alias per factor with a
        ``factor_id_alias`` field."""
        result = backfill_mod.backfill(populated_db, dry_run=False)
        assert result.scanned == 3
        assert result.inserted == 2
        assert result.skipped == 1
        assert result.conflicts == 0

    def test_second_run_is_noop(self, populated_db: str) -> None:
        """Running the backfill a second time must INSERT nothing —
        every alias is already present, so the rows count under
        ``conflicts``."""
        first = backfill_mod.backfill(populated_db, dry_run=False)
        assert first.inserted == 2
        second = backfill_mod.backfill(populated_db, dry_run=False)
        assert second.scanned == 3
        assert second.inserted == 0
        assert second.skipped == 1
        assert second.conflicts == 2

    def test_third_run_remains_noop(self, populated_db: str) -> None:
        """Three+ runs are all no-ops — no drift, no double-insert."""
        backfill_mod.backfill(populated_db, dry_run=False)
        backfill_mod.backfill(populated_db, dry_run=False)
        third = backfill_mod.backfill(populated_db, dry_run=False)
        assert third.inserted == 0
        assert third.conflicts == 2
        assert third.skipped == 1


class TestBackfillDryRun:
    """``--dry-run`` mode never mutates the DB."""

    def test_dry_run_proposes_inserts_without_writing(
        self, populated_db: str
    ) -> None:
        result = backfill_mod.backfill(populated_db, dry_run=True)
        assert result.inserted == 0
        assert len(result.proposed) == 2
        # Sort for deterministic comparison.
        proposed_sorted = sorted(result.proposed, key=lambda t: t[1])
        assert proposed_sorted == [
            (
                "urn:gl:factor:ipcc-2006-nggi:phase2:bf-record-1:v1",
                "EF:phase2:bf-1:v1",
            ),
            (
                "urn:gl:factor:ipcc-2006-nggi:phase2:bf-record-2:v1",
                "EF:phase2:bf-2:v1",
            ),
        ]

    def test_dry_run_does_not_block_subsequent_real_run(
        self, populated_db: str
    ) -> None:
        """A dry-run followed by a real run must still INSERT — the
        dry-run leaves no side effects in the DB."""
        dry = backfill_mod.backfill(populated_db, dry_run=True)
        real = backfill_mod.backfill(populated_db, dry_run=False)
        assert dry.inserted == 0
        assert real.inserted == 2

    def test_dry_run_after_real_run_is_clean(
        self, populated_db: str
    ) -> None:
        """After a real backfill, a dry-run reports zero proposed rows."""
        backfill_mod.backfill(populated_db, dry_run=False)
        dry = backfill_mod.backfill(populated_db, dry_run=True)
        assert dry.proposed == []
        assert dry.conflicts == 2


class TestBackfillCLI:
    """Smoke-test the CLI entry point."""

    def test_cli_dry_run_exits_zero(self, populated_db: str) -> None:
        rc = backfill_mod.main(["--dsn", populated_db, "--dry-run"])
        assert rc == 0

    def test_cli_real_run_exits_zero(self, populated_db: str) -> None:
        rc = backfill_mod.main(["--dsn", populated_db])
        assert rc == 0
        # Re-running must also exit 0 — the second run is a no-op.
        rc2 = backfill_mod.main(["--dsn", populated_db])
        assert rc2 == 0


class TestBackfillResolvesViaFindByAlias:
    """After backfill, ``find_by_alias`` resolves to the canonical record.

    This is the headline acceptance: clients that hit either
    ``GET /v1/factors/{urn}`` or ``GET /v1/factors/by-alias/{legacy_id}``
    surface the SAME canonical record post-backfill.
    """

    def test_aliased_records_resolve_after_backfill(
        self, populated_db: str
    ) -> None:
        backfill_mod.backfill(populated_db, dry_run=False)
        # Reopen the repo so we don't share the script's connection.
        repo = AlphaFactorRepository(dsn=populated_db)
        try:
            via_urn = repo.get_by_urn(
                "urn:gl:factor:ipcc-2006-nggi:phase2:bf-record-1:v1"
            )
            via_alias = repo.find_by_alias("EF:phase2:bf-1:v1")
            assert via_urn is not None
            assert via_alias is not None
            assert via_alias["urn"] == via_urn["urn"]
            assert via_alias["factor_id_alias"] == "EF:phase2:bf-1:v1"
        finally:
            repo.close()

    def test_record_without_alias_does_not_resolve_by_alias(
        self, populated_db: str
    ) -> None:
        backfill_mod.backfill(populated_db, dry_run=False)
        repo = AlphaFactorRepository(dsn=populated_db)
        try:
            # Record 3 had no alias — find_by_alias("EF:phase2:bf-3:v1")
            # must NOT resolve.
            assert repo.find_by_alias("EF:phase2:bf-3:v1") is None
        finally:
            repo.close()
