# -*- coding: utf-8 -*-
"""Phase 2 / WS2 — DB-level URN + alias uniqueness tests.

Per CTO Phase 2 brief Section 2.7 acceptance:

    "test_urn_uniqueness_db.py — sqlite/Postgres-parameterised: insert
    factor with URN X, second insert with same URN raises
    FactorURNAlreadyExistsError. Insert two distinct URNs, alias one to
    the other's legacy_id — alias UNIQUE constraint rejects duplicate
    legacy_id across factors."

This module asserts:

  1. The factor table's ``UNIQUE (urn)`` constraint causes a second
     publish of an identical URN to raise
     :class:`FactorURNAlreadyExistsError`.
  2. The ``factor_aliases`` mirror's ``UNIQUE (legacy_id)`` constraint
     prevents a second alias row from claiming the same ``legacy_id``,
     even when the conflicting alias points at a DIFFERENT factor URN.

The Postgres parameterisation runs only when ``GL_TEST_POSTGRES_DSN`` is
set (CI/local opt-in); the SQLite path is the default and always runs.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from greenlang.factors.repositories.alpha_v0_1_repository import (
    AlphaFactorRepository,
    FactorURNAlreadyExistsError,
)


# ---------------------------------------------------------------------------
# DSN parameterisation — sqlite always runs, Postgres opt-in via env var.
# ---------------------------------------------------------------------------


_PG_DSN = os.environ.get("GL_TEST_POSTGRES_DSN")
_PARAMS = ["sqlite:///:memory:"]
if _PG_DSN:
    _PARAMS.append(_PG_DSN)


# ---------------------------------------------------------------------------
# Fixture factory — minimal record that passes the alpha provenance gate.
#
# The provenance gate (AlphaProvenanceGate) is loaded by the repository on
# every publish call; its only requirement here is that the factor record
# matches the v0.1 schema. We construct a single ``_BASE_RECORD`` template
# and override the URN / source URN / pack URN per test.
# ---------------------------------------------------------------------------


def _base_record(urn: str, *, alias: str | None = None) -> Dict[str, Any]:
    """Produce a v0.1 factor record with the supplied URN and alias.

    Mirrors the seed-fixture shape from
    ``greenlang/factors/data/catalog_seed_v0_1/ipcc_2006_nggi/v1.json`` —
    every required field is populated so the AlphaProvenanceGate passes.
    """
    rec: Dict[str, Any] = {
        "urn": urn,
        "source_urn": "urn:gl:source:ipcc-2006-nggi",
        "factor_pack_urn": "urn:gl:pack:ipcc-2006-nggi:tier-1-defaults:v1",
        "name": "Phase 2 uniqueness test factor",
        "description": (
            "Synthetic factor used to exercise the V500 UNIQUE(urn) "
            "constraint and the V501 UNIQUE(legacy_id) constraint on "
            "factor_aliases."
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
            "source_record_id": "phase2-uniqueness-record",
            "source_publication": "Phase 2 / WS2 uniqueness fixture",
            "source_version": "0.1",
            "raw_artifact_uri": "s3://greenlang-factors-raw/test/phase2.json",
            "raw_artifact_sha256": "6ff38c51f0ffcb08b2057b90164c3f3e6b67a16bacffb27507526b4dab1271c6",
            "parser_id": "tests.factors.v0_1_alpha.phase2",
            "parser_version": "0.1.0",
            "parser_commit": "0000000000000000000000000000000000000000",
            "row_ref": "phase2-uniqueness-record",
            "ingested_at": "2026-04-25T07:42:30Z",
            "operator": "bot:test_urn_uniqueness_db",
        },
        "review": {
            "review_status": "approved",
            "reviewer": "human:methodology-lead@greenlang.io",
            "reviewed_at": "2026-04-25T07:42:30Z",
            "approved_by": "human:methodology-lead@greenlang.io",
            "approved_at": "2026-04-25T07:42:30Z",
        },
        "tags": ["phase2", "uniqueness-test"],
    }
    if alias:
        rec["factor_id_alias"] = alias
    return rec


@pytest.fixture(params=_PARAMS, ids=lambda p: "sqlite" if "sqlite" in p else "postgres")
def repo(request: pytest.FixtureRequest) -> AlphaFactorRepository:
    """Open a fresh repository per test (memory-resident on SQLite)."""
    dsn = request.param
    # legacy mode — Phase 1 provenance gate only; Phase 2 orchestrator covered by tests/factors/v0_1_alpha/phase2/test_publish_pipeline_e2e.py
    r = AlphaFactorRepository(dsn=dsn, publish_env="legacy")
    yield r
    r.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_duplicate_urn_publish_raises(repo: AlphaFactorRepository) -> None:
    """V500 UNIQUE(urn): second publish of the same URN must raise."""
    urn = (
        "urn:gl:factor:ipcc-2006-nggi:phase2:dup-urn-fixture-1:v1"
    )
    rec = _base_record(urn, alias="EF:phase2:dup-urn-1:v1")
    assert repo.publish(rec) == urn
    with pytest.raises(FactorURNAlreadyExistsError) as exc_info:
        repo.publish(rec)
    assert exc_info.value.urn == urn
    # The first publish should still be intact — uniqueness violation
    # must not corrupt the existing row.
    fetched = repo.get_by_urn(urn)
    assert fetched is not None
    assert fetched["urn"] == urn


def test_alias_unique_legacy_id_constraint(
    repo: AlphaFactorRepository,
) -> None:
    """V501 UNIQUE(legacy_id): two distinct factors cannot share alias.

    Steps:
      1. Publish factor A with URN X.
      2. Publish factor B with URN Y.
      3. Register alias ``EF:phase2:shared:v1`` for X — succeeds.
      4. Try to register the SAME alias for Y — must raise (UNIQUE
         legacy_id).
    """
    urn_a = "urn:gl:factor:ipcc-2006-nggi:phase2:alias-conflict-a:v1"
    urn_b = "urn:gl:factor:ipcc-2006-nggi:phase2:alias-conflict-b:v1"
    repo.publish(_base_record(urn_a, alias=None))
    repo.publish(_base_record(urn_b, alias=None))

    legacy_id = "EF:phase2:shared-alias:v1"
    pk_a = repo.register_alias(urn_a, legacy_id)
    assert pk_a >= 1

    # Resolution sanity: alias points at A.
    found = repo.find_by_alias(legacy_id)
    assert found is not None
    assert found["urn"] == urn_a

    with pytest.raises(FactorURNAlreadyExistsError):
        repo.register_alias(urn_b, legacy_id)

    # Resolution still resolves to A — the conflict didn't trample
    # the existing alias.
    still_a = repo.find_by_alias(legacy_id)
    assert still_a is not None
    assert still_a["urn"] == urn_a


def test_distinct_aliases_for_distinct_urns_succeed(
    repo: AlphaFactorRepository,
) -> None:
    """Sanity: two factors with two distinct legacy ids both register
    cleanly and resolve back to the right URN.

    Catches a regression where the UNIQUE(legacy_id) check was
    accidentally widened to UNIQUE(urn, legacy_id) — that would
    silently allow conflicting alias rows.
    """
    urn_a = "urn:gl:factor:ipcc-2006-nggi:phase2:alias-distinct-a:v1"
    urn_b = "urn:gl:factor:ipcc-2006-nggi:phase2:alias-distinct-b:v1"
    repo.publish(_base_record(urn_a))
    repo.publish(_base_record(urn_b))
    repo.register_alias(urn_a, "EF:phase2:distinct-a:v1")
    repo.register_alias(urn_b, "EF:phase2:distinct-b:v1")

    assert repo.find_by_alias("EF:phase2:distinct-a:v1")["urn"] == urn_a
    assert repo.find_by_alias("EF:phase2:distinct-b:v1")["urn"] == urn_b


def test_find_by_alias_returns_none_for_missing_legacy_id(
    repo: AlphaFactorRepository,
) -> None:
    """``find_by_alias`` must return None (not raise) on a miss."""
    assert repo.find_by_alias("EF:phase2:nonexistent:v1") is None
    assert repo.find_by_alias("") is None
