# -*- coding: utf-8 -*-
"""Tests for the USEEIO v2 parser (Wave 4-B catalog expansion)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenlang.factors.ingestion.bootstrap import (
    SOURCE_SPECS,
    SEED_INPUTS_DIR,
    SEED_DIR,
    bootstrap_catalog,
)
from greenlang.factors.ingestion.parsers.useeio import parse_useeio


SOURCE_ID = "useeio_v2"


def _seed_payload():
    path = SEED_INPUTS_DIR / "useeio.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Direct parser tests
# ---------------------------------------------------------------------------


def test_parser_emits_records_from_seed():
    payload = _seed_payload()
    records = parse_useeio(payload)
    assert len(records) >= 30, f"expected >=30 USEEIO records, got {len(records)}"


def test_factor_ids_match_gold_pattern():
    records = parse_useeio(_seed_payload())
    for r in records:
        fid = r["factor_id"]
        assert fid.startswith("EF:EEIO:"), f"{fid} must start with EF:EEIO:"
        assert fid.endswith(":v1"), f"{fid} must end with :v1"
        assert ":US:" in fid, f"{fid} must include :US:"


def test_every_record_carries_ghg_vectors_and_unit():
    records = parse_useeio(_seed_payload())
    for r in records:
        assert r["unit"] == "usd", r["factor_id"]
        assert r["vectors"]["CO2"] > 0, (
            f"{r['factor_id']} CO2 intensity must be positive"
        )
        assert r["geography"] == "US"


def test_parser_idempotent():
    r1 = parse_useeio(_seed_payload())
    r2 = parse_useeio(_seed_payload())
    assert [r["factor_id"] for r in r1] == [r["factor_id"] for r in r2]
    assert [r["vectors"]["CO2"] for r in r1] == [r["vectors"]["CO2"] for r in r2]


# ---------------------------------------------------------------------------
# Bootstrap-orchestrated tests (N5 gate + attribution)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _useeio_envelope():
    bootstrap_catalog(only_sources=[SOURCE_ID])
    envelope_path = SEED_DIR / SOURCE_ID / "v2.0.json"
    assert envelope_path.exists(), f"missing envelope {envelope_path}"
    with envelope_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_bootstrap_envelope_written(_useeio_envelope):
    env = _useeio_envelope
    assert env["source_id"] == SOURCE_ID
    assert env["factor_count"] >= 30
    assert env["redistribution_class"] == "open"
    assert "USEEIO" in env["attribution_text"]


def test_every_record_passes_n5_gate(_useeio_envelope):
    """valid_from, source_version, country, unit, status all present."""
    for rec in _useeio_envelope["factors"]:
        assert rec.get("valid_from"), rec.get("factor_id")
        assert rec.get("geography") == "US", rec.get("factor_id")
        assert rec.get("unit") == "usd", rec.get("factor_id")
        assert rec.get("factor_status") == "certified", rec.get("factor_id")
        provenance_version = rec.get("provenance", {}).get("version")
        source_release = rec.get("source_release")
        assert provenance_version or source_release, rec.get("factor_id")


def test_attribution_present_on_every_record(_useeio_envelope):
    for rec in _useeio_envelope["factors"]:
        vflags = rec.get("validation_flags") or {}
        assert vflags.get("attribution_text"), rec.get("factor_id")
        attr = vflags["attribution_text"]
        assert "EPA" in attr or "Environmental Protection Agency" in attr


def test_naics_coverage_includes_gold_targets(_useeio_envelope):
    """Critical gold-set NAICS codes must be present in the envelope."""
    required = {"453210", "541512", "541110", "541810", "722310", "561720",
                "334118", "325120", "323111"}
    present = {
        rec["factor_id"].split(":")[2]
        for rec in _useeio_envelope["factors"]
    }
    missing = required - present
    assert not missing, f"gold-required NAICS codes missing: {missing}"


def test_source_spec_registered():
    sids = {s.source_id for s in SOURCE_SPECS}
    assert SOURCE_ID in sids
