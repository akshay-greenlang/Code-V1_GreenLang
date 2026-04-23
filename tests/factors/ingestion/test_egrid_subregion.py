# -*- coding: utf-8 -*-
"""Tests for the eGRID subregion extension parser (Wave 4-B)."""
from __future__ import annotations

import json
import re

import pytest

from greenlang.factors.ingestion.bootstrap import (
    SOURCE_SPECS,
    SEED_INPUTS_DIR,
    SEED_DIR,
    bootstrap_catalog,
)
from greenlang.factors.ingestion.parsers.egrid_subregion import (
    parse_egrid_subregion,
)


SOURCE_ID = "egrid_subregion"


def _seed_payload():
    path = SEED_INPUTS_DIR / "egrid_subregion.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_parser_emits_expected_subregion_count():
    records = parse_egrid_subregion(_seed_payload())
    assert len(records) >= 26, (
        f"expected >=26 eGRID subregion records (one per subregion), got {len(records)}"
    )


def test_factor_ids_match_gold_pattern():
    """Gold pattern: ``EF:eGRID:<subregion_lowercase>:US:<year>:v1``."""
    records = parse_egrid_subregion(_seed_payload())
    pattern = re.compile(r"^EF:eGRID:[a-z0-9]+:US:\d{4}:v1$")
    for r in records:
        assert pattern.match(r["factor_id"]), r["factor_id"]


def test_rfce_camx_erct_present():
    """Core subregions required by gold-set electricity entries."""
    records = parse_egrid_subregion(_seed_payload())
    ids = {r["factor_id"] for r in records}
    # Actual eGRID subregion acronyms include SRMV/SRMW/SRSO/SRTV/SRVC
    # (SERC-prefixed, not the literal "serc" token).
    for core in ("rfce", "camx", "erct", "srmw", "nwpp", "newe"):
        hits = [fid for fid in ids if f":{core}:" in fid]
        assert hits, f"no eGRID factor for {core!r} subregion"


def test_target_years_fan_out():
    """When seed declares target_years, we emit one factor per year per sub."""
    records = parse_egrid_subregion(_seed_payload())
    # factor_id shape: EF:eGRID:<sub>:US:<year>:v1  -> year at split index 4
    rfce_years = sorted({
        int(r["factor_id"].split(":")[4])
        for r in records
        if ":rfce:" in r["factor_id"]
    })
    assert len(rfce_years) >= 2, (
        f"target_years fan-out failed for rfce: {rfce_years}"
    )


def test_parser_idempotent():
    r1 = parse_egrid_subregion(_seed_payload())
    r2 = parse_egrid_subregion(_seed_payload())
    assert [x["factor_id"] for x in r1] == [x["factor_id"] for x in r2]


@pytest.fixture(scope="module")
def _egrid_envelope():
    bootstrap_catalog(only_sources=[SOURCE_ID])
    envelope_path = SEED_DIR / SOURCE_ID / "v2022.1.json"
    assert envelope_path.exists()
    with envelope_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_envelope_n5_gate(_egrid_envelope):
    for rec in _egrid_envelope["factors"]:
        assert rec.get("valid_from"), rec["factor_id"]
        assert rec["unit"] == "kwh"
        assert rec["factor_status"] == "certified"
        assert rec["scope"] == "2"


def test_envelope_attribution(_egrid_envelope):
    for rec in _egrid_envelope["factors"]:
        vflags = rec.get("validation_flags") or {}
        assert "eGRID" in vflags.get("attribution_text", "")


def test_source_spec_registered():
    assert SOURCE_ID in {s.source_id for s in SOURCE_SPECS}
