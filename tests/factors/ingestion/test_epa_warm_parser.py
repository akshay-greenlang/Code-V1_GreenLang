# -*- coding: utf-8 -*-
"""Tests for the EPA WARM v15 parser (Wave 4-B catalog expansion)."""
from __future__ import annotations

import json

import pytest

from greenlang.factors.ingestion.bootstrap import (
    SOURCE_SPECS,
    SEED_INPUTS_DIR,
    SEED_DIR,
    bootstrap_catalog,
)
from greenlang.factors.ingestion.parsers.epa_warm import parse_epa_warm


SOURCE_ID = "epa_warm"


def _seed_payload():
    path = SEED_INPUTS_DIR / "epa_warm.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_parser_emits_records():
    records = parse_epa_warm(_seed_payload())
    assert len(records) >= 15, f"expected >=15 WARM records, got {len(records)}"


def test_factor_ids_match_gold_pattern():
    """EPA WARM IDs must match gold regex `EF:EPA:waste_<treatment>_<material>:US:.*`."""
    records = parse_epa_warm(_seed_payload())
    for r in records:
        fid = r["factor_id"]
        assert fid.startswith("EF:EPA:waste_"), f"{fid} must start with EF:EPA:waste_"
        assert ":US:" in fid
        assert fid.endswith(":v1")


def test_expected_treatment_material_combinations_present():
    """Every gold-waste pattern needs at least one matching factor_id."""
    records = parse_epa_warm(_seed_payload())
    ids = {r["factor_id"] for r in records}

    required_substrings = [
        "waste_landfill_msw",
        "waste_landfill_paper",
        "waste_landfill_plastic",
        "waste_compost_food",
        "waste_recycling_cardboard",
        "waste_recycling_aluminum",
        "waste_incineration_msw",
        "waste_recycling_glass",
        "waste_recycling_steel",
        "waste_recycling_ewaste",
    ]
    for needle in required_substrings:
        matches = [i for i in ids if needle in i]
        assert matches, f"no factor_id contains {needle!r}; had {sorted(ids)[:5]}"


def test_parser_idempotent():
    r1 = parse_epa_warm(_seed_payload())
    r2 = parse_epa_warm(_seed_payload())
    assert [x["factor_id"] for x in r1] == [x["factor_id"] for x in r2]


@pytest.fixture(scope="module")
def _warm_envelope():
    bootstrap_catalog(only_sources=[SOURCE_ID])
    envelope_path = SEED_DIR / SOURCE_ID / "v15.json"
    assert envelope_path.exists()
    with envelope_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_envelope_n5_gate(_warm_envelope):
    for rec in _warm_envelope["factors"]:
        assert rec.get("valid_from"), rec["factor_id"]
        assert rec["unit"] == "tonnes"
        assert rec["geography"] == "US"
        assert rec["factor_status"] == "certified"


def test_envelope_attribution(_warm_envelope):
    for rec in _warm_envelope["factors"]:
        vflags = rec.get("validation_flags") or {}
        assert "WARM" in vflags.get("attribution_text", "")


def test_biogenic_flag_on_compost_records(_warm_envelope):
    compost_recs = [
        r for r in _warm_envelope["factors"]
        if "compost" in r["factor_id"] or "anaerobic_digestion" in r["factor_id"]
    ]
    assert compost_recs, "no compost / anaerobic digestion records found"
    for r in compost_recs:
        assert r.get("biogenic_flag") is True, r["factor_id"]


def test_source_spec_registered():
    assert SOURCE_ID in {s.source_id for s in SOURCE_SPECS}
