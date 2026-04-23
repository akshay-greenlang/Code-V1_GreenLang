# -*- coding: utf-8 -*-
"""Tests for the India CEA FY27 vintage-extension parser (Wave 4-B)."""
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
from greenlang.factors.ingestion.parsers.cea_fy27 import parse_cea_fy27


SOURCE_ID = "india_cea_fy27"


def _seed_payload():
    path = SEED_INPUTS_DIR / "cea_fy27.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_parser_emits_records():
    records = parse_cea_fy27(_seed_payload())
    assert len(records) >= 15, f"expected >=15 CEA FY27 records, got {len(records)}"


def test_factor_ids_use_v20_0_vintage_tag():
    """The whole point of this parser: every emitted ID must end with ``cea-v20.0``."""
    records = parse_cea_fy27(_seed_payload())
    for r in records:
        assert r["factor_id"].endswith("cea-v20.0"), r["factor_id"]


def test_factor_ids_match_gold_pattern():
    """Gold pattern: ``^EF:IN:<grid>:2026\\-27:cea\\-v20\\.0$``."""
    records = parse_cea_fy27(_seed_payload())
    pattern = re.compile(r"^EF:IN:[a-z_]+:\d{4}-\d{2}:cea-v20\.0$")
    for r in records:
        assert pattern.match(r["factor_id"]), r["factor_id"]


def test_fy27_coverage_present():
    records = parse_cea_fy27(_seed_payload())
    fy27_ids = [r for r in records if "2026-27" in r["factor_id"]]
    assert fy27_ids, "no FY27 records emitted"
    grids = {r["factor_id"].split(":")[2] for r in fy27_ids}
    assert "all_india" in grids


def test_parser_idempotent():
    r1 = parse_cea_fy27(_seed_payload())
    r2 = parse_cea_fy27(_seed_payload())
    assert [x["factor_id"] for x in r1] == [x["factor_id"] for x in r2]


def test_illustrative_rows_flagged():
    records = parse_cea_fy27(_seed_payload())
    fy27 = [r for r in records if "2026-27" in r["factor_id"]]
    for r in fy27:
        assert "illustrative" in r.get("tags", []), (
            f"{r['factor_id']} should be flagged illustrative"
        )


@pytest.fixture(scope="module")
def _cea_envelope():
    bootstrap_catalog(only_sources=[SOURCE_ID])
    envelope_path = SEED_DIR / SOURCE_ID / "v20.0-fy27.json"
    assert envelope_path.exists()
    with envelope_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_envelope_n5_gate(_cea_envelope):
    for rec in _cea_envelope["factors"]:
        assert rec.get("valid_from"), rec["factor_id"]
        assert rec["geography"] == "IN"
        assert rec["unit"] in {"kWh", "kwh"}
        assert rec["factor_status"] == "certified"


def test_envelope_attribution(_cea_envelope):
    for rec in _cea_envelope["factors"]:
        vflags = rec.get("validation_flags") or {}
        attr = vflags.get("attribution_text", "")
        assert "Central Electricity Authority" in attr or "CEA" in attr


def test_source_spec_registered():
    assert SOURCE_ID in {s.source_id for s in SOURCE_SPECS}
