# -*- coding: utf-8 -*-
"""Tests for the IPCC refrigerant GWP promotion parser (Wave 4-B).

Validates the Wave 4-B promotion of HFC/PFC/SF6/NF3 100-year GWPs from
``status=preview`` -> ``status=certified`` and emission of both AR5 and
AR6 GWP-basis records per gas so the gold-set refrigerant family can
resolve.
"""
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
from greenlang.factors.ingestion.parsers.ipcc_refrigerants import (
    parse_ipcc_refrigerants,
)


SOURCE_ID = "ipcc_refrigerants_promoted"


def _seed_payload():
    path = SEED_INPUTS_DIR / "ipcc_refrigerants.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Direct parser tests
# ---------------------------------------------------------------------------


def test_parser_emits_two_records_per_gas():
    records = parse_ipcc_refrigerants(_seed_payload())
    # 16 gases × 2 GWP bases = 32 records (some HFOs have tiny AR6 values)
    assert len(records) >= 24, (
        f"expected >=24 records (12 gases x 2 GWP bases), got {len(records)}"
    )


def test_factor_ids_match_gold_pattern():
    records = parse_ipcc_refrigerants(_seed_payload())
    ar5_pattern = re.compile(r"^EF:IPCC:[a-z0-9_]+:GLOBAL:ipcc_ar5_100:v1$")
    ar6_pattern = re.compile(r"^EF:IPCC:[a-z0-9_]+:GLOBAL:ipcc_ar6_100:v1$")
    for r in records:
        fid = r["factor_id"]
        assert ar5_pattern.match(fid) or ar6_pattern.match(fid), fid


def test_all_core_refrigerants_present_both_bases():
    """Every gold-set gas slug must have AR5 + AR6 records."""
    records = parse_ipcc_refrigerants(_seed_payload())
    ids = {r["factor_id"] for r in records}

    required_gases = [
        "r_22", "r_32", "r_134a", "r_410a", "r_404a", "r_1234yf",
    ]
    for gas in required_gases:
        ar5 = f"EF:IPCC:{gas}:GLOBAL:ipcc_ar5_100:v1"
        ar6 = f"EF:IPCC:{gas}:GLOBAL:ipcc_ar6_100:v1"
        assert ar5 in ids, f"missing AR5 record for {gas}"
        assert ar6 in ids, f"missing AR6 record for {gas}"


def test_r_22_gwp_in_gold_range():
    """R-22 AR5 100yr GWP in published range [1760, 1820]."""
    records = parse_ipcc_refrigerants(_seed_payload())
    r22_ar5 = next(
        r for r in records
        if r["factor_id"] == "EF:IPCC:r_22:GLOBAL:ipcc_ar5_100:v1"
    )
    assert 1760.0 <= r22_ar5["vectors"]["CO2"] <= 1820.0, (
        f"R-22 AR5 GWP {r22_ar5['vectors']['CO2']} outside gold range"
    )


def test_factor_family_is_refrigerant_gwp():
    """Wave 3 family_inference requires factor_family='refrigerant_gwp'."""
    records = parse_ipcc_refrigerants(_seed_payload())
    for r in records:
        assert r["factor_family"] == "refrigerant_gwp", (
            f"{r['factor_id']} wrong factor_family: {r.get('factor_family')}"
        )


def test_parser_idempotent():
    r1 = parse_ipcc_refrigerants(_seed_payload())
    r2 = parse_ipcc_refrigerants(_seed_payload())
    assert [x["factor_id"] for x in r1] == [x["factor_id"] for x in r2]


# ---------------------------------------------------------------------------
# Bootstrap-orchestrated tests — the PROMOTION (preview -> certified)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ref_envelope():
    bootstrap_catalog(only_sources=[SOURCE_ID])
    envelope_path = SEED_DIR / SOURCE_ID / "var5_ar6_100yr.json"
    assert envelope_path.exists()
    with envelope_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_all_records_certified_not_preview(_ref_envelope):
    """The whole point of this promotion: status must be ``certified``.

    Legacy ``ipcc_defaults.py`` emits refrigerants as ``preview``. This
    Wave 4-B promotion ships them as ``certified`` because the N5 gate
    passes and the values are from public IPCC Assessment Reports with
    stable peer-reviewed provenance.
    """
    for rec in _ref_envelope["factors"]:
        assert rec["factor_status"] == "certified", (
            f"{rec['factor_id']} must be certified (got {rec['factor_status']})"
        )


def test_alias_resolution_every_gold_gas_present(_ref_envelope):
    """Every refrigerant gas in the gold-set must resolve to a concrete record."""
    ids = {r["factor_id"] for r in _ref_envelope["factors"]}
    gold_expected = [
        "EF:IPCC:r_22:GLOBAL:ipcc_ar5_100:v1",
        "EF:IPCC:r_22:GLOBAL:ipcc_ar6_100:v1",
        "EF:IPCC:r_32:GLOBAL:ipcc_ar5_100:v1",
        "EF:IPCC:r_32:GLOBAL:ipcc_ar6_100:v1",
        "EF:IPCC:r_134a:GLOBAL:ipcc_ar5_100:v1",
        "EF:IPCC:r_134a:GLOBAL:ipcc_ar6_100:v1",
        "EF:IPCC:r_410a:GLOBAL:ipcc_ar5_100:v1",
        "EF:IPCC:r_410a:GLOBAL:ipcc_ar6_100:v1",
        "EF:IPCC:r_404a:GLOBAL:ipcc_ar5_100:v1",
        "EF:IPCC:r_404a:GLOBAL:ipcc_ar6_100:v1",
        "EF:IPCC:r_1234yf:GLOBAL:ipcc_ar5_100:v1",
        "EF:IPCC:r_1234yf:GLOBAL:ipcc_ar6_100:v1",
    ]
    missing = [fid for fid in gold_expected if fid not in ids]
    assert not missing, f"gold-required refrigerant IDs missing: {missing}"


def test_twelve_plus_gases_covered(_ref_envelope):
    """At least 12 distinct gas slugs covered per Wave 4-B spec."""
    gas_slugs = {
        r["factor_id"].split(":")[2]
        for r in _ref_envelope["factors"]
    }
    assert len(gas_slugs) >= 12, (
        f"expected >=12 distinct gas slugs, got {len(gas_slugs)}: {gas_slugs}"
    )


def test_attribution_references_ipcc_ar5_and_ar6(_ref_envelope):
    for rec in _ref_envelope["factors"]:
        vflags = rec.get("validation_flags") or {}
        attr = vflags.get("attribution_text", "")
        assert "IPCC" in attr, rec["factor_id"]


def test_source_spec_registered():
    assert SOURCE_ID in {s.source_id for s in SOURCE_SPECS}
