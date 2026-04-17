# -*- coding: utf-8 -*-
"""Tests for parser plugin framework (F018)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers import (
    BaseSourceParser,
    ParserRegistry,
    build_default_registry,
    EPAGHGHubParser,
    EGridParser,
    DESNZUKParser,
    DEFRAParser,
    IPCCDefaultsParser,
    CBAMFullParser,
    GHGProtocolParser,
    TCRParser,
    GreenEParser,
)


# ---- BaseSourceParser ----

def test_base_parser_is_abstract():
    with pytest.raises(TypeError):
        BaseSourceParser()  # type: ignore


def test_concrete_parser_has_required_attrs():
    p = EPAGHGHubParser()
    assert p.source_id == "epa_hub"
    assert p.parser_id == "epa_ghg_hub_v1"
    assert p.parser_version == "1.0"
    assert "json" in p.supported_formats


def test_parser_repr():
    p = EPAGHGHubParser()
    assert "EPAGHGHubParser" in repr(p)
    assert "epa_hub" in repr(p)


# ---- ParserRegistry ----

def test_registry_register_and_get():
    reg = ParserRegistry()
    p = EPAGHGHubParser()
    reg.register(p)
    assert reg.get("epa_hub") is p
    assert reg.get("nonexistent") is None


def test_registry_list_source_ids():
    reg = ParserRegistry()
    reg.register(EPAGHGHubParser())
    reg.register(EGridParser())
    ids = reg.list_source_ids()
    assert "egrid" in ids
    assert "epa_hub" in ids
    assert ids == sorted(ids)


def test_registry_contains():
    reg = ParserRegistry()
    reg.register(EPAGHGHubParser())
    assert "epa_hub" in reg
    assert "nonexistent" not in reg


def test_registry_len():
    reg = ParserRegistry()
    assert len(reg) == 0
    reg.register(EPAGHGHubParser())
    assert len(reg) == 1


def test_registry_overwrite_warns(caplog):
    reg = ParserRegistry()
    reg.register(EPAGHGHubParser())
    reg.register(EPAGHGHubParser())
    assert "Overwriting" in caplog.text


# ---- Default registry ----

def test_default_registry_has_all_parsers():
    reg = build_default_registry()
    expected = [
        "epa_hub", "egrid", "desnz_ghg_conversion", "defra_conversion",
        "eu_cbam", "ghgp_method_refs", "tcr_grp_defaults", "green_e_residual",
    ]
    for sid in expected:
        assert sid in reg, f"Missing parser for {sid}"


def test_default_registry_has_ipcc():
    reg = build_default_registry()
    assert "ipcc_defaults" in reg


def test_default_registry_count():
    reg = build_default_registry()
    assert len(reg) == 9  # 8 unique sources + 1 alias (defra_conversion)


# ---- Schema validation ----

def test_epa_validate_schema_valid():
    p = EPAGHGHubParser()
    ok, issues = p.validate_schema({
        "metadata": {"source": "EPA"},
        "stationary_combustion": [],
    })
    assert ok
    assert issues == []


def test_epa_validate_schema_invalid():
    p = EPAGHGHubParser()
    ok, issues = p.validate_schema({})
    assert not ok
    assert len(issues) > 0


def test_cbam_validate_schema_valid():
    p = CBAMFullParser()
    ok, issues = p.validate_schema({"products": {}})
    assert ok


def test_cbam_validate_schema_invalid():
    p = CBAMFullParser()
    ok, issues = p.validate_schema({})
    assert not ok


def test_ghgp_validate_schema_valid():
    p = GHGProtocolParser()
    ok, issues = p.validate_schema({"cat1_purchased_goods": []})
    assert ok


def test_green_e_validate_schema_invalid():
    p = GreenEParser()
    ok, issues = p.validate_schema({})
    assert not ok


# ---- Parse through registry ----

def test_epa_parse_through_registry():
    reg = build_default_registry()
    parser = reg.get("epa_hub")
    assert parser is not None
    data = {
        "metadata": {"version": "2024"},
        "stationary_combustion": [
            {"fuel_type": "Natural Gas", "unit": "scf", "co2_factor": 0.054, "ch4_factor": 0.000001, "n2o_factor": 0.0000001},
        ],
    }
    factors = parser.parse(data)
    assert len(factors) == 1
    assert factors[0]["factor_id"].startswith("EF:EPA:")


def test_egrid_parse_through_registry():
    reg = build_default_registry()
    parser = reg.get("egrid")
    assert parser is not None
    data = {
        "metadata": {"version": "2022"},
        "subregions": [
            {"acronym": "CAMX", "co2_lb_mwh": 496.0, "ch4_lb_mwh": 0.038, "n2o_lb_mwh": 0.005},
        ],
    }
    factors = parser.parse(data)
    assert len(factors) == 1
