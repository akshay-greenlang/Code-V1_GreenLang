# -*- coding: utf-8 -*-
"""Tests for GWP set registry (AR4/AR5/AR6, 20/100-yr)."""

from __future__ import annotations

import pytest

from greenlang.factors.ontology.gwp_sets import (
    DEFAULT_GWP_SET,
    GWPSet,
    compare_sets,
    convert_co2e,
    get_gwp,
    list_gases,
    normalize_gas_code,
)


# ---------------------------------------------------------------------------
# Enum / defaults
# ---------------------------------------------------------------------------


def test_default_is_ar6_100():
    # CTO non-negotiable #1 - AR6 100-year is the only acceptable default.
    assert DEFAULT_GWP_SET == GWPSet.IPCC_AR6_100


def test_all_sets_present():
    expected = {
        GWPSet.IPCC_AR4_100, GWPSet.IPCC_AR4_20,
        GWPSet.IPCC_AR5_100, GWPSet.IPCC_AR5_20,
        GWPSet.IPCC_AR6_100, GWPSet.IPCC_AR6_20,
    }
    for s in expected:
        gases = list_gases(s)
        assert gases["CO2"] == 1.0, f"{s} CO2 must be 1.0"
        assert "CH4" in gases
        assert "N2O" in gases


# ---------------------------------------------------------------------------
# Lookup accuracy vs IPCC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gas,gwp_set,expected",
    [
        ("CO2", GWPSet.IPCC_AR6_100, 1.0),
        ("CH4", GWPSet.IPCC_AR6_100, 27.9),
        ("N2O", GWPSet.IPCC_AR6_100, 273.0),
        ("HFC-134a", GWPSet.IPCC_AR6_100, 1530.0),
        ("SF6", GWPSet.IPCC_AR6_100, 25200.0),
        ("NF3", GWPSet.IPCC_AR6_100, 17400.0),
        ("PFC-14", GWPSet.IPCC_AR6_100, 7380.0),
        # AR5
        ("CH4", GWPSet.IPCC_AR5_100, 28.0),
        ("N2O", GWPSet.IPCC_AR5_100, 265.0),
        # AR4
        ("CH4", GWPSet.IPCC_AR4_100, 25.0),
        ("N2O", GWPSet.IPCC_AR4_100, 298.0),
        # 20-year horizon
        ("CH4", GWPSet.IPCC_AR6_20, 81.2),
        ("CH4", GWPSet.IPCC_AR5_20, 84.0),
        ("CH4", GWPSet.IPCC_AR4_20, 72.0),
    ],
)
def test_get_gwp_values(gas, gwp_set, expected):
    assert get_gwp(gas, gwp_set) == pytest.approx(expected)


def test_co2_is_always_unity():
    for s in GWPSet:
        assert get_gwp("CO2", s) == 1.0


# ---------------------------------------------------------------------------
# Alias normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("CF4", "PFC-14"),
        ("C2F6", "PFC-116"),
        ("C3F8", "PFC-218"),
        ("HFC134a", "HFC-134a"),
        ("hfc-32", "HFC-32"),
        ("PFC14", "PFC-14"),
        ("methane", "CH4"),
        ("carbon_dioxide", "CO2"),
    ],
)
def test_normalize_gas_code(raw, canonical):
    assert normalize_gas_code(raw) == canonical


def test_alias_lookup_returns_same_value():
    assert get_gwp("CF4", GWPSet.IPCC_AR6_100) == get_gwp("PFC-14", GWPSet.IPCC_AR6_100)
    assert get_gwp("HFC134a", GWPSet.IPCC_AR6_100) == get_gwp("HFC-134a", GWPSet.IPCC_AR6_100)


def test_empty_gas_raises():
    with pytest.raises(ValueError):
        normalize_gas_code("")


def test_unknown_gas_raises_keyerror():
    with pytest.raises(KeyError):
        get_gwp("unobtainium", GWPSet.IPCC_AR6_100)


# ---------------------------------------------------------------------------
# convert_co2e
# ---------------------------------------------------------------------------


def test_convert_co2e_pure_co2():
    assert convert_co2e({"CO2": 100.0}) == pytest.approx(100.0)


def test_convert_co2e_mixed_vector():
    # 1000 kg CO2 + 5 kg CH4 + 0.2 kg N2O  (AR6-100)
    result = convert_co2e(
        {"CO2": 1000.0, "CH4": 5.0, "N2O": 0.2},
        to_set=GWPSet.IPCC_AR6_100,
    )
    expected = 1000.0 + 5.0 * 27.9 + 0.2 * 273.0
    assert result == pytest.approx(expected)


def test_convert_co2e_cross_set():
    # Gas vector expressed in AR4-100 CO2e, re-express in AR6-100 CO2e.
    # 1 kg CH4 under AR4=25 = 25 kg CO2e;
    # recovered mass = 25/25 = 1 kg, then * 27.9 = 27.9 kg CO2e AR6.
    result = convert_co2e(
        {"CH4": 25.0},
        from_set=GWPSet.IPCC_AR4_100,
        to_set=GWPSet.IPCC_AR6_100,
    )
    assert result == pytest.approx(27.9)


def test_convert_co2e_strict_unknown_raises():
    with pytest.raises(KeyError):
        convert_co2e({"unobtainium": 1.0}, to_set=GWPSet.IPCC_AR6_100)


def test_convert_co2e_nonstrict_skips_unknown():
    out = convert_co2e(
        {"CO2": 10.0, "unobtainium": 999.0},
        to_set=GWPSet.IPCC_AR6_100,
        strict=False,
    )
    assert out == pytest.approx(10.0)


def test_convert_co2e_empty_vector_returns_zero():
    assert convert_co2e({}, to_set=GWPSet.IPCC_AR6_100) == 0.0


# ---------------------------------------------------------------------------
# compare_sets
# ---------------------------------------------------------------------------


def test_compare_sets_returns_all_horizons():
    out = compare_sets("CH4")
    # Should have all 6 sets
    assert len(out) == 6
    assert all(v is not None for v in out.values())
    # AR6-20 should be highest for CH4
    assert out[GWPSet.IPCC_AR6_20.value] > out[GWPSet.IPCC_AR6_100.value]


def test_compare_sets_subset():
    out = compare_sets("CO2", sets=[GWPSet.IPCC_AR6_100, GWPSet.IPCC_AR5_100])
    assert set(out.keys()) == {"IPCC_AR6_100", "IPCC_AR5_100"}
    assert all(v == 1.0 for v in out.values())


# ---------------------------------------------------------------------------
# Error / edge
# ---------------------------------------------------------------------------


def test_unknown_set_raises_on_convert():
    with pytest.raises(ValueError):
        convert_co2e({"CO2": 1.0}, to_set="not_a_set")  # type: ignore[arg-type]


def test_list_gases_returns_copy():
    a = list_gases(GWPSet.IPCC_AR6_100)
    a["fake"] = 999.0
    b = list_gases(GWPSet.IPCC_AR6_100)
    assert "fake" not in b
