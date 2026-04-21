# -*- coding: utf-8 -*-
"""Tests for gas-level chemistry utilities."""

from __future__ import annotations

import pytest

from greenlang.factors.ontology.chemistry import (
    BiogenicFate,
    C_TO_CO2_RATIO,
    CO2Split,
    RefrigerantLeakageResult,
    aggregate_co2e,
    apply_oxidation_factor,
    biogenic_fate,
    build_combustion_gas_vector,
    carbon_to_co2,
    get_default_oxidation_factor,
    model_refrigerant_leakage,
    split_fossil_biogenic_co2,
)
from greenlang.factors.ontology.gwp_sets import GWPSet


# ---------------------------------------------------------------------------
# Stoichiometry
# ---------------------------------------------------------------------------


def test_c_to_co2_ratio_constant():
    assert C_TO_CO2_RATIO == pytest.approx(44.009 / 12.011)


def test_carbon_to_co2():
    # 12 kg C -> ~44 kg CO2
    assert carbon_to_co2(12.0) == pytest.approx(44.009, rel=1e-3)


def test_carbon_to_co2_zero():
    assert carbon_to_co2(0.0) == 0.0


def test_carbon_to_co2_negative_raises():
    with pytest.raises(ValueError):
        carbon_to_co2(-1.0)


def test_apply_oxidation_factor():
    assert apply_oxidation_factor(100.0, 0.98) == pytest.approx(98.0)


def test_apply_oxidation_factor_bounds():
    with pytest.raises(ValueError):
        apply_oxidation_factor(100.0, 1.1)
    with pytest.raises(ValueError):
        apply_oxidation_factor(100.0, -0.1)


# ---------------------------------------------------------------------------
# Default oxidation factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fuel,expected",
    [
        ("bituminous_coal", 0.98),
        ("natural_gas", 0.995),
        ("wood", 0.97),
        ("diesel", 0.99),
        ("municipal_solid_waste", 0.95),
    ],
)
def test_default_oxidation_factor(fuel, expected):
    assert get_default_oxidation_factor(fuel) == expected


def test_default_oxidation_fallback():
    # Unknown fuels get a conservative 0.99
    assert get_default_oxidation_factor("unknown_fuel_xyz") == 0.99


# ---------------------------------------------------------------------------
# Fossil / biogenic split
# ---------------------------------------------------------------------------


def test_split_fossil_diesel():
    # 1 kg diesel, 86.7% C, 0.99 ox -> 0.99*0.867 kg C -> * 44/12 kg CO2
    split = split_fossil_biogenic_co2("diesel", 1.0)
    assert isinstance(split, CO2Split)
    expected_co2 = 0.99 * 0.867 * C_TO_CO2_RATIO
    assert split.co2_fossil_kg == pytest.approx(expected_co2, rel=1e-3)
    assert split.co2_biogenic_kg == 0.0
    assert split.biogenic_fraction == 0.0


def test_split_biogenic_wood():
    # Wood is 100% biogenic
    split = split_fossil_biogenic_co2("wood", 10.0)
    assert split.biogenic_fraction == 1.0
    assert split.co2_fossil_kg == 0.0
    assert split.co2_biogenic_kg > 0


def test_split_msw_mixed_biogenic():
    # MSW default is 60% biogenic
    split = split_fossil_biogenic_co2("municipal_solid_waste", 100.0)
    assert split.biogenic_fraction == pytest.approx(0.60)
    # fossil + bio should equal total
    assert split.co2_fossil_kg + split.co2_biogenic_kg == pytest.approx(
        split.co2_total_kg
    )


def test_split_zero_carbon_fuel():
    # Hydrogen has zero carbon -> zero CO2
    split = split_fossil_biogenic_co2("green_hydrogen", 10.0)
    assert split.co2_fossil_kg == 0.0
    assert split.co2_biogenic_kg == 0.0


def test_split_custom_oxidation_factor():
    split = split_fossil_biogenic_co2("diesel", 1.0, oxidation_factor=1.0)
    # oxidation=1 means full combustion; factor applied = 1.0
    assert split.oxidation_factor == 1.0
    expected = 1.0 * 0.867 * C_TO_CO2_RATIO
    assert split.co2_fossil_kg == pytest.approx(expected, rel=1e-3)


def test_split_custom_biogenic_fraction():
    split = split_fossil_biogenic_co2(
        "natural_gas", 100.0, biogenic_fraction=0.10
    )
    # 10% treated as biogenic
    assert split.biogenic_fraction == 0.10
    total = split.co2_fossil_kg + split.co2_biogenic_kg
    assert split.co2_biogenic_kg == pytest.approx(total * 0.10, rel=1e-6)


def test_split_gas_vector_keys():
    split = split_fossil_biogenic_co2("wood", 5.0)
    vec = split.as_gas_vector()
    assert set(vec.keys()) == {"CO2_fossil", "CO2_biogenic"}
    assert vec["CO2_biogenic"] > 0
    assert vec["CO2_fossil"] == 0.0


def test_split_invalid_biogenic_fraction_raises():
    with pytest.raises(ValueError):
        split_fossil_biogenic_co2("natural_gas", 1.0, biogenic_fraction=1.5)


def test_split_negative_fuel_raises():
    with pytest.raises(ValueError):
        split_fossil_biogenic_co2("diesel", -1.0)


# ---------------------------------------------------------------------------
# Biogenic fate / sequestration
# ---------------------------------------------------------------------------


def test_biogenic_fate_cradle_to_gate():
    fate = biogenic_fate(100.0, treatment="cradle_to_gate")
    assert fate.treatment == "cradle_to_gate"
    # cradle-to-gate counts combustion only
    assert fate.net_biogenic_kg == pytest.approx(100.0)


def test_biogenic_fate_cradle_to_grave_neutral():
    fate = biogenic_fate(100.0, treatment="cradle_to_grave")
    # Default: sequestration equals release -> net 0 (carbon-neutral).
    assert fate.net_biogenic_kg == pytest.approx(0.0)


def test_biogenic_fate_with_partial_sequestration():
    fate = biogenic_fate(
        100.0, co2_absorbed_kg=70.0, co2_eol_kg=10.0,
        treatment="cradle_to_grave",
    )
    # net = 100 + 10 - 70 = 40
    assert fate.net_biogenic_kg == pytest.approx(40.0)


def test_biogenic_fate_invalid_treatment():
    with pytest.raises(ValueError):
        biogenic_fate(1.0, treatment="bogus")


# ---------------------------------------------------------------------------
# Refrigerant leakage
# ---------------------------------------------------------------------------


def test_refrigerant_leakage_basic():
    r = model_refrigerant_leakage(
        "HFC-134a",
        charge_kg=10.0,
        annual_leak_rate=0.02,
        installation_leak_rate=0.01,
        end_of_life_recovery_rate=0.70,
        years=1.0,
    )
    assert isinstance(r, RefrigerantLeakageResult)
    # install = 10*0.01 = 0.1
    # operational = 10*0.02*1 = 0.2
    # no EoL
    assert r.installation_leak_kg == pytest.approx(0.1)
    assert r.operational_leak_kg == pytest.approx(0.2)
    assert r.end_of_life_leak_kg == 0.0
    assert r.total_leak_kg == pytest.approx(0.3)
    # CO2e using AR6-100 HFC-134a GWP 1530
    assert r.co2e_kg == pytest.approx(0.3 * 1530.0, rel=1e-3)


def test_refrigerant_leakage_with_eol():
    r = model_refrigerant_leakage(
        "HFC-32",
        charge_kg=5.0,
        annual_leak_rate=0.05,
        installation_leak_rate=0.005,
        end_of_life_recovery_rate=0.80,
        end_of_life_flag=True,
        years=10.0,
    )
    # install 0.025, operational 2.5, eol 1.0 -> total 3.525
    assert r.end_of_life_leak_kg == pytest.approx(1.0)
    assert r.total_leak_kg == pytest.approx(3.525, rel=1e-4)


def test_refrigerant_leakage_invalid_rates():
    with pytest.raises(ValueError):
        model_refrigerant_leakage("HFC-134a", charge_kg=1.0, annual_leak_rate=1.5)
    with pytest.raises(ValueError):
        model_refrigerant_leakage("HFC-134a", charge_kg=-1.0)
    with pytest.raises(ValueError):
        model_refrigerant_leakage("HFC-134a", charge_kg=1.0, years=-1)


def test_refrigerant_leakage_gas_vector():
    r = model_refrigerant_leakage("SF6", charge_kg=2.0, annual_leak_rate=0.01)
    vec = r.as_gas_vector()
    assert "SF6" in vec
    assert vec["SF6"] > 0


def test_refrigerant_unknown_uses_zero_gwp():
    # Unknown refrigerant code falls back to 0 GWP with warning.
    r = model_refrigerant_leakage("R-410A", charge_kg=1.0, annual_leak_rate=0.01)
    assert r.co2e_kg == 0.0


# ---------------------------------------------------------------------------
# Combustion gas-vector assembly (CTO non-negotiable #1)
# ---------------------------------------------------------------------------


def test_build_combustion_gas_vector_diesel():
    vec = build_combustion_gas_vector(
        "diesel",
        fuel_mass_kg=100.0,
        ch4_kg_per_kg_fuel=1e-6,
        n2o_kg_per_kg_fuel=5e-7,
    )
    assert set(vec.keys()) == {"CO2_fossil", "CO2_biogenic", "CH4", "N2O"}
    assert vec["CO2_fossil"] > 0
    assert vec["CO2_biogenic"] == 0.0
    assert vec["CH4"] == pytest.approx(1e-4)
    assert vec["N2O"] == pytest.approx(5e-5)


def test_build_combustion_gas_vector_biogenic_wood():
    vec = build_combustion_gas_vector("wood", 100.0)
    assert vec["CO2_biogenic"] > 0
    assert vec["CO2_fossil"] == 0.0


def test_aggregate_co2e_excludes_biogenic_by_default():
    vec = build_combustion_gas_vector("wood", 100.0)
    # Biogenic CO2 default: NOT aggregated to CO2e under GHG Protocol.
    co2e = aggregate_co2e(vec, gwp_set=GWPSet.IPCC_AR6_100, include_biogenic=False)
    assert co2e == 0.0


def test_aggregate_co2e_includes_biogenic_when_requested():
    vec = build_combustion_gas_vector("wood", 100.0)
    co2e = aggregate_co2e(vec, gwp_set=GWPSet.IPCC_AR6_100, include_biogenic=True)
    assert co2e > 0
    # should equal the biogenic CO2 mass (GWP=1)
    assert co2e == pytest.approx(vec["CO2_biogenic"], rel=1e-6)


def test_aggregate_co2e_gas_mix_diesel():
    vec = build_combustion_gas_vector(
        "diesel", 1000.0,
        ch4_kg_per_kg_fuel=1e-5, n2o_kg_per_kg_fuel=1e-6,
    )
    co2e = aggregate_co2e(vec, gwp_set=GWPSet.IPCC_AR6_100)
    # CO2_fossil * 1 + CH4 * 27.9 + N2O * 273
    expected = (
        vec["CO2_fossil"] * 1.0
        + vec["CH4"] * 27.9
        + vec["N2O"] * 273.0
    )
    assert co2e == pytest.approx(expected, rel=1e-6)


def test_aggregate_co2e_empty_vector():
    assert aggregate_co2e({}, gwp_set=GWPSet.IPCC_AR6_100) == 0.0


# ---------------------------------------------------------------------------
# End-to-end: CTO non-negotiable #1 (store gas vector, derive CO2e)
# ---------------------------------------------------------------------------


def test_cto_non_negotiable_gas_vector_first():
    """Emission factor must store gas vector, not CO2e.

    Validates that build_combustion_gas_vector produces a vector that
    can be re-aggregated under any GWP set with deterministic results.
    """
    vec = build_combustion_gas_vector(
        "natural_gas", 100.0,
        ch4_kg_per_kg_fuel=5e-6, n2o_kg_per_kg_fuel=1e-7,
    )
    ar4 = aggregate_co2e(vec, gwp_set=GWPSet.IPCC_AR4_100)
    ar5 = aggregate_co2e(vec, gwp_set=GWPSet.IPCC_AR5_100)
    ar6 = aggregate_co2e(vec, gwp_set=GWPSet.IPCC_AR6_100)
    # CO2 is identical across sets (GWP=1). CH4 differs (25 vs 28 vs 27.9).
    assert ar4 != ar5
    assert ar5 != ar6
    # All positive
    assert ar4 > 0 and ar5 > 0 and ar6 > 0
