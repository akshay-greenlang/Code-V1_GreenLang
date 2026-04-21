# -*- coding: utf-8 -*-
"""Unit ontology conversion helpers (S2) + LHV/HHV, gas law, steam extensions."""

from __future__ import annotations

import pytest

from greenlang.factors.ontology.unit_graph import UnitGraph, UnitConversionError
from greenlang.factors.ontology.units import (
    STP_PRESSURE_PA,
    convert_energy,
    convert_energy_to_kwh,
    convert_fuel_to_kwh,
    fuel_energy_content,
    gas_volume_to_energy_mj,
    gas_volume_to_mass_kg,
    is_known_activity_unit,
    steam_energy_mj,
    steam_enthalpy_kj_per_kg,
    suggest_si_base,
)


# ---------------------------------------------------------------------------
# Original tests (kept for backward compatibility)
# ---------------------------------------------------------------------------


def test_is_known_activity_unit():
    assert is_known_activity_unit("kWh")
    assert not is_known_activity_unit("unknown_unit_xyz")


def test_convert_energy_round_trip():
    mwh = convert_energy(1.0, "mwh", "kwh")
    assert mwh == pytest.approx(1000.0)
    back = convert_energy(mwh, "kwh", "mwh")
    assert back == pytest.approx(1.0)


def test_convert_energy_to_kwh():
    assert convert_energy_to_kwh(2.0, "mwh") == pytest.approx(2000.0)


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        convert_energy_to_kwh(1.0, "not_a_unit")


# ---------------------------------------------------------------------------
# suggest_si_base extensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit,base",
    [
        ("kWh", "J"), ("MWh", "J"), ("GJ", "J"), ("MJ", "J"),
        ("MMBtu", "J"), ("therms", "J"),
        ("kg", "kg"), ("tonnes", "kg"), ("lb", "kg"),
        ("m3", "m3"), ("liters", "m3"), ("gallons", "m3"),
        ("USD", None),
    ],
)
def test_suggest_si_base(unit, base):
    assert suggest_si_base(unit) == base


# ---------------------------------------------------------------------------
# Ideal-gas law helpers
# ---------------------------------------------------------------------------


def test_gas_volume_to_mass_natural_gas_stp():
    # 1 m3 natural gas at STP ~= 0.72 kg (ideal-gas, M=17.04 g/mol)
    mass = gas_volume_to_mass_kg(1.0, "natural_gas")
    assert mass == pytest.approx(0.75, rel=0.1)  # allow 10% tolerance


def test_gas_volume_to_mass_hydrogen_stp():
    # 1 m3 H2 at STP ~= 0.09 kg
    mass = gas_volume_to_mass_kg(1.0, "hydrogen")
    assert mass == pytest.approx(0.09, rel=0.1)


def test_gas_volume_to_mass_temperature_effect():
    # At higher temperature, same volume has LESS mass (ideal gas law).
    cold = gas_volume_to_mass_kg(1.0, "natural_gas", temperature_C=0.0)
    hot = gas_volume_to_mass_kg(1.0, "natural_gas", temperature_C=100.0)
    assert hot < cold


def test_gas_volume_to_mass_pressure_effect():
    low = gas_volume_to_mass_kg(1.0, "natural_gas", pressure_pa=STP_PRESSURE_PA)
    high = gas_volume_to_mass_kg(1.0, "natural_gas", pressure_pa=STP_PRESSURE_PA * 2)
    assert high == pytest.approx(low * 2.0, rel=1e-6)


def test_gas_volume_to_mass_bad_inputs():
    with pytest.raises(ValueError):
        gas_volume_to_mass_kg(-1.0, "natural_gas")
    with pytest.raises(ValueError):
        gas_volume_to_mass_kg(1.0, "natural_gas", pressure_pa=-1.0)
    with pytest.raises(ValueError):
        gas_volume_to_mass_kg(1.0, "natural_gas", temperature_C=-300.0)
    with pytest.raises(ValueError):
        gas_volume_to_mass_kg(1.0, "unknown_gas")


def test_gas_volume_to_energy_mj():
    # 1 m3 natural gas at STP ~= 0.75 kg * 48 MJ/kg ~= 36 MJ (LHV)
    mj = gas_volume_to_energy_mj(1.0, "natural_gas", basis="LHV")
    assert mj == pytest.approx(36.0, rel=0.1)


def test_gas_volume_to_energy_hhv_higher_than_lhv():
    lhv = gas_volume_to_energy_mj(1.0, "natural_gas", basis="LHV")
    hhv = gas_volume_to_energy_mj(1.0, "natural_gas", basis="HHV")
    assert hhv > lhv


# ---------------------------------------------------------------------------
# Unified fuel_energy_content entry point
# ---------------------------------------------------------------------------


def test_fuel_energy_content_kg_diesel():
    mj = fuel_energy_content(100.0, "kg", "diesel")
    assert mj == pytest.approx(4300.0)  # 100 * 43.0


def test_fuel_energy_content_tonnes_coal():
    mj = fuel_energy_content(1.0, "tonnes", "bituminous_coal")
    assert mj == pytest.approx(1000.0 * 25.8)  # 25.8 MJ/kg LHV


def test_fuel_energy_content_liters_diesel():
    # 1 L diesel -> 0.835 kg -> 35.905 MJ
    mj = fuel_energy_content(1.0, "L", "diesel")
    assert mj == pytest.approx(35.905, rel=1e-3)


def test_fuel_energy_content_gallons_gasoline():
    # 1 US gal = 3.78541 L
    mj_l = fuel_energy_content(3.78541, "L", "gasoline")
    mj_gal = fuel_energy_content(1.0, "gallons", "gasoline")
    assert mj_gal == pytest.approx(mj_l, rel=1e-6)


def test_fuel_energy_content_m3_gas_uses_ideal_gas():
    # For natural gas, m3 should route via ideal-gas law (density<5 kg/m3)
    mj = fuel_energy_content(1.0, "m3", "natural_gas")
    assert mj > 30.0 and mj < 45.0  # around ~36 MJ


def test_fuel_energy_content_hhv_vs_lhv():
    lhv = fuel_energy_content(1.0, "kg", "natural_gas", basis="LHV")
    hhv = fuel_energy_content(1.0, "kg", "natural_gas", basis="HHV")
    assert hhv > lhv


def test_fuel_energy_content_unknown_unit():
    with pytest.raises(ValueError):
        fuel_energy_content(1.0, "barrel", "diesel")


def test_fuel_energy_content_negative_raises():
    with pytest.raises(ValueError):
        fuel_energy_content(-1.0, "kg", "diesel")


def test_convert_fuel_to_kwh():
    # 1 kg diesel = 43 MJ / 3.6 = 11.944 kWh
    kwh = convert_fuel_to_kwh(1.0, "kg", "diesel")
    assert kwh == pytest.approx(43.0 / 3.6, rel=1e-6)


# ---------------------------------------------------------------------------
# Advanced fuel support (green hydrogen, synthetic methane, SAF)
# ---------------------------------------------------------------------------


def test_green_hydrogen_lhv_120():
    mj = fuel_energy_content(1.0, "kg", "green_hydrogen", basis="LHV")
    assert mj == pytest.approx(120.0)


def test_green_hydrogen_hhv_141_8():
    mj = fuel_energy_content(1.0, "kg", "green_hydrogen", basis="HHV")
    assert mj == pytest.approx(141.8)


def test_synthetic_methane_lhv():
    mj = fuel_energy_content(1.0, "kg", "synthetic_methane")
    assert mj == pytest.approx(50.0)


def test_saf_lhv():
    mj = fuel_energy_content(1.0, "kg", "saf")
    assert mj == pytest.approx(44.0)


# ---------------------------------------------------------------------------
# Steam enthalpy / energy
# ---------------------------------------------------------------------------


def test_steam_enthalpy_at_atmospheric():
    # 0 bar gauge, saturated vapour ~= 2676 kJ/kg
    h = steam_enthalpy_kj_per_kg(0.0, phase="vapour")
    assert h == pytest.approx(2676.0, rel=1e-3)


def test_steam_enthalpy_monotonic_with_pressure():
    # Vapour enthalpy increases with pressure in the range [0, 10] bar
    prev = 0.0
    for bar in [0, 1, 2, 5, 8, 10]:
        h = steam_enthalpy_kj_per_kg(bar, phase="vapour")
        assert h >= prev
        prev = h


def test_steam_enthalpy_liquid_lower_than_vapour():
    h_l = steam_enthalpy_kj_per_kg(5.0, phase="liquid")
    h_v = steam_enthalpy_kj_per_kg(5.0, phase="vapour")
    assert h_l < h_v


def test_steam_enthalpy_interpolates():
    h_5 = steam_enthalpy_kj_per_kg(5.0, phase="vapour")
    h_6 = steam_enthalpy_kj_per_kg(6.0, phase="vapour")
    h_55 = steam_enthalpy_kj_per_kg(5.5, phase="vapour")
    # midpoint must be within [5 bar, 6 bar]
    assert min(h_5, h_6) <= h_55 <= max(h_5, h_6)


def test_steam_enthalpy_out_of_range():
    with pytest.raises(ValueError):
        steam_enthalpy_kj_per_kg(-1.0)
    with pytest.raises(ValueError):
        steam_enthalpy_kj_per_kg(20.0)


def test_steam_energy_mj_basic():
    # 1 kg steam at 5 bar g, feedwater 20 C ~ (2755 - 83.72) / 1000 = 2.671 MJ
    mj = steam_energy_mj(1.0, 5.0, feedwater_temperature_C=20.0)
    assert mj == pytest.approx(2.671, rel=1e-2)


def test_steam_energy_mj_zero_mass_is_zero():
    assert steam_energy_mj(0.0, 5.0) == 0.0


def test_steam_energy_mj_negative_mass_raises():
    with pytest.raises(ValueError):
        steam_energy_mj(-1.0, 5.0)


# ---------------------------------------------------------------------------
# UnitGraph extensions (moisture + oxidation)
# ---------------------------------------------------------------------------


def test_unit_graph_mass_to_energy_with_hv():
    g = UnitGraph()
    # 1 kg * 43 MJ/kg = 43 MJ
    mj = g.convert(value=1.0, from_unit="kg", to_unit="MJ",
                   heating_value_mj_per_kg=43.0)
    assert mj == pytest.approx(43.0)


def test_unit_graph_with_moisture_correction():
    g = UnitGraph()
    # Dry LHV 18, 20% moisture -> effective 13.91 MJ/kg
    mj = g.convert(value=1.0, from_unit="kg", to_unit="MJ",
                   heating_value_mj_per_kg=18.0, moisture_fraction=0.20)
    assert mj == pytest.approx(13.91, rel=1e-3)


def test_unit_graph_with_oxidation_factor():
    g = UnitGraph()
    # 1 kg * 43 * 0.99 = 42.57 MJ
    mj = g.convert(value=1.0, from_unit="kg", to_unit="MJ",
                   heating_value_mj_per_kg=43.0, oxidation_factor=0.99)
    assert mj == pytest.approx(42.57, rel=1e-3)


def test_unit_graph_moisture_and_oxidation_combined():
    g = UnitGraph()
    # Wood: dry LHV 18, moisture 0.2 -> 13.91, ox 0.97 -> 13.4927
    mj = g.convert(value=1.0, from_unit="kg", to_unit="MJ",
                   heating_value_mj_per_kg=18.0,
                   moisture_fraction=0.20, oxidation_factor=0.97)
    assert mj == pytest.approx(13.91 * 0.97, rel=1e-3)


def test_unit_graph_invalid_moisture_raises():
    g = UnitGraph()
    with pytest.raises(UnitConversionError):
        g.convert(value=1.0, from_unit="kg", to_unit="MJ",
                  heating_value_mj_per_kg=18.0, moisture_fraction=1.5)


def test_unit_graph_invalid_oxidation_raises():
    g = UnitGraph()
    with pytest.raises(UnitConversionError):
        g.convert(value=1.0, from_unit="kg", to_unit="MJ",
                  heating_value_mj_per_kg=18.0, oxidation_factor=1.5)


def test_unit_graph_mass_volume_via_density():
    g = UnitGraph()
    # 835 kg / 835 kg/L = 1 L (wait: density_kg_per_l for diesel is 0.835)
    L = g.convert(value=0.835, from_unit="kg", to_unit="L",
                  material="diesel", density_kg_per_l=0.835)
    assert L == pytest.approx(1.0, rel=1e-6)


def test_unit_graph_volume_mass_via_density():
    g = UnitGraph()
    kg = g.convert(value=1.0, from_unit="L", to_unit="kg",
                   material="diesel", density_kg_per_l=0.835)
    assert kg == pytest.approx(0.835, rel=1e-6)


def test_unit_graph_missing_hv_raises():
    g = UnitGraph()
    with pytest.raises(UnitConversionError):
        g.convert(value=1.0, from_unit="kg", to_unit="MJ")


def test_unit_graph_same_unit_noop():
    g = UnitGraph()
    # same-unit conversion: path is empty, returns value unchanged.
    assert g.convert(value=42.0, from_unit="kg", to_unit="kg") == 42.0


def test_unit_graph_no_path_raises():
    g = UnitGraph()
    with pytest.raises(UnitConversionError):
        g.convert(value=1.0, from_unit="kg", to_unit="foo_unit_xyz")


def test_unit_graph_missing_material_raises():
    g = UnitGraph()
    # mass -> L edge requires material or density
    with pytest.raises(UnitConversionError):
        g.convert(value=1.0, from_unit="kg", to_unit="L")


def test_unit_graph_energy_to_mass_reciprocal():
    g = UnitGraph()
    # 43 MJ / 43 MJ/kg = 1 kg
    kg = g.convert(value=43.0, from_unit="MJ", to_unit="kg",
                   heating_value_mj_per_kg=43.0)
    assert kg == pytest.approx(1.0, rel=1e-6)


def test_unit_graph_energy_to_mass_zero_hv_raises():
    g = UnitGraph()
    with pytest.raises(UnitConversionError):
        g.convert(value=1.0, from_unit="MJ", to_unit="kg",
                  heating_value_mj_per_kg=1.0,
                  moisture_fraction=0.5,
                  # force HV to collapse to zero via extreme latent heat
                  latent_heat_water_mj_per_kg=100.0)


def test_unit_graph_edges_accessor():
    g = UnitGraph()
    edges = g.edges()
    assert len(edges) > 0
    # bootstrap registers both kg->t and t->kg etc.
    assert any(e.source_unit == "kg" for e in edges)


def test_unit_graph_shortest_path_found():
    g = UnitGraph()
    # kWh -> MMBtu is reachable via MJ -> kWh path.
    path = g.shortest_path(from_unit="kWh", to_unit="MMBtu")
    assert path is not None and len(path) > 0

