# -*- coding: utf-8 -*-
"""Tests for the fuel heating-values registry."""

from __future__ import annotations

import pytest

from greenlang.factors.ontology.heating_values import (
    FuelHeatingValue,
    apply_moisture_correction,
    apply_temperature_correction,
    convert_mass_to_energy,
    convert_volume_to_energy,
    get_fuel,
    get_heating_value,
    list_fuels,
    with_overrides,
)


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


def test_registry_has_at_least_40_fuels():
    fuels = list_fuels()
    assert len(fuels) >= 40, f"expected >=40 fuels, got {len(fuels)}"


@pytest.mark.parametrize(
    "code",
    [
        "anthracite", "bituminous_coal", "lignite",
        "natural_gas", "lpg", "diesel", "gasoline", "jet_fuel",
        "bunker_fuel", "wood", "biodiesel", "ethanol",
        "green_hydrogen", "synthetic_methane", "biogas",
        "wood_pellets", "saf",
    ],
)
def test_required_fuels_present(code):
    fv = get_fuel(code)
    assert fv.fuel_code == code
    assert fv.LHV_MJ_per_kg > 0
    assert fv.HHV_MJ_per_kg >= fv.LHV_MJ_per_kg
    assert fv.source_citation, "every fuel must carry a citation"


# ---------------------------------------------------------------------------
# Heating value accuracy vs primary sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,basis,expected",
    [
        ("natural_gas", "LHV", 48.0),
        ("natural_gas", "HHV", 53.1),
        ("diesel", "LHV", 43.0),
        ("gasoline", "LHV", 44.3),
        ("green_hydrogen", "LHV", 120.0),
        ("green_hydrogen", "HHV", 141.8),
        ("wood_pellets", "LHV", 17.0),
        ("ethanol", "LHV", 26.7),
    ],
)
def test_heating_value_matches_source(code, basis, expected):
    assert get_heating_value(code, basis) == pytest.approx(expected)


def test_hhv_always_ge_lhv():
    for code in list_fuels():
        fv = get_fuel(code)
        assert fv.HHV_MJ_per_kg >= fv.LHV_MJ_per_kg - 1e-6


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("NG", "natural_gas"),
        ("pipeline_gas", "natural_gas"),
        ("petrol", "gasoline"),
        ("HVO", "renewable_diesel"),
        ("MSW", "municipal_solid_waste"),
        ("H2", "hydrogen"),
        ("brown_coal", "lignite"),
        ("hard_coal", "bituminous_coal"),
    ],
)
def test_aliases(alias, canonical):
    assert get_fuel(alias).fuel_code == canonical


def test_unknown_fuel_raises():
    with pytest.raises(KeyError):
        get_fuel("unobtanium_fuel")


def test_empty_fuel_raises():
    with pytest.raises(ValueError):
        get_fuel("")


# ---------------------------------------------------------------------------
# Mass / volume -> energy conversions
# ---------------------------------------------------------------------------


def test_convert_mass_to_energy_diesel():
    # 100 kg of diesel * 43.0 MJ/kg = 4300 MJ
    assert convert_mass_to_energy(100.0, "diesel") == pytest.approx(4300.0)


def test_convert_mass_to_energy_negative_raises():
    with pytest.raises(ValueError):
        convert_mass_to_energy(-1.0, "diesel")


def test_convert_volume_to_energy_uses_density():
    # 1 m3 diesel ~= 835 kg * 43.0 MJ/kg = 35905 MJ
    assert convert_volume_to_energy(1.0, "diesel") == pytest.approx(35905.0, rel=1e-3)


def test_convert_mass_lhv_vs_hhv_differ():
    lhv = convert_mass_to_energy(1.0, "natural_gas", "LHV")
    hhv = convert_mass_to_energy(1.0, "natural_gas", "HHV")
    assert hhv > lhv


def test_invalid_basis_raises():
    with pytest.raises(ValueError):
        get_fuel("diesel").get("NCV")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Moisture correction
# ---------------------------------------------------------------------------


def test_apply_moisture_correction_dry_basis():
    # Dry LHV 18 MJ/kg, 20% moisture:
    # 18*(1-0.2) - 2.45*0.2 = 14.4 - 0.49 = 13.91
    out = apply_moisture_correction(18.0, 0.20)
    assert out == pytest.approx(13.91, rel=1e-3)


def test_apply_moisture_correction_zero():
    assert apply_moisture_correction(18.0, 0.0) == pytest.approx(18.0)


def test_apply_moisture_correction_invert():
    # Round-trip: dry -> wet -> dry must return original (within floating tol).
    dry = 18.0
    m = 0.25
    wet = apply_moisture_correction(dry, m, dry_basis_value=True)
    back = apply_moisture_correction(wet, m, dry_basis_value=False)
    assert back == pytest.approx(dry, rel=1e-6)


def test_apply_moisture_correction_out_of_range():
    with pytest.raises(ValueError):
        apply_moisture_correction(18.0, 1.0)
    with pytest.raises(ValueError):
        apply_moisture_correction(18.0, -0.1)


def test_apply_moisture_correction_negative_hv_raises():
    with pytest.raises(ValueError):
        apply_moisture_correction(-1.0, 0.1)


def test_apply_moisture_correction_clamps_to_zero():
    # Very wet, weak fuel: corrected value should not go negative.
    out = apply_moisture_correction(1.0, 0.9)
    assert out >= 0.0


# ---------------------------------------------------------------------------
# Temperature correction
# ---------------------------------------------------------------------------


def test_apply_temperature_correction_at_ref_is_identity():
    out = apply_temperature_correction(48.0, 25.0, "natural_gas")
    assert out == pytest.approx(48.0)


def test_apply_temperature_correction_hotter_reduces_value():
    out = apply_temperature_correction(48.0, 100.0, "natural_gas")
    assert out < 48.0


def test_apply_temperature_correction_cooler_increases_value():
    out = apply_temperature_correction(48.0, -25.0, "natural_gas")
    assert out > 48.0


def test_apply_temperature_correction_coefficient():
    # Linear: delta 75 C, k=4e-4 -> factor 1 - 0.03 = 0.97
    out = apply_temperature_correction(100.0, 100.0, "natural_gas", coefficient_per_C=4e-4)
    assert out == pytest.approx(100.0 * 0.97, rel=1e-6)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_with_overrides_creates_copy():
    base = get_fuel("wood")
    patched = with_overrides("wood", LHV_MJ_per_kg=14.0, moisture_content_fraction=0.30)
    assert isinstance(patched, FuelHeatingValue)
    assert patched.LHV_MJ_per_kg == 14.0
    assert patched.moisture_content_fraction == 0.30
    # original untouched
    assert get_fuel("wood").LHV_MJ_per_kg == base.LHV_MJ_per_kg


def test_fuel_dataclass_is_frozen():
    fv = get_fuel("diesel")
    with pytest.raises(Exception):  # FrozenInstanceError inherits from AttributeError
        fv.LHV_MJ_per_kg = 99.0  # type: ignore[misc]
