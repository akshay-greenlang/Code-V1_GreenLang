# -*- coding: utf-8 -*-
"""Phase F5 — Unit / Chemistry engine tests."""
from __future__ import annotations

import pytest

from greenlang.data.biogenic_split import (
    BIOGENIC_SHARE_TABLE,
    BiogenicShareError,
    biogenic_share,
    split_emissions,
    split_fuel,
)
from greenlang.data.density_converter import (
    DENSITY_TABLE,
    DensityLookupError,
    convert,
    get_density,
    mass_to_volume_l,
    volume_l_to_mass,
)
from greenlang.data.moisture import (
    LATENT_HEAT_WATER_MJ_PER_KG,
    MoistureBasis,
    MoistureError,
    convert_lhv,
    convert_mass,
)
from greenlang.data.oxidation import (
    OXIDATION_TABLE,
    OxidationBasis,
    OxidationLookupError,
    apply_oxidation,
    get_oxidation_factor,
)
from greenlang.factors.ontology.unit_graph import (
    DEFAULT_GRAPH,
    UnitConversionError,
)


# --------------------------------------------------------------------------
# Density
# --------------------------------------------------------------------------


class TestDensity:
    def test_diesel_kg_per_l_at_15c(self):
        rec = get_density("diesel")
        assert rec.density_kg_per_l == pytest.approx(0.84)

    def test_mass_to_volume_diesel(self):
        # 840 kg of diesel at 15 °C = 1000 L
        litres = mass_to_volume_l(material="diesel", mass_kg=840.0)
        assert litres == pytest.approx(1000.0, rel=1e-3)

    def test_volume_to_mass_gasoline(self):
        kg = volume_l_to_mass(material="gasoline", volume_l=1000.0)
        assert kg == pytest.approx(745.0, rel=1e-3)

    def test_temperature_correction(self):
        # Warmer = less dense (thermal expansion), so 1 L contains less mass
        rec = get_density("diesel")
        cold = rec.density_at(temperature_c=0.0)
        hot = rec.density_at(temperature_c=30.0)
        assert cold > hot

    def test_convert_tonnes_to_litres(self):
        # 1 t diesel → ~1190 L
        litres = convert(material="diesel", value=1.0, from_unit="t", to_unit="L")
        assert litres == pytest.approx(1000 / 0.84, rel=1e-2)

    def test_unknown_material_raises(self):
        with pytest.raises(DensityLookupError):
            get_density("unobtainium")

    def test_table_contains_core_fuels(self):
        for fuel in ("diesel", "gasoline", "natural_gas", "coal", "fuel_oil", "hydrogen"):
            assert fuel in DENSITY_TABLE


# --------------------------------------------------------------------------
# Oxidation
# --------------------------------------------------------------------------


class TestOxidation:
    def test_ipcc_2006_default_diesel(self):
        rec = get_oxidation_factor(fuel="diesel")
        assert rec.factor == 1.00

    def test_ipcc_1996_default_coal(self):
        rec = get_oxidation_factor(fuel="coal", basis=OxidationBasis.IPCC_1996)
        assert rec.factor == 0.98

    def test_fluidized_bed_coal(self):
        rec = get_oxidation_factor(fuel="coal", technology="fluidized_bed")
        assert rec.factor == 0.98

    def test_biomass_default_has_unburned_carbon(self):
        rec = get_oxidation_factor(fuel="biomass")
        assert rec.factor == 0.92

    def test_apply_oxidation_math(self):
        # 1000 kg carbon × 0.98 = 980 kg oxidised
        oxidised = apply_oxidation(
            mass_carbon_kg=1000.0,
            fuel="coal",
            technology="fluidized_bed",
        )
        assert oxidised == pytest.approx(980.0)

    def test_fallback_to_default_tech(self):
        rec = get_oxidation_factor(fuel="diesel", technology="some_unknown_tech")
        assert rec.technology == "default"

    def test_unknown_fuel_raises(self):
        with pytest.raises(OxidationLookupError):
            get_oxidation_factor(fuel="unobtainium")

    def test_flaring_special_case(self):
        rec = get_oxidation_factor(fuel="natural_gas", technology="flaring")
        assert rec.factor == 0.995


# --------------------------------------------------------------------------
# Moisture
# --------------------------------------------------------------------------


class TestMoisture:
    def test_wet_to_dry_mass(self):
        # 100 kg wet coal with 10 % moisture → 90 kg dry
        result = convert_mass(
            mass_kg=100.0,
            moisture_fraction=0.10,
            from_basis=MoistureBasis.AS_RECEIVED,
            to_basis=MoistureBasis.DRY,
        )
        assert result.converted_mass_kg == pytest.approx(90.0)

    def test_dry_to_wet_mass(self):
        # 90 kg dry + 10 % moisture → 100 kg wet
        result = convert_mass(
            mass_kg=90.0,
            moisture_fraction=0.10,
            from_basis=MoistureBasis.DRY,
            to_basis=MoistureBasis.AS_RECEIVED,
        )
        assert result.converted_mass_kg == pytest.approx(100.0)

    def test_identity_conversion(self):
        result = convert_mass(
            mass_kg=50.0,
            moisture_fraction=0.15,
            from_basis=MoistureBasis.DRY,
            to_basis=MoistureBasis.DRY,
        )
        assert result.converted_mass_kg == 50.0

    def test_lhv_dry_to_wet_reduces(self):
        # Wet-basis LHV is always lower than dry-basis (latent heat loss)
        dry = 20.0
        wet = convert_lhv(
            lhv_mj_per_kg=dry,
            moisture_fraction=0.10,
            from_basis=MoistureBasis.DRY,
            to_basis=MoistureBasis.AS_RECEIVED,
        )
        assert wet < dry

    def test_lhv_round_trip(self):
        dry = 20.0
        wet = convert_lhv(
            lhv_mj_per_kg=dry,
            moisture_fraction=0.10,
            from_basis=MoistureBasis.DRY,
            to_basis=MoistureBasis.AS_RECEIVED,
        )
        back = convert_lhv(
            lhv_mj_per_kg=wet,
            moisture_fraction=0.10,
            from_basis=MoistureBasis.AS_RECEIVED,
            to_basis=MoistureBasis.DRY,
        )
        assert back == pytest.approx(dry, rel=1e-6)

    def test_invalid_moisture_rejected(self):
        with pytest.raises(MoistureError):
            convert_mass(
                mass_kg=1.0,
                moisture_fraction=1.0,          # 100% moisture = no fuel!
                from_basis=MoistureBasis.AS_RECEIVED,
                to_basis=MoistureBasis.DRY,
            )

    def test_ash_free_dry_not_supported(self):
        with pytest.raises(MoistureError):
            convert_mass(
                mass_kg=1.0,
                moisture_fraction=0.10,
                from_basis=MoistureBasis.AS_RECEIVED,
                to_basis=MoistureBasis.ASH_FREE_DRY,
            )


# --------------------------------------------------------------------------
# Biogenic split
# --------------------------------------------------------------------------


class TestBiogenicSplit:
    def test_pure_diesel_zero_biogenic(self):
        assert biogenic_share("diesel", "pure") == 0.0

    def test_b20_is_20_percent(self):
        assert biogenic_share("diesel", "b20") == 0.20

    def test_e85(self):
        assert biogenic_share("gasoline", "e85") == 0.85

    def test_msw_default_global(self):
        # IPCC 2006 MSW composition avg.
        assert biogenic_share("msw", "default_global") == 0.45

    def test_split_emissions_math(self):
        split = split_emissions(co2_total_kg=1000.0, biogenic_fraction=0.30)
        assert split.fossil_co2_kg == pytest.approx(700.0)
        assert split.biogenic_co2_kg == pytest.approx(300.0)
        assert split.total_co2_kg == pytest.approx(1000.0)

    def test_split_fuel_end_to_end(self):
        split = split_fuel(co2_total_kg=1000.0, fuel="diesel", blend="b20")
        assert split.biogenic_co2_kg == pytest.approx(200.0)
        assert split.fossil_co2_kg == pytest.approx(800.0)
        assert "diesel" in split.source

    def test_invalid_fraction_rejected(self):
        with pytest.raises(ValueError):
            split_emissions(co2_total_kg=100.0, biogenic_fraction=1.5)

    def test_unknown_fuel_raises(self):
        with pytest.raises(BiogenicShareError):
            biogenic_share("unobtainium", "pure")

    def test_unknown_blend_raises(self):
        with pytest.raises(BiogenicShareError):
            biogenic_share("diesel", "b999")


# --------------------------------------------------------------------------
# Unit Graph
# --------------------------------------------------------------------------


class TestUnitGraph:
    def test_kwh_to_mj(self):
        mj = DEFAULT_GRAPH.convert(value=1.0, from_unit="kWh", to_unit="MJ")
        assert mj == pytest.approx(3.6)

    def test_mmbtu_to_kwh(self):
        kwh = DEFAULT_GRAPH.convert(value=1.0, from_unit="MMBtu", to_unit="kWh")
        assert kwh == pytest.approx(293.07, abs=0.01)

    def test_multi_hop_gj_to_kwh(self):
        # GJ → MJ → kWh
        kwh = DEFAULT_GRAPH.convert(value=1.0, from_unit="GJ", to_unit="kWh")
        assert kwh == pytest.approx(277.78, abs=0.01)

    def test_tonnes_to_lb(self):
        lb = DEFAULT_GRAPH.convert(value=1.0, from_unit="t", to_unit="lb")
        assert lb == pytest.approx(2204.62, rel=1e-3)

    def test_energy_without_heating_value_fails(self):
        with pytest.raises(UnitConversionError):
            DEFAULT_GRAPH.convert(value=1.0, from_unit="kg", to_unit="MJ")

    def test_kg_to_mj_with_lhv(self):
        # 1 kg × 43 MJ/kg = 43 MJ (diesel LHV)
        mj = DEFAULT_GRAPH.convert(
            value=1.0,
            from_unit="kg",
            to_unit="MJ",
            heating_value_mj_per_kg=43.0,
        )
        assert mj == pytest.approx(43.0)

    def test_identity_conversion(self):
        assert DEFAULT_GRAPH.convert(value=42.0, from_unit="MJ", to_unit="MJ") == 42.0

    def test_unreachable_path_raises(self):
        with pytest.raises(UnitConversionError):
            DEFAULT_GRAPH.convert(value=1.0, from_unit="fake_unit", to_unit="MJ")
