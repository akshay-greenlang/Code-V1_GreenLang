# -*- coding: utf-8 -*-
"""
Unit tests for FuelDatabaseEngine (Engine 1) - AGENT-MRV-001

Tests all methods of FuelDatabaseEngine with 70+ tests covering:
- Initialization and built-in data loading
- Fuel property lookups for all 20 fuel types
- Emission factor retrieval across EPA, IPCC, DEFRA, EU ETS sources
- Heating value (HHV/NCV) lookups and HHV > NCV invariant
- Oxidation factor lookups with source-specific values
- GWP lookups for AR4, AR5, AR6 (100yr and 20yr)
- Custom factor registration, retrieval, and overwrite
- Factor listing, searching, and counting
- Biogenic fuel classification
- Unit conversion between kg/mmBtu, kg/GJ, kg/TJ, t/TJ
- HHV-to-NCV ratio calculation
- Thread safety for concurrent reads and writes

Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.stationary_combustion.fuel_database import (
    FuelDatabaseEngine,
    _EPA_FACTORS,
    _IPCC_FACTORS,
    _DEFRA_FACTORS,
    _EU_ETS_FACTORS,
    _FUEL_PROPERTIES,
    _HEATING_VALUES,
    _OXIDATION_FACTORS,
    _GWP_VALUES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Create a FuelDatabaseEngine instance with provenance disabled."""
    return FuelDatabaseEngine(config={"enable_provenance": False})


@pytest.fixture
def db_with_provenance():
    """Create a FuelDatabaseEngine instance with provenance enabled."""
    return FuelDatabaseEngine(config={"enable_provenance": True})


# ---------------------------------------------------------------------------
# TestFuelDatabaseInit
# ---------------------------------------------------------------------------

class TestFuelDatabaseInit:
    """Tests for FuelDatabaseEngine initialization."""

    def test_initializes_with_default_config(self):
        """Engine initializes without config argument."""
        db = FuelDatabaseEngine(config={"enable_provenance": False})
        assert db is not None

    def test_initializes_with_built_in_data(self, db):
        """Engine loads built-in emission factor data on construction."""
        count = db.get_factor_count()
        assert count > 0

    def test_factor_count_exceeds_300(self, db):
        """Built-in database contains 300+ emission factors across 4 sources."""
        count = db.get_factor_count()
        assert count >= 300, f"Expected >= 300 factors, got {count}"

    def test_custom_factors_initially_empty(self, db):
        """Custom factor registry starts empty."""
        assert len(db._custom_factors) == 0

    def test_repr_contains_counts(self, db):
        """__repr__ includes fuel type and factor counts."""
        r = repr(db)
        assert "FuelDatabaseEngine" in r
        assert "fuel_types=" in r
        assert "factors=" in r

    def test_provenance_disabled_when_configured(self, db):
        """Provenance tracker is None when disabled in config."""
        assert db._provenance is None

    def test_provenance_enabled_when_configured(self, db_with_provenance):
        """Provenance tracker is set when enabled in config."""
        assert db_with_provenance._provenance is not None


# ---------------------------------------------------------------------------
# TestGetFuelProperties
# ---------------------------------------------------------------------------

class TestGetFuelProperties:
    """Tests for get_fuel_properties method."""

    ALL_FUEL_TYPES = sorted(_FUEL_PROPERTIES.keys())

    @pytest.mark.parametrize("fuel_type", sorted(_FUEL_PROPERTIES.keys()))
    def test_all_fuel_types_return_valid_properties(self, db, fuel_type):
        """Every registered fuel type returns a non-empty properties dict."""
        props = db.get_fuel_properties(fuel_type)
        assert isinstance(props, dict)
        assert "category" in props
        assert "display_name" in props
        assert "is_biogenic" in props
        assert "hhv" in props
        assert "ncv" in props

    def test_natural_gas_properties(self, db):
        """Natural gas has GASEOUS category and is not biogenic."""
        props = db.get_fuel_properties("NATURAL_GAS")
        assert props["category"] == "GASEOUS"
        assert props["is_biogenic"] is False

    def test_wood_biomass_is_biogenic(self, db):
        """Wood/biomass is classified as biogenic."""
        props = db.get_fuel_properties("WOOD_BIOMASS")
        assert props["is_biogenic"] is True

    def test_biogas_is_biogenic(self, db):
        """Biogas is classified as biogenic."""
        props = db.get_fuel_properties("BIOGAS")
        assert props["is_biogenic"] is True

    def test_landfill_gas_is_biogenic(self, db):
        """Landfill gas is classified as biogenic."""
        props = db.get_fuel_properties("LANDFILL_GAS")
        assert props["is_biogenic"] is True

    def test_diesel_not_biogenic(self, db):
        """Diesel is NOT biogenic."""
        props = db.get_fuel_properties("DIESEL")
        assert props["is_biogenic"] is False

    def test_unknown_fuel_type_raises_key_error(self, db):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError, match="Unknown fuel type"):
            db.get_fuel_properties("UNOBTAINIUM")

    def test_case_insensitive_lookup(self, db):
        """Fuel type lookup is case-insensitive."""
        props_upper = db.get_fuel_properties("NATURAL_GAS")
        props_lower = db.get_fuel_properties("natural_gas")
        assert props_upper == props_lower

    def test_properties_include_heating_values(self, db):
        """Properties include HHV and NCV from heating values table."""
        props = db.get_fuel_properties("DIESEL")
        assert props["hhv"] > 0
        assert props["ncv"] > 0


# ---------------------------------------------------------------------------
# TestGetEmissionFactor
# ---------------------------------------------------------------------------

class TestGetEmissionFactor:
    """Tests for get_emission_factor method."""

    def test_epa_natural_gas_co2(self, db):
        """EPA natural gas CO2 factor is 53.06 kg/mmBtu."""
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="EPA")
        assert ef == Decimal("53.06")

    def test_ipcc_natural_gas_co2(self, db):
        """IPCC natural gas CO2 factor is 56100 kg/TJ."""
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="IPCC")
        assert ef == Decimal("56100")

    def test_epa_diesel_co2(self, db):
        """EPA diesel CO2 factor is 73.96 kg/mmBtu."""
        ef = db.get_emission_factor("DIESEL", "CO2", source="EPA")
        assert ef == Decimal("73.96")

    def test_epa_coal_bituminous_co2(self, db):
        """EPA coal bituminous CO2 factor is 93.28 kg/mmBtu."""
        ef = db.get_emission_factor("COAL_BITUMINOUS", "CO2", source="EPA")
        assert ef == Decimal("93.28")

    def test_epa_wood_biomass_co2_is_zero(self, db):
        """EPA wood/biomass CO2 factor is 0 (biogenic)."""
        ef = db.get_emission_factor("WOOD_BIOMASS", "CO2", source="EPA")
        assert ef == Decimal("0")

    def test_epa_biogas_co2_is_zero(self, db):
        """EPA biogas CO2 factor is 0 (biogenic)."""
        ef = db.get_emission_factor("BIOGAS", "CO2", source="EPA")
        assert ef == Decimal("0")

    def test_epa_landfill_gas_co2_is_zero(self, db):
        """EPA landfill gas CO2 factor is 0 (biogenic)."""
        ef = db.get_emission_factor("LANDFILL_GAS", "CO2", source="EPA")
        assert ef == Decimal("0")

    def test_ch4_factor_is_positive(self, db):
        """CH4 emission factor for natural gas is positive."""
        ef = db.get_emission_factor("NATURAL_GAS", "CH4", source="EPA")
        assert ef > 0

    def test_n2o_factor_is_positive(self, db):
        """N2O emission factor for natural gas is positive."""
        ef = db.get_emission_factor("NATURAL_GAS", "N2O", source="EPA")
        assert ef > 0

    def test_defra_natural_gas_co2(self, db):
        """DEFRA natural gas CO2 factor is 0.18316 kg/kWh."""
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="DEFRA")
        assert ef == Decimal("0.18316")

    def test_eu_ets_natural_gas_co2(self, db):
        """EU ETS natural gas CO2 factor matches IPCC: 56100 kg/TJ."""
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="EU_ETS")
        assert ef == Decimal("56100")

    def test_default_source_is_epa(self, db):
        """Default source for get_emission_factor is EPA."""
        ef_default = db.get_emission_factor("NATURAL_GAS", "CO2")
        ef_epa = db.get_emission_factor("NATURAL_GAS", "CO2", source="EPA")
        assert ef_default == ef_epa

    def test_unknown_fuel_raises_key_error(self, db):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_emission_factor("UNKNOWN_FUEL", "CO2", source="EPA")

    def test_unknown_gas_raises_key_error(self, db):
        """Unknown gas raises KeyError."""
        with pytest.raises(KeyError):
            db.get_emission_factor("NATURAL_GAS", "SF6", source="EPA")

    def test_unknown_source_raises_key_error(self, db):
        """Unknown source raises KeyError."""
        with pytest.raises(KeyError):
            db.get_emission_factor("NATURAL_GAS", "CO2", source="BOGUS")

    def test_emission_factor_returns_decimal(self, db):
        """All emission factors are returned as Decimal type."""
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="EPA")
        assert isinstance(ef, Decimal)


# ---------------------------------------------------------------------------
# TestGetHeatingValue
# ---------------------------------------------------------------------------

class TestGetHeatingValue:
    """Tests for get_heating_value method."""

    @pytest.mark.parametrize("fuel_type", sorted(_HEATING_VALUES.keys()))
    def test_hhv_returns_positive(self, db, fuel_type):
        """HHV for every fuel type is positive."""
        hhv = db.get_heating_value(fuel_type, basis="HHV")
        assert hhv > 0

    @pytest.mark.parametrize("fuel_type", sorted(_HEATING_VALUES.keys()))
    def test_ncv_returns_positive(self, db, fuel_type):
        """NCV for every fuel type is positive."""
        ncv = db.get_heating_value(fuel_type, basis="NCV")
        assert ncv > 0

    @pytest.mark.parametrize("fuel_type", sorted(_HEATING_VALUES.keys()))
    def test_hhv_greater_than_ncv(self, db, fuel_type):
        """HHV is greater than NCV for all fuels."""
        hhv = db.get_heating_value(fuel_type, basis="HHV")
        ncv = db.get_heating_value(fuel_type, basis="NCV")
        assert hhv > ncv, (
            f"{fuel_type}: HHV ({hhv}) should be > NCV ({ncv})"
        )

    def test_natural_gas_hhv(self, db):
        """Natural gas HHV is 1.028 mmBtu/Mscf."""
        hhv = db.get_heating_value("NATURAL_GAS", basis="HHV")
        assert hhv == Decimal("1.028")

    def test_unknown_fuel_raises_key_error(self, db):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_heating_value("UNKNOWN_FUEL", basis="HHV")

    def test_invalid_basis_raises_value_error(self, db):
        """Invalid basis raises ValueError."""
        with pytest.raises(ValueError, match="basis must be"):
            db.get_heating_value("NATURAL_GAS", basis="LHV")

    def test_default_basis_is_hhv(self, db):
        """Default basis is HHV."""
        default = db.get_heating_value("NATURAL_GAS")
        hhv = db.get_heating_value("NATURAL_GAS", basis="HHV")
        assert default == hhv


# ---------------------------------------------------------------------------
# TestGetOxidationFactor
# ---------------------------------------------------------------------------

class TestGetOxidationFactor:
    """Tests for get_oxidation_factor method."""

    def test_natural_gas_ipcc_oxidation(self, db):
        """IPCC natural gas oxidation factor is 0.995."""
        of = db.get_oxidation_factor("NATURAL_GAS", source="IPCC")
        assert of == Decimal("0.995")

    def test_coal_ipcc_oxidation(self, db):
        """IPCC coal oxidation factor is 0.98."""
        of = db.get_oxidation_factor("COAL_BITUMINOUS", source="IPCC")
        assert of == Decimal("0.98")

    def test_diesel_ipcc_oxidation(self, db):
        """IPCC diesel oxidation factor is 0.99."""
        of = db.get_oxidation_factor("DIESEL", source="IPCC")
        assert of == Decimal("0.99")

    def test_epa_oxidation_all_one(self, db):
        """EPA oxidation factors are all 1.0 (complete combustion assumed)."""
        for fuel_type in _OXIDATION_FACTORS:
            of = db.get_oxidation_factor(fuel_type, source="EPA")
            assert of == Decimal("1.0"), f"{fuel_type}: EPA OF should be 1.0"

    def test_unknown_fuel_raises_key_error(self, db):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_oxidation_factor("UNKNOWN_FUEL", source="EPA")

    def test_default_source_is_epa(self, db):
        """Default source is EPA."""
        default = db.get_oxidation_factor("NATURAL_GAS")
        epa = db.get_oxidation_factor("NATURAL_GAS", source="EPA")
        assert default == epa


# ---------------------------------------------------------------------------
# TestGetGWP
# ---------------------------------------------------------------------------

class TestGetGWP:
    """Tests for get_gwp method."""

    def test_ar4_ch4(self, db):
        """AR4 CH4 GWP is 25."""
        gwp = db.get_gwp("CH4", source="AR4")
        assert gwp == Decimal("25")

    def test_ar4_n2o(self, db):
        """AR4 N2O GWP is 298."""
        gwp = db.get_gwp("N2O", source="AR4")
        assert gwp == Decimal("298")

    def test_ar5_ch4(self, db):
        """AR5 CH4 GWP is 28."""
        gwp = db.get_gwp("CH4", source="AR5")
        assert gwp == Decimal("28")

    def test_ar5_n2o(self, db):
        """AR5 N2O GWP is 265."""
        gwp = db.get_gwp("N2O", source="AR5")
        assert gwp == Decimal("265")

    def test_ar6_ch4(self, db):
        """AR6 CH4 GWP is 29.8."""
        gwp = db.get_gwp("CH4", source="AR6")
        assert gwp == Decimal("29.8")

    def test_ar6_n2o(self, db):
        """AR6 N2O GWP is 273."""
        gwp = db.get_gwp("N2O", source="AR6")
        assert gwp == Decimal("273")

    def test_co2_gwp_always_one(self, db):
        """CO2 GWP is 1 regardless of AR source."""
        for source in ("AR4", "AR5", "AR6"):
            gwp = db.get_gwp("CO2", source=source)
            assert gwp == Decimal("1")

    def test_20yr_gwp_ch4(self, db):
        """20-year GWP for CH4 is 82.5 (AR6_20YR)."""
        gwp = db.get_gwp("CH4", source="AR6", timeframe="20yr")
        assert gwp == Decimal("82.5")

    def test_default_source_is_ar6(self, db):
        """Default GWP source is AR6."""
        default = db.get_gwp("CH4")
        ar6 = db.get_gwp("CH4", source="AR6")
        assert default == ar6

    def test_invalid_timeframe_raises_value_error(self, db):
        """Invalid timeframe raises ValueError."""
        with pytest.raises(ValueError, match="timeframe must be"):
            db.get_gwp("CH4", source="AR6", timeframe="50yr")

    def test_unknown_gas_raises_key_error(self, db):
        """Unknown gas raises KeyError."""
        with pytest.raises(KeyError):
            db.get_gwp("SF6", source="AR6")

    def test_unknown_source_raises_key_error(self, db):
        """Unknown source raises KeyError."""
        with pytest.raises(KeyError):
            db.get_gwp("CH4", source="AR7")


# ---------------------------------------------------------------------------
# TestRegisterCustomFactor
# ---------------------------------------------------------------------------

class TestRegisterCustomFactor:
    """Tests for register_custom_factor method."""

    def test_register_and_retrieve(self, db):
        """Register a custom factor and retrieve it."""
        reg_id = db.register_custom_factor(
            fuel_type="NATURAL_GAS",
            gas="CO2",
            value=Decimal("55.0"),
            unit="kg/mmBtu",
            source="CUSTOM",
        )
        assert reg_id.startswith("custom_")
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="CUSTOM")
        assert ef == Decimal("55.0")

    def test_overwrite_custom_factor(self, db):
        """Registering the same fuel/gas/source overwrites the previous value."""
        db.register_custom_factor("DIESEL", "CO2", Decimal("70.0"), "kg/mmBtu")
        db.register_custom_factor("DIESEL", "CO2", Decimal("72.0"), "kg/mmBtu")
        ef = db.get_emission_factor("DIESEL", "CO2", source="CUSTOM")
        assert ef == Decimal("72.0")

    def test_negative_value_raises_value_error(self, db):
        """Negative emission factor value raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            db.register_custom_factor(
                "NATURAL_GAS", "CO2", Decimal("-1.0"), "kg/mmBtu"
            )

    def test_zero_value_accepted(self, db):
        """Zero emission factor is accepted (e.g. biogenic CO2)."""
        reg_id = db.register_custom_factor(
            "WOOD_BIOMASS", "CO2", Decimal("0"), "kg/mmBtu"
        )
        assert reg_id.startswith("custom_")

    def test_custom_factor_count_increments(self, db):
        """Factor count increases after registering a custom factor."""
        initial = db.get_factor_count()
        db.register_custom_factor("NATURAL_GAS", "CO2", Decimal("55.0"), "kg/mmBtu")
        assert db.get_factor_count() == initial + 1

    def test_custom_factor_takes_priority(self, db):
        """Custom factor is returned instead of built-in when source=CUSTOM."""
        db.register_custom_factor(
            "NATURAL_GAS", "CO2", Decimal("99.99"), "kg/mmBtu", source="CUSTOM",
        )
        ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="CUSTOM")
        assert ef == Decimal("99.99")


# ---------------------------------------------------------------------------
# TestListFuelTypes
# ---------------------------------------------------------------------------

class TestListFuelTypes:
    """Tests for list_fuel_types method."""

    def test_returns_all_fuel_types(self, db):
        """Returns all registered fuel types."""
        fuel_types = db.list_fuel_types()
        assert len(fuel_types) == len(_FUEL_PROPERTIES)

    def test_returns_sorted_list(self, db):
        """Fuel types list is sorted alphabetically."""
        fuel_types = db.list_fuel_types()
        assert fuel_types == sorted(fuel_types)

    def test_contains_natural_gas(self, db):
        """List includes NATURAL_GAS."""
        assert "NATURAL_GAS" in db.list_fuel_types()

    def test_contains_all_expected_types(self, db):
        """List includes all 20 built-in fuel types."""
        fuel_types = db.list_fuel_types()
        expected = [
            "BIOGAS", "BLAST_FURNACE_GAS", "COAL_ANTHRACITE",
            "COAL_BITUMINOUS", "COAL_SUBBITUMINOUS", "COKE_OVEN_GAS",
            "DIESEL", "JET_FUEL", "KEROSENE", "LANDFILL_GAS",
            "LIGNITE", "MOTOR_GASOLINE", "MSW", "NATURAL_GAS",
            "PEAT", "PETROLEUM_COKE", "PROPANE_LPG",
            "RESIDUAL_FUEL_OIL", "WASTE_OIL", "WOOD_BIOMASS",
        ]
        for fuel in expected:
            assert fuel in fuel_types, f"Missing fuel type: {fuel}"


# ---------------------------------------------------------------------------
# TestListEmissionFactors
# ---------------------------------------------------------------------------

class TestListEmissionFactors:
    """Tests for list_emission_factors method."""

    def test_unfiltered_returns_all(self, db):
        """Unfiltered call returns all built-in factors."""
        factors = db.list_emission_factors()
        assert len(factors) > 200

    def test_filter_by_fuel_type(self, db):
        """Filter by fuel_type returns only matching factors."""
        factors = db.list_emission_factors(fuel_type="NATURAL_GAS")
        for f in factors:
            assert f["fuel_type"] == "NATURAL_GAS"

    def test_filter_by_source(self, db):
        """Filter by source returns only matching factors."""
        factors = db.list_emission_factors(source="EPA")
        for f in factors:
            assert f["source"] == "EPA"

    def test_filter_by_gas(self, db):
        """Filter by gas returns only matching factors."""
        factors = db.list_emission_factors(gas="CO2")
        for f in factors:
            assert f["gas"] == "CO2"

    def test_combined_filters(self, db):
        """Combined fuel_type and source filters work together."""
        factors = db.list_emission_factors(
            fuel_type="NATURAL_GAS", source="EPA",
        )
        assert len(factors) == 3  # CO2, CH4, N2O
        for f in factors:
            assert f["fuel_type"] == "NATURAL_GAS"
            assert f["source"] == "EPA"

    def test_includes_custom_factors(self, db):
        """Custom factors appear in list results."""
        db.register_custom_factor(
            "NATURAL_GAS", "CO2", Decimal("55.0"), "kg/mmBtu",
        )
        factors = db.list_emission_factors(source="CUSTOM")
        assert len(factors) >= 1


# ---------------------------------------------------------------------------
# TestSearchFactors
# ---------------------------------------------------------------------------

class TestSearchFactors:
    """Tests for search_factors method."""

    def test_search_by_fuel_key(self, db):
        """Search by fuel key returns matching factors."""
        results = db.search_factors("natural_gas")
        assert len(results) > 0
        fuel_types = {r["fuel_type"] for r in results}
        assert "NATURAL_GAS" in fuel_types

    def test_search_by_display_name(self, db):
        """Search by display name returns matching factors."""
        results = db.search_factors("diesel")
        assert len(results) > 0

    def test_search_case_insensitive(self, db):
        """Search is case-insensitive."""
        upper = db.search_factors("COAL")
        lower = db.search_factors("coal")
        assert len(upper) == len(lower)

    def test_search_no_results(self, db):
        """Non-matching query returns empty list."""
        results = db.search_factors("unobtainium")
        assert results == []

    def test_search_partial_match(self, db):
        """Partial name match works."""
        results = db.search_factors("wood")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# TestIsBiogenic
# ---------------------------------------------------------------------------

class TestIsBiogenic:
    """Tests for is_biogenic method."""

    @pytest.mark.parametrize("fuel_type", [
        "WOOD_BIOMASS", "BIOGAS", "LANDFILL_GAS",
    ])
    def test_biogenic_fuels_return_true(self, db, fuel_type):
        """Biogenic fuels return True."""
        assert db.is_biogenic(fuel_type) is True

    @pytest.mark.parametrize("fuel_type", [
        "NATURAL_GAS", "DIESEL", "COAL_BITUMINOUS", "MOTOR_GASOLINE",
        "KEROSENE", "PETROLEUM_COKE", "RESIDUAL_FUEL_OIL",
    ])
    def test_fossil_fuels_return_false(self, db, fuel_type):
        """Fossil fuels return False."""
        assert db.is_biogenic(fuel_type) is False

    def test_unknown_fuel_raises_key_error(self, db):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError):
            db.is_biogenic("UNKNOWN_FUEL")


# ---------------------------------------------------------------------------
# TestConvertEFUnits
# ---------------------------------------------------------------------------

class TestConvertEFUnits:
    """Tests for convert_ef_units method."""

    def test_same_unit_returns_identity(self, db):
        """Converting from a unit to itself returns the same value."""
        val = Decimal("53.06")
        result = db.convert_ef_units(val, "kg/mmBtu", "kg/mmBtu", "NATURAL_GAS")
        assert result == val

    def test_kg_mmbtu_to_kg_gj(self, db):
        """Convert kg/mmBtu to kg/GJ."""
        val = Decimal("53.06")
        result = db.convert_ef_units(val, "kg/mmBtu", "kg/GJ", "NATURAL_GAS")
        # kg/GJ = kg/mmBtu / 1.055056 (smaller number)
        assert result < val
        assert result > 0

    def test_kg_gj_to_kg_tj(self, db):
        """Convert kg/GJ to kg/TJ (multiply by 1000)."""
        val = Decimal("50.0")
        result = db.convert_ef_units(val, "kg/GJ", "kg/TJ", "NATURAL_GAS")
        assert result == Decimal("50000")

    def test_t_tj_to_kg_gj(self, db):
        """Convert t/TJ to kg/GJ (identity in implementation)."""
        val = Decimal("56.1")
        result = db.convert_ef_units(val, "t/TJ", "kg/GJ", "NATURAL_GAS")
        assert result == val

    def test_unsupported_from_unit_raises(self, db):
        """Unsupported source unit raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported source unit"):
            db.convert_ef_units(Decimal("1"), "kg/bbl", "kg/GJ", "DIESEL")

    def test_unsupported_to_unit_raises(self, db):
        """Unsupported target unit raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported target unit"):
            db.convert_ef_units(Decimal("1"), "kg/GJ", "kg/bbl", "DIESEL")

    def test_round_trip_kg_mmbtu_kg_gj(self, db):
        """Round-trip conversion kg/mmBtu -> kg/GJ -> kg/mmBtu preserves value."""
        original = Decimal("53.06")
        intermediate = db.convert_ef_units(
            original, "kg/mmBtu", "kg/GJ", "NATURAL_GAS",
        )
        recovered = db.convert_ef_units(
            intermediate, "kg/GJ", "kg/mmBtu", "NATURAL_GAS",
        )
        assert abs(recovered - original) < Decimal("0.001")


# ---------------------------------------------------------------------------
# TestGetHHVToNCVRatio
# ---------------------------------------------------------------------------

class TestGetHHVToNCVRatio:
    """Tests for get_hhv_to_ncv_ratio method."""

    def test_natural_gas_ratio(self, db):
        """Natural gas HHV/NCV ratio is approximately 1.105."""
        ratio = db.get_hhv_to_ncv_ratio("NATURAL_GAS")
        assert Decimal("1.05") < ratio < Decimal("1.15")

    def test_diesel_ratio(self, db):
        """Diesel HHV/NCV ratio is approximately 1.065."""
        ratio = db.get_hhv_to_ncv_ratio("DIESEL")
        assert Decimal("1.03") < ratio < Decimal("1.10")

    def test_coal_bituminous_ratio(self, db):
        """Coal bituminous HHV/NCV ratio is approximately 1.05."""
        ratio = db.get_hhv_to_ncv_ratio("COAL_BITUMINOUS")
        assert Decimal("1.02") < ratio < Decimal("1.10")

    def test_all_ratios_greater_than_one(self, db):
        """HHV/NCV ratio is > 1.0 for all fuels."""
        for fuel_type in _HEATING_VALUES:
            ratio = db.get_hhv_to_ncv_ratio(fuel_type)
            assert ratio > Decimal("1.0"), (
                f"{fuel_type}: HHV/NCV ratio ({ratio}) should be > 1.0"
            )

    def test_unknown_fuel_raises_key_error(self, db):
        """Unknown fuel type raises KeyError."""
        with pytest.raises(KeyError):
            db.get_hhv_to_ncv_ratio("UNKNOWN_FUEL")


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Thread safety tests for concurrent FuelDatabaseEngine access."""

    def test_concurrent_reads(self, db):
        """Concurrent reads from multiple threads do not raise errors."""
        errors = []

        def read_factors():
            try:
                for _ in range(100):
                    db.get_emission_factor("NATURAL_GAS", "CO2", source="EPA")
                    db.get_heating_value("DIESEL", basis="HHV")
                    db.get_gwp("CH4", source="AR6")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_factors) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_writes(self, db):
        """Concurrent custom factor registrations do not corrupt state."""
        errors = []

        def register_factors(thread_id):
            try:
                for i in range(50):
                    db.register_custom_factor(
                        fuel_type=f"CUSTOM_FUEL_{thread_id}_{i}",
                        gas="CO2",
                        value=Decimal(f"{50 + thread_id}.{i}"),
                        unit="kg/mmBtu",
                        source="CUSTOM",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_factors, args=(tid,))
            for tid in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # 5 threads x 50 factors = 250 custom factors
        assert len(db._custom_factors) == 250
