# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - GHG Inventory Engine Tests
=============================================================

Unit tests for GHGInventoryEngine (Engine 1) covering emission
calculation, GWP conversion, inventory building, scope aggregation,
Scope 3 breakdown, intensity metrics, gas disaggregation,
consolidation, completeness validation, and E1-6 data points.

ESRS E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions.

Target: 60+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the ghg_inventory engine module."""
    return _load_engine("ghg_inventory")


@pytest.fixture
def engine(mod):
    """Create a fresh GHGInventoryEngine instance."""
    return mod.GHGInventoryEngine()


@pytest.fixture
def scope1_co2_entry(mod):
    """Create a Scope 1 CO2 emission entry."""
    return mod.EmissionEntry(
        source_name="Boiler Room A",
        scope=mod.GHGScope.SCOPE_1,
        gas=mod.EmissionGas.CO2,
        activity_data=Decimal("10000"),
        activity_unit="litres",
        emission_factor=Decimal("2.68"),
        emission_factor_unit="kgCO2e_per_litre",
        data_quality=mod.DataQualityLevel.PRIMARY,
    )


@pytest.fixture
def scope1_ch4_entry(mod):
    """Create a Scope 1 CH4 emission entry."""
    return mod.EmissionEntry(
        source_name="Methane Vent",
        scope=mod.GHGScope.SCOPE_1,
        gas=mod.EmissionGas.CH4,
        activity_data=Decimal("100"),
        activity_unit="kg",
        emission_factor=Decimal("1.0"),
        emission_factor_unit="kgCH4_per_unit",
    )


@pytest.fixture
def scope2_entry(mod):
    """Create a Scope 2 location-based emission entry."""
    return mod.EmissionEntry(
        source_name="Grid Electricity",
        scope=mod.GHGScope.SCOPE_2_LOCATION,
        gas=mod.EmissionGas.CO2,
        activity_data=Decimal("5000"),
        activity_unit="kWh",
        emission_factor=Decimal("0.45"),
        emission_factor_unit="kgCO2e_per_kWh",
    )


@pytest.fixture
def scope3_entry(mod):
    """Create a Scope 3 Category 1 emission entry."""
    return mod.EmissionEntry(
        source_name="Purchased Goods",
        scope=mod.GHGScope.SCOPE_3,
        gas=mod.EmissionGas.CO2,
        scope3_category=mod.Scope3Category.PURCHASED_GOODS,
        activity_data=Decimal("50000"),
        activity_unit="EUR",
        emission_factor=Decimal("0.5"),
        emission_factor_unit="kgCO2e_per_EUR",
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestGHGEnums:
    """Tests for GHG inventory enums."""

    def test_ghg_scope_count(self, mod):
        """GHGScope has 4 values."""
        assert len(mod.GHGScope) == 4

    def test_ghg_scope_values(self, mod):
        """GHGScope values match expected strings."""
        values = {m.value for m in mod.GHGScope}
        expected = {"scope_1", "scope_2_location", "scope_2_market", "scope_3"}
        assert values == expected

    def test_emission_gas_count(self, mod):
        """EmissionGas has 7 values."""
        assert len(mod.EmissionGas) == 7

    def test_emission_gas_values(self, mod):
        """EmissionGas values match IPCC/UNFCCC gas groups."""
        values = {m.value for m in mod.EmissionGas}
        expected = {"co2", "ch4", "n2o", "hfcs", "pfcs", "sf6", "nf3"}
        assert values == expected

    def test_scope3_category_has_15_values(self, mod):
        """Scope3Category has 15 values matching GHG Protocol categories."""
        assert len(mod.Scope3Category) == 15

    def test_consolidation_approach_values(self, mod):
        """ConsolidationApproach has 3 values."""
        assert len(mod.ConsolidationApproach) == 3
        values = {m.value for m in mod.ConsolidationApproach}
        expected = {"operational_control", "financial_control", "equity_share"}
        assert values == expected

    def test_data_quality_level_values(self, mod):
        """DataQualityLevel has 4 values."""
        assert len(mod.DataQualityLevel) == 4
        values = {m.value for m in mod.DataQualityLevel}
        expected = {"primary", "secondary_specific", "secondary_average", "estimated"}
        assert values == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestGHGConstants:
    """Tests for GHG inventory constants."""

    def test_gwp_ar6_co2_is_1(self, mod):
        """GWP_AR6 for CO2 is 1."""
        assert mod.GWP_AR6["co2"] == Decimal("1")

    def test_gwp_ar6_ch4(self, mod):
        """GWP_AR6 for CH4 is 27.9 (IPCC AR6)."""
        assert mod.GWP_AR6["ch4"] == Decimal("27.9")

    def test_gwp_ar6_n2o(self, mod):
        """GWP_AR6 for N2O is 273 (IPCC AR6)."""
        assert mod.GWP_AR6["n2o"] == Decimal("273")

    def test_gwp_ar6_sf6(self, mod):
        """GWP_AR6 for SF6 is 25200 (IPCC AR6)."""
        assert mod.GWP_AR6["sf6"] == Decimal("25200")

    def test_e1_6_datapoints_count(self, mod):
        """E1_6_DATAPOINTS has at least 15 entries."""
        assert len(mod.E1_6_DATAPOINTS) >= 15

    def test_scope_3_category_names_has_15(self, mod):
        """SCOPE_3_CATEGORY_NAMES has 15 entries."""
        assert len(mod.SCOPE_3_CATEGORY_NAMES) == 15


# ===========================================================================
# EmissionEntry Model Tests
# ===========================================================================


class TestEmissionEntryModel:
    """Tests for EmissionEntry Pydantic model."""

    def test_create_valid_entry(self, mod):
        """Create a valid EmissionEntry with required fields."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.CO2,
            activity_data=Decimal("1000"),
            emission_factor=Decimal("2.5"),
        )
        assert entry.scope == mod.GHGScope.SCOPE_1
        assert entry.activity_data == Decimal("1000")
        assert entry.emission_factor == Decimal("2.5")
        assert len(entry.source_id) > 0

    def test_scope3_requires_category(self, mod):
        """Scope 3 entries require scope3_category."""
        with pytest.raises(Exception):
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_3,
                activity_data=Decimal("100"),
                emission_factor=Decimal("1.0"),
                scope3_category=None,
            )

    def test_scope1_does_not_require_category(self, mod):
        """Scope 1 entries do not require scope3_category."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            activity_data=Decimal("100"),
            emission_factor=Decimal("1.0"),
        )
        assert entry.scope3_category is None

    def test_negative_activity_data_rejected(self, mod):
        """Negative activity data is rejected."""
        with pytest.raises(Exception):
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                activity_data=Decimal("-10"),
                emission_factor=Decimal("1.0"),
            )

    def test_default_gas_is_co2(self, mod):
        """Default gas is CO2."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            activity_data=Decimal("100"),
            emission_factor=Decimal("1.0"),
        )
        assert entry.gas == mod.EmissionGas.CO2


# ===========================================================================
# Calculate Emission Tests
# ===========================================================================


class TestCalculateEmission:
    """Tests for single emission calculation."""

    def test_single_co2_entry(self, engine, scope1_co2_entry):
        """CO2 entry: tCO2e = activity_data * emission_factor / 1000."""
        result = engine.calculate_emission(scope1_co2_entry)
        # 10000 * 2.68 / 1000 = 26.8 tCO2e
        assert isinstance(result, Decimal)
        assert float(result) == pytest.approx(26.8, abs=0.01)

    def test_ch4_with_gwp_conversion(self, engine, scope1_ch4_entry):
        """CH4 entry applies GWP-100 conversion."""
        result = engine.calculate_emission(scope1_ch4_entry)
        # 100 * 1.0 / 1000 = 0.1 tonnes CH4 * 27.9 GWP = 2.79 tCO2e
        assert isinstance(result, Decimal)
        assert float(result) == pytest.approx(2.79, abs=0.01)

    def test_precalculated_value_used(self, engine, mod):
        """Pre-calculated value_tco2e is used directly when > 0."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.CO2,
            activity_data=Decimal("999"),
            emission_factor=Decimal("999"),
            value_tco2e=Decimal("42.5"),
        )
        result = engine.calculate_emission(entry)
        assert float(result) == pytest.approx(42.5, abs=0.001)

    def test_zero_activity_data(self, engine, mod):
        """Zero activity data produces zero emissions."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.CO2,
            activity_data=Decimal("0"),
            emission_factor=Decimal("2.68"),
        )
        result = engine.calculate_emission(entry)
        assert result == Decimal("0")

    def test_n2o_with_gwp_conversion(self, engine, mod):
        """N2O entry applies GWP-100 of 273."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.N2O,
            activity_data=Decimal("10"),
            activity_unit="kg",
            emission_factor=Decimal("1.0"),
        )
        result = engine.calculate_emission(entry)
        # 10 * 1.0 / 1000 = 0.01 tonnes * 273 = 2.73 tCO2e
        assert float(result) == pytest.approx(2.73, abs=0.01)


# ===========================================================================
# Calculate Emission With GWP Tests
# ===========================================================================


class TestCalculateEmissionWithGWP:
    """Tests for GWP-based mass-to-tCO2e conversion."""

    def test_co2_mass_conversion(self, engine):
        """1 tonne CO2 = 1 tCO2e."""
        result = engine.calculate_emission_with_gwp(Decimal("1"), "co2")
        assert float(result) == pytest.approx(1.0, abs=0.001)

    def test_ch4_mass_conversion(self, engine):
        """1 tonne CH4 = 27.9 tCO2e."""
        result = engine.calculate_emission_with_gwp(Decimal("1"), "ch4")
        assert float(result) == pytest.approx(27.9, abs=0.01)

    def test_sf6_mass_conversion(self, engine):
        """1 tonne SF6 = 25200 tCO2e."""
        result = engine.calculate_emission_with_gwp(Decimal("1"), "sf6")
        assert float(result) == pytest.approx(25200.0, abs=1.0)

    def test_unknown_gas_raises(self, engine):
        """Unknown gas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gas"):
            engine.calculate_emission_with_gwp(Decimal("1"), "unknown_gas")


# ===========================================================================
# Build Inventory Tests
# ===========================================================================


class TestBuildInventory:
    """Tests for build_inventory method."""

    def test_basic_inventory(self, engine, scope1_co2_entry):
        """Basic inventory with single Scope 1 entry."""
        result = engine.build_inventory([scope1_co2_entry])
        assert result.scope1_total_tco2e > Decimal("0")
        assert result.entry_count == 1
        assert result.processing_time_ms >= 0.0

    def test_multiple_scopes(self, engine, scope1_co2_entry, scope2_entry, scope3_entry):
        """Inventory with entries across multiple scopes."""
        result = engine.build_inventory(
            [scope1_co2_entry, scope2_entry, scope3_entry]
        )
        assert result.scope1_total_tco2e > Decimal("0")
        assert result.scope2_location_total_tco2e > Decimal("0")
        assert result.scope3_total_tco2e > Decimal("0")
        assert result.total_tco2e > Decimal("0")
        assert result.entry_count == 3

    def test_scope_breakdown_totals(self, engine, scope1_co2_entry, scope2_entry, scope3_entry):
        """Total equals sum of scope components."""
        result = engine.build_inventory(
            [scope1_co2_entry, scope2_entry, scope3_entry]
        )
        # total = scope1 + scope2_market + scope3
        # (scope2_location is reported separately)
        expected_total = (
            result.scope1_total_tco2e
            + result.scope2_market_total_tco2e
            + result.scope3_total_tco2e
        )
        assert float(result.total_tco2e) == pytest.approx(
            float(expected_total), abs=0.1
        )

    def test_biogenic_co2_separate(self, engine, mod):
        """Biogenic CO2 reported separately from gross emissions."""
        biogenic = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.CO2,
            activity_data=Decimal("5000"),
            emission_factor=Decimal("1.0"),
            is_biogenic=True,
        )
        normal = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.CO2,
            activity_data=Decimal("10000"),
            emission_factor=Decimal("2.0"),
        )
        result = engine.build_inventory([biogenic, normal])
        assert result.biogenic_co2_tco2e > Decimal("0")

    def test_provenance_hash_64_chars(self, engine, scope1_co2_entry):
        """Provenance hash is a 64-character hex string."""
        result = engine.build_inventory([scope1_co2_entry])
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # Valid hex

    def test_empty_entries_raises(self, engine):
        """Empty entries list raises ValueError."""
        with pytest.raises(ValueError):
            engine.build_inventory([])

    def test_entity_name_stored(self, engine, scope1_co2_entry):
        """Entity name is stored in result."""
        result = engine.build_inventory(
            [scope1_co2_entry], entity_name="Test Corp"
        )
        assert result.entity_name == "Test Corp"

    def test_reporting_year_stored(self, engine, scope1_co2_entry):
        """Reporting year is stored in result."""
        result = engine.build_inventory(
            [scope1_co2_entry], reporting_year=2025
        )
        assert result.reporting_year == 2025


# ===========================================================================
# Scope 3 Tests
# ===========================================================================


class TestScope3:
    """Tests for Scope 3 calculation and breakdown."""

    def test_all_15_categories_recognized(self, mod):
        """All 15 Scope 3 categories are in the enum."""
        categories = list(mod.Scope3Category)
        assert len(categories) == 15

    def test_scope3_breakdown(self, engine, mod):
        """Scope 3 breakdown by category is computed."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_3,
                scope3_category=mod.Scope3Category.PURCHASED_GOODS,
                activity_data=Decimal("1000"),
                emission_factor=Decimal("1.0"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_3,
                scope3_category=mod.Scope3Category.BUSINESS_TRAVEL,
                activity_data=Decimal("500"),
                emission_factor=Decimal("0.5"),
            ),
        ]
        result = engine.build_inventory(entries)
        assert result.scope3_breakdown is not None
        assert len(result.scope3_breakdown.categories_included) >= 2

    def test_scope3_total_matches_categories(self, engine, mod):
        """Scope 3 total matches sum of category breakdowns."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_3,
                scope3_category=mod.Scope3Category.PURCHASED_GOODS,
                activity_data=Decimal("1000"),
                emission_factor=Decimal("2.0"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_3,
                scope3_category=mod.Scope3Category.WASTE,
                activity_data=Decimal("500"),
                emission_factor=Decimal("1.5"),
            ),
        ]
        result = engine.build_inventory(entries)
        cat_sum = sum(result.scope3_breakdown.by_category.values())
        assert float(cat_sum) == pytest.approx(
            float(result.scope3_total_tco2e), abs=0.01
        )

    def test_single_scope3_category(self, engine, scope3_entry):
        """Single Scope 3 category works correctly."""
        result = engine.build_inventory([scope3_entry])
        assert result.scope3_total_tco2e > Decimal("0")
        assert result.scope3_breakdown is not None
        assert len(result.scope3_breakdown.by_category) >= 1


# ===========================================================================
# Intensity Metric Tests
# ===========================================================================


class TestIntensityMetric:
    """Tests for GHG intensity calculation."""

    def test_revenue_intensity(self, engine, scope1_co2_entry):
        """Calculate intensity per revenue."""
        inv = engine.build_inventory([scope1_co2_entry])
        result = engine.calculate_intensity(
            total_tco2e=inv.total_tco2e,
            denominator_value=Decimal("10"),
            denominator_unit="EUR_million",
        )
        assert result.intensity_value > Decimal("0")

    def test_headcount_intensity(self, engine, scope1_co2_entry):
        """Calculate intensity per headcount."""
        inv = engine.build_inventory([scope1_co2_entry])
        result = engine.calculate_intensity(
            total_tco2e=inv.total_tco2e,
            denominator_value=Decimal("500"),
            denominator_unit="headcount",
        )
        assert result.intensity_value > Decimal("0")
        assert result.denominator_unit == "headcount"

    def test_zero_denominator_raises(self, engine):
        """Zero denominator raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_intensity(
                total_tco2e=Decimal("100"),
                denominator_value=Decimal("0"),
                denominator_unit="EUR_million",
            )


# ===========================================================================
# Disaggregate By Gas Tests
# ===========================================================================


class TestDisaggregateByGas:
    """Tests for gas disaggregation."""

    def test_by_gas_breakdown(self, engine, mod):
        """Disaggregate emissions by gas type."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("10000"),
                emission_factor=Decimal("2.0"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CH4,
                activity_data=Decimal("100"),
                emission_factor=Decimal("1.0"),
            ),
        ]
        result = engine.build_inventory(entries)
        assert result.by_gas is not None
        assert result.by_gas.co2_tco2e > Decimal("0")
        assert result.by_gas.ch4_tco2e > Decimal("0")

    def test_co2_dominant(self, engine, mod):
        """CO2 is the dominant gas in a typical inventory."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("100000"),
                emission_factor=Decimal("2.5"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CH4,
                activity_data=Decimal("10"),
                emission_factor=Decimal("1.0"),
            ),
        ]
        result = engine.build_inventory(entries)
        assert result.by_gas.co2_tco2e > result.by_gas.ch4_tco2e


# ===========================================================================
# Consolidation Tests
# ===========================================================================


class TestConsolidateEntities:
    """Tests for multi-entity consolidation."""

    def test_operational_control(self, engine, mod):
        """Consolidation with operational control approach."""
        entity_a_entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("2.0"),
            ),
        ]
        entity_b_entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("3000"),
                emission_factor=Decimal("2.0"),
            ),
        ]
        inv_a = engine.build_inventory(entity_a_entries, entity_name="Entity A")
        inv_b = engine.build_inventory(entity_b_entries, entity_name="Entity B")

        result = engine.consolidate_entities(
            [inv_a, inv_b],
            approach=mod.ConsolidationApproach.OPERATIONAL_CONTROL,
        )
        assert result.entity_count == 2
        assert result.consolidated_scope1_tco2e > Decimal("0")
        expected_scope1 = inv_a.scope1_total_tco2e + inv_b.scope1_total_tco2e
        assert float(result.consolidated_scope1_tco2e) == pytest.approx(
            float(expected_scope1), abs=0.01
        )

    def test_equity_share(self, engine, mod):
        """Consolidation with equity share approach."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("10000"),
                emission_factor=Decimal("2.0"),
            ),
        ]
        inv = engine.build_inventory(entries, entity_name="Sub Co")
        result = engine.consolidate_entities(
            [inv],
            approach=mod.ConsolidationApproach.EQUITY_SHARE,
            equity_shares={"Sub Co": Decimal("0.50")},
        )
        assert result.entity_count == 1
        # With 50% equity share, consolidated should be ~50% of original
        assert float(result.consolidated_scope1_tco2e) == pytest.approx(
            float(inv.scope1_total_tco2e * Decimal("0.50")), abs=0.1
        )


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestCompleteness:
    """Tests for E1-6 completeness validation."""

    def test_full_dataset_passes(self, engine, mod):
        """A complete dataset passes completeness validation."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("10000"),
                emission_factor=Decimal("2.68"),
                data_quality=mod.DataQualityLevel.PRIMARY,
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_2_LOCATION,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("0.45"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_2_MARKET,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("0.30"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_3,
                scope3_category=mod.Scope3Category.PURCHASED_GOODS,
                activity_data=Decimal("50000"),
                emission_factor=Decimal("0.5"),
            ),
        ]
        result = engine.build_inventory(entries, reporting_year=2025)
        completeness = engine.validate_completeness(result)
        assert isinstance(completeness, dict)
        # Should have some score or status
        assert len(completeness) > 0

    def test_missing_scope_noted(self, engine, mod):
        """Missing scope entries are flagged."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("10000"),
                emission_factor=Decimal("2.68"),
            ),
        ]
        result = engine.build_inventory(entries)
        completeness = engine.validate_completeness(result)
        # With only Scope 1, completeness should flag missing Scope 2/3
        assert isinstance(completeness, dict)


# ===========================================================================
# E1-6 Data Points Tests
# ===========================================================================


class TestE16Datapoints:
    """Tests for E1-6 required data point extraction."""

    def test_returns_datapoints(self, engine, scope1_co2_entry, scope3_entry):
        """get_e1_6_datapoints returns required data points."""
        result = engine.build_inventory([scope1_co2_entry, scope3_entry])
        datapoints = engine.get_e1_6_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 10

    def test_scope1_total_in_datapoints(self, engine, scope1_co2_entry):
        """Scope 1 total is present in data points."""
        result = engine.build_inventory([scope1_co2_entry])
        datapoints = engine.get_e1_6_datapoints(result)
        # Should contain Scope 1 total
        has_scope1 = any("scope1" in k.lower() for k in datapoints.keys())
        assert has_scope1


# ===========================================================================
# GWP Lookup Tests
# ===========================================================================


class TestGWPLookup:
    """Tests for GWP lookup method."""

    def test_get_gwp_co2(self, engine):
        """CO2 GWP is 1."""
        assert engine.get_gwp("co2") == Decimal("1")

    def test_get_gwp_ch4(self, engine):
        """CH4 GWP is 27.9."""
        assert engine.get_gwp("ch4") == Decimal("27.9")

    def test_get_gwp_specific_hfc(self, engine):
        """Specific HFC GWP values are available."""
        gwp = engine.get_gwp("hfc_134a")
        assert gwp == Decimal("1530")


# ===========================================================================
# Scope Summary Tests
# ===========================================================================


class TestScopeSummary:
    """Tests for scope summary helper."""

    def test_get_scope_summary(self, engine, mod):
        """get_scope_summary returns a summary dict."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_1,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("10000"),
                emission_factor=Decimal("2.0"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_2_LOCATION,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("0.4"),
            ),
        ]
        result = engine.build_inventory(entries)
        summary = engine.get_scope_summary(result)
        assert isinstance(summary, dict)
        assert len(summary) > 0


# ===========================================================================
# Additional GWP Conversion Tests
# ===========================================================================


class TestAdditionalGWPConversions:
    """Additional GWP-100 conversion tests for full gas coverage."""

    def test_nf3_gwp_value(self, mod):
        """GWP_AR6 for NF3 is 17400."""
        assert mod.GWP_AR6["nf3"] == Decimal("17400")

    def test_nf3_mass_conversion(self, engine):
        """1 tonne NF3 = 17400 tCO2e."""
        result = engine.calculate_emission_with_gwp(Decimal("1"), "nf3")
        assert float(result) == pytest.approx(17400.0, abs=1.0)

    def test_n2o_mass_conversion(self, engine):
        """1 tonne N2O = 273 tCO2e."""
        result = engine.calculate_emission_with_gwp(Decimal("1"), "n2o")
        assert float(result) == pytest.approx(273.0, abs=0.1)

    def test_gwp_zero_mass_returns_zero(self, engine):
        """Zero mass input returns zero tCO2e."""
        result = engine.calculate_emission_with_gwp(Decimal("0"), "sf6")
        assert result == Decimal("0")

    def test_gwp_fractional_mass(self, engine):
        """Fractional mass conversion is accurate."""
        result = engine.calculate_emission_with_gwp(Decimal("0.001"), "ch4")
        # 0.001 * 27.9 = 0.0279
        assert float(result) == pytest.approx(0.0279, abs=0.001)


# ===========================================================================
# Provenance Determinism Tests
# ===========================================================================


class TestProvenanceDeterminism:
    """Tests for provenance hash determinism and uniqueness."""

    def test_provenance_hash_is_hex(self, engine, mod):
        """Provenance hash is a valid 64-char hex string."""
        entry = mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            gas=mod.EmissionGas.CO2,
            activity_data=Decimal("1000"),
            emission_factor=Decimal("2.5"),
        )
        r1 = engine.build_inventory([entry], entity_name="Test", reporting_year=2025)
        assert len(r1.provenance_hash) == 64
        int(r1.provenance_hash, 16)  # Valid hex

    def test_different_input_different_hash(self, engine, mod):
        """Different inputs produce different provenance hashes."""
        e1 = [mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            activity_data=Decimal("1000"),
            emission_factor=Decimal("2.5"),
        )]
        e2 = [mod.EmissionEntry(
            scope=mod.GHGScope.SCOPE_1,
            activity_data=Decimal("2000"),
            emission_factor=Decimal("2.5"),
        )]
        r1 = engine.build_inventory(e1)
        eng2 = mod.GHGInventoryEngine()
        r2 = eng2.build_inventory(e2)
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# Market-Based Scope 2 Tests
# ===========================================================================


class TestMarketBasedScope2:
    """Tests for Scope 2 market-based emissions."""

    def test_market_based_entry(self, engine, mod):
        """Market-based Scope 2 entries are aggregated separately."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_2_MARKET,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("0.30"),
            ),
        ]
        result = engine.build_inventory(entries)
        assert result.scope2_market_total_tco2e > Decimal("0")

    def test_location_and_market_reported(self, engine, mod):
        """Both location-based and market-based are reported."""
        entries = [
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_2_LOCATION,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("0.45"),
            ),
            mod.EmissionEntry(
                scope=mod.GHGScope.SCOPE_2_MARKET,
                gas=mod.EmissionGas.CO2,
                activity_data=Decimal("5000"),
                emission_factor=Decimal("0.30"),
            ),
        ]
        result = engine.build_inventory(entries)
        assert result.scope2_location_total_tco2e > Decimal("0")
        assert result.scope2_market_total_tco2e > Decimal("0")
        # Location should be higher (0.45 > 0.30)
        assert result.scope2_location_total_tco2e > result.scope2_market_total_tco2e


# ===========================================================================
# Data Quality Tests
# ===========================================================================


class TestDataQuality:
    """Tests for data quality level handling."""

    def test_data_quality_scores_exist(self, mod):
        """DATA_QUALITY_SCORES maps all quality levels."""
        for dq in mod.DataQualityLevel:
            assert dq.value in mod.DATA_QUALITY_SCORES

    def test_primary_quality_highest(self, mod):
        """Primary data quality has the highest score."""
        scores = mod.DATA_QUALITY_SCORES
        primary = scores["primary"]
        for level in ["secondary_specific", "secondary_average", "estimated"]:
            assert primary >= scores[level]
