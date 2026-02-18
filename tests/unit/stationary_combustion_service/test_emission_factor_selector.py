# -*- coding: utf-8 -*-
"""
Unit tests for EmissionFactorSelectorEngine (Engine 4) - AGENT-MRV-001

Tests all methods of EmissionFactorSelectorEngine with 50+ tests covering:
- Initialization with optional fuel_database
- Tier 1 IPCC default factor selection
- Tier 2 country-specific factor selection (EPA/DEFRA/EU ETS)
- Tier 3 custom/facility-specific factor selection
- Automatic tier recommendation based on data quality
- Fallback chain (Tier 3 -> Tier 2 -> Tier 1)
- Multi-gas factor selection (CO2, CH4, N2O)
- Geography-to-source coverage mapping
- Recommended source by geography
- Custom factor validation against IPCC plausible ranges
- Source comparison across all databases
- Selection trace and audit log
- Selection statistics aggregation
- Manual selection recording

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal

import pytest

from greenlang.stationary_combustion.emission_factor_selector import (
    EmissionFactorSelectorEngine,
    IPCC_DEFAULT_FACTORS,
    EPA_FACTORS,
    DEFRA_FACTORS,
    EU_ETS_FACTORS,
    _GEOGRAPHY_SOURCE_MAP,
    _SOURCE_GEOGRAPHY_COVERAGE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def selector():
    """Create an EmissionFactorSelectorEngine with no fuel database."""
    return EmissionFactorSelectorEngine()


@pytest.fixture
def selector_with_custom(selector):
    """Create a selector with one custom factor registered."""
    selector.register_custom_factor(
        "NATURAL_GAS", "CO2", Decimal("55.500"),
        unit="kg/GJ", reference="Lab test 2025-12",
    )
    return selector


# ---------------------------------------------------------------------------
# TestSelectorInit
# ---------------------------------------------------------------------------

class TestSelectorInit:
    """Tests for EmissionFactorSelectorEngine initialization."""

    def test_initializes_without_fuel_database(self):
        """Engine initializes with no fuel database."""
        engine = EmissionFactorSelectorEngine()
        assert engine._fuel_database is None

    def test_initializes_with_empty_custom_factors(self, selector):
        """Custom factors start empty."""
        assert selector._custom_factors == {}

    def test_initializes_with_empty_selection_log(self, selector):
        """Selection log starts empty."""
        assert selector.get_selection_trace() == []


# ---------------------------------------------------------------------------
# TestSelectFactorTier1
# ---------------------------------------------------------------------------

class TestSelectFactorTier1:
    """Tests for Tier 1 IPCC default factor selection."""

    def test_ipcc_natural_gas_co2(self, selector):
        """IPCC natural gas CO2 factor selected for unspecified geography."""
        result = selector.select_factor("NATURAL_GAS", "CO2")
        assert result["tier"] == 1
        assert result["source"] == "IPCC"
        assert result["value"] == Decimal("56.100")

    def test_ipcc_diesel_co2(self, selector):
        """IPCC diesel CO2 factor is 74.100 kg/GJ."""
        result = selector.select_factor("DIESEL", "CO2")
        assert result["value"] == Decimal("74.100")
        assert result["source"] == "IPCC"

    def test_ipcc_coal_bituminous_co2(self, selector):
        """IPCC coal bituminous CO2 factor is 94.600 kg/GJ."""
        result = selector.select_factor("COAL_BITUMINOUS", "CO2")
        assert result["value"] == Decimal("94.600")

    def test_tier1_explicit_override(self, selector):
        """Explicit tier=1 forces IPCC selection."""
        result = selector.select_factor(
            "NATURAL_GAS", "CO2", geography="US", tier=1,
        )
        assert result["tier"] == 1
        assert result["source"] == "IPCC"

    def test_tier1_ch4_factor(self, selector):
        """Tier 1 CH4 factor is available."""
        result = selector.select_factor("NATURAL_GAS", "CH4")
        assert result["value"] > 0

    def test_tier1_n2o_factor(self, selector):
        """Tier 1 N2O factor is available."""
        result = selector.select_factor("NATURAL_GAS", "N2O")
        assert result["value"] > 0


# ---------------------------------------------------------------------------
# TestSelectFactorTier2
# ---------------------------------------------------------------------------

class TestSelectFactorTier2:
    """Tests for Tier 2 country-specific factor selection."""

    def test_us_geography_selects_epa(self, selector):
        """US geography selects EPA as Tier 2 source."""
        result = selector.select_factor("NATURAL_GAS", "CO2", geography="US")
        assert result["tier"] == 2
        assert result["source"] == "EPA"
        assert result["value"] == Decimal("53.060")

    def test_uk_geography_selects_defra(self, selector):
        """UK geography selects DEFRA as Tier 2 source."""
        result = selector.select_factor("NATURAL_GAS", "CO2", geography="UK")
        assert result["tier"] == 2
        assert result["source"] == "DEFRA"
        assert result["value"] == Decimal("56.010")

    def test_eu_geography_selects_eu_ets(self, selector):
        """EU geography selects EU ETS as Tier 2 source."""
        result = selector.select_factor("NATURAL_GAS", "CO2", geography="DE")
        assert result["tier"] == 2
        assert result["source"] == "EU_ETS"

    def test_tier2_explicit_override(self, selector):
        """Explicit tier=2 with geography works."""
        result = selector.select_factor(
            "NATURAL_GAS", "CO2", geography="US", tier=2,
        )
        assert result["tier"] == 2
        assert result["source"] == "EPA"

    def test_source_override_epa(self, selector):
        """Explicit source=EPA forces EPA selection."""
        result = selector.select_factor(
            "NATURAL_GAS", "CO2", source="EPA",
        )
        assert result["source"] == "EPA"

    def test_source_override_defra(self, selector):
        """Explicit source=DEFRA forces DEFRA selection."""
        result = selector.select_factor(
            "NATURAL_GAS", "CO2", source="DEFRA",
        )
        assert result["source"] == "DEFRA"


# ---------------------------------------------------------------------------
# TestSelectFactorTier3
# ---------------------------------------------------------------------------

class TestSelectFactorTier3:
    """Tests for Tier 3 facility-specific factor selection."""

    def test_custom_factor_selected_as_tier3(self, selector_with_custom):
        """Registered custom factor is selected as Tier 3."""
        result = selector_with_custom.select_factor("NATURAL_GAS", "CO2")
        assert result["tier"] == 3
        assert result["source"] == "CUSTOM"
        assert result["value"] == Decimal("55.500")

    def test_custom_factor_with_geography(self, selector_with_custom):
        """Custom factor takes priority even when geography is specified."""
        result = selector_with_custom.select_factor(
            "NATURAL_GAS", "CO2", geography="US",
        )
        assert result["tier"] == 3
        assert result["source"] == "CUSTOM"

    def test_explicit_tier3_without_custom(self, selector):
        """Explicit tier=3 without custom factor falls back to None."""
        # No custom factor registered, tier 3 requested but not available
        # Should return None from _try_tier3 and then None overall
        with pytest.raises(ValueError, match="No emission factor found"):
            selector.select_factor(
                "NATURAL_GAS", "CO2", tier=3,
            )


# ---------------------------------------------------------------------------
# TestAutoTierSelection
# ---------------------------------------------------------------------------

class TestAutoTierSelection:
    """Tests for auto_select_tier recommendation."""

    def test_tier1_no_data(self, selector):
        """Recommends Tier 1 when no geography or measurement data."""
        tier = selector.auto_select_tier("NATURAL_GAS")
        assert tier == 1

    def test_tier2_with_geography(self, selector):
        """Recommends Tier 2 when geography with country-specific source."""
        tier = selector.auto_select_tier("NATURAL_GAS", geography="US")
        assert tier == 2

    def test_tier3_with_measured_ef(self, selector):
        """Recommends Tier 3 when has_measured_ef=True."""
        tier = selector.auto_select_tier(
            "NATURAL_GAS",
            available_data={"has_measured_ef": True},
        )
        assert tier == 3

    def test_tier3_with_cems_data(self, selector):
        """Recommends Tier 3 when has_cems_data=True."""
        tier = selector.auto_select_tier(
            "NATURAL_GAS",
            available_data={"has_cems_data": True},
        )
        assert tier == 3

    def test_tier3_with_fuel_analysis(self, selector):
        """Recommends Tier 3 when has_fuel_analysis=True."""
        tier = selector.auto_select_tier(
            "NATURAL_GAS",
            available_data={"has_fuel_analysis": True},
        )
        assert tier == 3

    def test_tier3_with_custom_factor(self, selector_with_custom):
        """Recommends Tier 3 when custom factor is registered."""
        tier = selector_with_custom.auto_select_tier("NATURAL_GAS")
        assert tier == 3


# ---------------------------------------------------------------------------
# TestFallbackChain
# ---------------------------------------------------------------------------

class TestFallbackChain:
    """Tests for the Tier 3 -> Tier 2 -> Tier 1 fallback chain."""

    def test_fallback_to_tier1_no_geography(self, selector):
        """Without geography and no custom factor, falls back to Tier 1."""
        result = selector.select_factor("NATURAL_GAS", "CO2")
        assert result["tier"] == 1

    def test_fallback_from_tier2_to_tier1(self, selector):
        """If Tier 2 source doesn't have fuel, falls back to Tier 1."""
        # Japan geography -> no country-specific source -> IPCC Tier 1
        result = selector.select_factor("NATURAL_GAS", "CO2", geography="JP")
        assert result["tier"] == 1
        assert result["source"] == "IPCC"

    def test_custom_overrides_tier2(self, selector_with_custom):
        """Custom Tier 3 factor takes priority over Tier 2."""
        result = selector_with_custom.select_factor(
            "NATURAL_GAS", "CO2", geography="US",
        )
        assert result["tier"] == 3
        assert result["source"] == "CUSTOM"


# ---------------------------------------------------------------------------
# TestSelectFactorsForFuel
# ---------------------------------------------------------------------------

class TestSelectFactorsForFuel:
    """Tests for multi-gas factor selection."""

    def test_returns_three_gases(self, selector):
        """select_factors_for_fuel returns CO2, CH4, N2O."""
        factors = selector.select_factors_for_fuel("NATURAL_GAS")
        assert "CO2" in factors
        assert "CH4" in factors
        assert "N2O" in factors

    def test_all_values_positive(self, selector):
        """All three gas factors have positive values."""
        factors = selector.select_factors_for_fuel("NATURAL_GAS")
        for gas, result in factors.items():
            assert result["value"] > 0, f"{gas} factor should be positive"


# ---------------------------------------------------------------------------
# TestGeographyCoverage
# ---------------------------------------------------------------------------

class TestGeographyCoverage:
    """Tests for geography-to-source coverage mapping."""

    def test_epa_covers_us(self, selector):
        """EPA covers US geographies."""
        coverage = selector.get_geography_coverage("EPA")
        assert "US" in coverage

    def test_defra_covers_uk(self, selector):
        """DEFRA covers UK geographies."""
        coverage = selector.get_geography_coverage("DEFRA")
        assert "UK" in coverage

    def test_eu_ets_covers_eu(self, selector):
        """EU ETS covers EU member state geographies."""
        coverage = selector.get_geography_coverage("EU_ETS")
        assert "DE" in coverage
        assert "FR" in coverage

    def test_ipcc_covers_global(self, selector):
        """IPCC covers GLOBAL geography."""
        coverage = selector.get_geography_coverage("IPCC")
        assert "GLOBAL" in coverage


# ---------------------------------------------------------------------------
# TestRecommendedSource
# ---------------------------------------------------------------------------

class TestRecommendedSource:
    """Tests for recommended source by geography."""

    def test_us_recommends_epa(self, selector):
        """US recommends EPA."""
        assert selector.get_recommended_source("US") == "EPA"

    def test_uk_recommends_defra(self, selector):
        """UK recommends DEFRA."""
        assert selector.get_recommended_source("UK") == "DEFRA"

    def test_de_recommends_eu_ets(self, selector):
        """Germany recommends EU ETS."""
        assert selector.get_recommended_source("DE") == "EU_ETS"

    def test_global_recommends_ipcc(self, selector):
        """Unknown geography falls back to IPCC."""
        assert selector.get_recommended_source("JP") == "IPCC"

    def test_case_insensitive(self, selector):
        """Geography lookup is case-insensitive."""
        assert selector.get_recommended_source("us") == "EPA"


# ---------------------------------------------------------------------------
# TestValidateCustomFactor
# ---------------------------------------------------------------------------

class TestValidateCustomFactor:
    """Tests for custom factor validation against IPCC ranges."""

    def test_within_range_passes(self, selector):
        """Custom factor within 50% of IPCC passes validation."""
        result = selector.validate_custom_factor(
            "NATURAL_GAS", "CO2", Decimal("55.0"), "Lab test",
        )
        assert result["is_valid"] is True

    def test_outside_range_fails(self, selector):
        """Custom factor > 50% above IPCC fails validation."""
        result = selector.validate_custom_factor(
            "NATURAL_GAS", "CO2", Decimal("200.0"), "Bad data",
        )
        assert result["is_valid"] is False

    def test_exactly_at_boundary(self, selector):
        """Custom factor exactly at 50% boundary passes."""
        ipcc_val = IPCC_DEFAULT_FACTORS["NATURAL_GAS"]["CO2"]["value"]
        boundary = ipcc_val * Decimal("1.50")
        result = selector.validate_custom_factor(
            "NATURAL_GAS", "CO2", boundary, "Boundary test",
        )
        assert result["is_valid"] is True

    def test_unknown_fuel_accepted(self, selector):
        """Unknown fuel type in IPCC is accepted without range check."""
        result = selector.validate_custom_factor(
            "CUSTOM_FUEL", "CO2", Decimal("100.0"), "Custom fuel",
        )
        assert result["is_valid"] is True

    def test_validation_result_fields(self, selector):
        """Validation result contains all expected fields."""
        result = selector.validate_custom_factor(
            "NATURAL_GAS", "CO2", Decimal("55.0"), "Test",
        )
        assert "is_valid" in result
        assert "value" in result
        assert "ipcc_reference" in result
        assert "deviation_pct" in result
        assert "tolerance_pct" in result
        assert "message" in result


# ---------------------------------------------------------------------------
# TestCompareFactorSources
# ---------------------------------------------------------------------------

class TestCompareFactorSources:
    """Tests for source comparison."""

    def test_compare_natural_gas_co2(self, selector):
        """Comparing natural gas CO2 returns multiple sources."""
        comparisons = selector.compare_sources("NATURAL_GAS", "CO2")
        sources = {c["source"] for c in comparisons}
        assert "IPCC" in sources
        assert "EPA" in sources

    def test_comparisons_sorted_by_source(self, selector):
        """Comparisons are sorted alphabetically by source."""
        comparisons = selector.compare_sources("NATURAL_GAS", "CO2")
        names = [c["source"] for c in comparisons]
        assert names == sorted(names)

    def test_includes_custom_if_registered(self, selector_with_custom):
        """Custom factor appears in comparison when registered."""
        comparisons = selector_with_custom.compare_sources("NATURAL_GAS", "CO2")
        sources = {c["source"] for c in comparisons}
        assert "CUSTOM" in sources


# ---------------------------------------------------------------------------
# TestSelectionTrace
# ---------------------------------------------------------------------------

class TestSelectionTrace:
    """Tests for selection trace audit log."""

    def test_selection_records_to_log(self, selector):
        """Each select_factor call records to the selection log."""
        selector.select_factor("NATURAL_GAS", "CO2")
        trace = selector.get_selection_trace()
        assert len(trace) == 1

    def test_trace_contains_fuel_and_gas(self, selector):
        """Trace entry contains fuel_type and gas."""
        selector.select_factor("NATURAL_GAS", "CO2")
        entry = selector.get_selection_trace()[0]
        assert entry["fuel_type"] == "NATURAL_GAS"
        assert entry["gas"] == "CO2"

    def test_trace_contains_provenance_hash(self, selector):
        """Trace entry contains provenance_hash."""
        selector.select_factor("NATURAL_GAS", "CO2")
        entry = selector.get_selection_trace()[0]
        assert "provenance_hash" in entry
        assert len(entry["provenance_hash"]) == 64

    def test_result_contains_selection_trace(self, selector):
        """Result dict contains selection_trace list."""
        result = selector.select_factor("NATURAL_GAS", "CO2")
        assert "selection_trace" in result
        assert isinstance(result["selection_trace"], list)
        assert len(result["selection_trace"]) > 0


# ---------------------------------------------------------------------------
# TestSelectionStatistics
# ---------------------------------------------------------------------------

class TestSelectionStatistics:
    """Tests for selection statistics aggregation."""

    def test_empty_statistics(self, selector):
        """Empty engine returns zero total_selections."""
        stats = selector.get_selection_statistics()
        assert stats["total_selections"] == 0

    def test_statistics_after_selections(self, selector):
        """Statistics reflect completed selections."""
        selector.select_factor("NATURAL_GAS", "CO2")
        selector.select_factor("DIESEL", "CO2", geography="US")
        stats = selector.get_selection_statistics()
        assert stats["total_selections"] == 2
        assert 1 in stats["by_tier"] or 2 in stats["by_tier"]


# ---------------------------------------------------------------------------
# TestRecordSelection
# ---------------------------------------------------------------------------

class TestRecordSelection:
    """Tests for manual selection recording."""

    def test_manual_record_added_to_log(self, selector):
        """Manually recorded selection appears in the trace."""
        selector.record_selection(
            fuel_type="NATURAL_GAS", gas="CO2",
            tier=2, source="EPA", value=Decimal("53.06"),
        )
        trace = selector.get_selection_trace()
        assert len(trace) == 1
        assert trace[0]["origin"] == "manual_record"

    def test_manual_record_fields(self, selector):
        """Manual record contains correct fields."""
        selector.record_selection(
            fuel_type="DIESEL", gas="CH4",
            tier=1, source="IPCC", value=Decimal("0.003"),
        )
        entry = selector.get_selection_trace()[0]
        assert entry["fuel_type"] == "DIESEL"
        assert entry["gas"] == "CH4"
        assert entry["tier"] == 1
        assert entry["source"] == "IPCC"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for the selector engine."""

    def test_unknown_fuel_no_fallback_raises(self, selector):
        """Completely unknown fuel type with no fallback raises ValueError."""
        with pytest.raises(ValueError, match="No emission factor found"):
            selector.select_factor("PLUTONIUM", "CO2")

    def test_case_insensitive_fuel_type(self, selector):
        """Fuel type lookup is case-insensitive."""
        result = selector.select_factor("natural_gas", "CO2")
        assert result["value"] > 0

    def test_register_and_select_custom(self, selector):
        """Register custom factor then select it."""
        selector.register_custom_factor(
            "DIESEL", "CO2", Decimal("74.5"), "kg/GJ",
        )
        result = selector.select_factor("DIESEL", "CO2")
        assert result["tier"] == 3
        assert result["value"] == Decimal("74.5")
