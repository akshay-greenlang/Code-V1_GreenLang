"""
Unit tests for DenominatorRegistryEngine (PACK-046 Engine 1).

Tests all public methods with 50+ tests covering:
  - Initialisation and built-in denominator loading
  - calculate() full pipeline
  - register_denominator() custom denominators
  - get_denominator() lookup
  - list_denominators() filtering by category/sector/framework
  - convert_value() unit conversions with exact Decimal factors
  - validate_value() positive/zero/unit/YoY checks
  - recommend_denominators() scoring formula verification
  - Utility methods (get_version, get_categories, etc.)
  - Edge cases and error handling

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.denominator_registry_engine import (
    BUILT_IN_DENOMINATORS,
    DenominatorCategory,
    DenominatorDefinition,
    DenominatorRegistryEngine,
    DenominatorUnit,
    DenominatorValue,
    RegistryInput,
    RegistryResult,
    UNIT_CONVERSION_FACTORS,
    ValidationFinding,
    ValidationSeverity,
    get_built_in_denominators,
    recommend_denominators,
)


class TestDenominatorRegistryEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = DenominatorRegistryEngine()
        assert engine is not None

    def test_init_loads_built_in_denominators(self):
        engine = DenominatorRegistryEngine()
        assert engine.get_registry_size() == len(BUILT_IN_DENOMINATORS)

    def test_init_version(self):
        engine = DenominatorRegistryEngine()
        assert engine.get_version() == "1.0.0"

    def test_built_in_count_is_27(self):
        assert len(BUILT_IN_DENOMINATORS) == 27


class TestGetDenominator:
    """Tests for get_denominator() lookup."""

    def test_get_existing_denominator(self, denominator_engine):
        d = denominator_engine.get_denominator("revenue_usd")
        assert d is not None
        assert d.denominator_id == "revenue_usd"
        assert d.category == DenominatorCategory.ECONOMIC

    def test_get_nonexistent_denominator_returns_none(self, denominator_engine):
        d = denominator_engine.get_denominator("nonexistent_id")
        assert d is None

    def test_get_physical_denominator(self, denominator_engine):
        d = denominator_engine.get_denominator("production_tonnes")
        assert d is not None
        assert d.category == DenominatorCategory.PHYSICAL
        assert d.unit == DenominatorUnit.TONNE.value

    def test_get_headcount_denominator(self, denominator_engine):
        d = denominator_engine.get_denominator("fte_employees")
        assert d is not None
        assert d.category == DenominatorCategory.HEADCOUNT
        assert d.is_universal is True

    def test_get_area_denominator(self, denominator_engine):
        d = denominator_engine.get_denominator("floor_area_m2")
        assert d is not None
        assert d.category == DenominatorCategory.AREA

    def test_get_capacity_denominator(self, denominator_engine):
        d = denominator_engine.get_denominator("installed_capacity_mw")
        assert d is not None
        assert d.category == DenominatorCategory.CAPACITY

    def test_get_activity_denominator(self, denominator_engine):
        d = denominator_engine.get_denominator("passenger_km")
        assert d is not None
        assert d.category == DenominatorCategory.ACTIVITY


class TestListDenominators:
    """Tests for list_denominators() filtering."""

    def test_list_all_returns_27(self, denominator_engine):
        result = denominator_engine.list_denominators()
        assert len(result) == 27

    def test_list_by_economic_category(self, denominator_engine):
        result = denominator_engine.list_denominators(category=DenominatorCategory.ECONOMIC)
        assert len(result) == 6
        assert all(d.category == DenominatorCategory.ECONOMIC for d in result)

    def test_list_by_physical_category(self, denominator_engine):
        result = denominator_engine.list_denominators(category=DenominatorCategory.PHYSICAL)
        assert len(result) == 8

    def test_list_by_sector_manufacturing(self, denominator_engine):
        result = denominator_engine.list_denominators(sector="manufacturing")
        # Universal + manufacturing-specific
        assert len(result) >= 8

    def test_list_by_framework_sbti_sda(self, denominator_engine):
        result = denominator_engine.list_denominators(framework="SBTi_SDA")
        assert len(result) >= 5
        assert all("SBTi_SDA" in d.frameworks for d in result)

    def test_list_by_sector_and_framework(self, denominator_engine):
        result = denominator_engine.list_denominators(
            sector="cement", framework="SBTi_SDA"
        )
        assert any(d.denominator_id == "clinker_tonnes" for d in result)

    def test_list_by_nonexistent_sector_returns_universals(self, denominator_engine):
        result = denominator_engine.list_denominators(sector="space_mining")
        # Only universals match
        assert all(d.is_universal or "all" in d.sectors for d in result)


class TestRegisterDenominator:
    """Tests for register_denominator() custom definitions."""

    def test_register_custom_denominator(self, denominator_engine):
        custom = DenominatorDefinition(
            denominator_id="custom_widgets",
            name="Widgets Produced",
            unit="widget",
            category=DenominatorCategory.ACTIVITY,
        )
        denominator_engine.register_denominator(custom)
        assert denominator_engine.get_denominator("custom_widgets") is not None
        assert denominator_engine.get_registry_size() == 28

    def test_register_builtin_raises_value_error(self, denominator_engine):
        custom = DenominatorDefinition(
            denominator_id="revenue_usd",
            name="Override Revenue",
            unit="USD_million",
            category=DenominatorCategory.ECONOMIC,
        )
        with pytest.raises(ValueError, match="Cannot override built-in"):
            denominator_engine.register_denominator(custom)


class TestConvertValue:
    """Tests for convert_value() unit conversions."""

    def test_same_unit_returns_identity(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("100"), "tonne", "tonne")
        assert result == Decimal("100")

    def test_m2_to_sqft(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("100"), "m2", "sq_ft")
        assert result is not None
        assert result == Decimal("1076.390000")

    def test_sqft_to_m2(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("1000"), "sq_ft", "m2")
        assert result is not None
        assert result == Decimal("92.903000")

    def test_tonne_to_kg(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("5"), "tonne", "kg")
        assert result is not None
        assert result == Decimal("5000.000000")

    def test_mwh_to_gj(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("10"), "MWh", "GJ")
        assert result is not None
        assert result == Decimal("36.000000")

    def test_usd_to_eur(self, denominator_engine):
        result = denominator_engine.convert_value(
            Decimal("100"), "USD_million", "EUR_million"
        )
        assert result is not None
        assert result == Decimal("92.000000")

    def test_incompatible_units_returns_none(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("100"), "tonne", "MWh")
        assert result is None

    def test_gwh_to_mwh(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("1"), "GWh", "MWh")
        assert result is not None
        assert result == Decimal("1000.000000")

    def test_barrel_to_litre(self, denominator_engine):
        result = denominator_engine.convert_value(Decimal("1"), "barrel", "litre")
        assert result is not None
        assert result == Decimal("158.987000")


class TestValidateValue:
    """Tests for validate_value() validation rules."""

    def test_valid_value_no_findings(self, denominator_engine):
        val = DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("500"),
            unit="USD_million",
        )
        findings = denominator_engine.validate_value(val)
        assert len(findings) == 0

    def test_zero_value_raises_error(self, denominator_engine):
        val = DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("0"),
            unit="USD_million",
        )
        findings = denominator_engine.validate_value(val)
        assert any(f.code == "VALUE_NON_POSITIVE" for f in findings)

    def test_negative_value_raises_error(self, denominator_engine):
        val = DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("-100"),
            unit="USD_million",
        )
        findings = denominator_engine.validate_value(val)
        assert any(f.code == "VALUE_NON_POSITIVE" for f in findings)

    def test_unit_mismatch_no_conversion_raises_error(self, denominator_engine):
        val = DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("500"),
            unit="tonne",
        )
        findings = denominator_engine.validate_value(val)
        assert any(f.code == "UNIT_MISMATCH" for f in findings)

    def test_unit_mismatch_convertible_raises_warning(self, denominator_engine):
        val = DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("500"),
            unit="EUR_million",
        )
        findings = denominator_engine.validate_value(val)
        assert any(f.code == "UNIT_MISMATCH_CONVERTIBLE" for f in findings)
        assert all(f.severity != ValidationSeverity.ERROR for f in findings)

    def test_unknown_denominator_raises_error(self, denominator_engine):
        val = DenominatorValue(
            denominator_id="nonexistent_denom",
            period="2024",
            value=Decimal("500"),
            unit="USD_million",
        )
        findings = denominator_engine.validate_value(val)
        assert any(f.code == "DENOM_NOT_FOUND" for f in findings)

    def test_yoy_change_within_limit_no_warning(self, denominator_engine):
        prev = DenominatorValue(
            denominator_id="revenue_usd", period="2023",
            value=Decimal("500"), unit="USD_million",
        )
        curr = DenominatorValue(
            denominator_id="revenue_usd", period="2024",
            value=Decimal("520"), unit="USD_million",
        )
        findings = denominator_engine.validate_value(curr, prev)
        assert not any(f.code == "YOY_CHANGE_EXCEEDED" for f in findings)

    def test_yoy_change_exceeds_limit_raises_warning(self, denominator_engine):
        prev = DenominatorValue(
            denominator_id="revenue_usd", period="2023",
            value=Decimal("500"), unit="USD_million",
        )
        curr = DenominatorValue(
            denominator_id="revenue_usd", period="2024",
            value=Decimal("900"), unit="USD_million",
        )
        findings = denominator_engine.validate_value(curr, prev)
        assert any(f.code == "YOY_CHANGE_EXCEEDED" for f in findings)


class TestRecommendDenominators:
    """Tests for recommend_denominators() scoring formula."""

    def test_manufacturing_sector_recommendations(self, denominator_engine):
        recs = denominator_engine.recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA", "ESRS_E1_6"],
            available_ids=["production_tonnes", "revenue_usd"],
        )
        assert len(recs) > 0
        assert recs[0].rank == 1
        assert recs[0].relevance_score > Decimal("0")

    def test_recommendations_are_ranked(self, denominator_engine):
        recs = denominator_engine.recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA"],
        )
        for i in range(len(recs) - 1):
            assert recs[i].relevance_score >= recs[i + 1].relevance_score

    def test_top_n_limits_results(self, denominator_engine):
        recs = denominator_engine.recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA"],
            top_n=3,
        )
        assert len(recs) <= 3

    def test_data_availability_boosts_score(self, denominator_engine):
        recs_without = denominator_engine.recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA"],
            available_ids=[],
        )
        recs_with = denominator_engine.recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA"],
            available_ids=["production_tonnes"],
        )
        prod_without = next(
            (r for r in recs_without if r.denominator_id == "production_tonnes"), None
        )
        prod_with = next(
            (r for r in recs_with if r.denominator_id == "production_tonnes"), None
        )
        assert prod_with is not None
        assert prod_without is not None
        assert prod_with.relevance_score > prod_without.relevance_score

    def test_sector_score_is_binary(self, denominator_engine):
        recs = denominator_engine.recommend_denominators(
            sector="cement",
            frameworks=["SBTi_SDA"],
        )
        clinker = next(
            (r for r in recs if r.denominator_id == "clinker_tonnes"), None
        )
        assert clinker is not None
        assert clinker.sector_score == Decimal("1")

    def test_framework_score_partial(self, denominator_engine):
        recs = denominator_engine.recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA", "ESRS_E1_6", "CDP_C6_10", "GRI_305_4"],
        )
        for r in recs:
            assert Decimal("0") <= r.framework_score <= Decimal("1")


class TestCalculate:
    """Tests for the full calculate() pipeline."""

    def test_calculate_returns_registry_result(
        self, denominator_engine, sample_registry_input
    ):
        result = denominator_engine.calculate(sample_registry_input)
        assert isinstance(result, RegistryResult)

    def test_calculate_provenance_hash(
        self, denominator_engine, sample_registry_input
    ):
        result = denominator_engine.calculate(sample_registry_input)
        assert len(result.provenance_hash) == 64

    def test_calculate_has_recommendations(
        self, denominator_engine, sample_registry_input
    ):
        result = denominator_engine.calculate(sample_registry_input)
        assert len(result.recommendations) > 0

    def test_calculate_has_summary(
        self, denominator_engine, sample_registry_input
    ):
        result = denominator_engine.calculate(sample_registry_input)
        assert "total_denominators_in_registry" in result.summary
        assert result.summary["sector"] == "manufacturing"

    def test_calculate_processes_custom_denominators(self, denominator_engine):
        custom = DenominatorDefinition(
            denominator_id="my_custom_metric",
            name="Custom Metric",
            unit="custom_unit",
            category=DenominatorCategory.ACTIVITY,
        )
        inp = RegistryInput(
            sector="manufacturing",
            target_frameworks=["GRI_305_4"],
            custom_denominators=[custom],
        )
        result = denominator_engine.calculate(inp)
        assert result.available_denominators == 28

    def test_calculate_warnings_for_validation_errors(self, denominator_engine):
        bad_val = DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("-10"),
            unit="USD_million",
        )
        inp = RegistryInput(
            sector="manufacturing",
            target_frameworks=["GRI_305_4"],
            denominator_values=[bad_val],
        )
        result = denominator_engine.calculate(inp)
        assert len(result.warnings) > 0
        assert any("validation error" in w.lower() for w in result.warnings)

    def test_calculate_processing_time(
        self, denominator_engine, sample_registry_input
    ):
        result = denominator_engine.calculate(sample_registry_input)
        assert result.processing_time_ms > 0


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_categories(self, denominator_engine):
        cats = denominator_engine.get_categories()
        assert "economic" in cats
        assert "physical" in cats
        assert len(cats) == 6

    def test_get_frameworks(self, denominator_engine):
        fws = denominator_engine.get_frameworks()
        assert "GRI_305_4" in fws
        assert "SBTi_SDA" in fws

    def test_get_sectors(self, denominator_engine):
        sectors = denominator_engine.get_sectors()
        assert "manufacturing" in sectors
        assert "cement" in sectors
        assert "all" not in sectors

    def test_get_conversion_pairs(self, denominator_engine):
        pairs = denominator_engine.get_conversion_pairs()
        assert len(pairs) == len(UNIT_CONVERSION_FACTORS)


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_built_in_denominators(self):
        denoms = get_built_in_denominators()
        assert len(denoms) == 27
        assert all(isinstance(d, DenominatorDefinition) for d in denoms)

    def test_recommend_denominators_convenience(self):
        recs = recommend_denominators(
            sector="manufacturing",
            frameworks=["SBTi_SDA"],
        )
        assert len(recs) > 0
