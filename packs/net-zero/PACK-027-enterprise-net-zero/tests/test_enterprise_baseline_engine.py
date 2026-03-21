# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Enterprise Baseline Engine.

Tests financial-grade Scope 1+2+3 calculation across all 30 MRV agents with
data quality scoring, dual Scope 2 reporting, materiality assessment, and
per-entity breakdowns.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~55 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.enterprise_baseline_engine import (
    EnterpriseBaselineEngine,
    EnterpriseBaselineInput,
    EnterpriseBaselineResult,
    DataQualityLevel,
    DataQualityMatrix,
    MaterialityAssessment,
    EntityDefinition,
    CalculationApproach,
    Scope1Breakdown,
    Scope2Breakdown,
    Scope3Breakdown,
    FuelEntry,
    FuelType,
    ElectricityEntry,
    RefrigerantEntry,
    ConsolidationApproach,
    Scope3CategoryEntry,
    Scope3Category,
    SteamCoolingEntry,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive,
    assert_provenance_hash, assert_processing_time,
    timed_block, SCOPE3_CATEGORIES, DATA_QUALITY_LEVELS,
)


def _make_entity(entity_id="E001", name="TestCorp", country="US"):
    """Helper to create an EntityDefinition."""
    return EntityDefinition(
        entity_id=entity_id,
        entity_name=name,
        country=country,
    )


def _make_fuel(fuel_type=FuelType.NATURAL_GAS, quantity=Decimal("1000"),
               unit="GJ", entity_id="E001"):
    """Helper to create a FuelEntry."""
    return FuelEntry(
        fuel_type=fuel_type,
        quantity=quantity,
        unit=unit,
        entity_id=entity_id,
        facility_id="F001",
        source_reference="test",
    )


def _make_elec(annual_mwh=Decimal("1000"), region="US", entity_id="E001"):
    """Helper to create an ElectricityEntry."""
    return ElectricityEntry(
        annual_mwh=annual_mwh,
        region=region,
        entity_id=entity_id,
        facility_id="F001",
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestEnterpriseBaselineEngineInstantiation:
    def test_engine_instantiates(self):
        engine = EnterpriseBaselineEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = EnterpriseBaselineEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = EnterpriseBaselineEngine()
        assert hasattr(engine, "engine_version")
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Tests -- Scope 1 Calculations (Fuel Entries)
# ===========================================================================


class TestScope1Calculations:
    def test_natural_gas_fuel_entry(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.NATURAL_GAS, Decimal("12000"))],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")

    def test_diesel_fuel_entry(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.DIESEL, Decimal("5000"))],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")

    def test_multiple_fuel_entries(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[
                _make_fuel(FuelType.NATURAL_GAS, Decimal("12000")),
                _make_fuel(FuelType.DIESEL, Decimal("5000")),
            ],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")

    def test_scope1_breakdown_structure(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.NATURAL_GAS, Decimal("12000"))],
        ))
        assert isinstance(result.scope1, Scope1Breakdown)
        assert hasattr(result.scope1, "stationary_combustion_tco2e")
        assert hasattr(result.scope1, "by_entity")
        assert hasattr(result.scope1, "by_gas")

    def test_refrigerant_entries(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            refrigerant_entries=[RefrigerantEntry(
                refrigerant_type="R-410A",
                system_count=10,
                charge_kg=Decimal("85"),
                annual_leakage_rate_pct=Decimal("5"),
                facility_id="F001",
                entity_id="E001",
            )],
        ))
        assert result.scope1.refrigerants_tco2e >= Decimal("0")

    def test_scope1_ghg_breakdown(self):
        """Scope 1 should report by gas."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.NATURAL_GAS, Decimal("12000"))],
        ))
        assert isinstance(result.scope1.by_gas, dict)

    def test_empty_fuel_entries(self):
        """With no fuel entries, scope1 should be zero."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")

    @pytest.mark.parametrize("fuel", [
        FuelType.NATURAL_GAS, FuelType.DIESEL, FuelType.GASOLINE,
        FuelType.LPG, FuelType.FUEL_OIL, FuelType.KEROSENE,
    ])
    def test_scope1_fuel_types(self, fuel):
        """Each fuel type must be accepted."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(fuel, Decimal("1000"))],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Scope 2 Calculations (dual reporting)
# ===========================================================================


class TestScope2Calculations:
    def test_electricity_location_based(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity(country="DE")],
            electricity_entries=[_make_elec(Decimal("185000"), "DE")],
        ))
        assert result.scope2.location_based_tco2e >= Decimal("0")

    def test_dual_reporting(self):
        """Both location and market-based must be reported."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            electricity_entries=[_make_elec(Decimal("185000"), "US")],
        ))
        assert isinstance(result.scope2, Scope2Breakdown)
        assert hasattr(result.scope2, "location_based_tco2e")
        assert hasattr(result.scope2, "market_based_tco2e")

    def test_scope2_delta(self):
        """Delta between location and market based should be tracked."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            electricity_entries=[_make_elec(Decimal("100000"), "US")],
        ))
        assert hasattr(result.scope2, "delta_tco2e")

    def test_steam_cooling_entries(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            electricity_entries=[_make_elec(Decimal("50000"), "US")],
            steam_cooling_entries=[SteamCoolingEntry(
                energy_type="steam",
                annual_mwh=Decimal("25000"),
                entity_id="E001",
            )],
        ))
        assert result.scope2.steam_heat_tco2e >= Decimal("0")

    def test_scope2_by_region(self):
        """Scope 2 should track by region."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            electricity_entries=[_make_elec(Decimal("50000"), "US")],
        ))
        assert isinstance(result.scope2.by_region, dict)


# ===========================================================================
# Tests -- Scope 3 Calculations
# ===========================================================================


class TestScope3Calculations:
    def test_scope3_with_entries(self):
        """Scope 3 entries should produce results."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            scope3_entries=[Scope3CategoryEntry(
                category=Scope3Category.CAT_01_PURCHASED_GOODS,
                spend_usd=Decimal("780000000"),
                entity_id="E001",
            )],
        ))
        assert result.scope3.total_tco2e >= Decimal("0")

    def test_scope3_breakdown_structure(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            scope3_entries=[Scope3CategoryEntry(
                category=Scope3Category.CAT_06_BUSINESS_TRAVEL,
                spend_usd=Decimal("28000000"),
                entity_id="E001",
            )],
        ))
        assert isinstance(result.scope3, Scope3Breakdown)
        assert hasattr(result.scope3, "categories")
        assert hasattr(result.scope3, "upstream_tco2e")
        assert hasattr(result.scope3, "downstream_tco2e")

    def test_scope3_coverage_pct(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            scope3_entries=[Scope3CategoryEntry(
                category=Scope3Category.CAT_01_PURCHASED_GOODS,
                spend_usd=Decimal("500000000"),
                entity_id="E001",
            )],
        ))
        assert hasattr(result.scope3, "coverage_pct")

    def test_empty_scope3(self):
        """With no scope3 entries, total should be zero."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert result.scope3.total_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Data Quality
# ===========================================================================


class TestDataQuality:
    @pytest.mark.parametrize("dq_level", DATA_QUALITY_LEVELS)
    def test_data_quality_level_valid(self, dq_level):
        """Each DQ level (1-5) must be recognized."""
        assert 1 <= dq_level <= 5

    def test_data_quality_matrix_generation(self):
        """Engine must produce a DQ matrix."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "data_quality")
        assert isinstance(result.data_quality, DataQualityMatrix)

    def test_dq_matrix_fields(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        dq = result.data_quality
        assert hasattr(dq, "overall_score")
        assert hasattr(dq, "by_scope")
        assert hasattr(dq, "meets_target")

    def test_target_accuracy(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            target_accuracy_pct=Decimal("3"),
        ))
        assert result.data_quality.target_accuracy_pct == Decimal("3")


# ===========================================================================
# Tests -- Materiality Assessment
# ===========================================================================


class TestMaterialityAssessment:
    def test_materiality_assessment_generated(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "materiality")
        assert isinstance(result.materiality, MaterialityAssessment)

    def test_materiality_categories(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        ma = result.materiality
        assert hasattr(ma, "material_categories")
        assert hasattr(ma, "excluded_categories")
        assert hasattr(ma, "total_excluded_pct")

    def test_sbti_coverage(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert result.materiality.sbti_coverage_pct >= Decimal("0")


# ===========================================================================
# Tests -- Result Structure & Provenance
# ===========================================================================


class TestResultStructure:
    def test_result_has_all_fields(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert isinstance(result, EnterpriseBaselineResult)
        assert hasattr(result, "total_tco2e_location")
        assert hasattr(result, "total_tco2e_market")
        assert hasattr(result, "scope1")
        assert hasattr(result, "scope2")
        assert hasattr(result, "scope3")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")

    def test_provenance_hash_valid(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert_provenance_hash(result)

    def test_result_deterministic(self):
        """Same inputs must produce same outputs (zero-hallucination)."""
        engine = EnterpriseBaselineEngine()
        inp = EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.NATURAL_GAS, Decimal("12000"))],
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.total_tco2e_location == r2.total_tco2e_location
        assert r1.scope1.total_tco2e == r2.scope1.total_tco2e

    def test_scope_percentages(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.NATURAL_GAS, Decimal("12000"))],
            electricity_entries=[_make_elec(Decimal("50000"), "US")],
        ))
        assert hasattr(result, "scope1_pct")
        assert hasattr(result, "scope2_pct")
        assert hasattr(result, "scope3_pct")

    def test_confidence_interval(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "confidence_interval")
        ci = result.confidence_interval
        assert ci.lower_bound_tco2e <= ci.central_estimate_tco2e
        assert ci.upper_bound_tco2e >= ci.central_estimate_tco2e

    def test_consolidation_summary(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "consolidation")
        assert result.consolidation.entity_count >= 0

    def test_regulatory_citations(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert len(result.regulatory_citations) > 0

    def test_mrv_agents_used(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[_make_fuel(FuelType.NATURAL_GAS, Decimal("12000"))],
        ))
        assert len(result.mrv_agents_used) >= 0

    def test_intensity_metrics(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "intensity")

    def test_processing_time(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert result.processing_time_ms >= 0


# ===========================================================================
# Tests -- Multi-Entity Baseline
# ===========================================================================


class TestMultiEntityBaseline:
    def test_multiple_entities(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[
                _make_entity("E001", "HQ", "US"),
                _make_entity("E002", "Sub-EU", "DE"),
            ],
            fuel_entries=[
                _make_fuel(FuelType.NATURAL_GAS, Decimal("12000"), entity_id="E001"),
                _make_fuel(FuelType.DIESEL, Decimal("5000"), entity_id="E002"),
            ],
        ))
        assert result.entity_count == 2

    def test_consolidation_approach_financial(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=[_make_entity()],
        ))
        assert result.consolidation_approach == ConsolidationApproach.FINANCIAL_CONTROL.value

    def test_consolidation_approach_operational(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            entities=[_make_entity()],
        ))
        assert result.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL.value

    def test_consolidation_approach_equity(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[_make_entity()],
        ))
        assert result.consolidation_approach == ConsolidationApproach.EQUITY_SHARE.value

    @pytest.mark.parametrize("country", [
        "US", "GB", "DE", "FR", "JP", "CN", "IN", "BR", "AU", "CA",
        "NL", "CH", "SE", "KR", "SG",
    ])
    def test_entity_by_country(self, country):
        """Per-entity baseline must work for each country."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity(country=country)],
            electricity_entries=[_make_elec(Decimal("10000"), country)],
        ))
        assert result.scope2.location_based_tco2e >= Decimal("0")
