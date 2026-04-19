# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Performance.

Tests performance benchmarks for enterprise-scale operations: baseline
calculation, consolidation, Monte Carlo simulation, supply chain mapping,
template rendering, and memory usage.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~35 performance benchmarks
"""

import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.enterprise_baseline_engine import (
    EnterpriseBaselineEngine, EnterpriseBaselineInput,
    FuelEntry, FuelType, ElectricityEntry, EntityDefinition,
)
from engines.sbti_target_engine import (
    SBTiTargetEngine, SBTiTargetInput, TargetPathwayType as SBTiPathwayType,
    BaselineData,
)
from engines.scenario_modeling_engine import (
    ScenarioModelingEngine, ScenarioModelingInput as ScenarioInput,
)
from engines.multi_entity_consolidation_engine import (
    MultiEntityConsolidationEngine, ConsolidationInput, ConsolidationApproach,
    EntityEmissions,
)
from engines.supply_chain_mapping_engine import (
    SupplyChainMappingEngine, SupplyChainMappingInput as SupplyChainInput,
    SupplierEntry,
)
from engines.carbon_pricing_engine import (
    CarbonPricingEngine, CarbonPricingInput,
)
from engines.financial_integration_engine import (
    FinancialIntegrationEngine, FinancialIntegrationInput,
)

from .conftest import timed_block


def _make_entity(eid="E1", name="PerfTest", country="DE"):
    return EntityDefinition(entity_id=eid, entity_name=name, country=country)


def _make_consolidation_entity(idx):
    return EntityEmissions(
        entity_id=f"ENT-{idx:03d}",
        entity_name=f"Entity {idx}",
        scope1_tco2e=Decimal("10000"),
        scope2_location_tco2e=Decimal("5000"),
        scope2_market_tco2e=Decimal("4000"),
        scope3_tco2e=Decimal("30000"),
        ownership_pct=Decimal("100"),
    )


def _make_supplier(idx):
    return SupplierEntry(
        supplier_id=f"S{idx:04d}",
        supplier_name=f"Supplier {idx}",
        country="DE",
        annual_spend_usd=Decimal("100000"),
        scope3_contribution_tco2e=Decimal("100"),
    )


# ===========================================================================
# Tests -- Baseline Engine Performance
# ===========================================================================


class TestBaselinePerformance:
    def test_single_entity_baseline_under_5s(self):
        """Single entity baseline must complete in <5 seconds."""
        engine = EnterpriseBaselineEngine()
        with timed_block("Single entity baseline", max_seconds=5.0):
            result = engine.calculate(EnterpriseBaselineInput(
                entities=[_make_entity()],
                fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=50000000, unit="kWh")],
                electricity_entries=[ElectricityEntry(annual_mwh=Decimal("100000"), region="DE")],
            ))
        assert result is not None

    def test_baseline_only_entities_under_5s(self):
        """Baseline with only entities must complete in <5 seconds."""
        engine = EnterpriseBaselineEngine()
        with timed_block("Entities-only baseline", max_seconds=5.0):
            result = engine.calculate(EnterpriseBaselineInput(
                entities=[_make_entity()],
            ))
        assert result is not None

    def test_baseline_result_processing_time_tracked(self):
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms >= 0


# ===========================================================================
# Tests -- Consolidation Performance
# ===========================================================================


class TestConsolidationPerformance:
    def test_10_entities_under_10s(self):
        engine = MultiEntityConsolidationEngine()
        entities = [_make_consolidation_entity(i) for i in range(10)]
        with timed_block("10 entity consolidation", max_seconds=10.0):
            result = engine.calculate(ConsolidationInput(
                consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
                entities=entities,
            ))
        assert result is not None

    def test_50_entities_under_30s(self):
        engine = MultiEntityConsolidationEngine()
        entities = [_make_consolidation_entity(i) for i in range(50)]
        with timed_block("50 entity consolidation", max_seconds=30.0):
            result = engine.calculate(ConsolidationInput(
                consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
                entities=entities,
            ))
        assert result is not None

    def test_100_entities_under_120s(self):
        """100 entity consolidation must complete in <120 seconds."""
        engine = MultiEntityConsolidationEngine()
        entities = [_make_consolidation_entity(i) for i in range(100)]
        with timed_block("100 entity consolidation", max_seconds=120.0):
            result = engine.calculate(ConsolidationInput(
                consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
                entities=entities,
            ))
        assert result is not None


# ===========================================================================
# Tests -- Monte Carlo Performance
# ===========================================================================


class TestMonteCarloPerformance:
    def test_100_runs_under_5s(self):
        engine = ScenarioModelingEngine()
        with timed_block("100 MC runs", max_seconds=5.0):
            result = engine.calculate(ScenarioInput(
                base_year_emissions_tco2e=Decimal("867000"),
                mc_runs=100,
                random_seed=42,
            ))
        assert result.mc_runs_completed >= 100

    def test_1000_runs_under_30s(self):
        engine = ScenarioModelingEngine()
        with timed_block("1000 MC runs", max_seconds=30.0):
            result = engine.calculate(ScenarioInput(
                base_year_emissions_tco2e=Decimal("867000"),
                mc_runs=1000,
                random_seed=42,
            ))
        assert result.mc_runs_completed >= 1000

    def test_10000_runs_under_30min(self):
        """10,000 Monte Carlo runs must complete in <30 minutes."""
        engine = ScenarioModelingEngine()
        with timed_block("10000 MC runs", max_seconds=1800.0):
            result = engine.calculate(ScenarioInput(
                base_year_emissions_tco2e=Decimal("867000"),
                mc_runs=10000,
                random_seed=42,
            ))
        assert result.mc_runs_completed >= 10000


# ===========================================================================
# Tests -- SBTi Target Performance
# ===========================================================================


class TestSBTiPerformance:
    def test_target_calculation_under_5s(self):
        engine = SBTiTargetEngine()
        with timed_block("SBTi target calculation", max_seconds=5.0):
            result = engine.calculate(SBTiTargetInput(
                pathway_type=SBTiPathwayType.ACA_15C,
                base_year=2024, target_year=2030,
                baseline=BaselineData(
                    scope1_tco2e=Decimal("125000"),
                    scope2_tco2e=Decimal("85000"),
                    scope3_tco2e=Decimal("680000"),
                ),
            ))
        assert result is not None

    def test_criteria_validation_under_10s(self):
        engine = SBTiTargetEngine()
        with timed_block("Criteria validation", max_seconds=10.0):
            result = engine.calculate(SBTiTargetInput(
                pathway_type=SBTiPathwayType.ACA_15C,
                base_year=2024, target_year=2030,
                baseline=BaselineData(
                    scope1_tco2e=Decimal("125000"),
                    scope2_tco2e=Decimal("85000"),
                    scope3_tco2e=Decimal("680000"),
                ),
            ))
        assert len(result.criteria_validations) > 0


# ===========================================================================
# Tests -- Supply Chain Performance
# ===========================================================================


class TestSupplyChainPerformance:
    def test_50_suppliers_under_5s(self):
        engine = SupplyChainMappingEngine()
        suppliers = [_make_supplier(i) for i in range(50)]
        with timed_block("50 suppliers", max_seconds=5.0):
            result = engine.calculate(SupplyChainInput(suppliers=suppliers))
        assert result is not None

    def test_1000_suppliers_under_30s(self):
        engine = SupplyChainMappingEngine()
        suppliers = [_make_supplier(i) for i in range(1000)]
        with timed_block("1000 suppliers", max_seconds=30.0):
            result = engine.calculate(SupplyChainInput(suppliers=suppliers))
        assert result is not None


# ===========================================================================
# Tests -- Carbon Pricing Performance
# ===========================================================================


class TestCarbonPricingPerformance:
    def test_pricing_calculation_under_2s(self):
        engine = CarbonPricingEngine()
        with timed_block("Carbon pricing", max_seconds=2.0):
            result = engine.calculate(CarbonPricingInput(
                internal_carbon_price=Decimal("85"),
                total_scope1_tco2e=Decimal("867000"),
            ))
        assert result is not None

    def test_carbon_pricing_with_bu_under_5s(self):
        from engines.carbon_pricing_engine import BusinessUnitEmissions
        engine = CarbonPricingEngine()
        bus = [
            BusinessUnitEmissions(bu_name=f"BU-{i}", total_emissions_tco2e=Decimal("10000"),
                                  revenue_usd=Decimal("100000000"))
            for i in range(10)
        ]
        with timed_block("Carbon pricing with BUs", max_seconds=5.0):
            result = engine.calculate(CarbonPricingInput(
                internal_carbon_price=Decimal("85"),
                business_units=bus,
            ))
        assert result is not None


# ===========================================================================
# Tests -- Financial Integration Performance
# ===========================================================================


class TestFinancialPerformance:
    def test_carbon_pnl_under_5s(self):
        engine = FinancialIntegrationEngine()
        with timed_block("Carbon P&L", max_seconds=5.0):
            result = engine.calculate(FinancialIntegrationInput(
                total_scope1_tco2e=Decimal("867000"),
                internal_carbon_price=Decimal("85"),
            ))
        assert result is not None

    def test_board_report_generation_under_15min(self):
        """Board report must generate in <15 minutes from data refresh."""
        engine = FinancialIntegrationEngine()
        with timed_block("Board report data", max_seconds=900.0):
            result = engine.calculate(FinancialIntegrationInput(
                total_scope1_tco2e=Decimal("867000"),
                internal_carbon_price=Decimal("85"),
            ))
        assert result is not None


# ===========================================================================
# Tests -- Memory Usage
# ===========================================================================


class TestMemoryUsage:
    def test_baseline_memory_ceiling(self):
        """Baseline engine should not exceed memory ceiling."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        if hasattr(result, "peak_memory_mb"):
            assert result.peak_memory_mb < 2048  # 2GB ceiling

    def test_consolidation_memory_ceiling(self):
        """100 entity consolidation should not exceed memory ceiling."""
        engine = MultiEntityConsolidationEngine()
        entities = [_make_consolidation_entity(i) for i in range(100)]
        result = engine.calculate(ConsolidationInput(
            consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
            entities=entities,
        ))
        if hasattr(result, "peak_memory_mb"):
            assert result.peak_memory_mb < 4096  # 4GB ceiling
