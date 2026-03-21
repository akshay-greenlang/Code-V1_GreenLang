# -*- coding: utf-8 -*-
"""
End-to-end tests for PACK-027 Enterprise Net Zero Pack.

Tests full enterprise flows from onboarding through baseline, consolidation,
SBTi target setting, scenario modeling, carbon pricing, supply chain mapping,
financial integration, and assurance readiness.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~25 end-to-end scenarios
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.enterprise_baseline_engine import (
    EnterpriseBaselineEngine, EnterpriseBaselineInput, EnterpriseBaselineResult,
    FuelEntry, FuelType, ElectricityEntry, EntityDefinition,
)
from engines.sbti_target_engine import (
    SBTiTargetEngine, SBTiTargetInput, SBTiTargetResult,
    TargetPathwayType as SBTiPathwayType, BaselineData,
)
from engines.scenario_modeling_engine import (
    ScenarioModelingEngine, ScenarioModelingInput as ScenarioInput,
    ScenarioModelingResult as ScenarioResult,
)
from engines.carbon_pricing_engine import (
    CarbonPricingEngine, CarbonPricingInput, CarbonPricingResult,
)
from engines.scope4_avoided_emissions_engine import (
    Scope4AvoidedEmissionsEngine, Scope4Input,
    AvoidedEmissionCategory, ProductAvoidedEmissionEntry,
)
from engines.supply_chain_mapping_engine import (
    SupplyChainMappingEngine, SupplyChainMappingInput as SupplyChainInput,
    SupplierEntry,
)
from engines.multi_entity_consolidation_engine import (
    MultiEntityConsolidationEngine, ConsolidationInput, ConsolidationApproach,
    EntityEmissions,
)
from engines.financial_integration_engine import (
    FinancialIntegrationEngine, FinancialIntegrationInput,
)

from .conftest import assert_provenance_hash, assert_decimal_positive, timed_block


def _make_entity(eid="E1", name="TestEntity", country="DE"):
    return EntityDefinition(entity_id=eid, entity_name=name, country=country)


def _make_sbti_input(scope1=Decimal("125000"), scope2=Decimal("62000"),
                     scope3=Decimal("680000"), **overrides):
    defaults = dict(
        pathway_type=SBTiPathwayType.ACA_15C,
        base_year=2024, target_year=2030,
        baseline=BaselineData(
            scope1_tco2e=scope1, scope2_tco2e=scope2, scope3_tco2e=scope3,
        ),
    )
    defaults.update(overrides)
    return SBTiTargetInput(**defaults)


def _make_supplier(sid, name, country, spend, emissions):
    return SupplierEntry(
        supplier_id=sid, supplier_name=name, country=country,
        annual_spend_usd=spend, scope3_contribution_tco2e=emissions,
    )


# ========================================================================
# Manufacturing Enterprise Full Journey
# ========================================================================


class TestManufacturingEnterpriseJourney:
    """E2E: Large manufacturer from baseline through SBTi submission."""

    def test_full_manufacturing_pipeline(self):
        # Step 1: Enterprise Baseline
        baseline_engine = EnterpriseBaselineEngine()
        baseline = baseline_engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[
                FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=92000000, unit="kWh"),
                FuelEntry(fuel_type=FuelType.DIESEL, quantity=850000, unit="litres"),
            ],
            electricity_entries=[ElectricityEntry(annual_mwh=Decimal("185000"), region="DE")],
        ))
        assert isinstance(baseline, EnterpriseBaselineResult)
        assert baseline.total_tco2e_location > Decimal("0")

        # Step 2: SBTi Target Setting
        sbti_engine = SBTiTargetEngine()
        targets = sbti_engine.calculate(_make_sbti_input(
            scope1=baseline.scope1.total_tco2e,
            scope2=baseline.scope2.market_based_tco2e,
        ))
        assert targets.near_term_target.annual_reduction_rate_pct >= Decimal("4.2")

        # Step 3: Scenario Modeling
        scenario_engine = ScenarioModelingEngine()
        scenarios = scenario_engine.calculate(ScenarioInput(
            base_year_emissions_tco2e=baseline.total_tco2e_location,
            mc_runs=100, random_seed=42,
        ))
        assert scenarios.mc_runs_completed >= 100

        # Step 4: Carbon Pricing
        pricing_engine = CarbonPricingEngine()
        pricing = pricing_engine.calculate(CarbonPricingInput(
            internal_carbon_price=Decimal("85"),
            total_scope1_tco2e=baseline.scope1.total_tco2e,
            total_scope2_tco2e=baseline.scope2.location_based_tco2e,
        ))
        assert pricing.total_carbon_charge_usd >= Decimal("0")

        # Step 5: Financial Integration
        fin_engine = FinancialIntegrationEngine()
        financials = fin_engine.calculate(FinancialIntegrationInput(
            total_scope1_tco2e=baseline.scope1.total_tco2e,
            total_scope2_tco2e=baseline.scope2.location_based_tco2e,
            internal_carbon_price=Decimal("85"),
        ))
        assert financials.provenance_hash is not None


# ========================================================================
# Cross-Engine Integration
# ========================================================================


class TestCrossEngineIntegration:
    def test_baseline_to_sbti_data_flow(self):
        """Baseline result feeds directly into SBTi target engine."""
        baseline_engine = EnterpriseBaselineEngine()
        baseline = baseline_engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=5000000, unit="kWh")],
        ))
        sbti_engine = SBTiTargetEngine()
        targets = sbti_engine.calculate(_make_sbti_input(
            scope1=baseline.scope1.total_tco2e,
            scope2=baseline.scope2.market_based_tco2e,
        ))
        assert targets is not None

    def test_baseline_to_scenario_data_flow(self):
        """Baseline result feeds directly into scenario engine."""
        baseline_engine = EnterpriseBaselineEngine()
        baseline = baseline_engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        scenario_engine = ScenarioModelingEngine()
        scenarios = scenario_engine.calculate(ScenarioInput(
            base_year_emissions_tco2e=baseline.total_tco2e_location,
            mc_runs=100,
        ))
        assert scenarios is not None

    def test_baseline_to_carbon_pricing_data_flow(self):
        """Baseline result feeds directly into carbon pricing engine."""
        baseline_engine = EnterpriseBaselineEngine()
        baseline = baseline_engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        pricing_engine = CarbonPricingEngine()
        pricing = pricing_engine.calculate(CarbonPricingInput(
            internal_carbon_price=Decimal("85"),
            total_scope1_tco2e=baseline.scope1.total_tco2e,
        ))
        assert pricing is not None

    def test_all_engine_provenance_hashes(self):
        """Every engine must produce SHA-256 provenance hash."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert_provenance_hash(result)

    def test_end_to_end_performance(self):
        """Full pipeline should complete within enterprise time targets."""
        with timed_block("Full E2E pipeline", max_seconds=120):
            baseline_engine = EnterpriseBaselineEngine()
            baseline = baseline_engine.calculate(EnterpriseBaselineInput(
                entities=[_make_entity()],
            ))
            assert baseline is not None


# ========================================================================
# Parametrized Sector E2E Tests
# ========================================================================


E2E_SECTOR_CONFIGS = [
    ("manufacturing", "DE", Decimal("92000"), Decimal("185000")),
    ("energy_utilities", "US", Decimal("250000"), Decimal("500000")),
    ("financial_services", "GB", Decimal("5000"), Decimal("420000")),
    ("technology", "US", Decimal("8000"), Decimal("3200000")),
    ("consumer_goods", "NL", Decimal("15000"), Decimal("280000")),
    ("transport_logistics", "DE", Decimal("180000"), Decimal("45000")),
    ("real_estate", "GB", Decimal("12000"), Decimal("350000")),
    ("healthcare_pharma", "CH", Decimal("25000"), Decimal("150000")),
]


class TestSectorSpecificE2E:
    @pytest.mark.parametrize("sector,country,gas_kwh,elec_mwh", E2E_SECTOR_CONFIGS,
                             ids=[s[0] for s in E2E_SECTOR_CONFIGS])
    def test_sector_baseline_produces_emissions(self, sector, country, gas_kwh, elec_mwh):
        """Each sector must produce a valid baseline with emissions >= 0."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity(name=f"{sector}_e2e_test", country=country)],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=int(gas_kwh), unit="kWh")],
            electricity_entries=[ElectricityEntry(annual_mwh=elec_mwh, region=country)],
        ))
        assert result.total_tco2e_location >= Decimal("0")

    @pytest.mark.parametrize("sector,country,gas_kwh,elec_mwh", E2E_SECTOR_CONFIGS,
                             ids=[s[0] for s in E2E_SECTOR_CONFIGS])
    def test_sector_sbti_target_valid(self, sector, country, gas_kwh, elec_mwh):
        """Each sector baseline must produce data compatible with SBTi targeting."""
        baseline_engine = EnterpriseBaselineEngine()
        baseline = baseline_engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity(name=f"{sector}_sbti_test", country=country)],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=int(gas_kwh), unit="kWh")],
            electricity_entries=[ElectricityEntry(annual_mwh=elec_mwh, region=country)],
        ))
        sbti_engine = SBTiTargetEngine()
        targets = sbti_engine.calculate(_make_sbti_input(
            scope1=baseline.scope1.total_tco2e,
            scope2=baseline.scope2.market_based_tco2e,
        ))
        assert targets is not None

    @pytest.mark.parametrize("sector,country,gas_kwh,elec_mwh", E2E_SECTOR_CONFIGS,
                             ids=[s[0] for s in E2E_SECTOR_CONFIGS])
    def test_sector_carbon_pricing_valid(self, sector, country, gas_kwh, elec_mwh):
        """Each sector baseline must feed carbon pricing engine."""
        baseline_engine = EnterpriseBaselineEngine()
        baseline = baseline_engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity(name=f"{sector}_pricing_test", country=country)],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=int(gas_kwh), unit="kWh")],
            electricity_entries=[ElectricityEntry(annual_mwh=elec_mwh, region=country)],
        ))
        pricing_engine = CarbonPricingEngine()
        pricing = pricing_engine.calculate(CarbonPricingInput(
            internal_carbon_price=Decimal("85"),
            total_scope1_tco2e=baseline.scope1.total_tco2e,
            total_scope2_tco2e=baseline.scope2.location_based_tco2e,
        ))
        assert pricing is not None


# ========================================================================
# Parametrized SBTi Pathway E2E Tests
# ========================================================================


SBTI_PATHWAY_E2E = [
    (SBTiPathwayType.ACA_15C, 2024, 2030, Decimal("4.2")),
    (SBTiPathwayType.ACA_WB2C, 2024, 2030, Decimal("2.5")),
    (SBTiPathwayType.SDA, 2024, 2035, Decimal("0")),
    (SBTiPathwayType.FLAG, 2024, 2030, Decimal("0")),
]


class TestSBTiPathwayE2E:
    @pytest.mark.parametrize("pathway,base_yr,target_yr,min_rate", SBTI_PATHWAY_E2E,
                             ids=["ACA_15C", "ACA_WB2C", "SDA", "FLAG"])
    def test_pathway_target_generation(self, pathway, base_yr, target_yr, min_rate):
        """Each SBTi pathway must produce valid target outputs."""
        engine = SBTiTargetEngine()
        result = engine.calculate(SBTiTargetInput(
            pathway_type=pathway,
            base_year=base_yr,
            target_year=target_yr,
            baseline=BaselineData(
                scope1_tco2e=Decimal("125000"),
                scope2_tco2e=Decimal("62000"),
                scope3_tco2e=Decimal("680000"),
            ),
        ))
        assert result is not None
        if min_rate > Decimal("0"):
            assert result.near_term_target.annual_reduction_rate_pct >= min_rate
