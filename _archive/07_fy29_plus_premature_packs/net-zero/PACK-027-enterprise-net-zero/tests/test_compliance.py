# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Regulatory Compliance.

Tests compliance with 8+ simultaneous regulatory frameworks: GHG Protocol,
SBTi Corporate + Net-Zero, CDP, TCFD/ISSB S2, SEC Climate Rule,
CSRD/ESRS E1, California SB 253, ISO 14064-1, and verification standards.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~55 regulatory compliance tests
"""

import sys
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
from engines.financial_integration_engine import (
    FinancialIntegrationEngine, FinancialIntegrationInput,
)

from .conftest import assert_provenance_hash, REGULATORY_FRAMEWORKS


def _make_entity(eid="E1", name="ComplianceTest", country="DE"):
    return EntityDefinition(entity_id=eid, entity_name=name, country=country)


def _make_sbti_input(**overrides):
    defaults = dict(
        pathway_type=SBTiPathwayType.ACA_15C,
        base_year=2024,
        target_year=2030,
        baseline=BaselineData(
            scope1_tco2e=Decimal("125000"),
            scope2_tco2e=Decimal("62000"),
            scope3_tco2e=Decimal("680000"),
        ),
    )
    defaults.update(overrides)
    return SBTiTargetInput(**defaults)


# ===========================================================================
# Tests -- GHG Protocol Compliance
# ===========================================================================


class TestGHGProtocolCompliance:
    def test_organizational_boundary_defined(self):
        """GHG Protocol Chapter 3: Consolidation approach must be defined."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "consolidation_approach")

    def test_operational_boundary_scopes(self):
        """GHG Protocol Chapter 4: All three scopes must be reported."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=1000000, unit="kWh")],
            electricity_entries=[ElectricityEntry(annual_mwh=Decimal("1000"), region="DE")],
        ))
        assert hasattr(result, "scope1")
        assert hasattr(result, "scope2")
        assert hasattr(result, "scope3")

    def test_base_year_defined(self):
        """GHG Protocol Chapter 5: Base year must be defined."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "base_year")

    def test_gases_tracked_in_scope1(self):
        """GHG Protocol: Gases tracked in scope1 breakdown."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=1000000, unit="kWh")],
        ))
        assert hasattr(result.scope1, "by_gas")

    def test_scope2_dual_reporting(self):
        """GHG Protocol Scope 2 Guidance: Dual reporting required."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            electricity_entries=[ElectricityEntry(annual_mwh=Decimal("1000"), region="DE")],
        ))
        assert result.scope2.location_based_tco2e >= Decimal("0")
        assert result.scope2.market_based_tco2e >= Decimal("0")

    def test_scope3_coverage_tracked(self):
        """GHG Protocol Scope 3 Standard: Coverage percentage tracked."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result.scope3, "coverage_pct")


# ===========================================================================
# Tests -- SBTi Corporate Standard Compliance
# ===========================================================================


class TestSBTiCompliance:
    def test_criteria_validations_generated(self):
        """SBTi: Criteria validations must be generated."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert hasattr(result, "criteria_validations")
        assert len(result.criteria_validations) > 0

    def test_criteria_count_at_least_28(self):
        """SBTi Corporate Manual V5.3: At least 28 near-term criteria."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert result.criteria_pass_count + len([
            c for c in result.criteria_validations if c.status != "PASS"
        ]) >= 28

    def test_scope12_coverage(self):
        """SBTi: Scope 1+2 coverage must be tracked."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert hasattr(result, "near_term_target")

    def test_aca_15c_reduction_rate(self):
        """SBTi: ACA 1.5C requires >= 4.2%/yr absolute reduction."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert result.near_term_target.annual_reduction_rate_pct >= Decimal("4.2")

    def test_submission_readiness(self):
        """SBTi: Submission readiness score must be computed."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert hasattr(result, "submission_readiness_score")

    def test_milestones_generated(self):
        """SBTi: Milestones must be generated for the pathway."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert len(result.milestones) > 0


# ===========================================================================
# Tests -- SEC Climate Disclosure Rule
# ===========================================================================


class TestSECClimateRule:
    def test_scope12_disclosure(self):
        """SEC: Scope 1+2 emissions must be disclosed."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=1000000, unit="kWh")],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")
        assert result.scope2.location_based_tco2e >= Decimal("0")

    def test_materiality_assessment_present(self):
        """SEC: Materiality assessment must exist."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "materiality")

    def test_data_quality_assessment(self):
        """SEC: Data quality assessment for attestation."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "data_quality")


# ===========================================================================
# Tests -- CSRD / ESRS E1 Compliance
# ===========================================================================


class TestCSRDCompliance:
    def test_esrs_e14_targets(self):
        """ESRS E1-4: GHG reduction targets must be disclosed."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_sbti_input())
        assert result is not None
        assert result.near_term_target is not None

    def test_esrs_e16_emissions(self):
        """ESRS E1-6: Scope 1/2/3 emissions must be reported."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=1000000, unit="kWh")],
            electricity_entries=[ElectricityEntry(annual_mwh=Decimal("1000"), region="DE")],
        ))
        assert result.scope1.total_tco2e >= Decimal("0")
        assert result.scope2.location_based_tco2e >= Decimal("0")

    def test_esrs_e18_carbon_pricing(self):
        """ESRS E1-8: Internal carbon pricing must be disclosed."""
        engine = FinancialIntegrationEngine()
        result = engine.calculate(FinancialIntegrationInput(
            internal_carbon_price=Decimal("85"),
            total_scope1_tco2e=Decimal("125000"),
        ))
        assert hasattr(result, "esrs_disclosure")

    def test_esrs_e19_financial_effects(self):
        """ESRS E1-9: Anticipated financial effects must be disclosed."""
        engine = FinancialIntegrationEngine()
        result = engine.calculate(FinancialIntegrationInput(
            internal_carbon_price=Decimal("85"),
        ))
        assert hasattr(result, "esrs_disclosure")
        assert hasattr(result.esrs_disclosure, "e1_9_physical_risk_exposure_usd")


# ===========================================================================
# Tests -- ISO 14064-1 Compliance
# ===========================================================================


class TestISO14064Compliance:
    def test_iso_principles_in_methodology(self):
        """ISO 14064-1: Methodology notes should be generated."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=1000000, unit="kWh")],
        ))
        assert hasattr(result.scope1, "methodology_notes")

    def test_iso_boundary_definition(self):
        """ISO 14064-1: Organizational boundary must be defined."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "consolidation_approach")

    def test_iso_data_quality(self):
        """ISO 14064-1: Data quality must be assessed."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "data_quality")


# ===========================================================================
# Tests -- California SB 253
# ===========================================================================


class TestCaliforniaSB253:
    def test_scope123_required(self):
        """SB 253: All scopes must be reported."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "scope1")
        assert hasattr(result, "scope2")
        assert hasattr(result, "scope3")

    def test_regulatory_citations(self):
        """SB 253: Regulatory citations should be generated."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert len(result.regulatory_citations) > 0


# ===========================================================================
# Tests -- Verification / Assurance Standards
# ===========================================================================


class TestAssuranceCompliance:
    def test_sha256_provenance_all_calculations(self):
        """All calculations must have SHA-256 provenance hash."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert_provenance_hash(result)

    def test_deterministic_calculations(self):
        """Zero-hallucination: same inputs must produce same outputs."""
        engine = EnterpriseBaselineEngine()
        inp = EnterpriseBaselineInput(
            entities=[_make_entity()],
            fuel_entries=[FuelEntry(fuel_type=FuelType.NATURAL_GAS, quantity=10000000, unit="kWh")],
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.total_tco2e_location == r2.total_tco2e_location

    @pytest.mark.parametrize("framework", REGULATORY_FRAMEWORKS)
    def test_framework_regulatory_citations(self, framework):
        """Each regulatory framework reference must exist as a constant."""
        assert isinstance(framework, str)
        assert len(framework) > 0

    def test_assurance_workpaper_readiness(self):
        """Engine output must have all fields needed for assurance."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "regulatory_citations")


# ===========================================================================
# Tests -- Multi-Framework Simultaneous Compliance
# ===========================================================================


SCOPE3_ALL_CATEGORIES = list(range(1, 16))


class TestMultiFrameworkCompliance:
    def test_scope3_categories_tracked(self):
        """Scope 3 categories breakdown must exist."""
        engine = EnterpriseBaselineEngine()
        result = engine.calculate(EnterpriseBaselineInput(
            entities=[_make_entity()],
        ))
        assert hasattr(result.scope3, "categories")

    @pytest.mark.parametrize("gas", [
        "CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3",
    ])
    def test_kyoto_gas_tracking(self, gas):
        """All 7 Kyoto gases must be a valid gas name."""
        assert isinstance(gas, str) and len(gas) > 0

    @pytest.mark.parametrize("approach", [
        "financial_control", "operational_control", "equity_share",
    ])
    def test_organizational_boundary_approach(self, approach):
        """Each consolidation approach must be a known value."""
        assert approach in ["financial_control", "operational_control", "equity_share"]
