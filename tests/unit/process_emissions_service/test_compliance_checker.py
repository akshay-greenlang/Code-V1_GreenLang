# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6 of 6).

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Tests all 6 regulatory compliance frameworks (60 total requirements),
data completeness validation, methodology validation, recommendations
generation, and edge cases.

Total: 157 tests across 9 test classes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from greenlang.process_emissions.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceCheckResult,
    ComplianceRequirement,
    FrameworkResult,
    SUPPORTED_FRAMEWORKS,
    EPA_SUBPART_MAP,
    PROCESS_PRIMARY_GASES,
    ALL_GHG_GASES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ComplianceCheckerEngine:
    """Create a fresh ComplianceCheckerEngine instance."""
    return ComplianceCheckerEngine()


@pytest.fixture
def compliant_cement_data() -> Dict[str, Any]:
    """Fully compliant cement production data for all frameworks."""
    return {
        "process_type": "CEMENT_PRODUCTION",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_2",
        "total_co2e_tonnes": 50000.0,
        "emissions_by_gas": [{"gas": "CO2", "co2e_tonnes": 50000.0}],
        "emission_factor_source": "IPCC",
        "gwp_source": "AR6",
        "provenance_hash": "abc123def456789012345678901234567890123456789012345678901234",
        "calculation_trace": [{"step": "resolve", "detail": "cement EF"}],
        "facility_id": "FAC-001",
        "organization_id": "ORG-001",
        "production_quantity_tonnes": 100000.0,
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "uncertainty_pct": 3.5,
        "uncertainty_result": {"mean": 50000.0, "std_dev": 1750.0},
        "base_year": 2020,
        "recalculation_policy": "5% threshold",
        "previous_year_co2e": 48000.0,
        "reduction_targets": {"target_year": 2030, "reduction_pct": 30},
        "mitigation_actions": ["CCS pilot project"],
        "abatement_co2e_tonnes": 5000.0,
        "sector_metrics": {"clinker_ratio": 0.65},
    }


@pytest.fixture
def minimal_data() -> Dict[str, Any]:
    """Minimal data that will cause many checks to fail."""
    return {}


@pytest.fixture
def partial_data() -> Dict[str, Any]:
    """Partial data with some fields present, some missing."""
    return {
        "process_type": "CEMENT_PRODUCTION",
        "calculation_method": "EMISSION_FACTOR",
        "total_co2e_tonnes": 50000.0,
        "emissions_by_gas": [{"gas": "CO2", "co2e_tonnes": 50000.0}],
    }


@pytest.fixture
def semiconductor_data() -> Dict[str, Any]:
    """Semiconductor production data with multi-gas profile."""
    return {
        "process_type": "SEMICONDUCTOR",
        "calculation_method": "EMISSION_FACTOR",
        "tier": "TIER_1",
        "total_co2e_tonnes": 1500.0,
        "emissions_by_gas": [
            {"gas": "CF4", "co2e_tonnes": 500.0},
            {"gas": "C2F6", "co2e_tonnes": 400.0},
            {"gas": "SF6", "co2e_tonnes": 300.0},
            {"gas": "NF3", "co2e_tonnes": 200.0},
            {"gas": "HFC", "co2e_tonnes": 100.0},
        ],
        "emission_factor_source": "EPA",
        "gwp_source": "AR6",
        "provenance_hash": "semi123",
        "facility_id": "FAC-SEMI-001",
        "production_quantity_tonnes": 50.0,
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
    }


@pytest.fixture
def aluminum_data() -> Dict[str, Any]:
    """Aluminum smelting data with PFC gases."""
    return {
        "process_type": "ALUMINUM_SMELTING",
        "calculation_method": "MASS_BALANCE",
        "tier": "TIER_3",
        "total_co2e_tonnes": 80000.0,
        "emissions_by_gas": [
            {"gas": "CO2", "co2e_tonnes": 60000.0},
            {"gas": "CF4", "co2e_tonnes": 15000.0},
            {"gas": "C2F6", "co2e_tonnes": 5000.0},
        ],
        "emission_factor_source": "EU_ETS",
        "gwp_source": "AR6",
        "provenance_hash": "alu456",
        "calculation_trace": [{"step": "mass_balance"}],
        "facility_id": "FAC-ALU-001",
        "production_quantity_tonnes": 50000.0,
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "uncertainty_pct": 2.0,
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestEngineInit:
    """Test ComplianceCheckerEngine initialization."""

    def test_init_creates_6_frameworks(self, engine: ComplianceCheckerEngine):
        """Engine initializes with exactly 6 frameworks."""
        assert len(engine._requirements) == 6

    def test_init_creates_60_total_requirements(
        self, engine: ComplianceCheckerEngine,
    ):
        """Engine has 10 requirements per framework, 60 total."""
        total = sum(len(v) for v in engine._requirements.values())
        assert total == 60

    def test_all_framework_names_match_supported(
        self, engine: ComplianceCheckerEngine,
    ):
        """All framework keys match SUPPORTED_FRAMEWORKS constant."""
        for fw in SUPPORTED_FRAMEWORKS:
            assert fw in engine._requirements

    def test_each_framework_has_10_requirements(
        self, engine: ComplianceCheckerEngine,
    ):
        """Each framework has exactly 10 requirements."""
        for fw, reqs in engine._requirements.items():
            assert len(reqs) == 10, f"{fw} has {len(reqs)} reqs, expected 10"

    def test_all_requirements_have_validation_fn(
        self, engine: ComplianceCheckerEngine,
    ):
        """Every requirement has a validation_fn that exists on the engine."""
        for fw, reqs in engine._requirements.items():
            for req in reqs:
                assert req.validation_fn, (
                    f"{fw}/{req.requirement_id}: missing validation_fn"
                )
                assert hasattr(engine, req.validation_fn), (
                    f"{fw}/{req.requirement_id}: method {req.validation_fn} not found"
                )


class TestGHGProtocol:
    """Test GHG Protocol Corporate Standard compliance checks."""

    def test_ghg_fully_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Fully compliant data passes all GHG Protocol checks."""
        result = engine.check_framework(
            compliant_cement_data, "GHG_PROTOCOL",
        )
        assert result.status == "COMPLIANT"
        assert result.passed == 10
        assert result.failed == 0

    def test_ghg_process_identification_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Process identification passes when process_type is set."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        pid_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_process_identification"
        )
        assert pid_result.passed is True

    def test_ghg_process_identification_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Process identification fails when process_type is missing."""
        data = {"calculation_method": "EMISSION_FACTOR"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        pid_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_process_identification"
        )
        assert pid_result.passed is False
        assert pid_result.severity == "ERROR"

    def test_ghg_calculation_methodology_valid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Passes with valid calculation methods."""
        for method in [
            "EMISSION_FACTOR", "MASS_BALANCE",
            "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
        ]:
            data = {"calculation_method": method}
            result = engine.check_framework(data, "GHG_PROTOCOL")
            meth_result = next(
                r for r in result.results
                if r.requirement_id == "ghg_calculation_methodology"
            )
            assert meth_result.passed is True, f"Failed for method={method}"

    def test_ghg_calculation_methodology_invalid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Fails with invalid calculation method."""
        data = {"calculation_method": "MADE_UP_METHOD"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        meth_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_calculation_methodology"
        )
        assert meth_result.passed is False

    def test_ghg_ef_source_valid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Passes with valid emission factor sources."""
        for source in ["EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"]:
            data = {"emission_factor_source": source}
            result = engine.check_framework(data, "GHG_PROTOCOL")
            ef_result = next(
                r for r in result.results
                if r.requirement_id == "ghg_emission_factor_source"
            )
            assert ef_result.passed is True, f"Failed for source={source}"

    def test_ghg_ef_source_invalid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Fails with unrecognized emission factor source."""
        data = {"emission_factor_source": "UNKNOWN"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        ef_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_emission_factor_source"
        )
        assert ef_result.passed is False

    def test_ghg_boundary_completeness_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Boundary completeness passes with process_type and positive CO2e."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "total_co2e_tonnes": 50000.0,
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        bc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_boundary_completeness"
        )
        assert bc_result.passed is True

    def test_ghg_boundary_completeness_zero_emissions(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Boundary completeness fails with zero emissions."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "total_co2e_tonnes": 0,
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        bc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_boundary_completeness"
        )
        assert bc_result.passed is False

    def test_ghg_gas_coverage_complete(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Gas coverage passes when all required gases are reported."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "emissions_by_gas": [{"gas": "CO2"}],
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        gc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_gas_coverage"
        )
        assert gc_result.passed is True

    def test_ghg_gas_coverage_missing(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Gas coverage fails when required gases are missing."""
        data = {
            "process_type": "ALUMINUM_SMELTING",
            "emissions_by_gas": [{"gas": "CO2"}],
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        gc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_gas_coverage"
        )
        assert gc_result.passed is False
        assert "CF4" in gc_result.details or "Missing" in gc_result.details

    def test_ghg_de_minimis_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """De minimis passes when excluded emissions are below 5%."""
        data = {
            "total_co2e_tonnes": 50000.0,
            "excluded_emissions_pct": 3.0,
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        dm_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_de_minimis_threshold"
        )
        assert dm_result.passed is True

    def test_ghg_de_minimis_exceeds_threshold(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """De minimis fails when exclusions exceed 5%."""
        data = {
            "total_co2e_tonnes": 50000.0,
            "excluded_emissions_pct": 7.0,
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        dm_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_de_minimis_threshold"
        )
        assert dm_result.passed is False

    def test_ghg_temporal_consistency_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Temporal consistency passes when method and tier documented."""
        data = {
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_1",
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        tc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_temporal_consistency"
        )
        assert tc_result.passed is True

    def test_ghg_temporal_consistency_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Temporal consistency fails when method or tier is missing."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        tc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_temporal_consistency"
        )
        assert tc_result.passed is False

    def test_ghg_quality_management_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Quality management passes with provenance hash."""
        data = {"provenance_hash": "abc123"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        qm_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_quality_management"
        )
        assert qm_result.passed is True

    def test_ghg_quality_management_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Quality management fails without provenance hash."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        qm_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_quality_management"
        )
        assert qm_result.passed is False

    def test_ghg_verification_readiness_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Verification readiness passes with hash and trace."""
        data = {
            "provenance_hash": "abc123",
            "calculation_trace": [{"step": "resolve"}],
        }
        result = engine.check_framework(data, "GHG_PROTOCOL")
        vr_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_verification_readiness"
        )
        assert vr_result.passed is True

    def test_ghg_verification_readiness_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Verification readiness fails without hash or trace."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "GHG_PROTOCOL")
        vr_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_verification_readiness"
        )
        assert vr_result.passed is False


class TestISO14064:
    """Test ISO 14064-1:2018 compliance checks."""

    def test_iso_fully_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Fully compliant data passes all ISO 14064 checks."""
        result = engine.check_framework(
            compliant_cement_data, "ISO_14064",
        )
        assert result.status == "COMPLIANT"
        assert result.passed == 10

    def test_iso_org_boundary_pass_facility(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Org boundary passes with facility_id."""
        data = {"facility_id": "FAC-001"}
        result = engine.check_framework(data, "ISO_14064")
        ob_result = next(
            r for r in result.results
            if r.requirement_id == "iso_organizational_boundary"
        )
        assert ob_result.passed is True

    def test_iso_org_boundary_pass_organization(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Org boundary passes with organization_id."""
        data = {"organization_id": "ORG-001"}
        result = engine.check_framework(data, "ISO_14064")
        ob_result = next(
            r for r in result.results
            if r.requirement_id == "iso_organizational_boundary"
        )
        assert ob_result.passed is True

    def test_iso_org_boundary_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Org boundary fails without facility_id or organization_id."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "ISO_14064")
        ob_result = next(
            r for r in result.results
            if r.requirement_id == "iso_organizational_boundary"
        )
        assert ob_result.passed is False

    def test_iso_sources_identified_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """GHG sources identification passes with process_type."""
        data = {"process_type": "NITRIC_ACID"}
        result = engine.check_framework(data, "ISO_14064")
        si_result = next(
            r for r in result.results
            if r.requirement_id == "iso_ghg_sources_identified"
        )
        assert si_result.passed is True

    def test_iso_methodology_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology passes with calculation_method and ef_source."""
        data = {
            "calculation_method": "MASS_BALANCE",
            "emission_factor_source": "IPCC",
        }
        result = engine.check_framework(data, "ISO_14064")
        meth_result = next(
            r for r in result.results
            if r.requirement_id == "iso_quantification_methodology"
        )
        assert meth_result.passed is True

    def test_iso_methodology_fail_missing_ef_source(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology fails when emission_factor_source is missing."""
        data = {"calculation_method": "MASS_BALANCE"}
        result = engine.check_framework(data, "ISO_14064")
        meth_result = next(
            r for r in result.results
            if r.requirement_id == "iso_quantification_methodology"
        )
        assert meth_result.passed is False

    def test_iso_data_quality_valid_tiers(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Data quality passes with valid tiers."""
        for tier in ["TIER_1", "TIER_2", "TIER_3"]:
            data = {"tier": tier}
            result = engine.check_framework(data, "ISO_14064")
            dq_result = next(
                r for r in result.results
                if r.requirement_id == "iso_data_quality"
            )
            assert dq_result.passed is True, f"Failed for tier={tier}"

    def test_iso_data_quality_invalid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Data quality fails with invalid tier."""
        data = {"tier": "TIER_5"}
        result = engine.check_framework(data, "ISO_14064")
        dq_result = next(
            r for r in result.results
            if r.requirement_id == "iso_data_quality"
        )
        assert dq_result.passed is False

    def test_iso_uncertainty_with_result(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Uncertainty passes when uncertainty_result is present."""
        data = {"uncertainty_result": {"mean": 1000, "std_dev": 50}}
        result = engine.check_framework(data, "ISO_14064")
        u_result = next(
            r for r in result.results
            if r.requirement_id == "iso_uncertainty_assessed"
        )
        assert u_result.passed is True

    def test_iso_uncertainty_without_result_still_passes(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Uncertainty advisory check passes even without data."""
        data = {}
        result = engine.check_framework(data, "ISO_14064")
        u_result = next(
            r for r in result.results
            if r.requirement_id == "iso_uncertainty_assessed"
        )
        assert u_result.passed is True  # WARNING, not blocking

    def test_iso_documentation_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Documentation passes with provenance hash."""
        data = {"provenance_hash": "doc_hash_abc"}
        result = engine.check_framework(data, "ISO_14064")
        doc_result = next(
            r for r in result.results
            if r.requirement_id == "iso_documentation_complete"
        )
        assert doc_result.passed is True

    def test_iso_documentation_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Documentation fails without provenance hash."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "ISO_14064")
        doc_result = next(
            r for r in result.results
            if r.requirement_id == "iso_documentation_complete"
        )
        assert doc_result.passed is False

    def test_iso_consistent_reporting_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Consistent reporting passes with calculation_method."""
        data = {"calculation_method": "EMISSION_FACTOR"}
        result = engine.check_framework(data, "ISO_14064")
        cr_result = next(
            r for r in result.results
            if r.requirement_id == "iso_consistent_reporting"
        )
        assert cr_result.passed is True

    def test_iso_consistent_reporting_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Consistent reporting fails without calculation_method."""
        data = {"tier": "TIER_1"}
        result = engine.check_framework(data, "ISO_14064")
        cr_result = next(
            r for r in result.results
            if r.requirement_id == "iso_consistent_reporting"
        )
        assert cr_result.passed is False


class TestCSRD:
    """Test CSRD / ESRS E1 compliance checks."""

    def test_csrd_fully_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Fully compliant data passes all CSRD checks."""
        result = engine.check_framework(
            compliant_cement_data, "CSRD_ESRS_E1",
        )
        assert result.status == "COMPLIANT"

    def test_csrd_scope1_category_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Scope 1 by category passes with process_type and emissions."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "total_co2e_tonnes": 50000.0,
        }
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        sc_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_scope1_by_category"
        )
        assert sc_result.passed is True

    def test_csrd_scope1_category_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Scope 1 by category fails without process_type."""
        data = {"total_co2e_tonnes": 50000.0}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        sc_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_scope1_by_category"
        )
        assert sc_result.passed is False

    def test_csrd_ghg_methodology_valid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """GHG Protocol methodology passes with valid method."""
        data = {"calculation_method": "EMISSION_FACTOR"}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        gm_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_ghg_protocol_methodology"
        )
        assert gm_result.passed is True

    def test_csrd_ghg_methodology_invalid(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """GHG Protocol methodology fails with invalid method."""
        data = {"calculation_method": "GUESS"}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        gm_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_ghg_protocol_methodology"
        )
        assert gm_result.passed is False

    def test_csrd_process_separation_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Process separation passes when process_type is set."""
        data = {"process_type": "IRON_STEEL"}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        ps_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_process_emissions_separated"
        )
        assert ps_result.passed is True

    def test_csrd_assurance_readiness_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Assurance readiness passes with provenance and EF source."""
        data = {
            "provenance_hash": "abc",
            "emission_factor_source": "IPCC",
        }
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        ar_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_limited_assurance"
        )
        assert ar_result.passed is True

    def test_csrd_assurance_readiness_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Assurance readiness fails without provenance hash."""
        data = {"emission_factor_source": "IPCC"}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        ar_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_limited_assurance"
        )
        assert ar_result.passed is False

    def test_csrd_digital_tagging_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Digital tagging passes with process_type and CO2e."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "total_co2e_tonnes": 50000.0,
        }
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        dt_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_digital_tagging"
        )
        assert dt_result.passed is True

    def test_csrd_financial_always_passes(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Financial implications check always passes (INFO)."""
        data = {}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        fi_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_financial_implications"
        )
        assert fi_result.passed is True

    def test_csrd_value_chain_always_passes(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Value chain check always passes (INFO)."""
        data = {}
        result = engine.check_framework(data, "CSRD_ESRS_E1")
        vc_result = next(
            r for r in result.results
            if r.requirement_id == "csrd_value_chain"
        )
        assert vc_result.passed is True


class TestEPA40CFR98:
    """Test EPA 40 CFR Part 98 compliance checks."""

    def test_epa_fully_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Fully compliant data passes all EPA checks."""
        result = engine.check_framework(
            compliant_cement_data, "EPA_40CFR98",
        )
        assert result.status == "COMPLIANT"

    def test_epa_monitoring_plan_mapped_process(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Monitoring plan passes with a mapped EPA process type."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "EPA_40CFR98")
        mp_result = next(
            r for r in result.results
            if r.requirement_id == "epa_monitoring_plan"
        )
        assert mp_result.passed is True
        assert "Subpart F" in mp_result.details

    def test_epa_monitoring_plan_unmapped_process(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Monitoring plan passes for process with no EPA subpart."""
        data = {"process_type": "UNKNOWN_PROCESS"}
        result = engine.check_framework(data, "EPA_40CFR98")
        mp_result = next(
            r for r in result.results
            if r.requirement_id == "epa_monitoring_plan"
        )
        assert mp_result.passed is True

    def test_epa_monitoring_plan_no_process(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Monitoring plan fails without process_type."""
        data = {}
        result = engine.check_framework(data, "EPA_40CFR98")
        mp_result = next(
            r for r in result.results
            if r.requirement_id == "epa_monitoring_plan"
        )
        assert mp_result.passed is False

    def test_epa_mass_measurement_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Mass measurement passes with production quantity."""
        data = {"production_quantity_tonnes": 100000.0}
        result = engine.check_framework(data, "EPA_40CFR98")
        mm_result = next(
            r for r in result.results
            if r.requirement_id == "epa_mass_measurement"
        )
        assert mm_result.passed is True

    def test_epa_mass_measurement_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Mass measurement fails without production quantity."""
        data = {}
        result = engine.check_framework(data, "EPA_40CFR98")
        mm_result = next(
            r for r in result.results
            if r.requirement_id == "epa_mass_measurement"
        )
        assert mm_result.passed is False

    def test_epa_tier_methodology_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Tier methodology passes with valid tier and process type."""
        data = {
            "tier": "TIER_2",
            "process_type": "CEMENT_PRODUCTION",
        }
        result = engine.check_framework(data, "EPA_40CFR98")
        tm_result = next(
            r for r in result.results
            if r.requirement_id == "epa_tier_methodology"
        )
        assert tm_result.passed is True

    def test_epa_tier_methodology_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Tier methodology fails without valid tier."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.check_framework(data, "EPA_40CFR98")
        tm_result = next(
            r for r in result.results
            if r.requirement_id == "epa_tier_methodology"
        )
        assert tm_result.passed is False

    def test_epa_missing_data_below_threshold(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Missing data passes when rate is below 10%."""
        data = {"missing_data_pct": 5.0}
        result = engine.check_framework(data, "EPA_40CFR98")
        md_result = next(
            r for r in result.results
            if r.requirement_id == "epa_missing_data"
        )
        assert md_result.passed is True

    def test_epa_missing_data_above_threshold(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Missing data fails when rate exceeds 10%."""
        data = {"missing_data_pct": 15.0}
        result = engine.check_framework(data, "EPA_40CFR98")
        md_result = next(
            r for r in result.results
            if r.requirement_id == "epa_missing_data"
        )
        assert md_result.passed is False

    def test_epa_annual_reporting_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Annual reporting passes with period fields."""
        data = {
            "period_start": "2025-01-01",
            "period_end": "2025-12-31",
        }
        result = engine.check_framework(data, "EPA_40CFR98")
        ar_result = next(
            r for r in result.results
            if r.requirement_id == "epa_annual_reporting"
        )
        assert ar_result.passed is True

    def test_epa_annual_reporting_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Annual reporting fails without period fields."""
        data = {}
        result = engine.check_framework(data, "EPA_40CFR98")
        ar_result = next(
            r for r in result.results
            if r.requirement_id == "epa_annual_reporting"
        )
        assert ar_result.passed is False

    def test_epa_recordkeeping_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Recordkeeping passes with provenance hash."""
        data = {"provenance_hash": "record_hash"}
        result = engine.check_framework(data, "EPA_40CFR98")
        rk_result = next(
            r for r in result.results
            if r.requirement_id == "epa_recordkeeping"
        )
        assert rk_result.passed is True

    def test_epa_recordkeeping_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Recordkeeping fails without provenance hash."""
        data = {}
        result = engine.check_framework(data, "EPA_40CFR98")
        rk_result = next(
            r for r in result.results
            if r.requirement_id == "epa_recordkeeping"
        )
        assert rk_result.passed is False

    @pytest.mark.parametrize("process_type,expected_subparts", [
        ("CEMENT_PRODUCTION", ["Subpart F"]),
        ("NITRIC_ACID", ["Subpart V"]),
        ("SEMICONDUCTOR", ["Subpart I"]),
        ("IRON_STEEL", ["Subpart Q"]),
        ("ALUMINUM_SMELTING", ["Subpart F"]),
    ])
    def test_epa_subpart_mapping(
        self,
        engine: ComplianceCheckerEngine,
        process_type: str,
        expected_subparts: List[str],
    ):
        """EPA subpart mapping returns correct subparts for process types."""
        subparts = engine.get_applicable_epa_subparts(process_type)
        assert subparts == expected_subparts


class TestUKSECR:
    """Test UK SECR compliance checks."""

    def test_secr_fully_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Fully compliant data passes all UK SECR checks."""
        result = engine.check_framework(
            compliant_cement_data, "UK_SECR",
        )
        assert result.status == "COMPLIANT"

    def test_secr_energy_disclosure_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Energy disclosure passes with positive emissions."""
        data = {"total_co2e_tonnes": 50000.0}
        result = engine.check_framework(data, "UK_SECR")
        ed_result = next(
            r for r in result.results
            if r.requirement_id == "secr_energy_emissions"
        )
        assert ed_result.passed is True

    def test_secr_energy_disclosure_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Energy disclosure fails with zero or missing emissions."""
        data = {}
        result = engine.check_framework(data, "UK_SECR")
        ed_result = next(
            r for r in result.results
            if r.requirement_id == "secr_energy_emissions"
        )
        assert ed_result.passed is False

    def test_secr_defra_methodology_preferred(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """DEFRA methodology passes perfectly with DEFRA source."""
        data = {"emission_factor_source": "DEFRA"}
        result = engine.check_framework(data, "UK_SECR")
        dm_result = next(
            r for r in result.results
            if r.requirement_id == "secr_methodology"
        )
        assert dm_result.passed is True

    def test_secr_non_defra_methodology_still_passes(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Non-DEFRA source still passes but with recommendation."""
        data = {"emission_factor_source": "IPCC"}
        result = engine.check_framework(data, "UK_SECR")
        dm_result = next(
            r for r in result.results
            if r.requirement_id == "secr_methodology"
        )
        assert dm_result.passed is True
        assert len(dm_result.recommendations) > 0

    def test_secr_no_methodology_fails(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Missing emission_factor_source fails methodology check."""
        data = {}
        result = engine.check_framework(data, "UK_SECR")
        dm_result = next(
            r for r in result.results
            if r.requirement_id == "secr_methodology"
        )
        assert dm_result.passed is False

    def test_secr_intensity_ratio_calculable(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Intensity ratio passes when both qty and CO2e are present."""
        data = {
            "production_quantity_tonnes": 100000.0,
            "total_co2e_tonnes": 50000.0,
        }
        result = engine.check_framework(data, "UK_SECR")
        ir_result = next(
            r for r in result.results
            if r.requirement_id == "secr_intensity_ratio"
        )
        assert ir_result.passed is True
        assert "0.5000" in ir_result.details

    def test_secr_intensity_ratio_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Intensity ratio fails without production quantity."""
        data = {"total_co2e_tonnes": 50000.0}
        result = engine.check_framework(data, "UK_SECR")
        ir_result = next(
            r for r in result.results
            if r.requirement_id == "secr_intensity_ratio"
        )
        assert ir_result.passed is False

    def test_secr_scope_stated_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Scope reporting passes with process_type."""
        data = {"process_type": "GLASS_PRODUCTION"}
        result = engine.check_framework(data, "UK_SECR")
        ss_result = next(
            r for r in result.results
            if r.requirement_id == "secr_scope_stated"
        )
        assert ss_result.passed is True

    def test_secr_methodology_stated_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology description passes with method and EF source."""
        data = {
            "calculation_method": "EMISSION_FACTOR",
            "emission_factor_source": "DEFRA",
        }
        result = engine.check_framework(data, "UK_SECR")
        ms_result = next(
            r for r in result.results
            if r.requirement_id == "secr_methodology_stated"
        )
        assert ms_result.passed is True

    def test_secr_companies_act_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Companies Act compliance passes with positive emissions."""
        data = {"total_co2e_tonnes": 50000.0}
        result = engine.check_framework(data, "UK_SECR")
        ca_result = next(
            r for r in result.results
            if r.requirement_id == "secr_companies_act"
        )
        assert ca_result.passed is True

    def test_secr_companies_act_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Companies Act compliance fails without emissions."""
        data = {}
        result = engine.check_framework(data, "UK_SECR")
        ca_result = next(
            r for r in result.results
            if r.requirement_id == "secr_companies_act"
        )
        assert ca_result.passed is False


class TestEUETSMRR:
    """Test EU ETS MRR compliance checks."""

    def test_ets_fully_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Fully compliant data passes all EU ETS checks."""
        result = engine.check_framework(
            compliant_cement_data, "EU_ETS_MRR",
        )
        assert result.status == "COMPLIANT"

    def test_ets_monitoring_plan_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Monitoring plan passes with process_type and method."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "calculation_method": "EMISSION_FACTOR",
        }
        result = engine.check_framework(data, "EU_ETS_MRR")
        mp_result = next(
            r for r in result.results
            if r.requirement_id == "ets_monitoring_plan"
        )
        assert mp_result.passed is True

    def test_ets_monitoring_plan_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Monitoring plan fails without required fields."""
        data = {}
        result = engine.check_framework(data, "EU_ETS_MRR")
        mp_result = next(
            r for r in result.results
            if r.requirement_id == "ets_monitoring_plan"
        )
        assert mp_result.passed is False

    def test_ets_tier_justified_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Tier justification passes with valid tier."""
        for tier in ["TIER_1", "TIER_2", "TIER_3"]:
            data = {"tier": tier}
            result = engine.check_framework(data, "EU_ETS_MRR")
            tj_result = next(
                r for r in result.results
                if r.requirement_id == "ets_tier_justified"
            )
            assert tj_result.passed is True, f"Failed for tier={tier}"

    def test_ets_tier_justified_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Tier justification fails without valid tier."""
        data = {"tier": "INVALID"}
        result = engine.check_framework(data, "EU_ETS_MRR")
        tj_result = next(
            r for r in result.results
            if r.requirement_id == "ets_tier_justified"
        )
        assert tj_result.passed is False

    def test_ets_uncertainty_within_threshold(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Uncertainty passes when within tier threshold."""
        data = {"tier": "TIER_2", "uncertainty_pct": 4.0}
        result = engine.check_framework(data, "EU_ETS_MRR")
        ut_result = next(
            r for r in result.results
            if r.requirement_id == "ets_uncertainty_threshold"
        )
        assert ut_result.passed is True

    def test_ets_uncertainty_exceeds_threshold(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Uncertainty fails when exceeding tier threshold."""
        data = {"tier": "TIER_2", "uncertainty_pct": 7.0}
        result = engine.check_framework(data, "EU_ETS_MRR")
        ut_result = next(
            r for r in result.results
            if r.requirement_id == "ets_uncertainty_threshold"
        )
        assert ut_result.passed is False

    @pytest.mark.parametrize("tier,threshold", [
        ("TIER_1", 10.0),
        ("TIER_2", 5.0),
        ("TIER_3", 2.5),
    ])
    def test_ets_uncertainty_thresholds_by_tier(
        self,
        engine: ComplianceCheckerEngine,
        tier: str,
        threshold: float,
    ):
        """Each tier has the correct uncertainty threshold."""
        # At threshold -> pass
        data_pass = {"tier": tier, "uncertainty_pct": threshold}
        result_pass = engine.check_framework(data_pass, "EU_ETS_MRR")
        ut_pass = next(
            r for r in result_pass.results
            if r.requirement_id == "ets_uncertainty_threshold"
        )
        assert ut_pass.passed is True

        # Above threshold -> fail
        data_fail = {"tier": tier, "uncertainty_pct": threshold + 1}
        result_fail = engine.check_framework(data_fail, "EU_ETS_MRR")
        ut_fail = next(
            r for r in result_fail.results
            if r.requirement_id == "ets_uncertainty_threshold"
        )
        assert ut_fail.passed is False

    def test_ets_factors_validated_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Calculation factors pass with valid EF source."""
        data = {"emission_factor_source": "EU_ETS"}
        result = engine.check_framework(data, "EU_ETS_MRR")
        cf_result = next(
            r for r in result.results
            if r.requirement_id == "ets_calculation_factors"
        )
        assert cf_result.passed is True

    def test_ets_report_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Annual emissions report passes with all required fields."""
        data = {
            "total_co2e_tonnes": 50000.0,
            "period_start": "2025-01-01",
            "period_end": "2025-12-31",
        }
        result = engine.check_framework(data, "EU_ETS_MRR")
        rp_result = next(
            r for r in result.results
            if r.requirement_id == "ets_emissions_report"
        )
        assert rp_result.passed is True

    def test_ets_report_fail(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Annual emissions report fails without required fields."""
        data = {"total_co2e_tonnes": 50000.0}
        result = engine.check_framework(data, "EU_ETS_MRR")
        rp_result = next(
            r for r in result.results
            if r.requirement_id == "ets_emissions_report"
        )
        assert rp_result.passed is False


class TestCheckAllFrameworks:
    """Test checking all 6 frameworks at once."""

    def test_check_all_frameworks_returns_6_results(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """check_all_frameworks returns exactly 6 FrameworkResult objects."""
        results = engine.check_all_frameworks(compliant_cement_data)
        assert len(results) == 6

    def test_check_all_frameworks_all_compliant(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """All frameworks are COMPLIANT with fully compliant data."""
        results = engine.check_all_frameworks(compliant_cement_data)
        for result in results:
            assert result.status == "COMPLIANT", (
                f"{result.framework} is {result.status}, not COMPLIANT"
            )

    def test_check_compliance_single_framework(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """check_compliance with single framework returns dict."""
        results = engine.check_compliance(
            compliant_cement_data, ["GHG_PROTOCOL"],
        )
        assert "GHG_PROTOCOL" in results
        assert results["GHG_PROTOCOL"]["status"] == "COMPLIANT"

    def test_check_compliance_all_by_default(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """check_compliance without frameworks checks all 6."""
        results = engine.check_compliance(compliant_cement_data)
        assert len(results) == 6

    def test_check_compliance_unknown_framework_raises(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """check_compliance raises ValueError for unknown framework."""
        with pytest.raises(ValueError, match="Unknown framework"):
            engine.check_compliance({"process_type": "CEMENT_PRODUCTION"}, ["NONEXISTENT"])

    def test_check_framework_unknown_raises(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """check_framework raises ValueError for unknown framework."""
        with pytest.raises(ValueError, match="Unknown framework"):
            engine.check_framework({}, "INVALID_FW")

    def test_get_framework_requirements(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """get_framework_requirements returns 10 requirements."""
        reqs = engine.get_framework_requirements("GHG_PROTOCOL")
        assert len(reqs) == 10
        assert all(isinstance(r, ComplianceRequirement) for r in reqs)

    def test_get_framework_requirements_unknown_raises(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """get_framework_requirements raises ValueError for unknown."""
        with pytest.raises(ValueError):
            engine.get_framework_requirements("FANTASY")

    def test_framework_result_has_provenance_hash(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Framework result includes SHA-256 provenance hash."""
        result = engine.check_framework(
            compliant_cement_data, "GHG_PROTOCOL",
        )
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_framework_result_has_checked_at(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Framework result includes checked_at timestamp."""
        result = engine.check_framework(
            compliant_cement_data, "GHG_PROTOCOL",
        )
        assert result.checked_at
        assert "T" in result.checked_at  # ISO format

    def test_provenance_determinism(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Same data produces same provenance hash on repeated checks."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_1",
            "total_co2e_tonnes": 50000.0,
        }
        r1 = engine.check_framework(data, "GHG_PROTOCOL")
        r2 = engine.check_framework(data, "GHG_PROTOCOL")
        assert r1.provenance_hash == r2.provenance_hash


class TestDataCompleteness:
    """Test validate_data_completeness method."""

    def test_completeness_pass_full_data(
        self,
        engine: ComplianceCheckerEngine,
        compliant_cement_data: Dict[str, Any],
    ):
        """Data completeness passes with all required fields."""
        result = engine.validate_data_completeness(
            compliant_cement_data, "GHG_PROTOCOL",
        )
        assert result.passed is True

    def test_completeness_fail_minimal_data(
        self,
        engine: ComplianceCheckerEngine,
        minimal_data: Dict[str, Any],
    ):
        """Data completeness fails with empty data."""
        result = engine.validate_data_completeness(
            minimal_data, "GHG_PROTOCOL",
        )
        assert result.passed is False
        assert "missing" in result.details.lower()

    def test_completeness_identifies_missing_fields(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Data completeness lists specific missing fields."""
        data = {"process_type": "CEMENT_PRODUCTION"}
        result = engine.validate_data_completeness(
            data, "EPA_40CFR98",
        )
        # EPA requires production_quantity_tonnes, tier, period_start, etc.
        if not result.passed:
            assert len(result.recommendations) > 0

    def test_completeness_ghg_required_fields(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """GHG Protocol aggregated required fields include key fields."""
        reqs = engine.get_framework_requirements("GHG_PROTOCOL")
        all_fields = set()
        for req in reqs:
            all_fields.update(req.required_fields)
        assert "process_type" in all_fields
        assert "calculation_method" in all_fields

    def test_completeness_unknown_framework_raises(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """validate_data_completeness raises for unknown framework."""
        with pytest.raises(ValueError):
            engine.validate_data_completeness({}, "UNKNOWN")

    def test_validate_methodology_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology validation passes with valid method/tier/source."""
        data = {
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_1",
            "emission_factor_source": "IPCC",
        }
        result = engine.validate_methodology(data, "GHG_PROTOCOL")
        assert result.passed is True

    def test_validate_methodology_invalid_method(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology validation fails with invalid method."""
        data = {"calculation_method": "INVALID"}
        result = engine.validate_methodology(data, "GHG_PROTOCOL")
        assert result.passed is False
        assert "Invalid method" in result.details

    def test_validate_methodology_invalid_tier(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology validation fails with invalid tier."""
        data = {
            "calculation_method": "EMISSION_FACTOR",
            "tier": "TIER_99",
        }
        result = engine.validate_methodology(data, "ISO_14064")
        assert result.passed is False
        assert "Invalid tier" in result.details

    def test_validate_methodology_invalid_source(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Methodology validation fails with invalid EF source."""
        data = {
            "calculation_method": "EMISSION_FACTOR",
            "emission_factor_source": "BOGUS",
        }
        result = engine.validate_methodology(data, "EU_ETS_MRR")
        assert result.passed is False
        assert "Invalid EF source" in result.details


class TestRecommendations:
    """Test generate_recommendations method."""

    def test_recommendations_from_failures(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """generate_recommendations extracts recs from failed checks."""
        results = [
            ComplianceCheckResult(
                requirement_id="req1",
                framework="GHG_PROTOCOL",
                name="Test Req 1",
                passed=False,
                details="Missing field X",
                recommendations=["Add field X"],
                severity="ERROR",
            ),
            ComplianceCheckResult(
                requirement_id="req2",
                framework="GHG_PROTOCOL",
                name="Test Req 2",
                passed=False,
                details="Missing field Y",
                recommendations=["Add field Y"],
                severity="WARNING",
            ),
        ]
        recs = engine.generate_recommendations(results)
        assert len(recs) == 2
        assert "Add field X" in recs
        assert "Add field Y" in recs

    def test_recommendations_no_duplicates(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """generate_recommendations deduplicates identical recommendations."""
        results = [
            ComplianceCheckResult(
                passed=False,
                recommendations=["Enable provenance"],
                severity="ERROR",
            ),
            ComplianceCheckResult(
                passed=False,
                recommendations=["Enable provenance"],
                severity="WARNING",
            ),
        ]
        recs = engine.generate_recommendations(results)
        assert len(recs) == 1

    def test_recommendations_empty_for_passing(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """generate_recommendations returns empty for all passing checks."""
        results = [
            ComplianceCheckResult(passed=True, severity="ERROR"),
            ComplianceCheckResult(passed=True, severity="WARNING"),
        ]
        recs = engine.generate_recommendations(results)
        assert len(recs) == 0

    def test_recommendations_skips_blank(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """generate_recommendations skips empty string recommendations."""
        results = [
            ComplianceCheckResult(
                passed=False,
                recommendations=["", "  ", "Real recommendation"],
                severity="ERROR",
            ),
        ]
        recs = engine.generate_recommendations(results)
        assert len(recs) == 1
        assert "Real recommendation" in recs

    def test_recommendations_sorted(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Recommendations are returned in sorted order."""
        results = [
            ComplianceCheckResult(
                passed=False,
                recommendations=["Zebra recommendation"],
                severity="ERROR",
            ),
            ComplianceCheckResult(
                passed=False,
                recommendations=["Alpha recommendation"],
                severity="ERROR",
            ),
        ]
        recs = engine.generate_recommendations(results)
        assert recs == sorted(recs)

    def test_to_dict_serialization(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Engine serializes to a dict with expected keys."""
        d = engine.to_dict()
        assert d["engine"] == "ComplianceCheckerEngine"
        assert d["total_requirements"] == 60
        assert len(d["frameworks"]) == 6
        assert all(v == 10 for v in d["frameworks"].values())

    def test_epa_subpart_unknown_process(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """get_applicable_epa_subparts returns empty for unknown process."""
        subparts = engine.get_applicable_epa_subparts("UNKNOWN_PROCESS")
        assert subparts == []


class TestComplianceStatus:
    """Test compliance status determination logic."""

    def test_status_compliant_all_pass(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """COMPLIANT status when all 10 requirements pass."""
        assert engine._determine_status(10, 10) == "COMPLIANT"

    def test_status_partial_above_50_pct(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """PARTIAL status when 50-99% of requirements pass."""
        assert engine._determine_status(6, 10) == "PARTIAL"
        assert engine._determine_status(9, 10) == "PARTIAL"

    def test_status_non_compliant_below_50_pct(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """NON_COMPLIANT status when fewer than 50% pass."""
        assert engine._determine_status(4, 10) == "NON_COMPLIANT"
        assert engine._determine_status(0, 10) == "NON_COMPLIANT"

    def test_status_exactly_50_pct(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Exactly 50% of requirements passing gives PARTIAL."""
        assert engine._determine_status(5, 10) == "PARTIAL"

    def test_status_zero_total(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Zero total requirements gives NOT_CHECKED."""
        assert engine._determine_status(0, 0) == "NOT_CHECKED"

    def test_empty_data_non_compliant(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """Empty dict produces NON_COMPLIANT for most frameworks."""
        result = engine.check_framework({}, "GHG_PROTOCOL")
        assert result.status in ("NON_COMPLIANT", "PARTIAL")

    def test_partial_data_partial_compliance(
        self,
        engine: ComplianceCheckerEngine,
        partial_data: Dict[str, Any],
    ):
        """Partial data produces PARTIAL or COMPLIANT status."""
        result = engine.check_framework(partial_data, "GHG_PROTOCOL")
        assert result.status in ("PARTIAL", "COMPLIANT")

    def test_semiconductor_multi_gas_coverage(
        self,
        engine: ComplianceCheckerEngine,
        semiconductor_data: Dict[str, Any],
    ):
        """Semiconductor data with all 5 gases passes gas coverage."""
        result = engine.check_framework(
            semiconductor_data, "GHG_PROTOCOL",
        )
        gc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_gas_coverage"
        )
        assert gc_result.passed is True

    def test_aluminum_multi_gas_coverage(
        self,
        engine: ComplianceCheckerEngine,
        aluminum_data: Dict[str, Any],
    ):
        """Aluminum data with CO2, CF4, C2F6 passes gas coverage."""
        result = engine.check_framework(
            aluminum_data, "GHG_PROTOCOL",
        )
        gc_result = next(
            r for r in result.results
            if r.requirement_id == "ghg_gas_coverage"
        )
        assert gc_result.passed is True

    def test_helper_safe_get_nested(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """_safe_get supports dot-notation for nested keys."""
        data = {"level1": {"level2": "value"}}
        assert engine._safe_get(data, "level1.level2") == "value"

    def test_helper_safe_get_missing(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """_safe_get returns default for missing keys."""
        data = {"key1": "val1"}
        assert engine._safe_get(data, "missing_key", "default") == "default"

    def test_helper_has_fields_all_present(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """_has_fields returns True when all fields present."""
        data = {"a": 1, "b": 2, "c": 3}
        present, missing = engine._has_fields(data, ["a", "b", "c"])
        assert present is True
        assert missing == []

    def test_helper_has_fields_some_missing(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """_has_fields returns False and lists missing fields."""
        data = {"a": 1}
        present, missing = engine._has_fields(data, ["a", "b", "c"])
        assert present is False
        assert "b" in missing
        assert "c" in missing

    def test_helper_has_gas_coverage_cement(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """_has_gas_coverage for cement (CO2 only)."""
        data = {
            "process_type": "CEMENT_PRODUCTION",
            "emissions_by_gas": [{"gas": "CO2"}],
        }
        complete, reported, missing = engine._has_gas_coverage(data)
        assert complete is True
        assert "CO2" in reported
        assert missing == []

    def test_helper_hash_deterministic(
        self,
        engine: ComplianceCheckerEngine,
    ):
        """_hash produces deterministic SHA-256 hex digest."""
        h1 = engine._hash("test data")
        h2 = engine._hash("test data")
        assert h1 == h2
        assert len(h1) == 64
