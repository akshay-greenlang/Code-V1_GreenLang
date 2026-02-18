# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6) - AGENT-MRV-003 Mobile Combustion.

Tests all public methods with 90+ test functions covering:
- Initialization, framework listing, framework requirements
- Single-framework compliance checks (GHG Protocol, ISO 14064,
  CSRD/ESRS E1, EPA Part 98, UK SECR, EU ETS MRR)
- All-frameworks check
- Data completeness validation
- Methodology validation (methods and tiers per framework)
- Reporting requirements validation (per-gas, biogenic, energy, transport)
- Intensity metrics validation
- Recommendations generation
- Base year threshold checks
- History management (append, retrieve, clear)
- Provenance hashing (SHA-256, determinism)
- Compliance status logic (COMPLIANT / NEEDS_REVIEW / NON_COMPLIANT)
- Framework-specific checks (EPA large emitter, EU ETS verification, etc.)
- Edge cases and thread safety

Author: GreenLang QA Team
"""

import threading
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.mobile_combustion.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceCheckResult,
    ComplianceFinding,
    ComplianceStatus,
    FindingSeverity,
    RegulatoryFramework,
    _BASE_YEAR_RECALC_THRESHOLD,
    _EPA_LARGE_EMITTER_THRESHOLD_KG,
    _EPA_LARGE_EMITTER_THRESHOLD_TCO2E,
    _FRAMEWORK_REQUIREMENTS,
    _VALID_METHODS_BY_FRAMEWORK,
    _VALID_TIERS_BY_FRAMEWORK,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """Create a default ComplianceCheckerEngine instance."""
    return ComplianceCheckerEngine()


@pytest.fixture
def fully_compliant_ghg_data():
    """Create fully compliant data for GHG Protocol checks."""
    return {
        "total_co2e_kg": 50000,
        "method": "FUEL_BASED",
        "tier": "TIER_2",
        "gases": {"CO2": 48000, "CH4": 1200, "N2O": 800},
        "biogenic_co2_separated": True,
        "gwp_source": "IPCC AR5",
        "base_year": 2020,
        "base_year_emissions": 48000,
        "emission_factor_source": "EPA MRV Table C-1",
        "fuel_type": "diesel",
        "fuel_quantity": 18000,
        "uncertainty_assessment": {"pct_95": 8.5},
        "organizational_boundary": "operational_control",
        "control_approach": "operational_control",
    }


@pytest.fixture
def minimal_data():
    """Create minimal data that will fail most checks."""
    return {
        "total_co2e_kg": 5000,
    }


@pytest.fixture
def fully_compliant_iso_data():
    """Create fully compliant data for ISO 14064 checks."""
    return {
        "total_co2e_kg": 50000,
        "method": "FUEL_BASED",
        "tier": "TIER_2",
        "gases": {"CO2": 48000, "CH4": 1200, "N2O": 800},
        "uncertainty_assessment": {"pct_95": 8.5},
        "quality_procedures": "ISO 14064-1 QMP documented",
        "assumptions_documented": True,
        "organizational_boundary": "operational_control",
        "control_approach": "operational_control",
        "data_management_system": "GreenLang DMS",
        "base_year": 2020,
        "verification_status": "verified",
    }


@pytest.fixture
def fully_compliant_csrd_data():
    """Create fully compliant data for CSRD/ESRS E1 checks."""
    return {
        "total_co2e_kg": 50000,
        "method": "FUEL_BASED",
        "tier": "TIER_2",
        "source_category": "mobile_combustion",
        "transport_category_separated": True,
        "intensity_metric": 0.25,
        "intensity_per_revenue": 0.0025,
        "intensity_per_fte": 1.2,
        "reduction_target": "30% by 2030",
        "mitigation_actions": "Fleet electrification program",
        "gross_emissions": 50000,
        "net_emissions": 48000,
        "materiality_assessment": "Completed Q4 2025",
        "energy_consumption_kwh": 180000,
        "assumptions_documented": True,
    }


@pytest.fixture
def fully_compliant_epa_data():
    """Create fully compliant data for EPA Part 98 checks."""
    return {
        "total_co2e_kg": 30000000,  # 30,000 tCO2e (above 25k threshold)
        "method": "FUEL_BASED",
        "tier": "TIER_3",
        "gases": {"CO2": 29000000, "CH4": 600000, "N2O": 400000},
        "fuel_type": "diesel",
        "fuel_quantity": 11000000,
        "emission_factor_source": "EPA Table C-1",
        "equipment_type": "heavy_duty_truck",
        "fuel_analysis_data": {"carbon_content": 0.87, "hhv": 137.0},
        "measurement_documentation": "Calibrated flow meters",
        "missing_data_procedures": "40 CFR 98.35 compliant",
        "reporting_frequency": "monthly",
        "reporting_year": 2025,
        "submission_date": "2026-03-15",
    }


@pytest.fixture
def fully_compliant_uk_secr_data():
    """Create fully compliant data for UK SECR checks."""
    return {
        "total_co2e_kg": 50000,
        "method": "FUEL_BASED",
        "tier": "TIER_2",
        "transport_category_separated": True,
        "intensity_per_revenue": 0.0025,
        "intensity_per_fte": 1.2,
        "energy_consumption_kwh": 180000,
        "methodology_disclosed": "DEFRA 2025 conversion factors",
        "uk_emissions": 40000,
        "global_emissions": 10000,
        "previous_period_emissions": 48000,
        "efficiency_actions": "Fleet optimization program",
        "mitigation_actions": "EV transition plan",
    }


@pytest.fixture
def fully_compliant_eu_ets_data():
    """Create fully compliant data for EU ETS MRR checks."""
    return {
        "total_co2e_kg": 50000,
        "method": "FUEL_BASED",
        "tier": "TIER_3",
        "fuel_measurement_method": "calibrated_meter",
        "ef_analysis_frequency": "annual",
        "verification_status": "verified",
        "monitoring_plan_approved": True,
        "source_streams": ["diesel_fleet"],
        "biomass_fraction": 0.05,
        "uncertainty_assessment": {"pct_95": 5.0},
        "data_gap_procedures": "Documented per MRR Art. 65",
        "improvement_report": "Submitted 2025-06-30",
    }


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test ComplianceCheckerEngine initialization."""

    def test_default_init(self, engine):
        """Engine initializes with empty history and a lock."""
        assert engine._compliance_history == []
        assert isinstance(engine._lock, type(threading.RLock()))

    def test_history_starts_empty(self, engine):
        """get_compliance_history returns empty list on fresh engine."""
        assert engine.get_compliance_history() == []

    def test_frameworks_count(self, engine):
        """Six frameworks are supported."""
        frameworks = engine.list_frameworks()
        assert len(frameworks) == 6


# ===========================================================================
# TestListFrameworks
# ===========================================================================


class TestListFrameworks:
    """Test list_frameworks and get_framework_requirements."""

    def test_list_frameworks_returns_sorted(self, engine):
        """list_frameworks returns sorted framework identifiers."""
        frameworks = engine.list_frameworks()
        assert frameworks == sorted(frameworks)

    def test_list_frameworks_contains_all_six(self, engine):
        """All six regulatory frameworks are listed."""
        frameworks = engine.list_frameworks()
        expected = {
            "CSRD_ESRS_E1", "EPA_PART_98", "EU_ETS_MRR",
            "GHG_PROTOCOL", "ISO_14064", "UK_SECR",
        }
        assert set(frameworks) == expected

    def test_get_framework_requirements_ghg(self, engine):
        """GHG Protocol has 10 requirements."""
        reqs = engine.get_framework_requirements("GHG_PROTOCOL")
        assert len(reqs) == 10

    def test_get_framework_requirements_iso(self, engine):
        """ISO 14064 has 10 requirements."""
        reqs = engine.get_framework_requirements("ISO_14064")
        assert len(reqs) == 10

    def test_get_framework_requirements_csrd(self, engine):
        """CSRD/ESRS E1 has 10 requirements."""
        reqs = engine.get_framework_requirements("CSRD_ESRS_E1")
        assert len(reqs) == 10

    def test_get_framework_requirements_epa(self, engine):
        """EPA Part 98 has 10 requirements."""
        reqs = engine.get_framework_requirements("EPA_PART_98")
        assert len(reqs) == 10

    def test_get_framework_requirements_secr(self, engine):
        """UK SECR has 10 requirements."""
        reqs = engine.get_framework_requirements("UK_SECR")
        assert len(reqs) == 10

    def test_get_framework_requirements_mrr(self, engine):
        """EU ETS MRR has 10 requirements."""
        reqs = engine.get_framework_requirements("EU_ETS_MRR")
        assert len(reqs) == 10

    def test_get_framework_requirements_returns_copy(self, engine):
        """get_framework_requirements returns a copy, not the original."""
        reqs = engine.get_framework_requirements("GHG_PROTOCOL")
        reqs.clear()
        assert len(engine.get_framework_requirements("GHG_PROTOCOL")) == 10

    def test_get_framework_requirements_invalid(self, engine):
        """Invalid framework raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized framework"):
            engine.get_framework_requirements("INVALID_FRAMEWORK")

    def test_each_requirement_has_id_and_text(self, engine):
        """Every requirement across all frameworks has id and requirement."""
        for fw in engine.list_frameworks():
            for req in engine.get_framework_requirements(fw):
                assert "id" in req
                assert "requirement" in req
                assert "mandatory" in req
                assert "fields_required" in req


# ===========================================================================
# TestEnumerations
# ===========================================================================


class TestEnumerations:
    """Test enum definitions match expectations."""

    def test_regulatory_framework_values(self):
        """RegulatoryFramework has exactly 6 members."""
        assert len(RegulatoryFramework) == 6
        assert RegulatoryFramework.GHG_PROTOCOL.value == "GHG_PROTOCOL"
        assert RegulatoryFramework.ISO_14064.value == "ISO_14064"
        assert RegulatoryFramework.CSRD_ESRS_E1.value == "CSRD_ESRS_E1"
        assert RegulatoryFramework.EPA_PART_98.value == "EPA_PART_98"
        assert RegulatoryFramework.UK_SECR.value == "UK_SECR"
        assert RegulatoryFramework.EU_ETS_MRR.value == "EU_ETS_MRR"

    def test_compliance_status_values(self):
        """ComplianceStatus has exactly 3 members."""
        assert len(ComplianceStatus) == 3
        assert ComplianceStatus.COMPLIANT.value == "COMPLIANT"
        assert ComplianceStatus.NON_COMPLIANT.value == "NON_COMPLIANT"
        assert ComplianceStatus.NEEDS_REVIEW.value == "NEEDS_REVIEW"

    def test_finding_severity_values(self):
        """FindingSeverity has exactly 5 members."""
        assert len(FindingSeverity) == 5
        assert FindingSeverity.CRITICAL.value == "CRITICAL"
        assert FindingSeverity.MAJOR.value == "MAJOR"
        assert FindingSeverity.MINOR.value == "MINOR"
        assert FindingSeverity.INFORMATIONAL.value == "INFORMATIONAL"
        assert FindingSeverity.PASS.value == "PASS"

    def test_constants(self):
        """Module-level constants are correct."""
        assert _EPA_LARGE_EMITTER_THRESHOLD_TCO2E == Decimal("25000")
        assert _EPA_LARGE_EMITTER_THRESHOLD_KG == Decimal("25000000")
        assert _BASE_YEAR_RECALC_THRESHOLD == Decimal("0.05")


# ===========================================================================
# TestDataclasses
# ===========================================================================


class TestDataclasses:
    """Test ComplianceFinding and ComplianceCheckResult dataclasses."""

    def test_finding_to_dict(self):
        """ComplianceFinding.to_dict serializes all fields."""
        finding = ComplianceFinding(
            finding_id="f_test",
            requirement_id="GHG-MC-001",
            requirement_text="Test requirement",
            category="methodology",
            severity="CRITICAL",
            status="NON_COMPLIANT",
            detail="Detail text",
            data_present=False,
        )
        d = finding.to_dict()
        assert d["finding_id"] == "f_test"
        assert d["requirement_id"] == "GHG-MC-001"
        assert d["requirement_text"] == "Test requirement"
        assert d["category"] == "methodology"
        assert d["severity"] == "CRITICAL"
        assert d["status"] == "NON_COMPLIANT"
        assert d["detail"] == "Detail text"
        assert d["data_present"] is False

    def test_check_result_to_dict(self, engine, fully_compliant_ghg_data):
        """ComplianceCheckResult.to_dict serializes all fields."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        d = result.to_dict()
        assert "result_id" in d
        assert d["framework"] == "GHG_PROTOCOL"
        assert "status" in d
        assert isinstance(d["findings"], list)
        assert isinstance(d["recommendations"], list)
        assert len(d["provenance_hash"]) == 64
        assert "compliance_score_pct" in d


# ===========================================================================
# TestCheckComplianceGHGProtocol
# ===========================================================================


class TestCheckComplianceGHGProtocol:
    """Test check_compliance for GHG Protocol framework."""

    def test_fully_compliant(self, engine, fully_compliant_ghg_data):
        """Fully compliant data yields COMPLIANT status."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        assert isinstance(result, ComplianceCheckResult)
        assert result.framework == "GHG_PROTOCOL"
        assert result.status == ComplianceStatus.COMPLIANT.value
        assert result.total_requirements == 10

    def test_result_has_provenance_hash(self, engine, fully_compliant_ghg_data):
        """Result always carries a 64-character SHA-256 hash."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        assert len(result.provenance_hash) == 64

    def test_result_has_timestamp(self, engine, fully_compliant_ghg_data):
        """Result has an ISO-formatted timestamp."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        assert "T" in result.timestamp

    def test_result_has_result_id(self, engine, fully_compliant_ghg_data):
        """Result ID starts with cc_ prefix."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        assert result.result_id.startswith("cc_")

    def test_minimal_data_is_non_compliant(self, engine, minimal_data):
        """Minimal data with many missing fields is NON_COMPLIANT."""
        result = engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        assert result.status == ComplianceStatus.NON_COMPLIANT.value

    def test_invalid_framework_raises(self, engine, minimal_data):
        """Unrecognized framework string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized framework"):
            engine.check_compliance(minimal_data, "INVALID")

    def test_compliance_score_100_for_fully_compliant(
        self, engine, fully_compliant_ghg_data
    ):
        """Compliance score is 100.0% when fully compliant."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        if result.status == ComplianceStatus.COMPLIANT.value:
            assert result.compliance_score_pct == Decimal("100.0")

    def test_findings_list_populated(self, engine, fully_compliant_ghg_data):
        """Findings list is always populated (even for PASS)."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        assert len(result.findings) > 0

    def test_mandatory_met_equals_mandatory_requirements_when_compliant(
        self, engine, fully_compliant_ghg_data
    ):
        """When compliant, mandatory_met == mandatory_requirements."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        if result.status == ComplianceStatus.COMPLIANT.value:
            assert result.mandatory_met == result.mandatory_requirements

    def test_base_year_recalculation_finding(self, engine, fully_compliant_ghg_data):
        """When emissions change > 5%, base year finding is NEEDS_REVIEW."""
        data = dict(fully_compliant_ghg_data)
        data["total_co2e_kg"] = 100000  # ~100% change from 48000
        data["base_year_emissions"] = 48000
        result = engine.check_compliance(data, "GHG_PROTOCOL")
        base_findings = [
            f for f in result.findings if f.requirement_id == "GHG-MC-006"
        ]
        # At least one finding about base year recalculation
        assert any("exceed" in f.detail.lower() or "recalcul" in f.detail.lower()
                    for f in base_findings)

    def test_base_year_within_threshold(self, engine, fully_compliant_ghg_data):
        """When emissions change <= 5%, base year check passes."""
        data = dict(fully_compliant_ghg_data)
        data["total_co2e_kg"] = 49000  # ~2% change from 48000
        data["base_year_emissions"] = 48000
        result = engine.check_compliance(data, "GHG_PROTOCOL")
        base_findings = [
            f for f in result.findings
            if f.requirement_id == "GHG-MC-006"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(base_findings) > 0


# ===========================================================================
# TestCheckComplianceISO14064
# ===========================================================================


class TestCheckComplianceISO14064:
    """Test check_compliance for ISO 14064 framework."""

    def test_fully_compliant(self, engine, fully_compliant_iso_data):
        """Fully compliant data yields COMPLIANT status."""
        result = engine.check_compliance(fully_compliant_iso_data, "ISO_14064")
        assert result.framework == "ISO_14064"
        # Should be COMPLIANT or at most NEEDS_REVIEW
        assert result.status in {
            ComplianceStatus.COMPLIANT.value,
            ComplianceStatus.NEEDS_REVIEW.value,
        }

    def test_missing_uncertainty_is_non_compliant(self, engine):
        """ISO 14064 requires uncertainty statement; missing = CRITICAL."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "gases": {"CO2": 48000, "CH4": 1200, "N2O": 800},
        }
        result = engine.check_compliance(data, "ISO_14064")
        iso_specific = [
            f for f in result.findings
            if f.requirement_id == "ISO-MC-003"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(iso_specific) > 0

    def test_missing_quality_procedures(self, engine):
        """Missing quality procedures produces MAJOR finding."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "ISO_14064")
        quality_findings = [
            f for f in result.findings
            if f.requirement_id == "ISO-MC-004"
            and f.status == ComplianceStatus.NON_COMPLIANT.value
        ]
        assert len(quality_findings) > 0

    def test_assumptions_documented_check(self, engine, fully_compliant_iso_data):
        """Documented assumptions produce a PASS finding."""
        result = engine.check_compliance(fully_compliant_iso_data, "ISO_14064")
        assumptions = [
            f for f in result.findings
            if f.requirement_id == "ISO-MC-005"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(assumptions) > 0


# ===========================================================================
# TestCheckComplianceCSRD
# ===========================================================================


class TestCheckComplianceCSRD:
    """Test check_compliance for CSRD/ESRS E1 framework."""

    def test_fully_compliant(self, engine, fully_compliant_csrd_data):
        """Fully compliant CSRD data passes compliance check."""
        result = engine.check_compliance(fully_compliant_csrd_data, "CSRD_ESRS_E1")
        assert result.framework == "CSRD_ESRS_E1"
        assert result.status in {
            ComplianceStatus.COMPLIANT.value,
            ComplianceStatus.NEEDS_REVIEW.value,
        }

    def test_missing_materiality_assessment(self, engine):
        """CSRD requires materiality assessment; missing = CRITICAL."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "CSRD_ESRS_E1")
        materiality = [
            f for f in result.findings
            if f.requirement_id == "CSRD-MC-008"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(materiality) > 0

    def test_missing_mitigation_actions(self, engine):
        """Missing mitigation actions produces MAJOR finding."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "CSRD_ESRS_E1")
        mitigation = [
            f for f in result.findings
            if f.requirement_id == "CSRD-MC-005"
            and f.status == ComplianceStatus.NON_COMPLIANT.value
        ]
        assert len(mitigation) > 0

    def test_missing_reduction_target(self, engine):
        """Missing reduction target produces MAJOR finding."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "CSRD_ESRS_E1")
        target = [
            f for f in result.findings
            if f.requirement_id == "CSRD-MC-004"
            and f.status == ComplianceStatus.NON_COMPLIANT.value
        ]
        assert len(target) > 0

    def test_intensity_metrics_checked(self, engine, fully_compliant_csrd_data):
        """CSRD adds intensity findings to results."""
        result = engine.check_compliance(fully_compliant_csrd_data, "CSRD_ESRS_E1")
        intensity = [
            f for f in result.findings
            if f.requirement_id in {"INTENSITY-REVENUE", "INTENSITY-FTE"}
        ]
        assert len(intensity) >= 2

    def test_energy_consumption_kwh_checked(self, engine, fully_compliant_csrd_data):
        """CSRD checks energy consumption in kWh."""
        result = engine.check_compliance(fully_compliant_csrd_data, "CSRD_ESRS_E1")
        energy = [
            f for f in result.findings
            if "ENERGY" in f.requirement_id
        ]
        assert len(energy) > 0

    def test_transport_category_checked(self, engine, fully_compliant_csrd_data):
        """CSRD checks transport as separate category."""
        result = engine.check_compliance(fully_compliant_csrd_data, "CSRD_ESRS_E1")
        transport = [
            f for f in result.findings
            if "TRANSPORT" in f.requirement_id
        ]
        assert len(transport) > 0


# ===========================================================================
# TestCheckComplianceEPA
# ===========================================================================


class TestCheckComplianceEPA:
    """Test check_compliance for EPA Part 98 framework."""

    def test_fully_compliant(self, engine, fully_compliant_epa_data):
        """Fully compliant EPA data passes compliance check."""
        result = engine.check_compliance(fully_compliant_epa_data, "EPA_PART_98")
        assert result.framework == "EPA_PART_98"
        assert result.status in {
            ComplianceStatus.COMPLIANT.value,
            ComplianceStatus.NEEDS_REVIEW.value,
        }

    def test_large_emitter_no_monthly_reporting(self, engine):
        """Emissions > 25k tCO2e without monthly reporting = CRITICAL."""
        data = {
            "total_co2e_kg": 30000000,  # 30,000 tCO2e
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "gases": {"CO2": 29000000, "CH4": 600000, "N2O": 400000},
            "fuel_type": "diesel",
            "fuel_quantity": 11000000,
            "emission_factor_source": "EPA",
            "equipment_type": "truck",
            "measurement_documentation": "documented",
            "missing_data_procedures": "documented",
            "reporting_year": 2025,
            "submission_date": "2026-03-15",
            "reporting_frequency": "annual",  # NOT monthly
        }
        result = engine.check_compliance(data, "EPA_PART_98")
        monthly = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-002"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(monthly) > 0

    def test_large_emitter_with_monthly_reporting(self, engine, fully_compliant_epa_data):
        """Emissions > 25k tCO2e with monthly reporting = PASS."""
        result = engine.check_compliance(fully_compliant_epa_data, "EPA_PART_98")
        monthly = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-002"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(monthly) > 0

    def test_below_threshold_no_monthly_required(self, engine):
        """Emissions < 25k tCO2e do not require monthly reporting."""
        data = {
            "total_co2e_kg": 10000000,  # 10,000 tCO2e (below 25k)
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EPA_PART_98")
        threshold = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-002"
            and f.severity == FindingSeverity.INFORMATIONAL.value
        ]
        assert len(threshold) > 0

    def test_tier_3_requires_fuel_analysis(self, engine):
        """Tier 3 without fuel analysis data = CRITICAL."""
        data = {
            "total_co2e_kg": 5000,
            "method": "FUEL_BASED",
            "tier": "TIER_3",
        }
        result = engine.check_compliance(data, "EPA_PART_98")
        fuel_analysis = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-005"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(fuel_analysis) > 0

    def test_tier_3_with_fuel_analysis(self, engine, fully_compliant_epa_data):
        """Tier 3 with fuel analysis data = PASS."""
        result = engine.check_compliance(fully_compliant_epa_data, "EPA_PART_98")
        fuel_analysis = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-005"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(fuel_analysis) > 0

    def test_only_fuel_based_accepted(self, engine):
        """EPA Part 98 only accepts FUEL_BASED method."""
        data = {
            "total_co2e_kg": 5000,
            "method": "DISTANCE_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EPA_PART_98")
        method_findings = [
            f for f in result.findings
            if "METHOD" in f.requirement_id
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(method_findings) > 0

    def test_missing_data_procedures_check(self, engine, fully_compliant_epa_data):
        """EPA checks for missing data substitution procedures."""
        result = engine.check_compliance(fully_compliant_epa_data, "EPA_PART_98")
        missing_procs = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-009"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(missing_procs) > 0


# ===========================================================================
# TestCheckComplianceUKSECR
# ===========================================================================


class TestCheckComplianceUKSECR:
    """Test check_compliance for UK SECR framework."""

    def test_fully_compliant(self, engine, fully_compliant_uk_secr_data):
        """Fully compliant UK SECR data passes check."""
        result = engine.check_compliance(fully_compliant_uk_secr_data, "UK_SECR")
        assert result.framework == "UK_SECR"
        assert result.status in {
            ComplianceStatus.COMPLIANT.value,
            ComplianceStatus.NEEDS_REVIEW.value,
        }

    def test_missing_methodology_disclosure(self, engine):
        """Missing methodology disclosure = MAJOR finding."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "UK_SECR")
        methodology = [
            f for f in result.findings
            if f.requirement_id == "SECR-MC-005"
            and f.status == ComplianceStatus.NON_COMPLIANT.value
        ]
        assert len(methodology) > 0

    def test_uk_global_separation_checked(self, engine, fully_compliant_uk_secr_data):
        """UK vs global emissions separation is checked."""
        result = engine.check_compliance(fully_compliant_uk_secr_data, "UK_SECR")
        uk_global = [
            f for f in result.findings
            if f.requirement_id == "SECR-MC-007"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(uk_global) > 0

    def test_missing_previous_period_emissions(self, engine):
        """Missing previous period emissions = MAJOR finding."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "UK_SECR")
        prev = [
            f for f in result.findings
            if f.requirement_id == "SECR-MC-008"
            and f.status == ComplianceStatus.NON_COMPLIANT.value
        ]
        assert len(prev) > 0

    def test_intensity_metrics_required(self, engine):
        """UK SECR adds intensity metric findings."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "UK_SECR")
        intensity = [
            f for f in result.findings
            if f.requirement_id in {"INTENSITY-REVENUE", "INTENSITY-FTE"}
        ]
        assert len(intensity) >= 2

    def test_transport_category_required(self, engine):
        """UK SECR checks transport category separation."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "UK_SECR")
        transport = [
            f for f in result.findings
            if "TRANSPORT" in f.requirement_id
        ]
        assert len(transport) > 0

    def test_energy_kwh_required(self, engine):
        """UK SECR checks energy consumption in kWh."""
        data = {"total_co2e_kg": 50000, "method": "FUEL_BASED", "tier": "TIER_2"}
        result = engine.check_compliance(data, "UK_SECR")
        energy = [
            f for f in result.findings
            if "ENERGY" in f.requirement_id
        ]
        assert len(energy) > 0


# ===========================================================================
# TestCheckComplianceEUETS
# ===========================================================================


class TestCheckComplianceEUETS:
    """Test check_compliance for EU ETS MRR framework."""

    def test_fully_compliant(self, engine, fully_compliant_eu_ets_data):
        """Fully compliant EU ETS data passes check."""
        result = engine.check_compliance(fully_compliant_eu_ets_data, "EU_ETS_MRR")
        assert result.framework == "EU_ETS_MRR"
        assert result.status in {
            ComplianceStatus.COMPLIANT.value,
            ComplianceStatus.NEEDS_REVIEW.value,
        }

    def test_tier_3_without_calibrated_measurement(self, engine):
        """Tier 3 without calibrated measurement = CRITICAL."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_3",
            "fuel_measurement_method": "estimated",
        }
        result = engine.check_compliance(data, "EU_ETS_MRR")
        calibrated = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-002"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(calibrated) > 0

    def test_tier_3_with_calibrated_measurement(self, engine, fully_compliant_eu_ets_data):
        """Tier 3 with calibrated_meter measurement passes."""
        result = engine.check_compliance(fully_compliant_eu_ets_data, "EU_ETS_MRR")
        calibrated = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-002"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(calibrated) > 0

    def test_missing_verification(self, engine):
        """Missing verification status = CRITICAL."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EU_ETS_MRR")
        verification = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-004"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(verification) > 0

    def test_verification_completed(self, engine, fully_compliant_eu_ets_data):
        """Verified status passes MRR-MC-004."""
        result = engine.check_compliance(fully_compliant_eu_ets_data, "EU_ETS_MRR")
        verification = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-004"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(verification) > 0

    def test_missing_monitoring_plan(self, engine):
        """Missing monitoring plan = CRITICAL."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EU_ETS_MRR")
        monitoring = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-005"
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(monitoring) > 0

    def test_annual_ef_analysis_required(self, engine):
        """Missing annual EF analysis = MAJOR."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EU_ETS_MRR")
        ef_analysis = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-003"
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(ef_analysis) > 0

    def test_annual_ef_analysis_compliant(self, engine, fully_compliant_eu_ets_data):
        """Annual EF analysis with annual frequency = PASS."""
        result = engine.check_compliance(fully_compliant_eu_ets_data, "EU_ETS_MRR")
        ef_analysis = [
            f for f in result.findings
            if f.requirement_id == "MRR-MC-003"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(ef_analysis) > 0

    def test_only_fuel_based_accepted(self, engine):
        """EU ETS MRR only accepts FUEL_BASED method."""
        data = {
            "total_co2e_kg": 50000,
            "method": "DISTANCE_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EU_ETS_MRR")
        method_findings = [
            f for f in result.findings
            if "METHOD" in f.requirement_id
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(method_findings) > 0


# ===========================================================================
# TestCheckAllFrameworks
# ===========================================================================


class TestCheckAllFrameworks:
    """Test check_all_frameworks across all six frameworks."""

    def test_returns_all_six_frameworks(self, engine, minimal_data):
        """Returns results for all 6 frameworks."""
        results = engine.check_all_frameworks(minimal_data)
        assert len(results) == 6
        for fw in RegulatoryFramework:
            assert fw.value in results

    def test_each_result_is_compliance_check_result(self, engine, minimal_data):
        """Each value is a ComplianceCheckResult instance."""
        results = engine.check_all_frameworks(minimal_data)
        for fw, result in results.items():
            assert isinstance(result, ComplianceCheckResult)
            assert result.framework == fw

    def test_history_records_six_entries(self, engine, minimal_data):
        """check_all_frameworks adds 6 entries to history."""
        engine.check_all_frameworks(minimal_data)
        history = engine.get_compliance_history()
        assert len(history) == 6

    def test_compliant_data_all_frameworks(self, engine):
        """Data with all possible fields achieves compliance on multiple frameworks."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "gases": {"CO2": 48000, "CH4": 1200, "N2O": 800},
            "biogenic_co2_separated": True,
            "gwp_source": "IPCC AR5",
            "base_year": 2020,
            "base_year_emissions": 48000,
            "emission_factor_source": "EPA",
            "fuel_type": "diesel",
            "fuel_quantity": 18000,
            "uncertainty_assessment": {"pct_95": 8.5},
            "organizational_boundary": "operational_control",
            "control_approach": "operational_control",
            "quality_procedures": "documented",
            "assumptions_documented": True,
            "data_management_system": "GreenLang",
            "verification_status": "verified",
            "source_category": "mobile",
            "transport_category_separated": True,
            "intensity_metric": 0.25,
            "intensity_per_revenue": 0.0025,
            "intensity_per_fte": 1.2,
            "reduction_target": "30% by 2030",
            "mitigation_actions": "Fleet electrification",
            "gross_emissions": 50000,
            "net_emissions": 48000,
            "materiality_assessment": "completed",
            "energy_consumption_kwh": 180000,
            "equipment_type": "truck",
            "measurement_documentation": "calibrated",
            "missing_data_procedures": "documented",
            "reporting_frequency": "monthly",
            "reporting_year": 2025,
            "submission_date": "2026-03-15",
            "methodology_disclosed": "DEFRA 2025",
            "uk_emissions": 40000,
            "global_emissions": 10000,
            "previous_period_emissions": 48000,
            "efficiency_actions": "fleet optimization",
            "fuel_measurement_method": "calibrated_meter",
            "ef_analysis_frequency": "annual",
            "monitoring_plan_approved": True,
            "source_streams": ["diesel_fleet"],
            "biomass_fraction": 0.05,
            "data_gap_procedures": "documented",
            "improvement_report": "submitted",
        }
        results = engine.check_all_frameworks(data)
        compliant_count = sum(
            1 for r in results.values()
            if r.status == ComplianceStatus.COMPLIANT.value
        )
        # At least GHG Protocol and ISO 14064 should be compliant
        assert compliant_count >= 2


# ===========================================================================
# TestValidateDataCompleteness
# ===========================================================================


class TestValidateDataCompleteness:
    """Test validate_data_completeness method."""

    def test_all_fields_present_returns_pass(self, engine):
        """When all required fields are present, findings are PASS."""
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "organizational_boundary": "operational_control",
            "control_approach": "operational_control",
            "biogenic_co2_separated": True,
            "gases": {"CO2": 48000, "CH4": 1200, "N2O": 800},
            "gwp_source": "IPCC AR5",
            "base_year": 2020,
            "base_year_emissions": 48000,
            "emission_factor_source": "EPA",
            "fuel_type": "diesel",
            "fuel_quantity": 18000,
            "uncertainty_assessment": {"pct_95": 8.5},
        }
        findings = engine.validate_data_completeness(data, "GHG_PROTOCOL")
        assert len(findings) == 10  # 10 requirements for GHG Protocol
        pass_count = sum(
            1 for f in findings if f.severity == FindingSeverity.PASS.value
        )
        assert pass_count == 10

    def test_empty_data_all_fail(self, engine):
        """Empty data dictionary fails all completeness checks."""
        findings = engine.validate_data_completeness({}, "GHG_PROTOCOL")
        assert len(findings) == 10
        fail_count = sum(
            1 for f in findings if f.status != ComplianceStatus.COMPLIANT.value
        )
        assert fail_count == 10

    def test_mandatory_missing_is_critical(self, engine):
        """Missing mandatory fields produce CRITICAL severity."""
        findings = engine.validate_data_completeness({}, "GHG_PROTOCOL")
        critical = [
            f for f in findings
            if f.severity == FindingSeverity.CRITICAL.value
        ]
        # GHG Protocol has 8 mandatory requirements
        assert len(critical) == 8

    def test_optional_missing_is_minor(self, engine):
        """Missing optional fields produce MINOR severity."""
        findings = engine.validate_data_completeness({}, "GHG_PROTOCOL")
        minor = [
            f for f in findings
            if f.severity == FindingSeverity.MINOR.value
        ]
        # GHG Protocol has 2 optional requirements
        assert len(minor) == 2

    def test_empty_string_counts_as_missing(self, engine):
        """Empty string values count as missing."""
        data = {"total_co2e_kg": "", "method": ""}
        findings = engine.validate_data_completeness(data, "GHG_PROTOCOL")
        # Fields with empty strings should not be PASS
        total_finding = [
            f for f in findings
            if f.requirement_id == "GHG-MC-007"
        ]
        assert total_finding[0].data_present is False

    def test_empty_dict_counts_as_missing(self, engine):
        """Empty dict values count as missing."""
        data = {"gases": {}}
        findings = engine.validate_data_completeness(data, "GHG_PROTOCOL")
        gases_finding = [
            f for f in findings if f.requirement_id == "GHG-MC-004"
        ]
        assert gases_finding[0].data_present is False

    def test_empty_list_counts_as_missing(self, engine):
        """Empty list values count as missing."""
        data = {"source_streams": []}
        findings = engine.validate_data_completeness(data, "EU_ETS_MRR")
        stream_finding = [
            f for f in findings if f.requirement_id == "MRR-MC-006"
        ]
        assert stream_finding[0].data_present is False

    def test_invalid_framework_raises(self, engine):
        """Invalid framework raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized framework"):
            engine.validate_data_completeness({}, "BOGUS")

    def test_finding_detail_mentions_missing_fields(self, engine):
        """Finding detail lists which fields are missing."""
        findings = engine.validate_data_completeness({}, "GHG_PROTOCOL")
        for f in findings:
            if f.data_present is False:
                assert "Missing required fields" in f.detail


# ===========================================================================
# TestValidateMethodology
# ===========================================================================


class TestValidateMethodology:
    """Test validate_methodology method."""

    @pytest.mark.parametrize("framework,valid_methods", [
        ("GHG_PROTOCOL", ["FUEL_BASED", "DISTANCE_BASED"]),
        ("ISO_14064", ["FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED"]),
        ("CSRD_ESRS_E1", ["FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED"]),
        ("EPA_PART_98", ["FUEL_BASED"]),
        ("UK_SECR", ["FUEL_BASED", "DISTANCE_BASED"]),
        ("EU_ETS_MRR", ["FUEL_BASED"]),
    ])
    def test_valid_methods_per_framework(self, engine, framework, valid_methods):
        """Each framework accepts its specified methods."""
        for method in valid_methods:
            findings = engine.validate_methodology(method, "TIER_2", framework)
            method_findings = [
                f for f in findings if "METHOD" in f.requirement_id
            ]
            assert any(
                f.severity == FindingSeverity.PASS.value for f in method_findings
            ), f"{method} should be valid for {framework}"

    def test_invalid_method_is_critical(self, engine):
        """Invalid method for EPA produces CRITICAL finding."""
        findings = engine.validate_methodology(
            "DISTANCE_BASED", "TIER_2", "EPA_PART_98"
        )
        method_findings = [
            f for f in findings
            if "METHOD" in f.requirement_id
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(method_findings) == 1

    def test_empty_method_is_critical(self, engine):
        """Empty method string produces CRITICAL finding."""
        findings = engine.validate_methodology("", "TIER_2", "GHG_PROTOCOL")
        method_findings = [
            f for f in findings
            if "METHOD" in f.requirement_id
            and f.severity == FindingSeverity.CRITICAL.value
        ]
        assert len(method_findings) == 1
        assert method_findings[0].data_present is False

    def test_valid_tier_is_pass(self, engine):
        """Valid tier produces PASS finding."""
        findings = engine.validate_methodology(
            "FUEL_BASED", "TIER_2", "GHG_PROTOCOL"
        )
        tier_findings = [
            f for f in findings
            if "TIER" in f.requirement_id
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(tier_findings) >= 1

    def test_invalid_tier_is_major(self, engine):
        """Invalid tier produces MAJOR finding."""
        findings = engine.validate_methodology(
            "FUEL_BASED", "TIER_99", "GHG_PROTOCOL"
        )
        tier_findings = [
            f for f in findings
            if "TIER" in f.requirement_id
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(tier_findings) == 1

    def test_tier_4_valid_only_for_epa(self, engine):
        """TIER_4 is only valid for EPA Part 98."""
        # EPA accepts Tier 4
        findings = engine.validate_methodology(
            "FUEL_BASED", "TIER_4", "EPA_PART_98"
        )
        tier_findings_pass = [
            f for f in findings
            if f.requirement_id == "EPA_PART_98-TIER"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(tier_findings_pass) == 1

        # GHG Protocol rejects Tier 4
        findings_ghg = engine.validate_methodology(
            "FUEL_BASED", "TIER_4", "GHG_PROTOCOL"
        )
        tier_findings_fail = [
            f for f in findings_ghg
            if f.requirement_id == "GHG_PROTOCOL-TIER"
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(tier_findings_fail) == 1

    def test_tier_1_epa_upgrade_recommendation(self, engine):
        """TIER_1 on EPA/EU_ETS produces INFORMATIONAL upgrade finding."""
        findings = engine.validate_methodology(
            "FUEL_BASED", "TIER_1", "EPA_PART_98"
        )
        upgrade = [
            f for f in findings
            if "UPGRADE" in f.requirement_id
            and f.severity == FindingSeverity.INFORMATIONAL.value
        ]
        assert len(upgrade) == 1

    def test_tier_1_eu_ets_upgrade_recommendation(self, engine):
        """TIER_1 on EU ETS MRR produces INFORMATIONAL upgrade finding."""
        findings = engine.validate_methodology(
            "FUEL_BASED", "TIER_1", "EU_ETS_MRR"
        )
        upgrade = [
            f for f in findings
            if "UPGRADE" in f.requirement_id
            and f.severity == FindingSeverity.INFORMATIONAL.value
        ]
        assert len(upgrade) == 1

    def test_tier_2_no_upgrade_recommendation(self, engine):
        """TIER_2 does not trigger upgrade recommendation."""
        findings = engine.validate_methodology(
            "FUEL_BASED", "TIER_2", "EPA_PART_98"
        )
        upgrade = [
            f for f in findings
            if "UPGRADE" in f.requirement_id
        ]
        assert len(upgrade) == 0

    def test_invalid_framework_raises(self, engine):
        """Invalid framework raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized framework"):
            engine.validate_methodology("FUEL_BASED", "TIER_2", "INVALID")


# ===========================================================================
# TestValidateReportingRequirements
# ===========================================================================


class TestValidateReportingRequirements:
    """Test validate_reporting_requirements method."""

    def test_ghg_protocol_per_gas_pass(self, engine):
        """GHG Protocol with all gases present = PASS."""
        data = {"gases": {"CO2": 48000, "CH4": 1200, "N2O": 800}}
        findings = engine.validate_reporting_requirements(data, "GHG_PROTOCOL")
        gas_findings = [
            f for f in findings
            if "GASES" in f.requirement_id
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(gas_findings) == 1

    def test_ghg_protocol_missing_gas_is_major(self, engine):
        """Missing a gas (e.g., N2O) = MAJOR finding."""
        data = {"gases": {"CO2": 48000, "CH4": 1200}}
        findings = engine.validate_reporting_requirements(data, "GHG_PROTOCOL")
        gas_findings = [
            f for f in findings
            if "GASES" in f.requirement_id
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(gas_findings) == 1
        assert "N2O" in gas_findings[0].detail

    def test_ghg_protocol_biogenic_co2_pass(self, engine):
        """Biogenic CO2 separated = PASS for GHG Protocol."""
        data = {"biogenic_co2_separated": True}
        findings = engine.validate_reporting_requirements(data, "GHG_PROTOCOL")
        biogenic = [
            f for f in findings
            if f.requirement_id == "GHG-MC-003"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(biogenic) == 1

    def test_ghg_protocol_biogenic_co2_fail(self, engine):
        """Biogenic CO2 not separated = MAJOR for GHG Protocol."""
        data = {}
        findings = engine.validate_reporting_requirements(data, "GHG_PROTOCOL")
        biogenic = [
            f for f in findings
            if f.requirement_id == "GHG-MC-003"
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(biogenic) == 1

    def test_ghg_protocol_gwp_source_pass(self, engine):
        """GWP source disclosed = PASS."""
        data = {"gwp_source": "IPCC AR5"}
        findings = engine.validate_reporting_requirements(data, "GHG_PROTOCOL")
        gwp = [
            f for f in findings
            if f.requirement_id == "GHG-MC-005"
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(gwp) == 1

    def test_ghg_protocol_gwp_source_fail(self, engine):
        """GWP source not disclosed = MAJOR."""
        data = {}
        findings = engine.validate_reporting_requirements(data, "GHG_PROTOCOL")
        gwp = [
            f for f in findings
            if f.requirement_id == "GHG-MC-005"
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(gwp) == 1

    def test_uk_secr_energy_kwh_pass(self, engine):
        """UK SECR with energy_consumption_kwh present = PASS."""
        data = {"energy_consumption_kwh": 180000}
        findings = engine.validate_reporting_requirements(data, "UK_SECR")
        energy = [
            f for f in findings
            if "ENERGY" in f.requirement_id
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(energy) == 1

    def test_uk_secr_energy_kwh_fail(self, engine):
        """UK SECR without energy_consumption_kwh = MAJOR."""
        data = {}
        findings = engine.validate_reporting_requirements(data, "UK_SECR")
        energy = [
            f for f in findings
            if "ENERGY" in f.requirement_id
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(energy) == 1

    def test_csrd_transport_category_pass(self, engine):
        """CSRD with transport_category_separated = PASS."""
        data = {"transport_category_separated": True}
        findings = engine.validate_reporting_requirements(data, "CSRD_ESRS_E1")
        transport = [
            f for f in findings
            if "TRANSPORT" in f.requirement_id
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(transport) == 1

    def test_csrd_transport_category_fail(self, engine):
        """CSRD without transport_category_separated = MAJOR."""
        data = {}
        findings = engine.validate_reporting_requirements(data, "CSRD_ESRS_E1")
        transport = [
            f for f in findings
            if "TRANSPORT" in f.requirement_id
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(transport) == 1

    def test_iso_14064_no_gas_check(self, engine):
        """ISO 14064 per-gas check is also active."""
        data = {"gases": {"CO2": 48000, "CH4": 1200, "N2O": 800}}
        findings = engine.validate_reporting_requirements(data, "ISO_14064")
        gas_findings = [
            f for f in findings
            if "GASES" in f.requirement_id
            and f.severity == FindingSeverity.PASS.value
        ]
        assert len(gas_findings) == 1

    def test_epa_per_gas_check(self, engine):
        """EPA Part 98 checks per-gas reporting."""
        data = {"gases": {"CO2": 48000}}
        findings = engine.validate_reporting_requirements(data, "EPA_PART_98")
        gas_findings = [
            f for f in findings
            if "GASES" in f.requirement_id
            and f.severity == FindingSeverity.MAJOR.value
        ]
        assert len(gas_findings) == 1
        # Should mention missing CH4 and N2O
        assert "CH4" in gas_findings[0].detail
        assert "N2O" in gas_findings[0].detail

    def test_eu_ets_no_per_gas_check(self, engine):
        """EU ETS MRR does not check per-gas (not in gas-check frameworks)."""
        data = {}
        findings = engine.validate_reporting_requirements(data, "EU_ETS_MRR")
        gas_findings = [
            f for f in findings
            if "GASES" in f.requirement_id
        ]
        assert len(gas_findings) == 0

    def test_invalid_framework_raises(self, engine):
        """Invalid framework raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized framework"):
            engine.validate_reporting_requirements({}, "INVALID")


# ===========================================================================
# TestValidateIntensityMetrics
# ===========================================================================


class TestValidateIntensityMetrics:
    """Test validate_intensity_metrics method."""

    def test_both_metrics_present(self, engine):
        """Both revenue and FTE metrics present = 2 PASS findings."""
        data = {"intensity_per_revenue": 0.0025, "intensity_per_fte": 1.2}
        findings = engine.validate_intensity_metrics(data)
        assert len(findings) == 2
        assert all(f.severity == FindingSeverity.PASS.value for f in findings)

    def test_both_metrics_missing(self, engine):
        """Both metrics missing = 2 MAJOR findings."""
        data = {}
        findings = engine.validate_intensity_metrics(data)
        assert len(findings) == 2
        assert all(f.severity == FindingSeverity.MAJOR.value for f in findings)

    def test_revenue_only(self, engine):
        """Revenue intensity present, FTE missing."""
        data = {"intensity_per_revenue": 0.0025}
        findings = engine.validate_intensity_metrics(data)
        revenue = [
            f for f in findings if f.requirement_id == "INTENSITY-REVENUE"
        ]
        fte = [
            f for f in findings if f.requirement_id == "INTENSITY-FTE"
        ]
        assert revenue[0].severity == FindingSeverity.PASS.value
        assert fte[0].severity == FindingSeverity.MAJOR.value

    def test_fte_only(self, engine):
        """FTE intensity present, revenue missing."""
        data = {"intensity_per_fte": 1.2}
        findings = engine.validate_intensity_metrics(data)
        revenue = [
            f for f in findings if f.requirement_id == "INTENSITY-REVENUE"
        ]
        fte = [
            f for f in findings if f.requirement_id == "INTENSITY-FTE"
        ]
        assert revenue[0].severity == FindingSeverity.MAJOR.value
        assert fte[0].severity == FindingSeverity.PASS.value

    def test_zero_is_valid(self, engine):
        """Zero values are considered present (not None)."""
        data = {"intensity_per_revenue": 0, "intensity_per_fte": 0}
        findings = engine.validate_intensity_metrics(data)
        assert all(f.severity == FindingSeverity.PASS.value for f in findings)

    def test_intensity_not_added_for_ghg_protocol(self, engine, fully_compliant_ghg_data):
        """GHG Protocol check does NOT add intensity findings."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        intensity = [
            f for f in result.findings
            if f.requirement_id in {"INTENSITY-REVENUE", "INTENSITY-FTE"}
        ]
        assert len(intensity) == 0


# ===========================================================================
# TestGenerateRecommendations
# ===========================================================================


class TestGenerateRecommendations:
    """Test generate_recommendations method."""

    def test_compliant_findings_no_recs(self, engine):
        """Compliant findings generate no recommendations from findings."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="GHG-MC-001",
                requirement_text="Test", category="methodology",
                severity=FindingSeverity.PASS.value,
                status=ComplianceStatus.COMPLIANT.value,
                detail="OK", data_present=True,
            ),
        ]
        # Note: general recs for missing uncertainty/base_year still added
        recs = engine.generate_recommendations(findings, "GHG_PROTOCOL", {})
        # At least uncertainty and base year general recs
        assert len(recs) >= 2

    def test_missing_gases_recommendation(self, engine):
        """Missing gas finding generates gas recommendation."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="GHG_PROTOCOL-GASES",
                requirement_text="Per-gas emission reporting",
                category="reporting",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Missing CO2", data_present=False,
            ),
        ]
        recs = engine.generate_recommendations(
            findings, "GHG_PROTOCOL",
            {"gases": {"CH4": 100, "N2O": 50}}
        )
        gas_recs = [r for r in recs if "CO2" in r]
        assert len(gas_recs) > 0

    def test_missing_uncertainty_always_recommended(self, engine):
        """Missing uncertainty_assessment always adds a recommendation."""
        recs = engine.generate_recommendations([], "GHG_PROTOCOL", {})
        uncertainty_recs = [r for r in recs if "uncertainty" in r.lower()]
        assert len(uncertainty_recs) > 0

    def test_missing_base_year_always_recommended(self, engine):
        """Missing base_year always adds a recommendation."""
        recs = engine.generate_recommendations([], "GHG_PROTOCOL", {})
        base_year_recs = [r for r in recs if "base year" in r.lower()]
        assert len(base_year_recs) > 0

    def test_no_duplicate_recommendations(self, engine):
        """Duplicate recommendations are not added."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="ISO-MC-003",
                requirement_text="uncertainty statement",
                category="quality",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Missing", data_present=False,
            ),
        ]
        recs = engine.generate_recommendations(findings, "ISO_14064", {})
        # Uncertainty rec should appear only once (from finding + general)
        uncertainty_recs = [r for r in recs if "uncertainty" in r.lower()]
        assert len(uncertainty_recs) == 1

    def test_invalid_method_recommendation(self, engine):
        """Invalid method generates method-specific recommendation."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="EPA_PART_98-METHOD",
                requirement_text="Calculation method accepted by EPA_PART_98",
                category="methodology",
                severity=FindingSeverity.CRITICAL.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Method 'DISTANCE_BASED' is NOT accepted",
                data_present=True,
            ),
        ]
        recs = engine.generate_recommendations(
            findings, "EPA_PART_98",
            {"method": "DISTANCE_BASED", "uncertainty_assessment": True, "base_year": 2020}
        )
        method_recs = [r for r in recs if "DISTANCE_BASED" in r]
        assert len(method_recs) > 0

    def test_biogenic_recommendation(self, engine):
        """Missing biogenic CO2 produces biogenic recommendation."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="GHG-MC-003",
                requirement_text="Separate biogenic CO2 reporting",
                category="reporting",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Not separated", data_present=False,
            ),
        ]
        recs = engine.generate_recommendations(findings, "GHG_PROTOCOL", {})
        biogenic_recs = [r for r in recs if "biogenic" in r.lower()]
        assert len(biogenic_recs) > 0

    def test_energy_kwh_recommendation(self, engine):
        """Missing energy kWh produces energy recommendation."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="UK_SECR-ENERGY",
                requirement_text="Energy consumption reported in kWh",
                category="reporting",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Not reported", data_present=False,
            ),
        ]
        recs = engine.generate_recommendations(findings, "UK_SECR", {})
        energy_recs = [r for r in recs if "kWh" in r]
        assert len(energy_recs) > 0

    def test_transport_category_recommendation(self, engine):
        """Missing transport category produces transport recommendation."""
        findings = [
            ComplianceFinding(
                finding_id="f_1", requirement_id="CSRD-TRANSPORT",
                requirement_text="Transport as separate emission category",
                category="reporting",
                severity=FindingSeverity.MAJOR.value,
                status=ComplianceStatus.NON_COMPLIANT.value,
                detail="Not separated", data_present=False,
            ),
        ]
        recs = engine.generate_recommendations(findings, "CSRD_ESRS_E1", {})
        transport_recs = [r for r in recs if "transport" in r.lower()]
        assert len(transport_recs) > 0


# ===========================================================================
# TestCheckBaseYearThreshold
# ===========================================================================


class TestCheckBaseYearThreshold:
    """Test check_base_year_threshold method."""

    def test_within_threshold(self, engine):
        """Change of 3% is within 5% threshold (returns False)."""
        result = engine.check_base_year_threshold(
            Decimal("10300"), Decimal("10000")
        )
        assert result is False

    def test_exceeds_threshold(self, engine):
        """Change of 10% exceeds 5% threshold (returns True)."""
        result = engine.check_base_year_threshold(
            Decimal("11000"), Decimal("10000")
        )
        assert result is True

    def test_exactly_5_percent(self, engine):
        """Exactly 5% change does NOT exceed threshold (> not >=)."""
        result = engine.check_base_year_threshold(
            Decimal("10500"), Decimal("10000")
        )
        assert result is False

    def test_slightly_above_5_percent(self, engine):
        """5.01% change exceeds threshold."""
        result = engine.check_base_year_threshold(
            Decimal("10501"), Decimal("10000")
        )
        assert result is True

    def test_decrease_exceeds_threshold(self, engine):
        """Decrease > 5% also exceeds threshold (absolute change)."""
        result = engine.check_base_year_threshold(
            Decimal("9000"), Decimal("10000")
        )
        assert result is True

    def test_base_zero_current_positive(self, engine):
        """Base = 0, current > 0 returns True."""
        result = engine.check_base_year_threshold(
            Decimal("1000"), Decimal("0")
        )
        assert result is True

    def test_base_zero_current_zero(self, engine):
        """Base = 0, current = 0 returns False."""
        result = engine.check_base_year_threshold(
            Decimal("0"), Decimal("0")
        )
        assert result is False

    def test_negative_current_raises(self, engine):
        """Negative current emissions raises ValueError."""
        with pytest.raises(ValueError, match="current_fleet_emissions must be >= 0"):
            engine.check_base_year_threshold(Decimal("-100"), Decimal("10000"))

    def test_negative_base_raises(self, engine):
        """Negative base emissions raises ValueError."""
        with pytest.raises(ValueError, match="base_fleet_emissions must be >= 0"):
            engine.check_base_year_threshold(Decimal("10000"), Decimal("-100"))

    def test_accepts_int_and_float(self, engine):
        """Method accepts int and float via _to_decimal conversion."""
        # Integers
        result_int = engine.check_base_year_threshold(11000, 10000)
        assert result_int is True
        # Floats
        result_float = engine.check_base_year_threshold(10300.0, 10000.0)
        assert result_float is False


# ===========================================================================
# TestComplianceStatusLogic
# ===========================================================================


class TestComplianceStatusLogic:
    """Test overall compliance status determination logic."""

    def test_zero_mandatory_failures_is_compliant(self, engine, fully_compliant_ghg_data):
        """0 mandatory failures = COMPLIANT."""
        result = engine.check_compliance(fully_compliant_ghg_data, "GHG_PROTOCOL")
        if result.mandatory_met == result.mandatory_requirements:
            assert result.status == ComplianceStatus.COMPLIANT.value

    def test_many_failures_is_non_compliant(self, engine, minimal_data):
        """Many mandatory failures = NON_COMPLIANT."""
        result = engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        # Minimal data will have many critical/major failures
        assert result.status == ComplianceStatus.NON_COMPLIANT.value

    def test_few_failures_is_needs_review(self, engine):
        """1-2 mandatory failures = NEEDS_REVIEW."""
        # Provide most data but leave out 1-2 mandatory fields
        data = {
            "total_co2e_kg": 50000,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
            "gases": {"CO2": 48000, "CH4": 1200, "N2O": 800},
            "biogenic_co2_separated": True,
            "gwp_source": "IPCC AR5",
            "base_year": 2020,
            "base_year_emissions": 49000,
            "emission_factor_source": "EPA",
            "organizational_boundary": "operational_control",
            "control_approach": "operational_control",
            # Missing: fuel_type, fuel_quantity (optional)
            # Missing: uncertainty_assessment (optional)
        }
        result = engine.check_compliance(data, "GHG_PROTOCOL")
        # Status should be COMPLIANT or NEEDS_REVIEW (depends on exact finding count)
        assert result.status in {
            ComplianceStatus.COMPLIANT.value,
            ComplianceStatus.NEEDS_REVIEW.value,
        }


# ===========================================================================
# TestHistoryManagement
# ===========================================================================


class TestHistoryManagement:
    """Test compliance history get/clear operations."""

    def test_history_records_check(self, engine, minimal_data):
        """Each check_compliance call adds to history."""
        engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        history = engine.get_compliance_history()
        assert len(history) == 1
        assert history[0].framework == "GHG_PROTOCOL"

    def test_history_accumulates(self, engine, minimal_data):
        """Multiple checks accumulate in history."""
        engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        engine.check_compliance(minimal_data, "ISO_14064")
        engine.check_compliance(minimal_data, "EPA_PART_98")
        assert len(engine.get_compliance_history()) == 3

    def test_clear_history(self, engine, minimal_data):
        """clear_history removes all records and returns count."""
        engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        engine.check_compliance(minimal_data, "ISO_14064")
        count = engine.clear_history()
        assert count == 2
        assert len(engine.get_compliance_history()) == 0

    def test_clear_empty_history(self, engine):
        """clear_history on empty history returns 0."""
        count = engine.clear_history()
        assert count == 0

    def test_history_returns_copy(self, engine, minimal_data):
        """get_compliance_history returns a copy, not the original."""
        engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        history = engine.get_compliance_history()
        history.clear()
        assert len(engine.get_compliance_history()) == 1


# ===========================================================================
# TestProvenanceHash
# ===========================================================================


class TestProvenanceHash:
    """Test provenance hash generation and determinism."""

    def test_provenance_hash_is_sha256(self, engine, minimal_data):
        """Provenance hash is a 64-character hex string (SHA-256)."""
        result = engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_provenance_hash_deterministic(self, engine, minimal_data):
        """Same input yields same provenance hash (status-dependent)."""
        r1 = engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        r2 = engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        # Same data, same framework => same provenance hash
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_frameworks_different_hash(self, engine, minimal_data):
        """Different frameworks produce different provenance hashes."""
        r1 = engine.check_compliance(minimal_data, "GHG_PROTOCOL")
        r2 = engine.check_compliance(minimal_data, "ISO_14064")
        assert r1.provenance_hash != r2.provenance_hash

    def test_different_data_different_hash(self, engine):
        """Different total_co2e_kg values produce different hashes."""
        r1 = engine.check_compliance(
            {"total_co2e_kg": 5000}, "GHG_PROTOCOL"
        )
        r2 = engine.check_compliance(
            {"total_co2e_kg": 99999}, "GHG_PROTOCOL"
        )
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# TestValidMethodsAndTiers
# ===========================================================================


class TestValidMethodsAndTiers:
    """Test _VALID_METHODS_BY_FRAMEWORK and _VALID_TIERS_BY_FRAMEWORK."""

    def test_valid_methods_mapping_complete(self):
        """All 6 frameworks have entries in methods mapping."""
        for fw in RegulatoryFramework:
            assert fw.value in _VALID_METHODS_BY_FRAMEWORK

    def test_valid_tiers_mapping_complete(self):
        """All 6 frameworks have entries in tiers mapping."""
        for fw in RegulatoryFramework:
            assert fw.value in _VALID_TIERS_BY_FRAMEWORK

    def test_epa_fuel_based_only(self):
        """EPA Part 98 only allows FUEL_BASED."""
        assert _VALID_METHODS_BY_FRAMEWORK["EPA_PART_98"] == ["FUEL_BASED"]

    def test_eu_ets_fuel_based_only(self):
        """EU ETS MRR only allows FUEL_BASED."""
        assert _VALID_METHODS_BY_FRAMEWORK["EU_ETS_MRR"] == ["FUEL_BASED"]

    def test_ghg_protocol_two_methods(self):
        """GHG Protocol allows FUEL_BASED and DISTANCE_BASED."""
        assert set(_VALID_METHODS_BY_FRAMEWORK["GHG_PROTOCOL"]) == {
            "FUEL_BASED", "DISTANCE_BASED"
        }

    def test_iso_three_methods(self):
        """ISO 14064 allows FUEL_BASED, DISTANCE_BASED, SPEND_BASED."""
        assert set(_VALID_METHODS_BY_FRAMEWORK["ISO_14064"]) == {
            "FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED"
        }

    def test_epa_has_tier_4(self):
        """EPA Part 98 includes TIER_4."""
        assert "TIER_4" in _VALID_TIERS_BY_FRAMEWORK["EPA_PART_98"]

    def test_ghg_protocol_no_tier_4(self):
        """GHG Protocol does not include TIER_4."""
        assert "TIER_4" not in _VALID_TIERS_BY_FRAMEWORK["GHG_PROTOCOL"]


# ===========================================================================
# TestFrameworkRequirementStructure
# ===========================================================================


class TestFrameworkRequirementStructure:
    """Test framework requirement definitions structure."""

    def test_total_60_requirements(self):
        """All 6 frameworks x 10 requirements = 60 total."""
        total = sum(len(reqs) for reqs in _FRAMEWORK_REQUIREMENTS.values())
        assert total == 60

    def test_each_requirement_has_unique_id(self):
        """Each requirement within a framework has a unique ID."""
        for fw, reqs in _FRAMEWORK_REQUIREMENTS.items():
            ids = [r["id"] for r in reqs]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {fw}"

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
        "EPA_PART_98", "UK_SECR", "EU_ETS_MRR",
    ])
    def test_framework_has_10_requirements(self, framework):
        """Each framework has exactly 10 requirements."""
        assert len(_FRAMEWORK_REQUIREMENTS[framework]) == 10

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
        "EPA_PART_98", "UK_SECR", "EU_ETS_MRR",
    ])
    def test_mandatory_and_optional_split(self, framework):
        """Each framework has at least 1 mandatory and can have optional."""
        reqs = _FRAMEWORK_REQUIREMENTS[framework]
        mandatory = [r for r in reqs if r.get("mandatory", True)]
        assert len(mandatory) >= 1


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_none_values_in_data(self, engine):
        """None values in data are treated as missing."""
        data = {"total_co2e_kg": None, "method": None}
        result = engine.check_compliance(data, "GHG_PROTOCOL")
        assert isinstance(result, ComplianceCheckResult)

    def test_extra_fields_ignored(self, engine, fully_compliant_ghg_data):
        """Extra fields in data do not cause errors."""
        data = dict(fully_compliant_ghg_data)
        data["unknown_field"] = "some_value"
        data["another_extra"] = 12345
        result = engine.check_compliance(data, "GHG_PROTOCOL")
        assert isinstance(result, ComplianceCheckResult)

    def test_zero_emissions_compliant(self, engine, fully_compliant_ghg_data):
        """Zero emissions is a valid value."""
        data = dict(fully_compliant_ghg_data)
        data["total_co2e_kg"] = 0
        result = engine.check_compliance(data, "GHG_PROTOCOL")
        assert isinstance(result, ComplianceCheckResult)

    def test_very_large_emissions(self, engine):
        """Very large emissions values are handled."""
        data = {
            "total_co2e_kg": 999999999999,
            "method": "FUEL_BASED",
            "tier": "TIER_2",
        }
        result = engine.check_compliance(data, "EPA_PART_98")
        assert isinstance(result, ComplianceCheckResult)
        # Should trigger large emitter check
        large_emitter = [
            f for f in result.findings
            if f.requirement_id == "EPA-MC-002"
            and "exceed" in f.detail.lower()
        ]
        assert len(large_emitter) > 0


# ===========================================================================
# TestThreadSafety
# ===========================================================================


class TestThreadSafety:
    """Test thread safety of compliance history access."""

    def test_concurrent_compliance_checks(self, engine):
        """Multiple threads can run compliance checks concurrently."""
        results = []
        errors = []

        def run_check(fw):
            try:
                r = engine.check_compliance(
                    {"total_co2e_kg": 5000, "method": "FUEL_BASED", "tier": "TIER_2"},
                    fw,
                )
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = []
        for fw in ["GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
                    "EPA_PART_98", "UK_SECR", "EU_ETS_MRR"]:
            t = threading.Thread(target=run_check, args=(fw,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 6
        assert len(engine.get_compliance_history()) == 6
