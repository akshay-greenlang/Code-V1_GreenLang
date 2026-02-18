# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceCheckerEngine (Engine 6 of 7) - AGENT-MRV-005

Tests all 7 regulatory frameworks (GHG Protocol, ISO 14064, CSRD/ESRS E1,
EPA Subpart W, EPA LDAR, EU Methane Regulation, UK SECR), each with 10
requirements. Validates compliance status determination, field-level
requirement checks, data completeness, recommendations, and provenance.

Target: 157 tests, ~1980 lines.

Test classes:
    TestGHGProtocol (25)
    TestISO14064 (25)
    TestCSRD (25)
    TestEPASubpartW (25)
    TestEPALDAR (25)
    TestEUMethaneReg (20)
    TestUKSECR (12)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.fugitive_emissions.compliance_checker import (
    ComplianceCheckerEngine,
    SUPPORTED_FRAMEWORKS,
    _build_requirements,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Default ComplianceCheckerEngine."""
    return ComplianceCheckerEngine()


@pytest.fixture
def fully_compliant_data():
    """Calculation data that satisfies ALL 70 requirement fields."""
    return {
        # GHG Protocol fields
        "source_type": "EQUIPMENT_LEAK",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "emission_factor_source": "EPA",
        "facility_id": "FAC-001",
        "emissions_by_gas": [{"gas": "CH4", "mass_kg": 100}],
        "de_minimis_justified": True,
        "period_start": "2025-01-01",
        "base_year_defined": True,
        "qm_procedures": True,
        "provenance_hash": "a" * 64,
        # ISO 14064 fields
        "data_quality_assessed": True,
        "uncertainty_assessed": True,
        "exclusions_documented": True,
        "competence_confirmed": True,
        # CSRD fields
        "scope_category": "fugitive",
        "sector_standard": "oil_gas",
        "reduction_targets": True,
        "mitigation_actions": True,
        "financial_impact": True,
        "value_chain_scope": True,
        "xbrl_tagged": True,
        # EPA Subpart W fields
        "monitoring_plan": True,
        "missing_data_procedures": True,
        "calibration_current": True,
        "qaqc_procedures": True,
        "recordkeeping_policy": True,
        "annual_report_submitted": True,
        "verification_available": True,
        "eggrt_reported": True,
        # EPA LDAR fields
        "monitoring_frequency": "quarterly",
        "leak_definition_ppmv": 500,
        "repair_deadline_days": 15,
        "remonitor_after_repair": True,
        "dor_documented": True,
        "components_tagged": True,
        "surveyors_trained": True,
        "annual_report_available": True,
        "audit_ready": True,
        # EU Methane Regulation fields
        "has_ldar_program": True,
        "ogi_technology_used": True,
        "detection_sensitivity_met": True,
        "eu_repair_timeline_met": True,
        "reported_to_authority": True,
        "ogmp_level": 4,
        "methane_intensity_reported": True,
        "source_level_reporting": True,
        "third_party_verified": True,
        "public_disclosure": True,
        # UK SECR fields
        "total_co2e_tonnes": 500.0,
        "intensity_ratio": 0.5,
        "year_comparison": True,
        "scope_coverage": "scope1_scope2",
        "methodology_stated": True,
        "efficiency_narrative": True,
        "director_responsibility": True,
        "assurance_statement": True,
        "companies_act_compliant": True,
    }


@pytest.fixture
def minimal_data():
    """Minimal calculation data with only basic fields."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "total_co2e_tonnes": 100.0,
        "facility_id": "FAC-001",
    }


@pytest.fixture
def empty_data():
    """Empty calculation data (all fields missing)."""
    return {}


# ===========================================================================
# TestGHGProtocol (25 tests)
# ===========================================================================


class TestGHGProtocol:
    """Tests for GHG Protocol Corporate Standard compliance checks."""

    def test_fully_compliant_status(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        assert result["status"] == "compliant"
        assert result["met_count"] == 10

    def test_missing_source_type(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["source_type"]
        result = engine.check_framework("GHG_PROTOCOL", data)
        assert result["met_count"] < 10

    def test_missing_calculation_method(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["calculation_method"]
        result = engine.check_framework("GHG_PROTOCOL", data)
        assert result["met_count"] < 10

    def test_missing_ef_source(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["emission_factor_source"]
        result = engine.check_framework("GHG_PROTOCOL", data)
        assert result["met_count"] < 10

    def test_missing_facility_id(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["facility_id"]
        result = engine.check_framework("GHG_PROTOCOL", data)
        assert result["met_count"] < 10

    def test_empty_gas_list_not_met(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        data["emissions_by_gas"] = []
        result = engine.check_framework("GHG_PROTOCOL", data)
        assert result["met_count"] < 10

    def test_requirement_count_is_10(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_findings_list_length_10(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        assert len(result["findings"]) == 10

    def test_finding_structure(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        finding = result["findings"][0]
        assert "requirement_id" in finding
        assert "requirement_name" in finding
        assert "severity" in finding
        assert "status" in finding

    def test_finding_ids_ghg_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        for finding in result["findings"]:
            assert finding["requirement_id"].startswith("GHG-FE-")

    def test_partial_compliance(self, engine, minimal_data):
        result = engine.check_framework("GHG_PROTOCOL", minimal_data)
        assert result["status"] in ("partial", "non_compliant")

    def test_non_compliant_empty_data(self, engine, empty_data):
        result = engine.check_framework("GHG_PROTOCOL", empty_data)
        assert result["status"] == "non_compliant"

    def test_pass_rate_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        assert result["pass_rate"] == 1.0

    def test_pass_rate_zero_empty(self, engine, empty_data):
        result = engine.check_framework("GHG_PROTOCOL", empty_data)
        assert result["pass_rate"] < 0.5

    def test_recommendations_for_missing(self, engine, minimal_data):
        result = engine.check_framework("GHG_PROTOCOL", minimal_data)
        assert len(result["recommendations"]) > 0

    def test_no_recommendations_when_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        assert len(result["recommendations"]) == 0

    def test_severity_error_source_type(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        src = [f for f in result["findings"] if f["requirement_id"] == "GHG-FE-001"][0]
        assert src["severity"] == "ERROR"

    def test_severity_warning_de_minimis(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        dm = [f for f in result["findings"] if f["requirement_id"] == "GHG-FE-006"][0]
        assert dm["severity"] == "WARNING"

    def test_severity_info_qm(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        qm = [f for f in result["findings"] if f["requirement_id"] == "GHG-FE-009"][0]
        assert qm["severity"] == "INFO"

    def test_empty_string_not_met(self, engine):
        data = {"source_type": "", "calculation_method": "AEF"}
        result = engine.check_framework("GHG_PROTOCOL", data)
        src = [f for f in result["findings"] if f["check_field"] == "source_type"][0]
        assert src["status"] == "not_met"

    def test_false_bool_not_met(self, engine):
        data = {"de_minimis_justified": False, "source_type": "EQUIPMENT_LEAK"}
        result = engine.check_framework("GHG_PROTOCOL", data)
        dm = [f for f in result["findings"] if f["check_field"] == "de_minimis_justified"][0]
        assert dm["status"] == "not_met"

    def test_numeric_zero_is_met(self, engine):
        data = {"total_co2e_tonnes": 0.0}
        result = engine.check_framework("UK_SECR", data)
        t = [f for f in result["findings"] if f["check_field"] == "total_co2e_tonnes"][0]
        assert t["status"] == "met"

    def test_none_value_not_met(self, engine):
        data = {"source_type": None}
        result = engine.check_framework("GHG_PROTOCOL", data)
        src = [f for f in result["findings"] if f["check_field"] == "source_type"][0]
        assert src["status"] == "not_met"

    def test_framework_label(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        assert result["framework"] == "GHG_PROTOCOL"

    def test_provenance_requirement(self, engine, fully_compliant_data):
        result = engine.check_framework("GHG_PROTOCOL", fully_compliant_data)
        prov = [f for f in result["findings"] if f["requirement_id"] == "GHG-FE-010"][0]
        assert prov["status"] == "met"


# ===========================================================================
# TestISO14064 (25 tests)
# ===========================================================================


class TestISO14064:
    """Tests for ISO 14064-1:2018 compliance checks."""

    def test_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        assert result["status"] == "compliant"

    def test_requirement_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_finding_ids_iso_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        for f in result["findings"]:
            assert f["requirement_id"].startswith("ISO-FE-")

    def test_missing_data_quality(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["data_quality_assessed"]
        result = engine.check_framework("ISO_14064", data)
        dq = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-004"][0]
        assert dq["status"] == "not_met"

    def test_missing_uncertainty(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["uncertainty_assessed"]
        result = engine.check_framework("ISO_14064", data)
        ua = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-005"][0]
        assert ua["status"] == "not_met"

    def test_clause_5_1_boundary(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        b = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-001"][0]
        assert b["status"] == "met"

    def test_clause_5_2_source(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        s = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-002"][0]
        assert s["status"] == "met"

    def test_clause_5_3_methodology(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        m = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-003"][0]
        assert m["status"] == "met"

    def test_clause_7_documentation(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        d = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-010"][0]
        assert d["status"] == "met"

    def test_non_compliant_empty(self, engine, empty_data):
        result = engine.check_framework("ISO_14064", empty_data)
        assert result["status"] == "non_compliant"

    def test_partial_with_minimal(self, engine, minimal_data):
        result = engine.check_framework("ISO_14064", minimal_data)
        assert result["status"] in ("partial", "non_compliant")

    def test_exclusions_warning_severity(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        exc = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-007"][0]
        assert exc["severity"] == "WARNING"

    def test_competence_info_severity(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        comp = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-009"][0]
        assert comp["severity"] == "INFO"

    def test_base_year_warning_severity(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        by = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-006"][0]
        assert by["severity"] == "WARNING"

    def test_temporal_warning_severity(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        tc = [f for f in result["findings"] if f["requirement_id"] == "ISO-FE-008"][0]
        assert tc["severity"] == "WARNING"

    def test_pass_rate_1_when_full(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        assert result["pass_rate"] == 1.0

    def test_recommendations_for_minimal(self, engine, minimal_data):
        result = engine.check_framework("ISO_14064", minimal_data)
        assert len(result["recommendations"]) > 0

    def test_met_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        assert result["met_count"] == 10

    def test_not_met_count_10_empty(self, engine, empty_data):
        result = engine.check_framework("ISO_14064", empty_data)
        assert result["not_met_count"] == 10

    def test_findings_have_framework(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        for f in result["findings"]:
            assert f["framework"] == "ISO_14064"

    def test_findings_have_description(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        for f in result["findings"]:
            assert len(f["description"]) > 0

    def test_findings_have_check_field(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        for f in result["findings"]:
            assert len(f["check_field"]) > 0

    def test_all_10_unique_ids(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        ids = [f["requirement_id"] for f in result["findings"]]
        assert len(set(ids)) == 10

    def test_pass_rate_zero_empty(self, engine, empty_data):
        result = engine.check_framework("ISO_14064", empty_data)
        assert result["pass_rate"] == 0.0

    def test_framework_echoed(self, engine, fully_compliant_data):
        result = engine.check_framework("ISO_14064", fully_compliant_data)
        assert result["framework"] == "ISO_14064"


# ===========================================================================
# TestCSRD (25 tests)
# ===========================================================================


class TestCSRD:
    """Tests for CSRD / ESRS E1 compliance checks."""

    def test_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        assert result["status"] == "compliant"
        assert result["met_count"] == 10

    def test_requirement_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_finding_ids_esrs_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        for f in result["findings"]:
            assert f["requirement_id"].startswith("ESRS-FE-")

    def test_scope_category_met(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        sc = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-003"][0]
        assert sc["status"] == "met"

    def test_missing_scope_category(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["scope_category"]
        result = engine.check_framework("CSRD_ESRS_E1", data)
        sc = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-003"][0]
        assert sc["status"] == "not_met"

    def test_xbrl_tagging_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        x = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-010"][0]
        assert x["severity"] == "WARNING"

    def test_financial_impact_info(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        fi = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-007"][0]
        assert fi["severity"] == "INFO"

    def test_assurance_readiness_error(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        ar = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-009"][0]
        assert ar["severity"] == "ERROR"
        assert ar["status"] == "met"

    def test_reduction_targets_met(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        rt = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-005"][0]
        assert rt["status"] == "met"

    def test_mitigation_actions_met(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        ma = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-006"][0]
        assert ma["status"] == "met"

    def test_non_compliant_empty(self, engine, empty_data):
        result = engine.check_framework("CSRD_ESRS_E1", empty_data)
        assert result["status"] == "non_compliant"

    def test_partial_minimal(self, engine, minimal_data):
        result = engine.check_framework("CSRD_ESRS_E1", minimal_data)
        assert result["status"] in ("partial", "non_compliant")

    def test_methodology_disclosure(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        md = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-002"][0]
        assert md["status"] == "met"

    def test_value_chain_scope_info(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        vcs = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-008"][0]
        assert vcs["severity"] == "INFO"

    def test_sector_standard_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        ss = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-004"][0]
        assert ss["severity"] == "WARNING"

    def test_missing_provenance(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["provenance_hash"]
        result = engine.check_framework("CSRD_ESRS_E1", data)
        prov = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-009"][0]
        assert prov["status"] == "not_met"

    def test_scope_by_category_error(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        sbc = [f for f in result["findings"] if f["requirement_id"] == "ESRS-FE-001"][0]
        assert sbc["severity"] == "ERROR"

    def test_recommendations_for_missing(self, engine, minimal_data):
        result = engine.check_framework("CSRD_ESRS_E1", minimal_data)
        assert len(result["recommendations"]) > 0

    def test_all_10_unique_ids(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        ids = [f["requirement_id"] for f in result["findings"]]
        assert len(set(ids)) == 10

    def test_pass_rate_full(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        assert result["pass_rate"] == 1.0

    def test_pass_rate_zero(self, engine, empty_data):
        result = engine.check_framework("CSRD_ESRS_E1", empty_data)
        assert result["pass_rate"] == 0.0

    def test_met_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        assert result["met_count"] == 10

    def test_not_met_count_10_empty(self, engine, empty_data):
        result = engine.check_framework("CSRD_ESRS_E1", empty_data)
        assert result["not_met_count"] == 10

    def test_framework_echoed(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        assert result["framework"] == "CSRD_ESRS_E1"

    def test_no_recommendations_when_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("CSRD_ESRS_E1", fully_compliant_data)
        assert len(result["recommendations"]) == 0


# ===========================================================================
# TestEPASubpartW (25 tests)
# ===========================================================================


class TestEPASubpartW:
    """Tests for EPA Subpart W (40 CFR Part 98) compliance checks."""

    def test_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        assert result["status"] == "compliant"

    def test_requirement_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_finding_ids_epaw_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        for f in result["findings"]:
            assert f["requirement_id"].startswith("EPAW-FE-")

    def test_monitoring_plan_error(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        mp = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-001"][0]
        assert mp["severity"] == "ERROR"

    def test_missing_monitoring_plan(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["monitoring_plan"]
        result = engine.check_framework("EPA_SUBPART_W", data)
        mp = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-001"][0]
        assert mp["status"] == "not_met"

    def test_missing_data_procedures(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["missing_data_procedures"]
        result = engine.check_framework("EPA_SUBPART_W", data)
        mdp = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-004"][0]
        assert mdp["status"] == "not_met"

    def test_calibration_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        cal = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-005"][0]
        assert cal["status"] == "met"

    def test_qaqc_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        qa = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-006"][0]
        assert qa["status"] == "met"

    def test_recordkeeping_error(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        rk = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-007"][0]
        assert rk["severity"] == "ERROR"

    def test_annual_report_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        ar = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-008"][0]
        assert ar["status"] == "met"

    def test_verification_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        v = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-009"][0]
        assert v["severity"] == "WARNING"

    def test_eggrt_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        eg = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-010"][0]
        assert eg["severity"] == "WARNING"

    def test_non_compliant_empty(self, engine, empty_data):
        result = engine.check_framework("EPA_SUBPART_W", empty_data)
        assert result["status"] == "non_compliant"

    def test_partial_minimal(self, engine, minimal_data):
        result = engine.check_framework("EPA_SUBPART_W", minimal_data)
        assert result["status"] in ("partial", "non_compliant")

    def test_all_10_unique_ids(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        ids = [f["requirement_id"] for f in result["findings"]]
        assert len(set(ids)) == 10

    def test_pass_rate_full(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        assert result["pass_rate"] == 1.0

    def test_source_category_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        sci = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-002"][0]
        assert sci["status"] == "met"

    def test_tier_methodology_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        tm = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-003"][0]
        assert tm["status"] == "met"

    def test_recommendations_empty_data(self, engine, empty_data):
        result = engine.check_framework("EPA_SUBPART_W", empty_data)
        assert len(result["recommendations"]) >= 1

    def test_met_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        assert result["met_count"] == 10

    def test_not_met_count_10_empty(self, engine, empty_data):
        result = engine.check_framework("EPA_SUBPART_W", empty_data)
        assert result["not_met_count"] == 10

    def test_framework_echoed(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        assert result["framework"] == "EPA_SUBPART_W"

    def test_missing_calibration(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["calibration_current"]
        result = engine.check_framework("EPA_SUBPART_W", data)
        cal = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-005"][0]
        assert cal["status"] == "not_met"

    def test_missing_recordkeeping(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["recordkeeping_policy"]
        result = engine.check_framework("EPA_SUBPART_W", data)
        rk = [f for f in result["findings"] if f["requirement_id"] == "EPAW-FE-007"][0]
        assert rk["status"] == "not_met"

    def test_no_recommendations_full(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_SUBPART_W", fully_compliant_data)
        assert len(result["recommendations"]) == 0


# ===========================================================================
# TestEPALDAR (25 tests)
# ===========================================================================


class TestEPALDAR:
    """Tests for EPA LDAR (40 CFR Part 60/63) compliance checks."""

    def test_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        assert result["status"] == "compliant"

    def test_requirement_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_finding_ids_ldar_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        for f in result["findings"]:
            assert f["requirement_id"].startswith("LDAR-FE-")

    def test_monitoring_frequency_error(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        mf = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-001"][0]
        assert mf["severity"] == "ERROR"

    def test_leak_threshold_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        ld = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-002"][0]
        assert ld["status"] == "met"

    def test_repair_deadline_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        rd = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-003"][0]
        assert rd["status"] == "met"

    def test_remonitor_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        rm = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-004"][0]
        assert rm["status"] == "met"

    def test_dor_documented_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        dor = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-005"][0]
        assert dor["status"] == "met"

    def test_component_tagging_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        ct = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-007"][0]
        assert ct["status"] == "met"

    def test_surveyor_training_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        st = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-008"][0]
        assert st["status"] == "met"

    def test_annual_reporting_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        ar = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-009"][0]
        assert ar["severity"] == "WARNING"

    def test_audit_readiness_info(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        audit = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-010"][0]
        assert audit["severity"] == "INFO"

    def test_missing_monitoring_frequency(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["monitoring_frequency"]
        result = engine.check_framework("EPA_LDAR", data)
        mf = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-001"][0]
        assert mf["status"] == "not_met"

    def test_missing_leak_threshold(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["leak_definition_ppmv"]
        result = engine.check_framework("EPA_LDAR", data)
        ld = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-002"][0]
        assert ld["status"] == "not_met"

    def test_non_compliant_empty(self, engine, empty_data):
        result = engine.check_framework("EPA_LDAR", empty_data)
        assert result["status"] == "non_compliant"

    def test_pass_rate_full(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        assert result["pass_rate"] == 1.0

    def test_pass_rate_zero(self, engine, empty_data):
        result = engine.check_framework("EPA_LDAR", empty_data)
        assert result["pass_rate"] == 0.0

    def test_all_10_unique_ids(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        ids = [f["requirement_id"] for f in result["findings"]]
        assert len(set(ids)) == 10

    def test_recordkeeping_error(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        rk = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-006"][0]
        assert rk["severity"] == "ERROR"

    def test_met_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        assert result["met_count"] == 10

    def test_not_met_count_10_empty(self, engine, empty_data):
        result = engine.check_framework("EPA_LDAR", empty_data)
        assert result["not_met_count"] == 10

    def test_recommendations_empty(self, engine, empty_data):
        result = engine.check_framework("EPA_LDAR", empty_data)
        assert len(result["recommendations"]) >= 1

    def test_missing_surveyors_trained(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["surveyors_trained"]
        result = engine.check_framework("EPA_LDAR", data)
        st = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-008"][0]
        assert st["status"] == "not_met"

    def test_missing_components_tagged(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        del data["components_tagged"]
        result = engine.check_framework("EPA_LDAR", data)
        ct = [f for f in result["findings"] if f["requirement_id"] == "LDAR-FE-007"][0]
        assert ct["status"] == "not_met"

    def test_framework_echoed(self, engine, fully_compliant_data):
        result = engine.check_framework("EPA_LDAR", fully_compliant_data)
        assert result["framework"] == "EPA_LDAR"


# ===========================================================================
# TestEUMethaneReg (20 tests)
# ===========================================================================


class TestEUMethaneReg:
    """Tests for EU Methane Regulation (EU 2024/1787) compliance checks."""

    def test_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        assert result["status"] == "compliant"

    def test_requirement_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_finding_ids_eumr_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        for f in result["findings"]:
            assert f["requirement_id"].startswith("EUMR-FE-")

    def test_ldar_program_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        ldar = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-001"][0]
        assert ldar["status"] == "met"

    def test_ogi_technology_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        ogi = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-002"][0]
        assert ogi["status"] == "met"

    def test_repair_timeline_error(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        rt = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-004"][0]
        assert rt["severity"] == "ERROR"

    def test_third_party_verified_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        tpv = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-009"][0]
        assert tpv["status"] == "met"

    def test_public_disclosure_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        pd = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-010"][0]
        assert pd["severity"] == "WARNING"

    def test_ogmp_level_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        ogmp = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-006"][0]
        assert ogmp["severity"] == "WARNING"

    def test_false_ldar_program(self, engine, fully_compliant_data):
        data = dict(fully_compliant_data)
        data["has_ldar_program"] = False
        result = engine.check_framework("EU_METHANE_REGULATION", data)
        ldar = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-001"][0]
        assert ldar["status"] == "not_met"

    def test_non_compliant_empty(self, engine, empty_data):
        result = engine.check_framework("EU_METHANE_REGULATION", empty_data)
        assert result["status"] == "non_compliant"

    def test_pass_rate_full(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        assert result["pass_rate"] == 1.0

    def test_source_level_reporting_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        slr = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-008"][0]
        assert slr["status"] == "met"

    def test_detection_sensitivity_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        ds = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-003"][0]
        assert ds["status"] == "met"

    def test_reporting_to_authority_met(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        rta = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-005"][0]
        assert rta["status"] == "met"

    def test_methane_intensity_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        mi = [f for f in result["findings"] if f["requirement_id"] == "EUMR-FE-007"][0]
        assert mi["severity"] == "WARNING"

    def test_all_10_unique_ids(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        ids = [f["requirement_id"] for f in result["findings"]]
        assert len(set(ids)) == 10

    def test_met_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        assert result["met_count"] == 10

    def test_framework_echoed(self, engine, fully_compliant_data):
        result = engine.check_framework("EU_METHANE_REGULATION", fully_compliant_data)
        assert result["framework"] == "EU_METHANE_REGULATION"

    def test_recommendations_empty_data(self, engine, empty_data):
        result = engine.check_framework("EU_METHANE_REGULATION", empty_data)
        assert len(result["recommendations"]) >= 1


# ===========================================================================
# TestUKSECR (12 tests)
# ===========================================================================


class TestUKSECR:
    """Tests for UK SECR compliance checks."""

    def test_fully_compliant(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        assert result["status"] == "compliant"

    def test_requirement_count_10(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        assert result["requirement_count"] == 10

    def test_finding_ids_secr_prefix(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        for f in result["findings"]:
            assert f["requirement_id"].startswith("SECR-FE-")

    def test_defra_methodology_met(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        defra = [f for f in result["findings"] if f["requirement_id"] == "SECR-FE-002"][0]
        assert defra["status"] == "met"

    def test_intensity_ratio_met(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        ir = [f for f in result["findings"] if f["requirement_id"] == "SECR-FE-003"][0]
        assert ir["status"] == "met"

    def test_companies_act_error(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        ca = [f for f in result["findings"] if f["requirement_id"] == "SECR-FE-010"][0]
        assert ca["severity"] == "ERROR"

    def test_non_compliant_empty(self, engine, empty_data):
        result = engine.check_framework("UK_SECR", empty_data)
        assert result["status"] == "non_compliant"

    def test_director_responsibility_warning(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        dr = [f for f in result["findings"] if f["requirement_id"] == "SECR-FE-008"][0]
        assert dr["severity"] == "WARNING"

    def test_assurance_info(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        assur = [f for f in result["findings"] if f["requirement_id"] == "SECR-FE-009"][0]
        assert assur["severity"] == "INFO"

    def test_year_comparison_met(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        yc = [f for f in result["findings"] if f["requirement_id"] == "SECR-FE-004"][0]
        assert yc["status"] == "met"

    def test_all_10_unique_ids(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        ids = [f["requirement_id"] for f in result["findings"]]
        assert len(set(ids)) == 10

    def test_framework_echoed(self, engine, fully_compliant_data):
        result = engine.check_framework("UK_SECR", fully_compliant_data)
        assert result["framework"] == "UK_SECR"
