# -*- coding: utf-8 -*-
"""
API tests for Audit Routes -- AGENT-EUDR-024

Tests audit planning, scheduling, execution, and auditor management
API endpoints including CRUD operations, request validation, response
format, error handling, and RBAC permission checks.

Target: ~40 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.models import (
    Audit,
    AuditScope,
    AuditModality,
    AuditStatus,
    CertificationScheme,
    Auditor,
    ScheduleAuditRequest,
    MatchAuditorRequest,
    SUPPORTED_COMMODITIES,
    EU_MEMBER_STATES,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    CERTIFICATION_SCHEMES,
)


class TestAuditScheduleEndpoint:
    """Test POST /v1/eudr-tam/audits/schedule endpoint logic."""

    def test_schedule_audits_valid_request(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        assert response is not None
        assert response.total_scheduled > 0

    def test_schedule_returns_risk_distribution(self, planning_engine, sample_schedule_request):
        response = planning_engine.schedule_audits(sample_schedule_request)
        assert "HIGH" in response.risk_distribution
        assert "STANDARD" in response.risk_distribution
        assert "LOW" in response.risk_distribution

    def test_schedule_empty_suppliers_returns_zero(self, planning_engine):
        request = ScheduleAuditRequest(
            operator_id="OP-001", supplier_ids=[], planning_year=2026,
        )
        response = planning_engine.schedule_audits(request)
        assert response.total_scheduled == 0

    def test_schedule_request_requires_operator_id(self):
        with pytest.raises((ValueError, Exception)):
            ScheduleAuditRequest(
                operator_id="", supplier_ids=["SUP-001"], planning_year=2026,
            )

    @pytest.mark.parametrize("year", [2025, 2026, 2027])
    def test_schedule_for_different_years(self, planning_engine, year):
        request = ScheduleAuditRequest(
            operator_id="OP-001",
            supplier_ids=["SUP-001"],
            planning_year=year,
        )
        response = planning_engine.schedule_audits(request)
        assert response is not None


class TestAuditCRUD:
    """Test audit CRUD operations."""

    def test_create_audit(self, sample_audit):
        assert sample_audit.audit_id == "AUD-TEST-001"
        assert sample_audit.status == AuditStatus.PLANNED

    def test_audit_requires_operator_id(self):
        with pytest.raises((ValueError, Exception)):
            Audit(
                operator_id="",
                supplier_id="SUP-001",
                planned_date=FROZEN_DATE,
                country_code="BR",
                commodity="wood",
            )

    def test_audit_requires_supplier_id(self):
        with pytest.raises((ValueError, Exception)):
            Audit(
                operator_id="OP-001",
                supplier_id="",
                planned_date=FROZEN_DATE,
                country_code="BR",
                commodity="wood",
            )

    def test_audit_country_code_uppercase(self):
        audit = Audit(
            operator_id="OP-001",
            supplier_id="SUP-001",
            planned_date=FROZEN_DATE,
            country_code="br",
            commodity="wood",
        )
        assert audit.country_code == "BR"

    def test_audit_default_status_is_planned(self):
        audit = Audit(
            operator_id="OP-001",
            supplier_id="SUP-001",
            planned_date=FROZEN_DATE,
            country_code="BR",
            commodity="wood",
        )
        assert audit.status == AuditStatus.PLANNED

    def test_audit_default_scope_is_full(self):
        audit = Audit(
            operator_id="OP-001",
            supplier_id="SUP-001",
            planned_date=FROZEN_DATE,
            country_code="BR",
            commodity="wood",
        )
        assert audit.audit_type == AuditScope.FULL

    @pytest.mark.parametrize("scope", [AuditScope.FULL, AuditScope.TARGETED, AuditScope.SURVEILLANCE, AuditScope.UNSCHEDULED])
    def test_all_audit_scopes(self, scope):
        audit = Audit(
            operator_id="OP-001", supplier_id="SUP-001",
            planned_date=FROZEN_DATE, country_code="BR",
            commodity="wood", audit_type=scope,
        )
        assert audit.audit_type == scope

    @pytest.mark.parametrize("modality", [AuditModality.ON_SITE, AuditModality.REMOTE, AuditModality.HYBRID, AuditModality.UNANNOUNCED])
    def test_all_audit_modalities(self, modality):
        audit = Audit(
            operator_id="OP-001", supplier_id="SUP-001",
            planned_date=FROZEN_DATE, country_code="BR",
            commodity="wood", modality=modality,
        )
        assert audit.modality == modality


class TestAuditorMatchEndpoint:
    """Test POST /v1/eudr-tam/auditors/match endpoint logic."""

    def test_match_auditors_returns_list(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001", commodity="wood", scheme="fsc",
            country_code="BR", required_language="pt", audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        assert isinstance(response.matched_auditors, list)

    def test_match_returns_scored_results(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001", commodity="wood", scheme="fsc",
            country_code="BR", required_language="pt", audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        for match in response.matched_auditors:
            assert "match_score" in match

    def test_match_excludes_inactive_auditors(self, auditor_registry_engine, sample_auditor_suspended):
        auditor_registry_engine.register_auditor(sample_auditor_suspended)
        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001", commodity="cocoa", scheme="rainforest_alliance",
            country_code="GH", required_language="en", audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        ids = [m["auditor_id"] for m in response.matched_auditors]
        assert sample_auditor_suspended.auditor_id not in ids


class TestAuditorCRUD:
    """Test auditor CRUD operations."""

    def test_create_auditor(self, sample_auditor_fsc):
        assert sample_auditor_fsc.full_name == "Maria Garcia"
        assert "FSC Lead Auditor" in sample_auditor_fsc.scheme_qualifications

    def test_auditor_requires_full_name(self):
        with pytest.raises((ValueError, Exception)):
            Auditor(full_name="", organization="Test Org")

    def test_auditor_requires_organization(self):
        with pytest.raises((ValueError, Exception)):
            Auditor(full_name="Test", organization="")

    def test_auditor_default_status_active(self):
        auditor = Auditor(full_name="Test", organization="Test Org")
        assert auditor.accreditation_status == "active"

    def test_auditor_performance_rating_range(self, sample_auditor_fsc):
        assert Decimal("0") <= sample_auditor_fsc.performance_rating <= Decimal("100")

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_auditor_with_each_commodity(self, commodity):
        auditor = Auditor(
            full_name="Test", organization="Test Org",
            commodity_competencies=[commodity],
        )
        assert commodity in auditor.commodity_competencies


class TestAuditListEndpoint:
    """Test GET /v1/eudr-tam/audits endpoint logic."""

    def test_list_audits_returns_list(self, batch_audits):
        assert isinstance(batch_audits, list)
        assert len(batch_audits) == 10

    def test_filter_by_status(self, batch_audits):
        planned = [a for a in batch_audits if a.status == AuditStatus.PLANNED]
        assert len(planned) == 10

    def test_filter_by_commodity(self, batch_audits):
        wood_audits = [a for a in batch_audits if a.commodity == "wood"]
        assert len(wood_audits) >= 1

    def test_filter_by_country(self, batch_audits):
        br_audits = [a for a in batch_audits if a.country_code == "BR"]
        assert len(br_audits) >= 1
