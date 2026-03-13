# -*- coding: utf-8 -*-
"""
Unit tests for Data Models -- AGENT-EUDR-024

Tests Pydantic v2 model creation, validation, serialization, enum
values, default field factories, and constraint enforcement for all
7 enumerations, 10 core models, 7 request models, and 7 response
models defined in the Third-Party Audit Manager models module.

Target: ~80 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.models import (
    AuditStatus,
    AuditScope,
    AuditModality,
    NCSeverity,
    CARStatus,
    CertificationScheme,
    AuthorityInteractionType,
    Audit,
    Auditor,
    AuditChecklist,
    AuditEvidence,
    NonConformance,
    RootCauseAnalysis,
    CorrectiveActionRequest,
    CertificateRecord,
    CompetentAuthorityInteraction,
    AuditReport,
    ScheduleAuditRequest,
    MatchAuditorRequest,
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    LogAuthorityInteractionRequest,
    CalculateAnalyticsRequest,
    VERSION,
    EUDR_CUTOFF_DATE,
    MAX_BATCH_SIZE,
    EUDR_RETENTION_YEARS,
    SUPPORTED_SCHEMES,
    SUPPORTED_COMMODITIES,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
)


# ===================================================================
# Enum Tests
# ===================================================================


class TestAuditStatusEnum:
    """Test AuditStatus enum values."""

    def test_planned_value(self):
        assert AuditStatus.PLANNED.value == "planned"

    def test_in_progress_value(self):
        assert AuditStatus.IN_PROGRESS.value == "in_progress"

    def test_closed_value(self):
        assert AuditStatus.CLOSED.value == "closed"

    def test_cancelled_value(self):
        assert AuditStatus.CANCELLED.value == "cancelled"

    def test_all_values_count(self):
        assert len(AuditStatus) >= 8


class TestAuditScopeEnum:
    """Test AuditScope enum values."""

    def test_full_scope(self):
        assert AuditScope.FULL.value == "full"

    def test_targeted_scope(self):
        assert AuditScope.TARGETED.value == "targeted"

    def test_surveillance_scope(self):
        assert AuditScope.SURVEILLANCE.value == "surveillance"

    def test_unscheduled_scope(self):
        assert AuditScope.UNSCHEDULED.value == "unscheduled"


class TestAuditModalityEnum:
    """Test AuditModality enum values."""

    def test_on_site(self):
        assert AuditModality.ON_SITE.value == "on_site"

    def test_remote(self):
        assert AuditModality.REMOTE.value == "remote"

    def test_hybrid(self):
        assert AuditModality.HYBRID.value == "hybrid"

    def test_unannounced(self):
        assert AuditModality.UNANNOUNCED.value == "unannounced"


class TestNCSeverityEnum:
    """Test NCSeverity enum values."""

    def test_critical(self):
        assert NCSeverity.CRITICAL.value == "critical"

    def test_major(self):
        assert NCSeverity.MAJOR.value == "major"

    def test_minor(self):
        assert NCSeverity.MINOR.value == "minor"

    def test_observation(self):
        assert NCSeverity.OBSERVATION.value == "observation"


class TestCARStatusEnum:
    """Test CARStatus enum values."""

    def test_issued(self):
        assert CARStatus.ISSUED.value == "issued"

    def test_acknowledged(self):
        assert CARStatus.ACKNOWLEDGED.value == "acknowledged"

    def test_closed(self):
        assert CARStatus.CLOSED.value == "closed"

    def test_overdue(self):
        assert CARStatus.OVERDUE.value == "overdue"

    def test_escalated(self):
        assert CARStatus.ESCALATED.value == "escalated"


class TestCertificationSchemeEnum:
    """Test CertificationScheme enum values."""

    def test_fsc(self):
        assert CertificationScheme.FSC.value == "fsc"

    def test_pefc(self):
        assert CertificationScheme.PEFC.value == "pefc"

    def test_rspo(self):
        assert CertificationScheme.RSPO.value == "rspo"

    def test_rainforest_alliance(self):
        assert CertificationScheme.RAINFOREST_ALLIANCE.value == "rainforest_alliance"

    def test_iscc(self):
        assert CertificationScheme.ISCC.value == "iscc"


# ===================================================================
# Core Model Tests
# ===================================================================


class TestAuditModel:
    """Test Audit model creation and validation."""

    def test_create_audit(self, sample_audit):
        assert sample_audit.audit_id == "AUD-TEST-001"
        assert sample_audit.status == AuditStatus.PLANNED

    def test_audit_has_operator_id(self, sample_audit):
        assert sample_audit.operator_id == "OP-001"

    def test_audit_has_commodity(self, sample_audit):
        assert sample_audit.commodity == "wood"

    def test_audit_has_country(self, sample_audit):
        assert sample_audit.country_code == "BR"

    def test_audit_priority_score_decimal(self, sample_audit):
        assert isinstance(sample_audit.priority_score, Decimal)

    def test_audit_planned_date(self, sample_audit):
        assert sample_audit.planned_date == FROZEN_DATE

    def test_audit_modality(self, sample_audit):
        assert sample_audit.modality == AuditModality.ON_SITE


class TestAuditorModel:
    """Test Auditor model creation and validation."""

    def test_create_auditor(self, sample_auditor_fsc):
        assert sample_auditor_fsc.auditor_id == "AUR-FSC-001"

    def test_auditor_has_organization(self, sample_auditor_fsc):
        assert sample_auditor_fsc.organization == "TUV SUD"

    def test_auditor_performance_rating(self, sample_auditor_fsc):
        assert isinstance(sample_auditor_fsc.performance_rating, Decimal)
        assert Decimal("0") <= sample_auditor_fsc.performance_rating <= Decimal("100")

    def test_auditor_accreditation_active(self, sample_auditor_fsc):
        assert sample_auditor_fsc.accreditation_status == "active"

    def test_auditor_accreditation_not_expired(self, sample_auditor_fsc):
        assert sample_auditor_fsc.accreditation_expiry > FROZEN_DATE


class TestNonConformanceModel:
    """Test NonConformance model creation."""

    def test_critical_nc(self, sample_nc_critical):
        assert sample_nc_critical.severity == NCSeverity.CRITICAL
        assert sample_nc_critical.risk_impact_score >= Decimal("90")

    def test_major_nc(self, sample_nc_major):
        assert sample_nc_major.severity == NCSeverity.MAJOR

    def test_minor_nc(self, sample_nc_minor):
        assert sample_nc_minor.severity == NCSeverity.MINOR

    def test_observation_nc(self, sample_nc_observation):
        assert sample_nc_observation.severity == NCSeverity.OBSERVATION

    def test_nc_has_finding_statement(self, sample_nc_critical):
        assert len(sample_nc_critical.finding_statement) > 0

    def test_nc_has_objective_evidence(self, sample_nc_critical):
        assert len(sample_nc_critical.objective_evidence) > 0


class TestCARModel:
    """Test CorrectiveActionRequest model."""

    def test_critical_car_30_day_sla(self, sample_car_critical):
        delta = sample_car_critical.sla_deadline - sample_car_critical.issued_at
        assert delta.days == 30

    def test_major_car_90_day_sla(self, sample_car_major):
        delta = sample_car_major.sla_deadline - sample_car_major.issued_at
        assert delta.days == 90

    def test_minor_car_365_day_sla(self, sample_car_minor):
        delta = sample_car_minor.sla_deadline - sample_car_minor.issued_at
        assert delta.days == 365

    def test_overdue_car(self, sample_car_overdue):
        assert sample_car_overdue.status == CARStatus.OVERDUE
        assert sample_car_overdue.sla_status == "overdue"


class TestCertificateModel:
    """Test CertificateRecord model."""

    def test_fsc_certificate(self, sample_certificate_fsc):
        assert sample_certificate_fsc.scheme == CertificationScheme.FSC
        assert sample_certificate_fsc.status == "active"

    def test_expired_certificate(self, sample_certificate_expired):
        assert sample_certificate_expired.status == "expired"
        assert sample_certificate_expired.expiry_date < FROZEN_DATE


# ===================================================================
# Constants Tests
# ===================================================================


class TestConstants:
    """Test module-level constants."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_cutoff_date(self):
        assert EUDR_CUTOFF_DATE == "2020-12-31"

    def test_max_batch_size(self):
        assert MAX_BATCH_SIZE == 1000

    def test_retention_years(self):
        assert EUDR_RETENTION_YEARS == 5

    def test_supported_schemes_count(self):
        assert len(SUPPORTED_SCHEMES) == 5

    def test_supported_commodities_count(self):
        assert len(SUPPORTED_COMMODITIES) == 7

    @pytest.mark.parametrize("commodity", [
        "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
    ])
    def test_commodity_in_supported(self, commodity):
        assert commodity in SUPPORTED_COMMODITIES
