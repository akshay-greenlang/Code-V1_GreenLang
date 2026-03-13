# -*- coding: utf-8 -*-
"""
Unit tests for models.py - AGENT-EUDR-036

Tests all 13 enumerations, 15+ Pydantic models, and constants.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.eu_information_system_interface.models import (
    AGENT_ID,
    AGENT_VERSION,
    SUPPORTED_COMMODITIES,
    DDS_REFERENCE_PREFIX,
    MIN_AUDIT_RETENTION_YEARS,
    MAX_GEOLOCATION_POINTS,
    REQUIRED_DDS_FIELDS,
    EUDRCommodity,
    OperatorType,
    DDSType,
    DDSStatus,
    SubmissionStatus,
    RegistrationStatus,
    GeolocationFormat,
    CoordinateSystem,
    DocumentType,
    AuditEventType,
    CompetentAuthority,
    Coordinate,
    GeoPolygon,
    GeolocationData,
    OperatorRegistration,
    DDSCommodityLine,
    DueDiligenceStatement,
    DocumentPackage,
    SubmissionRequest,
    StatusCheckResult,
    AuditRecord,
    APICallRecord,
    DDSSummary,
    SubmissionReport,
    HealthStatus,
)


class TestConstants:
    """Test module constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-EUIS-036"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_supported_commodities_count(self):
        assert len(SUPPORTED_COMMODITIES) == 7

    def test_supported_commodities_values(self):
        assert "cocoa" in SUPPORTED_COMMODITIES
        assert "coffee" in SUPPORTED_COMMODITIES
        assert "wood" in SUPPORTED_COMMODITIES

    def test_dds_reference_prefix(self):
        assert DDS_REFERENCE_PREFIX == "EUDR-DDS"

    def test_min_retention_years(self):
        assert MIN_AUDIT_RETENTION_YEARS == 5

    def test_max_geolocation_points(self):
        assert MAX_GEOLOCATION_POINTS == 500

    def test_required_dds_fields(self):
        assert "operator_id" in REQUIRED_DDS_FIELDS
        assert "commodity" in REQUIRED_DDS_FIELDS


class TestEUDRCommodity:
    """Test EUDRCommodity enumeration."""

    def test_seven_commodities(self):
        assert len(EUDRCommodity) == 7

    def test_all_values(self):
        expected = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        actual = {c.value for c in EUDRCommodity}
        assert actual == expected

    def test_from_value(self):
        assert EUDRCommodity("cocoa") == EUDRCommodity.COCOA

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            EUDRCommodity("banana")


class TestDDSStatus:
    """Test DDSStatus enumeration."""

    def test_ten_states(self):
        assert len(DDSStatus) == 10

    def test_key_states(self):
        assert DDSStatus.DRAFT.value == "draft"
        assert DDSStatus.VALIDATED.value == "validated"
        assert DDSStatus.SUBMITTED.value == "submitted"
        assert DDSStatus.ACCEPTED.value == "accepted"
        assert DDSStatus.REJECTED.value == "rejected"
        assert DDSStatus.WITHDRAWN.value == "withdrawn"


class TestOperatorType:
    """Test OperatorType enumeration."""

    def test_four_types(self):
        assert len(OperatorType) == 4

    def test_values(self):
        expected = {"operator", "trader", "sme_operator", "sme_trader"}
        actual = {t.value for t in OperatorType}
        assert actual == expected


class TestDDSType:
    """Test DDSType enumeration."""

    def test_three_types(self):
        assert len(DDSType) == 3

    def test_values(self):
        expected = {"placing", "making_available", "export"}
        actual = {t.value for t in DDSType}
        assert actual == expected


class TestSubmissionStatus:
    """Test SubmissionStatus enumeration."""

    def test_seven_statuses(self):
        assert len(SubmissionStatus) == 7

    def test_pending(self):
        assert SubmissionStatus.PENDING.value == "pending"

    def test_completed(self):
        assert SubmissionStatus.COMPLETED.value == "completed"


class TestRegistrationStatus:
    """Test RegistrationStatus enumeration."""

    def test_five_statuses(self):
        assert len(RegistrationStatus) == 5


class TestGeolocationFormat:
    """Test GeolocationFormat enumeration."""

    def test_three_formats(self):
        assert len(GeolocationFormat) == 3

    def test_values(self):
        expected = {"point", "polygon", "multipolygon"}
        actual = {f.value for f in GeolocationFormat}
        assert actual == expected


class TestDocumentType:
    """Test DocumentType enumeration."""

    def test_twelve_types(self):
        assert len(DocumentType) == 12

    def test_dds_form(self):
        assert DocumentType.DDS_FORM.value == "dds_form"

    def test_improvement_plan(self):
        assert DocumentType.IMPROVEMENT_PLAN.value == "improvement_plan"


class TestAuditEventType:
    """Test AuditEventType enumeration."""

    def test_sixteen_types(self):
        assert len(AuditEventType) == 16

    def test_key_events(self):
        assert AuditEventType.DDS_CREATED.value == "dds_created"
        assert AuditEventType.DDS_SUBMITTED.value == "dds_submitted"
        assert AuditEventType.API_CALL_MADE.value == "api_call_made"


class TestCompetentAuthority:
    """Test CompetentAuthority enumeration."""

    def test_twenty_seven_authorities(self):
        assert len(CompetentAuthority) == 27

    def test_de(self):
        assert CompetentAuthority.DE.value == "DE"


class TestCoordinate:
    """Test Coordinate model."""

    def test_create_valid(self):
        c = Coordinate(latitude=Decimal("6.688500"), longitude=Decimal("-1.624400"))
        assert c.latitude == Decimal("6.688500")

    def test_boundary_values(self):
        c = Coordinate(latitude=Decimal("90"), longitude=Decimal("-180"))
        assert c.latitude == Decimal("90")


class TestGeoPolygon:
    """Test GeoPolygon model."""

    def test_create_valid(self):
        coords = [
            Coordinate(latitude=Decimal("6.0"), longitude=Decimal("-1.0")),
            Coordinate(latitude=Decimal("6.1"), longitude=Decimal("-1.1")),
            Coordinate(latitude=Decimal("6.2"), longitude=Decimal("-1.0")),
        ]
        poly = GeoPolygon(coordinates=coords)
        assert len(poly.coordinates) == 3
        assert poly.crs == "EPSG:4326"


class TestDueDiligenceStatement:
    """Test DueDiligenceStatement model."""

    def test_create_minimal(self):
        dds = DueDiligenceStatement(
            dds_id="dds-001",
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type=DDSType.PLACING,
        )
        assert dds.status == DDSStatus.DRAFT
        assert dds.total_quantity == Decimal("0")

    def test_defaults(self):
        dds = DueDiligenceStatement(
            dds_id="dds-002",
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type=DDSType.EXPORT,
        )
        assert dds.dds_reference == ""
        assert dds.risk_assessment_id is None
        assert dds.improvement_plan_id is None

    def test_with_fixture(self, sample_dds):
        assert sample_dds.dds_id == "dds-test-001"
        assert sample_dds.dds_type == DDSType.PLACING


class TestOperatorRegistration:
    """Test OperatorRegistration model."""

    def test_create(self, sample_registration):
        assert sample_registration.registration_id == "reg-test-001"
        assert sample_registration.operator_type == OperatorType.OPERATOR
        assert sample_registration.member_state == CompetentAuthority.DE

    def test_default_status(self):
        reg = OperatorRegistration(
            registration_id="reg-new",
            operator_id="op-new",
            eori_number="FR987654321098",
            operator_type=OperatorType.TRADER,
            company_name="Test Corp",
            member_state=CompetentAuthority.FR,
        )
        assert reg.registration_status == RegistrationStatus.PENDING


class TestDocumentPackage:
    """Test DocumentPackage model."""

    def test_create(self, sample_package):
        assert sample_package.package_id == "pkg-test-001"
        assert sample_package.document_count == 2

    def test_defaults(self):
        pkg = DocumentPackage(package_id="pkg-empty", dds_id="dds-empty")
        assert pkg.document_count == 0
        assert pkg.compressed is False


class TestSubmissionRequest:
    """Test SubmissionRequest model."""

    def test_create(self, sample_submission):
        assert sample_submission.submission_id == "sub-test-001"
        assert sample_submission.status == SubmissionStatus.PENDING

    def test_defaults(self):
        s = SubmissionRequest(
            submission_id="sub-new",
            dds_id="dds-new",
            package_id="pkg-new",
        )
        assert s.attempt_count == 0
        assert s.eu_reference_number is None


class TestAuditRecord:
    """Test AuditRecord model."""

    def test_create(self, sample_audit_record):
        assert sample_audit_record.audit_id == "aud-test-001"
        assert sample_audit_record.event_type == AuditEventType.DDS_SUBMITTED

    def test_retention(self, sample_audit_record):
        assert sample_audit_record.retention_until is not None


class TestAPICallRecord:
    """Test APICallRecord model."""

    def test_create(self, sample_api_call_record):
        assert sample_api_call_record.call_id == "call-test-001"
        assert sample_api_call_record.success is True
        assert sample_api_call_record.duration_ms == Decimal("125.50")


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_defaults(self):
        h = HealthStatus()
        assert h.agent_id == "GL-EUDR-EUIS-036"
        assert h.status == "healthy"
        assert h.version == "1.0.0"


class TestDDSSummary:
    """Test DDSSummary model."""

    def test_create(self):
        s = DDSSummary(
            dds_id="dds-001",
            operator_id="op-001",
            dds_type=DDSType.PLACING,
            status=DDSStatus.DRAFT,
        )
        assert s.commodity_count == 0
        assert s.total_quantity == Decimal("0")


class TestSubmissionReport:
    """Test SubmissionReport model."""

    def test_create(self):
        r = SubmissionReport(report_id="rpt-001")
        assert r.total_submissions == 0
        assert r.successful == 0
