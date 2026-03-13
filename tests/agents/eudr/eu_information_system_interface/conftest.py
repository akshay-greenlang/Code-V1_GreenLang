# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-036 EU Information System Interface tests.

Provides reusable test fixtures for config, models, provenance tracking,
DDS records, operator registrations, geolocation data, document packages,
status checks, audit records, and API call records.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
    get_config,
    reset_config,
)
from greenlang.agents.eudr.eu_information_system_interface.models import (
    AGENT_ID,
    AGENT_VERSION,
    AuditEventType,
    AuditRecord,
    APICallRecord,
    CompetentAuthority,
    Coordinate,
    DDSCommodityLine,
    DDSStatus,
    DDSType,
    DocumentPackage,
    DocumentType,
    DueDiligenceStatement,
    EUDRCommodity,
    GeolocationData,
    GeolocationFormat,
    GeoPolygon,
    OperatorRegistration,
    OperatorType,
    RegistrationStatus,
    StatusCheckResult,
    SubmissionRequest,
    SubmissionStatus,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ---------------------------------------------------------------------------
# Auto-reset config singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> EUInformationSystemInterfaceConfig:
    """Create a default EUInformationSystemInterfaceConfig instance."""
    return EUInformationSystemInterfaceConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# DDS fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_commodity_line_data() -> List[Dict]:
    """Create sample commodity line dictionaries for DDS creation."""
    return [
        {
            "commodity": "cocoa",
            "description": "Cocoa beans, whole, unroasted",
            "hs_code": "1801.00",
            "quantity": "50000.00",
            "unit": "kg",
            "country_of_production": "GH",
            "geolocation": {
                "format": "point",
                "point": {"latitude": "6.6885", "longitude": "-1.6244"},
                "country_code": "GH",
            },
            "supplier_ids": ["sup-001", "sup-002"],
            "risk_assessment_conclusion": "negligible",
        },
        {
            "commodity": "cocoa",
            "description": "Cocoa butter",
            "hs_code": "1804.00",
            "quantity": "25000.00",
            "unit": "kg",
            "country_of_production": "GH",
            "geolocation": {
                "format": "point",
                "point": {"latitude": "6.6912", "longitude": "-1.6198"},
                "country_code": "GH",
            },
            "supplier_ids": ["sup-001"],
            "risk_assessment_conclusion": "low",
        },
    ]


@pytest.fixture
def sample_dds() -> DueDiligenceStatement:
    """Create a sample DDS in DRAFT status."""
    now = datetime.now(tz=timezone.utc)
    return DueDiligenceStatement(
        dds_id="dds-test-001",
        operator_id="operator-001",
        eori_number="DE123456789012",
        dds_type=DDSType.PLACING,
        status=DDSStatus.DRAFT,
        total_quantity=Decimal("75000.00"),
        risk_assessment_id="ra-001",
        created_at=now,
        updated_at=now,
        provenance_hash="a" * 64,
    )


@pytest.fixture
def validated_dds() -> DueDiligenceStatement:
    """Create a DDS in VALIDATED status."""
    now = datetime.now(tz=timezone.utc)
    return DueDiligenceStatement(
        dds_id="dds-validated-001",
        operator_id="operator-001",
        eori_number="DE123456789012",
        dds_type=DDSType.PLACING,
        status=DDSStatus.VALIDATED,
        total_quantity=Decimal("50000.00"),
        risk_assessment_id="ra-001",
        created_at=now - timedelta(days=1),
        updated_at=now,
        provenance_hash="b" * 64,
    )


@pytest.fixture
def submitted_dds() -> DueDiligenceStatement:
    """Create a DDS in SUBMITTED status."""
    now = datetime.now(tz=timezone.utc)
    return DueDiligenceStatement(
        dds_id="dds-submitted-001",
        dds_reference="EUDR-2026-DE-00012345",
        operator_id="operator-001",
        eori_number="DE123456789012",
        dds_type=DDSType.PLACING,
        status=DDSStatus.SUBMITTED,
        total_quantity=Decimal("25000.00"),
        submitted_at=now - timedelta(hours=6),
        created_at=now - timedelta(days=3),
        updated_at=now,
        provenance_hash="c" * 64,
    )


@pytest.fixture
def accepted_dds() -> DueDiligenceStatement:
    """Create a DDS in ACCEPTED status."""
    now = datetime.now(tz=timezone.utc)
    return DueDiligenceStatement(
        dds_id="dds-accepted-001",
        dds_reference="EUDR-2026-NL-00098765",
        operator_id="operator-002",
        eori_number="NL987654321098",
        dds_type=DDSType.EXPORT,
        status=DDSStatus.ACCEPTED,
        total_quantity=Decimal("15000.00"),
        submitted_at=now - timedelta(days=5),
        accepted_at=now - timedelta(days=3),
        created_at=now - timedelta(days=7),
        updated_at=now,
        provenance_hash="d" * 64,
    )


# ---------------------------------------------------------------------------
# Operator Registration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_registration() -> OperatorRegistration:
    """Create a sample operator registration in PENDING status."""
    now = datetime.now(tz=timezone.utc)
    return OperatorRegistration(
        registration_id="reg-test-001",
        operator_id="operator-001",
        eori_number="DE123456789012",
        operator_type=OperatorType.OPERATOR,
        company_name="Green Trading GmbH",
        member_state=CompetentAuthority.DE,
        address="Musterstr. 123, 10115 Berlin",
        contact_email="compliance@greentrading.de",
        registration_status=RegistrationStatus.PENDING,
        registered_at=now,
        expires_at=now + timedelta(days=365),
        provenance_hash="e" * 64,
    )


@pytest.fixture
def active_registration() -> OperatorRegistration:
    """Create an active operator registration."""
    now = datetime.now(tz=timezone.utc)
    return OperatorRegistration(
        registration_id="reg-active-001",
        operator_id="operator-001",
        eori_number="DE123456789012",
        operator_type=OperatorType.OPERATOR,
        company_name="Green Trading GmbH",
        member_state=CompetentAuthority.DE,
        registration_status=RegistrationStatus.ACTIVE,
        eu_system_id="EUIS-DE-2026-00001",
        registered_at=now - timedelta(days=30),
        expires_at=now + timedelta(days=335),
        provenance_hash="f" * 64,
    )


# ---------------------------------------------------------------------------
# Coordinate and Geolocation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_coordinates() -> List[Dict]:
    """Create sample coordinate dictionaries."""
    return [
        {"lat": 6.688500, "lng": -1.624400},
        {"lat": 6.691200, "lng": -1.619800},
        {"lat": 6.689000, "lng": -1.615000},
        {"lat": 6.685000, "lng": -1.620000},
    ]


@pytest.fixture
def single_coordinate() -> List[Dict]:
    """Create a single point coordinate."""
    return [{"lat": 4.5709, "lng": -75.6701}]


# ---------------------------------------------------------------------------
# Document Package fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> List[Dict]:
    """Create sample document list for package assembly."""
    return [
        {
            "type": "dds_form",
            "title": "Due Diligence Statement",
            "content": {"operator_id": "OP001", "commodity": "cocoa"},
            "size_bytes": 2048,
        },
        {
            "type": "geolocation_data",
            "title": "Plot Geolocation Data",
            "content": {"lat": 6.688, "lng": -1.624},
            "size_bytes": 1024,
        },
        {
            "type": "risk_assessment",
            "title": "Risk Assessment Report",
            "content": {"conclusion": "negligible"},
            "size_bytes": 4096,
        },
    ]


@pytest.fixture
def sample_package() -> DocumentPackage:
    """Create a sample document package."""
    now = datetime.now(tz=timezone.utc)
    return DocumentPackage(
        package_id="pkg-test-001",
        dds_id="dds-test-001",
        documents=[
            {"type": "dds_form", "title": "DDS", "size_bytes": 2048, "hash": "a" * 64},
            {"type": "geolocation_data", "title": "Geo", "size_bytes": 1024, "hash": "b" * 64},
        ],
        total_size_bytes=3072,
        document_count=2,
        compressed=False,
        assembled_at=now,
        provenance_hash="g" * 64,
    )


# ---------------------------------------------------------------------------
# Submission Request fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_submission() -> SubmissionRequest:
    """Create a sample submission request."""
    now = datetime.now(tz=timezone.utc)
    return SubmissionRequest(
        submission_id="sub-test-001",
        dds_id="dds-test-001",
        package_id="pkg-test-001",
        status=SubmissionStatus.PENDING,
        attempt_count=0,
        created_at=now,
        provenance_hash="h" * 64,
    )


@pytest.fixture
def old_submission() -> SubmissionRequest:
    """Create a submission that has been pending for a long time."""
    now = datetime.now(tz=timezone.utc)
    return SubmissionRequest(
        submission_id="sub-old-001",
        dds_id="dds-old-001",
        package_id="pkg-old-001",
        status=SubmissionStatus.PENDING,
        attempt_count=2,
        created_at=now - timedelta(hours=100),
        provenance_hash="i" * 64,
    )


# ---------------------------------------------------------------------------
# Audit Record fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_audit_record() -> AuditRecord:
    """Create a sample audit record."""
    now = datetime.now(tz=timezone.utc)
    return AuditRecord(
        audit_id="aud-test-001",
        event_type=AuditEventType.DDS_SUBMITTED,
        entity_type="dds",
        entity_id="dds-test-001",
        actor="system",
        action="submit",
        details={"commodity": "cocoa"},
        timestamp=now,
        retention_until=now + timedelta(days=1825),
        provenance_hash="j" * 64,
    )


# ---------------------------------------------------------------------------
# API Call Record fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_api_call_record() -> APICallRecord:
    """Create a sample API call record."""
    return APICallRecord(
        call_id="call-test-001",
        method="POST",
        endpoint="/dds/submit",
        status_code=200,
        duration_ms=Decimal("125.50"),
        success=True,
        retry_count=0,
    )
