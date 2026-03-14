# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-024 Third-Party Audit Manager test suite.

Provides reusable fixtures for configuration objects, engine instances,
audit records, auditor profiles, non-conformance samples, CAR lifecycle
data, certification scheme records, authority interactions, analytics
inputs, provenance tracking helpers, and shared constants used across
all test modules.

Fixture count: 75+ fixtures
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.third_party_audit_manager.config import (
    ThirdPartyAuditManagerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    # Enumerations
    AuditStatus,
    AuditScope,
    AuditModality,
    NCSeverity,
    CARStatus,
    CertificationScheme,
    AuthorityInteractionType,
    # Core Models
    Audit,
    Auditor,
    AuditChecklist,
    AuditEvidence,
    RootCauseAnalysis,
    NonConformance,
    CorrectiveActionRequest,
    CertificateRecord,
    CompetentAuthorityInteraction,
    AuditReport,
    # Request Models
    ScheduleAuditRequest,
    MatchAuditorRequest,
    ClassifyNCRequest,
    IssueCARRequest,
    GenerateReportRequest,
    LogAuthorityInteractionRequest,
    CalculateAnalyticsRequest,
    # Response Models
    ScheduleAuditResponse,
    MatchAuditorResponse,
    ClassifyNCResponse,
    IssueCARResponse,
    GenerateReportResponse,
    LogAuthorityInteractionResponse,
    CalculateAnalyticsResponse,
    # Constants
    VERSION,
    EUDR_CUTOFF_DATE,
    MAX_BATCH_SIZE,
    EUDR_RETENTION_YEARS,
    SUPPORTED_SCHEMES,
    SUPPORTED_COMMODITIES,
    SUPPORTED_REPORT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
    NC_SEVERITY_SLA_DAYS,
    EU_MEMBER_STATES,
)
from greenlang.agents.eudr.third_party_audit_manager.provenance import (
    ProvenanceRecord,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_tracker,
    reset_tracker,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_planning_scheduling_engine import (
    AuditPlanningSchedulingEngine,
    FREQUENCY_INTERVALS,
    FREQUENCY_MODALITY,
    FREQUENCY_SCOPE,
    SCHEME_RECERTIFICATION_CYCLES,
)
from greenlang.agents.eudr.third_party_audit_manager.auditor_registry_qualification_engine import (
    AuditorRegistryQualificationEngine,
    MATCH_WEIGHTS,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_execution_engine import (
    AuditExecutionEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.non_conformance_detection_engine import (
    NonConformanceDetectionEngine,
    CRITICAL_RULES,
    MAJOR_RULES,
)
from greenlang.agents.eudr.third_party_audit_manager.car_management_engine import (
    CARManagementEngine,
    VALID_CAR_TRANSITIONS,
)
from greenlang.agents.eudr.third_party_audit_manager.certification_integration_engine import (
    CertificationIntegrationEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_reporting_engine import (
    AuditReportingEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.audit_analytics_engine import (
    AuditAnalyticsEngine,
)


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------


class DeterministicUUID:
    """Generate sequential identifiers for deterministic testing."""

    def __init__(self, prefix: str = "test"):
        self._counter = 0
        self._prefix = prefix

    def next(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter:08d}"

    def reset(self):
        self._counter = 0


# ---------------------------------------------------------------------------
# Provenance hash helper
# ---------------------------------------------------------------------------


def compute_test_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH = 64
EUDR_DEFORESTATION_CUTOFF = "2020-12-31"
EUDR_CUTOFF_DATE_OBJ = date(2020, 12, 31)

EUDR_COMMODITIES = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

CERTIFICATION_SCHEMES = ["fsc", "pefc", "rspo", "rainforest_alliance", "iscc"]

NC_SEVERITIES = ["critical", "major", "minor", "observation"]

AUDIT_STATUSES = [
    "planned", "auditor_assigned", "in_preparation", "in_progress",
    "fieldwork_complete", "report_drafting", "report_issued",
    "car_follow_up", "closed", "cancelled",
]

CAR_STATUSES = [
    "issued", "acknowledged", "rca_submitted", "cap_submitted",
    "cap_approved", "in_progress", "evidence_submitted",
    "verification_pending", "closed", "rejected", "overdue", "escalated",
]

REPORT_FORMATS = ["pdf", "json", "html", "xlsx", "xml"]
REPORT_LANGUAGES = ["en", "fr", "de", "es", "pt"]

# Sample country codes for testing
HIGH_RISK_COUNTRIES = ["BR", "ID", "CO", "GH", "CI"]
STANDARD_RISK_COUNTRIES = ["MY", "PE", "EC", "NG", "CM"]
LOW_RISK_COUNTRIES = ["FI", "SE", "CA", "NZ", "CL"]

# Sample authority profiles
SAMPLE_AUTHORITIES = {
    "DE": "BMEL",
    "FR": "DGCCRF",
    "NL": "NVWA",
    "BE": "FOD Economie",
    "IT": "Ministero dell'Ambiente",
}


# ---------------------------------------------------------------------------
# Frozen datetime for deterministic testing
# ---------------------------------------------------------------------------

FROZEN_NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
FROZEN_DATE = date(2026, 3, 10)
FROZEN_PAST_30D = FROZEN_NOW - timedelta(days=30)
FROZEN_PAST_90D = FROZEN_NOW - timedelta(days=90)
FROZEN_PAST_365D = FROZEN_NOW - timedelta(days=365)
FROZEN_FUTURE_30D = FROZEN_NOW + timedelta(days=30)
FROZEN_FUTURE_90D = FROZEN_NOW + timedelta(days=90)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def default_config() -> ThirdPartyAuditManagerConfig:
    """Return a default ThirdPartyAuditManagerConfig instance."""
    cfg = get_config()
    return cfg


@pytest.fixture
def custom_config() -> ThirdPartyAuditManagerConfig:
    """Return a custom configuration with modified weights."""
    cfg = get_config()
    cfg.country_risk_weight = Decimal("0.30")
    cfg.supplier_risk_weight = Decimal("0.20")
    cfg.nc_history_weight = Decimal("0.20")
    cfg.certification_gap_weight = Decimal("0.15")
    cfg.deforestation_alert_weight = Decimal("0.15")
    return cfg


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def planning_engine(default_config) -> AuditPlanningSchedulingEngine:
    """Return an AuditPlanningSchedulingEngine instance."""
    return AuditPlanningSchedulingEngine(config=default_config)


@pytest.fixture
def auditor_registry_engine(default_config) -> AuditorRegistryQualificationEngine:
    """Return an AuditorRegistryQualificationEngine instance."""
    return AuditorRegistryQualificationEngine(config=default_config)


@pytest.fixture
def execution_engine(default_config) -> AuditExecutionEngine:
    """Return an AuditExecutionEngine instance."""
    return AuditExecutionEngine(config=default_config)


@pytest.fixture
def nc_engine(default_config) -> NonConformanceDetectionEngine:
    """Return a NonConformanceDetectionEngine instance."""
    return NonConformanceDetectionEngine(config=default_config)


@pytest.fixture
def car_engine(default_config) -> CARManagementEngine:
    """Return a CARManagementEngine instance."""
    return CARManagementEngine(config=default_config)


@pytest.fixture
def certification_engine(default_config) -> CertificationIntegrationEngine:
    """Return a CertificationIntegrationEngine instance."""
    return CertificationIntegrationEngine(config=default_config)


@pytest.fixture
def reporting_engine(default_config) -> AuditReportingEngine:
    """Return an AuditReportingEngine instance."""
    return AuditReportingEngine(config=default_config)


@pytest.fixture
def analytics_engine(default_config) -> AuditAnalyticsEngine:
    """Return an AuditAnalyticsEngine instance."""
    return AuditAnalyticsEngine(config=default_config)


# ---------------------------------------------------------------------------
# UUID Generator Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def uuid_gen() -> DeterministicUUID:
    """Return a deterministic UUID generator."""
    return DeterministicUUID(prefix="test")


# ---------------------------------------------------------------------------
# Provenance Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Return a fresh ProvenanceTracker instance."""
    reset_tracker()
    return get_tracker()


# ---------------------------------------------------------------------------
# Audit Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audit() -> Audit:
    """Return a sample Audit record for testing."""
    return Audit(
        audit_id="AUD-TEST-001",
        operator_id="OP-001",
        supplier_id="SUP-001",
        audit_type=AuditScope.FULL,
        modality=AuditModality.ON_SITE,
        certification_scheme=CertificationScheme.FSC,
        eudr_articles=["Art. 3", "Art. 4", "Art. 9", "Art. 10", "Art. 11"],
        planned_date=FROZEN_DATE,
        status=AuditStatus.PLANNED,
        priority_score=Decimal("75.50"),
        country_code="BR",
        commodity="wood",
        site_ids=["SITE-001", "SITE-002"],
        estimated_duration_days=5,
    )


@pytest.fixture
def sample_audit_in_progress() -> Audit:
    """Return an in-progress audit record."""
    return Audit(
        audit_id="AUD-TEST-002",
        operator_id="OP-001",
        supplier_id="SUP-002",
        audit_type=AuditScope.TARGETED,
        modality=AuditModality.HYBRID,
        eudr_articles=["Art. 9", "Art. 10"],
        planned_date=FROZEN_DATE - timedelta(days=5),
        actual_start_date=FROZEN_DATE - timedelta(days=3),
        status=AuditStatus.IN_PROGRESS,
        priority_score=Decimal("55.00"),
        country_code="ID",
        commodity="palm_oil",
        checklist_completion=Decimal("45.00"),
    )


@pytest.fixture
def sample_audit_closed() -> Audit:
    """Return a closed audit record."""
    return Audit(
        audit_id="AUD-TEST-003",
        operator_id="OP-001",
        supplier_id="SUP-003",
        audit_type=AuditScope.SURVEILLANCE,
        modality=AuditModality.REMOTE,
        planned_date=FROZEN_DATE - timedelta(days=60),
        actual_start_date=FROZEN_DATE - timedelta(days=55),
        actual_end_date=FROZEN_DATE - timedelta(days=50),
        status=AuditStatus.CLOSED,
        priority_score=Decimal("30.00"),
        country_code="FI",
        commodity="wood",
        checklist_completion=Decimal("100.00"),
        findings_count={"critical": 0, "major": 1, "minor": 2, "observation": 1},
    )


# ---------------------------------------------------------------------------
# Auditor Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_auditor_fsc() -> Auditor:
    """Return a qualified FSC lead auditor."""
    return Auditor(
        auditor_id="AUR-FSC-001",
        full_name="Maria Garcia",
        organization="TUV SUD",
        accreditation_body="DAkkS",
        accreditation_status="active",
        accreditation_expiry=date(2027, 12, 31),
        accreditation_scope=["forest_management", "chain_of_custody"],
        commodity_competencies=["wood", "rubber"],
        scheme_qualifications=["FSC Lead Auditor", "PEFC Auditor"],
        country_expertise=["BR", "PE", "CO"],
        languages=["en", "es", "pt"],
        audit_count=150,
        performance_rating=Decimal("88.50"),
        findings_per_audit=Decimal("4.2"),
        car_closure_rate=Decimal("92.00"),
        cpd_hours=48,
        cpd_compliant=True,
        contact_email="m.garcia@tuvsud.com",
        available_from=FROZEN_DATE,
    )


@pytest.fixture
def sample_auditor_rspo() -> Auditor:
    """Return a qualified RSPO lead auditor."""
    return Auditor(
        auditor_id="AUR-RSPO-001",
        full_name="Budi Santoso",
        organization="Control Union",
        accreditation_body="RvA",
        accreditation_status="active",
        accreditation_expiry=date(2027, 6, 30),
        commodity_competencies=["palm_oil"],
        scheme_qualifications=["RSPO Lead Auditor", "ISCC Auditor"],
        country_expertise=["ID", "MY"],
        languages=["en", "id", "ms"],
        audit_count=200,
        performance_rating=Decimal("91.00"),
        cpd_hours=52,
        cpd_compliant=True,
        available_from=FROZEN_DATE,
    )


@pytest.fixture
def sample_auditor_suspended() -> Auditor:
    """Return an auditor with suspended accreditation."""
    return Auditor(
        auditor_id="AUR-SUSP-001",
        full_name="Suspended Auditor",
        organization="Audit Corp",
        accreditation_status="suspended",
        accreditation_expiry=date(2025, 12, 31),
        commodity_competencies=["cocoa"],
        scheme_qualifications=["RA Auditor"],
        country_expertise=["GH"],
        languages=["en"],
        cpd_compliant=False,
    )


@pytest.fixture
def sample_auditor_expired() -> Auditor:
    """Return an auditor with expired accreditation."""
    return Auditor(
        auditor_id="AUR-EXP-001",
        full_name="Expired Auditor",
        organization="Audit Corp",
        accreditation_status="active",
        accreditation_expiry=date(2025, 1, 1),
        commodity_competencies=["coffee"],
        scheme_qualifications=["RA Auditor"],
        country_expertise=["CO"],
        languages=["en", "es"],
        cpd_compliant=True,
    )


@pytest.fixture
def sample_auditor_pool(
    sample_auditor_fsc,
    sample_auditor_rspo,
    sample_auditor_suspended,
    sample_auditor_expired,
) -> List[Auditor]:
    """Return a pool of auditors with mixed qualifications."""
    return [
        sample_auditor_fsc,
        sample_auditor_rspo,
        sample_auditor_suspended,
        sample_auditor_expired,
    ]


# ---------------------------------------------------------------------------
# Non-Conformance Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_nc_critical() -> NonConformance:
    """Return a critical non-conformance finding."""
    return NonConformance(
        nc_id="NC-CRIT-001",
        audit_id="AUD-TEST-001",
        finding_statement="Evidence of active deforestation detected on supplier plots after the December 31, 2020 cutoff date",
        objective_evidence="Satellite imagery from Sentinel-2 showing forest cover loss between 2021-06 and 2022-03 on plot PLT-BR-042",
        severity=NCSeverity.CRITICAL,
        eudr_article="Art. 3(a)",
        scheme_clause="FSC P9",
        article_2_40_category="environmental_crime",
        risk_impact_score=Decimal("95.00"),
        status="open",
        classification_rule="CRIT-003",
    )


@pytest.fixture
def sample_nc_major() -> NonConformance:
    """Return a major non-conformance finding."""
    return NonConformance(
        nc_id="NC-MAJ-001",
        audit_id="AUD-TEST-001",
        finding_statement="Incomplete risk assessment documentation for high-risk country supply chain per Article 10",
        objective_evidence="Risk assessment file REF-2025-042 missing country-specific risk evaluation for Indonesia",
        severity=NCSeverity.MAJOR,
        eudr_article="Art. 10",
        risk_impact_score=Decimal("65.00"),
        status="open",
        classification_rule="MAJ-001",
    )


@pytest.fixture
def sample_nc_minor() -> NonConformance:
    """Return a minor non-conformance finding."""
    return NonConformance(
        nc_id="NC-MIN-001",
        audit_id="AUD-TEST-001",
        finding_statement="Training records for supply chain staff not updated within the required 12-month cycle",
        objective_evidence="Last training date for 3 staff members was 15 months ago per HR system export",
        severity=NCSeverity.MINOR,
        eudr_article="Art. 2(40) Cat 5",
        risk_impact_score=Decimal("25.00"),
        status="open",
        classification_rule="MN-004",
    )


@pytest.fixture
def sample_nc_observation() -> NonConformance:
    """Return an observation (not a non-conformance)."""
    return NonConformance(
        nc_id="NC-OBS-001",
        audit_id="AUD-TEST-001",
        finding_statement="Supplier management system could benefit from digitized record keeping",
        objective_evidence="Paper-based records observed during site visit; no compliance issue identified",
        severity=NCSeverity.OBSERVATION,
        risk_impact_score=Decimal("5.00"),
        status="open",
    )


# ---------------------------------------------------------------------------
# CAR Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_car_critical() -> CorrectiveActionRequest:
    """Return a critical CAR with 30-day SLA."""
    issued_at = FROZEN_NOW
    return CorrectiveActionRequest(
        car_id="CAR-CRIT-001",
        nc_ids=["NC-CRIT-001"],
        audit_id="AUD-TEST-001",
        supplier_id="SUP-001",
        severity=NCSeverity.CRITICAL,
        sla_deadline=issued_at + timedelta(days=30),
        sla_status="on_track",
        status=CARStatus.ISSUED,
        issued_by="AUR-FSC-001",
        issued_at=issued_at,
    )


@pytest.fixture
def sample_car_major() -> CorrectiveActionRequest:
    """Return a major CAR with 90-day SLA."""
    issued_at = FROZEN_NOW
    return CorrectiveActionRequest(
        car_id="CAR-MAJ-001",
        nc_ids=["NC-MAJ-001"],
        audit_id="AUD-TEST-001",
        supplier_id="SUP-001",
        severity=NCSeverity.MAJOR,
        sla_deadline=issued_at + timedelta(days=90),
        sla_status="on_track",
        status=CARStatus.ISSUED,
        issued_by="AUR-FSC-001",
        issued_at=issued_at,
    )


@pytest.fixture
def sample_car_minor() -> CorrectiveActionRequest:
    """Return a minor CAR with 365-day SLA."""
    issued_at = FROZEN_NOW
    return CorrectiveActionRequest(
        car_id="CAR-MIN-001",
        nc_ids=["NC-MIN-001"],
        audit_id="AUD-TEST-001",
        supplier_id="SUP-001",
        severity=NCSeverity.MINOR,
        sla_deadline=issued_at + timedelta(days=365),
        sla_status="on_track",
        status=CARStatus.ISSUED,
        issued_by="AUR-FSC-001",
        issued_at=issued_at,
    )


@pytest.fixture
def sample_car_overdue() -> CorrectiveActionRequest:
    """Return an overdue CAR."""
    issued_at = FROZEN_NOW - timedelta(days=100)
    return CorrectiveActionRequest(
        car_id="CAR-OVER-001",
        nc_ids=["NC-MAJ-001"],
        audit_id="AUD-TEST-001",
        supplier_id="SUP-001",
        severity=NCSeverity.MAJOR,
        sla_deadline=issued_at + timedelta(days=90),
        sla_status="overdue",
        status=CARStatus.OVERDUE,
        issued_by="AUR-FSC-001",
        issued_at=issued_at,
    )


# ---------------------------------------------------------------------------
# Checklist Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_checklist_eudr() -> AuditChecklist:
    """Return a sample EUDR audit checklist."""
    return AuditChecklist(
        checklist_id="CHK-EUDR-001",
        audit_id="AUD-TEST-001",
        checklist_type="eudr",
        checklist_version="1.0.0",
        criteria=[
            {"criterion_id": "EUDR-ART3-001", "description": "Deforestation-free verification", "result": "pass"},
            {"criterion_id": "EUDR-ART9-GEO-001", "description": "Geolocation verification", "result": "fail"},
            {"criterion_id": "EUDR-ART10-RA-001", "description": "Risk assessment completeness", "result": None},
        ],
        total_criteria=17,
        passed_criteria=8,
        failed_criteria=2,
        na_criteria=1,
        completion_percentage=Decimal("64.71"),
    )


@pytest.fixture
def sample_checklist_fsc() -> AuditChecklist:
    """Return a sample FSC audit checklist."""
    return AuditChecklist(
        checklist_id="CHK-FSC-001",
        audit_id="AUD-TEST-001",
        checklist_type="fsc",
        checklist_version="5.3.0",
        total_criteria=12,
        passed_criteria=10,
        failed_criteria=1,
        na_criteria=1,
        completion_percentage=Decimal("100.00"),
    )


# ---------------------------------------------------------------------------
# Evidence Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_evidence_document() -> AuditEvidence:
    """Return a sample document evidence item."""
    return AuditEvidence(
        evidence_id="EV-DOC-001",
        audit_id="AUD-TEST-001",
        evidence_type="permit",
        file_name="forest_management_permit_2025.pdf",
        file_size_bytes=2_500_000,
        mime_type="application/pdf",
        sha256_hash="a" * 64,
        description="Forest management permit for site SITE-001",
        tags={"country": "BR", "year": "2025", "type": "permit"},
        collection_date=FROZEN_DATE,
        collector_id="AUR-FSC-001",
        location="Para, Brazil",
    )


@pytest.fixture
def sample_evidence_photo() -> AuditEvidence:
    """Return a sample photo evidence item."""
    return AuditEvidence(
        evidence_id="EV-PHOTO-001",
        audit_id="AUD-TEST-001",
        evidence_type="photo",
        file_name="site_inspection_001.jpg",
        file_size_bytes=5_000_000,
        mime_type="image/jpeg",
        sha256_hash="b" * 64,
        description="Site inspection photo showing forest boundary",
        collection_date=FROZEN_DATE,
        collector_id="AUR-FSC-001",
    )


# ---------------------------------------------------------------------------
# Certificate Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_certificate_fsc() -> CertificateRecord:
    """Return a sample FSC certificate record."""
    return CertificateRecord(
        certificate_id="CERT-FSC-001",
        scheme=CertificationScheme.FSC,
        certificate_number="FSC-C123456",
        holder_name="Test Supplier FSC Ltd.",
        holder_id="SUP-001",
        status="active",
        scope="chain_of_custody",
        issue_date=date(2023, 6, 1),
        expiry_date=date(2028, 5, 31),
        certified_products=["timber", "wood_products"],
        certified_sites=["SITE-001"],
    )


@pytest.fixture
def sample_certificate_rspo() -> CertificateRecord:
    """Return a sample RSPO certificate record."""
    return CertificateRecord(
        certificate_id="CERT-RSPO-001",
        scheme=CertificationScheme.RSPO,
        certificate_number="RSPO-2024-789",
        holder_name="Palm Oil Plantation RSPO Corp.",
        holder_id="SUP-002",
        status="active",
        scope="identity_preserved",
        issue_date=date(2024, 1, 1),
        expiry_date=date(2029, 12, 31),
        certified_products=["crude_palm_oil"],
    )


@pytest.fixture
def sample_certificate_expired() -> CertificateRecord:
    """Return an expired certificate record."""
    return CertificateRecord(
        certificate_id="CERT-EXP-001",
        scheme=CertificationScheme.PEFC,
        certificate_number="PEFC-XX-2020-001",
        holder_name="Expired Forest Products Inc.",
        holder_id="SUP-003",
        status="expired",
        scope="chain_of_custody",
        issue_date=date(2019, 1, 1),
        expiry_date=date(2024, 12, 31),
    )


# ---------------------------------------------------------------------------
# Authority Interaction Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_authority_document_request() -> CompetentAuthorityInteraction:
    """Return a sample competent authority document request."""
    return CompetentAuthorityInteraction(
        interaction_id="AUTH-DOC-001",
        operator_id="OP-001",
        authority_name="BMEL",
        member_state="DE",
        interaction_type=AuthorityInteractionType.DOCUMENT_REQUEST,
        subject="Request for DDS documentation for wood imports from Brazil",
        received_at=FROZEN_NOW,
        response_deadline=FROZEN_NOW + timedelta(days=30),
        status="received",
    )


@pytest.fixture
def sample_authority_inspection() -> CompetentAuthorityInteraction:
    """Return a sample competent authority inspection notification."""
    return CompetentAuthorityInteraction(
        interaction_id="AUTH-INSP-001",
        operator_id="OP-001",
        authority_name="NVWA",
        member_state="NL",
        interaction_type=AuthorityInteractionType.INSPECTION_NOTIFICATION,
        subject="Scheduled on-site inspection for palm oil supply chain",
        received_at=FROZEN_NOW,
        response_deadline=FROZEN_NOW + timedelta(days=14),
        status="received",
    )


# ---------------------------------------------------------------------------
# RCA Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rca_five_whys() -> RootCauseAnalysis:
    """Return a sample 5-Whys root cause analysis."""
    return RootCauseAnalysis(
        rca_id="RCA-5W-001",
        nc_id="NC-MAJ-001",
        framework="five_whys",
        five_whys=[
            {"why": "Why was the risk assessment incomplete?", "because": "Indonesia section was not included"},
            {"why": "Why was Indonesia section not included?", "because": "Staff member responsible was on leave"},
            {"why": "Why was there no backup?", "because": "No documented backup procedure exists"},
            {"why": "Why is there no backup procedure?", "because": "Process documentation was never formalized"},
            {"why": "Why was process not formalized?", "because": "Rapid team growth without process maturation"},
        ],
        direct_cause="Staff member responsible for Indonesia risk assessment was on extended leave",
        contributing_causes=["No backup procedure", "Rapid team growth"],
        root_cause="Lack of formalized process documentation for risk assessment coverage",
        recommended_actions=["Document risk assessment procedures", "Assign backup personnel", "Implement review checklist"],
        analyst_id="AUR-FSC-001",
    )


@pytest.fixture
def sample_rca_ishikawa() -> RootCauseAnalysis:
    """Return a sample Ishikawa root cause analysis."""
    return RootCauseAnalysis(
        rca_id="RCA-ISH-001",
        nc_id="NC-CRIT-001",
        framework="ishikawa",
        ishikawa_categories={
            "people": ["Insufficient training on satellite imagery interpretation"],
            "process": ["No automated deforestation alert monitoring"],
            "equipment": ["Outdated satellite imagery subscription"],
            "materials": ["Limited access to high-resolution imagery"],
            "environment": ["Remote forest area with cloud cover"],
            "management": ["No deforestation monitoring policy"],
        },
        direct_cause="Deforestation activity not detected during monitoring cycle",
        root_cause="No automated deforestation alert monitoring system integrated with supplier monitoring",
        recommended_actions=["Integrate EUDR-020 deforestation alerts", "Upgrade satellite subscription"],
    )


# ---------------------------------------------------------------------------
# Report Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audit_report() -> AuditReport:
    """Return a sample audit report."""
    return AuditReport(
        report_id="RPT-001",
        audit_id="AUD-TEST-001",
        report_format="pdf",
        language="en",
        report_version="1.0",
        iso_19011_compliant=True,
        sections=[
            "audit_objectives", "audit_scope", "audit_criteria",
            "audit_client", "audit_team", "dates_and_locations",
            "audit_findings", "audit_conclusions",
        ],
        findings_summary={"critical": 1, "major": 1, "minor": 1, "observation": 1},
        evidence_count=15,
        generated_at=FROZEN_NOW,
    )


# ---------------------------------------------------------------------------
# Schedule Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_schedule_request() -> ScheduleAuditRequest:
    """Return a sample audit schedule request."""
    return ScheduleAuditRequest(
        operator_id="OP-001",
        supplier_ids=["SUP-001", "SUP-002", "SUP-003"],
        planning_year=2026,
        quarter=1,
    )


@pytest.fixture
def large_schedule_request() -> ScheduleAuditRequest:
    """Return a schedule request with 500 suppliers for performance testing."""
    return ScheduleAuditRequest(
        operator_id="OP-PERF-001",
        supplier_ids=[f"SUP-PERF-{i:04d}" for i in range(500)],
        planning_year=2026,
        quarter=1,
    )


# ---------------------------------------------------------------------------
# Classify NC Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classify_nc_fraud_request() -> ClassifyNCRequest:
    """Return a classification request for fraud indicator."""
    return ClassifyNCRequest(
        audit_id="AUD-TEST-001",
        finding_statement="Evidence of document falsification detected",
        objective_evidence="Forged forest management permit with inconsistent signatures",
        indicators={"fraud_or_falsification": True},
    )


@pytest.fixture
def classify_nc_deforestation_request() -> ClassifyNCRequest:
    """Return a classification request for deforestation indicator."""
    return ClassifyNCRequest(
        audit_id="AUD-TEST-001",
        finding_statement="Active deforestation detected post-cutoff",
        objective_evidence="Satellite imagery confirms forest loss 2021-2022",
        indicators={"active_deforestation_post_cutoff": True},
    )


@pytest.fixture
def classify_nc_incomplete_risk_request() -> ClassifyNCRequest:
    """Return a classification request for incomplete risk assessment."""
    return ClassifyNCRequest(
        audit_id="AUD-TEST-001",
        finding_statement="Incomplete risk assessment for high-risk countries",
        objective_evidence="Risk assessment file missing country-specific evaluation",
        indicators={"incomplete_risk_assessment": True},
    )


@pytest.fixture
def classify_nc_minor_request() -> ClassifyNCRequest:
    """Return a classification request for a minor finding."""
    return ClassifyNCRequest(
        audit_id="AUD-TEST-001",
        finding_statement="Training records not current",
        objective_evidence="3 staff members have outdated training records",
        indicators={"training_records_not_current": True},
    )


# ---------------------------------------------------------------------------
# Issue CAR Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def issue_car_request_critical() -> IssueCARRequest:
    """Return a CAR issuance request for a critical NC."""
    return IssueCARRequest(
        nc_ids=["NC-CRIT-001"],
        audit_id="AUD-TEST-001",
        supplier_id="SUP-001",
        severity="critical",
        issued_by="AUR-FSC-001",
    )


@pytest.fixture
def issue_car_request_major() -> IssueCARRequest:
    """Return a CAR issuance request for a major NC."""
    return IssueCARRequest(
        nc_ids=["NC-MAJ-001"],
        audit_id="AUD-TEST-001",
        supplier_id="SUP-001",
        severity="major",
        issued_by="AUR-FSC-001",
    )


# ---------------------------------------------------------------------------
# Analytics Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analytics_request() -> CalculateAnalyticsRequest:
    """Return a sample analytics calculation request."""
    return CalculateAnalyticsRequest(
        operator_id="OP-001",
        time_period_start=FROZEN_DATE - timedelta(days=365),
        time_period_end=FROZEN_DATE,
    )


# ---------------------------------------------------------------------------
# Generate Report Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generate_report_request_pdf() -> GenerateReportRequest:
    """Return a report generation request for PDF format."""
    return GenerateReportRequest(
        audit_id="AUD-TEST-001",
        report_format="pdf",
        language="en",
    )


@pytest.fixture
def generate_report_request_json() -> GenerateReportRequest:
    """Return a report generation request for JSON format."""
    return GenerateReportRequest(
        audit_id="AUD-TEST-001",
        report_format="json",
        language="en",
    )


# ---------------------------------------------------------------------------
# Authority Interaction Request Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def authority_interaction_request() -> LogAuthorityInteractionRequest:
    """Return a sample authority interaction logging request."""
    return LogAuthorityInteractionRequest(
        operator_id="OP-001",
        authority_name="BMEL",
        member_state="DE",
        interaction_type="document_request",
        subject="Request for DDS documentation for wood imports",
    )


# ---------------------------------------------------------------------------
# Batch Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_audits() -> List[Audit]:
    """Return a batch of 10 audits with varied parameters."""
    audits = []
    for i in range(10):
        commodity = EUDR_COMMODITIES[i % len(EUDR_COMMODITIES)]
        country = ["BR", "ID", "CO", "GH", "CI", "MY", "PE", "FI", "SE", "NZ"][i]
        audits.append(
            Audit(
                audit_id=f"AUD-BATCH-{i:03d}",
                operator_id="OP-001",
                supplier_id=f"SUP-BATCH-{i:03d}",
                audit_type=AuditScope.FULL,
                modality=AuditModality.ON_SITE,
                planned_date=FROZEN_DATE + timedelta(days=i * 30),
                status=AuditStatus.PLANNED,
                priority_score=Decimal(str(20 + i * 8)),
                country_code=country,
                commodity=commodity,
            )
        )
    return audits


@pytest.fixture
def batch_ncs() -> List[NonConformance]:
    """Return a batch of NCs with mixed severities."""
    ncs = []
    severities = [NCSeverity.CRITICAL, NCSeverity.MAJOR, NCSeverity.MAJOR,
                  NCSeverity.MINOR, NCSeverity.MINOR, NCSeverity.OBSERVATION]
    for i, sev in enumerate(severities):
        ncs.append(
            NonConformance(
                nc_id=f"NC-BATCH-{i:03d}",
                audit_id="AUD-TEST-001",
                finding_statement=f"Test finding {i}",
                objective_evidence=f"Test evidence {i}",
                severity=sev,
                risk_impact_score=Decimal(str(90 - i * 15)),
                status="open",
            )
        )
    return ncs


@pytest.fixture
def batch_cars() -> List[CorrectiveActionRequest]:
    """Return a batch of CARs in various lifecycle stages."""
    cars = []
    statuses = [CARStatus.ISSUED, CARStatus.ACKNOWLEDGED, CARStatus.CAP_SUBMITTED,
                CARStatus.IN_PROGRESS, CARStatus.CLOSED, CARStatus.OVERDUE]
    severities = [NCSeverity.CRITICAL, NCSeverity.MAJOR, NCSeverity.MAJOR,
                  NCSeverity.MINOR, NCSeverity.MINOR, NCSeverity.MAJOR]
    for i, (status, severity) in enumerate(zip(statuses, severities)):
        issued_at = FROZEN_NOW - timedelta(days=i * 15)
        cars.append(
            CorrectiveActionRequest(
                car_id=f"CAR-BATCH-{i:03d}",
                nc_ids=[f"NC-BATCH-{i:03d}"],
                audit_id="AUD-TEST-001",
                supplier_id="SUP-001",
                severity=severity,
                sla_deadline=issued_at + timedelta(days=NC_SEVERITY_SLA_DAYS.get(severity.value, 90)),
                sla_status="on_track" if status != CARStatus.OVERDUE else "overdue",
                status=status,
                issued_by="AUR-FSC-001",
                issued_at=issued_at,
            )
        )
    return cars


# ---------------------------------------------------------------------------
# Commodity-specific Fixtures for Golden Scenarios
# ---------------------------------------------------------------------------


@pytest.fixture(params=EUDR_COMMODITIES)
def commodity(request) -> str:
    """Parametrize over all 7 EUDR commodities."""
    return request.param


@pytest.fixture(params=CERTIFICATION_SCHEMES)
def scheme(request) -> str:
    """Parametrize over all 5 certification schemes."""
    return request.param


@pytest.fixture(params=NC_SEVERITIES)
def severity(request) -> str:
    """Parametrize over all NC severity levels."""
    return request.param


@pytest.fixture(params=REPORT_FORMATS)
def report_format(request) -> str:
    """Parametrize over all 5 report formats."""
    return request.param


@pytest.fixture(params=REPORT_LANGUAGES)
def report_language(request) -> str:
    """Parametrize over all 5 report languages."""
    return request.param
