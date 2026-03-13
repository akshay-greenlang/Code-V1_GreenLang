# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-040 Authority Communication Manager tests.

Provides reusable test fixtures for config, models, provenance, engines,
communications, information requests, inspections, non-compliance cases,
appeals, documents, notifications, templates, authority definitions for all
27 EU member states, and multi-language template samples.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
    EU_LANGUAGES,
    EU_MEMBER_STATES,
    reset_config,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    AGENT_ID,
    AGENT_VERSION,
    DEADLINE_HOURS_MAP,
    EUDR_COMMODITIES,
    Appeal,
    AppealDecision,
    ApprovalWorkflow,
    Authority,
    AuthorityType,
    Communication,
    CommunicationPriority,
    CommunicationStatus,
    CommunicationThread,
    CommunicationType,
    DeadlineReminder,
    Document,
    DocumentType,
    HealthStatus,
    InformationRequest,
    InformationRequestType,
    Inspection,
    InspectionType,
    LanguageCode,
    NonCompliance,
    Notification,
    NotificationChannel,
    RecipientType,
    ResponseData,
    Template,
    ViolationSeverity,
    ViolationType,
)
from greenlang.agents.eudr.authority_communication_manager.provenance import (
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
def sample_config() -> AuthorityCommunicationManagerConfig:
    """Create a default AuthorityCommunicationManagerConfig instance."""
    return AuthorityCommunicationManagerConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Authority fixtures (all 27 EU member states)
# ---------------------------------------------------------------------------

@pytest.fixture
def authority_de() -> Authority:
    """Create German competent authority."""
    return Authority(
        authority_id="AUTH-DE-001",
        member_state="DE",
        name="German Federal Agency for Nature Conservation",
        authority_type=AuthorityType.NATIONAL_COMPETENT,
        contact_email="eudr@bfn.de",
        api_endpoint="https://eudr-portal.de/api/v1",
        preferred_language=LanguageCode.DE,
        timezone="Europe/Berlin",
    )


@pytest.fixture
def authority_fr() -> Authority:
    """Create French competent authority."""
    return Authority(
        authority_id="AUTH-FR-001",
        member_state="FR",
        name="French Ministry of Ecological Transition",
        authority_type=AuthorityType.ENVIRONMENTAL,
        contact_email="eudr@ecologie.gouv.fr",
        api_endpoint="https://eudr-portal.fr/api/v1",
        preferred_language=LanguageCode.FR,
        timezone="Europe/Paris",
    )


@pytest.fixture
def authority_nl() -> Authority:
    """Create Dutch competent authority."""
    return Authority(
        authority_id="AUTH-NL-001",
        member_state="NL",
        name="Netherlands Food and Consumer Product Safety Authority",
        authority_type=AuthorityType.NATIONAL_COMPETENT,
        contact_email="eudr@nvwa.nl",
        api_endpoint="https://eudr-portal.nl/api/v1",
        preferred_language=LanguageCode.NL,
        timezone="Europe/Amsterdam",
    )


@pytest.fixture
def authority_it() -> Authority:
    """Create Italian competent authority."""
    return Authority(
        authority_id="AUTH-IT-001",
        member_state="IT",
        name="Italian Ministry of Agriculture",
        authority_type=AuthorityType.NATIONAL_COMPETENT,
        contact_email="eudr@mipaaf.it",
        api_endpoint="https://eudr-portal.it/api/v1",
        preferred_language=LanguageCode.IT,
        timezone="Europe/Rome",
    )


@pytest.fixture
def all_27_authorities() -> List[Authority]:
    """Create authority records for all 27 EU member states."""
    authorities = []
    for code, info in EU_MEMBER_STATES.items():
        lang = info.get("language", "en")
        try:
            lang_code = LanguageCode(lang)
        except ValueError:
            lang_code = LanguageCode.EN
        authorities.append(
            Authority(
                authority_id=f"AUTH-{code}-001",
                member_state=code,
                name=info["authority"],
                authority_type=AuthorityType.NATIONAL_COMPETENT,
                contact_email=f"eudr@authority.{code.lower()}",
                api_endpoint=info["endpoint"],
                preferred_language=lang_code,
            )
        )
    return authorities


# ---------------------------------------------------------------------------
# Communication fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pending_communication() -> Communication:
    """Create a communication in PENDING status."""
    now = datetime.now(tz=timezone.utc)
    return Communication(
        communication_id="COMM-001",
        operator_id="OP-001",
        authority_id="AUTH-DE-001",
        member_state="DE",
        communication_type=CommunicationType.INFORMATION_REQUEST,
        status=CommunicationStatus.PENDING,
        priority=CommunicationPriority.NORMAL,
        subject="Information request regarding DDS-2026-001",
        body="Requesting additional details on supply chain documentation.",
        language=LanguageCode.EN,
        dds_reference="GL-DDS-20260313-ABCDEF",
        deadline=now + timedelta(days=5),
        created_at=now,
        updated_at=now,
        provenance_hash="a" * 64,
    )


@pytest.fixture
def sent_communication() -> Communication:
    """Create a communication in SENT status."""
    now = datetime.now(tz=timezone.utc)
    return Communication(
        communication_id="COMM-002",
        operator_id="OP-002",
        authority_id="AUTH-FR-001",
        member_state="FR",
        communication_type=CommunicationType.NON_COMPLIANCE_NOTICE,
        status=CommunicationStatus.SENT,
        priority=CommunicationPriority.HIGH,
        subject="Non-compliance notice - missing DDS",
        body="Formal notice of non-compliance under EUDR Article 16.",
        language=LanguageCode.FR,
        sent_at=now - timedelta(hours=6),
        created_at=now - timedelta(hours=8),
        updated_at=now,
        provenance_hash="b" * 64,
    )


@pytest.fixture
def closed_communication() -> Communication:
    """Create a communication in CLOSED status."""
    now = datetime.now(tz=timezone.utc)
    return Communication(
        communication_id="COMM-003",
        operator_id="OP-003",
        authority_id="AUTH-NL-001",
        member_state="NL",
        communication_type=CommunicationType.COMPLIANCE_CONFIRMATION,
        status=CommunicationStatus.CLOSED,
        priority=CommunicationPriority.ROUTINE,
        subject="Compliance confirmation for DDS-2025-789",
        body="Confirming operator compliance following inspection.",
        language=LanguageCode.NL,
        sent_at=now - timedelta(days=30),
        responded_at=now - timedelta(days=25),
        created_at=now - timedelta(days=35),
        updated_at=now,
        provenance_hash="c" * 64,
    )


@pytest.fixture
def overdue_communication() -> Communication:
    """Create an overdue communication."""
    now = datetime.now(tz=timezone.utc)
    return Communication(
        communication_id="COMM-004",
        operator_id="OP-004",
        authority_id="AUTH-DE-001",
        member_state="DE",
        communication_type=CommunicationType.INFORMATION_REQUEST,
        status=CommunicationStatus.OVERDUE,
        priority=CommunicationPriority.URGENT,
        subject="URGENT: Supply chain evidence required",
        body="Immediate evidence required per EUDR Article 17.",
        language=LanguageCode.DE,
        deadline=now - timedelta(hours=12),
        created_at=now - timedelta(days=3),
        updated_at=now,
        provenance_hash="d" * 64,
    )


@pytest.fixture
def sample_communications(
    pending_communication, sent_communication, closed_communication,
    overdue_communication,
) -> List[Communication]:
    """All sample communications."""
    return [
        pending_communication, sent_communication,
        closed_communication, overdue_communication,
    ]


# ---------------------------------------------------------------------------
# Information Request fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def urgent_info_request() -> InformationRequest:
    """Create an urgent information request."""
    now = datetime.now(tz=timezone.utc)
    return InformationRequest(
        request_id="REQ-001",
        communication_id="COMM-001",
        operator_id="OP-001",
        authority_id="AUTH-DE-001",
        request_type=InformationRequestType.DDS_CLARIFICATION,
        items_requested=[
            "Complete DDS statement copy",
            "Supply chain documentation for last 12 months",
            "Geolocation data for all production plots",
        ],
        dds_reference="GL-DDS-20260313-ABCDEF",
        commodity="cocoa",
        deadline=now + timedelta(hours=24),
        created_at=now,
        provenance_hash="e" * 64,
    )


@pytest.fixture
def normal_info_request() -> InformationRequest:
    """Create a normal priority information request."""
    now = datetime.now(tz=timezone.utc)
    return InformationRequest(
        request_id="REQ-002",
        communication_id="COMM-002",
        operator_id="OP-002",
        authority_id="AUTH-FR-001",
        request_type=InformationRequestType.SUPPLY_CHAIN_EVIDENCE,
        items_requested=[
            "Tier-1 supplier list",
            "Certificates of origin",
        ],
        dds_reference="GL-DDS-20260310-BCDEFG",
        commodity="coffee",
        deadline=now + timedelta(days=5),
        created_at=now,
        provenance_hash="f" * 64,
    )


@pytest.fixture
def routine_info_request() -> InformationRequest:
    """Create a routine information request."""
    now = datetime.now(tz=timezone.utc)
    return InformationRequest(
        request_id="REQ-003",
        communication_id="COMM-003",
        operator_id="OP-003",
        authority_id="AUTH-NL-001",
        request_type=InformationRequestType.RISK_ASSESSMENT_DETAILS,
        items_requested=["Annual risk assessment summary"],
        commodity="soya",
        deadline=now + timedelta(days=15),
        response_submitted=True,
        response_accepted=True,
        created_at=now - timedelta(days=10),
        provenance_hash="g" * 64,
    )


# ---------------------------------------------------------------------------
# Inspection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduled_inspection() -> Inspection:
    """Create a scheduled announced inspection."""
    now = datetime.now(tz=timezone.utc)
    return Inspection(
        inspection_id="INSP-001",
        communication_id="COMM-001",
        operator_id="OP-001",
        authority_id="AUTH-DE-001",
        inspection_type=InspectionType.ANNOUNCED,
        scheduled_date=now + timedelta(days=7),
        location="Warehouse Berlin-Mitte, Friedrichstr. 200",
        inspector_name="Dr. Klaus Mueller",
        scope="DDS documentation review and supply chain verification",
        status="scheduled",
        created_at=now,
        provenance_hash="h" * 64,
    )


@pytest.fixture
def in_progress_inspection() -> Inspection:
    """Create an in-progress inspection."""
    now = datetime.now(tz=timezone.utc)
    return Inspection(
        inspection_id="INSP-002",
        communication_id="COMM-002",
        operator_id="OP-002",
        authority_id="AUTH-FR-001",
        inspection_type=InspectionType.UNANNOUNCED,
        scheduled_date=now - timedelta(hours=3),
        actual_start=now - timedelta(hours=2),
        location="Port of Marseille, Terminal 3",
        inspector_name="Marie Dupont",
        scope="Physical inspection of coffee shipment and legality documentation",
        status="in_progress",
        created_at=now - timedelta(days=1),
        provenance_hash="i" * 64,
    )


@pytest.fixture
def completed_inspection() -> Inspection:
    """Create a completed inspection with findings."""
    now = datetime.now(tz=timezone.utc)
    return Inspection(
        inspection_id="INSP-003",
        communication_id="COMM-003",
        operator_id="OP-003",
        authority_id="AUTH-NL-001",
        inspection_type=InspectionType.FOLLOW_UP,
        scheduled_date=now - timedelta(days=14),
        actual_start=now - timedelta(days=14),
        actual_end=now - timedelta(days=14) + timedelta(hours=6),
        location="Amsterdam Distribution Center",
        inspector_name="Jan de Vries",
        scope="Follow-up on corrective actions from INSP-001",
        findings=[
            "DDS documentation now complete",
            "Geolocation data verified for all plots",
            "Supply chain mapping improved but gaps remain at Tier-3",
        ],
        corrective_actions=[
            "Extend supply chain mapping to Tier-3 within 90 days",
        ],
        follow_up_date=now + timedelta(days=90),
        status="completed",
        created_at=now - timedelta(days=20),
        provenance_hash="j" * 64,
    )


# ---------------------------------------------------------------------------
# Non-compliance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minor_non_compliance() -> NonCompliance:
    """Create a minor non-compliance record."""
    now = datetime.now(tz=timezone.utc)
    return NonCompliance(
        non_compliance_id="NC-001",
        communication_id="COMM-001",
        operator_id="OP-001",
        authority_id="AUTH-DE-001",
        violation_type=ViolationType.INCOMPLETE_DDS,
        severity=ViolationSeverity.MINOR,
        description="DDS statement missing geolocation data for 2 of 15 production plots.",
        evidence_references=["DOC-001", "DOC-002"],
        penalty_amount=Decimal("2500.00"),
        corrective_actions_required=[
            "Submit complete geolocation data within 30 days",
        ],
        corrective_deadline=now + timedelta(days=30),
        commodity="cocoa",
        dds_reference="GL-DDS-20260313-ABCDEF",
        issued_at=now,
        provenance_hash="k" * 64,
    )


@pytest.fixture
def major_non_compliance() -> NonCompliance:
    """Create a major non-compliance record."""
    now = datetime.now(tz=timezone.utc)
    return NonCompliance(
        non_compliance_id="NC-002",
        communication_id="COMM-002",
        operator_id="OP-002",
        authority_id="AUTH-FR-001",
        violation_type=ViolationType.DEFORESTATION_LINK,
        severity=ViolationSeverity.MAJOR,
        description="Satellite imagery shows deforestation activity on production plot GPS-2025-789 after December 2020 cutoff.",
        evidence_references=["DOC-003", "DOC-004", "DOC-005"],
        penalty_amount=Decimal("250000.00"),
        corrective_actions_required=[
            "Immediately suspend sourcing from plot GPS-2025-789",
            "Submit revised risk assessment within 14 days",
            "Implement enhanced due diligence for all suppliers",
        ],
        corrective_deadline=now + timedelta(days=14),
        commodity="coffee",
        dds_reference="GL-DDS-20260310-BCDEFG",
        issued_at=now,
        provenance_hash="l" * 64,
    )


@pytest.fixture
def critical_non_compliance() -> NonCompliance:
    """Create a critical non-compliance record."""
    now = datetime.now(tz=timezone.utc)
    return NonCompliance(
        non_compliance_id="NC-003",
        communication_id="COMM-003",
        operator_id="OP-003",
        authority_id="AUTH-NL-001",
        violation_type=ViolationType.FALSE_INFORMATION,
        severity=ViolationSeverity.CRITICAL,
        description="Deliberate falsification of supply chain origin data. Declared origin Indonesia but actual source traced to illegal logging in Myanmar.",
        evidence_references=["DOC-006", "DOC-007", "DOC-008", "DOC-009"],
        penalty_amount=Decimal("5000000.00"),
        corrective_actions_required=[
            "Full market withdrawal of affected products",
            "Complete re-audit of entire supply chain",
            "Third-party verification of all DDS statements",
        ],
        corrective_deadline=now + timedelta(days=7),
        commodity="wood",
        dds_reference="GL-DDS-20260305-CDEFGH",
        issued_at=now,
        provenance_hash="m" * 64,
    )


# ---------------------------------------------------------------------------
# Appeal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def filed_appeal() -> Appeal:
    """Create a newly filed appeal."""
    now = datetime.now(tz=timezone.utc)
    return Appeal(
        appeal_id="APP-001",
        communication_id="COMM-001",
        non_compliance_id="NC-001",
        operator_id="OP-001",
        authority_id="AUTH-DE-001",
        grounds="Geolocation data was submitted but not properly processed by the authority portal.",
        supporting_evidence=["DOC-010", "DOC-011"],
        decision=AppealDecision.PENDING,
        filing_date=now,
        deadline=now + timedelta(days=60),
        provenance_hash="n" * 64,
    )


@pytest.fixture
def under_review_appeal() -> Appeal:
    """Create an appeal under review."""
    now = datetime.now(tz=timezone.utc)
    return Appeal(
        appeal_id="APP-002",
        communication_id="COMM-002",
        non_compliance_id="NC-002",
        operator_id="OP-002",
        authority_id="AUTH-FR-001",
        grounds="Satellite imagery misinterpreted - area is agroforestry, not deforestation.",
        supporting_evidence=["DOC-012", "DOC-013", "DOC-014"],
        decision=AppealDecision.PENDING,
        filing_date=now - timedelta(days=20),
        deadline=now + timedelta(days=40),
        extensions_granted=0,
        provenance_hash="o" * 64,
    )


@pytest.fixture
def decided_appeal() -> Appeal:
    """Create an appeal with a decision."""
    now = datetime.now(tz=timezone.utc)
    return Appeal(
        appeal_id="APP-003",
        communication_id="COMM-003",
        non_compliance_id="NC-001",
        operator_id="OP-003",
        authority_id="AUTH-NL-001",
        grounds="Penalty disproportionate to the nature of the violation.",
        supporting_evidence=["DOC-015"],
        decision=AppealDecision.PARTIALLY_UPHELD,
        decision_reason="Penalty reduced to EUR 1,500 based on operator's cooperation and prompt corrective action.",
        decision_date=now - timedelta(days=5),
        filing_date=now - timedelta(days=45),
        deadline=now - timedelta(days=5),
        penalty_suspended=False,
        provenance_hash="p" * 64,
    )


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encrypted_document() -> Document:
    """Create an encrypted sensitive document."""
    now = datetime.now(tz=timezone.utc)
    return Document(
        document_id="DOC-001",
        communication_id="COMM-001",
        document_type=DocumentType.DDS_STATEMENT,
        title="Due Diligence Statement - DDS-2026-001",
        description="Complete DDS with supply chain mapping and risk assessment.",
        file_path="s3://gl-eudr-docs/dds/DDS-2026-001.pdf.enc",
        file_size_bytes=2_500_000,
        mime_type="application/pdf",
        language=LanguageCode.EN,
        encrypted=True,
        encryption_key_id="eudr-acm-doc-key-v1",
        integrity_hash="abc123" * 10 + "abcd",
        uploaded_by="OP-001",
        uploaded_at=now,
        provenance_hash="q" * 64,
    )


@pytest.fixture
def unencrypted_document() -> Document:
    """Create an unencrypted public document."""
    now = datetime.now(tz=timezone.utc)
    return Document(
        document_id="DOC-002",
        communication_id="COMM-002",
        document_type=DocumentType.CERTIFICATE,
        title="Sustainability Certificate - RSPO-2026-4567",
        description="RSPO certificate for palm oil supply chain.",
        file_path="s3://gl-eudr-docs/certs/RSPO-2026-4567.pdf",
        file_size_bytes=450_000,
        mime_type="application/pdf",
        language=LanguageCode.EN,
        encrypted=False,
        integrity_hash="def456" * 10 + "defg",
        uploaded_by="OP-002",
        uploaded_at=now,
        provenance_hash="r" * 64,
    )


@pytest.fixture
def satellite_document() -> Document:
    """Create a satellite imagery document."""
    now = datetime.now(tz=timezone.utc)
    return Document(
        document_id="DOC-003",
        communication_id="COMM-002",
        document_type=DocumentType.SATELLITE_IMAGERY,
        title="Sentinel-2 Analysis - Plot GPS-2025-789",
        description="Multi-temporal satellite analysis showing land use change.",
        file_path="s3://gl-eudr-docs/satellite/S2-GPS-2025-789.tiff.enc",
        file_size_bytes=15_000_000,
        mime_type="image/tiff",
        language=LanguageCode.EN,
        encrypted=True,
        encryption_key_id="eudr-acm-doc-key-v1",
        integrity_hash="ghi789" * 10 + "ghij",
        uploaded_by="AUTH-FR-001",
        uploaded_at=now,
        provenance_hash="s" * 64,
    )


# ---------------------------------------------------------------------------
# Template fixtures (multi-language)
# ---------------------------------------------------------------------------

@pytest.fixture
def template_en() -> Template:
    """Create an English information request template."""
    return Template(
        template_id="TPL-EN-001",
        template_name="information_request_en",
        communication_type=CommunicationType.INFORMATION_REQUEST,
        language=LanguageCode.EN,
        subject_template="Information Request - {dds_reference}",
        body_template=(
            "Dear {operator_name},\n\n"
            "Under EUDR Article 17, we request the following information "
            "regarding your Due Diligence Statement {dds_reference}:\n\n"
            "{items_list}\n\n"
            "Please respond by {deadline}.\n\n"
            "Regards,\n{authority_name}"
        ),
        placeholders=[
            "operator_name", "dds_reference", "items_list",
            "deadline", "authority_name",
        ],
    )


@pytest.fixture
def template_de() -> Template:
    """Create a German information request template."""
    return Template(
        template_id="TPL-DE-001",
        template_name="information_request_de",
        communication_type=CommunicationType.INFORMATION_REQUEST,
        language=LanguageCode.DE,
        subject_template="Auskunftsersuchen - {dds_reference}",
        body_template=(
            "Sehr geehrte/r {operator_name},\n\n"
            "gemaess EUDR Artikel 17 ersuchen wir um folgende Informationen "
            "zu Ihrer Sorgfaltserklaerung {dds_reference}:\n\n"
            "{items_list}\n\n"
            "Bitte antworten Sie bis {deadline}.\n\n"
            "Mit freundlichen Gruessen,\n{authority_name}"
        ),
        placeholders=[
            "operator_name", "dds_reference", "items_list",
            "deadline", "authority_name",
        ],
    )


@pytest.fixture
def template_fr() -> Template:
    """Create a French non-compliance notice template."""
    return Template(
        template_id="TPL-FR-001",
        template_name="non_compliance_notice_fr",
        communication_type=CommunicationType.NON_COMPLIANCE_NOTICE,
        language=LanguageCode.FR,
        subject_template="Avis de non-conformite - {dds_reference}",
        body_template=(
            "Cher/Chere {operator_name},\n\n"
            "Conformement a l'article 16 du reglement EUDR, nous vous "
            "informons de la non-conformite suivante concernant votre "
            "declaration {dds_reference}:\n\n"
            "{violation_description}\n\n"
            "Cordialement,\n{authority_name}"
        ),
        placeholders=[
            "operator_name", "dds_reference", "violation_description",
            "authority_name",
        ],
    )


@pytest.fixture
def all_language_templates() -> List[Template]:
    """Create templates in all 24 EU languages for testing."""
    templates = []
    for i, lang in enumerate(EU_LANGUAGES):
        try:
            lang_code = LanguageCode(lang)
        except ValueError:
            lang_code = LanguageCode.EN
        templates.append(
            Template(
                template_id=f"TPL-{lang.upper()}-001",
                template_name=f"information_request_{lang}",
                communication_type=CommunicationType.INFORMATION_REQUEST,
                language=lang_code,
                subject_template=f"[{lang.upper()}] Information Request - {{dds_reference}}",
                body_template=f"[{lang.upper()}] Body template for {{operator_name}}",
                placeholders=["operator_name", "dds_reference"],
            )
        )
    return templates


# ---------------------------------------------------------------------------
# Notification fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def email_notification() -> Notification:
    """Create an email notification."""
    now = datetime.now(tz=timezone.utc)
    return Notification(
        notification_id="NOTIF-001",
        communication_id="COMM-001",
        channel=NotificationChannel.EMAIL,
        recipient_type=RecipientType.OPERATOR,
        recipient_id="OP-001",
        recipient_address="compliance@acme-trading.com",
        subject="Information Request from German Competent Authority",
        body="You have received a new information request regarding DDS-2026-001.",
        language=LanguageCode.EN,
        delivery_status="sent",
        sent_at=now,
        provenance_hash="t" * 64,
    )


@pytest.fixture
def api_notification() -> Notification:
    """Create an API webhook notification."""
    now = datetime.now(tz=timezone.utc)
    return Notification(
        notification_id="NOTIF-002",
        communication_id="COMM-001",
        channel=NotificationChannel.API,
        recipient_type=RecipientType.SYSTEM,
        recipient_id="SYSTEM-001",
        recipient_address="https://erp.acme-trading.com/webhooks/eudr",
        subject="EUDR Communication Received",
        body='{"event": "communication_received", "communication_id": "COMM-001"}',
        language=LanguageCode.EN,
        delivery_status="delivered",
        sent_at=now - timedelta(minutes=5),
        delivered_at=now - timedelta(minutes=5),
        provenance_hash="u" * 64,
    )


@pytest.fixture
def portal_notification() -> Notification:
    """Create a portal notification."""
    return Notification(
        notification_id="NOTIF-003",
        communication_id="COMM-002",
        channel=NotificationChannel.PORTAL,
        recipient_type=RecipientType.COMPLIANCE_OFFICER,
        recipient_id="USER-CO-001",
        subject="Non-compliance notice requires attention",
        body="A non-compliance notice has been received from the French authority.",
        language=LanguageCode.EN,
        delivery_status="pending",
        provenance_hash="v" * 64,
    )


@pytest.fixture
def failed_notification() -> Notification:
    """Create a failed notification."""
    return Notification(
        notification_id="NOTIF-004",
        communication_id="COMM-004",
        channel=NotificationChannel.EMAIL,
        recipient_type=RecipientType.OPERATOR,
        recipient_id="OP-004",
        recipient_address="invalid@nonexistent.example",
        subject="URGENT: Overdue response",
        body="Your response is overdue.",
        language=LanguageCode.EN,
        delivery_status="failed",
        retry_count=3,
        max_retries=3,
        error_message="550 Mailbox not found",
        provenance_hash="w" * 64,
    )


# ---------------------------------------------------------------------------
# Deadline reminder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def upcoming_deadline_reminder() -> DeadlineReminder:
    """Create a deadline reminder for upcoming deadline."""
    now = datetime.now(tz=timezone.utc)
    return DeadlineReminder(
        reminder_id="REM-001",
        communication_id="COMM-001",
        operator_id="OP-001",
        deadline=now + timedelta(hours=48),
        hours_remaining=48,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Approval workflow fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pending_approval() -> ApprovalWorkflow:
    """Create a pending approval workflow."""
    return ApprovalWorkflow(
        workflow_id="WF-001",
        communication_id="COMM-001",
        initiated_by="USER-001",
        approver_id="USER-MGR-001",
        status="pending_review",
    )


# ---------------------------------------------------------------------------
# Communication thread fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_thread() -> CommunicationThread:
    """Create a sample communication thread."""
    now = datetime.now(tz=timezone.utc)
    return CommunicationThread(
        thread_id="THR-001",
        operator_id="OP-001",
        authority_id="AUTH-DE-001",
        subject="DDS-2026-001 Compliance Discussion",
        communication_ids=["COMM-001", "COMM-005"],
        status=CommunicationStatus.PENDING,
        priority=CommunicationPriority.NORMAL,
        created_at=now - timedelta(days=5),
        last_activity_at=now,
    )


# ---------------------------------------------------------------------------
# Response data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_response() -> ResponseData:
    """Create a sample response to a communication."""
    return ResponseData(
        response_id="RESP-001",
        communication_id="COMM-001",
        responder_id="OP-001",
        responder_type=RecipientType.OPERATOR,
        body="Please find the requested documentation attached.",
        document_ids=["DOC-001", "DOC-002"],
    )


# ---------------------------------------------------------------------------
# Engine fixtures (initialized with config only)
# ---------------------------------------------------------------------------

@pytest.fixture
def request_handler(sample_config):
    """Create a RequestHandler engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.request_handler import (
        RequestHandler,
    )
    return RequestHandler(config=sample_config)


@pytest.fixture
def inspection_coordinator(sample_config):
    """Create an InspectionCoordinator engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.inspection_coordinator import (
        InspectionCoordinator,
    )
    return InspectionCoordinator(config=sample_config)


@pytest.fixture
def non_compliance_manager(sample_config):
    """Create a NonComplianceManager engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.non_compliance_manager import (
        NonComplianceManager,
    )
    return NonComplianceManager(config=sample_config)


@pytest.fixture
def appeal_processor(sample_config):
    """Create an AppealProcessor engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.appeal_processor import (
        AppealProcessor,
    )
    return AppealProcessor(config=sample_config)


@pytest.fixture
def document_exchange(sample_config):
    """Create a DocumentExchange engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.document_exchange import (
        DocumentExchange,
    )
    return DocumentExchange(config=sample_config)


@pytest.fixture
def notification_router(sample_config):
    """Create a NotificationRouter engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.notification_router import (
        NotificationRouter,
    )
    return NotificationRouter(config=sample_config)


@pytest.fixture
def template_engine(sample_config):
    """Create a TemplateEngine engine instance."""
    from greenlang.agents.eudr.authority_communication_manager.template_engine import (
        TemplateEngine,
    )
    return TemplateEngine(config=sample_config)
