# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-031 Stakeholder Engagement Tool tests.

Provides reusable test fixtures for config, models, stakeholder records,
FPIC workflows, grievance mechanisms, consultation records, communication
hub, engagement assessments, and compliance reports across all test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
    reset_config,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    AuditAction,
    AuditEntry,
    CommunicationChannel,
    CommunicationRecord,
    CommunicationTemplate,
    ComplianceReport,
    ConsentStatus,
    ConsultationRecord,
    ConsultationType,
    DeliveryStatus,
    EngagementAssessment,
    EngagementDimension,
    EUDRCommodity,
    FPICStage,
    FPICWorkflow,
    GrievanceRecord,
    GrievanceSeverity,
    GrievanceStatus,
    HealthStatus,
    ReportFormat,
    ReportType,
    RightsClassification,
    StakeholderCategory,
    StakeholderRecord,
    StakeholderStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
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
def sample_config() -> StakeholderEngagementConfig:
    """Create a default StakeholderEngagementConfig instance."""
    return StakeholderEngagementConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Stakeholder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_contact_info() -> Dict:
    """Create sample contact information."""
    return {
        "primary_name": "Maria Gonzalez",
        "role": "Community Leader",
        "email": "maria.gonzalez@example.com",
        "phone": "+57-300-555-0101",
        "preferred_language": "es",
        "address": "Calle 12 #5-34, Antioquia, Colombia",
    }


@pytest.fixture
def sample_rights_classification() -> RightsClassification:
    """Create a sample rights classification."""
    return RightsClassification(
        has_land_rights=True,
        has_customary_rights=True,
        has_indigenous_status=True,
        fpic_required=True,
        applicable_conventions=["ILO 169", "UNDRIP"],
        legal_framework="Colombian Constitution Art. 330",
    )


@pytest.fixture
def sample_stakeholder_indigenous(
    sample_contact_info,
    sample_rights_classification,
) -> StakeholderRecord:
    """Create a sample indigenous stakeholder record."""
    return StakeholderRecord(
        stakeholder_id="STK-IND-001",
        operator_id="operator-001",
        name="Wayuu Community - La Guajira",
        category=StakeholderCategory.INDIGENOUS_COMMUNITY,
        status=StakeholderStatus.ACTIVE,
        country_code="CO",
        region="La Guajira",
        commodity=EUDRCommodity.COFFEE,
        contact_info=sample_contact_info,
        rights_classification=sample_rights_classification,
        population_estimate=1500,
        affected_area_hectares=Decimal("450.0"),
        engagement_history=[],
        notes="Traditional territory overlaps with coffee production area.",
    )


@pytest.fixture
def sample_stakeholder_community() -> StakeholderRecord:
    """Create a sample local community stakeholder record."""
    return StakeholderRecord(
        stakeholder_id="STK-COM-001",
        operator_id="operator-001",
        name="Vereda San Jose Community",
        category=StakeholderCategory.LOCAL_COMMUNITY,
        status=StakeholderStatus.ACTIVE,
        country_code="CO",
        region="Antioquia",
        commodity=EUDRCommodity.COFFEE,
        contact_info={
            "primary_name": "Carlos Ramirez",
            "role": "Community President",
            "email": "carlos.r@example.com",
            "phone": "+57-300-555-0202",
            "preferred_language": "es",
        },
        rights_classification=RightsClassification(
            has_land_rights=True,
            has_customary_rights=False,
            has_indigenous_status=False,
            fpic_required=False,
            applicable_conventions=[],
            legal_framework="Colombian Rural Development Law",
        ),
        population_estimate=350,
        affected_area_hectares=Decimal("120.0"),
    )


@pytest.fixture
def sample_stakeholder_cooperative() -> StakeholderRecord:
    """Create a sample cooperative stakeholder record."""
    return StakeholderRecord(
        stakeholder_id="STK-COOP-001",
        operator_id="operator-001",
        name="Cooperativa Cafe Verde",
        category=StakeholderCategory.COOPERATIVE,
        status=StakeholderStatus.ACTIVE,
        country_code="CO",
        region="Huila",
        commodity=EUDRCommodity.COFFEE,
        contact_info={
            "primary_name": "Ana Lopez",
            "role": "Cooperative Manager",
            "email": "ana.lopez@cafeverde.coop",
            "phone": "+57-300-555-0303",
            "preferred_language": "es",
        },
        rights_classification=RightsClassification(
            has_land_rights=True,
            has_customary_rights=False,
            has_indigenous_status=False,
            fpic_required=False,
            applicable_conventions=[],
            legal_framework="Colombian Cooperative Law",
        ),
        population_estimate=200,
        affected_area_hectares=Decimal("80.0"),
    )


@pytest.fixture
def sample_stakeholder_ngo() -> StakeholderRecord:
    """Create a sample NGO stakeholder record."""
    return StakeholderRecord(
        stakeholder_id="STK-NGO-001",
        operator_id="operator-001",
        name="Amazon Conservation Foundation",
        category=StakeholderCategory.NGO,
        status=StakeholderStatus.ACTIVE,
        country_code="BR",
        region="Amazonas",
        commodity=EUDRCommodity.SOYA,
        contact_info={
            "primary_name": "Dr. Elena Silva",
            "role": "Director",
            "email": "elena.silva@acf.org",
            "phone": "+55-92-555-0404",
            "preferred_language": "pt",
        },
        rights_classification=RightsClassification(
            has_land_rights=False,
            has_customary_rights=False,
            has_indigenous_status=False,
            fpic_required=False,
            applicable_conventions=[],
            legal_framework="",
        ),
        population_estimate=0,
        affected_area_hectares=Decimal("0"),
    )


@pytest.fixture
def multiple_stakeholders(
    sample_stakeholder_indigenous,
    sample_stakeholder_community,
    sample_stakeholder_cooperative,
    sample_stakeholder_ngo,
) -> List[StakeholderRecord]:
    """Provide multiple stakeholder records for testing."""
    return [
        sample_stakeholder_indigenous,
        sample_stakeholder_community,
        sample_stakeholder_cooperative,
        sample_stakeholder_ngo,
    ]


# ---------------------------------------------------------------------------
# FPIC fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fpic_stage_config() -> Dict:
    """Create sample FPIC stage configuration."""
    return {
        "notification_period_days": 30,
        "deliberation_period_days": 90,
        "consultation_min_sessions": 3,
        "documentation_language": "es",
        "independent_facilitator_required": True,
        "minimum_attendance_percentage": Decimal("60"),
    }


@pytest.fixture
def sample_fpic_workflow(
    sample_stakeholder_indigenous,
    sample_fpic_stage_config,
) -> FPICWorkflow:
    """Create a sample FPIC workflow in NOTIFICATION stage."""
    return FPICWorkflow(
        workflow_id="FPIC-001",
        stakeholder_id="STK-IND-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        current_stage=FPICStage.NOTIFICATION,
        stage_config=sample_fpic_stage_config,
        initiated_at=datetime.now(tz=timezone.utc) - timedelta(days=5),
        stage_history=[
            {
                "stage": FPICStage.NOTIFICATION.value,
                "entered_at": (datetime.now(tz=timezone.utc) - timedelta(days=5)).isoformat(),
                "notes": "Initial notification sent to community leaders.",
            }
        ],
        consent_status=ConsentStatus.PENDING,
        consultation_records=[],
        evidence_documents=[],
    )


@pytest.fixture
def fpic_workflow_all_stages() -> FPICWorkflow:
    """Create an FPIC workflow that has progressed through all stages."""
    now = datetime.now(tz=timezone.utc)
    return FPICWorkflow(
        workflow_id="FPIC-002",
        stakeholder_id="STK-IND-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        current_stage=FPICStage.MONITORING,
        stage_config={
            "notification_period_days": 30,
            "deliberation_period_days": 90,
        },
        initiated_at=now - timedelta(days=365),
        stage_history=[
            {"stage": FPICStage.NOTIFICATION.value, "entered_at": (now - timedelta(days=365)).isoformat()},
            {"stage": FPICStage.INFORMATION_SHARING.value, "entered_at": (now - timedelta(days=335)).isoformat()},
            {"stage": FPICStage.CONSULTATION.value, "entered_at": (now - timedelta(days=300)).isoformat()},
            {"stage": FPICStage.DELIBERATION.value, "entered_at": (now - timedelta(days=240)).isoformat()},
            {"stage": FPICStage.DECISION.value, "entered_at": (now - timedelta(days=150)).isoformat()},
            {"stage": FPICStage.AGREEMENT.value, "entered_at": (now - timedelta(days=120)).isoformat()},
            {"stage": FPICStage.MONITORING.value, "entered_at": (now - timedelta(days=90)).isoformat()},
        ],
        consent_status=ConsentStatus.GRANTED,
        consultation_records=["CON-001", "CON-002", "CON-003"],
        evidence_documents=["DOC-FPIC-001", "DOC-FPIC-002"],
    )


@pytest.fixture
def fpic_workflow_consented() -> FPICWorkflow:
    """Create an FPIC workflow with consent granted."""
    return FPICWorkflow(
        workflow_id="FPIC-003",
        stakeholder_id="STK-IND-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        current_stage=FPICStage.AGREEMENT,
        stage_config={"deliberation_period_days": 90},
        initiated_at=datetime.now(tz=timezone.utc) - timedelta(days=180),
        stage_history=[],
        consent_status=ConsentStatus.GRANTED,
        consent_recorded_at=datetime.now(tz=timezone.utc) - timedelta(days=30),
        consent_evidence="consent-doc-signed-2026.pdf",
        consultation_records=["CON-001", "CON-002"],
        evidence_documents=["DOC-CONSENT-001"],
    )


@pytest.fixture
def fpic_workflow_withheld() -> FPICWorkflow:
    """Create an FPIC workflow with consent withheld."""
    return FPICWorkflow(
        workflow_id="FPIC-004",
        stakeholder_id="STK-IND-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.PALM_OIL,
        current_stage=FPICStage.DECISION,
        stage_config={"deliberation_period_days": 90},
        initiated_at=datetime.now(tz=timezone.utc) - timedelta(days=200),
        stage_history=[],
        consent_status=ConsentStatus.WITHHELD,
        consent_recorded_at=datetime.now(tz=timezone.utc) - timedelta(days=10),
        consent_evidence="withhold-declaration-2026.pdf",
        consultation_records=["CON-003"],
        evidence_documents=["DOC-WITHHOLD-001"],
    )


# ---------------------------------------------------------------------------
# Grievance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_grievance_critical() -> GrievanceRecord:
    """Create a sample critical grievance record."""
    return GrievanceRecord(
        grievance_id="GRV-001",
        stakeholder_id="STK-IND-001",
        operator_id="operator-001",
        title="Unauthorized access to sacred site",
        description=(
            "Heavy machinery was observed operating within 200 meters of "
            "the sacred ancestral site without community permission."
        ),
        severity=GrievanceSeverity.CRITICAL,
        status=GrievanceStatus.SUBMITTED,
        channel="field_visit",
        submitted_at=datetime.now(tz=timezone.utc) - timedelta(days=3),
        sla_deadline=datetime.now(tz=timezone.utc) + timedelta(hours=21),
        category="land_rights_violation",
        investigation_notes=[],
        resolution_actions=[],
    )


@pytest.fixture
def sample_grievance_standard() -> GrievanceRecord:
    """Create a sample standard grievance record."""
    return GrievanceRecord(
        grievance_id="GRV-002",
        stakeholder_id="STK-COM-001",
        operator_id="operator-001",
        title="Water quality concern downstream",
        description=(
            "Community members report turbidity increase in the river downstream "
            "from the processing facility since operations began last month."
        ),
        severity=GrievanceSeverity.STANDARD,
        status=GrievanceStatus.SUBMITTED,
        channel="email",
        submitted_at=datetime.now(tz=timezone.utc) - timedelta(days=7),
        sla_deadline=datetime.now(tz=timezone.utc) + timedelta(days=7),
        category="environmental_impact",
        investigation_notes=[],
        resolution_actions=[],
    )


@pytest.fixture
def sample_investigation_notes() -> List[Dict]:
    """Create sample investigation notes."""
    return [
        {
            "note_id": "INV-NOTE-001",
            "investigator": "John Smith",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "finding": "Initial site visit confirmed machinery presence within restricted zone.",
            "evidence_ref": "PHOTO-GRV-001-01",
        },
        {
            "note_id": "INV-NOTE-002",
            "investigator": "John Smith",
            "timestamp": (datetime.now(tz=timezone.utc) + timedelta(hours=4)).isoformat(),
            "finding": "Contractor confirmed unauthorized entry, operations halted immediately.",
            "evidence_ref": "REPORT-GRV-001-01",
        },
    ]


@pytest.fixture
def sample_resolution_actions() -> List[Dict]:
    """Create sample resolution actions."""
    return [
        {
            "action_id": "RES-ACT-001",
            "type": "immediate_halt",
            "description": "All operations within 500m of sacred site halted.",
            "responsible_party": "Operations Manager",
            "deadline": (datetime.now(tz=timezone.utc) + timedelta(days=1)).isoformat(),
            "status": "completed",
        },
        {
            "action_id": "RES-ACT-002",
            "type": "corrective_action",
            "description": "Buffer zone of 1km established around sacred site.",
            "responsible_party": "Environmental Officer",
            "deadline": (datetime.now(tz=timezone.utc) + timedelta(days=14)).isoformat(),
            "status": "in_progress",
        },
    ]


@pytest.fixture
def multiple_grievances(
    sample_grievance_critical,
    sample_grievance_standard,
) -> List[GrievanceRecord]:
    """Provide multiple grievance records for testing."""
    return [
        sample_grievance_critical,
        sample_grievance_standard,
        GrievanceRecord(
            grievance_id="GRV-003",
            stakeholder_id="STK-COOP-001",
            operator_id="operator-001",
            title="Payment delay for certified product",
            description="Payment for Q4 2025 delivery not received after 60 days.",
            severity=GrievanceSeverity.MINOR,
            status=GrievanceStatus.RESOLVED,
            channel="phone",
            submitted_at=datetime.now(tz=timezone.utc) - timedelta(days=45),
            category="economic",
        ),
    ]


# ---------------------------------------------------------------------------
# Consultation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_participants() -> List[Dict]:
    """Create sample consultation participants."""
    return [
        {
            "participant_id": "PART-001",
            "name": "Maria Gonzalez",
            "role": "Community Leader",
            "stakeholder_id": "STK-IND-001",
            "organization": "Wayuu Community",
            "attended": True,
        },
        {
            "participant_id": "PART-002",
            "name": "Carlos Ramirez",
            "role": "Community President",
            "stakeholder_id": "STK-COM-001",
            "organization": "Vereda San Jose",
            "attended": True,
        },
        {
            "participant_id": "PART-003",
            "name": "Dr. Elena Silva",
            "role": "Observer",
            "stakeholder_id": "STK-NGO-001",
            "organization": "Amazon Conservation Foundation",
            "attended": True,
        },
        {
            "participant_id": "PART-004",
            "name": "Pedro Torres",
            "role": "Facilitator",
            "stakeholder_id": None,
            "organization": "Independent Mediation Service",
            "attended": True,
        },
    ]


@pytest.fixture
def sample_outcomes_commitments() -> List[Dict]:
    """Create sample consultation outcomes and commitments."""
    return [
        {
            "outcome_id": "OUT-001",
            "type": "agreement",
            "description": "Community agrees to allow monitoring stations on boundary.",
            "responsible_parties": ["operator-001", "STK-IND-001"],
            "deadline": (datetime.now(tz=timezone.utc) + timedelta(days=60)).isoformat(),
            "status": "pending",
        },
        {
            "outcome_id": "OUT-002",
            "type": "commitment",
            "description": "Operator commits to monthly water quality reports.",
            "responsible_parties": ["operator-001"],
            "deadline": None,
            "status": "ongoing",
        },
        {
            "outcome_id": "OUT-003",
            "type": "action_item",
            "description": "Schedule follow-up meeting within 30 days.",
            "responsible_parties": ["operator-001"],
            "deadline": (datetime.now(tz=timezone.utc) + timedelta(days=30)).isoformat(),
            "status": "pending",
        },
    ]


@pytest.fixture
def sample_consultation_community(
    sample_participants,
    sample_outcomes_commitments,
) -> ConsultationRecord:
    """Create a sample community consultation record."""
    return ConsultationRecord(
        consultation_id="CON-001",
        operator_id="operator-001",
        consultation_type=ConsultationType.COMMUNITY_MEETING,
        title="Quarterly Community Engagement Meeting - Q1 2026",
        description="Regular quarterly meeting to discuss operations impact and community concerns.",
        scheduled_at=datetime.now(tz=timezone.utc) - timedelta(days=7),
        conducted_at=datetime.now(tz=timezone.utc) - timedelta(days=7),
        location="Community Center, Vereda San Jose, Antioquia",
        stakeholder_ids=["STK-IND-001", "STK-COM-001", "STK-NGO-001"],
        participants=sample_participants,
        outcomes=sample_outcomes_commitments,
        evidence_refs=["PHOTO-CON-001", "MINUTES-CON-001"],
        language="es",
        facilitator="Pedro Torres",
        status="completed",
    )


@pytest.fixture
def sample_consultation_bilateral() -> ConsultationRecord:
    """Create a sample bilateral consultation record."""
    return ConsultationRecord(
        consultation_id="CON-002",
        operator_id="operator-001",
        consultation_type=ConsultationType.BILATERAL,
        title="Cooperative Partnership Review",
        description="Review of cooperative partnership terms and certification progress.",
        scheduled_at=datetime.now(tz=timezone.utc) - timedelta(days=3),
        conducted_at=datetime.now(tz=timezone.utc) - timedelta(days=3),
        location="Cooperativa Cafe Verde Office, Huila",
        stakeholder_ids=["STK-COOP-001"],
        participants=[
            {
                "participant_id": "PART-010",
                "name": "Ana Lopez",
                "role": "Cooperative Manager",
                "stakeholder_id": "STK-COOP-001",
                "organization": "Cooperativa Cafe Verde",
                "attended": True,
            },
        ],
        outcomes=[
            {
                "outcome_id": "OUT-010",
                "type": "agreement",
                "description": "Extend partnership for 2 additional years.",
                "responsible_parties": ["operator-001", "STK-COOP-001"],
                "deadline": None,
                "status": "agreed",
            }
        ],
        evidence_refs=["MINUTES-CON-002"],
        language="es",
        status="completed",
    )


@pytest.fixture
def consultation_with_evidence(sample_consultation_community) -> ConsultationRecord:
    """Create consultation with extensive evidence documentation."""
    record = sample_consultation_community
    record.evidence_refs = [
        "PHOTO-CON-001-01",
        "PHOTO-CON-001-02",
        "VIDEO-CON-001",
        "MINUTES-CON-001-SIGNED",
        "ATTENDANCE-CON-001",
        "MAP-CON-001",
    ]
    return record


# ---------------------------------------------------------------------------
# Communication fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_communication_email() -> CommunicationRecord:
    """Create a sample email communication record."""
    return CommunicationRecord(
        communication_id="COMM-001",
        operator_id="operator-001",
        stakeholder_ids=["STK-IND-001", "STK-COM-001"],
        channel=CommunicationChannel.EMAIL,
        subject="Upcoming Community Consultation - March 2026",
        body=(
            "Dear Community Leaders,\n\n"
            "We would like to invite you to the quarterly community engagement meeting "
            "scheduled for March 20, 2026. The meeting will cover environmental monitoring "
            "results and updates on the FPIC process.\n\n"
            "Please confirm your attendance.\n\nBest regards,\nOperations Team"
        ),
        sent_at=datetime.now(tz=timezone.utc) - timedelta(days=14),
        delivery_status=DeliveryStatus.DELIVERED,
        template_id="TPL-INVITE-001",
        language="es",
        campaign_id=None,
    )


@pytest.fixture
def sample_communication_sms() -> CommunicationRecord:
    """Create a sample SMS communication record."""
    return CommunicationRecord(
        communication_id="COMM-002",
        operator_id="operator-001",
        stakeholder_ids=["STK-COM-001"],
        channel=CommunicationChannel.SMS,
        subject="Meeting Reminder",
        body="Reminder: Community meeting tomorrow at 10AM, Community Center.",
        sent_at=datetime.now(tz=timezone.utc) - timedelta(days=8),
        delivery_status=DeliveryStatus.DELIVERED,
        language="es",
    )


@pytest.fixture
def sample_campaign() -> Dict:
    """Create a sample communication campaign."""
    return {
        "campaign_id": "CAMP-001",
        "name": "Q1 2026 Stakeholder Outreach",
        "description": "Quarterly outreach to all affected stakeholders",
        "target_stakeholders": ["STK-IND-001", "STK-COM-001", "STK-COOP-001", "STK-NGO-001"],
        "channels": [CommunicationChannel.EMAIL.value, CommunicationChannel.SMS.value],
        "scheduled_start": datetime.now(tz=timezone.utc).isoformat(),
        "scheduled_end": (datetime.now(tz=timezone.utc) + timedelta(days=7)).isoformat(),
        "status": "active",
        "communications_sent": 0,
        "communications_total": 8,
    }


@pytest.fixture
def communication_template() -> CommunicationTemplate:
    """Create a sample communication template."""
    return CommunicationTemplate(
        template_id="TPL-INVITE-001",
        name="Community Meeting Invitation",
        channel=CommunicationChannel.EMAIL,
        subject_template="Upcoming {{meeting_type}} - {{date}}",
        body_template=(
            "Dear {{stakeholder_name}},\n\n"
            "We would like to invite you to the {{meeting_type}} "
            "scheduled for {{date}} at {{location}}.\n\n"
            "Agenda: {{agenda}}\n\n"
            "Please confirm your attendance.\n\n"
            "Best regards,\n{{operator_name}}"
        ),
        language="es",
        variables=["stakeholder_name", "meeting_type", "date", "location", "agenda", "operator_name"],
        active=True,
    )


# ---------------------------------------------------------------------------
# Assessment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_engagement_assessment() -> EngagementAssessment:
    """Create a sample engagement assessment with moderate scores."""
    return EngagementAssessment(
        assessment_id="EA-001",
        operator_id="operator-001",
        stakeholder_id="STK-IND-001",
        assessment_date=datetime.now(tz=timezone.utc),
        dimension_scores={
            EngagementDimension.INCLUSIVENESS: Decimal("72"),
            EngagementDimension.TRANSPARENCY: Decimal("68"),
            EngagementDimension.RESPONSIVENESS: Decimal("75"),
            EngagementDimension.ACCOUNTABILITY: Decimal("65"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("80"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("78"),
        },
        composite_score=Decimal("73"),
        recommendations=[
            "Improve transparency by publishing regular impact reports in local language.",
            "Increase responsiveness by reducing grievance response time to 48 hours.",
        ],
        evidence_refs=["EA-SURVEY-001", "EA-INTERVIEW-001"],
    )


@pytest.fixture
def assessment_high_score() -> EngagementAssessment:
    """Create an engagement assessment with high scores."""
    return EngagementAssessment(
        assessment_id="EA-002",
        operator_id="operator-001",
        stakeholder_id="STK-COM-001",
        assessment_date=datetime.now(tz=timezone.utc),
        dimension_scores={
            EngagementDimension.INCLUSIVENESS: Decimal("92"),
            EngagementDimension.TRANSPARENCY: Decimal("88"),
            EngagementDimension.RESPONSIVENESS: Decimal("95"),
            EngagementDimension.ACCOUNTABILITY: Decimal("90"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("91"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("94"),
        },
        composite_score=Decimal("91.7"),
        recommendations=[],
        evidence_refs=["EA-SURVEY-002"],
    )


@pytest.fixture
def assessment_low_score() -> EngagementAssessment:
    """Create an engagement assessment with low scores."""
    return EngagementAssessment(
        assessment_id="EA-003",
        operator_id="operator-002",
        stakeholder_id="STK-IND-001",
        assessment_date=datetime.now(tz=timezone.utc),
        dimension_scores={
            EngagementDimension.INCLUSIVENESS: Decimal("25"),
            EngagementDimension.TRANSPARENCY: Decimal("30"),
            EngagementDimension.RESPONSIVENESS: Decimal("20"),
            EngagementDimension.ACCOUNTABILITY: Decimal("28"),
            EngagementDimension.CULTURAL_SENSITIVITY: Decimal("35"),
            EngagementDimension.RIGHTS_RESPECT: Decimal("22"),
        },
        composite_score=Decimal("26.7"),
        recommendations=[
            "Urgent: Establish meaningful consultation mechanisms with community.",
            "Urgent: Appoint culturally competent engagement officer.",
            "Urgent: Conduct rights impact assessment immediately.",
            "Develop community benefit sharing agreement.",
            "Implement grievance mechanism accessible in local languages.",
        ],
        evidence_refs=["EA-AUDIT-001"],
    )


# ---------------------------------------------------------------------------
# Report fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_compliance_report() -> ComplianceReport:
    """Create a sample compliance report."""
    return ComplianceReport(
        report_id="RPT-001",
        operator_id="operator-001",
        report_type=ReportType.ENGAGEMENT_SUMMARY,
        title="Stakeholder Engagement Compliance Report - Q1 2026",
        period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 3, 31, tzinfo=timezone.utc),
        generated_at=datetime.now(tz=timezone.utc),
        format=ReportFormat.JSON,
        sections={
            "stakeholder_overview": {"total": 4, "active": 4},
            "fpic_status": {"total_workflows": 2, "consented": 1, "pending": 1},
            "grievance_summary": {"total": 3, "resolved": 1, "open": 2},
            "consultation_register": {"total_sessions": 5, "stakeholders_engaged": 4},
            "engagement_scores": {"average_composite": "73.0"},
        },
        provenance_hash="",
    )


@pytest.fixture
def dds_summary_report() -> ComplianceReport:
    """Create a DDS summary report for stakeholder engagement."""
    return ComplianceReport(
        report_id="RPT-DDS-001",
        operator_id="operator-001",
        report_type=ReportType.DDS_SUMMARY,
        title="Due Diligence Statement - Stakeholder Engagement Section",
        period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 3, 31, tzinfo=timezone.utc),
        generated_at=datetime.now(tz=timezone.utc),
        format=ReportFormat.JSON,
        sections={
            "article_10_compliance": {
                "stakeholder_identification": "complete",
                "fpic_status": "in_progress",
                "grievance_mechanism": "operational",
            },
            "indigenous_rights": {
                "fpic_required": True,
                "fpic_status": "consent_granted",
                "applicable_conventions": ["ILO 169", "UNDRIP"],
            },
        },
        provenance_hash="abc" * 21 + "a",
    )


@pytest.fixture
def fpic_compliance_report() -> ComplianceReport:
    """Create an FPIC compliance report."""
    return ComplianceReport(
        report_id="RPT-FPIC-001",
        operator_id="operator-001",
        report_type=ReportType.FPIC_COMPLIANCE,
        title="FPIC Process Compliance Report",
        period_start=datetime(2025, 6, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 3, 31, tzinfo=timezone.utc),
        generated_at=datetime.now(tz=timezone.utc),
        format=ReportFormat.PDF,
        sections={
            "fpic_workflows": [
                {
                    "workflow_id": "FPIC-001",
                    "stakeholder": "Wayuu Community",
                    "stage": "monitoring",
                    "consent": "granted",
                },
            ],
            "consultation_log": {"total": 12, "with_indigenous": 8},
            "compliance_status": "compliant",
        },
        provenance_hash="def" * 21 + "d",
    )
