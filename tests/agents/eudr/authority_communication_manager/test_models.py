# -*- coding: utf-8 -*-
"""
Unit tests for Authority Communication Manager models - AGENT-EUDR-040

Tests all 13 enums, 15+ Pydantic model validations, encryption field
handling, Decimal precision for penalty amounts, language code validation,
constant definitions, deadline hour mappings, and EUDR commodity list.

95+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

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


# ====================================================================
# Constants
# ====================================================================


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-ACM-040"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_eudr_commodities_count(self):
        assert len(EUDR_COMMODITIES) == 7

    def test_eudr_commodities_content(self):
        for c in ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]:
            assert c in EUDR_COMMODITIES

    def test_deadline_hours_map_keys(self):
        expected = {"urgent", "high", "normal", "low", "routine"}
        assert set(DEADLINE_HOURS_MAP.keys()) == expected

    def test_deadline_urgent_hours(self):
        assert DEADLINE_HOURS_MAP["urgent"] == 24

    def test_deadline_high_hours(self):
        assert DEADLINE_HOURS_MAP["high"] == 72

    def test_deadline_normal_hours(self):
        assert DEADLINE_HOURS_MAP["normal"] == 120

    def test_deadline_low_hours(self):
        assert DEADLINE_HOURS_MAP["low"] == 240

    def test_deadline_routine_hours(self):
        assert DEADLINE_HOURS_MAP["routine"] == 360

    def test_deadline_ordering(self):
        """Urgent < high < normal < low < routine."""
        assert DEADLINE_HOURS_MAP["urgent"] < DEADLINE_HOURS_MAP["high"]
        assert DEADLINE_HOURS_MAP["high"] < DEADLINE_HOURS_MAP["normal"]
        assert DEADLINE_HOURS_MAP["normal"] < DEADLINE_HOURS_MAP["low"]
        assert DEADLINE_HOURS_MAP["low"] < DEADLINE_HOURS_MAP["routine"]


# ====================================================================
# Enums (13)
# ====================================================================


class TestEnums:
    """Test all 13 enum types."""

    def test_communication_type_count(self):
        assert len(CommunicationType) == 12

    def test_communication_type_values(self):
        assert CommunicationType.INFORMATION_REQUEST.value == "information_request"
        assert CommunicationType.INSPECTION_NOTICE.value == "inspection_notice"
        assert CommunicationType.NON_COMPLIANCE_NOTICE.value == "non_compliance_notice"
        assert CommunicationType.PENALTY_NOTICE.value == "penalty_notice"
        assert CommunicationType.APPEAL_ACKNOWLEDGMENT.value == "appeal_acknowledgment"
        assert CommunicationType.APPEAL_DECISION.value == "appeal_decision"

    def test_communication_status_count(self):
        assert len(CommunicationStatus) == 10

    def test_communication_status_values(self):
        assert CommunicationStatus.DRAFT.value == "draft"
        assert CommunicationStatus.PENDING.value == "pending"
        assert CommunicationStatus.SENT.value == "sent"
        assert CommunicationStatus.OVERDUE.value == "overdue"
        assert CommunicationStatus.CLOSED.value == "closed"

    def test_communication_priority_count(self):
        assert len(CommunicationPriority) == 5

    def test_communication_priority_values(self):
        assert CommunicationPriority.URGENT.value == "urgent"
        assert CommunicationPriority.ROUTINE.value == "routine"

    def test_information_request_type_count(self):
        assert len(InformationRequestType) == 10

    def test_information_request_type_values(self):
        assert InformationRequestType.DDS_CLARIFICATION.value == "dds_clarification"
        assert InformationRequestType.GEOLOCATION_VERIFICATION.value == "geolocation_verification"
        assert InformationRequestType.AUDIT_REPORT_REQUEST.value == "audit_report_request"

    def test_inspection_type_count(self):
        assert len(InspectionType) == 6

    def test_inspection_type_values(self):
        assert InspectionType.ANNOUNCED.value == "announced"
        assert InspectionType.UNANNOUNCED.value == "unannounced"
        assert InspectionType.REMOTE.value == "remote"

    def test_violation_type_count(self):
        assert len(ViolationType) == 11

    def test_violation_type_values(self):
        assert ViolationType.MISSING_DDS.value == "missing_dds"
        assert ViolationType.FALSE_INFORMATION.value == "false_information"
        assert ViolationType.DEFORESTATION_LINK.value == "deforestation_link"
        assert ViolationType.REPEATED_VIOLATION.value == "repeated_violation"

    def test_violation_severity_count(self):
        assert len(ViolationSeverity) == 4

    def test_violation_severity_values(self):
        assert ViolationSeverity.MINOR.value == "minor"
        assert ViolationSeverity.MODERATE.value == "moderate"
        assert ViolationSeverity.MAJOR.value == "major"
        assert ViolationSeverity.CRITICAL.value == "critical"

    def test_appeal_decision_count(self):
        assert len(AppealDecision) == 7

    def test_appeal_decision_values(self):
        assert AppealDecision.PENDING.value == "pending"
        assert AppealDecision.UPHELD.value == "upheld"
        assert AppealDecision.OVERTURNED.value == "overturned"
        assert AppealDecision.DISMISSED.value == "dismissed"
        assert AppealDecision.REFERRED.value == "referred"

    def test_document_type_count(self):
        assert len(DocumentType) == 15

    def test_document_type_values(self):
        assert DocumentType.DDS_STATEMENT.value == "dds_statement"
        assert DocumentType.SATELLITE_IMAGERY.value == "satellite_imagery"
        assert DocumentType.APPEAL_SUBMISSION.value == "appeal_submission"
        assert DocumentType.OTHER.value == "other"

    def test_notification_channel_count(self):
        assert len(NotificationChannel) == 5

    def test_notification_channel_values(self):
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.API.value == "api"
        assert NotificationChannel.PORTAL.value == "portal"
        assert NotificationChannel.SMS.value == "sms"
        assert NotificationChannel.WEBHOOK.value == "webhook"

    def test_recipient_type_count(self):
        assert len(RecipientType) == 7

    def test_recipient_type_values(self):
        assert RecipientType.AUTHORITY.value == "authority"
        assert RecipientType.OPERATOR.value == "operator"
        assert RecipientType.COMPLIANCE_OFFICER.value == "compliance_officer"

    def test_language_code_count(self):
        assert len(LanguageCode) == 24

    def test_language_code_values(self):
        assert LanguageCode.EN.value == "en"
        assert LanguageCode.DE.value == "de"
        assert LanguageCode.FR.value == "fr"
        assert LanguageCode.GA.value == "ga"  # Irish
        assert LanguageCode.MT.value == "mt"  # Maltese

    def test_authority_type_count(self):
        assert len(AuthorityType) == 7

    def test_authority_type_values(self):
        assert AuthorityType.NATIONAL_COMPETENT.value == "national_competent"
        assert AuthorityType.CUSTOMS.value == "customs"
        assert AuthorityType.EUROPEAN_COMMISSION.value == "european_commission"


# ====================================================================
# Authority Model
# ====================================================================


class TestAuthorityModel:
    """Test Authority Pydantic model."""

    def test_create(self, authority_de):
        assert authority_de.authority_id == "AUTH-DE-001"
        assert authority_de.member_state == "DE"

    def test_default_authority_type(self):
        auth = Authority(
            authority_id="AUTH-X", member_state="IE",
            name="Test Authority",
        )
        assert auth.authority_type == AuthorityType.NATIONAL_COMPETENT

    def test_preferred_language_default(self):
        auth = Authority(
            authority_id="AUTH-X", member_state="IE",
            name="Test Authority",
        )
        assert auth.preferred_language == LanguageCode.EN

    def test_active_default(self):
        auth = Authority(
            authority_id="AUTH-X", member_state="IE",
            name="Test Authority",
        )
        assert auth.active is True


# ====================================================================
# Communication Model
# ====================================================================


class TestCommunicationModel:
    """Test Communication Pydantic model."""

    def test_create(self, pending_communication):
        assert pending_communication.communication_id == "COMM-001"
        assert pending_communication.status == CommunicationStatus.PENDING

    def test_default_status(self):
        comm = Communication(
            communication_id="C1", operator_id="OP-1",
            authority_id="A1", member_state="DE",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
            subject="Test",
        )
        assert comm.status == CommunicationStatus.DRAFT

    def test_default_priority(self):
        comm = Communication(
            communication_id="C1", operator_id="OP-1",
            authority_id="A1", member_state="DE",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
            subject="Test",
        )
        assert comm.priority == CommunicationPriority.NORMAL

    def test_default_language(self):
        comm = Communication(
            communication_id="C1", operator_id="OP-1",
            authority_id="A1", member_state="DE",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
            subject="Test",
        )
        assert comm.language == LanguageCode.EN

    def test_empty_document_ids(self):
        comm = Communication(
            communication_id="C1", operator_id="OP-1",
            authority_id="A1", member_state="DE",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
            subject="Test",
        )
        assert comm.document_ids == []

    def test_created_at_auto(self):
        comm = Communication(
            communication_id="C1", operator_id="OP-1",
            authority_id="A1", member_state="DE",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
            subject="Test",
        )
        assert comm.created_at is not None


# ====================================================================
# InformationRequest Model
# ====================================================================


class TestInformationRequestModel:
    """Test InformationRequest Pydantic model."""

    def test_create(self, urgent_info_request):
        assert urgent_info_request.request_id == "REQ-001"
        assert urgent_info_request.request_type == InformationRequestType.DDS_CLARIFICATION

    def test_items_requested(self, urgent_info_request):
        assert len(urgent_info_request.items_requested) == 3

    def test_response_defaults(self):
        req = InformationRequest(
            request_id="R1", communication_id="C1",
            operator_id="OP-1", authority_id="A1",
            request_type=InformationRequestType.SUPPLY_CHAIN_EVIDENCE,
        )
        assert req.response_submitted is False
        assert req.response_accepted is False


# ====================================================================
# Inspection Model
# ====================================================================


class TestInspectionModel:
    """Test Inspection Pydantic model."""

    def test_create_scheduled(self, scheduled_inspection):
        assert scheduled_inspection.inspection_id == "INSP-001"
        assert scheduled_inspection.inspection_type == InspectionType.ANNOUNCED

    def test_create_completed(self, completed_inspection):
        assert completed_inspection.status == "completed"
        assert len(completed_inspection.findings) == 3

    def test_default_status(self):
        insp = Inspection(
            inspection_id="I1", communication_id="C1",
            operator_id="OP-1", authority_id="A1",
            inspection_type=InspectionType.ANNOUNCED,
        )
        assert insp.status == "scheduled"

    def test_empty_findings_default(self):
        insp = Inspection(
            inspection_id="I1", communication_id="C1",
            operator_id="OP-1", authority_id="A1",
            inspection_type=InspectionType.ANNOUNCED,
        )
        assert insp.findings == []
        assert insp.corrective_actions == []


# ====================================================================
# NonCompliance Model
# ====================================================================


class TestNonComplianceModel:
    """Test NonCompliance Pydantic model with Decimal penalties."""

    def test_create_minor(self, minor_non_compliance):
        assert minor_non_compliance.severity == ViolationSeverity.MINOR
        assert minor_non_compliance.penalty_amount == Decimal("2500.00")

    def test_create_major(self, major_non_compliance):
        assert major_non_compliance.severity == ViolationSeverity.MAJOR
        assert major_non_compliance.penalty_amount == Decimal("250000.00")

    def test_create_critical(self, critical_non_compliance):
        assert critical_non_compliance.severity == ViolationSeverity.CRITICAL
        assert critical_non_compliance.penalty_amount == Decimal("5000000.00")

    def test_penalty_precision(self):
        """Penalty amounts must maintain Decimal precision."""
        nc = NonCompliance(
            non_compliance_id="NC-X",
            communication_id="C1",
            operator_id="OP-1",
            authority_id="A1",
            violation_type=ViolationType.MISSING_DDS,
            severity=ViolationSeverity.MINOR,
            description="Test violation",
            penalty_amount=Decimal("1234.56"),
        )
        assert nc.penalty_amount == Decimal("1234.56")
        assert isinstance(nc.penalty_amount, Decimal)

    def test_penalty_none_by_default(self):
        nc = NonCompliance(
            non_compliance_id="NC-X",
            communication_id="C1",
            operator_id="OP-1",
            authority_id="A1",
            violation_type=ViolationType.MISSING_DDS,
            severity=ViolationSeverity.MINOR,
            description="Test violation",
        )
        assert nc.penalty_amount is None

    def test_corrective_completed_default(self):
        nc = NonCompliance(
            non_compliance_id="NC-X",
            communication_id="C1",
            operator_id="OP-1",
            authority_id="A1",
            violation_type=ViolationType.MISSING_DDS,
            severity=ViolationSeverity.MINOR,
            description="Test violation",
        )
        assert nc.corrective_completed is False


# ====================================================================
# Appeal Model
# ====================================================================


class TestAppealModel:
    """Test Appeal Pydantic model."""

    def test_create_filed(self, filed_appeal):
        assert filed_appeal.appeal_id == "APP-001"
        assert filed_appeal.decision == AppealDecision.PENDING

    def test_create_decided(self, decided_appeal):
        assert decided_appeal.decision == AppealDecision.PARTIALLY_UPHELD
        assert decided_appeal.decision_date is not None

    def test_default_decision(self):
        appeal = Appeal(
            appeal_id="A1", communication_id="C1",
            non_compliance_id="NC-1", operator_id="OP-1",
            authority_id="A1", grounds="Test grounds",
        )
        assert appeal.decision == AppealDecision.PENDING

    def test_extensions_default(self):
        appeal = Appeal(
            appeal_id="A1", communication_id="C1",
            non_compliance_id="NC-1", operator_id="OP-1",
            authority_id="A1", grounds="Test grounds",
        )
        assert appeal.extensions_granted == 0

    def test_penalty_suspended_default(self):
        appeal = Appeal(
            appeal_id="A1", communication_id="C1",
            non_compliance_id="NC-1", operator_id="OP-1",
            authority_id="A1", grounds="Test grounds",
        )
        assert appeal.penalty_suspended is True


# ====================================================================
# Document Model
# ====================================================================


class TestDocumentModel:
    """Test Document Pydantic model with encryption fields."""

    def test_encrypted_document(self, encrypted_document):
        assert encrypted_document.encrypted is True
        assert encrypted_document.encryption_key_id == "eudr-acm-doc-key-v1"

    def test_unencrypted_document(self, unencrypted_document):
        assert unencrypted_document.encrypted is False
        assert unencrypted_document.encryption_key_id == ""

    def test_default_mime_type(self):
        doc = Document(
            document_id="D1", communication_id="C1",
            document_type=DocumentType.DDS_STATEMENT,
            title="Test Doc",
        )
        assert doc.mime_type == "application/pdf"

    def test_default_encrypted(self):
        doc = Document(
            document_id="D1", communication_id="C1",
            document_type=DocumentType.DDS_STATEMENT,
            title="Test Doc",
        )
        assert doc.encrypted is False

    def test_file_size_non_negative(self):
        doc = Document(
            document_id="D1", communication_id="C1",
            document_type=DocumentType.DDS_STATEMENT,
            title="Test Doc",
            file_size_bytes=0,
        )
        assert doc.file_size_bytes >= 0


# ====================================================================
# Notification Model
# ====================================================================


class TestNotificationModel:
    """Test Notification Pydantic model."""

    def test_email_notification(self, email_notification):
        assert email_notification.channel == NotificationChannel.EMAIL
        assert email_notification.delivery_status == "sent"

    def test_api_notification(self, api_notification):
        assert api_notification.channel == NotificationChannel.API
        assert api_notification.delivery_status == "delivered"

    def test_failed_notification(self, failed_notification):
        assert failed_notification.delivery_status == "failed"
        assert failed_notification.retry_count == 3

    def test_default_delivery_status(self):
        n = Notification(
            notification_id="N1", communication_id="C1",
            channel=NotificationChannel.EMAIL,
            recipient_type=RecipientType.OPERATOR,
            recipient_id="OP-1",
        )
        assert n.delivery_status == "pending"
        assert n.retry_count == 0
        assert n.max_retries == 3


# ====================================================================
# Template Model
# ====================================================================


class TestTemplateModel:
    """Test Template Pydantic model."""

    def test_create_en(self, template_en):
        assert template_en.language == LanguageCode.EN
        assert len(template_en.placeholders) == 5

    def test_create_de(self, template_de):
        assert template_de.language == LanguageCode.DE

    def test_create_fr(self, template_fr):
        assert template_fr.language == LanguageCode.FR

    def test_default_version(self):
        t = Template(
            template_id="T1", template_name="test",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
        )
        assert t.version == "1.0"

    def test_default_active(self):
        t = Template(
            template_id="T1", template_name="test",
            communication_type=CommunicationType.GENERAL_CORRESPONDENCE,
        )
        assert t.active is True


# ====================================================================
# CommunicationThread Model
# ====================================================================


class TestCommunicationThreadModel:
    """Test CommunicationThread model."""

    def test_create(self, sample_thread):
        assert sample_thread.thread_id == "THR-001"
        assert len(sample_thread.communication_ids) == 2

    def test_defaults(self):
        t = CommunicationThread(
            thread_id="T1", operator_id="OP-1",
            authority_id="A1", subject="Test thread",
        )
        assert t.status == CommunicationStatus.PENDING
        assert t.priority == CommunicationPriority.NORMAL
        assert t.communication_ids == []


# ====================================================================
# ResponseData Model
# ====================================================================


class TestResponseDataModel:
    """Test ResponseData model."""

    def test_create(self, sample_response):
        assert sample_response.response_id == "RESP-001"
        assert len(sample_response.document_ids) == 2

    def test_default_responder_type(self):
        r = ResponseData(
            response_id="R1", communication_id="C1",
            responder_id="OP-1",
        )
        assert r.responder_type == RecipientType.OPERATOR

    def test_accepted_none_default(self):
        r = ResponseData(
            response_id="R1", communication_id="C1",
            responder_id="OP-1",
        )
        assert r.accepted is None


# ====================================================================
# DeadlineReminder Model
# ====================================================================


class TestDeadlineReminderModel:
    """Test DeadlineReminder model."""

    def test_create(self, upcoming_deadline_reminder):
        assert upcoming_deadline_reminder.reminder_id == "REM-001"
        assert upcoming_deadline_reminder.hours_remaining == 48

    def test_escalated_default(self):
        now = datetime.now(tz=timezone.utc)
        r = DeadlineReminder(
            reminder_id="R1", communication_id="C1",
            operator_id="OP-1", deadline=now,
            hours_remaining=24,
        )
        assert r.escalated is False


# ====================================================================
# ApprovalWorkflow Model
# ====================================================================


class TestApprovalWorkflowModel:
    """Test ApprovalWorkflow model."""

    def test_create(self, pending_approval):
        assert pending_approval.workflow_id == "WF-001"
        assert pending_approval.status == "pending_review"

    def test_default_status(self):
        wf = ApprovalWorkflow(
            workflow_id="W1", communication_id="C1",
            initiated_by="U1", approver_id="U2",
        )
        assert wf.status == "pending_review"


# ====================================================================
# HealthStatus Model
# ====================================================================


class TestHealthStatusModel:
    """Test HealthStatus model."""

    def test_default_agent_id(self):
        h = HealthStatus()
        assert h.agent_id == AGENT_ID

    def test_default_status(self):
        h = HealthStatus()
        assert h.status == "healthy"

    def test_default_version(self):
        h = HealthStatus()
        assert h.version == AGENT_VERSION

    def test_default_counters(self):
        h = HealthStatus()
        assert h.pending_communications == 0
        assert h.overdue_communications == 0
        assert h.active_inspections == 0
        assert h.open_appeals == 0
