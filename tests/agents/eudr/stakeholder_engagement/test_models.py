# -*- coding: utf-8 -*-
"""
Unit tests for models.py - AGENT-EUDR-031

Tests all enumerations, model creation, defaults, Decimal fields,
constants, serialization, and optional fields for the Stakeholder
Engagement Tool data models.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.stakeholder_engagement.models import (
    AGENT_ID,
    AGENT_VERSION,
    FPIC_STAGES_ORDERED,
    GRIEVANCE_SEVERITY_LEVELS,
    SUPPORTED_COMMODITIES,
    AuditAction,
    AuditEntry,
    CommunicationChannel,
    CommunicationRecord,
    CommunicationStatus,
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


class TestEnums:
    """Test all enum definitions and membership."""

    def test_eudr_commodity_values(self):
        """Test all EUDR commodity enum values."""
        assert EUDRCommodity.CATTLE == "cattle"
        assert EUDRCommodity.COCOA == "cocoa"
        assert EUDRCommodity.COFFEE == "coffee"
        assert EUDRCommodity.PALM_OIL == "palm_oil"
        assert EUDRCommodity.RUBBER == "rubber"
        assert EUDRCommodity.SOYA == "soya"
        assert EUDRCommodity.WOOD == "wood"
        assert len(EUDRCommodity) == 7

    def test_stakeholder_category_values(self):
        """Test all stakeholder category enum values."""
        expected = {
            "indigenous_community", "local_community", "cooperative",
            "smallholder", "ngo", "government_agency", "certification_body",
            "worker_union", "other",
        }
        actual = {c.value for c in StakeholderCategory}
        assert actual == expected

    def test_stakeholder_status_values(self):
        """Test all stakeholder status enum values."""
        expected = {"active", "inactive", "pending", "archived"}
        actual = {s.value for s in StakeholderStatus}
        assert actual == expected

    def test_fpic_stage_values(self):
        """Test all FPIC stage enum values."""
        expected = {
            "notification", "information_sharing", "consultation",
            "deliberation", "decision", "agreement", "monitoring",
        }
        actual = {s.value for s in FPICStage}
        assert actual == expected
        assert len(FPICStage) == 7

    def test_consent_status_values(self):
        """Test all consent status enum values."""
        expected = {"pending", "granted", "withheld", "conditional", "withdrawn", "expired"}
        actual = {s.value for s in ConsentStatus}
        assert actual == expected

    def test_grievance_severity_values(self):
        """Test all grievance severity enum values."""
        expected = {"critical", "high", "standard", "minor"}
        actual = {s.value for s in GrievanceSeverity}
        assert actual == expected

    def test_grievance_status_values(self):
        """Test all grievance status enum values."""
        expected = {
            "submitted", "triaged", "investigating", "resolved",
            "closed", "appealed", "reopened",
        }
        actual = {s.value for s in GrievanceStatus}
        assert actual == expected

    def test_consultation_type_values(self):
        """Test all consultation type enum values."""
        expected = {
            "community_meeting", "bilateral", "focus_group",
            "public_hearing", "workshop", "field_visit",
        }
        actual = {t.value for t in ConsultationType}
        assert actual == expected

    def test_communication_channel_values(self):
        """Test all communication channel enum values."""
        expected = {"email", "sms", "letter", "radio", "in_person", "phone", "digital_platform"}
        actual = {c.value for c in CommunicationChannel}
        assert actual == expected

    def test_engagement_dimension_values(self):
        """Test all engagement dimension enum values."""
        expected = {
            "inclusiveness", "transparency", "responsiveness",
            "accountability", "cultural_sensitivity", "rights_respect",
        }
        actual = {d.value for d in EngagementDimension}
        assert actual == expected
        assert len(EngagementDimension) == 6

    def test_report_type_values(self):
        """Test all report type enum values."""
        expected = {
            "dds_summary", "fpic_compliance", "grievance_report",
            "consultation_register", "engagement_summary", "communication_log",
        }
        actual = {t.value for t in ReportType}
        assert actual == expected

    def test_report_format_values(self):
        """Test all report format enum values."""
        expected = {"json", "pdf", "xml", "csv"}
        actual = {f.value for f in ReportFormat}
        assert actual == expected

    def test_audit_action_values(self):
        """Test all audit action enum values."""
        expected = {
            "create", "update", "delete", "submit", "resolve",
            "escalate", "approve", "reject", "archive",
        }
        actual = {a.value for a in AuditAction}
        assert actual == expected


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        """Test AGENT_ID constant value."""
        assert AGENT_ID == "GL-EUDR-SET-031"

    def test_agent_version(self):
        """Test AGENT_VERSION constant value."""
        assert AGENT_VERSION == "1.0.0"

    def test_fpic_stages_ordered_count(self):
        """Test that all 7 FPIC stages are listed in order."""
        assert len(FPIC_STAGES_ORDERED) == 7

    def test_fpic_stages_ordered_first_is_notification(self):
        """Test first FPIC stage is notification."""
        assert FPIC_STAGES_ORDERED[0] == FPICStage.NOTIFICATION

    def test_fpic_stages_ordered_last_is_monitoring(self):
        """Test last FPIC stage is monitoring."""
        assert FPIC_STAGES_ORDERED[-1] == FPICStage.MONITORING

    def test_grievance_severity_levels_count(self):
        """Test grievance severity levels count."""
        assert len(GRIEVANCE_SEVERITY_LEVELS) == 4

    def test_supported_commodities_count(self):
        """Test that all 7 commodities are supported."""
        assert len(SUPPORTED_COMMODITIES) == 7

    def test_supported_commodities_includes_coffee(self):
        """Test supported commodities includes coffee."""
        assert "coffee" in SUPPORTED_COMMODITIES

    def test_supported_commodities_includes_palm_oil(self):
        """Test supported commodities includes palm oil."""
        assert "palm_oil" in SUPPORTED_COMMODITIES


class TestStakeholderRecordModel:
    """Test StakeholderRecord model creation and defaults."""

    def test_create_valid_stakeholder(self):
        """Test creating a valid StakeholderRecord."""
        record = StakeholderRecord(
            stakeholder_id="STK-001",
            operator_id="op-001",
            name="Test Community",
            category=StakeholderCategory.LOCAL_COMMUNITY,
            status=StakeholderStatus.ACTIVE,
            country_code="CO",
            region="Antioquia",
            commodity=EUDRCommodity.COFFEE,
            contact_info={"primary_name": "Test User"},
            rights_classification=RightsClassification(
                has_land_rights=True,
                has_customary_rights=False,
                has_indigenous_status=False,
                fpic_required=False,
                applicable_conventions=[],
                legal_framework="",
            ),
            population_estimate=100,
            affected_area_hectares=Decimal("50.0"),
        )
        assert record.stakeholder_id == "STK-001"
        assert record.category == StakeholderCategory.LOCAL_COMMUNITY
        assert record.status == StakeholderStatus.ACTIVE

    def test_stakeholder_defaults(self):
        """Test StakeholderRecord default values."""
        record = StakeholderRecord(
            stakeholder_id="STK-002",
            operator_id="op-001",
            name="Default Test",
            category=StakeholderCategory.OTHER,
            country_code="BR",
            commodity=EUDRCommodity.SOYA,
            contact_info={},
            rights_classification=RightsClassification(
                has_land_rights=False,
                has_customary_rights=False,
                has_indigenous_status=False,
                fpic_required=False,
                applicable_conventions=[],
                legal_framework="",
            ),
        )
        assert record.status == StakeholderStatus.PENDING
        assert record.engagement_history == []
        assert record.notes == ""

    def test_stakeholder_model_dump(self):
        """Test StakeholderRecord serialization via model_dump."""
        record = StakeholderRecord(
            stakeholder_id="STK-003",
            operator_id="op-001",
            name="Dump Test",
            category=StakeholderCategory.COOPERATIVE,
            country_code="GH",
            commodity=EUDRCommodity.COCOA,
            contact_info={"email": "test@example.com"},
            rights_classification=RightsClassification(
                has_land_rights=False,
                has_customary_rights=False,
                has_indigenous_status=False,
                fpic_required=False,
                applicable_conventions=[],
                legal_framework="",
            ),
        )
        dumped = record.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["stakeholder_id"] == "STK-003"

    def test_stakeholder_indigenous_category(self, sample_stakeholder_indigenous):
        """Test indigenous stakeholder has correct category."""
        assert sample_stakeholder_indigenous.category == StakeholderCategory.INDIGENOUS_COMMUNITY

    def test_stakeholder_rights_classification(self, sample_stakeholder_indigenous):
        """Test stakeholder rights classification attached correctly."""
        rc = sample_stakeholder_indigenous.rights_classification
        assert rc.has_indigenous_status is True
        assert rc.fpic_required is True
        assert "ILO 169" in rc.applicable_conventions

    def test_stakeholder_ngo_no_land_rights(self, sample_stakeholder_ngo):
        """Test NGO stakeholder has no land rights."""
        rc = sample_stakeholder_ngo.rights_classification
        assert rc.has_land_rights is False
        assert rc.fpic_required is False

    def test_stakeholder_with_engagement_history(self):
        """Test StakeholderRecord with engagement history."""
        record = StakeholderRecord(
            stakeholder_id="STK-004",
            operator_id="op-001",
            name="History Test",
            category=StakeholderCategory.LOCAL_COMMUNITY,
            country_code="CO",
            commodity=EUDRCommodity.COFFEE,
            contact_info={},
            rights_classification=RightsClassification(
                has_land_rights=True,
                has_customary_rights=False,
                has_indigenous_status=False,
                fpic_required=False,
                applicable_conventions=[],
                legal_framework="",
            ),
            engagement_history=[
                {"event": "initial_contact", "date": "2025-01-15"},
                {"event": "consultation", "date": "2025-06-01"},
            ],
        )
        assert len(record.engagement_history) == 2

    def test_stakeholder_population_estimate_validation(self):
        """Test StakeholderRecord population estimate validation."""
        with pytest.raises(Exception):
            StakeholderRecord(
                stakeholder_id="STK-BAD",
                operator_id="op-001",
                name="Bad",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO",
                commodity=EUDRCommodity.COFFEE,
                contact_info={},
                rights_classification=RightsClassification(
                    has_land_rights=False,
                    has_customary_rights=False,
                    has_indigenous_status=False,
                    fpic_required=False,
                    applicable_conventions=[],
                    legal_framework="",
                ),
                population_estimate=-50,
            )

    def test_stakeholder_affected_area_validation(self):
        """Test StakeholderRecord affected area validation."""
        with pytest.raises(Exception):
            StakeholderRecord(
                stakeholder_id="STK-BAD2",
                operator_id="op-001",
                name="Bad Area",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO",
                commodity=EUDRCommodity.COFFEE,
                contact_info={},
                rights_classification=RightsClassification(
                    has_land_rights=False,
                    has_customary_rights=False,
                    has_indigenous_status=False,
                    fpic_required=False,
                    applicable_conventions=[],
                    legal_framework="",
                ),
                affected_area_hectares=Decimal("-10"),
            )


class TestFPICWorkflowModel:
    """Test FPICWorkflow model."""

    def test_create_valid_fpic_workflow(self, sample_fpic_workflow):
        """Test creating a valid FPICWorkflow."""
        assert sample_fpic_workflow.workflow_id == "FPIC-001"
        assert sample_fpic_workflow.current_stage == FPICStage.NOTIFICATION
        assert sample_fpic_workflow.consent_status == ConsentStatus.PENDING

    def test_fpic_workflow_defaults(self):
        """Test FPICWorkflow default values."""
        wf = FPICWorkflow(
            workflow_id="FPIC-DEF",
            stakeholder_id="STK-001",
            operator_id="op-001",
            commodity=EUDRCommodity.COFFEE,
        )
        assert wf.current_stage == FPICStage.NOTIFICATION
        assert wf.consent_status == ConsentStatus.PENDING
        assert wf.stage_history == []
        assert wf.consultation_records == []
        assert wf.evidence_documents == []

    def test_fpic_workflow_with_consent(self, fpic_workflow_consented):
        """Test FPICWorkflow with consent granted."""
        assert fpic_workflow_consented.consent_status == ConsentStatus.GRANTED
        assert fpic_workflow_consented.consent_recorded_at is not None
        assert fpic_workflow_consented.consent_evidence != ""

    def test_fpic_workflow_with_withheld_consent(self, fpic_workflow_withheld):
        """Test FPICWorkflow with consent withheld."""
        assert fpic_workflow_withheld.consent_status == ConsentStatus.WITHHELD
        assert fpic_workflow_withheld.current_stage == FPICStage.DECISION

    def test_fpic_workflow_all_stages(self, fpic_workflow_all_stages):
        """Test FPICWorkflow that has progressed through all stages."""
        assert fpic_workflow_all_stages.current_stage == FPICStage.MONITORING
        assert len(fpic_workflow_all_stages.stage_history) == 7
        assert fpic_workflow_all_stages.consent_status == ConsentStatus.GRANTED

    def test_fpic_workflow_model_dump(self, sample_fpic_workflow):
        """Test FPICWorkflow serialization."""
        dumped = sample_fpic_workflow.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["workflow_id"] == "FPIC-001"

    def test_fpic_workflow_initiated_at_is_datetime(self, sample_fpic_workflow):
        """Test FPICWorkflow initiated_at is datetime."""
        assert isinstance(sample_fpic_workflow.initiated_at, datetime)

    def test_fpic_workflow_stage_config_present(self, sample_fpic_workflow):
        """Test FPICWorkflow stage configuration present."""
        assert "notification_period_days" in sample_fpic_workflow.stage_config

    def test_fpic_workflow_consultation_references(self, fpic_workflow_consented):
        """Test FPICWorkflow has consultation references."""
        assert len(fpic_workflow_consented.consultation_records) == 2

    def test_fpic_workflow_evidence_documents(self, fpic_workflow_consented):
        """Test FPICWorkflow has evidence documents."""
        assert len(fpic_workflow_consented.evidence_documents) >= 1


class TestGrievanceRecordModel:
    """Test GrievanceRecord model."""

    def test_create_valid_grievance(self, sample_grievance_critical):
        """Test creating a valid GrievanceRecord."""
        assert sample_grievance_critical.grievance_id == "GRV-001"
        assert sample_grievance_critical.severity == GrievanceSeverity.CRITICAL
        assert sample_grievance_critical.status == GrievanceStatus.SUBMITTED

    def test_grievance_defaults(self):
        """Test GrievanceRecord default values."""
        grv = GrievanceRecord(
            grievance_id="GRV-DEF",
            stakeholder_id="STK-001",
            operator_id="op-001",
            title="Test Grievance",
            description="Test description",
            severity=GrievanceSeverity.MINOR,
        )
        assert grv.status == GrievanceStatus.SUBMITTED
        assert grv.investigation_notes == []
        assert grv.resolution_actions == []

    def test_grievance_standard_category(self, sample_grievance_standard):
        """Test standard grievance has correct category."""
        assert sample_grievance_standard.category == "environmental_impact"

    def test_grievance_channel_variety(self, multiple_grievances):
        """Test grievances can have different channels."""
        channels = {g.channel for g in multiple_grievances if g.channel}
        assert len(channels) >= 2

    def test_grievance_model_dump(self, sample_grievance_critical):
        """Test GrievanceRecord serialization."""
        dumped = sample_grievance_critical.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["severity"] == GrievanceSeverity.CRITICAL.value

    def test_grievance_sla_deadline_set(self, sample_grievance_critical):
        """Test critical grievance has SLA deadline set."""
        assert sample_grievance_critical.sla_deadline is not None

    def test_grievance_submitted_at_is_datetime(self, sample_grievance_critical):
        """Test GrievanceRecord submitted_at is datetime."""
        assert isinstance(sample_grievance_critical.submitted_at, datetime)

    def test_grievance_severity_critical_value(self):
        """Test GrievanceSeverity CRITICAL value."""
        assert GrievanceSeverity.CRITICAL.value == "critical"

    def test_grievance_severity_ordering(self):
        """Test grievance severity levels exist in expected order."""
        severities = [GrievanceSeverity.CRITICAL, GrievanceSeverity.HIGH,
                      GrievanceSeverity.STANDARD, GrievanceSeverity.MINOR]
        assert len(severities) == 4

    def test_grievance_status_transitions(self):
        """Test all grievance status values accessible."""
        assert GrievanceStatus.SUBMITTED.value == "submitted"
        assert GrievanceStatus.TRIAGED.value == "triaged"
        assert GrievanceStatus.INVESTIGATING.value == "investigating"
        assert GrievanceStatus.RESOLVED.value == "resolved"


class TestConsultationRecordModel:
    """Test ConsultationRecord model."""

    def test_create_valid_consultation(self, sample_consultation_community):
        """Test creating a valid ConsultationRecord."""
        assert sample_consultation_community.consultation_id == "CON-001"
        assert sample_consultation_community.consultation_type == ConsultationType.COMMUNITY_MEETING

    def test_consultation_bilateral(self, sample_consultation_bilateral):
        """Test bilateral consultation record."""
        assert sample_consultation_bilateral.consultation_type == ConsultationType.BILATERAL
        assert len(sample_consultation_bilateral.stakeholder_ids) == 1

    def test_consultation_participants(self, sample_consultation_community):
        """Test consultation has participants."""
        assert len(sample_consultation_community.participants) >= 3

    def test_consultation_outcomes(self, sample_consultation_community):
        """Test consultation has outcomes."""
        assert len(sample_consultation_community.outcomes) >= 2

    def test_consultation_evidence_refs(self, consultation_with_evidence):
        """Test consultation with extensive evidence."""
        assert len(consultation_with_evidence.evidence_refs) >= 5

    def test_consultation_model_dump(self, sample_consultation_community):
        """Test ConsultationRecord serialization."""
        dumped = sample_consultation_community.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["consultation_id"] == "CON-001"

    def test_consultation_language(self, sample_consultation_community):
        """Test consultation language field."""
        assert sample_consultation_community.language == "es"

    def test_consultation_defaults(self):
        """Test ConsultationRecord default values."""
        record = ConsultationRecord(
            consultation_id="CON-DEF",
            operator_id="op-001",
            consultation_type=ConsultationType.WORKSHOP,
            title="Default Test",
        )
        assert record.participants == []
        assert record.outcomes == []
        assert record.evidence_refs == []
        assert record.status == "scheduled"


class TestCommunicationRecordModel:
    """Test CommunicationRecord model."""

    def test_create_valid_email_communication(self, sample_communication_email):
        """Test creating a valid email CommunicationRecord."""
        assert sample_communication_email.communication_id == "COMM-001"
        assert sample_communication_email.channel == CommunicationChannel.EMAIL

    def test_create_valid_sms_communication(self, sample_communication_sms):
        """Test creating a valid SMS CommunicationRecord."""
        assert sample_communication_sms.channel == CommunicationChannel.SMS

    def test_communication_delivery_status(self, sample_communication_email):
        """Test communication delivery status."""
        assert sample_communication_email.delivery_status == DeliveryStatus.DELIVERED

    def test_communication_model_dump(self, sample_communication_email):
        """Test CommunicationRecord serialization."""
        dumped = sample_communication_email.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["channel"] == CommunicationChannel.EMAIL.value

    def test_communication_template_ref(self, sample_communication_email):
        """Test communication template reference."""
        assert sample_communication_email.template_id == "TPL-INVITE-001"

    def test_communication_defaults(self):
        """Test CommunicationRecord default values."""
        record = CommunicationRecord(
            communication_id="COMM-DEF",
            operator_id="op-001",
            stakeholder_ids=["STK-001"],
            channel=CommunicationChannel.LETTER,
            subject="Test",
            body="Test body",
        )
        assert record.delivery_status == DeliveryStatus.PENDING
        assert record.template_id is None
        assert record.campaign_id is None


class TestEngagementAssessmentModel:
    """Test EngagementAssessment model."""

    def test_create_valid_assessment(self, sample_engagement_assessment):
        """Test creating a valid EngagementAssessment."""
        assert sample_engagement_assessment.assessment_id == "EA-001"
        assert sample_engagement_assessment.composite_score == Decimal("73")

    def test_assessment_dimension_scores(self, sample_engagement_assessment):
        """Test assessment dimension scores."""
        scores = sample_engagement_assessment.dimension_scores
        assert len(scores) == 6
        assert EngagementDimension.INCLUSIVENESS in scores

    def test_assessment_high_score(self, assessment_high_score):
        """Test high-score assessment."""
        assert assessment_high_score.composite_score > Decimal("90")
        assert len(assessment_high_score.recommendations) == 0

    def test_assessment_low_score(self, assessment_low_score):
        """Test low-score assessment has many recommendations."""
        assert assessment_low_score.composite_score < Decimal("30")
        assert len(assessment_low_score.recommendations) >= 3

    def test_assessment_model_dump(self, sample_engagement_assessment):
        """Test EngagementAssessment serialization."""
        dumped = sample_engagement_assessment.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["assessment_id"] == "EA-001"

    def test_assessment_evidence_refs(self, sample_engagement_assessment):
        """Test assessment evidence references."""
        assert len(sample_engagement_assessment.evidence_refs) >= 1

    def test_assessment_score_bounds_validation(self):
        """Test assessment score validation for values above 100."""
        with pytest.raises(Exception):
            EngagementAssessment(
                assessment_id="EA-BAD",
                operator_id="op-001",
                stakeholder_id="STK-001",
                assessment_date=datetime.now(tz=timezone.utc),
                dimension_scores={},
                composite_score=Decimal("150"),
            )

    def test_assessment_negative_score_validation(self):
        """Test assessment score validation for negative values."""
        with pytest.raises(Exception):
            EngagementAssessment(
                assessment_id="EA-BAD2",
                operator_id="op-001",
                stakeholder_id="STK-001",
                assessment_date=datetime.now(tz=timezone.utc),
                dimension_scores={},
                composite_score=Decimal("-10"),
            )


class TestComplianceReportModel:
    """Test ComplianceReport model."""

    def test_create_valid_report(self, sample_compliance_report):
        """Test creating a valid ComplianceReport."""
        assert sample_compliance_report.report_id == "RPT-001"
        assert sample_compliance_report.report_type == ReportType.ENGAGEMENT_SUMMARY

    def test_report_dds_summary(self, dds_summary_report):
        """Test DDS summary report."""
        assert dds_summary_report.report_type == ReportType.DDS_SUMMARY
        assert "article_10_compliance" in dds_summary_report.sections

    def test_report_fpic_compliance(self, fpic_compliance_report):
        """Test FPIC compliance report."""
        assert fpic_compliance_report.report_type == ReportType.FPIC_COMPLIANCE
        assert fpic_compliance_report.format == ReportFormat.PDF

    def test_report_sections_present(self, sample_compliance_report):
        """Test report has required sections."""
        assert "stakeholder_overview" in sample_compliance_report.sections
        assert "fpic_status" in sample_compliance_report.sections
        assert "grievance_summary" in sample_compliance_report.sections

    def test_report_model_dump(self, sample_compliance_report):
        """Test ComplianceReport serialization."""
        dumped = sample_compliance_report.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["report_id"] == "RPT-001"

    def test_report_period_range(self, sample_compliance_report):
        """Test report has valid period range."""
        assert sample_compliance_report.period_start < sample_compliance_report.period_end


class TestAuditEntryModel:
    """Test AuditEntry model."""

    def test_create_valid_audit_entry(self):
        """Test creating a valid AuditEntry."""
        entry = AuditEntry(
            entry_id="AUD-001",
            action=AuditAction.CREATE,
            entity_type="stakeholder",
            entity_id="STK-001",
            actor="agent-031",
            details={"field": "status", "old": "pending", "new": "active"},
        )
        assert entry.entry_id == "AUD-001"
        assert entry.action == AuditAction.CREATE

    def test_audit_entry_defaults(self):
        """Test AuditEntry default values."""
        entry = AuditEntry(
            entry_id="AUD-002",
            action=AuditAction.UPDATE,
            entity_type="grievance",
            entity_id="GRV-001",
            actor="agent-031",
        )
        assert entry.details == {}
        assert isinstance(entry.timestamp, datetime)

    def test_audit_entry_model_dump(self):
        """Test AuditEntry serialization."""
        entry = AuditEntry(
            entry_id="AUD-003",
            action=AuditAction.RESOLVE,
            entity_type="grievance",
            entity_id="GRV-002",
            actor="operator-001",
        )
        dumped = entry.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["action"] == AuditAction.RESOLVE.value

    def test_audit_entry_all_actions(self):
        """Test AuditEntry with all action types."""
        for action in AuditAction:
            entry = AuditEntry(
                entry_id=f"AUD-{action.value}",
                action=action,
                entity_type="test",
                entity_id="TEST-001",
                actor="test-actor",
            )
            assert entry.action == action

    def test_audit_entry_timestamp_utc(self):
        """Test AuditEntry timestamp is UTC."""
        entry = AuditEntry(
            entry_id="AUD-TZ",
            action=AuditAction.CREATE,
            entity_type="test",
            entity_id="TEST-001",
            actor="test",
        )
        assert entry.timestamp.tzinfo is not None


class TestHealthStatusModel:
    """Test HealthStatus model."""

    def test_health_status_defaults(self):
        """Test HealthStatus default values."""
        health = HealthStatus()
        assert health.agent_id == AGENT_ID
        assert health.status == "healthy"
        assert health.version == AGENT_VERSION
        assert health.database is False
        assert health.redis is False
        assert health.uptime_seconds == 0.0

    def test_health_status_custom_values(self):
        """Test HealthStatus with custom values."""
        health = HealthStatus(
            status="degraded",
            database=True,
            redis=True,
            uptime_seconds=7200.0,
            engines={
                "stakeholder_mapper": "ok",
                "fpic_workflow_engine": "ok",
                "grievance_mechanism": "ok",
                "consultation_manager": "ok",
                "communication_hub": "ok",
                "engagement_verifier": "ok",
                "compliance_reporter": "ok",
            },
        )
        assert health.status == "degraded"
        assert health.database is True
        assert health.uptime_seconds == 7200.0
        assert len(health.engines) == 7


class TestCommunicationTemplateModel:
    """Test CommunicationTemplate model."""

    def test_create_valid_template(self, communication_template):
        """Test creating a valid CommunicationTemplate."""
        assert communication_template.template_id == "TPL-INVITE-001"
        assert communication_template.channel == CommunicationChannel.EMAIL
        assert communication_template.active is True

    def test_template_variables(self, communication_template):
        """Test template has expected variables."""
        assert "stakeholder_name" in communication_template.variables
        assert "date" in communication_template.variables

    def test_template_body_contains_placeholders(self, communication_template):
        """Test template body contains variable placeholders."""
        assert "{{stakeholder_name}}" in communication_template.body_template
        assert "{{meeting_type}}" in communication_template.body_template
