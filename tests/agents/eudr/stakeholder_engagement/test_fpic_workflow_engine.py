# -*- coding: utf-8 -*-
"""
Unit tests for FPICWorkflowEngine - AGENT-EUDR-031

Tests FPIC workflow initiation, stage advancement, deliberation
period management, consultation recording, consent recording,
compliance monitoring, and SLA compliance.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.fpic_workflow_engine import (
    FPICWorkflowEngine,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    ConsentStatus,
    EUDRCommodity,
    FPICStage,
    FPICWorkflow,
    StakeholderCategory,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def engine(config):
    return FPICWorkflowEngine(config=config)


# ---------------------------------------------------------------------------
# Test: InitiateFPIC
# ---------------------------------------------------------------------------

class TestInitiateFPIC:
    """Test FPIC workflow initiation."""

    @pytest.mark.asyncio
    async def test_initiate_fpic_success(self, engine):
        """Test successful FPIC workflow initiation."""
        wf = await engine.initiate_fpic(
            stakeholder_id="STK-IND-001",
            operator_id="OP-001",
            commodity=EUDRCommodity.COFFEE,
        )
        assert wf.workflow_id.startswith("FPIC-")
        assert wf.current_stage == FPICStage.NOTIFICATION
        assert wf.consent_status == ConsentStatus.PENDING

    @pytest.mark.asyncio
    async def test_initiate_fpic_sets_initiated_at(self, engine):
        """Test initiation sets initiated_at timestamp."""
        wf = await engine.initiate_fpic(
            stakeholder_id="STK-IND-001",
            operator_id="OP-001",
            commodity=EUDRCommodity.COFFEE,
        )
        assert isinstance(wf.initiated_at, datetime)
        assert wf.initiated_at <= datetime.now(tz=timezone.utc)

    @pytest.mark.asyncio
    async def test_initiate_fpic_missing_stakeholder_raises(self, engine):
        """Test initiation with missing stakeholder_id raises error."""
        with pytest.raises(ValueError, match="stakeholder_id is required"):
            await engine.initiate_fpic(
                stakeholder_id="",
                operator_id="OP-001",
                commodity=EUDRCommodity.COFFEE,
            )

    @pytest.mark.asyncio
    async def test_initiate_fpic_missing_operator_raises(self, engine):
        """Test initiation with missing operator_id raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await engine.initiate_fpic(
                stakeholder_id="STK-IND-001",
                operator_id="",
                commodity=EUDRCommodity.COFFEE,
            )

    @pytest.mark.asyncio
    async def test_initiate_fpic_generates_unique_id(self, engine):
        """Test each initiation generates unique workflow ID."""
        wf1 = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        wf2 = await engine.initiate_fpic("STK-IND-002", "OP-001", EUDRCommodity.COFFEE)
        assert wf1.workflow_id != wf2.workflow_id

    @pytest.mark.asyncio
    async def test_initiate_fpic_creates_stage_history(self, engine):
        """Test initiation creates initial stage history entry."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        assert len(wf.stage_history) >= 1
        assert wf.stage_history[0]["stage"] == FPICStage.NOTIFICATION.value

    @pytest.mark.asyncio
    async def test_initiate_fpic_all_commodities(self, engine):
        """Test initiation with all commodity types."""
        for commodity in EUDRCommodity:
            wf = await engine.initiate_fpic("STK-IND-001", "OP-001", commodity)
            assert wf.commodity == commodity

    @pytest.mark.asyncio
    async def test_initiate_fpic_with_custom_config(self, engine):
        """Test initiation applies stage configuration."""
        wf = await engine.initiate_fpic(
            stakeholder_id="STK-IND-001",
            operator_id="OP-001",
            commodity=EUDRCommodity.PALM_OIL,
            stage_config={"notification_period_days": 45},
        )
        assert wf.stage_config.get("notification_period_days") == 45


# ---------------------------------------------------------------------------
# Test: AdvanceStage
# ---------------------------------------------------------------------------

class TestAdvanceStage:
    """Test FPIC stage advancement through all 7 stages."""

    @pytest.mark.asyncio
    async def test_advance_notification_to_information_sharing(self, engine):
        """Test advancing from NOTIFICATION to INFORMATION_SHARING."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        advanced = await engine.advance_stage(wf.workflow_id)
        assert advanced.current_stage == FPICStage.INFORMATION_SHARING

    @pytest.mark.asyncio
    async def test_advance_information_sharing_to_consultation(self, engine):
        """Test advancing from INFORMATION_SHARING to CONSULTATION."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        wf = await engine.advance_stage(wf.workflow_id)  # -> INFO_SHARING
        wf = await engine.advance_stage(wf.workflow_id)  # -> CONSULTATION
        assert wf.current_stage == FPICStage.CONSULTATION

    @pytest.mark.asyncio
    async def test_advance_consultation_to_deliberation(self, engine):
        """Test advancing from CONSULTATION to DELIBERATION."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(3):  # Advance through first 3 stages
            wf = await engine.advance_stage(wf.workflow_id)
        assert wf.current_stage == FPICStage.DELIBERATION

    @pytest.mark.asyncio
    async def test_advance_deliberation_to_decision(self, engine):
        """Test advancing from DELIBERATION to DECISION."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        assert wf.current_stage == FPICStage.DECISION

    @pytest.mark.asyncio
    async def test_advance_decision_to_agreement(self, engine):
        """Test advancing from DECISION to AGREEMENT."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(5):
            wf = await engine.advance_stage(wf.workflow_id)
        assert wf.current_stage == FPICStage.AGREEMENT

    @pytest.mark.asyncio
    async def test_advance_agreement_to_monitoring(self, engine):
        """Test advancing from AGREEMENT to MONITORING."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(6):
            wf = await engine.advance_stage(wf.workflow_id)
        assert wf.current_stage == FPICStage.MONITORING

    @pytest.mark.asyncio
    async def test_advance_beyond_monitoring_raises(self, engine):
        """Test advancing beyond MONITORING raises error."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(6):
            wf = await engine.advance_stage(wf.workflow_id)
        assert wf.current_stage == FPICStage.MONITORING
        with pytest.raises(ValueError, match="already at final stage"):
            await engine.advance_stage(wf.workflow_id)

    @pytest.mark.asyncio
    async def test_advance_updates_stage_history(self, engine):
        """Test advancement updates stage history."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        initial_len = len(wf.stage_history)
        wf = await engine.advance_stage(wf.workflow_id)
        assert len(wf.stage_history) > initial_len

    @pytest.mark.asyncio
    async def test_advance_nonexistent_workflow_raises(self, engine):
        """Test advancing nonexistent workflow raises error."""
        with pytest.raises(ValueError, match="workflow not found"):
            await engine.advance_stage("FPIC-NONEXISTENT")

    @pytest.mark.asyncio
    async def test_advance_with_notes(self, engine):
        """Test advancement with additional notes."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        wf = await engine.advance_stage(wf.workflow_id, notes="Notification period completed.")
        last_entry = wf.stage_history[-1]
        assert "notes" in last_entry or last_entry.get("notes") is not None

    @pytest.mark.asyncio
    async def test_full_workflow_progression(self, engine):
        """Test complete workflow progression through all 7 stages."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        stages = [FPICStage.INFORMATION_SHARING, FPICStage.CONSULTATION,
                  FPICStage.DELIBERATION, FPICStage.DECISION,
                  FPICStage.AGREEMENT, FPICStage.MONITORING]
        for expected_stage in stages:
            wf = await engine.advance_stage(wf.workflow_id)
            assert wf.current_stage == expected_stage


# ---------------------------------------------------------------------------
# Test: DeliberationPeriod
# ---------------------------------------------------------------------------

class TestDeliberationPeriod:
    """Test deliberation period management."""

    @pytest.mark.asyncio
    async def test_deliberation_period_default_days(self, engine):
        """Test default deliberation period is 90 days."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        period = engine.get_deliberation_period(wf.workflow_id)
        assert period == 90

    @pytest.mark.asyncio
    async def test_deliberation_period_custom_days(self, engine):
        """Test custom deliberation period from stage config."""
        wf = await engine.initiate_fpic(
            "STK-IND-001", "OP-001", EUDRCommodity.COFFEE,
            stage_config={"deliberation_period_days": 120},
        )
        period = engine.get_deliberation_period(wf.workflow_id)
        assert period == 120

    @pytest.mark.asyncio
    async def test_deliberation_period_check_not_expired(self, engine):
        """Test deliberation period not yet expired."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        # Advance to DELIBERATION stage
        for _ in range(3):
            wf = await engine.advance_stage(wf.workflow_id)
        is_expired = engine.is_deliberation_expired(wf.workflow_id)
        assert is_expired is False

    @pytest.mark.asyncio
    async def test_deliberation_period_remaining_days(self, engine):
        """Test calculating remaining deliberation days."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(3):
            wf = await engine.advance_stage(wf.workflow_id)
        remaining = engine.get_remaining_deliberation_days(wf.workflow_id)
        assert remaining >= 0

    @pytest.mark.asyncio
    async def test_deliberation_not_applicable_before_stage(self, engine):
        """Test deliberation period not applicable before reaching deliberation stage."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        with pytest.raises(ValueError, match="not in deliberation stage"):
            engine.is_deliberation_expired(wf.workflow_id)

    @pytest.mark.asyncio
    async def test_deliberation_nonexistent_workflow(self, engine):
        """Test deliberation check for nonexistent workflow."""
        with pytest.raises(ValueError):
            engine.get_deliberation_period("FPIC-NONEXISTENT")


# ---------------------------------------------------------------------------
# Test: RecordConsultation
# ---------------------------------------------------------------------------

class TestRecordConsultation:
    """Test recording consultations within FPIC workflow."""

    @pytest.mark.asyncio
    async def test_record_consultation_success(self, engine):
        """Test successful consultation recording."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        # Advance to consultation stage
        for _ in range(2):
            wf = await engine.advance_stage(wf.workflow_id)
        updated = await engine.record_consultation(
            workflow_id=wf.workflow_id,
            consultation_id="CON-001",
        )
        assert "CON-001" in updated.consultation_records

    @pytest.mark.asyncio
    async def test_record_multiple_consultations(self, engine):
        """Test recording multiple consultations."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(2):
            wf = await engine.advance_stage(wf.workflow_id)
        for i in range(3):
            wf = await engine.record_consultation(wf.workflow_id, f"CON-{i:03d}")
        assert len(wf.consultation_records) == 3

    @pytest.mark.asyncio
    async def test_record_consultation_updates_evidence(self, engine):
        """Test consultation recording can attach evidence."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(2):
            wf = await engine.advance_stage(wf.workflow_id)
        updated = await engine.record_consultation(
            workflow_id=wf.workflow_id,
            consultation_id="CON-001",
            evidence_refs=["PHOTO-001", "MINUTES-001"],
        )
        assert len(updated.evidence_documents) >= 2

    @pytest.mark.asyncio
    async def test_record_consultation_nonexistent_workflow(self, engine):
        """Test recording consultation for nonexistent workflow raises error."""
        with pytest.raises(ValueError, match="workflow not found"):
            await engine.record_consultation("FPIC-NONEXISTENT", "CON-001")

    @pytest.mark.asyncio
    async def test_record_consultation_empty_id_raises(self, engine):
        """Test recording consultation with empty ID raises error."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        with pytest.raises(ValueError, match="consultation_id is required"):
            await engine.record_consultation(wf.workflow_id, "")

    @pytest.mark.asyncio
    async def test_record_consultation_duplicate_allowed(self, engine):
        """Test that duplicate consultation IDs are handled gracefully."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(2):
            wf = await engine.advance_stage(wf.workflow_id)
        await engine.record_consultation(wf.workflow_id, "CON-DUP")
        # Recording same consultation should not raise
        updated = await engine.record_consultation(wf.workflow_id, "CON-DUP")
        assert "CON-DUP" in updated.consultation_records


# ---------------------------------------------------------------------------
# Test: RecordConsent
# ---------------------------------------------------------------------------

class TestRecordConsent:
    """Test consent recording for all consent statuses."""

    @pytest.mark.asyncio
    async def test_record_consent_granted(self, engine):
        """Test recording granted consent."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):  # Advance to DECISION
            wf = await engine.advance_stage(wf.workflow_id)
        updated = await engine.record_consent(
            workflow_id=wf.workflow_id,
            consent_status=ConsentStatus.GRANTED,
            evidence="signed-consent-doc.pdf",
        )
        assert updated.consent_status == ConsentStatus.GRANTED
        assert updated.consent_recorded_at is not None

    @pytest.mark.asyncio
    async def test_record_consent_withheld(self, engine):
        """Test recording withheld consent."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        updated = await engine.record_consent(
            workflow_id=wf.workflow_id,
            consent_status=ConsentStatus.WITHHELD,
            evidence="withhold-declaration.pdf",
        )
        assert updated.consent_status == ConsentStatus.WITHHELD

    @pytest.mark.asyncio
    async def test_record_consent_conditional(self, engine):
        """Test recording conditional consent."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        updated = await engine.record_consent(
            workflow_id=wf.workflow_id,
            consent_status=ConsentStatus.CONDITIONAL,
            evidence="conditional-consent-terms.pdf",
            conditions=["Buffer zone of 1km", "Monthly water quality reports"],
        )
        assert updated.consent_status == ConsentStatus.CONDITIONAL

    @pytest.mark.asyncio
    async def test_record_consent_withdrawn(self, engine):
        """Test recording withdrawn consent."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        await engine.record_consent(wf.workflow_id, ConsentStatus.GRANTED, "consent.pdf")
        updated = await engine.record_consent(
            workflow_id=wf.workflow_id,
            consent_status=ConsentStatus.WITHDRAWN,
            evidence="withdrawal-notice.pdf",
        )
        assert updated.consent_status == ConsentStatus.WITHDRAWN

    @pytest.mark.asyncio
    async def test_record_consent_nonexistent_workflow_raises(self, engine):
        """Test recording consent for nonexistent workflow raises error."""
        with pytest.raises(ValueError, match="workflow not found"):
            await engine.record_consent("FPIC-NONEXISTENT", ConsentStatus.GRANTED, "doc.pdf")

    @pytest.mark.asyncio
    async def test_record_consent_sets_timestamp(self, engine):
        """Test consent recording sets timestamp."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        updated = await engine.record_consent(wf.workflow_id, ConsentStatus.GRANTED, "doc.pdf")
        assert updated.consent_recorded_at is not None
        assert isinstance(updated.consent_recorded_at, datetime)

    @pytest.mark.asyncio
    async def test_record_consent_missing_evidence_raises(self, engine):
        """Test recording consent without evidence raises error."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        with pytest.raises(ValueError, match="evidence is required"):
            await engine.record_consent(wf.workflow_id, ConsentStatus.GRANTED, "")

    @pytest.mark.asyncio
    async def test_record_consent_all_statuses(self, engine):
        """Test recording consent with all valid statuses."""
        for status in [ConsentStatus.GRANTED, ConsentStatus.WITHHELD, ConsentStatus.CONDITIONAL]:
            wf = await engine.initiate_fpic(f"STK-{status.value}", "OP-001", EUDRCommodity.COFFEE)
            for _ in range(4):
                wf = await engine.advance_stage(wf.workflow_id)
            updated = await engine.record_consent(wf.workflow_id, status, f"{status.value}.pdf")
            assert updated.consent_status == status


# ---------------------------------------------------------------------------
# Test: MonitorCompliance
# ---------------------------------------------------------------------------

class TestMonitorCompliance:
    """Test FPIC compliance monitoring."""

    @pytest.mark.asyncio
    async def test_monitor_compliance_returns_status(self, engine):
        """Test compliance monitoring returns status dict."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        status = await engine.monitor_compliance(wf.workflow_id)
        assert isinstance(status, dict)
        assert "compliant" in status

    @pytest.mark.asyncio
    async def test_monitor_compliance_active_workflow(self, engine):
        """Test compliance monitoring for active workflow."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(2):
            wf = await engine.advance_stage(wf.workflow_id)
        status = await engine.monitor_compliance(wf.workflow_id)
        assert "current_stage" in status

    @pytest.mark.asyncio
    async def test_monitor_compliance_nonexistent_raises(self, engine):
        """Test compliance monitoring for nonexistent workflow."""
        with pytest.raises(ValueError, match="workflow not found"):
            await engine.monitor_compliance("FPIC-NONEXISTENT")

    @pytest.mark.asyncio
    async def test_monitor_compliance_consented_workflow(self, engine):
        """Test compliance monitoring for consented workflow."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        for _ in range(4):
            wf = await engine.advance_stage(wf.workflow_id)
        await engine.record_consent(wf.workflow_id, ConsentStatus.GRANTED, "consent.pdf")
        status = await engine.monitor_compliance(wf.workflow_id)
        assert status.get("consent_status") == "granted"

    @pytest.mark.asyncio
    async def test_monitor_compliance_includes_timeline(self, engine):
        """Test compliance monitoring includes timeline information."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        status = await engine.monitor_compliance(wf.workflow_id)
        assert "initiated_at" in status or "elapsed_days" in status


# ---------------------------------------------------------------------------
# Test: SLACompliance
# ---------------------------------------------------------------------------

class TestSLACompliance:
    """Test FPIC SLA compliance checking."""

    @pytest.mark.asyncio
    async def test_sla_compliance_new_workflow(self, engine):
        """Test SLA compliance for newly initiated workflow."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        sla = engine.check_sla_compliance(wf.workflow_id)
        assert isinstance(sla, dict)
        assert sla.get("sla_breached") is False

    @pytest.mark.asyncio
    async def test_sla_compliance_includes_stage_duration(self, engine):
        """Test SLA check includes current stage duration."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        sla = engine.check_sla_compliance(wf.workflow_id)
        assert "current_stage_duration_days" in sla or "stage_elapsed" in sla

    @pytest.mark.asyncio
    async def test_sla_compliance_nonexistent_raises(self, engine):
        """Test SLA check for nonexistent workflow."""
        with pytest.raises(ValueError):
            engine.check_sla_compliance("FPIC-NONEXISTENT")

    @pytest.mark.asyncio
    async def test_sla_compliance_overall_duration(self, engine):
        """Test SLA check includes overall workflow duration."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        sla = engine.check_sla_compliance(wf.workflow_id)
        assert "total_duration_days" in sla or "workflow_age_days" in sla

    @pytest.mark.asyncio
    async def test_sla_compliance_returns_dict(self, engine):
        """Test SLA check returns a dictionary."""
        wf = await engine.initiate_fpic("STK-IND-001", "OP-001", EUDRCommodity.COFFEE)
        result = engine.check_sla_compliance(wf.workflow_id)
        assert isinstance(result, dict)
