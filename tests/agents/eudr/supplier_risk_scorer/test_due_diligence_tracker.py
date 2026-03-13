# -*- coding: utf-8 -*-
"""
Unit tests for DueDiligenceTracker - AGENT-EUDR-017 Engine 2

Tests comprehensive due diligence activity tracking per EUDR Articles 8-11
covering audit scheduling, activity logging, non-conformance recording,
corrective action management, completion rate calculation, gap identification,
and escalation workflow.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.supplier_risk_scorer.due_diligence_tracker import (
    DueDiligenceTracker,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    DDLevel,
    DDStatus,
    NonConformanceType,
)


# ============================================================================
# TestDueDiligenceTrackerInit
# ============================================================================


class TestDueDiligenceTrackerInit:
    """Tests for DueDiligenceTracker initialization."""

    @pytest.mark.unit
    def test_initialization_creates_empty_stores(self, mock_config):
        tracker = DueDiligenceTracker()
        assert tracker._dd_records == {}
        assert tracker._activities == {}

    @pytest.mark.unit
    def test_initialization_creates_lock(self, mock_config):
        tracker = DueDiligenceTracker()
        assert tracker._lock is not None


# ============================================================================
# TestRecordActivity
# ============================================================================


class TestRecordActivity:
    """Tests for record_activity method."""

    @pytest.mark.unit
    def test_record_activity_creates_record(
        self, due_diligence_tracker, sample_dd_record
    ):
        result = due_diligence_tracker.record_activity(
            supplier_id=sample_dd_record["supplier_id"],
            activity_type="audit",
            description="Annual audit conducted",
            auditor="auditor@example.com",
        )
        assert result is not None
        assert "activity_id" in result
        assert result["activity_type"] == "audit"

    @pytest.mark.unit
    def test_record_activity_all_types(self, due_diligence_tracker):
        activity_types = [
            "audit", "site_visit", "document_review",
            "questionnaire", "screening", "verification",
            "training", "interview"
        ]
        for activity_type in activity_types:
            result = due_diligence_tracker.record_activity(
                supplier_id="SUPP-001",
                activity_type=activity_type,
                description=f"{activity_type} completed",
                auditor="auditor@example.com",
            )
            assert result["activity_type"] == activity_type

    @pytest.mark.unit
    def test_record_activity_stores_timestamp(self, due_diligence_tracker):
        result = due_diligence_tracker.record_activity(
            supplier_id="SUPP-001",
            activity_type="audit",
            description="Test audit",
            auditor="auditor@example.com",
        )
        assert "timestamp" in result
        assert isinstance(result["timestamp"], datetime)


# ============================================================================
# TestGetDDHistory
# ============================================================================


class TestGetDDHistory:
    """Tests for get_dd_history method."""

    @pytest.mark.unit
    def test_get_dd_history_returns_all_activities(self, due_diligence_tracker):
        supplier_id = "SUPP-HIST"
        # Record multiple activities
        for i in range(5):
            due_diligence_tracker.record_activity(
                supplier_id=supplier_id,
                activity_type="audit",
                description=f"Audit {i}",
                auditor="auditor@example.com",
            )
        history = due_diligence_tracker.get_dd_history(supplier_id)
        assert len(history) == 5

    @pytest.mark.unit
    def test_get_dd_history_empty_for_new_supplier(self, due_diligence_tracker):
        history = due_diligence_tracker.get_dd_history("SUPP-NEW")
        assert history == []


# ============================================================================
# TestTrackNonConformance
# ============================================================================


class TestTrackNonConformance:
    """Tests for track_non_conformance method."""

    @pytest.mark.unit
    def test_track_minor_non_conformance(self, due_diligence_tracker):
        result = due_diligence_tracker.track_non_conformance(
            supplier_id="SUPP-001",
            nc_type=NonConformanceType.MINOR,
            description="Documentation formatting issue",
            detected_by="auditor@example.com",
        )
        assert result["nc_type"] == NonConformanceType.MINOR
        assert result["severity"] == "minor"

    @pytest.mark.unit
    def test_track_major_non_conformance(self, due_diligence_tracker):
        result = due_diligence_tracker.track_non_conformance(
            supplier_id="SUPP-001",
            nc_type=NonConformanceType.MAJOR,
            description="Missing geolocation data",
            detected_by="auditor@example.com",
        )
        assert result["nc_type"] == NonConformanceType.MAJOR
        assert result["severity"] == "major"

    @pytest.mark.unit
    def test_track_critical_non_conformance(self, due_diligence_tracker):
        result = due_diligence_tracker.track_non_conformance(
            supplier_id="SUPP-001",
            nc_type=NonConformanceType.CRITICAL,
            description="Deforestation detected in supply chain",
            detected_by="auditor@example.com",
        )
        assert result["nc_type"] == NonConformanceType.CRITICAL
        assert result["severity"] == "critical"

    @pytest.mark.unit
    def test_track_non_conformance_assigns_deadline(
        self, due_diligence_tracker, mock_config
    ):
        result = due_diligence_tracker.track_non_conformance(
            supplier_id="SUPP-001",
            nc_type=NonConformanceType.MAJOR,
            description="Test NC",
            detected_by="auditor@example.com",
        )
        assert "corrective_action_deadline" in result
        # Should be 90 days from now (config)
        expected_deadline = datetime.now(timezone.utc) + timedelta(
            days=mock_config.corrective_action_deadline_days
        )
        assert result["corrective_action_deadline"].date() == expected_deadline.date()


# ============================================================================
# TestCorrectiveAction
# ============================================================================


class TestCorrectiveAction:
    """Tests for corrective action management."""

    @pytest.mark.unit
    def test_create_corrective_action_plan(self, due_diligence_tracker):
        # First create NC
        nc = due_diligence_tracker.track_non_conformance(
            supplier_id="SUPP-001",
            nc_type=NonConformanceType.MAJOR,
            description="Test NC",
            detected_by="auditor@example.com",
        )
        # Create corrective action
        result = due_diligence_tracker.create_corrective_action_plan(
            nc_id=nc["nc_id"],
            plan_description="Implement new documentation process",
            responsible_party="supplier_contact@example.com",
            target_date=datetime.now(timezone.utc) + timedelta(days=60),
        )
        assert result is not None
        assert "action_id" in result
        assert result["status"] == "planned"

    @pytest.mark.unit
    def test_complete_corrective_action(self, due_diligence_tracker):
        nc = due_diligence_tracker.track_non_conformance(
            supplier_id="SUPP-001",
            nc_type=NonConformanceType.MINOR,
            description="Test NC",
            detected_by="auditor@example.com",
        )
        action = due_diligence_tracker.create_corrective_action_plan(
            nc_id=nc["nc_id"],
            plan_description="Fix issue",
            responsible_party="supplier@example.com",
            target_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        # Complete action
        result = due_diligence_tracker.complete_corrective_action(
            action_id=action["action_id"],
            completion_notes="Issue resolved",
        )
        assert result["status"] == "completed"
        assert "completion_date" in result


# ============================================================================
# TestCompletionRate
# ============================================================================


class TestCompletionRate:
    """Tests for DD completion rate calculation."""

    @pytest.mark.unit
    def test_calculate_completion_rate_simplified(
        self, due_diligence_tracker
    ):
        supplier_id = "SUPP-SIMPLE"
        # Record required activities for SIMPLIFIED level
        due_diligence_tracker.record_activity(
            supplier_id=supplier_id,
            activity_type="document_review",
            description="Documents reviewed",
            auditor="auditor@example.com",
        )
        due_diligence_tracker.record_activity(
            supplier_id=supplier_id,
            activity_type="screening",
            description="Screening completed",
            auditor="auditor@example.com",
        )
        rate = due_diligence_tracker.calculate_completion_rate(
            supplier_id=supplier_id,
            dd_level=DDLevel.SIMPLIFIED,
        )
        assert rate == Decimal("100.0")  # All required activities done

    @pytest.mark.unit
    def test_calculate_completion_rate_enhanced(
        self, due_diligence_tracker
    ):
        supplier_id = "SUPP-ENHANCED"
        # Record some but not all activities for ENHANCED level
        due_diligence_tracker.record_activity(
            supplier_id=supplier_id,
            activity_type="audit",
            description="Audit done",
            auditor="auditor@example.com",
        )
        rate = due_diligence_tracker.calculate_completion_rate(
            supplier_id=supplier_id,
            dd_level=DDLevel.ENHANCED,
        )
        # Should be < 100% as not all required activities completed
        assert Decimal("0.0") < rate < Decimal("100.0")


# ============================================================================
# TestIdentifyGaps
# ============================================================================


class TestIdentifyGaps:
    """Tests for gap identification."""

    @pytest.mark.unit
    def test_identify_gaps_returns_missing_activities(
        self, due_diligence_tracker
    ):
        supplier_id = "SUPP-GAPS"
        # Record only partial activities
        due_diligence_tracker.record_activity(
            supplier_id=supplier_id,
            activity_type="screening",
            description="Screening done",
            auditor="auditor@example.com",
        )
        gaps = due_diligence_tracker.identify_gaps(
            supplier_id=supplier_id,
            dd_level=DDLevel.STANDARD,
        )
        assert len(gaps) > 0
        # Should include document_review and questionnaire
        assert "document_review" in gaps or "questionnaire" in gaps


# ============================================================================
# TestEscalation
# ============================================================================


class TestEscalation:
    """Tests for escalation workflow."""

    @pytest.mark.unit
    def test_check_escalation_critical_nc(self, due_diligence_tracker):
        supplier_id = "SUPP-ESC"
        # Create critical NC
        due_diligence_tracker.track_non_conformance(
            supplier_id=supplier_id,
            nc_type=NonConformanceType.CRITICAL,
            description="Critical issue",
            detected_by="auditor@example.com",
        )
        escalation = due_diligence_tracker.check_escalation(supplier_id)
        assert escalation["requires_escalation"] is True
        assert "critical" in escalation["reason"].lower()

    @pytest.mark.unit
    def test_check_escalation_multiple_major_nc(self, due_diligence_tracker):
        supplier_id = "SUPP-ESC2"
        # Create multiple major NCs
        for i in range(3):
            due_diligence_tracker.track_non_conformance(
                supplier_id=supplier_id,
                nc_type=NonConformanceType.MAJOR,
                description=f"Major issue {i}",
                detected_by="auditor@example.com",
            )
        escalation = due_diligence_tracker.check_escalation(supplier_id)
        assert escalation["requires_escalation"] is True


# ============================================================================
# TestReadinessScore
# ============================================================================


class TestReadinessScore:
    """Tests for readiness scoring."""

    @pytest.mark.unit
    def test_calculate_readiness_score(self, due_diligence_tracker):
        supplier_id = "SUPP-READY"
        # Complete most DD activities
        for activity in ["audit", "site_visit", "document_review", "screening"]:
            due_diligence_tracker.record_activity(
                supplier_id=supplier_id,
                activity_type=activity,
                description=f"{activity} completed",
                auditor="auditor@example.com",
            )
        score = due_diligence_tracker.calculate_readiness_score(
            supplier_id=supplier_id,
            dd_level=DDLevel.ENHANCED,
        )
        assert Decimal("0.0") <= score <= Decimal("100.0")


# ============================================================================
# TestCostSummary
# ============================================================================


class TestCostSummary:
    """Tests for cost tracking."""

    @pytest.mark.unit
    def test_generate_cost_summary(self, due_diligence_tracker, mock_config):
        supplier_id = "SUPP-COST"
        summary = due_diligence_tracker.generate_cost_summary(
            supplier_id=supplier_id,
            dd_level=DDLevel.STANDARD,
        )
        assert "estimated_cost_eur" in summary
        # Standard DD should cost between min and max
        assert (
            mock_config.standard_cost_min_eur <=
            summary["estimated_cost_eur"] <=
            mock_config.standard_cost_max_eur
        )


# ============================================================================
# TestScheduleReassessment
# ============================================================================


class TestScheduleReassessment:
    """Tests for reassessment scheduling."""

    @pytest.mark.unit
    def test_schedule_reassessment(
        self, due_diligence_tracker, mock_config
    ):
        supplier_id = "SUPP-REASS"
        result = due_diligence_tracker.schedule_reassessment(
            supplier_id=supplier_id,
            dd_level=DDLevel.STANDARD,
        )
        assert "next_assessment_date" in result
        # Should be audit_interval_months from now
        expected_date = datetime.now(timezone.utc) + timedelta(
            days=mock_config.audit_interval_months * 30
        )
        assert result["next_assessment_date"].month == expected_date.month


# ============================================================================
# TestProvenance
# ============================================================================


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_activity_includes_provenance_hash(self, due_diligence_tracker):
        result = due_diligence_tracker.record_activity(
            supplier_id="SUPP-PROV",
            activity_type="audit",
            description="Test audit",
            auditor="auditor@example.com",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_supplier_id_raises_error(self, due_diligence_tracker):
        with pytest.raises(ValueError):
            due_diligence_tracker.record_activity(
                supplier_id="",
                activity_type="audit",
                description="Test",
                auditor="auditor@example.com",
            )

    @pytest.mark.unit
    def test_invalid_activity_type_raises_error(self, due_diligence_tracker):
        with pytest.raises(ValueError):
            due_diligence_tracker.record_activity(
                supplier_id="SUPP-001",
                activity_type="invalid_type",
                description="Test",
                auditor="auditor@example.com",
            )
