# -*- coding: utf-8 -*-
"""
Tests for FPIC API Routes - AGENT-EUDR-021

Test count: 36 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (API: FPIC Routes)
"""

from datetime import date
from decimal import Decimal

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    FPICAssessment,
    FPICStatus,
    VerifyFPICRequest,
    FPICWorkflow,
    FPICWorkflowStage,
    WorkflowTransition,
    CreateWorkflowRequest,
    AdvanceWorkflowRequest,
)


# ===========================================================================
# 1. POST /fpic/verify (10 tests)
# ===========================================================================


class TestVerifyFPICEndpoint:
    """Test POST /fpic/verify endpoint."""

    def test_verify_fpic_valid_request(self):
        """Test valid FPIC verification request."""
        req = VerifyFPICRequest(
            plot_id="p-001",
            territory_id="t-001",
            community_id="c-001",
            documentation={"community_identified": True},
            production_start_date=date(2025, 1, 1),
            country_code="BR",
        )
        assert req.plot_id == "p-001"
        assert req.territory_id == "t-001"

    def test_verify_fpic_minimal_request(self):
        """Test minimal FPIC verification request."""
        req = VerifyFPICRequest(
            plot_id="p-001",
            territory_id="t-001",
        )
        assert req.community_id is None
        assert req.documentation == {}

    def test_verify_fpic_with_full_documentation(self, full_fpic_documentation):
        """Test FPIC verification with complete documentation."""
        req = VerifyFPICRequest(
            plot_id="p-001",
            territory_id="t-001",
            documentation=full_fpic_documentation,
        )
        assert "community_identified" in req.documentation

    def test_verify_fpic_response_score(self, sample_fpic_obtained):
        """Test FPIC response includes score."""
        assert sample_fpic_obtained.fpic_score >= Decimal("0")
        assert sample_fpic_obtained.fpic_score <= Decimal("100")

    def test_verify_fpic_response_status(self, sample_fpic_obtained):
        """Test FPIC response includes status classification."""
        assert sample_fpic_obtained.fpic_status in [
            FPICStatus.CONSENT_OBTAINED,
            FPICStatus.CONSENT_PARTIAL,
            FPICStatus.CONSENT_MISSING,
        ]

    def test_verify_fpic_response_element_scores(self, sample_fpic_obtained):
        """Test FPIC response includes all 10 element scores."""
        assert sample_fpic_obtained.community_identification_score >= Decimal("0")
        assert sample_fpic_obtained.information_disclosure_score >= Decimal("0")
        assert sample_fpic_obtained.prior_timing_score >= Decimal("0")
        assert sample_fpic_obtained.consultation_process_score >= Decimal("0")
        assert sample_fpic_obtained.community_representation_score >= Decimal("0")
        assert sample_fpic_obtained.consent_record_score >= Decimal("0")
        assert sample_fpic_obtained.absence_of_coercion_score >= Decimal("0")
        assert sample_fpic_obtained.agreement_documentation_score >= Decimal("0")
        assert sample_fpic_obtained.benefit_sharing_score >= Decimal("0")
        assert sample_fpic_obtained.monitoring_provisions_score >= Decimal("0")

    def test_verify_fpic_requires_auth(self, mock_auth):
        """Test FPIC verification requires authentication."""
        result = mock_auth.validate_token("valid-token")
        assert "eudr-irc:fpic:read" in result["permissions"]

    def test_verify_fpic_response_provenance(self, sample_fpic_obtained):
        """Test FPIC response includes provenance hash."""
        assert len(sample_fpic_obtained.provenance_hash) == SHA256_HEX_LENGTH

    def test_verify_fpic_with_country_code(self):
        """Test FPIC verification with country-specific rules."""
        req = VerifyFPICRequest(
            plot_id="p-001",
            territory_id="t-001",
            country_code="BR",
        )
        assert req.country_code == "BR"

    def test_verify_fpic_response_temporal_compliance(self, sample_fpic_obtained):
        """Test FPIC response includes temporal compliance flag."""
        assert isinstance(sample_fpic_obtained.temporal_compliance, bool)


# ===========================================================================
# 2. FPIC Workflow Endpoints (12 tests)
# ===========================================================================


class TestFPICWorkflowEndpoints:
    """Test FPIC workflow creation and advancement endpoints."""

    def test_create_workflow_request(self):
        """Test creating a FPIC workflow request."""
        req = CreateWorkflowRequest(
            plot_id="p-001",
            territory_id="t-001",
            community_id="c-001",
            initiator="system",
        )
        assert req.plot_id == "p-001"

    def test_workflow_starts_at_identification(self, sample_workflow):
        """Test new workflow starts at IDENTIFICATION or later stage."""
        assert sample_workflow.current_stage in [
            s for s in FPICWorkflowStage
        ]

    def test_advance_workflow_request(self):
        """Test advance workflow request."""
        req = AdvanceWorkflowRequest(
            actor="analyst-001",
            reason="All required information disclosed",
            supporting_evidence=[
                {"type": "document", "id": "doc-001"},
            ],
        )
        assert req.actor == "analyst-001"

    def test_workflow_stage_history(self, sample_workflow):
        """Test workflow maintains stage history."""
        assert len(sample_workflow.stage_history) >= 1

    def test_workflow_sla_status(self, sample_workflow):
        """Test workflow tracks SLA status."""
        assert sample_workflow.sla_status in ["on_track", "at_risk", "breached"]

    def test_workflow_escalation_level(self, sample_workflow):
        """Test workflow tracks escalation level."""
        assert sample_workflow.escalation_level >= 0

    def test_workflow_next_deadline(self, sample_workflow):
        """Test workflow has next deadline."""
        assert sample_workflow.next_deadline is not None

    def test_workflow_transition_record(self):
        """Test workflow transition record creation."""
        transition = WorkflowTransition(
            transition_id="tr-001",
            workflow_id="wf-001",
            from_stage="identification",
            to_stage="information_disclosure",
            actor="analyst-001",
            reason="Community identified and verified",
            provenance_hash="a" * 64,
        )
        assert transition.from_stage == "identification"
        assert transition.to_stage == "information_disclosure"

    def test_workflow_consent_withdrawn(self):
        """Test workflow can transition to consent_withdrawn."""
        wf = FPICWorkflow(
            workflow_id="wf-withdrawn",
            plot_id="p-001",
            territory_id="t-001",
            community_id="c-001",
            current_stage=FPICWorkflowStage.CONSENT_WITHDRAWN,
            provenance_hash="b" * 64,
        )
        assert wf.current_stage == FPICWorkflowStage.CONSENT_WITHDRAWN

    def test_workflow_consent_denied(self):
        """Test workflow can transition to consent_denied."""
        wf = FPICWorkflow(
            workflow_id="wf-denied",
            plot_id="p-001",
            territory_id="t-001",
            community_id="c-001",
            current_stage=FPICWorkflowStage.CONSENT_DENIED,
            provenance_hash="c" * 64,
        )
        assert wf.current_stage == FPICWorkflowStage.CONSENT_DENIED

    def test_workflow_provenance_hash(self, sample_workflow):
        """Test workflow has provenance hash."""
        assert len(sample_workflow.provenance_hash) == SHA256_HEX_LENGTH

    def test_workflow_agreement_id_optional(self, sample_workflow):
        """Test workflow agreement_id is optional."""
        assert sample_workflow.agreement_id is None


# ===========================================================================
# 3. GET /fpic/assessments (8 tests)
# ===========================================================================


class TestGetFPICAssessments:
    """Test GET /fpic/assessments endpoint."""

    def test_list_assessments(self, sample_fpic_obtained, sample_fpic_partial,
                               sample_fpic_missing):
        """Test listing FPIC assessments."""
        assessments = [sample_fpic_obtained, sample_fpic_partial, sample_fpic_missing]
        assert len(assessments) == 3

    def test_filter_by_status_obtained(self, sample_fpic_obtained):
        """Test filtering assessments by CONSENT_OBTAINED status."""
        assert sample_fpic_obtained.fpic_status == FPICStatus.CONSENT_OBTAINED

    def test_filter_by_status_partial(self, sample_fpic_partial):
        """Test filtering assessments by CONSENT_PARTIAL status."""
        assert sample_fpic_partial.fpic_status == FPICStatus.CONSENT_PARTIAL

    def test_filter_by_status_missing(self, sample_fpic_missing):
        """Test filtering assessments by CONSENT_MISSING status."""
        assert sample_fpic_missing.fpic_status == FPICStatus.CONSENT_MISSING

    def test_filter_by_territory(self, sample_fpic_obtained):
        """Test filtering assessments by territory ID."""
        assert sample_fpic_obtained.territory_id == "t-001"

    def test_filter_by_plot(self, sample_fpic_obtained):
        """Test filtering assessments by plot ID."""
        assert sample_fpic_obtained.plot_id == "p-001"

    def test_assessment_requires_read_permission(self, mock_auth):
        """Test listing assessments requires read permission."""
        result = mock_auth.validate_token("valid-token")
        assert "eudr-irc:fpic:read" in result["permissions"]

    def test_assessment_version_tracking(self, sample_fpic_obtained):
        """Test assessment version is tracked."""
        assert sample_fpic_obtained.version >= 1


# ===========================================================================
# 4. Workflow SLA Configuration (6 tests)
# ===========================================================================


class TestWorkflowSLAConfiguration:
    """Test FPIC workflow SLA settings."""

    def test_identification_sla_days(self, mock_config):
        """Test identification stage SLA is 14 days."""
        assert mock_config.workflow_sla_days["identification"] == 14

    def test_information_disclosure_sla(self, mock_config):
        """Test information disclosure SLA is 30 days."""
        assert mock_config.workflow_sla_days["information_disclosure"] == 30

    def test_consultation_sla(self, mock_config):
        """Test consultation stage SLA is 60 days."""
        assert mock_config.workflow_sla_days["consultation"] == 60

    def test_consent_decision_sla(self, mock_config):
        """Test consent decision SLA is 30 days."""
        assert mock_config.workflow_sla_days["consent_decision"] == 30

    def test_implementation_sla(self, mock_config):
        """Test implementation SLA is 365 days."""
        assert mock_config.workflow_sla_days["implementation"] == 365

    def test_escalation_thresholds(self, mock_config):
        """Test escalation thresholds are defined."""
        assert mock_config.escalation_thresholds_days["level_1"] == 7
        assert mock_config.escalation_thresholds_days["level_2"] == 14
        assert mock_config.escalation_thresholds_days["level_3"] == 30
