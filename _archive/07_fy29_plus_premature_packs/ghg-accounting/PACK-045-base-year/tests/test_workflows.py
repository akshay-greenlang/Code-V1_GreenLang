# -*- coding: utf-8 -*-
"""
Tests for all 8 PACK-045 workflows.

Tests workflow instantiation, status tracking, and attribute checks.
All workflows take __init__(self, config=None) and have execute(input_data).
Target: ~60 tests.
"""

import pytest
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from workflows.base_year_establishment_workflow import (
    BaseYearEstablishmentWorkflow, WorkflowStatus,
)
from workflows.recalculation_assessment_workflow import (
    RecalculationAssessmentWorkflow,
)
from workflows.recalculation_execution_workflow import (
    RecalculationExecutionWorkflow,
)
from workflows.merger_acquisition_workflow import (
    MergerAcquisitionWorkflow,
)
from workflows.annual_review_workflow import (
    AnnualReviewWorkflow,
)
from workflows.target_rebasing_workflow import (
    TargetRebasingWorkflow,
)
from workflows.audit_verification_workflow import (
    AuditVerificationWorkflow,
)
from workflows.full_base_year_pipeline_workflow import (
    FullBaseYearPipelineWorkflow,
)


# ============================================================================
# BaseYearEstablishmentWorkflow
# ============================================================================

class TestBaseYearEstablishmentWorkflow:
    def test_create_workflow(self):
        wf = BaseYearEstablishmentWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = BaseYearEstablishmentWorkflow(config={"key": "value"})
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = BaseYearEstablishmentWorkflow()
        assert hasattr(wf, "execute")

    def test_workflow_status_enum_pending(self):
        assert WorkflowStatus.PENDING is not None

    def test_workflow_status_enum_completed(self):
        assert WorkflowStatus.COMPLETED is not None

    def test_workflow_status_enum_failed(self):
        assert WorkflowStatus.FAILED is not None

    def test_workflow_status_enum_running(self):
        assert WorkflowStatus.RUNNING is not None

    def test_workflow_status_enum_partial(self):
        assert WorkflowStatus.PARTIAL is not None

    def test_workflow_status_values(self):
        assert len(WorkflowStatus) == 5


# ============================================================================
# RecalculationAssessmentWorkflow
# ============================================================================

class TestRecalculationAssessmentWorkflow:
    def test_create_workflow(self):
        wf = RecalculationAssessmentWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = RecalculationAssessmentWorkflow(config={"key": "value"})
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = RecalculationAssessmentWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# RecalculationExecutionWorkflow
# ============================================================================

class TestRecalculationExecutionWorkflow:
    def test_create_workflow(self):
        wf = RecalculationExecutionWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = RecalculationExecutionWorkflow(config=None)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = RecalculationExecutionWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# MergerAcquisitionWorkflow
# ============================================================================

class TestMergerAcquisitionWorkflow:
    def test_create_workflow(self):
        wf = MergerAcquisitionWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = MergerAcquisitionWorkflow(config={"key": "value"})
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = MergerAcquisitionWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# AnnualReviewWorkflow
# ============================================================================

class TestAnnualReviewWorkflow:
    def test_create_workflow(self):
        wf = AnnualReviewWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = AnnualReviewWorkflow(config=None)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = AnnualReviewWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# TargetRebasingWorkflow
# ============================================================================

class TestTargetRebasingWorkflow:
    def test_create_workflow(self):
        wf = TargetRebasingWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = TargetRebasingWorkflow(config={"key": "value"})
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = TargetRebasingWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# AuditVerificationWorkflow
# ============================================================================

class TestAuditVerificationWorkflow:
    def test_create_workflow(self):
        wf = AuditVerificationWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = AuditVerificationWorkflow(config=None)
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = AuditVerificationWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# FullBaseYearPipelineWorkflow
# ============================================================================

class TestFullBaseYearPipelineWorkflow:
    def test_create_workflow(self):
        wf = FullBaseYearPipelineWorkflow()
        assert wf is not None

    def test_create_workflow_with_config(self):
        wf = FullBaseYearPipelineWorkflow(config={"key": "value"})
        assert wf is not None

    def test_workflow_has_execute(self):
        wf = FullBaseYearPipelineWorkflow()
        assert hasattr(wf, "execute")


# ============================================================================
# Cross-workflow tests
# ============================================================================

class TestAllWorkflowsCreation:
    """Ensure all 8 workflows can be instantiated without arguments."""

    def test_all_workflows_instantiate(self):
        workflows = [
            BaseYearEstablishmentWorkflow(),
            RecalculationAssessmentWorkflow(),
            RecalculationExecutionWorkflow(),
            MergerAcquisitionWorkflow(),
            AnnualReviewWorkflow(),
            TargetRebasingWorkflow(),
            AuditVerificationWorkflow(),
            FullBaseYearPipelineWorkflow(),
        ]
        assert len(workflows) == 8
        for wf in workflows:
            assert wf is not None

    def test_all_workflows_have_execute(self):
        workflows = [
            BaseYearEstablishmentWorkflow(),
            RecalculationAssessmentWorkflow(),
            RecalculationExecutionWorkflow(),
            MergerAcquisitionWorkflow(),
            AnnualReviewWorkflow(),
            TargetRebasingWorkflow(),
            AuditVerificationWorkflow(),
            FullBaseYearPipelineWorkflow(),
        ]
        for wf in workflows:
            assert hasattr(wf, "execute"), f"{type(wf).__name__} missing execute"


class TestWorkflowStatusEnum:
    def test_status_values(self):
        assert WorkflowStatus.PENDING is not None

    def test_status_is_string_enum(self):
        for status in WorkflowStatus:
            assert isinstance(status.value, str)

    def test_status_count(self):
        assert len(WorkflowStatus) == 5

    def test_status_pending_value(self):
        assert WorkflowStatus.PENDING.value == "pending"

    def test_status_completed_value(self):
        assert WorkflowStatus.COMPLETED.value == "completed"

    def test_status_failed_value(self):
        assert WorkflowStatus.FAILED.value == "failed"
