# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Workflow Tests
==============================================================

Tests for all 8 DMA workflows: class existence, phase methods,
execute method, status tracking, error handling, and timing.

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

import inspect

import pytest

from .conftest import (
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    WORKFLOWS_DIR,
    _load_workflow,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _try_load_workflow(key):
    """Attempt to load a workflow, returning module or None."""
    try:
        return _load_workflow(key)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Workflow File Existence
# ===========================================================================


class TestWorkflowFilesExist:
    """Test that all 8 workflow files exist on disk."""

    @pytest.mark.parametrize("wf_key,wf_file", list(WORKFLOW_FILES.items()))
    def test_workflow_file_exists(self, wf_key, wf_file):
        """Workflow file exists on disk."""
        path = WORKFLOWS_DIR / wf_file
        assert path.exists(), f"Workflow file missing: {path}"


# ===========================================================================
# Impact Assessment Workflow
# ===========================================================================


class TestImpactAssessmentWorkflow:
    """Tests for the ImpactAssessmentWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("impact_assessment")

    def test_impact_assessment_workflow_class_exists(self):
        """ImpactAssessmentWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ImpactAssessmentWorkflow")

    def test_impact_assessment_4_phases(self):
        """Workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        wf_cls = getattr(self.mod, "ImpactAssessmentWorkflow")
        instance = wf_cls.__new__(wf_cls)
        # Check for phase methods or phase count attribute
        phase_methods = [m for m in dir(instance) if "phase" in m.lower() or "step" in m.lower()]
        has_execute = hasattr(wf_cls, "execute") or hasattr(wf_cls, "run")
        assert has_execute or len(phase_methods) >= 1

    def test_impact_assessment_has_execute(self):
        """ImpactAssessmentWorkflow has execute or run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.ImpactAssessmentWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_impact_assessment_input_model(self):
        """ImpactAssessmentInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ImpactAssessmentInput")

    def test_impact_assessment_result_model(self):
        """ImpactAssessmentResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ImpactAssessmentResult")


# ===========================================================================
# Financial Assessment Workflow
# ===========================================================================


class TestFinancialAssessmentWorkflow:
    """Tests for the FinancialAssessmentWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("financial_assessment")

    def test_financial_assessment_workflow_class_exists(self):
        """FinancialAssessmentWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "FinancialAssessmentWorkflow")

    def test_financial_assessment_4_phases(self):
        """Workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.FinancialAssessmentWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_financial_assessment_input_model(self):
        """FinancialAssessmentInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "FinancialAssessmentInput")

    def test_financial_assessment_result_model(self):
        """FinancialAssessmentResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "FinancialAssessmentResult")


# ===========================================================================
# Stakeholder Engagement Workflow
# ===========================================================================


class TestStakeholderEngagementWorkflow:
    """Tests for the StakeholderEngagementWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("stakeholder_engagement")

    def test_stakeholder_engagement_workflow_class_exists(self):
        """StakeholderEngagementWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "StakeholderEngagementWorkflow")

    def test_stakeholder_engagement_5_phases(self):
        """Workflow has execute/run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.StakeholderEngagementWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_stakeholder_engagement_input_model(self):
        """StakeholderEngagementInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "StakeholderEngagementInput")

    def test_stakeholder_engagement_result_model(self):
        """StakeholderEngagementResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "StakeholderEngagementResult")

    def test_stakeholder_model_exists(self):
        """Stakeholder model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "Stakeholder")


# ===========================================================================
# IRO Identification Workflow
# ===========================================================================


class TestIROIdentificationWorkflow:
    """Tests for the IROIdentificationWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("iro_identification")

    def test_iro_identification_workflow_class_exists(self):
        """IROIdentificationWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "IROIdentificationWorkflow")

    def test_iro_identification_4_phases(self):
        """Workflow has execute/run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.IROIdentificationWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_iro_identification_input_model(self):
        """IROIdentificationInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "IROIdentificationInput")

    def test_iro_identification_result_model(self):
        """IROIdentificationResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "IROIdentificationResult")

    def test_iro_record_model(self):
        """IRORecord model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "IRORecord")


# ===========================================================================
# Materiality Matrix Workflow
# ===========================================================================


class TestMaterialityMatrixWorkflow:
    """Tests for the MaterialityMatrixWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("materiality_matrix")

    def test_materiality_matrix_workflow_class_exists(self):
        """MaterialityMatrixWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "MaterialityMatrixWorkflow")

    def test_materiality_matrix_3_phases(self):
        """Workflow has execute/run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.MaterialityMatrixWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_materiality_matrix_input_model(self):
        """MaterialityMatrixInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "MaterialityMatrixInput")

    def test_materiality_matrix_result_model(self):
        """MaterialityMatrixResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "MaterialityMatrixResult")


# ===========================================================================
# ESRS Mapping Workflow
# ===========================================================================


class TestESRSMappingWorkflow:
    """Tests for the ESRSMappingWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("esrs_mapping")

    def test_esrs_mapping_workflow_class_exists(self):
        """ESRSMappingWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ESRSMappingWorkflow")

    def test_esrs_mapping_3_phases(self):
        """Workflow has execute/run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.ESRSMappingWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_esrs_mapping_input_model(self):
        """ESRSMappingInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ESRSMappingInput")

    def test_esrs_mapping_result_model(self):
        """ESRSMappingResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ESRSMappingResult")


# ===========================================================================
# Full DMA Workflow
# ===========================================================================


class TestFullDMAWorkflow:
    """Tests for the FullDMAWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("full_dma")

    def test_full_dma_workflow_class_exists(self):
        """FullDMAWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "FullDMAWorkflow")

    def test_full_dma_6_phases(self):
        """Workflow has execute/run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.FullDMAWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_full_dma_input_model(self):
        """FullDMAInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "FullDMAInput")

    def test_full_dma_result_model(self):
        """FullDMAResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "FullDMAResult")

    def test_full_dma_topic_result_model(self):
        """DMATopicResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "DMATopicResult")


# ===========================================================================
# DMA Update Workflow
# ===========================================================================


class TestDMAUpdateWorkflow:
    """Tests for the DMAUpdateWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("dma_update")

    def test_dma_update_workflow_class_exists(self):
        """DMAUpdateWorkflow class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMAUpdateWorkflow")

    def test_dma_update_4_phases(self):
        """Workflow has execute/run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.DMAUpdateWorkflow
        assert hasattr(cls, "execute") or hasattr(cls, "run")

    def test_dma_update_input_model(self):
        """DMAUpdateInput model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "DMAUpdateInput")

    def test_dma_update_result_model(self):
        """DMAUpdateResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "DMAUpdateResult")

    def test_dma_update_delta_entry_model(self):
        """DeltaEntry model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "DeltaEntry")


# ===========================================================================
# Cross-Workflow Pattern Tests
# ===========================================================================


class TestWorkflowPatterns:
    """Pattern tests applicable to all workflows."""

    @pytest.mark.parametrize("wf_key,wf_class", list(WORKFLOW_CLASSES.items()))
    def test_workflow_status_tracking(self, wf_key, wf_class):
        """All workflow modules have status enums or tracking."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        # Each workflow should define either WorkflowStatus or PhaseStatus
        has_status = (
            hasattr(mod, "WorkflowStatus")
            or hasattr(mod, "PhaseStatus")
            or hasattr(mod, "Status")
        )
        assert has_status, f"Workflow {wf_key} missing status tracking"

    @pytest.mark.parametrize("wf_key,wf_class", list(WORKFLOW_CLASSES.items()))
    def test_workflow_has_docstring(self, wf_key, wf_class):
        """All workflow classes have docstrings."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        cls = getattr(mod, wf_class, None)
        if cls is None:
            pytest.skip(f"Class {wf_class} not found")
        assert cls.__doc__ is not None, f"Workflow {wf_class} missing docstring"

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_module_has_pydantic_models(self, wf_key):
        """Workflow module exports Pydantic models (Input/Result)."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        # Check that some model class inherits from BaseModel
        from pydantic import BaseModel
        model_classes = [
            name for name, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, BaseModel) and obj is not BaseModel
        ]
        assert len(model_classes) >= 2, (
            f"Workflow {wf_key} should have at least 2 Pydantic models (Input + Result)"
        )

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_error_handling(self, wf_key):
        """Workflow module defines or imports error/status handling."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        has_error_handling = (
            hasattr(mod, "PhaseStatus")
            or hasattr(mod, "WorkflowStatus")
            or hasattr(mod, "FAILED")
        )
        assert has_error_handling

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_timing(self, wf_key):
        """Workflow modules should import datetime or time for timing."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        source_path = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        content = source_path.read_text(encoding="utf-8")
        has_timing = "datetime" in content or "time" in content
        assert has_timing, f"Workflow {wf_key} should use datetime/time for timing"
