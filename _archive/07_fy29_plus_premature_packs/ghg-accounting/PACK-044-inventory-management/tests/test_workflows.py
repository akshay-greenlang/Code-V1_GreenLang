# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Workflow Tests
=======================================

Tests all 8 workflows: module loading, class instantiation, phase
definitions, step validation, and workflow metadata.

Target: 60+ test cases.
"""

from pathlib import Path

import pytest

from conftest import (
    _load_module,
    WORKFLOWS_DIR,
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    WORKFLOW_PHASE_COUNTS,
)


# ===================================================================
# Workflow File Existence Tests
# ===================================================================


class TestWorkflowFileExistence:
    """Tests that all workflow files exist on disk."""

    @pytest.mark.parametrize("wf_key,wf_file", list(WORKFLOW_FILES.items()))
    def test_workflow_file_exists(self, wf_key, wf_file):
        path = WORKFLOWS_DIR / wf_file
        assert path.exists(), f"Workflow file missing: {path}"

    def test_workflows_init_exists(self):
        init_path = WORKFLOWS_DIR / "__init__.py"
        assert init_path.exists(), "__init__.py missing in workflows/"


# ===================================================================
# Workflow Module Loading Tests
# ===================================================================


class TestWorkflowModuleLoading:
    """Tests that all workflow modules load without errors."""

    @pytest.mark.parametrize("wf_key,wf_file", list(WORKFLOW_FILES.items()))
    def test_workflow_module_loads(self, wf_key, wf_file):
        mod = _load_module(wf_key, wf_file, "workflows")
        assert mod is not None

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_CLASSES.keys()))
    def test_workflow_class_exists(self, wf_key):
        wf_file = WORKFLOW_FILES[wf_key]
        mod = _load_module(wf_key, wf_file, "workflows")
        cls_name = WORKFLOW_CLASSES[wf_key]
        assert hasattr(mod, cls_name), f"Class {cls_name} not found in {wf_file}"


# ===================================================================
# Workflow Class Instantiation Tests
# ===================================================================


class TestWorkflowInstantiation:
    """Tests that workflow classes can be instantiated."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_CLASSES.keys()))
    def test_workflow_instantiates(self, wf_key):
        wf_file = WORKFLOW_FILES[wf_key]
        mod = _load_module(wf_key, wf_file, "workflows")
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        assert instance is not None


# ===================================================================
# Workflow Phase Count Tests
# ===================================================================


class TestWorkflowPhaseCounts:
    """Tests that workflows have the expected number of phases."""

    @pytest.mark.parametrize("wf_key,expected_count", list(WORKFLOW_PHASE_COUNTS.items()))
    def test_phase_count(self, wf_key, expected_count):
        wf_file = WORKFLOW_FILES[wf_key]
        mod = _load_module(wf_key, wf_file, "workflows")
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        if hasattr(instance, "PHASE_SEQUENCE"):
            phases = instance.PHASE_SEQUENCE
        elif hasattr(instance, "phases"):
            phases = instance.phases
        elif hasattr(instance, "get_phases"):
            phases = instance.get_phases()
        elif hasattr(instance, "steps"):
            phases = instance.steps
        elif hasattr(instance, "get_steps"):
            phases = instance.get_steps()
        else:
            pytest.skip(f"Cannot determine phases for {wf_key}")
            return
        assert len(phases) == expected_count, (
            f"Workflow {wf_key} has {len(phases)} phases, expected {expected_count}"
        )


# ===================================================================
# Workflow Attribute Tests
# ===================================================================


class TestWorkflowAttributes:
    """Tests for common workflow attributes."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_CLASSES.keys()))
    def test_workflow_has_name(self, wf_key):
        wf_file = WORKFLOW_FILES[wf_key]
        mod = _load_module(wf_key, wf_file, "workflows")
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        has_name = (
            hasattr(instance, "name")
            or hasattr(instance, "workflow_name")
            or hasattr(instance, "NAME")
            or hasattr(instance, "workflow_id")
            or cls.__doc__ is not None
        )
        assert has_name, f"Workflow {wf_key} missing name attribute"

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_CLASSES.keys()))
    def test_workflow_has_description(self, wf_key):
        wf_file = WORKFLOW_FILES[wf_key]
        mod = _load_module(wf_key, wf_file, "workflows")
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        has_desc = (
            hasattr(instance, "description")
            or hasattr(instance, "DESCRIPTION")
            or cls.__doc__ is not None
        )
        assert has_desc, f"Workflow {wf_key} missing description"


# ===================================================================
# Individual Workflow Tests
# ===================================================================


class TestAnnualInventoryCycleWorkflow:
    """Tests for AnnualInventoryCycleWorkflow."""

    def test_load(self):
        mod = _load_module("annual_inventory_cycle",
                           WORKFLOW_FILES["annual_inventory_cycle"], "workflows")
        cls = getattr(mod, "AnnualInventoryCycleWorkflow")
        wf = cls()
        assert wf is not None

    def test_has_8_phases(self):
        mod = _load_module("annual_inventory_cycle",
                           WORKFLOW_FILES["annual_inventory_cycle"], "workflows")
        wf = getattr(mod, "AnnualInventoryCycleWorkflow")()
        phases = getattr(wf, "PHASE_SEQUENCE", getattr(wf, "phases", getattr(wf, "steps", [])))
        assert len(phases) == 8


class TestDataCollectionCampaignWorkflow:
    """Tests for DataCollectionCampaignWorkflow."""

    def test_load(self):
        mod = _load_module("data_collection_campaign",
                           WORKFLOW_FILES["data_collection_campaign"], "workflows")
        cls = getattr(mod, "DataCollectionCampaignWorkflow")
        wf = cls()
        assert wf is not None

    def test_has_5_phases(self):
        mod = _load_module("data_collection_campaign",
                           WORKFLOW_FILES["data_collection_campaign"], "workflows")
        wf = getattr(mod, "DataCollectionCampaignWorkflow")()
        phases = getattr(wf, "PHASE_SEQUENCE", getattr(wf, "phases", getattr(wf, "steps", [])))
        assert len(phases) == 5


class TestQualityReviewWorkflow:
    """Tests for QualityReviewWorkflow."""

    def test_load(self):
        mod = _load_module("quality_review",
                           WORKFLOW_FILES["quality_review"], "workflows")
        wf = getattr(mod, "QualityReviewWorkflow")()
        assert wf is not None


class TestChangeAssessmentWorkflow:
    """Tests for ChangeAssessmentWorkflow."""

    def test_load(self):
        mod = _load_module("change_assessment",
                           WORKFLOW_FILES["change_assessment"], "workflows")
        wf = getattr(mod, "ChangeAssessmentWorkflow")()
        assert wf is not None


class TestInventoryFinalizationWorkflow:
    """Tests for InventoryFinalizationWorkflow."""

    def test_load(self):
        mod = _load_module("inventory_finalization",
                           WORKFLOW_FILES["inventory_finalization"], "workflows")
        wf = getattr(mod, "InventoryFinalizationWorkflow")()
        assert wf is not None


class TestConsolidationWorkflow:
    """Tests for ConsolidationWorkflow."""

    def test_load(self):
        mod = _load_module("consolidation",
                           WORKFLOW_FILES["consolidation"], "workflows")
        wf = getattr(mod, "ConsolidationWorkflow")()
        assert wf is not None


class TestImprovementPlanningWorkflow:
    """Tests for ImprovementPlanningWorkflow."""

    def test_load(self):
        mod = _load_module("improvement_planning",
                           WORKFLOW_FILES["improvement_planning"], "workflows")
        wf = getattr(mod, "ImprovementPlanningWorkflow")()
        assert wf is not None


class TestFullManagementPipelineWorkflow:
    """Tests for FullManagementPipelineWorkflow."""

    def test_load(self):
        mod = _load_module("full_management_pipeline",
                           WORKFLOW_FILES["full_management_pipeline"], "workflows")
        wf = getattr(mod, "FullManagementPipelineWorkflow")()
        assert wf is not None

    def test_has_12_phases(self):
        mod = _load_module("full_management_pipeline",
                           WORKFLOW_FILES["full_management_pipeline"], "workflows")
        wf = getattr(mod, "FullManagementPipelineWorkflow")()
        phases = getattr(wf, "PHASE_SEQUENCE", getattr(wf, "phases", getattr(wf, "steps", [])))
        assert len(phases) == 12


# ===================================================================
# Workflows __init__ Tests
# ===================================================================


class TestWorkflowsInit:
    """Tests for the workflows __init__.py lazy-loading."""

    def test_init_module_loads(self):
        mod = _load_module("workflows_init", "__init__.py", "workflows")
        assert mod is not None

    def test_get_loaded_workflows(self):
        mod = _load_module("workflows_init", "__init__.py", "workflows")
        if hasattr(mod, "get_loaded_workflows"):
            loaded = mod.get_loaded_workflows()
            assert isinstance(loaded, dict)
            assert len(loaded) >= 1

    def test_all_8_workflows_loaded(self):
        mod = _load_module("workflows_init", "__init__.py", "workflows")
        if hasattr(mod, "get_loaded_workflows"):
            loaded = mod.get_loaded_workflows()
            assert len(loaded) == 8
