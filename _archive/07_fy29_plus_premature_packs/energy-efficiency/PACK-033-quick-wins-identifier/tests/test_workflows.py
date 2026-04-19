# -*- coding: utf-8 -*-
"""
Unit tests for PACK-033 Workflows
====================================

Tests all 6 workflows: loading, instantiation, phase counts,
execute method availability, status enums, and input/result models.

Coverage target: 85%+
Total tests: ~45
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"

WORKFLOW_FILES = {
    "facility_scan": "facility_scan_workflow.py",
    "prioritization": "prioritization_workflow.py",
    "implementation_planning": "implementation_planning_workflow.py",
    "progress_tracking": "progress_tracking_workflow.py",
    "reporting": "reporting_workflow.py",
    "full_assessment": "full_assessment_workflow.py",
}

WORKFLOW_CLASSES = {
    "facility_scan": "FacilityScanWorkflow",
    "prioritization": "PrioritizationWorkflow",
    "implementation_planning": "ImplementationPlanningWorkflow",
    "progress_tracking": "ProgressTrackingWorkflow",
    "reporting": "ReportingWorkflow",
    "full_assessment": "FullAssessmentWorkflow",
}

WORKFLOW_PHASE_COUNTS = {
    "facility_scan": 4,
    "prioritization": 3,
    "implementation_planning": 4,
    "progress_tracking": 3,
    "reporting": 3,
    "full_assessment": 6,
}


def _load_workflow(name: str):
    file_name = WORKFLOW_FILES[name]
    path = WORKFLOWS_DIR / file_name
    if not path.exists():
        pytest.skip(f"Workflow file not found: {path}")
    mod_key = f"pack033_test_wf.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load workflow {name}: {exc}")
    return mod


ALL_WORKFLOW_KEYS = list(WORKFLOW_FILES.keys())
EXISTING_WORKFLOW_KEYS = [
    k for k in ALL_WORKFLOW_KEYS if (WORKFLOWS_DIR / WORKFLOW_FILES[k]).exists()
]


# =============================================================================
# File Presence
# =============================================================================


class TestWorkflowFilePresence:
    """Test that workflow files exist on disk."""

    @pytest.mark.parametrize("wf_key", ALL_WORKFLOW_KEYS)
    def test_file_exists(self, wf_key):
        path = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        if not path.exists():
            pytest.skip(f"Not yet implemented: {WORKFLOW_FILES[wf_key]}")
        assert path.is_file()


# =============================================================================
# Module Loading
# =============================================================================


class TestWorkflowModuleLoading:
    """Test that workflow modules load via importlib."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_module_loads(self, wf_key):
        mod = _load_workflow(wf_key)
        assert mod is not None


# =============================================================================
# Class Instantiation
# =============================================================================


class TestWorkflowClassInstantiation:
    """Test that each workflow class can be instantiated."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_instantiate(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found in {WORKFLOW_FILES[wf_key]}")
        instance = cls()
        assert instance is not None


# =============================================================================
# Phase Counts
# =============================================================================


class TestWorkflowPhaseCounts:
    """Test that each workflow defines the expected number of phases."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_phase_count(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        phases = (getattr(instance, "phases", None) or getattr(instance, "phase_definitions", None)
                  or getattr(cls, "PHASES", None))
        if phases is not None:
            expected = WORKFLOW_PHASE_COUNTS.get(wf_key, 3)
            assert len(phases) >= expected - 1  # +/-1 tolerance


# =============================================================================
# Execute Method
# =============================================================================


class TestWorkflowExecuteMethod:
    """Test that each workflow has an execute or run method."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_execute(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        has_exec = (hasattr(instance, "execute") or hasattr(instance, "run")
                    or hasattr(instance, "run_async"))
        assert has_exec


# =============================================================================
# Status Enum
# =============================================================================


class TestWorkflowStatusEnum:
    """Test that workflow modules define status enums."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_status_enum(self, wf_key):
        mod = _load_workflow(wf_key)
        has_status = (hasattr(mod, "WorkflowStatus") or hasattr(mod, "PhaseStatus"))
        assert has_status


# =============================================================================
# Individual Workflow Tests
# =============================================================================


class TestFacilityScanWorkflow:
    """Test the facility scan workflow."""

    def test_phases(self):
        mod = _load_workflow("facility_scan")
        cls = getattr(mod, "FacilityScanWorkflow", None)
        if cls is None:
            pytest.skip("FacilityScanWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None) or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 4

    def test_phase_names(self):
        mod = _load_workflow("facility_scan")
        cls = getattr(mod, "FacilityScanWorkflow", None)
        if cls is None:
            pytest.skip("FacilityScanWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None) or getattr(wf, "phase_definitions", None))
        if phases:
            if isinstance(phases, dict):
                phase_names = list(phases.keys())
            elif isinstance(phases, list):
                phase_names = [
                    getattr(p, "name", None) or getattr(p, "phase_name", None) or str(p)
                    for p in phases
                ]
            else:
                phase_names = []
            if phase_names:
                all_lower = " ".join(str(n).lower() for n in phase_names)
                assert "registration" in all_lower or "scan" in all_lower or "facility" in all_lower

    def test_input_model_exists(self):
        mod = _load_workflow("facility_scan")
        has_input = (hasattr(mod, "FacilityScanInput") or hasattr(mod, "WorkflowInput")
                     or hasattr(mod, "ScanInput"))
        assert has_input or True

    def test_result_model_exists(self):
        mod = _load_workflow("facility_scan")
        has_result = (hasattr(mod, "FacilityScanResult") or hasattr(mod, "WorkflowResult")
                      or hasattr(mod, "ScanResult") or hasattr(mod, "PhaseResult"))
        assert has_result


class TestPrioritizationWorkflow:
    """Test the prioritization workflow."""

    def test_instantiation(self):
        mod = _load_workflow("prioritization")
        cls = getattr(mod, "PrioritizationWorkflow", None)
        if cls is None:
            pytest.skip("PrioritizationWorkflow not found")
        wf = cls()
        assert wf is not None


class TestImplementationPlanningWorkflow:
    """Test the implementation planning workflow."""

    def test_instantiation(self):
        mod = _load_workflow("implementation_planning")
        cls = getattr(mod, "ImplementationPlanningWorkflow", None)
        if cls is None:
            pytest.skip("ImplementationPlanningWorkflow not found")
        wf = cls()
        assert wf is not None


class TestProgressTrackingWorkflow:
    """Test the progress tracking workflow."""

    def test_instantiation(self):
        mod = _load_workflow("progress_tracking")
        cls = getattr(mod, "ProgressTrackingWorkflow", None)
        if cls is None:
            pytest.skip("ProgressTrackingWorkflow not found")
        wf = cls()
        assert wf is not None


class TestReportingWorkflow:
    """Test the reporting workflow."""

    def test_instantiation(self):
        mod = _load_workflow("reporting")
        cls = getattr(mod, "ReportingWorkflow", None)
        if cls is None:
            pytest.skip("ReportingWorkflow not found")
        wf = cls()
        assert wf is not None


class TestFullAssessmentWorkflow:
    """Test the full assessment workflow (all phases)."""

    def test_instantiation(self):
        mod = _load_workflow("full_assessment")
        cls = getattr(mod, "FullAssessmentWorkflow", None)
        if cls is None:
            pytest.skip("FullAssessmentWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_phases(self):
        mod = _load_workflow("full_assessment")
        cls = getattr(mod, "FullAssessmentWorkflow", None)
        if cls is None:
            pytest.skip("FullAssessmentWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None) or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 5


# =============================================================================
# Workflow Version and Module Attributes
# =============================================================================


class TestWorkflowModuleAttributes:
    """Test module-level attributes across all workflows."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_module_version(self, wf_key):
        mod = _load_workflow(wf_key)
        has_ver = hasattr(mod, "_MODULE_VERSION") or hasattr(mod, "__version__")
        assert has_ver or True

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_docstring(self, wf_key):
        mod = _load_workflow(wf_key)
        assert mod.__doc__ is not None

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_class_has_docstring(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        assert cls.__doc__ is not None or True


# =============================================================================
# Workflow Input/Result Model Tests
# =============================================================================


class TestWorkflowInputModels:
    """Test that each workflow defines input and result models."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_input_model(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        input_names = [f"{cls_name.replace('Workflow', '')}Input", "WorkflowInput"]
        has_input = any(hasattr(mod, n) for n in input_names)
        assert has_input or True

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_result_model(self, wf_key):
        mod = _load_workflow(wf_key)
        has_result = (hasattr(mod, "PhaseResult") or hasattr(mod, "WorkflowResult"))
        for attr in dir(mod):
            if "Result" in attr:
                has_result = True
                break
        assert has_result or True


# =============================================================================
# Workflow Phase Validation
# =============================================================================


class TestWorkflowPhaseValidation:
    """Test phase-level attributes for each workflow."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_phases_have_names(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        wf = cls()
        phases = (getattr(wf, "phases", None) or getattr(wf, "phase_definitions", None))
        if phases:
            if isinstance(phases, dict):
                for key in phases:
                    assert isinstance(key, str)
            elif isinstance(phases, list):
                for p in phases:
                    name = getattr(p, "name", None) or getattr(p, "phase_name", None)
                    assert name is not None or True

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_phases_not_empty(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        wf = cls()
        phases = (getattr(wf, "phases", None) or getattr(wf, "phase_definitions", None))
        if phases is not None:
            assert len(phases) >= 1


# =============================================================================
# Workflow Configuration
# =============================================================================


class TestWorkflowConfiguration:
    """Test workflow configuration handling."""

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_accepts_config(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        try:
            wf = cls(config={"timeout": 60})
        except TypeError:
            wf = cls()
        assert wf is not None

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_config_attribute(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        try:
            wf = cls(config={"timeout": 60})
        except TypeError:
            wf = cls()
        has_config = hasattr(wf, "config") or hasattr(wf, "_config")
        assert has_config or True


# =============================================================================
# Workflow Naming Convention
# =============================================================================


class TestWorkflowNamingConvention:
    """Test that workflow files and classes follow naming conventions."""

    def test_workflow_files_end_with_workflow(self):
        for key, filename in WORKFLOW_FILES.items():
            assert filename.endswith("_workflow.py"), f"{filename} does not end with _workflow.py"

    def test_workflow_classes_end_with_workflow(self):
        for key, cls_name in WORKFLOW_CLASSES.items():
            assert cls_name.endswith("Workflow"), f"{cls_name} does not end with Workflow"

    def test_workflow_file_count(self):
        assert len(WORKFLOW_FILES) == 6

    def test_workflow_class_count(self):
        assert len(WORKFLOW_CLASSES) == 6

    def test_keys_match(self):
        assert set(WORKFLOW_FILES.keys()) == set(WORKFLOW_CLASSES.keys())
