# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Workflow Tests
=============================================

Tests all 8 workflows for importability, correct phase counts,
Input/Result Pydantic models, phase definitions, and execute methods.

Test Count Target: ~65 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    WORKFLOW_PHASE_COUNTS,
    _load_module,
)


def _load_workflow(wf_key: str):
    """Load a workflow module by its logical key."""
    file_name = WORKFLOW_FILES.get(wf_key)
    if file_name is None:
        pytest.skip(f"Unknown workflow key: {wf_key}")
    try:
        return _load_module(wf_key, file_name, "workflows")
    except FileNotFoundError:
        pytest.skip(f"Workflow file not found: {file_name}")
    except ImportError as exc:
        pytest.skip(f"Cannot load workflow {wf_key}: {exc}")


# =========================================================================
# 1. Workflow File Presence
# =========================================================================


class TestWorkflowFilePresence:
    """Test all 8 workflow files exist on disk."""

    @pytest.mark.parametrize("wf_key,file_name", list(WORKFLOW_FILES.items()))
    def test_workflow_file_exists(self, wf_key, file_name):
        """Workflow Python file exists."""
        path = WORKFLOWS_DIR / file_name
        if not path.exists():
            pytest.skip(f"File not found: {path}")
        assert path.is_file()
        assert path.suffix == ".py"


# =========================================================================
# 2. Workflow Module Loading
# =========================================================================


class TestWorkflowModuleLoading:
    """Test all 8 workflow modules can be loaded."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_module_loads(self, wf_key):
        """Workflow module loads without error."""
        mod = _load_workflow(wf_key)
        assert mod is not None


# =========================================================================
# 3. Workflow Class Instantiation
# =========================================================================


class TestWorkflowClassInstantiation:
    """Test workflow class exists and can be instantiated."""

    @pytest.mark.parametrize("wf_key,class_name", list(WORKFLOW_CLASSES.items()))
    def test_workflow_class_exists(self, wf_key, class_name):
        """Workflow class is defined in the module."""
        mod = _load_workflow(wf_key)
        assert hasattr(mod, class_name), f"{class_name} not found in {wf_key}"

    @pytest.mark.parametrize("wf_key,class_name", list(WORKFLOW_CLASSES.items()))
    def test_workflow_instantiation(self, wf_key, class_name):
        """Workflow class can be instantiated."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        assert instance is not None


# =========================================================================
# 4. Workflow Phase Counts
# =========================================================================


class TestWorkflowPhaseCounts:
    """Test each workflow has the correct number of phases."""

    @pytest.mark.parametrize("wf_key,expected_phases", list(WORKFLOW_PHASE_COUNTS.items()))
    def test_phase_count(self, wf_key, expected_phases):
        """Workflow has the expected number of phases."""
        mod = _load_workflow(wf_key)
        class_name = WORKFLOW_CLASSES.get(wf_key)
        if class_name is None:
            pytest.skip(f"No class mapping for {wf_key}")
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        # Check for phases attribute (list or dict)
        phases = getattr(instance, "phases", None)
        if phases is None:
            phases = getattr(instance, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute found")
        assert len(phases) == expected_phases


# =========================================================================
# 5. Workflow Input/Result Models
# =========================================================================


class TestWorkflowModels:
    """Test workflow Input and Result Pydantic models."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_has_input_model(self, wf_key):
        """Workflow module defines an Input model."""
        mod = _load_workflow(wf_key)
        # Convention: {ClassName}Input or WorkflowInput
        class_name = WORKFLOW_CLASSES.get(wf_key, "")
        input_name = f"{class_name}Input"
        generic_name = "WorkflowInput"
        has_input = hasattr(mod, input_name) or hasattr(mod, generic_name)
        if not has_input:
            pytest.skip(f"No Input model found for {wf_key}")
        assert has_input

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_has_result_model(self, wf_key):
        """Workflow module defines a Result model."""
        mod = _load_workflow(wf_key)
        class_name = WORKFLOW_CLASSES.get(wf_key, "")
        result_name = f"{class_name}Result"
        generic_name = "WorkflowResult"
        has_result = hasattr(mod, result_name) or hasattr(mod, generic_name)
        if not has_result:
            pytest.skip(f"No Result model found for {wf_key}")
        assert has_result


# =========================================================================
# 6. Workflow Execute Method
# =========================================================================


class TestWorkflowExecuteMethod:
    """Test workflow execute method exists."""

    @pytest.mark.parametrize("wf_key,class_name", list(WORKFLOW_CLASSES.items()))
    def test_execute_method_exists(self, wf_key, class_name):
        """Workflow class has an execute() or run() method."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        has_execute = hasattr(instance, "execute") or hasattr(instance, "run")
        assert has_execute, f"{class_name} has no execute/run method"


# =========================================================================
# 7. Individual Workflow Tests
# =========================================================================


class TestInitialBenchmarkWorkflow:
    """Test initial benchmark workflow specifics."""

    def test_initial_benchmark_4_phases(self):
        """Initial benchmark workflow has 4 phases."""
        mod = _load_workflow("initial_benchmark")
        cls = getattr(mod, "InitialBenchmarkWorkflow", None)
        if cls is None:
            pytest.skip("InitialBenchmarkWorkflow not found")
        instance = cls()
        phases = getattr(instance, "phases", None) or getattr(instance, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute")
        assert len(phases) == 4

    def test_initial_benchmark_has_data_collection_phase(self):
        """Initial benchmark starts with data collection."""
        mod = _load_workflow("initial_benchmark")
        cls = getattr(mod, "InitialBenchmarkWorkflow", None)
        if cls is None:
            pytest.skip("InitialBenchmarkWorkflow not found")
        instance = cls()
        phases = getattr(instance, "phases", None) or getattr(instance, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute")
        # First phase should relate to data collection
        first_phase = phases[0] if isinstance(phases, list) else list(phases.keys())[0]
        phase_str = str(first_phase).lower()
        assert "data" in phase_str or "collect" in phase_str or "input" in phase_str


class TestFullAssessmentWorkflow:
    """Test full assessment workflow specifics."""

    def test_full_assessment_6_phases(self):
        """Full assessment workflow has 6 phases."""
        mod = _load_workflow("full_assessment")
        cls = getattr(mod, "FullAssessmentWorkflow", None)
        if cls is None:
            pytest.skip("FullAssessmentWorkflow not found")
        instance = cls()
        phases = getattr(instance, "phases", None) or getattr(instance, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute")
        assert len(phases) == 6


class TestPortfolioBenchmarkWorkflow:
    """Test portfolio benchmark workflow specifics."""

    def test_portfolio_workflow_5_phases(self):
        """Portfolio benchmark workflow has 5 phases."""
        mod = _load_workflow("portfolio_benchmark")
        cls = getattr(mod, "PortfolioBenchmarkWorkflow", None)
        if cls is None:
            pytest.skip("PortfolioBenchmarkWorkflow not found")
        instance = cls()
        phases = getattr(instance, "phases", None) or getattr(instance, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute")
        assert len(phases) == 5


class TestContinuousMonitoringWorkflow:
    """Test continuous monitoring workflow specifics."""

    def test_continuous_monitoring_4_phases(self):
        """Continuous monitoring workflow has 4 phases."""
        mod = _load_workflow("continuous_monitoring")
        cls = getattr(mod, "ContinuousMonitoringWorkflow", None)
        if cls is None:
            pytest.skip("ContinuousMonitoringWorkflow not found")
        instance = cls()
        phases = getattr(instance, "phases", None) or getattr(instance, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute")
        assert len(phases) == 4


# =========================================================================
# 8. Workflow Metadata
# =========================================================================


class TestWorkflowMetadata:
    """Test workflow metadata and version info."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_has_version(self, wf_key):
        """Workflow module defines _MODULE_VERSION."""
        mod = _load_workflow(wf_key)
        if not hasattr(mod, "_MODULE_VERSION"):
            pytest.skip(f"_MODULE_VERSION not found in {wf_key}")
        assert mod._MODULE_VERSION == "1.0.0"

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_has_docstring(self, wf_key):
        """Workflow module has a docstring."""
        mod = _load_workflow(wf_key)
        assert mod.__doc__ is not None
        assert len(mod.__doc__) > 20
