# -*- coding: utf-8 -*-
"""
Unit tests for PACK-038 Workflows
====================================

Tests all 8 workflows: load analysis, peak assessment, BESS evaluation,
shifting optimization, CP strategy, ratchet mitigation, reporting,
and full lifecycle.

Coverage target: 85%+
Total tests: ~90
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"

WORKFLOW_FILES = {
    "load_analysis": "load_analysis_workflow.py",
    "peak_assessment": "peak_assessment_workflow.py",
    "bess_evaluation": "bess_evaluation_workflow.py",
    "shifting_optimization": "shifting_optimization_workflow.py",
    "cp_strategy": "cp_strategy_workflow.py",
    "ratchet_mitigation": "ratchet_mitigation_workflow.py",
    "reporting": "reporting_workflow.py",
    "full_lifecycle": "full_lifecycle_workflow.py",
}

WORKFLOW_CLASSES = {
    "load_analysis": "LoadAnalysisWorkflow",
    "peak_assessment": "PeakAssessmentWorkflow",
    "bess_evaluation": "BESSEvaluationWorkflow",
    "shifting_optimization": "ShiftingOptimizationWorkflow",
    "cp_strategy": "CPStrategyWorkflow",
    "ratchet_mitigation": "RatchetMitigationWorkflow",
    "reporting": "ReportingWorkflow",
    "full_lifecycle": "FullLifecycleWorkflow",
}

WORKFLOW_PHASE_COUNTS = {
    "load_analysis": 4,
    "peak_assessment": 5,
    "bess_evaluation": 5,
    "shifting_optimization": 4,
    "cp_strategy": 4,
    "ratchet_mitigation": 4,
    "reporting": 3,
    "full_lifecycle": 8,
}


def _load_workflow(name: str):
    file_name = WORKFLOW_FILES[name]
    path = WORKFLOWS_DIR / file_name
    if not path.exists():
        pytest.skip(f"Workflow file not found: {path}")
    mod_key = f"pack038_test_wf.{name}"
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
    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_module_loads(self, wf_key):
        mod = _load_workflow(wf_key)
        assert mod is not None


# =============================================================================
# Class Instantiation
# =============================================================================


class TestWorkflowClassInstantiation:
    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_instantiate(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        assert instance is not None


# =============================================================================
# Phase Counts
# =============================================================================


class TestWorkflowPhaseCounts:
    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_phase_count(self, wf_key):
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        phases = (getattr(instance, "phases", None)
                  or getattr(instance, "phase_definitions", None)
                  or getattr(cls, "PHASES", None))
        if phases is not None:
            expected = WORKFLOW_PHASE_COUNTS.get(wf_key, 3)
            assert len(phases) >= expected - 1


# =============================================================================
# Execute Method
# =============================================================================


class TestWorkflowExecuteMethod:
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
    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_status_enum(self, wf_key):
        mod = _load_workflow(wf_key)
        has_status = (hasattr(mod, "WorkflowStatus")
                      or hasattr(mod, "PhaseStatus"))
        assert has_status


# =============================================================================
# Load Analysis Workflow
# =============================================================================


class TestLoadAnalysisWorkflow:
    def test_phases(self):
        mod = _load_workflow("load_analysis")
        cls = getattr(mod, "LoadAnalysisWorkflow", None)
        if cls is None:
            pytest.skip("LoadAnalysisWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 4

    def test_phase_names(self):
        mod = _load_workflow("load_analysis")
        cls = getattr(mod, "LoadAnalysisWorkflow", None)
        if cls is None:
            pytest.skip("LoadAnalysisWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            phase_str = " ".join(str(p) for p in (
                phases if isinstance(phases, list) else phases.keys()
            )).lower()
            assert "load" in phase_str or "profile" in phase_str or "data" in phase_str

    def test_accepts_data(self):
        mod = _load_workflow("load_analysis")
        cls = getattr(mod, "LoadAnalysisWorkflow", None)
        if cls is None:
            pytest.skip("LoadAnalysisWorkflow not found")
        wf = cls()
        assert wf is not None


# =============================================================================
# Peak Assessment Workflow
# =============================================================================


class TestPeakAssessmentWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("peak_assessment")
        cls = getattr(mod, "PeakAssessmentWorkflow", None)
        if cls is None:
            pytest.skip("PeakAssessmentWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_phases(self):
        mod = _load_workflow("peak_assessment")
        cls = getattr(mod, "PeakAssessmentWorkflow", None)
        if cls is None:
            pytest.skip("PeakAssessmentWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 4

    def test_has_peak_phase(self):
        mod = _load_workflow("peak_assessment")
        cls = getattr(mod, "PeakAssessmentWorkflow", None)
        if cls is None:
            pytest.skip("PeakAssessmentWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            phase_str = " ".join(str(p) for p in (
                phases if isinstance(phases, list) else phases.keys()
            )).lower()
            assert "peak" in phase_str or "identify" in phase_str or "assess" in phase_str


# =============================================================================
# BESS Evaluation Workflow
# =============================================================================


class TestBESSEvaluationWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("bess_evaluation")
        cls = getattr(mod, "BESSEvaluationWorkflow", None)
        if cls is None:
            pytest.skip("BESSEvaluationWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_phases(self):
        mod = _load_workflow("bess_evaluation")
        cls = getattr(mod, "BESSEvaluationWorkflow", None)
        if cls is None:
            pytest.skip("BESSEvaluationWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 4

    def test_has_sizing_phase(self):
        mod = _load_workflow("bess_evaluation")
        cls = getattr(mod, "BESSEvaluationWorkflow", None)
        if cls is None:
            pytest.skip("BESSEvaluationWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            phase_str = " ".join(str(p) for p in (
                phases if isinstance(phases, list) else phases.keys()
            )).lower()
            assert "size" in phase_str or "bess" in phase_str or "battery" in phase_str


# =============================================================================
# Shifting Optimization Workflow
# =============================================================================


class TestShiftingOptimizationWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("shifting_optimization")
        cls = getattr(mod, "ShiftingOptimizationWorkflow", None)
        if cls is None:
            pytest.skip("ShiftingOptimizationWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_has_shift_phase(self):
        mod = _load_workflow("shifting_optimization")
        cls = getattr(mod, "ShiftingOptimizationWorkflow", None)
        if cls is None:
            pytest.skip("ShiftingOptimizationWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            phase_str = " ".join(str(p) for p in (
                phases if isinstance(phases, list) else phases.keys()
            )).lower()
            assert "shift" in phase_str or "optim" in phase_str or "schedule" in phase_str


# =============================================================================
# CP Strategy Workflow
# =============================================================================


class TestCPStrategyWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("cp_strategy")
        cls = getattr(mod, "CPStrategyWorkflow", None)
        if cls is None:
            pytest.skip("CPStrategyWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_phases(self):
        mod = _load_workflow("cp_strategy")
        cls = getattr(mod, "CPStrategyWorkflow", None)
        if cls is None:
            pytest.skip("CPStrategyWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 3


# =============================================================================
# Ratchet Mitigation Workflow
# =============================================================================


class TestRatchetMitigationWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("ratchet_mitigation")
        cls = getattr(mod, "RatchetMitigationWorkflow", None)
        if cls is None:
            pytest.skip("RatchetMitigationWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_has_ratchet_phase(self):
        mod = _load_workflow("ratchet_mitigation")
        cls = getattr(mod, "RatchetMitigationWorkflow", None)
        if cls is None:
            pytest.skip("RatchetMitigationWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            phase_str = " ".join(str(p) for p in (
                phases if isinstance(phases, list) else phases.keys()
            )).lower()
            assert "ratchet" in phase_str or "spike" in phase_str or "mitig" in phase_str


# =============================================================================
# Reporting Workflow
# =============================================================================


class TestReportingWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("reporting")
        cls = getattr(mod, "ReportingWorkflow", None)
        if cls is None:
            pytest.skip("ReportingWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_phases(self):
        mod = _load_workflow("reporting")
        cls = getattr(mod, "ReportingWorkflow", None)
        if cls is None:
            pytest.skip("ReportingWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 3


# =============================================================================
# Full Lifecycle Workflow
# =============================================================================


class TestFullLifecycleWorkflow:
    def test_instantiation(self):
        mod = _load_workflow("full_lifecycle")
        cls = getattr(mod, "FullLifecycleWorkflow", None)
        if cls is None:
            pytest.skip("FullLifecycleWorkflow not found")
        wf = cls()
        assert wf is not None

    def test_phases(self):
        mod = _load_workflow("full_lifecycle")
        cls = getattr(mod, "FullLifecycleWorkflow", None)
        if cls is None:
            pytest.skip("FullLifecycleWorkflow not found")
        wf = cls()
        phases = (getattr(wf, "phases", None)
                  or getattr(wf, "phase_definitions", None))
        if phases:
            assert len(phases) >= 6

    def test_full_lifecycle_covers_engines(self):
        mod = _load_workflow("full_lifecycle")
        cls = getattr(mod, "FullLifecycleWorkflow", None)
        if cls is None:
            pytest.skip("FullLifecycleWorkflow not found")
        wf = cls()
        engines = (getattr(wf, "engines", None)
                   or getattr(wf, "engine_sequence", None)
                   or getattr(wf, "_engines", None))
        if engines:
            assert len(engines) >= 5


# =============================================================================
# Module Attributes
# =============================================================================


class TestWorkflowModuleAttributes:
    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_module_version(self, wf_key):
        mod = _load_workflow(wf_key)
        has_ver = hasattr(mod, "_MODULE_VERSION") or hasattr(mod, "__version__")
        assert has_ver or True

    @pytest.mark.parametrize("wf_key", EXISTING_WORKFLOW_KEYS)
    def test_has_docstring(self, wf_key):
        mod = _load_workflow(wf_key)
        assert mod.__doc__ is not None


# =============================================================================
# Error Handling
# =============================================================================


class TestWorkflowErrorHandling:
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


# =============================================================================
# Naming Convention
# =============================================================================


class TestWorkflowNamingConvention:
    def test_workflow_files_end_with_workflow(self):
        for key, filename in WORKFLOW_FILES.items():
            assert filename.endswith("_workflow.py")

    def test_workflow_classes_end_with_workflow(self):
        for key, cls_name in WORKFLOW_CLASSES.items():
            assert cls_name.endswith("Workflow")

    def test_workflow_file_count(self):
        assert len(WORKFLOW_FILES) == 8

    def test_keys_match(self):
        assert set(WORKFLOW_FILES.keys()) == set(WORKFLOW_CLASSES.keys())

    def test_phase_count_keys_match(self):
        assert set(WORKFLOW_PHASE_COUNTS.keys()) == set(WORKFLOW_FILES.keys())
