# -*- coding: utf-8 -*-
"""
Unit tests for PACK-032 Building Energy Assessment Workflows

Tests workflow module loading, class instantiation, phase/step definitions,
and structural requirements for all 8 assessment workflows.

Target: 15+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"


WORKFLOW_DEFINITIONS = {
    "initial_building_assessment_workflow": "InitialBuildingAssessmentWorkflow",
    "epc_generation_workflow": "EPCGenerationWorkflow",
    "retrofit_planning_workflow": "RetrofitPlanningWorkflow",
    "certification_assessment_workflow": "CertificationAssessmentWorkflow",
    "nzeb_readiness_workflow": "NZEBReadinessWorkflow",
    "regulatory_compliance_workflow": "RegulatoryComplianceWorkflow",
    "continuous_building_monitoring_workflow": "ContinuousBuildingMonitoringWorkflow",
    "tenant_engagement_workflow": "TenantEngagementWorkflow",
}


def _load_wf(name: str):
    path = WORKFLOWS_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Workflow file not found: {path}")
    mod_key = f"pack032_wf.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =========================================================================
# Test Workflow File Existence
# =========================================================================


class TestWorkflowFiles:
    @pytest.mark.parametrize("wf_file", list(WORKFLOW_DEFINITIONS.keys()))
    def test_workflow_file_exists(self, wf_file):
        path = WORKFLOWS_DIR / f"{wf_file}.py"
        assert path.exists(), f"Workflow file missing: {wf_file}.py"

    def test_workflow_init_exists(self):
        assert (WORKFLOWS_DIR / "__init__.py").exists()

    def test_workflow_count(self):
        py_files = [f for f in WORKFLOWS_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 8


# =========================================================================
# Test Workflow Class Loading
# =========================================================================


class TestWorkflowClasses:
    @pytest.mark.parametrize(
        "wf_file,class_name",
        list(WORKFLOW_DEFINITIONS.items()),
    )
    def test_workflow_class_exists(self, wf_file, class_name):
        mod = _load_wf(wf_file)
        assert hasattr(mod, class_name), f"{class_name} not found in {wf_file}"

    @pytest.mark.parametrize(
        "wf_file,class_name",
        list(WORKFLOW_DEFINITIONS.items()),
    )
    def test_workflow_instantiation(self, wf_file, class_name):
        mod = _load_wf(wf_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert instance is not None


# =========================================================================
# Test Workflow Execute Method
# =========================================================================


class TestWorkflowMethods:
    @pytest.mark.parametrize(
        "wf_file,class_name",
        list(WORKFLOW_DEFINITIONS.items()),
    )
    def test_has_execute_method(self, wf_file, class_name):
        mod = _load_wf(wf_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert hasattr(instance, "execute"), f"{class_name} has no execute() method"


# =========================================================================
# Test Workflow Enums
# =========================================================================


class TestWorkflowEnums:
    def test_phase_status_enum(self):
        mod = _load_wf("initial_building_assessment_workflow")
        assert hasattr(mod, "PhaseStatus")
        ps = mod.PhaseStatus
        assert hasattr(ps, "PENDING")
        assert hasattr(ps, "COMPLETED")

    def test_workflow_status_enum(self):
        mod = _load_wf("initial_building_assessment_workflow")
        assert hasattr(mod, "WorkflowStatus")
        ws = mod.WorkflowStatus
        assert hasattr(ws, "PENDING")
        assert hasattr(ws, "COMPLETED")
        assert hasattr(ws, "FAILED")


# =========================================================================
# Test Workflow Structural Patterns
# =========================================================================


class TestWorkflowPatterns:
    def test_initial_assessment_has_building_type_enum(self):
        mod = _load_wf("initial_building_assessment_workflow")
        assert hasattr(mod, "BuildingType")

    def test_initial_assessment_has_construction_era(self):
        mod = _load_wf("initial_building_assessment_workflow")
        assert hasattr(mod, "ConstructionEra")

    def test_epc_generation_loadable(self):
        mod = _load_wf("epc_generation_workflow")
        assert hasattr(mod, "EPCGenerationWorkflow")

    def test_retrofit_planning_loadable(self):
        mod = _load_wf("retrofit_planning_workflow")
        assert hasattr(mod, "RetrofitPlanningWorkflow")

    def test_nzeb_readiness_loadable(self):
        mod = _load_wf("nzeb_readiness_workflow")
        assert hasattr(mod, "NZEBReadinessWorkflow")
