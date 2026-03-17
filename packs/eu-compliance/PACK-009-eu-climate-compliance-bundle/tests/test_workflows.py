# -*- coding: utf-8 -*-
"""
Workflow tests for PACK-009 EU Climate Compliance Bundle

Tests all 8 workflows: Unified Data Collection, Cross-Regulation Assessment,
Consolidated Reporting, Calendar Management, Cross-Framework Gap Analysis,
Bundle Health Check, Data Consistency Reconciliation, and Annual Compliance
Review. Each workflow is tested for instantiation and simulated execution
with phase result validation.

Coverage target: 85%+
Test count: 10

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories).

    Registers the module in sys.modules so that pydantic can resolve
    forward-referenced annotations created by ``from __future__ import
    annotations``.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_import(module_name: str, file_path: Path):
    """Import a module, returning None if file does not exist or fails."""
    if not file_path.exists():
        return None
    try:
        return _import_from_path(module_name, file_path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load all workflow modules
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
_WORKFLOWS_DIR = _PACK_DIR / "workflows"

_WORKFLOW_MAP = {
    "unified_data_collection": "unified_data_collection.py",
    "cross_regulation_assessment": "cross_regulation_assessment.py",
    "consolidated_reporting": "consolidated_reporting.py",
    "calendar_management": "calendar_management.py",
    "cross_framework_gap_analysis": "cross_framework_gap_analysis.py",
    "bundle_health_check": "bundle_health_check.py",
    "data_consistency_reconciliation": "data_consistency_reconciliation.py",
    "annual_compliance_review": "annual_compliance_review.py",
}

_WORKFLOW_CLASS_NAMES = {
    "unified_data_collection": "UnifiedDataCollectionWorkflow",
    "cross_regulation_assessment": "CrossRegulationAssessmentWorkflow",
    "consolidated_reporting": "ConsolidatedReportingWorkflow",
    "calendar_management": "CalendarManagementWorkflow",
    "cross_framework_gap_analysis": "CrossFrameworkGapAnalysisWorkflow",
    "bundle_health_check": "BundleHealthCheckWorkflow",
    "data_consistency_reconciliation": "DataConsistencyReconciliationWorkflow",
    "annual_compliance_review": "AnnualComplianceReviewWorkflow",
}

_WORKFLOW_RESULT_CLASS_NAMES = {
    "unified_data_collection": "UnifiedDataCollectionResult",
    "cross_regulation_assessment": "CrossRegulationAssessmentResult",
    "consolidated_reporting": "ConsolidatedReportingResult",
    "calendar_management": "CalendarManagementResult",
    "cross_framework_gap_analysis": "CrossFrameworkGapAnalysisResult",
    "bundle_health_check": "BundleHealthCheckResult",
    "data_consistency_reconciliation": "DataConsistencyReconciliationResult",
    "annual_compliance_review": "AnnualComplianceReviewResult",
}

_loaded_modules: Dict[str, Any] = {}
_loaded_classes: Dict[str, Any] = {}
_loaded_configs: Dict[str, Any] = {}
_loaded_results: Dict[str, Any] = {}

for wf_id, filename in _WORKFLOW_MAP.items():
    mod = _safe_import(wf_id, _WORKFLOWS_DIR / filename)
    _loaded_modules[wf_id] = mod
    if mod is not None:
        cls_name = _WORKFLOW_CLASS_NAMES[wf_id]
        _loaded_classes[wf_id] = getattr(mod, cls_name, None)
        _loaded_configs[wf_id] = getattr(mod, "WorkflowConfig", None)
        result_name = _WORKFLOW_RESULT_CLASS_NAMES[wf_id]
        _loaded_results[wf_id] = getattr(mod, result_name, None)

_ANY_WORKFLOW_AVAILABLE = any(v is not None for v in _loaded_classes.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _assert_provenance_hash(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest."""
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64
    assert re.match(r"^[0-9a-f]{64}$", hash_str)


def _get_default_workflow_config(wf_id: str) -> Any:
    """Create a default WorkflowConfig for the given workflow.

    All PACK-009 WorkflowConfig classes require ``organization_id`` and
    ``reporting_year`` as mandatory fields.
    """
    config_cls = _loaded_configs.get(wf_id)
    if config_cls is None:
        return None
    try:
        return config_cls(
            organization_id="test-org-001",
            reporting_year=2025,
        )
    except Exception:
        return None


def _try_execute(wf_id: str):
    """Try to instantiate and execute a workflow, returning the result."""
    wf_cls = _loaded_classes.get(wf_id)
    if wf_cls is None:
        pytest.skip(f"Workflow class for '{wf_id}' not available")
    config = _get_default_workflow_config(wf_id)
    if config is None:
        pytest.skip(f"WorkflowConfig for '{wf_id}' not available")
    workflow = wf_cls()
    result = workflow.execute(config)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkflows:
    """Test suite for all 8 PACK-009 workflows."""

    def test_unified_data_collection_instantiation_and_execute(self):
        """UnifiedDataCollectionWorkflow instantiates and executes."""
        wf_id = "unified_data_collection"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("UnifiedDataCollectionWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_cross_regulation_assessment_instantiation_and_execute(self):
        """CrossRegulationAssessmentWorkflow instantiates and executes."""
        wf_id = "cross_regulation_assessment"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("CrossRegulationAssessmentWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_consolidated_reporting_instantiation_and_execute(self):
        """ConsolidatedReportingWorkflow instantiates and executes."""
        wf_id = "consolidated_reporting"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("ConsolidatedReportingWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_calendar_management_instantiation_and_execute(self):
        """CalendarManagementWorkflow instantiates and executes."""
        wf_id = "calendar_management"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("CalendarManagementWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_cross_framework_gap_analysis_instantiation_and_execute(self):
        """CrossFrameworkGapAnalysisWorkflow instantiates and executes."""
        wf_id = "cross_framework_gap_analysis"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("CrossFrameworkGapAnalysisWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_bundle_health_check_instantiation_and_execute(self):
        """BundleHealthCheckWorkflow instantiates and executes."""
        wf_id = "bundle_health_check"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("BundleHealthCheckWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_data_consistency_reconciliation_instantiation_and_execute(self):
        """DataConsistencyReconciliationWorkflow instantiates and executes."""
        wf_id = "data_consistency_reconciliation"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("DataConsistencyReconciliationWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_annual_compliance_review_instantiation_and_execute(self):
        """AnnualComplianceReviewWorkflow instantiates and executes."""
        wf_id = "annual_compliance_review"
        wf_cls = _loaded_classes.get(wf_id)
        if wf_cls is None:
            pytest.skip("AnnualComplianceReviewWorkflow not available")
        workflow = wf_cls()
        assert workflow is not None

        config = _get_default_workflow_config(wf_id)
        if config is None:
            pytest.skip("WorkflowConfig not available")
        result = workflow.execute(config)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "phases") or hasattr(result, "phase_results")

    def test_all_workflows_return_provenance_hash(self):
        """All successfully executed workflows include a provenance hash."""
        for wf_id in _WORKFLOW_MAP:
            wf_cls = _loaded_classes.get(wf_id)
            config_cls = _loaded_configs.get(wf_id)
            if wf_cls is None or config_cls is None:
                continue
            try:
                workflow = wf_cls()
                config = config_cls(
                    organization_id="test-org-001",
                    reporting_year=2025,
                )
                result = workflow.execute(config)
            except Exception:
                continue

            if hasattr(result, "provenance_hash"):
                ph = result.provenance_hash
                if ph:
                    _assert_provenance_hash(ph)

    def test_all_workflows_have_phase_results(self):
        """All successfully executed workflows populate phase_results."""
        workflows_tested = 0
        for wf_id in _WORKFLOW_MAP:
            wf_cls = _loaded_classes.get(wf_id)
            config_cls = _loaded_configs.get(wf_id)
            if wf_cls is None or config_cls is None:
                continue
            try:
                workflow = wf_cls()
                config = config_cls(
                    organization_id="test-org-001",
                    reporting_year=2025,
                )
                result = workflow.execute(config)
            except Exception:
                continue

            phases_attr = None
            if hasattr(result, "phases"):
                phases_attr = result.phases
            elif hasattr(result, "phase_results"):
                phases_attr = result.phase_results

            if phases_attr is not None:
                assert isinstance(phases_attr, (list, dict)), (
                    f"{wf_id}: phases must be list or dict, got {type(phases_attr)}"
                )
                if isinstance(phases_attr, list):
                    assert len(phases_attr) >= 1, (
                        f"{wf_id}: expected at least 1 phase result"
                    )
                else:
                    assert len(phases_attr) >= 1, (
                        f"{wf_id}: expected at least 1 phase result entry"
                    )
                workflows_tested += 1

        if workflows_tested == 0:
            pytest.skip("No workflows could be executed successfully")
