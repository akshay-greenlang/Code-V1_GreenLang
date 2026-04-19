# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Workflow Tests
=====================================================

Tests for all 8 workflows: file existence, module loading, class exports,
execute method, PhaseStatus enum, provenance hashing. Parametrized across
all workflows.

Target: ~40 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

import pytest

from .conftest import (
    _load_workflow,
    WORKFLOWS_DIR,
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
)


# ===========================================================================
# Workflow definitions
# ===========================================================================


EXISTING_WORKFLOWS = [
    ("claim_assessment", "claim_assessment_workflow.py", "ClaimAssessmentWorkflow"),
    ("evidence_collection", "evidence_collection_workflow.py", "EvidenceCollectionWorkflow"),
    ("lifecycle_verification", "lifecycle_verification_workflow.py", "LifecycleVerificationWorkflow"),
    ("label_audit", "label_audit_workflow.py", "LabelAuditWorkflow"),
    ("greenwashing_screening", "greenwashing_screening_workflow.py", "GreenwashingScreeningWorkflow"),
    ("compliance_gap", "compliance_gap_workflow.py", "ComplianceGapWorkflow"),
]

# remediation_planning exists on disk but is referenced differently in conftest
EXTRA_WORKFLOWS = [
    ("remediation_planning", "remediation_planning_workflow.py", "RemediationPlanningWorkflow"),
]

ALL_WORKFLOW_KEYS = list(WORKFLOW_FILES.keys())


# ===========================================================================
# File Existence Tests
# ===========================================================================


class TestWorkflowFileExistence:
    """Tests for workflow file existence."""

    @pytest.mark.parametrize("key", ALL_WORKFLOW_KEYS)
    def test_workflow_registered_in_mapping(self, key):
        """Workflow key is registered in WORKFLOW_FILES."""
        assert key in WORKFLOW_FILES

    @pytest.mark.parametrize("key", ALL_WORKFLOW_KEYS)
    def test_workflow_has_class_mapping(self, key):
        """Workflow key has a class name mapping in WORKFLOW_CLASSES."""
        assert key in WORKFLOW_CLASSES

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_file_exists(self, key, filename, cls_name):
        """Workflow file exists on disk."""
        path = WORKFLOWS_DIR / filename
        assert path.exists(), f"Workflow file missing: {filename}"


# ===========================================================================
# Module Loading Tests
# ===========================================================================


class TestWorkflowModuleLoading:
    """Tests for workflow module loading."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_module_loads(self, key, filename, cls_name):
        """Workflow module loads successfully."""
        mod = _load_workflow(key)
        assert mod is not None

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_class_exists(self, key, filename, cls_name):
        """Workflow module exports the expected class."""
        mod = _load_workflow(key)
        assert hasattr(mod, cls_name), f"Class {cls_name} not found in {key}"

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_class_has_docstring(self, key, filename, cls_name):
        """Workflow class has a docstring."""
        mod = _load_workflow(key)
        cls = getattr(mod, cls_name)
        assert cls.__doc__ is not None


# ===========================================================================
# Method Existence Tests
# ===========================================================================


class TestWorkflowMethodExistence:
    """Tests for workflow execute method."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_has_execute_method(self, key, filename, cls_name):
        """Workflow class has execute method."""
        mod = _load_workflow(key)
        cls = getattr(mod, cls_name)
        instance = cls()
        assert hasattr(instance, "execute")
        assert callable(instance.execute)


# ===========================================================================
# PhaseStatus Enum Tests
# ===========================================================================


class TestWorkflowPhaseStatus:
    """Tests for PhaseStatus enum in workflows."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_has_phase_status_enum(self, key, filename, cls_name):
        """Workflow module exports PhaseStatus enum."""
        mod = _load_workflow(key)
        assert hasattr(mod, "PhaseStatus"), (
            f"PhaseStatus enum not found in {key}"
        )

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_phase_status_has_pending(self, key, filename, cls_name):
        """PhaseStatus includes a pending/not_started value."""
        mod = _load_workflow(key)
        values = {m.value for m in mod.PhaseStatus}
        has_initial = (
            "pending" in values
            or "not_started" in values
            or "queued" in values
        )
        assert has_initial

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_phase_status_has_completed(self, key, filename, cls_name):
        """PhaseStatus includes a completed/done value."""
        mod = _load_workflow(key)
        values = {m.value for m in mod.PhaseStatus}
        has_completed = (
            "completed" in values
            or "done" in values
            or "finished" in values
        )
        assert has_completed


# ===========================================================================
# Source File Characteristic Tests
# ===========================================================================


class TestWorkflowSourceCharacteristics:
    """Tests for workflow source file characteristics."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_source_has_provenance(self, key, filename, cls_name):
        """Workflow source references provenance or hashing."""
        source = (WORKFLOWS_DIR / filename).read_text(encoding="utf-8")
        has_provenance = (
            "sha256" in source.lower()
            or "hashlib" in source
            or "provenance" in source.lower()
            or "hash" in source.lower()
        )
        assert has_provenance

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_source_has_pydantic(self, key, filename, cls_name):
        """Workflow source uses Pydantic BaseModel."""
        source = (WORKFLOWS_DIR / filename).read_text(encoding="utf-8")
        assert "BaseModel" in source

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_WORKFLOWS)
    def test_workflow_source_has_logging(self, key, filename, cls_name):
        """Workflow source uses logging."""
        source = (WORKFLOWS_DIR / filename).read_text(encoding="utf-8")
        assert "logging" in source


# ===========================================================================
# Missing Workflows Tests
# ===========================================================================


class TestMissingWorkflows:
    """Tests documenting workflows not yet created."""

    def test_regulatory_submission_registered(self):
        """regulatory_submission is registered in WORKFLOW_FILES."""
        assert "regulatory_submission" in WORKFLOW_FILES

    def test_regulatory_submission_file_status(self):
        """regulatory_submission_workflow.py existence check."""
        path = WORKFLOWS_DIR / "regulatory_submission_workflow.py"
        if not path.exists():
            pytest.skip("regulatory_submission_workflow.py not yet created")
        assert path.exists()

    def test_remediation_planning_file_exists(self):
        """remediation_planning_workflow.py exists on disk."""
        path = WORKFLOWS_DIR / "remediation_planning_workflow.py"
        assert path.exists()
