# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Workflow Tests
================================================

Tests all 8 workflow classes for instantiation, execution, phase
progression, and provenance tracking. Each workflow is loaded dynamically
via conftest helpers.

Test count target: ~40 tests
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    WORKFLOW_CLASSES,
    WORKFLOW_FILES,
    WORKFLOWS_DIR,
    _load_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKFLOW_KEYS = list(WORKFLOW_FILES.keys())


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


class TestWorkflowFilesExist:
    """Verify all workflow source files are present on disk."""

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_workflow_file_exists(self, wf_key: str):
        """Each workflow file must exist under workflows/."""
        filepath = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        assert filepath.exists(), f"Missing workflow file: {filepath}"


# ---------------------------------------------------------------------------
# 2. Module loading
# ---------------------------------------------------------------------------


class TestWorkflowModuleLoading:
    """Verify all workflow modules load without import errors."""

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_workflow_module_loads(self, wf_key: str):
        """Each workflow module must load successfully."""
        mod = _load_workflow(wf_key)
        assert mod is not None


# ---------------------------------------------------------------------------
# 3. Class instantiation
# ---------------------------------------------------------------------------


class TestWorkflowInstantiation:
    """Verify all workflow classes can be instantiated."""

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_workflow_class_exists(self, wf_key: str):
        """The expected class must be exported from the module."""
        mod = _load_workflow(wf_key)
        cls_name = WORKFLOW_CLASSES[wf_key]
        cls = getattr(mod, cls_name, None)
        assert cls is not None, f"Class {cls_name} not found in {wf_key}"

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_workflow_instantiation_no_args(self, wf_key: str):
        """Each workflow class must instantiate with no arguments."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        assert instance is not None

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_workflow_has_workflow_id(self, wf_key: str):
        """Each workflow instance must have a workflow_id attribute."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        assert hasattr(instance, "workflow_id")
        assert isinstance(instance.workflow_id, str)
        assert len(instance.workflow_id) > 0


# ---------------------------------------------------------------------------
# 4. get_phases()
# ---------------------------------------------------------------------------


class TestWorkflowPhases:
    """Verify get_phases() returns non-empty phase lists."""

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_get_phases_returns_list(self, wf_key: str):
        """get_phases() must return a non-empty list of dicts."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        phases = instance.get_phases()
        assert isinstance(phases, list)
        assert len(phases) >= 3, "Workflows should have at least 3 phases"

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_phases_have_name_and_description(self, wf_key: str):
        """Each phase dict must contain name and description keys."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        for phase in instance.get_phases():
            assert "name" in phase, f"Phase missing 'name': {phase}"
            assert "description" in phase, f"Phase missing 'description': {phase}"


# ---------------------------------------------------------------------------
# 5. execute() with defaults
# ---------------------------------------------------------------------------


class TestWorkflowExecution:
    """Verify execute() completes with default inputs."""

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_execute_returns_result(self, wf_key: str):
        """execute() must return a result object (not None)."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        result = _run(instance.execute())
        assert result is not None

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_result_has_workflow_id(self, wf_key: str):
        """Result must contain a workflow_id string."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        result = _run(instance.execute())
        assert hasattr(result, "workflow_id")
        assert isinstance(result.workflow_id, str)
        assert len(result.workflow_id) > 0

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_result_has_provenance_hash(self, wf_key: str):
        """Result must contain a 64-char SHA-256 provenance_hash."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        result = _run(instance.execute())
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_result_has_executed_at(self, wf_key: str):
        """Result must contain a non-empty executed_at timestamp."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        result = _run(instance.execute())
        assert hasattr(result, "executed_at")
        assert isinstance(result.executed_at, str)
        assert len(result.executed_at) > 0

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_result_has_status_completed(self, wf_key: str):
        """Result status must be 'completed' for default inputs."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        result = _run(instance.execute())
        status_val = result.status
        if hasattr(status_val, "value"):
            status_val = status_val.value
        assert status_val == "completed"

    @pytest.mark.parametrize("wf_key", WORKFLOW_KEYS)
    def test_result_phases_completed(self, wf_key: str):
        """phases_completed must match the number of defined phases."""
        mod = _load_workflow(wf_key)
        cls = getattr(mod, WORKFLOW_CLASSES[wf_key])
        instance = cls()
        expected_count = len(instance.get_phases())
        result = _run(instance.execute())
        assert result.phases_completed == expected_count
