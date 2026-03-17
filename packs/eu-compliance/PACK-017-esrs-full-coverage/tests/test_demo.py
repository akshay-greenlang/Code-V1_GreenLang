# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Demo / Smoke Tests
=======================================================

Smoke and importability tests that verify:
  - All 11 engine files exist on disk
  - All engine modules can be dynamically loaded
  - Each engine module exports its primary class
  - All 12 workflow files exist on disk
  - All 12 template files exist on disk
  - All 10 integration files exist on disk

These tests run without heavy dependencies and serve as a fast gate
to detect missing or broken source files before deeper test suites.

Target: 56+ parametrized test cases.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage
Date:    March 2026
"""

from pathlib import Path

import pytest

from .conftest import (
    ENGINES_DIR,
    WORKFLOWS_DIR,
    TEMPLATES_DIR,
    INTEGRATIONS_DIR,
    ENGINE_FILES,
    ENGINE_CLASSES,
    WORKFLOW_FILES,
    TEMPLATE_FILES,
    INTEGRATION_FILES,
    _load_engine,
)


# ===========================================================================
# Engine File Existence
# ===========================================================================


class TestEngineFilesExist:
    """Verify that all engine source files exist on disk."""

    @pytest.mark.parametrize("engine_key,file_name", list(ENGINE_FILES.items()))
    def test_engine_file_exists(self, engine_key, file_name):
        """Engine file {file_name} exists in the engines directory."""
        path = ENGINES_DIR / file_name
        assert path.exists(), (
            f"Engine file missing: {path} (key={engine_key})"
        )

    @pytest.mark.parametrize("engine_key,file_name", list(ENGINE_FILES.items()))
    def test_engine_file_non_empty(self, engine_key, file_name):
        """Engine file {file_name} is non-empty (>0 bytes)."""
        path = ENGINES_DIR / file_name
        if not path.exists():
            pytest.skip(f"Engine file not found: {path}")
        assert path.stat().st_size > 0, (
            f"Engine file is empty (0 bytes): {path} (key={engine_key})"
        )


# ===========================================================================
# Engine Module Loading
# ===========================================================================


class TestEngineModulesLoad:
    """Verify that all engine modules can be dynamically loaded."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_module_loads(self, engine_key):
        """Engine module for {engine_key} can be loaded via importlib."""
        path = ENGINES_DIR / ENGINE_FILES[engine_key]
        if not path.exists():
            pytest.skip(f"Engine file not found: {path}")
        if path.stat().st_size == 0:
            pytest.skip(f"Engine file is empty: {path}")
        module = _load_engine(engine_key)
        assert module is not None


# ===========================================================================
# Engine Class Exports
# ===========================================================================


class TestEngineClassExports:
    """Verify that each engine module exports its primary class."""

    @pytest.mark.parametrize(
        "engine_key,class_name",
        list(ENGINE_CLASSES.items()),
    )
    def test_engine_exports_class(self, engine_key, class_name):
        """Engine {engine_key} exports class {class_name}."""
        path = ENGINES_DIR / ENGINE_FILES[engine_key]
        if not path.exists():
            pytest.skip(f"Engine file not found: {path}")
        if path.stat().st_size == 0:
            pytest.skip(f"Engine file is empty: {path}")
        module = _load_engine(engine_key)
        assert hasattr(module, class_name), (
            f"Module for {engine_key} does not export class '{class_name}'. "
            f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}"
        )


# ===========================================================================
# Workflow File Existence
# ===========================================================================


class TestWorkflowFilesExist:
    """Verify that all workflow source files exist on disk."""

    @pytest.mark.parametrize("workflow_key,file_name", list(WORKFLOW_FILES.items()))
    def test_workflow_file_exists(self, workflow_key, file_name):
        """Workflow file {file_name} exists in the workflows directory."""
        path = WORKFLOWS_DIR / file_name
        assert path.exists(), (
            f"Workflow file missing: {path} (key={workflow_key})"
        )


# ===========================================================================
# Template File Existence
# ===========================================================================


class TestTemplateFilesExist:
    """Verify that all template source files exist on disk."""

    @pytest.mark.parametrize("template_key,file_name", list(TEMPLATE_FILES.items()))
    def test_template_file_exists(self, template_key, file_name):
        """Template file {file_name} exists in the templates directory."""
        path = TEMPLATES_DIR / file_name
        assert path.exists(), (
            f"Template file missing: {path} (key={template_key})"
        )


# ===========================================================================
# Integration File Existence
# ===========================================================================


class TestIntegrationFilesExist:
    """Verify that all integration source files exist on disk."""

    @pytest.mark.parametrize("integration_key,file_name", list(INTEGRATION_FILES.items()))
    def test_integration_file_exists(self, integration_key, file_name):
        """Integration file {file_name} exists in the integrations directory."""
        path = INTEGRATIONS_DIR / file_name
        assert path.exists(), (
            f"Integration file missing: {path} (key={integration_key})"
        )
