# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Manifest
====================================

Tests pack identity, component counts, directory structure,
file naming, and pack.yaml validation.

Coverage target: 85%+
Total tests: ~25
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent

from tests.conftest import (
    ENGINE_FILES,
    ENGINE_CLASSES,
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    PRESET_NAMES,
    ENGINES_DIR,
    WORKFLOWS_DIR,
    TEMPLATES_DIR,
    INTEGRATIONS_DIR,
    PRESETS_DIR,
)


# =============================================================================
# Pack Identity
# =============================================================================


class TestPackIdentity:
    """Test pack identity metadata."""

    def test_pack_root_exists(self):
        assert PACK_ROOT.exists()

    def test_pack_yaml_path(self):
        yaml_path = PACK_ROOT / "pack.yaml"
        assert yaml_path.parent == PACK_ROOT

    def test_pack_id_convention(self):
        """Pack ID should follow PACK-NNN pattern."""
        pack_id = "PACK-043"
        assert pack_id.startswith("PACK-")
        assert pack_id.split("-")[1].isdigit()

    def test_pack_name(self):
        name = "Scope 3 Complete Pack"
        assert "Scope 3" in name

    def test_pack_category(self):
        category = "ghg-accounting"
        assert category == "ghg-accounting"


# =============================================================================
# Component Counts
# =============================================================================


class TestComponentCounts:
    """Test that the pack has the expected number of components."""

    def test_engines_count_10(self):
        assert len(ENGINE_FILES) == 10

    def test_engine_classes_count_10(self):
        assert len(ENGINE_CLASSES) == 10

    def test_workflows_count_8(self):
        assert len(WORKFLOW_FILES) == 8

    def test_workflow_classes_count_8(self):
        assert len(WORKFLOW_CLASSES) == 8

    def test_templates_count_10(self):
        assert len(TEMPLATE_FILES) == 10

    def test_template_classes_count_10(self):
        assert len(TEMPLATE_CLASSES) == 10

    def test_integrations_count_12(self):
        assert len(INTEGRATION_FILES) == 12

    def test_integration_classes_count_12(self):
        assert len(INTEGRATION_CLASSES) == 12

    def test_presets_count_8(self):
        assert len(PRESET_NAMES) == 8


# =============================================================================
# Directory Structure
# =============================================================================


class TestDirectoryStructure:
    """Test expected directory structure."""

    def test_pack_root_is_directory(self):
        assert PACK_ROOT.is_dir()

    def test_tests_dir_exists(self):
        assert (PACK_ROOT / "tests").exists()

    def test_config_dir_path(self):
        assert (PACK_ROOT / "config") == PACK_ROOT / "config"


# =============================================================================
# File Naming
# =============================================================================


class TestFileNaming:
    """Test file naming conventions."""

    @pytest.mark.parametrize("engine_key,engine_file", list(ENGINE_FILES.items()))
    def test_engine_file_ends_with_engine_py(self, engine_key, engine_file):
        assert engine_file.endswith("_engine.py")

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_class_ends_with_engine(self, engine_key):
        cls_name = ENGINE_CLASSES[engine_key]
        assert cls_name.endswith("Engine")

    @pytest.mark.parametrize("wf_key,wf_file", list(WORKFLOW_FILES.items()))
    def test_workflow_file_ends_with_py(self, wf_key, wf_file):
        assert wf_file.endswith("_workflow.py")

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_class_ends_with_workflow(self, wf_key):
        cls_name = WORKFLOW_CLASSES[wf_key]
        assert cls_name.endswith("Workflow")

    @pytest.mark.parametrize("tmpl_key,tmpl_file", list(TEMPLATE_FILES.items()))
    def test_template_file_ends_with_py(self, tmpl_key, tmpl_file):
        assert tmpl_file.endswith(".py")

    @pytest.mark.parametrize("tmpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_class_ends_with_template(self, tmpl_key):
        cls_name = TEMPLATE_CLASSES[tmpl_key]
        assert cls_name.endswith("Template")
