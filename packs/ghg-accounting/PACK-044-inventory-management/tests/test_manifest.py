# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Pack Manifest Tests
============================================

Tests the pack.yaml manifest: required fields, engine/workflow/template
references, integration listings, preset configuration, and structural
completeness.

Target: 20+ test cases.
"""

from pathlib import Path

import pytest

from conftest import (
    PACK_ROOT,
    ENGINES_DIR,
    WORKFLOWS_DIR,
    TEMPLATES_DIR,
    INTEGRATIONS_DIR,
    CONFIG_DIR,
    PRESETS_DIR,
    ENGINE_FILES,
    WORKFLOW_FILES,
    TEMPLATE_FILES,
    INTEGRATION_FILES,
    PRESET_NAMES,
)


# ===================================================================
# Directory Structure Tests
# ===================================================================


class TestDirectoryStructure:
    """Tests for pack directory structure."""

    def test_pack_root_exists(self):
        assert PACK_ROOT.exists()

    def test_engines_dir_exists(self):
        assert ENGINES_DIR.exists()

    def test_workflows_dir_exists(self):
        assert WORKFLOWS_DIR.exists()

    def test_templates_dir_exists(self):
        assert TEMPLATES_DIR.exists()

    def test_integrations_dir_exists(self):
        assert INTEGRATIONS_DIR.exists()

    def test_config_dir_exists(self):
        assert CONFIG_DIR.exists()

    def test_presets_dir_exists(self):
        assert PRESETS_DIR.exists()


# ===================================================================
# Pack.yaml Tests
# ===================================================================


class TestPackYaml:
    """Tests for pack.yaml manifest content."""

    def test_pack_yaml_exists(self, pack_yaml_path):
        assert pack_yaml_path.exists()

    def test_pack_yaml_has_name(self, pack_yaml_data):
        # May be at top level or nested under 'metadata'
        meta = pack_yaml_data.get("metadata", pack_yaml_data)
        assert "name" in meta or "pack_name" in meta or "display_name" in meta

    def test_pack_yaml_has_version(self, pack_yaml_data):
        meta = pack_yaml_data.get("metadata", pack_yaml_data)
        assert "version" in meta

    def test_pack_yaml_has_description(self, pack_yaml_data):
        meta = pack_yaml_data.get("metadata", pack_yaml_data)
        assert "description" in meta

    def test_pack_yaml_has_engines(self, pack_yaml_data):
        assert "engines" in pack_yaml_data
        assert len(pack_yaml_data["engines"]) == 10

    def test_pack_yaml_has_workflows(self, pack_yaml_data):
        assert "workflows" in pack_yaml_data
        assert len(pack_yaml_data["workflows"]) == 8

    def test_pack_yaml_has_templates(self, pack_yaml_data):
        assert "templates" in pack_yaml_data
        assert len(pack_yaml_data["templates"]) == 10

    def test_pack_yaml_has_integrations(self, pack_yaml_data):
        assert "integrations" in pack_yaml_data
        assert len(pack_yaml_data["integrations"]) == 12

    def test_pack_yaml_has_presets(self, pack_yaml_data):
        key = "presets" if "presets" in pack_yaml_data else "config_presets"
        assert key in pack_yaml_data


# ===================================================================
# Engine File Completeness Tests
# ===================================================================


class TestEngineFileCompleteness:
    """Tests that all engine source files exist."""

    @pytest.mark.parametrize("key,filename", list(ENGINE_FILES.items()))
    def test_engine_file_exists(self, key, filename):
        path = ENGINES_DIR / filename
        assert path.exists(), f"Engine file missing: {path}"

    def test_engines_init_exists(self):
        assert (ENGINES_DIR / "__init__.py").exists()


# ===================================================================
# Template File Tests
# ===================================================================


class TestTemplateFiles:
    """Tests that all template source files exist."""

    @pytest.mark.parametrize("key,filename", list(TEMPLATE_FILES.items()))
    def test_template_file_exists(self, key, filename):
        path = TEMPLATES_DIR / filename
        assert path.exists(), f"Template file missing: {path}"

    def test_templates_init_exists(self):
        assert (TEMPLATES_DIR / "__init__.py").exists()


# ===================================================================
# Preset File Tests
# ===================================================================


class TestPresetFiles:
    """Tests that all preset YAML files exist."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_yaml_exists(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file missing: {path}"


# ===================================================================
# Pack Config Tests
# ===================================================================


class TestPackConfigFile:
    """Tests for pack_config.py presence."""

    def test_pack_config_exists(self):
        assert (CONFIG_DIR / "pack_config.py").exists()

    def test_config_init_exists(self):
        assert (CONFIG_DIR / "__init__.py").exists()
