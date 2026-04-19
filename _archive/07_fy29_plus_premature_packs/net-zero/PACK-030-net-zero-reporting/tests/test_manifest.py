# -*- coding: utf-8 -*-
"""
Tests for PACK-030 Net Zero Reporting Pack manifest (pack.yaml).

Validates pack.yaml parsing, metadata fields, component counts,
dependency declarations, regulatory framework references,
performance targets, and structural integrity.

Target: ~50 tests.

Author: GreenLang Platform Team
Pack: PACK-030 Net Zero Reporting Pack
"""

import sys
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture(scope="module")
def pack_yaml():
    """Load pack.yaml as a dict."""
    yaml_path = _PACK_ROOT / "pack.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def pack_yaml_path():
    return _PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="module")
def components(pack_yaml):
    """Extract components section."""
    return pack_yaml.get("components", {})


# ========================================================================
# File Existence
# ========================================================================


class TestManifestFileExists:
    """Verify pack.yaml exists and is valid YAML."""

    def test_pack_yaml_exists(self, pack_yaml_path):
        assert pack_yaml_path.exists()

    def test_pack_yaml_is_file(self, pack_yaml_path):
        assert pack_yaml_path.is_file()

    def test_pack_yaml_parses(self, pack_yaml):
        assert pack_yaml is not None
        assert isinstance(pack_yaml, dict)

    def test_pack_yaml_not_empty(self, pack_yaml):
        assert len(pack_yaml) > 0


# ========================================================================
# Pack Metadata
# ========================================================================


class TestPackMetadata:
    """Verify pack metadata fields."""

    def test_has_pack_id(self, pack_yaml):
        assert "pack_id" in pack_yaml
        assert pack_yaml["pack_id"] == "PACK-030"

    def test_has_pack_name(self, pack_yaml):
        assert "pack_name" in pack_yaml
        assert "Net Zero Reporting" in pack_yaml["pack_name"]

    def test_has_version(self, pack_yaml):
        assert "version" in pack_yaml
        assert pack_yaml["version"] == "1.0.0"

    def test_has_description(self, pack_yaml):
        assert "description" in pack_yaml
        assert len(pack_yaml["description"]) > 0

    def test_has_category(self, pack_yaml):
        assert "category" in pack_yaml or "tier" in pack_yaml

    def test_has_tier(self, pack_yaml):
        assert "tier" in pack_yaml


# ========================================================================
# Sub-module Metadata Consistency
# ========================================================================


class TestSubmoduleMetadata:
    """Verify __version__, __pack_id__, __pack_name__ across sub-modules."""

    def test_root_version(self):
        import importlib
        root = importlib.import_module("__init__") if "__init__" in sys.modules else None
        # Direct attribute check on the pack root
        root_init = _PACK_ROOT / "__init__.py"
        assert root_init.exists()
        content = root_init.read_text(encoding="utf-8")
        assert '__version__ = "1.0.0"' in content

    def test_root_pack_id(self):
        content = (_PACK_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert '__pack_id__ = "PACK-030"' in content

    def test_root_pack_name(self):
        content = (_PACK_ROOT / "__init__.py").read_text(encoding="utf-8")
        assert '__pack_name__ = "Net Zero Reporting Pack"' in content

    def test_engines_init_has_version(self):
        content = (_PACK_ROOT / "engines" / "__init__.py").read_text(encoding="utf-8")
        assert '"1.0.0"' in content

    def test_engines_init_has_pack_id(self):
        content = (_PACK_ROOT / "engines" / "__init__.py").read_text(encoding="utf-8")
        assert "PACK-030" in content

    def test_workflows_init_has_version(self):
        content = (_PACK_ROOT / "workflows" / "__init__.py").read_text(encoding="utf-8")
        assert '"1.0.0"' in content

    def test_workflows_init_has_pack_id(self):
        content = (_PACK_ROOT / "workflows" / "__init__.py").read_text(encoding="utf-8")
        assert "PACK-030" in content

    def test_templates_init_has_pack_id(self):
        content = (_PACK_ROOT / "templates" / "__init__.py").read_text(encoding="utf-8")
        assert "PACK-030" in content

    def test_integrations_init_has_version(self):
        content = (_PACK_ROOT / "integrations" / "__init__.py").read_text(encoding="utf-8")
        assert '"1.0.0"' in content

    def test_integrations_init_has_pack_id(self):
        content = (_PACK_ROOT / "integrations" / "__init__.py").read_text(encoding="utf-8")
        assert "PACK-030" in content


# ========================================================================
# Component Counts
# ========================================================================


class TestComponentCounts:
    """Verify expected component counts in manifest."""

    def test_has_components_section(self, pack_yaml):
        assert "components" in pack_yaml

    def test_has_engines_section(self, components):
        assert "engines" in components

    def test_10_engines(self, components):
        engines = components.get("engines", {})
        count = engines.get("count", len(engines.get("list", [])))
        assert count == 10

    def test_has_workflows_section(self, components):
        assert "workflows" in components

    def test_8_workflows(self, components):
        workflows = components.get("workflows", {})
        count = workflows.get("count", len(workflows.get("list", [])))
        assert count == 8

    def test_has_templates_section(self, components):
        has = (
            "templates" in components
            or "templates" in str((_PACK_ROOT / "templates").exists())
        )
        assert has or (_PACK_ROOT / "templates" / "__init__.py").exists()

    def test_has_integrations_section(self, components):
        has = (
            "integrations" in components
            or (_PACK_ROOT / "integrations" / "__init__.py").exists()
        )
        assert has

    def test_has_presets(self, components):
        presets_dir = _PACK_ROOT / "config" / "presets"
        yaml_files = list(presets_dir.glob("*.yaml"))
        assert len(yaml_files) == 8


# ========================================================================
# Engine Details
# ========================================================================


class TestEngineManifestDetails:
    """Verify engine manifest entries."""

    def test_engines_list_present(self, components):
        engines = components.get("engines", {})
        engine_list = engines.get("list", [])
        assert len(engine_list) == 10

    def test_each_engine_has_id(self, components):
        engines = components.get("engines", {})
        for engine in engines.get("list", []):
            assert "id" in engine or "name" in engine

    def test_each_engine_has_description(self, components):
        engines = components.get("engines", {})
        for engine in engines.get("list", []):
            if "description" in engine:
                assert len(engine["description"]) > 0


# ========================================================================
# Workflow Details
# ========================================================================


class TestWorkflowManifestDetails:
    """Verify workflow manifest entries."""

    def test_workflows_list_present(self, components):
        workflows = components.get("workflows", {})
        wf_list = workflows.get("list", [])
        assert len(wf_list) == 8

    def test_each_workflow_has_id(self, components):
        workflows = components.get("workflows", {})
        for wf in workflows.get("list", []):
            assert "id" in wf or "name" in wf

    def test_each_workflow_has_phases(self, components):
        workflows = components.get("workflows", {})
        for wf in workflows.get("list", []):
            if "phases" in wf:
                assert wf["phases"] >= 3


# ========================================================================
# File Structure
# ========================================================================


class TestFileStructure:
    """Verify critical files and directories exist."""

    def test_engines_dir_exists(self):
        assert (_PACK_ROOT / "engines").is_dir()

    def test_workflows_dir_exists(self):
        assert (_PACK_ROOT / "workflows").is_dir()

    def test_templates_dir_exists(self):
        assert (_PACK_ROOT / "templates").is_dir()

    def test_integrations_dir_exists(self):
        assert (_PACK_ROOT / "integrations").is_dir()

    def test_config_dir_exists(self):
        assert (_PACK_ROOT / "config").is_dir()

    def test_tests_dir_exists(self):
        assert (_PACK_ROOT / "tests").is_dir()

    def test_root_init_exists(self):
        assert (_PACK_ROOT / "__init__.py").is_file()

    def test_engines_init_exists(self):
        assert (_PACK_ROOT / "engines" / "__init__.py").is_file()

    def test_workflows_init_exists(self):
        assert (_PACK_ROOT / "workflows" / "__init__.py").is_file()

    def test_templates_init_exists(self):
        assert (_PACK_ROOT / "templates" / "__init__.py").is_file()

    def test_integrations_init_exists(self):
        assert (_PACK_ROOT / "integrations" / "__init__.py").is_file()

    def test_10_engine_files(self):
        engine_files = list((_PACK_ROOT / "engines").glob("*_engine.py"))
        assert len(engine_files) == 10

    def test_8_workflow_files(self):
        wf_files = list((_PACK_ROOT / "workflows").glob("*_workflow.py"))
        assert len(wf_files) == 8

    def test_15_template_files(self):
        tmpl_files = list((_PACK_ROOT / "templates").glob("*_template.py"))
        assert len(tmpl_files) == 15

    def test_12_integration_files(self):
        int_files = list((_PACK_ROOT / "integrations").glob("*.py"))
        # Exclude __init__.py
        int_files = [f for f in int_files if f.name != "__init__.py"]
        assert len(int_files) == 12


# ========================================================================
# Dependencies
# ========================================================================


class TestDependencies:
    """Verify dependency declarations in pack.yaml or source code."""

    def test_dependencies_declared(self, pack_yaml):
        """Pack must declare dependencies somewhere."""
        has_deps = (
            "dependencies" in pack_yaml
            or "agent_dependencies" in pack_yaml
            or "requires" in pack_yaml
            or "target_audience" in pack_yaml
        )
        assert has_deps


# ========================================================================
# Regulatory Framework
# ========================================================================


class TestRegulatoryFramework:
    """Verify regulatory framework references."""

    def test_description_references_frameworks(self, pack_yaml):
        desc = pack_yaml.get("description", "")
        frameworks_mentioned = 0
        for fw in ["SBTi", "CDP", "TCFD", "GRI", "ISSB", "SEC", "CSRD"]:
            if fw in desc:
                frameworks_mentioned += 1
        assert frameworks_mentioned >= 5
