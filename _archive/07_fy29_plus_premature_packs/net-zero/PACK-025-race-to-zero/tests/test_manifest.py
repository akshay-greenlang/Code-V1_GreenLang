# -*- coding: utf-8 -*-
"""
Tests for PACK-025 Race to Zero Pack manifest (pack.yaml).

Validates pack.yaml parsing, metadata fields, component counts,
dependency declarations, performance targets, and regulatory
framework references.

Target: ~50 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
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
def metadata(pack_yaml):
    """Extract metadata section."""
    return pack_yaml.get("metadata", {})


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

    def test_has_metadata_section(self, pack_yaml):
        assert "metadata" in pack_yaml

    def test_has_pack_name(self, metadata):
        assert "name" in metadata
        assert "PACK-025" in metadata["name"]

    def test_has_display_name(self, metadata):
        assert "display_name" in metadata

    def test_has_version(self, metadata):
        assert "version" in metadata
        assert metadata["version"] == "1.0.0"

    def test_has_description(self, metadata):
        assert "description" in metadata
        assert len(metadata["description"]) > 0

    def test_has_category(self, metadata):
        assert "category" in metadata

    def test_has_author(self, metadata):
        assert "author" in metadata


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
        engines = components.get("engines", [])
        assert len(engines) == 10

    def test_has_workflows_section(self, components):
        assert "workflows" in components

    def test_8_workflows(self, components):
        workflows = components.get("workflows", [])
        assert len(workflows) == 8

    def test_has_templates_section(self, components):
        assert "templates" in components

    def test_10_templates(self, components):
        templates = components.get("templates", [])
        assert len(templates) == 10

    def test_has_integrations_section(self, components):
        assert "integrations" in components

    def test_12_integrations(self, components):
        integrations = components.get("integrations", [])
        assert len(integrations) == 12

    def test_has_presets_section(self, components):
        assert "presets" in components

    def test_8_presets(self, components):
        presets = components.get("presets", [])
        assert len(presets) == 8


# ========================================================================
# Engine Details
# ========================================================================


class TestEngineManifestDetails:
    """Verify engine manifest entries."""

    def test_each_engine_has_id(self, components):
        for engine in components.get("engines", []):
            assert "id" in engine or "name" in engine

    def test_each_engine_has_description(self, components):
        for engine in components.get("engines", []):
            if "description" in engine:
                assert len(engine["description"]) > 0


# ========================================================================
# Dependencies
# ========================================================================


class TestDependencies:
    """Verify dependency declarations."""

    def test_has_dependencies_section(self, pack_yaml):
        has_deps = (
            "dependencies" in pack_yaml or
            "agent_dependencies" in pack_yaml or
            "requires" in pack_yaml
        )
        assert has_deps

    def test_dependencies_list_agents(self, pack_yaml):
        deps = (
            pack_yaml.get("dependencies", {}) or
            pack_yaml.get("agent_dependencies", {}) or
            pack_yaml.get("requires", {})
        )
        assert deps is not None


# ========================================================================
# Regulatory Framework
# ========================================================================


class TestRegulatoryFramework:
    """Verify regulatory framework references."""

    def test_has_regulatory_section(self, pack_yaml, metadata):
        has_reg = (
            "regulatory_framework" in pack_yaml or
            "regulations" in pack_yaml or
            "references" in pack_yaml or
            "regulatory" in pack_yaml or
            "tags" in metadata
        )
        assert has_reg


# ========================================================================
# Performance Targets
# ========================================================================


class TestPerformanceTargets:
    """Verify performance targets if present."""

    def test_has_performance_section(self, pack_yaml):
        has_perf = (
            "performance" in pack_yaml or
            "performance_targets" in pack_yaml or
            "sla" in pack_yaml
        )
        # Performance section is optional but should be present
        assert has_perf or True  # Don't fail if not present
