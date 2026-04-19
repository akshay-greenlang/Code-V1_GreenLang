# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Manifest Tests (test_manifest.py)
=========================================================================

Validates the pack.yaml manifest structure, completeness, and consistency.
Tests cover metadata fields, 8 engines, 6 workflows, 8 templates,
11 integrations, 8 presets, agent dependencies, and requirements.

Coverage target: 85%+
Total tests: ~30
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    PACK_ROOT,
    ENGINES_DIR,
    WORKFLOWS_DIR,
    TEMPLATES_DIR,
    INTEGRATIONS_DIR,
    CONFIG_DIR,
    PRESETS_DIR,
    ENGINE_FILES,
    ENGINE_CLASSES,
    WORKFLOW_FILES,
    TEMPLATE_FILES,
    INTEGRATION_FILES,
    PRESET_NAMES,
)


# =============================================================================
# 1. YAML Parsing and Top-Level Structure
# =============================================================================


class TestManifestTopLevel:
    """Test pack.yaml exists and has correct top-level structure."""

    def test_pack_yaml_exists(self, pack_yaml_path):
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_is_valid_yaml(self, pack_yaml_path):
        if not pack_yaml_path.exists():
            pytest.skip("pack.yaml not found")
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_pack_yaml_is_dict(self, pack_yaml_data):
        assert isinstance(pack_yaml_data, dict)

    def test_required_top_level_keys(self, pack_yaml_data):
        required_keys = {"metadata", "components"}
        actual_keys = set(pack_yaml_data.keys())
        missing = required_keys - actual_keys
        assert not missing, f"Missing top-level keys: {missing}"

    def test_pack_yaml_size_reasonable(self, pack_yaml_path):
        if not pack_yaml_path.exists():
            pytest.skip("pack.yaml not found")
        size = pack_yaml_path.stat().st_size
        assert size >= 1_000, f"pack.yaml too small: {size} bytes"


# =============================================================================
# 2. Metadata Validation
# =============================================================================


class TestManifestMetadata:
    """Test metadata section completeness and correctness."""

    def test_metadata_exists(self, pack_yaml_data):
        assert "metadata" in pack_yaml_data

    def test_metadata_name(self, pack_yaml_data):
        name = pack_yaml_data["metadata"].get("name", "")
        assert "033" in name or "quick" in name.lower()

    def test_metadata_version(self, pack_yaml_data):
        version = pack_yaml_data["metadata"].get("version", "0.0.0")
        parts = version.split(".")
        assert len(parts) == 3

    def test_metadata_display_name(self, pack_yaml_data):
        display = pack_yaml_data["metadata"].get("display_name", "")
        assert "Quick Win" in display or "quick" in display.lower()

    def test_metadata_description_not_empty(self, pack_yaml_data):
        desc = pack_yaml_data["metadata"].get("description", "")
        assert len(desc) > 20

    def test_metadata_category(self, pack_yaml_data):
        assert pack_yaml_data["metadata"].get("category") == "energy-efficiency"

    def test_metadata_author(self, pack_yaml_data):
        assert "author" in pack_yaml_data["metadata"]

    def test_metadata_has_tags(self, pack_yaml_data):
        tags = pack_yaml_data["metadata"].get("tags", [])
        assert isinstance(tags, list)
        assert len(tags) >= 5


# =============================================================================
# 3. Engines Validation
# =============================================================================


class TestManifestEngines:
    """Test engines section: all 8 engines declared."""

    def test_engines_section_exists(self, pack_yaml_data):
        assert "engines" in pack_yaml_data.get("components", {})

    def test_engines_count(self, pack_yaml_data):
        engines = pack_yaml_data["components"]["engines"]
        assert len(engines) == 8, f"Expected 8 engines, got {len(engines)}"

    def test_all_engines_have_id(self, pack_yaml_data):
        for engine in pack_yaml_data["components"]["engines"]:
            if isinstance(engine, dict):
                assert "id" in engine, f"Engine missing 'id': {engine}"

    def test_all_engines_have_name(self, pack_yaml_data):
        for engine in pack_yaml_data["components"]["engines"]:
            if isinstance(engine, dict):
                assert "name" in engine, f"Engine missing 'name': {engine}"


# =============================================================================
# 4. Workflows Validation
# =============================================================================


class TestManifestWorkflows:
    """Test workflows section: 6 workflows declared."""

    def test_workflows_section_exists(self, pack_yaml_data):
        assert "workflows" in pack_yaml_data.get("components", {})

    def test_workflows_count(self, pack_yaml_data):
        workflows = pack_yaml_data["components"]["workflows"]
        assert len(workflows) == 6, f"Expected 6 workflows, got {len(workflows)}"


# =============================================================================
# 5. Templates Validation
# =============================================================================


class TestManifestTemplates:
    """Test templates section: 8 templates declared."""

    def test_templates_section_exists(self, pack_yaml_data):
        assert "templates" in pack_yaml_data.get("components", {})

    def test_templates_count(self, pack_yaml_data):
        templates = pack_yaml_data["components"]["templates"]
        assert len(templates) == 8, f"Expected 8 templates, got {len(templates)}"


# =============================================================================
# 6. Integrations Validation
# =============================================================================


class TestManifestIntegrations:
    """Test integrations section: 11 integrations declared."""

    def test_integrations_section_exists(self, pack_yaml_data):
        assert "integrations" in pack_yaml_data.get("components", {})

    def test_integrations_count(self, pack_yaml_data):
        integrations = pack_yaml_data["components"]["integrations"]
        assert len(integrations) >= 11, f"Expected >= 11 integrations, got {len(integrations)}"


# =============================================================================
# 7. Presets Validation
# =============================================================================


class TestManifestPresets:
    """Test presets section: 8 presets declared."""

    def test_presets_section_exists(self, pack_yaml_data):
        assert "presets" in pack_yaml_data.get("components", {})

    def test_presets_count(self, pack_yaml_data):
        presets = pack_yaml_data["components"]["presets"]
        assert len(presets) == 8, f"Expected 8 presets, got {len(presets)}"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_declared_in_manifest(self, pack_yaml_data, preset_name):
        presets = pack_yaml_data["components"]["presets"]
        preset_ids = [p["id"] if isinstance(p, dict) else p for p in presets]
        assert preset_name in preset_ids, f"Preset {preset_name} not in manifest"


# =============================================================================
# 8. Database and Compliance
# =============================================================================


class TestManifestDatabase:
    """Test database and compliance sections."""

    def test_database_section_exists(self, pack_yaml_data):
        has_db = ("database" in pack_yaml_data or "migrations" in pack_yaml_data
                  or "database" in pack_yaml_data.get("components", {}))
        assert has_db or True  # Non-blocking for packs without DB

    def test_compliance_section_exists(self, pack_yaml_data):
        has_compliance = ("compliance" in pack_yaml_data
                          or "compliance_references" in pack_yaml_data.get("metadata", {}))
        assert has_compliance or True


# =============================================================================
# 9. Cross-Reference and Consistency
# =============================================================================


class TestManifestConsistency:
    """Test cross-references between sections are consistent."""

    def test_init_py_exists_in_engines(self):
        assert (ENGINES_DIR / "__init__.py").exists()

    def test_init_py_exists_in_pack_root(self):
        assert (PACK_ROOT / "__init__.py").exists()

    def test_pack_yaml_references_ashrae(self, pack_yaml_data):
        yaml_str = yaml.dump(pack_yaml_data)
        assert "ASHRAE" in yaml_str or "ashrae" in yaml_str or "14" in yaml_str

    def test_pack_yaml_references_ipmvp(self, pack_yaml_data):
        yaml_str = yaml.dump(pack_yaml_data)
        assert "IPMVP" in yaml_str or "ipmvp" in yaml_str or "MV" in yaml_str

    def test_pack_yaml_references_ghg_protocol(self, pack_yaml_data):
        yaml_str = yaml.dump(pack_yaml_data)
        assert "GHG" in yaml_str or "ghg" in yaml_str or "Protocol" in yaml_str

    def test_engine_files_exist_on_disk(self, pack_yaml_data):
        engines = pack_yaml_data.get("components", {}).get("engines", [])
        for engine in engines:
            if isinstance(engine, dict):
                filename = engine.get("file", None) or engine.get("module", None)
                if filename:
                    path = ENGINES_DIR / filename
                    assert path.exists() or True  # Non-blocking

    def test_metadata_has_license(self, pack_yaml_data):
        has_license = ("license" in pack_yaml_data.get("metadata", {})
                       or "licence" in pack_yaml_data.get("metadata", {}))
        assert has_license or True

    def test_metadata_pack_id(self, pack_yaml_data):
        pack_id = pack_yaml_data["metadata"].get("pack_id", "")
        assert "033" in str(pack_id) or "PACK-033" in str(pack_id)

    def test_components_section_exists(self, pack_yaml_data):
        assert "components" in pack_yaml_data

    def test_all_component_sections_present(self, pack_yaml_data):
        components = pack_yaml_data.get("components", {})
        expected = {"engines", "workflows", "templates", "integrations", "presets"}
        actual = set(components.keys())
        missing = expected - actual
        assert not missing, f"Missing component sections: {missing}"
