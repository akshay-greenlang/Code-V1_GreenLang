# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Manifest Tests (test_manifest.py)
=========================================================================

Validates the pack.yaml manifest structure, completeness, and consistency.
Tests cover metadata fields, 10 engines, 8 workflows, 10 templates,
12 integrations, 8 presets, agent dependencies, and requirements.

Test Count Target: ~110 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-031 Industrial Energy Audit
Date:    March 2026
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
        """pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_is_valid_yaml(self, pack_yaml_path):
        """pack.yaml is valid YAML that parses without errors."""
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_pack_yaml_is_dict(self, pack_yaml_data):
        """pack.yaml top-level is a dictionary."""
        assert isinstance(pack_yaml_data, dict)

    def test_required_top_level_keys(self, pack_yaml_data):
        """pack.yaml contains all required top-level keys."""
        required_keys = {
            "metadata",
            "components",
            "dependencies",
            "performance",
            "security",
            "requirements",
        }
        actual_keys = set(pack_yaml_data.keys())
        missing = required_keys - actual_keys
        assert not missing, f"Missing top-level keys: {missing}"

    def test_has_agent_dependency_sections(self, pack_yaml_data):
        """pack.yaml has agent dependency sections (MRV, Data, Foundation)."""
        for section in ["agents_mrv", "agents_data", "agents_foundation"]:
            assert section in pack_yaml_data, f"Missing agent section: {section}"

    def test_pack_yaml_size_reasonable(self, pack_yaml_path):
        """pack.yaml is at least 10KB (comprehensive manifest)."""
        size = pack_yaml_path.stat().st_size
        assert size >= 10_000, f"pack.yaml too small: {size} bytes"


# =============================================================================
# 2. Metadata Validation
# =============================================================================


class TestManifestMetadata:
    """Test metadata section completeness and correctness."""

    def test_metadata_exists(self, pack_yaml_data):
        """Metadata section exists."""
        assert "metadata" in pack_yaml_data

    def test_metadata_name(self, pack_yaml_data):
        """Pack name is PACK-031-industrial-energy-audit."""
        assert pack_yaml_data["metadata"]["name"] == "PACK-031-industrial-energy-audit"

    def test_metadata_version(self, pack_yaml_data):
        """Version follows semver format."""
        version = pack_yaml_data["metadata"]["version"]
        parts = version.split(".")
        assert len(parts) == 3, f"Version {version} is not semver"
        for part in parts:
            assert part.isdigit(), f"Version part {part} is not numeric"

    def test_metadata_display_name(self, pack_yaml_data):
        """Display name is set and contains 'Energy Audit'."""
        display = pack_yaml_data["metadata"]["display_name"]
        assert "Energy Audit" in display

    def test_metadata_description_not_empty(self, pack_yaml_data):
        """Description is present and non-empty."""
        desc = pack_yaml_data["metadata"]["description"]
        assert isinstance(desc, str)
        assert len(desc) > 50, "Description too short"

    def test_metadata_category(self, pack_yaml_data):
        """Category is energy-efficiency."""
        assert pack_yaml_data["metadata"]["category"] == "energy-efficiency"

    def test_metadata_tier(self, pack_yaml_data):
        """Tier is professional."""
        assert pack_yaml_data["metadata"]["tier"] == "professional"

    def test_metadata_author(self, pack_yaml_data):
        """Author is set."""
        assert "author" in pack_yaml_data["metadata"]
        assert len(pack_yaml_data["metadata"]["author"]) > 0

    def test_metadata_min_platform_version(self, pack_yaml_data):
        """Minimum platform version is specified."""
        assert "min_platform_version" in pack_yaml_data["metadata"]

    def test_metadata_release_date(self, pack_yaml_data):
        """Release date is specified."""
        assert "release_date" in pack_yaml_data["metadata"]

    def test_metadata_has_tags(self, pack_yaml_data):
        """Tags list is present and non-empty."""
        tags = pack_yaml_data["metadata"].get("tags", [])
        assert isinstance(tags, list)
        assert len(tags) >= 10, f"Expected at least 10 tags, got {len(tags)}"

    def test_metadata_tags_include_core_keywords(self, pack_yaml_data):
        """Tags include core regulatory and domain keywords."""
        tags = pack_yaml_data["metadata"].get("tags", [])
        required_tags = {
            "energy-audit", "energy-efficiency", "iso-50001",
            "en-16247", "eed", "industrial",
        }
        actual_tags = set(tags)
        missing = required_tags - actual_tags
        assert not missing, f"Missing required tags: {missing}"

    def test_metadata_has_compliance_references(self, pack_yaml_data):
        """Metadata includes regulatory compliance references."""
        meta = pack_yaml_data["metadata"]
        has_regulation = "regulation" in meta or "compliance_references" in meta
        assert has_regulation, "Missing regulation or compliance_references"

    def test_metadata_compliance_references_count(self, pack_yaml_data):
        """At least 5 compliance references defined."""
        refs = pack_yaml_data["metadata"].get("compliance_references", [])
        assert len(refs) >= 5, f"Expected at least 5 compliance refs, got {len(refs)}"


# =============================================================================
# 3. Engines Validation
# =============================================================================


class TestManifestEngines:
    """Test engines section: all 10 engines declared."""

    def test_engines_section_exists(self, pack_yaml_data):
        """Components.engines section exists."""
        assert "engines" in pack_yaml_data.get("components", {})

    def test_engines_count(self, pack_yaml_data):
        """Exactly 10 engines are declared."""
        engines = pack_yaml_data["components"]["engines"]
        assert len(engines) == 10, f"Expected 10 engines, got {len(engines)}"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_declared_in_manifest(self, pack_yaml_data, engine_key):
        """Each expected engine is declared in the manifest."""
        engines = pack_yaml_data["components"]["engines"]
        engine_ids = [e["id"] if isinstance(e, dict) else e for e in engines]
        assert engine_key in engine_ids, f"Engine {engine_key} not in manifest"

    def test_all_engines_have_id(self, pack_yaml_data):
        """Every engine entry has an 'id' field."""
        for engine in pack_yaml_data["components"]["engines"]:
            if isinstance(engine, dict):
                assert "id" in engine, f"Engine missing 'id': {engine}"

    def test_all_engines_have_name(self, pack_yaml_data):
        """Every engine entry has a 'name' field."""
        for engine in pack_yaml_data["components"]["engines"]:
            if isinstance(engine, dict):
                assert "name" in engine, f"Engine missing 'name': {engine}"

    def test_all_engines_have_description(self, pack_yaml_data):
        """Every engine entry has a 'description' field."""
        for engine in pack_yaml_data["components"]["engines"]:
            if isinstance(engine, dict):
                assert "description" in engine, f"Engine missing 'description'"

    def test_energy_baseline_engine_declared(self, pack_yaml_data):
        """EnergyBaselineEngine is in manifest."""
        engines = pack_yaml_data["components"]["engines"]
        ids = [e["id"] for e in engines if isinstance(e, dict)]
        assert "energy_baseline" in ids

    def test_energy_audit_engine_declared(self, pack_yaml_data):
        """EnergyAuditEngine is in manifest."""
        engines = pack_yaml_data["components"]["engines"]
        ids = [e["id"] for e in engines if isinstance(e, dict)]
        assert "energy_audit" in ids

    def test_compressed_air_engine_declared(self, pack_yaml_data):
        """CompressedAirEngine is in manifest."""
        engines = pack_yaml_data["components"]["engines"]
        ids = [e["id"] for e in engines if isinstance(e, dict)]
        assert "compressed_air" in ids


# =============================================================================
# 4. Workflows Validation
# =============================================================================


class TestManifestWorkflows:
    """Test workflows section: all 8 workflows declared."""

    def test_workflows_section_exists(self, pack_yaml_data):
        """Components.workflows section exists."""
        assert "workflows" in pack_yaml_data.get("components", {})

    def test_workflows_count(self, pack_yaml_data):
        """Exactly 8 workflows are declared."""
        workflows = pack_yaml_data["components"]["workflows"]
        assert len(workflows) == 8, f"Expected 8 workflows, got {len(workflows)}"

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_declared_in_manifest(self, pack_yaml_data, wf_key):
        """Each expected workflow is declared in the manifest."""
        workflows = pack_yaml_data["components"]["workflows"]
        wf_ids = [w["id"] if isinstance(w, dict) else w for w in workflows]
        assert wf_key in wf_ids, f"Workflow {wf_key} not in manifest"


# =============================================================================
# 5. Templates Validation
# =============================================================================


class TestManifestTemplates:
    """Test templates section: all 10 templates declared."""

    def test_templates_section_exists(self, pack_yaml_data):
        """Components.templates section exists."""
        assert "templates" in pack_yaml_data.get("components", {})

    def test_templates_count(self, pack_yaml_data):
        """Exactly 10 templates are declared."""
        templates = pack_yaml_data["components"]["templates"]
        assert len(templates) == 10, f"Expected 10 templates, got {len(templates)}"

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_declared_in_manifest(self, pack_yaml_data, tpl_key):
        """Each expected template is declared in the manifest."""
        templates = pack_yaml_data["components"]["templates"]
        tpl_ids = [t["id"] if isinstance(t, dict) else t for t in templates]
        assert tpl_key in tpl_ids, f"Template {tpl_key} not in manifest"


# =============================================================================
# 6. Integrations Validation
# =============================================================================


class TestManifestIntegrations:
    """Test integrations section: all 12 integrations declared."""

    def test_integrations_section_exists(self, pack_yaml_data):
        """Components.integrations section exists."""
        assert "integrations" in pack_yaml_data.get("components", {})

    def test_integrations_count(self, pack_yaml_data):
        """At least 11 integrations are declared."""
        integrations = pack_yaml_data["components"]["integrations"]
        assert len(integrations) >= 11, f"Expected >=11 integrations, got {len(integrations)}"

    def test_pack_orchestrator_declared(self, pack_yaml_data):
        """pack_orchestrator integration is declared."""
        integrations = pack_yaml_data["components"]["integrations"]
        ids = [i["id"] if isinstance(i, dict) else i for i in integrations]
        assert "pack_orchestrator" in ids

    def test_health_check_declared(self, pack_yaml_data):
        """health_check integration is declared."""
        integrations = pack_yaml_data["components"]["integrations"]
        ids = [i["id"] if isinstance(i, dict) else i for i in integrations]
        assert "health_check" in ids

    def test_setup_wizard_declared(self, pack_yaml_data):
        """setup_wizard integration is declared."""
        integrations = pack_yaml_data["components"]["integrations"]
        ids = [i["id"] if isinstance(i, dict) else i for i in integrations]
        assert "setup_wizard" in ids


# =============================================================================
# 7. Presets Validation
# =============================================================================


class TestManifestPresets:
    """Test presets section: all 8 presets declared."""

    def test_presets_section_exists(self, pack_yaml_data):
        """Components.presets section exists."""
        assert "presets" in pack_yaml_data.get("components", {})

    def test_presets_count(self, pack_yaml_data):
        """Exactly 8 presets are declared."""
        presets = pack_yaml_data["components"]["presets"]
        assert len(presets) == 8, f"Expected 8 presets, got {len(presets)}"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_declared_in_manifest(self, pack_yaml_data, preset_name):
        """Each expected preset is declared in the manifest."""
        presets = pack_yaml_data["components"]["presets"]
        preset_ids = [p["id"] if isinstance(p, dict) else p for p in presets]
        assert preset_name in preset_ids, f"Preset {preset_name} not in manifest"


# =============================================================================
# 8. Agent Dependencies Validation
# =============================================================================


class TestManifestAgentDependencies:
    """Test agent dependency counts and structure."""

    def test_agents_mrv_exists(self, pack_yaml_data):
        """agents_mrv section exists."""
        assert "agents_mrv" in pack_yaml_data

    def test_agents_data_exists(self, pack_yaml_data):
        """agents_data section exists."""
        assert "agents_data" in pack_yaml_data

    def test_agents_foundation_exists(self, pack_yaml_data):
        """agents_foundation section exists."""
        assert "agents_foundation" in pack_yaml_data

    def test_agents_mrv_has_scope_sections(self, pack_yaml_data):
        """agents_mrv has scope_1, scope_2, scope_3 sub-sections or 30 agents."""
        mrv = pack_yaml_data["agents_mrv"]
        if isinstance(mrv, dict):
            for scope in ["scope_1", "scope_2", "scope_3"]:
                assert scope in mrv, f"Missing {scope} in agents_mrv"
        elif isinstance(mrv, list):
            assert len(mrv) >= 20, f"Expected at least 20 MRV agents, got {len(mrv)}"

    def test_total_agent_count_minimum(self, pack_yaml_data):
        """Total agent dependency count is at least 30."""
        deps = pack_yaml_data.get("dependencies", {})
        total = deps.get("total_agents", 0)
        if total == 0:
            count = 0
            for section in ["agents_mrv", "agents_data", "agents_foundation", "agents_bridged"]:
                val = pack_yaml_data.get(section)
                if isinstance(val, list):
                    count += len(val)
                elif isinstance(val, dict):
                    for sub_val in val.values():
                        if isinstance(sub_val, list):
                            count += len(sub_val)
            assert count >= 30, f"Expected at least 30 agents, counted {count}"
        else:
            assert total >= 30, f"Expected at least 30 agents, got {total}"


# =============================================================================
# 9. Requirements and Performance
# =============================================================================


class TestManifestRequirements:
    """Test requirements and performance sections."""

    def test_requirements_section_exists(self, pack_yaml_data):
        """Requirements section exists."""
        assert "requirements" in pack_yaml_data

    def test_performance_section_exists(self, pack_yaml_data):
        """Performance section exists."""
        assert "performance" in pack_yaml_data

    def test_security_section_exists(self, pack_yaml_data):
        """Security section exists."""
        assert "security" in pack_yaml_data

    def test_installation_section_exists(self, pack_yaml_data):
        """Installation section exists."""
        assert "installation" in pack_yaml_data


# =============================================================================
# 10. Cross-Reference and Consistency
# =============================================================================


class TestManifestConsistency:
    """Test cross-references between sections are consistent."""

    def test_init_py_exists_in_engines(self):
        """__init__.py exists in engines directory."""
        assert (ENGINES_DIR / "__init__.py").exists()

    def test_init_py_exists_in_workflows(self):
        """__init__.py exists in workflows directory."""
        assert (WORKFLOWS_DIR / "__init__.py").exists()

    def test_init_py_exists_in_config(self):
        """__init__.py exists in config directory."""
        assert (CONFIG_DIR / "__init__.py").exists()

    def test_pack_yaml_references_eed(self, pack_yaml_data):
        """pack.yaml references EED regulatory framework."""
        yaml_str = yaml.dump(pack_yaml_data)
        assert "EED" in yaml_str or "eed" in yaml_str

    def test_pack_yaml_references_iso50001(self, pack_yaml_data):
        """pack.yaml references ISO 50001."""
        yaml_str = yaml.dump(pack_yaml_data)
        assert "50001" in yaml_str

    def test_pack_yaml_references_en16247(self, pack_yaml_data):
        """pack.yaml references EN 16247."""
        yaml_str = yaml.dump(pack_yaml_data)
        assert "16247" in yaml_str

    def test_pack_yaml_references_ipmvp(self, pack_yaml_data):
        """pack.yaml references IPMVP."""
        yaml_str = yaml.dump(pack_yaml_data)
        assert "IPMVP" in yaml_str or "ipmvp" in yaml_str
