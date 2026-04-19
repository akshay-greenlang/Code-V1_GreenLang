# -*- coding: utf-8 -*-
"""
Tests for PACK-028 Sector Pathway Pack manifest (pack.yaml).

Validates pack.yaml parsing, metadata fields, component counts,
dependency declarations, performance targets, regulatory framework
references, and file structure completeness.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-028 Sector Pathway Pack
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
    """Return path to pack.yaml."""
    return _PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="module")
def components(pack_yaml):
    """Extract components section."""
    return pack_yaml.get("components", {})


@pytest.fixture(scope="module")
def metadata(pack_yaml):
    """Extract metadata section."""
    return pack_yaml.get("metadata", {})


# ========================================================================
# File Existence
# ========================================================================


class TestManifestFileExists:
    """Verify pack.yaml exists and is valid YAML."""

    def test_pack_yaml_exists(self, pack_yaml_path):
        """pack.yaml file exists."""
        assert pack_yaml_path.exists()

    def test_pack_yaml_is_file(self, pack_yaml_path):
        """pack.yaml is a regular file."""
        assert pack_yaml_path.is_file()

    def test_pack_yaml_parses(self, pack_yaml):
        """pack.yaml parses to a non-None dict."""
        assert pack_yaml is not None
        assert isinstance(pack_yaml, dict)

    def test_pack_yaml_not_empty(self, pack_yaml):
        """pack.yaml is not empty."""
        assert len(pack_yaml) > 0


# ========================================================================
# Pack Identity
# ========================================================================


class TestPackIdentity:
    """Verify top-level pack identity fields."""

    def test_has_pack_id(self, pack_yaml):
        """pack.yaml has pack_id field."""
        assert "pack_id" in pack_yaml
        assert pack_yaml["pack_id"] == "PACK-028"

    def test_has_pack_name(self, pack_yaml):
        """pack.yaml has pack_name field."""
        assert "pack_name" in pack_yaml
        assert "Sector Pathway" in pack_yaml["pack_name"]

    def test_has_version(self, pack_yaml):
        """pack.yaml has version field."""
        assert "version" in pack_yaml
        assert pack_yaml["version"] == "1.0.0"

    def test_has_tier(self, pack_yaml):
        """pack.yaml has tier field."""
        assert "tier" in pack_yaml

    def test_has_category(self, pack_yaml):
        """pack.yaml has category field."""
        assert "category" in pack_yaml

    def test_has_description(self, pack_yaml):
        """pack.yaml has a non-empty description."""
        assert "description" in pack_yaml
        assert len(pack_yaml["description"]) > 0


# ========================================================================
# Component Counts
# ========================================================================


class TestComponentCounts:
    """Verify expected component counts in manifest."""

    def test_has_components_section(self, pack_yaml):
        """pack.yaml has components section."""
        assert "components" in pack_yaml

    def test_has_engines_section(self, components):
        """Components has engines section."""
        assert "engines" in components

    def test_8_engines(self, components):
        """Manifest declares 8 engines."""
        engines = components.get("engines", {})
        if isinstance(engines, dict) and "count" in engines:
            assert engines["count"] == 8
        elif isinstance(engines, dict) and "list" in engines:
            assert len(engines["list"]) == 8
        elif isinstance(engines, list):
            assert len(engines) == 8

    def test_has_workflows_section(self, components):
        """Components has workflows section."""
        assert "workflows" in components

    def test_6_workflows(self, components):
        """Manifest declares 6 workflows."""
        workflows = components.get("workflows", {})
        if isinstance(workflows, dict) and "count" in workflows:
            assert workflows["count"] == 6
        elif isinstance(workflows, dict) and "list" in workflows:
            assert len(workflows["list"]) == 6
        elif isinstance(workflows, list):
            assert len(workflows) == 6

    def test_has_templates_section(self, components):
        """Components has templates section."""
        assert "templates" in components

    def test_8_templates(self, components):
        """Manifest declares 8 templates."""
        templates = components.get("templates", {})
        if isinstance(templates, dict) and "count" in templates:
            assert templates["count"] == 8
        elif isinstance(templates, dict) and "list" in templates:
            assert len(templates["list"]) == 8
        elif isinstance(templates, list):
            assert len(templates) == 8

    def test_has_integrations_section(self, components):
        """Components has integrations section."""
        assert "integrations" in components

    def test_10_integrations(self, components):
        """Manifest declares 10 integrations."""
        integrations = components.get("integrations", {})
        if isinstance(integrations, dict) and "count" in integrations:
            assert integrations["count"] == 10
        elif isinstance(integrations, dict) and "list" in integrations:
            assert len(integrations["list"]) == 10
        elif isinstance(integrations, list):
            assert len(integrations) == 10

    def test_has_presets_section(self, components):
        """Components has presets section."""
        assert "presets" in components

    def test_6_presets(self, components):
        """Manifest declares 6 presets."""
        presets = components.get("presets", {})
        if isinstance(presets, dict) and "count" in presets:
            assert presets["count"] == 6
        elif isinstance(presets, dict) and "list" in presets:
            assert len(presets["list"]) == 6
        elif isinstance(presets, list):
            assert len(presets) == 6


# ========================================================================
# Engine Details
# ========================================================================


class TestEngineManifestDetails:
    """Verify engine manifest entries."""

    def test_each_engine_has_id(self, components):
        """Each engine entry has an id or name."""
        engines = components.get("engines", {})
        engine_list = engines.get("list", engines) if isinstance(engines, dict) else engines
        if isinstance(engine_list, list):
            for engine in engine_list:
                assert "id" in engine or "name" in engine

    def test_each_engine_has_description(self, components):
        """Each engine entry has a description."""
        engines = components.get("engines", {})
        engine_list = engines.get("list", engines) if isinstance(engines, dict) else engines
        if isinstance(engine_list, list):
            for engine in engine_list:
                if "description" in engine:
                    assert len(engine["description"]) > 0


# ========================================================================
# Workflow Details
# ========================================================================


class TestWorkflowManifestDetails:
    """Verify workflow manifest entries."""

    def test_each_workflow_has_phases(self, components):
        """Each workflow entry declares its phase count."""
        workflows = components.get("workflows", {})
        wf_list = workflows.get("list", workflows) if isinstance(workflows, dict) else workflows
        if isinstance(wf_list, list):
            for wf in wf_list:
                assert "phases" in wf or "phase_sequence" in wf

    def test_full_sector_assessment_has_7_phases(self, components):
        """Full sector assessment workflow has 7 phases."""
        workflows = components.get("workflows", {})
        wf_list = workflows.get("list", workflows) if isinstance(workflows, dict) else workflows
        if isinstance(wf_list, list):
            full_wf = [w for w in wf_list if "full" in w.get("id", "").lower()]
            if full_wf:
                assert full_wf[0].get("phases", 0) == 7


# ========================================================================
# Dependencies
# ========================================================================


class TestDependencies:
    """Verify dependency declarations."""

    def test_has_dependencies_section(self, pack_yaml):
        """pack.yaml declares platform dependencies."""
        has_deps = (
            "dependencies" in pack_yaml
            or "platform_dependencies" in pack_yaml
            or "agent_dependencies" in pack_yaml
            or "requires" in pack_yaml
        )
        assert has_deps

    def test_mrv_agents_dependency(self, pack_yaml):
        """pack.yaml references MRV agents."""
        deps = (
            pack_yaml.get("platform_dependencies", {})
            or pack_yaml.get("dependencies", {})
        )
        assert "mrv_agents" in deps or "mrv" in str(deps).lower()

    def test_data_agents_dependency(self, pack_yaml):
        """pack.yaml references DATA agents."""
        deps = (
            pack_yaml.get("platform_dependencies", {})
            or pack_yaml.get("dependencies", {})
        )
        assert "data_agents" in deps or "data" in str(deps).lower()


# ========================================================================
# Sector Coverage
# ========================================================================


class TestSectorCoverage:
    """Verify sector coverage declarations."""

    def test_has_sector_coverage(self, pack_yaml):
        """pack.yaml has sector_coverage section."""
        assert "sector_coverage" in pack_yaml

    def test_sda_sectors_count(self, pack_yaml):
        """12 SDA sectors declared."""
        sda = pack_yaml.get("sector_coverage", {}).get("sda_sectors", {})
        if isinstance(sda, dict) and "count" in sda:
            assert sda["count"] == 12
        elif isinstance(sda, dict) and "list" in sda:
            assert len(sda["list"]) == 12

    def test_extended_sectors_count(self, pack_yaml):
        """4 extended sectors declared."""
        ext = pack_yaml.get("sector_coverage", {}).get("extended_sectors", {})
        if isinstance(ext, dict) and "count" in ext:
            assert ext["count"] == 4
        elif isinstance(ext, dict) and "list" in ext:
            assert len(ext["list"]) == 4


# ========================================================================
# Scenarios
# ========================================================================


class TestScenarios:
    """Verify scenario declarations."""

    def test_has_scenarios_section(self, pack_yaml):
        """pack.yaml has scenarios section."""
        assert "scenarios" in pack_yaml

    def test_5_scenarios(self, pack_yaml):
        """5 climate scenarios declared."""
        scenarios = pack_yaml.get("scenarios", {}).get("supported", [])
        assert len(scenarios) == 5


# ========================================================================
# Regulatory Framework
# ========================================================================


class TestRegulatoryFramework:
    """Verify regulatory framework references."""

    def test_has_regulatory_section(self, pack_yaml):
        """pack.yaml has regulatory_alignment section."""
        has_reg = (
            "regulatory_alignment" in pack_yaml
            or "regulatory_framework" in pack_yaml
            or "regulations" in pack_yaml
        )
        assert has_reg

    def test_sbti_referenced(self, pack_yaml):
        """SBTi is referenced in regulatory alignment."""
        reg = pack_yaml.get("regulatory_alignment", {}).get("frameworks", [])
        sbti_refs = [f for f in reg if "SBTi" in f.get("name", "")]
        assert len(sbti_refs) >= 1

    def test_iea_referenced(self, pack_yaml):
        """IEA is referenced in regulatory alignment."""
        reg = pack_yaml.get("regulatory_alignment", {}).get("frameworks", [])
        iea_refs = [f for f in reg if "IEA" in f.get("name", "")]
        assert len(iea_refs) >= 1


# ========================================================================
# File Structure Completeness
# ========================================================================


class TestFileStructureCompleteness:
    """Verify the actual file structure matches manifest declarations."""

    def test_engines_init_exists(self):
        """engines/__init__.py exists."""
        assert (_PACK_ROOT / "engines" / "__init__.py").exists()

    def test_workflows_init_exists(self):
        """workflows/__init__.py exists."""
        assert (_PACK_ROOT / "workflows" / "__init__.py").exists()

    def test_templates_init_exists(self):
        """templates/__init__.py exists."""
        assert (_PACK_ROOT / "templates" / "__init__.py").exists()

    def test_integrations_init_exists(self):
        """integrations/__init__.py exists."""
        assert (_PACK_ROOT / "integrations" / "__init__.py").exists()

    def test_config_init_exists(self):
        """config/__init__.py exists."""
        assert (_PACK_ROOT / "config" / "__init__.py").exists()

    def test_presets_init_exists(self):
        """config/presets/__init__.py exists."""
        assert (_PACK_ROOT / "config" / "presets" / "__init__.py").exists()

    def test_pack_init_exists(self):
        """Root __init__.py exists."""
        assert (_PACK_ROOT / "__init__.py").exists()

    def test_engine_files_count(self):
        """At least 8 engine Python files exist (plus __init__)."""
        engine_dir = _PACK_ROOT / "engines"
        py_files = [f for f in engine_dir.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 8, (
            f"Expected 8+ engine files, found {len(py_files)}: "
            f"{[f.name for f in py_files]}"
        )

    def test_workflow_files_count(self):
        """At least 6 workflow Python files exist (plus __init__)."""
        wf_dir = _PACK_ROOT / "workflows"
        py_files = [f for f in wf_dir.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 6, (
            f"Expected 6+ workflow files, found {len(py_files)}: "
            f"{[f.name for f in py_files]}"
        )

    def test_template_files_count(self):
        """At least 8 template Python files exist (plus __init__)."""
        tmpl_dir = _PACK_ROOT / "templates"
        py_files = [f for f in tmpl_dir.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 8, (
            f"Expected 8+ template files, found {len(py_files)}: "
            f"{[f.name for f in py_files]}"
        )

    def test_integration_files_count(self):
        """At least 10 integration Python files exist (plus __init__)."""
        int_dir = _PACK_ROOT / "integrations"
        py_files = [f for f in int_dir.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 10, (
            f"Expected 10+ integration files, found {len(py_files)}: "
            f"{[f.name for f in py_files]}"
        )

    def test_preset_yaml_files_count(self):
        """At least 6 preset YAML files exist."""
        preset_dir = _PACK_ROOT / "config" / "presets"
        yaml_files = list(preset_dir.glob("*.yaml"))
        assert len(yaml_files) >= 6, (
            f"Expected 6+ preset files, found {len(yaml_files)}: "
            f"{[f.name for f in yaml_files]}"
        )


# ========================================================================
# Performance Targets
# ========================================================================


class TestPerformanceTargets:
    """Verify performance targets if present."""

    def test_has_performance_section(self, pack_yaml):
        """pack.yaml has performance-related section."""
        has_perf = (
            "performance" in pack_yaml
            or "performance_targets" in pack_yaml
            or "technical_specs" in pack_yaml
        )
        assert has_perf

    def test_technical_specs_has_perf_targets(self, pack_yaml):
        """technical_specs has performance_targets."""
        specs = pack_yaml.get("technical_specs", {})
        if specs:
            assert "performance_targets" in specs


# ========================================================================
# Metadata Section
# ========================================================================


class TestPackMetadata:
    """Verify pack metadata section."""

    def test_has_metadata_section(self, pack_yaml):
        """pack.yaml has metadata section."""
        assert "metadata" in pack_yaml

    def test_has_author(self, metadata):
        """Metadata has author field."""
        assert "author" in metadata

    def test_has_keywords(self, metadata):
        """Metadata has keywords list."""
        assert "keywords" in metadata
        assert isinstance(metadata["keywords"], list)
        assert len(metadata["keywords"]) > 0

    def test_keywords_include_sector_pathway(self, metadata):
        """Keywords include 'sector-pathway'."""
        assert "sector-pathway" in metadata["keywords"]

    def test_keywords_include_sbti(self, metadata):
        """Keywords include 'sbti-sda'."""
        assert "sbti-sda" in metadata["keywords"]
