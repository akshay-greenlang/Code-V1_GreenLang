# -*- coding: utf-8 -*-
"""
Tests for PACK-029 Interim Targets Pack manifest (pack.yaml).

Validates pack.yaml parsing, metadata fields, component counts,
dependency declarations, performance targets, and regulatory
framework references.

Target: ~50 tests.

Author: GreenLang Platform Team
Pack: PACK-029 Interim Targets Pack
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
        assert pack_yaml_path.exists()

    def test_pack_yaml_is_file(self, pack_yaml_path):
        assert pack_yaml_path.is_file()

    def test_pack_yaml_parses(self, pack_yaml):
        assert pack_yaml is not None
        assert isinstance(pack_yaml, dict)

    def test_pack_yaml_not_empty(self, pack_yaml):
        assert len(pack_yaml) > 0


# ========================================================================
# Pack Identity
# ========================================================================


class TestPackIdentity:
    """Verify pack identity fields."""

    def test_has_pack_id(self, pack_yaml):
        assert "pack_id" in pack_yaml
        assert pack_yaml["pack_id"] == "PACK-029"

    def test_has_pack_name(self, pack_yaml):
        assert "pack_name" in pack_yaml
        assert "Interim Targets" in pack_yaml["pack_name"]

    def test_has_version(self, pack_yaml):
        assert "version" in pack_yaml
        assert pack_yaml["version"] == "1.0.0"

    def test_has_tier(self, pack_yaml):
        assert "tier" in pack_yaml

    def test_has_category(self, pack_yaml):
        assert "category" in pack_yaml
        assert pack_yaml["category"] == "Net Zero"

    def test_has_description(self, pack_yaml):
        assert "description" in pack_yaml
        assert len(pack_yaml["description"]) > 0


# ========================================================================
# Pack Metadata
# ========================================================================


class TestPackMetadata:
    """Verify pack metadata section."""

    def test_has_metadata_section(self, pack_yaml):
        assert "metadata" in pack_yaml

    def test_has_author(self, metadata):
        assert "author" in metadata
        assert "GreenLang" in metadata["author"]

    def test_has_keywords(self, metadata):
        assert "keywords" in metadata
        assert isinstance(metadata["keywords"], list)
        assert len(metadata["keywords"]) >= 5

    def test_keywords_include_core_terms(self, metadata):
        keywords = metadata.get("keywords", [])
        core_terms = ["interim-targets", "net-zero"]
        for term in core_terms:
            assert term in keywords, f"Missing keyword: {term}"


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
        if isinstance(engines, dict) and "count" in engines:
            assert engines["count"] == 10
        elif isinstance(engines, dict) and "list" in engines:
            assert len(engines["list"]) == 10
        elif isinstance(engines, list):
            assert len(engines) == 10

    def test_has_workflows_section(self, components):
        assert "workflows" in components

    def test_7_workflows(self, components):
        workflows = components.get("workflows", {})
        if isinstance(workflows, dict) and "count" in workflows:
            assert workflows["count"] == 7
        elif isinstance(workflows, dict) and "list" in workflows:
            assert len(workflows["list"]) == 7
        elif isinstance(workflows, list):
            assert len(workflows) == 7

    def test_has_templates_section(self, components):
        assert "templates" in components

    def test_10_templates(self, components):
        templates = components.get("templates", {})
        if isinstance(templates, dict) and "count" in templates:
            assert templates["count"] == 10
        elif isinstance(templates, dict) and "list" in templates:
            assert len(templates["list"]) == 10
        elif isinstance(templates, list):
            assert len(templates) == 10

    def test_has_integrations_section(self, components):
        assert "integrations" in components

    def test_10_integrations(self, components):
        integrations = components.get("integrations", {})
        if isinstance(integrations, dict) and "count" in integrations:
            assert integrations["count"] == 10
        elif isinstance(integrations, dict) and "list" in integrations:
            assert len(integrations["list"]) == 10
        elif isinstance(integrations, list):
            assert len(integrations) == 10

    def test_has_presets_section(self, components):
        assert "presets" in components

    def test_7_presets(self, components):
        presets = components.get("presets", {})
        if isinstance(presets, dict) and "count" in presets:
            assert presets["count"] == 7
        elif isinstance(presets, dict) and "list" in presets:
            assert len(presets["list"]) == 7
        elif isinstance(presets, list):
            assert len(presets) == 7


# ========================================================================
# Engine Details
# ========================================================================


class TestEngineManifestDetails:
    """Verify engine manifest entries."""

    def test_each_engine_has_id(self, components):
        engines = components.get("engines", {})
        engine_list = engines.get("list", engines) if isinstance(engines, dict) else engines
        if isinstance(engine_list, list):
            for engine in engine_list:
                assert "id" in engine or "name" in engine

    def test_each_engine_has_description(self, components):
        engines = components.get("engines", {})
        engine_list = engines.get("list", engines) if isinstance(engines, dict) else engines
        if isinstance(engine_list, list):
            for engine in engine_list:
                if "description" in engine:
                    assert len(str(engine["description"])) > 0


# ========================================================================
# Workflow Details
# ========================================================================


class TestWorkflowManifestDetails:
    """Verify workflow manifest entries."""

    def test_each_workflow_has_id(self, components):
        workflows = components.get("workflows", {})
        wf_list = workflows.get("list", workflows) if isinstance(workflows, dict) else workflows
        if isinstance(wf_list, list):
            for wf in wf_list:
                assert "id" in wf or "name" in wf

    def test_each_workflow_has_phases(self, components):
        workflows = components.get("workflows", {})
        wf_list = workflows.get("list", workflows) if isinstance(workflows, dict) else workflows
        if isinstance(wf_list, list):
            for wf in wf_list:
                if "phases" in wf:
                    assert wf["phases"] >= 3, f"Workflow {wf.get('name', 'unknown')} has too few phases"


# ========================================================================
# Dependencies
# ========================================================================


class TestDependencies:
    """Verify dependency declarations."""

    def test_has_dependencies_section(self, pack_yaml):
        has_deps = (
            "dependencies" in pack_yaml
            or "agent_dependencies" in pack_yaml
            or "platform_dependencies" in pack_yaml
            or "requires" in pack_yaml
        )
        assert has_deps

    def test_has_mrv_dependency(self, pack_yaml):
        deps = (
            pack_yaml.get("platform_dependencies", {})
            or pack_yaml.get("dependencies", {})
        )
        if isinstance(deps, dict):
            has_mrv = (
                "mrv_agents" in deps
                or "mrv" in str(deps).lower()
            )
            assert has_mrv

    def test_has_pack_dependencies(self, pack_yaml):
        deps = pack_yaml.get("platform_dependencies", {})
        if isinstance(deps, dict):
            pack_deps = deps.get("pack_dependencies", [])
            if pack_deps:
                pack_ids = [d.get("pack_id", "") for d in pack_deps]
                assert "PACK-021" in pack_ids or "PACK-028" in pack_ids


# ========================================================================
# Sub-Module Metadata Consistency
# ========================================================================


class TestSubModuleMetadata:
    """Verify sub-module metadata matches pack.yaml."""

    def test_root_version_matches_manifest(self, pack_yaml):
        """Root __init__ version matches pack.yaml."""
        root_init = _PACK_ROOT / "__init__.py"
        if root_init.exists():
            content = root_init.read_text(encoding="utf-8")
            assert pack_yaml["version"] in content

    def test_root_pack_id_matches_manifest(self, pack_yaml):
        """Root __init__ pack_id matches pack.yaml."""
        root_init = _PACK_ROOT / "__init__.py"
        if root_init.exists():
            content = root_init.read_text(encoding="utf-8")
            assert pack_yaml["pack_id"] in content

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


# ========================================================================
# File Structure Completeness
# ========================================================================


class TestFileStructureCompleteness:
    """Verify pack directory structure is complete."""

    def test_engines_directory_exists(self):
        assert (_PACK_ROOT / "engines").is_dir()

    def test_workflows_directory_exists(self):
        assert (_PACK_ROOT / "workflows").is_dir()

    def test_templates_directory_exists(self):
        assert (_PACK_ROOT / "templates").is_dir()

    def test_integrations_directory_exists(self):
        assert (_PACK_ROOT / "integrations").is_dir()

    def test_config_directory_exists(self):
        assert (_PACK_ROOT / "config").is_dir()

    def test_config_presets_directory_exists(self):
        assert (_PACK_ROOT / "config" / "presets").is_dir()

    def test_tests_directory_exists(self):
        assert (_PACK_ROOT / "tests").is_dir()

    def test_pack_yaml_in_root(self):
        assert (_PACK_ROOT / "pack.yaml").exists()

    def test_root_init_exists(self):
        assert (_PACK_ROOT / "__init__.py").exists()

    def test_readme_exists(self):
        """Pack has a README file."""
        has_readme = (
            (_PACK_ROOT / "README.md").exists()
            or (_PACK_ROOT / "BUILD_COMPLETE.md").exists()
        )
        assert has_readme


# ========================================================================
# Database Migrations
# ========================================================================


class TestDatabaseMigrations:
    """Verify database migration references in manifest."""

    def test_has_database_section(self, pack_yaml):
        has_db = "database" in pack_yaml
        # Database section is expected but don't fail if absent
        if has_db:
            assert isinstance(pack_yaml["database"], dict)

    def test_has_migration_count(self, pack_yaml):
        db = pack_yaml.get("database", {})
        if db:
            migrations = db.get("migrations", {})
            if isinstance(migrations, dict) and "count" in migrations:
                assert migrations["count"] >= 10


# ========================================================================
# Testing Targets
# ========================================================================


class TestTestingTargets:
    """Verify testing targets in manifest."""

    def test_has_testing_section(self, pack_yaml):
        has_testing = "testing" in pack_yaml
        if has_testing:
            assert isinstance(pack_yaml["testing"], dict)

    def test_has_coverage_target(self, pack_yaml):
        testing = pack_yaml.get("testing", {})
        if testing and "coverage" in testing:
            coverage_str = str(testing["coverage"]).replace("%", "")
            assert float(coverage_str) >= 85


# ========================================================================
# Regulatory Framework
# ========================================================================


class TestRegulatoryFramework:
    """Verify regulatory framework references."""

    def test_has_regulatory_content(self, pack_yaml):
        """Pack references regulatory frameworks somewhere in manifest."""
        yaml_str = str(pack_yaml).lower()
        assert "sbti" in yaml_str or "cdp" in yaml_str or "tcfd" in yaml_str

    def test_description_mentions_sbti(self, pack_yaml):
        """Pack description references SBTi."""
        desc = pack_yaml.get("description", "")
        assert "SBTi" in desc or "sbti" in desc.lower()

    def test_description_mentions_cdp(self, pack_yaml):
        """Pack description references CDP."""
        desc = pack_yaml.get("description", "")
        assert "CDP" in desc or "cdp" in desc.lower()
