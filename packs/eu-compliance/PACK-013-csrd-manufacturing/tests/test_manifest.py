# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Manifest Tests (test_manifest.py)
=====================================================================

Validates the pack.yaml manifest file for PACK-013 CSRD Manufacturing Pack.
Ensures all required metadata, engines, workflows, templates, integrations,
and presets are correctly declared and that referenced files exist on disk.

Test count: 25 test definitions, ~65 collected with parametrize expansions.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-013 CSRD Manufacturing
Date:    March 2026
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


# =============================================================================
# Expected Manifest Counts
# =============================================================================

EXPECTED_ENGINE_COUNT = 8
EXPECTED_WORKFLOW_COUNT = 8
EXPECTED_TEMPLATE_COUNT = 8
EXPECTED_INTEGRATION_MIN_COUNT = 9
EXPECTED_PRESET_COUNT = 6
EXPECTED_AGENT_DEPENDENCY_MIN_COUNT = 30

# Expected engine IDs
EXPECTED_ENGINE_IDS = [
    "process_emissions",
    "energy_intensity",
    "product_carbon_footprint",
    "circular_economy",
    "water_pollution",
    "bat_compliance",
    "supply_chain_emissions",
    "manufacturing_benchmark",
]

# Expected workflow IDs
EXPECTED_WORKFLOW_IDS = [
    "manufacturing_emissions",
    "product_pcf",
    "circular_economy",
    "bat_compliance",
    "supply_chain_assessment",
    "esrs_manufacturing",
    "decarbonization_roadmap",
    "regulatory_compliance",
]

# Expected template IDs
EXPECTED_TEMPLATE_IDS = [
    "process_emissions_report",
    "product_pcf_label",
    "energy_performance_report",
    "circular_economy_report",
    "bat_compliance_report",
    "water_pollution_report",
    "manufacturing_scorecard",
    "decarbonization_roadmap",
]

# Expected integration IDs (minimum set)
EXPECTED_INTEGRATION_IDS = [
    "pack_orchestrator",
    "csrd_pack_bridge",
    "cbam_pack_bridge",
    "mrv_industrial_bridge",
    "data_manufacturing_bridge",
    "eu_ets_bridge",
    "taxonomy_bridge",
    "health_check",
    "setup_wizard",
]

# Expected preset IDs
EXPECTED_PRESET_IDS = [
    "heavy_industry",
    "discrete_manufacturing",
    "process_manufacturing",
    "light_manufacturing",
    "multi_site",
    "sme_manufacturer",
]


# =============================================================================
# Helper Functions
# =============================================================================


def _get_components(pack_yaml_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the components section from pack.yaml."""
    return pack_yaml_data.get("components", {})


def _get_engines(pack_yaml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the engines list from pack.yaml."""
    components = _get_components(pack_yaml_data)
    return components.get("engines", [])


def _get_workflows(pack_yaml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the workflows list from pack.yaml."""
    components = _get_components(pack_yaml_data)
    return components.get("workflows", [])


def _get_templates(pack_yaml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the templates list from pack.yaml."""
    components = _get_components(pack_yaml_data)
    return components.get("templates", [])


def _get_integrations(pack_yaml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the integrations list from pack.yaml."""
    components = _get_components(pack_yaml_data)
    return components.get("integrations", [])


def _get_presets(pack_yaml_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the presets list from pack.yaml."""
    components = _get_components(pack_yaml_data)
    return components.get("presets", [])


def _count_agent_dependencies(pack_yaml_data: Dict[str, Any]) -> int:
    """Count total agent dependencies across all sections."""
    count = 0
    for section_key in ["agents_mrv", "agents_data", "agents_foundation"]:
        section = pack_yaml_data.get(section_key, {})
        if isinstance(section, dict):
            for scope_key, agents in section.items():
                if isinstance(agents, list):
                    count += len(agents)
        elif isinstance(section, list):
            count += len(section)
    return count


# =============================================================================
# Test Class: Manifest Existence and Validity
# =============================================================================


class TestManifestExistence:
    """Tests that pack.yaml exists and is valid YAML."""

    def test_pack_yaml_exists(self, pack_yaml_path: Path):
        """Verify pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), (
            f"pack.yaml not found at {pack_yaml_path}"
        )

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path):
        """Verify pack.yaml parses as valid YAML without errors."""
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None, "pack.yaml parsed to None (empty file)"
        assert isinstance(data, dict), (
            f"pack.yaml root must be a dict, got {type(data).__name__}"
        )

    def test_pack_yaml_size_reasonable(self, pack_yaml_path: Path):
        """Verify pack.yaml size is within reasonable bounds (10KB - 200KB)."""
        size_bytes = pack_yaml_path.stat().st_size
        assert size_bytes >= 10_000, (
            f"pack.yaml is suspiciously small ({size_bytes} bytes)"
        )
        assert size_bytes <= 200_000, (
            f"pack.yaml is suspiciously large ({size_bytes} bytes)"
        )


# =============================================================================
# Test Class: Metadata Fields
# =============================================================================


class TestMetadata:
    """Tests for the metadata section of pack.yaml."""

    def test_metadata_fields(self, pack_yaml_data: Dict[str, Any]):
        """Verify all required metadata fields are present."""
        metadata = pack_yaml_data.get("metadata", {})
        required_fields = ["name", "version", "description", "category", "tier"]
        for field in required_fields:
            assert field in metadata, (
                f"Missing required metadata field: {field}"
            )

    def test_metadata_name(self, pack_yaml_data: Dict[str, Any]):
        """Verify metadata.name matches expected pack identifier."""
        name = pack_yaml_data["metadata"]["name"]
        assert name == "PACK-013-csrd-manufacturing", (
            f"Unexpected pack name: {name}"
        )

    def test_metadata_category(self, pack_yaml_data: Dict[str, Any]):
        """Verify metadata.category is eu-compliance."""
        category = pack_yaml_data["metadata"]["category"]
        assert category == "eu-compliance", (
            f"Unexpected category: {category}"
        )

    def test_metadata_tier(self, pack_yaml_data: Dict[str, Any]):
        """Verify metadata.tier is sector-specific."""
        tier = pack_yaml_data["metadata"]["tier"]
        assert tier == "sector-specific", (
            f"Unexpected tier: {tier}"
        )

    def test_version_semver(self, pack_yaml_data: Dict[str, Any]):
        """Verify metadata.version follows semantic versioning (X.Y.Z)."""
        version = pack_yaml_data["metadata"]["version"]
        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, version), (
            f"Version '{version}' does not match semver pattern X.Y.Z"
        )


# =============================================================================
# Test Class: Engines Section
# =============================================================================


class TestEngines:
    """Tests for the engines section of pack.yaml."""

    def test_engines_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify engines section exists and has expected count."""
        engines = _get_engines(pack_yaml_data)
        assert len(engines) == EXPECTED_ENGINE_COUNT, (
            f"Expected {EXPECTED_ENGINE_COUNT} engines, got {len(engines)}"
        )

    @pytest.mark.parametrize("engine_id", EXPECTED_ENGINE_IDS)
    def test_each_engine_has_fields(
        self, pack_yaml_data: Dict[str, Any], engine_id: str
    ):
        """Verify each engine entry has required fields: id, name, description."""
        engines = _get_engines(pack_yaml_data)
        engine = next((e for e in engines if e.get("id") == engine_id), None)
        assert engine is not None, (
            f"Engine '{engine_id}' not found in pack.yaml"
        )
        for field in ["id", "name", "description"]:
            assert field in engine, (
                f"Engine '{engine_id}' missing field: {field}"
            )

    @pytest.mark.parametrize("engine_id", EXPECTED_ENGINE_IDS)
    def test_engine_files_exist(
        self, pack_root: Path, engine_id: str
    ):
        """Verify each engine's Python file exists on disk."""
        from conftest import ENGINE_FILES

        file_name = ENGINE_FILES.get(engine_id)
        assert file_name is not None, (
            f"No file mapping for engine '{engine_id}'"
        )
        file_path = pack_root / "engines" / file_name
        assert file_path.exists(), (
            f"Engine file not found: {file_path}"
        )

    def test_no_duplicate_engine_ids(self, pack_yaml_data: Dict[str, Any]):
        """Verify no duplicate engine IDs in the manifest."""
        engines = _get_engines(pack_yaml_data)
        ids = [e.get("id") for e in engines]
        assert len(ids) == len(set(ids)), (
            f"Duplicate engine IDs found: {[x for x in ids if ids.count(x) > 1]}"
        )


# =============================================================================
# Test Class: Workflows Section
# =============================================================================


class TestWorkflows:
    """Tests for the workflows section of pack.yaml."""

    def test_workflows_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify workflows section exists and has expected count."""
        workflows = _get_workflows(pack_yaml_data)
        assert len(workflows) == EXPECTED_WORKFLOW_COUNT, (
            f"Expected {EXPECTED_WORKFLOW_COUNT} workflows, got {len(workflows)}"
        )

    @pytest.mark.parametrize("workflow_id", EXPECTED_WORKFLOW_IDS)
    def test_each_workflow_has_fields(
        self, pack_yaml_data: Dict[str, Any], workflow_id: str
    ):
        """Verify each workflow entry has required fields: id, name, description."""
        workflows = _get_workflows(pack_yaml_data)
        wf = next((w for w in workflows if w.get("id") == workflow_id), None)
        assert wf is not None, (
            f"Workflow '{workflow_id}' not found in pack.yaml"
        )
        for field in ["id", "name", "description"]:
            assert field in wf, (
                f"Workflow '{workflow_id}' missing field: {field}"
            )

    @pytest.mark.parametrize("workflow_id", EXPECTED_WORKFLOW_IDS)
    def test_workflow_files_exist(
        self, pack_root: Path, workflow_id: str
    ):
        """Verify each workflow's Python file exists on disk."""
        from conftest import WORKFLOW_FILES

        file_name = WORKFLOW_FILES.get(workflow_id)
        assert file_name is not None, (
            f"No file mapping for workflow '{workflow_id}'"
        )
        file_path = pack_root / "workflows" / file_name
        assert file_path.exists(), (
            f"Workflow file not found: {file_path}"
        )

    def test_no_duplicate_workflow_ids(self, pack_yaml_data: Dict[str, Any]):
        """Verify no duplicate workflow IDs in the manifest."""
        workflows = _get_workflows(pack_yaml_data)
        ids = [w.get("id") for w in workflows]
        assert len(ids) == len(set(ids)), (
            f"Duplicate workflow IDs found: {[x for x in ids if ids.count(x) > 1]}"
        )


# =============================================================================
# Test Class: Templates Section
# =============================================================================


class TestTemplates:
    """Tests for the templates section of pack.yaml."""

    def test_templates_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify templates section exists and has expected count."""
        templates = _get_templates(pack_yaml_data)
        assert len(templates) == EXPECTED_TEMPLATE_COUNT, (
            f"Expected {EXPECTED_TEMPLATE_COUNT} templates, got {len(templates)}"
        )

    @pytest.mark.parametrize("template_id", EXPECTED_TEMPLATE_IDS)
    def test_each_template_has_fields(
        self, pack_yaml_data: Dict[str, Any], template_id: str
    ):
        """Verify each template has required fields: id, display_name, description."""
        templates = _get_templates(pack_yaml_data)
        tmpl = next((t for t in templates if t.get("id") == template_id), None)
        assert tmpl is not None, (
            f"Template '{template_id}' not found in pack.yaml"
        )
        for field in ["id", "display_name", "description"]:
            assert field in tmpl, (
                f"Template '{template_id}' missing field: {field}"
            )

    @pytest.mark.parametrize("template_id", EXPECTED_TEMPLATE_IDS)
    def test_template_files_exist(
        self, pack_root: Path, template_id: str
    ):
        """Verify each template's Python file exists on disk."""
        from conftest import TEMPLATE_FILES

        file_name = TEMPLATE_FILES.get(template_id)
        assert file_name is not None, (
            f"No file mapping for template '{template_id}'"
        )
        file_path = pack_root / "templates" / file_name
        assert file_path.exists(), (
            f"Template file not found: {file_path}"
        )

    def test_no_duplicate_template_ids(self, pack_yaml_data: Dict[str, Any]):
        """Verify no duplicate template IDs in the manifest."""
        templates = _get_templates(pack_yaml_data)
        ids = [t.get("id") for t in templates]
        assert len(ids) == len(set(ids)), (
            f"Duplicate template IDs found: {[x for x in ids if ids.count(x) > 1]}"
        )


# =============================================================================
# Test Class: Integrations Section
# =============================================================================


class TestIntegrations:
    """Tests for the integrations section of pack.yaml."""

    def test_integrations_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify integrations section exists and has at least minimum count."""
        integrations = _get_integrations(pack_yaml_data)
        assert len(integrations) >= EXPECTED_INTEGRATION_MIN_COUNT, (
            f"Expected at least {EXPECTED_INTEGRATION_MIN_COUNT} integrations, "
            f"got {len(integrations)}"
        )

    @pytest.mark.parametrize("integration_id", EXPECTED_INTEGRATION_IDS)
    def test_integration_files_exist(
        self, pack_root: Path, integration_id: str
    ):
        """Verify each integration's Python file exists on disk."""
        from conftest import INTEGRATION_FILES

        file_name = INTEGRATION_FILES.get(integration_id)
        assert file_name is not None, (
            f"No file mapping for integration '{integration_id}'"
        )
        file_path = pack_root / "integrations" / file_name
        assert file_path.exists(), (
            f"Integration file not found: {file_path}"
        )


# =============================================================================
# Test Class: Presets Section
# =============================================================================


class TestPresets:
    """Tests for the presets section of pack.yaml."""

    def test_presets_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify presets section exists and has expected count."""
        presets = _get_presets(pack_yaml_data)
        assert len(presets) == EXPECTED_PRESET_COUNT, (
            f"Expected {EXPECTED_PRESET_COUNT} presets, got {len(presets)}"
        )

    @pytest.mark.parametrize("preset_id", EXPECTED_PRESET_IDS)
    def test_preset_files_exist(self, pack_root: Path, preset_id: str):
        """Verify each preset YAML file exists on disk."""
        file_path = pack_root / "config" / "presets" / f"{preset_id}.yaml"
        assert file_path.exists(), (
            f"Preset file not found: {file_path}"
        )


# =============================================================================
# Test Class: Agent Dependencies
# =============================================================================


class TestAgentDependencies:
    """Tests for the agent dependency declarations in pack.yaml."""

    def test_agent_dependencies_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify at least one agent dependency section exists."""
        has_mrv = "agents_mrv" in pack_yaml_data
        has_data = "agents_data" in pack_yaml_data
        has_found = "agents_foundation" in pack_yaml_data
        assert has_mrv or has_data or has_found, (
            "No agent dependency sections found in pack.yaml. "
            "Expected agents_mrv, agents_data, or agents_foundation."
        )

    def test_agent_dependencies_count(self, pack_yaml_data: Dict[str, Any]):
        """Verify total agent dependencies meet the minimum threshold."""
        count = _count_agent_dependencies(pack_yaml_data)
        assert count >= EXPECTED_AGENT_DEPENDENCY_MIN_COUNT, (
            f"Expected at least {EXPECTED_AGENT_DEPENDENCY_MIN_COUNT} agent "
            f"dependencies, got {count}"
        )


# =============================================================================
# Test Class: Requirements and Demo Config
# =============================================================================


class TestRequirementsAndDemo:
    """Tests for requirements and demo configuration references."""

    def test_requirements_section(self, pack_yaml_data: Dict[str, Any]):
        """Verify a compliance_references or regulation section exists."""
        metadata = pack_yaml_data.get("metadata", {})
        has_compliance = "compliance_references" in metadata
        has_regulation = "regulation" in metadata
        assert has_compliance or has_regulation, (
            "Neither compliance_references nor regulation found in metadata"
        )

    def test_demo_config_exists(self, pack_root: Path):
        """Verify demo configuration YAML file exists on disk."""
        demo_path = pack_root / "config" / "demo" / "demo_config.yaml"
        assert demo_path.exists(), (
            f"Demo config not found: {demo_path}"
        )
