# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Manifest Tests
=====================================================

Tests the pack.yaml manifest for PACK-008 to ensure:
- Pack metadata is correct (ID, version, tier, category)
- All 10 taxonomy engines are listed
- All 10 taxonomy workflows are listed
- All 10 taxonomy templates are listed
- All 12 integrations are listed
- All 30 MRV agents are present (Scope 1/2/3 + cross-cutting)
- All 10 data agents are present
- All 10 foundation agents are present
- Total agent count is 51 (1 app + 30 MRV + 10 data + 10 foundation)
- EU Taxonomy Regulation (EU) 2020/852 reference is correct
- Performance targets are specified
- Presets and sectors are defined
- All engine IDs are unique

Author: GreenLang QA Team
Version: 1.0.0
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

_PACK_008_DIR = Path(__file__).resolve().parent.parent
_PACK_YAML_PATH = _PACK_008_DIR / "pack.yaml"


@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    return _PACK_YAML_PATH


@pytest.fixture(scope="module")
def pack_yaml() -> Dict[str, Any]:
    if not _PACK_YAML_PATH.exists():
        pytest.skip("pack.yaml not found")
    with open(_PACK_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.unit
class TestPackManifest:
    """Test suite for PACK-008 pack.yaml manifest."""

    # -----------------------------------------------------------------
    # Existence and validity
    # -----------------------------------------------------------------

    def test_pack_yaml_exists(self, pack_yaml_path: Path):
        """Test that pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path):
        """Test that pack.yaml is valid YAML."""
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "pack.yaml must parse to a dictionary"
        assert len(data) > 0, "pack.yaml must not be empty"

    # -----------------------------------------------------------------
    # Metadata fields
    # -----------------------------------------------------------------

    def test_metadata_fields(self, pack_yaml: Dict[str, Any]):
        """Test that metadata contains name, version, display_name, category."""
        meta = pack_yaml.get("metadata", {})
        assert "name" in meta, "metadata.name missing"
        assert "version" in meta, "metadata.version missing"
        assert "display_name" in meta, "metadata.display_name missing"
        assert "category" in meta, "metadata.category missing"

        assert "PACK-008" in meta["name"], (
            f"Expected PACK-008 in metadata.name, got {meta['name']}"
        )
        assert meta["display_name"] == "EU Taxonomy Alignment Pack", (
            f"Expected 'EU Taxonomy Alignment Pack', got {meta['display_name']}"
        )
        assert meta["category"] == "eu-compliance", (
            f"Expected eu-compliance category, got {meta['category']}"
        )

    def test_metadata_version_format(self, pack_yaml: Dict[str, Any]):
        """Test that version follows semver format X.Y.Z."""
        meta = pack_yaml.get("metadata", {})
        version = meta.get("version", "")
        assert isinstance(version, str), f"version must be string, got {type(version)}"
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"Invalid semver version format: {version}"
        )

    def test_metadata_regulation_references(self, pack_yaml: Dict[str, Any]):
        """Test regulation references include all relevant delegated acts."""
        meta = pack_yaml.get("metadata", {})
        compliance_refs = meta.get("compliance_references", [])

        # Expect at least 4 regulation references
        assert len(compliance_refs) >= 4, (
            f"Expected at least 4 compliance references, found {len(compliance_refs)}"
        )

        ref_ids = [ref.get("id", "") for ref in compliance_refs]
        assert "EU-TAXONOMY" in ref_ids, "Missing EU-TAXONOMY reference"
        assert "ARTICLE-8-DA" in ref_ids, "Missing ARTICLE-8-DA reference"
        assert "CLIMATE-DA" in ref_ids, "Missing CLIMATE-DA reference"
        assert "ENVIRONMENTAL-DA" in ref_ids, "Missing ENVIRONMENTAL-DA reference"

    # -----------------------------------------------------------------
    # Components - Engines
    # -----------------------------------------------------------------

    def test_components_engines_count(self, pack_yaml: Dict[str, Any]):
        """Test that components contain exactly 10 taxonomy engines."""
        components = pack_yaml.get("components", {})
        engines = components.get("engines", [])
        assert len(engines) == 10, (
            f"Expected 10 engines, found {len(engines)}"
        )

    # -----------------------------------------------------------------
    # Components - Workflows
    # -----------------------------------------------------------------

    def test_components_workflows_count(self, pack_yaml: Dict[str, Any]):
        """Test that components contain exactly 10 taxonomy workflows."""
        components = pack_yaml.get("components", {})
        workflows = components.get("workflows", [])
        assert len(workflows) == 10, (
            f"Expected 10 workflows, found {len(workflows)}"
        )

    # -----------------------------------------------------------------
    # Components - Templates
    # -----------------------------------------------------------------

    def test_components_templates_count(self, pack_yaml: Dict[str, Any]):
        """Test that components contain exactly 10 taxonomy templates."""
        components = pack_yaml.get("components", {})
        templates = components.get("templates", [])
        assert len(templates) == 10, (
            f"Expected 10 templates, found {len(templates)}"
        )

    # -----------------------------------------------------------------
    # Components - Integrations
    # -----------------------------------------------------------------

    def test_components_integrations_count(self, pack_yaml: Dict[str, Any]):
        """Test that components contain exactly 12 integrations."""
        components = pack_yaml.get("components", {})
        integrations = components.get("integrations", [])
        assert len(integrations) == 12, (
            f"Expected 12 integrations, found {len(integrations)}"
        )

    # -----------------------------------------------------------------
    # Agent sections
    # -----------------------------------------------------------------

    def test_agents_mrv_section_exists(self, pack_yaml: Dict[str, Any]):
        """Test that agents_mrv section exists with scope_1/scope_2/scope_3."""
        components = pack_yaml.get("components", {})
        agents_mrv = components.get("agents_mrv", {})
        assert isinstance(agents_mrv, dict), "agents_mrv must be a dictionary"
        assert "scope_1" in agents_mrv, "Missing scope_1 in agents_mrv"
        assert "scope_2" in agents_mrv, "Missing scope_2 in agents_mrv"
        assert "scope_3" in agents_mrv, "Missing scope_3 in agents_mrv"
        assert "cross_cutting" in agents_mrv, "Missing cross_cutting in agents_mrv"

        # Count total MRV agents: 8 (scope1) + 5 (scope2) + 15 (scope3) + 2 (cross) = 30
        total_mrv = 0
        for category, agents in agents_mrv.items():
            if isinstance(agents, list):
                total_mrv += len(agents)
        assert total_mrv == 30, f"Expected 30 MRV agents total, found {total_mrv}"

    def test_agents_data_section_exists(self, pack_yaml: Dict[str, Any]):
        """Test that agents_data section exists with 10 data agents."""
        components = pack_yaml.get("components", {})
        agents_data = components.get("agents_data", [])
        assert isinstance(agents_data, list), "agents_data must be a list"
        assert len(agents_data) == 10, (
            f"Expected 10 data agents, found {len(agents_data)}"
        )

    def test_agents_foundation_section_exists(self, pack_yaml: Dict[str, Any]):
        """Test that agents_foundation section exists with 10 foundation agents."""
        components = pack_yaml.get("components", {})
        agents_foundation = components.get("agents_foundation", [])
        assert isinstance(agents_foundation, list), "agents_foundation must be a list"
        assert len(agents_foundation) == 10, (
            f"Expected 10 foundation agents, found {len(agents_foundation)}"
        )

    # -----------------------------------------------------------------
    # Pack summary
    # -----------------------------------------------------------------

    def test_pack_summary_total_agents(self, pack_yaml: Dict[str, Any]):
        """Test pack summary shows 51 total agents."""
        summary = pack_yaml.get("pack_summary", {})
        total = summary.get("total_agents", 0)
        assert total == 51, f"Expected 51 total agents, got {total}"

    def test_pack_summary_counts(self, pack_yaml: Dict[str, Any]):
        """Test pack summary has correct sub-counts."""
        summary = pack_yaml.get("pack_summary", {})
        assert summary.get("taxonomy_app", 0) == 1, "Expected taxonomy_app = 1"
        assert summary.get("mrv_agents", 0) == 30, "Expected mrv_agents = 30"
        assert summary.get("data_agents", 0) == 10, "Expected data_agents = 10"
        assert summary.get("foundation_agents", 0) == 10, "Expected foundation_agents = 10"
        assert summary.get("pack_engines", 0) == 10, "Expected pack_engines = 10"
        assert summary.get("pack_workflows", 0) == 10, "Expected pack_workflows = 10"
        assert summary.get("pack_templates", 0) == 10, "Expected pack_templates = 10"
        assert summary.get("pack_integrations", 0) == 12, "Expected pack_integrations = 12"

    # -----------------------------------------------------------------
    # Presets and sectors
    # -----------------------------------------------------------------

    def test_presets_section(self, pack_yaml: Dict[str, Any]):
        """Test presets section contains size and sector presets."""
        presets = pack_yaml.get("presets", {})
        assert "size_presets" in presets, "Missing size_presets in presets"
        assert "sector_presets" in presets, "Missing sector_presets in presets"

        size_presets = presets["size_presets"]
        assert len(size_presets) == 5, (
            f"Expected 5 size presets, found {len(size_presets)}"
        )

        size_ids = [p.get("id", "") for p in size_presets]
        expected_size_ids = [
            "non_financial_undertaking",
            "financial_institution",
            "asset_manager",
            "large_enterprise",
            "sme_simplified",
        ]
        for expected_id in expected_size_ids:
            assert expected_id in size_ids, (
                f"Missing size preset: {expected_id}"
            )

    def test_sectors_section(self, pack_yaml: Dict[str, Any]):
        """Test sector presets include all 6 required sectors."""
        presets = pack_yaml.get("presets", {})
        sector_presets = presets.get("sector_presets", [])
        assert len(sector_presets) == 6, (
            f"Expected 6 sector presets, found {len(sector_presets)}"
        )

        sector_ids = [s.get("id", "") for s in sector_presets]
        expected_sector_ids = [
            "energy",
            "manufacturing",
            "real_estate",
            "transport",
            "forestry_agriculture",
            "financial_services",
        ]
        for expected_id in expected_sector_ids:
            assert expected_id in sector_ids, (
                f"Missing sector preset: {expected_id}"
            )

    # -----------------------------------------------------------------
    # Requirements
    # -----------------------------------------------------------------

    def test_requirements_section(self, pack_yaml: Dict[str, Any]):
        """Test requirements section includes runtime and infrastructure."""
        requirements = pack_yaml.get("requirements", {})
        assert "runtime" in requirements, "Missing runtime in requirements"
        assert "python_packages" in requirements, "Missing python_packages"
        assert "infrastructure" in requirements, "Missing infrastructure"
        assert "database" in requirements, "Missing database requirements"

        runtime = requirements["runtime"]
        assert "python" in runtime, "Missing python version requirement"
        assert "postgresql" in runtime, "Missing postgresql version requirement"

    # -----------------------------------------------------------------
    # Performance targets
    # -----------------------------------------------------------------

    def test_performance_targets(self, pack_yaml: Dict[str, Any]):
        """Test performance targets section exists with key benchmarks."""
        perf = pack_yaml.get("performance", {})
        assert "activity_screening" in perf, "Missing activity_screening target"
        assert "alignment_assessment" in perf, "Missing alignment_assessment target"
        assert "kpi_calculation" in perf, "Missing kpi_calculation target"
        assert "gar_calculation" in perf, "Missing gar_calculation target"
        assert "report_generation" in perf, "Missing report_generation target"
        assert "api_latency" in perf, "Missing api_latency target"
        assert "availability" in perf, "Missing availability target"

        # Validate specific performance numbers
        assert perf["activity_screening"].get("single_activity_max_ms", 0) > 0
        assert perf["api_latency"].get("p99_ms", 0) > 0
        assert perf["availability"].get("target_percent", 0) >= 99.0

    # -----------------------------------------------------------------
    # Engine ID uniqueness
    # -----------------------------------------------------------------

    def test_all_engine_ids_unique(self, pack_yaml: Dict[str, Any]):
        """Test that all engine IDs are unique (no duplicates)."""
        components = pack_yaml.get("components", {})
        engines = components.get("engines", [])
        engine_ids = [e.get("id", "") for e in engines]

        assert len(engine_ids) == len(set(engine_ids)), (
            f"Duplicate engine IDs found: "
            f"{[eid for eid in engine_ids if engine_ids.count(eid) > 1]}"
        )

    # -----------------------------------------------------------------
    # Regulation reference
    # -----------------------------------------------------------------

    def test_regulation_reference(self, pack_yaml: Dict[str, Any]):
        """Test regulation reference includes (EU) 2020/852."""
        meta = pack_yaml.get("metadata", {})
        reg = meta.get("regulation", {})
        ref = reg.get("reference", "")
        assert "2020/852" in ref, (
            f"Expected (EU) 2020/852 regulation reference, got {ref}"
        )

        # Verify environmental objectives count
        env_obj_count = reg.get("environmental_objectives", 0)
        assert env_obj_count == 6, (
            f"Expected 6 environmental objectives, got {env_obj_count}"
        )
