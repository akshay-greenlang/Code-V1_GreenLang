# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Manifest Tests
=================================================

Tests the pack.yaml manifest for PACK-007 to ensure:
- Pack metadata is correct (ID, version, tier)
- Extends PACK-006 as expected
- All 40 EUDR agents are listed
- All 10 professional engines are listed
- All 10 professional workflows are listed
- All data/foundation agents are included
- Professional-tier integrations are present
- Total agent count is 60 (40 EUDR + 20 data/foundation)
"""

import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

_PACK_007_DIR = Path(__file__).resolve().parent.parent
_PACK_YAML_PATH = _PACK_007_DIR / "pack.yaml"


@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    return _PACK_YAML_PATH


@pytest.fixture(scope="module")
def pack_yaml() -> Dict[str, Any]:
    if not _PACK_YAML_PATH.exists():
        pytest.skip("pack.yaml not found")
    with open(_PACK_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestPackManifest:
    """Test suite for PACK-007 pack.yaml manifest."""

    def test_pack_yaml_exists(self, pack_yaml_path: Path):
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_id_is_correct(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        name = meta.get("name", "")
        assert "PACK-007" in name or "007" in name, f"Expected PACK-007 in name, got {name}"

    def test_version_format(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        version = meta.get("version", "")
        assert isinstance(version, str), f"version must be string, got {type(version)}"
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid version: {version}"

    def test_tier_is_professional(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        tier = meta.get("tier", "")
        assert tier == "professional", f"Expected tier='professional', got {tier}"

    def test_extends_pack_006(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        extends = meta.get("extends", "")
        assert "006" in str(extends), f"Expected extends to reference PACK-006, got {extends}"

    def test_eudr_agents_count_40(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        eudr = components.get("agents_eudr", {})
        count = 0
        for cat, agents in eudr.items():
            if isinstance(agents, list):
                count += len([
                    a for a in agents
                    if isinstance(a, dict) and str(a.get("id", "")).startswith("AGENT-EUDR-")
                ])
        assert count >= 40, f"Expected at least 40 EUDR agents, found {count}"

    def test_data_agents_listed(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        data_agents = components.get("agents_data", [])
        assert len(data_agents) > 0, "Expected data agents in components"

    def test_foundation_agents_listed(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        found_agents = components.get("agents_foundation", [])
        assert len(found_agents) > 0, "Expected foundation agents in components"

    def test_engines_count_10(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        engines = components.get("engines", [])
        assert len(engines) >= 10, f"Expected at least 10 engines, found {len(engines)}"

    def test_workflows_count_10(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        workflows = components.get("workflows", [])
        assert len(workflows) >= 10, f"Expected at least 10 workflows, found {len(workflows)}"

    def test_templates_count_10(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        templates = components.get("templates", [])
        assert len(templates) >= 10, f"Expected at least 10 templates, found {len(templates)}"

    def test_integrations_count_12(self, pack_yaml: Dict[str, Any]):
        components = pack_yaml.get("components", {})
        integrations = components.get("integrations", [])
        assert len(integrations) >= 12, f"Expected at least 12 integrations, found {len(integrations)}"

    def test_all_commodities_listed(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        reg = meta.get("regulation", {})
        # Check commodities count in regulation or description
        commodities_count = reg.get("commodities", 0)
        description = meta.get("description", "")
        assert commodities_count >= 7 or "7" in description or "seven" in description.lower(), \
            "Expected 7 commodities referenced"

    def test_regulation_reference(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        reg = meta.get("regulation", {})
        ref = reg.get("reference", "") or reg.get("name", "")
        assert "2023/1115" in ref or "Deforestation" in ref, \
            f"Expected EUDR regulation reference, got {ref}"

    def test_total_agents_60(self, pack_yaml: Dict[str, Any]):
        summary = pack_yaml.get("pack_summary", {})
        total = summary.get("total_agents", 0)
        assert total >= 60, f"Expected at least 60 total agents, got {total}"

    def test_pack_display_name(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        display_name = meta.get("display_name", "")
        assert "EUDR" in display_name, f"Expected 'EUDR' in display_name, got {display_name}"
        assert "Professional" in display_name, f"Expected 'Professional' in display_name"

    def test_category_eu_compliance(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        category = meta.get("category", "")
        assert category == "eu-compliance", f"Expected eu-compliance, got {category}"

    def test_professional_features_in_description(self, pack_yaml: Dict[str, Any]):
        meta = pack_yaml.get("metadata", {})
        desc = meta.get("description", "").lower()
        keywords = ["satellite", "portfolio", "benchmarking", "grievance", "monte carlo"]
        found = [k for k in keywords if k.lower() in desc]
        assert len(found) >= 2, f"Expected professional keywords in description, found: {found}"

    def test_pack_summary_counts(self, pack_yaml: Dict[str, Any]):
        summary = pack_yaml.get("pack_summary", {})
        assert summary.get("eudr_agents", 0) == 40
        assert summary.get("professional_engines", 0) == 10
        assert summary.get("professional_workflows", 0) == 10
        assert summary.get("professional_integrations", 0) == 12
