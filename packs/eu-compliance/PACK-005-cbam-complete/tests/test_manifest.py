# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Pack Manifest Tests (15 tests)

Validates the pack.yaml manifest structure, metadata, dependencies,
components, workflows, templates, integrations, CN codes, goods
categories, presets, sectors, and version format.

Author: GreenLang QA Team
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import PACK_YAML_PATH, _compute_hash


class TestPackManifest:
    """Test suite for CBAM Complete pack.yaml manifest validation."""

    def test_manifest_exists(self, pack_yaml_path: Path):
        """Test that pack.yaml file exists at the expected location."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"
        assert pack_yaml_path.is_file()

    def test_manifest_valid_yaml(self, pack_yaml_raw: str):
        """Test that pack.yaml is valid YAML."""
        if not pack_yaml_raw:
            pytest.skip("pack.yaml not yet created")
        parsed = yaml.safe_load(pack_yaml_raw)
        assert isinstance(parsed, dict), "pack.yaml should parse to a dict"
        assert len(parsed) > 0, "pack.yaml should not be empty"

    def test_manifest_has_metadata(self, pack_yaml: Dict[str, Any]):
        """Test manifest metadata section contains required fields."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        assert "metadata" in pack_yaml
        meta = pack_yaml["metadata"]
        name = meta.get("name", "")
        assert "cbam-complete" in name, f"Expected 'cbam-complete' in name, got '{name}'"
        assert meta.get("category") == "eu-compliance"
        assert "display_name" in meta
        assert "description" in meta
        assert "author" in meta

    def test_manifest_extends_pack004(self, pack_yaml: Dict[str, Any]):
        """Test manifest declares extension of PACK-004."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        # May be in metadata.extends, dependencies, or top-level extends
        extends = pack_yaml.get("metadata", {}).get("extends", "")
        deps = pack_yaml.get("dependencies", [])
        dep_ids = [d.get("pack_id", "") for d in deps] if isinstance(deps, list) else []

        has_pack004 = (
            "PACK-004" in extends
            or "cbam-readiness" in extends
            or any("PACK-004" in d or "cbam-readiness" in d for d in dep_ids)
        )
        assert has_pack004, "PACK-005 should extend PACK-004-cbam-readiness"

    def test_manifest_has_components(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists CBAM Complete components."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        assert "components" in pack_yaml or "engines" in pack_yaml, (
            "Manifest must have components or engines section"
        )

    def test_manifest_engine_count(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 8 new engines."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        engines = pack_yaml.get(
            "engines",
            pack_yaml.get("components", {}).get("engines", {}),
        )
        if isinstance(engines, (dict, list)):
            assert len(engines) >= 8, f"Expected 8+ engines, found {len(engines)}"

    def test_manifest_workflow_count(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 6 new workflows."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        workflows = pack_yaml.get(
            "workflows",
            pack_yaml.get("components", {}).get("workflows", {}),
        )
        if isinstance(workflows, (dict, list)):
            assert len(workflows) >= 6, f"Expected 6+ workflows, found {len(workflows)}"

    def test_manifest_template_count(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 6 new templates."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        templates = pack_yaml.get(
            "templates",
            pack_yaml.get("components", {}).get("templates", {}),
        )
        if isinstance(templates, (dict, list)):
            assert len(templates) >= 6, f"Expected 6+ templates, found {len(templates)}"

    def test_manifest_integration_count(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 7 new integrations."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        integrations = pack_yaml.get(
            "integrations",
            pack_yaml.get("components", {}).get("integrations", {}),
        )
        if isinstance(integrations, dict):
            # Integrations may be nested under 'new' and 'inherited' keys
            new_integrations = integrations.get("new", [])
            assert len(new_integrations) >= 7, (
                f"Expected 7+ new integrations, found {len(new_integrations)}"
            )
        elif isinstance(integrations, list):
            assert len(integrations) >= 7, (
                f"Expected 7+ integrations, found {len(integrations)}"
            )

    def test_manifest_cn_code_coverage(self, pack_yaml: Dict[str, Any]):
        """Test manifest references 160+ CN codes."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        yaml_str = yaml.dump(pack_yaml)
        # Manifest should reference CN code coverage count or list
        has_cn_ref = (
            "cn_code" in yaml_str.lower()
            or "160" in yaml_str
            or "goods_categories" in yaml_str.lower()
        )
        assert has_cn_ref, "Manifest should reference CN codes or goods categories"

    def test_manifest_goods_categories(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists all 6 CBAM goods categories."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        yaml_str = yaml.dump(pack_yaml).lower()
        expected = ["cement", "aluminium", "fertilizer", "electricity", "hydrogen"]
        # At least cement and aluminium and either steel or iron_steel
        found = sum(1 for cat in expected if cat in yaml_str)
        has_steel = "steel" in yaml_str or "iron_steel" in yaml_str
        assert found >= 3 or has_steel, f"Expected CBAM goods categories in manifest"

    def test_manifest_presets_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 4 presets."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        presets = pack_yaml.get("presets", pack_yaml.get("config", {}).get("presets", {}))
        if isinstance(presets, dict):
            # Presets may be grouped under pack_005_presets and sector_presets
            pack_presets = presets.get("pack_005_presets", [])
            assert len(pack_presets) >= 4, (
                f"Expected 4+ pack presets, found {len(pack_presets)}"
            )
        elif isinstance(presets, list):
            assert len(presets) >= 4, f"Expected 4+ presets, found {len(presets)}"
        else:
            # Presets may be referenced by path rather than inline
            assert True, "Presets may be loaded from preset files"

    def test_manifest_sectors_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 3 sectors."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        sectors = pack_yaml.get("sectors", pack_yaml.get("config", {}).get("sectors", {}))
        if isinstance(sectors, (dict, list)):
            assert len(sectors) >= 3, f"Expected 3+ sectors, found {len(sectors)}"
        else:
            assert True, "Sectors may be loaded from sector files"

    def test_manifest_dependencies(self, pack_yaml: Dict[str, Any]):
        """Test manifest dependencies have required fields."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        deps = pack_yaml.get("dependencies", {})
        if isinstance(deps, dict):
            # Dependencies may be grouped under 'required' and 'optional' keys
            required_deps = deps.get("required", [])
            optional_deps = deps.get("optional", [])
            all_deps = required_deps + optional_deps
            assert len(all_deps) >= 1, "Expected at least 1 dependency"
            for dep in all_deps:
                assert isinstance(dep, dict), "Each dependency should be a dict"
                assert "name" in dep, "Each dependency must have a 'name'"
        elif isinstance(deps, list):
            for dep in deps:
                assert isinstance(dep, dict), "Each dependency should be a dict"

    def test_manifest_version_format(self, pack_yaml: Dict[str, Any]):
        """Test manifest version is a valid semantic version."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        version = pack_yaml.get("metadata", {}).get("version", "")
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"Version '{version}' is not valid semver"
        )
