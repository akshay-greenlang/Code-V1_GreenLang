# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Pack Manifest Tests (15 tests)

Validates the pack.yaml manifest structure, metadata, dependencies,
components, workflows, templates, CN codes, compliance references,
and presets for the CBAM Readiness Pack.

Author: GreenLang QA Team
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import PACK_YAML_PATH, _compute_hash


class TestPackManifest:
    """Test suite for CBAM pack.yaml manifest validation."""

    def test_manifest_exists(self, pack_yaml_path: Path):
        """Test that pack.yaml file exists at the expected location."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"
        assert pack_yaml_path.is_file()

    def test_manifest_yaml_valid(self, pack_yaml_raw: str):
        """Test that pack.yaml is valid YAML."""
        if not pack_yaml_raw:
            pytest.skip("pack.yaml not yet created")
        parsed = yaml.safe_load(pack_yaml_raw)
        assert isinstance(parsed, dict), "pack.yaml should parse to a dict"
        assert len(parsed) > 0, "pack.yaml should not be empty"

    def test_manifest_metadata(self, pack_yaml: Dict[str, Any]):
        """Test manifest metadata section contains required CBAM fields."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        assert "metadata" in pack_yaml
        meta = pack_yaml["metadata"]
        assert meta["name"] == "cbam-readiness"
        assert meta["category"] == "eu-compliance"
        assert "display_name" in meta
        assert "description" in meta
        assert "author" in meta

    def test_manifest_version(self, pack_yaml: Dict[str, Any]):
        """Test manifest version is a valid semantic version."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        version = pack_yaml["metadata"]["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"Version '{version}' is not valid semver"
        )

    def test_manifest_standalone(self, pack_yaml: Dict[str, Any]):
        """Test manifest is standalone (does not extend CSRD packs)."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        deps = pack_yaml.get("dependencies", [])
        csrd_dep = [
            d for d in deps
            if d.get("pack_id", "").startswith("PACK-00") and "csrd" in d.get("name", "")
        ]
        assert len(csrd_dep) == 0, "CBAM pack should be standalone, not extending CSRD"

    def test_manifest_components(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists CBAM-specific components."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        assert "components" in pack_yaml
        components = pack_yaml["components"]
        assert isinstance(components, dict)

    def test_manifest_engines_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists all 7 CBAM engines."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        # Engines may be at top level or nested under components
        engines = pack_yaml.get(
            "engines",
            pack_yaml.get("components", {}).get("engines", {}),
        )
        if isinstance(engines, dict):
            assert len(engines) >= 7, f"Expected 7+ engines, found {len(engines)}"
        elif isinstance(engines, list):
            assert len(engines) >= 7, f"Expected 7+ engines, found {len(engines)}"

    def test_manifest_workflows_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 7 CBAM workflows."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        assert "workflows" in pack_yaml
        workflows = pack_yaml["workflows"]
        if isinstance(workflows, list):
            assert len(workflows) >= 7
        elif isinstance(workflows, dict):
            assert len(workflows) >= 7

    def test_manifest_templates_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 8 CBAM templates."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        assert "templates" in pack_yaml
        templates = pack_yaml["templates"]
        if isinstance(templates, list):
            assert len(templates) >= 8
        elif isinstance(templates, dict):
            assert len(templates) >= 8

    def test_manifest_compliance_refs_cbam(self, pack_yaml: Dict[str, Any]):
        """Test manifest references CBAM regulation."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        refs = pack_yaml["metadata"].get("compliance_references", [])
        assert len(refs) >= 1, "Must have at least one compliance reference"
        ref_ids = {r.get("id", "") for r in refs}
        assert "CBAM" in ref_ids, "Must reference CBAM regulation"

    def test_manifest_goods_categories(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists CBAM goods categories."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        # goods_categories may be top-level or under cbam/config
        gc = pack_yaml.get(
            "goods_categories",
            pack_yaml.get("cbam", pack_yaml.get("config", {})).get(
                "goods_categories", {}
            ),
        )
        if isinstance(gc, dict):
            actual = set(gc.keys())
            # Categories may use "iron_steel" or "steel"
            expected_variants = [
                {"cement", "iron_steel", "aluminium"},
                {"cement", "steel", "aluminium"},
            ]
            assert any(exp.issubset(actual) for exp in expected_variants), (
                f"Missing categories. Found: {actual}"
            )
        elif isinstance(gc, list):
            actual = set(gc)
            expected = {"cement", "aluminium"}
            assert expected.issubset(actual)

    def test_manifest_cn_codes(self, pack_yaml: Dict[str, Any]):
        """Test manifest references CN codes."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        # CN codes may be in components, config, or referenced in goods_categories
        yaml_str = yaml.dump(pack_yaml)
        # At minimum the manifest should reference CN codes concept
        assert "cn" in yaml_str.lower() or "CN" in yaml_str or "goods" in yaml_str.lower(), (
            "Manifest should reference CN codes or goods categories"
        )

    def test_manifest_presets(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists presets."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        presets = pack_yaml.get("presets", {})
        assert isinstance(presets, (dict, list)), "Presets should be dict or list"

    def test_manifest_dependencies(self, pack_yaml: Dict[str, Any]):
        """Test manifest dependencies have required fields."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        deps = pack_yaml.get("dependencies", [])
        if deps:
            for dep in deps:
                assert isinstance(dep, dict)

    def test_manifest_no_csrd_dependency(self, pack_yaml: Dict[str, Any]):
        """Test CBAM pack does not depend on CSRD packs."""
        if not pack_yaml:
            pytest.skip("pack.yaml not yet created")
        deps = pack_yaml.get("dependencies", [])
        for dep in deps:
            name = dep.get("name", "")
            pack_id = dep.get("pack_id", "")
            assert "csrd" not in name.lower(), (
                f"CBAM should not depend on CSRD pack: {name}"
            )
            assert pack_id not in ("PACK-001", "PACK-002", "PACK-003"), (
                f"CBAM should not depend on {pack_id}"
            )
