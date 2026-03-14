# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Pack Manifest Tests (20 tests)

Validates the pack.yaml manifest structure, metadata, dependencies,
components, workflows, templates, integrations, and presets.

Author: GreenLang QA Team
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


class TestPackManifest:
    """Test suite for pack.yaml manifest validation."""

    def test_manifest_exists(self, pack_yaml_path: Path):
        """Test that pack.yaml file exists at the expected location."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"
        assert pack_yaml_path.is_file()

    def test_manifest_yaml_valid(self, pack_yaml_raw: str):
        """Test that pack.yaml is valid YAML."""
        parsed = yaml.safe_load(pack_yaml_raw)
        assert isinstance(parsed, dict), "pack.yaml should parse to a dict"
        assert len(parsed) > 0, "pack.yaml should not be empty"

    def test_manifest_metadata(self, pack_yaml: Dict[str, Any]):
        """Test manifest metadata section contains required fields."""
        assert "metadata" in pack_yaml
        meta = pack_yaml["metadata"]
        assert meta["name"] == "csrd-enterprise"
        assert meta["tier"] == "enterprise"
        assert meta["category"] == "eu-compliance"
        assert "display_name" in meta
        assert "description" in meta
        assert "author" in meta

    def test_manifest_version(self, pack_yaml: Dict[str, Any]):
        """Test manifest version is a valid semantic version."""
        version = pack_yaml["metadata"]["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"Version '{version}' is not valid semver"
        )

    def test_manifest_extends_pack_002(self, pack_yaml: Dict[str, Any]):
        """Test manifest declares dependency on PACK-002."""
        assert "dependencies" in pack_yaml
        deps = pack_yaml["dependencies"]
        pack_002_found = any(
            d.get("pack_id") == "PACK-002" or d.get("name") == "csrd-professional"
            for d in deps
        )
        assert pack_002_found, "PACK-003 must extend PACK-002"

    def test_manifest_components_count(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists enterprise components with correct structure."""
        assert "components" in pack_yaml
        components = pack_yaml["components"]
        assert "inherited" in components
        assert components["inherited"]["agent_count"] >= 90, (
            "Should inherit 90+ agents from PACK-002"
        )

    def test_manifest_engines_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists all 10 enterprise engines."""
        components = pack_yaml["components"]
        engine_groups = [
            "predictive_engines", "narrative_engines", "iot_engines",
            "carbon_credit_engines", "supply_chain_engines", "filing_engines",
            "workflow_builder_engines",
        ]
        total_engines = 0
        for group in engine_groups:
            if group in components:
                engines = components[group]
                assert isinstance(engines, list), f"{group} should be a list"
                total_engines += len(engines)
        assert total_engines >= 10, f"Expected 10+ engines, found {total_engines}"

    def test_manifest_workflows_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 8 enterprise workflows."""
        assert "workflows" in pack_yaml
        workflows = pack_yaml["workflows"]
        enterprise_workflows = [
            k for k in workflows.keys()
            if k not in ("inherited",) and isinstance(workflows[k], dict)
        ]
        assert len(enterprise_workflows) >= 8, (
            f"Expected 8+ enterprise workflows, found {len(enterprise_workflows)}"
        )

    def test_manifest_templates_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 9 enterprise templates."""
        assert "templates" in pack_yaml
        templates = pack_yaml["templates"]
        template_list = [t for t in templates if isinstance(t, dict)]
        assert len(template_list) >= 9, (
            f"Expected 9+ enterprise templates, found {len(template_list)}"
        )

    def test_manifest_integrations_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists 9 enterprise integrations."""
        assert "integrations" in pack_yaml
        integrations = pack_yaml["integrations"]
        int_list = [i for i in integrations if isinstance(i, dict)]
        assert len(int_list) >= 9, (
            f"Expected 9+ enterprise integrations, found {len(int_list)}"
        )

    def test_manifest_platform_bridges(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists platform bridges."""
        assert "platform_bridges" in pack_yaml
        bridges = pack_yaml["platform_bridges"]
        assert len(bridges) >= 10, (
            f"Expected 10+ platform bridges, found {len(bridges)}"
        )

    def test_manifest_config_defaults(self, pack_yaml: Dict[str, Any]):
        """Test manifest contains performance and requirements sections."""
        assert "performance" in pack_yaml
        assert "requirements" in pack_yaml
        perf = pack_yaml["performance"]
        assert "data_ingestion" in perf
        assert "ghg_calculation" in perf
        assert "predictive_analytics" in perf

    def test_manifest_compliance_refs(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists compliance references."""
        refs = pack_yaml["metadata"].get("compliance_references", [])
        assert len(refs) >= 5, f"Expected 5+ compliance refs, found {len(refs)}"
        ref_ids = {r["id"] for r in refs}
        assert "CSRD" in ref_ids
        assert "ESRS" in ref_ids

    def test_manifest_presets_listed(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists size and sector presets."""
        assert "presets" in pack_yaml
        presets = pack_yaml["presets"]
        assert "size_presets" in presets
        assert "sector_presets" in presets
        assert len(presets["size_presets"]) >= 4
        assert len(presets["sector_presets"]) >= 5

    def test_manifest_sectors_listed(self, pack_yaml: Dict[str, Any]):
        """Test all 5 sector presets have correct IDs."""
        sector_ids = {
            s["id"] for s in pack_yaml["presets"]["sector_presets"]
        }
        expected = {
            "banking_enterprise", "oil_gas_enterprise",
            "automotive_enterprise", "pharma_enterprise", "conglomerate",
        }
        assert expected.issubset(sector_ids), (
            f"Missing sector presets: {expected - sector_ids}"
        )

    def test_manifest_dependencies(self, pack_yaml: Dict[str, Any]):
        """Test all dependencies have required fields."""
        for dep in pack_yaml["dependencies"]:
            assert "pack_id" in dep or "name" in dep
            assert "version" in dep or "required" in dep

    def test_manifest_security_requirements(self, pack_yaml: Dict[str, Any]):
        """Test manifest runtime requirements include Python 3.11+."""
        reqs = pack_yaml["requirements"]["runtime"]
        assert reqs["python"] == ">=3.11"
        assert reqs["postgresql"] == ">=16"

    def test_manifest_performance_targets(self, pack_yaml: Dict[str, Any]):
        """Test performance targets include predictive analytics targets."""
        perf = pack_yaml["performance"]
        assert "predictive_analytics" in perf
        pa = perf["predictive_analytics"]
        assert pa["forecast_max_seconds"] <= 120
        assert pa["anomaly_detection_max_seconds"] <= 60

    def test_manifest_tenant_config(self, pack_yaml: Dict[str, Any]):
        """Test manifest lists tenant onboarding workflow."""
        workflows = pack_yaml["workflows"]
        assert "tenant_onboarding" in workflows
        tenant_wf = workflows["tenant_onboarding"]
        assert "phases" in tenant_wf
        assert len(tenant_wf["phases"]) >= 2

    def test_manifest_provenance_hash(self, pack_yaml_raw: str):
        """Test that we can compute a provenance hash of the manifest."""
        manifest_hash = hashlib.sha256(pack_yaml_raw.encode("utf-8")).hexdigest()
        assert len(manifest_hash) == 64
        assert all(c in "0123456789abcdef" for c in manifest_hash)

    def test_manifest_schema_valid(self, pack_yaml: Dict[str, Any]):
        """Test manifest has all top-level sections."""
        required_sections = [
            "metadata", "dependencies", "components",
            "workflows", "templates", "integrations",
            "presets", "requirements", "performance",
        ]
        for section in required_sections:
            assert section in pack_yaml, f"Missing section: {section}"
