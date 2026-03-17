# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Manifest Tests
=================================================

Tests for pack.yaml structure: metadata, engines, workflows, templates,
integrations, presets, agent dependencies, performance targets, security
config, and regulatory references.

Target: 35+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

import re
from pathlib import Path

import pytest


# ===========================================================================
# Pack YAML Existence and Parse
# ===========================================================================


class TestPackYAMLFile:
    """Tests for pack.yaml file existence and parseability."""

    def test_pack_yaml_exists(self, pack_yaml_path):
        """pack.yaml file exists on disk."""
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_parseable(self, pack_yaml_data):
        """pack.yaml parses to a non-empty dictionary."""
        assert isinstance(pack_yaml_data, dict)
        assert len(pack_yaml_data) > 0

    def test_pack_yaml_has_required_keys(self, pack_yaml_data):
        """pack.yaml has metadata, components, and dependencies sections."""
        for key in ["metadata", "components", "dependencies"]:
            assert key in pack_yaml_data, f"Missing required key: {key}"


# ===========================================================================
# Metadata
# ===========================================================================


class TestPackMetadata:
    """Tests for pack.yaml metadata section."""

    def test_metadata_fields_present(self, pack_yaml_data):
        """Metadata section contains all required fields."""
        meta = pack_yaml_data.get("metadata", {})
        required = [
            "name", "version", "display_name", "description",
            "category", "tier", "author", "license",
        ]
        for field in required:
            assert field in meta, f"Missing metadata field: {field}"

    def test_pack_name_correct(self, pack_yaml_data):
        """Pack name is PACK-016-esrs-e1-climate."""
        meta = pack_yaml_data["metadata"]
        assert meta["name"] == "PACK-016-esrs-e1-climate"

    def test_pack_version_valid(self, pack_yaml_data):
        """Pack version follows semver pattern."""
        version = pack_yaml_data["metadata"]["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid version: {version}"

    def test_pack_version_is_1_0_0(self, pack_yaml_data):
        """Pack version is 1.0.0."""
        assert pack_yaml_data["metadata"]["version"] == "1.0.0"

    def test_pack_display_name_contains_e1(self, pack_yaml_data):
        """Display name contains E1 Climate."""
        meta = pack_yaml_data["metadata"]
        display = meta["display_name"]
        assert "E1" in display and "Climate" in display

    def test_pack_category(self, pack_yaml_data):
        """Category is eu-compliance."""
        assert pack_yaml_data["metadata"]["category"] == "eu-compliance"


# ===========================================================================
# Regulation References
# ===========================================================================


class TestRegulationReferences:
    """Tests for regulatory references in pack.yaml."""

    def test_regulation_references_present(self, pack_yaml_data):
        """Regulation section exists."""
        meta = pack_yaml_data["metadata"]
        assert "regulation" in meta

    def test_esrs_e1_referenced(self, pack_yaml_data):
        """ESRS E1 is the primary regulation."""
        reg = pack_yaml_data["metadata"]["regulation"]
        primary = reg.get("primary", {})
        assert "ESRS E1" in primary.get("reference", "")

    def test_csrd_referenced(self, pack_yaml_data):
        """CSRD is referenced in secondary regulations."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        csrd_found = any("CSRD" in s.get("name", "") or "2022/2464" in s.get("reference", "")
                         for s in secondary)
        assert csrd_found, "CSRD not found in secondary regulations"

    def test_ghg_protocol_referenced(self, pack_yaml_data):
        """GHG Protocol is referenced in secondary regulations."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        ghg_found = any("GHG Protocol" in s.get("name", "") for s in secondary)
        assert ghg_found, "GHG Protocol not found in secondary regulations"

    def test_sbti_referenced(self, pack_yaml_data):
        """SBTi is referenced in secondary regulations."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        sbti_found = any("SBTi" in s.get("name", "") or "SBTi" in s.get("reference", "")
                         for s in secondary)
        assert sbti_found, "SBTi not found in secondary regulations"

    def test_ipcc_ar6_referenced(self, pack_yaml_data):
        """IPCC AR6 is referenced in secondary regulations."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        ipcc_found = any("IPCC" in s.get("name", "") or "IPCC" in s.get("reference", "")
                         for s in secondary)
        assert ipcc_found, "IPCC AR6 not found in secondary regulations"


# ===========================================================================
# Components
# ===========================================================================


class TestComponents:
    """Tests for components section in pack.yaml."""

    def test_engines_listed(self, pack_yaml_data):
        """Components section lists 8 engines."""
        engines = pack_yaml_data["components"]["engines"]
        assert len(engines) == 8

    def test_workflows_listed(self, pack_yaml_data):
        """Components section lists 9 workflows."""
        workflows = pack_yaml_data["components"]["workflows"]
        assert len(workflows) == 9

    def test_templates_listed(self, pack_yaml_data):
        """Components section lists 9 templates."""
        templates = pack_yaml_data["components"]["templates"]
        assert len(templates) == 9

    def test_integrations_listed(self, pack_yaml_data):
        """Components section lists 8 integrations."""
        integrations = pack_yaml_data["components"]["integrations"]
        assert len(integrations) == 8

    def test_engine_ids_unique(self, pack_yaml_data):
        """Engine IDs are unique."""
        engines = pack_yaml_data["components"]["engines"]
        ids = [e["id"] for e in engines]
        assert len(ids) == len(set(ids))

    def test_ghg_inventory_engine_present(self, pack_yaml_data):
        """GHG inventory engine is present in components."""
        engines = pack_yaml_data["components"]["engines"]
        ids = [e["id"] for e in engines]
        assert "ghg_inventory" in ids

    def test_all_engines_have_required_fields(self, pack_yaml_data):
        """All engines have id, name, and description."""
        engines = pack_yaml_data["components"]["engines"]
        for engine in engines:
            assert "id" in engine, f"Engine missing id"
            assert "name" in engine, f"Engine {engine.get('id', '?')} missing name"
            assert "description" in engine, f"Engine {engine.get('id', '?')} missing description"


# ===========================================================================
# Dependencies
# ===========================================================================


class TestDependencies:
    """Tests for dependencies section in pack.yaml."""

    def test_pack_015_required(self, pack_yaml_data):
        """PACK-015 is listed as a required dependency."""
        deps = pack_yaml_data["dependencies"]
        required_packs = deps.get("required_packs", [])
        pack_015_found = any("PACK-015" in p.get("id", "") for p in required_packs)
        assert pack_015_found, "PACK-015 not in required_packs"

    def test_mrv_agents_referenced(self, pack_yaml_data):
        """AGENT-MRV agents are referenced."""
        agents_mrv = pack_yaml_data.get("agents_mrv", {})
        has_scope_1 = "scope_1" in agents_mrv
        has_scope_2 = "scope_2" in agents_mrv
        has_scope_3 = "scope_3" in agents_mrv
        assert has_scope_1 and has_scope_2 and has_scope_3

    def test_total_agents_70(self, pack_yaml_data):
        """Total agent count is 70."""
        deps = pack_yaml_data["dependencies"]
        assert deps.get("total_agents") == 70


# ===========================================================================
# Performance
# ===========================================================================


class TestPerformance:
    """Tests for performance targets in pack.yaml."""

    def test_performance_section_exists(self, pack_yaml_data):
        """Performance section exists."""
        assert "performance" in pack_yaml_data

    def test_ghg_inventory_targets_defined(self, pack_yaml_data):
        """GHG inventory performance targets are defined."""
        perf = pack_yaml_data["performance"]
        assert "ghg_inventory" in perf
        assert "max_emission_sources" in perf["ghg_inventory"]

    def test_memory_ceiling_defined(self, pack_yaml_data):
        """Memory ceiling is defined."""
        perf = pack_yaml_data["performance"]
        assert "memory_ceiling_mb" in perf
        assert perf["memory_ceiling_mb"] > 0


# ===========================================================================
# Security
# ===========================================================================


class TestSecurity:
    """Tests for security section in pack.yaml."""

    def test_security_section_exists(self, pack_yaml_data):
        """Security section exists."""
        assert "security" in pack_yaml_data

    def test_encryption_at_rest_specified(self, pack_yaml_data):
        """Encryption at rest is specified."""
        sec = pack_yaml_data["security"]
        assert "encryption_at_rest" in sec
        assert "AES" in sec["encryption_at_rest"]

    def test_encryption_in_transit_specified(self, pack_yaml_data):
        """Encryption in transit is specified."""
        sec = pack_yaml_data["security"]
        assert "encryption_in_transit" in sec
        assert "TLS" in sec["encryption_in_transit"]

    def test_authentication_specified(self, pack_yaml_data):
        """Authentication method is specified."""
        sec = pack_yaml_data["security"]
        assert "authentication" in sec
        assert "JWT" in sec["authentication"]
