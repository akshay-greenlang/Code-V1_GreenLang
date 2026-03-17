# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Manifest Tests
==============================================================

Tests for pack.yaml structure: metadata, engines, workflows,
templates, integrations, presets, agent dependencies, performance
targets, security config, and regulatory references.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
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
        """Pack name matches PACK-015-double-materiality."""
        meta = pack_yaml_data["metadata"]
        assert meta["name"] == "PACK-015-double-materiality"

    def test_pack_version_valid(self, pack_yaml_data):
        """Pack version follows semver pattern."""
        version = pack_yaml_data["metadata"]["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Invalid version: {version}"

    def test_pack_display_name(self, pack_yaml_data):
        """Display name is set."""
        meta = pack_yaml_data["metadata"]
        assert "Double Materiality" in meta["display_name"]

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

    def test_esrs_1_referenced(self, pack_yaml_data):
        """ESRS 1 is the primary regulation."""
        reg = pack_yaml_data["metadata"]["regulation"]
        primary = reg.get("primary", {})
        assert "ESRS 1" in primary.get("reference", "")

    def test_csrd_referenced(self, pack_yaml_data):
        """CSRD is referenced in secondary regulations."""
        reg = pack_yaml_data["metadata"]["regulation"]
        secondary = reg.get("secondary", [])
        refs = [r.get("reference", "") for r in secondary]
        assert any("2022/2464" in ref for ref in refs), "CSRD directive not found"

    def test_compliance_references(self, pack_yaml_data):
        """Compliance references section exists and is non-empty."""
        meta = pack_yaml_data["metadata"]
        refs = meta.get("compliance_references", [])
        assert len(refs) >= 5


# ===========================================================================
# Tags
# ===========================================================================


class TestTags:
    """Tests for tags in pack.yaml."""

    def test_tags_comprehensive(self, pack_yaml_data):
        """Tags include key DMA-related terms."""
        tags = pack_yaml_data["metadata"].get("tags", [])
        assert len(tags) >= 15
        expected_tags = [
            "double-materiality", "dma", "esrs", "csrd",
            "impact-materiality", "financial-materiality", "iro",
        ]
        for tag in expected_tags:
            assert tag in tags, f"Missing tag: {tag}"


# ===========================================================================
# Components - Engines
# ===========================================================================


class TestEngines:
    """Tests for engines section in pack.yaml."""

    def test_engines_count_8(self, pack_yaml_data):
        """Exactly 8 engines are defined."""
        engines = pack_yaml_data["components"]["engines"]
        assert len(engines) == 8

    def test_all_engine_ids_present(self, pack_yaml_data):
        """All 8 engine IDs are present."""
        engines = pack_yaml_data["components"]["engines"]
        engine_ids = {e["id"] for e in engines}
        expected = {
            "impact_materiality", "financial_materiality",
            "stakeholder_engagement", "iro_identification",
            "materiality_matrix", "esrs_topic_mapping",
            "threshold_scoring", "dma_report",
        }
        assert engine_ids == expected

    def test_engines_have_descriptions(self, pack_yaml_data):
        """Every engine has a non-empty description."""
        for engine in pack_yaml_data["components"]["engines"]:
            assert engine.get("description"), f"Engine {engine['id']} missing description"

    def test_engines_have_required_flag(self, pack_yaml_data):
        """Every engine has a required flag."""
        for engine in pack_yaml_data["components"]["engines"]:
            assert "required" in engine, f"Engine {engine['id']} missing required field"


# ===========================================================================
# Components - Workflows
# ===========================================================================


class TestWorkflows:
    """Tests for workflows section in pack.yaml."""

    def test_workflows_count_8(self, pack_yaml_data):
        """Exactly 8 workflows are defined."""
        workflows = pack_yaml_data["components"]["workflows"]
        assert len(workflows) == 8

    def test_all_workflow_ids_present(self, pack_yaml_data):
        """All 8 workflow IDs are present."""
        workflows = pack_yaml_data["components"]["workflows"]
        wf_ids = {w["id"] for w in workflows}
        expected = {
            "impact_assessment", "financial_assessment",
            "stakeholder_engagement", "iro_identification",
            "materiality_matrix", "esrs_mapping",
            "full_dma", "dma_update",
        }
        assert wf_ids == expected

    def test_workflows_have_phases(self, pack_yaml_data):
        """Every workflow declares a phase count."""
        for wf in pack_yaml_data["components"]["workflows"]:
            assert "phases" in wf, f"Workflow {wf['id']} missing phases"
            assert wf["phases"] >= 3


# ===========================================================================
# Components - Templates
# ===========================================================================


class TestTemplates:
    """Tests for templates section in pack.yaml."""

    def test_templates_count_8(self, pack_yaml_data):
        """Exactly 8 templates are defined."""
        templates = pack_yaml_data["components"]["templates"]
        assert len(templates) == 8

    def test_all_template_ids_present(self, pack_yaml_data):
        """All 8 template IDs are present."""
        templates = pack_yaml_data["components"]["templates"]
        tmpl_ids = {t["id"] for t in templates}
        expected = {
            "impact_materiality_report", "financial_materiality_report",
            "stakeholder_engagement_report", "materiality_matrix_report",
            "iro_register", "esrs_mapping_report",
            "executive_summary", "audit_trail_report",
        }
        assert tmpl_ids == expected


# ===========================================================================
# Components - Integrations
# ===========================================================================


class TestIntegrations:
    """Tests for integrations section in pack.yaml."""

    def test_integrations_present(self, pack_yaml_data):
        """Integrations section exists and has entries."""
        integrations = pack_yaml_data["components"]["integrations"]
        assert len(integrations) >= 8

    def test_orchestrator_present(self, pack_yaml_data):
        """pack_orchestrator integration exists."""
        integrations = pack_yaml_data["components"]["integrations"]
        ids = {i["id"] for i in integrations}
        assert "pack_orchestrator" in ids

    def test_health_check_present(self, pack_yaml_data):
        """health_check integration exists."""
        integrations = pack_yaml_data["components"]["integrations"]
        ids = {i["id"] for i in integrations}
        assert "health_check" in ids


# ===========================================================================
# Presets
# ===========================================================================


class TestPresets:
    """Tests for presets section in pack.yaml."""

    def test_presets_count_6(self, pack_yaml_data):
        """Exactly 6 presets are defined."""
        presets = pack_yaml_data["components"]["presets"]
        assert len(presets) == 6

    def test_preset_ids(self, pack_yaml_data):
        """All 6 preset IDs present."""
        presets = pack_yaml_data["components"]["presets"]
        preset_ids = {p["id"] for p in presets}
        expected = {
            "large_enterprise", "mid_market", "sme",
            "financial_services", "manufacturing", "multi_sector",
        }
        assert preset_ids == expected


# ===========================================================================
# Agent Dependencies
# ===========================================================================


class TestAgentDependencies:
    """Tests for agent dependencies in pack.yaml."""

    def test_agent_dependencies_present(self, pack_yaml_data):
        """Dependencies section exists."""
        assert "dependencies" in pack_yaml_data

    def test_mrv_agents_30(self, pack_yaml_data):
        """30 MRV agents declared."""
        deps = pack_yaml_data["dependencies"]
        assert deps["breakdown"]["mrv_agents"] == 30

    def test_data_agents_20(self, pack_yaml_data):
        """20 DATA agents declared."""
        deps = pack_yaml_data["dependencies"]
        assert deps["breakdown"]["data_agents"] == 20

    def test_foundation_agents_10(self, pack_yaml_data):
        """10 FOUND agents declared."""
        deps = pack_yaml_data["dependencies"]
        assert deps["breakdown"]["foundation_agents"] == 10

    def test_total_agents(self, pack_yaml_data):
        """Total agents equals 72."""
        deps = pack_yaml_data["dependencies"]
        assert deps["total_agents"] == 72


# ===========================================================================
# Performance and Security
# ===========================================================================


class TestPerformanceAndSecurity:
    """Tests for performance targets and security config in pack.yaml."""

    def test_performance_targets_present(self, pack_yaml_data):
        """Performance section exists."""
        assert "performance" in pack_yaml_data
        perf = pack_yaml_data["performance"]
        assert "full_dma" in perf

    def test_security_config_present(self, pack_yaml_data):
        """Security section exists."""
        assert "security" in pack_yaml_data
        sec = pack_yaml_data["security"]
        assert sec["authentication"] == "JWT (RS256)"
        assert sec["encryption_at_rest"] == "AES-256-GCM"

    def test_requirements_present(self, pack_yaml_data):
        """Requirements section exists with Python version."""
        assert "requirements" in pack_yaml_data
        reqs = pack_yaml_data["requirements"]
        assert "runtime" in reqs
        assert "python" in reqs["runtime"]
