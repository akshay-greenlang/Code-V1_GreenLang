# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Manifest (pack.yaml) Tests
============================================================

Tests the pack manifest YAML file for structural validity, required
sections, component counts, and metadata completeness.

Test count target: ~25 tests
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import MANIFEST_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    """Load pack.yaml once for all manifest tests."""
    assert MANIFEST_PATH.exists(), f"pack.yaml not found at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data is not None, "pack.yaml parsed as empty"
    return data


# ---------------------------------------------------------------------------
# 1. YAML validity
# ---------------------------------------------------------------------------


class TestManifestValidity:
    """Verify pack.yaml is valid YAML and parses correctly."""

    def test_yaml_file_exists(self):
        assert MANIFEST_PATH.exists()

    def test_yaml_parses(self, manifest):
        assert isinstance(manifest, dict)

    def test_yaml_not_empty(self, manifest):
        assert len(manifest) > 0


# ---------------------------------------------------------------------------
# 2. Metadata section
# ---------------------------------------------------------------------------


class TestManifestMetadata:
    """Verify metadata section is complete."""

    def test_metadata_exists(self, manifest):
        assert "metadata" in manifest

    def test_metadata_name(self, manifest):
        meta = manifest["metadata"]
        assert meta["name"] == "PACK-019-csddd-readiness"

    def test_metadata_version(self, manifest):
        meta = manifest["metadata"]
        assert "version" in meta
        assert meta["version"] == "1.0.0"

    def test_metadata_display_name(self, manifest):
        meta = manifest["metadata"]
        assert "display_name" in meta
        assert "CSDDD" in meta["display_name"]

    def test_metadata_description(self, manifest):
        meta = manifest["metadata"]
        assert "description" in meta
        assert len(meta["description"]) > 50

    def test_metadata_category(self, manifest):
        meta = manifest["metadata"]
        assert meta.get("category") == "eu-compliance"

    def test_metadata_author(self, manifest):
        meta = manifest["metadata"]
        assert "author" in meta

    def test_metadata_tags(self, manifest):
        meta = manifest["metadata"]
        assert "tags" in meta
        tags = meta["tags"]
        assert isinstance(tags, list)
        assert len(tags) >= 10
        assert "csddd" in tags
        assert "due-diligence" in tags


# ---------------------------------------------------------------------------
# 3. Regulation section
# ---------------------------------------------------------------------------


class TestManifestRegulation:
    """Verify regulation references are present."""

    def test_regulation_section_exists(self, manifest):
        meta = manifest.get("metadata", {})
        assert "regulation" in meta

    def test_primary_regulation(self, manifest):
        reg = manifest["metadata"]["regulation"]
        assert "primary" in reg
        primary = reg["primary"]
        assert "Directive" in primary.get("reference", "")

    def test_secondary_regulations(self, manifest):
        reg = manifest["metadata"]["regulation"]
        assert "secondary" in reg
        secondary = reg["secondary"]
        assert isinstance(secondary, list)
        assert len(secondary) >= 5


# ---------------------------------------------------------------------------
# 4. Compliance references
# ---------------------------------------------------------------------------


class TestManifestComplianceRefs:
    """Verify compliance reference entries."""

    def test_compliance_references_exist(self, manifest):
        meta = manifest.get("metadata", {})
        assert "compliance_references" in meta

    def test_compliance_references_count(self, manifest):
        refs = manifest["metadata"]["compliance_references"]
        assert isinstance(refs, list)
        assert len(refs) >= 10  # At least articles + OECD + UNGP


# ---------------------------------------------------------------------------
# 5. Components section
# ---------------------------------------------------------------------------


class TestManifestComponents:
    """Verify component counts for engines, workflows, templates, integrations."""

    def test_components_section_exists(self, manifest):
        assert "components" in manifest

    def test_engines_count(self, manifest):
        engines = manifest["components"].get("engines", [])
        assert isinstance(engines, list)
        assert len(engines) == 8, f"Expected 8 engines, got {len(engines)}"

    def test_workflows_count(self, manifest):
        workflows = manifest["components"].get("workflows", [])
        assert isinstance(workflows, list)
        assert len(workflows) == 8, f"Expected 8 workflows, got {len(workflows)}"

    def test_templates_count(self, manifest):
        templates = manifest["components"].get("templates", [])
        assert isinstance(templates, list)
        assert len(templates) == 8, f"Expected 8 templates, got {len(templates)}"

    def test_integrations_count(self, manifest):
        integrations = manifest["components"].get("integrations", [])
        assert isinstance(integrations, list)
        assert len(integrations) >= 8, f"Expected at least 8 integrations, got {len(integrations)}"

    def test_engine_ids_unique(self, manifest):
        engines = manifest["components"].get("engines", [])
        ids = [e["id"] for e in engines]
        assert len(ids) == len(set(ids)), "Engine IDs must be unique"

    def test_workflow_ids_unique(self, manifest):
        workflows = manifest["components"].get("workflows", [])
        ids = [w["id"] for w in workflows]
        assert len(ids) == len(set(ids)), "Workflow IDs must be unique"


# ---------------------------------------------------------------------------
# 6. Scope thresholds
# ---------------------------------------------------------------------------


class TestManifestScopeThresholds:
    """Verify scope threshold definitions match CSDDD phase-in schedule."""

    def test_scope_thresholds_exist(self, manifest):
        meta = manifest.get("metadata", {})
        assert "scope_thresholds" in meta

    def test_phase_1_thresholds(self, manifest):
        thresholds = manifest["metadata"]["scope_thresholds"]
        p1 = thresholds.get("phase_1", {})
        eu = p1.get("eu_companies", {})
        assert eu.get("employees") == 5000
        assert eu.get("turnover_eur") == 1_500_000_000

    def test_phase_3_thresholds(self, manifest):
        thresholds = manifest["metadata"]["scope_thresholds"]
        p3 = thresholds.get("phase_3", {})
        eu = p3.get("eu_companies", {})
        assert eu.get("employees") == 1000
        assert eu.get("turnover_eur") == 450_000_000
