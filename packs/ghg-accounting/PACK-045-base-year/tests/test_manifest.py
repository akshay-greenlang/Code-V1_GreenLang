# -*- coding: utf-8 -*-
"""
Tests for PACK-045 pack.yaml manifest and engines __init__.

Validates metadata, structure, and module initialization.
Target: ~40 tests.
"""

import pytest
from pathlib import Path
import sys

import yaml

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

PACK_YAML = PACK_ROOT / "pack.yaml"


@pytest.fixture
def manifest():
    """Load and parse pack.yaml."""
    with open(PACK_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# Manifest Metadata
# ============================================================================

class TestManifestMetadata:
    def test_manifest_exists(self):
        assert PACK_YAML.exists()

    def test_manifest_parseable(self, manifest):
        assert manifest is not None
        assert isinstance(manifest, dict)

    def test_metadata_section(self, manifest):
        assert "metadata" in manifest

    def test_metadata_name(self, manifest):
        assert manifest["metadata"]["name"] == "PACK-045-base-year"

    def test_metadata_version(self, manifest):
        assert manifest["metadata"]["version"] == "1.0.0"

    def test_metadata_display_name(self, manifest):
        assert "Base Year" in manifest["metadata"]["display_name"]

    def test_metadata_description(self, manifest):
        assert len(manifest["metadata"]["description"]) > 50

    def test_metadata_category(self, manifest):
        assert manifest["metadata"]["category"] == "ghg-accounting"

    def test_metadata_tier(self, manifest):
        assert manifest["metadata"]["tier"] == "enterprise"

    def test_metadata_author(self, manifest):
        assert manifest["metadata"]["author"] == "GreenLang Platform Team"

    def test_metadata_min_platform_version(self, manifest):
        assert "min_platform_version" in manifest["metadata"]

    def test_metadata_release_date(self, manifest):
        assert manifest["metadata"]["release_date"] == "2026-03-24"

    def test_metadata_tags(self, manifest):
        tags = manifest["metadata"]["tags"]
        assert isinstance(tags, list)
        assert len(tags) >= 10
        assert "base-year" in tags
        assert "ghg-protocol" in tags

    def test_metadata_regulation(self, manifest):
        reg = manifest["metadata"]["regulation"]
        assert "primary" in reg
        assert "GHG Protocol" in reg["primary"]["name"]

    def test_metadata_secondary_regulations(self, manifest):
        secondary = manifest["metadata"]["regulation"]["secondary"]
        assert isinstance(secondary, list)
        assert len(secondary) >= 2


# ============================================================================
# Manifest Structure
# ============================================================================

class TestManifestStructure:
    def test_engines_section(self, manifest):
        assert "engines" in manifest

    def test_engines_count(self, manifest):
        engines = manifest["engines"]
        assert len(engines) == 10

    def test_workflows_section(self, manifest):
        assert "workflows" in manifest

    def test_workflows_count(self, manifest):
        workflows = manifest["workflows"]
        assert len(workflows) == 8

    def test_templates_section(self, manifest):
        assert "templates" in manifest

    def test_templates_count(self, manifest):
        templates = manifest["templates"]
        assert len(templates) == 10

    def test_integrations_section(self, manifest):
        assert "integrations" in manifest

    def test_integrations_count(self, manifest):
        integrations = manifest["integrations"]
        assert len(integrations) == 12

    def test_presets_section(self, manifest):
        assert "presets" in manifest

    def test_presets_count(self, manifest):
        presets = manifest["presets"]
        assert len(presets) == 8


# ============================================================================
# Engines Module __init__
# ============================================================================

class TestEnginesInit:
    def test_import_engines_module(self):
        from engines import __version__, __pack__, __pack_name__, __engines_count__
        assert __version__ == "1.0.0"
        assert __pack__ == "PACK-045"
        assert __pack_name__ == "Base Year Management Pack"
        assert __engines_count__ == 10

    def test_get_loaded_engines(self):
        from engines import get_loaded_engines
        loaded = get_loaded_engines()
        assert isinstance(loaded, list)
        assert len(loaded) == 10

    def test_get_engine_count(self):
        from engines import get_engine_count
        count = get_engine_count()
        assert count == 10

    def test_all_engines_in_all(self):
        from engines import __all__
        expected = [
            "BaseYearSelectionEngine",
            "BaseYearInventoryEngine",
            "RecalculationPolicyEngine",
            "RecalculationTriggerEngine",
            "SignificanceAssessmentEngine",
            "BaseYearAdjustmentEngine",
            "TimeSeriesConsistencyEngine",
            "TargetTrackingEngine",
            "BaseYearAuditEngine",
            "BaseYearReportingEngine",
        ]
        for name in expected:
            assert name in __all__


# ============================================================================
# File Structure Validation
# ============================================================================

class TestFileStructure:
    def test_engines_directory(self):
        assert (PACK_ROOT / "engines").is_dir()

    def test_workflows_directory(self):
        assert (PACK_ROOT / "workflows").is_dir()

    def test_templates_directory(self):
        assert (PACK_ROOT / "templates").is_dir()

    def test_integrations_directory(self):
        assert (PACK_ROOT / "integrations").is_dir()

    def test_config_directory(self):
        assert (PACK_ROOT / "config").is_dir()

    def test_presets_directory(self):
        assert (PACK_ROOT / "config" / "presets").is_dir()

    def test_engine_files_exist(self):
        engine_files = [
            "base_year_selection_engine.py",
            "base_year_inventory_engine.py",
            "recalculation_policy_engine.py",
            "recalculation_trigger_engine.py",
            "significance_assessment_engine.py",
            "base_year_adjustment_engine.py",
            "time_series_consistency_engine.py",
            "target_tracking_engine.py",
            "base_year_audit_engine.py",
            "base_year_reporting_engine.py",
        ]
        for f in engine_files:
            assert (PACK_ROOT / "engines" / f).exists(), f"Missing: {f}"

    def test_workflow_files_exist(self):
        workflow_files = [
            "base_year_establishment_workflow.py",
            "recalculation_assessment_workflow.py",
            "recalculation_execution_workflow.py",
            "merger_acquisition_workflow.py",
            "annual_review_workflow.py",
            "target_rebasing_workflow.py",
            "audit_verification_workflow.py",
            "full_base_year_pipeline_workflow.py",
        ]
        for f in workflow_files:
            assert (PACK_ROOT / "workflows" / f).exists(), f"Missing: {f}"
