# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Manifest Tests

Tests pack.yaml parsing, component listing, migration and preset
file existence, and metadata correctness.

Target: 15-20 tests.
"""

import os
import pytest
import yaml


PACK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


@pytest.fixture(scope="module")
def manifest():
    """Load pack.yaml manifest."""
    manifest_path = os.path.join(PACK_ROOT, "pack.yaml")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestManifestMetadata:
    """Test pack metadata from manifest."""

    def test_pack_id(self, manifest):
        assert manifest["pack"]["id"] == "PACK-050"

    def test_pack_name(self, manifest):
        assert manifest["pack"]["name"] == "GHG Consolidation Pack"

    def test_pack_version(self, manifest):
        assert manifest["pack"]["version"] == "1.0.0"

    def test_pack_category(self, manifest):
        assert manifest["pack"]["category"] == "ghg-accounting"

    def test_pack_tier(self, manifest):
        assert manifest["pack"]["tier"] == "enterprise"

    def test_pack_status(self, manifest):
        assert manifest["pack"]["status"] == "production-ready"


class TestManifestComponents:
    """Test component listing in manifest."""

    def test_engine_count(self, manifest):
        assert manifest["components"]["engines"]["count"] == 10

    def test_engine_items_listed(self, manifest):
        engines = manifest["components"]["engines"]["items"]
        assert len(engines) == 10

    def test_workflow_count(self, manifest):
        assert manifest["components"]["workflows"]["count"] == 8

    def test_workflow_items_listed(self, manifest):
        workflows = manifest["components"]["workflows"]["items"]
        assert len(workflows) == 8

    def test_template_count(self, manifest):
        assert manifest["components"]["templates"]["count"] == 10

    def test_integration_count(self, manifest):
        assert manifest["components"]["integrations"]["count"] == 13

    def test_preset_count(self, manifest):
        presets = manifest["components"]["presets"]
        assert len(presets) == 8


class TestManifestDatabase:
    """Test database migration references."""

    def test_migration_start(self, manifest):
        assert manifest["database"]["migrations"]["start"] == "V416"

    def test_migration_end(self, manifest):
        assert manifest["database"]["migrations"]["end"] == "V425"

    def test_migration_count(self, manifest):
        assert manifest["database"]["migrations"]["count"] == 10

    def test_migration_files_listed(self, manifest):
        files = manifest["database"]["migrations"]["files"]
        assert len(files) == 10


class TestManifestCompliance:
    """Test compliance declarations."""

    def test_zero_hallucination(self, manifest):
        assert manifest["compliance"]["zero_hallucination"] is True

    def test_deterministic_calculations(self, manifest):
        assert manifest["compliance"]["deterministic_calculations"] is True

    def test_decimal_arithmetic(self, manifest):
        assert manifest["compliance"]["decimal_arithmetic"] is True

    def test_sha256_provenance(self, manifest):
        assert manifest["compliance"]["sha256_provenance"] is True

    def test_consolidation_approaches_listed(self, manifest):
        approaches = manifest["compliance"]["consolidation_approaches"]
        assert "Equity Share" in approaches
        assert "Operational Control" in approaches
        assert "Financial Control" in approaches


class TestManifestTesting:
    """Test testing metadata."""

    def test_test_file_target(self, manifest):
        assert manifest["testing"]["test_files"] >= 18

    def test_test_function_target(self, manifest):
        assert manifest["testing"]["test_functions"] >= 700

    def test_pass_rate_target(self, manifest):
        assert manifest["testing"]["pass_rate"] == "100%"


class TestPresetFilesExist:
    """Test that preset files referenced in manifest exist."""

    @pytest.mark.parametrize("preset_name", [
        "corporate_conglomerate",
        "financial_holding",
        "jv_partnership",
        "multinational",
        "private_equity",
        "real_estate_fund",
        "public_company",
        "sme_group",
    ])
    def test_preset_yaml_exists(self, preset_name):
        path = os.path.join(PACK_ROOT, "presets", f"{preset_name}.yaml")
        assert os.path.isfile(path), f"Preset file not found: {path}"
