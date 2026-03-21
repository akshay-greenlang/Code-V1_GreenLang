# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - pack.yaml manifest.

Validates the structure, metadata, component counts, regulatory references,
agent dependencies, performance targets, and security configuration of the
pack.yaml manifest file.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~150 lines, 35+ tests
"""

import re
from pathlib import Path

import pytest
import yaml

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_DIR / "pack.yaml"


@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    return PACK_YAML_PATH


@pytest.fixture(scope="module")
def pack_data() -> dict:
    with open(PACK_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "pack.yaml parsed as None (empty file?)"
    return data


@pytest.fixture(scope="module")
def metadata(pack_data: dict) -> dict:
    return pack_data.get("metadata", {})


@pytest.fixture(scope="module")
def components(pack_data: dict) -> dict:
    return pack_data.get("components", {})


# ===========================================================================
# Tests -- File Existence & YAML Validity
# ===========================================================================


class TestPackYamlStructure:
    def test_pack_yaml_exists(self, pack_yaml_path: Path) -> None:
        assert pack_yaml_path.exists(), f"pack.yaml not found at {pack_yaml_path}"

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path) -> None:
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_pack_yaml_not_empty(self, pack_data: dict) -> None:
        assert len(pack_data) > 0


# ===========================================================================
# Tests -- Metadata Fields
# ===========================================================================


class TestMetadataFields:
    def test_metadata_section_exists(self, pack_data: dict) -> None:
        assert "metadata" in pack_data

    @pytest.mark.parametrize("field", ["name", "version", "display_name", "category", "tier"])
    def test_required_metadata_fields(self, metadata: dict, field: str) -> None:
        assert field in metadata
        assert metadata[field]

    def test_metadata_name(self, metadata: dict) -> None:
        assert metadata["name"] == "PACK-026-sme-net-zero"

    def test_metadata_display_name(self, metadata: dict) -> None:
        assert "SME" in metadata["display_name"]
        assert "Net Zero" in metadata["display_name"]

    def test_metadata_category(self, metadata: dict) -> None:
        assert metadata["category"] == "net-zero"

    def test_metadata_tier(self, metadata: dict) -> None:
        assert metadata["tier"] == "sme-optimized"

    def test_version_format(self, metadata: dict) -> None:
        version = metadata["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version)

    def test_tags_present(self, metadata: dict) -> None:
        tags = metadata.get("tags", [])
        assert isinstance(tags, list)
        assert len(tags) >= 8

    @pytest.mark.parametrize("expected_tag", [
        "sme", "net-zero", "small-business", "quick-wins",
        "grant-finder", "accounting-integration", "simplified",
    ])
    def test_critical_tags_present(self, metadata: dict, expected_tag: str) -> None:
        tags = metadata.get("tags", [])
        assert expected_tag in tags, f"Missing critical tag: {expected_tag}"

    def test_author_present(self, metadata: dict) -> None:
        assert "author" in metadata and metadata["author"]

    def test_release_date_present(self, metadata: dict) -> None:
        assert "release_date" in metadata

    def test_target_audience_in_description(self, metadata: dict) -> None:
        desc = metadata.get("description", "")
        assert "SME" in desc or "sme" in desc.lower()


# ===========================================================================
# Tests -- Component Counts
# ===========================================================================


class TestComponentCounts:
    def test_engines_section(self, components: dict) -> None:
        engines = components.get("engines", {})
        assert engines.get("count") == 8

    def test_workflows_section(self, components: dict) -> None:
        workflows = components.get("workflows", {})
        assert workflows.get("count") == 6

    def test_templates_section(self, components: dict) -> None:
        templates = components.get("templates", {})
        assert templates.get("count") == 8

    def test_integrations_section(self, components: dict) -> None:
        integrations = components.get("integrations", {})
        assert integrations.get("count") == 13

    def test_presets_section(self, components: dict) -> None:
        presets = components.get("presets", {})
        assert presets.get("count") == 6

    def test_engines_items_have_id(self, components: dict) -> None:
        items = components.get("engines", {}).get("items", [])
        for engine in items:
            assert "id" in engine

    def test_workflows_items_have_id(self, components: dict) -> None:
        items = components.get("workflows", {}).get("items", [])
        for wf in items:
            assert "id" in wf


# ===========================================================================
# Tests -- SME-Specific Requirements
# ===========================================================================


class TestSMESpecificRequirements:
    def test_accounting_integrations(self, components: dict) -> None:
        items = components.get("integrations", {}).get("items", [])
        integration_ids = [i.get("id", "") for i in items]
        assert any("xero" in str(i).lower() for i in integration_ids)
        assert any("quickbooks" in str(i).lower() for i in integration_ids)
        assert any("sage" in str(i).lower() for i in integration_ids)


# ===========================================================================
# Tests -- Performance Targets
# ===========================================================================


class TestPerformanceTargets:
    def test_performance_section_present(self, pack_data: dict) -> None:
        assert "performance" in pack_data

    def test_baseline_calculation_defined(self, pack_data: dict) -> None:
        perf = pack_data.get("performance", {})
        baseline = perf.get("baseline_calculation", {})
        assert "bronze_tier" in baseline
        assert "silver_tier" in baseline
        assert "gold_tier" in baseline

    def test_memory_ceiling_defined(self, pack_data: dict) -> None:
        perf = pack_data.get("performance", {})
        assert "memory_ceiling" in perf


# ===========================================================================
# Tests -- Security Configuration
# ===========================================================================


class TestSecurityConfig:
    def test_security_section_present(self, pack_data: dict) -> None:
        assert "security" in pack_data

    def test_authentication_defined(self, pack_data: dict) -> None:
        sec = pack_data.get("security", {})
        auth = sec.get("authentication", "")
        assert "JWT" in str(auth)

    def test_encryption_at_rest(self, pack_data: dict) -> None:
        sec = pack_data.get("security", {})
        encryption = sec.get("encryption", {})
        assert "AES-256" in str(encryption.get("at_rest", ""))

    def test_encryption_in_transit(self, pack_data: dict) -> None:
        sec = pack_data.get("security", {})
        encryption = sec.get("encryption", {})
        assert "TLS 1.3" in str(encryption.get("in_transit", ""))

    def test_audit_trail_enabled(self, pack_data: dict) -> None:
        sec = pack_data.get("security", {})
        audit = sec.get("audit_trail", {})
        assert audit.get("enabled") is True
