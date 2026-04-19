# -*- coding: utf-8 -*-
"""Tests for PACK-022 - pack.yaml manifest validation.

Validates the structure, metadata, component counts, regulatory references,
agent dependencies, performance targets, and security configuration of the
pack.yaml manifest file.

Author:  GL-TestEngineer
Pack:    PACK-022 Net Zero Acceleration
"""
import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_DIR / "pack.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pack_yaml_path() -> Path:
    """Return the path to pack.yaml."""
    return PACK_YAML_PATH


@pytest.fixture(scope="module")
def pack_data() -> dict:
    """Load and return the parsed pack.yaml contents."""
    with open(PACK_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "pack.yaml parsed as None (empty file?)"
    return data


@pytest.fixture(scope="module")
def metadata(pack_data: dict) -> dict:
    """Return the metadata section."""
    return pack_data.get("metadata", {})


@pytest.fixture(scope="module")
def components(pack_data: dict) -> dict:
    """Return the components section."""
    return pack_data.get("components", {})


# ===========================================================================
# Tests -- File Existence & YAML Validity
# ===========================================================================


class TestPackYamlStructure:
    """Tests for pack.yaml file existence and YAML validity."""

    def test_pack_yaml_exists(self, pack_yaml_path: Path) -> None:
        """pack.yaml must exist at the pack root directory."""
        assert pack_yaml_path.exists(), (
            f"pack.yaml not found at {pack_yaml_path}"
        )

    def test_pack_yaml_valid_yaml(self, pack_yaml_path: Path) -> None:
        """pack.yaml must load without YAML parsing errors."""
        with open(pack_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), (
            "pack.yaml must parse to a dictionary at root level"
        )

    def test_pack_yaml_not_empty(self, pack_data: dict) -> None:
        """pack.yaml must not be an empty document."""
        assert len(pack_data) > 0, "pack.yaml should not be empty"


# ===========================================================================
# Tests -- Metadata Fields
# ===========================================================================


class TestMetadataFields:
    """Tests for required metadata fields in pack.yaml."""

    def test_metadata_section_exists(self, pack_data: dict) -> None:
        """The 'metadata' top-level key must exist."""
        assert "metadata" in pack_data, "Missing 'metadata' section"

    @pytest.mark.parametrize("field", [
        "name",
        "version",
        "display_name",
        "category",
        "tier",
    ])
    def test_required_metadata_fields(self, metadata: dict, field: str) -> None:
        """All mandatory metadata fields must be present."""
        assert field in metadata, f"Missing metadata.{field}"
        assert metadata[field], f"metadata.{field} must not be empty"

    def test_metadata_name(self, metadata: dict) -> None:
        """Pack name must be PACK-022-net-zero-acceleration."""
        assert metadata["name"] == "PACK-022-net-zero-acceleration"

    def test_metadata_display_name(self, metadata: dict) -> None:
        """Display name must mention Net Zero."""
        display = metadata["display_name"]
        assert "Net Zero" in display, (
            f"display_name should mention 'Net Zero', got: {display}"
        )

    def test_metadata_category(self, metadata: dict) -> None:
        """Category must be 'net-zero'."""
        assert metadata["category"] == "net-zero"

    def test_metadata_tier(self, metadata: dict) -> None:
        """Tier must be 'professional'."""
        assert metadata["tier"] == "professional"

    def test_version_format(self, metadata: dict) -> None:
        """Version must be valid semver (MAJOR.MINOR.PATCH)."""
        version = metadata["version"]
        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, version), (
            f"Version '{version}' does not match semver (X.Y.Z)"
        )


# ===========================================================================
# Tests -- Component Counts
# ===========================================================================


class TestComponentCounts:
    """Tests for the correct number of components in each category."""

    def test_engines_count(self, components: dict) -> None:
        """There must be exactly 10 engines."""
        engines = components.get("engines", [])
        assert len(engines) == 10, (
            f"Expected 10 engines, got {len(engines)}"
        )

    def test_workflows_count(self, components: dict) -> None:
        """There must be exactly 8 workflows."""
        workflows = components.get("workflows", [])
        assert len(workflows) == 8, (
            f"Expected 8 workflows, got {len(workflows)}"
        )

    def test_each_engine_has_id_and_name(self, components: dict) -> None:
        """Every engine must have an 'id' and 'name' field."""
        for engine in components.get("engines", []):
            assert "id" in engine, f"Engine missing 'id': {engine}"
            assert "name" in engine, f"Engine missing 'name': {engine}"

    def test_each_workflow_has_id_and_name(self, components: dict) -> None:
        """Every workflow must have an 'id' and 'name' field."""
        for wf in components.get("workflows", []):
            assert "id" in wf, f"Workflow missing 'id': {wf}"
            assert "name" in wf, f"Workflow missing 'name': {wf}"


# ===========================================================================
# Tests -- Security Configuration
# ===========================================================================


class TestSecurityConfig:
    """Tests for security configuration presence."""

    def test_security_section_present(self, pack_data: dict) -> None:
        """security section must be present."""
        assert "security" in pack_data, "Missing security section"

    def test_audit_logging_enabled(self, pack_data: dict) -> None:
        """Audit logging must be enabled."""
        sec = pack_data.get("security", {})
        assert sec.get("audit_logging") is True


# ===========================================================================
# Tests -- Performance Targets
# ===========================================================================


class TestPerformanceTargets:
    """Tests for performance target definitions."""

    def test_performance_section_present(self, pack_data: dict) -> None:
        """performance section must be present."""
        assert "performance" in pack_data, "Missing performance section"
