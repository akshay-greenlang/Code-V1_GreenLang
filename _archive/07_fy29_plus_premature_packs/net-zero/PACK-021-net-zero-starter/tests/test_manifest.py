# -*- coding: utf-8 -*-
"""
Test suite for PACK-021 Net Zero Starter Pack - pack.yaml manifest.

Validates the structure, metadata, component counts, regulatory references,
agent dependencies, performance targets, and security configuration of the
pack.yaml manifest file.

Author:  GreenLang Test Engineering
Pack:    PACK-021 Net Zero Starter
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
        """Pack name must be PACK-021-net-zero-starter."""
        assert metadata["name"] == "PACK-021-net-zero-starter"

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
        """Tier must be 'starter'."""
        assert metadata["tier"] == "starter"

    def test_version_format(self, metadata: dict) -> None:
        """Version must be valid semver (MAJOR.MINOR.PATCH)."""
        version = metadata["version"]
        semver_pattern = r"^\d+\.\d+\.\d+$"
        assert re.match(semver_pattern, version), (
            f"Version '{version}' does not match semver (X.Y.Z)"
        )

    def test_tags_present(self, metadata: dict) -> None:
        """Tags must be present and contain at least 10 entries."""
        tags = metadata.get("tags", [])
        assert isinstance(tags, list), "metadata.tags must be a list"
        assert len(tags) >= 10, (
            f"Expected at least 10 tags, got {len(tags)}"
        )

    @pytest.mark.parametrize("expected_tag", [
        "net-zero",
        "sbti",
        "ghg-protocol",
        "decarbonization",
        "macc",
        "paris-agreement",
        "ipcc-ar6",
    ])
    def test_critical_tags_present(self, metadata: dict, expected_tag: str) -> None:
        """Critical tags must be included in metadata.tags."""
        tags = metadata.get("tags", [])
        assert expected_tag in tags, (
            f"Missing critical tag: {expected_tag}"
        )

    def test_author_present(self, metadata: dict) -> None:
        """metadata.author must be present."""
        assert "author" in metadata
        assert metadata["author"]

    def test_release_date_present(self, metadata: dict) -> None:
        """metadata.release_date must be present."""
        assert "release_date" in metadata
        assert metadata["release_date"]

    def test_min_platform_version_present(self, metadata: dict) -> None:
        """metadata.min_platform_version must be present."""
        assert "min_platform_version" in metadata


# ===========================================================================
# Tests -- Component Counts
# ===========================================================================


class TestComponentCounts:
    """Tests for the correct number of components in each category."""

    def test_engines_count(self, components: dict) -> None:
        """There must be exactly 8 engines."""
        engines = components.get("engines", [])
        assert len(engines) == 8, (
            f"Expected 8 engines, got {len(engines)}"
        )

    def test_workflows_count(self, components: dict) -> None:
        """There must be exactly 6 workflows."""
        workflows = components.get("workflows", [])
        assert len(workflows) == 6, (
            f"Expected 6 workflows, got {len(workflows)}"
        )

    def test_templates_count(self, components: dict) -> None:
        """There must be exactly 8 templates."""
        templates = components.get("templates", [])
        assert len(templates) == 8, (
            f"Expected 8 templates, got {len(templates)}"
        )

    def test_integrations_count(self, components: dict) -> None:
        """There must be exactly 10 integrations."""
        integrations = components.get("integrations", [])
        assert len(integrations) == 10, (
            f"Expected 10 integrations, got {len(integrations)}"
        )

    def test_presets_count(self, components: dict) -> None:
        """There must be exactly 6 presets."""
        presets = components.get("presets", [])
        assert len(presets) == 6, (
            f"Expected 6 presets, got {len(presets)}"
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
# Tests -- Regulatory References
# ===========================================================================


class TestRegulatoryReferences:
    """Tests for regulatory and compliance reference sections."""

    def test_regulation_section_present(self, metadata: dict) -> None:
        """metadata.regulation must be present."""
        assert "regulation" in metadata, "Missing metadata.regulation section"

    def test_primary_regulation_present(self, metadata: dict) -> None:
        """Primary regulation must reference SBTi Net-Zero Standard."""
        reg = metadata.get("regulation", {})
        primary = reg.get("primary", {})
        assert primary, "Missing metadata.regulation.primary"
        assert "SBTi" in primary.get("name", ""), (
            "Primary regulation should reference SBTi"
        )

    def test_secondary_regulations_present(self, metadata: dict) -> None:
        """There must be at least 5 secondary regulatory references."""
        reg = metadata.get("regulation", {})
        secondary = reg.get("secondary", [])
        assert len(secondary) >= 5, (
            f"Expected at least 5 secondary regulations, got {len(secondary)}"
        )

    def test_compliance_references_present(self, metadata: dict) -> None:
        """metadata.compliance_references must be present with entries."""
        refs = metadata.get("compliance_references", [])
        assert len(refs) >= 5, (
            f"Expected at least 5 compliance references, got {len(refs)}"
        )

    @pytest.mark.parametrize("ref_id", [
        "SBTI_NZ",
        "GHG_PROTOCOL",
        "GHG_SCOPE3",
        "IPCC_AR6",
        "PARIS",
    ])
    def test_critical_compliance_references(
        self, metadata: dict, ref_id: str
    ) -> None:
        """Critical compliance reference IDs must be present."""
        refs = metadata.get("compliance_references", [])
        ids = [r.get("id") for r in refs]
        assert ref_id in ids, (
            f"Missing compliance reference: {ref_id}"
        )


# ===========================================================================
# Tests -- Agent Dependencies
# ===========================================================================


class TestAgentDependencies:
    """Tests for agent dependency declarations."""

    def test_agents_mrv_listed(self, pack_data: dict) -> None:
        """agents_mrv section must be present."""
        assert "agents_mrv" in pack_data, "Missing agents_mrv section"

    def test_mrv_scope1_agents(self, pack_data: dict) -> None:
        """At least 5 Scope 1 MRV agents must be declared."""
        scope1 = pack_data.get("agents_mrv", {}).get("scope_1", [])
        assert len(scope1) >= 5, (
            f"Expected at least 5 Scope 1 agents, got {len(scope1)}"
        )

    def test_mrv_scope2_agents(self, pack_data: dict) -> None:
        """At least 3 Scope 2 MRV agents must be declared."""
        scope2 = pack_data.get("agents_mrv", {}).get("scope_2", [])
        assert len(scope2) >= 3, (
            f"Expected at least 3 Scope 2 agents, got {len(scope2)}"
        )

    def test_mrv_scope3_agents(self, pack_data: dict) -> None:
        """At least 10 Scope 3 MRV agents must be declared."""
        scope3 = pack_data.get("agents_mrv", {}).get("scope_3", [])
        assert len(scope3) >= 10, (
            f"Expected at least 10 Scope 3 agents, got {len(scope3)}"
        )

    def test_agents_data_listed(self, pack_data: dict) -> None:
        """agents_data section must be present with entries."""
        data_agents = pack_data.get("agents_data", [])
        assert len(data_agents) >= 10, (
            f"Expected at least 10 data agents, got {len(data_agents)}"
        )

    def test_agents_foundation_listed(self, pack_data: dict) -> None:
        """agents_foundation must list all 10 foundation agents."""
        found = pack_data.get("agents_foundation", [])
        assert len(found) == 10, (
            f"Expected 10 foundation agents, got {len(found)}"
        )

    def test_dependencies_summary(self, pack_data: dict) -> None:
        """dependencies.total_agents must match declared agents."""
        deps = pack_data.get("dependencies", {})
        total = deps.get("total_agents", 0)
        assert total >= 50, (
            f"Expected at least 50 total agents, got {total}"
        )


# ===========================================================================
# Tests -- Performance Targets
# ===========================================================================


class TestPerformanceTargets:
    """Tests for performance target definitions."""

    def test_performance_section_present(self, pack_data: dict) -> None:
        """performance section must be present."""
        assert "performance" in pack_data, "Missing performance section"

    def test_baseline_performance_targets(self, pack_data: dict) -> None:
        """Baseline calculation performance targets must be defined."""
        perf = pack_data.get("performance", {})
        baseline = perf.get("baseline_calculation", {})
        assert "target_duration_minutes" in baseline
        assert baseline["target_duration_minutes"] > 0

    def test_target_setting_performance(self, pack_data: dict) -> None:
        """Target setting performance targets must be defined."""
        perf = pack_data.get("performance", {})
        ts = perf.get("target_setting", {})
        assert "target_duration_minutes" in ts

    def test_memory_ceiling_defined(self, pack_data: dict) -> None:
        """memory_ceiling_mb must be defined."""
        perf = pack_data.get("performance", {})
        assert "memory_ceiling_mb" in perf
        assert perf["memory_ceiling_mb"] > 0


# ===========================================================================
# Tests -- Security Configuration
# ===========================================================================


class TestSecurityConfig:
    """Tests for security configuration presence."""

    def test_security_section_present(self, pack_data: dict) -> None:
        """security section must be present."""
        assert "security" in pack_data, "Missing security section"

    def test_authentication_defined(self, pack_data: dict) -> None:
        """Authentication method must be specified."""
        sec = pack_data.get("security", {})
        assert "authentication" in sec
        assert "JWT" in sec["authentication"]

    def test_encryption_at_rest(self, pack_data: dict) -> None:
        """Encryption at rest must be AES-256-GCM."""
        sec = pack_data.get("security", {})
        assert "AES-256" in sec.get("encryption_at_rest", "")

    def test_encryption_in_transit(self, pack_data: dict) -> None:
        """Encryption in transit must be TLS 1.3."""
        sec = pack_data.get("security", {})
        assert "TLS 1.3" in sec.get("encryption_in_transit", "")

    def test_audit_logging_enabled(self, pack_data: dict) -> None:
        """Audit logging must be enabled."""
        sec = pack_data.get("security", {})
        assert sec.get("audit_logging") is True

    def test_required_roles_present(self, pack_data: dict) -> None:
        """At least 3 required roles must be declared."""
        sec = pack_data.get("security", {})
        roles = sec.get("required_roles", [])
        assert len(roles) >= 3, (
            f"Expected at least 3 required roles, got {len(roles)}"
        )

    def test_pii_redaction_enabled(self, pack_data: dict) -> None:
        """PII redaction must be enabled."""
        sec = pack_data.get("security", {})
        assert sec.get("pii_redaction") is True
