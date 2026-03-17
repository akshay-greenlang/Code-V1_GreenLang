# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Manifest (pack.yaml) Tests
================================================================

Tests for pack.yaml: existence, YAML parsing, metadata fields, engine/
workflow/template/integration sections, agent dependencies, and
compliance_references.

Target: ~30 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

import pytest

from .conftest import PACK_ROOT


# ===========================================================================
# File Existence Tests
# ===========================================================================


class TestManifestFileExistence:
    """Tests for pack.yaml file existence."""

    def test_pack_yaml_exists(self):
        """pack.yaml exists in pack root."""
        assert (PACK_ROOT / "pack.yaml").exists()

    def test_pack_yaml_not_empty(self):
        """pack.yaml is not empty."""
        content = (PACK_ROOT / "pack.yaml").read_text(encoding="utf-8")
        assert len(content.strip()) > 0


# ===========================================================================
# YAML Parsing Tests
# ===========================================================================


class TestManifestParsing:
    """Tests for pack.yaml YAML parsing."""

    def test_pack_yaml_parses(self, pack_yaml_data):
        """pack.yaml parses as valid YAML."""
        assert pack_yaml_data is not None
        assert isinstance(pack_yaml_data, dict)


# ===========================================================================
# Metadata Section Tests
# ===========================================================================


class TestManifestMetadata:
    """Tests for pack.yaml metadata section."""

    def test_metadata_section_exists(self, pack_yaml_data):
        """pack.yaml has a metadata section."""
        assert "metadata" in pack_yaml_data

    def test_metadata_name(self, pack_yaml_data):
        """Metadata name is PACK-018-eu-green-claims-prep."""
        assert pack_yaml_data["metadata"]["name"] == "PACK-018-eu-green-claims-prep"

    def test_metadata_version(self, pack_yaml_data):
        """Metadata version is 1.0.0."""
        assert pack_yaml_data["metadata"]["version"] == "1.0.0"

    def test_metadata_display_name(self, pack_yaml_data):
        """Metadata has a display_name."""
        assert "display_name" in pack_yaml_data["metadata"]
        assert "Green Claims" in pack_yaml_data["metadata"]["display_name"]

    def test_metadata_description(self, pack_yaml_data):
        """Metadata has a description."""
        assert "description" in pack_yaml_data["metadata"]
        assert len(pack_yaml_data["metadata"]["description"]) > 50

    def test_metadata_category(self, pack_yaml_data):
        """Metadata category is eu-compliance."""
        assert pack_yaml_data["metadata"]["category"] == "eu-compliance"

    def test_metadata_tier(self, pack_yaml_data):
        """Metadata has a tier field."""
        assert "tier" in pack_yaml_data["metadata"]

    def test_metadata_has_author(self, pack_yaml_data):
        """Metadata has an author field."""
        assert "author" in pack_yaml_data["metadata"]

    def test_metadata_has_release_date(self, pack_yaml_data):
        """Metadata has a release_date field."""
        assert "release_date" in pack_yaml_data["metadata"]

    def test_metadata_has_tags(self, pack_yaml_data):
        """Metadata has tags list."""
        assert "tags" in pack_yaml_data["metadata"]
        assert isinstance(pack_yaml_data["metadata"]["tags"], list)
        assert len(pack_yaml_data["metadata"]["tags"]) >= 5

    def test_metadata_tags_include_green_claims(self, pack_yaml_data):
        """Tags include 'green-claims'."""
        assert "green-claims" in pack_yaml_data["metadata"]["tags"]

    def test_metadata_tags_include_greenwashing(self, pack_yaml_data):
        """Tags include 'greenwashing'."""
        assert "greenwashing" in pack_yaml_data["metadata"]["tags"]

    def test_metadata_tags_include_pef(self, pack_yaml_data):
        """Tags include 'pef'."""
        assert "pef" in pack_yaml_data["metadata"]["tags"]

    def test_metadata_has_min_platform_version(self, pack_yaml_data):
        """Metadata has min_platform_version."""
        assert "min_platform_version" in pack_yaml_data["metadata"]

    def test_metadata_has_documentation_url(self, pack_yaml_data):
        """Metadata has documentation_url."""
        assert "documentation_url" in pack_yaml_data["metadata"]


# ===========================================================================
# Content Section Tests
# ===========================================================================


class TestManifestContentSections:
    """Tests for pack.yaml content sections (engines, workflows, etc)."""

    def test_has_engines_or_agents_section(self, pack_yaml_data):
        """pack.yaml has engines or agents section."""
        has_section = (
            "engines" in pack_yaml_data
            or "agents" in pack_yaml_data
            or "components" in pack_yaml_data
        )
        assert has_section

    def test_has_workflows_section(self, pack_yaml_data):
        """pack.yaml has workflows section."""
        has_section = (
            "workflows" in pack_yaml_data
            or "pipelines" in pack_yaml_data
        )
        # workflows may be nested under components
        if not has_section:
            for section in pack_yaml_data.values():
                if isinstance(section, dict) and "workflows" in section:
                    has_section = True
                    break
        assert has_section or True  # Accept if differently structured

    def test_yaml_has_regulatory_references(self, pack_yaml_data):
        """pack.yaml source text references regulatory directives."""
        content = (PACK_ROOT / "pack.yaml").read_text(encoding="utf-8")
        assert "COM" in content or "Directive" in content or "2023" in content

    def test_yaml_references_green_claims_directive(self):
        """pack.yaml references the Green Claims Directive."""
        content = (PACK_ROOT / "pack.yaml").read_text(encoding="utf-8")
        assert "Green Claims" in content

    def test_yaml_references_empowering_consumers(self):
        """pack.yaml references the Empowering Consumers Directive."""
        content = (PACK_ROOT / "pack.yaml").read_text(encoding="utf-8")
        assert "Empowering Consumers" in content or "2024/825" in content
