# -*- coding: utf-8 -*-
"""
Unit tests for GL-FOUND-X-002 Git-Backed Schema Registry.

This module tests the GitSchemaRegistry class and related functionality including:
- Schema path convention (schemas/{domain}/{name}@{version}.yaml)
- Version resolution and listing
- Semver comparison and constraint matching
- Caching of fetched schemas
- Git operations (pull, commit hash)

Test Categories:
- SemVer tests: Parsing and comparison
- VersionConstraint tests: Constraint matching
- GitSchemaRegistry tests: Core registry operations
- SchemaCache tests: Caching behavior

Author: GreenLang Team
Date: 2026-01-29
"""

import json
import time
import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from greenlang.schema.registry.git_backend import (
    GitSchemaRegistry,
    SchemaNotFoundError,
    InvalidSchemaIdError,
    VersionConstraintError,
    SchemaParseError,
    GitOperationError,
    SchemaSourceModel,
    SemVer,
    VersionConstraint,
    CachedSchema,
    SchemaCache,
    compare_versions,
    sort_versions,
    filter_versions,
    create_git_registry,
)


# ============================================================================
# SEMVER PARSING TESTS
# ============================================================================


class TestSemVerParsing:
    """Tests for SemVer parsing."""

    def test_parse_simple_version(self):
        """Parse a simple major.minor.patch version."""
        v = SemVer.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None
        assert v.build is None

    def test_parse_version_with_prerelease(self):
        """Parse a version with prerelease identifier."""
        v = SemVer.parse("1.0.0-beta.1")
        assert v.major == 1
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease == "beta.1"
        assert v.build is None

    def test_parse_version_with_build_metadata(self):
        """Parse a version with build metadata."""
        v = SemVer.parse("1.0.0+build.123")
        assert v.major == 1
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease is None
        assert v.build == "build.123"

    def test_parse_version_with_prerelease_and_build(self):
        """Parse a version with both prerelease and build."""
        v = SemVer.parse("1.2.3-alpha.1+build.456")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "alpha.1"
        assert v.build == "build.456"

    def test_parse_version_zero(self):
        """Parse version starting with 0."""
        v = SemVer.parse("0.1.0")
        assert v.major == 0
        assert v.minor == 1
        assert v.patch == 0

    def test_parse_large_version_numbers(self):
        """Parse large version numbers."""
        v = SemVer.parse("123.456.789")
        assert v.major == 123
        assert v.minor == 456
        assert v.patch == 789

    def test_parse_invalid_version_raises_error(self):
        """Invalid version string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            SemVer.parse("invalid")

    def test_parse_incomplete_version_raises_error(self):
        """Incomplete version string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            SemVer.parse("1.2")

    def test_parse_version_with_leading_v_raises_error(self):
        """Version with leading 'v' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid semver"):
            SemVer.parse("v1.2.3")

    def test_is_valid_returns_true_for_valid(self):
        """is_valid should return True for valid versions."""
        assert SemVer.is_valid("1.2.3") is True
        assert SemVer.is_valid("0.0.0") is True
        assert SemVer.is_valid("1.0.0-alpha") is True

    def test_is_valid_returns_false_for_invalid(self):
        """is_valid should return False for invalid versions."""
        assert SemVer.is_valid("invalid") is False
        assert SemVer.is_valid("1.2") is False
        assert SemVer.is_valid("") is False


# ============================================================================
# SEMVER COMPARISON TESTS
# ============================================================================


class TestSemVerComparison:
    """Tests for SemVer comparison."""

    def test_equal_versions(self):
        """Equal versions should be equal."""
        v1 = SemVer.parse("1.2.3")
        v2 = SemVer.parse("1.2.3")
        assert v1 == v2
        assert not (v1 < v2)
        assert not (v1 > v2)

    def test_major_version_comparison(self):
        """Major version differences should determine order."""
        v1 = SemVer.parse("1.0.0")
        v2 = SemVer.parse("2.0.0")
        assert v1 < v2
        assert v2 > v1

    def test_minor_version_comparison(self):
        """Minor version differences should determine order."""
        v1 = SemVer.parse("1.1.0")
        v2 = SemVer.parse("1.2.0")
        assert v1 < v2
        assert v2 > v1

    def test_patch_version_comparison(self):
        """Patch version differences should determine order."""
        v1 = SemVer.parse("1.0.1")
        v2 = SemVer.parse("1.0.2")
        assert v1 < v2
        assert v2 > v1

    def test_prerelease_less_than_release(self):
        """Prerelease versions should be less than release."""
        v1 = SemVer.parse("1.0.0-alpha")
        v2 = SemVer.parse("1.0.0")
        assert v1 < v2
        assert v2 > v1

    def test_prerelease_comparison(self):
        """Prereleases should be compared alphanumerically."""
        v1 = SemVer.parse("1.0.0-alpha")
        v2 = SemVer.parse("1.0.0-beta")
        assert v1 < v2

    def test_prerelease_numeric_comparison(self):
        """Numeric prerelease identifiers should be compared as numbers."""
        v1 = SemVer.parse("1.0.0-alpha.1")
        v2 = SemVer.parse("1.0.0-alpha.2")
        assert v1 < v2

    def test_le_comparison(self):
        """Less than or equal comparison."""
        v1 = SemVer.parse("1.0.0")
        v2 = SemVer.parse("1.0.0")
        v3 = SemVer.parse("2.0.0")
        assert v1 <= v2
        assert v1 <= v3

    def test_ge_comparison(self):
        """Greater than or equal comparison."""
        v1 = SemVer.parse("2.0.0")
        v2 = SemVer.parse("2.0.0")
        v3 = SemVer.parse("1.0.0")
        assert v1 >= v2
        assert v1 >= v3

    def test_hash_equality(self):
        """Equal versions should have equal hashes."""
        v1 = SemVer.parse("1.2.3")
        v2 = SemVer.parse("1.2.3")
        assert hash(v1) == hash(v2)

    def test_is_compatible_with(self):
        """is_compatible_with should check major version."""
        v1 = SemVer.parse("1.2.3")
        v2 = SemVer.parse("1.5.0")
        v3 = SemVer.parse("2.0.0")
        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)


# ============================================================================
# SEMVER STRING CONVERSION TESTS
# ============================================================================


class TestSemVerString:
    """Tests for SemVer string conversion."""

    def test_str_simple_version(self):
        """Simple version to string."""
        v = SemVer(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_str_with_prerelease(self):
        """Version with prerelease to string."""
        v = SemVer(1, 0, 0, prerelease="beta.1")
        assert str(v) == "1.0.0-beta.1"

    def test_str_with_build(self):
        """Version with build to string."""
        v = SemVer(1, 0, 0, build="build.123")
        assert str(v) == "1.0.0+build.123"

    def test_str_with_prerelease_and_build(self):
        """Version with both to string."""
        v = SemVer(1, 0, 0, prerelease="alpha", build="123")
        assert str(v) == "1.0.0-alpha+123"


# ============================================================================
# SEMVER BUMP TESTS
# ============================================================================


class TestSemVerBump:
    """Tests for SemVer version bumping."""

    def test_bump_major(self):
        """Bump major version."""
        v = SemVer(1, 2, 3)
        bumped = v.bump_major()
        assert str(bumped) == "2.0.0"

    def test_bump_minor(self):
        """Bump minor version."""
        v = SemVer(1, 2, 3)
        bumped = v.bump_minor()
        assert str(bumped) == "1.3.0"

    def test_bump_patch(self):
        """Bump patch version."""
        v = SemVer(1, 2, 3)
        bumped = v.bump_patch()
        assert str(bumped) == "1.2.4"


# ============================================================================
# COMPARE VERSIONS FUNCTION TESTS
# ============================================================================


class TestCompareVersions:
    """Tests for compare_versions function."""

    def test_compare_less_than(self):
        """v1 < v2 should return -1."""
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_compare_greater_than(self):
        """v1 > v2 should return 1."""
        assert compare_versions("2.0.0", "1.0.0") == 1

    def test_compare_equal(self):
        """v1 == v2 should return 0."""
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_compare_invalid_falls_back_to_string(self):
        """Invalid versions should fall back to string comparison."""
        assert compare_versions("a", "b") == -1
        assert compare_versions("b", "a") == 1


# ============================================================================
# SORT VERSIONS FUNCTION TESTS
# ============================================================================


class TestSortVersions:
    """Tests for sort_versions function."""

    def test_sort_versions_descending(self):
        """Sort versions newest first (default)."""
        versions = ["1.0.0", "2.0.0", "1.5.0", "0.9.0"]
        sorted_v = sort_versions(versions)
        assert sorted_v == ["2.0.0", "1.5.0", "1.0.0", "0.9.0"]

    def test_sort_versions_ascending(self):
        """Sort versions oldest first."""
        versions = ["1.0.0", "2.0.0", "1.5.0", "0.9.0"]
        sorted_v = sort_versions(versions, reverse=False)
        assert sorted_v == ["0.9.0", "1.0.0", "1.5.0", "2.0.0"]

    def test_sort_versions_with_prereleases(self):
        """Sort versions including prereleases."""
        versions = ["1.0.0", "1.0.0-beta", "1.0.0-alpha"]
        sorted_v = sort_versions(versions)
        assert sorted_v == ["1.0.0", "1.0.0-beta", "1.0.0-alpha"]

    def test_sort_empty_list(self):
        """Sort empty list."""
        assert sort_versions([]) == []

    def test_sort_single_version(self):
        """Sort single version."""
        assert sort_versions(["1.0.0"]) == ["1.0.0"]


# ============================================================================
# FILTER VERSIONS FUNCTION TESTS
# ============================================================================


class TestFilterVersions:
    """Tests for filter_versions function."""

    def test_filter_with_min_version(self):
        """Filter with minimum version."""
        versions = ["1.0.0", "1.5.0", "2.0.0", "0.9.0"]
        filtered = filter_versions(versions, min_version="1.2.0")
        assert "1.5.0" in filtered
        assert "2.0.0" in filtered
        assert "1.0.0" not in filtered
        assert "0.9.0" not in filtered

    def test_filter_with_max_version(self):
        """Filter with maximum version."""
        versions = ["1.0.0", "1.5.0", "2.0.0", "0.9.0"]
        filtered = filter_versions(versions, max_version="1.5.0")
        assert "1.0.0" in filtered
        assert "1.5.0" in filtered
        assert "0.9.0" in filtered
        assert "2.0.0" not in filtered

    def test_filter_with_range(self):
        """Filter with both min and max."""
        versions = ["1.0.0", "1.5.0", "2.0.0", "0.9.0"]
        filtered = filter_versions(versions, min_version="1.0.0", max_version="1.5.0")
        assert "1.0.0" in filtered
        assert "1.5.0" in filtered
        assert "2.0.0" not in filtered
        assert "0.9.0" not in filtered

    def test_filter_exclude_prereleases(self):
        """Filter should exclude prereleases by default."""
        versions = ["1.0.0", "1.0.0-beta", "2.0.0"]
        filtered = filter_versions(versions)
        assert "1.0.0" in filtered
        assert "2.0.0" in filtered
        assert "1.0.0-beta" not in filtered

    def test_filter_include_prereleases(self):
        """Filter can include prereleases."""
        versions = ["1.0.0", "1.0.0-beta", "2.0.0"]
        filtered = filter_versions(versions, include_prerelease=True)
        assert "1.0.0-beta" in filtered


# ============================================================================
# VERSION CONSTRAINT TESTS
# ============================================================================


class TestVersionConstraint:
    """Tests for VersionConstraint class."""

    def test_exact_match_constraint(self):
        """Exact version constraint."""
        c = VersionConstraint("1.2.3")
        assert c.matches("1.2.3")
        assert not c.matches("1.2.4")
        assert not c.matches("1.2.2")

    def test_caret_constraint(self):
        """Caret constraint (^) allows same major."""
        c = VersionConstraint("^1.0.0")
        assert c.matches("1.0.0")
        assert c.matches("1.5.0")
        assert c.matches("1.9.9")
        assert not c.matches("2.0.0")
        assert not c.matches("0.9.0")

    def test_tilde_constraint(self):
        """Tilde constraint (~) allows same minor."""
        c = VersionConstraint("~1.2.0")
        assert c.matches("1.2.0")
        assert c.matches("1.2.5")
        assert not c.matches("1.3.0")
        assert not c.matches("2.0.0")

    def test_gte_constraint(self):
        """Greater than or equal constraint."""
        c = VersionConstraint(">=1.2.0")
        assert c.matches("1.2.0")
        assert c.matches("1.5.0")
        assert c.matches("2.0.0")
        assert not c.matches("1.1.0")

    def test_gt_constraint(self):
        """Greater than constraint."""
        c = VersionConstraint(">1.2.0")
        assert not c.matches("1.2.0")
        assert c.matches("1.2.1")
        assert c.matches("2.0.0")

    def test_lte_constraint(self):
        """Less than or equal constraint."""
        c = VersionConstraint("<=1.2.0")
        assert c.matches("1.2.0")
        assert c.matches("1.0.0")
        assert c.matches("0.5.0")
        assert not c.matches("1.3.0")

    def test_lt_constraint(self):
        """Less than constraint."""
        c = VersionConstraint("<1.2.0")
        assert not c.matches("1.2.0")
        assert c.matches("1.1.9")
        assert c.matches("0.9.0")

    def test_wildcard_constraint(self):
        """Wildcard constraint matches everything."""
        c = VersionConstraint("*")
        assert c.matches("1.0.0")
        assert c.matches("2.0.0")
        assert c.matches("0.0.1")

    def test_latest_constraint(self):
        """'latest' is treated as wildcard."""
        c = VersionConstraint("latest")
        assert c.matches("1.0.0")
        assert c.matches("999.0.0")

    def test_invalid_constraint_raises_error(self):
        """Invalid constraint should raise VersionConstraintError."""
        with pytest.raises(VersionConstraintError, match="Invalid version"):
            VersionConstraint("^invalid")

    def test_empty_constraint_raises_error(self):
        """Empty constraint should raise VersionConstraintError."""
        with pytest.raises(VersionConstraintError, match="Empty"):
            VersionConstraint("")

    def test_str_representation(self):
        """String representation."""
        c = VersionConstraint("^1.0.0")
        assert str(c) == "^1.0.0"

    def test_repr_representation(self):
        """Repr representation."""
        c = VersionConstraint("^1.0.0")
        assert repr(c) == "VersionConstraint('^1.0.0')"


# ============================================================================
# SCHEMA CACHE TESTS
# ============================================================================


class TestSchemaCache:
    """Tests for SchemaCache class."""

    @pytest.fixture
    def cache(self):
        """Create a cache instance."""
        return SchemaCache(ttl_seconds=60)

    @pytest.fixture
    def sample_source(self) -> SchemaSourceModel:
        """Create a sample schema source."""
        return SchemaSourceModel(
            content='{"type": "object"}',
            content_type="application/json",
            schema_id="test/schema",
            version="1.0.0",
            path="schemas/test/schema@1.0.0.json",
        )

    def test_set_and_get(self, cache, sample_source):
        """Set and get a cached schema."""
        cache.set("test@1.0.0", sample_source)
        result = cache.get("test@1.0.0")
        assert result is not None
        assert result.schema_id == "test/schema"

    def test_get_missing_returns_none(self, cache):
        """Get missing key returns None."""
        result = cache.get("missing@1.0.0")
        assert result is None

    def test_expired_entry_returns_none(self, sample_source):
        """Expired entry should return None."""
        cache = SchemaCache(ttl_seconds=0)  # Expire immediately
        cache.set("test@1.0.0", sample_source)
        time.sleep(0.01)  # Small delay to ensure expiration
        result = cache.get("test@1.0.0")
        assert result is None

    def test_delete(self, cache, sample_source):
        """Delete a cached entry."""
        cache.set("test@1.0.0", sample_source)
        assert cache.delete("test@1.0.0") is True
        assert cache.get("test@1.0.0") is None

    def test_delete_missing_returns_false(self, cache):
        """Delete missing key returns False."""
        assert cache.delete("missing@1.0.0") is False

    def test_clear(self, cache, sample_source):
        """Clear all entries."""
        cache.set("test1@1.0.0", sample_source)
        cache.set("test2@1.0.0", sample_source)
        count = cache.clear()
        assert count == 2
        assert cache.get("test1@1.0.0") is None

    def test_invalidate_prefix(self, cache, sample_source):
        """Invalidate entries with prefix."""
        cache.set("test@1.0.0", sample_source)
        cache.set("test@2.0.0", sample_source)
        cache.set("other@1.0.0", sample_source)
        count = cache.invalidate_prefix("test@")
        assert count == 2
        assert cache.get("test@1.0.0") is None
        assert cache.get("other@1.0.0") is not None

    def test_max_size_eviction(self, sample_source):
        """Max size should trigger eviction."""
        cache = SchemaCache(ttl_seconds=60, max_size=2)
        cache.set("first@1.0.0", sample_source)
        cache.set("second@1.0.0", sample_source)
        cache.set("third@1.0.0", sample_source)  # Should evict first
        assert cache.get("first@1.0.0") is None  # Evicted
        assert cache.get("second@1.0.0") is not None
        assert cache.get("third@1.0.0") is not None

    def test_get_stats(self, cache, sample_source):
        """Get cache statistics."""
        cache.set("test@1.0.0", sample_source)
        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 0


# ============================================================================
# CACHED SCHEMA TESTS
# ============================================================================


class TestCachedSchema:
    """Tests for CachedSchema dataclass."""

    def test_is_expired_false_when_fresh(self):
        """Fresh entry should not be expired."""
        source = SchemaSourceModel(
            content='{}',
            content_type="application/json",
            schema_id="test",
            version="1.0.0",
            path="test@1.0.0.json",
        )
        entry = CachedSchema(source=source, ttl_seconds=60)
        assert entry.is_expired() is False

    def test_is_expired_true_when_old(self):
        """Old entry should be expired."""
        source = SchemaSourceModel(
            content='{}',
            content_type="application/json",
            schema_id="test",
            version="1.0.0",
            path="test@1.0.0.json",
        )
        entry = CachedSchema(
            source=source,
            cached_at=time.time() - 100,
            ttl_seconds=60
        )
        assert entry.is_expired() is True


# ============================================================================
# SCHEMA SOURCE MODEL TESTS
# ============================================================================


class TestSchemaSourceModel:
    """Tests for SchemaSourceModel Pydantic model."""

    def test_create_valid_model(self):
        """Create a valid model."""
        model = SchemaSourceModel(
            content='{"type": "object"}',
            content_type="application/json",
            schema_id="test/schema",
            version="1.0.0",
            path="schemas/test/schema@1.0.0.json",
        )
        assert model.schema_id == "test/schema"
        assert model.version == "1.0.0"

    def test_optional_fields(self):
        """Optional fields should default to None."""
        model = SchemaSourceModel(
            content='{}',
            content_type="application/yaml",
            schema_id="test",
            version="1.0.0",
            path="test.yaml",
        )
        assert model.etag is None
        assert model.commit_hash is None
        assert model.parsed_content is None

    def test_invalid_content_type_raises_error(self):
        """Invalid content type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid content_type"):
            SchemaSourceModel(
                content='{}',
                content_type="text/plain",
                schema_id="test",
                version="1.0.0",
                path="test.txt",
            )


# ============================================================================
# GIT SCHEMA REGISTRY TESTS
# ============================================================================


class TestGitSchemaRegistry:
    """Tests for GitSchemaRegistry class."""

    @pytest.fixture
    def registry(self, tmp_path) -> GitSchemaRegistry:
        """Create a registry with test schemas."""
        # Create schema directory structure
        schemas = tmp_path / "schemas"
        schemas.mkdir()
        (schemas / "emissions").mkdir()
        (schemas / "energy").mkdir()

        # Create test schema files
        activity_1_0 = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        activity_1_1 = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "number"}}
        }
        activity_2_0 = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"name": {"type": "string"}, "unit": {"type": "string"}}
        }

        import yaml
        (schemas / "emissions" / "activity@1.0.0.yaml").write_text(
            yaml.dump(activity_1_0)
        )
        (schemas / "emissions" / "activity@1.1.0.yaml").write_text(
            yaml.dump(activity_1_1)
        )
        (schemas / "emissions" / "activity@2.0.0.yaml").write_text(
            yaml.dump(activity_2_0)
        )
        (schemas / "energy" / "consumption@1.0.0.json").write_text(
            json.dumps({"type": "object"})
        )

        return GitSchemaRegistry(
            repo_path=str(tmp_path),
            schema_dir="schemas",
            cache_ttl_seconds=3600,
        )

    def test_resolve_existing_schema(self, registry):
        """Resolve an existing schema."""
        source = registry.resolve("emissions/activity", "1.0.0")
        assert source.schema_id == "emissions/activity"
        assert source.version == "1.0.0"
        assert "name" in str(source.content)

    def test_resolve_json_schema(self, registry):
        """Resolve a JSON schema file."""
        source = registry.resolve("energy/consumption", "1.0.0")
        assert source.schema_id == "energy/consumption"
        assert "object" in str(source.content)

    def test_resolve_missing_schema_raises_error(self, registry):
        """Missing schema should raise SchemaNotFoundError."""
        with pytest.raises(SchemaNotFoundError) as exc_info:
            registry.resolve("emissions/activity", "9.9.9")
        assert exc_info.value.schema_id == "emissions/activity"
        assert exc_info.value.version == "9.9.9"

    def test_resolve_invalid_schema_id_raises_error(self, registry):
        """Invalid schema ID should raise InvalidSchemaIdError."""
        with pytest.raises(InvalidSchemaIdError):
            registry.resolve("INVALID-ID", "1.0.0")

    def test_resolve_caches_result(self, registry):
        """Resolved schemas should be cached."""
        source1 = registry.resolve("emissions/activity", "1.0.0")
        source2 = registry.resolve("emissions/activity", "1.0.0")
        # Second call should use cache (stats should show 1 entry)
        stats = registry.get_cache_stats()
        assert stats["total_entries"] >= 1

    def test_list_versions(self, registry):
        """List available versions for a schema."""
        versions = registry.list_versions("emissions/activity")
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions
        # Should be sorted newest first
        assert versions[0] == "2.0.0"

    def test_list_versions_empty_for_missing_schema(self, registry):
        """List versions for missing schema returns empty list."""
        versions = registry.list_versions("nonexistent/schema")
        assert versions == []

    def test_get_latest(self, registry):
        """Get latest version without constraint."""
        latest = registry.get_latest("emissions/activity")
        assert latest == "2.0.0"

    def test_get_latest_with_caret_constraint(self, registry):
        """Get latest version with caret constraint."""
        latest = registry.get_latest("emissions/activity", "^1.0.0")
        assert latest == "1.1.0"

    def test_get_latest_with_exact_constraint(self, registry):
        """Get latest with exact version constraint."""
        latest = registry.get_latest("emissions/activity", "1.0.0")
        assert latest == "1.0.0"

    def test_get_latest_no_match_raises_error(self, registry):
        """No matching version should raise SchemaNotFoundError."""
        with pytest.raises(SchemaNotFoundError):
            registry.get_latest("emissions/activity", "^3.0.0")

    def test_exists_true_for_existing(self, registry):
        """exists should return True for existing schema."""
        assert registry.exists("emissions/activity", "1.0.0") is True

    def test_exists_false_for_missing(self, registry):
        """exists should return False for missing schema."""
        assert registry.exists("emissions/activity", "9.9.9") is False

    def test_exists_false_for_invalid_id(self, registry):
        """exists should return False for invalid ID."""
        assert registry.exists("INVALID", "1.0.0") is False

    def test_list_schemas(self, registry):
        """List all available schemas."""
        schemas = registry.list_schemas()
        assert "emissions/activity" in schemas
        assert "energy/consumption" in schemas

    def test_list_schemas_by_domain(self, registry):
        """List schemas filtered by domain."""
        schemas = registry.list_schemas(domain="emissions")
        assert "emissions/activity" in schemas
        assert "energy/consumption" not in schemas

    def test_invalidate_cache_all(self, registry):
        """Invalidate entire cache."""
        registry.resolve("emissions/activity", "1.0.0")
        registry.invalidate_cache()
        stats = registry.get_cache_stats()
        assert stats["total_entries"] == 0

    def test_invalidate_cache_specific(self, registry):
        """Invalidate specific schema."""
        registry.resolve("emissions/activity", "1.0.0")
        registry.resolve("emissions/activity", "1.1.0")
        registry.invalidate_cache("emissions/activity")
        stats = registry.get_cache_stats()
        assert stats["total_entries"] == 0


class TestGitSchemaRegistryParsing:
    """Tests for schema parsing in GitSchemaRegistry."""

    @pytest.fixture
    def schema_dir(self, tmp_path) -> Path:
        """Create schema directory with various files."""
        schemas = tmp_path / "schemas" / "test"
        schemas.mkdir(parents=True)
        return schemas

    def test_resolve_parsed_returns_dict(self, schema_dir):
        """resolve_parsed should return parsed content."""
        import yaml
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        (schema_dir / "schema@1.0.0.yaml").write_text(yaml.dump(schema))

        registry = GitSchemaRegistry(
            repo_path=str(schema_dir.parent.parent),
            schema_dir="schemas",
        )
        parsed = registry.resolve_parsed("test/schema", "1.0.0")
        assert parsed["type"] == "object"
        assert "name" in parsed["properties"]

    def test_auto_parse_enabled(self, schema_dir):
        """Auto parse should include parsed content."""
        import yaml
        schema = {"type": "string"}
        (schema_dir / "simple@1.0.0.yaml").write_text(yaml.dump(schema))

        registry = GitSchemaRegistry(
            repo_path=str(schema_dir.parent.parent),
            schema_dir="schemas",
            auto_parse=True,
        )
        source = registry.resolve("test/simple", "1.0.0")
        assert "_parsed" in source.content
        assert source.content["_parsed"]["type"] == "string"

    def test_auto_parse_disabled(self, schema_dir):
        """Disabled auto parse should not include parsed content."""
        import yaml
        schema = {"type": "string"}
        (schema_dir / "simple@1.0.0.yaml").write_text(yaml.dump(schema))

        registry = GitSchemaRegistry(
            repo_path=str(schema_dir.parent.parent),
            schema_dir="schemas",
            auto_parse=False,
        )
        source = registry.resolve("test/simple", "1.0.0")
        assert "_parsed" not in source.content


class TestGitSchemaRegistrySchemaId:
    """Tests for schema ID parsing in GitSchemaRegistry."""

    @pytest.fixture
    def registry(self, tmp_path) -> GitSchemaRegistry:
        """Create a registry."""
        schemas = tmp_path / "schemas"
        schemas.mkdir()
        return GitSchemaRegistry(repo_path=str(tmp_path), schema_dir="schemas")

    def test_parse_schema_id_with_domain(self, registry):
        """Parse schema ID with domain."""
        domain, name = registry._parse_schema_id("emissions/activity")
        assert domain == "emissions"
        assert name == "activity"

    def test_parse_schema_id_without_domain(self, registry):
        """Parse schema ID without domain."""
        domain, name = registry._parse_schema_id("simple")
        assert domain == ""
        assert name == "simple"

    def test_parse_nested_domain(self, registry):
        """Parse schema ID with nested domain."""
        domain, name = registry._parse_schema_id("org/sub/schema")
        assert domain == "org/sub"
        assert name == "schema"

    def test_parse_empty_schema_id_raises_error(self, registry):
        """Empty schema ID should raise error."""
        with pytest.raises(InvalidSchemaIdError, match="Empty"):
            registry._parse_schema_id("")

    def test_parse_invalid_characters_raises_error(self, registry):
        """Invalid characters should raise error."""
        with pytest.raises(InvalidSchemaIdError, match="lowercase"):
            registry._parse_schema_id("Invalid/Schema")


# ============================================================================
# GIT OPERATIONS TESTS
# ============================================================================


class TestGitOperations:
    """Tests for Git operations in GitSchemaRegistry."""

    @pytest.fixture
    def registry(self, tmp_path) -> GitSchemaRegistry:
        """Create a registry with remote configured."""
        schemas = tmp_path / "schemas"
        schemas.mkdir()
        return GitSchemaRegistry(
            repo_path=str(tmp_path),
            schema_dir="schemas",
            remote_url="https://example.com/schemas.git",
            branch="main",
        )

    def test_pull_without_remote_returns_false(self, tmp_path):
        """Pull without remote should return False."""
        registry = GitSchemaRegistry(
            repo_path=str(tmp_path),
            schema_dir="schemas",
        )
        assert registry.pull() is False

    @patch('subprocess.run')
    def test_get_commit_hash(self, mock_run, registry):
        """Get commit hash should call git rev-parse."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456\n"
        )
        commit_hash = registry._get_commit_hash()
        assert commit_hash == "abc123def456"

    @patch('subprocess.run')
    def test_get_commit_hash_caches_result(self, mock_run, registry):
        """Commit hash should be cached."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123\n"
        )
        registry._get_commit_hash()
        registry._get_commit_hash()
        # Should only call git once
        assert mock_run.call_count == 1


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestCreateGitRegistry:
    """Tests for create_git_registry factory function."""

    def test_create_basic_registry(self, tmp_path):
        """Create a basic registry."""
        schemas = tmp_path / "schemas"
        schemas.mkdir()
        registry = create_git_registry(str(tmp_path), schema_dir="schemas")
        assert isinstance(registry, GitSchemaRegistry)

    @patch.object(GitSchemaRegistry, 'pull')
    def test_auto_pull_on_creation(self, mock_pull, tmp_path):
        """Auto pull should call pull method."""
        schemas = tmp_path / "schemas"
        schemas.mkdir()
        mock_pull.return_value = True
        registry = create_git_registry(
            str(tmp_path),
            schema_dir="schemas",
            remote_url="https://example.com/schemas.git",
            auto_pull=True,
        )
        mock_pull.assert_called_once()

    @patch.object(GitSchemaRegistry, 'pull', side_effect=GitOperationError("pull", "failed"))
    def test_auto_pull_handles_failure(self, mock_pull, tmp_path):
        """Auto pull failure should not raise exception."""
        schemas = tmp_path / "schemas"
        schemas.mkdir()
        # Should not raise
        registry = create_git_registry(
            str(tmp_path),
            schema_dir="schemas",
            remote_url="https://example.com/schemas.git",
            auto_pull=True,
        )
        assert registry is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
