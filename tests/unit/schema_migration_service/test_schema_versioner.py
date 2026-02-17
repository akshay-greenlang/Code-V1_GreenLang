# -*- coding: utf-8 -*-
"""
Unit Tests for SchemaVersionerEngine (AGENT-DATA-017)

Tests semantic versioning lifecycle: version creation with automatic SemVer
bump classification, version retrieval, listing with filters and pagination,
latest version resolution, version comparison, deprecation / undeprecation
with sunset date management, consumer version-range pinning, changelog
retrieval, bump classification, statistics, reset, and edge cases.

Coverage target: 85%+ of schema_versioner.py (~100 tests)

Test classes:
    - TestSchemaVersionerInit           (5 tests)
    - TestCreateVersion                 (22 tests)
    - TestGetVersion                    (7 tests)
    - TestListVersions                  (12 tests)
    - TestGetLatestVersion              (6 tests)
    - TestCompareVersions               (10 tests)
    - TestDeprecateVersion              (9 tests)
    - TestVersionPinning                (11 tests)
    - TestClassifyBump                  (15 tests)
    - TestGetChangelog                  (7 tests)
    - TestSchemaVersionerEdgeCases      (8 tests)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from greenlang.schema_migration.schema_versioner import (
    BUMP_MAJOR,
    BUMP_MINOR,
    BUMP_PATCH,
    SchemaVersionerEngine,
    _bump_version,
    _classify_bump,
    _compute_provenance,
    _generate_id,
    _parse_version,
    _serialize_definition,
    _version_tuple_key,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine() -> SchemaVersionerEngine:
    """Create a fresh SchemaVersionerEngine for each test."""
    return SchemaVersionerEngine()


@pytest.fixture
def simple_definition() -> Dict[str, Any]:
    """A minimal JSON Schema definition."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
        },
    }


@pytest.fixture
def extended_definition() -> Dict[str, Any]:
    """A definition with an additional field (additive change)."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string"},
        },
    }


@pytest.fixture
def reduced_definition() -> Dict[str, Any]:
    """A definition with a field removed (breaking change)."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
        },
    }


@pytest.fixture
def cosmetic_definition() -> Dict[str, Any]:
    """Same fields, slightly different metadata (cosmetic change)."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Primary key"},
            "name": {"type": "string", "description": "Full name"},
        },
    }


@pytest.fixture
def breaking_changes() -> List[Dict[str, Any]]:
    """Change list classified as breaking."""
    return [
        {"field": "name", "severity": "breaking", "description": "field removed"},
    ]


@pytest.fixture
def non_breaking_changes() -> List[Dict[str, Any]]:
    """Change list classified as non-breaking (additive)."""
    return [
        {"field": "email", "severity": "non_breaking", "description": "field added"},
    ]


@pytest.fixture
def cosmetic_changes() -> List[Dict[str, Any]]:
    """Change list classified as cosmetic."""
    return [
        {"field": "id", "severity": "cosmetic", "description": "description updated"},
    ]


@pytest.fixture
def engine_with_versions(
    engine: SchemaVersionerEngine,
    simple_definition: Dict[str, Any],
    extended_definition: Dict[str, Any],
    non_breaking_changes: List[Dict[str, Any]],
) -> SchemaVersionerEngine:
    """Engine pre-loaded with two versions of schema 'sch-orders'."""
    engine.create_version("sch-orders", simple_definition, changelog_note="Initial")
    engine.create_version(
        "sch-orders",
        extended_definition,
        changes=non_breaking_changes,
        changelog_note="Added email",
    )
    return engine


# ============================================================================
# TestSchemaVersionerInit
# ============================================================================


class TestSchemaVersionerInit:
    """Tests for SchemaVersionerEngine initialisation."""

    def test_init_creates_engine(self) -> None:
        """Engine can be instantiated without arguments."""
        engine = SchemaVersionerEngine()
        assert engine is not None

    def test_init_empty_versions(self, engine: SchemaVersionerEngine) -> None:
        """Freshly initialised engine has no versions."""
        stats = engine.get_statistics()
        assert stats["total_versions"] == 0

    def test_init_empty_pins(self, engine: SchemaVersionerEngine) -> None:
        """Freshly initialised engine has no pins."""
        stats = engine.get_statistics()
        assert stats["total_pins"] == 0

    def test_init_zero_deprecated(self, engine: SchemaVersionerEngine) -> None:
        """Freshly initialised engine has zero deprecated versions."""
        stats = engine.get_statistics()
        assert stats["deprecated_count"] == 0

    def test_init_statistics_structure(self, engine: SchemaVersionerEngine) -> None:
        """Statistics dict has the expected top-level keys."""
        stats = engine.get_statistics()
        expected_keys = {
            "total_versions",
            "active_versions",
            "deprecated_count",
            "by_bump_type",
            "by_schema",
            "total_schemas",
            "total_pins",
            "collected_at",
        }
        assert expected_keys.issubset(set(stats.keys()))


# ============================================================================
# TestCreateVersion
# ============================================================================


class TestCreateVersion:
    """Tests for SchemaVersionerEngine.create_version."""

    def test_first_version_is_1_0_0(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """The very first version for a schema is always 1.0.0."""
        v = engine.create_version("sch-001", simple_definition)
        assert v["version"] == "1.0.0"

    def test_first_version_bump_type_is_patch(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """First version has bump_type 'patch' (no prior version to bump from)."""
        v = engine.create_version("sch-001", simple_definition)
        assert v["bump_type"] == BUMP_PATCH

    def test_auto_bump_major(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        reduced_definition: Dict[str, Any],
        breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """Breaking changes trigger a major bump."""
        engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version(
            "sch-001", reduced_definition, changes=breaking_changes
        )
        assert v2["version"] == "2.0.0"
        assert v2["bump_type"] == BUMP_MAJOR

    def test_auto_bump_minor(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        extended_definition: Dict[str, Any],
        non_breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """Additive / non-breaking changes trigger a minor bump."""
        engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version(
            "sch-001", extended_definition, changes=non_breaking_changes
        )
        assert v2["version"] == "1.1.0"
        assert v2["bump_type"] == BUMP_MINOR

    def test_auto_bump_patch(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        cosmetic_definition: Dict[str, Any],
        cosmetic_changes: List[Dict[str, Any]],
    ) -> None:
        """Cosmetic changes trigger a patch bump."""
        engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version(
            "sch-001", cosmetic_definition, changes=cosmetic_changes
        )
        assert v2["version"] == "1.0.1"
        assert v2["bump_type"] == BUMP_PATCH

    def test_no_changes_defaults_to_patch(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """When no changes list is provided, bump defaults to patch."""
        engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version("sch-001", simple_definition)
        assert v2["version"] == "1.0.1"

    def test_version_id_has_ver_prefix(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Version ID starts with 'VER-' prefix."""
        v = engine.create_version("sch-001", simple_definition)
        assert v["id"].startswith("VER-")

    def test_version_id_is_unique(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Each version receives a unique ID."""
        v1 = engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version("sch-001", simple_definition)
        assert v1["id"] != v2["id"]

    def test_version_string_format(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Version string matches X.Y.Z format."""
        v = engine.create_version("sch-001", simple_definition)
        assert re.match(r"^\d+\.\d+\.\d+$", v["version"])

    def test_schema_id_stored(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Version record stores the correct schema_id."""
        v = engine.create_version("sch-orders", simple_definition)
        assert v["schema_id"] == "sch-orders"

    def test_definition_stored(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Version record stores the original definition."""
        v = engine.create_version("sch-001", simple_definition)
        assert v["definition"] == simple_definition

    def test_definition_hash_is_sha256(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Definition hash is a 64-character hex string (SHA-256)."""
        v = engine.create_version("sch-001", simple_definition)
        assert len(v["definition_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in v["definition_hash"])

    def test_definition_hash_deterministic(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Same definition produces same hash across versions."""
        v1 = engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version("sch-001", simple_definition)
        assert v1["definition_hash"] == v2["definition_hash"]

    def test_changelog_note_stored(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """changelog_note is stored on the version record."""
        v = engine.create_version(
            "sch-001", simple_definition, changelog_note="Initial release"
        )
        assert v["changelog_note"] == "Initial release"

    def test_created_by_default(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Default created_by is 'system'."""
        v = engine.create_version("sch-001", simple_definition)
        assert v["created_by"] == "system"

    def test_created_by_custom(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Custom created_by is honoured."""
        v = engine.create_version(
            "sch-001", simple_definition, created_by="user-42"
        )
        assert v["created_by"] == "user-42"

    def test_created_at_is_iso8601(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """created_at is a valid ISO-8601 timestamp."""
        v = engine.create_version("sch-001", simple_definition)
        dt = datetime.fromisoformat(v["created_at"])
        assert dt.tzinfo is not None

    def test_is_deprecated_false_on_creation(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """New versions are not deprecated."""
        v = engine.create_version("sch-001", simple_definition)
        assert v["is_deprecated"] is False
        assert v["deprecated_at"] is None
        assert v["sunset_date"] is None
        assert v["deprecation_reason"] == ""

    def test_provenance_hash_on_creation(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Provenance hash is a 64-char hex SHA-256."""
        v = engine.create_version("sch-001", simple_definition)
        assert len(v["provenance_hash"]) == 64

    def test_statistics_updated_on_create(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Statistics counters increment after version creation."""
        engine.create_version("sch-001", simple_definition)
        stats = engine.get_statistics()
        assert stats["total_versions"] == 1
        assert stats["total_schemas"] == 1
        assert stats["by_schema"]["sch-001"] == 1

    def test_empty_schema_id_raises_value_error(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Empty schema_id raises ValueError."""
        with pytest.raises(ValueError, match="schema_id"):
            engine.create_version("", simple_definition)

    def test_non_serialisable_definition_raises_value_error(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Non-JSON-serialisable definition raises ValueError."""
        with pytest.raises(ValueError, match="JSON-serialisable"):
            engine.create_version("sch-001", {"fn": lambda: None})

    # -- Multi-version sequencing --

    def test_major_resets_minor_and_patch(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        breaking_changes: List[Dict[str, Any]],
        non_breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """Major bump after minor versions resets minor and patch to 0."""
        engine.create_version("sch-001", simple_definition)
        engine.create_version("sch-001", simple_definition, changes=non_breaking_changes)
        # Now at 1.1.0
        v3 = engine.create_version("sch-001", simple_definition, changes=breaking_changes)
        assert v3["version"] == "2.0.0"

    def test_minor_resets_patch(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        cosmetic_changes: List[Dict[str, Any]],
        non_breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """Minor bump after patch versions resets patch to 0."""
        engine.create_version("sch-001", simple_definition)
        engine.create_version("sch-001", simple_definition, changes=cosmetic_changes)
        # Now at 1.0.1
        v3 = engine.create_version(
            "sch-001", simple_definition, changes=non_breaking_changes
        )
        assert v3["version"] == "1.1.0"


# ============================================================================
# TestGetVersion
# ============================================================================


class TestGetVersion:
    """Tests for SchemaVersionerEngine.get_version."""

    def test_get_existing_version(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Retrieve a version by its ID."""
        v = engine.create_version("sch-001", simple_definition)
        result = engine.get_version(v["id"])
        assert result is not None
        assert result["id"] == v["id"]

    def test_get_nonexistent_version(self, engine: SchemaVersionerEngine) -> None:
        """Non-existent version_id returns None."""
        assert engine.get_version("VER-does-not-exist") is None

    def test_get_version_preserves_all_fields(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Returned record contains all expected keys."""
        v = engine.create_version("sch-001", simple_definition, changelog_note="test")
        result = engine.get_version(v["id"])
        expected_keys = {
            "id", "schema_id", "version", "bump_type", "definition",
            "definition_hash", "changes", "changelog", "changelog_note",
            "created_by", "created_at", "is_deprecated", "deprecated_at",
            "sunset_date", "deprecation_reason", "provenance_hash",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_get_version_after_deprecation(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Deprecated version is still retrievable by ID."""
        v = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v["id"])
        result = engine.get_version(v["id"])
        assert result is not None
        assert result["is_deprecated"] is True

    def test_get_version_returns_correct_definition(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Definition content is intact on retrieval."""
        v = engine.create_version("sch-001", simple_definition)
        result = engine.get_version(v["id"])
        assert result["definition"] == simple_definition

    def test_get_version_by_string(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Retrieve version by schema_id and SemVer string."""
        engine.create_version("sch-001", simple_definition)
        result = engine.get_version_by_string("sch-001", "1.0.0")
        assert result is not None
        assert result["version"] == "1.0.0"

    def test_get_version_by_string_nonexistent(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Non-existent SemVer string returns None."""
        engine.create_version("sch-001", simple_definition)
        assert engine.get_version_by_string("sch-001", "9.9.9") is None


# ============================================================================
# TestListVersions
# ============================================================================


class TestListVersions:
    """Tests for SchemaVersionerEngine.list_versions."""

    def test_list_versions_returns_all(
        self, engine_with_versions: SchemaVersionerEngine
    ) -> None:
        """list_versions returns all non-deprecated versions for a schema."""
        versions = engine_with_versions.list_versions("sch-orders")
        assert len(versions) == 2

    def test_list_versions_sorted_descending(
        self, engine_with_versions: SchemaVersionerEngine
    ) -> None:
        """Versions are sorted newest-first (descending SemVer)."""
        versions = engine_with_versions.list_versions("sch-orders")
        assert versions[0]["version"] == "1.1.0"
        assert versions[1]["version"] == "1.0.0"

    def test_list_versions_empty_schema(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Unknown schema returns empty list."""
        assert engine.list_versions("sch-nonexistent") == []

    def test_list_versions_exclude_deprecated_by_default(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Deprecated versions are excluded when include_deprecated=False (default)."""
        v1 = engine.create_version("sch-001", simple_definition)
        engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v1["id"])
        versions = engine.list_versions("sch-001", include_deprecated=False)
        assert len(versions) == 1

    def test_list_versions_include_deprecated(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """include_deprecated=True returns all versions."""
        v1 = engine.create_version("sch-001", simple_definition)
        engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v1["id"])
        versions = engine.list_versions("sch-001", include_deprecated=True)
        assert len(versions) == 2

    def test_list_versions_pagination_limit(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Limit parameter restricts number of returned records."""
        for _ in range(5):
            engine.create_version("sch-001", simple_definition)
        versions = engine.list_versions("sch-001", limit=3)
        assert len(versions) == 3

    def test_list_versions_pagination_offset(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Offset parameter skips records."""
        for _ in range(5):
            engine.create_version("sch-001", simple_definition)
        versions = engine.list_versions("sch-001", offset=2, limit=100)
        assert len(versions) == 3

    def test_list_versions_offset_beyond_total(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Offset exceeding total returns empty list."""
        engine.create_version("sch-001", simple_definition)
        versions = engine.list_versions("sch-001", offset=100)
        assert versions == []

    def test_list_versions_multiple_schemas_isolated(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Versions of different schemas are isolated."""
        engine.create_version("sch-a", simple_definition)
        engine.create_version("sch-a", simple_definition)
        engine.create_version("sch-b", simple_definition)
        assert len(engine.list_versions("sch-a")) == 2
        assert len(engine.list_versions("sch-b")) == 1

    def test_list_versions_combined_limit_and_offset(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Combined limit + offset produces correct window."""
        for _ in range(10):
            engine.create_version("sch-001", simple_definition)
        versions = engine.list_versions("sch-001", offset=3, limit=4)
        assert len(versions) == 4

    def test_list_versions_default_limit(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Default limit is 100."""
        for _ in range(5):
            engine.create_version("sch-001", simple_definition)
        versions = engine.list_versions("sch-001")
        assert len(versions) == 5  # All returned under default limit

    def test_list_versions_all_deprecated_returns_empty(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """When all versions are deprecated, default listing returns empty."""
        v1 = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v1["id"])
        versions = engine.list_versions("sch-001", include_deprecated=False)
        assert versions == []


# ============================================================================
# TestGetLatestVersion
# ============================================================================


class TestGetLatestVersion:
    """Tests for SchemaVersionerEngine.get_latest_version."""

    def test_latest_single_version(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """With one version, latest returns that version."""
        engine.create_version("sch-001", simple_definition)
        latest = engine.get_latest_version("sch-001")
        assert latest is not None
        assert latest["version"] == "1.0.0"

    def test_latest_multiple_versions(
        self,
        engine_with_versions: SchemaVersionerEngine,
    ) -> None:
        """With multiple versions, latest returns the highest SemVer."""
        latest = engine_with_versions.get_latest_version("sch-orders")
        assert latest["version"] == "1.1.0"

    def test_latest_no_versions(self, engine: SchemaVersionerEngine) -> None:
        """Unknown schema returns None."""
        assert engine.get_latest_version("sch-nonexistent") is None

    def test_latest_excludes_deprecated(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Latest skips deprecated versions."""
        engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v2["id"])
        latest = engine.get_latest_version("sch-001")
        assert latest is not None
        assert latest["version"] == "1.0.0"

    def test_latest_all_deprecated_returns_none(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """When all versions are deprecated, latest returns None."""
        v1 = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v1["id"])
        assert engine.get_latest_version("sch-001") is None

    def test_latest_after_major_bump(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """Latest correctly reflects a major bump."""
        engine.create_version("sch-001", simple_definition)
        engine.create_version("sch-001", simple_definition, changes=breaking_changes)
        latest = engine.get_latest_version("sch-001")
        assert latest["version"] == "2.0.0"


# ============================================================================
# TestCompareVersions
# ============================================================================


class TestCompareVersions:
    """Tests for SchemaVersionerEngine.compare_versions."""

    def test_compare_same_version(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Comparing a version with itself shows no definition change."""
        v = engine.create_version("sch-001", simple_definition)
        diff = engine.compare_versions(v["id"], v["id"])
        assert diff["definition_changed"] is False
        assert diff["version_relation"] == "equal"

    def test_compare_b_newer(
        self,
        engine_with_versions: SchemaVersionerEngine,
    ) -> None:
        """When version B is newer, relation is 'b_newer'."""
        versions = engine_with_versions.list_versions("sch-orders")
        # versions[0] = 1.1.0 (newer), versions[1] = 1.0.0 (older)
        diff = engine_with_versions.compare_versions(
            versions[1]["id"], versions[0]["id"]
        )
        assert diff["version_relation"] == "b_newer"

    def test_compare_a_newer(
        self,
        engine_with_versions: SchemaVersionerEngine,
    ) -> None:
        """When version A is newer, relation is 'a_newer'."""
        versions = engine_with_versions.list_versions("sch-orders")
        diff = engine_with_versions.compare_versions(
            versions[0]["id"], versions[1]["id"]
        )
        assert diff["version_relation"] == "a_newer"

    def test_compare_definition_changed(
        self,
        engine_with_versions: SchemaVersionerEngine,
    ) -> None:
        """Different definitions produce definition_changed=True."""
        versions = engine_with_versions.list_versions("sch-orders")
        diff = engine_with_versions.compare_versions(
            versions[1]["id"], versions[0]["id"]
        )
        assert diff["definition_changed"] is True

    def test_compare_added_keys(
        self,
        engine: SchemaVersionerEngine,
    ) -> None:
        """added_keys lists keys in B not in A."""
        v1 = engine.create_version("sch-001", {"id": 1})
        v2 = engine.create_version("sch-001", {"id": 1, "name": "x"})
        diff = engine.compare_versions(v1["id"], v2["id"])
        assert "name" in diff["added_keys"]

    def test_compare_removed_keys(
        self,
        engine: SchemaVersionerEngine,
    ) -> None:
        """removed_keys lists keys in A not in B."""
        v1 = engine.create_version("sch-001", {"id": 1, "name": "x"})
        v2 = engine.create_version("sch-001", {"id": 1})
        diff = engine.compare_versions(v1["id"], v2["id"])
        assert "name" in diff["removed_keys"]

    def test_compare_common_keys(
        self,
        engine: SchemaVersionerEngine,
    ) -> None:
        """common_keys lists keys in both A and B."""
        v1 = engine.create_version("sch-001", {"id": 1, "name": "x"})
        v2 = engine.create_version("sch-001", {"id": 2, "email": "e"})
        diff = engine.compare_versions(v1["id"], v2["id"])
        assert "id" in diff["common_keys"]

    def test_compare_same_schema_flag(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """same_schema is True when both versions are on the same schema."""
        v1 = engine.create_version("sch-001", simple_definition)
        v2 = engine.create_version("sch-001", simple_definition)
        diff = engine.compare_versions(v1["id"], v2["id"])
        assert diff["same_schema"] is True

    def test_compare_different_schemas(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """same_schema is False for cross-schema comparison."""
        v1 = engine.create_version("sch-a", simple_definition)
        v2 = engine.create_version("sch-b", simple_definition)
        diff = engine.compare_versions(v1["id"], v2["id"])
        assert diff["same_schema"] is False

    def test_compare_nonexistent_raises(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Comparing with a non-existent version raises ValueError."""
        v = engine.create_version("sch-001", simple_definition)
        with pytest.raises(ValueError, match="not found"):
            engine.compare_versions(v["id"], "VER-nonexistent")
        with pytest.raises(ValueError, match="not found"):
            engine.compare_versions("VER-nonexistent", v["id"])


# ============================================================================
# TestDeprecateVersion
# ============================================================================


class TestDeprecateVersion:
    """Tests for deprecate_version and undeprecate_version."""

    def test_deprecate_sets_flag(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Deprecating a version sets is_deprecated to True."""
        v = engine.create_version("sch-001", simple_definition)
        result = engine.deprecate_version(v["id"])
        assert result["is_deprecated"] is True

    def test_deprecate_with_sunset_date(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """sunset_date is stored when provided."""
        v = engine.create_version("sch-001", simple_definition)
        result = engine.deprecate_version(v["id"], sunset_date="2027-06-30")
        assert result["sunset_date"] == "2027-06-30"

    def test_deprecate_without_sunset_date(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """sunset_date is None when omitted."""
        v = engine.create_version("sch-001", simple_definition)
        result = engine.deprecate_version(v["id"])
        assert result["sunset_date"] is None

    def test_deprecate_with_reason(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Deprecation reason is stored."""
        v = engine.create_version("sch-001", simple_definition)
        result = engine.deprecate_version(v["id"], reason="Superseded by v2")
        assert result["deprecation_reason"] == "Superseded by v2"

    def test_deprecate_nonexistent_raises(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Deprecating a non-existent version raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.deprecate_version("VER-nonexistent")

    def test_deprecate_already_deprecated_raises(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Double deprecation raises ValueError."""
        v = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v["id"])
        with pytest.raises(ValueError, match="already deprecated"):
            engine.deprecate_version(v["id"])

    def test_deprecate_invalid_sunset_date_raises(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Invalid sunset_date format raises ValueError."""
        v = engine.create_version("sch-001", simple_definition)
        with pytest.raises(ValueError, match="not a valid ISO-8601"):
            engine.deprecate_version(v["id"], sunset_date="not-a-date")

    def test_undeprecate_clears_flag(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Undeprecating restores is_deprecated to False."""
        v = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v["id"], sunset_date="2027-01-01", reason="old")
        result = engine.undeprecate_version(v["id"])
        assert result["is_deprecated"] is False
        assert result["deprecated_at"] is None
        assert result["sunset_date"] is None
        assert result["deprecation_reason"] == ""

    def test_undeprecate_not_deprecated_raises(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Undeprecating a non-deprecated version raises ValueError."""
        v = engine.create_version("sch-001", simple_definition)
        with pytest.raises(ValueError, match="not deprecated"):
            engine.undeprecate_version(v["id"])


# ============================================================================
# TestVersionPinning
# ============================================================================


class TestVersionPinning:
    """Tests for pin_version and get_statistics (pins)."""

    def test_pin_version_returns_record(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """pin_version returns a record with expected fields."""
        pin = engine.pin_version("sch-001", "consumer-a", ">=1.0.0 <2.0.0")
        assert "pin_id" in pin
        assert pin["schema_id"] == "sch-001"
        assert pin["consumer_id"] == "consumer-a"
        assert pin["version_range"] == ">=1.0.0 <2.0.0"

    def test_pin_id_has_prefix(self, engine: SchemaVersionerEngine) -> None:
        """Pin ID starts with 'PIN-' prefix."""
        pin = engine.pin_version("sch-001", "consumer-a", "1.x")
        assert pin["pin_id"].startswith("PIN-")

    def test_pin_provenance_hash(self, engine: SchemaVersionerEngine) -> None:
        """Pin record has a 64-char provenance hash."""
        pin = engine.pin_version("sch-001", "consumer-a", "1.x")
        assert len(pin["provenance_hash"]) == 64

    def test_pin_timestamp(self, engine: SchemaVersionerEngine) -> None:
        """Pin record has a valid pinned_at timestamp."""
        pin = engine.pin_version("sch-001", "consumer-a", "1.x")
        dt = datetime.fromisoformat(pin["pinned_at"])
        assert dt.tzinfo is not None

    def test_pin_with_semver_range(self, engine: SchemaVersionerEngine) -> None:
        """Exact SemVer range expression is stored verbatim."""
        range_str = ">=2.0.0 <3.0.0"
        pin = engine.pin_version("sch-001", "svc-orders", range_str)
        assert pin["version_range"] == range_str

    def test_pin_with_tilde_range(self, engine: SchemaVersionerEngine) -> None:
        """Tilde range is accepted and stored."""
        pin = engine.pin_version("sch-001", "svc-x", "~1.2.0")
        assert pin["version_range"] == "~1.2.0"

    def test_multiple_pins_different_schemas(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """A consumer can pin different schemas independently."""
        engine.pin_version("sch-a", "consumer-1", "1.x")
        engine.pin_version("sch-b", "consumer-1", "2.x")
        stats = engine.get_statistics()
        assert stats["total_pins"] == 2

    def test_pin_overwrites_same_consumer_schema(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Re-pinning the same consumer+schema updates the range."""
        engine.pin_version("sch-001", "consumer-a", "1.x")
        engine.pin_version("sch-001", "consumer-a", "2.x")
        stats = engine.get_statistics()
        # total_pins counter increments each call (it counts creations)
        assert stats["total_pins"] == 2

    def test_pin_empty_schema_id_raises(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Empty schema_id raises ValueError."""
        with pytest.raises(ValueError, match="schema_id"):
            engine.pin_version("", "consumer-a", "1.x")

    def test_pin_empty_consumer_id_raises(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Empty consumer_id raises ValueError."""
        with pytest.raises(ValueError, match="consumer_id"):
            engine.pin_version("sch-001", "", "1.x")

    def test_pin_empty_version_range_raises(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Empty version_range raises ValueError."""
        with pytest.raises(ValueError, match="version_range"):
            engine.pin_version("sch-001", "consumer-a", "")


# ============================================================================
# TestClassifyBump
# ============================================================================


class TestClassifyBump:
    """Tests for classify_bump (public method) and _classify_bump (module-level)."""

    def test_empty_changes_returns_patch(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Empty change list returns 'patch'."""
        assert engine.classify_bump([]) == BUMP_PATCH

    def test_breaking_returns_major(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """A single breaking change returns 'major'."""
        changes = [{"severity": "breaking"}]
        assert engine.classify_bump(changes) == BUMP_MAJOR

    def test_major_alias(self, engine: SchemaVersionerEngine) -> None:
        """'major' severity alias maps to major bump."""
        assert engine.classify_bump([{"severity": "major"}]) == BUMP_MAJOR

    def test_breaking_change_alias(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """'breaking_change' severity alias maps to major bump."""
        assert engine.classify_bump([{"severity": "breaking_change"}]) == BUMP_MAJOR

    def test_incompatible_alias(self, engine: SchemaVersionerEngine) -> None:
        """'incompatible' severity alias maps to major bump."""
        assert engine.classify_bump([{"severity": "incompatible"}]) == BUMP_MAJOR

    def test_non_breaking_returns_minor(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """A single non-breaking change returns 'minor'."""
        changes = [{"severity": "non_breaking"}]
        assert engine.classify_bump(changes) == BUMP_MINOR

    def test_minor_alias(self, engine: SchemaVersionerEngine) -> None:
        """'minor' severity alias maps to minor bump."""
        assert engine.classify_bump([{"severity": "minor"}]) == BUMP_MINOR

    def test_additive_alias(self, engine: SchemaVersionerEngine) -> None:
        """'additive' severity alias maps to minor bump."""
        assert engine.classify_bump([{"severity": "additive"}]) == BUMP_MINOR

    def test_feature_alias(self, engine: SchemaVersionerEngine) -> None:
        """'feature' severity alias maps to minor bump."""
        assert engine.classify_bump([{"severity": "feature"}]) == BUMP_MINOR

    def test_backward_compatible_alias(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """'backward_compatible' severity alias maps to minor bump."""
        assert engine.classify_bump([{"severity": "backward_compatible"}]) == BUMP_MINOR

    def test_cosmetic_returns_patch(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Cosmetic change returns 'patch'."""
        assert engine.classify_bump([{"severity": "cosmetic"}]) == BUMP_PATCH

    def test_documentation_alias(self, engine: SchemaVersionerEngine) -> None:
        """'documentation' severity alias maps to patch bump."""
        assert engine.classify_bump([{"severity": "documentation"}]) == BUMP_PATCH

    def test_mixed_breaking_wins(self, engine: SchemaVersionerEngine) -> None:
        """If any change is breaking, result is major (breaking wins)."""
        changes = [
            {"severity": "cosmetic"},
            {"severity": "non_breaking"},
            {"severity": "breaking"},
        ]
        assert engine.classify_bump(changes) == BUMP_MAJOR

    def test_mixed_non_breaking_wins_over_cosmetic(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """non_breaking wins over cosmetic when no breaking present."""
        changes = [
            {"severity": "cosmetic"},
            {"severity": "non_breaking"},
        ]
        assert engine.classify_bump(changes) == BUMP_MINOR

    def test_missing_severity_defaults_to_patch(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Change without a severity key defaults to patch-level."""
        assert engine.classify_bump([{"field": "x"}]) == BUMP_PATCH


# ============================================================================
# TestGetChangelog
# ============================================================================


class TestGetChangelog:
    """Tests for SchemaVersionerEngine.get_changelog."""

    def test_changelog_ordered_ascending(
        self, engine_with_versions: SchemaVersionerEngine
    ) -> None:
        """Changelog entries are in ascending SemVer order (oldest first)."""
        log = engine_with_versions.get_changelog("sch-orders")
        assert len(log) == 2
        assert log[0]["version"] == "1.0.0"
        assert log[1]["version"] == "1.1.0"

    def test_changelog_empty_schema(self, engine: SchemaVersionerEngine) -> None:
        """Unknown schema returns empty changelog."""
        assert engine.get_changelog("sch-nonexistent") == []

    def test_changelog_single_version(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Single version produces single changelog entry."""
        engine.create_version("sch-001", simple_definition, changelog_note="Init")
        log = engine.get_changelog("sch-001")
        assert len(log) == 1
        assert log[0]["changelog_note"] == "Init"

    def test_changelog_entry_structure(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Each changelog entry has expected keys."""
        engine.create_version("sch-001", simple_definition, changelog_note="Init")
        entry = engine.get_changelog("sch-001")[0]
        expected_keys = {
            "version_id", "schema_id", "version", "bump_type",
            "changelog", "changelog_note", "created_by", "created_at",
            "is_deprecated",
        }
        assert expected_keys.issubset(set(entry.keys()))

    def test_changelog_from_version_filter(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        non_breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """from_version filters out older versions."""
        engine.create_version("sch-001", simple_definition)
        engine.create_version(
            "sch-001", simple_definition, changes=non_breaking_changes
        )
        log = engine.get_changelog("sch-001", from_version="1.1.0")
        assert len(log) == 1
        assert log[0]["version"] == "1.1.0"

    def test_changelog_to_version_filter(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        non_breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """to_version filters out newer versions."""
        engine.create_version("sch-001", simple_definition)
        engine.create_version(
            "sch-001", simple_definition, changes=non_breaking_changes
        )
        log = engine.get_changelog("sch-001", to_version="1.0.0")
        assert len(log) == 1
        assert log[0]["version"] == "1.0.0"

    def test_changelog_range_filter(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        non_breaking_changes: List[Dict[str, Any]],
        breaking_changes: List[Dict[str, Any]],
    ) -> None:
        """Combined from_version + to_version range filter."""
        engine.create_version("sch-001", simple_definition)
        engine.create_version(
            "sch-001", simple_definition, changes=non_breaking_changes
        )
        engine.create_version(
            "sch-001", simple_definition, changes=breaking_changes
        )
        # Versions: 1.0.0, 1.1.0, 2.0.0
        log = engine.get_changelog(
            "sch-001", from_version="1.1.0", to_version="1.1.0"
        )
        assert len(log) == 1
        assert log[0]["version"] == "1.1.0"


# ============================================================================
# TestSchemaVersionerEdgeCases
# ============================================================================


class TestSchemaVersionerEdgeCases:
    """Edge case and boundary tests."""

    def test_many_versions_sequential(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Engine handles many sequential versions (50+)."""
        for i in range(50):
            engine.create_version("sch-stress", simple_definition)
        stats = engine.get_statistics()
        assert stats["total_versions"] == 50
        assert stats["by_schema"]["sch-stress"] == 50

    def test_many_schemas(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Engine handles many distinct schemas."""
        for i in range(100):
            engine.create_version(f"sch-{i:04d}", simple_definition)
        stats = engine.get_statistics()
        assert stats["total_schemas"] == 100

    def test_reset_clears_all(
        self,
        engine_with_versions: SchemaVersionerEngine,
    ) -> None:
        """reset() clears all state."""
        engine_with_versions.reset()
        stats = engine_with_versions.get_statistics()
        assert stats["total_versions"] == 0
        assert stats["total_schemas"] == 0
        assert stats["total_pins"] == 0
        assert stats["deprecated_count"] == 0

    def test_thread_safety_concurrent_creation(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """Concurrent version creation does not corrupt state."""
        errors: List[Exception] = []

        def _create(schema_id: str, n: int) -> None:
            try:
                for _ in range(n):
                    engine.create_version(schema_id, simple_definition)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_create, args=(f"sch-t{i}", 10))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        stats = engine.get_statistics()
        assert stats["total_versions"] == 50
        assert stats["total_schemas"] == 5

    def test_sunset_warning_approaching(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """check_sunset_warnings detects versions approaching sunset."""
        v = engine.create_version("sch-001", simple_definition)
        tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
        engine.deprecate_version(v["id"], sunset_date=tomorrow)
        warnings = engine.check_sunset_warnings(warning_days=30)
        assert len(warnings) >= 1
        assert warnings[0]["version_id"] == v["id"]
        assert warnings[0]["days_until_sunset"] <= 30
        assert warnings[0]["is_overdue"] is False

    def test_sunset_warning_overdue(
        self, engine: SchemaVersionerEngine, simple_definition: Dict[str, Any]
    ) -> None:
        """check_sunset_warnings detects overdue sunsets."""
        v = engine.create_version("sch-001", simple_definition)
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        engine.deprecate_version(v["id"], sunset_date=yesterday)
        warnings = engine.check_sunset_warnings(warning_days=30)
        assert len(warnings) >= 1
        assert warnings[0]["is_overdue"] is True
        assert warnings[0]["days_until_sunset"] < 0

    def test_sunset_warning_negative_days_raises(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """Negative warning_days raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            engine.check_sunset_warnings(warning_days=-1)

    def test_parse_version_valid(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """parse_version correctly parses valid SemVer strings."""
        assert engine.parse_version("0.0.0") == (0, 0, 0)
        assert engine.parse_version("1.2.3") == (1, 2, 3)
        assert engine.parse_version("10.20.30") == (10, 20, 30)


# ============================================================================
# TestHelperFunctions
# ============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions exposed for completeness."""

    def test_parse_version_invalid_parts(self) -> None:
        """_parse_version rejects strings with wrong number of parts."""
        with pytest.raises(ValueError, match="expected exactly 3"):
            _parse_version("1.2")
        with pytest.raises(ValueError, match="expected exactly 3"):
            _parse_version("1.2.3.4")

    def test_parse_version_non_integer(self) -> None:
        """_parse_version rejects non-integer components."""
        with pytest.raises(ValueError, match="must be integers"):
            _parse_version("a.b.c")

    def test_parse_version_negative(self) -> None:
        """_parse_version rejects negative components."""
        with pytest.raises(ValueError, match="non-negative"):
            _parse_version("-1.0.0")

    def test_bump_version_major(self) -> None:
        """_bump_version major resets minor and patch."""
        assert _bump_version("1.2.3", "major") == "2.0.0"

    def test_bump_version_minor(self) -> None:
        """_bump_version minor resets patch."""
        assert _bump_version("1.2.3", "minor") == "1.3.0"

    def test_bump_version_patch(self) -> None:
        """_bump_version increments only patch."""
        assert _bump_version("1.2.3", "patch") == "1.2.4"

    def test_bump_version_invalid_type(self) -> None:
        """_bump_version rejects invalid bump types."""
        with pytest.raises(ValueError, match="Unknown bump_type"):
            _bump_version("1.0.0", "mega")

    def test_generate_id_prefix(self) -> None:
        """_generate_id produces correctly prefixed IDs."""
        assert _generate_id("TEST").startswith("TEST-")

    def test_generate_id_unique(self) -> None:
        """_generate_id produces unique IDs."""
        ids = {_generate_id("VER") for _ in range(100)}
        assert len(ids) == 100

    def test_serialize_definition_deterministic(self) -> None:
        """_serialize_definition is deterministic (key-sorted, compact)."""
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert _serialize_definition(d1) == _serialize_definition(d2)

    def test_compute_provenance_is_sha256(self) -> None:
        """_compute_provenance returns a 64-char hex SHA-256."""
        h = _compute_provenance("test_op", "payload")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_version_tuple_key_valid(self) -> None:
        """_version_tuple_key converts valid SemVer to tuple."""
        assert _version_tuple_key("3.14.1") == (3, 14, 1)

    def test_version_tuple_key_invalid_fallback(self) -> None:
        """_version_tuple_key returns (0,0,0) for invalid strings."""
        assert _version_tuple_key("not-semver") == (0, 0, 0)

    def test_classify_bump_module_function(self) -> None:
        """Module-level _classify_bump matches public method."""
        assert _classify_bump([{"severity": "breaking"}]) == BUMP_MAJOR
        assert _classify_bump([{"severity": "non_breaking"}]) == BUMP_MINOR
        assert _classify_bump([]) == BUMP_PATCH


# ============================================================================
# TestStatistics
# ============================================================================


class TestStatistics:
    """Tests for get_statistics aggregation correctness."""

    def test_statistics_bump_type_counts(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
        breaking_changes: List[Dict[str, Any]],
        non_breaking_changes: List[Dict[str, Any]],
        cosmetic_changes: List[Dict[str, Any]],
    ) -> None:
        """Bump type counters reflect actual bump distribution."""
        engine.create_version("sch-001", simple_definition)  # patch (first)
        engine.create_version(
            "sch-001", simple_definition, changes=non_breaking_changes
        )  # minor
        engine.create_version(
            "sch-001", simple_definition, changes=breaking_changes
        )  # major
        engine.create_version(
            "sch-001", simple_definition, changes=cosmetic_changes
        )  # patch
        stats = engine.get_statistics()
        assert stats["by_bump_type"]["major"] == 1
        assert stats["by_bump_type"]["minor"] == 1
        assert stats["by_bump_type"]["patch"] == 2

    def test_statistics_active_vs_deprecated(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """active_versions = total_versions - deprecated_count."""
        v1 = engine.create_version("sch-001", simple_definition)
        engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v1["id"])
        stats = engine.get_statistics()
        assert stats["total_versions"] == 2
        assert stats["deprecated_count"] == 1
        assert stats["active_versions"] == 1

    def test_statistics_collected_at_is_iso8601(
        self, engine: SchemaVersionerEngine
    ) -> None:
        """collected_at is a valid ISO-8601 timestamp."""
        stats = engine.get_statistics()
        dt = datetime.fromisoformat(stats["collected_at"])
        assert dt.tzinfo is not None

    def test_statistics_after_reset(
        self, engine_with_versions: SchemaVersionerEngine
    ) -> None:
        """Statistics are zeroed after reset."""
        engine_with_versions.reset()
        stats = engine_with_versions.get_statistics()
        assert stats["total_versions"] == 0
        assert stats["active_versions"] == 0
        assert stats["deprecated_count"] == 0
        assert stats["total_schemas"] == 0

    def test_statistics_by_schema_breakdown(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """by_schema dict accurately counts versions per schema."""
        engine.create_version("sch-a", simple_definition)
        engine.create_version("sch-a", simple_definition)
        engine.create_version("sch-b", simple_definition)
        stats = engine.get_statistics()
        assert stats["by_schema"]["sch-a"] == 2
        assert stats["by_schema"]["sch-b"] == 1

    def test_statistics_undeprecate_adjusts_counts(
        self,
        engine: SchemaVersionerEngine,
        simple_definition: Dict[str, Any],
    ) -> None:
        """Undeprecating a version decrements the deprecated counter."""
        v = engine.create_version("sch-001", simple_definition)
        engine.deprecate_version(v["id"])
        assert engine.get_statistics()["deprecated_count"] == 1
        engine.undeprecate_version(v["id"])
        assert engine.get_statistics()["deprecated_count"] == 0


# ============================================================================
# TestBumpVersionPublicProxy
# ============================================================================


class TestBumpVersionPublicProxy:
    """Tests for the public bump_version proxy method."""

    def test_bump_major(self, engine: SchemaVersionerEngine) -> None:
        assert engine.bump_version("1.2.3", "major") == "2.0.0"

    def test_bump_minor(self, engine: SchemaVersionerEngine) -> None:
        assert engine.bump_version("1.2.3", "minor") == "1.3.0"

    def test_bump_patch(self, engine: SchemaVersionerEngine) -> None:
        assert engine.bump_version("1.2.3", "patch") == "1.2.4"

    def test_bump_from_zero(self, engine: SchemaVersionerEngine) -> None:
        assert engine.bump_version("0.0.0", "major") == "1.0.0"
        assert engine.bump_version("0.0.0", "minor") == "0.1.0"
        assert engine.bump_version("0.0.0", "patch") == "0.0.1"

    def test_bump_invalid_raises(self, engine: SchemaVersionerEngine) -> None:
        with pytest.raises(ValueError):
            engine.bump_version("1.0.0", "invalid")
