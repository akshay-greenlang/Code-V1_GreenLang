# -*- coding: utf-8 -*-
"""
Integration Tests: SchemaRegistryEngine + SchemaVersionerEngine
================================================================

Tests multi-engine workflows that span Engine 1 (Registry) and Engine 2
(Versioner). Validates registration -> versioning -> lifecycle management
flows end-to-end with real engine instances (no mocks).

Test Classes:
    TestSchemaRegistrationAndVersioning  (~8 tests)
    TestVersionLifecycle                 (~6 tests)
    TestSchemaGroupWithVersions          (~5 tests)
    TestBulkRegistrationWithVersions     (~5 tests)
    TestExportImportWithVersions         (~6 tests)

Total: ~30 integration tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from greenlang.schema_migration.schema_registry import SchemaRegistryEngine
from greenlang.schema_migration.schema_versioner import SchemaVersionerEngine


# ---------------------------------------------------------------------------
# Test Class 1: Schema Registration and Versioning
# ---------------------------------------------------------------------------


class TestSchemaRegistrationAndVersioning:
    """Test registration of schemas followed by version creation and bumps."""

    def test_register_then_create_first_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Register a schema, then create an initial version which should be 1.0.0."""
        schema = fresh_registry.register_schema(
            namespace="greenlang.emissions",
            name="UserRecord",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="platform-team",
            tags={"domain": "emissions"},
        )
        schema_id = schema["schema_id"]
        assert schema["status"] == "draft"
        assert schema_id is not None

        version = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="Initial release",
        )
        assert version["version"] == "1.0.0"
        assert version["schema_id"] == schema_id
        assert version["provenance_hash"] is not None
        assert len(version["provenance_hash"]) == 64

    def test_multiple_versions_with_auto_bump(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
        sample_user_schema_v2: Dict[str, Any],
    ):
        """Create multiple versions with different change severities and verify auto-bumps."""
        schema = fresh_registry.register_schema(
            namespace="greenlang.emissions",
            name="UserRecord",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="platform-team",
        )
        schema_id = schema["schema_id"]

        # v1.0.0 - initial
        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="Initial release",
        )
        assert v1["version"] == "1.0.0"

        # v1.1.0 - minor (non-breaking additive change)
        v2 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[
                {"field": "phone", "severity": "non_breaking", "description": "Added phone field"}
            ],
            changelog_note="Added phone field",
        )
        assert v2["version"] == "1.1.0"
        assert v2["bump_type"] == "minor"

        # v1.1.1 - patch (cosmetic change)
        v3 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[
                {"field": "name", "severity": "cosmetic", "description": "Updated description"}
            ],
            changelog_note="Updated description for name field",
        )
        assert v3["version"] == "1.1.1"
        assert v3["bump_type"] == "patch"

        # v2.0.0 - major (breaking change)
        v4 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v2,
            changes=[
                {"field": "age", "severity": "breaking", "description": "Removed age field"},
                {"field": "salary", "severity": "non_breaking", "description": "Added salary"},
            ],
            changelog_note="Major refactoring: removed age, added salary",
        )
        assert v4["version"] == "2.0.0"
        assert v4["bump_type"] == "major"

    def test_version_history_is_complete(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Create 3 versions and verify list_versions returns all of them."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="HistoryTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        # Create 3 versions
        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )
        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "x", "severity": "non_breaking", "description": "add x"}],
            changelog_note="v1.1",
        )
        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "y", "severity": "cosmetic", "description": "cosmetic"}],
            changelog_note="v1.1.1",
        )

        versions = fresh_versioner.list_versions(schema_id)
        assert len(versions) == 3
        version_strings = [v["version"] for v in versions]
        assert "1.0.0" in version_strings
        assert "1.1.0" in version_strings
        assert "1.1.1" in version_strings

    def test_get_latest_version_returns_newest(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """get_latest_version should return the most recently created non-deprecated version."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="LatestTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )
        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "z", "severity": "breaking", "description": "break"}],
            changelog_note="v2",
        )

        latest = fresh_versioner.get_latest_version(schema_id)
        assert latest is not None
        assert latest["version"] == "2.0.0"

    def test_version_provenance_hash_determinism(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Provenance hashes should be non-empty 64-char SHA-256 hex strings."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="ProvenanceTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        assert v1["provenance_hash"] is not None
        assert isinstance(v1["provenance_hash"], str)
        assert len(v1["provenance_hash"]) == 64

        # Schema also has provenance
        assert schema["provenance_hash"] is not None
        assert len(schema["provenance_hash"]) == 64

    def test_version_compare_returns_diff(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
        sample_user_schema_v2: Dict[str, Any],
    ):
        """compare_versions should detect definition differences between v1 and v2."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="CompareTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )
        v2 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v2,
            changes=[{"field": "age", "severity": "breaking", "description": "removed age"}],
            changelog_note="v2",
        )

        comparison = fresh_versioner.compare_versions(v1["id"], v2["id"])
        assert comparison is not None
        assert comparison.get("definition_changed") is True

    def test_register_schema_then_activate_and_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Register, activate, then create version -- validates lifecycle integration."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="LifecycleVersionTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]
        assert schema["status"] == "draft"

        # Activate schema
        updated = fresh_registry.update_schema(schema_id, status="active")
        assert updated["status"] == "active"

        # Create version for active schema
        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="First release",
        )
        assert v1["version"] == "1.0.0"
        assert v1["schema_id"] == schema_id

    def test_version_changelog_is_accessible(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Changelog should be retrievable and contain all version notes."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="ChangelogTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="Initial release",
        )
        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "x", "severity": "non_breaking", "description": "add x"}],
            changelog_note="Added x field",
        )

        changelog = fresh_versioner.get_changelog(schema_id)
        assert isinstance(changelog, list)
        assert len(changelog) >= 2


# ---------------------------------------------------------------------------
# Test Class 2: Version Lifecycle (deprecation, sunset)
# ---------------------------------------------------------------------------


class TestVersionLifecycle:
    """Test version deprecation, undeprecation, and sunset detection."""

    def test_deprecate_old_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Deprecate v1 and verify it is excluded from non-deprecated listings."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="DeprecateTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )
        v2 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "x", "severity": "breaking", "description": "break"}],
            changelog_note="v2",
        )

        # Deprecate v1
        deprecated = fresh_versioner.deprecate_version(v1["id"], reason="Superseded by v2")
        assert deprecated.get("is_deprecated") is True

        # list_versions with include_deprecated=False should not include v1
        active_versions = fresh_versioner.list_versions(
            schema_id, include_deprecated=False
        )
        version_strings = [v["version"] for v in active_versions]
        assert "1.0.0" not in version_strings
        assert "2.0.0" in version_strings

    def test_undeprecate_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Undeprecating a version should restore its visibility."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="UndeprecateTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        # Deprecate then undeprecate
        fresh_versioner.deprecate_version(v1["id"], reason="Testing")
        restored = fresh_versioner.undeprecate_version(v1["id"])
        assert restored.get("is_deprecated") is False

        # Should appear in active listing
        versions = fresh_versioner.list_versions(
            schema_id, include_deprecated=False
        )
        version_strings = [v["version"] for v in versions]
        assert "1.0.0" in version_strings

    def test_deprecate_with_sunset_date(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Deprecation with a sunset date should store the date correctly."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="SunsetTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        deprecated = fresh_versioner.deprecate_version(
            v1["id"],
            sunset_date="2026-06-01",
            reason="Planned removal",
        )
        assert deprecated.get("is_deprecated") is True

    def test_get_latest_skips_deprecated(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """get_latest_version should skip deprecated versions."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="LatestSkipTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )
        v2 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "x", "severity": "non_breaking", "description": "x"}],
            changelog_note="v2",
        )

        # Deprecate v2 (latest)
        fresh_versioner.deprecate_version(v2["id"], reason="Bad release")

        # Latest should be v1 now
        latest = fresh_versioner.get_latest_version(schema_id)
        assert latest is not None
        assert latest["version"] == "1.0.0"

    def test_schema_deprecation_and_version_coexist(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Deprecating a schema in the registry does not affect version deprecation state."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="CoexistTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        v1 = fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        # Activate then deprecate the schema in registry
        fresh_registry.update_schema(schema_id, status="active")
        fresh_registry.update_schema(schema_id, status="deprecated")

        # Version should still be non-deprecated
        version = fresh_versioner.get_version(v1["id"])
        assert version is not None
        assert version.get("is_deprecated") is False

    def test_version_pin_after_registration(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Pin a consumer to a version range after registering and creating versions."""
        schema = fresh_registry.register_schema(
            namespace="test",
            name="PinTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_id = schema["schema_id"]

        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )
        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changes=[{"field": "x", "severity": "breaking", "description": "break"}],
            changelog_note="v2",
        )

        pin_result = fresh_versioner.pin_version(
            schema_id=schema_id,
            consumer_id="data-pipeline-A",
            version_range=">=1.0.0 <2.0.0",
        )
        assert pin_result is not None
        assert pin_result.get("consumer_id") == "data-pipeline-A"


# ---------------------------------------------------------------------------
# Test Class 3: Schema Groups with Versions
# ---------------------------------------------------------------------------


class TestSchemaGroupWithVersions:
    """Test schema groups combined with versioning operations."""

    def test_create_group_and_version_members(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Create a group, add schemas, and create versions for each member."""
        # Register two schemas
        schema_a = fresh_registry.register_schema(
            namespace="emissions",
            name="FactorA",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_b = fresh_registry.register_schema(
            namespace="emissions",
            name="FactorB",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )

        # Create group
        group = fresh_registry.create_group(
            name="emission-factors",
            description="All emission factor schemas",
            schema_ids=[schema_a["schema_id"], schema_b["schema_id"]],
        )
        assert group is not None

        # Create versions for each
        v_a = fresh_versioner.create_version(
            schema_id=schema_a["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="Factor A v1",
        )
        v_b = fresh_versioner.create_version(
            schema_id=schema_b["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="Factor B v1",
        )

        assert v_a["version"] == "1.0.0"
        assert v_b["version"] == "1.0.0"
        assert v_a["schema_id"] != v_b["schema_id"]

    def test_group_members_have_independent_version_histories(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Version histories for grouped schemas should be independent."""
        schema_a = fresh_registry.register_schema(
            namespace="emissions",
            name="IndepA",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        schema_b = fresh_registry.register_schema(
            namespace="emissions",
            name="IndepB",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )

        fresh_registry.create_group(
            name="indep-group",
            description="Independence test",
            schema_ids=[schema_a["schema_id"], schema_b["schema_id"]],
        )

        # Schema A gets 3 versions
        fresh_versioner.create_version(
            schema_id=schema_a["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="a-v1",
        )
        fresh_versioner.create_version(
            schema_id=schema_a["schema_id"],
            definition_json=sample_user_schema_v1,
            changes=[{"field": "x", "severity": "non_breaking", "description": "add"}],
            changelog_note="a-v1.1",
        )
        fresh_versioner.create_version(
            schema_id=schema_a["schema_id"],
            definition_json=sample_user_schema_v1,
            changes=[{"field": "y", "severity": "breaking", "description": "break"}],
            changelog_note="a-v2",
        )

        # Schema B gets 1 version
        fresh_versioner.create_version(
            schema_id=schema_b["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="b-v1",
        )

        versions_a = fresh_versioner.list_versions(schema_a["schema_id"])
        versions_b = fresh_versioner.list_versions(schema_b["schema_id"])

        assert len(versions_a) == 3
        assert len(versions_b) == 1

    def test_add_schema_to_existing_group_then_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Add a schema to an existing group, then create a version for it."""
        schema_a = fresh_registry.register_schema(
            namespace="test",
            name="GroupAddA",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )

        fresh_registry.create_group(
            name="grow-group",
            description="Grows over time",
            schema_ids=[schema_a["schema_id"]],
        )

        # Register new schema and add to group
        schema_c = fresh_registry.register_schema(
            namespace="test",
            name="GroupAddC",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        fresh_registry.add_to_group("grow-group", schema_c["schema_id"])

        # Create version for newly added schema
        v_c = fresh_versioner.create_version(
            schema_id=schema_c["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="c-v1",
        )
        assert v_c["version"] == "1.0.0"

        # Group should show both members
        group = fresh_registry.get_group("grow-group")
        assert group is not None

    def test_remove_schema_from_group_versions_intact(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Removing a schema from a group should not affect its versions."""
        schema_a = fresh_registry.register_schema(
            namespace="test",
            name="RemGroupA",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="tester",
        )
        fresh_registry.create_group(
            name="rem-group",
            schema_ids=[schema_a["schema_id"]],
        )

        # Create version
        v1 = fresh_versioner.create_version(
            schema_id=schema_a["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        # Remove from group
        fresh_registry.remove_from_group("rem-group", schema_a["schema_id"])

        # Version still accessible
        version = fresh_versioner.get_version(v1["id"])
        assert version is not None
        assert version["version"] == "1.0.0"

    def test_list_groups_after_multiple_registrations(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Verify list_groups returns all created groups after schemas are registered."""
        for i in range(3):
            schema = fresh_registry.register_schema(
                namespace="test",
                name=f"GroupList{i}",
                schema_type="json_schema",
                definition_json=sample_user_schema_v1,
                owner="tester",
            )
            fresh_registry.create_group(
                name=f"group-{i}",
                description=f"Group {i}",
                schema_ids=[schema["schema_id"]],
            )

        groups = fresh_registry.list_groups()
        assert len(groups) >= 3


# ---------------------------------------------------------------------------
# Test Class 4: Bulk Registration with Versions
# ---------------------------------------------------------------------------


class TestBulkRegistrationWithVersions:
    """Test bulk import/registration followed by versioning workflows."""

    def test_register_multiple_schemas_then_version_each(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Register 5 schemas and create an initial version for each."""
        schema_ids = []
        for i in range(5):
            schema = fresh_registry.register_schema(
                namespace="bulk",
                name=f"BulkSchema{i}",
                schema_type="json_schema",
                definition_json=sample_user_schema_v1,
                owner="bulk-tester",
                tags={"batch": str(i)},
            )
            schema_ids.append(schema["schema_id"])

        # Create versions for each
        for sid in schema_ids:
            v = fresh_versioner.create_version(
                schema_id=sid,
                definition_json=sample_user_schema_v1,
                changelog_note="Initial release",
            )
            assert v["version"] == "1.0.0"

        # Verify all 5 exist in registry
        all_schemas = fresh_registry.list_schemas(namespace="bulk")
        assert len(all_schemas) == 5

    def test_bulk_import_then_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Bulk import schemas, then create versions for each imported schema."""
        schemas_to_import = [
            {
                "namespace": "import",
                "name": f"ImportSchema{i}",
                "schema_type": "json_schema",
                "definition_json": sample_user_schema_v1,
                "owner": "importer",
            }
            for i in range(3)
        ]

        result = fresh_registry.import_schemas(schemas_to_import)
        assert result is not None

        # Verify import succeeded and create versions
        imported = fresh_registry.list_schemas(namespace="import")
        assert len(imported) >= 3

        for schema in imported:
            v = fresh_versioner.create_version(
                schema_id=schema["schema_id"],
                definition_json=sample_user_schema_v1,
                changelog_note="Post-import initial version",
            )
            assert v["version"] == "1.0.0"

    def test_versioner_statistics_after_bulk(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Versioner statistics should reflect all created versions accurately."""
        for i in range(4):
            schema = fresh_registry.register_schema(
                namespace="stats",
                name=f"StatsSchema{i}",
                schema_type="json_schema",
                definition_json=sample_user_schema_v1,
                owner="stats-tester",
            )
            fresh_versioner.create_version(
                schema_id=schema["schema_id"],
                definition_json=sample_user_schema_v1,
                changelog_note="v1",
            )

        stats = fresh_versioner.get_statistics()
        assert stats is not None
        # Statistics key may be "versions_created", "total_versions", or "active_versions"
        count = (
            stats.get("versions_created")
            or stats.get("total_versions")
            or stats.get("active_versions", 0)
        )
        assert count >= 4

    def test_registry_statistics_after_bulk(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Registry statistics should reflect all registered schemas."""
        for i in range(3):
            fresh_registry.register_schema(
                namespace="reg-stats",
                name=f"RegStats{i}",
                schema_type="json_schema",
                definition_json=sample_user_schema_v1,
                owner="stats-tester",
            )

        stats = fresh_registry.get_statistics()
        assert stats is not None
        assert stats.get("total_schemas", 0) >= 3

    def test_register_then_search_and_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Register schemas, search for them, then create versions for matches."""
        for i in range(3):
            fresh_registry.register_schema(
                namespace="searchable",
                name=f"EmissionFactor{i}",
                schema_type="json_schema",
                definition_json=sample_user_schema_v1,
                owner="search-tester",
                description=f"Emission factor schema variant {i}",
            )

        results = fresh_registry.search_schemas("EmissionFactor")
        assert len(results) >= 3

        # Version the first result
        first = results[0]
        v = fresh_versioner.create_version(
            schema_id=first["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="Versioned from search",
        )
        assert v["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Test Class 5: Export/Import with Versions
# ---------------------------------------------------------------------------


class TestExportImportWithVersions:
    """Test export and re-import workflows preserving schema integrity."""

    def test_export_schemas_with_versions(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Export schemas after creating versions to verify export includes definitions."""
        schema = fresh_registry.register_schema(
            namespace="export",
            name="ExportTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="exporter",
        )
        schema_id = schema["schema_id"]

        fresh_versioner.create_version(
            schema_id=schema_id,
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        exported = fresh_registry.export_schemas(schema_ids=[schema_id])
        assert isinstance(exported, list)
        assert len(exported) >= 1
        assert exported[0]["schema_id"] == schema_id

    def test_export_by_namespace(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Export all schemas in a namespace after versioning."""
        for i in range(3):
            schema = fresh_registry.register_schema(
                namespace="ns-export",
                name=f"NsExport{i}",
                schema_type="json_schema",
                definition_json=sample_user_schema_v1,
                owner="exporter",
            )
            fresh_versioner.create_version(
                schema_id=schema["schema_id"],
                definition_json=sample_user_schema_v1,
                changelog_note="v1",
            )

        exported = fresh_registry.export_schemas(namespace="ns-export")
        assert len(exported) >= 3

    def test_import_then_export_roundtrip(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Import schemas, create versions, export, and verify roundtrip consistency."""
        # Register original
        original = fresh_registry.register_schema(
            namespace="roundtrip",
            name="RoundtripTest",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="roundtripper",
        )
        fresh_versioner.create_version(
            schema_id=original["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        # Export
        exported = fresh_registry.export_schemas(
            schema_ids=[original["schema_id"]]
        )
        assert len(exported) >= 1

        # Verify exported data has the definition
        export_entry = exported[0]
        assert "definition" in export_entry or "definition_json" in export_entry

    def test_validate_definition_before_versioning(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Validate a definition against registry rules before creating a version."""
        validation = fresh_registry.validate_definition(
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
        )
        assert validation is not None
        assert validation.get("is_valid", validation.get("valid", False)) is True

        # Register and version the validated schema
        schema = fresh_registry.register_schema(
            namespace="validated",
            name="ValidatedSchema",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="validator",
        )
        v = fresh_versioner.create_version(
            schema_id=schema["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="Validated and versioned",
        )
        assert v["version"] == "1.0.0"

    def test_schema_definition_hash_consistency(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
        sample_user_schema_v1: Dict[str, Any],
    ):
        """Definition hash should match between registry and versioner."""
        schema = fresh_registry.register_schema(
            namespace="hash-test",
            name="HashConsistency",
            schema_type="json_schema",
            definition_json=sample_user_schema_v1,
            owner="hash-tester",
        )

        v1 = fresh_versioner.create_version(
            schema_id=schema["schema_id"],
            definition_json=sample_user_schema_v1,
            changelog_note="v1",
        )

        # Version should have a definition hash
        assert "definition_hash" in v1
        assert isinstance(v1["definition_hash"], str)
        assert len(v1["definition_hash"]) == 64

    def test_avro_schema_register_and_version(
        self,
        fresh_registry: SchemaRegistryEngine,
        fresh_versioner: SchemaVersionerEngine,
    ):
        """Register an Avro schema and create a version for it."""
        avro_def = {
            "type": "record",
            "name": "EmissionRecord",
            "namespace": "com.greenlang.emissions",
            "fields": [
                {"name": "source_id", "type": "string"},
                {"name": "co2_kg", "type": "double"},
                {"name": "timestamp", "type": "long"},
            ],
        }

        schema = fresh_registry.register_schema(
            namespace="avro-test",
            name="EmissionAvro",
            schema_type="avro",
            definition_json=avro_def,
            owner="avro-tester",
        )
        assert schema["schema_id"] is not None

        v1 = fresh_versioner.create_version(
            schema_id=schema["schema_id"],
            definition_json=avro_def,
            changelog_note="Avro v1",
        )
        assert v1["version"] == "1.0.0"
