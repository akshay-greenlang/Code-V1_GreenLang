"""
GL-001 ThermalCommand: Schema Registry Tests

Comprehensive tests for schema version management, compatibility checking,
and migration support.

Test Coverage:
- Semantic version parsing and comparison
- Schema registration and lookup
- Compatibility checking (backward/forward/full)
- Schema migration
- Change detection
"""

import pytest
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/tests/', 1)[0])

from data_contracts.schema_registry import (
    # Enums
    CompatibilityMode,
    SchemaStatus,
    ChangeType,
    # Models
    SemanticVersion,
    SchemaChange,
    SchemaDefinition,
    SchemaMigration,
    # Classes
    CompatibilityChecker,
    SchemaMigrator,
    SchemaRegistry,
    # Functions
    get_schema_registry,
)


# =============================================================================
# SemanticVersion Tests
# =============================================================================

class TestSemanticVersion:
    """Tests for SemanticVersion class."""

    def test_parse_valid_version(self):
        """Test parsing valid version strings."""
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_parse_zero_version(self):
        """Test parsing zero version."""
        version = SemanticVersion.parse("0.0.0")
        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0

    def test_parse_large_version(self):
        """Test parsing large version numbers."""
        version = SemanticVersion.parse("100.200.300")
        assert version.major == 100
        assert version.minor == 200
        assert version.patch == 300

    def test_parse_invalid_format(self):
        """Test parsing invalid version format."""
        with pytest.raises(ValueError, match="Invalid version"):
            SemanticVersion.parse("1.2")

        with pytest.raises(ValueError, match="Invalid version"):
            SemanticVersion.parse("v1.2.3")

        with pytest.raises(ValueError, match="Invalid version"):
            SemanticVersion.parse("1.2.3.4")

    def test_str_representation(self):
        """Test string representation."""
        version = SemanticVersion(1, 2, 3)
        assert str(version) == "1.2.3"

    def test_comparison_operators(self):
        """Test comparison operators."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        v3 = SemanticVersion(2, 0, 0)
        v4 = SemanticVersion(1, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 <= v4
        assert v3 > v2
        assert v2 >= v1
        assert v1 == v4
        assert v1 != v2

    def test_bump_major(self):
        """Test major version bump."""
        version = SemanticVersion(1, 2, 3)
        bumped = version.bump_major()
        assert bumped == SemanticVersion(2, 0, 0)

    def test_bump_minor(self):
        """Test minor version bump."""
        version = SemanticVersion(1, 2, 3)
        bumped = version.bump_minor()
        assert bumped == SemanticVersion(1, 3, 0)

    def test_bump_patch(self):
        """Test patch version bump."""
        version = SemanticVersion(1, 2, 3)
        bumped = version.bump_patch()
        assert bumped == SemanticVersion(1, 2, 4)

    def test_hash(self):
        """Test hashing for dict keys."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 0)
        d = {v1: "value"}
        assert d[v2] == "value"

    def test_compatibility_backward(self):
        """Test backward compatibility check."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)

        # Newer can read older
        assert v2.is_compatible_with(v1, CompatibilityMode.BACKWARD)
        # Older cannot read newer in backward mode
        assert not v1.is_compatible_with(v2, CompatibilityMode.BACKWARD)

    def test_compatibility_forward(self):
        """Test forward compatibility check."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)

        # Older can read newer in forward mode
        assert v1.is_compatible_with(v2, CompatibilityMode.FORWARD)
        # Newer cannot read older in forward mode
        assert not v2.is_compatible_with(v1, CompatibilityMode.FORWARD)

    def test_compatibility_major_break(self):
        """Test major version break compatibility."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)

        # Different major = incompatible
        assert not v2.is_compatible_with(v1, CompatibilityMode.BACKWARD)
        assert not v1.is_compatible_with(v2, CompatibilityMode.FORWARD)


# =============================================================================
# SchemaChange Tests
# =============================================================================

class TestSchemaChange:
    """Tests for SchemaChange class."""

    def test_create_change(self):
        """Test creating schema change."""
        change = SchemaChange(
            change_type=ChangeType.FIELD_ADDED,
            field_path="new_field",
            new_value={"type": "string"},
            is_breaking=False,
            description="Added new optional field",
        )
        assert change.change_type == ChangeType.FIELD_ADDED
        assert change.is_breaking is False

    def test_to_dict(self):
        """Test converting change to dict."""
        change = SchemaChange(
            change_type=ChangeType.FIELD_REMOVED,
            field_path="old_field",
            old_value={"type": "integer"},
            is_breaking=True,
            description="Removed deprecated field",
        )
        d = change.to_dict()
        assert d["change_type"] == "field_removed"
        assert d["is_breaking"] is True


# =============================================================================
# SchemaDefinition Tests
# =============================================================================

class TestSchemaDefinition:
    """Tests for SchemaDefinition class."""

    def test_create_definition(self):
        """Test creating schema definition."""
        schema = SchemaDefinition(
            name="TestSchema",
            version=SemanticVersion(1, 0, 0),
            schema_type="pydantic",
            schema_content={"properties": {"field1": {"type": "string"}}},
            description="Test schema",
        )
        assert schema.name == "TestSchema"
        assert schema.schema_id == "TestSchema:1.0.0"

    def test_fingerprint(self):
        """Test schema fingerprint computation."""
        schema1 = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={"properties": {"a": {"type": "string"}}},
        )
        schema2 = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={"properties": {"a": {"type": "string"}}},
        )
        schema3 = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={"properties": {"b": {"type": "integer"}}},
        )

        # Same content = same fingerprint
        assert schema1.fingerprint == schema2.fingerprint
        # Different content = different fingerprint
        assert schema1.fingerprint != schema3.fingerprint

    def test_to_dict(self):
        """Test converting to dict."""
        schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={"properties": {}},
            tags=["test", "example"],
        )
        d = schema.to_dict()
        assert d["name"] == "Test"
        assert d["version"] == "1.0.0"
        assert "test" in d["tags"]


# =============================================================================
# CompatibilityChecker Tests
# =============================================================================

class TestCompatibilityChecker:
    """Tests for CompatibilityChecker class."""

    @pytest.fixture
    def checker(self) -> CompatibilityChecker:
        """Create compatibility checker."""
        return CompatibilityChecker(CompatibilityMode.BACKWARD)

    def test_detect_field_added(self, checker):
        """Test detecting added field."""
        old_schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={
                "properties": {"field1": {"type": "string"}},
                "required": ["field1"],
            },
        )
        new_schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 1, 0),
            schema_type="json_schema",
            schema_content={
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "integer", "default": 0},
                },
                "required": ["field1"],
            },
        )

        is_compatible, changes = checker.check_compatibility(old_schema, new_schema)
        assert is_compatible is True  # Optional field addition is backward compatible
        assert any(c.change_type == ChangeType.FIELD_ADDED for c in changes)

    def test_detect_field_removed(self, checker):
        """Test detecting removed field."""
        old_schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={
                "properties": {
                    "field1": {"type": "string"},
                    "field2": {"type": "integer"},
                },
            },
        )
        new_schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(2, 0, 0),
            schema_type="json_schema",
            schema_content={
                "properties": {"field1": {"type": "string"}},
            },
        )

        is_compatible, changes = checker.check_compatibility(old_schema, new_schema)
        assert is_compatible is False  # Field removal breaks backward compatibility
        assert any(c.change_type == ChangeType.FIELD_REMOVED for c in changes)

    def test_detect_type_change(self, checker):
        """Test detecting type change."""
        old_schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={
                "properties": {"field1": {"type": "string"}},
            },
        )
        new_schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(2, 0, 0),
            schema_type="json_schema",
            schema_content={
                "properties": {"field1": {"type": "integer"}},
            },
        )

        is_compatible, changes = checker.check_compatibility(old_schema, new_schema)
        assert is_compatible is False
        assert any(c.change_type == ChangeType.FIELD_TYPE_CHANGED for c in changes)

    def test_no_changes(self, checker):
        """Test no changes detection."""
        schema = SchemaDefinition(
            name="Test",
            version=SemanticVersion(1, 0, 0),
            schema_type="json_schema",
            schema_content={"properties": {"field1": {"type": "string"}}},
        )

        is_compatible, changes = checker.check_compatibility(schema, schema)
        assert is_compatible is True
        assert len(changes) == 0


# =============================================================================
# SchemaMigrator Tests
# =============================================================================

class TestSchemaMigrator:
    """Tests for SchemaMigrator class."""

    @pytest.fixture
    def migrator(self) -> SchemaMigrator:
        """Create schema migrator."""
        return SchemaMigrator()

    def test_register_migration(self, migrator):
        """Test registering a migration."""
        migration = SchemaMigration(
            name="Test",
            from_version=SemanticVersion(1, 0, 0),
            to_version=SemanticVersion(1, 1, 0),
            changes=[
                SchemaChange(
                    change_type=ChangeType.FIELD_ADDED,
                    field_path="new_field",
                    new_value={"type": "string", "default": ""},
                )
            ],
        )
        migrator.register_migration(migration)
        assert migration.migration_id in migrator._migrations

    def test_register_transform(self, migrator):
        """Test registering transform function."""
        def transform(data):
            data["new_field"] = data.get("old_field", "").upper()
            return data

        migrator.register_transform("1.0.0", "1.1.0", transform)
        assert "1.0.0->1.1.0" in migrator._transform_functions

    def test_migrate_data_with_transform(self, migrator):
        """Test migrating data with transform function."""
        def transform(data):
            data["version"] = "2"
            return data

        migrator.register_transform("1.0.0", "1.1.0", transform)

        migration = SchemaMigration(
            name="Test",
            from_version=SemanticVersion(1, 0, 0),
            to_version=SemanticVersion(1, 1, 0),
            changes=[],
        )
        migrator.register_migration(migration)

        data = {"field1": "value", "version": "1"}
        result = migrator.migrate_data(
            data,
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 1, 0),
            "Test",
        )
        assert result["version"] == "2"

    def test_migrate_same_version(self, migrator):
        """Test migrating to same version returns same data."""
        data = {"field1": "value"}
        result = migrator.migrate_data(
            data,
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 0, 0),
            "Test",
        )
        assert result == data


# =============================================================================
# SchemaRegistry Tests
# =============================================================================

class TestSchemaRegistry:
    """Tests for SchemaRegistry class."""

    @pytest.fixture
    def registry(self) -> SchemaRegistry:
        """Create fresh schema registry."""
        return SchemaRegistry(compatibility_mode=CompatibilityMode.NONE)

    def test_register_schema(self, registry):
        """Test registering a schema."""
        schema = registry.register_schema(
            name="CustomSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={"properties": {"field": {"type": "string"}}},
            description="Test schema",
        )
        assert schema.name == "CustomSchema"
        assert str(schema.version) == "1.0.0"

    def test_get_schema(self, registry):
        """Test getting schema by name."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={"properties": {}},
        )
        schema = registry.get_schema("TestSchema")
        assert schema is not None
        assert schema.name == "TestSchema"

    def test_get_schema_specific_version(self, registry):
        """Test getting specific schema version."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={"version": "1"},
            check_compatibility=False,
        )
        registry.register_schema(
            name="TestSchema",
            version="1.1.0",
            schema_type="json_schema",
            schema_content={"version": "1.1"},
            check_compatibility=False,
        )

        schema_v1 = registry.get_schema("TestSchema", "1.0.0")
        schema_v11 = registry.get_schema("TestSchema", "1.1.0")

        assert schema_v1.schema_content["version"] == "1"
        assert schema_v11.schema_content["version"] == "1.1"

    def test_get_latest_version(self, registry):
        """Test getting latest version."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={},
            check_compatibility=False,
        )
        registry.register_schema(
            name="TestSchema",
            version="1.2.0",
            schema_type="json_schema",
            schema_content={},
            check_compatibility=False,
        )

        latest = registry.get_latest_version("TestSchema")
        assert latest == "1.2.0"

    def test_get_all_versions(self, registry):
        """Test getting all versions."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={},
            check_compatibility=False,
        )
        registry.register_schema(
            name="TestSchema",
            version="1.1.0",
            schema_type="json_schema",
            schema_content={},
            check_compatibility=False,
        )

        versions = registry.get_all_versions("TestSchema")
        assert len(versions) == 2

    def test_deprecate_schema(self, registry):
        """Test deprecating a schema."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={},
        )

        success = registry.deprecate_schema(
            "TestSchema",
            "1.0.0",
            "Replaced by v2.0.0"
        )
        assert success is True

        schema = registry.get_schema("TestSchema", "1.0.0")
        assert schema.status == SchemaStatus.DEPRECATED

    def test_compatibility_check(self, registry):
        """Test compatibility check."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={
                "properties": {"field1": {"type": "string"}},
            },
        )

        # Compatible change (add optional field)
        new_content = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "integer", "default": 0},
            },
        }
        is_compatible, changes = registry.check_compatibility(
            "TestSchema",
            new_content,
            "1.1.0"
        )
        # With mode=NONE, everything is compatible
        assert is_compatible is True

    def test_list_schemas(self, registry):
        """Test listing schemas."""
        registry.register_schema(
            name="Schema1",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={},
            tags=["test"],
        )
        registry.register_schema(
            name="Schema2",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={},
            tags=["production"],
        )

        all_schemas = registry.list_schemas()
        assert "Schema1" in all_schemas
        assert "Schema2" in all_schemas

        test_schemas = registry.list_schemas(tags=["test"])
        assert "Schema1" in test_schemas
        assert "Schema2" not in test_schemas

    def test_validate_data(self, registry):
        """Test data validation against schema."""
        registry.register_schema(
            name="TestSchema",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["name"],
            },
        )

        # Valid data
        is_valid, errors = registry.validate_data(
            "TestSchema",
            {"name": "test", "value": 42},
        )
        assert is_valid is True
        assert len(errors) == 0

        # Missing required field
        is_valid, errors = registry.validate_data(
            "TestSchema",
            {"value": 42},
        )
        assert is_valid is False
        assert any("name" in e for e in errors)

    def test_export_registry(self, registry):
        """Test exporting registry."""
        registry.register_schema(
            name="Schema1",
            version="1.0.0",
            schema_type="json_schema",
            schema_content={},
        )

        export = registry.export_registry()
        assert "schemas" in export
        assert "Schema1" in export["schemas"]
        assert "exported_at" in export

    def test_builtin_schemas_registered(self):
        """Test that builtin domain schemas are registered."""
        # Create registry with builtins
        registry = SchemaRegistry()

        # Check domain schemas are present
        domain_schemas = [
            "ProcessSensorData",
            "EnergyConsumptionData",
            "SafetySystemStatus",
            "ProductionSchedule",
            "WeatherForecast",
            "EnergyPrices",
            "EquipmentHealth",
            "AlarmState",
        ]

        for schema_name in domain_schemas:
            schema = registry.get_schema(schema_name)
            assert schema is not None, f"Missing builtin schema: {schema_name}"


# =============================================================================
# Singleton Tests
# =============================================================================

class TestGetSchemaRegistry:
    """Tests for get_schema_registry singleton."""

    def test_singleton_returns_same_instance(self):
        """Test singleton returns same instance."""
        registry1 = get_schema_registry()
        registry2 = get_schema_registry()
        assert registry1 is registry2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
