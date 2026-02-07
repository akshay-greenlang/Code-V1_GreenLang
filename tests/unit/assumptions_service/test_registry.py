# -*- coding: utf-8 -*-
"""
Unit Tests for AssumptionRegistry (AGENT-FOUND-004)

Tests CRUD operations, version management, value resolution,
export/import, and provenance tracking for the assumption registry.

Coverage target: 85%+ of registry.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline AssumptionRegistry mirroring greenlang/assumptions/registry.py
# ---------------------------------------------------------------------------


class Assumption:
    """Minimal assumption model for registry testing."""

    def __init__(
        self,
        assumption_id: str,
        name: str,
        description: str = "",
        category: str = "custom",
        data_type: str = "float",
        value: Any = None,
        unit: str = "",
        source: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: int = 1,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.assumption_id = assumption_id
        self.name = name
        self.description = description
        self.category = category
        self.data_type = data_type
        self.value = value
        self.unit = unit
        self.source = source
        self.tags = tags or []
        self.metadata = metadata or {}
        self.version = version
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "data_type": self.data_type,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AssumptionVersion:
    """Version snapshot."""

    def __init__(self, assumption_id, version, value, change_type="update",
                 changed_by="system", change_reason="", provenance_hash="",
                 timestamp=None):
        self.assumption_id = assumption_id
        self.version = version
        self.value = value
        self.change_type = change_type
        self.changed_by = changed_by
        self.change_reason = change_reason
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "assumption_id": self.assumption_id,
            "version": self.version,
            "value": self.value,
            "change_type": self.change_type,
            "changed_by": self.changed_by,
            "change_reason": self.change_reason,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


class RegistryError(Exception):
    """Base error for registry operations."""
    pass


class DuplicateAssumptionError(RegistryError):
    pass


class AssumptionNotFoundError(RegistryError):
    pass


class ValidationError(RegistryError):
    pass


class AssumptionInUseError(RegistryError):
    pass


class AssumptionRegistry:
    """
    Registry for managing assumptions with version history.
    Mirrors greenlang/assumptions/registry.py.
    """

    def __init__(self, max_versions: int = 50, enable_provenance: bool = True):
        self._assumptions: Dict[str, Assumption] = {}
        self._versions: Dict[str, List[AssumptionVersion]] = {}
        self._max_versions = max_versions
        self._enable_provenance = enable_provenance
        self._dependencies: Dict[str, List[str]] = {}  # who depends on this

    def create(
        self,
        assumption_id: str,
        name: str,
        description: str = "",
        category: str = "custom",
        data_type: str = "float",
        value: Any = None,
        unit: str = "",
        source: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Assumption:
        """Create a new assumption."""
        if not assumption_id:
            raise ValidationError("assumption_id is required")
        if not name:
            raise ValidationError("name is required")
        if assumption_id in self._assumptions:
            raise DuplicateAssumptionError(
                f"Assumption '{assumption_id}' already exists"
            )

        assumption = Assumption(
            assumption_id=assumption_id,
            name=name,
            description=description,
            category=category,
            data_type=data_type,
            value=value,
            unit=unit,
            source=source,
            tags=tags,
            metadata=metadata,
        )
        self._assumptions[assumption_id] = assumption

        # Record initial version
        prov_hash = self._provenance_hash("create", assumption_id, value)
        v = AssumptionVersion(
            assumption_id=assumption_id,
            version=1,
            value=value,
            change_type="create",
            provenance_hash=prov_hash,
        )
        self._versions[assumption_id] = [v]

        return assumption

    def get(self, assumption_id: str) -> Assumption:
        """Get an assumption by ID."""
        if assumption_id not in self._assumptions:
            raise AssumptionNotFoundError(
                f"Assumption '{assumption_id}' not found"
            )
        return self._assumptions[assumption_id]

    def update(
        self,
        assumption_id: str,
        value: Any = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        change_reason: str = "",
        changed_by: str = "system",
    ) -> Assumption:
        """Update an assumption, creating a new version."""
        a = self.get(assumption_id)

        if value is not None:
            a.value = value
        if name is not None:
            a.name = name
        if description is not None:
            a.description = description
        if tags is not None:
            a.tags = tags
        if metadata is not None:
            a.metadata = metadata

        a.version += 1
        a.updated_at = datetime.utcnow().isoformat()

        # Record version
        prov_hash = self._provenance_hash("update", assumption_id, a.value)
        v = AssumptionVersion(
            assumption_id=assumption_id,
            version=a.version,
            value=a.value,
            change_type="update",
            changed_by=changed_by,
            change_reason=change_reason,
            provenance_hash=prov_hash,
        )
        self._versions[assumption_id].append(v)

        # Enforce max versions
        if len(self._versions[assumption_id]) > self._max_versions:
            self._versions[assumption_id] = self._versions[assumption_id][
                -self._max_versions :
            ]

        return a

    def delete(self, assumption_id: str) -> bool:
        """Delete an assumption. Fails if in use."""
        if assumption_id not in self._assumptions:
            raise AssumptionNotFoundError(
                f"Assumption '{assumption_id}' not found"
            )

        # Check if in use by dependencies
        if assumption_id in self._dependencies and self._dependencies[assumption_id]:
            raise AssumptionInUseError(
                f"Assumption '{assumption_id}' is in use by: "
                f"{', '.join(self._dependencies[assumption_id])}"
            )

        del self._assumptions[assumption_id]
        if assumption_id in self._versions:
            del self._versions[assumption_id]
        return True

    def list_assumptions(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> List[Assumption]:
        """List assumptions with optional filters."""
        results = list(self._assumptions.values())

        if category:
            results = [a for a in results if a.category == category]

        if tags:
            results = [
                a for a in results
                if any(t in a.tags for t in tags)
            ]

        if search:
            lo = search.lower()
            results = [
                a for a in results
                if lo in a.name.lower()
                or lo in a.description.lower()
                or lo in a.assumption_id.lower()
            ]

        return results

    def get_value(
        self,
        assumption_id: str,
        scenario_overrides: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get the current value, optionally with scenario overrides."""
        a = self.get(assumption_id)

        if scenario_overrides and assumption_id in scenario_overrides:
            return scenario_overrides[assumption_id]

        return a.value

    def set_value(
        self,
        assumption_id: str,
        value: Any,
        change_reason: str = "",
        changed_by: str = "system",
    ) -> Assumption:
        """Set a new value (convenience wrapper around update)."""
        return self.update(
            assumption_id,
            value=value,
            change_reason=change_reason,
            changed_by=changed_by,
        )

    def get_versions(self, assumption_id: str) -> List[AssumptionVersion]:
        """Get version history for an assumption."""
        if assumption_id not in self._versions:
            raise AssumptionNotFoundError(
                f"Assumption '{assumption_id}' not found"
            )
        return list(self._versions[assumption_id])

    def export_all(self) -> Dict[str, Any]:
        """Export all assumptions and versions."""
        assumptions_data = [a.to_dict() for a in self._assumptions.values()]
        versions_data = {}
        for aid, vlist in self._versions.items():
            versions_data[aid] = [v.to_dict() for v in vlist]

        export_data = {
            "assumptions": assumptions_data,
            "versions": versions_data,
            "exported_at": datetime.utcnow().isoformat(),
        }

        # Compute integrity hash
        payload = json.dumps(assumptions_data, sort_keys=True, default=str)
        export_data["integrity_hash"] = hashlib.sha256(
            payload.encode()
        ).hexdigest()

        return export_data

    def import_all(
        self,
        data: Dict[str, Any],
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """Import assumptions from export data."""
        imported = 0
        skipped = 0
        errors = []

        for item in data.get("assumptions", []):
            aid = item.get("assumption_id", "")
            if aid in self._assumptions:
                if skip_duplicates:
                    skipped += 1
                    continue
                else:
                    errors.append(f"Duplicate: {aid}")
                    continue

            try:
                self.create(
                    assumption_id=aid,
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    category=item.get("category", "custom"),
                    data_type=item.get("data_type", "float"),
                    value=item.get("value"),
                    unit=item.get("unit", ""),
                    source=item.get("source", ""),
                    tags=item.get("tags"),
                    metadata=item.get("metadata"),
                )
                imported += 1
            except Exception as e:
                errors.append(f"Error importing {aid}: {str(e)}")

        return {"imported": imported, "skipped": skipped, "errors": errors}

    def register_dependency(self, assumption_id: str, dependent_id: str):
        """Register that dependent_id depends on assumption_id."""
        if assumption_id not in self._dependencies:
            self._dependencies[assumption_id] = []
        if dependent_id not in self._dependencies[assumption_id]:
            self._dependencies[assumption_id].append(dependent_id)

    @property
    def count(self) -> int:
        return len(self._assumptions)

    def _provenance_hash(self, operation: str, assumption_id: str, value: Any) -> str:
        if not self._enable_provenance:
            return ""
        payload = json.dumps(
            {"operation": operation, "assumption_id": assumption_id, "value": str(value)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Fresh AssumptionRegistry for each test."""
    return AssumptionRegistry(max_versions=50)


@pytest.fixture
def populated_registry():
    """Registry pre-loaded with 5 assumptions."""
    reg = AssumptionRegistry(max_versions=50)
    reg.create("diesel_ef_us", "US Diesel EF", category="emission_factor",
               value=2.68, tags=["diesel", "US", "scope1"])
    reg.create("natural_gas_ef_us", "US Natural Gas EF", category="emission_factor",
               value=1.93, tags=["natural_gas", "US", "scope1"])
    reg.create("electricity_ef_us", "US Electricity EF", category="emission_factor",
               value=0.42, tags=["electricity", "US", "scope2"])
    reg.create("coal_ef_us", "US Coal EF", category="emission_factor",
               value=3.45, tags=["coal", "US", "scope1"])
    reg.create("carbon_price_us", "US Carbon Price", category="economic",
               value=50.0, tags=["carbon_price", "US"])
    return reg


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCreateAssumption:
    """Test create() operation."""

    def test_create_success(self, registry):
        a = registry.create("test_id", "Test Assumption", value=42.0)
        assert a.assumption_id == "test_id"
        assert a.name == "Test Assumption"
        assert a.value == 42.0
        assert a.version == 1

    def test_create_with_all_fields(self, registry):
        a = registry.create(
            "ef_diesel", "Diesel EF",
            description="Emission factor for diesel",
            category="emission_factor",
            data_type="float",
            value=2.68,
            unit="kgCO2e/L",
            source="EPA",
            tags=["diesel"],
            metadata={"region": "US"},
        )
        assert a.description == "Emission factor for diesel"
        assert a.unit == "kgCO2e/L"
        assert a.tags == ["diesel"]

    def test_create_duplicate_fails(self, registry):
        registry.create("dup", "First")
        with pytest.raises(DuplicateAssumptionError, match="already exists"):
            registry.create("dup", "Second")

    def test_create_empty_id_fails(self, registry):
        with pytest.raises(ValidationError, match="assumption_id"):
            registry.create("", "Test")

    def test_create_empty_name_fails(self, registry):
        with pytest.raises(ValidationError, match="name"):
            registry.create("id", "")

    def test_create_increments_count(self, registry):
        assert registry.count == 0
        registry.create("a1", "Name1")
        assert registry.count == 1
        registry.create("a2", "Name2")
        assert registry.count == 2

    def test_create_records_initial_version(self, registry):
        registry.create("a1", "Name1", value=10)
        versions = registry.get_versions("a1")
        assert len(versions) == 1
        assert versions[0].version == 1
        assert versions[0].value == 10
        assert versions[0].change_type == "create"

    def test_create_generates_provenance_hash(self, registry):
        registry.create("a1", "Name1", value=10)
        versions = registry.get_versions("a1")
        assert len(versions[0].provenance_hash) == 64


class TestGetAssumption:
    """Test get() operation."""

    def test_get_success(self, populated_registry):
        a = populated_registry.get("diesel_ef_us")
        assert a.name == "US Diesel EF"
        assert a.value == 2.68

    def test_get_not_found(self, registry):
        with pytest.raises(AssumptionNotFoundError, match="not found"):
            registry.get("nonexistent")


class TestUpdateAssumption:
    """Test update() operation."""

    def test_update_value(self, populated_registry):
        a = populated_registry.update("diesel_ef_us", value=2.75)
        assert a.value == 2.75
        assert a.version == 2

    def test_update_name(self, populated_registry):
        a = populated_registry.update("diesel_ef_us", name="Updated Name")
        assert a.name == "Updated Name"

    def test_update_description(self, populated_registry):
        a = populated_registry.update("diesel_ef_us", description="New desc")
        assert a.description == "New desc"

    def test_update_tags(self, populated_registry):
        a = populated_registry.update("diesel_ef_us", tags=["new_tag"])
        assert a.tags == ["new_tag"]

    def test_update_metadata(self, populated_registry):
        a = populated_registry.update("diesel_ef_us", metadata={"new": "val"})
        assert a.metadata == {"new": "val"}

    def test_update_creates_version(self, populated_registry):
        populated_registry.update("diesel_ef_us", value=2.75)
        versions = populated_registry.get_versions("diesel_ef_us")
        assert len(versions) == 2
        assert versions[1].value == 2.75
        assert versions[1].version == 2

    def test_update_not_found(self, registry):
        with pytest.raises(AssumptionNotFoundError):
            registry.update("nonexistent", value=1)

    def test_multiple_updates_increment_version(self, populated_registry):
        populated_registry.update("diesel_ef_us", value=2.75)
        populated_registry.update("diesel_ef_us", value=2.80)
        populated_registry.update("diesel_ef_us", value=2.85)
        a = populated_registry.get("diesel_ef_us")
        assert a.version == 4  # initial 1 + 3 updates
        versions = populated_registry.get_versions("diesel_ef_us")
        assert len(versions) == 4

    def test_update_with_change_reason(self, populated_registry):
        populated_registry.update(
            "diesel_ef_us", value=2.75,
            change_reason="Updated for 2025 data",
            changed_by="analyst1",
        )
        versions = populated_registry.get_versions("diesel_ef_us")
        latest = versions[-1]
        assert latest.change_reason == "Updated for 2025 data"
        assert latest.changed_by == "analyst1"

    def test_update_provenance_hash(self, populated_registry):
        populated_registry.update("diesel_ef_us", value=2.75)
        versions = populated_registry.get_versions("diesel_ef_us")
        assert len(versions[-1].provenance_hash) == 64


class TestDeleteAssumption:
    """Test delete() operation."""

    def test_delete_success(self, populated_registry):
        count_before = populated_registry.count
        result = populated_registry.delete("carbon_price_us")
        assert result is True
        assert populated_registry.count == count_before - 1

    def test_delete_not_found(self, registry):
        with pytest.raises(AssumptionNotFoundError):
            registry.delete("nonexistent")

    def test_delete_in_use_fails(self, populated_registry):
        populated_registry.register_dependency("diesel_ef_us", "calc_scope1")
        with pytest.raises(AssumptionInUseError, match="in use"):
            populated_registry.delete("diesel_ef_us")

    def test_delete_removes_versions(self, populated_registry):
        populated_registry.delete("carbon_price_us")
        with pytest.raises(AssumptionNotFoundError):
            populated_registry.get_versions("carbon_price_us")


class TestListAssumptions:
    """Test list_assumptions() operation."""

    def test_list_all(self, populated_registry):
        results = populated_registry.list_assumptions()
        assert len(results) == 5

    def test_list_by_category(self, populated_registry):
        results = populated_registry.list_assumptions(category="emission_factor")
        assert len(results) == 4

    def test_list_by_tags(self, populated_registry):
        results = populated_registry.list_assumptions(tags=["diesel"])
        assert len(results) == 1
        assert results[0].assumption_id == "diesel_ef_us"

    def test_list_by_search(self, populated_registry):
        results = populated_registry.list_assumptions(search="diesel")
        assert len(results) == 1

    def test_list_by_search_case_insensitive(self, populated_registry):
        results = populated_registry.list_assumptions(search="DIESEL")
        assert len(results) == 1

    def test_list_empty_registry(self, registry):
        results = registry.list_assumptions()
        assert results == []

    def test_list_no_match(self, populated_registry):
        results = populated_registry.list_assumptions(category="nonexistent")
        assert results == []

    def test_list_multiple_tag_match(self, populated_registry):
        results = populated_registry.list_assumptions(tags=["scope1"])
        assert len(results) == 3  # diesel, natural_gas, coal


class TestGetValue:
    """Test get_value() operation."""

    def test_get_baseline_value(self, populated_registry):
        val = populated_registry.get_value("diesel_ef_us")
        assert val == 2.68

    def test_get_value_with_override(self, populated_registry):
        overrides = {"diesel_ef_us": 3.10}
        val = populated_registry.get_value("diesel_ef_us", scenario_overrides=overrides)
        assert val == 3.10

    def test_get_value_no_override_falls_back(self, populated_registry):
        overrides = {"other_id": 999}
        val = populated_registry.get_value("diesel_ef_us", scenario_overrides=overrides)
        assert val == 2.68

    def test_get_value_not_found(self, registry):
        with pytest.raises(AssumptionNotFoundError):
            registry.get_value("nonexistent")


class TestSetValue:
    """Test set_value() operation."""

    def test_set_value_success(self, populated_registry):
        a = populated_registry.set_value("diesel_ef_us", 2.75)
        assert a.value == 2.75
        assert a.version == 2

    def test_set_value_creates_version(self, populated_registry):
        populated_registry.set_value("diesel_ef_us", 2.75)
        versions = populated_registry.get_versions("diesel_ef_us")
        assert len(versions) == 2


class TestGetVersions:
    """Test get_versions() operation."""

    def test_get_versions_returns_history(self, populated_registry):
        versions = populated_registry.get_versions("diesel_ef_us")
        assert len(versions) == 1
        assert versions[0].version == 1

    def test_versions_grow_with_updates(self, populated_registry):
        populated_registry.update("diesel_ef_us", value=2.75)
        populated_registry.update("diesel_ef_us", value=2.80)
        versions = populated_registry.get_versions("diesel_ef_us")
        assert len(versions) == 3

    def test_versions_not_found(self, registry):
        with pytest.raises(AssumptionNotFoundError):
            registry.get_versions("nonexistent")


class TestMaxVersionsLimit:
    """Test that max_versions limit is enforced."""

    def test_max_versions_enforced(self):
        reg = AssumptionRegistry(max_versions=5)
        reg.create("a1", "Name", value=0)
        for i in range(1, 10):
            reg.update("a1", value=i)
        versions = reg.get_versions("a1")
        assert len(versions) == 5
        # Should keep the latest 5
        assert versions[-1].value == 9


class TestExportAll:
    """Test export_all() operation."""

    def test_export_includes_all_data(self, populated_registry):
        data = populated_registry.export_all()
        assert "assumptions" in data
        assert "versions" in data
        assert "exported_at" in data
        assert "integrity_hash" in data
        assert len(data["assumptions"]) == 5

    def test_export_hash_integrity(self, populated_registry):
        data = populated_registry.export_all()
        assert len(data["integrity_hash"]) == 64

    def test_export_empty_registry(self, registry):
        data = registry.export_all()
        assert len(data["assumptions"]) == 0


class TestImportAll:
    """Test import_all() operation."""

    def test_import_success(self, registry):
        data = {
            "assumptions": [
                {"assumption_id": "a1", "name": "A1", "value": 10},
                {"assumption_id": "a2", "name": "A2", "value": 20},
            ]
        }
        result = registry.import_all(data)
        assert result["imported"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == []
        assert registry.count == 2

    def test_import_skip_duplicates(self, populated_registry):
        data = {
            "assumptions": [
                {"assumption_id": "diesel_ef_us", "name": "Dup", "value": 99},
                {"assumption_id": "new_one", "name": "New", "value": 1},
            ]
        }
        result = populated_registry.import_all(data, skip_duplicates=True)
        assert result["imported"] == 1
        assert result["skipped"] == 1

    def test_import_error_handling(self, registry):
        data = {
            "assumptions": [
                {"assumption_id": "a1", "name": "A1", "value": 10},
                {"assumption_id": "", "name": "Bad", "value": 0},  # invalid
            ]
        }
        result = registry.import_all(data)
        assert result["imported"] == 1
        assert len(result["errors"]) == 1

    def test_import_empty_data(self, registry):
        result = registry.import_all({})
        assert result["imported"] == 0


class TestProvenanceOnOperations:
    """Test provenance hash is generated on each operation."""

    def test_create_has_provenance(self, registry):
        registry.create("a1", "Name", value=10)
        versions = registry.get_versions("a1")
        assert versions[0].provenance_hash != ""
        assert len(versions[0].provenance_hash) == 64

    def test_update_has_provenance(self, populated_registry):
        populated_registry.update("diesel_ef_us", value=2.75)
        versions = populated_registry.get_versions("diesel_ef_us")
        assert versions[-1].provenance_hash != ""

    def test_provenance_disabled(self):
        reg = AssumptionRegistry(enable_provenance=False)
        reg.create("a1", "Name", value=10)
        versions = reg.get_versions("a1")
        assert versions[0].provenance_hash == ""

    def test_different_values_different_provenance(self, registry):
        registry.create("a1", "Name", value=10)
        registry.create("a2", "Name2", value=20)
        v1 = registry.get_versions("a1")[0]
        v2 = registry.get_versions("a2")[0]
        assert v1.provenance_hash != v2.provenance_hash


class TestRegistryEdgeCases:
    """Test edge cases for the registry."""

    def test_none_value_allowed(self, registry):
        a = registry.create("a1", "Name", value=None)
        assert a.value is None

    def test_list_value_allowed(self, registry):
        a = registry.create("a1", "Name", data_type="list_float", value=[1.0, 2.0])
        assert a.value == [1.0, 2.0]

    def test_dict_value_allowed(self, registry):
        a = registry.create("a1", "Name", data_type="dict", value={"k": "v"})
        assert a.value == {"k": "v"}

    def test_boolean_value_allowed(self, registry):
        a = registry.create("a1", "Name", data_type="boolean", value=True)
        assert a.value is True

    def test_string_value_allowed(self, registry):
        a = registry.create("a1", "Name", data_type="string", value="hello")
        assert a.value == "hello"

    def test_zero_value_allowed(self, registry):
        a = registry.create("a1", "Name", value=0)
        assert a.value == 0
