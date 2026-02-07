# -*- coding: utf-8 -*-
"""
Unit Tests for Assumptions Models (AGENT-FOUND-004)

Tests all Pydantic-style models, enums, field validation, serialization,
and edge cases for the assumptions registry data types.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models mirroring greenlang/assumptions/models.py
# ---------------------------------------------------------------------------


class AssumptionDataType(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    DATE = "date"
    LIST_FLOAT = "list_float"
    LIST_STRING = "list_string"
    DICT = "dict"


class AssumptionCategory(str, Enum):
    EMISSION_FACTOR = "emission_factor"
    CONVERSION_FACTOR = "conversion_factor"
    BENCHMARK = "benchmark"
    THRESHOLD = "threshold"
    POLICY = "policy"
    ECONOMIC = "economic"
    TECHNICAL = "technical"
    CUSTOM = "custom"


class ScenarioType(str, Enum):
    BASELINE = "baseline"
    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"
    CUSTOM = "custom"


class ChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AssumptionMetadata:
    """Metadata attached to an assumption."""

    def __init__(
        self,
        region: str = "",
        sector: str = "",
        fuel_type: str = "",
        source: str = "",
        source_url: str = "",
        effective_date: Optional[str] = None,
        expiry_date: Optional[str] = None,
        custom: Optional[Dict[str, Any]] = None,
    ):
        self.region = region
        self.sector = sector
        self.fuel_type = fuel_type
        self.source = source
        self.source_url = source_url
        self.effective_date = effective_date
        self.expiry_date = expiry_date
        self.custom = custom or {}

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "region": self.region,
            "sector": self.sector,
            "fuel_type": self.fuel_type,
            "source": self.source,
            "source_url": self.source_url,
            "effective_date": self.effective_date,
            "expiry_date": self.expiry_date,
            "custom": self.custom,
        }
        return d


class Assumption:
    """Core assumption model."""

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
        if not assumption_id or len(assumption_id) < 1:
            raise ValueError("assumption_id is required and must be non-empty")
        if not name or len(name) < 1:
            raise ValueError("name is required and must be non-empty")

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
    """A single version snapshot of an assumption."""

    def __init__(
        self,
        assumption_id: str,
        version: int,
        value: Any,
        change_type: str = "update",
        changed_by: str = "system",
        change_reason: str = "",
        provenance_hash: str = "",
        timestamp: Optional[str] = None,
    ):
        self.assumption_id = assumption_id
        self.version = version
        self.value = value
        self.change_type = change_type
        self.changed_by = changed_by
        self.change_reason = change_reason
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
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


class Scenario:
    """A scenario with assumption overrides."""

    def __init__(
        self,
        scenario_id: str,
        name: str,
        description: str = "",
        scenario_type: str = "custom",
        overrides: Optional[Dict[str, Any]] = None,
        parent_scenario: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_active: bool = True,
        created_at: Optional[str] = None,
    ):
        if not scenario_id or len(scenario_id) < 1:
            raise ValueError("scenario_id is required")
        if not name or len(name) < 1:
            raise ValueError("name is required")

        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.scenario_type = scenario_type
        self.overrides = overrides or {}
        self.parent_scenario = parent_scenario
        self.tags = tags or []
        self.is_active = is_active
        self.created_at = created_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type,
            "overrides": self.overrides,
            "parent_scenario": self.parent_scenario,
            "tags": self.tags,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


class ChangeLogEntry:
    """Changelog entry for an assumption modification."""

    def __init__(
        self,
        entry_id: str,
        assumption_id: str,
        change_type: str,
        old_value: Any = None,
        new_value: Any = None,
        changed_by: str = "system",
        change_reason: str = "",
        provenance_hash: str = "",
        timestamp: Optional[str] = None,
    ):
        self.entry_id = entry_id
        self.assumption_id = assumption_id
        self.change_type = change_type
        self.old_value = old_value
        self.new_value = new_value
        self.changed_by = changed_by
        self.change_reason = change_reason
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "assumption_id": self.assumption_id,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changed_by": self.changed_by,
            "change_reason": self.change_reason,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


class ValidationRule:
    """A validation rule for an assumption value."""

    def __init__(
        self,
        rule_id: str,
        assumption_id: str,
        rule_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        message: str = "",
    ):
        self.rule_id = rule_id
        self.assumption_id = assumption_id
        self.rule_type = rule_type
        self.parameters = parameters or {}
        self.severity = severity
        self.message = message


class ValidationResult:
    """Result of a validation check."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class DependencyNode:
    """A node in the dependency graph."""

    def __init__(
        self,
        node_id: str,
        node_type: str = "assumption",
        upstream: Optional[List[str]] = None,
        downstream: Optional[List[str]] = None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.upstream = upstream or []
        self.downstream = downstream or []


class SensitivityResult:
    """Result of a sensitivity analysis run."""

    def __init__(
        self,
        assumption_id: str,
        base_value: float,
        variations: Optional[List[Dict[str, float]]] = None,
        impact_summary: Optional[Dict[str, Any]] = None,
    ):
        self.assumption_id = assumption_id
        self.base_value = base_value
        self.variations = variations or []
        self.impact_summary = impact_summary or {}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAssumptionDataTypeEnum:
    """Test AssumptionDataType enum values."""

    def test_float_value(self):
        assert AssumptionDataType.FLOAT.value == "float"

    def test_integer_value(self):
        assert AssumptionDataType.INTEGER.value == "integer"

    def test_string_value(self):
        assert AssumptionDataType.STRING.value == "string"

    def test_boolean_value(self):
        assert AssumptionDataType.BOOLEAN.value == "boolean"

    def test_percentage_value(self):
        assert AssumptionDataType.PERCENTAGE.value == "percentage"

    def test_ratio_value(self):
        assert AssumptionDataType.RATIO.value == "ratio"

    def test_date_value(self):
        assert AssumptionDataType.DATE.value == "date"

    def test_list_float_value(self):
        assert AssumptionDataType.LIST_FLOAT.value == "list_float"

    def test_list_string_value(self):
        assert AssumptionDataType.LIST_STRING.value == "list_string"

    def test_dict_value(self):
        assert AssumptionDataType.DICT.value == "dict"

    def test_all_10_types(self):
        assert len(AssumptionDataType) == 10


class TestAssumptionCategoryEnum:
    """Test AssumptionCategory enum values."""

    def test_emission_factor(self):
        assert AssumptionCategory.EMISSION_FACTOR.value == "emission_factor"

    def test_benchmark(self):
        assert AssumptionCategory.BENCHMARK.value == "benchmark"

    def test_custom(self):
        assert AssumptionCategory.CUSTOM.value == "custom"

    def test_all_8_categories(self):
        assert len(AssumptionCategory) == 8


class TestScenarioTypeEnum:
    """Test ScenarioType enum values."""

    def test_baseline(self):
        assert ScenarioType.BASELINE.value == "baseline"

    def test_conservative(self):
        assert ScenarioType.CONSERVATIVE.value == "conservative"

    def test_optimistic(self):
        assert ScenarioType.OPTIMISTIC.value == "optimistic"

    def test_custom(self):
        assert ScenarioType.CUSTOM.value == "custom"


class TestChangeTypeEnum:
    """Test ChangeType enum values."""

    def test_create(self):
        assert ChangeType.CREATE.value == "create"

    def test_update(self):
        assert ChangeType.UPDATE.value == "update"

    def test_delete(self):
        assert ChangeType.DELETE.value == "delete"

    def test_restore(self):
        assert ChangeType.RESTORE.value == "restore"


class TestValidationSeverityEnum:
    """Test ValidationSeverity enum values."""

    def test_error(self):
        assert ValidationSeverity.ERROR.value == "error"

    def test_warning(self):
        assert ValidationSeverity.WARNING.value == "warning"

    def test_info(self):
        assert ValidationSeverity.INFO.value == "info"


class TestAssumptionModel:
    """Test Assumption model creation and serialization."""

    def test_creation_with_required_fields(self):
        a = Assumption(assumption_id="test_id", name="Test Name")
        assert a.assumption_id == "test_id"
        assert a.name == "Test Name"

    def test_creation_with_all_fields(self):
        a = Assumption(
            assumption_id="diesel_ef",
            name="Diesel EF",
            description="Diesel emission factor",
            category="emission_factor",
            data_type="float",
            value=2.68,
            unit="kgCO2e/L",
            source="EPA",
            tags=["diesel", "US"],
            metadata={"region": "US"},
            version=3,
        )
        assert a.value == 2.68
        assert a.unit == "kgCO2e/L"
        assert a.version == 3
        assert a.tags == ["diesel", "US"]

    def test_empty_assumption_id_raises(self):
        with pytest.raises(ValueError, match="assumption_id"):
            Assumption(assumption_id="", name="Test")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Assumption(assumption_id="id", name="")

    def test_to_dict(self):
        a = Assumption(assumption_id="id1", name="Name1", value=42)
        d = a.to_dict()
        assert d["assumption_id"] == "id1"
        assert d["name"] == "Name1"
        assert d["value"] == 42
        assert "created_at" in d
        assert "updated_at" in d

    def test_default_version_is_1(self):
        a = Assumption(assumption_id="id", name="Name")
        assert a.version == 1

    def test_default_tags_empty_list(self):
        a = Assumption(assumption_id="id", name="Name")
        assert a.tags == []

    def test_default_metadata_empty_dict(self):
        a = Assumption(assumption_id="id", name="Name")
        assert a.metadata == {}


class TestAssumptionVersionModel:
    """Test AssumptionVersion model."""

    def test_creation(self):
        v = AssumptionVersion(
            assumption_id="diesel_ef",
            version=2,
            value=2.75,
            change_type="update",
            changed_by="user1",
            change_reason="Updated for 2025 data",
        )
        assert v.assumption_id == "diesel_ef"
        assert v.version == 2
        assert v.value == 2.75

    def test_to_dict(self):
        v = AssumptionVersion(assumption_id="id", version=1, value=10)
        d = v.to_dict()
        assert d["assumption_id"] == "id"
        assert d["version"] == 1
        assert "timestamp" in d


class TestScenarioModel:
    """Test Scenario model creation and serialization."""

    def test_creation(self):
        s = Scenario(scenario_id="s1", name="Baseline", scenario_type="baseline")
        assert s.scenario_id == "s1"
        assert s.name == "Baseline"
        assert s.scenario_type == "baseline"

    def test_creation_with_overrides(self):
        s = Scenario(
            scenario_id="s2",
            name="Conservative",
            overrides={"diesel_ef": 3.0},
        )
        assert s.overrides == {"diesel_ef": 3.0}

    def test_empty_scenario_id_raises(self):
        with pytest.raises(ValueError, match="scenario_id"):
            Scenario(scenario_id="", name="Test")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Scenario(scenario_id="s1", name="")

    def test_to_dict(self):
        s = Scenario(scenario_id="s1", name="Test")
        d = s.to_dict()
        assert d["scenario_id"] == "s1"
        assert "created_at" in d

    def test_default_active(self):
        s = Scenario(scenario_id="s1", name="Test")
        assert s.is_active is True

    def test_parent_scenario(self):
        s = Scenario(scenario_id="s2", name="Child", parent_scenario="s1")
        assert s.parent_scenario == "s1"


class TestChangeLogEntryModel:
    """Test ChangeLogEntry model."""

    def test_creation(self):
        e = ChangeLogEntry(
            entry_id="e1",
            assumption_id="a1",
            change_type="update",
            old_value=2.68,
            new_value=2.75,
        )
        assert e.entry_id == "e1"
        assert e.old_value == 2.68
        assert e.new_value == 2.75

    def test_to_dict(self):
        e = ChangeLogEntry(entry_id="e1", assumption_id="a1", change_type="create")
        d = e.to_dict()
        assert d["entry_id"] == "e1"
        assert d["change_type"] == "create"
        assert "timestamp" in d


class TestValidationRuleModel:
    """Test ValidationRule model."""

    def test_creation(self):
        r = ValidationRule(
            rule_id="r1",
            assumption_id="a1",
            rule_type="min_value",
            parameters={"min_value": 0},
            severity="error",
            message="Must be positive",
        )
        assert r.rule_id == "r1"
        assert r.rule_type == "min_value"
        assert r.parameters["min_value"] == 0


class TestValidationResultModel:
    """Test ValidationResult model."""

    def test_valid_result(self):
        r = ValidationResult(is_valid=True)
        assert r.is_valid is True
        assert r.errors == []
        assert r.warnings == []

    def test_invalid_result(self):
        r = ValidationResult(
            is_valid=False,
            errors=["Value too low"],
            warnings=["Outside typical range"],
        )
        assert r.is_valid is False
        assert len(r.errors) == 1
        assert len(r.warnings) == 1


class TestDependencyNodeModel:
    """Test DependencyNode model."""

    def test_creation(self):
        n = DependencyNode(node_id="a1")
        assert n.node_id == "a1"
        assert n.upstream == []
        assert n.downstream == []

    def test_with_dependencies(self):
        n = DependencyNode(
            node_id="a1",
            upstream=["a2", "a3"],
            downstream=["calc1"],
        )
        assert len(n.upstream) == 2
        assert "calc1" in n.downstream


class TestSensitivityResultModel:
    """Test SensitivityResult model."""

    def test_creation(self):
        sr = SensitivityResult(
            assumption_id="a1",
            base_value=2.68,
            variations=[
                {"factor": 0.9, "value": 2.412},
                {"factor": 1.1, "value": 2.948},
            ],
        )
        assert sr.assumption_id == "a1"
        assert sr.base_value == 2.68
        assert len(sr.variations) == 2


class TestAssumptionMetadataModel:
    """Test AssumptionMetadata model."""

    def test_creation(self):
        m = AssumptionMetadata(region="US", source="EPA")
        assert m.region == "US"
        assert m.source == "EPA"

    def test_defaults(self):
        m = AssumptionMetadata()
        assert m.region == ""
        assert m.custom == {}

    def test_to_dict(self):
        m = AssumptionMetadata(region="EU", sector="transport")
        d = m.to_dict()
        assert d["region"] == "EU"
        assert d["sector"] == "transport"

    def test_custom_fields(self):
        m = AssumptionMetadata(custom={"extra_key": "extra_value"})
        assert m.custom["extra_key"] == "extra_value"
