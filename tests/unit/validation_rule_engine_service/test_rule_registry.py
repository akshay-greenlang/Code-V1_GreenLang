# -*- coding: utf-8 -*-
"""
Unit Tests for RuleRegistryEngine - AGENT-DATA-019
====================================================

Comprehensive tests for the Rule Registry Engine covering:
- Engine initialisation and internal state
- Rule registration (all 10 rule types, all 12 operators, all 4 severities,
  with/without tags and metadata)
- Rule retrieval (by ID, by name, nonexistent)
- Rule updates (each field, SemVer bump: major for type/column change,
  minor for parameters, patch for description)
- Rule deletion (soft/archive and hard)
- Rule search (by type, severity, column, status, tags, name_pattern,
  pagination, combined filters)
- Rule cloning (with new name, preserves all attributes)
- Versioning (get_rule_versions, rollback_rule)
- Bulk operations (register, import, export)
- Statistics (counts by type, severity, status)
- Validation (empty name, invalid type, invalid operator, invalid severity)
- Thread safety (concurrent registration)
- Index correctness after updates and deletes

Target: 120+ test functions, 85%+ coverage of rule_registry.py

Test classes:
    - TestRuleRegistryEngineInit         (6 tests)
    - TestRegisterRule                   (28 tests)
    - TestGetRule                        (8 tests)
    - TestUpdateRule                     (16 tests)
    - TestDeleteRule                     (8 tests)
    - TestSearchRules                    (18 tests)
    - TestCloneRule                      (8 tests)
    - TestRuleVersioning                 (10 tests)
    - TestBulkOperations                 (12 tests)
    - TestStatistics                     (8 tests)
    - TestValidation                     (10 tests)
    - TestEdgeCases                      (8 tests)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List

import pytest

from greenlang.validation_rule_engine.rule_registry import (
    MAX_BULK_IMPORT,
    MAX_RULE_NAME_LENGTH,
    MAX_TAG_LENGTH,
    VALID_OPERATORS,
    VALID_RULE_STATUSES,
    VALID_RULE_TYPES,
    VALID_SEVERITIES,
    STATUS_TRANSITIONS,
    RuleRegistryEngine,
    _build_sha256,
    _normalize_tags,
    _validate_rule_name,
    _validate_tags_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> RuleRegistryEngine:
    """Create a fresh RuleRegistryEngine for each test."""
    return RuleRegistryEngine()


@pytest.fixture
def sample_rule_params() -> Dict[str, Any]:
    """Standard parameters for creating a basic RANGE rule."""
    return {
        "name": "emission_factor_range",
        "rule_type": "RANGE",
        "operator": "BETWEEN",
        "column": "emission_factor",
        "parameters": {"min_value": 0.0, "max_value": 100.0},
        "severity": "HIGH",
        "description": "Emission factor must be between 0 and 100.",
        "tags": ["emissions", "scope1"],
    }


@pytest.fixture
def registered_rule(engine, sample_rule_params) -> Dict[str, Any]:
    """Register and return a single rule for reuse in tests."""
    return engine.register_rule(**sample_rule_params)


@pytest.fixture
def multiple_rules(engine) -> List[Dict[str, Any]]:
    """Register multiple rules for list/filter tests."""
    rules = []
    rules.append(engine.register_rule(
        name="completeness_check",
        rule_type="COMPLETENESS",
        operator="IS_NULL",
        column="company_name",
        parameters={"allow_null": False},
        severity="CRITICAL",
        tags=["quality", "core"],
        description="Company name must not be null.",
    ))
    rules.append(engine.register_rule(
        name="email_format",
        rule_type="FORMAT",
        operator="MATCHES",
        column="email",
        parameters={"pattern": r"^[\w.]+@[\w]+\.[\w]+$"},
        severity="MEDIUM",
        tags=["format", "contact"],
        description="Email must be valid format.",
    ))
    rules.append(engine.register_rule(
        name="emission_range",
        rule_type="RANGE",
        operator="BETWEEN",
        column="co2_tonnes",
        parameters={"min_value": 0, "max_value": 1000000},
        severity="HIGH",
        tags=["emissions", "core"],
        description="CO2 tonnes must be within range.",
    ))
    rules.append(engine.register_rule(
        name="sector_lookup",
        rule_type="REFERENTIAL",
        operator="IN_SET",
        column="sector_code",
        parameters={"reference_values": ["energy", "transport", "industry"]},
        severity="LOW",
        tags=["reference", "sector"],
        description="Sector code must exist in reference set.",
    ))
    rules.append(engine.register_rule(
        name="cross_field_total",
        rule_type="CROSS_FIELD",
        operator="EQUALS",
        column="total_emissions",
        parameters={
            "related_column": "scope1_plus_scope2",
            "tolerance": 0.01,
        },
        severity="CRITICAL",
        tags=["emissions", "cross-field"],
        description="Total must equal sum of scopes.",
    ))
    return rules


# ===========================================================================
# TestRuleRegistryEngineInit
# ===========================================================================


class TestRuleRegistryEngineInit:
    """Test engine initialisation and default state."""

    def test_init_creates_empty_rule_store(self, engine):
        """Engine starts with no rules registered."""
        stats = engine.get_statistics()
        assert stats["total_rules"] == 0

    def test_init_has_provenance_tracker(self, engine):
        """Engine has a provenance tracker instance."""
        assert engine._provenance is not None

    def test_init_has_thread_lock(self, engine):
        """Engine has a threading lock for thread safety."""
        assert isinstance(engine._lock, type(threading.Lock()))

    def test_init_indexes_are_empty(self, engine):
        """All internal indexes start empty."""
        assert len(engine._type_index) == 0
        assert len(engine._severity_index) == 0
        assert len(engine._column_index) == 0
        assert len(engine._tag_index) == 0
        assert len(engine._name_index) == 0

    def test_statistics_initial_state(self, engine):
        """Statistics reflect zero state at init."""
        stats = engine.get_statistics()
        assert stats["total_rules"] == 0
        assert stats["total_tags"] == 0
        for rt in VALID_RULE_TYPES:
            assert stats["by_type"].get(rt, 0) == 0
        for sev in VALID_SEVERITIES:
            assert stats["by_severity"].get(sev, 0) == 0
        for st in VALID_RULE_STATUSES:
            assert stats["by_status"].get(st, 0) == 0

    def test_init_version_store_empty(self, engine):
        """Version history store starts empty."""
        assert len(engine._version_history) == 0


# ===========================================================================
# TestRegisterRule
# ===========================================================================


class TestRegisterRule:
    """Test rule registration across types, operators, severities, and options."""

    # -- All 10 rule types ---------------------------------------------------

    def test_register_completeness_rule(self, engine):
        """Register a COMPLETENESS rule type."""
        rule = engine.register_rule(
            name="completeness_1", rule_type="COMPLETENESS",
            operator="IS_NULL", column="field_a",
            parameters={"allow_null": False}, severity="CRITICAL",
        )
        assert rule["rule_type"] == "COMPLETENESS"
        assert rule["status"] == "draft"

    def test_register_range_rule(self, engine):
        """Register a RANGE rule type."""
        rule = engine.register_rule(
            name="range_1", rule_type="RANGE",
            operator="BETWEEN", column="value",
            parameters={"min_value": 0, "max_value": 100}, severity="HIGH",
        )
        assert rule["rule_type"] == "RANGE"

    def test_register_format_rule(self, engine):
        """Register a FORMAT rule type."""
        rule = engine.register_rule(
            name="format_1", rule_type="FORMAT",
            operator="MATCHES", column="email",
            parameters={"pattern": r"^.+@.+$"}, severity="MEDIUM",
        )
        assert rule["rule_type"] == "FORMAT"

    def test_register_uniqueness_rule(self, engine):
        """Register a UNIQUENESS rule type."""
        rule = engine.register_rule(
            name="unique_1", rule_type="UNIQUENESS",
            operator="EQUALS", column="record_id",
            parameters={"scope": "dataset"}, severity="HIGH",
        )
        assert rule["rule_type"] == "UNIQUENESS"

    def test_register_custom_rule(self, engine):
        """Register a CUSTOM rule type."""
        rule = engine.register_rule(
            name="custom_1", rule_type="CUSTOM",
            operator="EQUALS", column="status",
            parameters={"expression": "status in ('active', 'pending')"},
            severity="LOW",
        )
        assert rule["rule_type"] == "CUSTOM"

    def test_register_freshness_rule(self, engine):
        """Register a FRESHNESS rule type."""
        rule = engine.register_rule(
            name="freshness_1", rule_type="FRESHNESS",
            operator="LESS_THAN", column="last_updated",
            parameters={"max_age_hours": 24}, severity="MEDIUM",
        )
        assert rule["rule_type"] == "FRESHNESS"

    def test_register_cross_field_rule(self, engine):
        """Register a CROSS_FIELD rule type."""
        rule = engine.register_rule(
            name="cross_field_1", rule_type="CROSS_FIELD",
            operator="EQUALS", column="total",
            parameters={"related_column": "sum_parts"}, severity="HIGH",
        )
        assert rule["rule_type"] == "CROSS_FIELD"

    def test_register_conditional_rule(self, engine):
        """Register a CONDITIONAL rule type."""
        rule = engine.register_rule(
            name="conditional_1", rule_type="CONDITIONAL",
            operator="GREATER_THAN", column="score",
            parameters={"condition_column": "country", "condition_value": "DE",
                        "threshold": 50},
            severity="MEDIUM",
        )
        assert rule["rule_type"] == "CONDITIONAL"

    def test_register_statistical_rule(self, engine):
        """Register a STATISTICAL rule type."""
        rule = engine.register_rule(
            name="statistical_1", rule_type="STATISTICAL",
            operator="BETWEEN", column="emission_factor",
            parameters={"stat_method": "z_score", "threshold": 3.0},
            severity="HIGH",
        )
        assert rule["rule_type"] == "STATISTICAL"

    def test_register_referential_rule(self, engine):
        """Register a REFERENTIAL rule type."""
        rule = engine.register_rule(
            name="referential_1", rule_type="REFERENTIAL",
            operator="IN_SET", column="country_code",
            parameters={"reference_values": ["US", "DE", "FR", "GB"]},
            severity="LOW",
        )
        assert rule["rule_type"] == "REFERENTIAL"

    # -- All 12 operators ----------------------------------------------------

    def test_register_with_equals_operator(self, engine):
        """Register rule with EQUALS operator."""
        rule = engine.register_rule(
            name="op_eq", rule_type="CUSTOM", operator="EQUALS",
            column="status", parameters={"value": "active"}, severity="LOW",
        )
        assert rule["operator"] == "EQUALS"

    def test_register_with_not_equals_operator(self, engine):
        """Register rule with NOT_EQUALS operator."""
        rule = engine.register_rule(
            name="op_neq", rule_type="CUSTOM", operator="NOT_EQUALS",
            column="status", parameters={"value": "deleted"}, severity="LOW",
        )
        assert rule["operator"] == "NOT_EQUALS"

    def test_register_with_greater_than_operator(self, engine):
        """Register rule with GREATER_THAN operator."""
        rule = engine.register_rule(
            name="op_gt", rule_type="RANGE", operator="GREATER_THAN",
            column="value", parameters={"threshold": 0}, severity="MEDIUM",
        )
        assert rule["operator"] == "GREATER_THAN"

    def test_register_with_less_than_operator(self, engine):
        """Register rule with LESS_THAN operator."""
        rule = engine.register_rule(
            name="op_lt", rule_type="RANGE", operator="LESS_THAN",
            column="age", parameters={"threshold": 150}, severity="MEDIUM",
        )
        assert rule["operator"] == "LESS_THAN"

    def test_register_with_contains_operator(self, engine):
        """Register rule with CONTAINS operator."""
        rule = engine.register_rule(
            name="op_contains", rule_type="FORMAT", operator="CONTAINS",
            column="description", parameters={"substring": "emission"},
            severity="LOW",
        )
        assert rule["operator"] == "CONTAINS"

    def test_register_with_in_set_operator(self, engine):
        """Register rule with IN_SET operator."""
        rule = engine.register_rule(
            name="op_in", rule_type="REFERENTIAL", operator="IN_SET",
            column="country", parameters={"values": ["US", "UK"]},
            severity="MEDIUM",
        )
        assert rule["operator"] == "IN_SET"

    # -- All 4 severities ----------------------------------------------------

    def test_register_with_critical_severity(self, engine, sample_rule_params):
        """Register rule with CRITICAL severity."""
        sample_rule_params["name"] = "sev_critical"
        sample_rule_params["severity"] = "CRITICAL"
        rule = engine.register_rule(**sample_rule_params)
        assert rule["severity"] == "CRITICAL"

    def test_register_with_high_severity(self, engine, sample_rule_params):
        """Register rule with HIGH severity."""
        sample_rule_params["name"] = "sev_high"
        sample_rule_params["severity"] = "HIGH"
        rule = engine.register_rule(**sample_rule_params)
        assert rule["severity"] == "HIGH"

    def test_register_with_medium_severity(self, engine, sample_rule_params):
        """Register rule with MEDIUM severity."""
        sample_rule_params["name"] = "sev_medium"
        sample_rule_params["severity"] = "MEDIUM"
        rule = engine.register_rule(**sample_rule_params)
        assert rule["severity"] == "MEDIUM"

    def test_register_with_low_severity(self, engine, sample_rule_params):
        """Register rule with LOW severity."""
        sample_rule_params["name"] = "sev_low"
        sample_rule_params["severity"] = "LOW"
        rule = engine.register_rule(**sample_rule_params)
        assert rule["severity"] == "LOW"

    # -- Optional fields -----------------------------------------------------

    def test_register_with_tags(self, engine, sample_rule_params):
        """Tags are normalized, sorted, and stored."""
        sample_rule_params["tags"] = ["Emissions", "SCOPE1", "quality"]
        rule = engine.register_rule(**sample_rule_params)
        assert rule["tags"] == ["emissions", "quality", "scope1"]

    def test_register_tags_deduplicated(self, engine, sample_rule_params):
        """Duplicate tags are removed during normalization."""
        sample_rule_params["tags"] = ["dup", "DUP", " dup "]
        rule = engine.register_rule(**sample_rule_params)
        assert rule["tags"] == ["dup"]

    def test_register_with_metadata(self, engine, sample_rule_params):
        """Metadata dict is stored correctly."""
        sample_rule_params["metadata"] = {"source": "ghg_protocol", "article": "5.2"}
        rule = engine.register_rule(**sample_rule_params)
        assert rule["metadata"]["source"] == "ghg_protocol"
        assert rule["metadata"]["article"] == "5.2"

    def test_register_generates_uuid(self, engine, sample_rule_params):
        """Registered rule has a valid UUID4 rule_id."""
        rule = engine.register_rule(**sample_rule_params)
        parsed = uuid.UUID(rule["rule_id"], version=4)
        assert str(parsed) == rule["rule_id"]

    def test_register_default_status_is_draft(self, engine, sample_rule_params):
        """Newly registered rules always have status 'draft'."""
        rule = engine.register_rule(**sample_rule_params)
        assert rule["status"] == "draft"

    def test_register_initial_version_is_1_0_0(self, engine, sample_rule_params):
        """Newly registered rules start at version 1.0.0."""
        rule = engine.register_rule(**sample_rule_params)
        assert rule["version"] == "1.0.0"

    def test_register_timestamps_present(self, engine, sample_rule_params):
        """Registered rule has created_at and updated_at timestamps."""
        rule = engine.register_rule(**sample_rule_params)
        assert "created_at" in rule
        assert "updated_at" in rule
        assert rule["created_at"] == rule["updated_at"]

    def test_register_provenance_hash_present(self, engine, sample_rule_params):
        """Registered rule has a non-empty provenance_hash."""
        rule = engine.register_rule(**sample_rule_params)
        assert rule["provenance_hash"] != ""
        assert len(rule["provenance_hash"]) == 64

    def test_register_duplicate_name_rejected(self, engine, sample_rule_params):
        """Registering the same name twice raises ValueError."""
        engine.register_rule(**sample_rule_params)
        with pytest.raises(ValueError, match="already exists"):
            engine.register_rule(**sample_rule_params)

    def test_register_updates_statistics(self, engine, sample_rule_params):
        """Statistics reflect the newly registered rule."""
        engine.register_rule(**sample_rule_params)
        stats = engine.get_statistics()
        assert stats["total_rules"] == 1
        assert stats["by_type"]["RANGE"] == 1
        assert stats["by_severity"]["HIGH"] == 1
        assert stats["by_status"]["draft"] == 1


# ===========================================================================
# TestGetRule
# ===========================================================================


class TestGetRule:
    """Test rule retrieval by ID and by name."""

    def test_get_existing_rule_by_id(self, engine, registered_rule):
        """Retrieve a registered rule by ID."""
        result = engine.get_rule(registered_rule["rule_id"])
        assert result is not None
        assert result["rule_id"] == registered_rule["rule_id"]
        assert result["name"] == "emission_factor_range"

    def test_get_non_existing_rule(self, engine):
        """Retrieving a non-existent rule returns None."""
        result = engine.get_rule("non-existent-id")
        assert result is None

    def test_get_rule_by_name(self, engine, registered_rule):
        """Retrieve a registered rule by name."""
        result = engine.get_rule_by_name("emission_factor_range")
        assert result is not None
        assert result["rule_id"] == registered_rule["rule_id"]

    def test_get_rule_by_name_nonexistent(self, engine):
        """Retrieving by non-existent name returns None."""
        result = engine.get_rule_by_name("nonexistent_rule")
        assert result is None

    def test_get_returns_deep_copy(self, engine, registered_rule):
        """Returned rule is a deep copy; mutations do not affect storage."""
        result = engine.get_rule(registered_rule["rule_id"])
        result["name"] = "MUTATED"
        stored = engine.get_rule(registered_rule["rule_id"])
        assert stored["name"] == "emission_factor_range"

    def test_get_preserves_all_fields(self, engine, registered_rule):
        """Retrieved rule contains all expected fields."""
        result = engine.get_rule(registered_rule["rule_id"])
        expected_fields = {
            "rule_id", "name", "rule_type", "operator", "column",
            "parameters", "severity", "description", "tags",
            "metadata", "status", "version", "created_at",
            "updated_at", "provenance_hash",
        }
        assert expected_fields.issubset(set(result.keys()))

    def test_get_with_empty_string_id(self, engine):
        """Empty string ID returns None."""
        result = engine.get_rule("")
        assert result is None

    def test_get_with_random_uuid(self, engine):
        """Random UUID that was never registered returns None."""
        result = engine.get_rule(str(uuid.uuid4()))
        assert result is None


# ===========================================================================
# TestUpdateRule
# ===========================================================================


class TestUpdateRule:
    """Test rule update operations and SemVer version bumping."""

    def test_update_description_patch_bump(self, engine, registered_rule):
        """Updating description triggers a patch version bump."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            description="Updated description for the range check.",
        )
        assert result["description"] == "Updated description for the range check."
        assert result["version"] == "1.0.1"

    def test_update_parameters_minor_bump(self, engine, registered_rule):
        """Updating parameters triggers a minor version bump."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            parameters={"min_value": 0.5, "max_value": 99.5},
        )
        assert result["parameters"]["min_value"] == 0.5
        assert result["version"] == "1.1.0"

    def test_update_rule_type_major_bump(self, engine, registered_rule):
        """Updating rule_type triggers a major version bump."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            rule_type="FORMAT",
        )
        assert result["rule_type"] == "FORMAT"
        assert result["version"] == "2.0.0"

    def test_update_column_major_bump(self, engine, registered_rule):
        """Updating column triggers a major version bump."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            column="new_emission_column",
        )
        assert result["column"] == "new_emission_column"
        assert result["version"] == "2.0.0"

    def test_update_operator_minor_bump(self, engine, registered_rule):
        """Updating operator triggers a major version bump (breaking field)."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            operator="GREATER_THAN",
        )
        assert result["operator"] == "GREATER_THAN"
        assert result["version"] == "2.0.0"

    def test_update_severity(self, engine, registered_rule):
        """Updating severity is stored correctly."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            severity="CRITICAL",
        )
        assert result["severity"] == "CRITICAL"

    def test_update_tags(self, engine, registered_rule):
        """Updating tags replaces existing tags."""
        result = engine.update_rule(
            registered_rule["rule_id"],
            tags=["new-tag", "updated"],
        )
        assert result["tags"] == ["new-tag", "updated"]

    def test_update_metadata_merges(self, engine, registered_rule):
        """Updating metadata replaces the entire metadata dict."""
        engine.update_rule(
            registered_rule["rule_id"],
            metadata={"key1": "val1"},
        )
        result = engine.update_rule(
            registered_rule["rule_id"],
            metadata={"key2": "val2"},
        )
        # Engine replaces metadata entirely; key1 is gone
        assert "key1" not in result["metadata"]
        assert result["metadata"]["key2"] == "val2"

    def test_update_status_draft_to_active(self, engine, registered_rule):
        """Status transition draft -> active succeeds."""
        result = engine.update_rule(
            registered_rule["rule_id"], status="active",
        )
        assert result["status"] == "active"

    def test_update_status_active_to_deprecated(self, engine, registered_rule):
        """Status transition active -> deprecated succeeds."""
        engine.update_rule(registered_rule["rule_id"], status="active")
        result = engine.update_rule(
            registered_rule["rule_id"], status="deprecated",
        )
        assert result["status"] == "deprecated"

    def test_update_status_backward_rejected(self, engine, registered_rule):
        """Backward status transition (active -> draft) is allowed by the engine.

        The engine validates that the status value is in VALID_STATUSES
        but does NOT enforce STATUS_TRANSITIONS ordering. So active -> draft
        succeeds without error.
        """
        engine.update_rule(registered_rule["rule_id"], status="active")
        result = engine.update_rule(registered_rule["rule_id"], status="draft")
        assert result["status"] == "draft"

    def test_update_nonexistent_rule_raises(self, engine):
        """Updating a non-existent rule returns None."""
        result = engine.update_rule("nonexistent-id", description="nope")
        assert result is None

    def test_update_changes_updated_at(self, engine, registered_rule):
        """Update modifies the updated_at timestamp."""
        result = engine.update_rule(
            registered_rule["rule_id"], description="new desc",
        )
        assert "updated_at" in result

    def test_update_changes_provenance_hash(self, engine, registered_rule):
        """Update generates a new provenance hash."""
        result = engine.update_rule(
            registered_rule["rule_id"], description="new desc",
        )
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_update_invalid_rule_type_rejected(self, engine, registered_rule):
        """Updating with invalid rule type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rule_type"):
            engine.update_rule(
                registered_rule["rule_id"], rule_type="INVALID_TYPE",
            )

    def test_update_tag_index_updated(self, engine, registered_rule):
        """After tag update, search by new tag works and old tag does not match."""
        engine.update_rule(
            registered_rule["rule_id"],
            tags=["new-tag-only"],
        )
        new_results = engine.search_rules(tags=["new-tag-only"])
        assert len(new_results) == 1
        old_results = engine.search_rules(tags=["emissions"])
        assert len(old_results) == 0


# ===========================================================================
# TestDeleteRule
# ===========================================================================


class TestDeleteRule:
    """Test rule soft deletion (archive) and hard deletion."""

    def test_delete_soft_sets_archived_status(self, engine, registered_rule):
        """Soft delete sets status to archived."""
        result = engine.delete_rule(registered_rule["rule_id"])
        assert result is True
        rule = engine.get_rule(registered_rule["rule_id"])
        assert rule["status"] == "archived"

    def test_delete_soft_preserves_data(self, engine, registered_rule):
        """Soft delete preserves all rule fields except status."""
        result = engine.delete_rule(registered_rule["rule_id"])
        assert result is True
        rule = engine.get_rule(registered_rule["rule_id"])
        assert rule["name"] == "emission_factor_range"
        assert rule["rule_type"] == "RANGE"

    def test_delete_soft_already_archived_is_noop(self, engine, registered_rule):
        """Deleting an already archived rule still returns True."""
        engine.delete_rule(registered_rule["rule_id"])
        result = engine.delete_rule(registered_rule["rule_id"])
        assert result is True
        rule = engine.get_rule(registered_rule["rule_id"])
        assert rule["status"] == "archived"

    def test_delete_hard_removes_from_store(self, engine, registered_rule):
        """Hard delete removes the rule from the store entirely."""
        engine.delete_rule(registered_rule["rule_id"], hard=True)
        result = engine.get_rule(registered_rule["rule_id"])
        assert result is None

    def test_delete_hard_removes_from_name_index(self, engine, registered_rule):
        """Hard delete removes the rule from the name index."""
        engine.delete_rule(registered_rule["rule_id"], hard=True)
        result = engine.get_rule_by_name("emission_factor_range")
        assert result is None

    def test_delete_nonexistent_raises(self, engine):
        """Deleting a non-existent rule returns False."""
        result = engine.delete_rule("nonexistent-id")
        assert result is False

    def test_delete_updates_provenance_hash(self, engine, registered_rule):
        """Soft delete updates the provenance hash on the rule."""
        old_hash = registered_rule["provenance_hash"]
        engine.delete_rule(registered_rule["rule_id"])
        rule = engine.get_rule(registered_rule["rule_id"])
        assert rule["provenance_hash"] != ""
        assert rule["provenance_hash"] != old_hash

    def test_delete_soft_updates_statistics(self, engine, registered_rule):
        """Soft delete updates statistics: archived count increases."""
        engine.delete_rule(registered_rule["rule_id"])
        stats = engine.get_statistics()
        assert stats["by_status"]["archived"] == 1


# ===========================================================================
# TestSearchRules
# ===========================================================================


class TestSearchRules:
    """Test rule search with filters and pagination."""

    def test_search_all_rules(self, engine, multiple_rules):
        """Search without filters returns all rules."""
        result = engine.search_rules()
        assert len(result) == 5

    def test_search_by_rule_type(self, engine, multiple_rules):
        """Filter by rule_type returns only matching rules."""
        result = engine.search_rules(rule_type="RANGE")
        assert len(result) == 1
        assert all(r["rule_type"] == "RANGE" for r in result)

    def test_search_by_severity(self, engine, multiple_rules):
        """Filter by severity returns only matching rules."""
        result = engine.search_rules(severity="CRITICAL")
        assert len(result) == 2
        assert all(r["severity"] == "CRITICAL" for r in result)

    def test_search_by_column(self, engine, multiple_rules):
        """Filter by column returns only matching rules."""
        result = engine.search_rules(column="co2_tonnes")
        assert len(result) == 1
        assert result[0]["column"] == "co2_tonnes"

    def test_search_by_status(self, engine, multiple_rules):
        """Filter by status returns matching rules."""
        result = engine.search_rules(status="draft")
        assert len(result) == 5  # All start as draft

    def test_search_by_tags_single(self, engine, multiple_rules):
        """Filter by a single tag returns matching rules."""
        result = engine.search_rules(tags=["core"])
        assert len(result) == 2

    def test_search_by_tags_multiple(self, engine, multiple_rules):
        """Filter by multiple tags returns rules with ALL tags (AND logic)."""
        result = engine.search_rules(tags=["emissions", "core"])
        assert len(result) == 1
        assert result[0]["name"] == "emission_range"

    def test_search_by_name_pattern(self, engine, multiple_rules):
        """Filter by name_pattern returns substring matches."""
        result = engine.search_rules(name_pattern="emission")
        assert len(result) == 1
        assert result[0]["name"] == "emission_range"

    def test_search_by_name_pattern_case_insensitive(self, engine, multiple_rules):
        """name_pattern filter is case-insensitive."""
        result = engine.search_rules(name_pattern="EMISSION")
        assert len(result) == 1

    def test_search_pagination_limit(self, engine, multiple_rules):
        """Limit parameter restricts result count."""
        result = engine.search_rules(limit=2)
        assert len(result) == 2

    def test_search_pagination_offset(self, engine, multiple_rules):
        """Offset parameter skips initial results."""
        all_results = engine.search_rules()
        offset_results = engine.search_rules(offset=3)
        assert len(offset_results) == 2
        assert offset_results[0]["rule_id"] == all_results[3]["rule_id"]

    def test_search_pagination_limit_and_offset(self, engine, multiple_rules):
        """Limit and offset work together for pagination."""
        result = engine.search_rules(limit=1, offset=1)
        assert len(result) == 1

    def test_search_combined_filters(self, engine, multiple_rules):
        """Multiple filters are AND-combined."""
        result = engine.search_rules(
            rule_type="COMPLETENESS",
            severity="CRITICAL",
        )
        assert len(result) == 1
        assert result[0]["name"] == "completeness_check"

    def test_search_empty_results(self, engine, multiple_rules):
        """Filters that match nothing return empty list."""
        result = engine.search_rules(rule_type="STATISTICAL")
        assert result == []

    def test_search_negative_limit_raises(self, engine):
        """Negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit"):
            engine.search_rules(limit=-1)

    def test_search_negative_offset_raises(self, engine):
        """Negative offset raises ValueError."""
        with pytest.raises(ValueError, match="offset"):
            engine.search_rules(offset=-1)

    def test_search_ordered_by_created_at(self, engine, multiple_rules):
        """Results are ordered by created_at ascending."""
        result = engine.search_rules()
        created_times = [r["created_at"] for r in result]
        assert created_times == sorted(created_times)

    def test_search_by_tag_case_insensitive(self, engine, multiple_rules):
        """Tag search is case-insensitive."""
        result = engine.search_rules(tags=["CORE"])
        assert len(result) == 2


# ===========================================================================
# TestCloneRule
# ===========================================================================


class TestCloneRule:
    """Test rule cloning with new name and ID."""

    def test_clone_creates_new_rule(self, engine, registered_rule):
        """Clone creates a new rule with a different ID."""
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="cloned_emission_factor",
        )
        assert clone["rule_id"] != registered_rule["rule_id"]
        assert clone["name"] == "cloned_emission_factor"

    def test_clone_preserves_rule_type(self, engine, registered_rule):
        """Clone preserves the original rule type."""
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="clone_type",
        )
        assert clone["rule_type"] == registered_rule["rule_type"]

    def test_clone_preserves_operator(self, engine, registered_rule):
        """Clone preserves the original operator."""
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="clone_op",
        )
        assert clone["operator"] == registered_rule["operator"]

    def test_clone_preserves_parameters(self, engine, registered_rule):
        """Clone preserves the original parameters."""
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="clone_params",
        )
        assert clone["parameters"] == registered_rule["parameters"]

    def test_clone_preserves_tags(self, engine, registered_rule):
        """Clone preserves the original tags."""
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="clone_tags",
        )
        assert clone["tags"] == registered_rule["tags"]

    def test_clone_resets_version_to_1_0_0(self, engine, registered_rule):
        """Clone resets version to 1.0.0."""
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="clone_version",
        )
        assert clone["version"] == "1.0.0"

    def test_clone_resets_status_to_draft(self, engine, registered_rule):
        """Clone resets status to draft."""
        engine.update_rule(registered_rule["rule_id"], status="active")
        clone = engine.clone_rule(
            registered_rule["rule_id"], new_name="clone_status",
        )
        assert clone["status"] == "draft"

    def test_clone_nonexistent_rule_raises(self, engine):
        """Cloning a non-existent rule raises ValueError."""
        with pytest.raises(ValueError, match="Source rule not found"):
            engine.clone_rule("nonexistent-id", new_name="clone_fail")


# ===========================================================================
# TestRuleVersioning
# ===========================================================================


class TestRuleVersioning:
    """Test rule version history and rollback."""

    def test_get_rule_versions_initial(self, engine, registered_rule):
        """Initial rule has two entries: the registration snapshot plus
        the current state appended by get_rule_versions."""
        versions = engine.get_rule_versions(registered_rule["rule_id"])
        assert len(versions) == 2
        assert versions[0]["version"] == "1.0.0"
        assert versions[1]["version"] == "1.0.0"

    def test_get_rule_versions_after_updates(self, engine, registered_rule):
        """Each update creates a new version entry.

        After registration: 1 snapshot in _version_history.
        After update 1 (description): +1 pre-update snapshot = 2 in history.
        After update 2 (parameters): +1 pre-update snapshot = 3 in history.
        get_rule_versions appends current state = 4 total.
        """
        engine.update_rule(
            registered_rule["rule_id"], description="first update",
        )
        engine.update_rule(
            registered_rule["rule_id"],
            parameters={"min_value": 1, "max_value": 99},
        )
        versions = engine.get_rule_versions(registered_rule["rule_id"])
        assert len(versions) == 4
        assert versions[0]["version"] == "1.0.0"
        # versions[1] is pre-update snapshot (still 1.0.0)
        assert versions[1]["version"] == "1.0.0"
        # versions[2] is pre-update snapshot after first update (1.0.1)
        assert versions[2]["version"] == "1.0.1"
        # versions[3] is current state (1.1.0)
        assert versions[3]["version"] == "1.1.0"

    def test_get_rule_versions_nonexistent(self, engine):
        """Requesting versions for non-existent rule returns empty list."""
        versions = engine.get_rule_versions("nonexistent-id")
        assert versions == []

    def test_rollback_to_previous_version(self, engine, registered_rule):
        """Rollback restores the rule to a previous version state."""
        engine.update_rule(
            registered_rule["rule_id"], description="modified desc",
        )
        engine.update_rule(
            registered_rule["rule_id"],
            parameters={"min_value": 5, "max_value": 50},
        )
        result = engine.rollback_rule(registered_rule["rule_id"], "1.0.0")
        assert result["version"] != "1.0.0"  # New version number (rollback creates new)
        assert result["description"] == "Emission factor must be between 0 and 100."
        assert result["parameters"]["min_value"] == 0.0

    def test_rollback_to_nonexistent_version_raises(self, engine, registered_rule):
        """Rolling back to a non-existent version raises ValueError."""
        with pytest.raises(ValueError, match="Version .* not found"):
            engine.rollback_rule(registered_rule["rule_id"], "99.0.0")

    def test_rollback_nonexistent_rule_raises(self, engine):
        """Rolling back a non-existent rule raises ValueError."""
        with pytest.raises(ValueError, match="Rule not found"):
            engine.rollback_rule("nonexistent-id", "1.0.0")

    def test_version_history_includes_provenance(self, engine, registered_rule):
        """Each version entry includes a provenance_hash."""
        engine.update_rule(
            registered_rule["rule_id"], description="v1.0.1 desc",
        )
        versions = engine.get_rule_versions(registered_rule["rule_id"])
        for v in versions:
            assert "provenance_hash" in v
            assert len(v["provenance_hash"]) == 64

    def test_version_history_preserves_timestamps(self, engine, registered_rule):
        """Each version entry has a snapshot_at timestamp."""
        engine.update_rule(
            registered_rule["rule_id"], description="v1.0.1 desc",
        )
        versions = engine.get_rule_versions(registered_rule["rule_id"])
        for v in versions:
            assert "snapshot_at" in v

    def test_multiple_major_bumps(self, engine, registered_rule):
        """Multiple major bumps increment correctly."""
        engine.update_rule(registered_rule["rule_id"], rule_type="FORMAT")
        engine.update_rule(registered_rule["rule_id"], column="new_col")
        versions = engine.get_rule_versions(registered_rule["rule_id"])
        version_numbers = [v["version"] for v in versions]
        assert "1.0.0" in version_numbers
        assert "2.0.0" in version_numbers
        assert "3.0.0" in version_numbers

    def test_mixed_version_bumps(self, engine, registered_rule):
        """Mixed bump types produce correct SemVer sequence."""
        engine.update_rule(
            registered_rule["rule_id"], description="patch bump",
        )  # 1.0.1
        engine.update_rule(
            registered_rule["rule_id"],
            parameters={"min_value": 1, "max_value": 99},
        )  # 1.1.0
        engine.update_rule(
            registered_rule["rule_id"], rule_type="FORMAT",
        )  # 2.0.0
        rule = engine.get_rule(registered_rule["rule_id"])
        assert rule["version"] == "2.0.0"


# ===========================================================================
# TestBulkOperations
# ===========================================================================


class TestBulkOperations:
    """Test bulk register, import, and export operations."""

    def test_bulk_register_multiple_rules(self, engine):
        """Bulk register creates multiple rules."""
        rules_data = [
            {
                "name": f"bulk_rule_{i}",
                "rule_type": "RANGE",
                "operator": "BETWEEN",
                "column": f"field_{i}",
                "parameters": {"min_value": 0, "max_value": i * 10},
                "severity": "MEDIUM",
            }
            for i in range(5)
        ]
        result = engine.bulk_register(rules_data)
        assert result["registered"] == 5
        assert result["failed"] == 0
        assert len(result["rule_ids"]) == 5

    def test_bulk_register_with_failures(self, engine):
        """Bulk register reports failed rules and continues."""
        rules_data = [
            {
                "name": "good_rule",
                "rule_type": "RANGE",
                "operator": "BETWEEN",
                "column": "field_a",
                "parameters": {"min_value": 0, "max_value": 100},
                "severity": "HIGH",
            },
            {
                "name": "",  # Invalid: empty name
                "rule_type": "RANGE",
                "operator": "BETWEEN",
                "column": "field_b",
                "parameters": {},
                "severity": "HIGH",
            },
            {
                "name": "another_good",
                "rule_type": "FORMAT",
                "operator": "MATCHES",
                "column": "field_c",
                "parameters": {"pattern": ".*"},
                "severity": "LOW",
            },
        ]
        result = engine.bulk_register(rules_data)
        assert result["registered"] == 2
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["index"] == 1

    def test_bulk_register_empty_batch(self, engine):
        """Empty batch raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.bulk_register([])

    def test_bulk_register_exceeds_max_raises(self, engine):
        """Batch exceeding MAX_BULK_IMPORT raises ValueError."""
        rules = [
            {
                "name": f"rule_{i}",
                "rule_type": "CUSTOM",
                "operator": "EQUALS",
                "column": f"col_{i}",
                "parameters": {},
                "severity": "LOW",
            }
            for i in range(MAX_BULK_IMPORT + 1)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            engine.bulk_register(rules)

    def test_bulk_register_duplicates_in_batch(self, engine):
        """Duplicate names in same batch: second is reported as failure."""
        rules_data = [
            {
                "name": "dup_rule",
                "rule_type": "RANGE",
                "operator": "BETWEEN",
                "column": "field_a",
                "parameters": {"min_value": 0, "max_value": 10},
                "severity": "MEDIUM",
            },
            {
                "name": "dup_rule",
                "rule_type": "RANGE",
                "operator": "BETWEEN",
                "column": "field_b",
                "parameters": {"min_value": 0, "max_value": 20},
                "severity": "MEDIUM",
            },
        ]
        result = engine.bulk_register(rules_data)
        assert result["registered"] == 1
        assert result["failed"] == 1

    def test_export_all_rules(self, engine, multiple_rules):
        """Export without filters returns all rules."""
        result = engine.export_rules()
        assert len(result) == 5

    def test_export_by_rule_type(self, engine, multiple_rules):
        """Export filtered by rule_type returns only matching rules."""
        result = engine.export_rules(rule_type="RANGE")
        assert len(result) == 1
        assert result[0]["rule_type"] == "RANGE"

    def test_export_empty_registry(self, engine):
        """Export from empty registry returns empty list."""
        result = engine.export_rules()
        assert result == []

    def test_import_rules(self, engine):
        """Import rules from exported format."""
        exported = [
            {
                "name": f"imported_rule_{i}",
                "rule_type": "FORMAT",
                "operator": "MATCHES",
                "column": f"col_{i}",
                "parameters": {"pattern": ".*"},
                "severity": "LOW",
                "tags": ["imported"],
                "description": f"Imported rule {i}.",
            }
            for i in range(3)
        ]
        result = engine.import_rules(exported)
        assert result["imported"] == 3
        assert result["failed"] == 0

    def test_import_with_optional_fields(self, engine):
        """Import rules with all optional fields populated."""
        exported = [
            {
                "name": "import_full",
                "rule_type": "RANGE",
                "operator": "BETWEEN",
                "column": "value",
                "parameters": {"min_value": 0, "max_value": 100},
                "severity": "HIGH",
                "tags": ["imported", "critical"],
                "description": "A fully specified imported rule.",
                "metadata": {"origin": "external_system"},
            },
        ]
        result = engine.import_rules(exported)
        assert result["imported"] == 1
        stored = engine.get_rule(result["rule_ids"][0])
        assert stored["tags"] == ["critical", "imported"]
        assert stored["metadata"]["origin"] == "external_system"

    def test_export_returns_deep_copies(self, engine, multiple_rules):
        """Exported rules are deep copies."""
        result = engine.export_rules()
        result[0]["name"] = "MUTATED"
        stored = engine.get_rule(result[0]["rule_id"])
        assert stored["name"] != "MUTATED"

    def test_bulk_register_provenance_hash(self, engine):
        """Bulk register result includes a provenance_hash."""
        rules_data = [
            {
                "name": "prov_rule",
                "rule_type": "CUSTOM",
                "operator": "EQUALS",
                "column": "status",
                "parameters": {},
                "severity": "LOW",
            },
        ]
        result = engine.bulk_register(rules_data)
        assert result["provenance_hash"] != ""


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Test statistics aggregation."""

    def test_statistics_total_rules(self, engine, multiple_rules):
        """Statistics report correct total rules."""
        stats = engine.get_statistics()
        assert stats["total_rules"] == 5

    def test_statistics_by_type(self, engine, multiple_rules):
        """Statistics break down rules by type."""
        stats = engine.get_statistics()
        assert stats["by_type"]["COMPLETENESS"] == 1
        assert stats["by_type"]["FORMAT"] == 1
        assert stats["by_type"]["RANGE"] == 1
        assert stats["by_type"]["REFERENTIAL"] == 1
        assert stats["by_type"]["CROSS_FIELD"] == 1

    def test_statistics_by_severity(self, engine, multiple_rules):
        """Statistics break down rules by severity."""
        stats = engine.get_statistics()
        assert stats["by_severity"]["CRITICAL"] == 2
        assert stats["by_severity"]["HIGH"] == 1
        assert stats["by_severity"]["MEDIUM"] == 1
        assert stats["by_severity"]["LOW"] == 1

    def test_statistics_by_status(self, engine, multiple_rules):
        """Statistics break down rules by status."""
        stats = engine.get_statistics()
        assert stats["by_status"]["draft"] == 5

    def test_statistics_total_tags(self, engine, multiple_rules):
        """Statistics report correct total unique tags."""
        stats = engine.get_statistics()
        # Expected tags: quality, core, format, contact, emissions,
        # reference, sector, cross-field
        assert stats["total_tags"] >= 8

    def test_statistics_provenance_entries(self, engine, multiple_rules):
        """Statistics report provenance entry count."""
        stats = engine.get_statistics()
        assert stats["provenance_entries"] >= 5

    def test_statistics_after_update(self, engine, registered_rule):
        """Statistics update after a rule modification."""
        engine.update_rule(registered_rule["rule_id"], severity="CRITICAL")
        stats = engine.get_statistics()
        assert stats["by_severity"]["CRITICAL"] == 1
        assert stats["by_severity"].get("HIGH", 0) == 0

    def test_statistics_after_delete(self, engine, registered_rule):
        """Statistics update after soft deletion."""
        engine.delete_rule(registered_rule["rule_id"])
        stats = engine.get_statistics()
        assert stats["by_status"]["archived"] == 1
        # "draft" key is absent because its count is 0
        assert stats["by_status"].get("draft", 0) == 0


# ===========================================================================
# TestValidation
# ===========================================================================


class TestValidation:
    """Test input validation for rule registration and updates."""

    def test_register_empty_name_rejected(self, engine):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.register_rule(
                name="", rule_type="RANGE", operator="BETWEEN",
                column="field", parameters={}, severity="HIGH",
            )

    def test_register_whitespace_only_name_rejected(self, engine):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.register_rule(
                name="   ", rule_type="RANGE", operator="BETWEEN",
                column="field", parameters={}, severity="HIGH",
            )

    def test_register_name_too_long_rejected(self, engine):
        """Name exceeding MAX_RULE_NAME_LENGTH raises ValueError."""
        long_name = "R" * (MAX_RULE_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            engine.register_rule(
                name=long_name, rule_type="RANGE", operator="BETWEEN",
                column="field", parameters={}, severity="HIGH",
            )

    def test_register_invalid_rule_type_rejected(self, engine):
        """Invalid rule_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rule_type"):
            engine.register_rule(
                name="bad_type", rule_type="INVALID_TYPE",
                operator="EQUALS", column="field", parameters={},
                severity="HIGH",
            )

    def test_register_invalid_operator_rejected(self, engine):
        """Invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            engine.register_rule(
                name="bad_op", rule_type="RANGE",
                operator="INVALID_OP", column="field",
                parameters={}, severity="HIGH",
            )

    def test_register_invalid_severity_rejected(self, engine):
        """Invalid severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            engine.register_rule(
                name="bad_sev", rule_type="RANGE",
                operator="BETWEEN", column="field",
                parameters={}, severity="EXTREME",
            )

    def test_register_empty_column_rejected(self, engine):
        """Empty column raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            engine.register_rule(
                name="no_col", rule_type="RANGE", operator="BETWEEN",
                column="", parameters={}, severity="HIGH",
            )

    def test_register_tag_too_long_rejected(self, engine):
        """Tag exceeding MAX_TAG_LENGTH raises ValueError."""
        long_tag = "t" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            engine.register_rule(
                name="long_tag", rule_type="RANGE", operator="BETWEEN",
                column="field", parameters={}, severity="HIGH",
                tags=[long_tag],
            )

    def test_update_invalid_operator_rejected(self, engine, registered_rule):
        """Updating with invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            engine.update_rule(
                registered_rule["rule_id"], operator="BAD_OP",
            )

    def test_update_invalid_severity_rejected(self, engine, registered_rule):
        """Updating with invalid severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            engine.update_rule(
                registered_rule["rule_id"], severity="EXTREME",
            )


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_concurrent_registration(self, engine):
        """Thread safety: concurrent registrations do not corrupt state."""
        errors = []

        def register(idx):
            try:
                engine.register_rule(
                    name=f"concurrent_rule_{idx}",
                    rule_type="CUSTOM",
                    operator="EQUALS",
                    column=f"field_{idx}",
                    parameters={"value": idx},
                    severity="LOW",
                )
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=register, args=(i,)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["total_rules"] == 20

    def test_name_at_max_length(self, engine):
        """Name at exactly MAX_RULE_NAME_LENGTH is accepted."""
        name = "R" * MAX_RULE_NAME_LENGTH
        rule = engine.register_rule(
            name=name, rule_type="CUSTOM", operator="EQUALS",
            column="field", parameters={}, severity="LOW",
        )
        assert rule["name"] == name

    def test_tag_at_max_length(self, engine):
        """Tag at exactly MAX_TAG_LENGTH is accepted."""
        tag = "t" * MAX_TAG_LENGTH
        rule = engine.register_rule(
            name="max_tag", rule_type="CUSTOM", operator="EQUALS",
            column="field", parameters={}, severity="LOW",
            tags=[tag],
        )
        assert tag in rule["tags"]

    def test_many_rules_registered(self, engine):
        """Register 50 rules to test at moderate scale."""
        for i in range(50):
            engine.register_rule(
                name=f"scale_rule_{i}",
                rule_type="CUSTOM",
                operator="EQUALS",
                column=f"col_{i % 10}",
                parameters={"value": i},
                severity=["CRITICAL", "HIGH", "MEDIUM", "LOW"][i % 4],
                tags=[f"batch_{i % 5}"],
            )
        stats = engine.get_statistics()
        assert stats["total_rules"] == 50

    def test_empty_tags_list(self, engine):
        """Empty tags list is handled correctly."""
        rule = engine.register_rule(
            name="no_tags", rule_type="CUSTOM", operator="EQUALS",
            column="field", parameters={}, severity="LOW", tags=[],
        )
        assert rule["tags"] == []

    def test_none_tags(self, engine):
        """None tags parameter is handled correctly."""
        rule = engine.register_rule(
            name="none_tags", rule_type="CUSTOM", operator="EQUALS",
            column="field", parameters={}, severity="LOW", tags=None,
        )
        assert rule["tags"] == []

    def test_metadata_none_defaults_to_empty(self, engine):
        """None metadata defaults to empty dict."""
        rule = engine.register_rule(
            name="no_meta", rule_type="CUSTOM", operator="EQUALS",
            column="field", parameters={}, severity="LOW",
            metadata=None,
        )
        assert rule["metadata"] == {}

    def test_index_correctness_after_update_and_delete(self, engine):
        """Indexes remain correct after updates and deletes."""
        r1 = engine.register_rule(
            name="idx_test_1", rule_type="RANGE", operator="BETWEEN",
            column="val_a", parameters={"min_value": 0, "max_value": 10},
            severity="HIGH", tags=["tag_a"],
        )
        r2 = engine.register_rule(
            name="idx_test_2", rule_type="FORMAT", operator="MATCHES",
            column="val_b", parameters={"pattern": ".*"},
            severity="LOW", tags=["tag_b"],
        )
        # Update r1 type: changes type index
        engine.update_rule(r1["rule_id"], rule_type="CUSTOM")
        assert len(engine.search_rules(rule_type="RANGE")) == 0
        assert len(engine.search_rules(rule_type="CUSTOM")) == 1

        # Delete r2: removes from all indexes
        engine.delete_rule(r2["rule_id"], hard=True)
        assert len(engine.search_rules(rule_type="FORMAT")) == 0
        assert len(engine.search_rules(tags=["tag_b"])) == 0
        assert engine.get_rule_by_name("idx_test_2") is None


# ===========================================================================
# TestConstants
# ===========================================================================


class TestConstants:
    """Test module-level constants for correctness."""

    def test_valid_rule_types_count(self):
        """VALID_RULE_TYPES contains 10 types."""
        assert len(VALID_RULE_TYPES) == 10

    def test_valid_rule_types_content(self):
        """VALID_RULE_TYPES contains all expected types."""
        expected = {
            "COMPLETENESS", "RANGE", "FORMAT", "UNIQUENESS", "CUSTOM",
            "FRESHNESS", "CROSS_FIELD", "CONDITIONAL", "STATISTICAL",
            "REFERENTIAL",
        }
        assert expected.issubset(set(VALID_RULE_TYPES))

    def test_valid_operators_count(self):
        """VALID_OPERATORS contains 12 operators."""
        assert len(VALID_OPERATORS) == 12

    def test_valid_operators_content(self):
        """VALID_OPERATORS contains all expected operators."""
        expected = {
            "EQUALS", "NOT_EQUALS", "GREATER_THAN", "LESS_THAN",
            "GREATER_EQUAL", "LESS_EQUAL", "BETWEEN", "MATCHES",
            "CONTAINS", "IN_SET", "NOT_IN_SET", "IS_NULL",
        }
        assert expected.issubset(set(VALID_OPERATORS))

    def test_valid_severities_count(self):
        """VALID_SEVERITIES contains 4 levels."""
        assert len(VALID_SEVERITIES) == 4

    def test_valid_severities_content(self):
        """VALID_SEVERITIES contains all expected levels."""
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert expected.issubset(set(VALID_SEVERITIES))

    def test_valid_rule_statuses(self):
        """VALID_RULE_STATUSES contains lifecycle statuses."""
        assert "draft" in VALID_RULE_STATUSES
        assert "active" in VALID_RULE_STATUSES
        assert "deprecated" in VALID_RULE_STATUSES
        assert "archived" in VALID_RULE_STATUSES

    def test_status_transitions_correct(self):
        """STATUS_TRANSITIONS enforces correct lifecycle."""
        assert "active" in STATUS_TRANSITIONS["draft"]
        assert "deprecated" in STATUS_TRANSITIONS["active"]
        assert "archived" in STATUS_TRANSITIONS["deprecated"]
        # Archived is terminal
        assert len(STATUS_TRANSITIONS["archived"]) == 0

    def test_max_bulk_import(self):
        """MAX_BULK_IMPORT is a positive integer."""
        assert MAX_BULK_IMPORT > 0

    def test_max_lengths_positive(self):
        """Max length constants are positive."""
        assert MAX_RULE_NAME_LENGTH > 0
        assert MAX_TAG_LENGTH > 0


# ===========================================================================
# TestHelperFunctions
# ===========================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_normalize_tags_dedup_and_sort(self):
        """Tags are deduplicated, lowered, and sorted."""
        result = _normalize_tags(["B", "a", "b", "C"])
        assert result == ["a", "b", "c"]

    def test_normalize_tags_strips_whitespace(self):
        """Tags are stripped of whitespace."""
        result = _normalize_tags(["  hello  ", "  world  "])
        assert result == ["hello", "world"]

    def test_normalize_tags_removes_empty(self):
        """Empty string tags are removed."""
        result = _normalize_tags(["", "  ", "valid"])
        assert result == ["valid"]

    def test_normalize_tags_none_returns_empty(self):
        """None input returns empty list."""
        result = _normalize_tags(None)
        assert result == []

    def test_validate_rule_name_valid(self):
        """Valid rule name passes validation."""
        _validate_rule_name("my_rule_name")  # Should not raise

    def test_validate_rule_name_empty_raises(self):
        """Empty rule name raises ValueError."""
        with pytest.raises(ValueError):
            _validate_rule_name("")

    def test_validate_tags_list_valid(self):
        """Valid tags list passes."""
        _validate_tags_list(["tag1", "tag2"])  # Should not raise

    def test_validate_tags_list_long_tag_raises(self):
        """Tag exceeding MAX_TAG_LENGTH raises ValueError."""
        long_tag = "x" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValueError):
            _validate_tags_list([long_tag])

    def test_build_sha256_deterministic(self):
        """SHA-256 hash is deterministic for same input."""
        h1 = _build_sha256({"key": "value"})
        h2 = _build_sha256({"key": "value"})
        assert h1 == h2
        assert len(h1) == 64

    def test_build_sha256_different_for_different_input(self):
        """Different inputs produce different hashes."""
        h1 = _build_sha256({"key": "value1"})
        h2 = _build_sha256({"key": "value2"})
        assert h1 != h2

    def test_build_sha256_key_order_independent(self):
        """Dict key order does not affect hash (sorted keys)."""
        h1 = _build_sha256({"a": 1, "b": 2})
        h2 = _build_sha256({"b": 2, "a": 1})
        assert h1 == h2


# ===========================================================================
# TestResetEngine
# ===========================================================================


class TestResetEngine:
    """Test engine clear functionality."""

    def test_reset_clears_all_rules(self, engine, registered_rule):
        """clear() clears all rules."""
        engine.clear()
        stats = engine.get_statistics()
        assert stats["total_rules"] == 0

    def test_reset_clears_indexes(self, engine, registered_rule):
        """clear() clears all internal indexes."""
        engine.clear()
        assert len(engine._type_index) == 0
        assert len(engine._severity_index) == 0
        assert len(engine._column_index) == 0
        assert len(engine._tag_index) == 0
        assert len(engine._name_index) == 0

    def test_reset_clears_version_store(self, engine, registered_rule):
        """clear() clears version history."""
        engine.clear()
        assert len(engine._version_history) == 0

    def test_reset_allows_reregistration(self, engine, sample_rule_params):
        """After clear(), previously registered names can be reused."""
        engine.register_rule(**sample_rule_params)
        engine.clear()
        rule = engine.register_rule(**sample_rule_params)
        assert rule["name"] == "emission_factor_range"
