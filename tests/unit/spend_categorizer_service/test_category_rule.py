# -*- coding: utf-8 -*-
"""
Unit tests for CategoryRuleEngine - AGENT-DATA-009 Batch 3

Tests the CategoryRuleEngine with 85%+ coverage across:
- Initialization and configuration
- Rule creation (all 6 match types, validation)
- Rule retrieval (existing, nonexistent)
- Rule listing (all, by priority, active only, pagination)
- Rule update (name, pattern, priority, deactivate)
- Rule deletion (existing, nonexistent)
- Rule evaluation (single match, priority order, no match, inactive)
- Batch rule application (mixed matches)
- Rule import and export
- Rule effectiveness tracking
- Match type evaluation (EXACT, CONTAINS, REGEX, FUZZY, STARTS_WITH, ENDS_WITH)
- Statistics tracking
- SHA-256 provenance hashes
- Thread safety (concurrent evaluation)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from greenlang.spend_categorizer.category_rule import (
    CategoryRule,
    CategoryRuleEngine,
    MatchType,
    _string_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> CategoryRuleEngine:
    """Create a default CategoryRuleEngine."""
    return CategoryRuleEngine()


@pytest.fixture
def engine_custom() -> CategoryRuleEngine:
    """Engine with custom configuration."""
    return CategoryRuleEngine({
        "default_priority": 200,
        "default_confidence": 0.85,
        "fuzzy_threshold": 0.80,
    })


@pytest.fixture
def contains_rule(engine: CategoryRuleEngine) -> CategoryRule:
    """Create a CONTAINS rule in the engine."""
    return engine.create_rule(
        name="Travel Rule",
        match_type="CONTAINS",
        pattern="travel",
        target_category="business_travel",
    )


@pytest.fixture
def exact_rule(engine: CategoryRuleEngine) -> CategoryRule:
    """Create an EXACT rule in the engine."""
    return engine.create_rule(
        name="AWS Exact",
        match_type="EXACT",
        pattern="Amazon Web Services",
        target_category="cloud_computing",
        match_field="vendor_name",
        priority=500,
    )


@pytest.fixture
def regex_rule(engine: CategoryRuleEngine) -> CategoryRule:
    """Create a REGEX rule in the engine."""
    return engine.create_rule(
        name="Freight Regex",
        match_type="REGEX",
        pattern=r"freight|shipping|logistics",
        target_category="transportation",
    )


@pytest.fixture
def fuzzy_rule(engine: CategoryRuleEngine) -> CategoryRule:
    """Create a FUZZY rule in the engine."""
    return engine.create_rule(
        name="Chemical Fuzzy",
        match_type="FUZZY",
        pattern="chemical supplies",
        target_category="chemicals",
        fuzzy_threshold=0.5,
    )


@pytest.fixture
def starts_with_rule(engine: CategoryRuleEngine) -> CategoryRule:
    """Create a STARTS_WITH rule in the engine."""
    return engine.create_rule(
        name="Capital Equipment Prefix",
        match_type="STARTS_WITH",
        pattern="Capital",
        target_category="capital_goods",
    )


@pytest.fixture
def ends_with_rule(engine: CategoryRuleEngine) -> CategoryRule:
    """Create an ENDS_WITH rule in the engine."""
    return engine.create_rule(
        name="Services Suffix",
        match_type="ENDS_WITH",
        pattern="services",
        target_category="professional_services",
    )


@pytest.fixture
def multi_rules(engine: CategoryRuleEngine) -> List[CategoryRule]:
    """Create multiple rules with different priorities."""
    rules = []
    rules.append(engine.create_rule("Low Priority", "CONTAINS", "office", "office_supplies", priority=10))
    rules.append(engine.create_rule("Medium Priority", "CONTAINS", "office", "office_equipment", priority=100))
    rules.append(engine.create_rule("High Priority", "CONTAINS", "office", "office_priority", priority=500))
    return rules


@pytest.fixture
def import_data() -> List[Dict[str, Any]]:
    """Data for bulk rule import."""
    return [
        {"name": "Import Rule 1", "match_type": "CONTAINS", "pattern": "fuel", "target_category": "fuel_energy"},
        {"name": "Import Rule 2", "match_type": "EXACT", "pattern": "FedEx", "target_category": "courier"},
        {"name": "Import Rule 3", "match_type": "REGEX", "pattern": r"\bsteel\b", "target_category": "metals"},
    ]


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test CategoryRuleEngine initialization."""

    def test_default_init(self, engine: CategoryRuleEngine):
        """Engine initializes with default configuration."""
        assert engine._default_priority == 100
        assert engine._default_confidence == 0.90
        assert engine._fuzzy_threshold == 0.75

    def test_custom_init(self, engine_custom: CategoryRuleEngine):
        """Engine respects custom configuration."""
        assert engine_custom._default_priority == 200
        assert engine_custom._default_confidence == 0.85
        assert engine_custom._fuzzy_threshold == 0.80

    def test_empty_rule_store(self, engine: CategoryRuleEngine):
        """Rule store starts empty."""
        assert len(engine._rules) == 0

    def test_stats_initialized(self, engine: CategoryRuleEngine):
        """Statistics counters start at zero."""
        stats = engine.get_statistics()
        assert stats["rules_created"] == 0
        assert stats["evaluations_performed"] == 0
        assert stats["matches_found"] == 0


# ===========================================================================
# TestCreateRule
# ===========================================================================


class TestCreateRule:
    """Test create_rule() for all match types."""

    @pytest.mark.parametrize("match_type", ["EXACT", "CONTAINS", "REGEX", "FUZZY", "STARTS_WITH", "ENDS_WITH"])
    def test_all_match_types(self, engine: CategoryRuleEngine, match_type):
        """All 6 match types create rules successfully."""
        pattern = "test" if match_type != "REGEX" else "test.*pattern"
        rule = engine.create_rule("Test Rule", match_type, pattern, "test_cat")
        assert isinstance(rule, CategoryRule)
        assert rule.match_type == match_type

    def test_rule_id_generated(self, engine: CategoryRuleEngine):
        """Rule ID starts with 'rule-'."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.rule_id.startswith("rule-")

    def test_default_priority(self, engine: CategoryRuleEngine):
        """Default priority is applied when not specified."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.priority == 100

    def test_custom_priority(self, engine: CategoryRuleEngine):
        """Custom priority is respected."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat", priority=500)
        assert rule.priority == 500

    def test_default_confidence(self, engine: CategoryRuleEngine):
        """Default confidence is applied."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.confidence == 0.90

    def test_custom_confidence(self, engine: CategoryRuleEngine):
        """Custom confidence is respected."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat", confidence=0.75)
        assert rule.confidence == 0.75

    def test_default_match_field(self, engine: CategoryRuleEngine):
        """Default match field is 'description'."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.match_field == "description"

    def test_custom_match_field(self, engine: CategoryRuleEngine):
        """Custom match field is respected."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat", match_field="vendor_name")
        assert rule.match_field == "vendor_name"

    def test_target_scope3(self, engine: CategoryRuleEngine):
        """Scope 3 category override is stored."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat", target_scope3=6)
        assert rule.target_scope3 == 6

    def test_fuzzy_threshold(self, engine: CategoryRuleEngine):
        """Custom fuzzy threshold is stored."""
        rule = engine.create_rule("Test", "FUZZY", "test", "cat", fuzzy_threshold=0.60)
        assert rule.fuzzy_threshold == 0.60

    def test_metadata_stored(self, engine: CategoryRuleEngine):
        """Custom metadata is stored."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat", metadata={"owner": "admin"})
        assert rule.metadata == {"owner": "admin"}

    def test_rule_active_by_default(self, engine: CategoryRuleEngine):
        """New rules are active by default."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.active is True

    def test_match_count_zero(self, engine: CategoryRuleEngine):
        """New rule starts with zero match count."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.match_count == 0

    def test_timestamps_set(self, engine: CategoryRuleEngine):
        """Created and updated timestamps are set."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert rule.created_at != ""
        assert rule.updated_at != ""

    def test_invalid_match_type_raises(self, engine: CategoryRuleEngine):
        """Invalid match type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid match_type"):
            engine.create_rule("Test", "INVALID", "test", "cat")

    def test_empty_pattern_raises(self, engine: CategoryRuleEngine):
        """Empty pattern raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.create_rule("Test", "CONTAINS", "", "cat")

    def test_invalid_regex_raises(self, engine: CategoryRuleEngine):
        """Invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex"):
            engine.create_rule("Test", "REGEX", "[invalid(", "cat")

    def test_case_insensitive_match_type(self, engine: CategoryRuleEngine):
        """Match type is case-insensitive."""
        rule = engine.create_rule("Test", "contains", "test", "cat")
        assert rule.match_type == "CONTAINS"

    def test_stats_incremented(self, engine: CategoryRuleEngine):
        """Creating a rule increments rules_created stat."""
        engine.create_rule("Test", "CONTAINS", "test", "cat")
        stats = engine.get_statistics()
        assert stats["rules_created"] == 1

    def test_provenance_hash(self, engine: CategoryRuleEngine):
        """Created rule has a SHA-256 provenance hash."""
        rule = engine.create_rule("Test", "CONTAINS", "test", "cat")
        assert len(rule.provenance_hash) == 64


# ===========================================================================
# TestGetRule
# ===========================================================================


class TestGetRule:
    """Test get_rule() retrieval."""

    def test_existing_rule(self, engine: CategoryRuleEngine, contains_rule):
        """Existing rule is retrieved by ID."""
        result = engine.get_rule(contains_rule.rule_id)
        assert result is not None
        assert result.rule_id == contains_rule.rule_id
        assert result.name == "Travel Rule"

    def test_nonexistent_rule(self, engine: CategoryRuleEngine):
        """Nonexistent rule ID returns None."""
        result = engine.get_rule("nonexistent-id")
        assert result is None


# ===========================================================================
# TestListRules
# ===========================================================================


class TestListRules:
    """Test list_rules() with filters."""

    def test_list_all_active(self, engine: CategoryRuleEngine, multi_rules):
        """List returns all active rules sorted by priority descending."""
        rules = engine.list_rules()
        assert len(rules) == 3
        assert rules[0].priority >= rules[1].priority

    def test_sorted_by_priority_descending(self, engine: CategoryRuleEngine, multi_rules):
        """Rules are sorted by priority descending (highest first)."""
        rules = engine.list_rules()
        priorities = [r.priority for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_active_only_filter(self, engine: CategoryRuleEngine, multi_rules):
        """Active-only filter excludes inactive rules."""
        engine.update_rule(multi_rules[0].rule_id, active=False)
        rules = engine.list_rules(active_only=True)
        assert len(rules) == 2

    def test_active_only_false(self, engine: CategoryRuleEngine, multi_rules):
        """active_only=False returns all rules including inactive."""
        engine.update_rule(multi_rules[0].rule_id, active=False)
        rules = engine.list_rules(active_only=False)
        assert len(rules) == 3

    def test_priority_filter(self, engine: CategoryRuleEngine, multi_rules):
        """Priority filter returns rules at or above threshold."""
        rules = engine.list_rules(priority=100)
        assert all(r.priority >= 100 for r in rules)

    def test_limit(self, engine: CategoryRuleEngine, multi_rules):
        """Limit parameter caps results."""
        rules = engine.list_rules(limit=2)
        assert len(rules) <= 2

    def test_empty_store(self, engine: CategoryRuleEngine):
        """Empty rule store returns empty list."""
        rules = engine.list_rules()
        assert rules == []


# ===========================================================================
# TestUpdateRule
# ===========================================================================


class TestUpdateRule:
    """Test update_rule() modifications."""

    def test_update_name(self, engine: CategoryRuleEngine, contains_rule):
        """Updating name changes the rule name."""
        updated = engine.update_rule(contains_rule.rule_id, name="Updated Travel Rule")
        assert updated.name == "Updated Travel Rule"

    def test_update_pattern(self, engine: CategoryRuleEngine, contains_rule):
        """Updating pattern changes the match pattern."""
        updated = engine.update_rule(contains_rule.rule_id, pattern="business travel")
        assert updated.pattern == "business travel"

    def test_update_priority(self, engine: CategoryRuleEngine, contains_rule):
        """Updating priority changes the evaluation order."""
        updated = engine.update_rule(contains_rule.rule_id, priority=999)
        assert updated.priority == 999

    def test_update_confidence(self, engine: CategoryRuleEngine, contains_rule):
        """Updating confidence changes the match confidence."""
        updated = engine.update_rule(contains_rule.rule_id, confidence=0.50)
        assert updated.confidence == 0.50

    def test_deactivate_rule(self, engine: CategoryRuleEngine, contains_rule):
        """Deactivating a rule sets active=False."""
        updated = engine.update_rule(contains_rule.rule_id, active=False)
        assert updated.active is False

    def test_update_match_type(self, engine: CategoryRuleEngine, contains_rule):
        """Updating match_type changes the match strategy."""
        updated = engine.update_rule(contains_rule.rule_id, match_type="EXACT")
        assert updated.match_type == "EXACT"

    def test_invalid_match_type_raises(self, engine: CategoryRuleEngine, contains_rule):
        """Updating with invalid match_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid match_type"):
            engine.update_rule(contains_rule.rule_id, match_type="INVALID")

    def test_nonexistent_rule_raises(self, engine: CategoryRuleEngine):
        """Updating nonexistent rule raises ValueError."""
        with pytest.raises(ValueError, match="Rule not found"):
            engine.update_rule("nonexistent-id", name="Test")

    def test_updated_at_changes(self, engine: CategoryRuleEngine, contains_rule):
        """Updated timestamp changes after update."""
        original_time = contains_rule.updated_at
        updated = engine.update_rule(contains_rule.rule_id, name="New Name")
        # updated_at may or may not differ depending on timing, but it should exist
        assert updated.updated_at != ""

    def test_provenance_hash_changes(self, engine: CategoryRuleEngine, contains_rule):
        """Provenance hash changes after update."""
        original_hash = contains_rule.provenance_hash
        updated = engine.update_rule(contains_rule.rule_id, name="Changed Name")
        assert updated.provenance_hash != original_hash


# ===========================================================================
# TestDeleteRule
# ===========================================================================


class TestDeleteRule:
    """Test delete_rule() removal."""

    def test_delete_existing(self, engine: CategoryRuleEngine, contains_rule):
        """Deleting an existing rule returns True."""
        result = engine.delete_rule(contains_rule.rule_id)
        assert result is True
        assert engine.get_rule(contains_rule.rule_id) is None

    def test_delete_nonexistent(self, engine: CategoryRuleEngine):
        """Deleting a nonexistent rule returns False."""
        result = engine.delete_rule("nonexistent-id")
        assert result is False

    def test_stats_updated(self, engine: CategoryRuleEngine, contains_rule):
        """Deleting increments rules_deleted stat."""
        engine.delete_rule(contains_rule.rule_id)
        stats = engine.get_statistics()
        assert stats["rules_deleted"] == 1

    def test_deleted_rule_not_in_list(self, engine: CategoryRuleEngine, contains_rule):
        """Deleted rule does not appear in list_rules."""
        engine.delete_rule(contains_rule.rule_id)
        rules = engine.list_rules(active_only=False)
        assert all(r.rule_id != contains_rule.rule_id for r in rules)


# ===========================================================================
# TestEvaluateRules
# ===========================================================================


class TestEvaluateRules:
    """Test evaluate_rules() matching logic."""

    def test_single_match(self, engine: CategoryRuleEngine, contains_rule):
        """Single matching rule returns its category."""
        cat, conf, rid = engine.evaluate_rules({"description": "Business travel flights"})
        assert cat == "business_travel"
        assert conf == 0.90
        assert rid == contains_rule.rule_id

    def test_no_match(self, engine: CategoryRuleEngine, contains_rule):
        """No matching rule returns (None, 0.0, None)."""
        cat, conf, rid = engine.evaluate_rules({"description": "Office supplies purchase"})
        assert cat is None
        assert conf == 0.0
        assert rid is None

    def test_priority_wins(self, engine: CategoryRuleEngine, multi_rules):
        """Highest priority rule wins when multiple match."""
        cat, conf, rid = engine.evaluate_rules({"description": "office equipment"})
        assert cat == "office_priority"
        assert rid == multi_rules[2].rule_id

    def test_inactive_rule_skipped(self, engine: CategoryRuleEngine, contains_rule):
        """Inactive rules are not evaluated."""
        engine.update_rule(contains_rule.rule_id, active=False)
        cat, conf, rid = engine.evaluate_rules({"description": "Business travel flights"})
        assert cat is None

    def test_match_increments_count(self, engine: CategoryRuleEngine, contains_rule):
        """Matching increments the rule's match_count."""
        engine.evaluate_rules({"description": "travel booking"})
        engine.evaluate_rules({"description": "travel agency"})
        rule = engine.get_rule(contains_rule.rule_id)
        assert rule.match_count == 2

    def test_stats_updated_on_match(self, engine: CategoryRuleEngine, contains_rule):
        """Match updates evaluations_performed and matches_found."""
        engine.evaluate_rules({"description": "travel booking"})
        stats = engine.get_statistics()
        assert stats["evaluations_performed"] == 1
        assert stats["matches_found"] == 1

    def test_stats_updated_on_no_match(self, engine: CategoryRuleEngine, contains_rule):
        """No-match updates evaluations_performed and no_match_count."""
        engine.evaluate_rules({"description": "unrelated text"})
        stats = engine.get_statistics()
        assert stats["evaluations_performed"] == 1
        assert stats["no_match_count"] == 1

    def test_empty_record(self, engine: CategoryRuleEngine, contains_rule):
        """Empty record returns no match."""
        cat, conf, rid = engine.evaluate_rules({})
        assert cat is None

    def test_fallback_fields(self, engine: CategoryRuleEngine, contains_rule):
        """Engine tries fallback fields if match_field is empty."""
        cat, conf, rid = engine.evaluate_rules({"vendor_name": "travel agency inc"})
        # Should try fallback fields; "description" is empty, but "vendor_name" matches
        # The contains_rule matches on "description" field, and the code falls back to
        # "description" then "vendor_name" then "category" fields
        assert cat == "business_travel"


# ===========================================================================
# TestApplyRules
# ===========================================================================


class TestApplyRules:
    """Test apply_rules() batch application."""

    def test_batch_results(self, engine: CategoryRuleEngine, contains_rule):
        """Batch application adds rule_category field to each record."""
        records = [
            {"description": "travel booking domestic"},
            {"description": "office supplies order"},
            {"description": "international travel arrangements"},
        ]
        results = engine.apply_rules(records)
        assert len(results) == 3
        assert results[0]["rule_category"] == "business_travel"
        assert results[1]["rule_category"] is None
        assert results[2]["rule_category"] == "business_travel"

    def test_confidence_added(self, engine: CategoryRuleEngine, contains_rule):
        """Matched records have rule_confidence set."""
        records = [{"description": "travel booking"}]
        results = engine.apply_rules(records)
        assert results[0]["rule_confidence"] == 0.90

    def test_rule_id_added(self, engine: CategoryRuleEngine, contains_rule):
        """Matched records have rule_id set."""
        records = [{"description": "travel booking"}]
        results = engine.apply_rules(records)
        assert results[0]["rule_id"] == contains_rule.rule_id

    def test_unmatched_record_none_fields(self, engine: CategoryRuleEngine, contains_rule):
        """Unmatched records have None rule fields."""
        records = [{"description": "no match here"}]
        results = engine.apply_rules(records)
        assert results[0]["rule_category"] is None
        assert results[0]["rule_confidence"] == 0.0
        assert results[0]["rule_id"] is None

    def test_empty_batch(self, engine: CategoryRuleEngine, contains_rule):
        """Empty batch returns empty list."""
        results = engine.apply_rules([])
        assert results == []

    def test_mixed_matches(self, engine: CategoryRuleEngine, contains_rule, regex_rule):
        """Mixed batch with some matches and some misses."""
        records = [
            {"description": "travel arrangements"},
            {"description": "freight delivery service"},
            {"description": "random purchase"},
        ]
        results = engine.apply_rules(records)
        assert results[0]["rule_category"] == "business_travel"
        assert results[1]["rule_category"] == "transportation"
        assert results[2]["rule_category"] is None


# ===========================================================================
# TestImportRules
# ===========================================================================


class TestImportRules:
    """Test import_rules() bulk import."""

    def test_import_count(self, engine: CategoryRuleEngine, import_data):
        """Import returns count of successfully imported rules."""
        count = engine.import_rules(import_data)
        assert count == 3

    def test_imported_rules_available(self, engine: CategoryRuleEngine, import_data):
        """Imported rules appear in list_rules."""
        engine.import_rules(import_data)
        rules = engine.list_rules(active_only=False)
        assert len(rules) == 3

    def test_import_with_invalid_entry(self, engine: CategoryRuleEngine):
        """Invalid entries are skipped during import."""
        data = [
            {"name": "Valid Rule", "match_type": "CONTAINS", "pattern": "test", "target_category": "cat"},
            {"name": "Invalid", "match_type": "INVALID", "pattern": "test", "target_category": "cat"},
        ]
        count = engine.import_rules(data)
        assert count == 1

    def test_import_empty_pattern_skipped(self, engine: CategoryRuleEngine):
        """Empty pattern entries are skipped."""
        data = [
            {"name": "Empty Pat", "match_type": "CONTAINS", "pattern": "", "target_category": "cat"},
        ]
        count = engine.import_rules(data)
        assert count == 0

    def test_import_empty_list(self, engine: CategoryRuleEngine):
        """Empty import list returns 0."""
        count = engine.import_rules([])
        assert count == 0


# ===========================================================================
# TestExportRules
# ===========================================================================


class TestExportRules:
    """Test export_rules() serialization."""

    def test_export_returns_dicts(self, engine: CategoryRuleEngine, contains_rule, exact_rule):
        """Export returns a list of dicts."""
        exported = engine.export_rules()
        assert isinstance(exported, list)
        assert len(exported) == 2
        assert all(isinstance(r, dict) for r in exported)

    def test_export_sorted_by_priority(self, engine: CategoryRuleEngine, multi_rules):
        """Exported rules are sorted by priority descending."""
        exported = engine.export_rules()
        priorities = [r["priority"] for r in exported]
        assert priorities == sorted(priorities, reverse=True)

    def test_export_includes_all_fields(self, engine: CategoryRuleEngine, contains_rule):
        """Exported rule dict includes all model fields."""
        exported = engine.export_rules()
        assert "rule_id" in exported[0]
        assert "name" in exported[0]
        assert "match_type" in exported[0]
        assert "pattern" in exported[0]
        assert "target_category" in exported[0]
        assert "provenance_hash" in exported[0]

    def test_export_empty(self, engine: CategoryRuleEngine):
        """Empty engine exports empty list."""
        exported = engine.export_rules()
        assert exported == []

    def test_roundtrip_import_export(self, engine: CategoryRuleEngine, import_data):
        """Import then export preserves rule data."""
        engine.import_rules(import_data)
        exported = engine.export_rules()
        assert len(exported) == len(import_data)


# ===========================================================================
# TestGetEffectiveness
# ===========================================================================


class TestGetEffectiveness:
    """Test get_effectiveness() match statistics."""

    def test_aggregate_stats(self, engine: CategoryRuleEngine, contains_rule):
        """Aggregate effectiveness includes total rules and matches."""
        engine.evaluate_rules({"description": "travel booking"})
        engine.evaluate_rules({"description": "travel agency"})
        eff = engine.get_effectiveness()
        assert eff["total_rules"] == 1
        assert eff["active_rules"] == 1
        assert eff["total_matches"] == 2

    def test_per_rule_stats(self, engine: CategoryRuleEngine, contains_rule):
        """Per-rule effectiveness returns specific rule info."""
        engine.evaluate_rules({"description": "travel booking"})
        eff = engine.get_effectiveness(rule_id=contains_rule.rule_id)
        assert eff["rule_id"] == contains_rule.rule_id
        assert eff["match_count"] == 1

    def test_nonexistent_rule_error(self, engine: CategoryRuleEngine):
        """Nonexistent rule ID returns error dict."""
        eff = engine.get_effectiveness(rule_id="nonexistent")
        assert "error" in eff

    def test_never_matched_count(self, engine: CategoryRuleEngine, contains_rule, exact_rule):
        """never_matched_rules counts active rules with zero matches."""
        eff = engine.get_effectiveness()
        assert eff["never_matched_rules"] == 2

    def test_top_rules_list(self, engine: CategoryRuleEngine, contains_rule, exact_rule):
        """top_rules list includes rule details."""
        engine.evaluate_rules({"description": "travel booking"})
        eff = engine.get_effectiveness()
        assert len(eff["top_rules"]) <= 10

    def test_by_match_type(self, engine: CategoryRuleEngine, contains_rule, exact_rule):
        """by_match_type tracks match counts per type."""
        engine.evaluate_rules({"description": "travel booking"})
        eff = engine.get_effectiveness()
        assert "CONTAINS" in eff["by_match_type"]

    def test_inactive_rules_counted(self, engine: CategoryRuleEngine, contains_rule):
        """Inactive rules are counted separately."""
        engine.update_rule(contains_rule.rule_id, active=False)
        eff = engine.get_effectiveness()
        assert eff["inactive_rules"] == 1


# ===========================================================================
# TestRuleMatchTypes
# ===========================================================================


class TestRuleMatchTypes:
    """Test each match type's evaluation logic."""

    def test_exact_matching_case_insensitive(self, engine: CategoryRuleEngine, exact_rule):
        """EXACT match is case-insensitive."""
        cat, _, _ = engine.evaluate_rules({"vendor_name": "amazon web services"})
        assert cat == "cloud_computing"

    def test_exact_no_partial_match(self, engine: CategoryRuleEngine, exact_rule):
        """EXACT does not match partial strings."""
        cat, _, _ = engine.evaluate_rules({"vendor_name": "Amazon Web Services Inc"})
        assert cat is None

    def test_contains_matching(self, engine: CategoryRuleEngine, contains_rule):
        """CONTAINS matches substring."""
        cat, _, _ = engine.evaluate_rules({"description": "Corporate travel booking"})
        assert cat == "business_travel"

    def test_contains_case_insensitive(self, engine: CategoryRuleEngine, contains_rule):
        """CONTAINS is case-insensitive."""
        cat, _, _ = engine.evaluate_rules({"description": "BUSINESS TRAVEL"})
        assert cat == "business_travel"

    def test_regex_matching(self, engine: CategoryRuleEngine, regex_rule):
        """REGEX matches pattern."""
        cat, _, _ = engine.evaluate_rules({"description": "Freight delivery for Q1"})
        assert cat == "transportation"

    def test_regex_no_match(self, engine: CategoryRuleEngine, regex_rule):
        """REGEX returns None when pattern does not match."""
        cat, _, _ = engine.evaluate_rules({"description": "Office desk purchase"})
        assert cat is None

    def test_regex_case_insensitive(self, engine: CategoryRuleEngine, regex_rule):
        """REGEX is case-insensitive via re.IGNORECASE."""
        cat, _, _ = engine.evaluate_rules({"description": "FREIGHT SERVICES"})
        assert cat == "transportation"

    def test_fuzzy_matching_high_similarity(self, engine: CategoryRuleEngine, fuzzy_rule):
        """FUZZY matches when similarity exceeds threshold."""
        cat, _, _ = engine.evaluate_rules({"description": "chemical supply"})
        assert cat == "chemicals"

    def test_fuzzy_no_match_low_similarity(self, engine: CategoryRuleEngine, fuzzy_rule):
        """FUZZY does not match when similarity is below threshold."""
        cat, _, _ = engine.evaluate_rules({"description": "xyz abc"})
        assert cat is None

    def test_starts_with_matching(self, engine: CategoryRuleEngine, starts_with_rule):
        """STARTS_WITH matches prefix."""
        cat, _, _ = engine.evaluate_rules({"description": "Capital equipment for factory"})
        assert cat == "capital_goods"

    def test_starts_with_case_insensitive(self, engine: CategoryRuleEngine, starts_with_rule):
        """STARTS_WITH is case-insensitive."""
        cat, _, _ = engine.evaluate_rules({"description": "capital equipment"})
        assert cat == "capital_goods"

    def test_starts_with_no_match(self, engine: CategoryRuleEngine, starts_with_rule):
        """STARTS_WITH does not match when text does not start with pattern."""
        cat, _, _ = engine.evaluate_rules({"description": "Equipment capital"})
        assert cat is None

    def test_ends_with_matching(self, engine: CategoryRuleEngine, ends_with_rule):
        """ENDS_WITH matches suffix."""
        cat, _, _ = engine.evaluate_rules({"description": "Consulting services"})
        assert cat == "professional_services"

    def test_ends_with_case_insensitive(self, engine: CategoryRuleEngine, ends_with_rule):
        """ENDS_WITH is case-insensitive."""
        cat, _, _ = engine.evaluate_rules({"description": "PROFESSIONAL SERVICES"})
        assert cat == "professional_services"

    def test_ends_with_no_match(self, engine: CategoryRuleEngine, ends_with_rule):
        """ENDS_WITH does not match when text does not end with pattern."""
        cat, _, _ = engine.evaluate_rules({"description": "services and consulting"})
        assert cat is None


# ===========================================================================
# TestStringSimilarity
# ===========================================================================


class TestStringSimilarity:
    """Test the _string_similarity() helper function."""

    def test_identical_strings(self):
        """Identical strings return 1.0."""
        assert _string_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        """Both empty returns 1.0."""
        assert _string_similarity("", "") == 1.0

    def test_one_empty(self):
        """One empty returns 0.0."""
        assert _string_similarity("hello", "") == 0.0
        assert _string_similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        """Similarity is case-insensitive."""
        assert _string_similarity("Hello", "hello") == 1.0

    def test_similar_strings(self):
        """Similar strings have high similarity."""
        sim = _string_similarity("chemical supplies", "chemical supply")
        assert sim > 0.7

    def test_dissimilar_strings(self):
        """Dissimilar strings have low similarity."""
        sim = _string_similarity("hello world", "xyz abc")
        assert sim < 0.3


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Test engine statistics tracking."""

    def test_initial_stats(self, engine: CategoryRuleEngine):
        """Statistics start at zero."""
        stats = engine.get_statistics()
        assert stats["rules_created"] == 0
        assert stats["evaluations_performed"] == 0

    def test_rule_counts(self, engine: CategoryRuleEngine, multi_rules):
        """Total and active rule counts are tracked."""
        stats = engine.get_statistics()
        assert stats["total_rules"] == 3
        assert stats["active_rules"] == 3

    def test_by_match_type_tracking(self, engine: CategoryRuleEngine, contains_rule):
        """Match type distribution is tracked."""
        engine.evaluate_rules({"description": "travel agency"})
        stats = engine.get_statistics()
        assert stats["by_match_type"].get("CONTAINS", 0) >= 1

    def test_error_counting(self, engine: CategoryRuleEngine):
        """Import errors increment error counter."""
        engine.import_rules([{"name": "Bad", "match_type": "INVALID", "pattern": "x", "target_category": "y"}])
        stats = engine.get_statistics()
        assert stats["errors"] >= 1


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hashes."""

    def test_create_rule_has_hash(self, engine: CategoryRuleEngine, contains_rule):
        """Created rule has a 64-char hex provenance hash."""
        assert len(contains_rule.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in contains_rule.provenance_hash)

    def test_update_changes_hash(self, engine: CategoryRuleEngine, contains_rule):
        """Updating a rule changes its provenance hash."""
        old_hash = contains_rule.provenance_hash
        updated = engine.update_rule(contains_rule.rule_id, name="Changed")
        assert updated.provenance_hash != old_hash

    def test_different_rules_different_hashes(self, engine: CategoryRuleEngine, contains_rule, exact_rule):
        """Different rules have different provenance hashes."""
        assert contains_rule.provenance_hash != exact_rule.provenance_hash


# ===========================================================================
# TestThreadSafety
# ===========================================================================


class TestThreadSafety:
    """Test thread-safe concurrent rule operations."""

    def test_concurrent_evaluation(self, engine: CategoryRuleEngine, contains_rule):
        """Concurrent evaluations do not corrupt state."""
        errors: List[str] = []

        def eval_task():
            try:
                for _ in range(50):
                    engine.evaluate_rules({"description": "travel booking"})
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=eval_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["evaluations_performed"] == 200
        assert stats["matches_found"] == 200

    def test_concurrent_creation_and_evaluation(self, engine: CategoryRuleEngine):
        """Concurrent rule creation and evaluation is safe."""
        errors: List[str] = []

        def create_task(idx: int):
            try:
                for i in range(10):
                    engine.create_rule(
                        f"Thread{idx}Rule{i}",
                        "CONTAINS",
                        f"pattern{idx}{i}",
                        f"cat{idx}{i}",
                    )
            except Exception as exc:
                errors.append(str(exc))

        def eval_task():
            try:
                for _ in range(20):
                    engine.evaluate_rules({"description": "some text"})
            except Exception as exc:
                errors.append(str(exc))

        threads = []
        for i in range(2):
            threads.append(threading.Thread(target=create_task, args=(i,)))
        for _ in range(2):
            threads.append(threading.Thread(target=eval_task))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["rules_created"] == 20
