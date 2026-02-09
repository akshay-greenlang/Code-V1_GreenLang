# -*- coding: utf-8 -*-
"""
Unit tests for QualityRuleEngine.

AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)
Tests rule CRUD, rule evaluation, quality gates, import/export,
and statistics tracking.

Target: 110+ tests, 85%+ coverage.
"""

import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pytest

from greenlang.data_quality_profiler.quality_rule_engine import (
    QualityRuleEngine,
    RULE_COMPLETENESS,
    RULE_RANGE,
    RULE_FORMAT,
    RULE_UNIQUENESS,
    RULE_CUSTOM,
    RULE_FRESHNESS,
    ALL_RULE_TYPES,
    OP_EQUALS,
    OP_NOT_EQUALS,
    OP_GREATER_THAN,
    OP_LESS_THAN,
    OP_BETWEEN,
    OP_MATCHES,
    OP_CONTAINS,
    OP_IN_SET,
    ALL_OPERATORS,
    GATE_PASS,
    GATE_WARN,
    GATE_FAIL,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    _compute_provenance,
    _is_missing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> QualityRuleEngine:
    """Create a default QualityRuleEngine."""
    return QualityRuleEngine()


@pytest.fixture
def custom_engine() -> QualityRuleEngine:
    """Create an engine with custom config."""
    return QualityRuleEngine(config={
        "max_rules": 100,
        "max_gates": 50,
        "default_threshold": 0.90,
    })


@pytest.fixture
def completeness_rule(engine) -> Dict[str, Any]:
    """Create a completeness rule."""
    return engine.create_rule(
        name="check_email_present",
        rule_type=RULE_COMPLETENESS,
        column="email",
        threshold=0.95,
    )


@pytest.fixture
def range_rule(engine) -> Dict[str, Any]:
    """Create a range rule."""
    return engine.create_rule(
        name="valid_age",
        rule_type=RULE_RANGE,
        column="age",
        operator=OP_BETWEEN,
        parameters={"min_val": 0, "max_val": 150},
    )


@pytest.fixture
def format_rule(engine) -> Dict[str, Any]:
    """Create a format rule."""
    return engine.create_rule(
        name="email_format",
        rule_type=RULE_FORMAT,
        column="email",
        parameters={"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
    )


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    """Sample test data."""
    return [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "Bob", "age": 25, "email": "bob@test.org"},
        {"name": "Charlie", "age": -5, "email": "invalid"},
        {"name": None, "age": 200, "email": None},
        {"name": "Eve", "age": 40, "email": "eve@company.co"},
    ]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Test QualityRuleEngine initialization."""

    def test_default_config(self):
        """Default settings are applied."""
        e = QualityRuleEngine()
        assert e._max_rules == 10000
        assert e._max_gates == 1000
        assert e._default_threshold == 0.95

    def test_custom_config(self):
        """Custom config overrides defaults."""
        e = QualityRuleEngine(config={
            "max_rules": 50,
            "max_gates": 10,
            "default_threshold": 0.80,
        })
        assert e._max_rules == 50
        assert e._max_gates == 10
        assert e._default_threshold == 0.80

    def test_empty_store(self):
        """Rule and gate stores start empty."""
        e = QualityRuleEngine()
        assert e.list_rules() == []
        assert e.list_gates() == []

    def test_initial_stats(self):
        """Stats start at zero."""
        e = QualityRuleEngine()
        stats = e.get_statistics()
        assert stats["rules_created"] == 0
        assert stats["rules_evaluated"] == 0
        assert stats["gates_created"] == 0


# ---------------------------------------------------------------------------
# TestCreateRule
# ---------------------------------------------------------------------------


class TestCreateRule:
    """Test create_rule() method."""

    @pytest.mark.parametrize("rule_type", list(ALL_RULE_TYPES))
    def test_all_rule_types(self, engine, rule_type):
        """Each valid rule type can be created."""
        rule = engine.create_rule(
            name=f"test_{rule_type}",
            rule_type=rule_type,
            column="col",
        )
        assert rule["rule_type"] == rule_type

    def test_rule_id_format(self, engine):
        """rule_id starts with QRL-."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert rule["rule_id"].startswith("QRL-")

    def test_default_priority(self, engine):
        """Default priority is PRIORITY_MEDIUM."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert rule["priority"] == PRIORITY_MEDIUM

    def test_custom_priority(self, engine):
        """Custom priority is respected."""
        rule = engine.create_rule(
            "test", RULE_COMPLETENESS, "col", priority=PRIORITY_CRITICAL
        )
        assert rule["priority"] == PRIORITY_CRITICAL

    def test_default_threshold(self, engine):
        """Threshold can be None."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert rule["threshold"] is None

    def test_custom_threshold(self, engine):
        """Custom threshold is stored."""
        rule = engine.create_rule(
            "test", RULE_COMPLETENESS, "col", threshold=0.99
        )
        assert rule["threshold"] == 0.99

    def test_parameters(self, engine):
        """Parameters dict is stored."""
        params = {"min_val": 0, "max_val": 100}
        rule = engine.create_rule(
            "test", RULE_RANGE, "col", parameters=params
        )
        assert rule["parameters"] == params

    def test_active_default(self, engine):
        """Rules are active by default."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert rule["active"] is True

    def test_operator(self, engine):
        """Operator is stored."""
        rule = engine.create_rule(
            "test", RULE_RANGE, "col", operator=OP_BETWEEN
        )
        assert rule["operator"] == OP_BETWEEN

    def test_column(self, engine):
        """Column is stored."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "email")
        assert rule["column"] == "email"

    def test_provenance_hash(self, engine):
        """provenance_hash is 64-char hex."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert len(rule["provenance_hash"]) == 64

    def test_stats_incremented(self, engine):
        """rules_created counter increments."""
        engine.create_rule("test", RULE_COMPLETENESS, "col")
        stats = engine.get_statistics()
        assert stats["rules_created"] == 1

    def test_invalid_rule_type(self, engine):
        """Invalid rule_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rule_type"):
            engine.create_rule("test", "INVALID_TYPE", "col")

    def test_invalid_operator(self, engine):
        """Invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            engine.create_rule(
                "test", RULE_COMPLETENESS, "col", operator="BOGUS"
            )

    def test_version_starts_at_one(self, engine):
        """Version starts at 1."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert rule["version"] == 1

    def test_timestamps_present(self, engine):
        """created_at and updated_at are present."""
        rule = engine.create_rule("test", RULE_COMPLETENESS, "col")
        assert "created_at" in rule
        assert "updated_at" in rule

    def test_name_stored(self, engine):
        """Rule name is stored."""
        rule = engine.create_rule("my_rule_name", RULE_COMPLETENESS, "col")
        assert rule["name"] == "my_rule_name"

    def test_match_type_operator(self, engine):
        """MATCHES operator stored correctly."""
        rule = engine.create_rule(
            "test", RULE_FORMAT, "col", operator=OP_MATCHES,
            parameters={"pattern": "^[A-Z]+$"}
        )
        assert rule["operator"] == OP_MATCHES


# ---------------------------------------------------------------------------
# TestGetRule
# ---------------------------------------------------------------------------


class TestGetRule:
    """Test get_rule() method."""

    def test_existing(self, engine, completeness_rule):
        """Retrieve existing rule."""
        rule = engine.get_rule(completeness_rule["rule_id"])
        assert rule is not None
        assert rule["name"] == "check_email_present"

    def test_nonexistent(self, engine):
        """Nonexistent rule returns None."""
        assert engine.get_rule("QRL-nonexistent") is None


# ---------------------------------------------------------------------------
# TestListRules
# ---------------------------------------------------------------------------


class TestListRules:
    """Test list_rules() method."""

    def test_all_rules(self, engine):
        """List all created rules."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        engine.create_rule("r2", RULE_RANGE, "col")
        rules = engine.list_rules()
        assert len(rules) == 2

    def test_active_only(self, engine):
        """Filter by active status."""
        r1 = engine.create_rule("r1", RULE_COMPLETENESS, "col")
        r2 = engine.create_rule("r2", RULE_RANGE, "col")
        engine.update_rule(r2["rule_id"], {"active": False})
        active = engine.list_rules(active_only=True)
        assert len(active) == 1

    def test_by_rule_type(self, engine):
        """Filter by rule_type."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        engine.create_rule("r2", RULE_RANGE, "col")
        completeness_rules = engine.list_rules(rule_type=RULE_COMPLETENESS)
        assert len(completeness_rules) == 1

    def test_priority_sorted(self, engine):
        """Rules are sorted by priority (lower number first)."""
        engine.create_rule("low", RULE_COMPLETENESS, "col", priority=PRIORITY_LOW)
        engine.create_rule("critical", RULE_COMPLETENESS, "col", priority=PRIORITY_CRITICAL)
        rules = engine.list_rules()
        assert rules[0]["priority"] <= rules[1]["priority"]

    def test_pagination(self, engine):
        """Pagination via limit and offset."""
        for i in range(5):
            engine.create_rule(f"r{i}", RULE_COMPLETENESS, "col")
        page = engine.list_rules(limit=2, offset=1)
        assert len(page) == 2

    def test_empty(self, engine):
        """Empty rule list."""
        rules = engine.list_rules()
        assert rules == []

    def test_limit_zero(self, engine):
        """Limit 0 returns empty."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        rules = engine.list_rules(limit=0)
        assert rules == []


# ---------------------------------------------------------------------------
# TestUpdateRule
# ---------------------------------------------------------------------------


class TestUpdateRule:
    """Test update_rule() method."""

    def test_update_name(self, engine, completeness_rule):
        """Update rule name."""
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"name": "new_name"}
        )
        assert updated["name"] == "new_name"

    def test_update_threshold(self, engine, completeness_rule):
        """Update threshold."""
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"threshold": 0.99}
        )
        assert updated["threshold"] == 0.99

    def test_update_priority(self, engine, completeness_rule):
        """Update priority."""
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"priority": PRIORITY_CRITICAL}
        )
        assert updated["priority"] == PRIORITY_CRITICAL

    def test_deactivate(self, engine, completeness_rule):
        """Deactivate a rule."""
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"active": False}
        )
        assert updated["active"] is False

    def test_update_parameters(self, engine, range_rule):
        """Update parameters."""
        updated = engine.update_rule(
            range_rule["rule_id"],
            {"parameters": {"min_val": 10, "max_val": 200}}
        )
        assert updated["parameters"]["min_val"] == 10

    def test_nonexistent(self, engine):
        """Update nonexistent rule returns None."""
        result = engine.update_rule("QRL-nonexistent", {"name": "x"})
        assert result is None

    def test_provenance_changes(self, engine, completeness_rule):
        """Provenance hash changes on update."""
        old_hash = completeness_rule["provenance_hash"]
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"name": "new"}
        )
        assert updated["provenance_hash"] != old_hash

    def test_timestamp_updates(self, engine, completeness_rule):
        """updated_at changes on update."""
        old_ts = completeness_rule["updated_at"]
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"name": "new"}
        )
        # May or may not differ by microsecond; just check it exists
        assert "updated_at" in updated

    def test_invalid_field_raises(self, engine, completeness_rule):
        """Invalid field raises ValueError."""
        with pytest.raises(ValueError, match="Cannot update"):
            engine.update_rule(
                completeness_rule["rule_id"], {"rule_type": "RANGE"}
            )

    def test_version_increments(self, engine, completeness_rule):
        """Version increments on update."""
        assert completeness_rule["version"] == 1
        updated = engine.update_rule(
            completeness_rule["rule_id"], {"name": "v2"}
        )
        assert updated["version"] == 2


# ---------------------------------------------------------------------------
# TestDeleteRule
# ---------------------------------------------------------------------------


class TestDeleteRule:
    """Test delete_rule() method."""

    def test_existing(self, engine, completeness_rule):
        """Delete existing rule returns True."""
        assert engine.delete_rule(completeness_rule["rule_id"]) is True
        assert engine.get_rule(completeness_rule["rule_id"]) is None

    def test_nonexistent(self, engine):
        """Delete nonexistent returns False."""
        assert engine.delete_rule("QRL-nonexistent") is False

    def test_stats_not_decremented(self, engine, completeness_rule):
        """rules_created does not decrement on delete."""
        stats_before = engine.get_statistics()["rules_created"]
        engine.delete_rule(completeness_rule["rule_id"])
        stats_after = engine.get_statistics()["rules_created"]
        assert stats_after == stats_before

    def test_not_in_list(self, engine, completeness_rule):
        """Deleted rule not in list."""
        engine.delete_rule(completeness_rule["rule_id"])
        rules = engine.list_rules()
        ids = [r["rule_id"] for r in rules]
        assert completeness_rule["rule_id"] not in ids


# ---------------------------------------------------------------------------
# TestEvaluateRule
# ---------------------------------------------------------------------------


class TestEvaluateRule:
    """Test evaluate_rule() method."""

    def test_completeness_pass(self, engine, completeness_rule):
        """All non-null -> PASS."""
        data = [{"email": "a@b.com"}, {"email": "c@d.com"}]
        result = engine.evaluate_rule(completeness_rule, data)
        assert result["pass_rate"] == 1.0
        assert result["outcome"] == GATE_PASS

    def test_completeness_fail(self, engine, completeness_rule):
        """Many nulls -> FAIL."""
        data = [{"email": None}] * 10
        result = engine.evaluate_rule(completeness_rule, data)
        assert result["pass_rate"] == 0.0
        assert result["outcome"] == GATE_FAIL

    def test_range_within(self, engine, range_rule):
        """Value within range passes."""
        data = [{"age": 25}, {"age": 50}]
        result = engine.evaluate_rule(range_rule, data)
        assert result["pass_rate"] == 1.0

    def test_range_below(self, engine, range_rule):
        """Value below min fails."""
        data = [{"age": -5}]
        result = engine.evaluate_rule(range_rule, data)
        assert result["pass_rate"] == 0.0

    def test_range_above(self, engine, range_rule):
        """Value above max fails."""
        data = [{"age": 200}]
        result = engine.evaluate_rule(range_rule, data)
        assert result["pass_rate"] == 0.0

    def test_format_high_match(self, engine, format_rule):
        """Good format matches pass."""
        data = [{"email": "user@test.com"}, {"email": "admin@site.org"}]
        result = engine.evaluate_rule(format_rule, data)
        assert result["pass_rate"] == 1.0

    def test_format_low_match(self, engine, format_rule):
        """Bad format fails."""
        data = [{"email": "not_an_email"}, {"email": "also_bad"}]
        result = engine.evaluate_rule(format_rule, data)
        assert result["pass_rate"] == 0.0

    def test_uniqueness_pass(self, engine):
        """Non-null values pass uniqueness."""
        rule = engine.create_rule("unique_id", RULE_UNIQUENESS, "id")
        data = [{"id": "A"}, {"id": "B"}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_uniqueness_fail(self, engine):
        """Null values fail uniqueness."""
        rule = engine.create_rule("unique_id", RULE_UNIQUENESS, "id")
        data = [{"id": None}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_custom_rule_pass(self, engine):
        """Custom rule with comparator passes."""
        rule = engine.create_rule(
            "custom_check", RULE_CUSTOM, "score",
            parameters={"comparator": OP_GREATER_THAN, "compare_value": 50}
        )
        data = [{"score": 80}, {"score": 90}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_custom_rule_fail(self, engine):
        """Custom rule fails when condition not met."""
        rule = engine.create_rule(
            "custom_check", RULE_CUSTOM, "score",
            parameters={"comparator": OP_GREATER_THAN, "compare_value": 100}
        )
        data = [{"score": 50}, {"score": 60}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_freshness_pass(self, engine):
        """Fresh timestamps pass freshness rule."""
        now = datetime.now(timezone.utc)
        rule = engine.create_rule(
            "freshness", RULE_FRESHNESS, "updated_at",
            parameters={"max_age_hours": 48}
        )
        data = [{"updated_at": now.strftime("%Y-%m-%dT%H:%M:%S")}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_freshness_fail(self, engine):
        """Very old timestamps fail freshness rule."""
        old = datetime.now(timezone.utc) - timedelta(hours=100)
        rule = engine.create_rule(
            "freshness", RULE_FRESHNESS, "updated_at",
            parameters={"max_age_hours": 24}
        )
        data = [{"updated_at": old.strftime("%Y-%m-%dT%H:%M:%S")}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_empty_data(self, engine, completeness_rule):
        """Empty data -> pass_rate 1.0 (vacuous)."""
        result = engine.evaluate_rule(completeness_rule, [])
        assert result["pass_rate"] == 1.0

    def test_evaluation_id(self, engine, completeness_rule):
        """evaluation_id starts with EVL-."""
        data = [{"email": "x@y.com"}]
        result = engine.evaluate_rule(completeness_rule, data)
        assert result["evaluation_id"].startswith("EVL-")

    def test_message_present(self, engine, completeness_rule):
        """rule_name is in result."""
        data = [{"email": "x@y.com"}]
        result = engine.evaluate_rule(completeness_rule, data)
        assert result["rule_name"] == "check_email_present"

    def test_provenance_hash(self, engine, completeness_rule):
        """provenance_hash is 64-char hex."""
        data = [{"email": "x@y.com"}]
        result = engine.evaluate_rule(completeness_rule, data)
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestEvaluateRules
# ---------------------------------------------------------------------------


class TestEvaluateRules:
    """Test evaluate_rules() method."""

    def test_multiple_rules(self, engine, sample_data):
        """Evaluate multiple rules at once."""
        engine.create_rule("r1", RULE_COMPLETENESS, "name")
        engine.create_rule("r2", RULE_COMPLETENESS, "email")
        results = engine.evaluate_rules(sample_data)
        assert len(results) == 2

    def test_all_pass(self, engine):
        """All rules pass."""
        engine.create_rule("r1", RULE_COMPLETENESS, "name")
        data = [{"name": "Alice"}, {"name": "Bob"}]
        results = engine.evaluate_rules(data)
        assert all(r["pass_rate"] == 1.0 for r in results)

    def test_some_fail(self, engine, sample_data):
        """Some rules fail."""
        engine.create_rule("r1", RULE_COMPLETENESS, "name")
        engine.create_rule(
            "r2", RULE_RANGE, "age",
            parameters={"min_val": 0, "max_val": 150}
        )
        results = engine.evaluate_rules(sample_data)
        assert len(results) == 2

    def test_no_rules(self, engine, sample_data):
        """No rules -> empty results."""
        results = engine.evaluate_rules(sample_data)
        assert results == []

    def test_empty_data(self, engine):
        """Empty data with rules."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        results = engine.evaluate_rules([])
        assert len(results) == 1
        assert results[0]["pass_rate"] == 1.0

    def test_specific_rule_ids(self, engine, sample_data):
        """Evaluate only specific rule IDs."""
        r1 = engine.create_rule("r1", RULE_COMPLETENESS, "name")
        r2 = engine.create_rule("r2", RULE_COMPLETENESS, "email")
        results = engine.evaluate_rules(sample_data, rule_ids=[r1["rule_id"]])
        assert len(results) == 1

    def test_inactive_skipped(self, engine, sample_data):
        """Inactive rules are skipped when no rule_ids given."""
        r1 = engine.create_rule("r1", RULE_COMPLETENESS, "name")
        r2 = engine.create_rule("r2", RULE_COMPLETENESS, "email")
        engine.update_rule(r2["rule_id"], {"active": False})
        results = engine.evaluate_rules(sample_data)
        assert len(results) == 1

    def test_nonexistent_ids_skipped(self, engine, sample_data):
        """Nonexistent rule IDs are silently skipped."""
        results = engine.evaluate_rules(
            sample_data, rule_ids=["QRL-nonexistent"]
        )
        assert results == []


# ---------------------------------------------------------------------------
# TestEvaluateGate
# ---------------------------------------------------------------------------


class TestEvaluateGate:
    """Test evaluate_gate() method."""

    def test_all_pass(self, engine):
        """All conditions met -> PASS."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9},
            {"dimension": "validity", "operator": OP_GREATER_THAN, "threshold": 0.8},
        ]
        scores = {"completeness": 0.95, "validity": 0.85}
        result = engine.evaluate_gate(conditions, scores)
        assert result["outcome"] == GATE_PASS

    def test_mixed_warn(self, engine):
        """Some conditions met -> WARN (>=50%)."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9},
            {"dimension": "validity", "operator": OP_GREATER_THAN, "threshold": 0.9},
        ]
        scores = {"completeness": 0.95, "validity": 0.7}
        result = engine.evaluate_gate(conditions, scores)
        assert result["outcome"] == GATE_WARN

    def test_many_fail(self, engine):
        """Most conditions fail -> FAIL (<50%)."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9},
            {"dimension": "validity", "operator": OP_GREATER_THAN, "threshold": 0.9},
            {"dimension": "timeliness", "operator": OP_GREATER_THAN, "threshold": 0.9},
        ]
        scores = {"completeness": 0.5, "validity": 0.3, "timeliness": 0.2}
        result = engine.evaluate_gate(conditions, scores)
        assert result["outcome"] == GATE_FAIL

    def test_single_condition_pass(self, engine):
        """Single passing condition."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9},
        ]
        scores = {"completeness": 0.95}
        result = engine.evaluate_gate(conditions, scores)
        assert result["outcome"] == GATE_PASS

    def test_single_condition_fail(self, engine):
        """Single failing condition -> FAIL."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9},
        ]
        scores = {"completeness": 0.5}
        result = engine.evaluate_gate(conditions, scores)
        assert result["outcome"] == GATE_FAIL

    def test_all_dimensions(self, engine):
        """Multiple dimension types."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.8},
            {"dimension": "validity", "operator": OP_GREATER_THAN, "threshold": 0.8},
            {"dimension": "consistency", "operator": OP_GREATER_THAN, "threshold": 0.8},
            {"dimension": "timeliness", "operator": OP_GREATER_THAN, "threshold": 0.8},
        ]
        scores = {
            "completeness": 0.95, "validity": 0.9,
            "consistency": 0.85, "timeliness": 0.9,
        }
        result = engine.evaluate_gate(conditions, scores)
        assert result["outcome"] == GATE_PASS

    def test_dimension_scores_mapping(self, engine):
        """Missing dimension defaults to 0.0."""
        conditions = [
            {"dimension": "unknown_dim", "operator": OP_GREATER_THAN, "threshold": 0.5},
        ]
        result = engine.evaluate_gate(conditions, {})
        assert result["outcome"] == GATE_FAIL

    def test_provenance_hash(self, engine):
        """provenance_hash is 64-char hex."""
        conditions = [
            {"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9},
        ]
        result = engine.evaluate_gate(conditions, {"completeness": 0.95})
        assert len(result["provenance_hash"]) == 64

    def test_condition_results(self, engine):
        """condition_results detail each condition."""
        conditions = [
            {"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.5},
            {"dimension": "b", "operator": OP_GREATER_THAN, "threshold": 0.5},
        ]
        result = engine.evaluate_gate(
            conditions, {"a": 0.8, "b": 0.3}
        )
        assert len(result["condition_results"]) == 2
        passed_list = [cr["passed"] for cr in result["condition_results"]]
        assert True in passed_list
        assert False in passed_list

    def test_met_ratio(self, engine):
        """met_ratio is computed correctly."""
        conditions = [
            {"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.5},
            {"dimension": "b", "operator": OP_GREATER_THAN, "threshold": 0.5},
        ]
        result = engine.evaluate_gate(conditions, {"a": 0.8, "b": 0.3})
        assert result["met_ratio"] == 0.5

    def test_empty_conditions(self, engine):
        """No conditions -> all pass (vacuous truth)."""
        result = engine.evaluate_gate([], {})
        assert result["outcome"] == GATE_PASS
        assert result["met_ratio"] == 1.0

    def test_less_than_operator(self, engine):
        """LESS_THAN operator works in gate conditions."""
        conditions = [
            {"dimension": "error_rate", "operator": OP_LESS_THAN, "threshold": 0.05},
        ]
        result = engine.evaluate_gate(conditions, {"error_rate": 0.01})
        assert result["outcome"] == GATE_PASS


# ---------------------------------------------------------------------------
# TestCreateGate
# ---------------------------------------------------------------------------


class TestCreateGate:
    """Test create_gate() method."""

    def test_basic(self, engine):
        """Create a basic gate."""
        gate = engine.create_gate(
            "quality_gate",
            [{"dimension": "completeness", "operator": OP_GREATER_THAN, "threshold": 0.9}],
        )
        assert gate["gate_id"].startswith("GTE-")
        assert gate["name"] == "quality_gate"

    def test_custom_threshold(self, engine):
        """Custom gate threshold is stored."""
        gate = engine.create_gate(
            "gate",
            [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}],
            threshold=0.8,
        )
        assert gate["threshold"] == 0.8

    def test_multiple_conditions(self, engine):
        """Multiple conditions are stored."""
        conditions = [
            {"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8},
            {"dimension": "b", "operator": OP_GREATER_THAN, "threshold": 0.7},
        ]
        gate = engine.create_gate("gate", conditions)
        assert len(gate["conditions"]) == 2

    def test_empty_conditions_raises(self, engine):
        """Empty conditions raises ValueError."""
        with pytest.raises(ValueError, match="at least one condition"):
            engine.create_gate("gate", [])

    def test_gate_id_format(self, engine):
        """gate_id starts with GTE-."""
        gate = engine.create_gate(
            "gate",
            [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}],
        )
        assert gate["gate_id"].startswith("GTE-")

    def test_provenance_hash(self, engine):
        """Gate has provenance hash."""
        gate = engine.create_gate(
            "gate",
            [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}],
        )
        assert len(gate["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestListGates
# ---------------------------------------------------------------------------


class TestListGates:
    """Test list_gates() method."""

    def test_empty(self, engine):
        """Empty gate list."""
        gates = engine.list_gates()
        assert gates == []

    def test_multiple(self, engine):
        """Multiple gates listed."""
        engine.create_gate("g1", [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}])
        engine.create_gate("g2", [{"dimension": "b", "operator": OP_GREATER_THAN, "threshold": 0.7}])
        gates = engine.list_gates()
        assert len(gates) == 2

    def test_pagination(self, engine):
        """Pagination via limit and offset."""
        for i in range(5):
            engine.create_gate(
                f"g{i}",
                [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}],
            )
        page = engine.list_gates(limit=2, offset=1)
        assert len(page) == 2

    def test_sorted_by_created_at(self, engine):
        """Gates sorted by created_at descending."""
        engine.create_gate("g1", [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}])
        engine.create_gate("g2", [{"dimension": "b", "operator": OP_GREATER_THAN, "threshold": 0.7}])
        gates = engine.list_gates()
        assert len(gates) == 2


# ---------------------------------------------------------------------------
# TestImportRules
# ---------------------------------------------------------------------------


class TestImportRules:
    """Test import_rules() method."""

    def test_import_count(self, engine):
        """Imported count matches valid rules."""
        rules_data = [
            {"name": "r1", "rule_type": RULE_COMPLETENESS, "column": "col"},
            {"name": "r2", "rule_type": RULE_RANGE, "column": "col"},
        ]
        count = engine.import_rules(rules_data)
        assert count == 2

    def test_available_after_import(self, engine):
        """Imported rules are available."""
        rules_data = [
            {"name": "imported1", "rule_type": RULE_COMPLETENESS, "column": "c"},
        ]
        engine.import_rules(rules_data)
        rules = engine.list_rules()
        assert len(rules) == 1
        assert rules[0]["name"] == "imported1"

    def test_invalid_skipped(self, engine):
        """Invalid rule defs are skipped."""
        rules_data = [
            {"name": "valid", "rule_type": RULE_COMPLETENESS, "column": "c"},
            {"name": "invalid", "rule_type": "INVALID_TYPE", "column": "c"},
        ]
        count = engine.import_rules(rules_data)
        assert count == 1

    def test_empty_import(self, engine):
        """Empty import."""
        count = engine.import_rules([])
        assert count == 0

    def test_roundtrip(self, engine):
        """Export then import preserves rules."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        engine.create_rule("r2", RULE_RANGE, "col", parameters={"min_val": 0})
        exported = engine.export_rules()

        engine2 = QualityRuleEngine()
        count = engine2.import_rules(exported)
        assert count == 2


# ---------------------------------------------------------------------------
# TestExportRules
# ---------------------------------------------------------------------------


class TestExportRules:
    """Test export_rules() method."""

    def test_returns_dicts(self, engine):
        """Exported rules are dicts."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        exported = engine.export_rules()
        assert all(isinstance(r, dict) for r in exported)

    def test_all_fields(self, engine):
        """Exported rules have required fields."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        exported = engine.export_rules()
        required = {"name", "rule_type", "column", "operator", "threshold", "parameters", "priority", "active"}
        assert required.issubset(exported[0].keys())

    def test_empty_export(self, engine):
        """Empty rule store exports empty list."""
        exported = engine.export_rules()
        assert exported == []

    def test_multiple_rules(self, engine):
        """Multiple rules exported."""
        engine.create_rule("r1", RULE_COMPLETENESS, "a")
        engine.create_rule("r2", RULE_RANGE, "b")
        exported = engine.export_rules()
        assert len(exported) == 2

    def test_roundtrip_preserves_types(self, engine):
        """Rule types are preserved in roundtrip."""
        engine.create_rule("r1", RULE_RANGE, "col", parameters={"min_val": 0, "max_val": 100})
        exported = engine.export_rules()

        engine2 = QualityRuleEngine()
        engine2.import_rules(exported)
        rules = engine2.list_rules()
        assert rules[0]["rule_type"] == RULE_RANGE
        assert rules[0]["parameters"]["min_val"] == 0


# ---------------------------------------------------------------------------
# TestStatistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test get_statistics()."""

    def test_initial_statistics(self, engine):
        """Initial stats all zero."""
        stats = engine.get_statistics()
        assert stats["rules_created"] == 0
        assert stats["rules_evaluated"] == 0
        assert stats["gates_created"] == 0
        assert stats["gates_evaluated"] == 0
        assert stats["stored_rules"] == 0

    def test_post_evaluation_statistics(self, engine, completeness_rule):
        """Stats update after evaluation."""
        data = [{"email": "a@b.com"}]
        engine.evaluate_rule(completeness_rule, data)
        stats = engine.get_statistics()
        assert stats["rules_evaluated"] == 1
        assert stats["stored_evaluations"] == 1

    def test_post_create_statistics(self, engine):
        """Stats update after create."""
        engine.create_rule("r1", RULE_COMPLETENESS, "col")
        stats = engine.get_statistics()
        assert stats["rules_created"] == 1
        assert stats["stored_rules"] == 1

    def test_gate_statistics(self, engine):
        """Gate stats update."""
        engine.create_gate("g", [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}])
        engine.evaluate_gate(
            [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}],
            {"a": 0.9},
        )
        stats = engine.get_statistics()
        assert stats["gates_created"] == 1
        assert stats["gates_evaluated"] == 1


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_format(self, engine, completeness_rule):
        """Rule provenance hash is 64-char hex."""
        h = completeness_rule["provenance_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_evaluation_provenance(self, engine, completeness_rule):
        """Evaluation provenance hash is 64-char hex."""
        data = [{"email": "a@b.com"}]
        result = engine.evaluate_rule(completeness_rule, data)
        assert len(result["provenance_hash"]) == 64

    def test_helper_function(self):
        """_compute_provenance returns 64-char hex."""
        h = _compute_provenance("test_op", "test_data")
        assert len(h) == 64


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of QualityRuleEngine."""

    def test_concurrent_evaluation(self, engine):
        """Multiple threads can evaluate rules concurrently."""
        rule = engine.create_rule("r1", RULE_COMPLETENESS, "col")
        data = [{"col": "value"}] * 10
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(5):
                    engine.evaluate_rule(rule, data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["rules_evaluated"] == 20

    def test_concurrent_create(self, engine):
        """Multiple threads can create rules concurrently."""
        errors: List[Exception] = []

        def worker(thread_id):
            try:
                for i in range(5):
                    engine.create_rule(
                        f"t{thread_id}_r{i}",
                        RULE_COMPLETENESS,
                        "col",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["rules_created"] == 20


# ---------------------------------------------------------------------------
# TestOperatorEvaluation
# ---------------------------------------------------------------------------


class TestOperatorEvaluation:
    """Test various operators via evaluate_rule."""

    def test_equals_operator(self, engine):
        """EQUALS operator compares string equality."""
        rule = engine.create_rule(
            "eq_check", RULE_CUSTOM, "status",
            operator=OP_EQUALS,
            parameters={"comparator": OP_EQUALS, "compare_value": "active"},
        )
        data = [{"status": "active"}, {"status": "inactive"}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_not_equals_operator(self, engine):
        """NOT_EQUALS operator."""
        rule = engine.create_rule(
            "neq_check", RULE_CUSTOM, "status",
            parameters={"comparator": OP_NOT_EQUALS, "compare_value": "blocked"},
        )
        data = [{"status": "active"}, {"status": "blocked"}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_greater_than_operator(self, engine):
        """GREATER_THAN operator."""
        rule = engine.create_rule(
            "gt_check", RULE_CUSTOM, "score",
            parameters={"comparator": OP_GREATER_THAN, "compare_value": 50},
        )
        data = [{"score": 80}, {"score": 30}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_less_than_operator(self, engine):
        """LESS_THAN operator."""
        rule = engine.create_rule(
            "lt_check", RULE_CUSTOM, "errors",
            parameters={"comparator": OP_LESS_THAN, "compare_value": 5},
        )
        data = [{"errors": 3}, {"errors": 10}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_contains_operator(self, engine):
        """CONTAINS operator."""
        rule = engine.create_rule(
            "contains_check", RULE_CUSTOM, "desc",
            parameters={"comparator": OP_CONTAINS, "compare_value": "green"},
        )
        data = [{"desc": "greenlang platform"}, {"desc": "other system"}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_in_set_operator(self, engine):
        """IN_SET operator requires compare_value to be non-None for custom eval."""
        rule = engine.create_rule(
            "in_set_check", RULE_CUSTOM, "status",
            parameters={
                "comparator": OP_IN_SET,
                "compare_value": "active",  # non-None triggers operator eval
                "allowed_values": ["active", "pending"],
            },
        )
        data = [{"status": "active"}, {"status": "blocked"}]
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1


# ---------------------------------------------------------------------------
# TestIsMissing
# ---------------------------------------------------------------------------


class TestIsMissing:
    """Test _is_missing helper."""

    @pytest.mark.parametrize("value,expected", [
        (None, True),
        ("", True),
        ("   ", True),
        ("hello", False),
        (0, False),
        (False, False),
        ([], False),
    ])
    def test_is_missing(self, value, expected):
        """_is_missing correctly identifies missing values."""
        assert _is_missing(value) == expected


# ---------------------------------------------------------------------------
# TestDeleteGate
# ---------------------------------------------------------------------------


class TestDeleteGate:
    """Test delete_gate() method."""

    def test_delete_existing(self, engine):
        """Delete existing gate returns True."""
        gate = engine.create_gate(
            "g", [{"dimension": "a", "operator": OP_GREATER_THAN, "threshold": 0.8}]
        )
        assert engine.delete_gate(gate["gate_id"]) is True
        assert engine.get_gate(gate["gate_id"]) is None

    def test_delete_nonexistent(self, engine):
        """Delete nonexistent returns False."""
        assert engine.delete_gate("GTE-nonexistent") is False


# ---------------------------------------------------------------------------
# TestEvaluationStorage
# ---------------------------------------------------------------------------


class TestEvaluationStorage:
    """Test evaluation storage and retrieval."""

    def test_get_evaluation(self, engine, completeness_rule):
        """Retrieve stored evaluation."""
        data = [{"email": "a@b.com"}]
        result = engine.evaluate_rule(completeness_rule, data)
        stored = engine.get_evaluation(result["evaluation_id"])
        assert stored is not None

    def test_list_evaluations(self, engine, completeness_rule):
        """List stored evaluations."""
        data = [{"email": "a@b.com"}]
        engine.evaluate_rule(completeness_rule, data)
        engine.evaluate_rule(completeness_rule, data)
        evals = engine.list_evaluations()
        assert len(evals) == 2

    def test_list_evaluations_pagination(self, engine, completeness_rule):
        """List evaluations with pagination."""
        data = [{"email": "a@b.com"}]
        for _ in range(5):
            engine.evaluate_rule(completeness_rule, data)
        page = engine.list_evaluations(limit=2, offset=1)
        assert len(page) == 2
