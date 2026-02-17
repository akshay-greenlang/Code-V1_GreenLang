# -*- coding: utf-8 -*-
"""
Unit tests for RuleBasedImputerEngine - AGENT-DATA-012

Tests evaluate_rules (8 condition types, priority ordering, first-match-wins),
lookup_imputation, regulatory_defaults (3 frameworks), create_rule,
validate_rule (contradiction detection), apply_rule_set, helper functions,
and edge cases.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.rule_based_imputer import (
    RuleBasedImputerEngine,
    _is_missing,
    _classify_confidence,
    _PRIORITY_ORDER,
    _PRIORITY_CONFIDENCE,
)
from greenlang.missing_value_imputer.models import (
    ConfidenceLevel,
    ImputationRule,
    ImputationStrategy,
    LookupEntry,
    LookupTable,
    RuleCondition,
    RuleConditionType,
    RulePriority,
)


@pytest.fixture
def engine():
    return RuleBasedImputerEngine(MissingValueImputerConfig())


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_missing_none(self):
        assert _is_missing(None) is True

    def test_is_missing_empty_str(self):
        assert _is_missing("") is True

    def test_is_missing_whitespace(self):
        assert _is_missing("   ") is True

    def test_is_missing_nan(self):
        assert _is_missing(float("nan")) is True

    def test_is_missing_valid_value(self):
        assert _is_missing(42) is False

    def test_classify_confidence_high(self):
        assert _classify_confidence(0.90) == ConfidenceLevel.HIGH

    def test_classify_confidence_medium(self):
        assert _classify_confidence(0.75) == ConfidenceLevel.MEDIUM

    def test_classify_confidence_low(self):
        assert _classify_confidence(0.55) == ConfidenceLevel.LOW

    def test_classify_confidence_very_low(self):
        assert _classify_confidence(0.30) == ConfidenceLevel.VERY_LOW

    def test_priority_order_critical(self):
        assert _PRIORITY_ORDER[RulePriority.CRITICAL] == 5

    def test_priority_order_default(self):
        assert _PRIORITY_ORDER[RulePriority.DEFAULT] == 1

    def test_priority_confidence_critical(self):
        assert _PRIORITY_CONFIDENCE[RulePriority.CRITICAL] == 0.95

    def test_priority_confidence_low(self):
        assert _PRIORITY_CONFIDENCE[RulePriority.LOW] == 0.70


# ---------------------------------------------------------------------------
# evaluate_rules tests
# ---------------------------------------------------------------------------


def _make_rule(
    name="test_rule",
    target_column="val",
    conditions=None,
    impute_value=99.0,
    priority="medium",
    active=True,
):
    """Build an ImputationRule for testing."""
    if conditions is None:
        conditions = []
    cond_objs = []
    for c in conditions:
        cond_objs.append(
            RuleCondition(
                field_name=c.get("field_name", ""),
                condition_type=RuleConditionType(c.get("condition_type", "equals")),
                value=c.get("value"),
                case_sensitive=c.get("case_sensitive", False),
            )
        )
    rule = ImputationRule(
        name=name,
        target_column=target_column,
        conditions=cond_objs,
        impute_value=impute_value,
        priority=RulePriority(priority),
    )
    rule.active = active
    return rule


class TestEvaluateRules:
    def test_basic_match(self, engine):
        records = [{"val": None, "cat": "A"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "equals", "value": "A"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1
        assert result[0].imputed_value == 99.0
        assert result[0].strategy == ImputationStrategy.RULE_BASED

    def test_no_active_rules_returns_empty(self, engine):
        records = [{"val": None}]
        rule = _make_rule(active=False)
        result = engine.evaluate_rules(records, "val", [rule])
        assert result == []

    def test_no_matching_column_rules_returns_empty(self, engine):
        records = [{"val": None}]
        rule = _make_rule(target_column="other")
        result = engine.evaluate_rules(records, "val", [rule])
        assert result == []

    def test_no_missing_returns_empty(self, engine):
        records = [{"val": 10.0}]
        rule = _make_rule()
        result = engine.evaluate_rules(records, "val", [rule])
        assert result == []

    def test_priority_ordering(self, engine):
        records = [{"val": None}]
        low_rule = _make_rule(name="low", priority="low", impute_value=1.0, conditions=[])
        high_rule = _make_rule(name="high", priority="high", impute_value=2.0, conditions=[])
        result = engine.evaluate_rules(records, "val", [low_rule, high_rule])
        assert len(result) == 1
        assert result[0].imputed_value == 2.0  # High priority wins

    def test_first_match_wins(self, engine):
        records = [{"val": None, "cat": "A"}]
        rule1 = _make_rule(name="r1", impute_value=10.0, conditions=[{"field_name": "cat", "condition_type": "equals", "value": "A"}])
        rule2 = _make_rule(name="r2", impute_value=20.0, conditions=[{"field_name": "cat", "condition_type": "equals", "value": "A"}])
        result = engine.evaluate_rules(records, "val", [rule1, rule2])
        assert len(result) == 1

    def test_confidence_set_by_priority(self, engine):
        records = [{"val": None}]
        rule = _make_rule(priority="critical", conditions=[])
        result = engine.evaluate_rules(records, "val", [rule])
        assert result[0].confidence == 0.95

    def test_provenance_hash_present(self, engine):
        records = [{"val": None}]
        rule = _make_rule(conditions=[])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result[0].provenance_hash) == 64

    def test_multiple_missing_records(self, engine):
        records = [{"val": None}, {"val": 5.0}, {"val": None}]
        rule = _make_rule(conditions=[])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 2
        assert result[0].record_index == 0
        assert result[1].record_index == 2


# ---------------------------------------------------------------------------
# Condition type tests
# ---------------------------------------------------------------------------


class TestConditionTypes:
    def test_equals(self, engine):
        records = [{"val": None, "cat": "A"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "equals", "value": "A"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_equals_case_insensitive(self, engine):
        records = [{"val": None, "cat": "Office"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "equals", "value": "office", "case_sensitive": False}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_not_equals(self, engine):
        records = [{"val": None, "cat": "B"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "not_equals", "value": "A"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_not_equals_no_match(self, engine):
        records = [{"val": None, "cat": "A"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "not_equals", "value": "A"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 0

    def test_contains(self, engine):
        records = [{"val": None, "desc": "Natural Gas Boiler"}]
        rule = _make_rule(conditions=[{"field_name": "desc", "condition_type": "contains", "value": "gas"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_greater_than(self, engine):
        records = [{"val": None, "temp": 30.0}]
        rule = _make_rule(conditions=[{"field_name": "temp", "condition_type": "greater_than", "value": 25}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_greater_than_no_match(self, engine):
        records = [{"val": None, "temp": 20.0}]
        rule = _make_rule(conditions=[{"field_name": "temp", "condition_type": "greater_than", "value": 25}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 0

    def test_less_than(self, engine):
        records = [{"val": None, "temp": 10.0}]
        rule = _make_rule(conditions=[{"field_name": "temp", "condition_type": "less_than", "value": 25}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_in_list(self, engine):
        records = [{"val": None, "cat": "office"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "in_list", "value": ["office", "warehouse"]}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_in_list_no_match(self, engine):
        records = [{"val": None, "cat": "factory"}]
        rule = _make_rule(conditions=[{"field_name": "cat", "condition_type": "in_list", "value": ["office", "warehouse"]}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 0

    def test_regex(self, engine):
        records = [{"val": None, "code": "EF-2024-001"}]
        rule = _make_rule(conditions=[{"field_name": "code", "condition_type": "regex", "value": r"^EF-\d{4}-\d{3}$"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_regex_no_match(self, engine):
        records = [{"val": None, "code": "invalid"}]
        rule = _make_rule(conditions=[{"field_name": "code", "condition_type": "regex", "value": r"^EF-\d{4}-\d{3}$"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 0

    def test_is_null(self, engine):
        records = [{"val": None, "other": None}]
        rule = _make_rule(conditions=[{"field_name": "other", "condition_type": "is_null", "value": None}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_missing_field_fails_condition(self, engine):
        records = [{"val": None}]
        rule = _make_rule(conditions=[{"field_name": "nonexistent", "condition_type": "equals", "value": "X"}])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 0

    def test_multiple_conditions_all_must_match(self, engine):
        records = [{"val": None, "cat": "office", "temp": 30.0}]
        rule = _make_rule(conditions=[
            {"field_name": "cat", "condition_type": "equals", "value": "office"},
            {"field_name": "temp", "condition_type": "greater_than", "value": 25},
        ])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 1

    def test_multiple_conditions_partial_match_fails(self, engine):
        records = [{"val": None, "cat": "office", "temp": 20.0}]
        rule = _make_rule(conditions=[
            {"field_name": "cat", "condition_type": "equals", "value": "office"},
            {"field_name": "temp", "condition_type": "greater_than", "value": 25},
        ])
        result = engine.evaluate_rules(records, "val", [rule])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# lookup_imputation tests
# ---------------------------------------------------------------------------


class TestLookupImputation:
    def _make_lookup_table(self):
        entries = [
            LookupEntry(key="diesel", value=2.68),
            LookupEntry(key="petrol", value=2.31),
            LookupEntry(key="natural_gas", value=2.02),
        ]
        return LookupTable(
            name="emission_factors",
            key_column="fuel_type",
            target_column="emission_factor",
            entries=entries,
            default_value=1.0,
        )

    def test_direct_match(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert len(result) == 1
        assert result[0].imputed_value == 2.68
        assert result[0].strategy == ImputationStrategy.LOOKUP_TABLE

    def test_case_insensitive_match(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": None, "fuel_type": "DIESEL"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert result[0].imputed_value == 2.68

    def test_default_fallback(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": None, "fuel_type": "unknown_fuel"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert result[0].imputed_value == 1.0

    def test_direct_match_high_confidence(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert result[0].confidence == 0.92

    def test_default_lower_confidence(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": None, "fuel_type": "unknown"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert result[0].confidence == 0.65

    def test_no_missing_returns_empty(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": 2.0, "fuel_type": "diesel"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert result == []

    def test_provenance_hash(self, engine):
        table = self._make_lookup_table()
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.lookup_imputation(records, "emission_factor", table)
        assert len(result[0].provenance_hash) == 64


# ---------------------------------------------------------------------------
# regulatory_defaults tests
# ---------------------------------------------------------------------------


class TestRegulatoryDefaults:
    def test_ghg_protocol_diesel(self, engine):
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.regulatory_defaults(records, "emission_factor", "ghg_protocol")
        assert len(result) == 1
        assert result[0].imputed_value == 2.68
        assert result[0].strategy == ImputationStrategy.REGULATORY_DEFAULT

    def test_defra_diesel(self, engine):
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.regulatory_defaults(records, "emission_factor", "defra")
        assert len(result) == 1
        assert result[0].imputed_value == 2.71

    def test_epa_diesel(self, engine):
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.regulatory_defaults(records, "emission_factor", "epa")
        assert len(result) == 1
        assert result[0].imputed_value == 2.69

    def test_unknown_framework_returns_empty(self, engine):
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.regulatory_defaults(records, "emission_factor", "nonexistent")
        assert result == []

    def test_no_matching_field_returns_empty(self, engine):
        records = [{"emission_factor": None, "irrelevant": "data"}]
        result = engine.regulatory_defaults(records, "emission_factor", "ghg_protocol")
        assert result == []

    def test_confidence_is_moderate(self, engine):
        records = [{"emission_factor": None, "fuel_type": "diesel"}]
        result = engine.regulatory_defaults(records, "emission_factor", "ghg_protocol")
        assert result[0].confidence == 0.60

    def test_activity_type_match(self, engine):
        records = [{"emission_factor": None, "activity_type": "natural_gas"}]
        result = engine.regulatory_defaults(records, "emission_factor", "ghg_protocol")
        assert len(result) == 1
        assert result[0].imputed_value == 2.02

    def test_no_missing_returns_empty(self, engine):
        records = [{"emission_factor": 5.0, "fuel_type": "diesel"}]
        result = engine.regulatory_defaults(records, "emission_factor", "ghg_protocol")
        assert result == []


# ---------------------------------------------------------------------------
# create_rule tests
# ---------------------------------------------------------------------------


class TestCreateRule:
    def test_basic_creation(self, engine):
        rule = engine.create_rule(
            name="test_rule",
            conditions=[{"field_name": "cat", "condition_type": "equals", "value": "A"}],
            imputed_value=0.5,
            priority="high",
            justification="Test justification",
        )
        assert isinstance(rule, ImputationRule)
        assert rule.name == "test_rule"
        assert rule.priority == RulePriority.HIGH
        assert rule.impute_value == 0.5
        assert len(rule.conditions) == 1

    def test_rule_has_provenance(self, engine):
        rule = engine.create_rule("r", [], 1.0)
        assert len(rule.provenance_hash) == 64

    def test_rule_id_set(self, engine):
        rule = engine.create_rule("r", [], 1.0)
        assert len(rule.rule_id) > 0


# ---------------------------------------------------------------------------
# validate_rule tests
# ---------------------------------------------------------------------------


class TestValidateRule:
    def test_valid_rule(self, engine):
        rule = ImputationRule(
            name="valid",
            target_column="col",
            conditions=[
                RuleCondition(field_name="x", condition_type=RuleConditionType.EQUALS, value="A"),
            ],
            impute_value=10,
            priority=RulePriority.MEDIUM,
            provenance_hash="a" * 64,
        )
        result = engine.validate_rule(rule)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_name(self, engine):
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="name"):
            ImputationRule(
                name="",
                target_column="col",
                conditions=[],
                impute_value=1,
                priority=RulePriority.LOW,
                provenance_hash="a" * 64,
            )

    def test_missing_target_column(self, engine):
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="target_column"):
            ImputationRule(
                name="x",
                target_column="",
                conditions=[],
                impute_value=1,
                priority=RulePriority.LOW,
                provenance_hash="a" * 64,
            )

    def test_no_conditions_warning(self, engine):
        rule = ImputationRule(
            name="x",
            target_column="col",
            conditions=[],
            impute_value=1,
            priority=RulePriority.LOW,
            provenance_hash="a" * 64,
        )
        result = engine.validate_rule(rule)
        assert any("no conditions" in w.lower() for w in result["warnings"])

    def test_contradictory_conditions(self, engine):
        rule = ImputationRule(
            name="x",
            target_column="col",
            conditions=[
                RuleCondition(field_name="f", condition_type=RuleConditionType.EQUALS, value="A"),
                RuleCondition(field_name="f", condition_type=RuleConditionType.NOT_EQUALS, value="A"),
            ],
            impute_value=1,
            priority=RulePriority.MEDIUM,
            provenance_hash="a" * 64,
        )
        result = engine.validate_rule(rule)
        assert result["valid"] is False
        assert any("contradictory" in e.lower() for e in result["errors"])

    def test_invalid_regex(self, engine):
        rule = ImputationRule(
            name="x",
            target_column="col",
            conditions=[
                RuleCondition(field_name="f", condition_type=RuleConditionType.REGEX, value="[invalid"),
            ],
            impute_value=1,
            priority=RulePriority.LOW,
            provenance_hash="a" * 64,
        )
        result = engine.validate_rule(rule)
        assert result["valid"] is False

    def test_in_list_non_list_warning(self, engine):
        rule = ImputationRule(
            name="x",
            target_column="col",
            conditions=[
                RuleCondition(field_name="f", condition_type=RuleConditionType.IN_LIST, value="single"),
            ],
            impute_value=1,
            priority=RulePriority.LOW,
            provenance_hash="a" * 64,
        )
        result = engine.validate_rule(rule)
        assert any("IN_LIST" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# apply_rule_set tests
# ---------------------------------------------------------------------------


class TestApplyRuleSet:
    def test_multi_column(self, engine):
        records = [{"a": None, "b": None, "cat": "X"}]
        rule_a = _make_rule(name="ra", target_column="a", impute_value=10, conditions=[])
        rule_b = _make_rule(name="rb", target_column="b", impute_value=20, conditions=[])
        rule_set = {"a": [rule_a], "b": [rule_b]}
        result = engine.apply_rule_set(records, rule_set)
        assert "a" in result
        assert "b" in result
        assert len(result["a"]) == 1
        assert len(result["b"]) == 1

    def test_empty_records(self, engine):
        result = engine.apply_rule_set([], {"a": []})
        assert result == {"a": []}
