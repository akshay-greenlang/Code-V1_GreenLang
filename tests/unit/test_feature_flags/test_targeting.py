# -*- coding: utf-8 -*-
"""
Unit Tests for Feature Flag Targeting Subsystem - INFRA-008

Tests the three targeting components: PercentageRollout (consistent hashing),
SegmentMatcher (attribute-based conditions), and RuleEvaluator (priority-based
rule dispatch).
"""

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.infrastructure.feature_flags.models import (
    EvaluationContext,
    FlagRule,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.targeting.percentage import (
    PercentageRollout,
)
from greenlang.infrastructure.feature_flags.targeting.rules import RuleEvaluator
from greenlang.infrastructure.feature_flags.targeting.segments import SegmentMatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rollout():
    """Create a PercentageRollout instance."""
    return PercentageRollout()


@pytest.fixture
def matcher():
    """Create a SegmentMatcher instance."""
    return SegmentMatcher()


@pytest.fixture
def evaluator():
    """Create a RuleEvaluator instance."""
    return RuleEvaluator()


@pytest.fixture
def base_context():
    """Create a standard EvaluationContext for targeting tests."""
    return EvaluationContext(
        user_id="user-42",
        tenant_id="tenant-acme",
        environment="staging",
        user_segments=["enterprise", "early_adopter"],
        user_attributes={
            "plan_type": "enterprise",
            "seats": 100,
            "region": "us-east",
            "company_name": "Acme Corp",
        },
    )


# ---------------------------------------------------------------------------
# PercentageRollout Tests
# ---------------------------------------------------------------------------


class TestPercentageRollout:
    """Tests for PercentageRollout consistent hashing."""

    def test_zero_percentage_always_false(self, rollout):
        """A 0% rollout always returns False."""
        assert rollout.evaluate("flag-a", "user-1", 0.0) is False
        assert rollout.evaluate("flag-a", "user-2", 0.0) is False

    def test_hundred_percentage_always_true(self, rollout):
        """A 100% rollout always returns True."""
        assert rollout.evaluate("flag-a", "user-1", 100.0) is True
        assert rollout.evaluate("flag-a", "user-2", 100.0) is True

    def test_determinism_same_inputs(self, rollout):
        """Same (flag_key, user_id) pair always produces the same result."""
        results = [
            rollout.evaluate("enable-scope3", "user-42", 50.0)
            for _ in range(50)
        ]
        assert len(set(results)) == 1, "Percentage rollout must be deterministic"

    def test_different_users_get_different_results(self, rollout):
        """Different users should not all get the same result at 50%."""
        results = set()
        for i in range(200):
            result = rollout.evaluate("flag-a", f"user-{i}", 50.0)
            results.add(result)
        assert True in results and False in results, (
            "At 50% rollout, some users should be included and some excluded"
        )

    def test_hash_to_bucket_returns_valid_range(self, rollout):
        """_hash_to_bucket must return a value in [0, 99]."""
        for i in range(500):
            bucket = rollout._hash_to_bucket(f"test-input-{i}")
            assert 0 <= bucket <= 99, f"Bucket {bucket} out of range for input {i}"

    def test_hash_to_bucket_is_deterministic(self, rollout):
        """Same input always produces the same bucket."""
        bucket1 = rollout._hash_to_bucket("flag-a:user-42")
        bucket2 = rollout._hash_to_bucket("flag-a:user-42")
        assert bucket1 == bucket2

    def test_no_user_id_uses_random_bucket(self, rollout):
        """When user_id is None, each call may return a different result."""
        results = set()
        for _ in range(100):
            result = rollout.evaluate("flag-a", None, 50.0)
            results.add(result)
        # With 100 random calls at 50%, we should see both True and False
        assert len(results) == 2, "Anonymous users should get random bucketing"

    def test_empty_user_id_treated_as_anonymous(self, rollout):
        """An empty string user_id is treated the same as None (random)."""
        results = set()
        for _ in range(100):
            result = rollout.evaluate("flag-a", "", 50.0)
            results.add(result)
        assert len(results) == 2

    def test_get_variant_empty_list(self, rollout):
        """get_variant returns None when variants list is empty."""
        result = rollout.get_variant("flag-a", "user-42", [])
        assert result is None

    def test_get_variant_zero_weight(self, rollout):
        """get_variant returns None when all variants have zero weight."""
        variants = [
            FlagVariant(variant_key="control", flag_key="ab-test.ui", weight=0.0),
            FlagVariant(variant_key="treatment", flag_key="ab-test.ui", weight=0.0),
        ]
        result = rollout.get_variant("ab-test.ui", "user-42", variants)
        assert result is None

    def test_get_variant_single_variant(self, rollout):
        """get_variant always returns the only variant when there is one."""
        variants = [
            FlagVariant(
                variant_key="only-one",
                flag_key="ab-test.ui",
                weight=100.0,
            ),
        ]
        result = rollout.get_variant("ab-test.ui", "user-42", variants)
        assert result == "only-one"

    def test_get_variant_weighted_selection_deterministic(self, rollout):
        """Variant selection is deterministic for the same (flag, user) pair."""
        variants = [
            FlagVariant(variant_key="control", flag_key="ab-test.ui", weight=50.0),
            FlagVariant(variant_key="treatment", flag_key="ab-test.ui", weight=50.0),
        ]
        results = [
            rollout.get_variant("ab-test.ui", "user-42", variants)
            for _ in range(20)
        ]
        assert len(set(results)) == 1, "Variant selection must be deterministic"

    def test_get_variant_distribution_roughly_correct(self, rollout):
        """With many users, variant distribution should roughly match weights."""
        variants = [
            FlagVariant(variant_key="control", flag_key="ab-test.ui", weight=70.0),
            FlagVariant(variant_key="treatment", flag_key="ab-test.ui", weight=30.0),
        ]
        counts = {"control": 0, "treatment": 0}
        num_users = 1000
        for i in range(num_users):
            selected = rollout.get_variant("ab-test.ui", f"user-{i}", variants)
            counts[selected] += 1

        # Allow a generous margin for hash distribution variance
        control_pct = counts["control"] / num_users * 100
        assert 50.0 < control_pct < 90.0, (
            f"Expected ~70% control, got {control_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# SegmentMatcher Tests
# ---------------------------------------------------------------------------


class TestSegmentMatcher:
    """Tests for SegmentMatcher attribute-based targeting."""

    def test_empty_conditions_match_everything(self, matcher, base_context):
        """Empty conditions dict matches all contexts."""
        assert matcher.matches(base_context, {}) is True

    def test_eq_operator(self, matcher, base_context):
        """'eq' operator matches equal string values (case-insensitive)."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.plan_type", "operator": "eq", "value": "Enterprise"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_neq_operator(self, matcher, base_context):
        """'neq' operator matches non-equal values."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.plan_type", "operator": "neq", "value": "free"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_in_operator(self, matcher, base_context):
        """'in' operator checks membership in a list."""
        conditions = {
            "conditions": [
                {"attribute": "environment", "operator": "in", "value": ["staging", "prod"]},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_not_in_operator(self, matcher, base_context):
        """'not_in' operator checks non-membership in a list."""
        conditions = {
            "conditions": [
                {"attribute": "environment", "operator": "not_in", "value": ["prod"]},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_gt_operator(self, matcher, base_context):
        """'gt' operator performs numeric greater-than comparison."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.seats", "operator": "gt", "value": 50},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_lt_operator(self, matcher, base_context):
        """'lt' operator performs numeric less-than comparison."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.seats", "operator": "lt", "value": 200},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_contains_operator_string(self, matcher, base_context):
        """'contains' operator checks for substring match (case-insensitive)."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.company_name", "operator": "contains", "value": "acme"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_starts_with_operator(self, matcher, base_context):
        """'starts_with' operator checks string prefix."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.region", "operator": "starts_with", "value": "us"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_ends_with_operator(self, matcher, base_context):
        """'ends_with' operator checks string suffix."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.region", "operator": "ends_with", "value": "east"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_regex_operator(self, matcher, base_context):
        """'regex' operator matches a regular expression."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.region", "operator": "regex", "value": "^us-.*$"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_attribute_path_resolution_nested(self, matcher, base_context):
        """Dot-notation paths resolve nested user_attributes."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.plan_type", "operator": "eq", "value": "enterprise"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_attribute_path_direct_field(self, matcher, base_context):
        """Direct context fields (environment, user_id) are resolved."""
        conditions = {
            "conditions": [
                {"attribute": "environment", "operator": "eq", "value": "staging"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_and_logic_all_must_match(self, matcher, base_context):
        """AND conditions require all conditions to match."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.plan_type", "operator": "eq", "value": "enterprise"},
                {"attribute": "user_attributes.seats", "operator": "gt", "value": 50},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_and_logic_one_fails(self, matcher, base_context):
        """AND conditions fail if any single condition does not match."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.plan_type", "operator": "eq", "value": "enterprise"},
                {"attribute": "user_attributes.seats", "operator": "gt", "value": 500},
            ]
        }
        assert matcher.matches(base_context, conditions) is False

    def test_or_logic_at_least_one_matches(self, matcher, base_context):
        """OR conditions pass if at least one condition matches."""
        conditions = {
            "or": [
                {"attribute": "environment", "operator": "eq", "value": "prod"},
                {"attribute": "environment", "operator": "eq", "value": "staging"},
            ]
        }
        assert matcher.matches(base_context, conditions) is True

    def test_or_logic_none_match(self, matcher, base_context):
        """OR conditions fail if no condition matches."""
        conditions = {
            "or": [
                {"attribute": "environment", "operator": "eq", "value": "prod"},
                {"attribute": "environment", "operator": "eq", "value": "dev"},
            ]
        }
        assert matcher.matches(base_context, conditions) is False

    def test_missing_attribute_returns_false(self, matcher, base_context):
        """Conditions targeting a non-existent attribute return False."""
        conditions = {
            "conditions": [
                {"attribute": "user_attributes.nonexistent_field", "operator": "eq", "value": "x"},
            ]
        }
        assert matcher.matches(base_context, conditions) is False

    def test_unsupported_operator_returns_false(self, matcher, base_context):
        """An unsupported operator name returns False."""
        conditions = {
            "conditions": [
                {"attribute": "environment", "operator": "like", "value": "stag%"},
            ]
        }
        assert matcher.matches(base_context, conditions) is False

    def test_none_actual_value_with_eq(self, matcher):
        """eq comparison with None actual and non-None expected returns False."""
        ctx = EvaluationContext(user_id=None)
        conditions = {
            "conditions": [
                {"attribute": "user_id", "operator": "eq", "value": "some-user"},
            ]
        }
        assert matcher.matches(ctx, conditions) is False


# ---------------------------------------------------------------------------
# RuleEvaluator Tests
# ---------------------------------------------------------------------------


class TestRuleEvaluator:
    """Tests for RuleEvaluator priority-based rule dispatch."""

    def test_empty_rules_returns_none(self, evaluator, base_context):
        """No rules means no match."""
        assert evaluator.evaluate([], base_context) is None

    def test_user_list_rule_matches_user(self, evaluator, base_context):
        """A user_list rule matches when the user is in the list."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-42", "user-99"]},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is not None
        assert result.rule_id == rule.rule_id

    def test_user_list_rule_no_match(self, evaluator, base_context):
        """A user_list rule does not match when user is not in the list."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-99", "user-100"]},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is None

    def test_environment_rule_matches(self, evaluator, base_context):
        """An environment rule matches when context environment is listed."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="environment",
            priority=10,
            conditions={"environments": ["staging", "prod"]},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is not None

    def test_environment_rule_no_match(self, evaluator, base_context):
        """An environment rule does not match when environment is not listed."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="environment",
            priority=10,
            conditions={"environments": ["prod"]},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is None

    def test_segment_rule_matches(self, evaluator, base_context):
        """A segment rule matches when segment conditions are satisfied."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="segment",
            priority=10,
            conditions={
                "conditions": [
                    {"attribute": "user_attributes.plan_type", "operator": "eq", "value": "enterprise"},
                ]
            },
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is not None

    def test_percentage_rule_deterministic(self, evaluator, base_context):
        """A percentage rule is deterministic for the same context."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="percentage",
            priority=10,
            conditions={"percentage": 50.0},
        )
        results = [evaluator.evaluate([rule], base_context) for _ in range(20)]
        first_result = results[0]
        for r in results[1:]:
            # All should be the same (either all match or all None)
            assert (r is None) == (first_result is None)

    def test_percentage_rule_100_always_matches(self, evaluator, base_context):
        """A percentage rule at 100% always matches."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="percentage",
            priority=10,
            conditions={"percentage": 100.0},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is not None

    def test_disabled_rules_are_skipped(self, evaluator, base_context):
        """Disabled rules are not evaluated."""
        disabled_rule = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=1,
            conditions={"users": ["user-42"]},
            enabled=False,
        )
        enabled_rule = FlagRule(
            flag_key="test-flag",
            rule_type="environment",
            priority=100,
            conditions={"environments": ["staging"]},
            enabled=True,
        )
        result = evaluator.evaluate([disabled_rule, enabled_rule], base_context)
        assert result is not None
        assert result.rule_id == enabled_rule.rule_id

    def test_priority_ordering_lower_number_wins(self, evaluator, base_context):
        """Rules are evaluated in priority order; lower number wins."""
        low_priority = FlagRule(
            flag_key="test-flag",
            rule_type="environment",
            priority=100,
            conditions={"environments": ["staging"]},
        )
        high_priority = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=1,
            conditions={"users": ["user-42"]},
        )
        # Pass in reverse order to confirm sorting happens internally
        result = evaluator.evaluate([low_priority, high_priority], base_context)
        assert result is not None
        assert result.rule_id == high_priority.rule_id

    def test_first_matching_rule_wins(self, evaluator, base_context):
        """When multiple rules match, the highest priority (lowest number) wins."""
        rule_a = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-42"]},
        )
        rule_b = FlagRule(
            flag_key="test-flag",
            rule_type="environment",
            priority=20,
            conditions={"environments": ["staging"]},
        )
        result = evaluator.evaluate([rule_a, rule_b], base_context)
        assert result.rule_id == rule_a.rule_id

    def test_no_matching_rules_returns_none(self, evaluator, base_context):
        """When no rules match, None is returned."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-999"]},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is None

    def test_unknown_rule_type_returns_no_match(self, evaluator, base_context):
        """An unknown rule_type does not match (returns False internally)."""
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="unknown_type",
            priority=10,
            conditions={},
        )
        result = evaluator.evaluate([rule], base_context)
        assert result is None

    def test_user_list_case_insensitive(self, evaluator):
        """User list matching is case-insensitive."""
        ctx = EvaluationContext(user_id="User-42")
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-42"]},
        )
        result = evaluator.evaluate([rule], ctx)
        assert result is not None

    def test_user_list_no_user_id_returns_no_match(self, evaluator):
        """user_list rule does not match when user_id is None."""
        ctx = EvaluationContext(user_id=None)
        rule = FlagRule(
            flag_key="test-flag",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-42"]},
        )
        result = evaluator.evaluate([rule], ctx)
        assert result is None
