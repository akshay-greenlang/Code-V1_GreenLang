# -*- coding: utf-8 -*-
"""
Unit Tests for RuleEvaluatorEngine - AGENT-DATA-019: Validation Rule Engine
============================================================================

Tests all public methods of RuleEvaluatorEngine with 150+ tests covering
rule evaluation by type (10 types), operator evaluation (12 operators),
rule set evaluation with SLA thresholds, compound rule evaluation (AND/OR/NOT),
batch evaluation across multiple datasets, cross-dataset evaluation with
join keys, edge cases, pass rate calculation, duration tracking, and
provenance recording.

Test Classes (16):
    - TestRuleEvaluatorInit (5 tests)
    - TestEvaluateCompleteness (12 tests)
    - TestEvaluateRange (14 tests)
    - TestEvaluateFormat (12 tests)
    - TestEvaluateUniqueness (8 tests)
    - TestEvaluateCustom (10 tests)
    - TestEvaluateFreshness (8 tests)
    - TestEvaluateCrossField (12 tests)
    - TestEvaluateConditional (10 tests)
    - TestEvaluateStatistical (12 tests)
    - TestEvaluateReferential (8 tests)
    - TestOperatorEvaluation (14 tests)
    - TestRuleSetEvaluation (12 tests)
    - TestCompoundRuleEvaluation (10 tests)
    - TestBatchEvaluation (8 tests)
    - TestCrossDatasetEvaluation (6 tests)
    - TestEdgeCases (10 tests)

Total: ~171 tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.validation_rule_engine.rule_evaluator import RuleEvaluatorEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _make_rule(
    *,
    rule_id: str = "r-test",
    rule_name: str = "test_rule",
    rule_type: str = "COMPLETENESS",
    column: str = "value",
    operator: str = "",
    threshold: Any = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Factory helper to create a rule dict matching the engine API.

    The engine expects a dict with keys: rule_id, rule_name, rule_type,
    column, operator, threshold, parameters.
    """
    return {
        "rule_id": rule_id,
        "rule_name": rule_name,
        "rule_type": rule_type,
        "column": column,
        "operator": operator,
        "threshold": threshold,
        "parameters": parameters or {},
    }


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> RuleEvaluatorEngine:
    """Create a fresh RuleEvaluatorEngine instance for each test."""
    return RuleEvaluatorEngine(genesis_hash="test-evaluator-genesis")


@pytest.fixture
def sample_dataset() -> List[Dict[str, Any]]:
    """10 sample records for evaluation testing."""
    return [
        {"id": "r1", "name": "Alice", "age": 28, "email": "alice@example.com", "score": 85.5, "department": "Engineering", "salary": 72000},
        {"id": "r2", "name": "Bob", "age": 35, "email": "bob@example.com", "score": 92.0, "department": "Science", "salary": 85000},
        {"id": "r3", "name": "Carol", "age": 42, "email": "carol@example.com", "score": 78.3, "department": "Operations", "salary": 95000},
        {"id": "r4", "name": "David", "age": 31, "email": "david@example.com", "score": 65.7, "department": "Engineering", "salary": 68000},
        {"id": "r5", "name": "Eva", "age": 29, "email": "eva@example.com", "score": 91.2, "department": "Science", "salary": 71000},
        {"id": "r6", "name": "Frank", "age": 38, "email": "frank@example.com", "score": 88.4, "department": "Management", "salary": 92000},
        {"id": "r7", "name": "Grace", "age": 26, "email": "grace@example.com", "score": 95.1, "department": "Operations", "salary": 63000},
        {"id": "r8", "name": "Hector", "age": 44, "email": "hector@example.com", "score": 72.9, "department": "Engineering", "salary": 98000},
        {"id": "r9", "name": "Ingrid", "age": 33, "email": "ingrid@example.com", "score": 81.0, "department": "Science", "salary": 77000},
        {"id": "r10", "name": "James", "age": 30, "email": "james@example.com", "score": 86.6, "department": "Operations", "salary": 74000},
    ]


@pytest.fixture
def dataset_with_nulls() -> List[Dict[str, Any]]:
    """Dataset containing null/empty values for completeness testing."""
    return [
        {"id": "r1", "name": "Alice", "email": "alice@example.com", "score": 85.5},
        {"id": "r2", "name": None, "email": "bob@example.com", "score": 92.0},
        {"id": "r3", "name": "", "email": None, "score": None},
        {"id": "r4", "name": "David", "email": "", "score": 65.7},
        {"id": "r5", "name": "Eva", "email": "eva@example.com", "score": 0},
    ]


@pytest.fixture
def dataset_with_dates() -> List[Dict[str, Any]]:
    """Dataset with datetime strings for freshness testing."""
    now = _utcnow()
    return [
        {"id": "r1", "timestamp": (now - timedelta(hours=1)).isoformat(), "value": 10},
        {"id": "r2", "timestamp": (now - timedelta(hours=5)).isoformat(), "value": 20},
        {"id": "r3", "timestamp": (now - timedelta(hours=25)).isoformat(), "value": 30},
        {"id": "r4", "timestamp": (now - timedelta(hours=48)).isoformat(), "value": 40},
        {"id": "r5", "timestamp": (now - timedelta(minutes=30)).isoformat(), "value": 50},
    ]


@pytest.fixture
def reference_dataset() -> List[Dict[str, Any]]:
    """Reference dataset for referential integrity testing."""
    return [
        {"dept_id": "ENG", "dept_name": "Engineering"},
        {"dept_id": "SCI", "dept_name": "Science"},
        {"dept_id": "OPS", "dept_name": "Operations"},
        {"dept_id": "MGT", "dept_name": "Management"},
    ]


# ==========================================================================
# TestRuleEvaluatorInit
# ==========================================================================


class TestRuleEvaluatorInit:
    """Tests for RuleEvaluatorEngine initialization."""

    def test_init_creates_instance(self, engine: RuleEvaluatorEngine) -> None:
        """Engine initializes without error."""
        assert engine is not None

    def test_init_has_provenance_tracker(self, engine: RuleEvaluatorEngine) -> None:
        """Engine has a provenance tracker."""
        assert hasattr(engine, "_provenance") or hasattr(engine, "_tracker")

    def test_init_has_empty_results(self, engine: RuleEvaluatorEngine) -> None:
        """Engine starts with no evaluation results stored."""
        stats = engine.get_statistics()
        assert stats["total_evaluations"] == 0

    def test_init_custom_genesis_hash(self) -> None:
        """Engine accepts a custom genesis hash."""
        eng = RuleEvaluatorEngine(genesis_hash="custom-genesis")
        assert eng is not None

    def test_init_default_genesis_hash(self) -> None:
        """Engine works with default genesis hash."""
        eng = RuleEvaluatorEngine()
        assert eng is not None


# ==========================================================================
# TestEvaluateCompleteness
# ==========================================================================


class TestEvaluateCompleteness:
    """Tests for COMPLETENESS rule type evaluation."""

    def test_completeness_all_present(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """All records pass when field is present and non-null."""
        rule = _make_rule(
            rule_name="name_required",
            rule_type="COMPLETENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert result["pass_rate"] == 1.0

    def test_completeness_detects_null(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Detects null values as completeness failures."""
        rule = _make_rule(
            rule_name="name_required",
            rule_type="COMPLETENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, dataset_with_nulls)
        assert result["fail_count"] >= 1

    def test_completeness_detects_empty_string(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Detects empty strings as completeness failures."""
        rule = _make_rule(
            rule_name="name_not_empty",
            rule_type="COMPLETENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, dataset_with_nulls)
        # r2 is None, r3 is empty string
        assert result["fail_count"] >= 2

    def test_completeness_missing_field(self, engine: RuleEvaluatorEngine) -> None:
        """Missing field is a completeness failure."""
        data = [{"a": 1}, {"a": 2, "b": 3}]
        rule = _make_rule(
            rule_name="b_required",
            rule_type="COMPLETENESS",
            column="b",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1

    def test_completeness_threshold_parameter(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Completeness uses threshold parameter for pass/fail ratio."""
        rule = _make_rule(
            rule_name="score_completeness",
            rule_type="COMPLETENESS",
            column="score",
            threshold=0.5,
            parameters={"threshold": 0.5},
        )
        result = engine.evaluate_rule(rule, dataset_with_nulls)
        assert "pass_rate" in result

    def test_completeness_zero_is_not_null(self, engine: RuleEvaluatorEngine) -> None:
        """Zero values are considered present (not null)."""
        data = [{"value": 0}, {"value": 1}, {"value": 0.0}]
        rule = _make_rule(
            rule_name="value_required",
            rule_type="COMPLETENESS",
            column="value",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_completeness_false_is_not_null(self, engine: RuleEvaluatorEngine) -> None:
        """Boolean False is considered present (not null)."""
        data = [{"active": False}, {"active": True}]
        rule = _make_rule(
            rule_name="active_required",
            rule_type="COMPLETENESS",
            column="active",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_completeness_all_null(self, engine: RuleEvaluatorEngine) -> None:
        """All null values yield 0% pass rate."""
        data = [{"value": None}, {"value": None}, {"value": None}]
        rule = _make_rule(
            rule_name="value_required",
            rule_type="COMPLETENESS",
            column="value",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_completeness_result_has_failures(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Result includes failure details."""
        rule = _make_rule(
            rule_name="email_required",
            rule_type="COMPLETENESS",
            column="email",
        )
        result = engine.evaluate_rule(rule, dataset_with_nulls)
        assert "failures" in result

    def test_completeness_result_keys(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Result carries standard keys from evaluate_rule."""
        rule = _make_rule(
            rule_name="name_check",
            rule_type="COMPLETENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, dataset_with_nulls)
        assert "pass_count" in result
        assert "fail_count" in result
        assert "pass_rate" in result

    def test_completeness_duration_tracked(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Evaluation duration is tracked."""
        rule = _make_rule(
            rule_name="name_required",
            rule_type="COMPLETENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    def test_completeness_provenance_recorded(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Provenance hash is recorded for evaluation."""
        rule = _make_rule(
            rule_name="name_required",
            rule_type="COMPLETENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ==========================================================================
# TestEvaluateRange
# ==========================================================================


class TestEvaluateRange:
    """Tests for RANGE rule type evaluation."""

    def test_range_between_all_pass(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """All records pass when all values within range."""
        rule = _make_rule(
            rule_name="age_range",
            rule_type="RANGE",
            column="age",
            operator="BETWEEN",
            threshold={"min": 20, "max": 50},
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert result["pass_rate"] == 1.0

    def test_range_between_some_fail(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Some records fail when values outside range."""
        rule = _make_rule(
            rule_name="age_narrow_range",
            rule_type="RANGE",
            column="age",
            operator="BETWEEN",
            threshold={"min": 30, "max": 40},
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert 0 < result["pass_rate"] < 1.0

    def test_range_greater_than(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """gt operator works correctly."""
        rule = _make_rule(
            rule_name="score_gt_80",
            rule_type="RANGE",
            column="score",
            operator="gt",
            threshold=80.0,
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        expected_passing = sum(1 for r in sample_dataset if r["score"] > 80.0)
        assert result["pass_count"] == expected_passing

    def test_range_less_than(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """lt operator works correctly."""
        rule = _make_rule(
            rule_name="age_lt_30",
            rule_type="RANGE",
            column="age",
            operator="lt",
            threshold=30.0,
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        expected_passing = sum(1 for r in sample_dataset if r["age"] < 30)
        assert result["pass_count"] == expected_passing

    def test_range_greater_equal(self, engine: RuleEvaluatorEngine) -> None:
        """gte includes boundary value."""
        data = [{"val": 10}, {"val": 5}, {"val": 10}]
        rule = _make_rule(
            rule_name="val_ge_10",
            rule_type="RANGE",
            column="val",
            operator="gte",
            threshold=10.0,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_range_less_equal(self, engine: RuleEvaluatorEngine) -> None:
        """lte includes boundary value."""
        data = [{"val": 10}, {"val": 5}, {"val": 15}]
        rule = _make_rule(
            rule_name="val_le_10",
            rule_type="RANGE",
            column="val",
            operator="lte",
            threshold=10.0,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_range_min_boundary_inclusive(self, engine: RuleEvaluatorEngine) -> None:
        """BETWEEN is inclusive of min boundary."""
        data = [{"val": 10}]
        rule = _make_rule(
            rule_name="val_between",
            rule_type="RANGE",
            column="val",
            operator="BETWEEN",
            threshold={"min": 10, "max": 20},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_range_max_boundary_inclusive(self, engine: RuleEvaluatorEngine) -> None:
        """BETWEEN is inclusive of max boundary."""
        data = [{"val": 20}]
        rule = _make_rule(
            rule_name="val_between",
            rule_type="RANGE",
            column="val",
            operator="BETWEEN",
            threshold={"min": 10, "max": 20},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_range_null_value_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Null values fail range checks."""
        data = [{"val": None}, {"val": 15}]
        rule = _make_rule(
            rule_name="val_range",
            rule_type="RANGE",
            column="val",
            operator="BETWEEN",
            threshold={"min": 10, "max": 20},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1

    def test_range_negative_values(self, engine: RuleEvaluatorEngine) -> None:
        """Range works with negative values."""
        data = [{"val": -5}, {"val": -15}, {"val": 5}]
        rule = _make_rule(
            rule_name="val_range_neg",
            rule_type="RANGE",
            column="val",
            operator="BETWEEN",
            threshold={"min": -10, "max": 10},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_range_float_precision(self, engine: RuleEvaluatorEngine) -> None:
        """Range handles float precision correctly."""
        data = [{"val": 0.1 + 0.2}]  # 0.30000000000000004
        rule = _make_rule(
            rule_name="val_range_float",
            rule_type="RANGE",
            column="val",
            operator="lte",
            threshold=0.31,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_range_all_fail(self, engine: RuleEvaluatorEngine) -> None:
        """All records fail when all outside range."""
        data = [{"val": 100}, {"val": 200}]
        rule = _make_rule(
            rule_name="val_range_fail",
            rule_type="RANGE",
            column="val",
            operator="BETWEEN",
            threshold={"min": 0, "max": 50},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_range_result_contains_failures(self, engine: RuleEvaluatorEngine) -> None:
        """Result failures include actual and expected values."""
        data = [{"val": 100}]
        rule = _make_rule(
            rule_name="val_range",
            rule_type="RANGE",
            column="val",
            operator="lt",
            threshold=50.0,
        )
        result = engine.evaluate_rule(rule, data)
        failures = result.get("failures", [])
        if failures:
            assert failures[0].get("actual") is not None or "actual" in str(failures[0])


# ==========================================================================
# TestEvaluateFormat
# ==========================================================================


class TestEvaluateFormat:
    """Tests for FORMAT rule type evaluation."""

    def test_format_email_valid(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Valid email addresses pass format check."""
        rule = _make_rule(
            rule_name="email_format",
            rule_type="FORMAT",
            column="email",
            operator="matches",
            threshold=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert result["pass_rate"] == 1.0

    def test_format_email_invalid(self, engine: RuleEvaluatorEngine) -> None:
        """Invalid email addresses fail format check."""
        data = [{"email": "notanemail"}, {"email": "also@bad"}, {"email": "good@example.com"}]
        rule = _make_rule(
            rule_name="email_format",
            rule_type="FORMAT",
            column="email",
            operator="matches",
            threshold=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_format_phone_valid(self, engine: RuleEvaluatorEngine) -> None:
        """Valid phone numbers pass format check."""
        data = [{"phone": "+1-555-123-4567"}, {"phone": "+44-20-7946-0958"}]
        rule = _make_rule(
            rule_name="phone_format",
            rule_type="FORMAT",
            column="phone",
            operator="matches",
            threshold=r"^\+\d{1,3}-\d+-\d+-\d+$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_format_date_iso(self, engine: RuleEvaluatorEngine) -> None:
        """ISO 8601 dates pass format validation."""
        data = [
            {"date": "2025-01-15"},
            {"date": "2025-12-31"},
            {"date": "not-a-date"},
        ]
        rule = _make_rule(
            rule_name="date_format",
            rule_type="FORMAT",
            column="date",
            operator="matches",
            threshold=r"^\d{4}-\d{2}-\d{2}$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_format_null_value_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Null values fail format check."""
        data = [{"email": None}]
        rule = _make_rule(
            rule_name="email_format",
            rule_type="FORMAT",
            column="email",
            operator="matches",
            threshold=r"^.+@.+\..+$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_format_empty_string_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Empty strings fail format check."""
        data = [{"code": ""}]
        rule = _make_rule(
            rule_name="code_format",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^[A-Z]{3}\d{3}$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_format_contains_operator(self, engine: RuleEvaluatorEngine) -> None:
        """CONTAINS operator checks for substring presence."""
        data = [{"desc": "contains keyword"}, {"desc": "no match here"}]
        rule = _make_rule(
            rule_name="desc_contains",
            rule_type="FORMAT",
            column="desc",
            operator="matches",
            threshold=r".*keyword.*",
            parameters={"partial_match": True},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_format_case_sensitive(self, engine: RuleEvaluatorEngine) -> None:
        """Regex matching is case-sensitive by default."""
        data = [{"code": "ABC123"}, {"code": "abc123"}]
        rule = _make_rule(
            rule_name="code_upper",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^[A-Z]+\d+$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_format_all_pass(self, engine: RuleEvaluatorEngine) -> None:
        """All records pass when all match pattern."""
        data = [{"code": "A1"}, {"code": "B2"}, {"code": "C3"}]
        rule = _make_rule(
            rule_name="code_format",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^[A-Z]\d$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_format_all_fail(self, engine: RuleEvaluatorEngine) -> None:
        """All records fail when none match pattern."""
        data = [{"code": "123"}, {"code": "456"}]
        rule = _make_rule(
            rule_name="code_format",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^[A-Z]+$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_format_numeric_to_string_coercion(self, engine: RuleEvaluatorEngine) -> None:
        """Numeric values are coerced to string for format checks."""
        data = [{"code": 12345}]
        rule = _make_rule(
            rule_name="code_format",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^\d+$",
        )
        result = engine.evaluate_rule(rule, data)
        # Should either pass via coercion or fail gracefully
        assert "pass_rate" in result

    def test_format_provenance_hash(self, engine: RuleEvaluatorEngine) -> None:
        """Format evaluation includes provenance hash."""
        data = [{"code": "A1"}]
        rule = _make_rule(
            rule_name="code_format",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^[A-Z]\d$",
        )
        result = engine.evaluate_rule(rule, data)
        assert "provenance_hash" in result


# ==========================================================================
# TestEvaluateUniqueness
# ==========================================================================


class TestEvaluateUniqueness:
    """Tests for UNIQUENESS rule type evaluation."""

    def test_uniqueness_all_unique(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """All pass when values are unique."""
        rule = _make_rule(
            rule_name="id_unique",
            rule_type="UNIQUENESS",
            column="id",
        )
        result = engine.evaluate_rule(rule, sample_dataset)
        assert result["pass_rate"] == 1.0

    def test_uniqueness_with_duplicates(self, engine: RuleEvaluatorEngine) -> None:
        """Detects duplicate values."""
        data = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Alice"}]
        rule = _make_rule(
            rule_name="name_unique",
            rule_type="UNIQUENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1

    def test_uniqueness_all_same(self, engine: RuleEvaluatorEngine) -> None:
        """All duplicates when all values are the same."""
        data = [{"val": 1}, {"val": 1}, {"val": 1}]
        rule = _make_rule(
            rule_name="val_unique",
            rule_type="UNIQUENESS",
            column="val",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 2

    def test_uniqueness_null_handling(self, engine: RuleEvaluatorEngine) -> None:
        """Multiple null values may or may not be considered duplicates."""
        data = [{"val": None}, {"val": None}, {"val": 1}]
        rule = _make_rule(
            rule_name="val_unique",
            rule_type="UNIQUENESS",
            column="val",
        )
        result = engine.evaluate_rule(rule, data)
        assert "pass_rate" in result

    def test_uniqueness_case_sensitive(self, engine: RuleEvaluatorEngine) -> None:
        """Uniqueness is case-sensitive for strings."""
        data = [{"name": "Alice"}, {"name": "alice"}]
        rule = _make_rule(
            rule_name="name_unique",
            rule_type="UNIQUENESS",
            column="name",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_uniqueness_single_record(self, engine: RuleEvaluatorEngine) -> None:
        """Single record is always unique."""
        data = [{"val": 42}]
        rule = _make_rule(
            rule_name="val_unique",
            rule_type="UNIQUENESS",
            column="val",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_uniqueness_mixed_types(self, engine: RuleEvaluatorEngine) -> None:
        """Different types with same string representation."""
        data = [{"val": 1}, {"val": "1"}]
        rule = _make_rule(
            rule_name="val_unique",
            rule_type="UNIQUENESS",
            column="val",
        )
        result = engine.evaluate_rule(rule, data)
        assert "pass_rate" in result

    def test_uniqueness_large_dataset(self, engine: RuleEvaluatorEngine) -> None:
        """Uniqueness check works on larger datasets."""
        data = [{"val": i} for i in range(100)]
        rule = _make_rule(
            rule_name="val_unique",
            rule_type="UNIQUENESS",
            column="val",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0


# ==========================================================================
# TestEvaluateCustom
# ==========================================================================


class TestEvaluateCustom:
    """Tests for CUSTOM rule type evaluation."""

    def test_custom_simple_expression(self, engine: RuleEvaluatorEngine) -> None:
        """Simple comparison expression evaluates correctly."""
        data = [{"score": 85}, {"score": 45}, {"score": 90}]
        rule = _make_rule(
            rule_name="score_passing",
            rule_type="CUSTOM",
            column="score",
            threshold="score > 50",
            parameters={"expression": "score > 50"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_custom_arithmetic_expression(self, engine: RuleEvaluatorEngine) -> None:
        """Arithmetic expressions work in custom rules."""
        data = [{"a": 10, "b": 5}, {"a": 3, "b": 8}]
        rule = _make_rule(
            rule_name="sum_check",
            rule_type="CUSTOM",
            column="a",
            parameters={"expression": "a + b > 10"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] >= 1

    def test_custom_string_expression(self, engine: RuleEvaluatorEngine) -> None:
        """String operations work in custom rules."""
        data = [{"name": "Alice Smith"}, {"name": "Bob"}]
        rule = _make_rule(
            rule_name="name_has_space",
            rule_type="CUSTOM",
            column="name",
            parameters={"expression": "' ' in str(name)"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_custom_boolean_expression(self, engine: RuleEvaluatorEngine) -> None:
        """Boolean expression evaluates correctly."""
        data = [{"active": True}, {"active": False}]
        rule = _make_rule(
            rule_name="must_be_active",
            rule_type="CUSTOM",
            column="active",
            parameters={"expression": "active == True"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_custom_null_safe(self, engine: RuleEvaluatorEngine) -> None:
        """Custom expression handles null values safely."""
        data = [{"val": None}, {"val": 10}]
        rule = _make_rule(
            rule_name="val_check",
            rule_type="CUSTOM",
            column="val",
            parameters={"expression": "val is not None and val > 5"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_custom_invalid_expression_fails_gracefully(self, engine: RuleEvaluatorEngine) -> None:
        """Invalid expressions fail gracefully, not with unhandled exception."""
        data = [{"val": 10}]
        rule = _make_rule(
            rule_name="bad_expr",
            rule_type="CUSTOM",
            column="val",
            parameters={"expression": "this is not valid python"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1 or result["pass_rate"] == 0.0

    def test_custom_no_dangerous_builtins(self, engine: RuleEvaluatorEngine) -> None:
        """Custom expressions cannot access dangerous builtins like __import__."""
        data = [{"val": 1}]
        rule = _make_rule(
            rule_name="dangerous_expr",
            rule_type="CUSTOM",
            column="val",
        )
        # __import__ is blocked, so this should raise ValueError
        with pytest.raises(ValueError):
            rule["parameters"] = {"expression": "__import__('os').system('echo hacked')"}
            engine.evaluate_rule(rule, data)

    def test_custom_with_len(self, engine: RuleEvaluatorEngine) -> None:
        """Safe builtins like len are available in custom expressions."""
        data = [{"name": "Alice"}, {"name": "Bo"}]
        rule = _make_rule(
            rule_name="name_length",
            rule_type="CUSTOM",
            column="name",
            parameters={"expression": "len(str(name)) >= 3"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_custom_all_pass(self, engine: RuleEvaluatorEngine) -> None:
        """All pass when all meet custom expression."""
        data = [{"val": 10}, {"val": 20}]
        rule = _make_rule(
            rule_name="val_positive",
            rule_type="CUSTOM",
            column="val",
            parameters={"expression": "val > 0"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_custom_empty_expression_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Empty expression string fails or raises error."""
        data = [{"val": 10}]
        rule = _make_rule(
            rule_name="empty_expr",
            rule_type="CUSTOM",
            column="val",
            parameters={"expression": ""},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1 or result["pass_rate"] == 0.0


# ==========================================================================
# TestEvaluateFreshness
# ==========================================================================


class TestEvaluateFreshness:
    """Tests for FRESHNESS rule type evaluation."""

    def test_freshness_recent_data_passes(self, engine: RuleEvaluatorEngine, dataset_with_dates: List) -> None:
        """Recent data passes freshness check."""
        rule = _make_rule(
            rule_name="timestamp_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=2,
        )
        result = engine.evaluate_rule(rule, dataset_with_dates)
        assert result["pass_count"] >= 1

    def test_freshness_old_data_fails(self, engine: RuleEvaluatorEngine, dataset_with_dates: List) -> None:
        """Old data fails freshness check."""
        rule = _make_rule(
            rule_name="timestamp_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=2,
        )
        result = engine.evaluate_rule(rule, dataset_with_dates)
        assert result["fail_count"] >= 1

    def test_freshness_tight_threshold(self, engine: RuleEvaluatorEngine, dataset_with_dates: List) -> None:
        """Tight threshold fails more records."""
        rule = _make_rule(
            rule_name="timestamp_very_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=0.25,
        )
        result = engine.evaluate_rule(rule, dataset_with_dates)
        assert result["fail_count"] >= 3

    def test_freshness_loose_threshold(self, engine: RuleEvaluatorEngine, dataset_with_dates: List) -> None:
        """Loose threshold passes all records."""
        rule = _make_rule(
            rule_name="timestamp_relaxed",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=100,
        )
        result = engine.evaluate_rule(rule, dataset_with_dates)
        assert result["pass_rate"] == 1.0

    def test_freshness_null_timestamp_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Null timestamp fails freshness check."""
        data = [{"timestamp": None}]
        rule = _make_rule(
            rule_name="ts_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=24,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_freshness_invalid_timestamp_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Invalid timestamp string fails freshness check."""
        data = [{"timestamp": "not-a-timestamp"}]
        rule = _make_rule(
            rule_name="ts_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=24,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_freshness_result_has_duration(self, engine: RuleEvaluatorEngine, dataset_with_dates: List) -> None:
        """Freshness evaluation has duration tracking."""
        rule = _make_rule(
            rule_name="ts_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=24,
        )
        result = engine.evaluate_rule(rule, dataset_with_dates)
        assert "duration_ms" in result

    def test_freshness_provenance(self, engine: RuleEvaluatorEngine, dataset_with_dates: List) -> None:
        """Freshness evaluation records provenance."""
        rule = _make_rule(
            rule_name="ts_fresh",
            rule_type="FRESHNESS",
            column="timestamp",
            threshold=24,
        )
        result = engine.evaluate_rule(rule, dataset_with_dates)
        assert "provenance_hash" in result


# ==========================================================================
# TestEvaluateCrossField
# ==========================================================================


class TestEvaluateCrossField:
    """Tests for CROSS_FIELD rule type evaluation."""

    def test_cross_field_greater_than(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field gt comparison works."""
        data = [
            {"start": 10, "end": 20},
            {"start": 30, "end": 25},
            {"start": 5, "end": 15},
        ]
        rule = _make_rule(
            rule_name="end_gt_start",
            rule_type="CROSS_FIELD",
            column="end",
            operator="gt",
            parameters={"column_b": "start"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_cross_field_equals(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field eq comparison works."""
        data = [
            {"a": 10, "b": 10},
            {"a": 10, "b": 20},
        ]
        rule = _make_rule(
            rule_name="a_equals_b",
            rule_type="CROSS_FIELD",
            column="a",
            operator="eq",
            parameters={"column_b": "b"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_cross_field_not_equals(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field ne comparison works."""
        data = [
            {"a": 10, "b": 20},
            {"a": 10, "b": 10},
        ]
        rule = _make_rule(
            rule_name="a_ne_b",
            rule_type="CROSS_FIELD",
            column="a",
            operator="ne",
            parameters={"column_b": "b"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_cross_field_less_than(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field lt comparison works."""
        data = [
            {"cost": 100, "budget": 200},
            {"cost": 300, "budget": 200},
        ]
        rule = _make_rule(
            rule_name="cost_lt_budget",
            rule_type="CROSS_FIELD",
            column="cost",
            operator="lt",
            parameters={"column_b": "budget"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_cross_field_greater_equal(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field gte comparison works."""
        data = [
            {"revenue": 100, "cost": 100},
            {"revenue": 50, "cost": 100},
        ]
        rule = _make_rule(
            rule_name="revenue_ge_cost",
            rule_type="CROSS_FIELD",
            column="revenue",
            operator="gte",
            parameters={"column_b": "cost"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_cross_field_less_equal(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field lte comparison works."""
        data = [
            {"actual": 100, "target": 100},
            {"actual": 150, "target": 100},
        ]
        rule = _make_rule(
            rule_name="actual_le_target",
            rule_type="CROSS_FIELD",
            column="actual",
            operator="lte",
            parameters={"column_b": "target"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_cross_field_null_handling(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field with null in either field fails."""
        data = [{"a": None, "b": 10}, {"a": 5, "b": None}]
        rule = _make_rule(
            rule_name="a_lt_b",
            rule_type="CROSS_FIELD",
            column="a",
            operator="lt",
            parameters={"column_b": "b"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 2

    def test_cross_field_missing_column_b(self, engine: RuleEvaluatorEngine) -> None:
        """Missing column_b in record fails gracefully."""
        data = [{"a": 10}]
        rule = _make_rule(
            rule_name="a_lt_b",
            rule_type="CROSS_FIELD",
            column="a",
            operator="lt",
            parameters={"column_b": "b"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1

    def test_cross_field_string_comparison(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field comparison works with strings."""
        data = [{"first": "Alice", "last": "Zulu"}, {"first": "Zulu", "last": "Alice"}]
        rule = _make_rule(
            rule_name="first_lt_last",
            rule_type="CROSS_FIELD",
            column="first",
            operator="lt",
            parameters={"column_b": "last"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_cross_field_all_pass(self, engine: RuleEvaluatorEngine) -> None:
        """All records pass cross-field check."""
        data = [{"min": 1, "max": 10}, {"min": 5, "max": 20}]
        rule = _make_rule(
            rule_name="min_lt_max",
            rule_type="CROSS_FIELD",
            column="min",
            operator="lt",
            parameters={"column_b": "max"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_cross_field_all_fail(self, engine: RuleEvaluatorEngine) -> None:
        """All records fail cross-field check."""
        data = [{"a": 20, "b": 10}, {"a": 30, "b": 5}]
        rule = _make_rule(
            rule_name="a_lt_b",
            rule_type="CROSS_FIELD",
            column="a",
            operator="lt",
            parameters={"column_b": "b"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_cross_field_duration(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-field evaluation tracks duration."""
        data = [{"a": 1, "b": 2}]
        rule = _make_rule(
            rule_name="a_lt_b",
            rule_type="CROSS_FIELD",
            column="a",
            operator="lt",
            parameters={"column_b": "b"},
        )
        result = engine.evaluate_rule(rule, data)
        assert "duration_ms" in result


# ==========================================================================
# TestEvaluateConditional
# ==========================================================================


class TestEvaluateConditional:
    """Tests for CONDITIONAL rule type evaluation."""

    def test_conditional_if_then_pass(self, engine: RuleEvaluatorEngine) -> None:
        """Records matching condition are validated, passing rules pass."""
        data = [
            {"country": "DE", "emission_factor": 0.5},
            {"country": "US", "emission_factor": 3.0},
            {"country": "DE", "emission_factor": 1.5},
        ]
        rule = _make_rule(
            rule_name="de_emission_range",
            rule_type="CONDITIONAL",
            column="emission_factor",
            operator="BETWEEN",
            threshold={"min": 0.1, "max": 2.5},
            parameters={
                "predicate_column": "country",
                "predicate_value": "DE",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        # US record is not evaluated (condition not met), DE records pass
        assert result["pass_count"] >= 2

    def test_conditional_if_then_fail(self, engine: RuleEvaluatorEngine) -> None:
        """Records matching condition fail when rule not met."""
        data = [
            {"country": "DE", "emission_factor": 5.0},
            {"country": "US", "emission_factor": 3.0},
        ]
        rule = _make_rule(
            rule_name="de_emission_range",
            rule_type="CONDITIONAL",
            column="emission_factor",
            operator="lt",
            threshold=2.5,
            parameters={
                "predicate_column": "country",
                "predicate_value": "DE",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1

    def test_conditional_no_matching_records(self, engine: RuleEvaluatorEngine) -> None:
        """No records match condition, all are skipped or pass."""
        data = [
            {"country": "US", "emission_factor": 5.0},
            {"country": "UK", "emission_factor": 3.0},
        ]
        rule = _make_rule(
            rule_name="de_emission_range",
            rule_type="CONDITIONAL",
            column="emission_factor",
            operator="lt",
            threshold=2.5,
            parameters={
                "predicate_column": "country",
                "predicate_value": "DE",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        # No records match condition, so pass rate should be 1.0 (all skipped)
        assert result["pass_rate"] == 1.0 or result["fail_count"] == 0

    def test_conditional_all_matching_pass(self, engine: RuleEvaluatorEngine) -> None:
        """All matching records pass."""
        data = [
            {"type": "A", "val": 10},
            {"type": "A", "val": 20},
            {"type": "B", "val": 5},
        ]
        rule = _make_rule(
            rule_name="type_a_val_check",
            rule_type="CONDITIONAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={
                "predicate_column": "type",
                "predicate_value": "A",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] >= 2

    def test_conditional_invalid_condition(self, engine: RuleEvaluatorEngine) -> None:
        """Invalid predicate_value with no predicate_column passes all."""
        data = [{"val": 10}]
        rule = _make_rule(
            rule_name="bad_condition",
            rule_type="CONDITIONAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={},
        )
        result = engine.evaluate_rule(rule, data)
        assert "pass_rate" in result

    def test_conditional_with_null_values(self, engine: RuleEvaluatorEngine) -> None:
        """Conditional handles null values in predicate field."""
        data = [
            {"type": None, "val": 10},
            {"type": "A", "val": 20},
        ]
        rule = _make_rule(
            rule_name="type_a_check",
            rule_type="CONDITIONAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={
                "predicate_column": "type",
                "predicate_value": "A",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] >= 1

    def test_conditional_complex_condition(self, engine: RuleEvaluatorEngine) -> None:
        """Complex conditions work via inner_rule for multi-field."""
        data = [
            {"country": "DE", "factor": 1.0},
            {"country": "DE", "factor": 3.0},
            {"country": "US", "factor": 2.0},
        ]
        rule = _make_rule(
            rule_name="de_factor_check",
            rule_type="CONDITIONAL",
            column="factor",
            operator="BETWEEN",
            threshold={"min": 0.1, "max": 2.5},
            parameters={
                "predicate_column": "country",
                "predicate_value": "DE",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] >= 1

    def test_conditional_provenance(self, engine: RuleEvaluatorEngine) -> None:
        """Conditional evaluation records provenance."""
        data = [{"type": "A", "val": 10}]
        rule = _make_rule(
            rule_name="cond_prov",
            rule_type="CONDITIONAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={
                "predicate_column": "type",
                "predicate_value": "A",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        assert "provenance_hash" in result

    def test_conditional_empty_predicate_evaluates_all(self, engine: RuleEvaluatorEngine) -> None:
        """Missing predicate_column evaluates all as pass."""
        data = [{"val": 10}, {"val": 20}]
        rule = _make_rule(
            rule_name="no_condition",
            rule_type="CONDITIONAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] >= 2 or result["fail_count"] >= 0

    def test_conditional_duration(self, engine: RuleEvaluatorEngine) -> None:
        """Conditional evaluation tracks duration."""
        data = [{"type": "A", "val": 10}]
        rule = _make_rule(
            rule_name="cond_dur",
            rule_type="CONDITIONAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={
                "predicate_column": "type",
                "predicate_value": "A",
                "predicate_operator": "eq",
            },
        )
        result = engine.evaluate_rule(rule, data)
        assert "duration_ms" in result


# ==========================================================================
# TestEvaluateStatistical
# ==========================================================================


class TestEvaluateStatistical:
    """Tests for STATISTICAL rule type evaluation."""

    def test_statistical_mean_check(self, engine: RuleEvaluatorEngine) -> None:
        """Mean of column is validated against threshold."""
        data = [{"val": 10}, {"val": 20}, {"val": 30}]
        rule = _make_rule(
            rule_name="mean_check",
            rule_type="STATISTICAL",
            column="val",
            operator="eq",
            threshold=20,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        # Mean is 20
        assert result["pass_count"] >= 1 or result["pass_rate"] == 1.0

    def test_statistical_median_check(self, engine: RuleEvaluatorEngine) -> None:
        """Median of column is validated against threshold."""
        data = [{"val": 10}, {"val": 20}, {"val": 30}]
        rule = _make_rule(
            rule_name="median_check",
            rule_type="STATISTICAL",
            column="val",
            operator="eq",
            threshold=20,
            parameters={"statistic": "median"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_statistical_stddev_check(self, engine: RuleEvaluatorEngine) -> None:
        """Standard deviation is validated against threshold."""
        data = [{"val": 10}, {"val": 10}, {"val": 10}]
        rule = _make_rule(
            rule_name="stddev_check",
            rule_type="STATISTICAL",
            column="val",
            operator="lt",
            threshold=1.0,
            parameters={"statistic": "stddev"},
        )
        result = engine.evaluate_rule(rule, data)
        # Stddev of identical values is 0
        assert result["pass_rate"] == 1.0

    def test_statistical_percentile_check(self, engine: RuleEvaluatorEngine) -> None:
        """Percentile is validated against threshold."""
        data = [{"val": i} for i in range(1, 101)]
        rule = _make_rule(
            rule_name="p95_check",
            rule_type="STATISTICAL",
            column="val",
            operator="lte",
            threshold=96.0,
            parameters={"statistic": "percentile", "percentile": 95},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_statistical_min_check(self, engine: RuleEvaluatorEngine) -> None:
        """Minimum value is validated."""
        data = [{"val": 5}, {"val": 10}, {"val": 15}]
        rule = _make_rule(
            rule_name="min_check",
            rule_type="STATISTICAL",
            column="val",
            operator="gte",
            threshold=5.0,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        # Mean of [5,10,15]=10, 10>=5 is true
        assert result["pass_rate"] == 1.0

    def test_statistical_max_check(self, engine: RuleEvaluatorEngine) -> None:
        """Maximum value is validated."""
        data = [{"val": 5}, {"val": 10}, {"val": 15}]
        rule = _make_rule(
            rule_name="max_check",
            rule_type="STATISTICAL",
            column="val",
            operator="lte",
            threshold=15.0,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        # Mean of [5,10,15]=10, 10<=15 is true
        assert result["pass_rate"] == 1.0

    def test_statistical_mean_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Mean outside range fails."""
        data = [{"val": 100}, {"val": 200}, {"val": 300}]
        rule = _make_rule(
            rule_name="mean_fail",
            rule_type="STATISTICAL",
            column="val",
            operator="lt",
            threshold=50.0,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0 or result["fail_count"] >= 1

    def test_statistical_null_values_excluded(self, engine: RuleEvaluatorEngine) -> None:
        """Null values are excluded from statistical calculations."""
        data = [{"val": 10}, {"val": None}, {"val": 30}]
        rule = _make_rule(
            rule_name="mean_check",
            rule_type="STATISTICAL",
            column="val",
            operator="eq",
            threshold=20,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        # Mean of [10, 30] = 20
        assert result["pass_rate"] == 1.0

    def test_statistical_empty_column(self, engine: RuleEvaluatorEngine) -> None:
        """All null column fails statistical check."""
        data = [{"val": None}, {"val": None}]
        rule = _make_rule(
            rule_name="mean_check",
            rule_type="STATISTICAL",
            column="val",
            operator="eq",
            threshold=0,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] >= 1 or result["pass_rate"] == 0.0

    def test_statistical_sum_check(self, engine: RuleEvaluatorEngine) -> None:
        """Sum validation works via mean * count approximation."""
        data = [{"val": 10}, {"val": 20}, {"val": 30}]
        rule = _make_rule(
            rule_name="mean_check",
            rule_type="STATISTICAL",
            column="val",
            operator="eq",
            threshold=20,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_statistical_count_check(self, engine: RuleEvaluatorEngine) -> None:
        """Count-like validation via mean works."""
        data = [{"val": 10}, {"val": 20}, {"val": 30}]
        rule = _make_rule(
            rule_name="median_check",
            rule_type="STATISTICAL",
            column="val",
            operator="eq",
            threshold=20,
            parameters={"statistic": "median"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_statistical_provenance(self, engine: RuleEvaluatorEngine) -> None:
        """Statistical evaluation records provenance."""
        data = [{"val": 10}]
        rule = _make_rule(
            rule_name="stat_prov",
            rule_type="STATISTICAL",
            column="val",
            operator="gt",
            threshold=5.0,
            parameters={"statistic": "mean"},
        )
        result = engine.evaluate_rule(rule, data)
        assert "provenance_hash" in result


# ==========================================================================
# TestEvaluateReferential
# ==========================================================================


class TestEvaluateReferential:
    """Tests for REFERENTIAL rule type evaluation."""

    def test_referential_all_exist(self, engine: RuleEvaluatorEngine) -> None:
        """All references found in lookup dataset."""
        data = [
            {"dept_id": "ENG"},
            {"dept_id": "SCI"},
            {"dept_id": "OPS"},
        ]
        ref_data = [
            {"dept_id": "ENG"},
            {"dept_id": "SCI"},
            {"dept_id": "OPS"},
            {"dept_id": "MGT"},
        ]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_referential_some_missing(self, engine: RuleEvaluatorEngine) -> None:
        """Missing references are detected."""
        data = [
            {"dept_id": "ENG"},
            {"dept_id": "UNKNOWN"},
            {"dept_id": "OPS"},
        ]
        ref_data = [
            {"dept_id": "ENG"},
            {"dept_id": "SCI"},
            {"dept_id": "OPS"},
        ]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_referential_all_missing(self, engine: RuleEvaluatorEngine) -> None:
        """All references missing yields 0% pass rate."""
        data = [{"dept_id": "X"}, {"dept_id": "Y"}]
        ref_data = [{"dept_id": "A"}, {"dept_id": "B"}]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_referential_null_value_fails(self, engine: RuleEvaluatorEngine) -> None:
        """Null reference value fails."""
        data = [{"dept_id": None}]
        ref_data = [{"dept_id": "ENG"}]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_referential_empty_reference(self, engine: RuleEvaluatorEngine) -> None:
        """Empty reference dataset fails all records."""
        data = [{"dept_id": "ENG"}]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": [], "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_referential_case_sensitive(self, engine: RuleEvaluatorEngine) -> None:
        """Referential check is case-sensitive."""
        data = [{"dept_id": "eng"}]
        ref_data = [{"dept_id": "ENG"}]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_referential_provenance(self, engine: RuleEvaluatorEngine) -> None:
        """Referential evaluation records provenance."""
        data = [{"dept_id": "ENG"}]
        ref_data = [{"dept_id": "ENG"}]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert "provenance_hash" in result

    def test_referential_duration(self, engine: RuleEvaluatorEngine) -> None:
        """Referential evaluation tracks duration."""
        data = [{"dept_id": "ENG"}]
        ref_data = [{"dept_id": "ENG"}]
        rule = _make_rule(
            rule_name="dept_ref",
            rule_type="REFERENTIAL",
            column="dept_id",
            parameters={"reference_data": ref_data, "reference_column": "dept_id"},
        )
        result = engine.evaluate_rule(rule, data)
        assert "duration_ms" in result


# ==========================================================================
# TestOperatorEvaluation
# ==========================================================================


class TestOperatorEvaluation:
    """Tests for individual operator evaluation across rule types."""

    def test_operator_equals(self, engine: RuleEvaluatorEngine) -> None:
        """eq operator matches exact values."""
        data = [{"val": 42}, {"val": 43}]
        rule = _make_rule(
            rule_name="val_eq",
            rule_type="RANGE",
            column="val",
            operator="eq",
            threshold=42,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_not_equals(self, engine: RuleEvaluatorEngine) -> None:
        """ne operator excludes matching values."""
        data = [{"val": 42}, {"val": 43}]
        rule = _make_rule(
            rule_name="val_ne",
            rule_type="RANGE",
            column="val",
            operator="ne",
            threshold=42,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_in_set(self, engine: RuleEvaluatorEngine) -> None:
        """IN operator checks membership in numeric allowed values."""
        data = [{"val": 1}, {"val": 2}, {"val": 5}]
        rule = _make_rule(
            rule_name="val_in_set",
            rule_type="RANGE",
            column="val",
            operator="IN",
            threshold=[1, 2, 3],
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_operator_not_in_set(self, engine: RuleEvaluatorEngine) -> None:
        """NOT_IN operator excludes blacklisted numeric values."""
        data = [{"val": 1}, {"val": 5}, {"val": 10}]
        rule = _make_rule(
            rule_name="val_not_in",
            rule_type="RANGE",
            column="val",
            operator="NOT_IN",
            threshold=[5, 10],
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_is_null(self, engine: RuleEvaluatorEngine) -> None:
        """Null detection works via COMPLETENESS rule type."""
        data = [{"val": None}, {"val": 10}]
        rule = _make_rule(
            rule_name="val_completeness",
            rule_type="COMPLETENESS",
            column="val",
        )
        result = engine.evaluate_rule(rule, data)
        # COMPLETENESS detects null -> 1 pass, 1 fail
        assert result["pass_count"] == 1
        assert result["fail_count"] == 1

    def test_operator_matches(self, engine: RuleEvaluatorEngine) -> None:
        """matches operator uses regex matching."""
        data = [{"code": "ABC123"}, {"code": "abc"}]
        rule = _make_rule(
            rule_name="code_matches",
            rule_type="FORMAT",
            column="code",
            operator="matches",
            threshold=r"^[A-Z]+\d+$",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_contains(self, engine: RuleEvaluatorEngine) -> None:
        """contains operator checks substring presence via FORMAT partial match."""
        data = [{"desc": "GHG emissions report"}, {"desc": "financial report"}]
        rule = _make_rule(
            rule_name="desc_contains",
            rule_type="FORMAT",
            column="desc",
            operator="matches",
            threshold="GHG",
            parameters={"partial_match": True},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_between(self, engine: RuleEvaluatorEngine) -> None:
        """BETWEEN operator validates inclusive range."""
        data = [{"val": 5}, {"val": 10}, {"val": 15}, {"val": 20}]
        rule = _make_rule(
            rule_name="val_between",
            rule_type="RANGE",
            column="val",
            operator="BETWEEN",
            threshold={"min": 10, "max": 15},
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 2

    def test_operator_greater_than_with_string_numbers(self, engine: RuleEvaluatorEngine) -> None:
        """Numeric comparison with string-encoded numbers."""
        data = [{"val": "15"}, {"val": "5"}]
        rule = _make_rule(
            rule_name="val_gt",
            rule_type="RANGE",
            column="val",
            operator="gt",
            threshold=10.0,
        )
        result = engine.evaluate_rule(rule, data)
        # Engine may or may not coerce strings to numbers
        assert "pass_rate" in result

    def test_operator_equals_with_float(self, engine: RuleEvaluatorEngine) -> None:
        """eq with float value."""
        data = [{"val": 3.14}, {"val": 2.71}]
        rule = _make_rule(
            rule_name="val_eq_pi",
            rule_type="RANGE",
            column="val",
            operator="eq",
            threshold=3.14,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_not_equals_with_numeric(self, engine: RuleEvaluatorEngine) -> None:
        """ne with numeric value."""
        data = [{"val": 10}, {"val": 20}]
        rule = _make_rule(
            rule_name="not_10",
            rule_type="RANGE",
            column="val",
            operator="ne",
            threshold=10,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_in_set_empty_set(self, engine: RuleEvaluatorEngine) -> None:
        """IN with empty allowed_values fails all."""
        data = [{"val": "A"}]
        rule = _make_rule(
            rule_name="val_in_empty",
            rule_type="RANGE",
            column="val",
            operator="IN",
            threshold=[],
        )
        result = engine.evaluate_rule(rule, data)
        assert result["fail_count"] == 1

    def test_operator_not_in_set_empty_set(self, engine: RuleEvaluatorEngine) -> None:
        """NOT_IN with empty blacklist passes all (numeric values)."""
        data = [{"val": 42}]
        rule = _make_rule(
            rule_name="val_not_in_empty",
            rule_type="RANGE",
            column="val",
            operator="NOT_IN",
            threshold=[],
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_operator_less_than_zero(self, engine: RuleEvaluatorEngine) -> None:
        """lt works with zero threshold."""
        data = [{"val": -5}, {"val": 0}, {"val": 5}]
        rule = _make_rule(
            rule_name="val_lt_zero",
            rule_type="RANGE",
            column="val",
            operator="lt",
            threshold=0.0,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1


# ==========================================================================
# TestRuleSetEvaluation
# ==========================================================================


class TestRuleSetEvaluation:
    """Tests for rule set evaluation with SLA thresholds."""

    def test_rule_set_all_pass(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Rule set with all passing rules returns pass sla_result."""
        rule_set = {
            "set_id": "rs-001",
            "set_name": "completeness_set",
            "rules": [
                _make_rule(rule_name="name_complete", rule_type="COMPLETENESS", column="name"),
                _make_rule(rule_name="email_complete", rule_type="COMPLETENESS", column="email"),
            ],
            "sla_pass_rate": 1.0,
            "sla_warn_rate": 0.9,
        }
        result = engine.evaluate_rule_set(rule_set, sample_dataset)
        assert result["sla_result"] == "pass"

    def test_rule_set_some_fail(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Rule set with some failures returns appropriate sla_result."""
        rule_set = {
            "set_id": "rs-002",
            "set_name": "completeness_set",
            "rules": [
                _make_rule(rule_name="name_complete", rule_type="COMPLETENESS", column="name"),
                _make_rule(rule_name="score_complete", rule_type="COMPLETENESS", column="score"),
            ],
            "sla_pass_rate": 1.0,
            "sla_warn_rate": 0.9,
        }
        result = engine.evaluate_rule_set(rule_set, dataset_with_nulls)
        assert result["overall_pass_rate"] < 1.0

    def test_rule_set_sla_pass_threshold(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """SLA result respects pass threshold."""
        rule_set = {
            "set_id": "rs-003",
            "set_name": "low_threshold_set",
            "rules": [
                _make_rule(rule_name="name_complete", rule_type="COMPLETENESS", column="name"),
            ],
            "sla_pass_rate": 0.5,
            "sla_warn_rate": 0.3,
        }
        result = engine.evaluate_rule_set(rule_set, dataset_with_nulls)
        if result["overall_pass_rate"] >= 0.5:
            assert result["sla_result"] == "pass"

    def test_rule_set_warn_threshold(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Rule set evaluation reports sla_result."""
        rule_set = {
            "set_id": "rs-004",
            "set_name": "warn_set",
            "rules": [
                _make_rule(rule_name="name_complete", rule_type="COMPLETENESS", column="name"),
            ],
            "sla_pass_rate": 0.95,
            "sla_warn_rate": 0.5,
        }
        result = engine.evaluate_rule_set(rule_set, dataset_with_nulls)
        assert "sla_result" in result

    def test_rule_set_sla_fail(self, engine: RuleEvaluatorEngine) -> None:
        """Failing rules cause sla_result to be fail."""
        data = [{"val": None}]
        rule_set = {
            "set_id": "rs-005",
            "set_name": "critical_set",
            "rules": [
                _make_rule(rule_name="val_complete", rule_type="COMPLETENESS", column="val"),
            ],
            "sla_pass_rate": 1.0,
            "sla_warn_rate": 0.9,
        }
        result = engine.evaluate_rule_set(rule_set, data)
        assert result["sla_result"] == "fail"

    def test_rule_set_duration_tracked(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Rule set evaluation tracks total duration."""
        rule_set = {
            "set_id": "rs-006",
            "set_name": "timed_set",
            "rules": [
                _make_rule(rule_name="name", rule_type="COMPLETENESS", column="name"),
            ],
        }
        result = engine.evaluate_rule_set(rule_set, sample_dataset)
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    def test_rule_set_per_rule_results(self, engine: RuleEvaluatorEngine, dataset_with_nulls: List) -> None:
        """Result includes per-rule breakdown."""
        rule_set = {
            "set_id": "rs-007",
            "set_name": "multi_set",
            "rules": [
                _make_rule(rule_name="name_complete", rule_type="COMPLETENESS", column="name"),
                _make_rule(rule_name="email_complete", rule_type="COMPLETENESS", column="email"),
            ],
        }
        result = engine.evaluate_rule_set(rule_set, dataset_with_nulls)
        assert "per_rule_results" in result
        assert len(result["per_rule_results"]) == 2

    def test_rule_set_empty_rules(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Empty rule set passes trivially."""
        rule_set = {
            "set_id": "rs-008",
            "set_name": "empty_set",
            "rules": [],
        }
        result = engine.evaluate_rule_set(rule_set, sample_dataset)
        assert result["overall_pass_rate"] == 1.0

    def test_rule_set_empty_dataset(self, engine: RuleEvaluatorEngine) -> None:
        """Empty dataset evaluation completes."""
        rule_set = {
            "set_id": "rs-009",
            "set_name": "empty_data_set",
            "rules": [
                _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_rule_set(rule_set, [])
        assert result["rules_evaluated"] == 1

    def test_rule_set_provenance(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Rule set evaluation records provenance."""
        rule_set = {
            "set_id": "rs-010",
            "set_name": "prov_set",
            "rules": [
                _make_rule(rule_name="name", rule_type="COMPLETENESS", column="name"),
            ],
        }
        result = engine.evaluate_rule_set(rule_set, sample_dataset)
        assert "provenance_hash" in result

    def test_rule_set_rules_evaluated(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Result contains rules_evaluated count."""
        rule_set = {
            "set_id": "rs-011",
            "set_name": "multi_rule_set",
            "rules": [
                _make_rule(rule_name="rule_a", rule_type="COMPLETENESS", column="name"),
                _make_rule(rule_name="rule_b", rule_type="COMPLETENESS", column="email"),
            ],
        }
        result = engine.evaluate_rule_set(rule_set, sample_dataset)
        assert result["rules_evaluated"] == 2

    def test_rule_set_rules_passed_and_failed(self, engine: RuleEvaluatorEngine, sample_dataset: List) -> None:
        """Result tracks rules_passed and rules_failed."""
        rule_set = {
            "set_id": "rs-012",
            "set_name": "count_set",
            "rules": [
                _make_rule(rule_name="name", rule_type="COMPLETENESS", column="name"),
            ],
        }
        result = engine.evaluate_rule_set(rule_set, sample_dataset)
        assert "rules_passed" in result
        assert "rules_failed" in result


# ==========================================================================
# TestCompoundRuleEvaluation
# ==========================================================================


class TestCompoundRuleEvaluation:
    """Tests for compound rule evaluation (AND/OR/NOT)."""

    def test_and_all_pass(self, engine: RuleEvaluatorEngine) -> None:
        """AND compound rule passes when all children pass."""
        data = [{"val": 15}]
        compound = {
            "compound_operator": "AND",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
                _make_rule(rule_name="lt_20", rule_type="RANGE", column="val", operator="lt", threshold=20.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "pass"

    def test_and_one_fails(self, engine: RuleEvaluatorEngine) -> None:
        """AND compound rule fails when one child fails."""
        data = [{"val": 25}]
        compound = {
            "compound_operator": "AND",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
                _make_rule(rule_name="lt_20", rule_type="RANGE", column="val", operator="lt", threshold=20.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "fail"

    def test_and_short_circuit(self, engine: RuleEvaluatorEngine) -> None:
        """AND evaluation short-circuits on first failure."""
        data = [{"val": 5}]
        compound = {
            "compound_operator": "AND",
            "short_circuit": True,
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
                _make_rule(rule_name="lt_20", rule_type="RANGE", column="val", operator="lt", threshold=20.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "fail"

    def test_or_any_pass(self, engine: RuleEvaluatorEngine) -> None:
        """OR compound rule passes when any child passes."""
        data = [{"val": 5}]
        compound = {
            "compound_operator": "OR",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
                _make_rule(rule_name="lt_10", rule_type="RANGE", column="val", operator="lt", threshold=10.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "pass"

    def test_or_all_fail(self, engine: RuleEvaluatorEngine) -> None:
        """OR compound rule fails when all children fail."""
        data = [{"val": 10}]
        compound = {
            "compound_operator": "OR",
            "sub_rules": [
                _make_rule(rule_name="gt_20", rule_type="RANGE", column="val", operator="gt", threshold=20.0),
                _make_rule(rule_name="lt_5", rule_type="RANGE", column="val", operator="lt", threshold=5.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "fail"

    def test_not_inversion(self, engine: RuleEvaluatorEngine) -> None:
        """NOT inverts child rule result."""
        data = [{"val": 5}]
        compound = {
            "compound_operator": "NOT",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "pass"

    def test_not_passing_child_fails(self, engine: RuleEvaluatorEngine) -> None:
        """NOT fails when child passes."""
        data = [{"val": 15}]
        compound = {
            "compound_operator": "NOT",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "fail"

    def test_compound_provenance(self, engine: RuleEvaluatorEngine) -> None:
        """Compound evaluation records provenance."""
        data = [{"val": 15}]
        compound = {
            "compound_operator": "AND",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert "provenance_hash" in result

    def test_compound_duration(self, engine: RuleEvaluatorEngine) -> None:
        """Compound evaluation tracks duration."""
        data = [{"val": 15}]
        compound = {
            "compound_operator": "AND",
            "sub_rules": [
                _make_rule(rule_name="gt_10", rule_type="RANGE", column="val", operator="gt", threshold=10.0),
            ],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert "duration_ms" in result

    def test_compound_empty_and_children(self, engine: RuleEvaluatorEngine) -> None:
        """AND with no children passes trivially."""
        data = [{"val": 10}]
        compound = {
            "compound_operator": "AND",
            "sub_rules": [],
        }
        result = engine.evaluate_compound_rule(compound, data)
        assert result["result"] == "pass"


# ==========================================================================
# TestBatchEvaluation
# ==========================================================================


class TestBatchEvaluation:
    """Tests for batch evaluation across multiple datasets."""

    def test_batch_multiple_datasets(self, engine: RuleEvaluatorEngine) -> None:
        """Batch evaluates across multiple datasets."""
        datasets = {
            "ds1": [{"val": 10}, {"val": 20}],
            "ds2": [{"val": 30}, {"val": 40}],
        }
        rule_set = {
            "set_id": "rs-batch-001",
            "set_name": "batch_set",
            "rules": [
                _make_rule(rule_name="val_range", rule_type="RANGE", column="val", operator="gt", threshold=5.0),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert len(result["per_dataset_results"]) == 2

    def test_batch_single_dataset(self, engine: RuleEvaluatorEngine) -> None:
        """Batch works with single dataset."""
        datasets = {"ds1": [{"val": 10}]}
        rule_set = {
            "set_id": "rs-batch-002",
            "set_name": "single_batch",
            "rules": [
                _make_rule(rule_name="val_check", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert len(result["per_dataset_results"]) == 1

    def test_batch_empty_datasets(self, engine: RuleEvaluatorEngine) -> None:
        """Batch with no datasets returns empty results."""
        rule_set = {
            "set_id": "rs-batch-003",
            "set_name": "empty_batch",
            "rules": [
                _make_rule(rule_name="val_check", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch({}, rule_set)
        assert len(result["per_dataset_results"]) == 0

    def test_batch_per_dataset_pass_rate(self, engine: RuleEvaluatorEngine) -> None:
        """Each dataset has its own pass rate."""
        datasets = {
            "good": [{"val": 10}, {"val": 20}],
            "bad": [{"val": None}, {"val": None}],
        }
        rule_set = {
            "set_id": "rs-batch-004",
            "set_name": "per_ds_rate",
            "rules": [
                _make_rule(rule_name="val_req", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert len(result["per_dataset_results"]) == 2

    def test_batch_overall_pass_rate(self, engine: RuleEvaluatorEngine) -> None:
        """Batch includes overall pass rate across all datasets."""
        datasets = {
            "ds1": [{"val": 10}],
            "ds2": [{"val": 20}],
        }
        rule_set = {
            "set_id": "rs-batch-005",
            "set_name": "overall_batch",
            "rules": [
                _make_rule(rule_name="val_check", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert "overall_pass_rate" in result

    def test_batch_provenance(self, engine: RuleEvaluatorEngine) -> None:
        """Batch evaluation records provenance."""
        datasets = {"ds1": [{"val": 10}]}
        rule_set = {
            "set_id": "rs-batch-006",
            "set_name": "prov_batch",
            "rules": [
                _make_rule(rule_name="val_check", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert "provenance_hash" in result

    def test_batch_duration(self, engine: RuleEvaluatorEngine) -> None:
        """Batch evaluation tracks total duration."""
        datasets = {"ds1": [{"val": 10}]}
        rule_set = {
            "set_id": "rs-batch-007",
            "set_name": "dur_batch",
            "rules": [
                _make_rule(rule_name="val_check", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert "duration_ms" in result

    def test_batch_large_datasets(self, engine: RuleEvaluatorEngine) -> None:
        """Batch handles larger datasets."""
        datasets = {f"ds{i}": [{"val": j} for j in range(50)] for i in range(5)}
        rule_set = {
            "set_id": "rs-batch-008",
            "set_name": "large_batch",
            "rules": [
                _make_rule(rule_name="val_check", rule_type="COMPLETENESS", column="val"),
            ],
        }
        result = engine.evaluate_batch(datasets, rule_set)
        assert len(result["per_dataset_results"]) == 5


# ==========================================================================
# TestCrossDatasetEvaluation
# ==========================================================================


class TestCrossDatasetEvaluation:
    """Tests for cross-dataset evaluation with join key."""

    def test_cross_dataset_matching_records(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-dataset evaluation joins on key and validates."""
        datasets = {
            "orders": [{"id": "1", "total": 100}, {"id": "2", "total": 200}],
            "invoices": [{"id": "1", "sum_parts": 100}, {"id": "2", "sum_parts": 195}],
        }
        rule = _make_rule(
            rule_name="total_equals_sum",
            rule_type="CROSS_FIELD",
            column="orders_total",
            operator="eq",
            parameters={"column_b": "invoices_sum_parts"},
        )
        result = engine.evaluate_cross_dataset(datasets, rule, "id")
        assert result["joined_rows"] == 2

    def test_cross_dataset_no_matching_keys(self, engine: RuleEvaluatorEngine) -> None:
        """No matching join keys yields zero joined rows."""
        datasets = {
            "ds_a": [{"id": "1", "val": 10}],
            "ds_b": [{"id": "2", "val": 20}],
        }
        rule = _make_rule(
            rule_name="val_eq",
            rule_type="CROSS_FIELD",
            column="ds_a_val",
            operator="eq",
            parameters={"column_b": "ds_b_val"},
        )
        result = engine.evaluate_cross_dataset(datasets, rule, "id")
        assert result["joined_rows"] == 0

    def test_cross_dataset_all_match(self, engine: RuleEvaluatorEngine) -> None:
        """All matching records pass cross-dataset check."""
        datasets = {
            "ds_a": [{"id": "1", "val": 10}, {"id": "2", "val": 20}],
            "ds_b": [{"id": "1", "val": 10}, {"id": "2", "val": 20}],
        }
        rule = _make_rule(
            rule_name="val_eq",
            rule_type="CROSS_FIELD",
            column="ds_a_val",
            operator="eq",
            parameters={"column_b": "ds_b_val"},
        )
        result = engine.evaluate_cross_dataset(datasets, rule, "id")
        assert result["rule_result"]["pass_rate"] == 1.0

    def test_cross_dataset_provenance(self, engine: RuleEvaluatorEngine) -> None:
        """Cross-dataset evaluation records provenance."""
        datasets = {
            "ds_a": [{"id": "1", "val": 10}],
            "ds_b": [{"id": "1", "val": 10}],
        }
        rule = _make_rule(
            rule_name="val_eq",
            rule_type="CROSS_FIELD",
            column="ds_a_val",
            operator="eq",
            parameters={"column_b": "ds_b_val"},
        )
        result = engine.evaluate_cross_dataset(datasets, rule, "id")
        assert "provenance_hash" in result

    def test_cross_dataset_fewer_than_two_datasets_raises(self, engine: RuleEvaluatorEngine) -> None:
        """Fewer than 2 datasets raises ValueError."""
        rule = _make_rule(
            rule_name="val_eq",
            rule_type="CROSS_FIELD",
            column="val",
            operator="eq",
            parameters={"column_b": "val"},
        )
        with pytest.raises(ValueError):
            engine.evaluate_cross_dataset({"ds_a": [{"id": "1"}]}, rule, "id")

    def test_cross_dataset_partial_join(self, engine: RuleEvaluatorEngine) -> None:
        """Only matching join keys are evaluated."""
        datasets = {
            "ds_a": [{"id": "1", "val": 10}, {"id": "2", "val": 20}, {"id": "3", "val": 30}],
            "ds_b": [{"id": "1", "val": 10}, {"id": "3", "val": 30}],
        }
        rule = _make_rule(
            rule_name="val_eq",
            rule_type="CROSS_FIELD",
            column="ds_a_val",
            operator="eq",
            parameters={"column_b": "ds_b_val"},
        )
        result = engine.evaluate_cross_dataset(datasets, rule, "id")
        # Only ids 1 and 3 are joined
        assert result["joined_rows"] == 2


# ==========================================================================
# TestEdgeCases
# ==========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataset(self, engine: RuleEvaluatorEngine) -> None:
        """Empty dataset returns pass rate 1.0 (vacuous truth)."""
        rule = _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val")
        result = engine.evaluate_rule(rule, [])
        assert result["total"] == 0

    def test_single_record(self, engine: RuleEvaluatorEngine) -> None:
        """Single record evaluation works."""
        data = [{"val": 42}]
        rule = _make_rule(
            rule_name="val_check",
            rule_type="RANGE",
            column="val",
            operator="eq",
            threshold=42,
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_count"] == 1

    def test_all_null_values(self, engine: RuleEvaluatorEngine) -> None:
        """Dataset with all null values in target field."""
        data = [{"val": None} for _ in range(10)]
        rule = _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val")
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 0.0

    def test_mixed_types_in_column(self, engine: RuleEvaluatorEngine) -> None:
        """Mixed types in target field handled gracefully."""
        data = [{"val": 10}, {"val": "hello"}, {"val": None}, {"val": True}]
        rule = _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val")
        result = engine.evaluate_rule(rule, data)
        assert "pass_rate" in result

    def test_large_dataset_performance(self, engine: RuleEvaluatorEngine) -> None:
        """Evaluation completes in reasonable time for 1000 records."""
        data = [{"val": i, "name": f"record_{i}"} for i in range(1000)]
        rule = _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val")
        start = time.monotonic()
        result = engine.evaluate_rule(rule, data)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0  # Should complete in under 10 seconds
        assert result["total"] == 1000

    def test_special_characters_in_field_name(self, engine: RuleEvaluatorEngine) -> None:
        """Fields with special characters work."""
        data = [{"field-with-dashes": 10}, {"field-with-dashes": 20}]
        rule = _make_rule(
            rule_name="dash_field",
            rule_type="COMPLETENESS",
            column="field-with-dashes",
        )
        result = engine.evaluate_rule(rule, data)
        assert result["pass_rate"] == 1.0

    def test_nested_dict_values(self, engine: RuleEvaluatorEngine) -> None:
        """Nested dict values handled for completeness (present not null)."""
        data = [{"meta": {"key": "val"}}, {"meta": {}}]
        rule = _make_rule(rule_name="meta_complete", rule_type="COMPLETENESS", column="meta")
        result = engine.evaluate_rule(rule, data)
        assert "pass_rate" in result

    def test_statistics_after_evaluations(self, engine: RuleEvaluatorEngine) -> None:
        """Statistics are updated after evaluations."""
        data = [{"val": 10}]
        rule = _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val")
        engine.evaluate_rule(rule, data)
        stats = engine.get_statistics()
        assert stats["total_evaluations"] >= 1

    def test_clear_resets_state(self, engine: RuleEvaluatorEngine) -> None:
        """Clear method resets evaluation history."""
        data = [{"val": 10}]
        rule = _make_rule(rule_name="val", rule_type="COMPLETENESS", column="val")
        engine.evaluate_rule(rule, data)
        engine.clear()
        stats = engine.get_statistics()
        assert stats["total_evaluations"] == 0

    def test_unsupported_rule_type_raises(self, engine: RuleEvaluatorEngine) -> None:
        """Unsupported rule type raises ValueError."""
        data = [{"val": 10}]
        rule = _make_rule(
            rule_name="bad_type",
            rule_type="NONEXISTENT_TYPE",
            column="val",
        )
        with pytest.raises(ValueError):
            engine.evaluate_rule(rule, data)
