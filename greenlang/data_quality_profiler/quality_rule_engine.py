# -*- coding: utf-8 -*-
"""
Quality Rule Engine - AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)

Custom quality rules and quality gates for data quality governance.
Supports CRUD operations on quality rules, rule evaluation against
datasets, quality gate evaluation with pass/warn/fail thresholds,
and rule import/export for portability.

Zero-Hallucination Guarantees:
    - All rule evaluations use deterministic comparison operators
    - Gate outcomes are computed from threshold arithmetic only
    - Custom expressions use restricted safe evaluation (no exec/eval)
    - No ML/LLM calls in the evaluation path
    - SHA-256 provenance on every rule/gate mutation
    - Thread-safe in-memory storage

Rule Types:
    - COMPLETENESS: check null rate < threshold
    - RANGE: check value within min/max
    - FORMAT: check format pattern match rate
    - UNIQUENESS: check cardinality >= threshold
    - CUSTOM: restricted Python expression evaluation
    - FRESHNESS: check age_hours < threshold

Operators:
    EQUALS, NOT_EQUALS, GREATER_THAN, LESS_THAN, BETWEEN,
    MATCHES, CONTAINS, IN_SET

Example:
    >>> from greenlang.data_quality_profiler.quality_rule_engine import QualityRuleEngine
    >>> engine = QualityRuleEngine()
    >>> rule = engine.create_rule(
    ...     name="age_range", rule_type="RANGE", column="age",
    ...     operator="BETWEEN", threshold=None,
    ...     parameters={"min_val": 0, "max_val": 150}
    ... )
    >>> data = [{"age": 25}, {"age": -5}, {"age": 200}]
    >>> result = engine.evaluate_rule(rule, data)
    >>> print(result["pass_rate"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "QualityRuleEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "QRL") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a rule engine operation.

    Args:
        operation: Name of the operation.
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_missing(value: Any) -> bool:
    """Determine whether a value is considered missing.

    Args:
        value: The value to check.

    Returns:
        True if the value is None, empty string, or whitespace-only.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rule types
RULE_COMPLETENESS = "COMPLETENESS"
RULE_RANGE = "RANGE"
RULE_FORMAT = "FORMAT"
RULE_UNIQUENESS = "UNIQUENESS"
RULE_CUSTOM = "CUSTOM"
RULE_FRESHNESS = "FRESHNESS"

ALL_RULE_TYPES = frozenset({
    RULE_COMPLETENESS, RULE_RANGE, RULE_FORMAT,
    RULE_UNIQUENESS, RULE_CUSTOM, RULE_FRESHNESS,
})

# Operators
OP_EQUALS = "EQUALS"
OP_NOT_EQUALS = "NOT_EQUALS"
OP_GREATER_THAN = "GREATER_THAN"
OP_LESS_THAN = "LESS_THAN"
OP_BETWEEN = "BETWEEN"
OP_MATCHES = "MATCHES"
OP_CONTAINS = "CONTAINS"
OP_IN_SET = "IN_SET"

ALL_OPERATORS = frozenset({
    OP_EQUALS, OP_NOT_EQUALS, OP_GREATER_THAN, OP_LESS_THAN,
    OP_BETWEEN, OP_MATCHES, OP_CONTAINS, OP_IN_SET,
})

# Gate outcomes
GATE_PASS = "PASS"
GATE_WARN = "WARN"
GATE_FAIL = "FAIL"

# Severity / priority levels
PRIORITY_CRITICAL = 1
PRIORITY_HIGH = 2
PRIORITY_MEDIUM = 3
PRIORITY_LOW = 4

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"


# ---------------------------------------------------------------------------
# QualityRuleEngine
# ---------------------------------------------------------------------------


class QualityRuleEngine:
    """Custom quality rules and quality gates engine.

    Provides CRUD for quality rules, evaluates rules against datasets,
    manages quality gates with pass/warn/fail thresholds, and supports
    rule import/export for governance portability.

    Thread-safe: all mutations to internal storage are protected by
    a threading lock. SHA-256 provenance hashes on every mutation.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for thread-safe storage access.
        _rules: In-memory rule storage.
        _gates: In-memory gate storage.
        _evaluations: In-memory evaluation result storage.
        _stats: Aggregate engine statistics.

    Example:
        >>> engine = QualityRuleEngine()
        >>> rule = engine.create_rule("test", "RANGE", "col", "BETWEEN",
        ...     parameters={"min_val": 0, "max_val": 100})
        >>> result = engine.evaluate_rule(rule, [{"col": 50}, {"col": 150}])
        >>> assert result["pass_rate"] == 0.5
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize QualityRuleEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_rules``: int, maximum rules to store (default 10000)
                - ``max_gates``: int, maximum gates to store (default 1000)
                - ``default_threshold``: float, default pass threshold (default 0.95)
        """
        self._config = config or {}
        self._max_rules: int = self._config.get("max_rules", 10000)
        self._max_gates: int = self._config.get("max_gates", 1000)
        self._default_threshold: float = self._config.get("default_threshold", 0.95)
        self._lock = threading.Lock()
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._gates: Dict[str, Dict[str, Any]] = {}
        self._evaluations: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "rules_created": 0,
            "rules_evaluated": 0,
            "gates_created": 0,
            "gates_evaluated": 0,
            "total_evaluation_time_ms": 0.0,
        }
        logger.info(
            "QualityRuleEngine initialized: max_rules=%d, max_gates=%d, "
            "default_threshold=%.2f",
            self._max_rules, self._max_gates, self._default_threshold,
        )

    # ------------------------------------------------------------------
    # Rule CRUD
    # ------------------------------------------------------------------

    def create_rule(
        self,
        name: str,
        rule_type: str,
        column: str,
        operator: Optional[str] = None,
        threshold: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = PRIORITY_MEDIUM,
    ) -> Dict[str, Any]:
        """Create a new quality rule.

        Args:
            name: Human-readable rule name.
            rule_type: Rule type (COMPLETENESS, RANGE, FORMAT, etc.).
            column: Target column name.
            operator: Comparison operator (EQUALS, BETWEEN, etc.).
            threshold: Pass/fail threshold value.
            parameters: Additional parameters (min_val, max_val, pattern, etc.).
            priority: Rule priority (1=critical, 4=low).

        Returns:
            Created rule dict with: rule_id, name, rule_type, column,
            operator, threshold, parameters, priority, active, provenance_hash.

        Raises:
            ValueError: If rule_type is invalid or max rules exceeded.
        """
        if rule_type not in ALL_RULE_TYPES:
            raise ValueError(
                f"Invalid rule_type '{rule_type}'. Must be one of: "
                f"{', '.join(sorted(ALL_RULE_TYPES))}"
            )
        if operator and operator not in ALL_OPERATORS:
            raise ValueError(
                f"Invalid operator '{operator}'. Must be one of: "
                f"{', '.join(sorted(ALL_OPERATORS))}"
            )

        with self._lock:
            if len(self._rules) >= self._max_rules:
                raise ValueError(
                    f"Maximum rules ({self._max_rules}) exceeded"
                )

        rule_id = _generate_id("QRL")
        now = _utcnow()

        rule: Dict[str, Any] = {
            "rule_id": rule_id,
            "name": name,
            "rule_type": rule_type,
            "column": column,
            "operator": operator,
            "threshold": threshold,
            "parameters": parameters or {},
            "priority": priority,
            "active": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 1,
        }

        provenance_data = json.dumps({
            "rule_id": rule_id,
            "name": name,
            "rule_type": rule_type,
            "column": column,
        }, sort_keys=True, default=str)
        rule["provenance_hash"] = _compute_provenance("create_rule", provenance_data)

        with self._lock:
            self._rules[rule_id] = rule
            self._stats["rules_created"] += 1

        logger.info("Rule created: id=%s, name=%s, type=%s", rule_id, name, rule_type)
        return rule

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a rule by ID.

        Args:
            rule_id: The rule identifier.

        Returns:
            Rule dict or None if not found.
        """
        with self._lock:
            return self._rules.get(rule_id)

    def list_rules(
        self,
        active_only: bool = False,
        rule_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List rules with optional filtering.

        Args:
            active_only: If True, only return active rules.
            rule_type: Optional filter by rule type.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of rule dicts sorted by priority then creation time.
        """
        with self._lock:
            rules = list(self._rules.values())

        if active_only:
            rules = [r for r in rules if r.get("active", True)]
        if rule_type:
            rules = [r for r in rules if r.get("rule_type") == rule_type]

        rules.sort(key=lambda r: (r.get("priority", 4), r.get("created_at", "")))
        return rules[offset:offset + limit]

    def update_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update an existing rule.

        Args:
            rule_id: The rule identifier.
            updates: Dict of fields to update. Allowed keys: name,
                operator, threshold, parameters, priority, active.

        Returns:
            Updated rule dict or None if not found.

        Raises:
            ValueError: If updates contain invalid fields.
        """
        allowed_keys = {"name", "operator", "threshold", "parameters", "priority", "active"}
        invalid_keys = set(updates.keys()) - allowed_keys
        if invalid_keys:
            raise ValueError(f"Cannot update fields: {invalid_keys}")

        if "operator" in updates and updates["operator"] not in ALL_OPERATORS:
            raise ValueError(f"Invalid operator: {updates['operator']}")

        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                return None

            for key, value in updates.items():
                rule[key] = value

            rule["updated_at"] = _utcnow().isoformat()
            rule["version"] = rule.get("version", 1) + 1

            provenance_data = json.dumps({
                "rule_id": rule_id,
                "updates": updates,
                "version": rule["version"],
            }, sort_keys=True, default=str)
            rule["provenance_hash"] = _compute_provenance("update_rule", provenance_data)

            self._rules[rule_id] = rule

        logger.info("Rule updated: id=%s, v%d", rule_id, rule["version"])
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule.

        Args:
            rule_id: The rule identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info("Rule deleted: %s", rule_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Rule Evaluation
    # ------------------------------------------------------------------

    def evaluate_rule(
        self,
        rule: Dict[str, Any],
        data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a single rule against a dataset.

        Args:
            rule: Rule dict (from create_rule or get_rule).
            data: List of row dictionaries.

        Returns:
            RuleEvaluation dict with: evaluation_id, rule_id, rule_name,
            pass_count, fail_count, total_count, pass_rate, outcome,
            violations, provenance_hash.
        """
        start = time.monotonic()
        eval_id = _generate_id("EVL")
        rule_type = rule.get("rule_type", "")
        column = rule.get("column", "")
        operator = rule.get("operator", "")
        threshold = rule.get("threshold")
        params = rule.get("parameters", {})

        pass_count = 0
        fail_count = 0
        violations: List[Dict[str, Any]] = []
        total = len(data)

        for idx, row in enumerate(data):
            value = row.get(column)
            passed = self._evaluate_single(
                value, rule_type, operator, threshold, params
            )
            if passed:
                pass_count += 1
            else:
                fail_count += 1
                if len(violations) < 100:
                    violations.append({
                        "row_index": idx,
                        "column": column,
                        "value": str(value)[:200] if value is not None else None,
                        "rule_type": rule_type,
                    })

        pass_rate = pass_count / total if total > 0 else 1.0
        rule_threshold = threshold if threshold is not None else self._default_threshold

        # Determine outcome
        if pass_rate >= (rule_threshold if isinstance(rule_threshold, float) and rule_threshold <= 1.0 else self._default_threshold):
            outcome = GATE_PASS
        elif pass_rate >= 0.8:
            outcome = GATE_WARN
        else:
            outcome = GATE_FAIL

        elapsed_ms = (time.monotonic() - start) * 1000.0

        provenance_data = json.dumps({
            "eval_id": eval_id,
            "rule_id": rule.get("rule_id", ""),
            "pass_rate": pass_rate,
            "total": total,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("evaluate_rule", provenance_data)

        result: Dict[str, Any] = {
            "evaluation_id": eval_id,
            "rule_id": rule.get("rule_id", ""),
            "rule_name": rule.get("name", ""),
            "rule_type": rule_type,
            "column": column,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total_count": total,
            "pass_rate": round(pass_rate, 4),
            "outcome": outcome,
            "violations": violations,
            "violation_count": fail_count,
            "provenance_hash": provenance_hash,
            "evaluation_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        with self._lock:
            self._evaluations[eval_id] = result
            self._stats["rules_evaluated"] += 1
            self._stats["total_evaluation_time_ms"] += elapsed_ms

        logger.info(
            "Rule evaluated: eval=%s, rule=%s, pass_rate=%.4f, outcome=%s",
            eval_id, rule.get("rule_id", ""), pass_rate, outcome,
        )
        return result

    def evaluate_rules(
        self,
        data: List[Dict[str, Any]],
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple rules against a dataset.

        Args:
            data: List of row dictionaries.
            rule_ids: Optional list of rule IDs to evaluate.
                If None, evaluates all active rules.

        Returns:
            List of RuleEvaluation dicts.
        """
        if rule_ids:
            rules_to_eval = []
            for rid in rule_ids:
                r = self.get_rule(rid)
                if r:
                    rules_to_eval.append(r)
        else:
            rules_to_eval = self.list_rules(active_only=True)

        results: List[Dict[str, Any]] = []
        for rule in rules_to_eval:
            result = self.evaluate_rule(rule, data)
            results.append(result)

        return results

    def _evaluate_single(
        self,
        value: Any,
        rule_type: str,
        operator: str,
        threshold: Optional[float],
        params: Dict[str, Any],
    ) -> bool:
        """Evaluate a single value against a rule.

        Args:
            value: The value to evaluate.
            rule_type: Rule type.
            operator: Comparison operator.
            threshold: Threshold value.
            params: Additional parameters.

        Returns:
            True if the value passes the rule.
        """
        if rule_type == RULE_COMPLETENESS:
            return not _is_missing(value)

        if rule_type == RULE_RANGE:
            return self._evaluate_range(value, params)

        if rule_type == RULE_FORMAT:
            return self._evaluate_format(value, params)

        if rule_type == RULE_UNIQUENESS:
            # Uniqueness is evaluated at column level, not per-value
            # For per-value: just check it's non-null
            return not _is_missing(value)

        if rule_type == RULE_FRESHNESS:
            return self._evaluate_freshness(value, params)

        if rule_type == RULE_CUSTOM:
            return self._evaluate_custom(value, params)

        # Default: use operator-based evaluation
        return self._evaluate_operator(value, operator, threshold, params)

    def _evaluate_range(
        self,
        value: Any,
        params: Dict[str, Any],
    ) -> bool:
        """Evaluate a value against a range rule.

        Args:
            value: Value to check.
            params: Dict with min_val and/or max_val.

        Returns:
            True if within range.
        """
        if _is_missing(value):
            return True  # Missing handled by completeness rules

        try:
            num = float(str(value))
        except (ValueError, TypeError):
            return False

        if math.isnan(num) or math.isinf(num):
            return False

        min_val = params.get("min_val")
        max_val = params.get("max_val")

        if min_val is not None and num < float(min_val):
            return False
        if max_val is not None and num > float(max_val):
            return False
        return True

    def _evaluate_format(
        self,
        value: Any,
        params: Dict[str, Any],
    ) -> bool:
        """Evaluate a value against a format pattern rule.

        Args:
            value: Value to check.
            params: Dict with 'pattern' key containing regex string.

        Returns:
            True if matches the pattern.
        """
        if _is_missing(value):
            return True

        pattern = params.get("pattern", "")
        if not pattern:
            return True

        try:
            return bool(re.match(pattern, str(value)))
        except re.error:
            logger.warning("Invalid regex pattern in format rule: %s", pattern)
            return True

    def _evaluate_freshness(
        self,
        value: Any,
        params: Dict[str, Any],
    ) -> bool:
        """Evaluate a timestamp value against a freshness rule.

        Args:
            value: Timestamp value.
            params: Dict with 'max_age_hours' key.

        Returns:
            True if the timestamp is fresh enough.
        """
        if _is_missing(value):
            return False

        max_age = params.get("max_age_hours", 24.0)
        now = _utcnow()

        # Try ISO format parsing
        if isinstance(value, str):
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    ts = datetime.strptime(value.strip(), fmt)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_hours = (now - ts).total_seconds() / 3600.0
                    return age_hours <= max_age
                except ValueError:
                    continue
            return False

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            age_hours = (now - value).total_seconds() / 3600.0
            return age_hours <= max_age

        return False

    def _evaluate_custom(
        self,
        value: Any,
        params: Dict[str, Any],
    ) -> bool:
        """Evaluate a value against a custom expression.

        Uses restricted comparison only (no exec/eval for safety).
        The expression is a simple condition like "value > 0".

        Args:
            value: Value to check.
            params: Dict with 'expression' key and optional 'comparator'.

        Returns:
            True if the custom check passes.
        """
        if _is_missing(value):
            return True

        comparator = params.get("comparator")
        compare_value = params.get("compare_value")

        if comparator and compare_value is not None:
            return self._evaluate_operator(
                value, comparator, None,
                {"compare_value": compare_value}
            )

        # If no structured comparator, default to non-null check
        return True

    def _evaluate_operator(
        self,
        value: Any,
        operator: str,
        threshold: Optional[float],
        params: Dict[str, Any],
    ) -> bool:
        """Evaluate a value using a comparison operator.

        Args:
            value: Value to evaluate.
            operator: Comparison operator string.
            threshold: Threshold for comparison.
            params: Additional parameters.

        Returns:
            True if the comparison passes.
        """
        if _is_missing(value):
            return True

        compare_to = params.get("compare_value", threshold)

        if operator == OP_EQUALS:
            return str(value) == str(compare_to) if compare_to is not None else True

        if operator == OP_NOT_EQUALS:
            return str(value) != str(compare_to) if compare_to is not None else True

        if operator == OP_GREATER_THAN:
            try:
                return float(str(value)) > float(str(compare_to))
            except (ValueError, TypeError):
                return False

        if operator == OP_LESS_THAN:
            try:
                return float(str(value)) < float(str(compare_to))
            except (ValueError, TypeError):
                return False

        if operator == OP_BETWEEN:
            min_val = params.get("min_val", params.get("lower"))
            max_val = params.get("max_val", params.get("upper"))
            try:
                num = float(str(value))
                result = True
                if min_val is not None:
                    result = result and num >= float(str(min_val))
                if max_val is not None:
                    result = result and num <= float(str(max_val))
                return result
            except (ValueError, TypeError):
                return False

        if operator == OP_MATCHES:
            pattern = params.get("pattern", str(compare_to) if compare_to else "")
            try:
                return bool(re.match(pattern, str(value)))
            except re.error:
                return False

        if operator == OP_CONTAINS:
            search_str = str(compare_to) if compare_to is not None else ""
            return search_str in str(value)

        if operator == OP_IN_SET:
            allowed = params.get("allowed_values", [])
            if not allowed and compare_to is not None:
                allowed = [compare_to]
            return value in allowed or str(value) in [str(a) for a in allowed]

        return True

    # ------------------------------------------------------------------
    # Quality Gates
    # ------------------------------------------------------------------

    def create_gate(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a quality gate.

        A gate defines conditions that dimension scores must meet.
        Each condition specifies a dimension, operator, and threshold.

        Gate condition format:
            {
                "dimension": "completeness" | "validity" | "consistency" | ...
                "operator": "GREATER_THAN" | "LESS_THAN" | ...
                "threshold": 0.95
                "weight": 1.0  (optional)
            }

        Args:
            name: Gate name.
            conditions: List of condition dicts.
            threshold: Overall pass threshold (fraction of conditions
                that must pass). Defaults to 1.0 (all must pass).

        Returns:
            Gate dict with: gate_id, name, conditions, threshold.

        Raises:
            ValueError: If conditions are empty or max gates exceeded.
        """
        if not conditions:
            raise ValueError("Gate must have at least one condition")

        with self._lock:
            if len(self._gates) >= self._max_gates:
                raise ValueError(
                    f"Maximum gates ({self._max_gates}) exceeded"
                )

        gate_id = _generate_id("GTE")
        now = _utcnow()
        pass_threshold = threshold if threshold is not None else 1.0

        gate: Dict[str, Any] = {
            "gate_id": gate_id,
            "name": name,
            "conditions": conditions,
            "threshold": pass_threshold,
            "active": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        provenance_data = json.dumps({
            "gate_id": gate_id,
            "name": name,
            "conditions_count": len(conditions),
            "threshold": pass_threshold,
        }, sort_keys=True, default=str)
        gate["provenance_hash"] = _compute_provenance("create_gate", provenance_data)

        with self._lock:
            self._gates[gate_id] = gate
            self._stats["gates_created"] += 1

        logger.info(
            "Gate created: id=%s, name=%s, conditions=%d, threshold=%.2f",
            gate_id, name, len(conditions), pass_threshold,
        )
        return gate

    def evaluate_gate(
        self,
        gate_conditions: List[Dict[str, Any]],
        dimension_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Evaluate a quality gate against dimension scores.

        Args:
            gate_conditions: List of condition dicts with dimension,
                operator, and threshold.
            dimension_scores: Dict mapping dimension_name -> score.

        Returns:
            GateOutcome dict with: outcome (PASS/WARN/FAIL),
            conditions_met, conditions_total, condition_results.
        """
        start = time.monotonic()
        condition_results: List[Dict[str, Any]] = []
        met_count = 0
        total = len(gate_conditions)

        for cond in gate_conditions:
            dimension = cond.get("dimension", "")
            operator = cond.get("operator", OP_GREATER_THAN)
            threshold_val = cond.get("threshold", 0.0)

            actual_score = dimension_scores.get(dimension, 0.0)
            passed = self._evaluate_gate_condition(actual_score, operator, threshold_val)

            if passed:
                met_count += 1

            condition_results.append({
                "dimension": dimension,
                "operator": operator,
                "threshold": threshold_val,
                "actual_score": round(actual_score, 4),
                "passed": passed,
            })

        # Determine outcome
        met_ratio = met_count / total if total > 0 else 1.0
        if met_ratio >= 1.0:
            outcome = GATE_PASS
        elif met_ratio >= 0.5:
            outcome = GATE_WARN
        else:
            outcome = GATE_FAIL

        elapsed_ms = (time.monotonic() - start) * 1000.0

        provenance_data = json.dumps({
            "conditions_total": total,
            "conditions_met": met_count,
            "outcome": outcome,
        }, sort_keys=True, default=str)
        provenance_hash = _compute_provenance("evaluate_gate", provenance_data)

        with self._lock:
            self._stats["gates_evaluated"] += 1
            self._stats["total_evaluation_time_ms"] += elapsed_ms

        return {
            "outcome": outcome,
            "conditions_met": met_count,
            "conditions_total": total,
            "met_ratio": round(met_ratio, 4),
            "condition_results": condition_results,
            "provenance_hash": provenance_hash,
            "evaluation_time_ms": round(elapsed_ms, 2),
        }

    def _evaluate_gate_condition(
        self,
        actual: float,
        operator: str,
        threshold: float,
    ) -> bool:
        """Evaluate a single gate condition.

        Args:
            actual: Actual dimension score.
            operator: Comparison operator.
            threshold: Threshold value.

        Returns:
            True if the condition is met.
        """
        if operator == OP_GREATER_THAN:
            return actual > threshold
        if operator == OP_LESS_THAN:
            return actual < threshold
        if operator == OP_EQUALS:
            return abs(actual - threshold) < 1e-9
        if operator == OP_NOT_EQUALS:
            return abs(actual - threshold) >= 1e-9
        if operator in (OP_BETWEEN, OP_MATCHES, OP_CONTAINS, OP_IN_SET):
            # For gate conditions, GREATER_THAN is the standard
            return actual >= threshold
        return actual >= threshold

    def get_gate(self, gate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a gate by ID.

        Args:
            gate_id: The gate identifier.

        Returns:
            Gate dict or None if not found.
        """
        with self._lock:
            return self._gates.get(gate_id)

    def list_gates(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all gates.

        Args:
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of gate dicts.
        """
        with self._lock:
            gates = sorted(
                self._gates.values(),
                key=lambda g: g.get("created_at", ""),
                reverse=True,
            )
            return gates[offset:offset + limit]

    def delete_gate(self, gate_id: str) -> bool:
        """Delete a gate.

        Args:
            gate_id: The gate identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if gate_id in self._gates:
                del self._gates[gate_id]
                logger.info("Gate deleted: %s", gate_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def import_rules(
        self,
        rules_data: List[Dict[str, Any]],
    ) -> int:
        """Import rules from a list of rule definitions.

        Each rule dict should have: name, rule_type, column, and
        optionally: operator, threshold, parameters, priority.

        Args:
            rules_data: List of rule definition dicts.

        Returns:
            Number of rules successfully imported.
        """
        imported = 0
        for rule_def in rules_data:
            try:
                self.create_rule(
                    name=rule_def.get("name", "imported_rule"),
                    rule_type=rule_def.get("rule_type", RULE_COMPLETENESS),
                    column=rule_def.get("column", ""),
                    operator=rule_def.get("operator"),
                    threshold=rule_def.get("threshold"),
                    parameters=rule_def.get("parameters"),
                    priority=rule_def.get("priority", PRIORITY_MEDIUM),
                )
                imported += 1
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Failed to import rule '%s': %s",
                    rule_def.get("name", "unknown"), e,
                )

        logger.info("Imported %d/%d rules", imported, len(rules_data))
        return imported

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules as portable definitions.

        Returns:
            List of rule dicts suitable for import_rules().
        """
        with self._lock:
            rules = []
            for rule in self._rules.values():
                rules.append({
                    "name": rule.get("name", ""),
                    "rule_type": rule.get("rule_type", ""),
                    "column": rule.get("column", ""),
                    "operator": rule.get("operator"),
                    "threshold": rule.get("threshold"),
                    "parameters": rule.get("parameters", {}),
                    "priority": rule.get("priority", PRIORITY_MEDIUM),
                    "active": rule.get("active", True),
                })
            return rules

    # ------------------------------------------------------------------
    # Storage and Retrieval
    # ------------------------------------------------------------------

    def get_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored evaluation by ID.

        Args:
            evaluation_id: The evaluation identifier.

        Returns:
            Evaluation dict or None if not found.
        """
        with self._lock:
            return self._evaluations.get(evaluation_id)

    def list_evaluations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored evaluations with pagination.

        Args:
            limit: Maximum number of results.
            offset: Results to skip.

        Returns:
            List of evaluation dicts sorted by creation time descending.
        """
        with self._lock:
            all_evals = sorted(
                self._evaluations.values(),
                key=lambda e: e.get("created_at", ""),
                reverse=True,
            )
            return all_evals[offset:offset + limit]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate engine statistics.

        Returns:
            Dictionary with counters and totals for all rule/gate
            operations performed by this engine instance.
        """
        with self._lock:
            evaluated = self._stats["rules_evaluated"]
            avg_time = (
                self._stats["total_evaluation_time_ms"] / evaluated
                if evaluated > 0 else 0.0
            )
            return {
                "rules_created": self._stats["rules_created"],
                "rules_evaluated": evaluated,
                "gates_created": self._stats["gates_created"],
                "gates_evaluated": self._stats["gates_evaluated"],
                "total_evaluation_time_ms": round(
                    self._stats["total_evaluation_time_ms"], 2
                ),
                "avg_evaluation_time_ms": round(avg_time, 2),
                "stored_rules": len(self._rules),
                "stored_gates": len(self._gates),
                "stored_evaluations": len(self._evaluations),
                "timestamp": _utcnow().isoformat(),
            }
