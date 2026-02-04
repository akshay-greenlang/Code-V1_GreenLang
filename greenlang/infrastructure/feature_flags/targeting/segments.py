# -*- coding: utf-8 -*-
"""
Segment Matcher - INFRA-008 Targeting Subsystem

Evaluates whether an EvaluationContext matches a set of segment conditions.
Conditions use a rich operator language (eq, neq, in, not_in, gt, lt, gte,
lte, contains, starts_with, ends_with, regex) applied to context attributes
resolved via dot-notation path expressions.

Supports both AND logic (all conditions must match) and OR groups (any
condition in an OR group must match). Missing attributes gracefully evaluate
to False -- they never raise exceptions.

Design principles:
    - Zero external dependencies beyond stdlib ``re``.
    - Defensive: missing or mistyped attributes return False, never crash.
    - Composable: AND and OR groups can be nested for complex targeting.

Example:
    >>> from greenlang.infrastructure.feature_flags.models import EvaluationContext
    >>> matcher = SegmentMatcher()
    >>> conditions = {
    ...     "conditions": [
    ...         {"attribute": "user_attributes.plan_type", "operator": "eq", "value": "enterprise"},
    ...         {"attribute": "environment", "operator": "in", "value": ["staging", "prod"]},
    ...     ]
    ... }
    >>> ctx = EvaluationContext(
    ...     user_id="u-1",
    ...     environment="staging",
    ...     user_attributes={"plan_type": "enterprise"},
    ... )
    >>> matcher.matches(ctx, conditions)
    True
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from greenlang.infrastructure.feature_flags.models import EvaluationContext

logger = logging.getLogger(__name__)

# Operators that compare against a single scalar value
_SCALAR_OPERATORS = {
    "eq", "neq", "gt", "lt", "gte", "lte",
    "contains", "starts_with", "ends_with", "regex",
}
# Operators that compare against a collection
_COLLECTION_OPERATORS = {"in", "not_in"}
# All supported operators
SUPPORTED_OPERATORS = _SCALAR_OPERATORS | _COLLECTION_OPERATORS


class SegmentMatcher:
    """Evaluates context attributes against segment targeting conditions.

    Conditions are structured dicts with the following layout::

        {
            "conditions": [
                {"attribute": "path.to.field", "operator": "eq", "value": "expected"},
                ...
            ]
        }

    Top-level conditions are combined with AND logic (all must match).
    For OR logic, use the ``"or"`` key::

        {
            "or": [
                {"attribute": "environment", "operator": "eq", "value": "staging"},
                {"attribute": "environment", "operator": "eq", "value": "prod"},
            ]
        }

    AND and OR can be combined. In that case the AND conditions AND the OR
    group must both be satisfied.
    """

    def matches(self, context: EvaluationContext, conditions: Dict[str, Any]) -> bool:
        """Check if context matches all segment conditions.

        Args:
            context: The evaluation context containing user/tenant/env data.
            conditions: Targeting conditions dict with "conditions" and/or "or" keys.

        Returns:
            True if the context satisfies all conditions, False otherwise.
            Returns True for empty condition sets (matches everything).
        """
        if not conditions:
            return True

        try:
            # Evaluate AND conditions
            and_conditions: List[Dict[str, Any]] = conditions.get("conditions", [])
            if and_conditions:
                if not self._evaluate_and_group(context, and_conditions):
                    return False

            # Evaluate OR group
            or_conditions: List[Dict[str, Any]] = conditions.get("or", [])
            if or_conditions:
                if not self._evaluate_or_group(context, or_conditions):
                    return False

            # If neither key is present, check for a single inline condition
            if not and_conditions and not or_conditions:
                if "attribute" in conditions and "operator" in conditions:
                    return self._evaluate_condition(context, conditions)

            return True

        except Exception as exc:
            logger.warning(
                "SegmentMatcher.matches failed with exception: %s. "
                "Returning False for safety.",
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Group evaluators
    # ------------------------------------------------------------------

    def _evaluate_and_group(
        self,
        context: EvaluationContext,
        conditions: List[Dict[str, Any]],
    ) -> bool:
        """Evaluate a list of conditions with AND logic (all must match).

        Args:
            context: Evaluation context.
            conditions: List of condition dicts.

        Returns:
            True if every condition matches.
        """
        for condition in conditions:
            # Support nested OR within AND
            if "or" in condition and "attribute" not in condition:
                if not self._evaluate_or_group(context, condition["or"]):
                    return False
            else:
                if not self._evaluate_condition(context, condition):
                    return False
        return True

    def _evaluate_or_group(
        self,
        context: EvaluationContext,
        conditions: List[Dict[str, Any]],
    ) -> bool:
        """Evaluate a list of conditions with OR logic (at least one must match).

        Args:
            context: Evaluation context.
            conditions: List of condition dicts.

        Returns:
            True if at least one condition matches. False if the list is empty.
        """
        if not conditions:
            return False

        for condition in conditions:
            # Support nested AND within OR
            if "conditions" in condition and "attribute" not in condition:
                if self._evaluate_and_group(context, condition["conditions"]):
                    return True
            else:
                if self._evaluate_condition(context, condition):
                    return True
        return False

    # ------------------------------------------------------------------
    # Single condition evaluation
    # ------------------------------------------------------------------

    def _evaluate_condition(
        self,
        context: EvaluationContext,
        condition: Dict[str, Any],
    ) -> bool:
        """Evaluate a single condition against the context.

        A condition dict must contain:
            - ``attribute``: Dot-notation path to the context field.
            - ``operator``: One of the SUPPORTED_OPERATORS.
            - ``value``: The expected value to compare against.

        Args:
            context: Evaluation context.
            condition: Single condition dict.

        Returns:
            True if the condition matches. False on any error or missing data.
        """
        attribute_path: Optional[str] = condition.get("attribute")
        operator: Optional[str] = condition.get("operator")
        expected_value: Any = condition.get("value")

        if not attribute_path or not operator:
            logger.debug(
                "SegmentMatcher: skipping malformed condition "
                "(missing attribute or operator): %s",
                condition,
            )
            return False

        if operator not in SUPPORTED_OPERATORS:
            logger.debug(
                "SegmentMatcher: unsupported operator '%s' in condition: %s",
                operator,
                condition,
            )
            return False

        # Resolve the attribute value from the context
        actual_value = self._resolve_attribute(context, attribute_path)

        return self._apply_operator(operator, actual_value, expected_value)

    # ------------------------------------------------------------------
    # Attribute resolution
    # ------------------------------------------------------------------

    def _resolve_attribute(
        self,
        context: EvaluationContext,
        attribute_path: str,
    ) -> Any:
        """Resolve a dot-notation attribute path against the evaluation context.

        Supported top-level paths:
            - ``user_id`` -> context.user_id
            - ``tenant_id`` -> context.tenant_id
            - ``environment`` -> context.environment
            - ``user_segments`` -> context.user_segments
            - ``user_attributes.X`` -> context.user_attributes["X"]
            - ``user_attributes.X.Y`` -> nested dict lookup

        Args:
            context: Evaluation context.
            attribute_path: Dot-notation path (e.g. "user_attributes.plan_type").

        Returns:
            The resolved value, or None if the path cannot be resolved.
        """
        parts = attribute_path.split(".")

        if not parts:
            return None

        root = parts[0]

        # Direct scalar fields on the context model
        direct_fields = {
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "environment": context.environment,
            "request_id": context.request_id,
            "user_segments": context.user_segments,
        }
        if root in direct_fields:
            if len(parts) == 1:
                return direct_fields[root]
            # If someone writes user_id.something, return None
            return None

        # Dict-based fields: user_attributes
        if root == "user_attributes":
            return self._resolve_nested(context.user_attributes, parts[1:])

        # Fallback: try user_attributes as implicit root for convenience
        return self._resolve_nested(context.user_attributes, parts)

    def _resolve_nested(self, data: Any, keys: Sequence[str]) -> Any:
        """Walk a nested dict/object by a sequence of keys.

        Args:
            data: The root dict or object.
            keys: Remaining path segments to traverse.

        Returns:
            The resolved value, or None if any key is missing.
        """
        current = data
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key, None)
            else:
                return None
        return current

    # ------------------------------------------------------------------
    # Operator application
    # ------------------------------------------------------------------

    def _apply_operator(
        self,
        operator: str,
        actual: Any,
        expected: Any,
    ) -> bool:
        """Apply a comparison operator to an actual and expected value.

        All comparisons handle None/missing gracefully by returning False.

        Args:
            operator: Operator name (e.g. "eq", "gt", "contains").
            actual: The resolved context value.
            expected: The expected/target value from the condition.

        Returns:
            True if the comparison succeeds.
        """
        dispatch = {
            "eq": self._op_eq,
            "neq": self._op_neq,
            "in": self._op_in,
            "not_in": self._op_not_in,
            "gt": self._op_gt,
            "lt": self._op_lt,
            "gte": self._op_gte,
            "lte": self._op_lte,
            "contains": self._op_contains,
            "starts_with": self._op_starts_with,
            "ends_with": self._op_ends_with,
            "regex": self._op_regex,
        }
        handler = dispatch.get(operator)
        if handler is None:
            return False
        return handler(actual, expected)

    # -- Individual operators -------------------------------------------

    @staticmethod
    def _op_eq(actual: Any, expected: Any) -> bool:
        """Equality check with case-insensitive string support."""
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.lower() == expected.lower()
        return actual == expected

    @staticmethod
    def _op_neq(actual: Any, expected: Any) -> bool:
        """Inequality check with case-insensitive string support."""
        if actual is None and expected is None:
            return False
        if actual is None or expected is None:
            return True
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.lower() != expected.lower()
        return actual != expected

    @staticmethod
    def _op_in(actual: Any, expected: Any) -> bool:
        """Check if actual is in the expected collection."""
        if actual is None:
            return False
        if not isinstance(expected, (list, tuple, set, frozenset)):
            return False
        if isinstance(actual, str):
            actual_lower = actual.lower()
            return any(
                (isinstance(e, str) and e.lower() == actual_lower) or e == actual
                for e in expected
            )
        return actual in expected

    @staticmethod
    def _op_not_in(actual: Any, expected: Any) -> bool:
        """Check if actual is NOT in the expected collection."""
        if actual is None:
            return True
        if not isinstance(expected, (list, tuple, set, frozenset)):
            return True
        if isinstance(actual, str):
            actual_lower = actual.lower()
            return not any(
                (isinstance(e, str) and e.lower() == actual_lower) or e == actual
                for e in expected
            )
        return actual not in expected

    @staticmethod
    def _op_gt(actual: Any, expected: Any) -> bool:
        """Greater than numeric comparison."""
        try:
            return float(actual) > float(expected)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _op_lt(actual: Any, expected: Any) -> bool:
        """Less than numeric comparison."""
        try:
            return float(actual) < float(expected)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _op_gte(actual: Any, expected: Any) -> bool:
        """Greater than or equal numeric comparison."""
        try:
            return float(actual) >= float(expected)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _op_lte(actual: Any, expected: Any) -> bool:
        """Less than or equal numeric comparison."""
        try:
            return float(actual) <= float(expected)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _op_contains(actual: Any, expected: Any) -> bool:
        """Check if actual contains expected (substring or element).

        Works for strings (case-insensitive substring) and lists (element check).
        """
        if actual is None:
            return False
        if isinstance(actual, str) and isinstance(expected, str):
            return expected.lower() in actual.lower()
        if isinstance(actual, (list, tuple)):
            return expected in actual
        return False

    @staticmethod
    def _op_starts_with(actual: Any, expected: Any) -> bool:
        """Case-insensitive string prefix check."""
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        return actual.lower().startswith(expected.lower())

    @staticmethod
    def _op_ends_with(actual: Any, expected: Any) -> bool:
        """Case-insensitive string suffix check."""
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        return actual.lower().endswith(expected.lower())

    @staticmethod
    def _op_regex(actual: Any, expected: Any) -> bool:
        """Regex pattern match (case-insensitive, partial via re.search).

        Returns False for invalid patterns instead of raising.
        """
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        try:
            return bool(re.search(expected, actual, re.IGNORECASE))
        except re.error:
            logger.debug(
                "SegmentMatcher: invalid regex pattern '%s'", expected
            )
            return False
