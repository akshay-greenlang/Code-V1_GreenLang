# -*- coding: utf-8 -*-
"""
Rule Evaluator - INFRA-008 Targeting Subsystem

Evaluates an ordered list of FlagRule instances against an EvaluationContext
to find the first matching rule. Rules are sorted by priority (ascending --
lower number means higher priority) and evaluated in order. The first rule
whose conditions match the context determines the flag evaluation outcome.

Supported rule types:
    - ``percentage``: Delegates to PercentageRollout for consistent hashing.
    - ``user_list``: Checks if the user ID is in an explicit allow list.
    - ``segment``: Delegates to SegmentMatcher for attribute-based targeting.
    - ``environment``: Checks if the deployment environment matches.
    - ``schedule``: Checks if the current time falls within a time window.

Design principles:
    - Deterministic: same inputs always produce the same output.
    - Fail-safe: if a rule evaluation errors, it is skipped (logged at WARNING).
    - Composable: each rule type delegates to its specialized matcher.

Example:
    >>> from greenlang.infrastructure.feature_flags.targeting.rules import RuleEvaluator
    >>> evaluator = RuleEvaluator()
    >>> matched = evaluator.evaluate(rules, context)
    >>> if matched:
    ...     print(f"Matched rule {matched.rule_id}")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.feature_flags.models import (
    EvaluationContext,
    FlagRule,
)
from greenlang.infrastructure.feature_flags.targeting.percentage import (
    PercentageRollout,
)
from greenlang.infrastructure.feature_flags.targeting.segments import (
    SegmentMatcher,
)

logger = logging.getLogger(__name__)


class RuleEvaluator:
    """Evaluates targeting rules in priority order against an evaluation context.

    Maintains stateless references to the PercentageRollout and SegmentMatcher
    helpers. Rule evaluation is synchronous (no I/O) and safe for concurrent
    use from multiple asyncio tasks.

    Attributes:
        _percentage_rollout: Consistent-hash percentage evaluator.
        _segment_matcher: Attribute-based segment condition evaluator.
    """

    def __init__(self) -> None:
        """Initialize RuleEvaluator with its targeting delegates."""
        self._percentage_rollout = PercentageRollout()
        self._segment_matcher = SegmentMatcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        rules: List[FlagRule],
        context: EvaluationContext,
    ) -> Optional[FlagRule]:
        """Find the first matching rule for the given context.

        Rules are sorted by priority (ascending) before evaluation. Only
        enabled rules are considered. The first rule whose conditions match
        the context is returned.

        Args:
            rules: List of FlagRule instances to evaluate.
            context: The evaluation context with user/tenant/env data.

        Returns:
            The first matching FlagRule, or None if no rule matches.
        """
        if not rules:
            return None

        # Sort by priority ascending (lower number = higher priority)
        sorted_rules = sorted(rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            if not rule.enabled:
                logger.debug(
                    "RuleEvaluator: skipping disabled rule %s", rule.rule_id
                )
                continue

            try:
                matched = self._evaluate_single_rule(rule, context)
                if matched:
                    logger.debug(
                        "RuleEvaluator: rule %s (type=%s, priority=%d) matched "
                        "for user=%s",
                        rule.rule_id,
                        rule.rule_type,
                        rule.priority,
                        context.user_id,
                    )
                    return rule
            except Exception as exc:
                logger.warning(
                    "RuleEvaluator: error evaluating rule %s (type=%s): %s. "
                    "Skipping this rule.",
                    rule.rule_id,
                    rule.rule_type,
                    exc,
                )
                continue

        logger.debug(
            "RuleEvaluator: no rules matched for user=%s (evaluated %d rules)",
            context.user_id,
            len(sorted_rules),
        )
        return None

    # ------------------------------------------------------------------
    # Rule type dispatch
    # ------------------------------------------------------------------

    def _evaluate_single_rule(
        self,
        rule: FlagRule,
        context: EvaluationContext,
    ) -> bool:
        """Dispatch to the appropriate evaluator for the rule type.

        Args:
            rule: The rule to evaluate.
            context: The evaluation context.

        Returns:
            True if the rule matches the context.
        """
        rule_type = rule.rule_type.lower().strip()

        if rule_type == "percentage":
            return self._evaluate_percentage(rule, context)
        if rule_type == "user_list":
            return self._evaluate_user_list(rule, context)
        if rule_type == "segment":
            return self._evaluate_segment(rule, context)
        if rule_type == "environment":
            return self._evaluate_environment(rule, context)
        if rule_type == "schedule":
            return self._evaluate_schedule(rule, context)

        logger.debug(
            "RuleEvaluator: unknown rule_type '%s' for rule %s",
            rule_type,
            rule.rule_id,
        )
        return False

    # ------------------------------------------------------------------
    # Percentage rule
    # ------------------------------------------------------------------

    def _evaluate_percentage(
        self,
        rule: FlagRule,
        context: EvaluationContext,
    ) -> bool:
        """Evaluate a percentage-type rule using consistent hashing.

        Conditions should contain:
            - ``percentage`` (float): Rollout percentage 0.0 - 100.0.

        Args:
            rule: The percentage rule.
            context: The evaluation context.

        Returns:
            True if the user falls within the rollout percentage.
        """
        conditions: Dict[str, Any] = rule.conditions
        percentage = float(conditions.get("percentage", 0.0))
        return self._percentage_rollout.evaluate(
            flag_key=rule.flag_key,
            user_id=context.identity_key,
            rollout_percentage=percentage,
        )

    # ------------------------------------------------------------------
    # User list rule
    # ------------------------------------------------------------------

    def _evaluate_user_list(
        self,
        rule: FlagRule,
        context: EvaluationContext,
    ) -> bool:
        """Evaluate a user-list-type rule.

        Conditions should contain:
            - ``users`` (list[str]): List of user IDs that should match.

        Args:
            rule: The user list rule.
            context: The evaluation context.

        Returns:
            True if the user ID is in the allowed list.
        """
        if not context.user_id:
            return False

        conditions: Dict[str, Any] = rule.conditions
        user_list: List[str] = conditions.get("users", [])

        if not user_list:
            return False

        # Case-insensitive comparison for user IDs
        user_id_lower = context.user_id.lower()
        return any(u.lower() == user_id_lower for u in user_list)

    # ------------------------------------------------------------------
    # Segment rule
    # ------------------------------------------------------------------

    def _evaluate_segment(
        self,
        rule: FlagRule,
        context: EvaluationContext,
    ) -> bool:
        """Evaluate a segment-type rule using the SegmentMatcher.

        The entire rule.conditions dict is passed to SegmentMatcher.matches().

        Args:
            rule: The segment rule.
            context: The evaluation context.

        Returns:
            True if the context matches the segment conditions.
        """
        return self._segment_matcher.matches(context, rule.conditions)

    # ------------------------------------------------------------------
    # Environment rule
    # ------------------------------------------------------------------

    def _evaluate_environment(
        self,
        rule: FlagRule,
        context: EvaluationContext,
    ) -> bool:
        """Evaluate an environment-type rule.

        Conditions should contain:
            - ``environments`` (list[str]): Allowed environment names.

        Args:
            rule: The environment rule.
            context: The evaluation context.

        Returns:
            True if the context environment is in the allowed list.
        """
        conditions: Dict[str, Any] = rule.conditions
        environments: List[str] = conditions.get("environments", [])

        if not environments:
            return False

        ctx_env = context.environment.lower()
        return any(env.lower() == ctx_env for env in environments)

    # ------------------------------------------------------------------
    # Schedule rule
    # ------------------------------------------------------------------

    def _evaluate_schedule(
        self,
        rule: FlagRule,
        context: EvaluationContext,
    ) -> bool:
        """Evaluate a schedule-type rule based on a time window.

        Conditions should contain:
            - ``start_time`` (str): ISO 8601 UTC start time.
            - ``end_time`` (str): ISO 8601 UTC end time.

        The current UTC time is used for comparison. If only start_time is
        provided, the rule matches if now >= start_time. If only end_time is
        provided, the rule matches if now <= end_time.

        Args:
            rule: The schedule rule.
            context: The evaluation context.

        Returns:
            True if the current time is within the schedule window.
        """
        conditions: Dict[str, Any] = rule.conditions
        now = datetime.now(timezone.utc)

        start_str: Optional[str] = conditions.get("start_time")
        end_str: Optional[str] = conditions.get("end_time")

        start_time: Optional[datetime] = None
        end_time: Optional[datetime] = None

        if start_str:
            start_time = self._parse_datetime(start_str)
        if end_str:
            end_time = self._parse_datetime(end_str)

        # If neither is set, the rule does not match
        if start_time is None and end_time is None:
            return False

        if start_time is not None and now < start_time:
            return False

        if end_time is not None and now > end_time:
            return False

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        """Parse an ISO 8601 datetime string into a timezone-aware datetime.

        Args:
            value: ISO 8601 datetime string.

        Returns:
            Parsed datetime in UTC, or None if parsing fails.
        """
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError) as exc:
            logger.debug(
                "RuleEvaluator: failed to parse datetime '%s': %s",
                value,
                exc,
            )
            return None
