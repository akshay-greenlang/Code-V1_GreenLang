# -*- coding: utf-8 -*-
"""
Feature Flag Engine - INFRA-008

Core evaluation engine for the GreenLang feature flag system. Implements an
11-step evaluation pipeline that deterministically resolves a flag key and
evaluation context into an enabled/disabled result with full provenance:

    1. Check kill switch (instant, local memory)
    2. Check local overrides (for testing/development)
    3. Get flag from storage
    4. Verify flag is active
    5. Check user blacklist (overrides with enabled=False)
    6. Check user whitelist (overrides with enabled=True)
    7. Check tenant overrides
    8. Check scheduling (start_time / end_time)
    9. Evaluate targeting rules (sorted by priority)
    10. Evaluate percentage rollout
    11. Return default value

Also provides batch evaluation (``evaluate_batch``) and full-flag-set
evaluation (``evaluate_all``) for dashboard and SDK bootstrap use cases.

Design principles:
    - Deterministic: same inputs always produce the same output.
    - Observable: every evaluation records its reason and duration.
    - Fail-safe: if storage or rules error, the flag returns its default value.
    - Fast: local checks short-circuit before any I/O.

Example:
    >>> engine = FeatureFlagEngine(storage=memory_store, config=config)
    >>> result = await engine.evaluate("enable-scope3-calc", context)
    >>> print(result.enabled, result.reason)
    True rule:r-42
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig
from greenlang.infrastructure.feature_flags.kill_switch import KillSwitch
from greenlang.infrastructure.feature_flags.models import (
    EvaluationContext,
    FeatureFlag,
    FlagEvaluationResult,
    FlagOverride,
    FlagStatus,
    FlagType,
)
from greenlang.infrastructure.feature_flags.storage.base import IFlagStorage
from greenlang.infrastructure.feature_flags.targeting.percentage import (
    PercentageRollout,
)
from greenlang.infrastructure.feature_flags.targeting.rules import RuleEvaluator

logger = logging.getLogger(__name__)


class FeatureFlagEngine:
    """Core evaluation engine for feature flags.

    Orchestrates the full evaluation pipeline from kill switch check through
    default value fallback. Maintains a local override dict for testing,
    a kill switch instance for emergency shutoff, and delegates targeting
    logic to the RuleEvaluator and PercentageRollout helpers.

    Attributes:
        _storage: Storage backend implementing IFlagStorage.
        _config: Engine configuration.
        _kill_switch: Emergency kill switch (Redis Pub/Sub backed).
        _rule_evaluator: Targeting rule evaluation engine.
        _percentage_rollout: Consistent-hash percentage rollout.
        _local_overrides: Local overrides dict (flag_key -> bool) for testing.
        _metrics: Simple evaluation metrics (count, total duration).
    """

    def __init__(
        self,
        storage: IFlagStorage,
        config: Optional[FeatureFlagConfig] = None,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        """Initialize the FeatureFlagEngine.

        Args:
            storage: Storage backend implementing IFlagStorage.
            config: Engine configuration. Uses defaults if not provided.
            kill_switch: Optional pre-configured KillSwitch instance. If not
                provided, one is created from config.redis_url.
        """
        self._storage = storage
        self._config = config or FeatureFlagConfig()

        self._kill_switch = kill_switch or KillSwitch(
            redis_url=self._config.redis_url or "redis://localhost:6379/0",
            channel=self._config.pubsub_channel,
        )
        self._rule_evaluator = RuleEvaluator()
        self._percentage_rollout = PercentageRollout()

        self._local_overrides: Dict[str, bool] = {}

        # Evaluation metrics
        self._metrics: Dict[str, Any] = {
            "evaluation_count": 0,
            "total_duration_ns": 0,
            "errors": 0,
        }

        logger.info(
            "FeatureFlagEngine initialized (metrics=%s, timeout=%dms)",
            self._config.enable_metrics,
            self._config.evaluation_timeout_ms,
        )

    # ------------------------------------------------------------------
    # Override management (for testing and development)
    # ------------------------------------------------------------------

    def set_override(self, flag_key: str, value: bool) -> None:
        """Set a local override for a flag (bypasses all evaluation logic).

        Useful for unit tests and local development. Overrides are checked
        at step 2 of the evaluation pipeline.

        Args:
            flag_key: The flag key to override.
            value: The override value (True = enabled, False = disabled).
        """
        self._local_overrides[flag_key] = value
        logger.info(
            "FeatureFlagEngine: local override set: %s=%s", flag_key, value
        )

    def remove_override(self, flag_key: str) -> None:
        """Remove a local override for a flag.

        Args:
            flag_key: The flag key whose override to remove.
        """
        removed = self._local_overrides.pop(flag_key, None)
        if removed is not None:
            logger.info(
                "FeatureFlagEngine: local override removed: %s", flag_key
            )

    def clear_overrides(self) -> None:
        """Remove all local overrides."""
        count = len(self._local_overrides)
        self._local_overrides.clear()
        logger.info(
            "FeatureFlagEngine: cleared %d local overrides", count
        )

    # ------------------------------------------------------------------
    # Kill switch access
    # ------------------------------------------------------------------

    @property
    def kill_switch(self) -> KillSwitch:
        """Access the kill switch instance.

        Returns:
            The KillSwitch used by this engine.
        """
        return self._kill_switch

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return evaluation metrics.

        Returns:
            Dict with evaluation_count, total_duration_ns, errors, and
            avg_duration_us (average evaluation duration in microseconds).
        """
        count = self._metrics["evaluation_count"]
        total_ns = self._metrics["total_duration_ns"]
        avg_us = (total_ns / count / 1000) if count > 0 else 0
        return {
            **self._metrics,
            "avg_duration_us": round(avg_us, 2),
        }

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        flag_key: str,
        context: EvaluationContext,
    ) -> FlagEvaluationResult:
        """Evaluate a single feature flag.

        Executes the full 11-step evaluation pipeline and returns a result
        with the enabled state, the reason for the decision, and timing
        metadata.

        Args:
            flag_key: Unique flag identifier.
            context: Evaluation context with user/tenant/env data.

        Returns:
            FlagEvaluationResult with the resolved state and reason.
        """
        start_ns = time.perf_counter_ns()

        try:
            result = await self._evaluate_pipeline(flag_key, context)
        except Exception as exc:
            logger.error(
                "FeatureFlagEngine: unexpected error evaluating '%s': %s",
                flag_key,
                exc,
                exc_info=True,
            )
            self._metrics["errors"] += 1
            result = self._make_result(
                flag_key=flag_key,
                enabled=self._config.default_value,
                reason="error",
            )

        # Record timing
        elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_us = elapsed_ns // 1000
        result = result.model_copy(update={"duration_us": elapsed_us})

        # Update metrics
        if self._config.enable_metrics:
            self._metrics["evaluation_count"] += 1
            self._metrics["total_duration_ns"] += elapsed_ns

        # Warn on slow evaluations
        elapsed_ms = elapsed_ns / 1_000_000
        if elapsed_ms > self._config.evaluation_timeout_ms:
            logger.warning(
                "FeatureFlagEngine: slow evaluation for '%s' "
                "(%.2f ms > %d ms threshold)",
                flag_key,
                elapsed_ms,
                self._config.evaluation_timeout_ms,
            )

        if self._config.enable_metrics:
            logger.debug(
                "FeatureFlagEngine: evaluate '%s' -> enabled=%s "
                "reason=%s duration=%dus",
                flag_key,
                result.enabled,
                result.reason,
                elapsed_us,
            )

        return result

    async def evaluate_all(
        self,
        context: EvaluationContext,
    ) -> Dict[str, bool]:
        """Evaluate all active flags and return a dict of flag_key -> enabled.

        Intended for SDK bootstrap and dashboard views. Only flags with
        status ACTIVE, ROLLED_OUT, or PERMANENT are included.

        Args:
            context: Evaluation context.

        Returns:
            Dict mapping flag_key to its enabled boolean.
        """
        all_flags = await self._storage.get_all_flags()
        results: Dict[str, bool] = {}

        active_statuses = {FlagStatus.ACTIVE, FlagStatus.ROLLED_OUT, FlagStatus.PERMANENT}

        for flag in all_flags:
            if flag.status not in active_statuses:
                continue
            evaluation = await self.evaluate(flag.key, context)
            results[flag.key] = evaluation.enabled

        return results

    async def evaluate_batch(
        self,
        flag_keys: List[str],
        context: EvaluationContext,
    ) -> Dict[str, FlagEvaluationResult]:
        """Evaluate a batch of flag keys.

        Args:
            flag_keys: List of flag keys to evaluate.
            context: Evaluation context.

        Returns:
            Dict mapping flag_key to its FlagEvaluationResult.
        """
        max_keys = self._config.max_flags_per_request
        keys_to_eval = flag_keys[:max_keys]
        if len(flag_keys) > max_keys:
            logger.warning(
                "FeatureFlagEngine: batch size %d exceeds max %d, truncating",
                len(flag_keys),
                max_keys,
            )

        results: Dict[str, FlagEvaluationResult] = {}
        for key in keys_to_eval:
            results[key] = await self.evaluate(key, context)
        return results

    # ------------------------------------------------------------------
    # 11-Step evaluation pipeline
    # ------------------------------------------------------------------

    async def _evaluate_pipeline(
        self,
        flag_key: str,
        context: EvaluationContext,
    ) -> FlagEvaluationResult:
        """Execute the full 11-step evaluation pipeline.

        Steps:
            1. Kill switch check
            2. Local override check
            3. Load flag from storage
            4. Verify flag is active
            5. User blacklist (overrides with enabled=False)
            6. User whitelist (overrides with enabled=True)
            7. Tenant overrides
            8. Schedule check (start_time / end_time)
            9. Targeting rules
            10. Percentage rollout
            11. Default value

        Args:
            flag_key: Flag identifier.
            context: Evaluation context.

        Returns:
            FlagEvaluationResult from the first matching step.
        """
        # Step 1: Kill switch (instant, no I/O)
        if self._kill_switch.is_killed(flag_key):
            return self._make_result(
                flag_key=flag_key,
                enabled=False,
                reason="kill_switch",
            )

        # Step 2: Local overrides (for testing, no I/O)
        if flag_key in self._local_overrides:
            return self._make_result(
                flag_key=flag_key,
                enabled=self._local_overrides[flag_key],
                reason="local_override",
            )

        # Step 3: Load flag from storage
        flag = await self._storage.get_flag(flag_key)
        if flag is None:
            return self._make_result(
                flag_key=flag_key,
                enabled=self._config.default_value,
                reason="flag_not_found",
            )

        # Step 4: Verify flag is active
        result = self._check_flag_status(flag)
        if result is not None:
            return result

        # Steps 5-7: Check overrides (user blacklist, whitelist, tenant)
        overrides = await self._storage.get_overrides(flag_key)
        override_result = self._evaluate_overrides(flag, overrides, context)
        if override_result is not None:
            return override_result

        # Step 8: Schedule check
        schedule_result = self._check_schedule(flag)
        if schedule_result is not None:
            return schedule_result

        # Step 9: Targeting rules
        rules = await self._storage.get_rules(flag_key)
        if rules:
            matched_rule = self._rule_evaluator.evaluate(rules, context)
            if matched_rule is not None:
                return self._make_result(
                    flag_key=flag_key,
                    enabled=True,
                    reason=f"rule:{matched_rule.rule_id}",
                    rule_id=matched_rule.rule_id,
                )

        # Step 10: Percentage rollout
        if flag.flag_type in (FlagType.PERCENTAGE, FlagType.MULTIVARIATE):
            if flag.rollout_percentage > 0.0:
                in_rollout = self._percentage_rollout.evaluate(
                    flag_key=flag.key,
                    user_id=context.identity_key,
                    rollout_percentage=flag.rollout_percentage,
                )
                if in_rollout:
                    # For multivariate, also resolve the variant
                    variant_key = None
                    if flag.flag_type == FlagType.MULTIVARIATE:
                        variants = await self._storage.get_variants(flag_key)
                        if variants:
                            variant_key = self._percentage_rollout.get_variant(
                                flag_key=flag.key,
                                user_id=context.identity_key,
                                variants=variants,
                            )
                    return self._make_result(
                        flag_key=flag_key,
                        enabled=True,
                        reason="percentage_rollout",
                        variant_key=variant_key,
                    )
                else:
                    return self._make_result(
                        flag_key=flag_key,
                        enabled=False,
                        reason="percentage_rollout_excluded",
                    )

        # Step 11: Default value
        return self._make_result(
            flag_key=flag_key,
            enabled=bool(flag.default_value),
            reason="default",
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _check_flag_status(self, flag: FeatureFlag) -> Optional[FlagEvaluationResult]:
        """Step 4: Check flag lifecycle status.

        Returns a result for non-evaluable statuses (DRAFT, ARCHIVED, KILLED).
        Returns None if the flag should continue through the pipeline.

        Args:
            flag: The loaded feature flag.

        Returns:
            FlagEvaluationResult if the flag is not active, or None to continue.
        """
        if flag.status == FlagStatus.KILLED:
            return self._make_result(
                flag_key=flag.key,
                enabled=False,
                reason="status_killed",
            )

        if flag.status == FlagStatus.ARCHIVED:
            return self._make_result(
                flag_key=flag.key,
                enabled=bool(flag.default_value),
                reason="status_archived",
            )

        if flag.status == FlagStatus.DRAFT:
            return self._make_result(
                flag_key=flag.key,
                enabled=bool(flag.default_value),
                reason="status_draft",
            )

        # ACTIVE, ROLLED_OUT, PERMANENT continue through the pipeline
        return None

    def _evaluate_overrides(
        self,
        flag: FeatureFlag,
        overrides: List[FlagOverride],
        context: EvaluationContext,
    ) -> Optional[FlagEvaluationResult]:
        """Steps 5-7: Evaluate user and tenant overrides.

        Checks overrides in order:
            5. User blacklist (scope_type=user, enabled=False)
            6. User whitelist (scope_type=user, enabled=True)
            7. Tenant overrides (scope_type=tenant)

        Expired overrides are silently skipped.

        Args:
            flag: The loaded feature flag.
            overrides: All overrides for this flag.
            context: The evaluation context.

        Returns:
            FlagEvaluationResult if an override matches, or None to continue.
        """
        if not overrides:
            return None

        now = datetime.now(timezone.utc)

        for override in overrides:
            # Skip expired overrides
            if override.expires_at is not None and override.expires_at < now:
                continue

            # Step 5 & 6: User-scoped overrides
            if override.scope_type == "user" and context.user_id:
                if override.scope_value.lower() == context.user_id.lower():
                    if not override.enabled:
                        # Step 5: Blacklist
                        return self._make_result(
                            flag_key=flag.key,
                            enabled=False,
                            reason="override_user_blacklist",
                            variant_key=override.variant_key,
                        )
                    else:
                        # Step 6: Whitelist
                        return self._make_result(
                            flag_key=flag.key,
                            enabled=True,
                            reason="override_user_whitelist",
                            variant_key=override.variant_key,
                        )

            # Step 7: Tenant-scoped overrides
            if override.scope_type == "tenant" and context.tenant_id:
                if override.scope_value.lower() == context.tenant_id.lower():
                    return self._make_result(
                        flag_key=flag.key,
                        enabled=override.enabled,
                        reason="override_tenant",
                        variant_key=override.variant_key,
                    )

            # Environment-scoped overrides
            if override.scope_type == "environment":
                if override.scope_value.lower() == context.environment.lower():
                    return self._make_result(
                        flag_key=flag.key,
                        enabled=override.enabled,
                        reason="override_environment",
                        variant_key=override.variant_key,
                    )

        return None

    def _check_schedule(self, flag: FeatureFlag) -> Optional[FlagEvaluationResult]:
        """Step 8: Check flag scheduling window.

        For SCHEDULED flags, returns disabled if the current time is outside
        the [start_time, end_time] window. For non-scheduled flags, also
        respects start_time/end_time if they are set.

        Args:
            flag: The loaded feature flag.

        Returns:
            FlagEvaluationResult if the schedule excludes the current time,
            or None to continue evaluation.
        """
        now = datetime.now(timezone.utc)

        if flag.start_time is not None and now < flag.start_time:
            return self._make_result(
                flag_key=flag.key,
                enabled=False,
                reason="schedule_not_started",
            )

        if flag.end_time is not None and now > flag.end_time:
            return self._make_result(
                flag_key=flag.key,
                enabled=False,
                reason="schedule_expired",
            )

        return None

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(
        flag_key: str,
        enabled: bool,
        reason: str,
        rule_id: Optional[str] = None,
        variant_key: Optional[str] = None,
    ) -> FlagEvaluationResult:
        """Build a FlagEvaluationResult.

        Args:
            flag_key: The flag key.
            enabled: Resolved enabled state.
            reason: Reason string for observability.
            rule_id: Matched rule ID, if any.
            variant_key: Selected variant key, if any.

        Returns:
            A fully populated FlagEvaluationResult.
        """
        return FlagEvaluationResult(
            flag_key=flag_key,
            enabled=enabled,
            reason=reason,
            rule_id=rule_id,
            variant_key=variant_key,
        )
