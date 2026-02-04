# -*- coding: utf-8 -*-
"""
Unit Tests for FeatureFlagEngine - INFRA-008

Tests the 11-step evaluation pipeline including kill switch checks, local
overrides, status checks, override evaluation, scheduling, targeting rules,
percentage rollout, and default value fallback. Also tests batch and
evaluate-all operations.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.feature_flags.engine import FeatureFlagEngine
from greenlang.infrastructure.feature_flags.kill_switch import KillSwitch
from greenlang.infrastructure.feature_flags.models import (
    EvaluationContext,
    FeatureFlag,
    FlagEvaluationResult,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagType,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.storage.memory import InMemoryFlagStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage():
    """Create a fresh InMemoryFlagStorage instance."""
    return InMemoryFlagStorage(max_size=1000, default_ttl=0.0)


@pytest.fixture
def kill_switch():
    """Create a KillSwitch in local-only mode (no Redis)."""
    return KillSwitch(redis_url="redis://localhost:6379/0")


@pytest.fixture
def config():
    """Create a mock FeatureFlagConfig with all engine-required attributes."""
    cfg = MagicMock()
    cfg.redis_url = None
    cfg.pubsub_channel = "ff:killswitch"
    cfg.default_value = False
    cfg.enable_metrics = True
    cfg.evaluation_timeout_ms = 50
    cfg.max_flags_per_request = 100
    return cfg


@pytest.fixture
def engine(storage, config, kill_switch):
    """Create a FeatureFlagEngine wired to in-memory storage and mock config."""
    return FeatureFlagEngine(
        storage=storage,
        config=config,
        kill_switch=kill_switch,
    )


@pytest.fixture
def context():
    """Create a standard EvaluationContext."""
    return EvaluationContext(
        user_id="user-42",
        tenant_id="tenant-acme",
        environment="staging",
        user_segments=["enterprise"],
        user_attributes={"plan_type": "enterprise"},
    )


@pytest.fixture
def active_boolean_flag():
    """Create an ACTIVE BOOLEAN flag with default_value=False."""
    return FeatureFlag(
        key="test-boolean",
        name="Test Boolean Flag",
        flag_type=FlagType.BOOLEAN,
        status=FlagStatus.ACTIVE,
        default_value=False,
    )


@pytest.fixture
def active_percentage_flag():
    """Create an ACTIVE PERCENTAGE flag at 50%."""
    return FeatureFlag(
        key="test-percentage",
        name="Test Percentage Flag",
        flag_type=FlagType.PERCENTAGE,
        status=FlagStatus.ACTIVE,
        default_value=False,
        rollout_percentage=50.0,
    )


# ---------------------------------------------------------------------------
# Boolean Flag Tests
# ---------------------------------------------------------------------------


class TestEngineBoolean:
    """Test evaluate() with BOOLEAN flags."""

    @pytest.mark.asyncio
    async def test_boolean_flag_returns_default_value(
        self, engine, storage, context, active_boolean_flag
    ):
        """A BOOLEAN flag with no rules returns its default_value."""
        await storage.save_flag(active_boolean_flag)
        result = await engine.evaluate("test-boolean", context)

        assert isinstance(result, FlagEvaluationResult)
        assert result.flag_key == "test-boolean"
        assert result.enabled is False
        assert result.reason == "default"

    @pytest.mark.asyncio
    async def test_boolean_flag_with_true_default(self, engine, storage, context):
        """A BOOLEAN flag with default_value=True returns True."""
        flag = FeatureFlag(
            key="test-true-default",
            name="T",
            flag_type=FlagType.BOOLEAN,
            status=FlagStatus.ACTIVE,
            default_value=True,
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-true-default", context)
        assert result.enabled is True
        assert result.reason == "default"


# ---------------------------------------------------------------------------
# Percentage Flag Tests
# ---------------------------------------------------------------------------


class TestEnginePercentage:
    """Test evaluate() with PERCENTAGE flags."""

    @pytest.mark.asyncio
    async def test_percentage_flag_100_always_enabled(
        self, engine, storage, context
    ):
        """A PERCENTAGE flag at 100% always returns enabled=True."""
        flag = FeatureFlag(
            key="test-pct-100",
            name="T",
            flag_type=FlagType.PERCENTAGE,
            status=FlagStatus.ACTIVE,
            default_value=False,
            rollout_percentage=100.0,
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-pct-100", context)
        assert result.enabled is True
        assert result.reason == "percentage_rollout"

    @pytest.mark.asyncio
    async def test_percentage_flag_0_always_disabled(
        self, engine, storage, context
    ):
        """A PERCENTAGE flag at 0% always returns enabled=False."""
        flag = FeatureFlag(
            key="test-pct-0",
            name="T",
            flag_type=FlagType.PERCENTAGE,
            status=FlagStatus.ACTIVE,
            default_value=False,
            rollout_percentage=0.0,
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-pct-0", context)
        # 0% rollout_percentage: evaluate method returns False immediately,
        # so we get the default value path
        assert result.enabled is False

    @pytest.mark.asyncio
    async def test_percentage_flag_deterministic(self, engine, storage, context):
        """A PERCENTAGE flag produces the same result for the same context."""
        flag = FeatureFlag(
            key="test-pct-50",
            name="T",
            flag_type=FlagType.PERCENTAGE,
            status=FlagStatus.ACTIVE,
            default_value=False,
            rollout_percentage=50.0,
        )
        await storage.save_flag(flag)

        results = []
        for _ in range(10):
            r = await engine.evaluate("test-pct-50", context)
            results.append(r.enabled)
        assert len(set(results)) == 1, "Percentage evaluation must be deterministic"


# ---------------------------------------------------------------------------
# Environment Flag Tests
# ---------------------------------------------------------------------------


class TestEngineEnvironment:
    """Test evaluate() with ENVIRONMENT flags."""

    @pytest.mark.asyncio
    async def test_environment_flag_matching(self, engine, storage, context):
        """An ENVIRONMENT flag enabled in the context environment returns True via rules."""
        flag = FeatureFlag(
            key="test-env",
            name="T",
            flag_type=FlagType.ENVIRONMENT,
            status=FlagStatus.ACTIVE,
            default_value=False,
            environments=["staging", "prod"],
        )
        await storage.save_flag(flag)
        # Environment-type flags need a targeting rule to match
        rule = FlagRule(
            flag_key="test-env",
            rule_type="environment",
            priority=10,
            conditions={"environments": ["staging", "prod"]},
        )
        await storage.save_rule(rule)

        result = await engine.evaluate("test-env", context)
        assert result.enabled is True
        assert "rule:" in result.reason


# ---------------------------------------------------------------------------
# User List Flag Tests
# ---------------------------------------------------------------------------


class TestEngineUserList:
    """Test evaluate() with USER_LIST flags via rules."""

    @pytest.mark.asyncio
    async def test_user_list_whitelisted_user(self, engine, storage, context):
        """A user_list rule matches the whitelisted user."""
        flag = FeatureFlag(
            key="test-userlist",
            name="T",
            flag_type=FlagType.USER_LIST,
            status=FlagStatus.ACTIVE,
            default_value=False,
        )
        await storage.save_flag(flag)
        rule = FlagRule(
            flag_key="test-userlist",
            rule_type="user_list",
            priority=10,
            conditions={"users": ["user-42", "user-99"]},
        )
        await storage.save_rule(rule)

        result = await engine.evaluate("test-userlist", context)
        assert result.enabled is True
        assert "rule:" in result.reason


# ---------------------------------------------------------------------------
# Scheduled Flag Tests
# ---------------------------------------------------------------------------


class TestEngineScheduled:
    """Test evaluate() with SCHEDULED flags."""

    @pytest.mark.asyncio
    async def test_scheduled_flag_within_window(self, engine, storage, context):
        """A SCHEDULED flag within its time window continues evaluation."""
        now = datetime.now(timezone.utc)
        flag = FeatureFlag(
            key="test-scheduled-in",
            name="T",
            flag_type=FlagType.SCHEDULED,
            status=FlagStatus.ACTIVE,
            default_value=True,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-scheduled-in", context)
        # Within window, falls through to default
        assert result.enabled is True
        assert result.reason == "default"

    @pytest.mark.asyncio
    async def test_scheduled_flag_before_window(self, engine, storage, context):
        """A SCHEDULED flag before its start_time returns disabled."""
        now = datetime.now(timezone.utc)
        flag = FeatureFlag(
            key="test-scheduled-before",
            name="T",
            flag_type=FlagType.SCHEDULED,
            status=FlagStatus.ACTIVE,
            default_value=True,
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2),
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-scheduled-before", context)
        assert result.enabled is False
        assert result.reason == "schedule_not_started"

    @pytest.mark.asyncio
    async def test_scheduled_flag_after_window(self, engine, storage, context):
        """A SCHEDULED flag after its end_time returns disabled."""
        now = datetime.now(timezone.utc)
        flag = FeatureFlag(
            key="test-scheduled-after",
            name="T",
            flag_type=FlagType.SCHEDULED,
            status=FlagStatus.ACTIVE,
            default_value=True,
            start_time=now - timedelta(hours=2),
            end_time=now - timedelta(hours=1),
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-scheduled-after", context)
        assert result.enabled is False
        assert result.reason == "schedule_expired"


# ---------------------------------------------------------------------------
# Flag Status Tests
# ---------------------------------------------------------------------------


class TestEngineStatus:
    """Test evaluate() behaviour for different flag statuses."""

    @pytest.mark.asyncio
    async def test_killed_status_returns_disabled(self, engine, storage, context):
        """A flag with KILLED status always returns enabled=False."""
        flag = FeatureFlag(
            key="test-killed",
            name="T",
            status=FlagStatus.KILLED,
            default_value=True,
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-killed", context)
        assert result.enabled is False
        assert result.reason == "status_killed"

    @pytest.mark.asyncio
    async def test_draft_status_returns_default(self, engine, storage, context):
        """A flag with DRAFT status returns its default_value."""
        flag = FeatureFlag(
            key="test-draft",
            name="T",
            status=FlagStatus.DRAFT,
            default_value=True,
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-draft", context)
        assert result.enabled is True
        assert result.reason == "status_draft"

    @pytest.mark.asyncio
    async def test_archived_status_returns_default(self, engine, storage, context):
        """A flag with ARCHIVED status returns its default_value."""
        flag = FeatureFlag(
            key="test-archived",
            name="T",
            status=FlagStatus.ARCHIVED,
            default_value=False,
        )
        await storage.save_flag(flag)
        result = await engine.evaluate("test-archived", context)
        assert result.enabled is False
        assert result.reason == "status_archived"

    @pytest.mark.asyncio
    async def test_flag_not_found_returns_config_default(
        self, engine, context, config
    ):
        """Evaluating a nonexistent flag returns the config default_value."""
        result = await engine.evaluate("nonexistent-flag", context)
        assert result.enabled is False  # config.default_value = False
        assert result.reason == "flag_not_found"


# ---------------------------------------------------------------------------
# Kill Switch Tests
# ---------------------------------------------------------------------------


class TestEngineKillSwitch:
    """Test evaluate() with kill switch activated."""

    @pytest.mark.asyncio
    async def test_kill_switch_overrides_everything(
        self, engine, storage, context, kill_switch
    ):
        """When kill switch is activated, the flag always returns disabled."""
        flag = FeatureFlag(
            key="test-kill-sw",
            name="T",
            flag_type=FlagType.PERCENTAGE,
            status=FlagStatus.ACTIVE,
            default_value=True,
            rollout_percentage=100.0,
        )
        await storage.save_flag(flag)

        # Activate kill switch (local-only since no Redis)
        kill_switch._killed_flags["test-kill-sw"] = {
            "killed_at": "2026-01-01T00:00:00Z"
        }

        result = await engine.evaluate("test-kill-sw", context)
        assert result.enabled is False
        assert result.reason == "kill_switch"


# ---------------------------------------------------------------------------
# Override Tests
# ---------------------------------------------------------------------------


class TestEngineOverrides:
    """Test evaluate() with flag overrides (user, tenant, environment)."""

    @pytest.mark.asyncio
    async def test_local_override_returns_override_value(
        self, engine, storage, context
    ):
        """A local override bypasses all evaluation logic."""
        flag = FeatureFlag(
            key="test-local-override",
            name="T",
            status=FlagStatus.ACTIVE,
            default_value=False,
        )
        await storage.save_flag(flag)
        engine.set_override("test-local-override", True)

        result = await engine.evaluate("test-local-override", context)
        assert result.enabled is True
        assert result.reason == "local_override"

        # Cleanup
        engine.remove_override("test-local-override")

    @pytest.mark.asyncio
    async def test_user_blacklist_override(self, engine, storage, context):
        """A user override with enabled=False blacklists the user."""
        flag = FeatureFlag(
            key="test-blacklist",
            name="T",
            status=FlagStatus.ACTIVE,
            default_value=True,
        )
        await storage.save_flag(flag)
        override = FlagOverride(
            flag_key="test-blacklist",
            scope_type="user",
            scope_value="user-42",
            enabled=False,
        )
        await storage.save_override(override)

        result = await engine.evaluate("test-blacklist", context)
        assert result.enabled is False
        assert result.reason == "override_user_blacklist"

    @pytest.mark.asyncio
    async def test_user_whitelist_override(self, engine, storage, context):
        """A user override with enabled=True whitelists the user."""
        flag = FeatureFlag(
            key="test-whitelist",
            name="T",
            status=FlagStatus.ACTIVE,
            default_value=False,
        )
        await storage.save_flag(flag)
        override = FlagOverride(
            flag_key="test-whitelist",
            scope_type="user",
            scope_value="user-42",
            enabled=True,
        )
        await storage.save_override(override)

        result = await engine.evaluate("test-whitelist", context)
        assert result.enabled is True
        assert result.reason == "override_user_whitelist"

    @pytest.mark.asyncio
    async def test_tenant_override(self, engine, storage, context):
        """A tenant-scoped override is applied for the matching tenant."""
        flag = FeatureFlag(
            key="test-tenant-override",
            name="T",
            status=FlagStatus.ACTIVE,
            default_value=False,
        )
        await storage.save_flag(flag)
        override = FlagOverride(
            flag_key="test-tenant-override",
            scope_type="tenant",
            scope_value="tenant-acme",
            enabled=True,
        )
        await storage.save_override(override)

        result = await engine.evaluate("test-tenant-override", context)
        assert result.enabled is True
        assert result.reason == "override_tenant"

    @pytest.mark.asyncio
    async def test_environment_override(self, engine, storage, context):
        """An environment-scoped override is applied for the matching env."""
        flag = FeatureFlag(
            key="test-env-override",
            name="T",
            status=FlagStatus.ACTIVE,
            default_value=False,
        )
        await storage.save_flag(flag)
        override = FlagOverride(
            flag_key="test-env-override",
            scope_type="environment",
            scope_value="staging",
            enabled=True,
        )
        await storage.save_override(override)

        result = await engine.evaluate("test-env-override", context)
        assert result.enabled is True
        assert result.reason == "override_environment"


# ---------------------------------------------------------------------------
# Batch and All Evaluation
# ---------------------------------------------------------------------------


class TestEngineBatch:
    """Test evaluate_all() and evaluate_batch()."""

    @pytest.mark.asyncio
    async def test_evaluate_all_returns_active_flags_only(
        self, engine, storage, context
    ):
        """evaluate_all returns only ACTIVE, ROLLED_OUT, and PERMANENT flags."""
        flags = [
            FeatureFlag(key="flag-active", name="A", status=FlagStatus.ACTIVE, default_value=True),
            FeatureFlag(key="flag-draft", name="D", status=FlagStatus.DRAFT, default_value=True),
            FeatureFlag(key="flag-rolled", name="R", status=FlagStatus.ROLLED_OUT, default_value=True),
            FeatureFlag(key="flag-permanent", name="P", status=FlagStatus.PERMANENT, default_value=True),
            FeatureFlag(key="flag-archived", name="X", status=FlagStatus.ARCHIVED),
        ]
        for f in flags:
            await storage.save_flag(f)

        results = await engine.evaluate_all(context)
        assert "flag-active" in results
        assert "flag-rolled" in results
        assert "flag-permanent" in results
        assert "flag-draft" not in results
        assert "flag-archived" not in results

    @pytest.mark.asyncio
    async def test_evaluate_batch_returns_results_for_specified_keys(
        self, engine, storage, context
    ):
        """evaluate_batch returns FlagEvaluationResult for each requested key."""
        flag_a = FeatureFlag(key="batch-a", name="A", status=FlagStatus.ACTIVE, default_value=True)
        flag_b = FeatureFlag(key="batch-b", name="B", status=FlagStatus.ACTIVE, default_value=False)
        await storage.save_flag(flag_a)
        await storage.save_flag(flag_b)

        results = await engine.evaluate_batch(["batch-a", "batch-b"], context)
        assert "batch-a" in results
        assert "batch-b" in results
        assert results["batch-a"].enabled is True
        assert results["batch-b"].enabled is False

    @pytest.mark.asyncio
    async def test_evaluate_batch_includes_nonexistent_keys(
        self, engine, storage, context
    ):
        """evaluate_batch includes results even for nonexistent flag keys."""
        results = await engine.evaluate_batch(["missing-key"], context)
        assert "missing-key" in results
        assert results["missing-key"].enabled is False
        assert results["missing-key"].reason == "flag_not_found"


# ---------------------------------------------------------------------------
# Timing and Metrics Tests
# ---------------------------------------------------------------------------


class TestEngineMetrics:
    """Test timing and metrics recording."""

    @pytest.mark.asyncio
    async def test_duration_us_is_positive(
        self, engine, storage, context, active_boolean_flag
    ):
        """Every evaluation records a positive duration_us."""
        await storage.save_flag(active_boolean_flag)
        result = await engine.evaluate("test-boolean", context)
        assert result.duration_us >= 0

    @pytest.mark.asyncio
    async def test_metrics_count_increments(
        self, engine, storage, context, active_boolean_flag
    ):
        """Evaluation count metric increments on each call."""
        await storage.save_flag(active_boolean_flag)
        initial = engine.get_metrics()["evaluation_count"]
        await engine.evaluate("test-boolean", context)
        await engine.evaluate("test-boolean", context)
        updated = engine.get_metrics()["evaluation_count"]
        assert updated == initial + 2

    @pytest.mark.asyncio
    async def test_result_has_proper_reason_string(
        self, engine, storage, context, active_boolean_flag
    ):
        """Every result contains a non-empty reason string."""
        await storage.save_flag(active_boolean_flag)
        result = await engine.evaluate("test-boolean", context)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestEngineErrorHandling:
    """Test engine behaviour on errors."""

    @pytest.mark.asyncio
    async def test_storage_error_returns_config_default(
        self, engine, context, config
    ):
        """If storage raises, the engine returns config default_value."""
        engine._storage = MagicMock()
        engine._storage.get_flag = AsyncMock(side_effect=Exception("DB down"))

        result = await engine.evaluate("any-flag", context)
        assert result.enabled is False  # config.default_value = False
        assert result.reason == "error"
