# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Versioning: SemVer parsing and comparison,
breaking change detection, compatibility matrix, canary deployment,
rollback control, and version migration.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    @classmethod
    def parse(cls, version: str) -> SemVer:
        pattern = re.compile(
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-([\w.]+))?(?:\+([\w.]+))?$"
        )
        m = pattern.match(version)
        if not m:
            raise ValueError(f"Invalid semantic version: {version}")
        return cls(
            major=int(m.group(1)),
            minor=int(m.group(2)),
            patch=int(m.group(3)),
            prerelease=m.group(4) or "",
            build=m.group(5) or "",
        )

    def __str__(self) -> str:
        v = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            v += f"-{self.prerelease}"
        if self.build:
            v += f"+{self.build}"
        return v

    def _tuple(self) -> Tuple[int, int, int, bool, str]:
        return (
            self.major,
            self.minor,
            self.patch,
            self.prerelease == "",
            self.prerelease,
        )

    def __lt__(self, other: SemVer) -> bool:
        return self._tuple() < other._tuple()

    def __le__(self, other: SemVer) -> bool:
        return self._tuple() <= other._tuple()

    def __gt__(self, other: SemVer) -> bool:
        return self._tuple() > other._tuple()

    def __ge__(self, other: SemVer) -> bool:
        return self._tuple() >= other._tuple()

    def bump_major(self) -> SemVer:
        return SemVer(self.major + 1, 0, 0)

    def bump_minor(self) -> SemVer:
        return SemVer(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemVer:
        return SemVer(self.major, self.minor, self.patch + 1)

    def satisfies(self, constraint: str) -> bool:
        if constraint == "*":
            return True
        if constraint.startswith("^"):
            base = SemVer.parse(constraint[1:])
            if self.major != base.major:
                return False
            return self >= base
        if constraint.startswith("~"):
            base = SemVer.parse(constraint[1:])
            if self.major != base.major or self.minor != base.minor:
                return False
            return self >= base
        if constraint.startswith(">="):
            base = SemVer.parse(constraint[2:])
            return self >= base
        return str(self) == constraint


class BreakingChangeType(str, Enum):
    REMOVED_FIELD = "removed_field"
    TYPE_CHANGE = "type_change"
    REMOVED_ENDPOINT = "removed_endpoint"
    NONE = "none"


@dataclass
class BreakingChange:
    change_type: BreakingChangeType
    field: str
    description: str


def detect_breaking_changes(
    old_schema: Dict[str, Any],
    new_schema: Dict[str, Any],
) -> List[BreakingChange]:
    changes: List[BreakingChange] = []
    old_fields = old_schema.get("fields", {})
    new_fields = new_schema.get("fields", {})
    for field_name, field_type in old_fields.items():
        if field_name not in new_fields:
            changes.append(BreakingChange(
                BreakingChangeType.REMOVED_FIELD,
                field_name,
                f"Field '{field_name}' was removed",
            ))
        elif new_fields[field_name] != field_type:
            changes.append(BreakingChange(
                BreakingChangeType.TYPE_CHANGE,
                field_name,
                f"Field '{field_name}' type changed from {field_type} to {new_fields[field_name]}",
            ))
    return changes


@dataclass
class CompatibilityMatrix:
    entries: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def set_compatible(self, agent_a: str, agent_b: str, compatible: bool) -> None:
        self.entries.setdefault(agent_a, {})[agent_b] = compatible
        self.entries.setdefault(agent_b, {})[agent_a] = compatible

    def is_compatible(self, agent_a: str, agent_b: str) -> bool:
        return self.entries.get(agent_a, {}).get(agent_b, False)


class CanaryController:
    def __init__(
        self,
        agent_key: str,
        steps: List[float],
        metric_fn: Optional[Callable[[], Coroutine[Any, Any, Dict[str, float]]]] = None,
        error_threshold: float = 0.05,
    ) -> None:
        self.agent_key = agent_key
        self.steps = steps
        self.current_step = 0
        self.traffic_pct = 0.0
        self.status = "pending"
        self._metric_fn = metric_fn
        self._error_threshold = error_threshold

    async def start(self) -> None:
        self.status = "running"
        self.current_step = 0
        self.traffic_pct = self.steps[0]

    async def promote_step(self) -> bool:
        if self._metric_fn:
            metrics = await self._metric_fn()
            if metrics.get("error_rate", 0) > self._error_threshold:
                await self.auto_rollback()
                return False
        self.current_step += 1
        if self.current_step >= len(self.steps):
            self.status = "completed"
            self.traffic_pct = 100.0
            return True
        self.traffic_pct = self.steps[self.current_step]
        return True

    async def auto_rollback(self) -> None:
        self.status = "rolled_back"
        self.traffic_pct = 0.0
        self.current_step = 0


class RollbackController:
    def __init__(
        self,
        error_rate_threshold: float = 0.05,
        latency_threshold_ms: float = 500.0,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.error_rate_threshold = error_rate_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.cooldown_seconds = cooldown_seconds
        self.last_rollback_at: Optional[float] = None
        self.rollback_count: int = 0

    def should_rollback(
        self, error_rate: float, latency_ms: float
    ) -> Tuple[bool, str]:
        if self.last_rollback_at is not None:
            elapsed = time.time() - self.last_rollback_at
            if elapsed < self.cooldown_seconds:
                return False, "in cooldown"

        if error_rate > self.error_rate_threshold:
            return True, f"error_rate {error_rate:.2%} exceeds threshold"
        if latency_ms > self.latency_threshold_ms:
            return True, f"latency {latency_ms:.0f}ms exceeds threshold"
        return False, "healthy"

    def record_rollback(self) -> None:
        self.last_rollback_at = time.time()
        self.rollback_count += 1


@dataclass
class MigrationStep:
    version: str
    up: Callable[[], Coroutine[Any, Any, None]]
    down: Optional[Callable[[], Coroutine[Any, Any, None]]] = None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def v1() -> SemVer:
    return SemVer.parse("1.2.3")


@pytest.fixture
def v2() -> SemVer:
    return SemVer.parse("2.0.0")


@pytest.fixture
def compat_matrix() -> CompatibilityMatrix:
    return CompatibilityMatrix()


# ============================================================================
# Tests
# ============================================================================


class TestSemVer:
    """Tests for semantic version parsing, comparison, and range matching."""

    def test_semver_parse(self) -> None:
        v = SemVer.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_semver_parse_prerelease(self) -> None:
        v = SemVer.parse("1.0.0-alpha.1")
        assert v.prerelease == "alpha.1"

    def test_semver_parse_build(self) -> None:
        v = SemVer.parse("1.0.0+build.42")
        assert v.build == "build.42"

    def test_semver_parse_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid semantic version"):
            SemVer.parse("not-valid")

    def test_semver_comparison(self) -> None:
        assert SemVer.parse("1.0.0") < SemVer.parse("2.0.0")
        assert SemVer.parse("1.1.0") > SemVer.parse("1.0.0")
        assert SemVer.parse("1.0.1") > SemVer.parse("1.0.0")

    def test_semver_prerelease_comparison(self) -> None:
        assert SemVer.parse("1.0.0-alpha") < SemVer.parse("1.0.0")

    def test_semver_range_satisfies_caret(self) -> None:
        v = SemVer.parse("1.2.3")
        assert v.satisfies("^1.0.0") is True
        assert v.satisfies("^2.0.0") is False

    def test_semver_range_satisfies_tilde(self) -> None:
        v = SemVer.parse("1.2.3")
        assert v.satisfies("~1.2.0") is True
        assert v.satisfies("~1.3.0") is False

    def test_semver_range_satisfies_gte(self) -> None:
        v = SemVer.parse("2.0.0")
        assert v.satisfies(">=1.0.0") is True
        assert v.satisfies(">=3.0.0") is False

    def test_semver_range_satisfies_wildcard(self) -> None:
        assert SemVer.parse("99.99.99").satisfies("*") is True

    def test_semver_bump_major(self, v1: SemVer) -> None:
        bumped = v1.bump_major()
        assert bumped == SemVer(2, 0, 0)

    def test_semver_bump_minor(self, v1: SemVer) -> None:
        bumped = v1.bump_minor()
        assert bumped == SemVer(1, 3, 0)

    def test_semver_bump_patch(self, v1: SemVer) -> None:
        bumped = v1.bump_patch()
        assert bumped == SemVer(1, 2, 4)

    def test_semver_str_roundtrip(self) -> None:
        v = SemVer.parse("1.2.3-beta.1+build.5")
        assert str(v) == "1.2.3-beta.1+build.5"


class TestBreakingChangeDetection:
    """Tests for detecting breaking API/schema changes."""

    def test_breaking_change_detection_removed_field(self) -> None:
        old = {"fields": {"co2": "float", "scope": "int"}}
        new = {"fields": {"co2": "float"}}  # scope removed
        changes = detect_breaking_changes(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == BreakingChangeType.REMOVED_FIELD
        assert changes[0].field == "scope"

    def test_breaking_change_detection_type_change(self) -> None:
        old = {"fields": {"co2": "float"}}
        new = {"fields": {"co2": "string"}}
        changes = detect_breaking_changes(old, new)
        assert len(changes) == 1
        assert changes[0].change_type == BreakingChangeType.TYPE_CHANGE

    def test_breaking_change_detection_no_changes(self) -> None:
        schema = {"fields": {"co2": "float", "scope": "int"}}
        changes = detect_breaking_changes(schema, schema)
        assert len(changes) == 0

    def test_breaking_change_detection_added_field(self) -> None:
        old = {"fields": {"co2": "float"}}
        new = {"fields": {"co2": "float", "scope": "int"}}
        changes = detect_breaking_changes(old, new)
        assert len(changes) == 0  # additions are not breaking


class TestCompatibilityMatrix:
    """Tests for the version compatibility matrix."""

    def test_compatibility_matrix_check(
        self, compat_matrix: CompatibilityMatrix
    ) -> None:
        compat_matrix.set_compatible("agent-a@1.0", "agent-b@2.0", True)
        assert compat_matrix.is_compatible("agent-a@1.0", "agent-b@2.0") is True
        assert compat_matrix.is_compatible("agent-b@2.0", "agent-a@1.0") is True

    def test_compatibility_matrix_unknown_pair(
        self, compat_matrix: CompatibilityMatrix
    ) -> None:
        assert compat_matrix.is_compatible("x", "y") is False


class TestCanaryController:
    """Tests for canary deployment progression."""

    @pytest.mark.asyncio
    async def test_canary_controller_start(self) -> None:
        canary = CanaryController("agent-a", steps=[5, 25, 50, 100])
        await canary.start()
        assert canary.status == "running"
        assert canary.traffic_pct == 5

    @pytest.mark.asyncio
    async def test_canary_controller_promote_step(self) -> None:
        canary = CanaryController("agent-a", steps=[5, 25, 50, 100])
        await canary.start()
        success = await canary.promote_step()
        assert success is True
        assert canary.traffic_pct == 25

    @pytest.mark.asyncio
    async def test_canary_controller_complete(self) -> None:
        canary = CanaryController("agent-a", steps=[50, 100])
        await canary.start()
        await canary.promote_step()  # -> 100
        assert canary.status == "completed"
        assert canary.traffic_pct == 100.0

    @pytest.mark.asyncio
    async def test_canary_controller_auto_rollback(self) -> None:
        async def bad_metrics() -> Dict[str, float]:
            return {"error_rate": 0.10, "latency_p99_ms": 200}

        canary = CanaryController(
            "agent-a",
            steps=[5, 25],
            metric_fn=bad_metrics,
            error_threshold=0.05,
        )
        await canary.start()
        success = await canary.promote_step()
        assert success is False
        assert canary.status == "rolled_back"
        assert canary.traffic_pct == 0.0


class TestRollbackController:
    """Tests for automated rollback triggering."""

    def test_rollback_controller_trigger_error_rate(self) -> None:
        controller = RollbackController(
            error_rate_threshold=0.05, latency_threshold_ms=500
        )
        should, reason = controller.should_rollback(
            error_rate=0.10, latency_ms=100
        )
        assert should is True
        assert "error_rate" in reason

    def test_rollback_controller_trigger_latency(self) -> None:
        controller = RollbackController(
            error_rate_threshold=0.05, latency_threshold_ms=500
        )
        should, reason = controller.should_rollback(
            error_rate=0.01, latency_ms=1000
        )
        assert should is True
        assert "latency" in reason

    def test_rollback_controller_healthy(self) -> None:
        controller = RollbackController()
        should, reason = controller.should_rollback(
            error_rate=0.01, latency_ms=100
        )
        assert should is False
        assert reason == "healthy"

    def test_rollback_controller_cooldown(self) -> None:
        controller = RollbackController(cooldown_seconds=60.0)
        controller.record_rollback()
        should, reason = controller.should_rollback(
            error_rate=0.50, latency_ms=5000
        )
        assert should is False
        assert "cooldown" in reason


class TestVersionMigration:
    """Tests for version migration steps."""

    @pytest.mark.asyncio
    async def test_version_migration_up(self) -> None:
        executed: List[str] = []

        async def up_v1():
            executed.append("v1.0.0_up")

        async def up_v2():
            executed.append("v2.0.0_up")

        migrations = [
            MigrationStep(version="1.0.0", up=up_v1),
            MigrationStep(version="2.0.0", up=up_v2),
        ]
        for m in migrations:
            await m.up()

        assert executed == ["v1.0.0_up", "v2.0.0_up"]

    @pytest.mark.asyncio
    async def test_version_migration_down(self) -> None:
        executed: List[str] = []

        async def down_v2():
            executed.append("v2.0.0_down")

        migration = MigrationStep(
            version="2.0.0",
            up=AsyncMock(),
            down=down_v2,
        )
        if migration.down:
            await migration.down()

        assert executed == ["v2.0.0_down"]
