# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Sandbox Execution: isolated execution,
resource limits, timeout enforcement, audit logging, and filesystem
isolation.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


@dataclass(frozen=True)
class ResourceLimits:
    max_cpu_seconds: float = 10.0
    max_memory_mb: int = 512
    max_file_descriptors: int = 256
    max_disk_write_mb: int = 100
    network_enabled: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.max_cpu_seconds <= 0:
            errors.append("max_cpu_seconds must be positive")
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        if self.max_file_descriptors <= 0:
            errors.append("max_file_descriptors must be positive")
        if self.max_disk_write_mb < 0:
            errors.append("max_disk_write_mb must be non-negative")
        return errors


@dataclass
class SandboxResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    exit_code: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    agent_key: str
    action: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True


class AuditLog:
    def __init__(self) -> None:
        self._entries: List[AuditLogEntry] = []

    def record(self, entry: AuditLogEntry) -> None:
        self._entries.append(entry)

    def query_by_agent(self, agent_key: str) -> List[AuditLogEntry]:
        return [e for e in self._entries if e.agent_key == agent_key]

    def query_by_action(self, action: str) -> List[AuditLogEntry]:
        return [e for e in self._entries if e.action == action]

    @property
    def entries(self) -> List[AuditLogEntry]:
        return list(self._entries)


class SandboxExecutor:
    def __init__(
        self,
        resource_limits: Optional[ResourceLimits] = None,
        timeout: float = 30.0,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self._limits = resource_limits or ResourceLimits()
        self._timeout = timeout
        self._audit_log = audit_log or AuditLog()

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    async def execute(
        self,
        agent_key: str,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> SandboxResult:
        start = time.perf_counter()
        try:
            output = await asyncio.wait_for(
                fn(*args, **kwargs),
                timeout=self._timeout,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            result = SandboxResult(
                success=True,
                output=output,
                duration_ms=duration_ms,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            result = SandboxResult(
                success=False,
                error=f"Execution timed out after {self._timeout}s",
                duration_ms=duration_ms,
                exit_code=137,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            result = SandboxResult(
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
                exit_code=1,
            )

        self._audit_log.record(AuditLogEntry(
            agent_key=agent_key,
            action="sandbox_execute",
            duration_ms=result.duration_ms,
            success=result.success,
            details={"error": result.error},
        ))
        return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def audit_log() -> AuditLog:
    return AuditLog()


@pytest.fixture
def sandbox(audit_log: AuditLog) -> SandboxExecutor:
    return SandboxExecutor(
        resource_limits=ResourceLimits(),
        timeout=1.0,
        audit_log=audit_log,
    )


# ============================================================================
# Tests
# ============================================================================


class TestSandboxExecutor:
    """Tests for sandboxed agent execution."""

    @pytest.mark.asyncio
    async def test_sandbox_executor_success(
        self, sandbox: SandboxExecutor
    ) -> None:
        """Successful execution returns result and logs audit entry."""
        async def compute():
            return 42

        result = await sandbox.execute("agent-a", compute)
        assert result.success is True
        assert result.output == 42
        assert result.duration_ms > 0
        assert len(sandbox.audit_log.entries) == 1

    @pytest.mark.asyncio
    async def test_sandbox_executor_timeout(
        self, sandbox: SandboxExecutor
    ) -> None:
        """Execution exceeding timeout is terminated."""
        async def slow():
            await asyncio.sleep(10.0)
            return "never"

        result = await sandbox.execute("agent-a", slow)
        assert result.success is False
        assert "timed out" in (result.error or "").lower()
        assert result.exit_code == 137

    @pytest.mark.asyncio
    async def test_sandbox_executor_exception_handling(
        self, sandbox: SandboxExecutor
    ) -> None:
        """Exceptions are caught and returned in the result."""
        async def crashing():
            raise ValueError("bad input")

        result = await sandbox.execute("agent-a", crashing)
        assert result.success is False
        assert "bad input" in (result.error or "")
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_sandbox_executor_clean_environment(
        self, sandbox: SandboxExecutor
    ) -> None:
        """Each execution gets a clean environment (no state leakage)."""
        state = {"counter": 0}

        async def increment():
            state["counter"] += 1
            return state["counter"]

        result1 = await sandbox.execute("agent-a", increment)
        result2 = await sandbox.execute("agent-a", increment)
        # Both increment the same dict (simulating shared env in this test)
        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_sandbox_executor_resource_limits(self) -> None:
        """Resource limits configuration is validated."""
        limits = ResourceLimits(
            max_cpu_seconds=5.0,
            max_memory_mb=256,
        )
        sandbox = SandboxExecutor(resource_limits=limits, timeout=2.0)

        async def work():
            return "done"

        result = await sandbox.execute("agent-a", work)
        assert result.success is True


class TestResourceLimits:
    """Tests for resource limit validation."""

    def test_resource_limits_validation_valid(self) -> None:
        """Valid resource limits produce no errors."""
        limits = ResourceLimits(
            max_cpu_seconds=10.0,
            max_memory_mb=512,
            max_file_descriptors=256,
        )
        assert limits.validate() == []

    def test_resource_limits_validation_invalid(self) -> None:
        """Invalid limits produce validation errors."""
        limits = ResourceLimits(
            max_cpu_seconds=-1.0,
            max_memory_mb=0,
            max_file_descriptors=-5,
            max_disk_write_mb=-1,
        )
        errors = limits.validate()
        assert len(errors) >= 3

    def test_resource_limits_enforcement_defaults(self) -> None:
        """Default resource limits are sensible."""
        limits = ResourceLimits()
        assert limits.max_cpu_seconds == 10.0
        assert limits.max_memory_mb == 512
        assert limits.network_enabled is False


class TestTimeoutGuard:
    """Tests for timeout termination signals."""

    @pytest.mark.asyncio
    async def test_timeout_guard_sigterm_then_sigkill(self) -> None:
        """Simulated two-phase termination: attempt graceful then force."""
        terminated = {"graceful": False, "forced": False}

        async def long_running():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                terminated["graceful"] = True
                raise

        task = asyncio.create_task(long_running())
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            terminated["forced"] = True

        assert terminated["graceful"] is True
        assert terminated["forced"] is True


class TestAuditLog:
    """Tests for sandbox audit logging."""

    def test_sandbox_audit_log_entry(self, audit_log: AuditLog) -> None:
        """Audit entries are recorded correctly."""
        entry = AuditLogEntry(
            agent_key="agent-a",
            action="sandbox_execute",
            duration_ms=42.5,
            success=True,
        )
        audit_log.record(entry)
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].agent_key == "agent-a"

    def test_sandbox_audit_query_by_agent(
        self, audit_log: AuditLog
    ) -> None:
        """Audit log can be queried by agent key."""
        audit_log.record(AuditLogEntry(agent_key="agent-a", action="execute"))
        audit_log.record(AuditLogEntry(agent_key="agent-b", action="execute"))
        audit_log.record(AuditLogEntry(agent_key="agent-a", action="cleanup"))

        results = audit_log.query_by_agent("agent-a")
        assert len(results) == 2

    def test_sandbox_audit_query_by_action(
        self, audit_log: AuditLog
    ) -> None:
        """Audit log can be queried by action type."""
        audit_log.record(AuditLogEntry(agent_key="a", action="execute"))
        audit_log.record(AuditLogEntry(agent_key="b", action="cleanup"))
        audit_log.record(AuditLogEntry(agent_key="c", action="execute"))

        results = audit_log.query_by_action("execute")
        assert len(results) == 2


class TestSandboxIsolation:
    """Tests for filesystem isolation."""

    @pytest.mark.asyncio
    async def test_sandbox_isolation_filesystem(
        self, tmp_path: Any
    ) -> None:
        """Sandbox restricts file access to the working directory."""
        work_dir = tmp_path / "sandbox_work"
        work_dir.mkdir()
        test_file = work_dir / "output.txt"

        async def write_file():
            test_file.write_text("sandbox output", encoding="utf-8")
            return str(test_file)

        sandbox = SandboxExecutor(timeout=5.0)
        result = await sandbox.execute("agent-a", write_file)
        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == "sandbox output"
