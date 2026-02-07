"""
Agent Factory Sandbox Module - INFRA-010

Provides process-level isolation for agent execution in the GreenLang
Climate OS platform. Each agent invocation runs in a clean subprocess
with filtered environment variables, resource limits, timeout enforcement,
and full audit trail.

Public API:
    - SandboxExecutor: Process-level isolation engine.
    - SandboxResult: Outcome of a sandboxed execution.
    - ResourceLimits: Frozen resource limit configuration.
    - ResourceLimitEnforcer: Validates and applies resource limits.
    - SandboxTimeoutGuard: Subprocess timeout with graceful termination.
    - SandboxAudit: Audit logging service for sandbox executions.
    - AuditEntry: Immutable record of a sandbox execution.

Example:
    >>> from greenlang.infrastructure.agent_factory.sandbox import (
    ...     SandboxExecutor, ResourceLimits, SandboxAudit,
    ... )
    >>> executor = SandboxExecutor()
    >>> limits = ResourceLimits.for_environment("prod")
    >>> result = await executor.execute("greenlang.agents.intake", data, limits)
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.sandbox.audit import (
    AuditEntry,
    SandboxAudit,
    SandboxAuditRecord,
)
from greenlang.infrastructure.agent_factory.sandbox.executor import (
    ResourceUsage,
    SandboxExecutor,
    SandboxResult,
)
from greenlang.infrastructure.agent_factory.sandbox.resource_limits import (
    ResourceLimitEnforcer,
    ResourceLimits,
)
from greenlang.infrastructure.agent_factory.sandbox.timeout_guard import (
    SandboxTimeoutEvent,
    SandboxTimeoutGuard,
)

__all__ = [
    # Executor
    "SandboxExecutor",
    "SandboxResult",
    "ResourceUsage",
    # Resource Limits
    "ResourceLimits",
    "ResourceLimitEnforcer",
    # Timeout
    "SandboxTimeoutGuard",
    "SandboxTimeoutEvent",
    # Audit
    "SandboxAudit",
    "SandboxAuditRecord",
    "AuditEntry",
]
