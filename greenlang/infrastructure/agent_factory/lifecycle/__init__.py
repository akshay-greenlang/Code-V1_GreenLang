# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory - Lifecycle sub-package.

Exports the core lifecycle management components: state machine, health
checks, warmup strategies, graceful shutdown, and the lifecycle manager.
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.lifecycle.states import (
    AgentState,
    AgentStateMachine,
    AgentStateTransition,
    InvalidTransitionError,
    TransitionCallback,
    VALID_TRANSITIONS,
)
from greenlang.infrastructure.agent_factory.lifecycle.health import (
    AgentHealthSummary,
    HealthCheck,
    HealthCheckFn,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthStatus,
    ProbeType,
)
from greenlang.infrastructure.agent_factory.lifecycle.warmup import (
    CachePrimingWarmup,
    ConnectionPoolWarmup,
    ModelLoadWarmup,
    WarmupManager,
    WarmupReport,
    WarmupStepResult,
    WarmupStrategy,
)
from greenlang.infrastructure.agent_factory.lifecycle.shutdown import (
    AgentShutdownResult,
    AgentShutdownSpec,
    GracefulShutdownCoordinator,
    ShutdownReport,
)
from greenlang.infrastructure.agent_factory.lifecycle.manager import (
    AgentLifecycleManager,
    LifecycleEventCallback,
    LifecycleManagerConfig,
)

__all__ = [
    # States
    "AgentState",
    "AgentStateMachine",
    "AgentStateTransition",
    "InvalidTransitionError",
    "TransitionCallback",
    "VALID_TRANSITIONS",
    # Health
    "AgentHealthSummary",
    "HealthCheck",
    "HealthCheckFn",
    "HealthCheckRegistry",
    "HealthCheckResult",
    "HealthStatus",
    "ProbeType",
    # Warmup
    "CachePrimingWarmup",
    "ConnectionPoolWarmup",
    "ModelLoadWarmup",
    "WarmupManager",
    "WarmupReport",
    "WarmupStepResult",
    "WarmupStrategy",
    # Shutdown
    "AgentShutdownResult",
    "AgentShutdownSpec",
    "GracefulShutdownCoordinator",
    "ShutdownReport",
    # Manager
    "AgentLifecycleManager",
    "LifecycleEventCallback",
    "LifecycleManagerConfig",
]
