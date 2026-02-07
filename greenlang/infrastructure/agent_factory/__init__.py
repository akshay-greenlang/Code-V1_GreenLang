# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory v1.0 - Phase 1: Core Infrastructure.

This package provides the Agent Lifecycle Manager and Distributed Task Queue
that form the foundation of the GreenLang Agent Factory.  It is designed for
production deployment on Kubernetes with Redis-backed distributed locking
and task streaming.

Sub-packages:
    lifecycle - Agent state machine, health checks, warmup, shutdown, manager.
    queue     - Distributed task queue, priority scheduling, DLQ, worker pool.

Quick start:
    >>> from greenlang.infrastructure.agent_factory import (
    ...     AgentLifecycleManager,
    ...     LifecycleManagerConfig,
    ...     DistributedTaskQueue,
    ...     TaskQueueConfig,
    ...     WorkerPool,
    ... )
    >>> config = LifecycleManagerConfig(redis_url="redis://localhost:6379")
    >>> manager = await AgentLifecycleManager.create(config)
    >>> await manager.register_agent("carbon-agent")
    >>> await manager.start_agent("carbon-agent")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

# -- Lifecycle ---------------------------------------------------------------
from greenlang.infrastructure.agent_factory.lifecycle import (
    # States
    AgentState,
    AgentStateMachine,
    AgentStateTransition,
    InvalidTransitionError,
    TransitionCallback,
    VALID_TRANSITIONS,
    # Health
    AgentHealthSummary,
    HealthCheck,
    HealthCheckFn,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthStatus,
    ProbeType,
    # Warmup
    CachePrimingWarmup,
    ConnectionPoolWarmup,
    ModelLoadWarmup,
    WarmupManager,
    WarmupReport,
    WarmupStepResult,
    WarmupStrategy,
    # Shutdown
    AgentShutdownResult,
    AgentShutdownSpec,
    GracefulShutdownCoordinator,
    ShutdownReport,
    # Manager
    AgentLifecycleManager,
    LifecycleEventCallback,
    LifecycleManagerConfig,
)

# -- Queue -------------------------------------------------------------------
from greenlang.infrastructure.agent_factory.queue import (
    # Task queue
    DistributedTaskQueue,
    TaskItem,
    TaskPriority,
    TaskQueueConfig,
    TaskStatus,
    # Priority
    PriorityScheduler,
    PrioritySchedulerConfig,
    # Dead letter
    DeadLetterQueue,
    DeadLetterQueueConfig,
    DLQMetrics,
    # Workers
    TaskHandler,
    WorkerInfo,
    WorkerPool,
    WorkerPoolConfig,
    WorkerState,
    # Scheduler
    CronExpression,
    EventTrigger,
    ScheduleEntry,
    ScheduleStatus,
    TaskScheduler,
)

__all__ = [
    # Lifecycle - states
    "AgentState",
    "AgentStateMachine",
    "AgentStateTransition",
    "InvalidTransitionError",
    "TransitionCallback",
    "VALID_TRANSITIONS",
    # Lifecycle - health
    "AgentHealthSummary",
    "HealthCheck",
    "HealthCheckFn",
    "HealthCheckRegistry",
    "HealthCheckResult",
    "HealthStatus",
    "ProbeType",
    # Lifecycle - warmup
    "CachePrimingWarmup",
    "ConnectionPoolWarmup",
    "ModelLoadWarmup",
    "WarmupManager",
    "WarmupReport",
    "WarmupStepResult",
    "WarmupStrategy",
    # Lifecycle - shutdown
    "AgentShutdownResult",
    "AgentShutdownSpec",
    "GracefulShutdownCoordinator",
    "ShutdownReport",
    # Lifecycle - manager
    "AgentLifecycleManager",
    "LifecycleEventCallback",
    "LifecycleManagerConfig",
    # Queue - task queue
    "DistributedTaskQueue",
    "TaskItem",
    "TaskPriority",
    "TaskQueueConfig",
    "TaskStatus",
    # Queue - priority
    "PriorityScheduler",
    "PrioritySchedulerConfig",
    # Queue - dead letter
    "DeadLetterQueue",
    "DeadLetterQueueConfig",
    "DLQMetrics",
    # Queue - workers
    "TaskHandler",
    "WorkerInfo",
    "WorkerPool",
    "WorkerPoolConfig",
    "WorkerState",
    # Queue - scheduler
    "CronExpression",
    "EventTrigger",
    "ScheduleEntry",
    "ScheduleStatus",
    "TaskScheduler",
]
