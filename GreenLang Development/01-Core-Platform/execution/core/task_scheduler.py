# -*- coding: utf-8 -*-
"""
TaskScheduler - Task scheduling and load balancing for agent orchestration.

This module implements task scheduling with priority queuing, load balancing,
timeout management, and task lifecycle tracking for distributed agent execution.

Example:
    >>> scheduler = TaskScheduler(config=TaskSchedulerConfig())
    >>> task_id = await scheduler.schedule(Task(
    ...     task_type="thermal_calculation",
    ...     payload={"temp": 450},
    ...     priority=TaskPriority.HIGH
    ... ))
    >>> result = await scheduler.wait_for_completion(task_id)

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from greenlang.orchestrator.quotas.manager import QuotaManager

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

    @property
    def weight(self) -> int:
        """Get numeric weight for priority comparison."""
        weights = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4,
        }
        return weights[self]


class TaskState(str, Enum):
    """Task execution states."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategies for task distribution."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY_WEIGHTED = "priority_weighted"
    CAPABILITY_MATCH = "capability_match"
    RANDOM = "random"


@dataclass
class Task:
    """
    Represents a schedulable task for agent execution.

    Attributes:
        task_id: Unique task identifier
        task_type: Type/category of task for routing
        payload: Task data/parameters
        priority: Task priority level
        timeout_seconds: Maximum execution time
        retry_count: Number of retries on failure
        retry_delay_seconds: Delay between retries
        dependencies: List of task IDs this task depends on
        assigned_agent: ID of agent assigned to execute
        created_at: Task creation timestamp
        started_at: Execution start timestamp
        completed_at: Completion timestamp
        state: Current task state
        result: Task result (on completion)
        error: Error message (on failure)
        metadata: Additional task metadata
        namespace: Namespace for quota management (FR-024)
    """

    task_type: str
    payload: Dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: float = 60.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    state: TaskState = TaskState.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: str = field(default="default")  # FR-024: Namespace for quota management
    _attempts: int = field(default=0, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "state": self.state.value,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create Task from dictionary."""
        task = cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            task_type=data["task_type"],
            payload=data.get("payload", {}),
            priority=TaskPriority(data.get("priority", "normal")),
            timeout_seconds=data.get("timeout_seconds", 60.0),
            retry_count=data.get("retry_count", 3),
            dependencies=data.get("dependencies", []),
            assigned_agent=data.get("assigned_agent"),
            state=TaskState(data.get("state", "pending")),
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace", "default"),
        )
        task.created_at = data.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        )
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.result = data.get("result")
        task.error = data.get("error")
        return task

    def mark_started(self, agent_id: Optional[str] = None) -> None:
        """Mark task as started."""
        self.state = TaskState.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        if agent_id:
            self.assigned_agent = agent_id
        self._attempts += 1

    def mark_completed(self, result: Any) -> None:
        """Mark task as completed with result."""
        self.state = TaskState.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error."""
        self.state = TaskState.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.error = error

    def mark_timeout(self) -> None:
        """Mark task as timed out."""
        self.state = TaskState.TIMEOUT
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.error = f"Task timed out after {self.timeout_seconds} seconds"

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self._attempts < self.retry_count

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        content = f"{self.task_id}{self.task_type}{self.payload}{self.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AgentCapacity:
    """
    Represents an agent's capacity and capabilities for load balancing.

    Attributes:
        agent_id: Agent identifier
        capabilities: Set of task types the agent can handle
        max_concurrent_tasks: Maximum concurrent tasks
        current_load: Current number of running tasks
        available: Whether agent is available
        last_heartbeat: Last heartbeat timestamp
        performance_score: Performance score (0-100)
        metadata: Additional agent metadata
    """

    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 10
    current_load: int = 0
    available: bool = True
    last_heartbeat: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    performance_score: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_accept_task(self, task: Task) -> bool:
        """Check if agent can accept the given task."""
        if not self.available:
            return False
        if self.current_load >= self.max_concurrent_tasks:
            return False
        if self.capabilities and task.task_type not in self.capabilities:
            return False
        return True

    def is_healthy(self, heartbeat_timeout_seconds: float = 30.0) -> bool:
        """Check if agent is healthy based on heartbeat."""
        last = datetime.fromisoformat(self.last_heartbeat.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return age < heartbeat_timeout_seconds

    @property
    def load_ratio(self) -> float:
        """Get load ratio (0.0 to 1.0)."""
        if self.max_concurrent_tasks == 0:
            return 1.0
        return self.current_load / self.max_concurrent_tasks


@dataclass
class TaskSchedulerConfig:
    """
    Configuration for TaskScheduler.

    Attributes:
        max_queue_size: Maximum tasks in queue
        default_timeout_seconds: Default task timeout
        load_balance_strategy: Strategy for distributing tasks
        enable_dependency_tracking: Track task dependencies
        heartbeat_interval_seconds: Interval for agent heartbeats
        heartbeat_timeout_seconds: Timeout for agent heartbeats
        max_concurrent_tasks: Maximum concurrent tasks globally
        metrics_enabled: Enable metrics collection
        cleanup_interval_seconds: Interval for cleaning completed tasks
        task_retention_seconds: How long to keep completed tasks
    """

    max_queue_size: int = 10000
    default_timeout_seconds: float = 60.0
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED
    enable_dependency_tracking: bool = True
    heartbeat_interval_seconds: float = 10.0
    heartbeat_timeout_seconds: float = 30.0
    max_concurrent_tasks: int = 100
    metrics_enabled: bool = True
    cleanup_interval_seconds: float = 60.0
    task_retention_seconds: float = 3600.0


@dataclass
class TaskSchedulerMetrics:
    """Metrics for task scheduler monitoring."""

    tasks_scheduled: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timeout: int = 0
    tasks_cancelled: int = 0
    tasks_pending: int = 0
    tasks_running: int = 0
    avg_wait_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    agents_available: int = 0
    agents_total: int = 0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # FR-024: Quota metrics
    quota_checks_passed: int = 0
    quota_checks_failed: int = 0
    tasks_queued_by_quota: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "tasks_scheduled": self.tasks_scheduled,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_timeout": self.tasks_timeout,
            "tasks_cancelled": self.tasks_cancelled,
            "tasks_pending": self.tasks_pending,
            "tasks_running": self.tasks_running,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "agents_available": self.agents_available,
            "agents_total": self.agents_total,
            "last_updated": self.last_updated,
            "quota_checks_passed": self.quota_checks_passed,
            "quota_checks_failed": self.quota_checks_failed,
            "tasks_queued_by_quota": self.tasks_queued_by_quota,
        }


# Type alias for task executor functions
TaskExecutor = Callable[[Task], Coroutine[Any, Any, Any]]


class TaskScheduler:
    """
    Task scheduler with priority queuing and load balancing.

    Provides:
    - Priority-based task scheduling
    - Multiple load balancing strategies
    - Task dependency resolution
    - Timeout and retry handling
    - Agent capacity management
    - Comprehensive metrics
    - FR-024: Namespace concurrency quota integration

    Example:
        >>> config = TaskSchedulerConfig(
        ...     load_balance_strategy=LoadBalanceStrategy.LEAST_LOADED
        ... )
        >>> scheduler = TaskScheduler(config)
        >>>
        >>> # Register agents
        >>> scheduler.register_agent(AgentCapacity(
        ...     agent_id="agent-1",
        ...     capabilities={"thermal_calculation", "energy_balance"},
        ...     max_concurrent_tasks=5
        ... ))
        >>>
        >>> # Schedule a task
        >>> task_id = await scheduler.schedule(Task(
        ...     task_type="thermal_calculation",
        ...     payload={"temperature": 450},
        ...     priority=TaskPriority.HIGH
        ... ))
        >>>
        >>> # Wait for completion
        >>> result = await scheduler.wait_for_completion(task_id, timeout=30.0)
    """

    def __init__(
        self,
        config: Optional[TaskSchedulerConfig] = None,
        quota_manager: Optional["QuotaManager"] = None,
    ) -> None:
        """
        Initialize TaskScheduler.

        Args:
            config: Configuration options
            quota_manager: Optional QuotaManager for namespace concurrency quotas (FR-024)
        """
        self.config = config or TaskSchedulerConfig()
        self._tasks: Dict[str, Task] = {}
        self._pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._agents: Dict[str, AgentCapacity] = {}
        self._executors: Dict[str, TaskExecutor] = {}
        self._completion_events: Dict[str, asyncio.Event] = {}
        self._metrics = TaskSchedulerMetrics()
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._round_robin_index = 0
        self._wait_times: List[float] = []
        self._execution_times: List[float] = []

        # FR-024: Namespace concurrency quota integration
        self._quota_manager: Optional["QuotaManager"] = quota_manager
        self._namespace_wait_times: Dict[str, List[float]] = {}

        logger.info(
            f"TaskScheduler initialized with strategy: {self.config.load_balance_strategy.value}"
        )
        if quota_manager:
            logger.info("TaskScheduler initialized with QuotaManager for namespace quotas")

    async def start(self) -> None:
        """Start the task scheduler."""
        if self._running:
            logger.warning("TaskScheduler is already running")
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TaskScheduler started")

    async def stop(self) -> None:
        """Stop the task scheduler gracefully."""
        logger.info("Stopping TaskScheduler...")
        self._running = False

        for task in [self._scheduler_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("TaskScheduler stopped")

    def register_agent(self, capacity: AgentCapacity) -> None:
        """
        Register an agent with the scheduler.

        Args:
            capacity: Agent capacity information
        """
        self._agents[capacity.agent_id] = capacity
        self._metrics.agents_total = len(self._agents)
        self._update_available_agents()
        logger.info(f"Registered agent: {capacity.agent_id}")

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if agent was removed
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._metrics.agents_total = len(self._agents)
            self._update_available_agents()
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def register_executor(self, task_type: str, executor: TaskExecutor) -> None:
        """
        Register an executor function for a task type.

        Args:
            task_type: Type of task this executor handles
            executor: Async function to execute tasks
        """
        self._executors[task_type] = executor
        logger.debug(f"Registered executor for task type: {task_type}")

    def set_quota_manager(self, quota_manager: "QuotaManager") -> None:
        """
        Set or update the quota manager.

        Args:
            quota_manager: QuotaManager instance for namespace quotas
        """
        self._quota_manager = quota_manager
        logger.info("QuotaManager set for TaskScheduler")

    async def schedule(self, task: Task) -> str:
        """
        Schedule a task for execution.

        FR-024: If a QuotaManager is configured, checks namespace quotas before
        scheduling. Tasks may be queued if the namespace quota is exceeded.

        Args:
            task: Task to schedule

        Returns:
            Task ID
        """
        async with self._lock:
            # FR-024: Check namespace quota if QuotaManager is configured
            quota_allowed = True
            if self._quota_manager:
                quota_allowed = await self._check_namespace_quota(task)
                if not quota_allowed:
                    self._metrics.tasks_queued_by_quota += 1
                    logger.info(
                        f"Task {task.task_id} queued by quota manager for namespace {task.namespace}"
                    )

            self._tasks[task.task_id] = task
            self._completion_events[task.task_id] = asyncio.Event()

            # Check dependencies
            if self.config.enable_dependency_tracking and task.dependencies:
                unmet_deps = [
                    dep_id
                    for dep_id in task.dependencies
                    if dep_id not in self._tasks
                    or self._tasks[dep_id].state != TaskState.COMPLETED
                ]
                if unmet_deps:
                    task.state = TaskState.PENDING
                    logger.debug(
                        f"Task {task.task_id} waiting for dependencies: {unmet_deps}"
                    )
                else:
                    task.state = TaskState.SCHEDULED

            # Add to priority queue
            queue_item = (task.priority.weight, time.time(), task.task_id)
            await self._pending_queue.put(queue_item)

            self._metrics.tasks_scheduled += 1
            self._metrics.tasks_pending = self._pending_queue.qsize()

        logger.debug(f"Scheduled task: {task.task_id} ({task.task_type})")
        return task.task_id

    async def _check_namespace_quota(self, task: Task) -> bool:
        """
        Check namespace quota for a task.

        Args:
            task: Task to check quota for

        Returns:
            True if quota allows immediate execution, False if queued
        """
        if not self._quota_manager:
            return True

        # Convert TaskPriority to numeric priority (1-10)
        priority_map = {
            TaskPriority.CRITICAL: 10,
            TaskPriority.HIGH: 8,
            TaskPriority.NORMAL: 5,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 1,
        }
        numeric_priority = priority_map.get(task.priority, 5)

        # Try to acquire run slot
        acquired = await self._quota_manager.acquire_run_slot(
            namespace=task.namespace,
            run_id=task.task_id,
            priority=numeric_priority,
            wait_for_slot=True,
        )

        if acquired:
            self._metrics.quota_checks_passed += 1
        else:
            self._metrics.quota_checks_failed += 1

        return acquired

    async def schedule_batch(self, tasks: List[Task]) -> List[str]:
        """
        Schedule multiple tasks.

        Args:
            tasks: List of tasks to schedule

        Returns:
            List of task IDs
        """
        task_ids = []
        for task in tasks:
            task_id = await self.schedule(task)
            task_ids.append(task_id)
        return task_ids

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: Task to cancel

        Returns:
            True if task was cancelled
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            if task.state in [TaskState.PENDING, TaskState.SCHEDULED]:
                task.state = TaskState.CANCELLED
                self._metrics.tasks_cancelled += 1
                if task_id in self._completion_events:
                    self._completion_events[task_id].set()
                logger.info(f"Cancelled task: {task_id}")
                return True

        return False

    async def wait_for_completion(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Task]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Completed task or None if timeout
        """
        if task_id not in self._completion_events:
            return None

        try:
            await asyncio.wait_for(
                self._completion_events[task_id].wait(),
                timeout=timeout,
            )
            return self._tasks.get(task_id)
        except asyncio.TimeoutError:
            return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    async def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """Get task state by ID."""
        task = self._tasks.get(task_id)
        return task.state if task else None

    async def heartbeat(self, agent_id: str) -> bool:
        """
        Update agent heartbeat.

        Args:
            agent_id: Agent sending heartbeat

        Returns:
            True if agent is registered
        """
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = (
                datetime.now(timezone.utc).isoformat()
            )
            self._update_available_agents()
            return True
        return False

    async def update_agent_load(self, agent_id: str, current_load: int) -> bool:
        """
        Update agent's current load.

        Args:
            agent_id: Agent to update
            current_load: Current number of tasks

        Returns:
            True if agent exists
        """
        if agent_id in self._agents:
            self._agents[agent_id].current_load = current_load
            return True
        return False

    def _update_available_agents(self) -> None:
        """Update count of available agents."""
        self._metrics.agents_available = sum(
            1 for a in self._agents.values()
            if a.available and a.is_healthy(self.config.heartbeat_timeout_seconds)
        )

    def _select_agent(self, task: Task) -> Optional[str]:
        """
        Select an agent for task execution based on load balance strategy.

        Args:
            task: Task to assign

        Returns:
            Agent ID or None if no suitable agent
        """
        # Filter eligible agents
        eligible = [
            a for a in self._agents.values()
            if a.can_accept_task(task)
            and a.is_healthy(self.config.heartbeat_timeout_seconds)
        ]

        if not eligible:
            return None

        strategy = self.config.load_balance_strategy

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            self._round_robin_index = (self._round_robin_index + 1) % len(eligible)
            return eligible[self._round_robin_index].agent_id

        elif strategy == LoadBalanceStrategy.LEAST_LOADED:
            agent = min(eligible, key=lambda a: a.load_ratio)
            return agent.agent_id

        elif strategy == LoadBalanceStrategy.PRIORITY_WEIGHTED:
            # Higher performance score and lower load = higher chance
            agent = min(
                eligible,
                key=lambda a: a.load_ratio * (100 - a.performance_score),
            )
            return agent.agent_id

        elif strategy == LoadBalanceStrategy.CAPABILITY_MATCH:
            # Prefer agents with exact capability match
            exact_match = [
                a for a in eligible
                if task.task_type in a.capabilities
            ]
            candidates = exact_match if exact_match else eligible
            agent = min(candidates, key=lambda a: a.load_ratio)
            return agent.agent_id

        elif strategy == LoadBalanceStrategy.RANDOM:
            import random
            return random.choice(eligible).agent_id

        return eligible[0].agent_id if eligible else None

    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            try:
                # Get next task from queue
                try:
                    _, _, task_id = await asyncio.wait_for(
                        self._pending_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                self._metrics.tasks_pending = self._pending_queue.qsize()

                async with self._lock:
                    task = self._tasks.get(task_id)
                    if not task or task.state == TaskState.CANCELLED:
                        continue

                    # Check dependencies
                    if self.config.enable_dependency_tracking and task.dependencies:
                        unmet = [
                            d for d in task.dependencies
                            if d not in self._tasks
                            or self._tasks[d].state != TaskState.COMPLETED
                        ]
                        if unmet:
                            # Re-queue task
                            queue_item = (task.priority.weight, time.time(), task_id)
                            await self._pending_queue.put(queue_item)
                            continue

                    # Select agent
                    agent_id = self._select_agent(task)
                    if not agent_id:
                        # Re-queue task
                        queue_item = (task.priority.weight, time.time(), task_id)
                        await self._pending_queue.put(queue_item)
                        await asyncio.sleep(0.5)  # Wait before retry
                        continue

                    # Assign and execute
                    task.mark_started(agent_id)
                    self._agents[agent_id].current_load += 1
                    self._metrics.tasks_running += 1

                # Execute task (outside lock)
                asyncio.create_task(self._execute_task(task))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduling error: {e}", exc_info=True)

    async def _execute_task(self, task: Task) -> None:
        """
        Execute a task with timeout and retry handling.

        Args:
            task: Task to execute
        """
        start_time = time.perf_counter()

        try:
            # Get executor
            executor = self._executors.get(task.task_type)
            if not executor:
                raise ValueError(f"No executor for task type: {task.task_type}")

            # Execute with timeout
            result = await asyncio.wait_for(
                executor(task),
                timeout=task.timeout_seconds,
            )

            # Mark completed
            task.mark_completed(result)
            self._metrics.tasks_completed += 1

            # Track execution time
            exec_time = (time.perf_counter() - start_time) * 1000
            self._execution_times.append(exec_time)
            if len(self._execution_times) > 1000:
                self._execution_times = self._execution_times[-1000:]
            self._metrics.avg_execution_time_ms = (
                sum(self._execution_times) / len(self._execution_times)
            )

            logger.debug(f"Task {task.task_id} completed in {exec_time:.2f}ms")

        except asyncio.TimeoutError:
            task.mark_timeout()
            self._metrics.tasks_timeout += 1
            logger.warning(f"Task {task.task_id} timed out")

            # Retry if possible
            if task.can_retry():
                await self._retry_task(task)

        except Exception as e:
            task.mark_failed(str(e))
            self._metrics.tasks_failed += 1
            logger.error(f"Task {task.task_id} failed: {e}")

            # Retry if possible
            if task.can_retry():
                await self._retry_task(task)

        finally:
            # Update agent load
            if task.assigned_agent and task.assigned_agent in self._agents:
                self._agents[task.assigned_agent].current_load = max(
                    0, self._agents[task.assigned_agent].current_load - 1
                )

            self._metrics.tasks_running = max(0, self._metrics.tasks_running - 1)
            self._metrics.last_updated = datetime.now(timezone.utc).isoformat()

            # FR-024: Release namespace quota slot
            if self._quota_manager:
                await self._quota_manager.release_run_slot(task.namespace, task.task_id)

            # Signal completion
            if task.task_id in self._completion_events:
                self._completion_events[task.task_id].set()

    async def _retry_task(self, task: Task) -> None:
        """
        Retry a failed task.

        Args:
            task: Task to retry
        """
        task.state = TaskState.RETRYING
        task.assigned_agent = None

        await asyncio.sleep(task.retry_delay_seconds)

        # Re-schedule
        queue_item = (task.priority.weight, time.time(), task.task_id)
        await self._pending_queue.put(queue_item)
        self._metrics.tasks_pending = self._pending_queue.qsize()

        logger.info(f"Retrying task {task.task_id} (attempt {task._attempts + 1})")

    async def _cleanup_loop(self) -> None:
        """Cleanup loop for removing old completed tasks."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)

                now = datetime.now(timezone.utc)
                retention = timedelta(seconds=self.config.task_retention_seconds)

                async with self._lock:
                    to_remove = []
                    for task_id, task in self._tasks.items():
                        if task.state in [
                            TaskState.COMPLETED,
                            TaskState.FAILED,
                            TaskState.CANCELLED,
                            TaskState.TIMEOUT,
                        ]:
                            if task.completed_at:
                                completed = datetime.fromisoformat(
                                    task.completed_at.replace("Z", "+00:00")
                                )
                                if now - completed > retention:
                                    to_remove.append(task_id)

                    for task_id in to_remove:
                        del self._tasks[task_id]
                        if task_id in self._completion_events:
                            del self._completion_events[task_id]

                    if to_remove:
                        logger.debug(f"Cleaned up {len(to_remove)} completed tasks")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_metrics(self) -> TaskSchedulerMetrics:
        """Get current scheduler metrics."""
        return self._metrics

    def get_agents(self) -> Dict[str, AgentCapacity]:
        """Get all registered agents."""
        return self._agents.copy()

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [
            t for t in self._tasks.values()
            if t.state in [TaskState.PENDING, TaskState.SCHEDULED]
        ]

    def get_running_tasks(self) -> List[Task]:
        """Get all running tasks."""
        return [t for t in self._tasks.values() if t.state == TaskState.RUNNING]

    # =========================================================================
    # FR-024: QUOTA METRICS METHODS
    # =========================================================================

    def get_namespace_metrics(self, namespace: str) -> Dict[str, Any]:
        """
        Get metrics for a specific namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Dictionary with namespace-specific metrics
        """
        namespace_tasks = [t for t in self._tasks.values() if t.namespace == namespace]
        running = len([t for t in namespace_tasks if t.state == TaskState.RUNNING])
        pending = len([t for t in namespace_tasks if t.state in [TaskState.PENDING, TaskState.SCHEDULED]])
        completed = len([t for t in namespace_tasks if t.state == TaskState.COMPLETED])
        failed = len([t for t in namespace_tasks if t.state == TaskState.FAILED])

        # Get quota metrics if quota manager is available
        quota_metrics = {}
        if self._quota_manager:
            try:
                qm = self._quota_manager.get_metrics(namespace)
                quota_metrics = {
                    "quota_usage_percent": qm.quota_usage_percent,
                    "queue_depth": qm.queue_depth,
                    "queue_wait_time_seconds": qm.queue_wait_time_seconds,
                    "runs_utilization_percent": qm.runs_utilization_percent,
                    "steps_utilization_percent": qm.steps_utilization_percent,
                }
            except Exception as e:
                logger.warning(f"Failed to get quota metrics for namespace {namespace}: {e}")

        return {
            "namespace": namespace,
            "tasks_running": running,
            "tasks_pending": pending,
            "tasks_completed": completed,
            "tasks_failed": failed,
            "total_tasks": len(namespace_tasks),
            **quota_metrics,
        }

    def get_all_namespace_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all namespaces.

        Returns:
            Dictionary of namespace -> metrics
        """
        namespaces = set(t.namespace for t in self._tasks.values())
        return {ns: self.get_namespace_metrics(ns) for ns in namespaces}

    def get_quota_utilization(self) -> Dict[str, float]:
        """
        Get quota utilization percentages for all namespaces.

        Returns:
            Dictionary of namespace -> utilization percentage
        """
        if not self._quota_manager:
            return {}

        result = {}
        try:
            for metrics in self._quota_manager.get_all_metrics():
                result[metrics.namespace] = metrics.quota_usage_percent
        except Exception as e:
            logger.warning(f"Failed to get quota utilization: {e}")

        return result


# Factory function
def create_task_scheduler(
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED,
    max_concurrent: int = 100,
    default_timeout: float = 60.0,
) -> TaskScheduler:
    """
    Create a task scheduler with common configurations.

    Args:
        strategy: Load balancing strategy
        max_concurrent: Maximum concurrent tasks
        default_timeout: Default task timeout

    Returns:
        Configured TaskScheduler instance
    """
    config = TaskSchedulerConfig(
        load_balance_strategy=strategy,
        max_concurrent_tasks=max_concurrent,
        default_timeout_seconds=default_timeout,
    )
    return TaskScheduler(config)


__all__ = [
    "AgentCapacity",
    "LoadBalanceStrategy",
    "Task",
    "TaskExecutor",
    "TaskPriority",
    "TaskScheduler",
    "TaskSchedulerConfig",
    "TaskSchedulerMetrics",
    "TaskState",
    "create_task_scheduler",
]
