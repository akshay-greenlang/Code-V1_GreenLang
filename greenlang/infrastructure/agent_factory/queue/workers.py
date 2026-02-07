# -*- coding: utf-8 -*-
"""
Worker Pool - Configurable pool of async workers consuming from the task queue.

Each worker is a long-lived async task that reads from the
DistributedTaskQueue, executes the task through a registered handler,
and acknowledges or fails the result.  The pool supports start / stop /
pause / resume, per-agent concurrency limits, graceful drain mode, and
health monitoring with auto-restart of dead workers.

Example:
    >>> pool = WorkerPool(
    ...     task_queue=queue,
    ...     handler_registry={"carbon-agent": carbon_handler},
    ...     config=WorkerPoolConfig(worker_count=5),
    ... )
    >>> await pool.start()
    >>> await pool.drain_and_stop()

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from greenlang.infrastructure.agent_factory.queue.task_queue import (
    DistributedTaskQueue,
    TaskItem,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TaskHandler = Callable[[TaskItem], Coroutine[Any, Any, Any]]


# ---------------------------------------------------------------------------
# Enums and config
# ---------------------------------------------------------------------------

class WorkerState(str, Enum):
    """Runtime state of an individual worker."""

    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerPoolConfig:
    """Configuration for the WorkerPool.

    Attributes:
        worker_count: Number of workers to spawn.
        dequeue_poll_interval: Seconds between dequeue attempts when idle.
        per_agent_concurrency: Maximum concurrent tasks per agent_key
            (0 = unlimited).
        drain_timeout: Seconds to wait for in-flight tasks during drain.
        health_check_interval: Seconds between worker health checks.
        auto_restart: Restart workers that exit unexpectedly.
    """

    worker_count: int = 5
    dequeue_poll_interval: float = 0.5
    per_agent_concurrency: int = 0
    drain_timeout: float = 60.0
    health_check_interval: float = 10.0
    auto_restart: bool = True


# ---------------------------------------------------------------------------
# Worker info
# ---------------------------------------------------------------------------

@dataclass
class WorkerInfo:
    """Observable state for a single worker.

    Attributes:
        worker_id: Unique identifier.
        state: Current worker state.
        tasks_completed: Total tasks successfully processed.
        tasks_failed: Total tasks that errored.
        current_task_id: ID of the task being processed (if any).
        started_at: UTC ISO-8601 start timestamp.
        last_active: UTC ISO-8601 timestamp of last activity.
    """

    worker_id: str
    state: WorkerState = WorkerState.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task_id: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_active: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "worker_id": self.worker_id,
            "state": self.state.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "current_task_id": self.current_task_id,
            "started_at": self.started_at,
            "last_active": self.last_active,
        }


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Pool of async worker tasks consuming from a DistributedTaskQueue.

    Lifecycle:
        1. ``start()``  - spawns ``worker_count`` workers plus a monitor.
        2. Workers loop: dequeue -> execute handler -> ack/fail.
        3. ``pause()``  - workers enter PAUSED and stop dequeuing.
        4. ``resume()`` - workers resume dequeuing.
        5. ``drain_and_stop()`` - stop accepting new tasks, wait for
           in-flight work, then shut down.
    """

    def __init__(
        self,
        task_queue: DistributedTaskQueue,
        handler_registry: Dict[str, TaskHandler],
        config: Optional[WorkerPoolConfig] = None,
    ) -> None:
        """Initialize the worker pool.

        Args:
            task_queue: The queue to consume from.
            handler_registry: Mapping of agent_key -> async handler.
            config: Pool configuration.
        """
        self._queue = task_queue
        self._handlers = handler_registry
        self._config = config or WorkerPoolConfig()
        self._workers: Dict[str, WorkerInfo] = {}
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._paused: bool = False
        self._draining: bool = False
        # Per-agent concurrency tracking
        self._agent_inflight: Dict[str, int] = {}
        self._agent_lock = asyncio.Lock()
        logger.debug(
            "WorkerPool initialized (workers=%d)", self._config.worker_count
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn worker tasks and the health monitor."""
        if self._running:
            logger.warning("WorkerPool is already running")
            return

        self._running = True
        self._paused = False
        self._draining = False

        for i in range(self._config.worker_count):
            wid = f"worker-{i}"
            await self._spawn_worker(wid)

        self._monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info(
            "WorkerPool started with %d workers", self._config.worker_count
        )

    async def stop(self) -> None:
        """Immediately stop all workers (does NOT drain)."""
        self._running = False
        await self._cancel_all_tasks()
        logger.info("WorkerPool stopped (immediate)")

    async def drain_and_stop(self) -> None:
        """Drain mode: finish in-flight work then shut down.

        Workers stop dequeuing new tasks.  After ``drain_timeout``
        seconds (or when all tasks complete) the pool shuts down.
        """
        self._draining = True
        logger.info(
            "WorkerPool entering drain mode (timeout=%.1fs)",
            self._config.drain_timeout,
        )

        # Wait for in-flight tasks to complete
        start = time.perf_counter()
        while True:
            inflight = sum(
                1 for w in self._workers.values()
                if w.state == WorkerState.PROCESSING
            )
            if inflight == 0:
                break
            elapsed = time.perf_counter() - start
            if elapsed >= self._config.drain_timeout:
                logger.warning(
                    "Drain timeout reached with %d in-flight workers", inflight
                )
                break
            await asyncio.sleep(0.25)

        self._running = False
        await self._cancel_all_tasks()
        logger.info("WorkerPool drained and stopped")

    async def pause(self) -> None:
        """Pause all workers (they stop dequeuing but keep running)."""
        self._paused = True
        for winfo in self._workers.values():
            if winfo.state == WorkerState.IDLE:
                winfo.state = WorkerState.PAUSED
        logger.info("WorkerPool paused")

    async def resume(self) -> None:
        """Resume paused workers."""
        self._paused = False
        for winfo in self._workers.values():
            if winfo.state == WorkerState.PAUSED:
                winfo.state = WorkerState.IDLE
        logger.info("WorkerPool resumed")

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, agent_key: str, handler: TaskHandler) -> None:
        """Register or replace a task handler for an agent key.

        Args:
            agent_key: Agent identifier.
            handler: Async callable that processes a TaskItem.
        """
        self._handlers[agent_key] = handler
        logger.info("Handler registered for agent %s", agent_key)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_workers(self) -> List[Dict[str, Any]]:
        """Return status info for all workers."""
        return [w.to_dict() for w in self._workers.values()]

    def active_count(self) -> int:
        """Return the number of workers currently processing a task."""
        return sum(
            1 for w in self._workers.values()
            if w.state == WorkerState.PROCESSING
        )

    @property
    def is_running(self) -> bool:
        """Whether the pool is in a running state."""
        return self._running

    @property
    def is_draining(self) -> bool:
        """Whether the pool is currently draining."""
        return self._draining

    # ------------------------------------------------------------------
    # Internal: worker loop
    # ------------------------------------------------------------------

    async def _spawn_worker(self, worker_id: str) -> None:
        """Create and register a single worker async task."""
        info = WorkerInfo(worker_id=worker_id)
        self._workers[worker_id] = info
        task = asyncio.create_task(self._worker_loop(worker_id))
        self._tasks[worker_id] = task
        await self._queue.add_consumer(worker_id)

    async def _worker_loop(self, worker_id: str) -> None:
        """Main loop for a single worker."""
        info = self._workers[worker_id]
        logger.debug("Worker %s started", worker_id)

        while self._running:
            try:
                # Paused or draining: do not dequeue
                if self._paused or self._draining:
                    if info.state not in {
                        WorkerState.PROCESSING,
                        WorkerState.STOPPING,
                    }:
                        info.state = WorkerState.PAUSED
                    await asyncio.sleep(self._config.dequeue_poll_interval)
                    continue

                info.state = WorkerState.IDLE

                # Dequeue
                task_item = await self._queue.dequeue(worker_id)
                if task_item is None:
                    await asyncio.sleep(self._config.dequeue_poll_interval)
                    continue

                # Per-agent concurrency check
                if not await self._acquire_agent_slot(task_item.agent_key):
                    # Cannot acquire slot; re-fail so it re-enters queue
                    await self._queue.fail(
                        task_item.id,
                        "agent concurrency limit reached",
                    )
                    continue

                # Process
                info.state = WorkerState.PROCESSING
                info.current_task_id = task_item.id
                info.last_active = datetime.now(timezone.utc).isoformat()

                try:
                    handler = self._handlers.get(task_item.agent_key)
                    if handler is None:
                        raise ValueError(
                            f"No handler for agent_key={task_item.agent_key}"
                        )
                    await handler(task_item)
                    await self._queue.acknowledge(task_item.id)
                    info.tasks_completed += 1
                except Exception as exc:
                    await self._queue.fail(task_item.id, str(exc))
                    info.tasks_failed += 1
                    logger.error(
                        "Worker %s task %s failed: %s",
                        worker_id,
                        task_item.id,
                        exc,
                    )
                finally:
                    await self._release_agent_slot(task_item.agent_key)
                    info.current_task_id = None
                    info.state = WorkerState.IDLE

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Unexpected error in worker %s", worker_id)
                await asyncio.sleep(self._config.dequeue_poll_interval)

        info.state = WorkerState.STOPPED
        logger.debug("Worker %s stopped", worker_id)

    # ------------------------------------------------------------------
    # Per-agent concurrency
    # ------------------------------------------------------------------

    async def _acquire_agent_slot(self, agent_key: str) -> bool:
        """Try to acquire a per-agent concurrency slot."""
        if self._config.per_agent_concurrency <= 0:
            return True  # unlimited

        async with self._agent_lock:
            current = self._agent_inflight.get(agent_key, 0)
            if current >= self._config.per_agent_concurrency:
                return False
            self._agent_inflight[agent_key] = current + 1
            return True

    async def _release_agent_slot(self, agent_key: str) -> None:
        """Release a per-agent concurrency slot."""
        if self._config.per_agent_concurrency <= 0:
            return

        async with self._agent_lock:
            current = self._agent_inflight.get(agent_key, 0)
            self._agent_inflight[agent_key] = max(0, current - 1)

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _health_monitor_loop(self) -> None:
        """Periodically check worker health and restart dead workers."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await self._check_worker_health()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker health monitor error")

    async def _check_worker_health(self) -> None:
        """Restart any workers whose async task has terminated."""
        if not self._config.auto_restart:
            return

        for wid, task in list(self._tasks.items()):
            if task.done() and self._running and not self._draining:
                exc = task.exception() if not task.cancelled() else None
                logger.warning(
                    "Worker %s exited unexpectedly (exc=%s), restarting",
                    wid,
                    exc,
                )
                await self._spawn_worker(wid)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cancel_all_tasks(self) -> None:
        """Cancel worker tasks and the monitor."""
        all_tasks: List[asyncio.Task[None]] = list(self._tasks.values())
        if self._monitor_task and not self._monitor_task.done():
            all_tasks.append(self._monitor_task)

        for t in all_tasks:
            if not t.done():
                t.cancel()

        await asyncio.gather(*all_tasks, return_exceptions=True)

        for winfo in self._workers.values():
            winfo.state = WorkerState.STOPPED

        self._tasks.clear()
        self._monitor_task = None


__all__ = [
    "TaskHandler",
    "WorkerInfo",
    "WorkerPool",
    "WorkerPoolConfig",
    "WorkerState",
]
