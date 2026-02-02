# -*- coding: utf-8 -*-
"""
GraphQL Subscriptions for Real-Time Updates
WebSocket-based subscriptions with connection management
"""

from __future__ import annotations
import strawberry
from typing import Optional, AsyncGenerator, Dict, Any, Set
from datetime import datetime
import asyncio
import logging
from collections import defaultdict

from greenlang.utilities.determinism import DeterministicClock
from greenlang.api.graphql.types import (
    ExecutionUpdate,
    ExecutionStatusUpdate,
    ExecutionProgress,
    AgentUpdate,
    WorkflowUpdate,
    SystemMetrics,
    SubscriptionEvent,
    ExecutionStatus,
)
from greenlang.api.graphql.context import GraphQLContext

logger = logging.getLogger(__name__)


# ==============================================================================
# Subscription Event Manager
# ==============================================================================

class SubscriptionManager:
    """
    Manages subscription events and broadcasting

    Provides:
    - Event queue management
    - Subscriber tracking
    - Event broadcasting
    - Connection management
    """

    def __init__(self):
        self.execution_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.agent_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.workflow_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.system_subscribers: Set[asyncio.Queue] = set()

        # Heartbeat tracking
        self.heartbeat_interval = 30  # seconds
        self.connection_timeouts: Dict[asyncio.Queue, float] = {}

        logger.info("SubscriptionManager initialized")

    async def subscribe_execution(
        self,
        execution_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Subscribe to execution updates

        Args:
            execution_id: Specific execution ID (None for all)

        Yields:
            Execution update events
        """
        queue = asyncio.Queue()

        # Register subscriber
        key = execution_id or "*"
        self.execution_subscribers[key].add(queue)

        logger.info(f"New execution subscription: {key}")

        try:
            while True:
                # Wait for event with timeout (for heartbeat)
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=self.heartbeat_interval
                    )
                    yield event
                except asyncio.TimeoutError:
                    # Send heartbeat (None means no event)
                    continue

        finally:
            # Cleanup on disconnect
            self.execution_subscribers[key].discard(queue)
            logger.info(f"Execution subscription ended: {key}")

    async def subscribe_agent(
        self,
        agent_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Subscribe to agent updates

        Args:
            agent_id: Specific agent ID (None for all)

        Yields:
            Agent update events
        """
        queue = asyncio.Queue()

        key = agent_id or "*"
        self.agent_subscribers[key].add(queue)

        logger.info(f"New agent subscription: {key}")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=self.heartbeat_interval
                    )
                    yield event
                except asyncio.TimeoutError:
                    continue

        finally:
            self.agent_subscribers[key].discard(queue)
            logger.info(f"Agent subscription ended: {key}")

    async def subscribe_workflow(
        self,
        workflow_id: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Subscribe to workflow updates

        Args:
            workflow_id: Specific workflow ID (None for all)

        Yields:
            Workflow update events
        """
        queue = asyncio.Queue()

        key = workflow_id or "*"
        self.workflow_subscribers[key].add(queue)

        logger.info(f"New workflow subscription: {key}")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=self.heartbeat_interval
                    )
                    yield event
                except asyncio.TimeoutError:
                    continue

        finally:
            self.workflow_subscribers[key].discard(queue)
            logger.info(f"Workflow subscription ended: {key}")

    async def subscribe_system_metrics(
        self,
        interval: int = 5,
    ) -> AsyncGenerator[SystemMetrics, None]:
        """
        Subscribe to system metrics

        Args:
            interval: Update interval in seconds

        Yields:
            System metrics
        """
        queue = asyncio.Queue()
        self.system_subscribers.add(queue)

        logger.info(f"New system metrics subscription (interval: {interval}s)")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=interval)
                    yield event
                except asyncio.TimeoutError:
                    # Generate metrics on timeout
                    metrics = await self._collect_system_metrics()
                    yield metrics

        finally:
            self.system_subscribers.discard(queue)
            logger.info("System metrics subscription ended")

    async def publish_execution_update(
        self,
        execution_id: str,
        event: SubscriptionEvent,
        execution: Any,
    ):
        """
        Publish execution update to subscribers

        Args:
            execution_id: Execution ID
            event: Event type
            execution: Execution object
        """
        update = ExecutionUpdate(
            event=event,
            execution=execution,
            timestamp=DeterministicClock.utcnow(),
        )

        # Broadcast to specific subscribers
        await self._broadcast(self.execution_subscribers[execution_id], update)

        # Broadcast to wildcard subscribers
        await self._broadcast(self.execution_subscribers["*"], update)

        logger.debug(f"Published execution update: {execution_id} ({event.value})")

    async def publish_execution_status(
        self,
        execution_id: str,
        old_status: Optional[ExecutionStatus],
        new_status: ExecutionStatus,
    ):
        """
        Publish execution status change

        Args:
            execution_id: Execution ID
            old_status: Previous status
            new_status: New status
        """
        update = ExecutionStatusUpdate(
            execution_id=strawberry.ID(execution_id),
            old_status=old_status,
            new_status=new_status,
            timestamp=DeterministicClock.utcnow(),
        )

        await self._broadcast(self.execution_subscribers[execution_id], update)
        await self._broadcast(self.execution_subscribers["*"], update)

        logger.debug(f"Published status change: {execution_id} {old_status} -> {new_status}")

    async def publish_execution_progress(
        self,
        execution_id: str,
        current_step: Optional[str],
        completed_steps: int,
        total_steps: int,
        estimated_time: Optional[float] = None,
    ):
        """
        Publish execution progress

        Args:
            execution_id: Execution ID
            current_step: Currently executing step
            completed_steps: Number of completed steps
            total_steps: Total number of steps
            estimated_time: Estimated time remaining in seconds
        """
        progress = ExecutionProgress(
            execution_id=strawberry.ID(execution_id),
            current_step=current_step,
            completed_steps=completed_steps,
            total_steps=total_steps,
            progress=completed_steps / total_steps if total_steps > 0 else 0.0,
            estimated_time_remaining=estimated_time,
            timestamp=DeterministicClock.utcnow(),
        )

        await self._broadcast(self.execution_subscribers[execution_id], progress)
        await self._broadcast(self.execution_subscribers["*"], progress)

    async def publish_agent_update(
        self,
        agent_id: str,
        event: SubscriptionEvent,
        agent: Any,
    ):
        """
        Publish agent update

        Args:
            agent_id: Agent ID
            event: Event type
            agent: Agent object
        """
        update = AgentUpdate(
            event=event,
            agent=agent,
            timestamp=DeterministicClock.utcnow(),
        )

        await self._broadcast(self.agent_subscribers[agent_id], update)
        await self._broadcast(self.agent_subscribers["*"], update)

        logger.debug(f"Published agent update: {agent_id} ({event.value})")

    async def publish_workflow_update(
        self,
        workflow_id: str,
        event: SubscriptionEvent,
        workflow: Any,
    ):
        """
        Publish workflow update

        Args:
            workflow_id: Workflow ID
            event: Event type
            workflow: Workflow object
        """
        update = WorkflowUpdate(
            event=event,
            workflow=workflow,
            timestamp=DeterministicClock.utcnow(),
        )

        await self._broadcast(self.workflow_subscribers[workflow_id], update)
        await self._broadcast(self.workflow_subscribers["*"], update)

        logger.debug(f"Published workflow update: {workflow_id} ({event.value})")

    async def publish_system_metrics(self, metrics: SystemMetrics):
        """
        Publish system metrics

        Args:
            metrics: System metrics
        """
        await self._broadcast(self.system_subscribers, metrics)

    async def _broadcast(self, subscribers: Set[asyncio.Queue], event: Any):
        """
        Broadcast event to subscribers

        Args:
            subscribers: Set of subscriber queues
            event: Event to broadcast
        """
        if not subscribers:
            return

        # Put event in all subscriber queues
        for queue in list(subscribers):
            try:
                await queue.put(event)
            except Exception as e:
                logger.error(f"Failed to broadcast to subscriber: {e}")
                # Remove dead subscriber
                subscribers.discard(queue)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system metrics

        Returns:
            SystemMetrics object
        """
        # Placeholder implementation
        import psutil

        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
        except Exception:
            cpu_usage = 0.0
            memory_usage = 0.0

        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_executions=len(self.execution_subscribers),
            requests_per_second=0.0,
            timestamp=DeterministicClock.utcnow(),
        )

    def get_subscriber_counts(self) -> Dict[str, int]:
        """Get subscriber counts by type"""
        return {
            "executions": sum(len(subs) for subs in self.execution_subscribers.values()),
            "agents": sum(len(subs) for subs in self.agent_subscribers.values()),
            "workflows": sum(len(subs) for subs in self.workflow_subscribers.values()),
            "system": len(self.system_subscribers),
        }


# Global subscription manager instance
subscription_manager = SubscriptionManager()


# ==============================================================================
# GraphQL Subscription Type
# ==============================================================================

@strawberry.type
class Subscription:
    """GraphQL Subscription root with real-time updates"""

    @strawberry.subscription
    async def execution_updated(
        self,
        info: strawberry.Info[GraphQLContext],
        execution_id: Optional[strawberry.ID] = None,
        workflow_id: Optional[strawberry.ID] = None,
        agent_id: Optional[strawberry.ID] = None,
        user_id: Optional[strawberry.ID] = None,
    ) -> AsyncGenerator[ExecutionUpdate, None]:
        """
        Subscribe to execution updates

        Args:
            execution_id: Filter by specific execution
            workflow_id: Filter by workflow
            agent_id: Filter by agent
            user_id: Filter by user

        Yields:
            ExecutionUpdate events
        """
        context = info.context

        # Permission check
        if not context.has_permission("execution", "read"):
            raise PermissionError("Access denied to execution updates")

        # Subscribe to updates
        async for update in subscription_manager.subscribe_execution(execution_id):
            # Apply filters
            if workflow_id and update.execution.workflow_id != workflow_id:
                continue
            if agent_id and update.execution.agent_id != agent_id:
                continue
            if user_id and update.execution.user_id != user_id:
                continue

            yield update

    @strawberry.subscription
    async def execution_status_changed(
        self,
        info: strawberry.Info[GraphQLContext],
        execution_id: Optional[strawberry.ID] = None,
    ) -> AsyncGenerator[ExecutionStatusUpdate, None]:
        """
        Subscribe to execution status changes

        Args:
            execution_id: Specific execution ID

        Yields:
            ExecutionStatusUpdate events
        """
        context = info.context

        # Permission check
        if not context.has_permission("execution", "read"):
            raise PermissionError("Access denied to execution status")

        async for update in subscription_manager.subscribe_execution(execution_id):
            # Only yield status updates
            if isinstance(update, ExecutionStatusUpdate):
                yield update

    @strawberry.subscription
    async def execution_progress(
        self,
        info: strawberry.Info[GraphQLContext],
        execution_id: strawberry.ID,
    ) -> AsyncGenerator[ExecutionProgress, None]:
        """
        Subscribe to execution progress

        Args:
            execution_id: Execution ID to track

        Yields:
            ExecutionProgress events
        """
        context = info.context

        # Permission check
        if not context.has_permission(f"execution:{execution_id}", "read"):
            raise PermissionError(f"Access denied to execution {execution_id}")

        async for update in subscription_manager.subscribe_execution(execution_id):
            # Only yield progress updates
            if isinstance(update, ExecutionProgress):
                yield update

    @strawberry.subscription
    async def agent_updated(
        self,
        info: strawberry.Info[GraphQLContext],
        agent_id: Optional[strawberry.ID] = None,
    ) -> AsyncGenerator[AgentUpdate, None]:
        """
        Subscribe to agent updates

        Args:
            agent_id: Specific agent ID (None for all)

        Yields:
            AgentUpdate events
        """
        context = info.context

        # Permission check
        if not context.has_permission("agent", "read"):
            raise PermissionError("Access denied to agent updates")

        async for update in subscription_manager.subscribe_agent(agent_id):
            yield update

    @strawberry.subscription
    async def workflow_updated(
        self,
        info: strawberry.Info[GraphQLContext],
        workflow_id: Optional[strawberry.ID] = None,
    ) -> AsyncGenerator[WorkflowUpdate, None]:
        """
        Subscribe to workflow updates

        Args:
            workflow_id: Specific workflow ID (None for all)

        Yields:
            WorkflowUpdate events
        """
        context = info.context

        # Permission check
        if not context.has_permission("workflow", "read"):
            raise PermissionError("Access denied to workflow updates")

        async for update in subscription_manager.subscribe_workflow(workflow_id):
            yield update

    @strawberry.subscription
    async def system_metrics(
        self,
        info: strawberry.Info[GraphQLContext],
        interval: int = 5000,
    ) -> AsyncGenerator[SystemMetrics, None]:
        """
        Subscribe to system metrics

        Args:
            interval: Update interval in milliseconds

        Yields:
            SystemMetrics every interval
        """
        context = info.context

        # Permission check
        if not context.has_permission("metrics", "read"):
            raise PermissionError("Access denied to system metrics")

        # Convert ms to seconds
        interval_seconds = interval / 1000.0

        async for metrics in subscription_manager.subscribe_system_metrics(
            int(interval_seconds)
        ):
            yield metrics


# ==============================================================================
# Subscription Helpers
# ==============================================================================

async def notify_execution_created(execution_id: str, execution: Any):
    """Notify subscribers of new execution"""
    await subscription_manager.publish_execution_update(
        execution_id,
        SubscriptionEvent.CREATED,
        execution,
    )


async def notify_execution_updated(execution_id: str, execution: Any):
    """Notify subscribers of execution update"""
    await subscription_manager.publish_execution_update(
        execution_id,
        SubscriptionEvent.UPDATED,
        execution,
    )


async def notify_execution_status_changed(
    execution_id: str,
    old_status: Optional[ExecutionStatus],
    new_status: ExecutionStatus,
):
    """Notify subscribers of status change"""
    await subscription_manager.publish_execution_status(
        execution_id,
        old_status,
        new_status,
    )


async def notify_execution_progress(
    execution_id: str,
    current_step: Optional[str],
    completed_steps: int,
    total_steps: int,
    estimated_time: Optional[float] = None,
):
    """Notify subscribers of progress update"""
    await subscription_manager.publish_execution_progress(
        execution_id,
        current_step,
        completed_steps,
        total_steps,
        estimated_time,
    )


async def notify_agent_created(agent_id: str, agent: Any):
    """Notify subscribers of new agent"""
    await subscription_manager.publish_agent_update(
        agent_id,
        SubscriptionEvent.CREATED,
        agent,
    )


async def notify_agent_updated(agent_id: str, agent: Any):
    """Notify subscribers of agent update"""
    await subscription_manager.publish_agent_update(
        agent_id,
        SubscriptionEvent.UPDATED,
        agent,
    )


async def notify_workflow_created(workflow_id: str, workflow: Any):
    """Notify subscribers of new workflow"""
    await subscription_manager.publish_workflow_update(
        workflow_id,
        SubscriptionEvent.CREATED,
        workflow,
    )


async def notify_workflow_updated(workflow_id: str, workflow: Any):
    """Notify subscribers of workflow update"""
    await subscription_manager.publish_workflow_update(
        workflow_id,
        SubscriptionEvent.UPDATED,
        workflow,
    )
