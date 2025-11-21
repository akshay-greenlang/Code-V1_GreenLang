# -*- coding: utf-8 -*-
"""
AgentCoordinator - Central coordinator for managing multiple agents.

This module implements a production-ready agent coordinator that manages
the lifecycle, communication, and coordination of 10,000+ concurrent agents
with high availability and fault tolerance.

Example:
    >>> coordinator = AgentCoordinator(message_bus, config)
    >>> await coordinator.initialize()
    >>>
    >>> # Register and start agents
    >>> await coordinator.register_agent(agent1)
    >>> await coordinator.start_agent("agent-001")
    >>>
    >>> # Coordinate agent workflow
    >>> result = await coordinator.coordinate_workflow(workflow_spec)
"""

from typing import Dict, List, Optional, Any, Set, Callable, AsyncIterator, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
import logging
from datetime import datetime, timezone
import uuid
from dataclasses import dataclass, field
import json

from prometheus_client import Counter, Gauge, Histogram
import networkx as nx

from .message_bus import MessageBus, Message, MessageType, Priority
# Import Redis Streams broker for production deployments (NEW)
from ..messaging.redis_streams_broker import RedisStreamsBroker
from ..messaging.message import Message as RedisMessage
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)

# Metrics
agent_count_gauge = Gauge('coordinator_agent_count', 'Number of registered agents', ['state'])
coordination_latency_histogram = Histogram('coordinator_latency_ms', 'Coordination latency', ['operation'])
workflow_counter = Counter('coordinator_workflows_total', 'Total workflows executed', ['status'])
agent_state_transitions = Counter('coordinator_state_transitions', 'Agent state transitions', ['from_state', 'to_state'])


class AgentState(str, Enum):
    """Agent lifecycle states."""
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"


class CoordinationPattern(str, Enum):
    """Agent coordination patterns."""
    SINGLE = "SINGLE"
    PIPELINE = "PIPELINE"
    PARALLEL = "PARALLEL"
    ORCHESTRATION = "ORCHESTRATION"
    CHOREOGRAPHY = "CHOREOGRAPHY"
    HIERARCHICAL = "HIERARCHICAL"


class AgentInfo(BaseModel):
    """Agent registration information."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    state: AgentState = Field(default=AgentState.CREATED)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: Optional[str] = Field(None)
    error_count: int = Field(default=0, ge=0)
    message_count: int = Field(default=0, ge=0)
    resource_usage: Dict[str, float] = Field(default_factory=dict)

    @validator('agent_id')
    def validate_agent_id(cls, v):
        """Validate agent ID format."""
        if not v.startswith("agent-"):
            raise ValueError(f"Agent ID must start with 'agent-': {v}")
        return v


class WorkflowSpec(BaseModel):
    """Workflow specification for agent coordination."""

    workflow_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = Field(..., description="Workflow name")
    pattern: CoordinationPattern = Field(..., description="Coordination pattern")
    agents: List[str] = Field(..., description="Participating agent IDs")
    tasks: List[Dict[str, Any]] = Field(..., description="Task definitions")
    timeout_ms: int = Field(default=30000, ge=0, description="Workflow timeout")
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecution(BaseModel):
    """Workflow execution state."""

    workflow_id: str = Field(...)
    spec: WorkflowSpec = Field(...)
    state: str = Field(default="PENDING")
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(None)
    current_task: Optional[str] = Field(None)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


@dataclass
class CoordinatorConfig:
    """Coordinator configuration."""
    max_agents: int = 10000
    heartbeat_interval_ms: int = 5000
    state_transition_timeout_ms: int = 10000
    max_retry_attempts: int = 3
    enable_auto_scaling: bool = True
    enable_health_checks: bool = True
    enable_metrics: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=dict)


class AgentCoordinator:
    """
    Central coordinator for managing multiple agents.

    Provides lifecycle management, resource allocation, workflow orchestration,
    and fault tolerance for 10,000+ concurrent agents.

    Infrastructure:
    - Supports both in-memory MessageBus (development) and Redis Streams (production)
    - Distributed coordination via Redis Streams with consumer groups
    - Pub/sub, request-response, and work queue patterns
    - Automatic failover and at-least-once delivery guarantees
    """

    def __init__(
        self,
        message_bus: Union[MessageBus, RedisStreamsBroker],
        config: Optional[CoordinatorConfig] = None,
        use_redis_streams: bool = False
    ):
        """
        Initialize agent coordinator.

        Args:
            message_bus: Message bus for agent communication
                        - MessageBus: In-memory bus for development/testing
                        - RedisStreamsBroker: Production-ready distributed messaging
            config: Coordinator configuration
            use_redis_streams: Flag indicating Redis Streams is in use (auto-detected)
        """
        self.message_bus = message_bus
        self.config = config or CoordinatorConfig()

        # Detect if using Redis Streams
        self.use_redis_streams = use_redis_streams or isinstance(message_bus, RedisStreamsBroker)

        if self.use_redis_streams:
            logger.info("AgentCoordinator initialized with Redis Streams broker (production mode)")
        else:
            logger.info("AgentCoordinator initialized with in-memory MessageBus (development mode)")

        # Agent registry
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_handlers: Dict[str, Any] = {}

        # Workflow management
        self.workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_graph = nx.DiGraph()

        # State management
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._locks: Dict[str, asyncio.Lock] = {}

        # Performance tracking
        self._coordination_count = 0
        self._error_count = 0

    async def initialize(self) -> None:
        """Initialize coordinator and start background tasks."""
        logger.info("Initializing AgentCoordinator")

        self._running = True

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._heartbeat_monitor())
        )
        self._tasks.append(
            asyncio.create_task(self._state_manager())
        )
        self._tasks.append(
            asyncio.create_task(self._message_handler())
        )

        if self.config.enable_health_checks:
            self._tasks.append(
                asyncio.create_task(self._health_checker())
            )

        logger.info(f"AgentCoordinator initialized with capacity for {self.config.max_agents} agents")

    async def shutdown(self) -> None:
        """Gracefully shutdown coordinator."""
        logger.info("Shutting down AgentCoordinator")
        self._running = False

        # Stop all agents
        for agent_id in list(self.agents.keys()):
            await self.stop_agent(agent_id)

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("AgentCoordinator shutdown complete")

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: Optional[List[str]] = None,
        handler: Optional[Any] = None
    ) -> bool:
        """
        Register new agent with coordinator.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            capabilities: Agent capabilities
            handler: Agent implementation object

        Returns:
            Success status
        """
        if len(self.agents) >= self.config.max_agents:
            logger.error(f"Maximum agent capacity ({self.config.max_agents}) reached")
            return False

        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered")
            return False

        try:
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities or [],
                state=AgentState.CREATED
            )

            self.agents[agent_id] = agent_info
            if handler:
                self.agent_handlers[agent_id] = handler

            self._locks[agent_id] = asyncio.Lock()

            # Update metrics
            agent_count_gauge.labels(state=AgentState.CREATED).inc()

            # Notify via message bus
            await self._send_lifecycle_event(agent_id, "REGISTERED")

            logger.info(f"Registered agent {agent_id} (type: {agent_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister agent from coordinator."""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False

        try:
            # Stop agent if running
            if self.agents[agent_id].state in [AgentState.READY, AgentState.RUNNING]:
                await self.stop_agent(agent_id)

            # Remove from registry
            agent_info = self.agents.pop(agent_id)
            self.agent_handlers.pop(agent_id, None)
            self._locks.pop(agent_id, None)

            # Update metrics
            agent_count_gauge.labels(state=agent_info.state).dec()

            # Notify
            await self._send_lifecycle_event(agent_id, "UNREGISTERED")

            logger.info(f"Unregistered agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def start_agent(self, agent_id: str) -> bool:
        """Start agent and transition to READY state."""
        return await self._transition_agent_state(
            agent_id,
            [AgentState.CREATED, AgentState.PAUSED],
            AgentState.INITIALIZING
        ) and await self._transition_agent_state(
            agent_id,
            [AgentState.INITIALIZING],
            AgentState.READY
        )

    async def stop_agent(self, agent_id: str) -> bool:
        """Stop agent and transition to TERMINATED state."""
        return await self._transition_agent_state(
            agent_id,
            [AgentState.READY, AgentState.RUNNING, AgentState.PAUSED, AgentState.ERROR],
            AgentState.STOPPING
        ) and await self._transition_agent_state(
            agent_id,
            [AgentState.STOPPING],
            AgentState.TERMINATED
        )

    async def pause_agent(self, agent_id: str) -> bool:
        """Pause agent execution."""
        return await self._transition_agent_state(
            agent_id,
            [AgentState.READY, AgentState.RUNNING],
            AgentState.PAUSED
        )

    async def resume_agent(self, agent_id: str) -> bool:
        """Resume paused agent."""
        return await self._transition_agent_state(
            agent_id,
            [AgentState.PAUSED],
            AgentState.READY
        )

    async def coordinate_workflow(self, spec: WorkflowSpec) -> WorkflowExecution:
        """
        Coordinate multi-agent workflow execution.

        Args:
            spec: Workflow specification

        Returns:
            Workflow execution result
        """
        start_time = datetime.now(timezone.utc)
        execution = WorkflowExecution(
            workflow_id=spec.workflow_id,
            spec=spec,
            state="RUNNING"
        )

        self.workflows[spec.workflow_id] = execution

        try:
            logger.info(f"Starting workflow {spec.workflow_id} with pattern {spec.pattern}")

            # Execute based on pattern
            if spec.pattern == CoordinationPattern.SINGLE:
                await self._execute_single(execution)
            elif spec.pattern == CoordinationPattern.PIPELINE:
                await self._execute_pipeline(execution)
            elif spec.pattern == CoordinationPattern.PARALLEL:
                await self._execute_parallel(execution)
            elif spec.pattern == CoordinationPattern.ORCHESTRATION:
                await self._execute_orchestration(execution)
            else:
                raise ValueError(f"Unsupported pattern: {spec.pattern}")

            execution.state = "COMPLETED"
            workflow_counter.labels(status="success").inc()

        except asyncio.TimeoutError:
            execution.state = "TIMEOUT"
            execution.errors.append("Workflow timeout exceeded")
            workflow_counter.labels(status="timeout").inc()

        except Exception as e:
            execution.state = "FAILED"
            execution.errors.append(str(e))
            workflow_counter.labels(status="failure").inc()
            logger.error(f"Workflow {spec.workflow_id} failed: {e}")

        finally:
            execution.completed_at = datetime.now(timezone.utc).isoformat()
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            execution.metrics["duration_ms"] = duration_ms
            coordination_latency_histogram.labels(operation="workflow").observe(duration_ms)

        return execution

    async def _execute_single(self, execution: WorkflowExecution) -> None:
        """Execute single agent task."""
        if not execution.spec.agents:
            raise ValueError("No agents specified")

        agent_id = execution.spec.agents[0]
        task = execution.spec.tasks[0] if execution.spec.tasks else {}

        result = await self._execute_agent_task(agent_id, task)
        execution.results.append(result)

    async def _execute_pipeline(self, execution: WorkflowExecution) -> None:
        """Execute pipeline of sequential tasks."""
        previous_result = None

        for i, (agent_id, task) in enumerate(zip(execution.spec.agents, execution.spec.tasks)):
            execution.current_task = f"stage_{i}"

            # Pass previous result as input
            if previous_result:
                task["input"] = previous_result.get("output")

            result = await self._execute_agent_task(agent_id, task)
            execution.results.append(result)
            previous_result = result

    async def _execute_parallel(self, execution: WorkflowExecution) -> None:
        """Execute tasks in parallel."""
        tasks = []

        for agent_id, task in zip(execution.spec.agents, execution.spec.tasks):
            tasks.append(self._execute_agent_task(agent_id, task))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                execution.errors.append(str(result))
            else:
                execution.results.append(result)

    async def _execute_orchestration(self, execution: WorkflowExecution) -> None:
        """Execute orchestrated workflow with dependencies."""
        # Build dependency graph
        graph = nx.DiGraph()
        for task in execution.spec.tasks:
            task_id = task.get("id", str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
            graph.add_node(task_id, **task)

            for dep in task.get("depends_on", []):
                graph.add_edge(dep, task_id)

        # Execute in topological order
        for task_id in nx.topological_sort(graph):
            task = graph.nodes[task_id]
            agent_id = task.get("agent_id")

            if agent_id:
                result = await self._execute_agent_task(agent_id, task)
                execution.results.append(result)

    async def _execute_agent_task(
        self,
        agent_id: str,
        task: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute task on specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        agent_info = self.agents[agent_id]

        # Ensure agent is ready
        if agent_info.state != AgentState.READY:
            await self.start_agent(agent_id)

        # Update state
        await self._transition_agent_state(agent_id, [AgentState.READY], AgentState.RUNNING)

        try:
            # Send task message
            message = Message(
                sender_id="coordinator",
                recipient_id=agent_id,
                message_type=MessageType.REQUEST,
                priority=Priority.HIGH,
                payload=task
            )

            # Use request-response pattern
            response = await self.message_bus.request_response(
                message,
                timeout_ms or self.config.state_transition_timeout_ms
            )

            if response:
                agent_info.message_count += 1
                return response.payload
            else:
                raise TimeoutError(f"Agent {agent_id} did not respond")

        finally:
            # Return to ready state
            await self._transition_agent_state(agent_id, [AgentState.RUNNING], AgentState.READY)

    async def _transition_agent_state(
        self,
        agent_id: str,
        from_states: List[AgentState],
        to_state: AgentState
    ) -> bool:
        """Transition agent to new state."""
        if agent_id not in self.agents:
            return False

        async with self._locks[agent_id]:
            agent_info = self.agents[agent_id]

            if agent_info.state not in from_states:
                logger.warning(
                    f"Cannot transition agent {agent_id} from {agent_info.state} to {to_state}"
                )
                return False

            old_state = agent_info.state
            agent_info.state = to_state

            # Update metrics
            agent_count_gauge.labels(state=old_state).dec()
            agent_count_gauge.labels(state=to_state).inc()
            agent_state_transitions.labels(from_state=old_state, to_state=to_state).inc()

            # Send state change event
            await self._send_lifecycle_event(agent_id, f"STATE_CHANGE:{to_state}")

            logger.debug(f"Agent {agent_id} transitioned from {old_state} to {to_state}")
            return True

    async def _send_lifecycle_event(self, agent_id: str, event: str) -> None:
        """Send agent lifecycle event."""
        message = Message(
            sender_id="coordinator",
            recipient_id="broadcast",
            message_type=MessageType.EVENT,
            payload={
                "agent_id": agent_id,
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        await self.message_bus.publish(message, topic="agent.lifecycle")

    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                for agent_id, agent_info in self.agents.items():
                    if agent_info.state in [AgentState.READY, AgentState.RUNNING]:
                        # Check last heartbeat
                        if agent_info.last_heartbeat:
                            last_heartbeat = datetime.fromisoformat(agent_info.last_heartbeat)
                            elapsed_ms = (now - last_heartbeat).total_seconds() * 1000

                            if elapsed_ms > self.config.heartbeat_interval_ms * 2:
                                logger.warning(f"Agent {agent_id} heartbeat timeout")
                                await self._transition_agent_state(
                                    agent_id,
                                    [AgentState.READY, AgentState.RUNNING],
                                    AgentState.ERROR
                                )

                await asyncio.sleep(self.config.heartbeat_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(1)

    async def _state_manager(self) -> None:
        """Manage agent state transitions."""
        while self._running:
            try:
                # Process state transition queue
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"State manager error: {e}")

    async def _message_handler(self) -> None:
        """Handle incoming messages from agents."""
        try:
            async for message in self.message_bus.subscribe(["agent.lifecycle", "agent.messages"]):
                if message.sender_id in self.agents:
                    # Update heartbeat
                    self.agents[message.sender_id].last_heartbeat = message.metadata.timestamp

                    # Process based on type
                    if message.message_type == MessageType.HEARTBEAT:
                        pass  # Heartbeat already recorded
                    elif message.message_type == MessageType.ERROR:
                        self.agents[message.sender_id].error_count += 1
                        logger.error(f"Agent {message.sender_id} error: {message.payload}")

        except Exception as e:
            logger.error(f"Message handler error: {e}")

    async def _health_checker(self) -> None:
        """Perform health checks on agents."""
        while self._running:
            try:
                unhealthy_agents = []

                for agent_id, agent_info in self.agents.items():
                    if agent_info.error_count > 10:
                        unhealthy_agents.append(agent_id)

                for agent_id in unhealthy_agents:
                    logger.warning(f"Agent {agent_id} marked unhealthy")
                    await self._transition_agent_state(
                        agent_id,
                        [AgentState.READY, AgentState.RUNNING],
                        AgentState.ERROR
                    )

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Health checker error: {e}")

    async def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information."""
        return self.agents.get(agent_id)

    async def list_agents(
        self,
        state: Optional[AgentState] = None,
        agent_type: Optional[str] = None
    ) -> List[AgentInfo]:
        """List registered agents with optional filters."""
        agents = list(self.agents.values())

        if state:
            agents = [a for a in agents if a.state == state]

        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]

        return agents

    async def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator metrics."""
        state_distribution = {}
        for state in AgentState:
            count = len([a for a in self.agents.values() if a.state == state])
            state_distribution[state] = count

        return {
            "total_agents": len(self.agents),
            "state_distribution": state_distribution,
            "active_workflows": len(self.workflows),
            "coordination_count": self._coordination_count,
            "error_count": self._error_count,
            "max_capacity": self.config.max_agents,
            "utilization": len(self.agents) / self.config.max_agents
        }