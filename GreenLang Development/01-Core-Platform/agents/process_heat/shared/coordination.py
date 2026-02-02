"""
MultiAgentCoordinator - Multi-agent coordination and orchestration utilities.

This module provides coordination primitives for multi-agent systems in the
GreenLang process heat ecosystem. It implements Contract Net Protocol (CNP),
blackboard architecture, and publish-subscribe patterns for agent communication.

Features:
    - Contract Net Protocol for task allocation
    - Blackboard pattern for shared knowledge
    - Pub/Sub messaging for events
    - Conflict resolution and consensus
    - Distributed locking
    - Agent discovery and registration

Example:
    >>> from greenlang.agents.process_heat.shared import MultiAgentCoordinator
    >>>
    >>> coordinator = MultiAgentCoordinator()
    >>> coordinator.register_agent("boiler_agent_1", BoilerAgent)
    >>> coordinator.broadcast_task(task, priority=Priority.HIGH)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import asyncio
import hashlib
import logging
import threading
import uuid
from collections import defaultdict
from queue import PriorityQueue

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MessageType(Enum):
    """Types of coordination messages."""
    TASK_ANNOUNCEMENT = auto()
    BID = auto()
    AWARD = auto()
    REJECT = auto()
    RESULT = auto()
    FAILURE = auto()
    HEARTBEAT = auto()
    SUBSCRIBE = auto()
    UNSUBSCRIBE = auto()
    EVENT = auto()
    QUERY = auto()
    RESPONSE = auto()
    LOCK_REQUEST = auto()
    LOCK_GRANT = auto()
    LOCK_RELEASE = auto()


class Priority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AgentRole(Enum):
    """Agent roles in coordination."""
    ORCHESTRATOR = auto()
    WORKER = auto()
    MONITOR = auto()
    GATEWAY = auto()
    SPECIALIST = auto()


class CoordinationProtocol(Enum):
    """Supported coordination protocols."""
    CONTRACT_NET = "contract_net"
    BLACKBOARD = "blackboard"
    PUB_SUB = "pub_sub"
    CONSENSUS = "consensus"
    AUCTION = "auction"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    PRIORITY = auto()
    TIMESTAMP = auto()
    VOTING = auto()
    ARBITRATION = auto()


# =============================================================================
# DATA MODELS
# =============================================================================

class CoordinationMessage(BaseModel):
    """Message for agent coordination."""

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier"
    )
    message_type: MessageType = Field(
        ...,
        description="Type of coordination message"
    )
    sender_id: str = Field(..., description="Sender agent ID")
    recipient_id: Optional[str] = Field(
        default=None,
        description="Recipient agent ID (None for broadcast)"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request-response"
    )
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Message priority"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp"
    )
    ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Time to live in seconds"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload"
    )

    class Config:
        use_enum_values = True

    def is_expired(self) -> bool:
        """Check if message has expired."""
        expiry_time = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry_time


class TaskSpec(BaseModel):
    """Specification for a coordination task."""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier"
    )
    task_type: str = Field(..., description="Task type identifier")
    description: str = Field(default="", description="Task description")
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Task priority"
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Task deadline"
    )
    required_capabilities: Set[str] = Field(
        default_factory=set,
        description="Required agent capabilities"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task parameters"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task constraints"
    )

    class Config:
        use_enum_values = True


class Bid(BaseModel):
    """Bid from an agent for a task."""

    bid_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique bid identifier"
    )
    task_id: str = Field(..., description="Task being bid on")
    agent_id: str = Field(..., description="Bidding agent ID")
    cost: float = Field(
        ...,
        ge=0,
        description="Estimated cost/time to complete"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in completing task"
    )
    estimated_completion_time: float = Field(
        ...,
        ge=0,
        description="Estimated completion time in seconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Bid timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional bid metadata"
    )


class TaskResult(BaseModel):
    """Result from task execution."""

    task_id: str = Field(..., description="Task ID")
    agent_id: str = Field(..., description="Executing agent ID")
    status: str = Field(..., description="success, failure, partial")
    result: Any = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Execution time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )


class AgentRegistration(BaseModel):
    """Agent registration information."""

    agent_id: str = Field(..., description="Agent unique identifier")
    agent_type: str = Field(..., description="Agent type (e.g., GL-002)")
    name: str = Field(..., description="Human-readable name")
    role: AgentRole = Field(
        default=AgentRole.WORKER,
        description="Agent role"
    )
    capabilities: Set[str] = Field(
        default_factory=set,
        description="Agent capabilities"
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="Agent communication endpoint"
    )
    status: str = Field(default="active", description="Agent status")
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Registration timestamp"
    )
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last heartbeat timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# COORDINATION COMPONENTS
# =============================================================================

class Blackboard:
    """
    Blackboard for shared knowledge between agents.

    The blackboard pattern allows agents to share information through
    a common data structure. Agents can read and write to specific
    sections of the blackboard.
    """

    def __init__(self) -> None:
        """Initialize the blackboard."""
        self._data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: List[Dict[str, Any]] = []

    def write(
        self,
        section: str,
        key: str,
        value: Any,
        writer_id: str,
    ) -> None:
        """
        Write a value to the blackboard.

        Args:
            section: Blackboard section
            key: Data key
            value: Value to write
            writer_id: ID of writing agent
        """
        with self._lock:
            old_value = self._data[section].get(key)
            self._data[section][key] = value

            # Record history
            self._history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "section": section,
                "key": key,
                "writer_id": writer_id,
                "operation": "write",
            })

            # Notify watchers
            for watcher in self._watchers.get(section, []):
                try:
                    watcher(section, key, old_value, value)
                except Exception as e:
                    logger.error(f"Watcher error: {e}")

    def read(self, section: str, key: str) -> Optional[Any]:
        """Read a value from the blackboard."""
        with self._lock:
            return self._data[section].get(key)

    def read_section(self, section: str) -> Dict[str, Any]:
        """Read all values from a section."""
        with self._lock:
            return dict(self._data[section])

    def watch(self, section: str, callback: Callable) -> None:
        """Register a watcher for a section."""
        with self._lock:
            self._watchers[section].append(callback)

    def delete(self, section: str, key: str, writer_id: str) -> bool:
        """Delete a value from the blackboard."""
        with self._lock:
            if key in self._data[section]:
                del self._data[section][key]
                self._history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "section": section,
                    "key": key,
                    "writer_id": writer_id,
                    "operation": "delete",
                })
                return True
            return False


class MessageBroker:
    """
    Message broker for pub/sub communication.

    Provides topic-based publish-subscribe messaging for agents.
    """

    def __init__(self) -> None:
        """Initialize the message broker."""
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._handlers: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self._message_queue: Dict[str, List[CoordinationMessage]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(
        self,
        topic: str,
        agent_id: str,
        handler: Callable[[CoordinationMessage], None],
    ) -> None:
        """
        Subscribe to a topic.

        Args:
            topic: Topic to subscribe to
            agent_id: Subscribing agent ID
            handler: Message handler callback
        """
        with self._lock:
            self._subscriptions[topic].add(agent_id)
            self._handlers[topic][agent_id] = handler
            logger.debug(f"Agent {agent_id} subscribed to {topic}")

    def unsubscribe(self, topic: str, agent_id: str) -> None:
        """Unsubscribe from a topic."""
        with self._lock:
            self._subscriptions[topic].discard(agent_id)
            self._handlers[topic].pop(agent_id, None)

    def publish(self, topic: str, message: CoordinationMessage) -> int:
        """
        Publish a message to a topic.

        Args:
            topic: Topic to publish to
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        with self._lock:
            subscribers = self._subscriptions.get(topic, set())
            delivered = 0

            for agent_id in subscribers:
                handler = self._handlers[topic].get(agent_id)
                if handler:
                    try:
                        handler(message)
                        delivered += 1
                    except Exception as e:
                        logger.error(f"Handler error for {agent_id}: {e}")

            return delivered

    def get_subscribers(self, topic: str) -> Set[str]:
        """Get subscribers for a topic."""
        with self._lock:
            return set(self._subscriptions.get(topic, set()))


class ContractNetManager:
    """
    Contract Net Protocol (CNP) manager.

    Implements the FIPA Contract Net Protocol for task allocation:
    1. Manager announces task
    2. Contractors submit bids
    3. Manager awards task to best bidder
    4. Contractor executes and reports result
    """

    def __init__(self, bid_timeout_s: float = 30.0) -> None:
        """Initialize the CNP manager."""
        self._pending_tasks: Dict[str, TaskSpec] = {}
        self._bids: Dict[str, List[Bid]] = defaultdict(list)
        self._awards: Dict[str, str] = {}  # task_id -> agent_id
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.RLock()
        self._bid_timeout = bid_timeout_s

    def announce_task(self, task: TaskSpec) -> str:
        """
        Announce a task for bidding.

        Args:
            task: Task specification

        Returns:
            Task ID
        """
        with self._lock:
            self._pending_tasks[task.task_id] = task
            self._bids[task.task_id] = []
            logger.info(f"Task announced: {task.task_id} ({task.task_type})")
            return task.task_id

    def submit_bid(self, bid: Bid) -> bool:
        """
        Submit a bid for a task.

        Args:
            bid: Bid submission

        Returns:
            True if bid accepted, False otherwise
        """
        with self._lock:
            task = self._pending_tasks.get(bid.task_id)
            if task is None:
                logger.warning(f"Bid submitted for unknown task: {bid.task_id}")
                return False

            if bid.task_id in self._awards:
                logger.warning(f"Bid submitted for already awarded task: {bid.task_id}")
                return False

            self._bids[bid.task_id].append(bid)
            logger.debug(f"Bid received from {bid.agent_id} for task {bid.task_id}")
            return True

    def evaluate_bids(
        self,
        task_id: str,
        strategy: str = "lowest_cost",
    ) -> Optional[str]:
        """
        Evaluate bids and award task.

        Args:
            task_id: Task to evaluate
            strategy: Evaluation strategy (lowest_cost, highest_confidence, balanced)

        Returns:
            Winning agent ID or None if no valid bids
        """
        with self._lock:
            bids = self._bids.get(task_id, [])
            if not bids:
                logger.warning(f"No bids received for task {task_id}")
                return None

            # Select winner based on strategy
            if strategy == "lowest_cost":
                winner = min(bids, key=lambda b: b.cost)
            elif strategy == "highest_confidence":
                winner = max(bids, key=lambda b: b.confidence)
            elif strategy == "balanced":
                # Weighted score: 40% cost, 40% confidence, 20% time
                def score(b: Bid) -> float:
                    max_cost = max(bid.cost for bid in bids) or 1
                    max_time = max(bid.estimated_completion_time for bid in bids) or 1
                    return (
                        0.4 * (1 - b.cost / max_cost) +
                        0.4 * b.confidence +
                        0.2 * (1 - b.estimated_completion_time / max_time)
                    )
                winner = max(bids, key=score)
            else:
                winner = bids[0]

            self._awards[task_id] = winner.agent_id
            logger.info(f"Task {task_id} awarded to {winner.agent_id}")
            return winner.agent_id

    def report_result(self, result: TaskResult) -> None:
        """Report task execution result."""
        with self._lock:
            self._results[result.task_id] = result
            # Clean up
            self._pending_tasks.pop(result.task_id, None)
            logger.info(
                f"Task {result.task_id} completed by {result.agent_id}: "
                f"{result.status}"
            )

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task."""
        with self._lock:
            if task_id in self._results:
                return {"status": "completed", "result": self._results[task_id]}
            if task_id in self._awards:
                return {
                    "status": "in_progress",
                    "assigned_to": self._awards[task_id],
                }
            if task_id in self._pending_tasks:
                return {
                    "status": "bidding",
                    "bid_count": len(self._bids[task_id]),
                }
            return {"status": "unknown"}


class DistributedLock:
    """
    Distributed lock for resource coordination.

    Provides mutual exclusion for shared resources across agents.
    """

    def __init__(self, lock_timeout_s: float = 60.0) -> None:
        """Initialize the distributed lock manager."""
        self._locks: Dict[str, Tuple[str, datetime]] = {}  # resource -> (holder, expiry)
        self._lock = threading.RLock()
        self._lock_timeout = lock_timeout_s
        self._waiters: Dict[str, List[str]] = defaultdict(list)

    def acquire(
        self,
        resource: str,
        agent_id: str,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """
        Attempt to acquire a lock on a resource.

        Args:
            resource: Resource identifier
            agent_id: Requesting agent ID
            timeout_s: Custom timeout (None uses default)

        Returns:
            True if lock acquired, False otherwise
        """
        timeout = timeout_s or self._lock_timeout

        with self._lock:
            # Check existing lock
            if resource in self._locks:
                holder, expiry = self._locks[resource]

                # Check if expired
                if datetime.now(timezone.utc) > expiry:
                    logger.debug(f"Lock on {resource} expired, releasing")
                    del self._locks[resource]
                elif holder == agent_id:
                    # Extend existing lock
                    self._locks[resource] = (
                        agent_id,
                        datetime.now(timezone.utc) + timedelta(seconds=timeout),
                    )
                    return True
                else:
                    # Lock held by another agent
                    return False

            # Acquire lock
            self._locks[resource] = (
                agent_id,
                datetime.now(timezone.utc) + timedelta(seconds=timeout),
            )
            logger.debug(f"Lock acquired on {resource} by {agent_id}")
            return True

    def release(self, resource: str, agent_id: str) -> bool:
        """
        Release a lock on a resource.

        Args:
            resource: Resource identifier
            agent_id: Holder agent ID

        Returns:
            True if released, False if not held
        """
        with self._lock:
            if resource not in self._locks:
                return False

            holder, _ = self._locks[resource]
            if holder != agent_id:
                logger.warning(
                    f"Agent {agent_id} tried to release lock held by {holder}"
                )
                return False

            del self._locks[resource]
            logger.debug(f"Lock released on {resource} by {agent_id}")
            return True

    def is_locked(self, resource: str) -> bool:
        """Check if a resource is locked."""
        with self._lock:
            if resource not in self._locks:
                return False

            _, expiry = self._locks[resource]
            if datetime.now(timezone.utc) > expiry:
                del self._locks[resource]
                return False

            return True

    def get_holder(self, resource: str) -> Optional[str]:
        """Get the agent holding a lock."""
        with self._lock:
            if resource in self._locks:
                holder, expiry = self._locks[resource]
                if datetime.now(timezone.utc) <= expiry:
                    return holder
            return None


# =============================================================================
# MULTI-AGENT COORDINATOR
# =============================================================================

class MultiAgentCoordinator:
    """
    Central coordinator for multi-agent systems.

    This class provides comprehensive coordination capabilities for
    the GreenLang process heat agent ecosystem, including task allocation,
    shared knowledge management, and conflict resolution.

    Features:
        - Agent registration and discovery
        - Contract Net Protocol for task allocation
        - Blackboard for shared knowledge
        - Pub/Sub messaging
        - Distributed locking
        - Conflict resolution

    Example:
        >>> coordinator = MultiAgentCoordinator()
        >>> coordinator.register_agent(registration)
        >>> task_id = coordinator.announce_task(task_spec)
        >>> result = await coordinator.wait_for_result(task_id)
    """

    def __init__(
        self,
        name: str = "ProcessHeatCoordinator",
        conflict_resolution: ConflictResolution = ConflictResolution.PRIORITY,
    ) -> None:
        """
        Initialize the multi-agent coordinator.

        Args:
            name: Coordinator name
            conflict_resolution: Default conflict resolution strategy
        """
        self.name = name
        self.conflict_resolution = conflict_resolution

        # Components
        self._agents: Dict[str, AgentRegistration] = {}
        self._blackboard = Blackboard()
        self._message_broker = MessageBroker()
        self._contract_net = ContractNetManager()
        self._distributed_lock = DistributedLock()

        # State
        self._lock = threading.RLock()
        self._event_log: List[Dict[str, Any]] = []
        self._coordinator_id = str(uuid.uuid4())

        logger.info(f"MultiAgentCoordinator '{name}' initialized")

    # =========================================================================
    # AGENT MANAGEMENT
    # =========================================================================

    def register_agent(self, registration: AgentRegistration) -> bool:
        """
        Register an agent with the coordinator.

        Args:
            registration: Agent registration information

        Returns:
            True if registered successfully
        """
        with self._lock:
            if registration.agent_id in self._agents:
                logger.warning(f"Agent {registration.agent_id} already registered")
                return False

            self._agents[registration.agent_id] = registration
            self._log_event("agent_registered", {
                "agent_id": registration.agent_id,
                "agent_type": registration.agent_type,
            })

            logger.info(
                f"Agent registered: {registration.agent_id} "
                f"({registration.agent_type})"
            )
            return True

    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent from the coordinator."""
        with self._lock:
            if agent_id not in self._agents:
                return False

            del self._agents[agent_id]
            self._log_event("agent_deregistered", {"agent_id": agent_id})
            logger.info(f"Agent deregistered: {agent_id}")
            return True

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID."""
        return self._agents.get(agent_id)

    def get_agents_by_capability(self, capability: str) -> List[AgentRegistration]:
        """Get all agents with a specific capability."""
        return [
            agent for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    def get_agents_by_type(self, agent_type: str) -> List[AgentRegistration]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]

    def update_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat timestamp."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = datetime.now(timezone.utc)
                return True
            return False

    # =========================================================================
    # CONTRACT NET PROTOCOL
    # =========================================================================

    def announce_task(
        self,
        task: TaskSpec,
        target_agents: Optional[List[str]] = None,
    ) -> str:
        """
        Announce a task for bidding.

        Args:
            task: Task specification
            target_agents: Optional list of specific agents to notify

        Returns:
            Task ID
        """
        task_id = self._contract_net.announce_task(task)

        # Broadcast task announcement
        message = CoordinationMessage(
            message_type=MessageType.TASK_ANNOUNCEMENT,
            sender_id=self._coordinator_id,
            priority=task.priority,
            payload=task.dict(),
        )

        if target_agents:
            for agent_id in target_agents:
                self._send_message(agent_id, message)
        else:
            # Broadcast to all agents with required capabilities
            eligible_agents = self._find_eligible_agents(task)
            for agent in eligible_agents:
                self._send_message(agent.agent_id, message)

        return task_id

    def submit_bid(self, bid: Bid) -> bool:
        """Submit a bid for a task."""
        return self._contract_net.submit_bid(bid)

    def evaluate_and_award(
        self,
        task_id: str,
        strategy: str = "balanced",
    ) -> Optional[str]:
        """Evaluate bids and award task."""
        winner = self._contract_net.evaluate_bids(task_id, strategy)

        if winner:
            # Send award message
            message = CoordinationMessage(
                message_type=MessageType.AWARD,
                sender_id=self._coordinator_id,
                recipient_id=winner,
                payload={"task_id": task_id},
            )
            self._send_message(winner, message)

        return winner

    def report_result(self, result: TaskResult) -> None:
        """Report task execution result."""
        self._contract_net.report_result(result)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        return self._contract_net.get_task_status(task_id)

    def _find_eligible_agents(self, task: TaskSpec) -> List[AgentRegistration]:
        """Find agents eligible for a task."""
        if not task.required_capabilities:
            return list(self._agents.values())

        return [
            agent for agent in self._agents.values()
            if task.required_capabilities.issubset(agent.capabilities)
        ]

    # =========================================================================
    # BLACKBOARD
    # =========================================================================

    def write_to_blackboard(
        self,
        section: str,
        key: str,
        value: Any,
        writer_id: str,
    ) -> None:
        """Write to shared blackboard."""
        self._blackboard.write(section, key, value, writer_id)

    def read_from_blackboard(self, section: str, key: str) -> Optional[Any]:
        """Read from shared blackboard."""
        return self._blackboard.read(section, key)

    def read_blackboard_section(self, section: str) -> Dict[str, Any]:
        """Read entire blackboard section."""
        return self._blackboard.read_section(section)

    def watch_blackboard(self, section: str, callback: Callable) -> None:
        """Watch a blackboard section for changes."""
        self._blackboard.watch(section, callback)

    # =========================================================================
    # PUB/SUB MESSAGING
    # =========================================================================

    def subscribe(
        self,
        topic: str,
        agent_id: str,
        handler: Callable[[CoordinationMessage], None],
    ) -> None:
        """Subscribe to a topic."""
        self._message_broker.subscribe(topic, agent_id, handler)

    def unsubscribe(self, topic: str, agent_id: str) -> None:
        """Unsubscribe from a topic."""
        self._message_broker.unsubscribe(topic, agent_id)

    def publish(self, topic: str, message: CoordinationMessage) -> int:
        """Publish message to topic."""
        return self._message_broker.publish(topic, message)

    def broadcast(self, message: CoordinationMessage) -> int:
        """Broadcast message to all agents."""
        count = 0
        for agent_id in self._agents.keys():
            self._send_message(agent_id, message)
            count += 1
        return count

    # =========================================================================
    # DISTRIBUTED LOCKING
    # =========================================================================

    def acquire_lock(
        self,
        resource: str,
        agent_id: str,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Acquire distributed lock."""
        return self._distributed_lock.acquire(resource, agent_id, timeout_s)

    def release_lock(self, resource: str, agent_id: str) -> bool:
        """Release distributed lock."""
        return self._distributed_lock.release(resource, agent_id)

    def is_locked(self, resource: str) -> bool:
        """Check if resource is locked."""
        return self._distributed_lock.is_locked(resource)

    # =========================================================================
    # CONFLICT RESOLUTION
    # =========================================================================

    def resolve_conflict(
        self,
        conflicting_values: List[Tuple[str, Any, Priority, datetime]],
        strategy: Optional[ConflictResolution] = None,
    ) -> Any:
        """
        Resolve a conflict between competing values.

        Args:
            conflicting_values: List of (agent_id, value, priority, timestamp)
            strategy: Resolution strategy (None uses default)

        Returns:
            Winning value
        """
        strategy = strategy or self.conflict_resolution

        if not conflicting_values:
            return None

        if strategy == ConflictResolution.PRIORITY:
            # Highest priority wins
            winner = max(conflicting_values, key=lambda x: x[2].value)
        elif strategy == ConflictResolution.TIMESTAMP:
            # Most recent wins
            winner = max(conflicting_values, key=lambda x: x[3])
        elif strategy == ConflictResolution.VOTING:
            # Most common value wins
            from collections import Counter
            values = [v[1] for v in conflicting_values]
            most_common = Counter(values).most_common(1)[0][0]
            winner = next(v for v in conflicting_values if v[1] == most_common)
        else:
            # Default to first
            winner = conflicting_values[0]

        self._log_event("conflict_resolved", {
            "strategy": strategy.name,
            "winner_agent": winner[0],
            "winner_value": str(winner[1]),
        })

        return winner[1]

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _send_message(self, agent_id: str, message: CoordinationMessage) -> bool:
        """Send message to a specific agent."""
        # In production, this would use actual transport
        # Here we log and could trigger registered handlers
        logger.debug(f"Message sent to {agent_id}: {message.message_type}")
        return True

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a coordination event."""
        self._event_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        })

    # =========================================================================
    # MONITORING
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "coordinator_id": self._coordinator_id,
            "name": self.name,
            "registered_agents": len(self._agents),
            "agent_types": list(set(a.agent_type for a in self._agents.values())),
            "event_count": len(self._event_log),
        }

    def get_agent_health(self, timeout_s: float = 60.0) -> Dict[str, str]:
        """Get health status of all agents."""
        now = datetime.now(timezone.utc)
        health = {}

        for agent_id, agent in self._agents.items():
            time_since_heartbeat = (now - agent.last_heartbeat).total_seconds()
            if time_since_heartbeat < timeout_s:
                health[agent_id] = "healthy"
            elif time_since_heartbeat < timeout_s * 2:
                health[agent_id] = "degraded"
            else:
                health[agent_id] = "unhealthy"

        return health
