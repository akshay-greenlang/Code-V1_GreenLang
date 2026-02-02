# -*- coding: utf-8 -*-
"""
CoordinationLayer - Agent coordination patterns for orchestration.

This module implements coordination patterns for multi-agent systems including
hierarchical master-slave coordination, consensus-based decisions, distributed
locking, and transaction management with saga patterns.

Example:
    >>> coordinator = CoordinationLayer(config=CoordinationConfig())
    >>> async with coordinator.acquire_lock("resource-1"):
    ...     await coordinator.coordinate_agents(["agent-1", "agent-2"], task)

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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)


class CoordinationPattern(str, Enum):
    """Coordination patterns for agent interaction."""

    MASTER_SLAVE = "master_slave"
    PEER_TO_PEER = "peer_to_peer"
    CONSENSUS = "consensus"
    CHOREOGRAPHY = "choreography"
    ORCHESTRATION = "orchestration"


class LockState(str, Enum):
    """States for distributed locks."""

    AVAILABLE = "available"
    ACQUIRED = "acquired"
    WAITING = "waiting"
    EXPIRED = "expired"


class TransactionState(str, Enum):
    """States for distributed transactions."""

    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class ConsensusResult(str, Enum):
    """Results of consensus operations."""

    ACHIEVED = "achieved"
    NOT_ACHIEVED = "not_achieved"
    TIMEOUT = "timeout"
    INSUFFICIENT_VOTERS = "insufficient_voters"


@dataclass
class AgentInfo:
    """
    Information about a coordinated agent.

    Attributes:
        agent_id: Unique agent identifier
        role: Agent's role in coordination (master, slave, peer)
        capabilities: Set of capabilities the agent provides
        status: Current agent status
        last_seen: Last heartbeat timestamp
        metadata: Additional agent metadata
    """

    agent_id: str
    role: str = "slave"
    capabilities: Set[str] = field(default_factory=set)
    status: str = "active"
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self, timeout_seconds: float = 30.0) -> bool:
        """Check if agent is healthy based on last seen time."""
        last = datetime.fromisoformat(self.last_seen.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return age < timeout_seconds and self.status == "active"


@dataclass
class DistributedLock:
    """
    Represents a distributed lock for resource coordination.

    Attributes:
        lock_id: Unique lock identifier
        resource_id: ID of the locked resource
        holder_id: ID of the current lock holder
        state: Current lock state
        acquired_at: When lock was acquired
        expires_at: When lock expires
        waiters: List of agents waiting for the lock
    """

    resource_id: str
    lock_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    holder_id: Optional[str] = None
    state: LockState = LockState.AVAILABLE
    acquired_at: Optional[str] = None
    expires_at: Optional[str] = None
    waiters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if not self.expires_at:
            return False
        expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expires

    def acquire(self, holder_id: str, ttl_seconds: float = 30.0) -> bool:
        """
        Attempt to acquire the lock.

        Args:
            holder_id: ID of the requester
            ttl_seconds: Lock time-to-live

        Returns:
            True if lock was acquired
        """
        if self.state == LockState.AVAILABLE or self.is_expired():
            self.holder_id = holder_id
            self.state = LockState.ACQUIRED
            self.acquired_at = datetime.now(timezone.utc).isoformat()
            self.expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            ).isoformat()
            return True
        return False

    def release(self, holder_id: str) -> bool:
        """
        Release the lock.

        Args:
            holder_id: ID of the holder releasing the lock

        Returns:
            True if lock was released
        """
        if self.holder_id == holder_id:
            self.holder_id = None
            self.state = LockState.AVAILABLE
            self.acquired_at = None
            self.expires_at = None
            return True
        return False


@dataclass
class SagaStep:
    """
    A step in a saga transaction.

    Attributes:
        step_id: Unique step identifier
        name: Step name
        agent_id: Agent responsible for this step
        action: Action to execute
        compensation: Compensation action for rollback
        timeout_seconds: Step timeout
        retries: Number of retries
        state: Current step state
        result: Step result
        error: Step error if any
    """

    name: str
    agent_id: str
    action: Dict[str, Any]
    compensation: Dict[str, Any]
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout_seconds: float = 30.0
    retries: int = 3
    state: TransactionState = TransactionState.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "agent_id": self.agent_id,
            "action": self.action,
            "compensation": self.compensation,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "state": self.state.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Saga:
    """
    Saga transaction for distributed coordination.

    Implements the saga pattern for long-running transactions with
    compensation-based rollback on failures.

    Attributes:
        saga_id: Unique saga identifier
        name: Saga name
        steps: Ordered list of saga steps
        state: Current saga state
        current_step_index: Index of current step
        started_at: When saga started
        completed_at: When saga completed
        metadata: Additional saga metadata
    """

    name: str
    steps: List[SagaStep]
    saga_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TransactionState = TransactionState.PENDING
    current_step_index: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert saga to dictionary."""
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "state": self.state.value,
            "current_step_index": self.current_step_index,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass
class ConsensusVote:
    """
    A vote in a consensus decision.

    Attributes:
        voter_id: ID of the voting agent
        proposal_id: ID of the proposal being voted on
        vote: Vote value (approve, reject, abstain)
        weight: Vote weight
        timestamp: When vote was cast
        reason: Optional reason for vote
    """

    voter_id: str
    proposal_id: str
    vote: str  # approve, reject, abstain
    weight: float = 1.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason: Optional[str] = None


@dataclass
class ConsensusProposal:
    """
    A proposal for consensus voting.

    Attributes:
        proposal_id: Unique proposal identifier
        topic: What is being decided
        proposed_by: Agent proposing
        required_approval: Required approval ratio (0-1)
        deadline: Voting deadline
        votes: Collected votes
        result: Final consensus result
    """

    topic: str
    proposed_by: str
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    required_approval: float = 0.5
    deadline: Optional[str] = None
    votes: List[ConsensusVote] = field(default_factory=list)
    result: Optional[ConsensusResult] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def add_vote(self, vote: ConsensusVote) -> None:
        """Add a vote to the proposal."""
        # Remove any existing vote from this voter
        self.votes = [v for v in self.votes if v.voter_id != vote.voter_id]
        self.votes.append(vote)

    def calculate_result(self, total_voters: int) -> ConsensusResult:
        """Calculate consensus result."""
        if not self.votes:
            return ConsensusResult.INSUFFICIENT_VOTERS

        if len(self.votes) < total_voters * 0.5:  # Need at least half to vote
            return ConsensusResult.INSUFFICIENT_VOTERS

        total_weight = sum(v.weight for v in self.votes)
        approval_weight = sum(
            v.weight for v in self.votes if v.vote == "approve"
        )

        if total_weight == 0:
            return ConsensusResult.NOT_ACHIEVED

        approval_ratio = approval_weight / total_weight
        return (
            ConsensusResult.ACHIEVED
            if approval_ratio >= self.required_approval
            else ConsensusResult.NOT_ACHIEVED
        )


@dataclass
class CoordinationConfig:
    """
    Configuration for CoordinationLayer.

    Attributes:
        pattern: Default coordination pattern
        lock_ttl_seconds: Default lock TTL
        consensus_timeout_seconds: Timeout for consensus operations
        saga_timeout_seconds: Default saga timeout
        max_retries: Maximum retries for operations
        heartbeat_interval_seconds: Agent heartbeat interval
        agent_timeout_seconds: Agent health timeout
    """

    pattern: CoordinationPattern = CoordinationPattern.ORCHESTRATION
    lock_ttl_seconds: float = 30.0
    consensus_timeout_seconds: float = 60.0
    saga_timeout_seconds: float = 300.0
    max_retries: int = 3
    heartbeat_interval_seconds: float = 10.0
    agent_timeout_seconds: float = 30.0


@dataclass
class CoordinationMetrics:
    """Metrics for coordination monitoring."""

    agents_registered: int = 0
    agents_active: int = 0
    locks_held: int = 0
    locks_waiting: int = 0
    sagas_active: int = 0
    sagas_completed: int = 0
    sagas_failed: int = 0
    consensus_achieved: int = 0
    consensus_failed: int = 0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "agents_registered": self.agents_registered,
            "agents_active": self.agents_active,
            "locks_held": self.locks_held,
            "locks_waiting": self.locks_waiting,
            "sagas_active": self.sagas_active,
            "sagas_completed": self.sagas_completed,
            "sagas_failed": self.sagas_failed,
            "consensus_achieved": self.consensus_achieved,
            "consensus_failed": self.consensus_failed,
            "last_updated": self.last_updated,
        }


# Type aliases
ActionHandler = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]]


class CoordinationLayer:
    """
    Coordination layer for multi-agent orchestration.

    Provides:
    - Agent registration and health monitoring
    - Distributed locking
    - Saga-based transactions
    - Consensus-based decision making
    - Hierarchical coordination patterns

    Example:
        >>> config = CoordinationConfig(pattern=CoordinationPattern.ORCHESTRATION)
        >>> coordinator = CoordinationLayer(config)
        >>>
        >>> # Register agents
        >>> coordinator.register_agent(AgentInfo(
        ...     agent_id="thermal-agent",
        ...     role="slave",
        ...     capabilities={"thermal_calculation"}
        ... ))
        >>>
        >>> # Acquire lock
        >>> async with coordinator.acquire_lock("boiler-1"):
        ...     result = await coordinator.execute_action("thermal-agent", action)
        >>>
        >>> # Run saga
        >>> saga = Saga(
        ...     name="thermal_optimization",
        ...     steps=[step1, step2, step3]
        ... )
        >>> result = await coordinator.run_saga(saga)
    """

    def __init__(self, config: Optional[CoordinationConfig] = None) -> None:
        """
        Initialize CoordinationLayer.

        Args:
            config: Configuration options
        """
        self.config = config or CoordinationConfig()
        self._agents: Dict[str, AgentInfo] = {}
        self._locks: Dict[str, DistributedLock] = {}
        self._sagas: Dict[str, Saga] = {}
        self._proposals: Dict[str, ConsensusProposal] = {}
        self._action_handlers: Dict[str, ActionHandler] = {}
        self._metrics = CoordinationMetrics()
        self._lock = asyncio.Lock()
        self._master_id: Optional[str] = None

        logger.info(f"CoordinationLayer initialized with pattern: {config.pattern.value if config else 'orchestration'}")

    def register_agent(self, agent: AgentInfo) -> None:
        """
        Register an agent with the coordination layer.

        Args:
            agent: Agent information
        """
        self._agents[agent.agent_id] = agent
        self._update_metrics()
        logger.info(f"Registered agent: {agent.agent_id} (role: {agent.role})")

        # Set master if this is the first or it claims master role
        if agent.role == "master" or not self._master_id:
            if agent.role == "master":
                self._master_id = agent.agent_id

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

            # Release any locks held by this agent
            for lock in self._locks.values():
                if lock.holder_id == agent_id:
                    lock.release(agent_id)

            # Elect new master if needed
            if self._master_id == agent_id:
                self._elect_master()

            self._update_metrics()
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def register_action_handler(
        self,
        agent_id: str,
        handler: ActionHandler,
    ) -> None:
        """
        Register an action handler for an agent.

        Args:
            agent_id: Agent ID
            handler: Async function to handle actions
        """
        self._action_handlers[agent_id] = handler
        logger.debug(f"Registered action handler for agent: {agent_id}")

    async def heartbeat(self, agent_id: str) -> bool:
        """
        Update agent heartbeat.

        Args:
            agent_id: Agent sending heartbeat

        Returns:
            True if agent is registered
        """
        if agent_id in self._agents:
            self._agents[agent_id].last_seen = datetime.now(timezone.utc).isoformat()
            self._agents[agent_id].status = "active"
            self._update_metrics()
            return True
        return False

    async def acquire_lock(
        self,
        resource_id: str,
        holder_id: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> "LockContext":
        """
        Acquire a distributed lock.

        Args:
            resource_id: Resource to lock
            holder_id: ID of the lock requester
            ttl_seconds: Lock TTL

        Returns:
            Lock context manager
        """
        holder = holder_id or self._master_id or "anonymous"
        ttl = ttl_seconds or self.config.lock_ttl_seconds

        return LockContext(self, resource_id, holder, ttl)

    async def _acquire_lock_internal(
        self,
        resource_id: str,
        holder_id: str,
        ttl_seconds: float,
    ) -> bool:
        """
        Internal lock acquisition.

        Args:
            resource_id: Resource to lock
            holder_id: Lock holder ID
            ttl_seconds: Lock TTL

        Returns:
            True if lock acquired
        """
        async with self._lock:
            if resource_id not in self._locks:
                self._locks[resource_id] = DistributedLock(resource_id=resource_id)

            lock = self._locks[resource_id]

            # Check for expired locks
            if lock.is_expired():
                lock.state = LockState.AVAILABLE
                lock.holder_id = None

            if lock.acquire(holder_id, ttl_seconds):
                self._update_metrics()
                logger.debug(f"Lock acquired: {resource_id} by {holder_id}")
                return True

            # Add to waiters
            if holder_id not in lock.waiters:
                lock.waiters.append(holder_id)
                self._update_metrics()

            return False

    async def _release_lock_internal(
        self,
        resource_id: str,
        holder_id: str,
    ) -> bool:
        """
        Internal lock release.

        Args:
            resource_id: Resource to unlock
            holder_id: Lock holder ID

        Returns:
            True if lock released
        """
        async with self._lock:
            if resource_id not in self._locks:
                return False

            lock = self._locks[resource_id]
            if lock.release(holder_id):
                # Grant to first waiter if any
                if lock.waiters:
                    next_holder = lock.waiters.pop(0)
                    lock.acquire(next_holder, self.config.lock_ttl_seconds)

                self._update_metrics()
                logger.debug(f"Lock released: {resource_id} by {holder_id}")
                return True

            return False

    async def run_saga(
        self,
        saga: Saga,
        on_step_complete: Optional[Callable[[SagaStep], None]] = None,
    ) -> Saga:
        """
        Execute a saga transaction.

        Args:
            saga: Saga to execute
            on_step_complete: Optional callback for step completion

        Returns:
            Completed saga with results
        """
        saga.state = TransactionState.PREPARING
        saga.started_at = datetime.now(timezone.utc).isoformat()
        self._sagas[saga.saga_id] = saga
        self._update_metrics()

        logger.info(f"Starting saga: {saga.name} ({saga.saga_id})")

        try:
            # Execute forward steps
            for i, step in enumerate(saga.steps):
                saga.current_step_index = i
                step.state = TransactionState.PREPARING

                try:
                    result = await self._execute_saga_step(step)
                    step.result = result
                    step.state = TransactionState.COMMITTED

                    if on_step_complete:
                        on_step_complete(step)

                    logger.debug(f"Saga step completed: {step.name}")

                except Exception as e:
                    step.error = str(e)
                    step.state = TransactionState.ABORTED
                    logger.error(f"Saga step failed: {step.name} - {e}")

                    # Start compensation
                    await self._compensate_saga(saga, i)
                    return saga

            # All steps completed
            saga.state = TransactionState.COMMITTED
            saga.completed_at = datetime.now(timezone.utc).isoformat()
            self._metrics.sagas_completed += 1
            logger.info(f"Saga completed: {saga.name}")

        except Exception as e:
            saga.state = TransactionState.ABORTED
            saga.completed_at = datetime.now(timezone.utc).isoformat()
            self._metrics.sagas_failed += 1
            logger.error(f"Saga failed: {saga.name} - {e}")

        finally:
            self._update_metrics()

        return saga

    async def _execute_saga_step(self, step: SagaStep) -> Any:
        """Execute a single saga step."""
        handler = self._action_handlers.get(step.agent_id)
        if not handler:
            raise ValueError(f"No handler for agent: {step.agent_id}")

        # Execute with timeout and retries
        for attempt in range(step.retries + 1):
            try:
                result = await asyncio.wait_for(
                    handler(step.agent_id, step.action),
                    timeout=step.timeout_seconds,
                )
                return result
            except asyncio.TimeoutError:
                if attempt == step.retries:
                    raise
                await asyncio.sleep(1.0)
            except Exception:
                if attempt == step.retries:
                    raise
                await asyncio.sleep(1.0)

    async def _compensate_saga(self, saga: Saga, failed_step_index: int) -> None:
        """
        Run compensation for failed saga.

        Args:
            saga: Saga to compensate
            failed_step_index: Index where failure occurred
        """
        saga.state = TransactionState.COMPENSATING
        logger.info(f"Compensating saga: {saga.name}")

        # Compensate in reverse order
        for i in range(failed_step_index - 1, -1, -1):
            step = saga.steps[i]
            if step.state == TransactionState.COMMITTED:
                try:
                    handler = self._action_handlers.get(step.agent_id)
                    if handler:
                        await asyncio.wait_for(
                            handler(step.agent_id, step.compensation),
                            timeout=step.timeout_seconds,
                        )
                    step.state = TransactionState.COMPENSATED
                    logger.debug(f"Compensated step: {step.name}")
                except Exception as e:
                    logger.error(f"Compensation failed for step {step.name}: {e}")

        saga.state = TransactionState.COMPENSATED
        saga.completed_at = datetime.now(timezone.utc).isoformat()
        self._metrics.sagas_failed += 1

    async def propose_consensus(
        self,
        topic: str,
        proposed_by: str,
        data: Dict[str, Any],
        required_approval: float = 0.5,
        timeout_seconds: Optional[float] = None,
    ) -> ConsensusProposal:
        """
        Start a consensus proposal.

        Args:
            topic: What is being decided
            proposed_by: Proposing agent
            data: Proposal data
            required_approval: Required approval ratio
            timeout_seconds: Voting timeout

        Returns:
            Consensus proposal
        """
        timeout = timeout_seconds or self.config.consensus_timeout_seconds
        deadline = (
            datetime.now(timezone.utc) + timedelta(seconds=timeout)
        ).isoformat()

        proposal = ConsensusProposal(
            topic=topic,
            proposed_by=proposed_by,
            required_approval=required_approval,
            deadline=deadline,
            data=data,
        )

        self._proposals[proposal.proposal_id] = proposal
        logger.info(f"Consensus proposal created: {topic} ({proposal.proposal_id})")

        return proposal

    async def vote_on_proposal(
        self,
        proposal_id: str,
        voter_id: str,
        vote: str,
        weight: float = 1.0,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: Proposal to vote on
            voter_id: Voting agent
            vote: Vote value (approve, reject, abstain)
            weight: Vote weight
            reason: Optional reason

        Returns:
            True if vote was recorded
        """
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]

        # Check deadline
        if proposal.deadline:
            deadline = datetime.fromisoformat(proposal.deadline.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > deadline:
                return False

        consensus_vote = ConsensusVote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            vote=vote,
            weight=weight,
            reason=reason,
        )

        proposal.add_vote(consensus_vote)
        logger.debug(f"Vote recorded: {voter_id} voted {vote} on {proposal_id}")
        return True

    async def resolve_consensus(
        self,
        proposal_id: str,
    ) -> ConsensusResult:
        """
        Resolve a consensus proposal.

        Args:
            proposal_id: Proposal to resolve

        Returns:
            Consensus result
        """
        if proposal_id not in self._proposals:
            return ConsensusResult.NOT_ACHIEVED

        proposal = self._proposals[proposal_id]
        total_voters = len([a for a in self._agents.values() if a.is_healthy()])

        result = proposal.calculate_result(total_voters)
        proposal.result = result

        if result == ConsensusResult.ACHIEVED:
            self._metrics.consensus_achieved += 1
        else:
            self._metrics.consensus_failed += 1

        self._update_metrics()
        logger.info(f"Consensus resolved: {proposal.topic} - {result.value}")
        return result

    async def coordinate_agents(
        self,
        agent_ids: List[str],
        task: Dict[str, Any],
        pattern: Optional[CoordinationPattern] = None,
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents on a task.

        Args:
            agent_ids: Agents to coordinate
            task: Task to coordinate on
            pattern: Coordination pattern to use

        Returns:
            Coordination result
        """
        coord_pattern = pattern or self.config.pattern
        task_id = task.get("task_id", str(uuid.uuid4()))

        logger.info(
            f"Coordinating {len(agent_ids)} agents with pattern: {coord_pattern.value}"
        )

        result = {
            "task_id": task_id,
            "pattern": coord_pattern.value,
            "agents": agent_ids,
            "assignments": {},
            "status": "pending",
        }

        # Validate agents
        valid_agents = [
            aid for aid in agent_ids
            if aid in self._agents and self._agents[aid].is_healthy()
        ]

        if not valid_agents:
            result["status"] = "failed"
            result["error"] = "No healthy agents available"
            return result

        if coord_pattern == CoordinationPattern.MASTER_SLAVE:
            result = await self._coordinate_master_slave(valid_agents, task, result)
        elif coord_pattern == CoordinationPattern.PEER_TO_PEER:
            result = await self._coordinate_peer_to_peer(valid_agents, task, result)
        elif coord_pattern == CoordinationPattern.CONSENSUS:
            result = await self._coordinate_consensus(valid_agents, task, result)
        else:
            # Default orchestration
            result = await self._coordinate_orchestration(valid_agents, task, result)

        return result

    async def _coordinate_master_slave(
        self,
        agents: List[str],
        task: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Master-slave coordination pattern."""
        master = self._master_id or agents[0]
        slaves = [a for a in agents if a != master]

        result["assignments"][master] = {
            "role": "master",
            "task": "coordinate",
            "subordinates": slaves,
        }

        for slave in slaves:
            result["assignments"][slave] = {
                "role": "slave",
                "task": task.get("subtask", task),
                "reports_to": master,
            }

        result["status"] = "coordinated"
        return result

    async def _coordinate_peer_to_peer(
        self,
        agents: List[str],
        task: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Peer-to-peer coordination pattern."""
        # Distribute task equally among peers
        for i, agent_id in enumerate(agents):
            result["assignments"][agent_id] = {
                "role": "peer",
                "task": task,
                "peers": [a for a in agents if a != agent_id],
                "partition": i,
                "total_partitions": len(agents),
            }

        result["status"] = "coordinated"
        return result

    async def _coordinate_consensus(
        self,
        agents: List[str],
        task: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Consensus-based coordination pattern."""
        # Create proposal for the task
        proposal = await self.propose_consensus(
            topic=task.get("topic", "coordination_decision"),
            proposed_by=agents[0],
            data=task,
            required_approval=0.5,
        )

        result["proposal_id"] = proposal.proposal_id
        result["status"] = "awaiting_votes"

        for agent_id in agents:
            result["assignments"][agent_id] = {
                "role": "voter",
                "proposal_id": proposal.proposal_id,
                "action": "vote",
            }

        return result

    async def _coordinate_orchestration(
        self,
        agents: List[str],
        task: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Orchestration coordination pattern."""
        # Assign tasks based on capabilities
        subtasks = task.get("subtasks", [task])

        for i, subtask in enumerate(subtasks):
            required_capability = subtask.get("capability")

            # Find matching agent
            assigned = None
            for agent_id in agents:
                agent = self._agents.get(agent_id)
                if agent:
                    if not required_capability or required_capability in agent.capabilities:
                        assigned = agent_id
                        break

            if assigned:
                result["assignments"][assigned] = {
                    "role": "executor",
                    "subtask": subtask,
                    "sequence": i,
                }

        result["status"] = "coordinated"
        return result

    def _elect_master(self) -> None:
        """Elect a new master from available agents."""
        healthy_agents = [
            a for a in self._agents.values()
            if a.is_healthy() and a.role != "observer"
        ]

        if healthy_agents:
            # Simple election: first available with master capability or first overall
            masters = [a for a in healthy_agents if "master" in a.capabilities]
            new_master = masters[0] if masters else healthy_agents[0]
            self._master_id = new_master.agent_id
            new_master.role = "master"
            logger.info(f"Elected new master: {self._master_id}")
        else:
            self._master_id = None
            logger.warning("No agents available for master election")

    def _update_metrics(self) -> None:
        """Update coordination metrics."""
        self._metrics.agents_registered = len(self._agents)
        self._metrics.agents_active = len(
            [a for a in self._agents.values() if a.is_healthy()]
        )
        self._metrics.locks_held = len(
            [l for l in self._locks.values() if l.state == LockState.ACQUIRED]
        )
        self._metrics.locks_waiting = sum(
            len(l.waiters) for l in self._locks.values()
        )
        self._metrics.sagas_active = len(
            [s for s in self._sagas.values()
             if s.state not in [TransactionState.COMMITTED, TransactionState.ABORTED, TransactionState.COMPENSATED]]
        )
        self._metrics.last_updated = datetime.now(timezone.utc).isoformat()

    def get_metrics(self) -> CoordinationMetrics:
        """Get current coordination metrics."""
        return self._metrics

    def get_agents(self) -> Dict[str, AgentInfo]:
        """Get all registered agents."""
        return self._agents.copy()

    def get_master(self) -> Optional[str]:
        """Get current master agent ID."""
        return self._master_id


class LockContext:
    """
    Async context manager for distributed locks.

    Usage:
        >>> async with coordinator.acquire_lock("resource-1") as lock:
        ...     # Do work with locked resource
        ...     pass
    """

    def __init__(
        self,
        coordinator: CoordinationLayer,
        resource_id: str,
        holder_id: str,
        ttl_seconds: float,
    ) -> None:
        self._coordinator = coordinator
        self._resource_id = resource_id
        self._holder_id = holder_id
        self._ttl_seconds = ttl_seconds
        self._acquired = False

    async def __aenter__(self) -> "LockContext":
        """Acquire lock on enter."""
        # Wait for lock with timeout
        timeout = self._ttl_seconds * 2  # Wait up to 2x TTL
        start = time.time()

        while time.time() - start < timeout:
            if await self._coordinator._acquire_lock_internal(
                self._resource_id, self._holder_id, self._ttl_seconds
            ):
                self._acquired = True
                return self
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Failed to acquire lock on {self._resource_id}")

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release lock on exit."""
        if self._acquired:
            await self._coordinator._release_lock_internal(
                self._resource_id, self._holder_id
            )


# Factory function
def create_coordination_layer(
    pattern: CoordinationPattern = CoordinationPattern.ORCHESTRATION,
    lock_ttl: float = 30.0,
) -> CoordinationLayer:
    """
    Create a coordination layer with common configurations.

    Args:
        pattern: Coordination pattern
        lock_ttl: Lock time-to-live

    Returns:
        Configured CoordinationLayer instance
    """
    config = CoordinationConfig(
        pattern=pattern,
        lock_ttl_seconds=lock_ttl,
    )
    return CoordinationLayer(config)


__all__ = [
    "AgentInfo",
    "ConsensusProposal",
    "ConsensusResult",
    "ConsensusVote",
    "CoordinationConfig",
    "CoordinationLayer",
    "CoordinationMetrics",
    "CoordinationPattern",
    "DistributedLock",
    "LockContext",
    "LockState",
    "Saga",
    "SagaStep",
    "TransactionState",
    "create_coordination_layer",
]
