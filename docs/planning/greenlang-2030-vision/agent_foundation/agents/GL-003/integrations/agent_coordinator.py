# -*- coding: utf-8 -*-
"""
Agent Coordination Module for GL-002 BoilerEfficiencyOptimizer

Enables communication and coordination with other GreenLang agents,
particularly GL-001 ProcessHeatOrchestrator and related heat system agents.

Features:
- Inter-agent messaging
- Task distribution
- Resource coordination
- State synchronization
- Event broadcasting
- Collaborative optimization
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import uuid
import hashlib
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    STATUS = "status"
    SYNC = "sync"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 1  # Immediate handling required
    HIGH = 2      # Process soon
    NORMAL = 3    # Standard priority
    LOW = 4       # Process when available
    BACKGROUND = 5  # Background processing


class AgentRole(Enum):
    """Agent roles in the heat system."""
    ORCHESTRATOR = "orchestrator"  # GL-001
    BOILER_OPTIMIZER = "boiler_optimizer"  # GL-002
    HEAT_RECOVERY = "heat_recovery"  # GL-003
    BURNER_CONTROLLER = "burner_controller"  # GL-004
    FEEDWATER_OPTIMIZER = "feedwater_optimizer"  # GL-005
    STEAM_DISTRIBUTION = "steam_distribution"  # GL-006
    MONITORING = "monitoring"
    ANALYTICS = "analytics"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """Inter-agent message structure."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    priority: MessagePriority
    timestamp: datetime
    payload: Dict[str, Any]
    requires_response: bool = False
    correlation_id: Optional[str] = None
    ttl: int = 3600  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentTask:
    """Task definition for agent coordination."""
    task_id: str
    task_type: str
    requester_id: str
    assignee_id: Optional[str]
    priority: MessagePriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentCapability:
    """Agent capability definition."""
    capability_name: str
    description: str
    parameters: List[str]
    performance_metrics: Dict[str, float]
    availability: bool = True
    max_concurrent_tasks: int = 10


@dataclass
class AgentProfile:
    """Agent profile and capabilities."""
    agent_id: str
    role: AgentRole
    capabilities: List[AgentCapability]
    status: str = "online"
    location: str = ""
    version: str = "1.0.0"
    last_heartbeat: Optional[datetime] = None
    performance_score: float = 100.0
    current_load: int = 0
    max_load: int = 100


class MessageBus:
    """
    Central message bus for inter-agent communication.

    Implements publish-subscribe pattern with routing.
    """

    def __init__(self):
        """Initialize message bus."""
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history = deque(maxlen=10000)
        self.routing_table: Dict[str, str] = {}
        self.processing = False
        self._process_task = None

    async def start(self):
        """Start message processing."""
        self.processing = True
        self._process_task = asyncio.create_task(self._process_messages())
        logger.info("Message bus started")

    async def stop(self):
        """Stop message processing."""
        self.processing = False
        if self._process_task:
            self._process_task.cancel()
        logger.info("Message bus stopped")

    async def publish(self, message: AgentMessage):
        """
        Publish message to bus.

        Args:
            message: Message to publish
        """
        await self.message_queue.put(message)
        self.message_history.append({
            'message_id': message.message_id,
            'timestamp': message.timestamp,
            'type': message.message_type.value,
            'sender': message.sender_id,
            'recipient': message.recipient_id
        })

    def subscribe(self, topic: str, callback: Callable):
        """
        Subscribe to message topic.

        Args:
            topic: Topic pattern (agent_id, role, or wildcard)
            callback: Async callback function
        """
        self.subscribers[topic].append(callback)
        logger.debug(f"Added subscriber for topic: {topic}")

    async def _process_messages(self):
        """Process messages from queue."""
        while self.processing:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Route message
                await self._route_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _route_message(self, message: AgentMessage):
        """Route message to appropriate subscribers."""
        # Direct recipient
        if message.recipient_id in self.subscribers:
            for callback in self.subscribers[message.recipient_id]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")

        # Broadcast messages
        if message.message_type == MessageType.BROADCAST:
            for topic, callbacks in self.subscribers.items():
                if topic != message.sender_id:  # Don't send to self
                    for callback in callbacks:
                        try:
                            await callback(message)
                        except Exception as e:
                            logger.error(f"Broadcast callback error: {e}")

        # Role-based routing
        if message.recipient_id.startswith("role:"):
            role = message.recipient_id.replace("role:", "")
            for topic, callbacks in self.subscribers.items():
                if topic.endswith(f":{role}"):
                    for callback in callbacks:
                        try:
                            await callback(message)
                        except Exception as e:
                            logger.error(f"Role callback error: {e}")


class TaskScheduler:
    """
    Task scheduling and distribution system.

    Manages task assignment and load balancing across agents.
    """

    def __init__(self):
        """Initialize task scheduler."""
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.agent_loads: Dict[str, int] = defaultdict(int)
        self.task_history = deque(maxlen=1000)
        self.scheduling_policies = {
            'round_robin': self._round_robin_assignment,
            'least_loaded': self._least_loaded_assignment,
            'capability_based': self._capability_based_assignment,
            'priority_weighted': self._priority_weighted_assignment
        }
        self.current_policy = 'least_loaded'

    async def submit_task(self, task: AgentTask) -> str:
        """
        Submit task for scheduling.

        Args:
            task: Task to schedule

        Returns:
            Task ID
        """
        self.tasks[task.task_id] = task
        await self.task_queue.put(task)
        logger.info(f"Task submitted: {task.task_id} ({task.task_type})")
        return task.task_id

    async def assign_task(
        self,
        task: AgentTask,
        available_agents: List[AgentProfile]
    ) -> Optional[str]:
        """
        Assign task to appropriate agent.

        Args:
            task: Task to assign
            available_agents: List of available agents

        Returns:
            Assigned agent ID or None
        """
        if not available_agents:
            return None

        # Use current scheduling policy
        assignment_func = self.scheduling_policies[self.current_policy]
        agent_id = await assignment_func(task, available_agents)

        if agent_id:
            task.assignee_id = agent_id
            task.status = TaskStatus.ASSIGNED
            self.agent_loads[agent_id] += 1
            logger.info(f"Task {task.task_id} assigned to {agent_id}")

        return agent_id

    async def _round_robin_assignment(
        self,
        task: AgentTask,
        agents: List[AgentProfile]
    ) -> Optional[str]:
        """Round-robin task assignment."""
        if not agents:
            return None

        # Simple round-robin based on task count
        min_tasks = min(self.agent_loads.values()) if self.agent_loads else 0
        for agent in agents:
            if self.agent_loads[agent.agent_id] == min_tasks:
                return agent.agent_id

        return agents[0].agent_id

    async def _least_loaded_assignment(
        self,
        task: AgentTask,
        agents: List[AgentProfile]
    ) -> Optional[str]:
        """Assign to least loaded agent."""
        if not agents:
            return None

        # Sort by current load
        sorted_agents = sorted(
            agents,
            key=lambda a: (a.current_load / a.max_load if a.max_load > 0 else 0)
        )

        return sorted_agents[0].agent_id

    async def _capability_based_assignment(
        self,
        task: AgentTask,
        agents: List[AgentProfile]
    ) -> Optional[str]:
        """Assign based on agent capabilities."""
        if not agents:
            return None

        # Find agents with required capability
        capable_agents = []
        for agent in agents:
            for capability in agent.capabilities:
                if capability.capability_name == task.task_type:
                    capable_agents.append(agent)
                    break

        if not capable_agents:
            return None

        # Among capable, choose least loaded
        return await self._least_loaded_assignment(task, capable_agents)

    async def _priority_weighted_assignment(
        self,
        task: AgentTask,
        agents: List[AgentProfile]
    ) -> Optional[str]:
        """Priority-weighted assignment."""
        if not agents:
            return None

        # High priority tasks go to high performance agents
        if task.priority == MessagePriority.CRITICAL:
            sorted_agents = sorted(
                agents,
                key=lambda a: a.performance_score,
                reverse=True
            )
            return sorted_agents[0].agent_id

        # Normal priority uses least loaded
        return await self._least_loaded_assignment(task, agents)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Update task status."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.status = status

        if status == TaskStatus.IN_PROGRESS:
            task.started_at = DeterministicClock.utcnow()
        elif status == TaskStatus.COMPLETED:
            task.completed_at = DeterministicClock.utcnow()
            task.result = result
            if task.assignee_id:
                self.agent_loads[task.assignee_id] = max(0, self.agent_loads[task.assignee_id] - 1)
        elif status == TaskStatus.FAILED:
            task.completed_at = DeterministicClock.utcnow()
            task.error = error
            if task.assignee_id:
                self.agent_loads[task.assignee_id] = max(0, self.agent_loads[task.assignee_id] - 1)

        # Add to history
        self.task_history.append({
            'task_id': task_id,
            'status': status.value,
            'timestamp': DeterministicClock.utcnow()
        })

    def get_task_status(self, task_id: str) -> Optional[AgentTask]:
        """Get task status."""
        return self.tasks.get(task_id)


class StateManager:
    """
    Manage shared state across agents.

    Provides consistent view of system state.
    """

    def __init__(self):
        """Initialize state manager."""
        self.global_state: Dict[str, Any] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.state_history = deque(maxlen=1000)
        self.state_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.state_version = 0
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

    async def update_state(
        self,
        agent_id: str,
        state_key: str,
        value: Any,
        broadcast: bool = True
    ):
        """
        Update shared state.

        Args:
            agent_id: Agent updating state
            state_key: State key
            value: New value
            broadcast: Broadcast change to subscribers
        """
        async with self.state_locks[state_key]:
            # Store previous value
            previous = self.global_state.get(state_key)

            # Update state
            self.global_state[state_key] = value
            self.state_version += 1

            # Update agent-specific state
            if agent_id not in self.agent_states:
                self.agent_states[agent_id] = {}
            self.agent_states[agent_id][state_key] = value

            # Record in history
            self.state_history.append({
                'version': self.state_version,
                'timestamp': DeterministicClock.utcnow(),
                'agent_id': agent_id,
                'key': state_key,
                'previous': previous,
                'new': value
            })

            # Broadcast to subscribers
            if broadcast:
                await self._broadcast_state_change(state_key, value, agent_id)

    async def _broadcast_state_change(
        self,
        state_key: str,
        value: Any,
        agent_id: str
    ):
        """Broadcast state change to subscribers."""
        for callback in self.subscribers.get(state_key, []):
            try:
                await callback(state_key, value, agent_id)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def subscribe_to_state(self, state_key: str, callback: Callable):
        """Subscribe to state changes."""
        self.subscribers[state_key].append(callback)

    def get_state(self, state_key: str) -> Any:
        """Get current state value."""
        return self.global_state.get(state_key)

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get all state for an agent."""
        return self.agent_states.get(agent_id, {})

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot."""
        return {
            'version': self.state_version,
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'global_state': self.global_state.copy(),
            'agent_states': self.agent_states.copy()
        }


class CollaborativeOptimizer:
    """
    Coordinate optimization across multiple agents.

    Implements distributed optimization algorithms.
    """

    def __init__(self):
        """Initialize collaborative optimizer."""
        self.optimization_sessions: Dict[str, Dict] = {}
        self.agent_contributions: Dict[str, Dict] = defaultdict(dict)
        self.consensus_threshold = 0.8

    async def start_optimization(
        self,
        session_id: str,
        objective: str,
        participating_agents: List[str],
        constraints: Dict[str, Any],
        timeout: int = 300
    ) -> str:
        """
        Start collaborative optimization session.

        Args:
            session_id: Unique session identifier
            objective: Optimization objective
            participating_agents: List of agent IDs
            constraints: Optimization constraints
            timeout: Session timeout in seconds

        Returns:
            Session ID
        """
        self.optimization_sessions[session_id] = {
            'objective': objective,
            'agents': participating_agents,
            'constraints': constraints,
            'start_time': DeterministicClock.utcnow(),
            'timeout': timeout,
            'status': 'active',
            'proposals': {},
            'consensus': None
        }

        logger.info(f"Started optimization session: {session_id}")
        return session_id

    async def submit_proposal(
        self,
        session_id: str,
        agent_id: str,
        proposal: Dict[str, Any]
    ) -> bool:
        """
        Submit optimization proposal from agent.

        Args:
            session_id: Session ID
            agent_id: Agent submitting proposal
            proposal: Optimization proposal

        Returns:
            Success status
        """
        if session_id not in self.optimization_sessions:
            return False

        session = self.optimization_sessions[session_id]

        if session['status'] != 'active':
            return False

        # Store proposal
        session['proposals'][agent_id] = {
            'proposal': proposal,
            'timestamp': DeterministicClock.utcnow(),
            'score': self._evaluate_proposal(proposal, session['constraints'])
        }

        # Check if all agents have submitted
        if len(session['proposals']) == len(session['agents']):
            await self._evaluate_consensus(session_id)

        return True

    def _evaluate_proposal(
        self,
        proposal: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> float:
        """Evaluate proposal against constraints."""
        score = 100.0

        # Check constraints
        for constraint, limit in constraints.items():
            if constraint in proposal:
                value = proposal[constraint]
                if isinstance(limit, tuple):  # Range constraint
                    min_val, max_val = limit
                    if value < min_val or value > max_val:
                        score -= 20
                elif isinstance(limit, (int, float)):  # Upper limit
                    if value > limit:
                        score -= 20

        return max(0, score)

    async def _evaluate_consensus(self, session_id: str):
        """Evaluate consensus among proposals."""
        session = self.optimization_sessions[session_id]
        proposals = session['proposals']

        if not proposals:
            return

        # Find best proposal
        best_agent = max(proposals, key=lambda a: proposals[a]['score'])
        best_proposal = proposals[best_agent]['proposal']
        best_score = proposals[best_agent]['score']

        # Check consensus threshold
        agreeing_agents = sum(
            1 for p in proposals.values()
            if p['score'] >= best_score * self.consensus_threshold
        )

        consensus_reached = agreeing_agents / len(proposals) >= self.consensus_threshold

        if consensus_reached:
            session['consensus'] = {
                'reached': True,
                'solution': best_proposal,
                'score': best_score,
                'agreement_rate': agreeing_agents / len(proposals)
            }
            session['status'] = 'completed'
            logger.info(f"Consensus reached for session {session_id}")
        else:
            # Need another round
            session['consensus'] = {
                'reached': False,
                'best_proposal': best_proposal,
                'disagreement_level': 1 - (agreeing_agents / len(proposals))
            }
            logger.info(f"No consensus for session {session_id}, continuing")

    def get_optimization_result(self, session_id: str) -> Optional[Dict]:
        """Get optimization result."""
        if session_id not in self.optimization_sessions:
            return None

        session = self.optimization_sessions[session_id]
        return session.get('consensus')


class AgentCoordinator:
    """
    Main coordination system for GL-002 BoilerEfficiencyOptimizer.

    Manages all inter-agent communication and coordination.
    """

    def __init__(self, agent_id: str = "GL-002", role: AgentRole = AgentRole.BOILER_OPTIMIZER):
        """Initialize agent coordinator."""
        self.agent_id = agent_id
        self.role = role
        self.message_bus = MessageBus()
        self.task_scheduler = TaskScheduler()
        self.state_manager = StateManager()
        self.optimizer = CollaborativeOptimizer()
        self.registered_agents: Dict[str, AgentProfile] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.running = False
        self._heartbeat_task = None
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup message handlers."""
        self.message_handlers = {
            MessageType.REQUEST: self._handle_request,
            MessageType.COMMAND: self._handle_command,
            MessageType.STATUS: self._handle_status,
            MessageType.SYNC: self._handle_sync,
            MessageType.HEARTBEAT: self._handle_heartbeat
        }

    async def start(self):
        """Start coordinator."""
        self.running = True

        # Start message bus
        await self.message_bus.start()

        # Subscribe to own messages
        self.message_bus.subscribe(self.agent_id, self._handle_message)

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Register with orchestrator
        await self.register_with_orchestrator()

        logger.info(f"Agent coordinator started: {self.agent_id}")

    async def stop(self):
        """Stop coordinator."""
        self.running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        await self.message_bus.stop()

        logger.info(f"Agent coordinator stopped: {self.agent_id}")

    async def register_with_orchestrator(self):
        """Register with GL-001 ProcessHeatOrchestrator."""
        # Define capabilities
        capabilities = [
            AgentCapability(
                capability_name="boiler_optimization",
                description="Optimize boiler efficiency and performance",
                parameters=["load", "fuel_type", "constraints"],
                performance_metrics={"accuracy": 0.95, "speed": 0.9}
            ),
            AgentCapability(
                capability_name="emissions_optimization",
                description="Minimize emissions while maintaining efficiency",
                parameters=["target_emissions", "load_requirement"],
                performance_metrics={"reduction": 0.3, "compliance": 0.99}
            ),
            AgentCapability(
                capability_name="fuel_optimization",
                description="Optimize fuel mix and consumption",
                parameters=["available_fuels", "cost_constraints"],
                performance_metrics={"cost_reduction": 0.15, "efficiency": 0.92}
            )
        ]

        profile = AgentProfile(
            agent_id=self.agent_id,
            role=self.role,
            capabilities=capabilities,
            status="online",
            version="1.0.0"
        )

        # Send registration message
        message = AgentMessage(
            message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            sender_id=self.agent_id,
            recipient_id="GL-001",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.HIGH,
            timestamp=DeterministicClock.utcnow(),
            payload={
                'action': 'register',
                'profile': {
                    'agent_id': profile.agent_id,
                    'role': profile.role.value,
                    'capabilities': [
                        {
                            'name': c.capability_name,
                            'description': c.description,
                            'parameters': c.parameters
                        }
                        for c in profile.capabilities
                    ]
                }
            },
            requires_response=True
        )

        await self.send_message(message)
        self.registered_agents[self.agent_id] = profile

        logger.info("Registered with orchestrator")

    async def send_message(self, message: AgentMessage):
        """Send message to another agent."""
        await self.message_bus.publish(message)

    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message."""
        handler = self.message_handlers.get(message.message_type)

        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.message_id}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")

    async def _handle_request(self, message: AgentMessage):
        """Handle request message."""
        action = message.payload.get('action')

        if action == 'optimize_boiler':
            # Handle boiler optimization request
            result = await self._perform_optimization(message.payload)
            await self._send_response(message, result)

        elif action == 'get_status':
            # Return current status
            status = await self._get_system_status()
            await self._send_response(message, status)

        elif action == 'coordinate_task':
            # Coordinate with other agents
            task = AgentTask(
                task_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                task_type=message.payload.get('task_type'),
                requester_id=message.sender_id,
                assignee_id=None,
                priority=MessagePriority.NORMAL,
                status=TaskStatus.PENDING,
                created_at=DeterministicClock.utcnow(),
                parameters=message.payload.get('parameters', {})
            )

            task_id = await self.task_scheduler.submit_task(task)
            await self._send_response(message, {'task_id': task_id})

    async def _handle_command(self, message: AgentMessage):
        """Handle command message."""
        command = message.payload.get('command')

        if command == 'start_optimization':
            # Start optimization session
            session_id = await self.optimizer.start_optimization(
                session_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                objective=message.payload.get('objective'),
                participating_agents=message.payload.get('agents', []),
                constraints=message.payload.get('constraints', {})
            )

            logger.info(f"Started optimization session: {session_id}")

        elif command == 'update_parameters':
            # Update operating parameters
            parameters = message.payload.get('parameters')
            await self.state_manager.update_state(
                self.agent_id,
                'operating_parameters',
                parameters
            )

    async def _handle_status(self, message: AgentMessage):
        """Handle status message."""
        # Update agent status
        sender_id = message.sender_id
        status = message.payload.get('status')

        if sender_id in self.registered_agents:
            self.registered_agents[sender_id].status = status
            self.registered_agents[sender_id].last_heartbeat = DeterministicClock.utcnow()

    async def _handle_sync(self, message: AgentMessage):
        """Handle synchronization message."""
        # Synchronize state with other agents
        state_data = message.payload.get('state')

        if state_data:
            for key, value in state_data.items():
                await self.state_manager.update_state(
                    message.sender_id,
                    key,
                    value,
                    broadcast=False
                )

    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle heartbeat message."""
        sender_id = message.sender_id

        if sender_id not in self.registered_agents:
            # New agent discovered
            profile_data = message.payload.get('profile')
            if profile_data:
                profile = AgentProfile(
                    agent_id=sender_id,
                    role=AgentRole(profile_data.get('role', 'monitoring')),
                    capabilities=[],
                    status='online',
                    last_heartbeat=DeterministicClock.utcnow()
                )
                self.registered_agents[sender_id] = profile
        else:
            # Update heartbeat
            self.registered_agents[sender_id].last_heartbeat = DeterministicClock.utcnow()

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.running:
            try:
                # Send heartbeat
                message = AgentMessage(
                    message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                    sender_id=self.agent_id,
                    recipient_id="broadcast",
                    message_type=MessageType.HEARTBEAT,
                    priority=MessagePriority.LOW,
                    timestamp=DeterministicClock.utcnow(),
                    payload={
                        'agent_id': self.agent_id,
                        'role': self.role.value,
                        'status': 'online',
                        'load': self.task_scheduler.agent_loads.get(self.agent_id, 0)
                    },
                    requires_response=False
                )

                await self.send_message(message)

                # Check for stale agents
                now = DeterministicClock.utcnow()
                for agent_id, profile in self.registered_agents.items():
                    if profile.last_heartbeat:
                        if (now - profile.last_heartbeat).total_seconds() > 60:
                            profile.status = 'offline'

                await asyncio.sleep(30)  # Heartbeat every 30 seconds

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)

    async def _send_response(self, request: AgentMessage, result: Dict[str, Any]):
        """Send response to request."""
        response = AgentMessage(
            message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            sender_id=self.agent_id,
            recipient_id=request.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request.priority,
            timestamp=DeterministicClock.utcnow(),
            payload=result,
            requires_response=False,
            correlation_id=request.message_id
        )

        await self.send_message(response)

    async def _perform_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform boiler optimization."""
        # Simulate optimization
        return {
            'success': True,
            'optimized_parameters': {
                'steam_pressure': 105.0,
                'steam_temperature': 495.0,
                'o2_content': 3.5,
                'efficiency': 91.5
            },
            'expected_savings': {
                'fuel_reduction': 5.2,  # %
                'emission_reduction': 8.3,  # %
                'cost_savings': 1250  # $/day
            }
        }

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'status': 'operational',
            'current_load': self.task_scheduler.agent_loads.get(self.agent_id, 0),
            'active_tasks': len([
                t for t in self.task_scheduler.tasks.values()
                if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
            ]),
            'registered_agents': len(self.registered_agents),
            'state_version': self.state_manager.state_version
        }

    async def request_collaboration(
        self,
        task_type: str,
        required_agents: List[str],
        parameters: Dict[str, Any]
    ) -> str:
        """
        Request collaboration from other agents.

        Args:
            task_type: Type of collaborative task
            required_agents: List of required agent IDs
            parameters: Task parameters

        Returns:
            Collaboration session ID
        """
        session_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Send collaboration request to each agent
        for agent_id in required_agents:
            message = AgentMessage(
                message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                sender_id=self.agent_id,
                recipient_id=agent_id,
                message_type=MessageType.REQUEST,
                priority=MessagePriority.HIGH,
                timestamp=DeterministicClock.utcnow(),
                payload={
                    'action': 'collaborate',
                    'session_id': session_id,
                    'task_type': task_type,
                    'parameters': parameters
                },
                requires_response=True
            )

            await self.send_message(message)

        logger.info(f"Initiated collaboration session: {session_id}")
        return session_id


# Example usage
async def main():
    """Example usage of agent coordinator."""

    # Initialize coordinator for GL-002
    coordinator = AgentCoordinator(
        agent_id="GL-002",
        role=AgentRole.BOILER_OPTIMIZER
    )

    # Start coordinator
    await coordinator.start()

    # Example: Send optimization request to GL-001
    optimization_request = AgentMessage(
        message_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        sender_id="GL-002",
        recipient_id="GL-001",
        message_type=MessageType.REQUEST,
        priority=MessagePriority.NORMAL,
        timestamp=DeterministicClock.utcnow(),
        payload={
            'action': 'optimize_system',
            'target': 'efficiency',
            'constraints': {
                'min_load': 80,  # MW
                'max_load': 120,  # MW
                'max_emissions': 100  # kg/hr CO2
            }
        },
        requires_response=True
    )

    await coordinator.send_message(optimization_request)

    # Example: Request collaboration for complex optimization
    collaboration_id = await coordinator.request_collaboration(
        task_type="multi_boiler_optimization",
        required_agents=["GL-001", "GL-003", "GL-004"],
        parameters={
            'total_steam_demand': 200,  # t/hr
            'available_boilers': 3,
            'optimization_window': 24  # hours
        }
    )

    print(f"Started collaboration: {collaboration_id}")

    # Update shared state
    await coordinator.state_manager.update_state(
        "GL-002",
        "boiler_efficiency",
        89.5
    )

    # Get system status
    status = await coordinator._get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")

    # Let it run for a bit
    await asyncio.sleep(10)

    # Stop coordinator
    await coordinator.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())