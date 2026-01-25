"""
GreenLang Framework - Agent-to-Agent (A2A) Protocol
Multi-Agent Collaboration and Communication

Based on:
- Google A2A Protocol Specification
- Microsoft AutoGen Multi-Agent Dialogue
- CrewAI Role-Based Collaboration
- AgentScope (Alibaba) Async Agent Coordination

This module enables seamless collaboration between GreenLang agents
through standardized message passing, task delegation, and result sharing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar
import hashlib
import json
import logging
import uuid
from collections import defaultdict


logger = logging.getLogger(__name__)

T = TypeVar('T')


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = auto()      # Request for action/information
    RESPONSE = auto()     # Response to a request
    BROADCAST = auto()    # Broadcast to all agents
    HANDOFF = auto()      # Transfer of task ownership
    STATUS = auto()       # Status update
    ERROR = auto()        # Error notification
    HEARTBEAT = auto()    # Health check


class AgentRole(Enum):
    """Roles agents can play in collaboration."""
    ORCHESTRATOR = "orchestrator"  # Coordinates other agents
    WORKER = "worker"              # Performs specific tasks
    REVIEWER = "reviewer"          # Reviews and validates work
    ADVISOR = "advisor"            # Provides recommendations
    MONITOR = "monitor"            # Monitors and reports


class TaskStatus(Enum):
    """Status of delegated tasks."""
    PENDING = auto()
    ACCEPTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class AgentCard:
    """
    Agent Card - Identity and capability description.

    Similar to the A2A Protocol Agent Card specification.
    """
    agent_id: str
    name: str
    description: str
    role: AgentRole
    capabilities: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    endpoint: str = ""
    max_concurrent_tasks: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "version": self.version,
            "endpoint": self.endpoint,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "metadata": self.metadata
        }

    def can_handle(self, capability: str) -> bool:
        """Check if agent has a capability."""
        return capability in self.capabilities


@dataclass
class A2AMessage:
    """
    Message for agent-to-agent communication.

    Follows the A2A Protocol message specification.
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str
    payload: Any
    correlation_id: str = ""  # For request-response linking
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 300
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = self.message_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.name,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "priority": self.priority,
            "metadata": self.metadata
        }

    def create_response(self, payload: Any) -> 'A2AMessage':
        """Create a response message."""
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            payload=payload,
            correlation_id=self.correlation_id
        )


@dataclass
class DelegatedTask:
    """
    Task delegated between agents.

    Tracks the full lifecycle of a task from delegation to completion.
    """
    task_id: str
    name: str
    description: str
    required_capability: str
    input_data: Any
    output_data: Any = None
    status: TaskStatus = TaskStatus.PENDING
    delegator_id: str = ""
    assignee_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    error_message: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.provenance_hash:
            data = f"{self.task_id}:{json.dumps(self.input_data, sort_keys=True, default=str)}"
            self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()

    def start(self, assignee_id: str) -> None:
        """Mark task as started."""
        self.assignee_id = assignee_id
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)

    def complete(self, output_data: Any) -> None:
        """Mark task as completed."""
        self.output_data = output_data
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def fail(self, error_message: str) -> None:
        """Mark task as failed."""
        self.error_message = error_message
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.status = TaskStatus.FAILED
        else:
            self.status = TaskStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "required_capability": self.required_capability,
            "status": self.status.name,
            "delegator_id": self.delegator_id,
            "assignee_id": self.assignee_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "provenance_hash": self.provenance_hash
        }


class A2AAgent(ABC):
    """
    Base class for A2A-enabled agents.

    Provides the infrastructure for agent registration, message handling,
    and task processing.
    """

    def __init__(self, card: AgentCard):
        self.card = card
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._pending_tasks: Dict[str, DelegatedTask] = {}
        self._completed_tasks: List[DelegatedTask] = []

        # Register default handlers
        self._message_handlers[MessageType.REQUEST] = self._handle_request
        self._message_handlers[MessageType.HANDOFF] = self._handle_handoff
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat

    @property
    def agent_id(self) -> str:
        return self.card.agent_id

    @abstractmethod
    def process_task(self, task: DelegatedTask) -> Any:
        """Process a delegated task. Must be implemented by subclasses."""
        pass

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[A2AMessage], Optional[A2AMessage]]
    ) -> None:
        """Register a custom message handler."""
        self._message_handlers[message_type] = handler

    def receive_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Receive and process a message."""
        logger.debug(f"Agent {self.agent_id} received message: {message.message_id}")

        handler = self._message_handlers.get(message.message_type)
        if handler:
            return handler(message)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            return None

    def _handle_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle a request message."""
        payload = message.payload
        if isinstance(payload, dict) and "task" in payload:
            task_data = payload["task"]
            task = DelegatedTask(
                task_id=task_data.get("task_id", str(uuid.uuid4())),
                name=task_data.get("name", ""),
                description=task_data.get("description", ""),
                required_capability=task_data.get("required_capability", ""),
                input_data=task_data.get("input_data"),
                delegator_id=message.sender_id
            )

            # Check if we can handle this task
            if not self.card.can_handle(task.required_capability):
                return message.create_response({
                    "accepted": False,
                    "reason": f"Agent lacks capability: {task.required_capability}"
                })

            # Accept and process task
            task.start(self.agent_id)
            self._pending_tasks[task.task_id] = task

            try:
                result = self.process_task(task)
                task.complete(result)
                self._completed_tasks.append(task)
                del self._pending_tasks[task.task_id]

                return message.create_response({
                    "accepted": True,
                    "task_id": task.task_id,
                    "status": task.status.name,
                    "result": result,
                    "provenance_hash": task.provenance_hash
                })
            except Exception as e:
                task.fail(str(e))
                return message.create_response({
                    "accepted": True,
                    "task_id": task.task_id,
                    "status": task.status.name,
                    "error": str(e)
                })

        return message.create_response({"error": "Invalid request format"})

    def _handle_handoff(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle a task handoff."""
        logger.info(f"Agent {self.agent_id} received handoff from {message.sender_id}")
        return self._handle_request(message)

    def _handle_heartbeat(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle a heartbeat message."""
        return message.create_response({
            "status": "alive",
            "pending_tasks": len(self._pending_tasks),
            "completed_tasks": len(self._completed_tasks)
        })

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.card.name,
            "role": self.card.role.value,
            "pending_tasks": len(self._pending_tasks),
            "completed_tasks": len(self._completed_tasks),
            "capabilities": self.card.capabilities
        }


class A2ARouter:
    """
    Routes messages between agents.

    Acts as a message broker for the agent network.
    """

    def __init__(self):
        self._agents: Dict[str, A2AAgent] = {}
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)
        self._message_queue: List[A2AMessage] = []

    def register_agent(self, agent: A2AAgent) -> None:
        """Register an agent with the router."""
        self._agents[agent.agent_id] = agent
        for capability in agent.card.capabilities:
            self._capability_index[capability].add(agent.agent_id)
        logger.info(f"Registered agent: {agent.agent_id}")

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            for capability in agent.card.capabilities:
                self._capability_index[capability].discard(agent_id)
            del self._agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[A2AAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents with a specific capability."""
        return list(self._capability_index.get(capability, set()))

    def route_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Route a message to its recipient."""
        if message.message_type == MessageType.BROADCAST:
            # Broadcast to all agents
            responses = []
            for agent in self._agents.values():
                if agent.agent_id != message.sender_id:
                    response = agent.receive_message(message)
                    if response:
                        responses.append(response)
            return None  # Broadcast doesn't return a single response

        # Route to specific recipient
        recipient = self._agents.get(message.recipient_id)
        if recipient:
            return recipient.receive_message(message)
        else:
            logger.warning(f"Recipient not found: {message.recipient_id}")
            return None

    def delegate_task(
        self,
        delegator_id: str,
        task: DelegatedTask
    ) -> Optional[A2AMessage]:
        """
        Delegate a task to an available agent.

        Finds an agent with the required capability and sends the task.
        """
        # Find capable agents
        capable_agents = self.find_agents_by_capability(task.required_capability)

        if not capable_agents:
            logger.warning(f"No agents found for capability: {task.required_capability}")
            return None

        # Select agent (simple round-robin or load-based selection)
        selected_agent_id = capable_agents[0]

        # Create and route message
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=delegator_id,
            recipient_id=selected_agent_id,
            payload={"task": task.to_dict()}
        )

        return self.route_message(message)

    def get_network_status(self) -> Dict[str, Any]:
        """Get status of the agent network."""
        return {
            "total_agents": len(self._agents),
            "agents": [agent.get_status() for agent in self._agents.values()],
            "capabilities": {
                cap: list(agents) for cap, agents in self._capability_index.items()
            }
        }


# ============================================================================
# CREW-STYLE MULTI-AGENT ORCHESTRATION
# ============================================================================

@dataclass
class CrewTask:
    """Task definition for crew-style orchestration."""
    task_id: str
    name: str
    description: str
    expected_output: str
    agent_role: AgentRole
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class AgentCrew:
    """
    CrewAI-style multi-agent collaboration.

    Orchestrates a team of specialized agents to accomplish complex tasks.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._router = A2ARouter()
        self._agents: Dict[AgentRole, A2AAgent] = {}
        self._task_results: Dict[str, Any] = {}

    def add_agent(self, agent: A2AAgent) -> 'AgentCrew':
        """Add an agent to the crew."""
        self._router.register_agent(agent)
        self._agents[agent.card.role] = agent
        return self

    def get_agent(self, role: AgentRole) -> Optional[A2AAgent]:
        """Get agent by role."""
        return self._agents.get(role)

    def kickoff(self, tasks: List[CrewTask]) -> Dict[str, Any]:
        """
        Execute a series of tasks using the crew.

        Tasks are executed in dependency order.
        """
        # Build dependency graph
        task_map = {t.task_id: t for t in tasks}
        completed = set()
        results = {}

        # Simple topological sort execution
        while len(completed) < len(tasks):
            for task in tasks:
                if task.task_id in completed:
                    continue

                # Check dependencies
                if all(dep in completed for dep in task.dependencies):
                    # Get dependency results as context
                    context = task.context.copy()
                    for dep in task.dependencies:
                        context[dep] = results.get(dep)

                    # Find and execute with appropriate agent
                    agent = self._agents.get(task.agent_role)
                    if agent:
                        delegated = DelegatedTask(
                            task_id=task.task_id,
                            name=task.name,
                            description=task.description,
                            required_capability=task.agent_role.value,
                            input_data={"task": task, "context": context},
                            delegator_id="crew_orchestrator"
                        )

                        try:
                            result = agent.process_task(delegated)
                            results[task.task_id] = result
                            completed.add(task.task_id)
                            logger.info(f"Completed task: {task.task_id}")
                        except Exception as e:
                            logger.error(f"Task failed: {task.task_id}: {e}")
                            results[task.task_id] = {"error": str(e)}
                            completed.add(task.task_id)
                    else:
                        logger.warning(f"No agent for role: {task.agent_role}")
                        completed.add(task.task_id)

        self._task_results = results
        return results

    def get_results(self) -> Dict[str, Any]:
        """Get all task results."""
        return self._task_results


# ============================================================================
# GLOBAL ROUTER INSTANCE
# ============================================================================

GREENLANG_A2A_ROUTER = A2ARouter()
