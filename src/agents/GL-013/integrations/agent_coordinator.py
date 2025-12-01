"""
Agent Coordinator Module for GL-013 PREDICTMAINT (Predictive Maintenance Agent).

Provides multi-agent coordination capabilities for integration with other GreenLang
agents (GL-001 to GL-012). Implements message bus integration, task distribution,
response aggregation, consensus mechanisms, and priority-based routing.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectorError,
    TimeoutError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class AgentID(str, Enum):
    """GreenLang agent identifiers."""

    GL_001_THERMOSYNC = "GL-001"  # Thermal Optimization
    GL_002_GRIDBALANCE = "GL-002"  # Grid Balancing
    GL_003_CARBONTRACK = "GL-003"  # Carbon Tracking
    GL_004_WATERWISE = "GL-004"  # Water Management
    GL_005_WASTEWATCH = "GL-005"  # Waste Management
    GL_006_AIRQUALITY = "GL-006"  # Air Quality
    GL_007_ENERGYOPT = "GL-007"  # Energy Optimization
    GL_008_SUPPLYCHAIN = "GL-008"  # Supply Chain
    GL_009_COMPLIANCE = "GL-009"  # Compliance
    GL_010_CARBONSCOPE = "GL-010"  # Emissions Compliance
    GL_011_SAFETYGUARD = "GL-011"  # Safety Monitoring
    GL_012_ASSETTRACK = "GL-012"  # Asset Tracking
    GL_013_PREDICTMAINT = "GL-013"  # Predictive Maintenance (self)


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"
    COMMAND = "command"
    QUERY = "query"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    CONSENSUS_PROPOSE = "consensus_propose"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_COMMIT = "consensus_commit"


class MessagePriority(str, Enum):
    """Message priority levels."""

    CRITICAL = "critical"  # Immediate handling required
    HIGH = "high"  # Priority processing
    NORMAL = "normal"  # Standard processing
    LOW = "low"  # Background processing
    BULK = "bulk"  # Batch processing allowed


class TaskStatus(str, Enum):
    """Distributed task status."""

    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ConsensusState(str, Enum):
    """Consensus protocol state."""

    IDLE = "idle"
    PROPOSING = "proposing"
    VOTING = "voting"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTED = "aborted"


class RoutingStrategy(str, Enum):
    """Message routing strategies."""

    DIRECT = "direct"  # Direct to specific agent
    BROADCAST = "broadcast"  # All agents
    MULTICAST = "multicast"  # Specific group of agents
    ROUND_ROBIN = "round_robin"  # Load balanced
    PRIORITY = "priority"  # Priority-based routing
    CAPABILITY = "capability"  # Route based on agent capabilities


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    WEIGHTED = "weighted"
    HASH_BASED = "hash_based"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class MessageBusConfig(BaseModel):
    """Message bus configuration."""

    model_config = ConfigDict(extra="forbid")

    bus_type: str = Field(default="redis", description="Message bus type (redis, rabbitmq, kafka)")
    host: str = Field(default="localhost", description="Bus host")
    port: int = Field(default=6379, ge=1, le=65535, description="Bus port")
    password: Optional[str] = Field(default=None, description="Bus password")
    database: int = Field(default=0, ge=0, description="Database number (Redis)")

    # Channel configuration
    channel_prefix: str = Field(default="greenlang:", description="Channel prefix")
    request_channel: str = Field(default="requests", description="Request channel")
    response_channel: str = Field(default="responses", description="Response channel")
    event_channel: str = Field(default="events", description="Event channel")
    broadcast_channel: str = Field(default="broadcast", description="Broadcast channel")

    # Connection settings
    connection_timeout_seconds: float = Field(default=10.0, description="Connection timeout")
    read_timeout_seconds: float = Field(default=30.0, description="Read timeout")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max connections")

    # Message settings
    message_ttl_seconds: int = Field(default=3600, ge=60, description="Message TTL")
    max_message_size_bytes: int = Field(default=1048576, description="Max message size (1MB)")


class AgentCoordinatorConfig(BaseConnectorConfig):
    """Configuration for agent coordinator."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    agent_id: AgentID = Field(default=AgentID.GL_013_PREDICTMAINT, description="This agent's ID")
    agent_name: str = Field(default="PREDICTMAINT", description="Agent name")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Message bus
    message_bus_config: MessageBusConfig = Field(
        default_factory=MessageBusConfig,
        description="Message bus configuration"
    )

    # Routing
    default_routing_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.DIRECT,
        description="Default routing strategy"
    )
    load_balance_strategy: LoadBalanceStrategy = Field(
        default=LoadBalanceStrategy.ROUND_ROBIN,
        description="Load balancing strategy"
    )

    # Timeouts
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout"
    )
    consensus_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Consensus timeout"
    )
    heartbeat_interval_seconds: float = Field(
        default=10.0,
        ge=5.0,
        le=60.0,
        description="Heartbeat interval"
    )
    agent_timeout_seconds: float = Field(
        default=60.0,
        ge=30.0,
        le=300.0,
        description="Agent offline threshold"
    )

    # Task distribution
    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max concurrent tasks"
    )
    task_queue_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Task queue size"
    )
    task_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Task retry count"
    )

    # Consensus
    consensus_enabled: bool = Field(default=True, description="Enable consensus protocol")
    consensus_quorum_percentage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quorum percentage for consensus"
    )

    # Capabilities this agent offers
    capabilities: List[str] = Field(
        default_factory=lambda: [
            "predictive_maintenance",
            "equipment_health",
            "failure_prediction",
            "maintenance_scheduling",
            "vibration_analysis",
        ],
        description="Agent capabilities"
    )

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.AGENT_COORDINATOR


# =============================================================================
# Pydantic Models - Messages
# =============================================================================


class AgentMessage(BaseModel):
    """Inter-agent message."""

    model_config = ConfigDict(extra="allow")

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message ID"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request/response"
    )
    message_type: MessageType = Field(..., description="Message type")
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="Message priority"
    )

    # Routing
    source_agent: AgentID = Field(..., description="Source agent ID")
    target_agent: Optional[AgentID] = Field(default=None, description="Target agent ID")
    target_agents: Optional[List[AgentID]] = Field(
        default=None,
        description="Target agents for multicast"
    )

    # Content
    action: str = Field(..., description="Action/operation name")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload")

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp"
    )
    expires_at: Optional[datetime] = Field(default=None, description="Message expiration")
    retry_count: int = Field(default=0, ge=0, description="Retry count")

    # Tracing
    trace_id: Optional[str] = Field(default=None, description="Distributed trace ID")
    span_id: Optional[str] = Field(default=None, description="Span ID")
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentResponse(BaseModel):
    """Response to an agent message."""

    model_config = ConfigDict(extra="allow")

    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Response ID"
    )
    correlation_id: str = Field(..., description="Original message ID")
    source_agent: AgentID = Field(..., description="Responding agent")
    target_agent: AgentID = Field(..., description="Original requester")

    # Response content
    success: bool = Field(..., description="Whether request succeeded")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response time")
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Processing time in ms"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentStatus(BaseModel):
    """Agent status information."""

    model_config = ConfigDict(extra="allow")

    agent_id: AgentID = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    agent_version: str = Field(..., description="Agent version")

    # Status
    status: str = Field(default="online", description="Agent status")
    last_heartbeat: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last heartbeat"
    )
    uptime_seconds: int = Field(default=0, ge=0, description="Uptime in seconds")

    # Capabilities
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")

    # Load
    current_tasks: int = Field(default=0, ge=0, description="Current task count")
    max_tasks: int = Field(default=10, ge=1, description="Max concurrent tasks")
    queue_size: int = Field(default=0, ge=0, description="Task queue size")
    load_percentage: float = Field(default=0.0, ge=0, le=100, description="Load percentage")

    # Health
    health_status: HealthStatus = Field(
        default=HealthStatus.HEALTHY,
        description="Health status"
    )
    error_count: int = Field(default=0, ge=0, description="Recent error count")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DistributedTask(BaseModel):
    """Distributed task for multi-agent execution."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Task ID"
    )
    task_type: str = Field(..., description="Task type")
    description: Optional[str] = Field(default=None, description="Task description")

    # Assignment
    owner_agent: AgentID = Field(..., description="Task owner")
    assigned_agent: Optional[AgentID] = Field(default=None, description="Assigned agent")
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="Required capabilities"
    )

    # Execution
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="Task priority"
    )
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    deadline: Optional[datetime] = Field(default=None, description="Task deadline")
    timeout_seconds: float = Field(default=60.0, ge=1.0, description="Task timeout")

    # Result
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message")

    # Retry
    retry_count: int = Field(default=0, ge=0, description="Retry count")
    max_retries: int = Field(default=3, ge=0, description="Max retries")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConsensusProposal(BaseModel):
    """Consensus protocol proposal."""

    model_config = ConfigDict(extra="allow")

    proposal_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Proposal ID"
    )
    proposer: AgentID = Field(..., description="Proposing agent")
    proposal_type: str = Field(..., description="Type of proposal")
    description: str = Field(..., description="Proposal description")

    # Content
    proposal_data: Dict[str, Any] = Field(..., description="Proposal content")

    # Voting
    voters: List[AgentID] = Field(default_factory=list, description="Required voters")
    votes: Dict[str, bool] = Field(default_factory=dict, description="Votes received")
    quorum_required: float = Field(default=0.5, ge=0, le=1, description="Quorum required")

    # State
    state: ConsensusState = Field(default=ConsensusState.IDLE, description="Consensus state")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    expires_at: datetime = Field(..., description="Expiration time")
    decided_at: Optional[datetime] = Field(default=None, description="Decision time")

    # Result
    approved: Optional[bool] = Field(default=None, description="Whether approved")
    result_message: Optional[str] = Field(default=None, description="Result message")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Message Handler Registry
# =============================================================================


class MessageHandlerRegistry:
    """Registry for message handlers."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_handlers: Dict[str, List[Callable]] = defaultdict(list)

    def register(
        self,
        action: str,
        handler: Callable[[AgentMessage], Any],
        is_async: bool = True,
    ) -> None:
        """
        Register a message handler.

        Args:
            action: Action name to handle
            handler: Handler function
            is_async: Whether handler is async
        """
        if is_async:
            self._async_handlers[action].append(handler)
        else:
            self._handlers[action].append(handler)

    def unregister(self, action: str, handler: Callable) -> None:
        """Unregister a handler."""
        if handler in self._handlers[action]:
            self._handlers[action].remove(handler)
        if handler in self._async_handlers[action]:
            self._async_handlers[action].remove(handler)

    async def dispatch(self, message: AgentMessage) -> List[Any]:
        """
        Dispatch message to registered handlers.

        Returns:
            List of handler results
        """
        results = []
        action = message.action

        # Call sync handlers
        for handler in self._handlers.get(action, []):
            try:
                result = handler(message)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler error for {action}: {e}")

        # Call async handlers
        for handler in self._async_handlers.get(action, []):
            try:
                result = await handler(message)
                results.append(result)
            except Exception as e:
                logger.error(f"Async handler error for {action}: {e}")

        return results


# =============================================================================
# Load Balancer
# =============================================================================


class LoadBalancer:
    """Load balancer for task distribution."""

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ) -> None:
        """Initialize load balancer."""
        self._strategy = strategy
        self._round_robin_index = 0
        self._agent_loads: Dict[AgentID, float] = {}
        self._agent_weights: Dict[AgentID, float] = {}

    def set_agent_load(self, agent_id: AgentID, load: float) -> None:
        """Update agent load."""
        self._agent_loads[agent_id] = load

    def set_agent_weight(self, agent_id: AgentID, weight: float) -> None:
        """Set agent weight for weighted balancing."""
        self._agent_weights[agent_id] = weight

    def select_agent(
        self,
        available_agents: List[AgentID],
        task: Optional[DistributedTask] = None,
    ) -> Optional[AgentID]:
        """
        Select agent for task assignment.

        Args:
            available_agents: List of available agents
            task: Optional task for hash-based routing

        Returns:
            Selected agent ID
        """
        if not available_agents:
            return None

        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_agents)
        elif self._strategy == LoadBalanceStrategy.LEAST_LOADED:
            return self._least_loaded_select(available_agents)
        elif self._strategy == LoadBalanceStrategy.RANDOM:
            return self._random_select(available_agents)
        elif self._strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted_select(available_agents)
        elif self._strategy == LoadBalanceStrategy.HASH_BASED:
            return self._hash_based_select(available_agents, task)

        return available_agents[0]

    def _round_robin_select(self, agents: List[AgentID]) -> AgentID:
        """Round robin selection."""
        agent = agents[self._round_robin_index % len(agents)]
        self._round_robin_index += 1
        return agent

    def _least_loaded_select(self, agents: List[AgentID]) -> AgentID:
        """Select least loaded agent."""
        return min(agents, key=lambda a: self._agent_loads.get(a, 0))

    def _random_select(self, agents: List[AgentID]) -> AgentID:
        """Random selection."""
        import random
        return random.choice(agents)

    def _weighted_select(self, agents: List[AgentID]) -> AgentID:
        """Weighted random selection."""
        import random
        weights = [self._agent_weights.get(a, 1.0) for a in agents]
        return random.choices(agents, weights=weights, k=1)[0]

    def _hash_based_select(
        self,
        agents: List[AgentID],
        task: Optional[DistributedTask],
    ) -> AgentID:
        """Hash-based selection for consistent routing."""
        if task:
            hash_key = task.task_id
        else:
            hash_key = str(uuid.uuid4())

        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        return agents[hash_value % len(agents)]


# =============================================================================
# Response Aggregator
# =============================================================================


class ResponseAggregator:
    """Aggregates responses from multiple agents."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._pending_requests: Dict[str, Dict[str, Any]] = {}

    def create_aggregation(
        self,
        correlation_id: str,
        expected_agents: List[AgentID],
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Create response aggregation for a request.

        Args:
            correlation_id: Request correlation ID
            expected_agents: Agents expected to respond
            timeout_seconds: Aggregation timeout
        """
        self._pending_requests[correlation_id] = {
            "expected": set(expected_agents),
            "received": {},
            "created_at": time.time(),
            "timeout": timeout_seconds,
            "event": asyncio.Event(),
        }

    def add_response(
        self,
        correlation_id: str,
        response: AgentResponse,
    ) -> bool:
        """
        Add response to aggregation.

        Returns:
            True if all responses received
        """
        if correlation_id not in self._pending_requests:
            return False

        agg = self._pending_requests[correlation_id]
        agg["received"][response.source_agent] = response

        if agg["expected"] <= set(agg["received"].keys()):
            agg["event"].set()
            return True

        return False

    async def wait_for_responses(
        self,
        correlation_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[AgentID, AgentResponse]:
        """
        Wait for all responses or timeout.

        Returns:
            Dictionary of agent_id -> response
        """
        if correlation_id not in self._pending_requests:
            return {}

        agg = self._pending_requests[correlation_id]
        timeout = timeout or agg["timeout"]

        try:
            await asyncio.wait_for(agg["event"].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Response aggregation timeout for {correlation_id}")

        responses = agg["received"]
        del self._pending_requests[correlation_id]
        return responses

    def get_partial_responses(
        self,
        correlation_id: str,
    ) -> Dict[AgentID, AgentResponse]:
        """Get responses received so far."""
        if correlation_id not in self._pending_requests:
            return {}
        return self._pending_requests[correlation_id]["received"]


# =============================================================================
# Agent Coordinator Implementation
# =============================================================================


class AgentCoordinator(BaseConnector):
    """
    Multi-Agent Coordinator for GL-013 PREDICTMAINT.

    Provides coordination with other GreenLang agents:
    - GL-001 to GL-012 agent integration
    - Message bus integration (pub/sub)
    - Task distribution and load balancing
    - Response aggregation
    - Consensus mechanisms
    - Timeout handling
    - Priority-based routing
    """

    def __init__(self, config: AgentCoordinatorConfig) -> None:
        """
        Initialize agent coordinator.

        Args:
            config: Coordinator configuration
        """
        super().__init__(config)
        self._coord_config = config

        # Message bus client
        self._bus_client = None
        self._bus_pubsub = None

        # Components
        self._handler_registry = MessageHandlerRegistry()
        self._load_balancer = LoadBalancer(config.load_balance_strategy)
        self._response_aggregator = ResponseAggregator()

        # State tracking
        self._known_agents: Dict[AgentID, AgentStatus] = {}
        self._pending_tasks: Dict[str, DistributedTask] = {}
        self._active_proposals: Dict[str, ConsensusProposal] = {}

        # Task queue
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=config.task_queue_size
        )
        self._running_tasks: Set[str] = set()

        # Background tasks
        self._message_listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._task_processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Timing
        self._start_time = time.time()

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self._handler_registry.register("heartbeat", self._handle_heartbeat)
        self._handler_registry.register("discovery", self._handle_discovery)
        self._handler_registry.register("status_request", self._handle_status_request)
        self._handler_registry.register("consensus_propose", self._handle_consensus_propose)
        self._handler_registry.register("consensus_vote", self._handle_consensus_vote)
        self._handler_registry.register("consensus_commit", self._handle_consensus_commit)
        self._handler_registry.register("task_assign", self._handle_task_assign)
        self._handler_registry.register("task_result", self._handle_task_result)

    async def connect(self) -> None:
        """Establish connection to message bus."""
        self._logger.info(
            f"Connecting agent coordinator for {self._coord_config.agent_id.value}..."
        )

        bus_config = self._coord_config.message_bus_config

        try:
            import redis.asyncio as redis

            self._bus_client = redis.Redis(
                host=bus_config.host,
                port=bus_config.port,
                password=bus_config.password,
                db=bus_config.database,
                socket_timeout=bus_config.connection_timeout_seconds,
            )

            # Test connection
            await self._bus_client.ping()

            # Subscribe to channels
            self._bus_pubsub = self._bus_client.pubsub()
            await self._subscribe_to_channels()

            # Start background tasks
            self._message_listener_task = asyncio.create_task(self._message_listener_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._task_processor_task = asyncio.create_task(self._task_processor_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Announce presence
            await self._announce_agent()

            self._logger.info("Agent coordinator connected and ready")

        except ImportError:
            raise ConfigurationError("redis package not installed. Install with: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to message bus: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from message bus."""
        # Stop background tasks
        for task in [
            self._message_listener_task,
            self._heartbeat_task,
            self._task_processor_task,
            self._cleanup_task,
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Unsubscribe and disconnect
        if self._bus_pubsub:
            await self._bus_pubsub.unsubscribe()
            await self._bus_pubsub.close()

        if self._bus_client:
            await self._bus_client.close()

        self._logger.info("Agent coordinator disconnected")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on coordinator."""
        start_time = time.time()

        try:
            if not self._bus_client:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Message bus client not initialized",
                    latency_ms=0.0,
                )

            # Ping message bus
            await self._bus_client.ping()
            latency_ms = (time.time() - start_time) * 1000

            # Check agent connectivity
            online_agents = sum(
                1 for a in self._known_agents.values()
                if a.status == "online"
            )
            total_agents = len(self._known_agents)

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message=f"Coordinator healthy. {online_agents}/{total_agents} agents online.",
                latency_ms=latency_ms,
                details={
                    "agent_id": self._coord_config.agent_id.value,
                    "online_agents": online_agents,
                    "total_known_agents": total_agents,
                    "pending_tasks": len(self._pending_tasks),
                    "running_tasks": len(self._running_tasks),
                    "queue_size": self._task_queue.qsize(),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def validate_configuration(self) -> bool:
        """Validate coordinator configuration."""
        if not self._coord_config.message_bus_config:
            raise ConfigurationError("Message bus configuration required")
        return True

    async def _subscribe_to_channels(self) -> None:
        """Subscribe to message bus channels."""
        bus_config = self._coord_config.message_bus_config
        prefix = bus_config.channel_prefix

        channels = [
            f"{prefix}{bus_config.broadcast_channel}",
            f"{prefix}{self._coord_config.agent_id.value}:*",
        ]

        for channel in channels:
            await self._bus_pubsub.psubscribe(channel)
            self._logger.debug(f"Subscribed to channel pattern: {channel}")

    async def _announce_agent(self) -> None:
        """Announce agent presence to the network."""
        status = self._get_own_status()

        message = AgentMessage(
            message_type=MessageType.EVENT,
            source_agent=self._coord_config.agent_id,
            action="agent_online",
            payload=status.model_dump(),
        )

        await self._publish_broadcast(message)

    # =========================================================================
    # Message Publishing
    # =========================================================================

    async def _publish_message(
        self,
        channel: str,
        message: AgentMessage,
    ) -> None:
        """Publish message to channel."""
        message_json = json.dumps(message.model_dump(), default=str)
        await self._bus_client.publish(channel, message_json)

    async def _publish_broadcast(self, message: AgentMessage) -> None:
        """Publish to broadcast channel."""
        prefix = self._coord_config.message_bus_config.channel_prefix
        channel = f"{prefix}{self._coord_config.message_bus_config.broadcast_channel}"
        await self._publish_message(channel, message)

    async def _publish_to_agent(
        self,
        target_agent: AgentID,
        message: AgentMessage,
    ) -> None:
        """Publish message to specific agent."""
        prefix = self._coord_config.message_bus_config.channel_prefix
        channel = f"{prefix}{target_agent.value}:messages"
        await self._publish_message(channel, message)

    # =========================================================================
    # Message Handling
    # =========================================================================

    async def _message_listener_loop(self) -> None:
        """Background loop for listening to messages."""
        try:
            async for message in self._bus_pubsub.listen():
                if message["type"] not in ["message", "pmessage"]:
                    continue

                try:
                    data = json.loads(message["data"])
                    agent_message = AgentMessage(**data)

                    # Don't process own messages
                    if agent_message.source_agent == self._coord_config.agent_id:
                        continue

                    await self._process_message(agent_message)

                except json.JSONDecodeError:
                    logger.warning("Failed to parse message")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Message listener error: {e}")

    async def _process_message(self, message: AgentMessage) -> None:
        """Process incoming message."""
        # Check if message is expired
        if message.expires_at and datetime.utcnow() > message.expires_at:
            logger.debug(f"Ignoring expired message {message.message_id}")
            return

        # Handle response aggregation
        if message.message_type == MessageType.RESPONSE and message.correlation_id:
            response = AgentResponse(
                correlation_id=message.correlation_id,
                source_agent=message.source_agent,
                target_agent=message.target_agent or self._coord_config.agent_id,
                success=message.payload.get("success", True),
                result=message.payload.get("result"),
                error=message.payload.get("error"),
            )
            self._response_aggregator.add_response(message.correlation_id, response)
            return

        # Dispatch to handlers
        await self._handler_registry.dispatch(message)

    # =========================================================================
    # Default Message Handlers
    # =========================================================================

    async def _handle_heartbeat(self, message: AgentMessage) -> None:
        """Handle heartbeat message."""
        agent_id = message.source_agent
        status_data = message.payload

        if agent_id not in self._known_agents:
            logger.info(f"Discovered agent: {agent_id.value}")

        self._known_agents[agent_id] = AgentStatus(
            agent_id=agent_id,
            agent_name=status_data.get("agent_name", agent_id.value),
            agent_version=status_data.get("agent_version", "unknown"),
            status="online",
            last_heartbeat=datetime.utcnow(),
            capabilities=status_data.get("capabilities", []),
            current_tasks=status_data.get("current_tasks", 0),
            max_tasks=status_data.get("max_tasks", 10),
            load_percentage=status_data.get("load_percentage", 0),
        )

        # Update load balancer
        self._load_balancer.set_agent_load(
            agent_id,
            status_data.get("load_percentage", 0)
        )

    async def _handle_discovery(self, message: AgentMessage) -> None:
        """Handle agent discovery request."""
        # Respond with our status
        response = AgentMessage(
            message_type=MessageType.RESPONSE,
            source_agent=self._coord_config.agent_id,
            target_agent=message.source_agent,
            correlation_id=message.message_id,
            action="discovery_response",
            payload=self._get_own_status().model_dump(),
        )
        await self._publish_to_agent(message.source_agent, response)

    async def _handle_status_request(self, message: AgentMessage) -> None:
        """Handle status request."""
        status = self._get_own_status()

        response = AgentMessage(
            message_type=MessageType.RESPONSE,
            source_agent=self._coord_config.agent_id,
            target_agent=message.source_agent,
            correlation_id=message.message_id,
            action="status_response",
            payload={"success": True, "result": status.model_dump()},
        )
        await self._publish_to_agent(message.source_agent, response)

    async def _handle_consensus_propose(self, message: AgentMessage) -> None:
        """Handle consensus proposal."""
        proposal = ConsensusProposal(**message.payload)

        if self._coord_config.agent_id not in proposal.voters:
            return

        # Auto-vote (in real implementation, this would involve decision logic)
        vote = True  # For now, always vote yes

        response = AgentMessage(
            message_type=MessageType.CONSENSUS_VOTE,
            source_agent=self._coord_config.agent_id,
            target_agent=message.source_agent,
            correlation_id=proposal.proposal_id,
            action="consensus_vote",
            payload={
                "proposal_id": proposal.proposal_id,
                "vote": vote,
                "voter": self._coord_config.agent_id.value,
            },
        )
        await self._publish_to_agent(message.source_agent, response)

    async def _handle_consensus_vote(self, message: AgentMessage) -> None:
        """Handle consensus vote."""
        proposal_id = message.payload.get("proposal_id")
        vote = message.payload.get("vote")
        voter = message.payload.get("voter")

        if proposal_id in self._active_proposals:
            proposal = self._active_proposals[proposal_id]
            proposal.votes[voter] = vote

            # Check if quorum reached
            if len(proposal.votes) >= len(proposal.voters) * proposal.quorum_required:
                await self._finalize_consensus(proposal)

    async def _handle_consensus_commit(self, message: AgentMessage) -> None:
        """Handle consensus commit notification."""
        proposal_id = message.payload.get("proposal_id")
        approved = message.payload.get("approved")
        logger.info(
            f"Consensus {proposal_id} {'approved' if approved else 'rejected'}"
        )

    async def _handle_task_assign(self, message: AgentMessage) -> None:
        """Handle task assignment."""
        task = DistributedTask(**message.payload)

        # Check if we have capacity
        if len(self._running_tasks) >= self._coord_config.max_concurrent_tasks:
            # Send rejection
            response = AgentMessage(
                message_type=MessageType.RESPONSE,
                source_agent=self._coord_config.agent_id,
                target_agent=message.source_agent,
                correlation_id=message.message_id,
                action="task_rejected",
                payload={
                    "success": False,
                    "error": "Agent at capacity",
                    "task_id": task.task_id,
                },
            )
            await self._publish_to_agent(message.source_agent, response)
            return

        # Accept task
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = self._coord_config.agent_id

        self._pending_tasks[task.task_id] = task

        # Queue for processing
        priority = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3,
            MessagePriority.BULK: 4,
        }.get(task.priority, 2)

        await self._task_queue.put((priority, task.task_id))

        # Send acceptance
        response = AgentMessage(
            message_type=MessageType.RESPONSE,
            source_agent=self._coord_config.agent_id,
            target_agent=message.source_agent,
            correlation_id=message.message_id,
            action="task_accepted",
            payload={"success": True, "task_id": task.task_id},
        )
        await self._publish_to_agent(message.source_agent, response)

    async def _handle_task_result(self, message: AgentMessage) -> None:
        """Handle task result notification."""
        task_id = message.payload.get("task_id")
        success = message.payload.get("success")
        result = message.payload.get("result")
        error = message.payload.get("error")

        if task_id in self._pending_tasks:
            task = self._pending_tasks[task_id]
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.result = result
            task.error = error
            task.completed_at = datetime.utcnow()

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while True:
            try:
                await asyncio.sleep(self._coord_config.heartbeat_interval_seconds)

                message = AgentMessage(
                    message_type=MessageType.HEARTBEAT,
                    source_agent=self._coord_config.agent_id,
                    action="heartbeat",
                    payload=self._get_own_status().model_dump(),
                )
                await self._publish_broadcast(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _task_processor_loop(self) -> None:
        """Process tasks from queue."""
        while True:
            try:
                # Get task from queue
                priority, task_id = await self._task_queue.get()

                if task_id not in self._pending_tasks:
                    continue

                task = self._pending_tasks[task_id]

                # Execute task
                self._running_tasks.add(task_id)
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()

                try:
                    result = await self._execute_task(task)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                task.completed_at = datetime.utcnow()
                self._running_tasks.discard(task_id)

                # Send result to owner
                if task.owner_agent != self._coord_config.agent_id:
                    result_message = AgentMessage(
                        message_type=MessageType.EVENT,
                        source_agent=self._coord_config.agent_id,
                        target_agent=task.owner_agent,
                        action="task_result",
                        payload={
                            "task_id": task.task_id,
                            "success": task.status == TaskStatus.COMPLETED,
                            "result": task.result,
                            "error": task.error,
                        },
                    )
                    await self._publish_to_agent(task.owner_agent, result_message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processor error: {e}")

    async def _execute_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute a distributed task. Override in subclass for actual implementation."""
        # Default implementation - should be overridden
        logger.info(f"Executing task {task.task_id}: {task.task_type}")
        return {"status": "completed", "message": "Task executed by default handler"}

    async def _cleanup_loop(self) -> None:
        """Cleanup expired agents and tasks."""
        cleanup_interval = 60.0

        while True:
            try:
                await asyncio.sleep(cleanup_interval)

                now = datetime.utcnow()
                timeout = timedelta(seconds=self._coord_config.agent_timeout_seconds)

                # Check for offline agents
                for agent_id, status in list(self._known_agents.items()):
                    if now - status.last_heartbeat > timeout:
                        if status.status != "offline":
                            logger.warning(f"Agent {agent_id.value} went offline")
                            status.status = "offline"

                # Check for expired proposals
                for proposal_id, proposal in list(self._active_proposals.items()):
                    if now > proposal.expires_at and proposal.state not in [
                        ConsensusState.COMMITTED,
                        ConsensusState.ABORTED,
                    ]:
                        proposal.state = ConsensusState.ABORTED
                        proposal.result_message = "Timeout"
                        logger.warning(f"Consensus {proposal_id} expired")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    # =========================================================================
    # Public API - Messaging
    # =========================================================================

    async def send_request(
        self,
        target_agent: AgentID,
        action: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> AgentResponse:
        """
        Send request to specific agent and wait for response.

        Args:
            target_agent: Target agent ID
            action: Action name
            payload: Request payload
            timeout: Response timeout
            priority: Message priority

        Returns:
            Agent response
        """
        timeout = timeout or self._coord_config.request_timeout_seconds

        message = AgentMessage(
            message_type=MessageType.REQUEST,
            source_agent=self._coord_config.agent_id,
            target_agent=target_agent,
            action=action,
            payload=payload,
            priority=priority,
            expires_at=datetime.utcnow() + timedelta(seconds=timeout),
        )

        # Setup response aggregation
        self._response_aggregator.create_aggregation(
            message.message_id,
            [target_agent],
            timeout,
        )

        # Send message
        await self._publish_to_agent(target_agent, message)

        # Wait for response
        responses = await self._response_aggregator.wait_for_responses(
            message.message_id,
            timeout,
        )

        if target_agent in responses:
            return responses[target_agent]

        return AgentResponse(
            correlation_id=message.message_id,
            source_agent=target_agent,
            target_agent=self._coord_config.agent_id,
            success=False,
            error="Request timeout",
        )

    async def broadcast_request(
        self,
        action: str,
        payload: Dict[str, Any],
        target_agents: Optional[List[AgentID]] = None,
        timeout: Optional[float] = None,
        require_all: bool = False,
    ) -> Dict[AgentID, AgentResponse]:
        """
        Broadcast request to multiple agents.

        Args:
            action: Action name
            payload: Request payload
            target_agents: Target agents (all known if not specified)
            timeout: Response timeout
            require_all: Wait for all responses

        Returns:
            Dictionary of agent_id -> response
        """
        timeout = timeout or self._coord_config.request_timeout_seconds

        if target_agents is None:
            target_agents = [
                aid for aid, status in self._known_agents.items()
                if status.status == "online"
            ]

        if not target_agents:
            return {}

        message = AgentMessage(
            message_type=MessageType.REQUEST,
            source_agent=self._coord_config.agent_id,
            target_agents=target_agents,
            action=action,
            payload=payload,
            expires_at=datetime.utcnow() + timedelta(seconds=timeout),
        )

        # Setup aggregation
        self._response_aggregator.create_aggregation(
            message.message_id,
            target_agents,
            timeout,
        )

        # Send to all targets
        for target in target_agents:
            message.target_agent = target
            await self._publish_to_agent(target, message)

        # Wait for responses
        return await self._response_aggregator.wait_for_responses(
            message.message_id,
            timeout,
        )

    async def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target_agent: Optional[AgentID] = None,
    ) -> None:
        """
        Publish event to the network.

        Args:
            event_type: Event type/name
            payload: Event payload
            target_agent: Optional specific target
        """
        message = AgentMessage(
            message_type=MessageType.EVENT,
            source_agent=self._coord_config.agent_id,
            target_agent=target_agent,
            action=event_type,
            payload=payload,
        )

        if target_agent:
            await self._publish_to_agent(target_agent, message)
        else:
            await self._publish_broadcast(message)

    # =========================================================================
    # Public API - Task Distribution
    # =========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout_seconds: float = 60.0,
    ) -> DistributedTask:
        """
        Submit task for distributed execution.

        Args:
            task_type: Task type
            payload: Task payload
            required_capabilities: Required agent capabilities
            priority: Task priority
            timeout_seconds: Task timeout

        Returns:
            Created task
        """
        # Find capable agents
        capable_agents = self._find_capable_agents(required_capabilities or [])

        if not capable_agents:
            raise ValidationError(
                f"No agents available with required capabilities: {required_capabilities}"
            )

        # Select agent
        task = DistributedTask(
            task_type=task_type,
            owner_agent=self._coord_config.agent_id,
            required_capabilities=required_capabilities or [],
            priority=priority,
            payload=payload,
            timeout_seconds=timeout_seconds,
            deadline=datetime.utcnow() + timedelta(seconds=timeout_seconds),
        )

        selected_agent = self._load_balancer.select_agent(capable_agents, task)
        task.assigned_agent = selected_agent

        self._pending_tasks[task.task_id] = task

        # Send task assignment
        message = AgentMessage(
            message_type=MessageType.COMMAND,
            source_agent=self._coord_config.agent_id,
            target_agent=selected_agent,
            action="task_assign",
            payload=task.model_dump(),
            priority=priority,
        )
        await self._publish_to_agent(selected_agent, message)

        return task

    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get status of a task."""
        return self._pending_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if task_id not in self._pending_tasks:
            return False

        task = self._pending_tasks[task_id]

        if task.status not in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.ASSIGNED]:
            return False

        task.status = TaskStatus.CANCELLED

        # Notify assigned agent
        if task.assigned_agent:
            message = AgentMessage(
                message_type=MessageType.COMMAND,
                source_agent=self._coord_config.agent_id,
                target_agent=task.assigned_agent,
                action="task_cancel",
                payload={"task_id": task_id},
            )
            await self._publish_to_agent(task.assigned_agent, message)

        return True

    # =========================================================================
    # Public API - Consensus
    # =========================================================================

    async def propose_consensus(
        self,
        proposal_type: str,
        description: str,
        proposal_data: Dict[str, Any],
        voters: Optional[List[AgentID]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ConsensusProposal:
        """
        Initiate consensus proposal.

        Args:
            proposal_type: Type of proposal
            description: Proposal description
            proposal_data: Proposal content
            voters: List of voting agents
            timeout_seconds: Voting timeout

        Returns:
            Consensus proposal
        """
        if not self._coord_config.consensus_enabled:
            raise ConfigurationError("Consensus protocol disabled")

        timeout = timeout_seconds or self._coord_config.consensus_timeout_seconds

        if voters is None:
            voters = [
                aid for aid, status in self._known_agents.items()
                if status.status == "online"
            ]

        proposal = ConsensusProposal(
            proposer=self._coord_config.agent_id,
            proposal_type=proposal_type,
            description=description,
            proposal_data=proposal_data,
            voters=voters,
            quorum_required=self._coord_config.consensus_quorum_percentage,
            state=ConsensusState.PROPOSING,
            expires_at=datetime.utcnow() + timedelta(seconds=timeout),
        )

        self._active_proposals[proposal.proposal_id] = proposal

        # Send proposal to voters
        message = AgentMessage(
            message_type=MessageType.CONSENSUS_PROPOSE,
            source_agent=self._coord_config.agent_id,
            action="consensus_propose",
            payload=proposal.model_dump(),
        )

        for voter in voters:
            await self._publish_to_agent(voter, message)

        proposal.state = ConsensusState.VOTING

        return proposal

    async def _finalize_consensus(self, proposal: ConsensusProposal) -> None:
        """Finalize consensus decision."""
        yes_votes = sum(1 for v in proposal.votes.values() if v)
        total_votes = len(proposal.votes)

        approved = (yes_votes / total_votes) >= proposal.quorum_required if total_votes > 0 else False

        proposal.state = ConsensusState.COMMITTING
        proposal.approved = approved
        proposal.decided_at = datetime.utcnow()
        proposal.result_message = f"Approved {yes_votes}/{total_votes}" if approved else f"Rejected {yes_votes}/{total_votes}"

        # Notify all voters
        commit_message = AgentMessage(
            message_type=MessageType.CONSENSUS_COMMIT,
            source_agent=self._coord_config.agent_id,
            action="consensus_commit",
            payload={
                "proposal_id": proposal.proposal_id,
                "approved": approved,
                "votes": proposal.votes,
                "result": proposal.result_message,
            },
        )

        for voter in proposal.voters:
            await self._publish_to_agent(voter, commit_message)

        proposal.state = ConsensusState.COMMITTED

    async def get_proposal_status(self, proposal_id: str) -> Optional[ConsensusProposal]:
        """Get status of a consensus proposal."""
        return self._active_proposals.get(proposal_id)

    # =========================================================================
    # Public API - Agent Discovery
    # =========================================================================

    async def discover_agents(self) -> List[AgentStatus]:
        """
        Discover all online agents.

        Returns:
            List of agent statuses
        """
        message = AgentMessage(
            message_type=MessageType.DISCOVERY,
            source_agent=self._coord_config.agent_id,
            action="discovery",
            payload={"request_status": True},
        )
        await self._publish_broadcast(message)

        # Wait briefly for responses
        await asyncio.sleep(2.0)

        return list(self._known_agents.values())

    def get_known_agents(self) -> Dict[AgentID, AgentStatus]:
        """Get all known agents."""
        return self._known_agents.copy()

    def get_agent_status(self, agent_id: AgentID) -> Optional[AgentStatus]:
        """Get status of specific agent."""
        return self._known_agents.get(agent_id)

    def get_online_agents(self) -> List[AgentID]:
        """Get list of online agent IDs."""
        return [
            aid for aid, status in self._known_agents.items()
            if status.status == "online"
        ]

    # =========================================================================
    # Handler Registration
    # =========================================================================

    def register_handler(
        self,
        action: str,
        handler: Callable[[AgentMessage], Any],
        is_async: bool = True,
    ) -> None:
        """
        Register message handler.

        Args:
            action: Action name to handle
            handler: Handler function
            is_async: Whether handler is async
        """
        self._handler_registry.register(action, handler, is_async)

    def unregister_handler(self, action: str, handler: Callable) -> None:
        """Unregister message handler."""
        self._handler_registry.unregister(action, handler)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_own_status(self) -> AgentStatus:
        """Get this agent's status."""
        uptime = int(time.time() - self._start_time)
        current_tasks = len(self._running_tasks)
        max_tasks = self._coord_config.max_concurrent_tasks
        load = (current_tasks / max_tasks * 100) if max_tasks > 0 else 0

        return AgentStatus(
            agent_id=self._coord_config.agent_id,
            agent_name=self._coord_config.agent_name,
            agent_version=self._coord_config.agent_version,
            status="online",
            uptime_seconds=uptime,
            capabilities=self._coord_config.capabilities,
            current_tasks=current_tasks,
            max_tasks=max_tasks,
            queue_size=self._task_queue.qsize(),
            load_percentage=load,
            health_status=HealthStatus.HEALTHY,
        )

    def _find_capable_agents(self, required_capabilities: List[str]) -> List[AgentID]:
        """Find agents with required capabilities."""
        capable = []
        required_set = set(required_capabilities)

        for agent_id, status in self._known_agents.items():
            if status.status != "online":
                continue

            agent_caps = set(status.capabilities)
            if required_set <= agent_caps:
                capable.append(agent_id)

        return capable


# =============================================================================
# Factory Function
# =============================================================================


def create_agent_coordinator(
    agent_id: AgentID = AgentID.GL_013_PREDICTMAINT,
    agent_name: str = "PREDICTMAINT",
    message_bus_config: Optional[MessageBusConfig] = None,
    **kwargs,
) -> AgentCoordinator:
    """
    Factory function to create agent coordinator.

    Args:
        agent_id: This agent's ID
        agent_name: Agent name
        message_bus_config: Message bus configuration
        **kwargs: Additional configuration options

    Returns:
        Configured agent coordinator
    """
    config = AgentCoordinatorConfig(
        connector_name=f"{agent_id.value}-coordinator",
        connector_type=ConnectorType.AGENT_COORDINATOR,
        agent_id=agent_id,
        agent_name=agent_name,
        message_bus_config=message_bus_config or MessageBusConfig(),
        **kwargs,
    )

    return AgentCoordinator(config)
