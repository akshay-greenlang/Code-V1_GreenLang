"""
Agent Coordinator Module for GL-014 EXCHANGER-PRO (Heat Exchanger Optimization Agent).

Provides multi-agent coordination capabilities for integration with other GreenLang
agents, specifically:
- GL-001 THERMOSYNC (Thermal efficiency data)
- GL-006 HEATRECLAIM (Heat recovery opportunities)
- GL-013 PREDICTMAINT (Maintenance coordination)

Implements message bus integration via gRPC/REST, async request/response patterns,
data sharing protocols, and collaborative optimization.

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

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    AuthenticationType,
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
    """GreenLang agent identifiers relevant to heat exchanger optimization."""

    GL_001_THERMOSYNC = "GL-001"  # Thermal Optimization
    GL_002_GRIDBALANCE = "GL-002"  # Grid Balancing
    GL_003_CARBONTRACK = "GL-003"  # Carbon Tracking
    GL_004_WATERWISE = "GL-004"  # Water Management
    GL_005_WASTEWATCH = "GL-005"  # Waste Management
    GL_006_HEATRECLAIM = "GL-006"  # Heat Recovery (key partner)
    GL_007_ENERGYOPT = "GL-007"  # Energy Optimization
    GL_008_SUPPLYCHAIN = "GL-008"  # Supply Chain
    GL_009_COMPLIANCE = "GL-009"  # Compliance
    GL_010_CARBONSCOPE = "GL-010"  # Emissions Compliance
    GL_011_SAFETYGUARD = "GL-011"  # Safety Monitoring
    GL_012_ASSETTRACK = "GL-012"  # Asset Tracking
    GL_013_PREDICTMAINT = "GL-013"  # Predictive Maintenance (key partner)
    GL_014_EXCHANGER_PRO = "GL-014"  # Heat Exchanger Optimization (self)


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"
    COMMAND = "command"
    QUERY = "query"
    DATA_SHARE = "data_share"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    SUBSCRIPTION = "subscription"
    UNSUBSCRIPTION = "unsubscription"


class MessagePriority(str, Enum):
    """Message priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BULK = "bulk"


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


class RoutingStrategy(str, Enum):
    """Message routing strategies."""

    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    ROUND_ROBIN = "round_robin"


class CommunicationProtocol(str, Enum):
    """Communication protocols for agent messaging."""

    REST = "rest"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    WEBSOCKET = "websocket"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class AgentEndpointConfig(BaseModel):
    """Configuration for a single agent endpoint."""

    model_config = ConfigDict(extra="forbid")

    agent_id: AgentID = Field(..., description="Agent identifier")
    protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.REST,
        description="Communication protocol"
    )

    # REST/HTTP configuration
    base_url: Optional[str] = Field(default=None, description="Base URL for REST API")
    api_version: str = Field(default="v1", description="API version")

    # gRPC configuration
    grpc_host: Optional[str] = Field(default=None, description="gRPC host")
    grpc_port: Optional[int] = Field(default=None, ge=1, le=65535, description="gRPC port")

    # Message queue configuration
    queue_name: Optional[str] = Field(default=None, description="Message queue name")
    exchange_name: Optional[str] = Field(default=None, description="Exchange name")

    # Authentication
    api_key: Optional[str] = Field(default=None, description="API key for this agent")
    use_mtls: bool = Field(default=False, description="Use mutual TLS")

    # Timeouts
    timeout_seconds: float = Field(default=30.0, ge=1.0, description="Request timeout")


class MessageBusConfig(BaseModel):
    """Message bus configuration."""

    model_config = ConfigDict(extra="forbid")

    bus_type: str = Field(
        default="redis",
        description="Message bus type (redis, rabbitmq, kafka)"
    )
    host: str = Field(default="localhost", description="Bus host")
    port: int = Field(default=6379, ge=1, le=65535, description="Bus port")
    password: Optional[str] = Field(default=None, description="Bus password")
    database: int = Field(default=0, ge=0, description="Database number")

    # Channel configuration
    channel_prefix: str = Field(default="greenlang:", description="Channel prefix")
    request_channel: str = Field(default="requests", description="Request channel")
    response_channel: str = Field(default="responses", description="Response channel")
    event_channel: str = Field(default="events", description="Event channel")

    # Settings
    message_ttl_seconds: int = Field(default=3600, ge=60, description="Message TTL")
    max_message_size_bytes: int = Field(default=1048576, description="Max message size")


class AgentCoordinatorConfig(BaseConnectorConfig):
    """Configuration for agent coordinator."""

    model_config = ConfigDict(extra="forbid")

    # Self identity
    agent_id: AgentID = Field(
        default=AgentID.GL_014_EXCHANGER_PRO,
        description="This agent's ID"
    )
    agent_name: str = Field(default="EXCHANGER-PRO", description="Agent name")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Partner agent configurations
    agent_endpoints: Dict[str, AgentEndpointConfig] = Field(
        default_factory=dict,
        description="Partner agent endpoint configurations"
    )

    # Default communication
    default_protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.REST,
        description="Default communication protocol"
    )

    # Message bus (optional)
    message_bus_config: Optional[MessageBusConfig] = Field(
        default=None,
        description="Message bus configuration"
    )

    # Routing
    default_routing_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.DIRECT,
        description="Default routing strategy"
    )

    # Timeouts
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout"
    )
    heartbeat_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Heartbeat interval"
    )
    agent_timeout_seconds: float = Field(
        default=90.0,
        ge=30.0,
        le=600.0,
        description="Agent offline threshold"
    )

    # Capabilities
    capabilities: List[str] = Field(
        default_factory=lambda: [
            "heat_exchanger_optimization",
            "fouling_prediction",
            "cleaning_scheduling",
            "thermal_performance",
            "ua_calculation",
        ],
        description="Agent capabilities"
    )

    # Data sharing
    enable_data_sharing: bool = Field(default=True, description="Enable data sharing")
    shared_data_topics: List[str] = Field(
        default_factory=lambda: [
            "heat_exchanger.performance",
            "heat_exchanger.fouling",
            "heat_exchanger.cleaning_schedule",
            "heat_exchanger.alerts",
        ],
        description="Topics this agent shares"
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

    # Tracing
    trace_id: Optional[str] = Field(default=None, description="Distributed trace ID")
    span_id: Optional[str] = Field(default=None, description="Span ID")

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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(default=None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentStatus(BaseModel):
    """Agent status information."""

    model_config = ConfigDict(extra="allow")

    agent_id: AgentID = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    agent_version: str = Field(..., description="Agent version")

    # Status
    status: str = Field(default="online", description="Agent status")
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: int = Field(default=0, ge=0)

    # Capabilities
    capabilities: List[str] = Field(default_factory=list)

    # Health
    health_status: HealthStatus = Field(default=HealthStatus.HEALTHY)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Heat Exchanger Specific Data Sharing Models
# =============================================================================


class HeatExchangerPerformanceData(BaseModel):
    """Heat exchanger performance data for sharing with other agents."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Performance metrics
    current_duty_kw: Optional[float] = Field(default=None, description="Current heat duty")
    design_duty_kw: Optional[float] = Field(default=None, description="Design duty")
    duty_ratio: Optional[float] = Field(default=None, description="Duty/design ratio")

    current_ua_w_k: Optional[float] = Field(default=None, description="Current UA")
    design_ua_w_k: Optional[float] = Field(default=None, description="Design UA")
    ua_ratio: Optional[float] = Field(default=None, description="UA/design ratio")

    effectiveness: Optional[float] = Field(default=None, description="Effectiveness")
    lmtd: Optional[float] = Field(default=None, description="Log mean temp difference")

    # Fouling
    fouling_factor: Optional[float] = Field(default=None, description="Fouling factor")
    fouling_trend: Optional[str] = Field(
        default=None,
        description="Trend: increasing, stable, decreasing"
    )
    estimated_days_to_cleaning: Optional[int] = Field(default=None)

    # Energy impact
    energy_loss_kw: Optional[float] = Field(default=None, description="Energy loss due to fouling")
    annual_energy_loss_kwh: Optional[float] = Field(default=None)
    cost_impact_per_day: Optional[float] = Field(default=None)

    # Temperatures
    hot_inlet_temp: Optional[float] = Field(default=None)
    hot_outlet_temp: Optional[float] = Field(default=None)
    cold_inlet_temp: Optional[float] = Field(default=None)
    cold_outlet_temp: Optional[float] = Field(default=None)


class HeatRecoveryOpportunity(BaseModel):
    """Heat recovery opportunity from GL-006 HEATRECLAIM."""

    model_config = ConfigDict(extra="allow")

    opportunity_id: str = Field(..., description="Opportunity ID")
    heat_exchanger_id: str = Field(..., description="Related heat exchanger")

    # Heat source
    source_stream_id: str = Field(..., description="Heat source stream")
    source_temperature: float = Field(..., description="Source temperature (C)")
    source_flow_rate: float = Field(..., description="Source flow (kg/s)")
    source_heat_content_kw: float = Field(..., description="Heat content (kW)")

    # Heat sink
    sink_stream_id: str = Field(..., description="Heat sink stream")
    sink_temperature: float = Field(..., description="Sink temperature (C)")
    sink_flow_rate: float = Field(..., description="Sink flow (kg/s)")
    sink_heat_demand_kw: float = Field(..., description="Heat demand (kW)")

    # Recovery potential
    recoverable_heat_kw: float = Field(..., description="Recoverable heat (kW)")
    recovery_efficiency: float = Field(..., ge=0, le=1, description="Recovery efficiency")
    annual_energy_savings_mwh: float = Field(..., description="Annual savings")
    annual_cost_savings: float = Field(..., description="Cost savings")
    co2_reduction_tonnes: float = Field(..., description="CO2 reduction")

    # Implementation
    implementation_cost: Optional[float] = Field(default=None)
    payback_period_months: Optional[float] = Field(default=None)
    priority_score: float = Field(default=0.0, ge=0, le=100)

    # Status
    status: str = Field(default="identified")
    identified_date: datetime = Field(default_factory=datetime.utcnow)


class MaintenancePrediction(BaseModel):
    """Maintenance prediction from GL-013 PREDICTMAINT."""

    model_config = ConfigDict(extra="allow")

    prediction_id: str = Field(..., description="Prediction ID")
    equipment_id: str = Field(..., description="Equipment ID")
    prediction_type: str = Field(..., description="Prediction type (fouling, failure, etc.)")

    # Prediction
    predicted_date: datetime = Field(..., description="Predicted event date")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence")
    probability: float = Field(..., ge=0, le=1, description="Event probability")

    # Severity
    severity: str = Field(default="medium", description="Impact severity")
    estimated_impact: Optional[str] = Field(default=None, description="Impact description")
    estimated_downtime_hours: Optional[float] = Field(default=None)
    estimated_cost: Optional[float] = Field(default=None)

    # Recommendation
    recommended_action: Optional[str] = Field(default=None)
    optimal_maintenance_date: Optional[datetime] = Field(default=None)

    # Model info
    model_version: str = Field(default="1.0.0")
    features_used: List[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ThermalEfficiencyData(BaseModel):
    """Thermal efficiency data from GL-001 THERMOSYNC."""

    model_config = ConfigDict(extra="allow")

    system_id: str = Field(..., description="Thermal system ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Related heat exchangers
    heat_exchanger_ids: List[str] = Field(default_factory=list)

    # System efficiency
    overall_thermal_efficiency: float = Field(..., ge=0, le=1)
    target_efficiency: float = Field(default=0.85, ge=0, le=1)
    efficiency_gap: Optional[float] = Field(default=None)

    # Component contributions
    heat_exchanger_efficiency: Optional[float] = Field(default=None)
    boiler_efficiency: Optional[float] = Field(default=None)
    distribution_efficiency: Optional[float] = Field(default=None)

    # Energy flows
    total_heat_input_kw: Optional[float] = Field(default=None)
    useful_heat_output_kw: Optional[float] = Field(default=None)
    heat_losses_kw: Optional[float] = Field(default=None)

    # Optimization recommendations
    optimization_potential_kw: Optional[float] = Field(default=None)
    recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# Agent Coordinator Implementation
# =============================================================================


class AgentCoordinator(BaseConnector):
    """
    Agent Coordinator for GL-014 EXCHANGER-PRO.

    Provides multi-agent coordination for heat exchanger optimization
    including integration with:
    - GL-001 THERMOSYNC (thermal efficiency)
    - GL-006 HEATRECLAIM (heat recovery opportunities)
    - GL-013 PREDICTMAINT (maintenance coordination)

    Features:
    - Message-based communication (REST/gRPC)
    - Data sharing protocols
    - Async request/response
    - Event subscription
    - Collaborative optimization
    """

    def __init__(self, config: AgentCoordinatorConfig) -> None:
        """Initialize agent coordinator."""
        super().__init__(config)
        self._coord_config = config

        # HTTP session for REST communication
        self._session: Optional[aiohttp.ClientSession] = None

        # Agent registry
        self._agent_registry: Dict[AgentID, AgentStatus] = {}
        self._agent_sessions: Dict[AgentID, aiohttp.ClientSession] = {}

        # Pending requests
        self._pending_requests: Dict[str, asyncio.Future] = {}

        # Event subscriptions
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)

        # Data cache for shared data
        self._shared_data_cache: Dict[str, Any] = {}

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish connections to partner agents."""
        self._logger.info("Initializing agent coordinator...")

        # Create main HTTP session
        timeout = aiohttp.ClientTimeout(
            total=self._coord_config.request_timeout_seconds,
            connect=30
        )

        connector = aiohttp.TCPConnector(
            limit=self._config.pool_max_size,
            limit_per_host=10
        )

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )

        # Connect to configured partner agents
        for agent_id_str, endpoint in self._coord_config.agent_endpoints.items():
            try:
                agent_id = AgentID(agent_id_str)
                await self._connect_to_agent(agent_id, endpoint)
            except Exception as e:
                self._logger.warning(
                    f"Failed to connect to agent {agent_id_str}: {e}"
                )

        self._state = ConnectionState.CONNECTED

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._logger.info(
            f"Agent coordinator initialized. Connected to "
            f"{len(self._agent_registry)} agents."
        )

    async def disconnect(self) -> None:
        """Disconnect from all partner agents."""
        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close agent sessions
        for session in self._agent_sessions.values():
            await session.close()
        self._agent_sessions.clear()

        # Close main session
        if self._session:
            await self._session.close()
            self._session = None

        self._agent_registry.clear()
        self._state = ConnectionState.DISCONNECTED
        self._logger.info("Agent coordinator disconnected")

    async def health_check(self) -> HealthCheckResult:
        """Check health of agent connections."""
        start_time = time.time()

        component_health = {}
        for agent_id, status in self._agent_registry.items():
            age = (datetime.utcnow() - status.last_heartbeat).total_seconds()
            if age < self._coord_config.agent_timeout_seconds:
                component_health[agent_id.value] = HealthStatus.HEALTHY
            else:
                component_health[agent_id.value] = HealthStatus.UNHEALTHY

        unhealthy_count = sum(
            1 for h in component_health.values()
            if h != HealthStatus.HEALTHY
        )

        if unhealthy_count == 0:
            overall_status = HealthStatus.HEALTHY
        elif unhealthy_count < len(component_health):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            status=overall_status,
            checked_at=datetime.utcnow(),
            latency_ms=(time.time() - start_time) * 1000,
            message=f"{len(self._agent_registry)} agents connected",
            component_health=component_health,
            details={
                "connected_agents": [a.value for a in self._agent_registry.keys()],
                "unhealthy_count": unhealthy_count,
            }
        )

    async def validate_configuration(self) -> bool:
        """Validate coordinator configuration."""
        if not self._coord_config.agent_id:
            raise ConfigurationError("Agent ID required")
        return True

    async def _connect_to_agent(
        self,
        agent_id: AgentID,
        endpoint: AgentEndpointConfig
    ) -> None:
        """Establish connection to a partner agent."""
        if endpoint.protocol == CommunicationProtocol.REST:
            if not endpoint.base_url:
                raise ConfigurationError(f"Base URL required for agent {agent_id}")

            # Create session for this agent
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            headers = {}
            if endpoint.api_key:
                headers["X-API-Key"] = endpoint.api_key

            session = aiohttp.ClientSession(
                base_url=endpoint.base_url,
                timeout=timeout,
                headers=headers
            )
            self._agent_sessions[agent_id] = session

            # Register agent
            self._agent_registry[agent_id] = AgentStatus(
                agent_id=agent_id,
                agent_name=agent_id.value,
                agent_version="unknown",
                status="connected",
                last_heartbeat=datetime.utcnow()
            )

            self._logger.info(f"Connected to agent {agent_id.value}")

        elif endpoint.protocol == CommunicationProtocol.GRPC:
            # gRPC connection would be implemented here
            self._logger.warning(f"gRPC not yet implemented for {agent_id}")

    async def _heartbeat_loop(self) -> None:
        """Background task for sending heartbeats."""
        while True:
            try:
                await asyncio.sleep(self._coord_config.heartbeat_interval_seconds)

                # Send heartbeats to all connected agents
                for agent_id in list(self._agent_registry.keys()):
                    try:
                        await self._send_heartbeat(agent_id)
                    except Exception as e:
                        self._logger.warning(
                            f"Heartbeat failed for {agent_id}: {e}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Heartbeat loop error: {e}")

    async def _send_heartbeat(self, agent_id: AgentID) -> None:
        """Send heartbeat to an agent."""
        message = AgentMessage(
            message_type=MessageType.HEARTBEAT,
            source_agent=self._coord_config.agent_id,
            target_agent=agent_id,
            action="heartbeat",
            payload={
                "status": "online",
                "capabilities": self._coord_config.capabilities,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        response = await self.send_message(message, wait_for_response=False)
        if response:
            self._agent_registry[agent_id].last_heartbeat = datetime.utcnow()

    # =========================================================================
    # Message Operations
    # =========================================================================

    async def send_message(
        self,
        message: AgentMessage,
        wait_for_response: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[AgentResponse]:
        """
        Send a message to another agent.

        Args:
            message: Message to send
            wait_for_response: Wait for response
            timeout: Response timeout

        Returns:
            Agent response if wait_for_response=True
        """
        timeout = timeout or self._coord_config.request_timeout_seconds

        if not message.target_agent:
            raise ValidationError("Target agent required")

        endpoint = self._coord_config.agent_endpoints.get(message.target_agent.value)
        if not endpoint:
            raise ConfigurationError(
                f"No endpoint configured for agent {message.target_agent}"
            )

        if endpoint.protocol == CommunicationProtocol.REST:
            return await self._send_rest_message(message, endpoint, wait_for_response, timeout)
        else:
            raise ConfigurationError(f"Protocol {endpoint.protocol} not implemented")

    async def _send_rest_message(
        self,
        message: AgentMessage,
        endpoint: AgentEndpointConfig,
        wait_for_response: bool,
        timeout: float
    ) -> Optional[AgentResponse]:
        """Send message via REST API."""
        session = self._agent_sessions.get(message.target_agent)
        if not session:
            # Use main session with full URL
            session = self._session
            url = f"{endpoint.base_url}/api/{endpoint.api_version}/messages"
        else:
            url = f"/api/{endpoint.api_version}/messages"

        headers = {"Content-Type": "application/json"}
        if endpoint.api_key:
            headers["X-API-Key"] = endpoint.api_key

        try:
            async with session.post(
                url,
                json=message.model_dump(mode="json"),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    self._logger.error(
                        f"Message send failed: {response.status} - {error_text}"
                    )
                    return AgentResponse(
                        correlation_id=message.message_id,
                        source_agent=message.target_agent,
                        target_agent=message.source_agent,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )

                if wait_for_response:
                    data = await response.json()
                    return AgentResponse(**data)
                return None

        except asyncio.TimeoutError:
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=False,
                error="Request timeout"
            )
        except Exception as e:
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=False,
                error=str(e)
            )

    async def broadcast_message(
        self,
        message: AgentMessage,
        exclude_agents: Optional[List[AgentID]] = None
    ) -> Dict[AgentID, Optional[AgentResponse]]:
        """Broadcast message to all connected agents."""
        exclude = set(exclude_agents or [])
        exclude.add(self._coord_config.agent_id)  # Don't send to self

        responses = {}
        tasks = []

        for agent_id in self._agent_registry.keys():
            if agent_id in exclude:
                continue

            msg_copy = message.model_copy()
            msg_copy.target_agent = agent_id
            tasks.append((agent_id, self.send_message(msg_copy)))

        results = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True
        )

        for (agent_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                responses[agent_id] = AgentResponse(
                    correlation_id=message.message_id,
                    source_agent=agent_id,
                    target_agent=self._coord_config.agent_id,
                    success=False,
                    error=str(result)
                )
            else:
                responses[agent_id] = result

        return responses

    # =========================================================================
    # GL-001 THERMOSYNC Integration
    # =========================================================================

    async def get_thermal_efficiency_data(
        self,
        system_id: str,
        heat_exchanger_ids: Optional[List[str]] = None
    ) -> Optional[ThermalEfficiencyData]:
        """
        Request thermal efficiency data from GL-001 THERMOSYNC.

        Args:
            system_id: Thermal system ID
            heat_exchanger_ids: Optional list of heat exchangers

        Returns:
            Thermal efficiency data
        """
        if AgentID.GL_001_THERMOSYNC not in self._agent_registry:
            self._logger.warning("GL-001 THERMOSYNC not connected")
            return None

        message = AgentMessage(
            message_type=MessageType.QUERY,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_001_THERMOSYNC,
            action="get_thermal_efficiency",
            payload={
                "system_id": system_id,
                "heat_exchanger_ids": heat_exchanger_ids or [],
                "include_components": True,
                "include_recommendations": True,
            }
        )

        response = await self.send_message(message)

        if response and response.success and response.result:
            return ThermalEfficiencyData(**response.result)

        return None

    async def share_heat_exchanger_performance(
        self,
        performance_data: HeatExchangerPerformanceData
    ) -> bool:
        """
        Share heat exchanger performance data with GL-001 THERMOSYNC.

        Args:
            performance_data: Performance data to share

        Returns:
            True if successful
        """
        if AgentID.GL_001_THERMOSYNC not in self._agent_registry:
            return False

        message = AgentMessage(
            message_type=MessageType.DATA_SHARE,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_001_THERMOSYNC,
            action="heat_exchanger_performance_update",
            payload=performance_data.model_dump(mode="json")
        )

        response = await self.send_message(message, wait_for_response=False)
        return response is None or response.success

    # =========================================================================
    # GL-006 HEATRECLAIM Integration
    # =========================================================================

    async def get_heat_recovery_opportunities(
        self,
        heat_exchanger_id: Optional[str] = None,
        min_savings_kw: float = 0.0
    ) -> List[HeatRecoveryOpportunity]:
        """
        Request heat recovery opportunities from GL-006 HEATRECLAIM.

        Args:
            heat_exchanger_id: Filter by heat exchanger
            min_savings_kw: Minimum savings threshold

        Returns:
            List of heat recovery opportunities
        """
        if AgentID.GL_006_HEATRECLAIM not in self._agent_registry:
            self._logger.warning("GL-006 HEATRECLAIM not connected")
            return []

        message = AgentMessage(
            message_type=MessageType.QUERY,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_006_HEATRECLAIM,
            action="get_heat_recovery_opportunities",
            payload={
                "heat_exchanger_id": heat_exchanger_id,
                "min_savings_kw": min_savings_kw,
                "include_implementation_details": True,
            }
        )

        response = await self.send_message(message)

        if response and response.success and response.result:
            opportunities = response.result.get("opportunities", [])
            return [HeatRecoveryOpportunity(**opp) for opp in opportunities]

        return []

    async def notify_heat_exchanger_optimization(
        self,
        equipment_id: str,
        optimization_type: str,
        details: Dict[str, Any]
    ) -> bool:
        """
        Notify GL-006 about heat exchanger optimization.

        Args:
            equipment_id: Equipment ID
            optimization_type: Type of optimization
            details: Optimization details

        Returns:
            True if notification sent
        """
        if AgentID.GL_006_HEATRECLAIM not in self._agent_registry:
            return False

        message = AgentMessage(
            message_type=MessageType.EVENT,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_006_HEATRECLAIM,
            action="heat_exchanger_optimization",
            payload={
                "equipment_id": equipment_id,
                "optimization_type": optimization_type,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        response = await self.send_message(message, wait_for_response=False)
        return response is None or response.success

    # =========================================================================
    # GL-013 PREDICTMAINT Integration
    # =========================================================================

    async def get_maintenance_predictions(
        self,
        equipment_id: str,
        prediction_types: Optional[List[str]] = None
    ) -> List[MaintenancePrediction]:
        """
        Request maintenance predictions from GL-013 PREDICTMAINT.

        Args:
            equipment_id: Equipment ID
            prediction_types: Filter by prediction types

        Returns:
            List of maintenance predictions
        """
        if AgentID.GL_013_PREDICTMAINT not in self._agent_registry:
            self._logger.warning("GL-013 PREDICTMAINT not connected")
            return []

        message = AgentMessage(
            message_type=MessageType.QUERY,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_013_PREDICTMAINT,
            action="get_predictions",
            payload={
                "equipment_id": equipment_id,
                "prediction_types": prediction_types or ["fouling", "failure", "cleaning"],
                "include_confidence": True,
            }
        )

        response = await self.send_message(message)

        if response and response.success and response.result:
            predictions = response.result.get("predictions", [])
            return [MaintenancePrediction(**pred) for pred in predictions]

        return []

    async def request_cleaning_work_order(
        self,
        equipment_id: str,
        cleaning_reason: str,
        current_fouling: float,
        current_ua_percent: float,
        predicted_critical_date: Optional[datetime] = None,
        urgency: str = "normal"
    ) -> Optional[str]:
        """
        Request GL-013 to create a cleaning work order.

        Args:
            equipment_id: Equipment ID
            cleaning_reason: Reason for cleaning
            current_fouling: Current fouling factor
            current_ua_percent: Current UA percentage
            predicted_critical_date: Predicted critical date
            urgency: Urgency level

        Returns:
            Work order ID if created
        """
        if AgentID.GL_013_PREDICTMAINT not in self._agent_registry:
            self._logger.warning("GL-013 PREDICTMAINT not connected")
            return None

        message = AgentMessage(
            message_type=MessageType.REQUEST,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_013_PREDICTMAINT,
            action="create_cleaning_work_order",
            priority=MessagePriority.HIGH if urgency in ["urgent", "critical"] else MessagePriority.NORMAL,
            payload={
                "equipment_id": equipment_id,
                "equipment_type": "heat_exchanger",
                "maintenance_type": "cleaning",
                "reason": cleaning_reason,
                "urgency": urgency,
                "metrics": {
                    "current_fouling_factor": current_fouling,
                    "current_ua_percent": current_ua_percent,
                    "predicted_critical_date": (
                        predicted_critical_date.isoformat()
                        if predicted_critical_date else None
                    ),
                },
                "requested_by": self._coord_config.agent_id.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        response = await self.send_message(message)

        if response and response.success and response.result:
            return response.result.get("work_order_id")

        return None

    async def share_fouling_prediction(
        self,
        equipment_id: str,
        current_fouling: float,
        predicted_fouling_rate: float,
        predicted_critical_date: datetime,
        confidence: float,
        model_info: Dict[str, Any]
    ) -> bool:
        """
        Share fouling prediction with GL-013 PREDICTMAINT.

        Args:
            equipment_id: Equipment ID
            current_fouling: Current fouling factor
            predicted_fouling_rate: Predicted rate of increase
            predicted_critical_date: Predicted date to reach threshold
            confidence: Prediction confidence
            model_info: Model information

        Returns:
            True if shared successfully
        """
        if AgentID.GL_013_PREDICTMAINT not in self._agent_registry:
            return False

        message = AgentMessage(
            message_type=MessageType.DATA_SHARE,
            source_agent=self._coord_config.agent_id,
            target_agent=AgentID.GL_013_PREDICTMAINT,
            action="fouling_prediction_update",
            payload={
                "equipment_id": equipment_id,
                "equipment_type": "heat_exchanger",
                "prediction": {
                    "type": "fouling",
                    "current_value": current_fouling,
                    "rate_of_change": predicted_fouling_rate,
                    "predicted_critical_date": predicted_critical_date.isoformat(),
                    "confidence": confidence,
                },
                "model_info": model_info,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        response = await self.send_message(message, wait_for_response=False)
        return response is None or response.success

    # =========================================================================
    # Event Subscription
    # =========================================================================

    async def subscribe_to_events(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None],
        source_agent: Optional[AgentID] = None
    ) -> str:
        """
        Subscribe to events from other agents.

        Args:
            event_type: Event type to subscribe to
            callback: Callback function
            source_agent: Filter by source agent

        Returns:
            Subscription ID
        """
        subscription_key = f"{source_agent.value if source_agent else '*'}:{event_type}"
        self._subscriptions[subscription_key].append(callback)

        # Notify source agent if specified
        if source_agent and source_agent in self._agent_registry:
            message = AgentMessage(
                message_type=MessageType.SUBSCRIPTION,
                source_agent=self._coord_config.agent_id,
                target_agent=source_agent,
                action="subscribe",
                payload={
                    "event_type": event_type,
                    "subscriber": self._coord_config.agent_id.value,
                }
            )
            await self.send_message(message, wait_for_response=False)

        return subscription_key

    async def handle_incoming_event(
        self,
        source_agent: AgentID,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """Handle an incoming event from another agent."""
        # Check for specific subscription
        specific_key = f"{source_agent.value}:{event_type}"
        wildcard_key = f"*:{event_type}"

        callbacks = (
            self._subscriptions.get(specific_key, []) +
            self._subscriptions.get(wildcard_key, [])
        )

        for callback in callbacks:
            try:
                callback(payload)
            except Exception as e:
                self._logger.error(f"Event callback error: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_connected_agents(self) -> List[AgentID]:
        """Get list of connected agents."""
        return list(self._agent_registry.keys())

    def is_agent_connected(self, agent_id: AgentID) -> bool:
        """Check if an agent is connected."""
        if agent_id not in self._agent_registry:
            return False

        status = self._agent_registry[agent_id]
        age = (datetime.utcnow() - status.last_heartbeat).total_seconds()
        return age < self._coord_config.agent_timeout_seconds


# =============================================================================
# Factory Function
# =============================================================================


def create_agent_coordinator(
    connector_name: str = "GL-014 Agent Coordinator",
    thermosync_url: Optional[str] = None,
    heatreclaim_url: Optional[str] = None,
    predictmaint_url: Optional[str] = None,
    **kwargs
) -> AgentCoordinator:
    """
    Factory function to create agent coordinator.

    Args:
        connector_name: Connector name
        thermosync_url: GL-001 THERMOSYNC base URL
        heatreclaim_url: GL-006 HEATRECLAIM base URL
        predictmaint_url: GL-013 PREDICTMAINT base URL
        **kwargs: Additional configuration options

    Returns:
        Configured AgentCoordinator
    """
    agent_endpoints = {}

    if thermosync_url:
        agent_endpoints[AgentID.GL_001_THERMOSYNC.value] = AgentEndpointConfig(
            agent_id=AgentID.GL_001_THERMOSYNC,
            protocol=CommunicationProtocol.REST,
            base_url=thermosync_url,
        )

    if heatreclaim_url:
        agent_endpoints[AgentID.GL_006_HEATRECLAIM.value] = AgentEndpointConfig(
            agent_id=AgentID.GL_006_HEATRECLAIM,
            protocol=CommunicationProtocol.REST,
            base_url=heatreclaim_url,
        )

    if predictmaint_url:
        agent_endpoints[AgentID.GL_013_PREDICTMAINT.value] = AgentEndpointConfig(
            agent_id=AgentID.GL_013_PREDICTMAINT,
            protocol=CommunicationProtocol.REST,
            base_url=predictmaint_url,
        )

    config = AgentCoordinatorConfig(
        connector_name=connector_name,
        connector_type=ConnectorType.AGENT_COORDINATOR,
        agent_endpoints=agent_endpoints,
        **kwargs
    )

    return AgentCoordinator(config)
