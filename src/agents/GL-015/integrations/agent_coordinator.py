"""
Agent Coordinator Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides integration with other GreenLang agents for comprehensive energy optimization:
- GL-001 THERMOSYNC (Thermal Efficiency) - Thermal efficiency context
- GL-006 HEATRECLAIM (Heat Recovery) - Heat recovery opportunities
- GL-014 EXCHANGER-PRO (Heat Exchanger) - Heat exchanger insulation context

Implements data sharing protocols, message routing, and coordinated analytics
across the GreenLang agent ecosystem.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
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
import json
import logging
import uuid

import aiohttp
from pydantic import BaseModel, Field, ConfigDict

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConfigurationError,
    ConnectionError,
    ConnectionState,
    ConnectorError,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class AgentID(str, Enum):
    """GreenLang agent identifiers."""

    # Current agent
    GL_015_INSULSCAN = "GL-015"

    # Related agents for insulation context
    GL_001_THERMOSYNC = "GL-001"  # Thermal Efficiency Optimization
    GL_006_HEATRECLAIM = "GL-006"  # Heat Recovery Optimization
    GL_014_EXCHANGER_PRO = "GL-014"  # Heat Exchanger Optimization

    # Additional potentially relevant agents
    GL_002_ENERGYAUDIT = "GL-002"  # Energy Audit
    GL_003_STEAMTRAP = "GL-003"  # Steam Trap Monitor
    GL_005_BOILEROPT = "GL-005"  # Boiler Optimization


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"  # Request for data/action
    RESPONSE = "response"  # Response to request
    NOTIFICATION = "notification"  # One-way notification
    BROADCAST = "broadcast"  # Broadcast to all agents
    HEARTBEAT = "heartbeat"  # Health/availability check
    SUBSCRIPTION = "subscription"  # Subscribe to events
    EVENT = "event"  # Event notification


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Status of coordinated tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RoutingStrategy(str, Enum):
    """Message routing strategies."""

    DIRECT = "direct"  # Point-to-point
    BROADCAST = "broadcast"  # To all subscribers
    ROUND_ROBIN = "round_robin"  # Load balanced
    PRIORITY = "priority"  # Priority-based


class CommunicationProtocol(str, Enum):
    """Communication protocols."""

    HTTP_REST = "http_rest"
    GRPC = "grpc"
    AMQP = "amqp"
    KAFKA = "kafka"
    REDIS_PUBSUB = "redis_pubsub"
    INTERNAL = "internal"  # In-process


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class AgentEndpointConfig(BaseModel):
    """Configuration for an agent endpoint."""

    model_config = ConfigDict(extra="forbid")

    agent_id: AgentID = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Agent display name")
    base_url: str = Field(..., description="Agent API base URL")
    api_version: str = Field(default="v1", description="API version")
    protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.HTTP_REST,
        description="Communication protocol"
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token"
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Request timeout"
    )
    enabled: bool = Field(default=True, description="Is agent enabled")


class MessageBusConfig(BaseModel):
    """Message bus configuration."""

    model_config = ConfigDict(extra="forbid")

    protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.HTTP_REST,
        description="Message bus protocol"
    )
    broker_url: Optional[str] = Field(
        default=None,
        description="Message broker URL (for AMQP/Kafka)"
    )
    exchange_name: str = Field(
        default="greenlang_agents",
        description="Exchange/topic name"
    )
    queue_name: Optional[str] = Field(
        default=None,
        description="Queue name for this agent"
    )
    prefetch_count: int = Field(
        default=10,
        ge=1,
        description="Message prefetch count"
    )


class AgentCoordinatorConfig(BaseConnectorConfig):
    """Configuration for agent coordinator."""

    model_config = ConfigDict(extra="forbid")

    # This agent's identity
    self_agent_id: AgentID = Field(
        default=AgentID.GL_015_INSULSCAN,
        description="This agent's ID"
    )
    self_agent_name: str = Field(
        default="INSULSCAN",
        description="This agent's name"
    )

    # Agent endpoints
    agent_endpoints: Dict[AgentID, AgentEndpointConfig] = Field(
        default_factory=dict,
        description="Configured agent endpoints"
    )

    # Message bus (for pub/sub)
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
        description="Request timeout"
    )
    response_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Response wait timeout"
    )

    # Health monitoring
    agent_health_check_interval_seconds: float = Field(
        default=60.0,
        ge=10.0,
        description="Agent health check interval"
    )
    max_consecutive_failures: int = Field(
        default=3,
        ge=1,
        description="Max failures before marking agent unhealthy"
    )

    # Event subscriptions
    subscribed_events: List[str] = Field(
        default_factory=list,
        description="Events to subscribe to"
    )

    def __init__(self, **data):
        """Initialize with connector type set."""
        data['connector_type'] = ConnectorType.AGENT_COORDINATOR
        super().__init__(**data)


# =============================================================================
# Data Models - Messages
# =============================================================================


class AgentMessage(BaseModel):
    """Inter-agent message."""

    model_config = ConfigDict(frozen=False)

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message ID"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request-response"
    )

    # Routing
    source_agent: AgentID = Field(..., description="Source agent")
    target_agent: Optional[AgentID] = Field(
        default=None,
        description="Target agent (None for broadcast)"
    )
    message_type: MessageType = Field(
        default=MessageType.REQUEST,
        description="Message type"
    )
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="Priority"
    )

    # Content
    action: str = Field(..., description="Action/method name")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message metadata"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Message expiration"
    )

    # Tracking
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)


class AgentResponse(BaseModel):
    """Response to an agent message."""

    model_config = ConfigDict(frozen=False)

    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Response ID"
    )
    correlation_id: str = Field(..., description="Original message ID")

    # Routing
    source_agent: AgentID = Field(..., description="Responding agent")
    target_agent: AgentID = Field(..., description="Original requester")

    # Result
    success: bool = Field(..., description="Was request successful")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response data"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Error code if failed"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Processing time in milliseconds"
    )


class AgentStatus(BaseModel):
    """Status of a GreenLang agent."""

    model_config = ConfigDict(frozen=False)

    agent_id: AgentID = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    agent_version: str = Field(..., description="Agent version")
    status: str = Field(
        default="unknown",
        description="Agent status"
    )
    health_status: HealthStatus = Field(
        default=HealthStatus.UNKNOWN,
        description="Health status"
    )
    last_heartbeat: Optional[datetime] = Field(
        default=None,
        description="Last heartbeat"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="Agent capabilities"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# =============================================================================
# Data Models - Insulation Context Sharing
# =============================================================================


class InsulationDefectData(BaseModel):
    """Insulation defect data for sharing with other agents."""

    model_config = ConfigDict(frozen=True)

    defect_id: str = Field(..., description="Defect identifier")
    equipment_id: str = Field(..., description="Equipment ID")
    equipment_type: str = Field(..., description="Equipment type")
    location_id: str = Field(..., description="Location ID")

    # Defect details
    defect_type: str = Field(..., description="Type of defect")
    severity: str = Field(..., description="Severity level")
    affected_area_m2: float = Field(..., ge=0, description="Affected area")

    # Temperature data
    surface_temperature_c: float = Field(
        ...,
        description="Surface temperature"
    )
    ambient_temperature_c: float = Field(
        ...,
        description="Ambient temperature"
    )
    expected_surface_temperature_c: Optional[float] = Field(
        default=None,
        description="Expected surface temperature"
    )
    process_temperature_c: Optional[float] = Field(
        default=None,
        description="Process temperature"
    )

    # Heat loss
    heat_loss_kw: float = Field(..., ge=0, description="Heat loss in kW")
    annual_energy_loss_mwh: float = Field(
        ...,
        ge=0,
        description="Annual energy loss"
    )
    annual_cost_impact: float = Field(..., ge=0, description="Annual cost")
    annual_co2_tonnes: float = Field(..., ge=0, description="CO2 impact")

    # Timestamps
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    thermal_image_id: Optional[str] = Field(default=None)


class ThermalEfficiencyContext(BaseModel):
    """Thermal efficiency context from GL-001 THERMOSYNC."""

    model_config = ConfigDict(frozen=True)

    system_id: str = Field(..., description="System identifier")
    overall_thermal_efficiency: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall thermal efficiency (0-1)"
    )
    target_efficiency: float = Field(
        ...,
        ge=0,
        le=1,
        description="Target efficiency"
    )
    efficiency_gap: float = Field(
        ...,
        description="Gap to target"
    )

    # Insulation contribution
    insulation_efficiency: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Insulation contribution to efficiency"
    )
    insulation_loss_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat loss from insulation"
    )
    insulation_loss_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Insulation loss as % of total loss"
    )

    # Recommendations
    priority_areas: List[str] = Field(
        default_factory=list,
        description="Priority improvement areas"
    )
    improvement_potential_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Improvement potential"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HeatRecoveryOpportunity(BaseModel):
    """Heat recovery opportunity from GL-006 HEATRECLAIM."""

    model_config = ConfigDict(frozen=True)

    opportunity_id: str = Field(..., description="Opportunity ID")
    source_equipment_id: str = Field(
        ...,
        description="Heat source equipment"
    )
    sink_equipment_id: Optional[str] = Field(
        default=None,
        description="Heat sink equipment"
    )

    # Heat details
    available_heat_kw: float = Field(..., ge=0, description="Available heat")
    recoverable_heat_kw: float = Field(
        ...,
        ge=0,
        description="Recoverable heat"
    )
    source_temperature_c: float = Field(
        ...,
        description="Source temperature"
    )
    sink_temperature_c: Optional[float] = Field(
        default=None,
        description="Sink temperature"
    )

    # Economics
    annual_savings_kwh: float = Field(..., ge=0, description="Annual savings")
    annual_cost_savings: float = Field(..., ge=0, description="Cost savings")
    implementation_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Implementation cost"
    )
    payback_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period"
    )

    # Insulation relevance
    insulation_improvement_required: bool = Field(
        default=False,
        description="Does opportunity require insulation work"
    )
    related_insulation_defects: List[str] = Field(
        default_factory=list,
        description="Related insulation defect IDs"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HeatExchangerInsulationContext(BaseModel):
    """Heat exchanger insulation context from GL-014 EXCHANGER-PRO."""

    model_config = ConfigDict(frozen=True)

    heat_exchanger_id: str = Field(..., description="Heat exchanger ID")
    heat_exchanger_type: str = Field(..., description="Type")

    # Operating conditions
    hot_side_temperature_c: float = Field(..., description="Hot side temp")
    cold_side_temperature_c: float = Field(..., description="Cold side temp")
    duty_kw: float = Field(..., ge=0, description="Current duty")
    design_duty_kw: float = Field(..., ge=0, description="Design duty")
    efficiency: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current efficiency"
    )

    # Insulation status
    is_insulated: bool = Field(
        default=True,
        description="Has insulation"
    )
    insulation_thickness_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Insulation thickness"
    )
    insulation_condition: Optional[str] = Field(
        default=None,
        description="Insulation condition"
    )
    shell_heat_loss_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Shell heat loss"
    )

    # Recommendations
    insulation_recommendation: Optional[str] = Field(
        default=None,
        description="Insulation recommendation"
    )
    potential_savings_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Potential savings from insulation"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Agent Coordinator
# =============================================================================


class AgentCoordinator(BaseConnector):
    """
    Agent Coordinator for GL-015 INSULSCAN.

    Manages communication with other GreenLang agents for coordinated
    energy optimization including thermal efficiency context, heat recovery
    opportunities, and heat exchanger insulation data.

    Features:
    - Inter-agent messaging (request/response, pub/sub)
    - Agent health monitoring
    - Data sharing protocols
    - Event subscription and handling
    """

    def __init__(self, config: AgentCoordinatorConfig) -> None:
        """
        Initialize agent coordinator.

        Args:
            config: Agent coordinator configuration
        """
        super().__init__(config)
        self._coordinator_config = config

        # HTTP session for REST communication
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Agent status tracking
        self._agent_status: Dict[AgentID, AgentStatus] = {}
        self._agent_health_failures: Dict[AgentID, int] = {}

        # Pending requests (correlation_id -> future)
        self._pending_requests: Dict[str, asyncio.Future] = {}

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Establish connections to configured agents."""
        self._logger.info("Initializing agent coordinator")

        try:
            self._state = ConnectionState.CONNECTING

            # Get HTTP session from pool
            self._http_session = await self._pool.get_session()

            # Discover and verify agents
            await self._discover_agents()

            # Start health monitoring
            self._health_check_task = asyncio.create_task(
                self._agent_health_check_loop()
            )

            self._state = ConnectionState.CONNECTED
            self._logger.info(
                f"Agent coordinator connected. {len(self._agent_status)} agents discovered."
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to initialize agent coordinator: {e}")
            raise ConnectionError(f"Agent coordinator initialization failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from all agents."""
        self._logger.info("Disconnecting agent coordinator")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests
        for correlation_id, future in self._pending_requests.items():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        self._agent_status.clear()
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Check health of agent coordinator and connected agents."""
        import time
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Not connected: {self._state.value}"
                )

            # Count healthy agents
            healthy_count = sum(
                1 for status in self._agent_status.values()
                if status.health_status == HealthStatus.HEALTHY
            )
            total_agents = len(self._agent_status)

            latency_ms = (time.time() - start_time) * 1000

            if healthy_count == total_agents:
                overall_status = HealthStatus.HEALTHY
                message = f"All {total_agents} agents healthy"
            elif healthy_count > 0:
                overall_status = HealthStatus.DEGRADED
                message = f"{healthy_count}/{total_agents} agents healthy"
            else:
                overall_status = HealthStatus.UNHEALTHY
                message = "No healthy agents"

            component_health = {
                agent_id.value: status.health_status
                for agent_id, status in self._agent_status.items()
            }

            return HealthCheckResult(
                status=overall_status,
                latency_ms=latency_ms,
                message=message,
                component_health=component_health,
                details={
                    "total_agents": total_agents,
                    "healthy_agents": healthy_count,
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}"
            )

    async def validate_configuration(self) -> bool:
        """Validate coordinator configuration."""
        # At least one agent should be configured
        if not self._coordinator_config.agent_endpoints:
            self._logger.warning("No agent endpoints configured")

        return True

    # =========================================================================
    # Agent Discovery and Health
    # =========================================================================

    async def _discover_agents(self) -> None:
        """Discover and verify configured agents."""
        for agent_id, endpoint in self._coordinator_config.agent_endpoints.items():
            if not endpoint.enabled:
                continue

            try:
                status = await self._get_agent_status(agent_id, endpoint)
                self._agent_status[agent_id] = status
                self._agent_health_failures[agent_id] = 0
                self._logger.info(
                    f"Discovered agent: {agent_id.value} ({status.agent_name})"
                )
            except Exception as e:
                self._logger.warning(
                    f"Failed to discover agent {agent_id.value}: {e}"
                )
                self._agent_status[agent_id] = AgentStatus(
                    agent_id=agent_id,
                    agent_name=endpoint.agent_name,
                    agent_version="unknown",
                    status="unreachable",
                    health_status=HealthStatus.UNHEALTHY,
                )

    async def _get_agent_status(
        self,
        agent_id: AgentID,
        endpoint: AgentEndpointConfig
    ) -> AgentStatus:
        """Get status from an agent."""
        url = f"{endpoint.base_url}/api/{endpoint.api_version}/status"
        headers = {}
        if endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"

        async with self._http_session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"Agent returned {response.status}")

            data = await response.json()

            return AgentStatus(
                agent_id=agent_id,
                agent_name=data.get("name", endpoint.agent_name),
                agent_version=data.get("version", "unknown"),
                status=data.get("status", "online"),
                health_status=HealthStatus(data.get("health", "healthy")),
                last_heartbeat=datetime.utcnow(),
                capabilities=data.get("capabilities", []),
            )

    async def _agent_health_check_loop(self) -> None:
        """Background task for agent health monitoring."""
        while True:
            try:
                await asyncio.sleep(
                    self._coordinator_config.agent_health_check_interval_seconds
                )

                for agent_id, endpoint in self._coordinator_config.agent_endpoints.items():
                    if not endpoint.enabled:
                        continue

                    try:
                        status = await self._get_agent_status(agent_id, endpoint)
                        self._agent_status[agent_id] = status
                        self._agent_health_failures[agent_id] = 0
                    except Exception as e:
                        self._agent_health_failures[agent_id] = \
                            self._agent_health_failures.get(agent_id, 0) + 1

                        if self._agent_health_failures[agent_id] >= \
                                self._coordinator_config.max_consecutive_failures:
                            if agent_id in self._agent_status:
                                self._agent_status[agent_id].health_status = \
                                    HealthStatus.UNHEALTHY
                            self._logger.warning(
                                f"Agent {agent_id.value} marked unhealthy after "
                                f"{self._agent_health_failures[agent_id]} failures"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check loop error: {e}")

    # =========================================================================
    # Messaging
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def send_message(
        self,
        target_agent: AgentID,
        action: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        wait_for_response: bool = True,
        timeout_seconds: Optional[float] = None
    ) -> Optional[AgentResponse]:
        """
        Send message to another agent.

        Args:
            target_agent: Target agent ID
            action: Action/method to invoke
            payload: Message payload
            message_type: Type of message
            priority: Message priority
            wait_for_response: Wait for response
            timeout_seconds: Response timeout

        Returns:
            Agent response if wait_for_response=True
        """
        # Create message
        message = AgentMessage(
            source_agent=self._coordinator_config.self_agent_id,
            target_agent=target_agent,
            message_type=message_type,
            priority=priority,
            action=action,
            payload=payload,
        )

        async def _send():
            return await self._send_http_message(message, wait_for_response, timeout_seconds)

        return await self.execute_with_protection(
            operation=_send,
            operation_name=f"send_message_{action}",
            validate_result=False
        )

    async def _send_http_message(
        self,
        message: AgentMessage,
        wait_for_response: bool,
        timeout_seconds: Optional[float]
    ) -> Optional[AgentResponse]:
        """Send message via HTTP REST."""
        target_agent = message.target_agent
        if target_agent not in self._coordinator_config.agent_endpoints:
            raise ConfigurationError(f"No endpoint configured for {target_agent}")

        endpoint = self._coordinator_config.agent_endpoints[target_agent]

        url = f"{endpoint.base_url}/api/{endpoint.api_version}/messages"
        headers = {"Content-Type": "application/json"}
        if endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"

        timeout = timeout_seconds or self._coordinator_config.request_timeout_seconds

        async with self._http_session.post(
            url,
            headers=headers,
            json=message.model_dump(mode="json"),
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status not in [200, 201, 202]:
                text = await response.text()
                raise ConnectorError(
                    f"Agent {target_agent.value} returned {response.status}: {text}"
                )

            if not wait_for_response:
                return None

            if response.status == 202:
                # Async processing - need to poll or wait
                return await self._wait_for_async_response(
                    message.message_id,
                    target_agent,
                    timeout
                )

            data = await response.json()
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=target_agent,
                target_agent=self._coordinator_config.self_agent_id,
                success=data.get("success", True),
                result=data.get("result"),
                error_code=data.get("error_code"),
                error_message=data.get("error_message"),
            )

    async def _wait_for_async_response(
        self,
        correlation_id: str,
        target_agent: AgentID,
        timeout: float
    ) -> AgentResponse:
        """Wait for async response using polling."""
        endpoint = self._coordinator_config.agent_endpoints[target_agent]
        url = f"{endpoint.base_url}/api/{endpoint.api_version}/messages/{correlation_id}/response"

        headers = {}
        if endpoint.auth_token:
            headers["Authorization"] = f"Bearer {endpoint.auth_token}"

        end_time = asyncio.get_event_loop().time() + timeout
        poll_interval = 0.5

        while asyncio.get_event_loop().time() < end_time:
            async with self._http_session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return AgentResponse(
                        correlation_id=correlation_id,
                        source_agent=target_agent,
                        target_agent=self._coordinator_config.self_agent_id,
                        success=data.get("success", True),
                        result=data.get("result"),
                        error_code=data.get("error_code"),
                        error_message=data.get("error_message"),
                    )
                elif response.status == 202:
                    # Still processing
                    await asyncio.sleep(poll_interval)
                    poll_interval = min(poll_interval * 1.5, 5.0)
                else:
                    raise ConnectorError(f"Response poll failed: {response.status}")

        raise TimeoutError(f"Response timeout for {correlation_id}")

    async def broadcast_message(
        self,
        action: str,
        payload: Dict[str, Any],
        target_agents: Optional[List[AgentID]] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Dict[AgentID, Optional[AgentResponse]]:
        """
        Broadcast message to multiple agents.

        Args:
            action: Action to invoke
            payload: Message payload
            target_agents: Specific agents (None = all)
            priority: Message priority

        Returns:
            Dict of agent ID to response
        """
        targets = target_agents or list(self._coordinator_config.agent_endpoints.keys())
        targets = [
            t for t in targets
            if t != self._coordinator_config.self_agent_id
        ]

        tasks = []
        for agent_id in targets:
            task = asyncio.create_task(
                self.send_message(
                    target_agent=agent_id,
                    action=action,
                    payload=payload,
                    message_type=MessageType.BROADCAST,
                    priority=priority,
                    wait_for_response=True
                )
            )
            tasks.append((agent_id, task))

        results = {}
        for agent_id, task in tasks:
            try:
                results[agent_id] = await task
            except Exception as e:
                self._logger.warning(
                    f"Broadcast to {agent_id.value} failed: {e}"
                )
                results[agent_id] = None

        return results

    # =========================================================================
    # GL-001 THERMOSYNC Integration
    # =========================================================================

    async def get_thermal_efficiency_context(
        self,
        system_id: str
    ) -> Optional[ThermalEfficiencyContext]:
        """
        Get thermal efficiency context from GL-001 THERMOSYNC.

        Args:
            system_id: System/plant identifier

        Returns:
            Thermal efficiency context or None
        """
        if AgentID.GL_001_THERMOSYNC not in self._coordinator_config.agent_endpoints:
            self._logger.warning("GL-001 THERMOSYNC not configured")
            return None

        response = await self.send_message(
            target_agent=AgentID.GL_001_THERMOSYNC,
            action="get_thermal_efficiency",
            payload={"system_id": system_id}
        )

        if not response or not response.success:
            self._logger.warning(
                f"Failed to get thermal efficiency from GL-001: "
                f"{response.error_message if response else 'No response'}"
            )
            return None

        result = response.result
        return ThermalEfficiencyContext(
            system_id=result.get("system_id", system_id),
            overall_thermal_efficiency=result.get("overall_thermal_efficiency", 0),
            target_efficiency=result.get("target_efficiency", 0.85),
            efficiency_gap=result.get("efficiency_gap", 0),
            insulation_efficiency=result.get("insulation_efficiency"),
            insulation_loss_kw=result.get("insulation_loss_kw"),
            insulation_loss_percent=result.get("insulation_loss_percent"),
            priority_areas=result.get("priority_areas", []),
            improvement_potential_kw=result.get("improvement_potential_kw"),
        )

    async def share_insulation_defect(
        self,
        defect: InsulationDefectData
    ) -> bool:
        """
        Share insulation defect data with GL-001 THERMOSYNC.

        Args:
            defect: Insulation defect data

        Returns:
            True if shared successfully
        """
        if AgentID.GL_001_THERMOSYNC not in self._coordinator_config.agent_endpoints:
            return False

        response = await self.send_message(
            target_agent=AgentID.GL_001_THERMOSYNC,
            action="report_insulation_defect",
            payload=defect.model_dump(mode="json"),
            message_type=MessageType.NOTIFICATION
        )

        return response is not None and response.success

    # =========================================================================
    # GL-006 HEATRECLAIM Integration
    # =========================================================================

    async def get_heat_recovery_opportunities(
        self,
        location_id: Optional[str] = None,
        equipment_ids: Optional[List[str]] = None
    ) -> List[HeatRecoveryOpportunity]:
        """
        Get heat recovery opportunities from GL-006 HEATRECLAIM.

        Args:
            location_id: Filter by location
            equipment_ids: Filter by equipment

        Returns:
            List of heat recovery opportunities
        """
        if AgentID.GL_006_HEATRECLAIM not in self._coordinator_config.agent_endpoints:
            self._logger.warning("GL-006 HEATRECLAIM not configured")
            return []

        payload = {}
        if location_id:
            payload["location_id"] = location_id
        if equipment_ids:
            payload["equipment_ids"] = equipment_ids

        response = await self.send_message(
            target_agent=AgentID.GL_006_HEATRECLAIM,
            action="get_heat_recovery_opportunities",
            payload=payload
        )

        if not response or not response.success:
            self._logger.warning(
                f"Failed to get heat recovery opportunities from GL-006: "
                f"{response.error_message if response else 'No response'}"
            )
            return []

        opportunities = []
        for opp_data in response.result.get("opportunities", []):
            opportunities.append(HeatRecoveryOpportunity(
                opportunity_id=opp_data.get("opportunity_id", str(uuid.uuid4())),
                source_equipment_id=opp_data.get("source_equipment_id", ""),
                sink_equipment_id=opp_data.get("sink_equipment_id"),
                available_heat_kw=opp_data.get("available_heat_kw", 0),
                recoverable_heat_kw=opp_data.get("recoverable_heat_kw", 0),
                source_temperature_c=opp_data.get("source_temperature_c", 0),
                sink_temperature_c=opp_data.get("sink_temperature_c"),
                annual_savings_kwh=opp_data.get("annual_savings_kwh", 0),
                annual_cost_savings=opp_data.get("annual_cost_savings", 0),
                implementation_cost=opp_data.get("implementation_cost"),
                payback_years=opp_data.get("payback_years"),
                insulation_improvement_required=opp_data.get(
                    "insulation_improvement_required", False
                ),
                related_insulation_defects=opp_data.get(
                    "related_insulation_defects", []
                ),
            ))

        return opportunities

    # =========================================================================
    # GL-014 EXCHANGER-PRO Integration
    # =========================================================================

    async def get_heat_exchanger_insulation_context(
        self,
        heat_exchanger_id: str
    ) -> Optional[HeatExchangerInsulationContext]:
        """
        Get heat exchanger insulation context from GL-014 EXCHANGER-PRO.

        Args:
            heat_exchanger_id: Heat exchanger identifier

        Returns:
            Heat exchanger insulation context or None
        """
        if AgentID.GL_014_EXCHANGER_PRO not in self._coordinator_config.agent_endpoints:
            self._logger.warning("GL-014 EXCHANGER-PRO not configured")
            return None

        response = await self.send_message(
            target_agent=AgentID.GL_014_EXCHANGER_PRO,
            action="get_insulation_context",
            payload={"heat_exchanger_id": heat_exchanger_id}
        )

        if not response or not response.success:
            self._logger.warning(
                f"Failed to get HX insulation context from GL-014: "
                f"{response.error_message if response else 'No response'}"
            )
            return None

        result = response.result
        return HeatExchangerInsulationContext(
            heat_exchanger_id=result.get("heat_exchanger_id", heat_exchanger_id),
            heat_exchanger_type=result.get("heat_exchanger_type", "unknown"),
            hot_side_temperature_c=result.get("hot_side_temperature_c", 0),
            cold_side_temperature_c=result.get("cold_side_temperature_c", 0),
            duty_kw=result.get("duty_kw", 0),
            design_duty_kw=result.get("design_duty_kw", 0),
            efficiency=result.get("efficiency", 0),
            is_insulated=result.get("is_insulated", True),
            insulation_thickness_mm=result.get("insulation_thickness_mm"),
            insulation_condition=result.get("insulation_condition"),
            shell_heat_loss_kw=result.get("shell_heat_loss_kw"),
            insulation_recommendation=result.get("insulation_recommendation"),
            potential_savings_kw=result.get("potential_savings_kw"),
        )

    async def report_heat_exchanger_insulation_issue(
        self,
        heat_exchanger_id: str,
        defect: InsulationDefectData
    ) -> bool:
        """
        Report insulation issue on heat exchanger to GL-014.

        Args:
            heat_exchanger_id: Heat exchanger ID
            defect: Insulation defect data

        Returns:
            True if reported successfully
        """
        if AgentID.GL_014_EXCHANGER_PRO not in self._coordinator_config.agent_endpoints:
            return False

        response = await self.send_message(
            target_agent=AgentID.GL_014_EXCHANGER_PRO,
            action="report_insulation_issue",
            payload={
                "heat_exchanger_id": heat_exchanger_id,
                "defect": defect.model_dump(mode="json")
            },
            message_type=MessageType.NOTIFICATION
        )

        return response is not None and response.success

    # =========================================================================
    # Event Handling
    # =========================================================================

    def subscribe_event(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        self._logger.info(f"Subscribed to event: {event_type}")

    def unsubscribe_event(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def handle_incoming_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Handle an incoming event."""
        handlers = self._event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                await handler(event_data)
            except Exception as e:
                self._logger.error(
                    f"Event handler error for {event_type}: {e}"
                )

    # =========================================================================
    # Agent Status
    # =========================================================================

    def get_connected_agents(self) -> List[AgentStatus]:
        """Get list of connected agents."""
        return list(self._agent_status.values())

    def get_agent_status(self, agent_id: AgentID) -> Optional[AgentStatus]:
        """Get status of specific agent."""
        return self._agent_status.get(agent_id)

    def is_agent_available(self, agent_id: AgentID) -> bool:
        """Check if agent is available."""
        status = self._agent_status.get(agent_id)
        return status is not None and status.health_status == HealthStatus.HEALTHY


# =============================================================================
# Factory Function
# =============================================================================


def create_agent_coordinator(
    connector_name: str = "AgentCoordinator",
    **kwargs
) -> AgentCoordinator:
    """
    Factory function to create agent coordinator.

    Args:
        connector_name: Connector name
        **kwargs: Additional configuration options

    Returns:
        Configured AgentCoordinator instance
    """
    config = AgentCoordinatorConfig(
        connector_name=connector_name,
        **kwargs
    )
    return AgentCoordinator(config)


# Convenience for configuring agent endpoints
def create_agent_endpoint(
    agent_id: AgentID,
    base_url: str,
    agent_name: Optional[str] = None,
    auth_token: Optional[str] = None,
    **kwargs
) -> AgentEndpointConfig:
    """Create an agent endpoint configuration."""
    return AgentEndpointConfig(
        agent_id=agent_id,
        agent_name=agent_name or agent_id.value,
        base_url=base_url,
        auth_token=auth_token,
        **kwargs
    )


# Import TimeoutError for async response waiting
from .base_connector import TimeoutError
