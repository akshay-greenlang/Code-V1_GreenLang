"""
ERP/MES Integration Module for GL-019 HEATSCHEDULER

Provides enterprise-grade connectivity to ERP and MES systems for production
planning, work order management, and maintenance schedule synchronization.

Supported Systems:
- SAP S/4HANA (RFC/BAPI interface)
- SAP ECC (RFC/BAPI interface)
- Oracle ERP Cloud (REST API)
- Oracle E-Business Suite (REST API)
- Generic MES systems (OPC UA, REST)

Features:
- Production schedule extraction
- Work order integration (create, update, status)
- Maintenance schedule synchronization
- Equipment master data retrieval
- Real-time schedule updates via webhooks/subscriptions

Protocols:
- RFC (SAP Remote Function Call)
- BAPI (SAP Business API)
- REST/HTTPS (Oracle, Generic)
- OPC UA (MES systems)

Author: GreenLang Data Integration Engineering Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
import logging
import time
import json
from collections import defaultdict, deque

from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ERPConnectionStatus(str, Enum):
    """ERP connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class SAPAuthMethod(str, Enum):
    """SAP authentication methods."""
    BASIC = "basic"
    SSO = "sso"
    X509 = "x509"
    OAUTH2 = "oauth2"


class OracleAuthMethod(str, Enum):
    """Oracle ERP authentication methods."""
    BASIC = "basic"
    OAUTH2 = "oauth2"
    JWT = "jwt"


class WorkOrderStatus(str, Enum):
    """Work order status values."""
    CREATED = "created"
    PLANNED = "planned"
    RELEASED = "released"
    IN_PROGRESS = "in_progress"
    PARTIALLY_COMPLETE = "partially_complete"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class MaintenanceType(str, Enum):
    """Maintenance types."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    EMERGENCY = "emergency"
    SCHEDULED = "scheduled"


class SchedulePriority(str, Enum):
    """Schedule priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ROUTINE = "routine"


class ERPSystem(str, Enum):
    """Supported ERP systems."""
    SAP_S4HANA = "sap_s4hana"
    SAP_ECC = "sap_ecc"
    ORACLE_CLOUD = "oracle_cloud"
    ORACLE_EBS = "oracle_ebs"
    GENERIC_MES = "generic_mes"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class ERPConfig(BaseModel):
    """Base configuration for ERP connectors."""

    model_config = ConfigDict(extra="forbid")

    # System identification
    erp_system: ERPSystem = Field(..., description="ERP system type")
    system_id: str = Field(..., min_length=1, description="System identifier")
    client_id: str = Field(default="100", description="SAP client or Oracle tenant")

    # Connection settings
    host: str = Field(..., description="ERP server host")
    port: int = Field(default=443, ge=1, le=65535, description="Server port")
    use_ssl: bool = Field(default=True, description="Use SSL/TLS")

    # Authentication
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password (use vault)")

    # Timeouts and retries
    connection_timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="Connection timeout seconds"
    )
    request_timeout: float = Field(
        default=60.0, ge=5.0, le=300.0, description="Request timeout seconds"
    )
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry attempts")
    retry_delay: float = Field(default=2.0, ge=0.5, le=30.0, description="Retry delay seconds")

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(
        default=100, ge=1, le=1000, description="Rate limit"
    )
    rate_limit_burst: int = Field(default=10, ge=1, le=100, description="Burst limit")

    # Connection pooling
    pool_size: int = Field(default=5, ge=1, le=20, description="Connection pool size")
    pool_timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="Pool timeout")

    # Circuit breaker
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=20, description="Failures before open"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Circuit timeout seconds"
    )

    # Caching
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600, description="Cache TTL")


class SAPConfig(ERPConfig):
    """SAP-specific configuration."""

    erp_system: ERPSystem = Field(default=ERPSystem.SAP_S4HANA)

    # SAP-specific settings
    sap_router: Optional[str] = Field(default=None, description="SAP Router string")
    system_number: str = Field(default="00", pattern=r"^\d{2}$", description="System number")
    auth_method: SAPAuthMethod = Field(default=SAPAuthMethod.BASIC)

    # RFC settings
    rfc_trace_level: int = Field(default=0, ge=0, le=3, description="RFC trace level")
    rfc_codepage: str = Field(default="UTF-8", description="RFC codepage")

    # Language
    language: str = Field(default="EN", pattern=r"^[A-Z]{2}$", description="Language code")

    # SAP gateway
    gateway_host: Optional[str] = Field(default=None, description="Gateway host")
    gateway_service: Optional[str] = Field(default=None, description="Gateway service")

    # X.509 certificate settings
    certificate_path: Optional[str] = Field(default=None, description="Certificate path")
    private_key_path: Optional[str] = Field(default=None, description="Private key path")

    # OAuth2 settings
    oauth_token_url: Optional[str] = Field(default=None, description="OAuth token URL")
    oauth_client_id: Optional[str] = Field(default=None, description="OAuth client ID")
    oauth_client_secret: Optional[str] = Field(default=None, description="OAuth client secret")


class OracleConfig(ERPConfig):
    """Oracle ERP Cloud configuration."""

    erp_system: ERPSystem = Field(default=ERPSystem.ORACLE_CLOUD)

    # Oracle-specific settings
    auth_method: OracleAuthMethod = Field(default=OracleAuthMethod.BASIC)
    base_url: Optional[HttpUrl] = Field(default=None, description="Base API URL")
    api_version: str = Field(default="v1", description="API version")

    # OAuth2 settings
    oauth_token_url: Optional[HttpUrl] = Field(default=None, description="OAuth token URL")
    oauth_client_id: Optional[str] = Field(default=None, description="OAuth client ID")
    oauth_client_secret: Optional[str] = Field(default=None, description="OAuth client secret")
    oauth_scope: Optional[str] = Field(default=None, description="OAuth scope")

    # JWT settings
    jwt_key_path: Optional[str] = Field(default=None, description="JWT private key path")
    jwt_issuer: Optional[str] = Field(default=None, description="JWT issuer")

    # API modules
    manufacturing_module: str = Field(default="manufacturingWorkOrders", description="Manufacturing API")
    maintenance_module: str = Field(default="maintenanceWorkOrders", description="Maintenance API")
    planning_module: str = Field(default="supplyChainPlanning", description="Planning API")


# =============================================================================
# Pydantic Models - Data
# =============================================================================


class ScheduleItem(BaseModel):
    """Individual schedule item (operation or task)."""

    model_config = ConfigDict(extra="allow")

    item_id: str = Field(..., description="Schedule item ID")
    sequence: int = Field(default=0, ge=0, description="Sequence number")

    # Timing
    planned_start: datetime = Field(..., description="Planned start time")
    planned_end: datetime = Field(..., description="Planned end time")
    actual_start: Optional[datetime] = Field(default=None, description="Actual start")
    actual_end: Optional[datetime] = Field(default=None, description="Actual end")

    # Duration
    planned_duration_hours: float = Field(default=0.0, ge=0, description="Planned duration")
    actual_duration_hours: Optional[float] = Field(default=None, description="Actual duration")

    # Resource requirements
    equipment_id: Optional[str] = Field(default=None, description="Equipment ID")
    work_center: Optional[str] = Field(default=None, description="Work center")
    resource_type: Optional[str] = Field(default=None, description="Resource type")

    # Energy requirements
    estimated_power_kw: Optional[float] = Field(default=None, description="Estimated power (kW)")
    estimated_energy_kwh: Optional[float] = Field(default=None, description="Estimated energy (kWh)")
    heating_required: bool = Field(default=False, description="Heating operation required")
    temperature_setpoint_c: Optional[float] = Field(default=None, description="Temperature setpoint")

    # Status
    status: str = Field(default="planned", description="Item status")
    progress_percent: float = Field(default=0.0, ge=0, le=100, description="Progress")

    description: Optional[str] = Field(default=None, description="Description")
    notes: Optional[str] = Field(default=None, description="Notes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProductionSchedule(BaseModel):
    """Production schedule from ERP/MES."""

    model_config = ConfigDict(extra="allow")

    schedule_id: str = Field(..., description="Schedule ID")
    schedule_name: Optional[str] = Field(default=None, description="Schedule name")
    version: int = Field(default=1, ge=1, description="Schedule version")

    # Timing
    schedule_date: datetime = Field(..., description="Schedule date")
    horizon_start: datetime = Field(..., description="Planning horizon start")
    horizon_end: datetime = Field(..., description="Planning horizon end")

    # Plant/location
    plant_code: str = Field(..., description="Plant code")
    plant_name: Optional[str] = Field(default=None, description="Plant name")
    area_code: Optional[str] = Field(default=None, description="Production area")

    # Schedule items
    items: List[ScheduleItem] = Field(default_factory=list, description="Schedule items")

    # Aggregations
    total_operations: int = Field(default=0, ge=0, description="Total operations")
    heating_operations: int = Field(default=0, ge=0, description="Heating operations")
    total_energy_kwh: float = Field(default=0.0, ge=0, description="Total energy estimate")
    peak_power_kw: float = Field(default=0.0, ge=0, description="Peak power estimate")

    # Status
    status: str = Field(default="active", description="Schedule status")
    frozen: bool = Field(default=False, description="Schedule frozen (locked)")

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Created timestamp"
    )
    modified_at: Optional[datetime] = Field(default=None, description="Modified timestamp")
    created_by: Optional[str] = Field(default=None, description="Created by user")
    source_system: Optional[str] = Field(default=None, description="Source ERP system")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkOrder(BaseModel):
    """Work order from ERP/MES."""

    model_config = ConfigDict(extra="allow")

    order_id: str = Field(..., description="Work order ID")
    order_number: str = Field(..., description="Work order number")
    order_type: str = Field(default="production", description="Order type")

    # Status
    status: WorkOrderStatus = Field(default=WorkOrderStatus.CREATED)
    priority: SchedulePriority = Field(default=SchedulePriority.MEDIUM)

    # Product information
    material_number: Optional[str] = Field(default=None, description="Material number")
    material_description: Optional[str] = Field(default=None, description="Material description")
    quantity_planned: float = Field(default=0.0, ge=0, description="Planned quantity")
    quantity_completed: float = Field(default=0.0, ge=0, description="Completed quantity")
    unit_of_measure: str = Field(default="EA", description="Unit of measure")

    # Timing
    basic_start_date: datetime = Field(..., description="Basic start date")
    basic_end_date: datetime = Field(..., description="Basic end date")
    scheduled_start: Optional[datetime] = Field(default=None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(default=None, description="Scheduled end")
    actual_start: Optional[datetime] = Field(default=None, description="Actual start")
    actual_end: Optional[datetime] = Field(default=None, description="Actual end")

    # Location
    plant_code: str = Field(..., description="Plant code")
    work_center: Optional[str] = Field(default=None, description="Work center")
    production_line: Optional[str] = Field(default=None, description="Production line")

    # Operations
    operations: List[ScheduleItem] = Field(default_factory=list, description="Operations")
    current_operation: Optional[str] = Field(default=None, description="Current operation")

    # Energy requirements
    estimated_energy_kwh: Optional[float] = Field(default=None, description="Estimated energy")
    heating_operations_count: int = Field(default=0, ge=0, description="Heating operations")
    max_temperature_c: Optional[float] = Field(default=None, description="Max temperature")

    # References
    sales_order: Optional[str] = Field(default=None, description="Sales order reference")
    customer: Optional[str] = Field(default=None, description="Customer")

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Created timestamp"
    )
    modified_at: Optional[datetime] = Field(default=None, description="Modified timestamp")
    source_system: Optional[str] = Field(default=None, description="Source ERP system")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MaintenanceSchedule(BaseModel):
    """Maintenance schedule from ERP/CMMS."""

    model_config = ConfigDict(extra="allow")

    schedule_id: str = Field(..., description="Schedule ID")
    maintenance_type: MaintenanceType = Field(default=MaintenanceType.PREVENTIVE)

    # Equipment
    equipment_id: str = Field(..., description="Equipment ID")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")
    functional_location: Optional[str] = Field(default=None, description="Functional location")

    # Timing
    planned_start: datetime = Field(..., description="Planned start")
    planned_end: datetime = Field(..., description="Planned end")
    actual_start: Optional[datetime] = Field(default=None, description="Actual start")
    actual_end: Optional[datetime] = Field(default=None, description="Actual end")
    duration_hours: float = Field(default=1.0, ge=0, description="Duration hours")

    # Status
    status: str = Field(default="planned", description="Status")
    priority: SchedulePriority = Field(default=SchedulePriority.MEDIUM)

    # Impact on production
    production_impact: bool = Field(default=True, description="Impacts production")
    equipment_unavailable: bool = Field(default=True, description="Equipment unavailable")
    power_shutdown_required: bool = Field(default=False, description="Power shutdown needed")
    heating_equipment: bool = Field(default=False, description="Is heating equipment")

    # Work details
    work_order_number: Optional[str] = Field(default=None, description="Work order number")
    task_description: Optional[str] = Field(default=None, description="Task description")
    technician: Optional[str] = Field(default=None, description="Assigned technician")

    # Location
    plant_code: str = Field(..., description="Plant code")
    area: Optional[str] = Field(default=None, description="Plant area")

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Created timestamp"
    )
    source_system: Optional[str] = Field(default=None, description="Source system")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing if system recovered
    """

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.CircuitBreaker")

    @property
    def state(self) -> "CircuitBreaker.State":
        """Get current circuit state."""
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            if self._state == self.State.CLOSED:
                return True

            if self._state == self.State.OPEN:
                # Check if recovery timeout elapsed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = self.State.HALF_OPEN
                        self._half_open_calls = 0
                        self._logger.info("Circuit breaker moving to HALF_OPEN")
                        return True
                return False

            if self._state == self.State.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state == self.State.HALF_OPEN:
                self._state = self.State.CLOSED
                self._failure_count = 0
                self._logger.info("Circuit breaker CLOSED (recovered)")
            elif self._state == self.State.CLOSED:
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.State.HALF_OPEN:
                self._state = self.State.OPEN
                self._logger.warning("Circuit breaker OPEN (half-open test failed)")
            elif self._state == self.State.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.State.OPEN
                    self._logger.warning(
                        f"Circuit breaker OPEN (failures: {self._failure_count})"
                    )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


# =============================================================================
# Rate Limiter Implementation
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter.

    Limits requests to a specified rate with burst capacity.
    """

    def __init__(self, requests_per_minute: int = 100, burst: int = 10):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token, waiting if necessary.

        Args:
            timeout: Maximum time to wait for a token

        Returns:
            True if token acquired, False if timeout
        """
        deadline = time.time() + timeout

        while True:
            async with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

            if time.time() >= deadline:
                return False

            # Wait for next token
            wait_time = min(1.0 / self.rate, deadline - time.time())
            if wait_time > 0:
                await asyncio.sleep(wait_time)


# =============================================================================
# Abstract Base Class
# =============================================================================


class ERPConnectorBase(ABC):
    """
    Abstract base class for ERP connectors.

    Provides common functionality:
    - Connection management
    - Authentication
    - Rate limiting
    - Circuit breaker
    - Retry logic
    - Caching
    """

    def __init__(self, config: ERPConfig) -> None:
        """Initialize ERP connector."""
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Connection state
        self._status = ERPConnectionStatus.DISCONNECTED
        self._client: Any = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # Circuit breaker
        if config.circuit_breaker_enabled:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout
            )
        else:
            self._circuit_breaker = None

        # Rate limiter
        self._rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_requests_per_minute,
            burst=config.rate_limit_burst
        )

        # Response cache
        self._cache: Dict[str, Tuple[Any, float]] = {}

        # Statistics
        self._stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "cache_hits": 0,
            "rate_limited": 0,
            "circuit_breaks": 0,
            "last_request": None,
            "last_error": None,
        }

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    async def _connect(self) -> bool:
        """Establish connection to ERP system."""
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """Disconnect from ERP system."""
        pass

    @abstractmethod
    async def _authenticate(self) -> bool:
        """Authenticate with ERP system."""
        pass

    @abstractmethod
    async def _fetch_production_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[ProductionSchedule]:
        """Fetch production schedules from ERP."""
        pass

    @abstractmethod
    async def _fetch_work_orders(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        status_filter: Optional[List[WorkOrderStatus]] = None,
        **kwargs
    ) -> List[WorkOrder]:
        """Fetch work orders from ERP."""
        pass

    @abstractmethod
    async def _fetch_maintenance_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        equipment_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[MaintenanceSchedule]:
        """Fetch maintenance schedules from ERP."""
        pass

    @abstractmethod
    async def _update_work_order_status(
        self,
        order_id: str,
        new_status: WorkOrderStatus,
        **kwargs
    ) -> bool:
        """Update work order status in ERP."""
        pass

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish connection to ERP system.

        Returns:
            True if connection successful
        """
        try:
            self._status = ERPConnectionStatus.CONNECTING
            self._logger.info(
                f"Connecting to {self._config.erp_system.value}: "
                f"{self._config.host}:{self._config.port}"
            )

            # Connect
            if await self._connect():
                self._status = ERPConnectionStatus.CONNECTED
                self._logger.info("ERP connection established")

                # Authenticate
                if await self._authenticate():
                    self._status = ERPConnectionStatus.AUTHENTICATED
                    self._logger.info("ERP authentication successful")
                    self._reconnect_attempts = 0
                    return True
                else:
                    self._status = ERPConnectionStatus.ERROR
                    self._logger.error("ERP authentication failed")
                    return False
            else:
                self._status = ERPConnectionStatus.ERROR
                self._logger.error("ERP connection failed")
                return False

        except Exception as e:
            self._status = ERPConnectionStatus.ERROR
            self._logger.error(f"Connection error: {e}")
            self._stats["last_error"] = str(e)

            # Schedule reconnection
            await self._schedule_reconnection()
            return False

    async def disconnect(self) -> None:
        """Disconnect from ERP system."""
        self._logger.info("Disconnecting from ERP...")

        # Cancel reconnection task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        # Disconnect
        await self._disconnect()
        self._status = ERPConnectionStatus.DISCONNECTED
        self._logger.info("ERP disconnected")

    async def _schedule_reconnection(self) -> None:
        """Schedule automatic reconnection."""
        if self._reconnect_task and not self._reconnect_task.done():
            return

        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Background task for reconnection attempts."""
        max_attempts = self._config.max_retries * 3  # More attempts for reconnection

        while self._reconnect_attempts < max_attempts:
            self._reconnect_attempts += 1
            self._status = ERPConnectionStatus.RECONNECTING

            # Exponential backoff
            wait_time = min(
                self._config.retry_delay * (2 ** (self._reconnect_attempts - 1)),
                60.0  # Max 60 seconds
            )

            self._logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/{max_attempts} "
                f"in {wait_time:.1f}s"
            )

            await asyncio.sleep(wait_time)

            if await self.connect():
                self._logger.info("Reconnection successful")
                return

        self._logger.error("Max reconnection attempts reached")
        self._status = ERPConnectionStatus.ERROR

    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._status == ERPConnectionStatus.AUTHENTICATED

    @property
    def status(self) -> ERPConnectionStatus:
        """Get current connection status."""
        return self._status

    # =========================================================================
    # Request Execution with Resilience
    # =========================================================================

    async def _execute_with_resilience(
        self,
        operation: Callable,
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute operation with circuit breaker, rate limiting, and retry logic.

        Args:
            operation: Async function to execute
            *args: Positional arguments
            cache_key: Optional cache key for response caching
            **kwargs: Keyword arguments

        Returns:
            Operation result

        Raises:
            Exception: If all retries fail
        """
        # Check cache
        if cache_key and self._config.cache_enabled:
            cached = self._get_cached(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached

        # Check circuit breaker
        if self._circuit_breaker:
            if not await self._circuit_breaker.can_execute():
                self._stats["circuit_breaks"] += 1
                raise Exception("Circuit breaker is open - service unavailable")

        # Rate limiting
        if not await self._rate_limiter.acquire(timeout=30.0):
            self._stats["rate_limited"] += 1
            raise Exception("Rate limit exceeded")

        # Retry loop
        last_exception = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                self._stats["requests"] += 1
                self._stats["last_request"] = datetime.now(timezone.utc)

                # Execute operation
                result = await operation(*args, **kwargs)

                # Record success
                self._stats["successes"] += 1
                if self._circuit_breaker:
                    await self._circuit_breaker.record_success()

                # Cache result
                if cache_key and self._config.cache_enabled:
                    self._set_cached(cache_key, result)

                return result

            except Exception as e:
                last_exception = e
                self._stats["failures"] += 1
                self._stats["last_error"] = str(e)

                if self._circuit_breaker:
                    await self._circuit_breaker.record_failure()

                if attempt < self._config.max_retries:
                    self._stats["retries"] += 1
                    wait_time = self._config.retry_delay * (2 ** (attempt - 1))
                    self._logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)

        raise last_exception or Exception("Operation failed after all retries")

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, cache_time = self._cache[key]
            if time.time() - cache_time < self._config.cache_ttl_seconds:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, time.time())

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def get_production_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        include_heating_only: bool = False,
        **kwargs
    ) -> List[ProductionSchedule]:
        """
        Retrieve production schedules from ERP.

        Args:
            plant_code: Plant/facility code
            start_date: Start of planning horizon
            end_date: End of planning horizon
            include_heating_only: Filter to heating operations only
            **kwargs: Additional parameters

        Returns:
            List of production schedules
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to ERP system")

        cache_key = f"schedules:{plant_code}:{start_date.date()}:{end_date.date()}"

        schedules = await self._execute_with_resilience(
            self._fetch_production_schedules,
            plant_code,
            start_date,
            end_date,
            cache_key=cache_key,
            **kwargs
        )

        if include_heating_only:
            for schedule in schedules:
                schedule.items = [
                    item for item in schedule.items if item.heating_required
                ]

        return schedules

    async def get_work_orders(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        status_filter: Optional[List[WorkOrderStatus]] = None,
        heating_operations_only: bool = False,
        **kwargs
    ) -> List[WorkOrder]:
        """
        Retrieve work orders from ERP.

        Args:
            plant_code: Plant/facility code
            start_date: Start date
            end_date: End date
            status_filter: Optional status filter
            heating_operations_only: Filter to orders with heating operations
            **kwargs: Additional parameters

        Returns:
            List of work orders
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to ERP system")

        cache_key = f"workorders:{plant_code}:{start_date.date()}:{end_date.date()}"

        work_orders = await self._execute_with_resilience(
            self._fetch_work_orders,
            plant_code,
            start_date,
            end_date,
            status_filter,
            cache_key=cache_key,
            **kwargs
        )

        if heating_operations_only:
            work_orders = [
                wo for wo in work_orders if wo.heating_operations_count > 0
            ]

        return work_orders

    async def get_maintenance_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        equipment_ids: Optional[List[str]] = None,
        heating_equipment_only: bool = False,
        **kwargs
    ) -> List[MaintenanceSchedule]:
        """
        Retrieve maintenance schedules from ERP/CMMS.

        Args:
            plant_code: Plant/facility code
            start_date: Start date
            end_date: End date
            equipment_ids: Optional equipment ID filter
            heating_equipment_only: Filter to heating equipment only
            **kwargs: Additional parameters

        Returns:
            List of maintenance schedules
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to ERP system")

        cache_key = f"maintenance:{plant_code}:{start_date.date()}:{end_date.date()}"

        schedules = await self._execute_with_resilience(
            self._fetch_maintenance_schedules,
            plant_code,
            start_date,
            end_date,
            equipment_ids,
            cache_key=cache_key,
            **kwargs
        )

        if heating_equipment_only:
            schedules = [s for s in schedules if s.heating_equipment]

        return schedules

    async def update_work_order(
        self,
        order_id: str,
        new_status: WorkOrderStatus,
        **kwargs
    ) -> bool:
        """
        Update work order status in ERP.

        Args:
            order_id: Work order ID
            new_status: New status value
            **kwargs: Additional parameters

        Returns:
            True if update successful
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to ERP system")

        result = await self._execute_with_resilience(
            self._update_work_order_status,
            order_id,
            new_status,
            **kwargs
        )

        # Invalidate related cache
        self.clear_cache()

        return result

    # =========================================================================
    # Statistics and Health
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "status": self._status.value,
            "erp_system": self._config.erp_system.value,
            "cache_size": len(self._cache),
            "circuit_breaker_state": (
                self._circuit_breaker.state.value if self._circuit_breaker else "disabled"
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self.is_connected(),
            "status": self._status.value,
            "erp_system": self._config.erp_system.value,
            "host": self._config.host,
            "statistics": self.get_statistics(),
        }


# =============================================================================
# SAP ERP Connector Implementation
# =============================================================================


class SAPERPConnector(ERPConnectorBase):
    """
    SAP ERP Connector for S/4HANA and ECC systems.

    Supports:
    - RFC/BAPI calls for direct SAP integration
    - Production order retrieval (PP module)
    - Maintenance order retrieval (PM module)
    - Work center and routing data
    """

    def __init__(self, config: SAPConfig) -> None:
        """Initialize SAP connector."""
        super().__init__(config)
        self._sap_config = config
        self._rfc_connection: Any = None

    async def _connect(self) -> bool:
        """Establish SAP RFC connection."""
        try:
            # Import pyrfc (optional dependency)
            try:
                from pyrfc import Connection
            except ImportError:
                self._logger.warning(
                    "pyrfc not installed. Using REST API fallback. "
                    "Install pyrfc for RFC/BAPI access: pip install pyrfc"
                )
                return await self._connect_rest()

            # Build connection parameters
            conn_params = {
                "ashost": self._sap_config.host,
                "sysnr": self._sap_config.system_number,
                "client": self._sap_config.client_id,
                "user": self._sap_config.username,
                "passwd": self._sap_config.password,
                "lang": self._sap_config.language,
            }

            # Add SAP router if specified
            if self._sap_config.sap_router:
                conn_params["saprouter"] = self._sap_config.sap_router

            # Create connection (synchronous, wrap in executor)
            loop = asyncio.get_event_loop()
            self._rfc_connection = await loop.run_in_executor(
                None, lambda: Connection(**conn_params)
            )

            return True

        except Exception as e:
            self._logger.error(f"SAP RFC connection failed: {e}")
            return False

    async def _connect_rest(self) -> bool:
        """Fallback REST API connection for SAP."""
        try:
            import httpx

            base_url = f"https://{self._sap_config.host}:{self._sap_config.port}"

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._sap_config.request_timeout,
                verify=self._sap_config.use_ssl,
            )

            return True

        except Exception as e:
            self._logger.error(f"SAP REST connection failed: {e}")
            return False

    async def _disconnect(self) -> None:
        """Disconnect from SAP."""
        if self._rfc_connection:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._rfc_connection.close)
            except Exception as e:
                self._logger.warning(f"Error closing RFC connection: {e}")
            self._rfc_connection = None

        if self._client:
            await self._client.aclose()
            self._client = None

    async def _authenticate(self) -> bool:
        """Authenticate with SAP (implicit in RFC connection)."""
        # RFC connection includes authentication
        if self._rfc_connection:
            return True

        # REST API authentication
        if self._client and self._sap_config.auth_method == SAPAuthMethod.OAUTH2:
            return await self._authenticate_oauth2()

        return self._client is not None

    async def _authenticate_oauth2(self) -> bool:
        """Authenticate via OAuth2."""
        try:
            if not self._sap_config.oauth_token_url:
                return False

            response = await self._client.post(
                self._sap_config.oauth_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._sap_config.oauth_client_id,
                    "client_secret": self._sap_config.oauth_client_secret,
                }
            )
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            return True

        except Exception as e:
            self._logger.error(f"SAP OAuth2 authentication failed: {e}")
            return False

    async def _fetch_production_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[ProductionSchedule]:
        """Fetch production schedules from SAP PP module."""
        schedules = []

        if self._rfc_connection:
            # Use RFC/BAPI
            schedules = await self._fetch_schedules_rfc(plant_code, start_date, end_date)
        else:
            # Use REST API
            schedules = await self._fetch_schedules_rest(plant_code, start_date, end_date)

        return schedules

    async def _fetch_schedules_rfc(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ProductionSchedule]:
        """Fetch schedules via RFC/BAPI."""
        try:
            loop = asyncio.get_event_loop()

            # Call BAPI_PRODORD_GET_LIST
            result = await loop.run_in_executor(
                None,
                lambda: self._rfc_connection.call(
                    "BAPI_PRODORD_GET_LIST",
                    PLANT=plant_code,
                    BASIC_START_DATE_FROM=start_date.strftime("%Y%m%d"),
                    BASIC_START_DATE_TO=end_date.strftime("%Y%m%d"),
                )
            )

            orders = result.get("ORDER_LIST", [])

            # Convert to schedule
            schedule = ProductionSchedule(
                schedule_id=f"SAP-{plant_code}-{start_date.strftime('%Y%m%d')}",
                schedule_date=datetime.now(timezone.utc),
                horizon_start=start_date,
                horizon_end=end_date,
                plant_code=plant_code,
                source_system="SAP",
                items=[],
            )

            for order in orders:
                item = ScheduleItem(
                    item_id=order.get("ORDER_NUMBER", ""),
                    planned_start=datetime.strptime(
                        order.get("BASIC_START_DATE", "19700101"), "%Y%m%d"
                    ).replace(tzinfo=timezone.utc),
                    planned_end=datetime.strptime(
                        order.get("BASIC_END_DATE", "19700101"), "%Y%m%d"
                    ).replace(tzinfo=timezone.utc),
                    equipment_id=order.get("WORK_CENTER", ""),
                    work_center=order.get("WORK_CENTER", ""),
                    status=order.get("SYSTEM_STATUS", "planned"),
                    description=order.get("MATERIAL_DESCRIPTION", ""),
                )
                schedule.items.append(item)

            schedule.total_operations = len(schedule.items)
            return [schedule]

        except Exception as e:
            self._logger.error(f"SAP RFC schedule fetch failed: {e}")
            raise

    async def _fetch_schedules_rest(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ProductionSchedule]:
        """Fetch schedules via REST API (OData)."""
        try:
            headers = {}
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"

            # SAP OData endpoint for production orders
            url = "/sap/opu/odata/sap/API_PRODUCTION_ORDER_2_SRV/A_ProductionOrder"
            params = {
                "$filter": (
                    f"ProductionPlant eq '{plant_code}' and "
                    f"MfgOrderPlannedStartDate ge datetime'{start_date.isoformat()}' and "
                    f"MfgOrderPlannedStartDate le datetime'{end_date.isoformat()}'"
                ),
                "$format": "json",
            }

            response = await self._client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            orders = data.get("d", {}).get("results", [])

            schedule = ProductionSchedule(
                schedule_id=f"SAP-{plant_code}-{start_date.strftime('%Y%m%d')}",
                schedule_date=datetime.now(timezone.utc),
                horizon_start=start_date,
                horizon_end=end_date,
                plant_code=plant_code,
                source_system="SAP",
                items=[],
            )

            for order in orders:
                item = ScheduleItem(
                    item_id=order.get("ManufacturingOrder", ""),
                    planned_start=datetime.fromisoformat(
                        order.get("MfgOrderPlannedStartDate", "").replace("Z", "+00:00")
                    ),
                    planned_end=datetime.fromisoformat(
                        order.get("MfgOrderPlannedEndDate", "").replace("Z", "+00:00")
                    ),
                    work_center=order.get("ProductionWorkCenter", ""),
                    status=order.get("ManufacturingOrderStatus", "planned"),
                )
                schedule.items.append(item)

            schedule.total_operations = len(schedule.items)
            return [schedule]

        except Exception as e:
            self._logger.error(f"SAP REST schedule fetch failed: {e}")
            raise

    async def _fetch_work_orders(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        status_filter: Optional[List[WorkOrderStatus]] = None,
        **kwargs
    ) -> List[WorkOrder]:
        """Fetch work orders from SAP."""
        work_orders = []

        try:
            if self._rfc_connection:
                # RFC call for production orders
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._rfc_connection.call(
                        "BAPI_PRODORD_GET_LIST",
                        PLANT=plant_code,
                        BASIC_START_DATE_FROM=start_date.strftime("%Y%m%d"),
                        BASIC_START_DATE_TO=end_date.strftime("%Y%m%d"),
                    )
                )

                for order in result.get("ORDER_LIST", []):
                    wo = WorkOrder(
                        order_id=order.get("ORDER_NUMBER", ""),
                        order_number=order.get("ORDER_NUMBER", ""),
                        order_type="production",
                        status=self._map_sap_status(order.get("SYSTEM_STATUS", "")),
                        material_number=order.get("MATERIAL", ""),
                        material_description=order.get("MATERIAL_DESCRIPTION", ""),
                        quantity_planned=float(order.get("TARGET_QUANTITY", 0)),
                        quantity_completed=float(order.get("CONFIRMED_QUANTITY", 0)),
                        basic_start_date=datetime.strptime(
                            order.get("BASIC_START_DATE", "19700101"), "%Y%m%d"
                        ).replace(tzinfo=timezone.utc),
                        basic_end_date=datetime.strptime(
                            order.get("BASIC_END_DATE", "19700101"), "%Y%m%d"
                        ).replace(tzinfo=timezone.utc),
                        plant_code=plant_code,
                        work_center=order.get("WORK_CENTER", ""),
                        source_system="SAP",
                    )
                    work_orders.append(wo)

        except Exception as e:
            self._logger.error(f"SAP work order fetch failed: {e}")
            raise

        # Apply status filter
        if status_filter:
            work_orders = [wo for wo in work_orders if wo.status in status_filter]

        return work_orders

    async def _fetch_maintenance_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        equipment_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[MaintenanceSchedule]:
        """Fetch maintenance schedules from SAP PM module."""
        schedules = []

        try:
            if self._rfc_connection:
                loop = asyncio.get_event_loop()

                # Call PM BAPI for maintenance orders
                result = await loop.run_in_executor(
                    None,
                    lambda: self._rfc_connection.call(
                        "BAPI_ALM_ORDER_GET_LIST",
                        PLANT=plant_code,
                        DATE_FROM=start_date.strftime("%Y%m%d"),
                        DATE_TO=end_date.strftime("%Y%m%d"),
                    )
                )

                for order in result.get("ORDER_LIST", []):
                    schedule = MaintenanceSchedule(
                        schedule_id=order.get("ORDERID", ""),
                        maintenance_type=MaintenanceType.PREVENTIVE,
                        equipment_id=order.get("EQUIPMENT", ""),
                        equipment_name=order.get("EQUIPMENT_DESCRIPTION", ""),
                        planned_start=datetime.strptime(
                            order.get("START_DATE", "19700101"), "%Y%m%d"
                        ).replace(tzinfo=timezone.utc),
                        planned_end=datetime.strptime(
                            order.get("END_DATE", "19700101"), "%Y%m%d"
                        ).replace(tzinfo=timezone.utc),
                        plant_code=plant_code,
                        work_order_number=order.get("ORDERID", ""),
                        task_description=order.get("SHORT_TEXT", ""),
                        source_system="SAP",
                    )
                    schedules.append(schedule)

        except Exception as e:
            self._logger.error(f"SAP maintenance schedule fetch failed: {e}")
            raise

        # Filter by equipment IDs
        if equipment_ids:
            schedules = [s for s in schedules if s.equipment_id in equipment_ids]

        return schedules

    async def _update_work_order_status(
        self,
        order_id: str,
        new_status: WorkOrderStatus,
        **kwargs
    ) -> bool:
        """Update work order status in SAP."""
        try:
            if self._rfc_connection:
                loop = asyncio.get_event_loop()

                # Map status to SAP status
                sap_status = self._map_status_to_sap(new_status)

                result = await loop.run_in_executor(
                    None,
                    lambda: self._rfc_connection.call(
                        "BAPI_PRODORD_CHANGE",
                        ORDERID=order_id,
                        ORDERDATA={"SYSTEM_STATUS": sap_status},
                    )
                )

                # Check for errors
                messages = result.get("RETURN", [])
                for msg in messages:
                    if msg.get("TYPE") == "E":
                        self._logger.error(f"SAP error: {msg.get('MESSAGE')}")
                        return False

                return True

        except Exception as e:
            self._logger.error(f"SAP work order update failed: {e}")
            raise

        return False

    def _map_sap_status(self, sap_status: str) -> WorkOrderStatus:
        """Map SAP system status to WorkOrderStatus."""
        status_map = {
            "CRTD": WorkOrderStatus.CREATED,
            "REL": WorkOrderStatus.RELEASED,
            "PCNF": WorkOrderStatus.PARTIALLY_COMPLETE,
            "CNF": WorkOrderStatus.COMPLETED,
            "TECO": WorkOrderStatus.COMPLETED,
            "DLT": WorkOrderStatus.CANCELLED,
        }
        return status_map.get(sap_status.upper(), WorkOrderStatus.PLANNED)

    def _map_status_to_sap(self, status: WorkOrderStatus) -> str:
        """Map WorkOrderStatus to SAP system status."""
        status_map = {
            WorkOrderStatus.CREATED: "CRTD",
            WorkOrderStatus.PLANNED: "CRTD",
            WorkOrderStatus.RELEASED: "REL",
            WorkOrderStatus.IN_PROGRESS: "REL",
            WorkOrderStatus.PARTIALLY_COMPLETE: "PCNF",
            WorkOrderStatus.COMPLETED: "CNF",
            WorkOrderStatus.CANCELLED: "DLT",
        }
        return status_map.get(status, "CRTD")


# =============================================================================
# Oracle ERP Cloud Connector Implementation
# =============================================================================


class OracleERPConnector(ERPConnectorBase):
    """
    Oracle ERP Cloud Connector.

    Supports:
    - Oracle ERP Cloud REST APIs
    - Manufacturing work orders
    - Maintenance management
    - Supply chain planning
    """

    def __init__(self, config: OracleConfig) -> None:
        """Initialize Oracle connector."""
        super().__init__(config)
        self._oracle_config = config

    async def _connect(self) -> bool:
        """Establish connection to Oracle ERP Cloud."""
        try:
            import httpx

            base_url = str(self._oracle_config.base_url) if self._oracle_config.base_url else (
                f"https://{self._oracle_config.host}:{self._oracle_config.port}"
            )

            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self._oracle_config.request_timeout,
                verify=self._oracle_config.use_ssl,
            )

            return True

        except Exception as e:
            self._logger.error(f"Oracle connection failed: {e}")
            return False

    async def _disconnect(self) -> None:
        """Disconnect from Oracle ERP Cloud."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _authenticate(self) -> bool:
        """Authenticate with Oracle ERP Cloud."""
        if self._oracle_config.auth_method == OracleAuthMethod.BASIC:
            return True  # Basic auth is passed with each request

        elif self._oracle_config.auth_method == OracleAuthMethod.OAUTH2:
            return await self._authenticate_oauth2()

        elif self._oracle_config.auth_method == OracleAuthMethod.JWT:
            return await self._authenticate_jwt()

        return False

    async def _authenticate_oauth2(self) -> bool:
        """Authenticate via OAuth2."""
        try:
            if not self._oracle_config.oauth_token_url:
                return False

            response = await self._client.post(
                str(self._oracle_config.oauth_token_url),
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._oracle_config.oauth_client_id,
                    "client_secret": self._oracle_config.oauth_client_secret,
                    "scope": self._oracle_config.oauth_scope or "",
                }
            )
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            return True

        except Exception as e:
            self._logger.error(f"Oracle OAuth2 authentication failed: {e}")
            return False

    async def _authenticate_jwt(self) -> bool:
        """Authenticate via JWT."""
        try:
            import jwt

            if not self._oracle_config.jwt_key_path:
                return False

            with open(self._oracle_config.jwt_key_path, "r") as f:
                private_key = f.read()

            now = datetime.now(timezone.utc)
            payload = {
                "iss": self._oracle_config.jwt_issuer,
                "sub": self._oracle_config.username,
                "aud": f"https://{self._oracle_config.host}",
                "iat": now,
                "exp": now + timedelta(hours=1),
            }

            self._access_token = jwt.encode(payload, private_key, algorithm="RS256")
            self._token_expires_at = now + timedelta(hours=1)

            return True

        except Exception as e:
            self._logger.error(f"Oracle JWT authentication failed: {e}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        headers = {"Content-Type": "application/json"}

        if self._oracle_config.auth_method == OracleAuthMethod.BASIC:
            import base64
            credentials = base64.b64encode(
                f"{self._oracle_config.username}:{self._oracle_config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        else:
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    async def _fetch_production_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[ProductionSchedule]:
        """Fetch production schedules from Oracle Manufacturing."""
        try:
            headers = self._get_auth_headers()

            # Oracle Manufacturing Work Orders API
            url = f"/fscmRestApi/resources/latest/{self._oracle_config.manufacturing_module}"
            params = {
                "q": (
                    f"PlannedStartDate >= '{start_date.strftime('%Y-%m-%d')}' and "
                    f"PlannedStartDate <= '{end_date.strftime('%Y-%m-%d')}' and "
                    f"OrganizationCode = '{plant_code}'"
                ),
                "limit": 500,
            }

            response = await self._client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            schedule = ProductionSchedule(
                schedule_id=f"ORACLE-{plant_code}-{start_date.strftime('%Y%m%d')}",
                schedule_date=datetime.now(timezone.utc),
                horizon_start=start_date,
                horizon_end=end_date,
                plant_code=plant_code,
                source_system="Oracle",
                items=[],
            )

            for item in items:
                schedule_item = ScheduleItem(
                    item_id=item.get("WorkOrderNumber", ""),
                    planned_start=datetime.fromisoformat(
                        item.get("PlannedStartDate", "").replace("Z", "+00:00")
                    ),
                    planned_end=datetime.fromisoformat(
                        item.get("PlannedCompletionDate", "").replace("Z", "+00:00")
                    ),
                    work_center=item.get("WorkArea", ""),
                    status=item.get("WorkOrderStatus", "planned"),
                    description=item.get("WorkOrderDescription", ""),
                )
                schedule.items.append(schedule_item)

            schedule.total_operations = len(schedule.items)
            return [schedule]

        except Exception as e:
            self._logger.error(f"Oracle schedule fetch failed: {e}")
            raise

    async def _fetch_work_orders(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        status_filter: Optional[List[WorkOrderStatus]] = None,
        **kwargs
    ) -> List[WorkOrder]:
        """Fetch work orders from Oracle Manufacturing."""
        work_orders = []

        try:
            headers = self._get_auth_headers()

            url = f"/fscmRestApi/resources/latest/{self._oracle_config.manufacturing_module}"
            params = {
                "q": (
                    f"PlannedStartDate >= '{start_date.strftime('%Y-%m-%d')}' and "
                    f"PlannedStartDate <= '{end_date.strftime('%Y-%m-%d')}' and "
                    f"OrganizationCode = '{plant_code}'"
                ),
                "limit": 500,
            }

            response = await self._client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            for item in items:
                wo = WorkOrder(
                    order_id=str(item.get("WorkOrderId", "")),
                    order_number=item.get("WorkOrderNumber", ""),
                    order_type=item.get("WorkOrderType", "production"),
                    status=self._map_oracle_status(item.get("WorkOrderStatus", "")),
                    material_number=item.get("ItemNumber", ""),
                    material_description=item.get("ItemDescription", ""),
                    quantity_planned=float(item.get("PlannedQuantity", 0)),
                    quantity_completed=float(item.get("CompletedQuantity", 0)),
                    basic_start_date=datetime.fromisoformat(
                        item.get("PlannedStartDate", "").replace("Z", "+00:00")
                    ),
                    basic_end_date=datetime.fromisoformat(
                        item.get("PlannedCompletionDate", "").replace("Z", "+00:00")
                    ),
                    plant_code=plant_code,
                    work_center=item.get("WorkArea", ""),
                    source_system="Oracle",
                )
                work_orders.append(wo)

        except Exception as e:
            self._logger.error(f"Oracle work order fetch failed: {e}")
            raise

        # Apply status filter
        if status_filter:
            work_orders = [wo for wo in work_orders if wo.status in status_filter]

        return work_orders

    async def _fetch_maintenance_schedules(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime,
        equipment_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[MaintenanceSchedule]:
        """Fetch maintenance schedules from Oracle Maintenance."""
        schedules = []

        try:
            headers = self._get_auth_headers()

            url = f"/fscmRestApi/resources/latest/{self._oracle_config.maintenance_module}"
            params = {
                "q": (
                    f"ScheduledStartDate >= '{start_date.strftime('%Y-%m-%d')}' and "
                    f"ScheduledStartDate <= '{end_date.strftime('%Y-%m-%d')}' and "
                    f"OrganizationCode = '{plant_code}'"
                ),
                "limit": 500,
            }

            response = await self._client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            for item in items:
                schedule = MaintenanceSchedule(
                    schedule_id=str(item.get("WorkOrderId", "")),
                    maintenance_type=MaintenanceType.PREVENTIVE,
                    equipment_id=item.get("AssetNumber", ""),
                    equipment_name=item.get("AssetDescription", ""),
                    planned_start=datetime.fromisoformat(
                        item.get("ScheduledStartDate", "").replace("Z", "+00:00")
                    ),
                    planned_end=datetime.fromisoformat(
                        item.get("ScheduledCompletionDate", "").replace("Z", "+00:00")
                    ),
                    plant_code=plant_code,
                    work_order_number=item.get("WorkOrderNumber", ""),
                    task_description=item.get("WorkOrderDescription", ""),
                    source_system="Oracle",
                )
                schedules.append(schedule)

        except Exception as e:
            self._logger.error(f"Oracle maintenance schedule fetch failed: {e}")
            raise

        # Filter by equipment IDs
        if equipment_ids:
            schedules = [s for s in schedules if s.equipment_id in equipment_ids]

        return schedules

    async def _update_work_order_status(
        self,
        order_id: str,
        new_status: WorkOrderStatus,
        **kwargs
    ) -> bool:
        """Update work order status in Oracle."""
        try:
            headers = self._get_auth_headers()

            url = f"/fscmRestApi/resources/latest/{self._oracle_config.manufacturing_module}/{order_id}"

            payload = {
                "WorkOrderStatus": self._map_status_to_oracle(new_status)
            }

            response = await self._client.patch(url, json=payload, headers=headers)
            response.raise_for_status()

            return True

        except Exception as e:
            self._logger.error(f"Oracle work order update failed: {e}")
            raise

    def _map_oracle_status(self, oracle_status: str) -> WorkOrderStatus:
        """Map Oracle status to WorkOrderStatus."""
        status_map = {
            "Unreleased": WorkOrderStatus.CREATED,
            "Released": WorkOrderStatus.RELEASED,
            "On Hold": WorkOrderStatus.ON_HOLD,
            "Complete": WorkOrderStatus.COMPLETED,
            "Closed": WorkOrderStatus.COMPLETED,
            "Canceled": WorkOrderStatus.CANCELLED,
        }
        return status_map.get(oracle_status, WorkOrderStatus.PLANNED)

    def _map_status_to_oracle(self, status: WorkOrderStatus) -> str:
        """Map WorkOrderStatus to Oracle status."""
        status_map = {
            WorkOrderStatus.CREATED: "Unreleased",
            WorkOrderStatus.PLANNED: "Unreleased",
            WorkOrderStatus.RELEASED: "Released",
            WorkOrderStatus.IN_PROGRESS: "Released",
            WorkOrderStatus.ON_HOLD: "On Hold",
            WorkOrderStatus.COMPLETED: "Complete",
            WorkOrderStatus.CANCELLED: "Canceled",
        }
        return status_map.get(status, "Unreleased")


# =============================================================================
# Factory Functions
# =============================================================================


def create_erp_connector(
    erp_system: ERPSystem,
    host: str,
    username: str,
    password: str,
    **kwargs
) -> ERPConnectorBase:
    """
    Factory function to create ERP connector.

    Args:
        erp_system: ERP system type
        host: Server host
        username: Username
        password: Password
        **kwargs: Additional configuration

    Returns:
        Configured ERP connector
    """
    if erp_system in [ERPSystem.SAP_S4HANA, ERPSystem.SAP_ECC]:
        config = SAPConfig(
            erp_system=erp_system,
            system_id=kwargs.get("system_id", "PRD"),
            host=host,
            username=username,
            password=password,
            **{k: v for k, v in kwargs.items() if k != "system_id"}
        )
        return SAPERPConnector(config)

    elif erp_system in [ERPSystem.ORACLE_CLOUD, ERPSystem.ORACLE_EBS]:
        config = OracleConfig(
            erp_system=erp_system,
            system_id=kwargs.get("system_id", "PRD"),
            host=host,
            username=username,
            password=password,
            **{k: v for k, v in kwargs.items() if k != "system_id"}
        )
        return OracleERPConnector(config)

    else:
        raise ValueError(f"Unsupported ERP system: {erp_system}")
