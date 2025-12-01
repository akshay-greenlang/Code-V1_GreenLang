"""
CMMS Connector Module for GL-013 PREDICTMAINT (Predictive Maintenance Agent).

Provides integration with Computerized Maintenance Management Systems including
SAP PM, IBM Maximo, and Oracle EAM. Supports REST API with OAuth 2.0 authentication,
work order management, equipment synchronization, and maintenance history retrieval.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from urllib.parse import urlencode, urljoin

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    DataQualityLevel,
    DataQualityResult,
    HealthCheckResult,
    HealthStatus,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectorError,
    RateLimitError,
    TimeoutError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class CMSProvider(str, Enum):
    """Supported CMMS providers."""

    SAP_PM = "sap_pm"
    IBM_MAXIMO = "ibm_maximo"
    ORACLE_EAM = "oracle_eam"
    INFOR_EAM = "infor_eam"
    FIIX = "fiix"
    UPKEEP = "upkeep"
    MAINTENANCE_CONNECTION = "maintenance_connection"


class WorkOrderStatus(str, Enum):
    """Work order status values."""

    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CLOSED = "closed"


class WorkOrderPriority(str, Enum):
    """Work order priority levels."""

    EMERGENCY = "emergency"
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ROUTINE = "routine"


class WorkOrderType(str, Enum):
    """Work order types."""

    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    CALIBRATION = "calibration"
    CONDITION_BASED = "condition_based"


class EquipmentStatus(str, Enum):
    """Equipment status values."""

    OPERATIONAL = "operational"
    RUNNING = "running"
    IDLE = "idle"
    STANDBY = "standby"
    UNDER_MAINTENANCE = "under_maintenance"
    FAILED = "failed"
    DECOMMISSIONED = "decommissioned"


class EquipmentCriticality(str, Enum):
    """Equipment criticality levels."""

    CRITICAL = "critical"
    ESSENTIAL = "essential"
    IMPORTANT = "important"
    NORMAL = "normal"
    LOW = "low"


class MaintenanceType(str, Enum):
    """Types of maintenance activities."""

    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    RUN_TO_FAILURE = "run_to_failure"
    DESIGN_OUT = "design_out"


class AuthenticationType(str, Enum):
    """Authentication types supported."""

    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_PASSWORD = "oauth2_password"
    OAUTH2_AUTHORIZATION_CODE = "oauth2_authorization_code"
    API_KEY = "api_key"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    SAML = "saml"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class OAuth2Config(BaseModel):
    """OAuth 2.0 configuration."""

    model_config = ConfigDict(extra="forbid")

    token_url: str = Field(..., description="OAuth2 token endpoint URL")
    client_id: str = Field(..., min_length=1, description="OAuth2 client ID")
    client_secret: str = Field(..., min_length=1, description="OAuth2 client secret")
    scope: str = Field(default="", description="OAuth2 scopes")
    audience: Optional[str] = Field(default=None, description="OAuth2 audience")
    grant_type: str = Field(default="client_credentials", description="OAuth2 grant type")

    # For password grant
    username: Optional[str] = Field(default=None, description="Username for password grant")
    password: Optional[str] = Field(default=None, description="Password for password grant")

    # Token management
    token_refresh_buffer_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Refresh token this many seconds before expiry"
    )


class APIKeyConfig(BaseModel):
    """API Key configuration."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(..., min_length=1, description="API key")
    header_name: str = Field(default="X-API-Key", description="Header name for API key")
    query_param_name: Optional[str] = Field(default=None, description="Query param name if used")


class BasicAuthConfig(BaseModel):
    """Basic authentication configuration."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class CMSSConnectorConfig(BaseConnectorConfig):
    """Configuration for CMMS connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: CMSProvider = Field(..., description="CMMS provider")
    base_url: str = Field(..., description="Base URL for CMMS API")
    api_version: str = Field(default="v1", description="API version")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
        description="Authentication type"
    )
    oauth2_config: Optional[OAuth2Config] = Field(default=None, description="OAuth2 configuration")
    api_key_config: Optional[APIKeyConfig] = Field(default=None, description="API key configuration")
    basic_auth_config: Optional[BasicAuthConfig] = Field(default=None, description="Basic auth configuration")

    # API settings
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Request timeout"
    )
    max_page_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum page size for paginated requests"
    )

    # Retry settings for API calls
    api_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for API calls"
    )
    api_retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Delay between API retries"
    )

    # Provider-specific settings
    sap_client: Optional[str] = Field(default=None, description="SAP client number")
    sap_system_id: Optional[str] = Field(default=None, description="SAP system ID")
    maximo_site_id: Optional[str] = Field(default=None, description="Maximo site ID")
    maximo_org_id: Optional[str] = Field(default=None, description="Maximo organization ID")
    oracle_business_unit: Optional[str] = Field(default=None, description="Oracle business unit")

    # Feature flags
    enable_work_orders: bool = Field(default=True, description="Enable work order operations")
    enable_equipment_sync: bool = Field(default=True, description="Enable equipment synchronization")
    enable_spare_parts: bool = Field(default=True, description="Enable spare parts queries")
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    enable_notifications: bool = Field(default=True, description="Enable notifications")

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.CMMS


# =============================================================================
# Pydantic Models - Data Objects
# =============================================================================


class Equipment(BaseModel):
    """Equipment/Asset master data model."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_name: str = Field(..., description="Equipment name")
    description: Optional[str] = Field(default=None, description="Equipment description")

    # Classification
    equipment_type: Optional[str] = Field(default=None, description="Equipment type/class")
    equipment_category: Optional[str] = Field(default=None, description="Equipment category")
    criticality: EquipmentCriticality = Field(
        default=EquipmentCriticality.NORMAL,
        description="Equipment criticality"
    )
    status: EquipmentStatus = Field(
        default=EquipmentStatus.OPERATIONAL,
        description="Equipment status"
    )

    # Location
    location_id: Optional[str] = Field(default=None, description="Location identifier")
    location_name: Optional[str] = Field(default=None, description="Location name")
    plant_id: Optional[str] = Field(default=None, description="Plant identifier")
    area: Optional[str] = Field(default=None, description="Area/zone")

    # Technical specifications
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer")
    model: Optional[str] = Field(default=None, description="Model number")
    serial_number: Optional[str] = Field(default=None, description="Serial number")
    asset_tag: Optional[str] = Field(default=None, description="Asset tag")

    # Dates
    installation_date: Optional[datetime] = Field(default=None, description="Installation date")
    warranty_expiry_date: Optional[datetime] = Field(default=None, description="Warranty expiry")
    last_maintenance_date: Optional[datetime] = Field(default=None, description="Last maintenance date")
    next_maintenance_date: Optional[datetime] = Field(default=None, description="Next scheduled maintenance")

    # Financial
    acquisition_cost: Optional[float] = Field(default=None, ge=0, description="Acquisition cost")
    current_value: Optional[float] = Field(default=None, ge=0, description="Current book value")
    cost_center: Optional[str] = Field(default=None, description="Cost center")

    # Relationships
    parent_equipment_id: Optional[str] = Field(default=None, description="Parent equipment ID")
    children_ids: List[str] = Field(default_factory=list, description="Child equipment IDs")

    # Custom attributes
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class WorkOrder(BaseModel):
    """Work order model."""

    model_config = ConfigDict(extra="allow")

    work_order_id: str = Field(..., description="Unique work order identifier")
    work_order_number: Optional[str] = Field(default=None, description="Work order number")
    title: str = Field(..., description="Work order title")
    description: Optional[str] = Field(default=None, description="Detailed description")

    # Classification
    work_order_type: WorkOrderType = Field(..., description="Work order type")
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority level"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Current status"
    )

    # Equipment
    equipment_id: str = Field(..., description="Equipment ID")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")
    location_id: Optional[str] = Field(default=None, description="Location ID")

    # Scheduling
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    scheduled_start: Optional[datetime] = Field(default=None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(default=None, description="Scheduled end")
    actual_start: Optional[datetime] = Field(default=None, description="Actual start")
    actual_end: Optional[datetime] = Field(default=None, description="Actual end")
    due_date: Optional[datetime] = Field(default=None, description="Due date")

    # Assignment
    assigned_to: Optional[str] = Field(default=None, description="Assigned technician ID")
    assigned_team: Optional[str] = Field(default=None, description="Assigned team/crew")
    created_by: Optional[str] = Field(default=None, description="Created by user ID")

    # Effort
    estimated_hours: Optional[float] = Field(default=None, ge=0, description="Estimated hours")
    actual_hours: Optional[float] = Field(default=None, ge=0, description="Actual hours worked")

    # Cost
    estimated_cost: Optional[float] = Field(default=None, ge=0, description="Estimated cost")
    actual_cost: Optional[float] = Field(default=None, ge=0, description="Actual cost")
    cost_center: Optional[str] = Field(default=None, description="Cost center")

    # Predictive maintenance fields
    prediction_id: Optional[str] = Field(default=None, description="Related prediction ID")
    predicted_failure_date: Optional[datetime] = Field(default=None, description="Predicted failure date")
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1, description="Prediction confidence")
    remaining_useful_life_hours: Optional[float] = Field(default=None, ge=0, description="RUL in hours")

    # Tasks
    tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Work order tasks")
    spare_parts: List[Dict[str, Any]] = Field(default_factory=list, description="Required spare parts")

    # Completion
    completion_notes: Optional[str] = Field(default=None, description="Completion notes")
    failure_code: Optional[str] = Field(default=None, description="Failure code if corrective")
    cause_code: Optional[str] = Field(default=None, description="Root cause code")
    action_code: Optional[str] = Field(default=None, description="Action taken code")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MaintenanceHistory(BaseModel):
    """Maintenance history record."""

    model_config = ConfigDict(extra="allow")

    history_id: str = Field(..., description="History record ID")
    equipment_id: str = Field(..., description="Equipment ID")
    work_order_id: Optional[str] = Field(default=None, description="Related work order ID")

    # Maintenance details
    maintenance_type: MaintenanceType = Field(..., description="Type of maintenance")
    maintenance_date: datetime = Field(..., description="Maintenance date")
    description: str = Field(..., description="Description of work performed")

    # Outcome
    downtime_hours: Optional[float] = Field(default=None, ge=0, description="Downtime hours")
    labor_hours: Optional[float] = Field(default=None, ge=0, description="Labor hours")
    total_cost: Optional[float] = Field(default=None, ge=0, description="Total cost")

    # Parts used
    parts_used: List[Dict[str, Any]] = Field(default_factory=list, description="Parts used")

    # Technician
    technician_id: Optional[str] = Field(default=None, description="Technician ID")
    technician_name: Optional[str] = Field(default=None, description="Technician name")

    # Failure analysis
    failure_code: Optional[str] = Field(default=None, description="Failure code")
    cause_code: Optional[str] = Field(default=None, description="Cause code")
    action_code: Optional[str] = Field(default=None, description="Action code")
    failure_description: Optional[str] = Field(default=None, description="Failure description")

    # Readings/measurements
    readings: Dict[str, Any] = Field(default_factory=dict, description="Readings/measurements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SparePart(BaseModel):
    """Spare part/inventory item model."""

    model_config = ConfigDict(extra="allow")

    part_id: str = Field(..., description="Part ID")
    part_number: str = Field(..., description="Part number")
    part_name: str = Field(..., description="Part name")
    description: Optional[str] = Field(default=None, description="Description")

    # Classification
    category: Optional[str] = Field(default=None, description="Part category")
    subcategory: Optional[str] = Field(default=None, description="Part subcategory")

    # Inventory
    quantity_on_hand: int = Field(default=0, ge=0, description="Quantity on hand")
    quantity_reserved: int = Field(default=0, ge=0, description="Quantity reserved")
    quantity_available: int = Field(default=0, ge=0, description="Quantity available")
    reorder_point: int = Field(default=0, ge=0, description="Reorder point")
    reorder_quantity: int = Field(default=0, ge=0, description="Reorder quantity")

    # Location
    warehouse_id: Optional[str] = Field(default=None, description="Warehouse ID")
    bin_location: Optional[str] = Field(default=None, description="Bin location")

    # Pricing
    unit_cost: Optional[float] = Field(default=None, ge=0, description="Unit cost")
    currency: str = Field(default="USD", description="Currency")

    # Supplier
    supplier_id: Optional[str] = Field(default=None, description="Primary supplier ID")
    supplier_part_number: Optional[str] = Field(default=None, description="Supplier part number")
    lead_time_days: Optional[int] = Field(default=None, ge=0, description="Lead time in days")

    # Compatibility
    compatible_equipment_ids: List[str] = Field(
        default_factory=list,
        description="Compatible equipment IDs"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Notification(BaseModel):
    """Notification/alert from CMMS."""

    model_config = ConfigDict(extra="allow")

    notification_id: str = Field(..., description="Notification ID")
    notification_type: str = Field(..., description="Notification type")
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority level"
    )
    status: str = Field(default="new", description="Notification status")

    # Content
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")

    # Target
    equipment_id: Optional[str] = Field(default=None, description="Related equipment ID")
    work_order_id: Optional[str] = Field(default=None, description="Related work order ID")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")
    acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledged timestamp")
    resolved_at: Optional[datetime] = Field(default=None, description="Resolved timestamp")

    # Assignment
    assigned_to: Optional[str] = Field(default=None, description="Assigned user ID")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CostRecord(BaseModel):
    """Cost tracking record."""

    model_config = ConfigDict(extra="allow")

    record_id: str = Field(..., description="Cost record ID")
    equipment_id: str = Field(..., description="Equipment ID")
    work_order_id: Optional[str] = Field(default=None, description="Work order ID")

    # Cost breakdown
    cost_type: str = Field(..., description="Cost type (labor, parts, external, etc.)")
    amount: float = Field(..., ge=0, description="Cost amount")
    currency: str = Field(default="USD", description="Currency")

    # Timing
    cost_date: datetime = Field(..., description="Cost date")
    period_start: Optional[datetime] = Field(default=None, description="Period start")
    period_end: Optional[datetime] = Field(default=None, description="Period end")

    # Categorization
    cost_center: Optional[str] = Field(default=None, description="Cost center")
    account_code: Optional[str] = Field(default=None, description="Account code")

    description: Optional[str] = Field(default=None, description="Description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Request/Response Models
# =============================================================================


class WorkOrderCreateRequest(BaseModel):
    """Request to create a work order."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=255, description="Work order title")
    description: Optional[str] = Field(default=None, max_length=4000, description="Description")
    work_order_type: WorkOrderType = Field(..., description="Work order type")
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority"
    )

    equipment_id: str = Field(..., description="Equipment ID")
    location_id: Optional[str] = Field(default=None, description="Location ID")

    scheduled_start: Optional[datetime] = Field(default=None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(default=None, description="Scheduled end")
    due_date: Optional[datetime] = Field(default=None, description="Due date")

    assigned_to: Optional[str] = Field(default=None, description="Assigned technician")
    assigned_team: Optional[str] = Field(default=None, description="Assigned team")

    estimated_hours: Optional[float] = Field(default=None, ge=0, description="Estimated hours")
    estimated_cost: Optional[float] = Field(default=None, ge=0, description="Estimated cost")
    cost_center: Optional[str] = Field(default=None, description="Cost center")

    # Predictive maintenance fields
    prediction_id: Optional[str] = Field(default=None, description="Prediction ID")
    predicted_failure_date: Optional[datetime] = Field(default=None, description="Predicted failure")
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1, description="Confidence")
    remaining_useful_life_hours: Optional[float] = Field(default=None, ge=0, description="RUL hours")

    tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Tasks")
    spare_parts: List[Dict[str, Any]] = Field(default_factory=list, description="Spare parts")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class WorkOrderUpdateRequest(BaseModel):
    """Request to update a work order."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=4000)
    priority: Optional[WorkOrderPriority] = Field(default=None)
    status: Optional[WorkOrderStatus] = Field(default=None)

    scheduled_start: Optional[datetime] = Field(default=None)
    scheduled_end: Optional[datetime] = Field(default=None)
    actual_start: Optional[datetime] = Field(default=None)
    actual_end: Optional[datetime] = Field(default=None)

    assigned_to: Optional[str] = Field(default=None)
    assigned_team: Optional[str] = Field(default=None)

    actual_hours: Optional[float] = Field(default=None, ge=0)
    actual_cost: Optional[float] = Field(default=None, ge=0)

    completion_notes: Optional[str] = Field(default=None, max_length=4000)
    failure_code: Optional[str] = Field(default=None)
    cause_code: Optional[str] = Field(default=None)
    action_code: Optional[str] = Field(default=None)

    metadata: Optional[Dict[str, Any]] = Field(default=None)


class EquipmentQueryParams(BaseModel):
    """Query parameters for equipment listing."""

    model_config = ConfigDict(extra="forbid")

    plant_id: Optional[str] = Field(default=None, description="Filter by plant")
    location_id: Optional[str] = Field(default=None, description="Filter by location")
    equipment_type: Optional[str] = Field(default=None, description="Filter by type")
    status: Optional[EquipmentStatus] = Field(default=None, description="Filter by status")
    criticality: Optional[EquipmentCriticality] = Field(default=None, description="Filter by criticality")
    search: Optional[str] = Field(default=None, description="Search term")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=500, description="Page size")
    sort_by: str = Field(default="equipment_name", description="Sort field")
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")


class MaintenanceHistoryQueryParams(BaseModel):
    """Query parameters for maintenance history."""

    model_config = ConfigDict(extra="forbid")

    equipment_id: str = Field(..., description="Equipment ID")
    start_date: Optional[datetime] = Field(default=None, description="Start date")
    end_date: Optional[datetime] = Field(default=None, description="End date")
    maintenance_type: Optional[MaintenanceType] = Field(default=None, description="Maintenance type")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=500, description="Page size")


# =============================================================================
# Authentication Handler
# =============================================================================


class AuthenticationHandler:
    """Handles authentication for CMMS API calls."""

    def __init__(
        self,
        auth_type: AuthenticationType,
        oauth2_config: Optional[OAuth2Config] = None,
        api_key_config: Optional[APIKeyConfig] = None,
        basic_auth_config: Optional[BasicAuthConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize authentication handler."""
        self._auth_type = auth_type
        self._oauth2_config = oauth2_config
        self._api_key_config = api_key_config
        self._basic_auth_config = basic_auth_config
        self._http_client = http_client

        # Token state
        self._access_token: Optional[str] = None
        self._token_type: str = "Bearer"
        self._token_expires_at: Optional[datetime] = None
        self._refresh_token: Optional[str] = None
        self._lock = asyncio.Lock()

    async def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.

        Returns:
            Dictionary of headers to include in request
        """
        if self._auth_type == AuthenticationType.API_KEY:
            if not self._api_key_config:
                raise ConfigurationError("API key configuration required")
            return {self._api_key_config.header_name: self._api_key_config.api_key}

        elif self._auth_type == AuthenticationType.BASIC_AUTH:
            if not self._basic_auth_config:
                raise ConfigurationError("Basic auth configuration required")
            credentials = base64.b64encode(
                f"{self._basic_auth_config.username}:{self._basic_auth_config.password}".encode()
            ).decode()
            return {"Authorization": f"Basic {credentials}"}

        elif self._auth_type == AuthenticationType.BEARER_TOKEN:
            if not self._access_token:
                raise AuthenticationError("No bearer token available")
            return {"Authorization": f"Bearer {self._access_token}"}

        elif self._auth_type in [
            AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
            AuthenticationType.OAUTH2_PASSWORD,
            AuthenticationType.OAUTH2_AUTHORIZATION_CODE,
        ]:
            token = await self._ensure_valid_token()
            return {"Authorization": f"{self._token_type} {token}"}

        else:
            raise ConfigurationError(f"Unsupported auth type: {self._auth_type}")

    async def _ensure_valid_token(self) -> str:
        """Ensure we have a valid OAuth2 token, refreshing if needed."""
        async with self._lock:
            if self._access_token and self._token_expires_at:
                buffer = timedelta(seconds=self._oauth2_config.token_refresh_buffer_seconds)
                if datetime.utcnow() < self._token_expires_at - buffer:
                    return self._access_token

            # Need to get new token
            await self._refresh_oauth2_token()
            return self._access_token

    async def _refresh_oauth2_token(self) -> None:
        """Refresh OAuth2 access token."""
        if not self._oauth2_config:
            raise ConfigurationError("OAuth2 configuration required")

        if not self._http_client:
            raise ConfigurationError("HTTP client required for OAuth2")

        # Build token request
        token_data = {
            "client_id": self._oauth2_config.client_id,
            "client_secret": self._oauth2_config.client_secret,
        }

        if self._auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            token_data["grant_type"] = "client_credentials"
            if self._oauth2_config.scope:
                token_data["scope"] = self._oauth2_config.scope
            if self._oauth2_config.audience:
                token_data["audience"] = self._oauth2_config.audience

        elif self._auth_type == AuthenticationType.OAUTH2_PASSWORD:
            token_data["grant_type"] = "password"
            token_data["username"] = self._oauth2_config.username
            token_data["password"] = self._oauth2_config.password
            if self._oauth2_config.scope:
                token_data["scope"] = self._oauth2_config.scope

        elif self._refresh_token:
            token_data["grant_type"] = "refresh_token"
            token_data["refresh_token"] = self._refresh_token

        try:
            response = await self._http_client.post(
                self._oauth2_config.token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()

            token_response = response.json()
            self._access_token = token_response["access_token"]
            self._token_type = token_response.get("token_type", "Bearer")

            expires_in = token_response.get("expires_in", 3600)
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

            if "refresh_token" in token_response:
                self._refresh_token = token_response["refresh_token"]

            logger.info("OAuth2 token refreshed successfully")

        except httpx.HTTPStatusError as e:
            raise AuthenticationError(
                f"OAuth2 token refresh failed: {e.response.status_code}",
                details={"response": e.response.text},
            )
        except Exception as e:
            raise AuthenticationError(f"OAuth2 token refresh error: {str(e)}")

    def set_bearer_token(self, token: str, token_type: str = "Bearer") -> None:
        """Set bearer token directly."""
        self._access_token = token
        self._token_type = token_type

    async def invalidate_token(self) -> None:
        """Invalidate current token, forcing refresh on next request."""
        async with self._lock:
            self._access_token = None
            self._token_expires_at = None


# =============================================================================
# Provider-Specific Adapters
# =============================================================================


class CMSSProviderAdapter:
    """Base adapter for CMMS provider-specific API transformations."""

    def __init__(self, config: CMSSConnectorConfig) -> None:
        self._config = config

    def get_base_url(self) -> str:
        """Get the base URL for API calls."""
        return self._config.base_url.rstrip("/")

    def get_headers(self) -> Dict[str, str]:
        """Get provider-specific headers."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def transform_work_order_to_api(self, request: WorkOrderCreateRequest) -> Dict[str, Any]:
        """Transform work order create request to API format."""
        return request.model_dump(exclude_none=True)

    def transform_api_to_work_order(self, data: Dict[str, Any]) -> WorkOrder:
        """Transform API response to work order model."""
        return WorkOrder(**data)

    def transform_api_to_equipment(self, data: Dict[str, Any]) -> Equipment:
        """Transform API response to equipment model."""
        return Equipment(**data)

    def get_work_orders_endpoint(self) -> str:
        """Get work orders API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/workorders"

    def get_equipment_endpoint(self) -> str:
        """Get equipment API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/equipment"

    def get_maintenance_history_endpoint(self) -> str:
        """Get maintenance history API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/maintenance-history"

    def get_spare_parts_endpoint(self) -> str:
        """Get spare parts API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/spare-parts"


class SAPPMAdapter(CMSSProviderAdapter):
    """Adapter for SAP Plant Maintenance (PM)."""

    def get_headers(self) -> Dict[str, str]:
        headers = super().get_headers()
        if self._config.sap_client:
            headers["sap-client"] = self._config.sap_client
        return headers

    def get_work_orders_endpoint(self) -> str:
        return f"{self.get_base_url()}/sap/opu/odata/sap/API_MAINTORDER_SRV/MaintenanceOrder"

    def get_equipment_endpoint(self) -> str:
        return f"{self.get_base_url()}/sap/opu/odata/sap/API_EQUIPMENT_SRV/Equipment"

    def transform_work_order_to_api(self, request: WorkOrderCreateRequest) -> Dict[str, Any]:
        """Transform to SAP PM format."""
        return {
            "MaintenanceOrderType": self._map_work_order_type(request.work_order_type),
            "MaintenanceOrderDesc": request.title,
            "LongTextString": request.description or "",
            "Equipment": request.equipment_id,
            "FunctionalLocation": request.location_id or "",
            "MaintPriority": self._map_priority(request.priority),
            "BasicSchedulingStartDate": request.scheduled_start.isoformat() if request.scheduled_start else None,
            "BasicSchedulingEndDate": request.scheduled_end.isoformat() if request.scheduled_end else None,
            "PlannedWorkQuantity": request.estimated_hours or 0,
            "ProfitCenter": request.cost_center or "",
        }

    def transform_api_to_work_order(self, data: Dict[str, Any]) -> WorkOrder:
        """Transform SAP PM response to work order model."""
        return WorkOrder(
            work_order_id=data.get("MaintenanceOrder", ""),
            work_order_number=data.get("MaintenanceOrder"),
            title=data.get("MaintenanceOrderDesc", ""),
            description=data.get("LongTextString"),
            work_order_type=self._reverse_map_work_order_type(data.get("MaintenanceOrderType", "")),
            priority=self._reverse_map_priority(data.get("MaintPriority", "")),
            status=self._map_sap_status(data.get("SystemStatus", "")),
            equipment_id=data.get("Equipment", ""),
            location_id=data.get("FunctionalLocation"),
            scheduled_start=self._parse_date(data.get("BasicSchedulingStartDate")),
            scheduled_end=self._parse_date(data.get("BasicSchedulingEndDate")),
            estimated_hours=data.get("PlannedWorkQuantity"),
            cost_center=data.get("ProfitCenter"),
            metadata={"sap_raw": data},
        )

    def _map_work_order_type(self, wot: WorkOrderType) -> str:
        mapping = {
            WorkOrderType.PREVENTIVE: "PM01",
            WorkOrderType.PREDICTIVE: "PM02",
            WorkOrderType.CORRECTIVE: "PM03",
            WorkOrderType.EMERGENCY: "PM04",
            WorkOrderType.INSPECTION: "PM05",
            WorkOrderType.CALIBRATION: "PM06",
            WorkOrderType.CONDITION_BASED: "PM02",
        }
        return mapping.get(wot, "PM01")

    def _reverse_map_work_order_type(self, sap_type: str) -> WorkOrderType:
        mapping = {
            "PM01": WorkOrderType.PREVENTIVE,
            "PM02": WorkOrderType.PREDICTIVE,
            "PM03": WorkOrderType.CORRECTIVE,
            "PM04": WorkOrderType.EMERGENCY,
            "PM05": WorkOrderType.INSPECTION,
            "PM06": WorkOrderType.CALIBRATION,
        }
        return mapping.get(sap_type, WorkOrderType.CORRECTIVE)

    def _map_priority(self, priority: WorkOrderPriority) -> str:
        mapping = {
            WorkOrderPriority.EMERGENCY: "1",
            WorkOrderPriority.URGENT: "2",
            WorkOrderPriority.HIGH: "3",
            WorkOrderPriority.MEDIUM: "4",
            WorkOrderPriority.LOW: "5",
            WorkOrderPriority.ROUTINE: "6",
        }
        return mapping.get(priority, "4")

    def _reverse_map_priority(self, sap_priority: str) -> WorkOrderPriority:
        mapping = {
            "1": WorkOrderPriority.EMERGENCY,
            "2": WorkOrderPriority.URGENT,
            "3": WorkOrderPriority.HIGH,
            "4": WorkOrderPriority.MEDIUM,
            "5": WorkOrderPriority.LOW,
            "6": WorkOrderPriority.ROUTINE,
        }
        return mapping.get(sap_priority, WorkOrderPriority.MEDIUM)

    def _map_sap_status(self, status: str) -> WorkOrderStatus:
        status_lower = status.lower()
        if "crtd" in status_lower:
            return WorkOrderStatus.DRAFT
        elif "rel" in status_lower:
            return WorkOrderStatus.APPROVED
        elif "pcnf" in status_lower:
            return WorkOrderStatus.IN_PROGRESS
        elif "cnf" in status_lower:
            return WorkOrderStatus.COMPLETED
        elif "teco" in status_lower:
            return WorkOrderStatus.CLOSED
        elif "dlfl" in status_lower:
            return WorkOrderStatus.CANCELLED
        return WorkOrderStatus.PENDING

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            # Handle SAP date format
            if date_str.startswith("/Date("):
                timestamp = int(date_str[6:-2]) / 1000
                return datetime.utcfromtimestamp(timestamp)
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None


class IBMMaximoAdapter(CMSSProviderAdapter):
    """Adapter for IBM Maximo."""

    def get_headers(self) -> Dict[str, str]:
        headers = super().get_headers()
        if self._config.maximo_site_id:
            headers["x-public-uri-site-id"] = self._config.maximo_site_id
        return headers

    def get_work_orders_endpoint(self) -> str:
        return f"{self.get_base_url()}/maximo/oslc/os/mxwo"

    def get_equipment_endpoint(self) -> str:
        return f"{self.get_base_url()}/maximo/oslc/os/mxasset"

    def transform_work_order_to_api(self, request: WorkOrderCreateRequest) -> Dict[str, Any]:
        """Transform to Maximo format."""
        return {
            "description": request.title,
            "description_longdescription": request.description or "",
            "worktype": self._map_work_order_type(request.work_order_type),
            "assetnum": request.equipment_id,
            "location": request.location_id or "",
            "wopriority": self._map_priority(request.priority),
            "siteid": self._config.maximo_site_id or "",
            "orgid": self._config.maximo_org_id or "",
            "schedstart": request.scheduled_start.isoformat() if request.scheduled_start else None,
            "schedfinish": request.scheduled_end.isoformat() if request.scheduled_end else None,
            "estlabhrs": request.estimated_hours or 0,
        }

    def transform_api_to_work_order(self, data: Dict[str, Any]) -> WorkOrder:
        """Transform Maximo response to work order model."""
        return WorkOrder(
            work_order_id=str(data.get("workorderid", "")),
            work_order_number=data.get("wonum"),
            title=data.get("description", ""),
            description=data.get("description_longdescription"),
            work_order_type=self._reverse_map_work_order_type(data.get("worktype", "")),
            priority=self._reverse_map_priority(data.get("wopriority", 0)),
            status=self._map_maximo_status(data.get("status", "")),
            equipment_id=data.get("assetnum", ""),
            location_id=data.get("location"),
            scheduled_start=self._parse_date(data.get("schedstart")),
            scheduled_end=self._parse_date(data.get("schedfinish")),
            actual_start=self._parse_date(data.get("actstart")),
            actual_end=self._parse_date(data.get("actfinish")),
            estimated_hours=data.get("estlabhrs"),
            actual_hours=data.get("actlabhrs"),
            metadata={"maximo_raw": data},
        )

    def _map_work_order_type(self, wot: WorkOrderType) -> str:
        mapping = {
            WorkOrderType.PREVENTIVE: "PM",
            WorkOrderType.PREDICTIVE: "PDM",
            WorkOrderType.CORRECTIVE: "CM",
            WorkOrderType.EMERGENCY: "EM",
            WorkOrderType.INSPECTION: "INS",
            WorkOrderType.CALIBRATION: "CAL",
            WorkOrderType.CONDITION_BASED: "CBM",
        }
        return mapping.get(wot, "CM")

    def _reverse_map_work_order_type(self, maximo_type: str) -> WorkOrderType:
        mapping = {
            "PM": WorkOrderType.PREVENTIVE,
            "PDM": WorkOrderType.PREDICTIVE,
            "CM": WorkOrderType.CORRECTIVE,
            "EM": WorkOrderType.EMERGENCY,
            "INS": WorkOrderType.INSPECTION,
            "CAL": WorkOrderType.CALIBRATION,
            "CBM": WorkOrderType.CONDITION_BASED,
        }
        return mapping.get(maximo_type.upper(), WorkOrderType.CORRECTIVE)

    def _map_priority(self, priority: WorkOrderPriority) -> int:
        mapping = {
            WorkOrderPriority.EMERGENCY: 1,
            WorkOrderPriority.URGENT: 2,
            WorkOrderPriority.HIGH: 3,
            WorkOrderPriority.MEDIUM: 4,
            WorkOrderPriority.LOW: 5,
            WorkOrderPriority.ROUTINE: 6,
        }
        return mapping.get(priority, 4)

    def _reverse_map_priority(self, maximo_priority: int) -> WorkOrderPriority:
        mapping = {
            1: WorkOrderPriority.EMERGENCY,
            2: WorkOrderPriority.URGENT,
            3: WorkOrderPriority.HIGH,
            4: WorkOrderPriority.MEDIUM,
            5: WorkOrderPriority.LOW,
            6: WorkOrderPriority.ROUTINE,
        }
        return mapping.get(maximo_priority, WorkOrderPriority.MEDIUM)

    def _map_maximo_status(self, status: str) -> WorkOrderStatus:
        status_upper = status.upper()
        mapping = {
            "WAPPR": WorkOrderStatus.PENDING,
            "APPR": WorkOrderStatus.APPROVED,
            "WSCH": WorkOrderStatus.APPROVED,
            "WMATL": WorkOrderStatus.APPROVED,
            "INPRG": WorkOrderStatus.IN_PROGRESS,
            "WPCOND": WorkOrderStatus.ON_HOLD,
            "COMP": WorkOrderStatus.COMPLETED,
            "CLOSE": WorkOrderStatus.CLOSED,
            "CAN": WorkOrderStatus.CANCELLED,
        }
        return mapping.get(status_upper, WorkOrderStatus.PENDING)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None


class OracleEAMAdapter(CMSSProviderAdapter):
    """Adapter for Oracle Enterprise Asset Management."""

    def get_work_orders_endpoint(self) -> str:
        return f"{self.get_base_url()}/fscmRestApi/resources/{self._config.api_version}/maintenanceWorkOrders"

    def get_equipment_endpoint(self) -> str:
        return f"{self.get_base_url()}/fscmRestApi/resources/{self._config.api_version}/assets"

    def transform_work_order_to_api(self, request: WorkOrderCreateRequest) -> Dict[str, Any]:
        """Transform to Oracle EAM format."""
        return {
            "WorkOrderDescription": request.title,
            "WorkOrderLongDescription": request.description or "",
            "WorkOrderType": self._map_work_order_type(request.work_order_type),
            "AssetNumber": request.equipment_id,
            "MaintenanceOrganizationCode": self._config.oracle_business_unit or "",
            "Priority": self._map_priority(request.priority),
            "ScheduledStartDate": request.scheduled_start.isoformat() if request.scheduled_start else None,
            "ScheduledCompletionDate": request.scheduled_end.isoformat() if request.scheduled_end else None,
        }

    def transform_api_to_work_order(self, data: Dict[str, Any]) -> WorkOrder:
        """Transform Oracle EAM response to work order model."""
        return WorkOrder(
            work_order_id=str(data.get("WorkOrderId", "")),
            work_order_number=data.get("WorkOrderNumber"),
            title=data.get("WorkOrderDescription", ""),
            description=data.get("WorkOrderLongDescription"),
            work_order_type=self._reverse_map_work_order_type(data.get("WorkOrderType", "")),
            priority=self._reverse_map_priority(data.get("Priority", "")),
            status=self._map_oracle_status(data.get("StatusCode", "")),
            equipment_id=data.get("AssetNumber", ""),
            scheduled_start=self._parse_date(data.get("ScheduledStartDate")),
            scheduled_end=self._parse_date(data.get("ScheduledCompletionDate")),
            actual_start=self._parse_date(data.get("ActualStartDate")),
            actual_end=self._parse_date(data.get("ActualCompletionDate")),
            metadata={"oracle_raw": data},
        )

    def _map_work_order_type(self, wot: WorkOrderType) -> str:
        mapping = {
            WorkOrderType.PREVENTIVE: "PREVENTIVE",
            WorkOrderType.PREDICTIVE: "PREDICTIVE",
            WorkOrderType.CORRECTIVE: "CORRECTIVE",
            WorkOrderType.EMERGENCY: "EMERGENCY",
            WorkOrderType.INSPECTION: "INSPECTION",
            WorkOrderType.CALIBRATION: "CALIBRATION",
            WorkOrderType.CONDITION_BASED: "CONDITION_BASED",
        }
        return mapping.get(wot, "CORRECTIVE")

    def _reverse_map_work_order_type(self, oracle_type: str) -> WorkOrderType:
        mapping = {
            "PREVENTIVE": WorkOrderType.PREVENTIVE,
            "PREDICTIVE": WorkOrderType.PREDICTIVE,
            "CORRECTIVE": WorkOrderType.CORRECTIVE,
            "EMERGENCY": WorkOrderType.EMERGENCY,
            "INSPECTION": WorkOrderType.INSPECTION,
            "CALIBRATION": WorkOrderType.CALIBRATION,
            "CONDITION_BASED": WorkOrderType.CONDITION_BASED,
        }
        return mapping.get(oracle_type.upper(), WorkOrderType.CORRECTIVE)

    def _map_priority(self, priority: WorkOrderPriority) -> str:
        mapping = {
            WorkOrderPriority.EMERGENCY: "1",
            WorkOrderPriority.URGENT: "2",
            WorkOrderPriority.HIGH: "3",
            WorkOrderPriority.MEDIUM: "4",
            WorkOrderPriority.LOW: "5",
            WorkOrderPriority.ROUTINE: "6",
        }
        return mapping.get(priority, "4")

    def _reverse_map_priority(self, oracle_priority: str) -> WorkOrderPriority:
        mapping = {
            "1": WorkOrderPriority.EMERGENCY,
            "2": WorkOrderPriority.URGENT,
            "3": WorkOrderPriority.HIGH,
            "4": WorkOrderPriority.MEDIUM,
            "5": WorkOrderPriority.LOW,
            "6": WorkOrderPriority.ROUTINE,
        }
        return mapping.get(oracle_priority, WorkOrderPriority.MEDIUM)

    def _map_oracle_status(self, status: str) -> WorkOrderStatus:
        status_upper = status.upper()
        mapping = {
            "DRAFT": WorkOrderStatus.DRAFT,
            "PENDING_APPROVAL": WorkOrderStatus.PENDING,
            "APPROVED": WorkOrderStatus.APPROVED,
            "RELEASED": WorkOrderStatus.APPROVED,
            "IN_PROGRESS": WorkOrderStatus.IN_PROGRESS,
            "ON_HOLD": WorkOrderStatus.ON_HOLD,
            "COMPLETED": WorkOrderStatus.COMPLETED,
            "CLOSED": WorkOrderStatus.CLOSED,
            "CANCELLED": WorkOrderStatus.CANCELLED,
        }
        return mapping.get(status_upper, WorkOrderStatus.PENDING)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return None


# =============================================================================
# CMMS Connector Implementation
# =============================================================================


class CMSSConnector(BaseConnector):
    """
    CMMS (Computerized Maintenance Management System) Connector.

    Provides integration with enterprise CMMS platforms including:
    - SAP Plant Maintenance (PM)
    - IBM Maximo
    - Oracle Enterprise Asset Management (EAM)

    Features:
    - OAuth 2.0 and API key authentication
    - Work order creation and management
    - Equipment master data synchronization
    - Maintenance history retrieval
    - Spare parts inventory queries
    - Cost center integration
    - Notification handling
    """

    def __init__(self, config: CMSSConnectorConfig) -> None:
        """
        Initialize CMMS connector.

        Args:
            config: CMMS connector configuration
        """
        super().__init__(config)
        self._cmms_config = config

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # Provider adapter
        self._adapter = self._create_adapter()

        # Authentication handler
        self._auth_handler: Optional[AuthenticationHandler] = None

    def _create_adapter(self) -> CMSSProviderAdapter:
        """Create provider-specific adapter."""
        if self._cmms_config.provider == CMSProvider.SAP_PM:
            return SAPPMAdapter(self._cmms_config)
        elif self._cmms_config.provider == CMSProvider.IBM_MAXIMO:
            return IBMMaximoAdapter(self._cmms_config)
        elif self._cmms_config.provider == CMSProvider.ORACLE_EAM:
            return OracleEAMAdapter(self._cmms_config)
        else:
            return CMSSProviderAdapter(self._cmms_config)

    async def connect(self) -> None:
        """Establish connection to CMMS."""
        self._logger.info(f"Connecting to {self._cmms_config.provider.value} CMMS...")

        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self._cmms_config.connection_timeout_seconds,
                read=self._cmms_config.read_timeout_seconds,
                write=self._cmms_config.write_timeout_seconds,
                pool=self._cmms_config.pool_acquire_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=self._cmms_config.pool_max_size,
                max_keepalive_connections=self._cmms_config.pool_min_size,
            ),
        )

        # Initialize authentication handler
        self._auth_handler = AuthenticationHandler(
            auth_type=self._cmms_config.auth_type,
            oauth2_config=self._cmms_config.oauth2_config,
            api_key_config=self._cmms_config.api_key_config,
            basic_auth_config=self._cmms_config.basic_auth_config,
            http_client=self._http_client,
        )

        # Test connection with authentication
        try:
            await self._test_connection()
            self._logger.info(f"Successfully connected to {self._cmms_config.provider.value} CMMS")
        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to CMMS: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from CMMS."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._auth_handler = None
        self._logger.info("Disconnected from CMMS")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on CMMS connection."""
        start_time = time.time()

        try:
            if not self._http_client:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="HTTP client not initialized",
                    latency_ms=0.0,
                )

            await self._test_connection()
            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="CMMS connection healthy",
                latency_ms=latency_ms,
                details={
                    "provider": self._cmms_config.provider.value,
                    "base_url": self._cmms_config.base_url,
                },
            )

        except AuthenticationError as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Authentication failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def validate_configuration(self) -> bool:
        """Validate CMMS connector configuration."""
        # Validate base URL
        if not self._cmms_config.base_url:
            raise ConfigurationError("Base URL is required")

        # Validate authentication configuration
        if self._cmms_config.auth_type in [
            AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
            AuthenticationType.OAUTH2_PASSWORD,
        ]:
            if not self._cmms_config.oauth2_config:
                raise ConfigurationError("OAuth2 configuration required")

        elif self._cmms_config.auth_type == AuthenticationType.API_KEY:
            if not self._cmms_config.api_key_config:
                raise ConfigurationError("API key configuration required")

        elif self._cmms_config.auth_type == AuthenticationType.BASIC_AUTH:
            if not self._cmms_config.basic_auth_config:
                raise ConfigurationError("Basic auth configuration required")

        # Validate provider-specific settings
        if self._cmms_config.provider == CMSProvider.SAP_PM:
            if not self._cmms_config.sap_client:
                self._logger.warning("SAP client not specified, using default")

        elif self._cmms_config.provider == CMSProvider.IBM_MAXIMO:
            if not self._cmms_config.maximo_site_id:
                self._logger.warning("Maximo site ID not specified")

        return True

    async def _test_connection(self) -> None:
        """Test connection with a simple API call."""
        headers = await self._auth_handler.get_auth_headers()
        headers.update(self._adapter.get_headers())

        # Try equipment endpoint as health check
        url = self._adapter.get_equipment_endpoint()

        # Add limit to make it a quick check
        params = {"$top": "1"} if "sap" in self._cmms_config.provider.value else {"limit": "1"}

        response = await self._http_client.get(url, headers=headers, params=params)
        response.raise_for_status()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        operation_name: str = "api_call",
    ) -> Dict[str, Any]:
        """
        Make authenticated API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            operation_name: Name for logging

        Returns:
            API response data
        """
        async def _do_request() -> Dict[str, Any]:
            headers = await self._auth_handler.get_auth_headers()
            headers.update(self._adapter.get_headers())

            response = await self._http_client.request(
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                json=data,
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                raise RateLimitError(
                    f"Rate limit exceeded",
                    retry_after_seconds=float(retry_after),
                )

            # Handle authentication errors
            if response.status_code == 401:
                await self._auth_handler.invalidate_token()
                raise AuthenticationError("Authentication failed")

            response.raise_for_status()

            if response.content:
                return response.json()
            return {}

        return await self.execute_with_protection(
            _do_request,
            operation_name=operation_name,
            use_cache=method == "GET",
            cache_key=self._generate_cache_key(endpoint, params) if method == "GET" else None,
        )

    # =========================================================================
    # Work Order Operations
    # =========================================================================

    async def create_work_order(self, request: WorkOrderCreateRequest) -> WorkOrder:
        """
        Create a new work order in CMMS.

        Args:
            request: Work order creation request

        Returns:
            Created work order
        """
        if not self._cmms_config.enable_work_orders:
            raise ConfigurationError("Work order operations disabled")

        self._logger.info(f"Creating work order for equipment {request.equipment_id}")

        endpoint = self._adapter.get_work_orders_endpoint()
        data = self._adapter.transform_work_order_to_api(request)

        response = await self._make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            operation_name="create_work_order",
        )

        work_order = self._adapter.transform_api_to_work_order(response)
        self._logger.info(f"Created work order {work_order.work_order_id}")

        return work_order

    async def get_work_order(self, work_order_id: str) -> WorkOrder:
        """
        Get work order by ID.

        Args:
            work_order_id: Work order identifier

        Returns:
            Work order details
        """
        if not self._cmms_config.enable_work_orders:
            raise ConfigurationError("Work order operations disabled")

        endpoint = f"{self._adapter.get_work_orders_endpoint()}/{work_order_id}"

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            operation_name="get_work_order",
        )

        return self._adapter.transform_api_to_work_order(response)

    async def update_work_order(
        self,
        work_order_id: str,
        request: WorkOrderUpdateRequest,
    ) -> WorkOrder:
        """
        Update an existing work order.

        Args:
            work_order_id: Work order identifier
            request: Update request

        Returns:
            Updated work order
        """
        if not self._cmms_config.enable_work_orders:
            raise ConfigurationError("Work order operations disabled")

        self._logger.info(f"Updating work order {work_order_id}")

        endpoint = f"{self._adapter.get_work_orders_endpoint()}/{work_order_id}"
        data = request.model_dump(exclude_none=True)

        response = await self._make_request(
            method="PATCH",
            endpoint=endpoint,
            data=data,
            operation_name="update_work_order",
        )

        # Clear cache for this work order
        await self._cache.delete(self._generate_cache_key(endpoint, None))

        return self._adapter.transform_api_to_work_order(response)

    async def list_work_orders(
        self,
        equipment_id: Optional[str] = None,
        status: Optional[WorkOrderStatus] = None,
        work_order_type: Optional[WorkOrderType] = None,
        priority: Optional[WorkOrderPriority] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> List[WorkOrder]:
        """
        List work orders with optional filters.

        Args:
            equipment_id: Filter by equipment
            status: Filter by status
            work_order_type: Filter by type
            priority: Filter by priority
            start_date: Filter by start date
            end_date: Filter by end date
            page: Page number
            page_size: Page size

        Returns:
            List of work orders
        """
        if not self._cmms_config.enable_work_orders:
            raise ConfigurationError("Work order operations disabled")

        endpoint = self._adapter.get_work_orders_endpoint()
        page_size = page_size or self._cmms_config.max_page_size

        # Build query parameters
        params: Dict[str, Any] = {
            "page": page,
            "limit": page_size,
        }

        if equipment_id:
            params["equipment_id"] = equipment_id
        if status:
            params["status"] = status.value
        if work_order_type:
            params["type"] = work_order_type.value
        if priority:
            params["priority"] = priority.value
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            operation_name="list_work_orders",
        )

        # Handle different response formats
        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [self._adapter.transform_api_to_work_order(item) for item in items]

    async def complete_work_order(
        self,
        work_order_id: str,
        actual_hours: float,
        completion_notes: Optional[str] = None,
        failure_code: Optional[str] = None,
        cause_code: Optional[str] = None,
        action_code: Optional[str] = None,
    ) -> WorkOrder:
        """
        Complete a work order.

        Args:
            work_order_id: Work order identifier
            actual_hours: Actual hours worked
            completion_notes: Completion notes
            failure_code: Failure code (for corrective maintenance)
            cause_code: Root cause code
            action_code: Action taken code

        Returns:
            Completed work order
        """
        update_request = WorkOrderUpdateRequest(
            status=WorkOrderStatus.COMPLETED,
            actual_hours=actual_hours,
            actual_end=datetime.utcnow(),
            completion_notes=completion_notes,
            failure_code=failure_code,
            cause_code=cause_code,
            action_code=action_code,
        )

        return await self.update_work_order(work_order_id, update_request)

    # =========================================================================
    # Equipment Operations
    # =========================================================================

    async def get_equipment(self, equipment_id: str) -> Equipment:
        """
        Get equipment by ID.

        Args:
            equipment_id: Equipment identifier

        Returns:
            Equipment details
        """
        if not self._cmms_config.enable_equipment_sync:
            raise ConfigurationError("Equipment sync disabled")

        endpoint = f"{self._adapter.get_equipment_endpoint()}/{equipment_id}"

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            operation_name="get_equipment",
        )

        return self._adapter.transform_api_to_equipment(response)

    async def list_equipment(
        self,
        params: Optional[EquipmentQueryParams] = None,
    ) -> List[Equipment]:
        """
        List equipment with optional filters.

        Args:
            params: Query parameters

        Returns:
            List of equipment
        """
        if not self._cmms_config.enable_equipment_sync:
            raise ConfigurationError("Equipment sync disabled")

        endpoint = self._adapter.get_equipment_endpoint()
        query_params = params.model_dump(exclude_none=True) if params else {}

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=query_params,
            operation_name="list_equipment",
        )

        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [self._adapter.transform_api_to_equipment(item) for item in items]

    async def sync_equipment(
        self,
        equipment_ids: Optional[List[str]] = None,
        full_sync: bool = False,
        since: Optional[datetime] = None,
    ) -> List[Equipment]:
        """
        Synchronize equipment master data.

        Args:
            equipment_ids: Specific equipment IDs to sync
            full_sync: Whether to do full sync
            since: Sync changes since this datetime

        Returns:
            List of synchronized equipment
        """
        if not self._cmms_config.enable_equipment_sync:
            raise ConfigurationError("Equipment sync disabled")

        self._logger.info("Starting equipment synchronization")

        if equipment_ids:
            # Sync specific equipment
            equipment_list = []
            for eq_id in equipment_ids:
                try:
                    equipment = await self.get_equipment(eq_id)
                    equipment_list.append(equipment)
                except Exception as e:
                    self._logger.warning(f"Failed to sync equipment {eq_id}: {e}")
            return equipment_list

        # Full or incremental sync
        params = EquipmentQueryParams()
        if since and not full_sync:
            # Add modified date filter (provider-specific)
            pass

        all_equipment = []
        page = 1

        while True:
            params.page = page
            batch = await self.list_equipment(params)

            if not batch:
                break

            all_equipment.extend(batch)
            page += 1

            if len(batch) < params.page_size:
                break

        self._logger.info(f"Synchronized {len(all_equipment)} equipment records")
        return all_equipment

    # =========================================================================
    # Maintenance History Operations
    # =========================================================================

    async def get_maintenance_history(
        self,
        equipment_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        maintenance_type: Optional[MaintenanceType] = None,
        limit: int = 100,
    ) -> List[MaintenanceHistory]:
        """
        Get maintenance history for equipment.

        Args:
            equipment_id: Equipment identifier
            start_date: History start date
            end_date: History end date
            maintenance_type: Filter by maintenance type
            limit: Maximum records to return

        Returns:
            List of maintenance history records
        """
        endpoint = self._adapter.get_maintenance_history_endpoint()

        params: Dict[str, Any] = {
            "equipment_id": equipment_id,
            "limit": limit,
        }

        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if maintenance_type:
            params["type"] = maintenance_type.value

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            operation_name="get_maintenance_history",
        )

        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [MaintenanceHistory(**item) for item in items]

    # =========================================================================
    # Spare Parts Operations
    # =========================================================================

    async def search_spare_parts(
        self,
        search_term: Optional[str] = None,
        part_number: Optional[str] = None,
        equipment_id: Optional[str] = None,
        category: Optional[str] = None,
        in_stock_only: bool = False,
        page: int = 1,
        page_size: int = 50,
    ) -> List[SparePart]:
        """
        Search spare parts inventory.

        Args:
            search_term: Search term
            part_number: Part number filter
            equipment_id: Compatible equipment filter
            category: Category filter
            in_stock_only: Only return in-stock parts
            page: Page number
            page_size: Page size

        Returns:
            List of spare parts
        """
        if not self._cmms_config.enable_spare_parts:
            raise ConfigurationError("Spare parts operations disabled")

        endpoint = self._adapter.get_spare_parts_endpoint()

        params: Dict[str, Any] = {
            "page": page,
            "limit": page_size,
        }

        if search_term:
            params["search"] = search_term
        if part_number:
            params["part_number"] = part_number
        if equipment_id:
            params["equipment_id"] = equipment_id
        if category:
            params["category"] = category
        if in_stock_only:
            params["in_stock"] = "true"

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            operation_name="search_spare_parts",
        )

        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [SparePart(**item) for item in items]

    async def check_part_availability(
        self,
        part_id: str,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """
        Check spare part availability.

        Args:
            part_id: Part identifier
            quantity: Required quantity

        Returns:
            Availability information
        """
        if not self._cmms_config.enable_spare_parts:
            raise ConfigurationError("Spare parts operations disabled")

        endpoint = f"{self._adapter.get_spare_parts_endpoint()}/{part_id}/availability"

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params={"quantity": quantity},
            operation_name="check_part_availability",
        )

        return response

    # =========================================================================
    # Cost Tracking Operations
    # =========================================================================

    async def get_equipment_costs(
        self,
        equipment_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cost_type: Optional[str] = None,
    ) -> List[CostRecord]:
        """
        Get cost records for equipment.

        Args:
            equipment_id: Equipment identifier
            start_date: Period start date
            end_date: Period end date
            cost_type: Filter by cost type

        Returns:
            List of cost records
        """
        if not self._cmms_config.enable_cost_tracking:
            raise ConfigurationError("Cost tracking disabled")

        endpoint = f"{self._adapter.get_base_url()}/api/{self._cmms_config.api_version}/costs"

        params: Dict[str, Any] = {
            "equipment_id": equipment_id,
        }

        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if cost_type:
            params["cost_type"] = cost_type

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            operation_name="get_equipment_costs",
        )

        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [CostRecord(**item) for item in items]

    async def get_total_maintenance_cost(
        self,
        equipment_id: str,
        period_months: int = 12,
    ) -> Dict[str, float]:
        """
        Get total maintenance cost summary for equipment.

        Args:
            equipment_id: Equipment identifier
            period_months: Period in months

        Returns:
            Cost summary by category
        """
        start_date = datetime.utcnow() - timedelta(days=period_months * 30)
        costs = await self.get_equipment_costs(equipment_id, start_date=start_date)

        summary: Dict[str, float] = {
            "labor": 0.0,
            "parts": 0.0,
            "external": 0.0,
            "other": 0.0,
            "total": 0.0,
        }

        for cost in costs:
            cost_type = cost.cost_type.lower()
            if cost_type in summary:
                summary[cost_type] += cost.amount
            else:
                summary["other"] += cost.amount
            summary["total"] += cost.amount

        return summary

    # =========================================================================
    # Notification Operations
    # =========================================================================

    async def get_notifications(
        self,
        equipment_id: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[WorkOrderPriority] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[Notification]:
        """
        Get notifications from CMMS.

        Args:
            equipment_id: Filter by equipment
            status: Filter by status
            priority: Filter by priority
            unacknowledged_only: Only return unacknowledged
            limit: Maximum records

        Returns:
            List of notifications
        """
        if not self._cmms_config.enable_notifications:
            raise ConfigurationError("Notifications disabled")

        endpoint = f"{self._adapter.get_base_url()}/api/{self._cmms_config.api_version}/notifications"

        params: Dict[str, Any] = {"limit": limit}

        if equipment_id:
            params["equipment_id"] = equipment_id
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority.value
        if unacknowledged_only:
            params["acknowledged"] = "false"

        response = await self._make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            operation_name="get_notifications",
        )

        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [Notification(**item) for item in items]

    async def create_notification(
        self,
        notification_type: str,
        title: str,
        message: str,
        equipment_id: Optional[str] = None,
        priority: WorkOrderPriority = WorkOrderPriority.MEDIUM,
        assigned_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """
        Create a notification in CMMS.

        Args:
            notification_type: Notification type
            title: Notification title
            message: Notification message
            equipment_id: Related equipment ID
            priority: Priority level
            assigned_to: Assigned user ID
            metadata: Additional metadata

        Returns:
            Created notification
        """
        if not self._cmms_config.enable_notifications:
            raise ConfigurationError("Notifications disabled")

        endpoint = f"{self._adapter.get_base_url()}/api/{self._cmms_config.api_version}/notifications"

        data = {
            "notification_type": notification_type,
            "title": title,
            "message": message,
            "priority": priority.value,
            "status": "new",
        }

        if equipment_id:
            data["equipment_id"] = equipment_id
        if assigned_to:
            data["assigned_to"] = assigned_to
        if metadata:
            data["metadata"] = metadata

        response = await self._make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            operation_name="create_notification",
        )

        return Notification(**response)

    async def acknowledge_notification(self, notification_id: str) -> Notification:
        """
        Acknowledge a notification.

        Args:
            notification_id: Notification identifier

        Returns:
            Updated notification
        """
        if not self._cmms_config.enable_notifications:
            raise ConfigurationError("Notifications disabled")

        endpoint = (
            f"{self._adapter.get_base_url()}/api/{self._cmms_config.api_version}"
            f"/notifications/{notification_id}/acknowledge"
        )

        response = await self._make_request(
            method="POST",
            endpoint=endpoint,
            operation_name="acknowledge_notification",
        )

        return Notification(**response)

    # =========================================================================
    # Predictive Maintenance Integration
    # =========================================================================

    async def create_predictive_work_order(
        self,
        equipment_id: str,
        prediction_id: str,
        predicted_failure_date: datetime,
        confidence_score: float,
        remaining_useful_life_hours: float,
        recommended_action: str,
        priority: Optional[WorkOrderPriority] = None,
    ) -> WorkOrder:
        """
        Create a predictive maintenance work order based on ML predictions.

        Args:
            equipment_id: Equipment identifier
            prediction_id: ID of the prediction that triggered this
            predicted_failure_date: Predicted failure date
            confidence_score: Prediction confidence (0-1)
            remaining_useful_life_hours: Remaining useful life in hours
            recommended_action: Recommended maintenance action
            priority: Priority (auto-calculated if not provided)

        Returns:
            Created work order
        """
        # Auto-calculate priority based on RUL and confidence
        if priority is None:
            if remaining_useful_life_hours < 24 and confidence_score > 0.8:
                priority = WorkOrderPriority.EMERGENCY
            elif remaining_useful_life_hours < 72 and confidence_score > 0.7:
                priority = WorkOrderPriority.URGENT
            elif remaining_useful_life_hours < 168 and confidence_score > 0.6:
                priority = WorkOrderPriority.HIGH
            else:
                priority = WorkOrderPriority.MEDIUM

        # Calculate scheduled dates
        schedule_buffer_hours = remaining_useful_life_hours * 0.7  # Schedule at 70% of RUL
        scheduled_start = datetime.utcnow() + timedelta(hours=schedule_buffer_hours)
        due_date = predicted_failure_date - timedelta(hours=24)  # Due 24h before predicted failure

        request = WorkOrderCreateRequest(
            title=f"Predictive Maintenance - {recommended_action}",
            description=(
                f"Predictive maintenance work order generated by GL-013 PREDICTMAINT.\n\n"
                f"Prediction ID: {prediction_id}\n"
                f"Predicted Failure Date: {predicted_failure_date.isoformat()}\n"
                f"Confidence Score: {confidence_score:.1%}\n"
                f"Remaining Useful Life: {remaining_useful_life_hours:.1f} hours\n\n"
                f"Recommended Action: {recommended_action}"
            ),
            work_order_type=WorkOrderType.PREDICTIVE,
            priority=priority,
            equipment_id=equipment_id,
            scheduled_start=scheduled_start,
            due_date=due_date,
            prediction_id=prediction_id,
            predicted_failure_date=predicted_failure_date,
            confidence_score=confidence_score,
            remaining_useful_life_hours=remaining_useful_life_hours,
            metadata={
                "generated_by": "GL-013_PREDICTMAINT",
                "prediction_id": prediction_id,
                "algorithm": "predictive_maintenance",
            },
        )

        return await self.create_work_order(request)

    async def get_equipment_prediction_history(
        self,
        equipment_id: str,
        limit: int = 10,
    ) -> List[WorkOrder]:
        """
        Get history of predictive maintenance work orders for equipment.

        Args:
            equipment_id: Equipment identifier
            limit: Maximum records

        Returns:
            List of predictive maintenance work orders
        """
        return await self.list_work_orders(
            equipment_id=equipment_id,
            work_order_type=WorkOrderType.PREDICTIVE,
            page_size=limit,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_cmms_connector(
    provider: CMSProvider,
    base_url: str,
    connector_name: str,
    auth_type: AuthenticationType = AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
    oauth2_config: Optional[OAuth2Config] = None,
    api_key_config: Optional[APIKeyConfig] = None,
    basic_auth_config: Optional[BasicAuthConfig] = None,
    **kwargs,
) -> CMSSConnector:
    """
    Factory function to create CMMS connector.

    Args:
        provider: CMMS provider
        base_url: Base URL
        connector_name: Connector name
        auth_type: Authentication type
        oauth2_config: OAuth2 configuration
        api_key_config: API key configuration
        basic_auth_config: Basic auth configuration
        **kwargs: Additional configuration options

    Returns:
        Configured CMMS connector
    """
    config = CMSSConnectorConfig(
        connector_name=connector_name,
        connector_type=ConnectorType.CMMS,
        provider=provider,
        base_url=base_url,
        auth_type=auth_type,
        oauth2_config=oauth2_config,
        api_key_config=api_key_config,
        basic_auth_config=basic_auth_config,
        **kwargs,
    )

    return CMSSConnector(config)
