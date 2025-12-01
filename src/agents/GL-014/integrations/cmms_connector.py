"""
CMMS Connector Module for GL-014 EXCHANGER-PRO (Heat Exchanger Optimization Agent).

Provides integration with Computerized Maintenance Management Systems including:
- SAP Plant Maintenance (PM)
- IBM Maximo
- Oracle Enterprise Asset Management (EAM)

Supports work order creation for cleaning schedules, equipment master data sync,
maintenance history retrieval, and notification creation.

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

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

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
    AuthenticationType,
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
    """Work order types for heat exchanger maintenance."""

    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    CLEANING = "cleaning"  # Heat exchanger specific
    CONDITION_BASED = "condition_based"
    FOULING_MITIGATION = "fouling_mitigation"  # Heat exchanger specific


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
    CLEANING = "cleaning"


class CleaningMethod(str, Enum):
    """Heat exchanger cleaning methods."""

    CHEMICAL_CLEANING = "chemical_cleaning"
    MECHANICAL_CLEANING = "mechanical_cleaning"
    HYDRO_BLASTING = "hydro_blasting"
    STEAM_CLEANING = "steam_cleaning"
    PIGGING = "pigging"
    ONLINE_CLEANING = "online_cleaning"
    OFFLINE_CLEANING = "offline_cleaning"
    THERMAL_SHOCK = "thermal_shock"
    ULTRASONIC = "ultrasonic"


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
    username: Optional[str] = Field(default=None, description="Username for password grant")
    password: Optional[str] = Field(default=None, description="Password for password grant")
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
    query_param_name: Optional[str] = Field(default=None, description="Query param name")


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
    oauth2_config: Optional[OAuth2Config] = Field(default=None, description="OAuth2 config")
    api_key_config: Optional[APIKeyConfig] = Field(default=None, description="API key config")
    basic_auth_config: Optional[BasicAuthConfig] = Field(default=None, description="Basic auth")

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

    # Provider-specific settings
    sap_client: Optional[str] = Field(default=None, description="SAP client number")
    sap_system_id: Optional[str] = Field(default=None, description="SAP system ID")
    maximo_site_id: Optional[str] = Field(default=None, description="Maximo site ID")
    maximo_org_id: Optional[str] = Field(default=None, description="Maximo organization ID")
    oracle_business_unit: Optional[str] = Field(default=None, description="Oracle business unit")

    # Feature flags
    enable_work_orders: bool = Field(default=True, description="Enable work order operations")
    enable_equipment_sync: bool = Field(default=True, description="Enable equipment sync")
    enable_spare_parts: bool = Field(default=True, description="Enable spare parts queries")
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    enable_notifications: bool = Field(default=True, description="Enable notifications")

    # Heat exchanger specific
    enable_cleaning_schedules: bool = Field(
        default=True,
        description="Enable cleaning schedule management"
    )
    default_cleaning_method: CleaningMethod = Field(
        default=CleaningMethod.CHEMICAL_CLEANING,
        description="Default cleaning method"
    )

    @field_validator('connector_type', mode='before')
    @classmethod
    def set_connector_type(cls, v):
        return ConnectorType.CMMS


# =============================================================================
# Pydantic Models - Data Objects
# =============================================================================


class HeatExchangerEquipment(BaseModel):
    """Heat exchanger equipment master data model."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_name: str = Field(..., description="Equipment name")
    description: Optional[str] = Field(default=None, description="Equipment description")
    equipment_tag: Optional[str] = Field(default=None, description="Equipment tag number")

    # Classification
    equipment_type: str = Field(default="heat_exchanger", description="Equipment type")
    equipment_subtype: Optional[str] = Field(
        default=None,
        description="Subtype (shell_tube, plate, air_cooled, etc.)"
    )
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
    process_unit: Optional[str] = Field(default=None, description="Process unit")

    # Technical specifications
    manufacturer: Optional[str] = Field(default=None, description="Manufacturer")
    model: Optional[str] = Field(default=None, description="Model number")
    serial_number: Optional[str] = Field(default=None, description="Serial number")
    asset_tag: Optional[str] = Field(default=None, description="Asset tag")

    # Heat exchanger specific
    design_duty_kw: Optional[float] = Field(default=None, description="Design duty (kW)")
    design_area_m2: Optional[float] = Field(default=None, description="Design area (m2)")
    design_ua_w_k: Optional[float] = Field(default=None, description="Design UA (W/K)")
    shell_material: Optional[str] = Field(default=None, description="Shell material")
    tube_material: Optional[str] = Field(default=None, description="Tube material")
    number_of_tubes: Optional[int] = Field(default=None, description="Number of tubes")
    tube_length_m: Optional[float] = Field(default=None, description="Tube length (m)")
    shell_passes: Optional[int] = Field(default=None, description="Shell passes")
    tube_passes: Optional[int] = Field(default=None, description="Tube passes")

    # Operating conditions
    hot_side_fluid: Optional[str] = Field(default=None, description="Hot side fluid")
    cold_side_fluid: Optional[str] = Field(default=None, description="Cold side fluid")
    max_pressure_kpa: Optional[float] = Field(default=None, description="Max pressure (kPa)")
    max_temperature_c: Optional[float] = Field(default=None, description="Max temp (C)")

    # Fouling
    design_fouling_factor: Optional[float] = Field(
        default=None,
        description="Design fouling factor (m2K/W)"
    )
    cleaning_interval_days: Optional[int] = Field(
        default=None,
        description="Recommended cleaning interval"
    )
    last_cleaning_date: Optional[datetime] = Field(default=None, description="Last cleaning")
    next_cleaning_date: Optional[datetime] = Field(default=None, description="Next cleaning")

    # Dates
    installation_date: Optional[datetime] = Field(default=None, description="Installation date")
    warranty_expiry_date: Optional[datetime] = Field(default=None, description="Warranty expiry")
    last_maintenance_date: Optional[datetime] = Field(default=None, description="Last maintenance")
    next_maintenance_date: Optional[datetime] = Field(default=None, description="Next maintenance")

    # Financial
    acquisition_cost: Optional[float] = Field(default=None, ge=0, description="Acquisition cost")
    current_value: Optional[float] = Field(default=None, ge=0, description="Current book value")
    cost_center: Optional[str] = Field(default=None, description="Cost center")

    # Relationships
    parent_equipment_id: Optional[str] = Field(default=None, description="Parent equipment ID")
    children_ids: List[str] = Field(default_factory=list, description="Child equipment IDs")

    # Metadata
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class CleaningWorkOrder(BaseModel):
    """Work order model for heat exchanger cleaning."""

    model_config = ConfigDict(extra="allow")

    work_order_id: str = Field(..., description="Unique work order identifier")
    work_order_number: Optional[str] = Field(default=None, description="Work order number")
    title: str = Field(..., description="Work order title")
    description: Optional[str] = Field(default=None, description="Detailed description")

    # Classification
    work_order_type: WorkOrderType = Field(
        default=WorkOrderType.CLEANING,
        description="Work order type"
    )
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

    # Cleaning specific
    cleaning_method: CleaningMethod = Field(
        default=CleaningMethod.CHEMICAL_CLEANING,
        description="Cleaning method"
    )
    cleaning_reason: Optional[str] = Field(
        default=None,
        description="Reason for cleaning (fouling level, schedule, etc.)"
    )
    current_fouling_factor: Optional[float] = Field(
        default=None,
        description="Current fouling factor before cleaning"
    )
    target_fouling_factor: Optional[float] = Field(
        default=None,
        description="Target fouling factor after cleaning"
    )
    current_ua_percent: Optional[float] = Field(
        default=None,
        description="Current UA as % of design"
    )
    estimated_efficiency_gain: Optional[float] = Field(
        default=None,
        description="Estimated efficiency improvement (%)"
    )

    # Scheduling
    created_date: datetime = Field(default_factory=datetime.utcnow, description="Creation date")
    scheduled_start: Optional[datetime] = Field(default=None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(default=None, description="Scheduled end")
    actual_start: Optional[datetime] = Field(default=None, description="Actual start")
    actual_end: Optional[datetime] = Field(default=None, description="Actual end")
    due_date: Optional[datetime] = Field(default=None, description="Due date")

    # Downtime
    estimated_downtime_hours: Optional[float] = Field(
        default=None,
        description="Estimated downtime"
    )
    actual_downtime_hours: Optional[float] = Field(
        default=None,
        description="Actual downtime"
    )

    # Assignment
    assigned_to: Optional[str] = Field(default=None, description="Assigned technician ID")
    assigned_team: Optional[str] = Field(default=None, description="Assigned team/crew")
    created_by: Optional[str] = Field(default=None, description="Created by user ID")

    # Effort
    estimated_hours: Optional[float] = Field(default=None, ge=0, description="Estimated hours")
    actual_hours: Optional[float] = Field(default=None, ge=0, description="Actual hours")

    # Cost
    estimated_cost: Optional[float] = Field(default=None, ge=0, description="Estimated cost")
    actual_cost: Optional[float] = Field(default=None, ge=0, description="Actual cost")
    cost_center: Optional[str] = Field(default=None, description="Cost center")

    # Chemical/materials for cleaning
    chemicals_required: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chemicals required for cleaning"
    )
    spare_parts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Spare parts required"
    )

    # Predictive maintenance fields
    prediction_id: Optional[str] = Field(default=None, description="Related prediction ID")
    predicted_fouling_date: Optional[datetime] = Field(
        default=None,
        description="Predicted critical fouling date"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Prediction confidence"
    )

    # Safety
    isolation_required: bool = Field(default=True, description="Isolation required")
    permit_required: bool = Field(default=True, description="Work permit required")
    safety_notes: Optional[str] = Field(default=None, description="Safety notes")

    # Completion
    completion_notes: Optional[str] = Field(default=None, description="Completion notes")
    post_cleaning_fouling: Optional[float] = Field(
        default=None,
        description="Post-cleaning fouling factor"
    )
    post_cleaning_ua_percent: Optional[float] = Field(
        default=None,
        description="Post-cleaning UA %"
    )
    efficiency_improvement: Optional[float] = Field(
        default=None,
        description="Actual efficiency improvement (%)"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MaintenanceHistory(BaseModel):
    """Maintenance history record for heat exchanger."""

    model_config = ConfigDict(extra="allow")

    history_id: str = Field(..., description="History record ID")
    equipment_id: str = Field(..., description="Equipment ID")
    work_order_id: Optional[str] = Field(default=None, description="Related work order ID")

    # Maintenance details
    maintenance_type: MaintenanceType = Field(..., description="Type of maintenance")
    maintenance_date: datetime = Field(..., description="Maintenance date")
    description: str = Field(..., description="Description of work performed")

    # Cleaning specific
    cleaning_method: Optional[CleaningMethod] = Field(
        default=None,
        description="Cleaning method used"
    )
    pre_cleaning_fouling: Optional[float] = Field(
        default=None,
        description="Fouling factor before cleaning"
    )
    post_cleaning_fouling: Optional[float] = Field(
        default=None,
        description="Fouling factor after cleaning"
    )
    fouling_removed_percent: Optional[float] = Field(
        default=None,
        description="Percentage of fouling removed"
    )
    pre_cleaning_ua_percent: Optional[float] = Field(
        default=None,
        description="UA % before cleaning"
    )
    post_cleaning_ua_percent: Optional[float] = Field(
        default=None,
        description="UA % after cleaning"
    )

    # Outcome
    downtime_hours: Optional[float] = Field(default=None, ge=0, description="Downtime hours")
    labor_hours: Optional[float] = Field(default=None, ge=0, description="Labor hours")
    total_cost: Optional[float] = Field(default=None, ge=0, description="Total cost")

    # Parts/chemicals used
    materials_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Materials used"
    )

    # Technician
    technician_id: Optional[str] = Field(default=None, description="Technician ID")
    technician_name: Optional[str] = Field(default=None, description="Technician name")

    # Findings
    findings: Optional[str] = Field(default=None, description="Maintenance findings")
    recommendations: Optional[str] = Field(default=None, description="Recommendations")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CleaningSchedule(BaseModel):
    """Cleaning schedule for heat exchanger."""

    model_config = ConfigDict(extra="allow")

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Schedule ID"
    )
    equipment_id: str = Field(..., description="Equipment ID")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")

    # Schedule parameters
    cleaning_method: CleaningMethod = Field(..., description="Cleaning method")
    interval_days: int = Field(..., ge=1, description="Cleaning interval in days")
    based_on: str = Field(
        default="time",
        description="Based on: time, condition, predictive"
    )

    # Thresholds for condition-based
    fouling_threshold: Optional[float] = Field(
        default=None,
        description="Fouling factor threshold for trigger"
    )
    ua_percent_threshold: Optional[float] = Field(
        default=None,
        description="UA % threshold for trigger"
    )
    pressure_drop_threshold: Optional[float] = Field(
        default=None,
        description="Pressure drop threshold (kPa)"
    )

    # Schedule tracking
    last_cleaning_date: Optional[datetime] = Field(default=None, description="Last cleaning")
    next_cleaning_date: Optional[datetime] = Field(default=None, description="Next scheduled")
    auto_generate_work_orders: bool = Field(
        default=True,
        description="Auto-generate work orders"
    )
    lead_time_days: int = Field(
        default=14,
        description="Days before to create work order"
    )

    # Status
    is_active: bool = Field(default=True, description="Schedule is active")
    suspended_until: Optional[datetime] = Field(default=None, description="Suspended until")

    # Metadata
    created_date: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created")
    acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledged")
    resolved_at: Optional[datetime] = Field(default=None, description="Resolved")

    # Assignment
    assigned_to: Optional[str] = Field(default=None, description="Assigned user ID")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Request/Response Models
# =============================================================================


class CleaningWorkOrderCreateRequest(BaseModel):
    """Request to create a cleaning work order."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=255, description="Work order title")
    description: Optional[str] = Field(default=None, max_length=4000, description="Description")
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority"
    )

    equipment_id: str = Field(..., description="Equipment ID")
    location_id: Optional[str] = Field(default=None, description="Location ID")

    # Cleaning specific
    cleaning_method: CleaningMethod = Field(
        default=CleaningMethod.CHEMICAL_CLEANING,
        description="Cleaning method"
    )
    cleaning_reason: Optional[str] = Field(default=None, description="Cleaning reason")
    current_fouling_factor: Optional[float] = Field(default=None, description="Current fouling")
    current_ua_percent: Optional[float] = Field(default=None, description="Current UA %")

    # Scheduling
    scheduled_start: Optional[datetime] = Field(default=None, description="Scheduled start")
    scheduled_end: Optional[datetime] = Field(default=None, description="Scheduled end")
    due_date: Optional[datetime] = Field(default=None, description="Due date")

    # Assignment
    assigned_to: Optional[str] = Field(default=None, description="Assigned technician")
    assigned_team: Optional[str] = Field(default=None, description="Assigned team")

    # Effort
    estimated_hours: Optional[float] = Field(default=None, ge=0, description="Estimated hours")
    estimated_downtime_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated downtime"
    )
    estimated_cost: Optional[float] = Field(default=None, ge=0, description="Estimated cost")

    # Safety
    isolation_required: bool = Field(default=True, description="Isolation required")
    permit_required: bool = Field(default=True, description="Work permit required")
    safety_notes: Optional[str] = Field(default=None, description="Safety notes")

    # Predictive maintenance
    prediction_id: Optional[str] = Field(default=None, description="Prediction ID")
    predicted_fouling_date: Optional[datetime] = Field(default=None, description="Predicted date")
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1, description="Confidence")

    # Chemicals/materials
    chemicals_required: List[Dict[str, Any]] = Field(default_factory=list)
    spare_parts: List[Dict[str, Any]] = Field(default_factory=list)

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
    actual_downtime_hours: Optional[float] = Field(default=None, ge=0)
    actual_cost: Optional[float] = Field(default=None, ge=0)

    completion_notes: Optional[str] = Field(default=None, max_length=4000)
    post_cleaning_fouling: Optional[float] = Field(default=None)
    post_cleaning_ua_percent: Optional[float] = Field(default=None)
    efficiency_improvement: Optional[float] = Field(default=None)

    metadata: Optional[Dict[str, Any]] = Field(default=None)


class EquipmentQueryParams(BaseModel):
    """Query parameters for equipment listing."""

    model_config = ConfigDict(extra="forbid")

    plant_id: Optional[str] = Field(default=None, description="Filter by plant")
    location_id: Optional[str] = Field(default=None, description="Filter by location")
    equipment_type: Optional[str] = Field(default=None, description="Filter by type")
    status: Optional[EquipmentStatus] = Field(default=None, description="Filter by status")
    criticality: Optional[EquipmentCriticality] = Field(default=None, description="Criticality")
    search: Optional[str] = Field(default=None, description="Search term")
    needs_cleaning: Optional[bool] = Field(default=None, description="Needs cleaning soon")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=500, description="Page size")
    sort_by: str = Field(default="equipment_name", description="Sort field")
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")


# =============================================================================
# Authentication Handler
# =============================================================================


class CMSSAuthenticationHandler:
    """Handles authentication for CMMS API calls."""

    def __init__(
        self,
        auth_type: AuthenticationType,
        oauth2_config: Optional[OAuth2Config] = None,
        api_key_config: Optional[APIKeyConfig] = None,
        basic_auth_config: Optional[BasicAuthConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Initialize authentication handler."""
        self._auth_type = auth_type
        self._oauth2_config = oauth2_config
        self._api_key_config = api_key_config
        self._basic_auth_config = basic_auth_config
        self._session = session

        # Token state
        self._access_token: Optional[str] = None
        self._token_type: str = "Bearer"
        self._token_expires_at: Optional[datetime] = None
        self._refresh_token: Optional[str] = None
        self._lock = asyncio.Lock()

    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if self._auth_type == AuthenticationType.API_KEY:
            if not self._api_key_config:
                raise ConfigurationError("API key configuration required")
            return {self._api_key_config.header_name: self._api_key_config.api_key}

        elif self._auth_type == AuthenticationType.BASIC:
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
        ]:
            token = await self._ensure_valid_token()
            return {"Authorization": f"{self._token_type} {token}"}

        else:
            return {}

    async def _ensure_valid_token(self) -> str:
        """Ensure we have a valid OAuth2 token, refreshing if needed."""
        async with self._lock:
            if self._access_token and self._token_expires_at:
                buffer = timedelta(
                    seconds=self._oauth2_config.token_refresh_buffer_seconds
                )
                if datetime.utcnow() < self._token_expires_at - buffer:
                    return self._access_token

            await self._refresh_oauth2_token()
            return self._access_token

    async def _refresh_oauth2_token(self) -> None:
        """Refresh OAuth2 access token."""
        if not self._oauth2_config:
            raise ConfigurationError("OAuth2 configuration required")

        if not self._session:
            raise ConfigurationError("HTTP session required for OAuth2")

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
            async with self._session.post(
                self._oauth2_config.token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                response.raise_for_status()
                token_response = await response.json()

                self._access_token = token_response["access_token"]
                self._token_type = token_response.get("token_type", "Bearer")

                expires_in = token_response.get("expires_in", 3600)
                self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                if "refresh_token" in token_response:
                    self._refresh_token = token_response["refresh_token"]

                logger.info("OAuth2 token refreshed successfully")

        except aiohttp.ClientResponseError as e:
            raise AuthenticationError(
                f"OAuth2 token refresh failed: {e.status}",
                details={"message": str(e)},
            )
        except Exception as e:
            raise AuthenticationError(f"OAuth2 token refresh error: {str(e)}")

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

    def transform_work_order_to_api(
        self,
        request: CleaningWorkOrderCreateRequest
    ) -> Dict[str, Any]:
        """Transform work order create request to API format."""
        return request.model_dump(exclude_none=True)

    def transform_api_to_work_order(self, data: Dict[str, Any]) -> CleaningWorkOrder:
        """Transform API response to work order model."""
        return CleaningWorkOrder(**data)

    def transform_api_to_equipment(self, data: Dict[str, Any]) -> HeatExchangerEquipment:
        """Transform API response to equipment model."""
        return HeatExchangerEquipment(**data)

    def get_work_orders_endpoint(self) -> str:
        """Get work orders API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/workorders"

    def get_equipment_endpoint(self) -> str:
        """Get equipment API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/equipment"

    def get_maintenance_history_endpoint(self) -> str:
        """Get maintenance history API endpoint."""
        return f"{self.get_base_url()}/api/{self._config.api_version}/maintenance-history"


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

    def transform_work_order_to_api(
        self,
        request: CleaningWorkOrderCreateRequest
    ) -> Dict[str, Any]:
        """Transform to SAP PM format."""
        # Map cleaning to SAP PM order type
        order_type = "PM01"  # Default preventive
        if request.cleaning_method == CleaningMethod.CHEMICAL_CLEANING:
            order_type = "PM07"  # Chemical cleaning
        elif request.cleaning_method == CleaningMethod.MECHANICAL_CLEANING:
            order_type = "PM08"  # Mechanical cleaning

        return {
            "MaintenanceOrderType": order_type,
            "MaintenanceOrderDesc": request.title,
            "LongTextString": request.description or "",
            "Equipment": request.equipment_id,
            "FunctionalLocation": request.location_id or "",
            "MaintPriority": self._map_priority(request.priority),
            "BasicSchedulingStartDate": (
                request.scheduled_start.isoformat() if request.scheduled_start else None
            ),
            "BasicSchedulingEndDate": (
                request.scheduled_end.isoformat() if request.scheduled_end else None
            ),
            "PlannedWorkQuantity": request.estimated_hours or 0,
            "ProfitCenter": request.cost_center if hasattr(request, 'cost_center') else "",
        }

    def transform_api_to_work_order(self, data: Dict[str, Any]) -> CleaningWorkOrder:
        """Transform SAP PM response to work order model."""
        return CleaningWorkOrder(
            work_order_id=data.get("MaintenanceOrder", ""),
            work_order_number=data.get("MaintenanceOrder"),
            title=data.get("MaintenanceOrderDesc", ""),
            description=data.get("LongTextString"),
            work_order_type=WorkOrderType.CLEANING,
            priority=self._reverse_map_priority(data.get("MaintPriority", "")),
            status=self._map_sap_status(data.get("SystemStatus", "")),
            equipment_id=data.get("Equipment", ""),
            location_id=data.get("FunctionalLocation"),
            scheduled_start=self._parse_date(data.get("BasicSchedulingStartDate")),
            scheduled_end=self._parse_date(data.get("BasicSchedulingEndDate")),
            estimated_hours=data.get("PlannedWorkQuantity"),
            metadata={"sap_raw": data},
        )

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

    def transform_work_order_to_api(
        self,
        request: CleaningWorkOrderCreateRequest
    ) -> Dict[str, Any]:
        """Transform to Maximo format."""
        return {
            "description": request.title,
            "description_longdescription": request.description or "",
            "worktype": "CLN",  # Cleaning work type
            "assetnum": request.equipment_id,
            "location": request.location_id or "",
            "wopriority": self._map_priority(request.priority),
            "siteid": self._config.maximo_site_id or "",
            "orgid": self._config.maximo_org_id or "",
            "schedstart": (
                request.scheduled_start.isoformat() if request.scheduled_start else None
            ),
            "schedfinish": (
                request.scheduled_end.isoformat() if request.scheduled_end else None
            ),
            "estlabhrs": request.estimated_hours or 0,
        }

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


class OracleEAMAdapter(CMSSProviderAdapter):
    """Adapter for Oracle Enterprise Asset Management."""

    def get_work_orders_endpoint(self) -> str:
        return (
            f"{self.get_base_url()}/fscmRestApi/resources/"
            f"{self._config.api_version}/maintenanceWorkOrders"
        )

    def get_equipment_endpoint(self) -> str:
        return f"{self.get_base_url()}/fscmRestApi/resources/{self._config.api_version}/assets"

    def transform_work_order_to_api(
        self,
        request: CleaningWorkOrderCreateRequest
    ) -> Dict[str, Any]:
        """Transform to Oracle EAM format."""
        return {
            "WorkOrderDescription": request.title,
            "WorkOrderLongDescription": request.description or "",
            "WorkOrderType": "CLEANING",
            "AssetNumber": request.equipment_id,
            "MaintenanceOrganizationCode": self._config.oracle_business_unit or "",
            "Priority": self._map_priority(request.priority),
            "ScheduledStartDate": (
                request.scheduled_start.isoformat() if request.scheduled_start else None
            ),
            "ScheduledCompletionDate": (
                request.scheduled_end.isoformat() if request.scheduled_end else None
            ),
        }

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


# =============================================================================
# CMMS Connector Implementation
# =============================================================================


class CMSSConnector(BaseConnector):
    """
    CMMS Connector for GL-014 EXCHANGER-PRO.

    Provides integration with enterprise CMMS platforms for heat exchanger
    maintenance management including:
    - SAP Plant Maintenance (PM)
    - IBM Maximo
    - Oracle Enterprise Asset Management (EAM)

    Features:
    - Cleaning work order creation and management
    - Heat exchanger equipment master data sync
    - Maintenance history retrieval
    - Cleaning schedule management
    - Fouling-based predictive work orders
    """

    def __init__(self, config: CMSSConnectorConfig) -> None:
        """Initialize CMMS connector."""
        super().__init__(config)
        self._cmms_config = config

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Provider adapter
        self._adapter = self._create_adapter()

        # Authentication handler
        self._auth_handler: Optional[CMSSAuthenticationHandler] = None

        # Cleaning schedules cache
        self._cleaning_schedules: Dict[str, CleaningSchedule] = {}

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

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(
            total=self._cmms_config.request_timeout_seconds,
            connect=30,
            sock_read=60
        )

        connector = aiohttp.TCPConnector(
            limit=self._config.pool_max_size,
            limit_per_host=self._config.pool_max_size,
            keepalive_timeout=self._config.pool_keepalive_timeout_seconds,
        )

        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        )

        # Initialize authentication handler
        self._auth_handler = CMSSAuthenticationHandler(
            auth_type=self._cmms_config.auth_type,
            oauth2_config=self._cmms_config.oauth2_config,
            api_key_config=self._cmms_config.api_key_config,
            basic_auth_config=self._cmms_config.basic_auth_config,
            session=self._session,
        )

        # Test connection
        try:
            await self._test_connection()
            self._state = ConnectionState.CONNECTED
            self._logger.info(
                f"Successfully connected to {self._cmms_config.provider.value} CMMS"
            )
        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Failed to connect to CMMS: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from CMMS."""
        if self._session:
            await self._session.close()
            self._session = None
        self._auth_handler = None
        self._state = ConnectionState.DISCONNECTED
        self._logger.info("Disconnected from CMMS")

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on CMMS connection."""
        start_time = time.time()

        try:
            if not self._session:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Session not initialized",
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
        if not self._cmms_config.base_url:
            raise ConfigurationError("Base URL is required")

        if self._cmms_config.auth_type in [
            AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
            AuthenticationType.OAUTH2_PASSWORD,
        ]:
            if not self._cmms_config.oauth2_config:
                raise ConfigurationError("OAuth2 configuration required")

        elif self._cmms_config.auth_type == AuthenticationType.API_KEY:
            if not self._cmms_config.api_key_config:
                raise ConfigurationError("API key configuration required")

        elif self._cmms_config.auth_type == AuthenticationType.BASIC:
            if not self._cmms_config.basic_auth_config:
                raise ConfigurationError("Basic auth configuration required")

        return True

    async def _test_connection(self) -> None:
        """Test connection with a simple API call."""
        headers = await self._auth_handler.get_auth_headers()
        headers.update(self._adapter.get_headers())

        url = self._adapter.get_equipment_endpoint()
        params = {"$top": "1"} if "sap" in self._cmms_config.provider.value else {"limit": "1"}

        async with self._session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        operation_name: str = "api_call",
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        async def _do_request() -> Dict[str, Any]:
            headers = await self._auth_handler.get_auth_headers()
            headers.update(self._adapter.get_headers())

            async with self._session.request(
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                json=data,
            ) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    raise RateLimitError(
                        "Rate limit exceeded",
                        retry_after_seconds=float(retry_after),
                    )

                if response.status == 401:
                    await self._auth_handler.invalidate_token()
                    raise AuthenticationError("Authentication failed")

                response.raise_for_status()

                if response.content_length and response.content_length > 0:
                    return await response.json()
                return {}

        return await self.execute_with_protection(
            _do_request,
            operation_name=operation_name,
            use_cache=method == "GET",
            cache_key=(
                self._generate_cache_key(endpoint, params)
                if method == "GET" else None
            ),
        )

    # =========================================================================
    # Cleaning Work Order Operations
    # =========================================================================

    async def create_cleaning_work_order(
        self,
        request: CleaningWorkOrderCreateRequest
    ) -> CleaningWorkOrder:
        """
        Create a cleaning work order for a heat exchanger.

        Args:
            request: Cleaning work order creation request

        Returns:
            Created cleaning work order
        """
        if not self._cmms_config.enable_work_orders:
            raise ConfigurationError("Work order operations disabled")

        self._logger.info(
            f"Creating cleaning work order for equipment {request.equipment_id}"
        )

        endpoint = self._adapter.get_work_orders_endpoint()
        data = self._adapter.transform_work_order_to_api(request)

        response = await self._make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            operation_name="create_cleaning_work_order",
        )

        work_order = self._adapter.transform_api_to_work_order(response)

        # Set cleaning-specific fields
        work_order.work_order_type = WorkOrderType.CLEANING
        work_order.cleaning_method = request.cleaning_method
        work_order.cleaning_reason = request.cleaning_reason
        work_order.current_fouling_factor = request.current_fouling_factor
        work_order.current_ua_percent = request.current_ua_percent

        self._logger.info(f"Created cleaning work order {work_order.work_order_id}")

        return work_order

    async def get_work_order(self, work_order_id: str) -> CleaningWorkOrder:
        """Get work order by ID."""
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
    ) -> CleaningWorkOrder:
        """Update an existing work order."""
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

        return self._adapter.transform_api_to_work_order(response)

    async def complete_cleaning_work_order(
        self,
        work_order_id: str,
        actual_hours: float,
        actual_downtime_hours: float,
        post_cleaning_fouling: Optional[float] = None,
        post_cleaning_ua_percent: Optional[float] = None,
        completion_notes: Optional[str] = None,
    ) -> CleaningWorkOrder:
        """
        Complete a cleaning work order with results.

        Args:
            work_order_id: Work order identifier
            actual_hours: Actual hours worked
            actual_downtime_hours: Actual downtime
            post_cleaning_fouling: Post-cleaning fouling factor
            post_cleaning_ua_percent: Post-cleaning UA percentage
            completion_notes: Completion notes

        Returns:
            Completed work order
        """
        # Get current work order to calculate improvement
        current = await self.get_work_order(work_order_id)
        efficiency_improvement = None

        if current.current_ua_percent and post_cleaning_ua_percent:
            efficiency_improvement = post_cleaning_ua_percent - current.current_ua_percent

        update_request = WorkOrderUpdateRequest(
            status=WorkOrderStatus.COMPLETED,
            actual_hours=actual_hours,
            actual_downtime_hours=actual_downtime_hours,
            actual_end=datetime.utcnow(),
            completion_notes=completion_notes,
            post_cleaning_fouling=post_cleaning_fouling,
            post_cleaning_ua_percent=post_cleaning_ua_percent,
            efficiency_improvement=efficiency_improvement,
        )

        return await self.update_work_order(work_order_id, update_request)

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
    ) -> List[CleaningWorkOrder]:
        """List work orders with optional filters."""
        if not self._cmms_config.enable_work_orders:
            raise ConfigurationError("Work order operations disabled")

        endpoint = self._adapter.get_work_orders_endpoint()
        page_size = page_size or self._cmms_config.max_page_size

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

        items = response.get("items", response.get("value", response.get("results", [])))
        if isinstance(response, list):
            items = response

        return [self._adapter.transform_api_to_work_order(item) for item in items]

    # =========================================================================
    # Heat Exchanger Equipment Operations
    # =========================================================================

    async def get_equipment(self, equipment_id: str) -> HeatExchangerEquipment:
        """Get heat exchanger equipment by ID."""
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
    ) -> List[HeatExchangerEquipment]:
        """List heat exchanger equipment with optional filters."""
        if not self._cmms_config.enable_equipment_sync:
            raise ConfigurationError("Equipment sync disabled")

        endpoint = self._adapter.get_equipment_endpoint()
        query_params = params.model_dump(exclude_none=True) if params else {}

        # Add heat exchanger filter
        if "equipment_type" not in query_params:
            query_params["equipment_type"] = "heat_exchanger"

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

    async def get_equipment_needing_cleaning(
        self,
        fouling_threshold: Optional[float] = None,
        ua_threshold: Optional[float] = None,
        days_until_scheduled: int = 30,
    ) -> List[HeatExchangerEquipment]:
        """
        Get heat exchangers that need cleaning.

        Args:
            fouling_threshold: Fouling factor threshold
            ua_threshold: UA percentage threshold
            days_until_scheduled: Days until scheduled cleaning

        Returns:
            List of equipment needing cleaning
        """
        all_equipment = await self.list_equipment()
        needs_cleaning = []

        now = datetime.utcnow()
        cutoff_date = now + timedelta(days=days_until_scheduled)

        for equip in all_equipment:
            should_add = False

            # Check scheduled cleaning
            if equip.next_cleaning_date and equip.next_cleaning_date <= cutoff_date:
                should_add = True

            # Check cleaning interval
            if equip.cleaning_interval_days and equip.last_cleaning_date:
                next_due = equip.last_cleaning_date + timedelta(
                    days=equip.cleaning_interval_days
                )
                if next_due <= cutoff_date:
                    should_add = True

            if should_add:
                needs_cleaning.append(equip)

        return needs_cleaning

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
        """Get maintenance history for heat exchanger."""
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

    async def get_cleaning_history(
        self,
        equipment_id: str,
        limit: int = 50,
    ) -> List[MaintenanceHistory]:
        """Get cleaning history for heat exchanger."""
        return await self.get_maintenance_history(
            equipment_id=equipment_id,
            maintenance_type=MaintenanceType.CLEANING,
            limit=limit,
        )

    # =========================================================================
    # Cleaning Schedule Operations
    # =========================================================================

    async def create_cleaning_schedule(
        self,
        schedule: CleaningSchedule
    ) -> CleaningSchedule:
        """Create or update cleaning schedule for equipment."""
        self._cleaning_schedules[schedule.equipment_id] = schedule

        # Calculate next cleaning date if not set
        if not schedule.next_cleaning_date:
            if schedule.last_cleaning_date:
                schedule.next_cleaning_date = (
                    schedule.last_cleaning_date +
                    timedelta(days=schedule.interval_days)
                )
            else:
                schedule.next_cleaning_date = (
                    datetime.utcnow() + timedelta(days=schedule.interval_days)
                )

        self._logger.info(
            f"Created cleaning schedule for {schedule.equipment_id}: "
            f"every {schedule.interval_days} days"
        )

        return schedule

    async def get_cleaning_schedule(
        self,
        equipment_id: str
    ) -> Optional[CleaningSchedule]:
        """Get cleaning schedule for equipment."""
        return self._cleaning_schedules.get(equipment_id)

    async def check_cleaning_schedules(self) -> List[CleaningWorkOrderCreateRequest]:
        """
        Check all cleaning schedules and return work orders to create.

        Returns:
            List of work order creation requests for due cleanings
        """
        work_orders_to_create = []
        now = datetime.utcnow()

        for equipment_id, schedule in self._cleaning_schedules.items():
            if not schedule.is_active:
                continue

            if schedule.suspended_until and schedule.suspended_until > now:
                continue

            if not schedule.auto_generate_work_orders:
                continue

            # Check if work order should be created
            if schedule.next_cleaning_date:
                lead_time = timedelta(days=schedule.lead_time_days)
                create_date = schedule.next_cleaning_date - lead_time

                if now >= create_date:
                    # Create work order request
                    request = CleaningWorkOrderCreateRequest(
                        title=f"Scheduled Cleaning - {equipment_id}",
                        description=(
                            f"Scheduled cleaning based on {schedule.based_on} schedule. "
                            f"Interval: {schedule.interval_days} days. "
                            f"Method: {schedule.cleaning_method.value}"
                        ),
                        equipment_id=equipment_id,
                        cleaning_method=schedule.cleaning_method,
                        cleaning_reason=f"scheduled_{schedule.based_on}",
                        scheduled_start=schedule.next_cleaning_date,
                        due_date=schedule.next_cleaning_date,
                    )
                    work_orders_to_create.append(request)

        return work_orders_to_create

    # =========================================================================
    # Predictive Cleaning Operations
    # =========================================================================

    async def create_predictive_cleaning_work_order(
        self,
        equipment_id: str,
        prediction_id: str,
        predicted_fouling_date: datetime,
        current_fouling_factor: float,
        current_ua_percent: float,
        confidence_score: float,
        cleaning_method: Optional[CleaningMethod] = None,
    ) -> CleaningWorkOrder:
        """
        Create a predictive cleaning work order based on fouling prediction.

        Args:
            equipment_id: Equipment identifier
            prediction_id: ID of the fouling prediction
            predicted_fouling_date: Predicted date fouling reaches critical
            current_fouling_factor: Current fouling factor
            current_ua_percent: Current UA percentage
            confidence_score: Prediction confidence (0-1)
            cleaning_method: Cleaning method to use

        Returns:
            Created cleaning work order
        """
        # Determine priority based on time to predicted fouling
        days_until_fouling = (predicted_fouling_date - datetime.utcnow()).days

        if days_until_fouling < 7 and confidence_score > 0.8:
            priority = WorkOrderPriority.URGENT
        elif days_until_fouling < 14 and confidence_score > 0.7:
            priority = WorkOrderPriority.HIGH
        elif days_until_fouling < 30:
            priority = WorkOrderPriority.MEDIUM
        else:
            priority = WorkOrderPriority.LOW

        # Schedule cleaning before predicted critical fouling
        schedule_buffer_days = max(7, int(days_until_fouling * 0.3))
        scheduled_start = predicted_fouling_date - timedelta(days=schedule_buffer_days)

        # Use default cleaning method if not specified
        cleaning_method = cleaning_method or self._cmms_config.default_cleaning_method

        request = CleaningWorkOrderCreateRequest(
            title=f"Predictive Cleaning - {equipment_id}",
            description=(
                f"Predictive cleaning work order generated by GL-014 EXCHANGER-PRO.\n\n"
                f"Prediction ID: {prediction_id}\n"
                f"Predicted Critical Fouling Date: {predicted_fouling_date.isoformat()}\n"
                f"Current Fouling Factor: {current_fouling_factor:.6f}\n"
                f"Current UA: {current_ua_percent:.1f}%\n"
                f"Confidence Score: {confidence_score:.1%}\n\n"
                f"Recommended cleaning method: {cleaning_method.value}"
            ),
            priority=priority,
            equipment_id=equipment_id,
            cleaning_method=cleaning_method,
            cleaning_reason="predictive_fouling",
            current_fouling_factor=current_fouling_factor,
            current_ua_percent=current_ua_percent,
            scheduled_start=scheduled_start,
            due_date=predicted_fouling_date - timedelta(days=1),
            prediction_id=prediction_id,
            predicted_fouling_date=predicted_fouling_date,
            confidence_score=confidence_score,
            metadata={
                "generated_by": "GL-014_EXCHANGER-PRO",
                "prediction_id": prediction_id,
                "algorithm": "fouling_prediction",
            },
        )

        return await self.create_cleaning_work_order(request)

    # =========================================================================
    # Notification Operations
    # =========================================================================

    async def create_fouling_alert(
        self,
        equipment_id: str,
        current_fouling: float,
        threshold_fouling: float,
        current_ua_percent: float,
        severity: WorkOrderPriority = WorkOrderPriority.HIGH,
    ) -> Notification:
        """Create a fouling alert notification."""
        if not self._cmms_config.enable_notifications:
            raise ConfigurationError("Notifications disabled")

        endpoint = (
            f"{self._adapter.get_base_url()}/api/"
            f"{self._cmms_config.api_version}/notifications"
        )

        data = {
            "notification_type": "fouling_alert",
            "title": f"Fouling Alert - {equipment_id}",
            "message": (
                f"Heat exchanger {equipment_id} has exceeded fouling threshold.\n"
                f"Current fouling factor: {current_fouling:.6f}\n"
                f"Threshold: {threshold_fouling:.6f}\n"
                f"Current UA: {current_ua_percent:.1f}%\n"
                f"Recommend scheduling cleaning."
            ),
            "priority": severity.value,
            "status": "new",
            "equipment_id": equipment_id,
            "metadata": {
                "current_fouling": current_fouling,
                "threshold_fouling": threshold_fouling,
                "current_ua_percent": current_ua_percent,
            },
        }

        response = await self._make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            operation_name="create_fouling_alert",
        )

        return Notification(**response)


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
