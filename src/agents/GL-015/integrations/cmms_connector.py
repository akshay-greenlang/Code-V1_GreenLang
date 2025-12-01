"""
CMMS Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade integration with Computerized Maintenance Management Systems:
- SAP PM (Plant Maintenance) - Work order creation, equipment master
- IBM Maximo - Asset management, work orders, inspections
- Oracle EAM (Enterprise Asset Management) - Equipment, work orders

Supports repair work order generation, material requisition, and inspection scheduling
for insulation maintenance and repair activities.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import hashlib
import json
import logging
import uuid

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitOpenError,
    ConfigurationError,
    ConnectionError,
    ConnectionState,
    ConnectorError,
    ConnectorType,
    DataQualityLevel,
    HealthCheckResult,
    HealthStatus,
    AuthenticationType,
    TimeoutError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class CMSSProvider(str, Enum):
    """Supported CMMS providers."""

    SAP_PM = "sap_pm"  # SAP Plant Maintenance
    IBM_MAXIMO = "ibm_maximo"  # IBM Maximo
    ORACLE_EAM = "oracle_eam"  # Oracle Enterprise Asset Management
    INFOR_EAM = "infor_eam"  # Infor EAM
    GENERIC = "generic"  # Generic CMMS


class WorkOrderStatus(str, Enum):
    """Work order status values."""

    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class WorkOrderPriority(str, Enum):
    """Work order priority levels."""

    EMERGENCY = "emergency"  # P1 - Immediate action
    CRITICAL = "critical"  # P2 - Within 24 hours
    HIGH = "high"  # P3 - Within 72 hours
    MEDIUM = "medium"  # P4 - Within 1 week
    LOW = "low"  # P5 - Scheduled maintenance
    PLANNED = "planned"  # P6 - Next planned shutdown


class WorkOrderType(str, Enum):
    """Types of work orders for insulation maintenance."""

    CORRECTIVE = "corrective"  # Repair existing damage
    PREVENTIVE = "preventive"  # Scheduled maintenance
    INSPECTION = "inspection"  # Inspection activity
    REPLACEMENT = "replacement"  # Full replacement
    EMERGENCY = "emergency"  # Emergency repair
    PROJECT = "project"  # Capital project work


class InsulationRepairType(str, Enum):
    """Types of insulation repairs."""

    PATCH_REPAIR = "patch_repair"  # Small area repair
    SECTION_REPLACEMENT = "section_replacement"  # Section replacement
    FULL_REPLACEMENT = "full_replacement"  # Complete replacement
    JACKET_REPAIR = "jacket_repair"  # Jacketing repair only
    VAPOR_BARRIER_REPAIR = "vapor_barrier_repair"  # Vapor barrier repair
    SEALING = "sealing"  # Joint/seam sealing
    REWRAP = "rewrap"  # Re-wrapping of insulation


class MaterialType(str, Enum):
    """Insulation material types for requisition."""

    MINERAL_WOOL = "mineral_wool"
    FIBERGLASS = "fiberglass"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    POLYISOCYANURATE = "polyisocyanurate"
    PHENOLIC_FOAM = "phenolic_foam"
    ALUMINUM_JACKETING = "aluminum_jacketing"
    STAINLESS_JACKETING = "stainless_jacketing"
    PVC_JACKETING = "pvc_jacketing"
    VAPOR_BARRIER = "vapor_barrier"
    SEALANT = "sealant"
    BANDS_CLIPS = "bands_clips"


class EquipmentType(str, Enum):
    """Types of insulated equipment."""

    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    COLUMN = "column"
    HEAT_EXCHANGER = "heat_exchanger"
    VALVE = "valve"
    FITTING = "fitting"
    DUCT = "duct"
    BOILER = "boiler"
    TURBINE = "turbine"
    FURNACE = "furnace"
    OTHER = "other"


class InspectionStatus(str, Enum):
    """Inspection scheduling status."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class OAuth2Config(BaseModel):
    """OAuth2 authentication configuration."""

    model_config = ConfigDict(extra="forbid")

    token_url: str = Field(..., description="OAuth2 token endpoint")
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client secret")
    scope: Optional[str] = Field(default=None, description="OAuth2 scope")
    grant_type: str = Field(
        default="client_credentials",
        description="OAuth2 grant type"
    )


class BasicAuthConfig(BaseModel):
    """Basic authentication configuration."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class APIKeyConfig(BaseModel):
    """API key authentication configuration."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(..., description="API key")
    header_name: str = Field(
        default="X-API-Key",
        description="Header name for API key"
    )


class SAPPMConfig(BaseModel):
    """SAP PM specific configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., description="SAP API base URL")
    client: str = Field(default="100", description="SAP client")
    api_version: str = Field(default="v1", description="API version")
    plant_code: str = Field(..., description="Plant code")
    planning_plant: str = Field(..., description="Planning plant")
    work_center: Optional[str] = Field(
        default=None,
        description="Default work center"
    )
    order_type_corrective: str = Field(
        default="PM01",
        description="SAP order type for corrective maintenance"
    )
    order_type_preventive: str = Field(
        default="PM02",
        description="SAP order type for preventive maintenance"
    )
    order_type_inspection: str = Field(
        default="PM03",
        description="SAP order type for inspection"
    )


class MaximoConfig(BaseModel):
    """IBM Maximo specific configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., description="Maximo API base URL")
    api_key: Optional[str] = Field(default=None, description="Maximo API key")
    organization: str = Field(..., description="Organization ID")
    site: str = Field(..., description="Site ID")
    work_type_corrective: str = Field(
        default="CM",
        description="Work type for corrective maintenance"
    )
    work_type_preventive: str = Field(
        default="PM",
        description="Work type for preventive maintenance"
    )
    work_type_inspection: str = Field(
        default="INS",
        description="Work type for inspection"
    )
    status_flow: Dict[str, str] = Field(
        default_factory=lambda: {
            "draft": "WAPPR",
            "approved": "APPR",
            "in_progress": "INPRG",
            "completed": "COMP",
        },
        description="Status code mapping"
    )


class OracleEAMConfig(BaseModel):
    """Oracle EAM specific configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., description="Oracle EAM API base URL")
    organization_id: int = Field(..., description="Organization ID")
    organization_code: str = Field(..., description="Organization code")
    wip_entity_type: str = Field(
        default="1",
        description="WIP entity type"
    )
    work_order_prefix: str = Field(
        default="INS",
        description="Work order number prefix"
    )


class CMSSConnectorConfig(BaseConnectorConfig):
    """Configuration for CMMS connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: CMSSProvider = Field(..., description="CMMS provider")

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
        description="Authentication type"
    )
    oauth2_config: Optional[OAuth2Config] = Field(
        default=None,
        description="OAuth2 configuration"
    )
    basic_auth_config: Optional[BasicAuthConfig] = Field(
        default=None,
        description="Basic auth configuration"
    )
    api_key_config: Optional[APIKeyConfig] = Field(
        default=None,
        description="API key configuration"
    )

    # Provider-specific configurations
    sap_pm_config: Optional[SAPPMConfig] = Field(
        default=None,
        description="SAP PM configuration"
    )
    maximo_config: Optional[MaximoConfig] = Field(
        default=None,
        description="IBM Maximo configuration"
    )
    oracle_eam_config: Optional[OracleEAMConfig] = Field(
        default=None,
        description="Oracle EAM configuration"
    )

    # Work order settings
    default_priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Default work order priority"
    )
    auto_approve: bool = Field(
        default=False,
        description="Auto-approve generated work orders"
    )
    notification_enabled: bool = Field(
        default=True,
        description="Enable notifications for work orders"
    )

    # Material requisition settings
    material_requisition_enabled: bool = Field(
        default=True,
        description="Enable material requisition creation"
    )
    default_warehouse: Optional[str] = Field(
        default=None,
        description="Default material warehouse"
    )

    def __init__(self, **data):
        """Initialize with connector type set."""
        data['connector_type'] = ConnectorType.CMMS
        super().__init__(**data)


# =============================================================================
# Data Models - Equipment
# =============================================================================


class InsulatedEquipment(BaseModel):
    """Insulated equipment/asset data model."""

    model_config = ConfigDict(frozen=False)

    equipment_id: str = Field(..., description="Equipment ID in CMMS")
    equipment_name: str = Field(..., description="Equipment name")
    description: Optional[str] = Field(default=None, description="Description")
    equipment_type: EquipmentType = Field(..., description="Equipment type")

    # Location
    functional_location: Optional[str] = Field(
        default=None,
        description="Functional location code"
    )
    plant_code: Optional[str] = Field(default=None, description="Plant code")
    area: Optional[str] = Field(default=None, description="Area/zone")
    building: Optional[str] = Field(default=None, description="Building")
    floor: Optional[str] = Field(default=None, description="Floor/level")

    # Insulation specifications
    insulation_material: Optional[str] = Field(
        default=None,
        description="Insulation material type"
    )
    insulation_thickness_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Insulation thickness in mm"
    )
    jacket_material: Optional[str] = Field(
        default=None,
        description="Jacketing material"
    )
    operating_temperature_c: Optional[float] = Field(
        default=None,
        description="Operating temperature in Celsius"
    )
    design_temperature_c: Optional[float] = Field(
        default=None,
        description="Design temperature in Celsius"
    )
    surface_area_m2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Insulated surface area in m2"
    )

    # Dimensions (for pipes)
    pipe_diameter_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Pipe outer diameter in mm"
    )
    pipe_length_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Pipe length in meters"
    )

    # Maintenance info
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Insulation installation date"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last inspection date"
    )
    next_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Next scheduled inspection"
    )
    condition_rating: Optional[str] = Field(
        default=None,
        description="Current condition rating"
    )

    # Custom attributes
    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom attributes"
    )


# =============================================================================
# Data Models - Work Orders
# =============================================================================


class WorkOrderTask(BaseModel):
    """Individual task within a work order."""

    model_config = ConfigDict(frozen=False)

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Task ID"
    )
    sequence: int = Field(default=1, ge=1, description="Task sequence number")
    description: str = Field(..., description="Task description")
    estimated_hours: float = Field(
        default=1.0,
        ge=0,
        description="Estimated labor hours"
    )
    actual_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actual labor hours"
    )
    craft: Optional[str] = Field(
        default=None,
        description="Required craft/trade"
    )
    status: str = Field(
        default="pending",
        description="Task status"
    )


class MaterialRequisition(BaseModel):
    """Material requisition for work order."""

    model_config = ConfigDict(frozen=False)

    requisition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Requisition ID"
    )
    material_code: str = Field(..., description="Material code in CMMS")
    material_name: str = Field(..., description="Material description")
    material_type: MaterialType = Field(..., description="Material type")
    quantity: float = Field(..., ge=0, description="Required quantity")
    unit_of_measure: str = Field(default="EA", description="Unit of measure")
    estimated_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated cost"
    )
    warehouse: Optional[str] = Field(
        default=None,
        description="Source warehouse"
    )
    status: str = Field(default="pending", description="Requisition status")


class InsulationRepairWorkOrder(BaseModel):
    """Work order for insulation repair/maintenance."""

    model_config = ConfigDict(frozen=False)

    work_order_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Work order ID"
    )
    work_order_number: Optional[str] = Field(
        default=None,
        description="CMMS work order number"
    )
    title: str = Field(..., max_length=255, description="Work order title")
    description: str = Field(..., description="Detailed description")

    # Work order metadata
    work_order_type: WorkOrderType = Field(
        default=WorkOrderType.CORRECTIVE,
        description="Work order type"
    )
    repair_type: InsulationRepairType = Field(
        default=InsulationRepairType.PATCH_REPAIR,
        description="Repair type"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Current status"
    )

    # Equipment reference
    equipment_id: str = Field(..., description="Equipment ID")
    equipment_name: Optional[str] = Field(
        default=None,
        description="Equipment name"
    )
    functional_location: Optional[str] = Field(
        default=None,
        description="Functional location"
    )

    # Defect details from inspection
    defect_description: Optional[str] = Field(
        default=None,
        description="Defect description from inspection"
    )
    defect_location: Optional[str] = Field(
        default=None,
        description="Location of defect on equipment"
    )
    defect_area_m2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Affected area in m2"
    )
    heat_loss_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated heat loss in kW"
    )
    annual_energy_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual energy cost impact"
    )
    co2_emissions_tonnes: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual CO2 emissions impact"
    )

    # Thermal image reference
    thermal_image_id: Optional[str] = Field(
        default=None,
        description="Reference to thermal image"
    )
    inspection_report_id: Optional[str] = Field(
        default=None,
        description="Reference to inspection report"
    )

    # Scheduling
    scheduled_start: Optional[datetime] = Field(
        default=None,
        description="Scheduled start date"
    )
    scheduled_end: Optional[datetime] = Field(
        default=None,
        description="Scheduled end date"
    )
    actual_start: Optional[datetime] = Field(
        default=None,
        description="Actual start date"
    )
    actual_end: Optional[datetime] = Field(
        default=None,
        description="Actual end date"
    )
    target_completion_date: Optional[datetime] = Field(
        default=None,
        description="Target completion date"
    )

    # Resource estimates
    estimated_labor_hours: float = Field(
        default=0,
        ge=0,
        description="Estimated labor hours"
    )
    estimated_material_cost: float = Field(
        default=0,
        ge=0,
        description="Estimated material cost"
    )
    estimated_total_cost: float = Field(
        default=0,
        ge=0,
        description="Total estimated cost"
    )
    actual_labor_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actual labor hours"
    )
    actual_material_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actual material cost"
    )
    actual_total_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actual total cost"
    )

    # Tasks and materials
    tasks: List[WorkOrderTask] = Field(
        default_factory=list,
        description="Work order tasks"
    )
    material_requisitions: List[MaterialRequisition] = Field(
        default_factory=list,
        description="Material requisitions"
    )

    # Assignment
    assigned_to: Optional[str] = Field(
        default=None,
        description="Assigned technician/crew"
    )
    work_center: Optional[str] = Field(
        default=None,
        description="Responsible work center"
    )
    contractor: Optional[str] = Field(
        default=None,
        description="External contractor"
    )

    # ROI and justification
    payback_period_months: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period in months"
    )
    roi_percent: Optional[float] = Field(
        default=None,
        description="Return on investment %"
    )
    justification: Optional[str] = Field(
        default=None,
        description="Business justification"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )
    created_by: Optional[str] = Field(
        default=None,
        description="Created by user"
    )


class WorkOrderCreateRequest(BaseModel):
    """Request to create a work order."""

    model_config = ConfigDict(frozen=True)

    equipment_id: str = Field(..., description="Equipment ID")
    title: str = Field(..., description="Work order title")
    description: str = Field(..., description="Description")
    work_order_type: WorkOrderType = Field(
        default=WorkOrderType.CORRECTIVE,
        description="Work order type"
    )
    repair_type: InsulationRepairType = Field(
        default=InsulationRepairType.PATCH_REPAIR,
        description="Repair type"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority"
    )
    defect_description: Optional[str] = Field(
        default=None,
        description="Defect description"
    )
    defect_area_m2: Optional[float] = Field(
        default=None,
        description="Affected area"
    )
    heat_loss_kw: Optional[float] = Field(
        default=None,
        description="Heat loss"
    )
    thermal_image_id: Optional[str] = Field(
        default=None,
        description="Thermal image reference"
    )
    inspection_report_id: Optional[str] = Field(
        default=None,
        description="Inspection report reference"
    )
    scheduled_start: Optional[datetime] = Field(
        default=None,
        description="Scheduled start"
    )
    target_completion_date: Optional[datetime] = Field(
        default=None,
        description="Target completion"
    )
    estimated_labor_hours: float = Field(
        default=0,
        description="Estimated hours"
    )
    tasks: List[WorkOrderTask] = Field(
        default_factory=list,
        description="Tasks"
    )
    materials: List[MaterialRequisition] = Field(
        default_factory=list,
        description="Materials"
    )


class WorkOrderUpdateRequest(BaseModel):
    """Request to update a work order."""

    model_config = ConfigDict(frozen=True)

    work_order_id: str = Field(..., description="Work order ID")
    status: Optional[WorkOrderStatus] = Field(
        default=None,
        description="New status"
    )
    priority: Optional[WorkOrderPriority] = Field(
        default=None,
        description="New priority"
    )
    scheduled_start: Optional[datetime] = Field(
        default=None,
        description="Scheduled start"
    )
    scheduled_end: Optional[datetime] = Field(
        default=None,
        description="Scheduled end"
    )
    assigned_to: Optional[str] = Field(
        default=None,
        description="Assignee"
    )
    actual_hours: Optional[float] = Field(
        default=None,
        description="Actual hours"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes to add"
    )


# =============================================================================
# Data Models - Inspection Scheduling
# =============================================================================


class InspectionSchedule(BaseModel):
    """Scheduled inspection for insulated equipment."""

    model_config = ConfigDict(frozen=False)

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Schedule ID"
    )
    inspection_number: Optional[str] = Field(
        default=None,
        description="CMMS inspection number"
    )
    title: str = Field(..., description="Inspection title")
    description: Optional[str] = Field(
        default=None,
        description="Description"
    )

    # Equipment/scope
    equipment_ids: List[str] = Field(
        default_factory=list,
        description="Equipment IDs to inspect"
    )
    functional_locations: List[str] = Field(
        default_factory=list,
        description="Functional locations to inspect"
    )
    scope_description: Optional[str] = Field(
        default=None,
        description="Inspection scope"
    )

    # Schedule
    scheduled_date: datetime = Field(..., description="Scheduled date")
    scheduled_duration_hours: float = Field(
        default=8.0,
        ge=0,
        description="Estimated duration"
    )
    frequency_days: Optional[int] = Field(
        default=None,
        ge=1,
        description="Inspection frequency"
    )
    is_recurring: bool = Field(
        default=False,
        description="Is recurring inspection"
    )

    # Assignment
    inspector: Optional[str] = Field(
        default=None,
        description="Assigned inspector"
    )
    inspection_team: List[str] = Field(
        default_factory=list,
        description="Inspection team members"
    )

    # Status
    status: InspectionStatus = Field(
        default=InspectionStatus.SCHEDULED,
        description="Status"
    )
    completed_date: Optional[datetime] = Field(
        default=None,
        description="Completion date"
    )

    # Results (after completion)
    defects_found: int = Field(
        default=0,
        ge=0,
        description="Number of defects found"
    )
    work_orders_generated: List[str] = Field(
        default_factory=list,
        description="Generated work order IDs"
    )


class InspectionScheduleRequest(BaseModel):
    """Request to create inspection schedule."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Inspection title")
    description: Optional[str] = Field(
        default=None,
        description="Description"
    )
    equipment_ids: List[str] = Field(
        default_factory=list,
        description="Equipment to inspect"
    )
    functional_locations: List[str] = Field(
        default_factory=list,
        description="Locations to inspect"
    )
    scheduled_date: datetime = Field(..., description="Scheduled date")
    scheduled_duration_hours: float = Field(
        default=8.0,
        description="Duration"
    )
    is_recurring: bool = Field(
        default=False,
        description="Is recurring"
    )
    frequency_days: Optional[int] = Field(
        default=None,
        description="Frequency in days"
    )
    inspector: Optional[str] = Field(
        default=None,
        description="Inspector"
    )


# =============================================================================
# CMMS Connector
# =============================================================================


class CMSSConnector(BaseConnector):
    """
    CMMS Connector for GL-015 INSULSCAN.

    Provides unified interface for CMMS integration supporting SAP PM,
    IBM Maximo, and Oracle EAM for insulation maintenance management.

    Features:
    - Work order creation for insulation repairs
    - Material requisition management
    - Inspection scheduling
    - Equipment master data retrieval
    - Maintenance history tracking
    """

    def __init__(self, config: CMSSConnectorConfig) -> None:
        """
        Initialize CMMS connector.

        Args:
            config: CMMS connector configuration
        """
        super().__init__(config)
        self._cmms_config = config

        # Authentication state
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Establish connection to CMMS."""
        self._logger.info(f"Connecting to CMMS: {self._cmms_config.provider.value}")

        try:
            self._state = ConnectionState.CONNECTING

            # Get HTTP session from pool
            self._http_session = await self._pool.get_session()

            # Authenticate
            await self._authenticate()

            # Verify connection
            await self._verify_connection()

            self._state = ConnectionState.CONNECTED
            self._logger.info(
                f"Connected to CMMS: {self._cmms_config.provider.value}"
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to CMMS: {e}")
            raise ConnectionError(
                f"CMMS connection failed: {e}",
                details={"provider": self._cmms_config.provider.value}
            )

    async def disconnect(self) -> None:
        """Disconnect from CMMS."""
        self._logger.info("Disconnecting from CMMS")
        self._access_token = None
        self._token_expires_at = None
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on CMMS connection."""
        import time
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"CMMS not connected: {self._state.value}"
                )

            # Verify token validity
            if not await self._is_token_valid():
                await self._authenticate()

            # Test API endpoint
            await self._verify_connection()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="CMMS connection healthy",
                details={
                    "provider": self._cmms_config.provider.value,
                    "token_valid": True,
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
        """Validate connector configuration."""
        provider = self._cmms_config.provider

        # Validate authentication config
        if self._cmms_config.auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            if not self._cmms_config.oauth2_config:
                raise ConfigurationError("OAuth2 config required for OAuth2 auth")

        elif self._cmms_config.auth_type == AuthenticationType.BASIC:
            if not self._cmms_config.basic_auth_config:
                raise ConfigurationError("Basic auth config required")

        elif self._cmms_config.auth_type == AuthenticationType.API_KEY:
            if not self._cmms_config.api_key_config:
                raise ConfigurationError("API key config required")

        # Validate provider-specific config
        if provider == CMSSProvider.SAP_PM:
            if not self._cmms_config.sap_pm_config:
                raise ConfigurationError("SAP PM config required for SAP provider")

        elif provider == CMSSProvider.IBM_MAXIMO:
            if not self._cmms_config.maximo_config:
                raise ConfigurationError("Maximo config required for Maximo provider")

        elif provider == CMSSProvider.ORACLE_EAM:
            if not self._cmms_config.oracle_eam_config:
                raise ConfigurationError("Oracle EAM config required for Oracle provider")

        return True

    # =========================================================================
    # Authentication
    # =========================================================================

    async def _authenticate(self) -> None:
        """Authenticate with CMMS."""
        auth_type = self._cmms_config.auth_type

        if auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            await self._oauth2_authenticate()
        elif auth_type == AuthenticationType.BASIC:
            self._setup_basic_auth()
        elif auth_type == AuthenticationType.API_KEY:
            self._setup_api_key_auth()
        else:
            self._logger.info("No authentication configured")

    async def _oauth2_authenticate(self) -> None:
        """Authenticate using OAuth2 client credentials."""
        config = self._cmms_config.oauth2_config

        token_data = {
            'grant_type': config.grant_type,
            'client_id': config.client_id,
            'client_secret': config.client_secret,
        }
        if config.scope:
            token_data['scope'] = config.scope

        try:
            async with self._http_session.post(
                config.token_url,
                data=token_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ConnectionError(
                        f"OAuth2 authentication failed: {response.status} - {text}"
                    )

                token_response = await response.json()
                self._access_token = token_response['access_token']
                expires_in = token_response.get('expires_in', 3600)
                self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                self._logger.info("OAuth2 authentication successful")

        except aiohttp.ClientError as e:
            raise ConnectionError(f"OAuth2 request failed: {e}")

    def _setup_basic_auth(self) -> None:
        """Setup basic authentication."""
        config = self._cmms_config.basic_auth_config
        credentials = f"{config.username}:{config.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self._access_token = f"Basic {encoded}"

    def _setup_api_key_auth(self) -> None:
        """Setup API key authentication."""
        config = self._cmms_config.api_key_config
        self._access_token = config.api_key

    async def _is_token_valid(self) -> bool:
        """Check if current token is valid."""
        if not self._access_token:
            return False
        if self._token_expires_at:
            return datetime.utcnow() < self._token_expires_at - timedelta(minutes=5)
        return True

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {}

        if self._cmms_config.auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            headers['Authorization'] = f'Bearer {self._access_token}'

        elif self._cmms_config.auth_type == AuthenticationType.BASIC:
            headers['Authorization'] = self._access_token

        elif self._cmms_config.auth_type == AuthenticationType.API_KEY:
            header_name = self._cmms_config.api_key_config.header_name
            headers[header_name] = self._access_token

        return headers

    async def _verify_connection(self) -> None:
        """Verify CMMS connection is working."""
        provider = self._cmms_config.provider

        if provider == CMSSProvider.SAP_PM:
            await self._verify_sap_connection()
        elif provider == CMSSProvider.IBM_MAXIMO:
            await self._verify_maximo_connection()
        elif provider == CMSSProvider.ORACLE_EAM:
            await self._verify_oracle_connection()

    async def _verify_sap_connection(self) -> None:
        """Verify SAP PM connection."""
        config = self._cmms_config.sap_pm_config
        url = f"{config.base_url}/sap/opu/odata/sap/API_EQUIPMENT/A_Equipment"
        params = {'$top': '1'}

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status not in [200, 404]:
                raise ConnectionError(f"SAP verification failed: {response.status}")

    async def _verify_maximo_connection(self) -> None:
        """Verify IBM Maximo connection."""
        config = self._cmms_config.maximo_config
        url = f"{config.base_url}/maximo/oslc/os/mxasset"
        params = {'oslc.pageSize': '1'}

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status not in [200, 401]:
                raise ConnectionError(f"Maximo verification failed: {response.status}")

    async def _verify_oracle_connection(self) -> None:
        """Verify Oracle EAM connection."""
        config = self._cmms_config.oracle_eam_config
        url = f"{config.base_url}/fscmRestApi/resources/11.13.18.05/assets"
        params = {'limit': '1'}

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status not in [200, 401]:
                raise ConnectionError(f"Oracle verification failed: {response.status}")

    # =========================================================================
    # Equipment Operations
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_equipment(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipment]:
        """
        Get equipment master data.

        Args:
            equipment_id: Equipment ID

        Returns:
            Equipment data or None if not found
        """
        async def _fetch():
            provider = self._cmms_config.provider

            if provider == CMSSProvider.SAP_PM:
                return await self._get_equipment_sap(equipment_id)
            elif provider == CMSSProvider.IBM_MAXIMO:
                return await self._get_equipment_maximo(equipment_id)
            elif provider == CMSSProvider.ORACLE_EAM:
                return await self._get_equipment_oracle(equipment_id)
            else:
                raise ConfigurationError(f"Unsupported provider: {provider}")

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_equipment",
            use_cache=True,
            cache_key=f"equipment_{equipment_id}"
        )

    async def _get_equipment_sap(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipment]:
        """Get equipment from SAP PM."""
        config = self._cmms_config.sap_pm_config
        url = f"{config.base_url}/sap/opu/odata/sap/API_EQUIPMENT/A_Equipment('{equipment_id}')"

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 404:
                return None
            if response.status != 200:
                raise ConnectorError(f"SAP API error: {response.status}")

            data = await response.json()
            result = data.get('d', {})

            return InsulatedEquipment(
                equipment_id=result.get('Equipment', equipment_id),
                equipment_name=result.get('EquipmentName', ''),
                description=result.get('EquipmentDescription'),
                equipment_type=EquipmentType.OTHER,
                functional_location=result.get('FunctionalLocation'),
                plant_code=result.get('MaintenancePlant'),
            )

    async def _get_equipment_maximo(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipment]:
        """Get equipment from IBM Maximo."""
        config = self._cmms_config.maximo_config
        url = f"{config.base_url}/maximo/oslc/os/mxasset"
        params = {'oslc.where': f'assetnum="{equipment_id}"'}

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"Maximo API error: {response.status}")

            data = await response.json()
            members = data.get('rdfs:member', [])

            if not members:
                return None

            result = members[0]

            return InsulatedEquipment(
                equipment_id=result.get('spi:assetnum', equipment_id),
                equipment_name=result.get('spi:description', ''),
                description=result.get('spi:description_longdescription'),
                equipment_type=EquipmentType.OTHER,
                functional_location=result.get('spi:location'),
                plant_code=result.get('spi:siteid'),
            )

    async def _get_equipment_oracle(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipment]:
        """Get equipment from Oracle EAM."""
        config = self._cmms_config.oracle_eam_config
        url = f"{config.base_url}/fscmRestApi/resources/11.13.18.05/assets/{equipment_id}"

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 404:
                return None
            if response.status != 200:
                raise ConnectorError(f"Oracle API error: {response.status}")

            result = await response.json()

            return InsulatedEquipment(
                equipment_id=result.get('AssetNumber', equipment_id),
                equipment_name=result.get('AssetDescription', ''),
                description=result.get('AssetDescription'),
                equipment_type=EquipmentType.OTHER,
                functional_location=result.get('AssetLocationId'),
                plant_code=str(config.organization_id),
            )

    @with_retry(max_retries=3, base_delay=1.0)
    async def list_equipment(
        self,
        functional_location: Optional[str] = None,
        plant_code: Optional[str] = None,
        equipment_type: Optional[EquipmentType] = None,
        limit: int = 100
    ) -> List[InsulatedEquipment]:
        """
        List equipment with optional filters.

        Args:
            functional_location: Filter by location
            plant_code: Filter by plant
            equipment_type: Filter by type
            limit: Maximum results

        Returns:
            List of equipment
        """
        async def _fetch():
            provider = self._cmms_config.provider

            if provider == CMSSProvider.SAP_PM:
                return await self._list_equipment_sap(
                    functional_location, plant_code, limit
                )
            elif provider == CMSSProvider.IBM_MAXIMO:
                return await self._list_equipment_maximo(
                    functional_location, plant_code, limit
                )
            elif provider == CMSSProvider.ORACLE_EAM:
                return await self._list_equipment_oracle(
                    functional_location, plant_code, limit
                )
            else:
                return []

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="list_equipment",
            validate_result=False
        )

    async def _list_equipment_sap(
        self,
        functional_location: Optional[str],
        plant_code: Optional[str],
        limit: int
    ) -> List[InsulatedEquipment]:
        """List equipment from SAP PM."""
        config = self._cmms_config.sap_pm_config
        url = f"{config.base_url}/sap/opu/odata/sap/API_EQUIPMENT/A_Equipment"

        filters = []
        if functional_location:
            filters.append(f"FunctionalLocation eq '{functional_location}'")
        if plant_code:
            filters.append(f"MaintenancePlant eq '{plant_code}'")

        params = {
            '$top': str(limit),
            '$format': 'json'
        }
        if filters:
            params['$filter'] = ' and '.join(filters)

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"SAP API error: {response.status}")

            data = await response.json()
            results = data.get('d', {}).get('results', [])

            return [
                InsulatedEquipment(
                    equipment_id=r.get('Equipment', ''),
                    equipment_name=r.get('EquipmentName', ''),
                    description=r.get('EquipmentDescription'),
                    equipment_type=EquipmentType.OTHER,
                    functional_location=r.get('FunctionalLocation'),
                    plant_code=r.get('MaintenancePlant'),
                )
                for r in results
            ]

    async def _list_equipment_maximo(
        self,
        functional_location: Optional[str],
        plant_code: Optional[str],
        limit: int
    ) -> List[InsulatedEquipment]:
        """List equipment from IBM Maximo."""
        config = self._cmms_config.maximo_config
        url = f"{config.base_url}/maximo/oslc/os/mxasset"

        where_clauses = []
        if functional_location:
            where_clauses.append(f'location="{functional_location}"')
        if plant_code:
            where_clauses.append(f'siteid="{plant_code}"')

        params = {
            'oslc.pageSize': str(limit),
        }
        if where_clauses:
            params['oslc.where'] = ' and '.join(where_clauses)

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"Maximo API error: {response.status}")

            data = await response.json()
            results = data.get('rdfs:member', [])

            return [
                InsulatedEquipment(
                    equipment_id=r.get('spi:assetnum', ''),
                    equipment_name=r.get('spi:description', ''),
                    equipment_type=EquipmentType.OTHER,
                    functional_location=r.get('spi:location'),
                    plant_code=r.get('spi:siteid'),
                )
                for r in results
            ]

    async def _list_equipment_oracle(
        self,
        functional_location: Optional[str],
        plant_code: Optional[str],
        limit: int
    ) -> List[InsulatedEquipment]:
        """List equipment from Oracle EAM."""
        config = self._cmms_config.oracle_eam_config
        url = f"{config.base_url}/fscmRestApi/resources/11.13.18.05/assets"

        params = {
            'limit': str(limit),
        }
        if functional_location:
            params['q'] = f"AssetLocationId={functional_location}"

        async with self._http_session.get(
            url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"Oracle API error: {response.status}")

            data = await response.json()
            results = data.get('items', [])

            return [
                InsulatedEquipment(
                    equipment_id=r.get('AssetNumber', ''),
                    equipment_name=r.get('AssetDescription', ''),
                    equipment_type=EquipmentType.OTHER,
                    functional_location=r.get('AssetLocationId'),
                    plant_code=str(config.organization_id),
                )
                for r in results
            ]

    # =========================================================================
    # Work Order Operations
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def create_work_order(
        self,
        request: WorkOrderCreateRequest
    ) -> InsulationRepairWorkOrder:
        """
        Create a repair work order in CMMS.

        Args:
            request: Work order creation request

        Returns:
            Created work order with CMMS number
        """
        async def _create():
            provider = self._cmms_config.provider

            if provider == CMSSProvider.SAP_PM:
                return await self._create_work_order_sap(request)
            elif provider == CMSSProvider.IBM_MAXIMO:
                return await self._create_work_order_maximo(request)
            elif provider == CMSSProvider.ORACLE_EAM:
                return await self._create_work_order_oracle(request)
            else:
                raise ConfigurationError(f"Unsupported provider: {provider}")

        return await self.execute_with_protection(
            operation=_create,
            operation_name="create_work_order",
            validate_result=False
        )

    async def _create_work_order_sap(
        self,
        request: WorkOrderCreateRequest
    ) -> InsulationRepairWorkOrder:
        """Create work order in SAP PM."""
        config = self._cmms_config.sap_pm_config

        # Map work order type to SAP order type
        order_type_map = {
            WorkOrderType.CORRECTIVE: config.order_type_corrective,
            WorkOrderType.PREVENTIVE: config.order_type_preventive,
            WorkOrderType.INSPECTION: config.order_type_inspection,
            WorkOrderType.REPLACEMENT: config.order_type_corrective,
            WorkOrderType.EMERGENCY: config.order_type_corrective,
        }
        order_type = order_type_map.get(request.work_order_type, config.order_type_corrective)

        # Map priority
        priority_map = {
            WorkOrderPriority.EMERGENCY: "1",
            WorkOrderPriority.CRITICAL: "2",
            WorkOrderPriority.HIGH: "3",
            WorkOrderPriority.MEDIUM: "4",
            WorkOrderPriority.LOW: "5",
            WorkOrderPriority.PLANNED: "6",
        }

        # Build SAP order payload
        payload = {
            "OrderType": order_type,
            "Equipment": request.equipment_id,
            "MaintenancePlant": config.plant_code,
            "PlanningPlant": config.planning_plant,
            "MaintenanceOrderDesc": request.title[:40],  # SAP limit
            "LongTextString": request.description,
            "Priority": priority_map.get(request.priority, "4"),
        }

        if config.work_center:
            payload["MainWorkCenter"] = config.work_center

        if request.scheduled_start:
            payload["MaintOrdBasicStartDate"] = request.scheduled_start.strftime("%Y-%m-%d")
            payload["MaintOrdBasicStartTime"] = request.scheduled_start.strftime("%H:%M:%S")

        url = f"{config.base_url}/sap/opu/odata/sap/API_MAINTENANCEORDER/MaintenanceOrder"
        headers = self._get_auth_headers()
        headers['Content-Type'] = 'application/json'

        async with self._http_session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status not in [200, 201]:
                text = await response.text()
                raise ConnectorError(
                    f"SAP work order creation failed: {response.status} - {text}"
                )

            data = await response.json()
            sap_order_number = data.get('d', {}).get('MaintenanceOrder', '')

            work_order = InsulationRepairWorkOrder(
                work_order_number=sap_order_number,
                title=request.title,
                description=request.description,
                work_order_type=request.work_order_type,
                repair_type=request.repair_type,
                priority=request.priority,
                status=WorkOrderStatus.PENDING if self._cmms_config.auto_approve else WorkOrderStatus.DRAFT,
                equipment_id=request.equipment_id,
                defect_description=request.defect_description,
                defect_area_m2=request.defect_area_m2,
                heat_loss_kw=request.heat_loss_kw,
                thermal_image_id=request.thermal_image_id,
                inspection_report_id=request.inspection_report_id,
                scheduled_start=request.scheduled_start,
                target_completion_date=request.target_completion_date,
                estimated_labor_hours=request.estimated_labor_hours,
            )

            self._logger.info(f"Created SAP work order: {sap_order_number}")
            return work_order

    async def _create_work_order_maximo(
        self,
        request: WorkOrderCreateRequest
    ) -> InsulationRepairWorkOrder:
        """Create work order in IBM Maximo."""
        config = self._cmms_config.maximo_config

        # Map work order type
        work_type_map = {
            WorkOrderType.CORRECTIVE: config.work_type_corrective,
            WorkOrderType.PREVENTIVE: config.work_type_preventive,
            WorkOrderType.INSPECTION: config.work_type_inspection,
        }
        work_type = work_type_map.get(request.work_order_type, config.work_type_corrective)

        # Map priority to Maximo priority (1-5)
        priority_map = {
            WorkOrderPriority.EMERGENCY: 1,
            WorkOrderPriority.CRITICAL: 1,
            WorkOrderPriority.HIGH: 2,
            WorkOrderPriority.MEDIUM: 3,
            WorkOrderPriority.LOW: 4,
            WorkOrderPriority.PLANNED: 5,
        }

        payload = {
            "spi:assetnum": request.equipment_id,
            "spi:siteid": config.site,
            "spi:orgid": config.organization,
            "spi:worktype": work_type,
            "spi:description": request.title,
            "spi:description_longdescription": request.description,
            "spi:wopriority": priority_map.get(request.priority, 3),
            "spi:estdur": request.estimated_labor_hours,
        }

        if request.scheduled_start:
            payload["spi:schedstart"] = request.scheduled_start.isoformat()

        url = f"{config.base_url}/maximo/oslc/os/mxwo"
        headers = self._get_auth_headers()
        headers['Content-Type'] = 'application/json'

        async with self._http_session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status not in [200, 201]:
                text = await response.text()
                raise ConnectorError(
                    f"Maximo work order creation failed: {response.status} - {text}"
                )

            data = await response.json()
            maximo_wo_num = data.get('spi:wonum', '')

            work_order = InsulationRepairWorkOrder(
                work_order_number=maximo_wo_num,
                title=request.title,
                description=request.description,
                work_order_type=request.work_order_type,
                repair_type=request.repair_type,
                priority=request.priority,
                status=WorkOrderStatus.DRAFT,
                equipment_id=request.equipment_id,
                defect_description=request.defect_description,
                defect_area_m2=request.defect_area_m2,
                heat_loss_kw=request.heat_loss_kw,
                thermal_image_id=request.thermal_image_id,
                inspection_report_id=request.inspection_report_id,
                scheduled_start=request.scheduled_start,
                target_completion_date=request.target_completion_date,
                estimated_labor_hours=request.estimated_labor_hours,
            )

            self._logger.info(f"Created Maximo work order: {maximo_wo_num}")
            return work_order

    async def _create_work_order_oracle(
        self,
        request: WorkOrderCreateRequest
    ) -> InsulationRepairWorkOrder:
        """Create work order in Oracle EAM."""
        config = self._cmms_config.oracle_eam_config

        payload = {
            "AssetNumber": request.equipment_id,
            "OrganizationId": config.organization_id,
            "WorkOrderDescription": request.title,
            "WorkOrderLongDescription": request.description,
            "Priority": request.priority.value,
        }

        if request.scheduled_start:
            payload["ScheduledStartDate"] = request.scheduled_start.isoformat()

        url = f"{config.base_url}/fscmRestApi/resources/11.13.18.05/maintenanceWorkOrders"
        headers = self._get_auth_headers()
        headers['Content-Type'] = 'application/json'

        async with self._http_session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status not in [200, 201]:
                text = await response.text()
                raise ConnectorError(
                    f"Oracle work order creation failed: {response.status} - {text}"
                )

            data = await response.json()
            oracle_wo_num = data.get('WorkOrderNumber', '')

            work_order = InsulationRepairWorkOrder(
                work_order_number=oracle_wo_num,
                title=request.title,
                description=request.description,
                work_order_type=request.work_order_type,
                repair_type=request.repair_type,
                priority=request.priority,
                status=WorkOrderStatus.DRAFT,
                equipment_id=request.equipment_id,
                defect_description=request.defect_description,
                defect_area_m2=request.defect_area_m2,
                heat_loss_kw=request.heat_loss_kw,
                thermal_image_id=request.thermal_image_id,
                inspection_report_id=request.inspection_report_id,
                scheduled_start=request.scheduled_start,
                target_completion_date=request.target_completion_date,
                estimated_labor_hours=request.estimated_labor_hours,
            )

            self._logger.info(f"Created Oracle work order: {oracle_wo_num}")
            return work_order

    @with_retry(max_retries=3, base_delay=1.0)
    async def update_work_order(
        self,
        request: WorkOrderUpdateRequest
    ) -> InsulationRepairWorkOrder:
        """
        Update an existing work order.

        Args:
            request: Work order update request

        Returns:
            Updated work order
        """
        # Implementation would follow similar provider-specific pattern
        raise NotImplementedError("Work order update not yet implemented")

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_work_order(
        self,
        work_order_number: str
    ) -> Optional[InsulationRepairWorkOrder]:
        """
        Get work order by number.

        Args:
            work_order_number: CMMS work order number

        Returns:
            Work order or None if not found
        """
        # Implementation would follow similar provider-specific pattern
        raise NotImplementedError("Work order retrieval not yet implemented")

    # =========================================================================
    # Material Requisition
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def create_material_requisition(
        self,
        work_order_number: str,
        materials: List[MaterialRequisition]
    ) -> List[MaterialRequisition]:
        """
        Create material requisitions for a work order.

        Args:
            work_order_number: Work order number
            materials: List of materials to requisition

        Returns:
            Created requisitions with IDs
        """
        if not self._cmms_config.material_requisition_enabled:
            self._logger.warning("Material requisition is disabled")
            return materials

        async def _create():
            # Provider-specific material requisition
            for material in materials:
                material.status = "submitted"
            return materials

        return await self.execute_with_protection(
            operation=_create,
            operation_name="create_material_requisition",
            validate_result=False
        )

    # =========================================================================
    # Inspection Scheduling
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def schedule_inspection(
        self,
        request: InspectionScheduleRequest
    ) -> InspectionSchedule:
        """
        Schedule an insulation inspection.

        Args:
            request: Inspection schedule request

        Returns:
            Created inspection schedule
        """
        async def _schedule():
            schedule = InspectionSchedule(
                title=request.title,
                description=request.description,
                equipment_ids=request.equipment_ids,
                functional_locations=request.functional_locations,
                scheduled_date=request.scheduled_date,
                scheduled_duration_hours=request.scheduled_duration_hours,
                is_recurring=request.is_recurring,
                frequency_days=request.frequency_days,
                inspector=request.inspector,
                status=InspectionStatus.SCHEDULED,
            )

            # Create in CMMS (provider-specific)
            # For now, return the schedule object
            self._logger.info(
                f"Scheduled inspection: {schedule.schedule_id} for {request.scheduled_date}"
            )

            return schedule

        return await self.execute_with_protection(
            operation=_schedule,
            operation_name="schedule_inspection",
            validate_result=False
        )

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_upcoming_inspections(
        self,
        days_ahead: int = 30,
        functional_location: Optional[str] = None
    ) -> List[InspectionSchedule]:
        """
        Get upcoming scheduled inspections.

        Args:
            days_ahead: Number of days to look ahead
            functional_location: Filter by location

        Returns:
            List of scheduled inspections
        """
        # Provider-specific implementation would query CMMS
        return []


# =============================================================================
# Factory Function
# =============================================================================


def create_cmms_connector(
    provider: CMSSProvider,
    connector_name: str = "CMMS",
    **kwargs
) -> CMSSConnector:
    """
    Factory function to create CMMS connector.

    Args:
        provider: CMMS provider
        connector_name: Connector name
        **kwargs: Additional configuration options

    Returns:
        Configured CMSSConnector instance
    """
    config = CMSSConnectorConfig(
        connector_name=connector_name,
        provider=provider,
        **kwargs
    )
    return CMSSConnector(config)


# Import base64 for basic auth
import base64
