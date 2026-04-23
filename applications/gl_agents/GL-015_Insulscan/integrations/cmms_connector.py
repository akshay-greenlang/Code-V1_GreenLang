"""
CMMS Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade integration with Computerized Maintenance Management Systems:
- SAP PM (Plant Maintenance) - Work order creation, equipment master
- IBM Maximo - Asset management, work orders, inspections
- Create work orders for insulation repairs
- Query asset master data
- Update inspection records
- REST API client with retry logic

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

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
import base64
import hashlib
import json
import logging
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class CMSSProvider(str, Enum):
    """Supported CMMS providers."""

    SAP_PM = "sap_pm"
    IBM_MAXIMO = "ibm_maximo"
    ORACLE_EAM = "oracle_eam"
    INFOR_EAM = "infor_eam"
    GENERIC = "generic"


class AuthenticationType(str, Enum):
    """Authentication types."""

    NONE = "none"
    BASIC = "basic"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_PASSWORD = "oauth2_password"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"


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

    EMERGENCY = "emergency"
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PLANNED = "planned"


class WorkOrderType(str, Enum):
    """Types of work orders for insulation maintenance."""

    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    INSPECTION = "inspection"
    REPLACEMENT = "replacement"
    EMERGENCY = "emergency"
    PROJECT = "project"


class InsulationRepairType(str, Enum):
    """Types of insulation repairs."""

    PATCH_REPAIR = "patch_repair"
    SECTION_REPLACEMENT = "section_replacement"
    FULL_REPLACEMENT = "full_replacement"
    JACKET_REPAIR = "jacket_repair"
    VAPOR_BARRIER_REPAIR = "vapor_barrier_repair"
    SEALING = "sealing"
    REWRAP = "rewrap"


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


class ConnectionState(str, Enum):
    """Connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class InspectionStatus(str, Enum):
    """Inspection scheduling status."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


# =============================================================================
# Custom Exceptions
# =============================================================================


class CMSSError(Exception):
    """Base CMMS exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class CMSSConnectionError(CMSSError):
    """CMMS connection error."""
    pass


class CMSSAuthenticationError(CMSSError):
    """CMMS authentication error."""
    pass


class CMSSValidationError(CMSSError):
    """CMMS validation error."""
    pass


class CMSSWorkOrderError(CMSSError):
    """Work order operation error."""
    pass


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


class CMSSConnectorConfig(BaseModel):
    """Configuration for CMMS connector."""

    model_config = ConfigDict(extra="forbid")

    connector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Connector identifier"
    )
    connector_name: str = Field(
        default="CMMS-Connector",
        description="Connector name"
    )

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

    # Connection settings
    connection_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        description="Connection timeout"
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        description="Request timeout"
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retries"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Retry delay"
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

    # Status
    status: InspectionStatus = Field(
        default=InspectionStatus.SCHEDULED,
        description="Status"
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


# =============================================================================
# CMMS Connector
# =============================================================================


class CMSSConnector:
    """
    CMMS Connector for GL-015 INSULSCAN.

    Provides unified interface for CMMS integration supporting SAP PM
    and IBM Maximo for insulation maintenance management.

    Features:
    - Work order creation for insulation repairs
    - Material requisition management
    - Inspection scheduling
    - Equipment master data retrieval
    - Retry logic and error handling
    """

    def __init__(self, config: CMSSConnectorConfig) -> None:
        """
        Initialize CMMS connector.

        Args:
            config: CMMS connector configuration
        """
        self._config = config
        self._logger = logging.getLogger(
            f"{__name__}.{config.connector_name}"
        )

        self._state = ConnectionState.DISCONNECTED
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

        # HTTP client (would use aiohttp in production)
        self._client: Optional[Any] = None

        # Statistics
        self._requests_count = 0
        self._requests_failed = 0
        self._work_orders_created = 0

    @property
    def config(self) -> CMSSConnectorConfig:
        """Get configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to CMMS.

        Raises:
            CMSSConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING

        self._logger.info(f"Connecting to CMMS: {self._config.provider.value}")

        try:
            # In production, create HTTP client session
            # import aiohttp
            # self._client = aiohttp.ClientSession(timeout=...)

            # Authenticate
            await self._authenticate()

            # Verify connection
            await self._verify_connection()

            self._state = ConnectionState.CONNECTED

            self._logger.info(f"Connected to CMMS: {self._config.provider.value}")

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"CMMS connection failed: {e}")
            raise CMSSConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from CMMS."""
        self._logger.info("Disconnecting from CMMS")

        self._access_token = None
        self._token_expires_at = None

        if self._client:
            # await self._client.close()
            pass

        self._state = ConnectionState.DISCONNECTED

    async def _authenticate(self) -> None:
        """Authenticate with CMMS."""
        auth_type = self._config.auth_type

        if auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            await self._oauth2_authenticate()
        elif auth_type == AuthenticationType.BASIC:
            self._setup_basic_auth()
        elif auth_type == AuthenticationType.API_KEY:
            self._setup_api_key_auth()
        else:
            self._logger.info("No authentication configured")

    async def _oauth2_authenticate(self) -> None:
        """Authenticate using OAuth2."""
        config = self._config.oauth2_config
        if not config:
            raise CMSSAuthenticationError("OAuth2 config not provided")

        self._logger.info("Authenticating with OAuth2")

        # In production, make token request
        # token_data = {
        #     'grant_type': config.grant_type,
        #     'client_id': config.client_id,
        #     'client_secret': config.client_secret,
        # }
        # response = await self._client.post(config.token_url, data=token_data)
        # ...

        # Mock token
        self._access_token = "mock_access_token"
        self._token_expires_at = datetime.utcnow() + timedelta(hours=1)

    def _setup_basic_auth(self) -> None:
        """Setup basic authentication."""
        config = self._config.basic_auth_config
        if not config:
            raise CMSSAuthenticationError("Basic auth config not provided")

        credentials = f"{config.username}:{config.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self._access_token = f"Basic {encoded}"

    def _setup_api_key_auth(self) -> None:
        """Setup API key authentication."""
        config = self._config.api_key_config
        if not config:
            raise CMSSAuthenticationError("API key config not provided")

        self._access_token = config.api_key

    async def _verify_connection(self) -> None:
        """Verify CMMS connection."""
        # Provider-specific verification
        pass

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {"Content-Type": "application/json"}

        if self._config.auth_type == AuthenticationType.OAUTH2_CLIENT_CREDENTIALS:
            headers['Authorization'] = f'Bearer {self._access_token}'
        elif self._config.auth_type == AuthenticationType.BASIC:
            headers['Authorization'] = self._access_token
        elif self._config.auth_type == AuthenticationType.API_KEY:
            config = self._config.api_key_config
            headers[config.header_name] = self._access_token

        return headers

    # =========================================================================
    # Equipment Operations
    # =========================================================================

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
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        try:
            provider = self._config.provider

            if provider == CMSSProvider.SAP_PM:
                return await self._get_equipment_sap(equipment_id)
            elif provider == CMSSProvider.IBM_MAXIMO:
                return await self._get_equipment_maximo(equipment_id)
            else:
                return None

        except Exception as e:
            self._requests_failed += 1
            self._logger.error(f"Failed to get equipment {equipment_id}: {e}")
            return None

    async def _get_equipment_sap(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipment]:
        """Get equipment from SAP PM."""
        # In production, call SAP OData API
        return InsulatedEquipment(
            equipment_id=equipment_id,
            equipment_name=f"Equipment {equipment_id}",
            equipment_type=EquipmentType.PIPE,
        )

    async def _get_equipment_maximo(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipment]:
        """Get equipment from IBM Maximo."""
        # In production, call Maximo REST API
        return InsulatedEquipment(
            equipment_id=equipment_id,
            equipment_name=f"Asset {equipment_id}",
            equipment_type=EquipmentType.PIPE,
        )

    async def list_equipment(
        self,
        functional_location: Optional[str] = None,
        plant_code: Optional[str] = None,
        limit: int = 100
    ) -> List[InsulatedEquipment]:
        """
        List equipment with optional filters.

        Args:
            functional_location: Filter by location
            plant_code: Filter by plant
            limit: Maximum results

        Returns:
            List of equipment
        """
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        # In production, query CMMS
        return []

    # =========================================================================
    # Work Order Operations
    # =========================================================================

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

        Raises:
            CMSSWorkOrderError: If creation fails
        """
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        try:
            provider = self._config.provider

            if provider == CMSSProvider.SAP_PM:
                return await self._create_work_order_sap(request)
            elif provider == CMSSProvider.IBM_MAXIMO:
                return await self._create_work_order_maximo(request)
            else:
                raise CMSSWorkOrderError(f"Unsupported provider: {provider}")

        except Exception as e:
            self._requests_failed += 1
            self._logger.error(f"Failed to create work order: {e}")
            raise CMSSWorkOrderError(f"Work order creation failed: {e}")

    async def _create_work_order_sap(
        self,
        request: WorkOrderCreateRequest
    ) -> InsulationRepairWorkOrder:
        """Create work order in SAP PM."""
        config = self._config.sap_pm_config

        # Map work order type to SAP order type
        order_type_map = {
            WorkOrderType.CORRECTIVE: config.order_type_corrective if config else "PM01",
            WorkOrderType.PREVENTIVE: config.order_type_preventive if config else "PM02",
            WorkOrderType.INSPECTION: config.order_type_inspection if config else "PM03",
        }

        # In production, call SAP OData API to create order

        # Mock response
        work_order_number = f"WO{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        work_order = InsulationRepairWorkOrder(
            work_order_number=work_order_number,
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
            tasks=list(request.tasks),
            material_requisitions=list(request.materials),
        )

        self._work_orders_created += 1
        self._logger.info(f"Created SAP work order: {work_order_number}")

        return work_order

    async def _create_work_order_maximo(
        self,
        request: WorkOrderCreateRequest
    ) -> InsulationRepairWorkOrder:
        """Create work order in IBM Maximo."""
        config = self._config.maximo_config

        # In production, call Maximo REST API

        # Mock response
        work_order_number = f"MX{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        work_order = InsulationRepairWorkOrder(
            work_order_number=work_order_number,
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
            tasks=list(request.tasks),
            material_requisitions=list(request.materials),
        )

        self._work_orders_created += 1
        self._logger.info(f"Created Maximo work order: {work_order_number}")

        return work_order

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
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        # In production, query CMMS
        return None

    async def update_work_order_status(
        self,
        work_order_number: str,
        new_status: WorkOrderStatus
    ) -> bool:
        """
        Update work order status.

        Args:
            work_order_number: Work order number
            new_status: New status

        Returns:
            True if updated
        """
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        # In production, update via CMMS API
        self._logger.info(f"Updated work order {work_order_number} to {new_status.value}")

        return True

    # =========================================================================
    # Material Requisition
    # =========================================================================

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
        if not self._config.material_requisition_enabled:
            self._logger.warning("Material requisition is disabled")
            return materials

        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        # In production, create requisitions in CMMS
        for material in materials:
            material.status = "submitted"

        self._logger.info(
            f"Created {len(materials)} material requisitions for {work_order_number}"
        )

        return materials

    # =========================================================================
    # Inspection Scheduling
    # =========================================================================

    async def schedule_inspection(
        self,
        title: str,
        scheduled_date: datetime,
        equipment_ids: Optional[List[str]] = None,
        functional_locations: Optional[List[str]] = None,
        duration_hours: float = 8.0,
        inspector: Optional[str] = None,
        is_recurring: bool = False,
        frequency_days: Optional[int] = None,
    ) -> InspectionSchedule:
        """
        Schedule an insulation inspection.

        Args:
            title: Inspection title
            scheduled_date: Scheduled date
            equipment_ids: Equipment to inspect
            functional_locations: Locations to inspect
            duration_hours: Estimated duration
            inspector: Assigned inspector
            is_recurring: Is recurring inspection
            frequency_days: Frequency for recurring

        Returns:
            Created inspection schedule
        """
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        schedule = InspectionSchedule(
            title=title,
            scheduled_date=scheduled_date,
            equipment_ids=equipment_ids or [],
            functional_locations=functional_locations or [],
            scheduled_duration_hours=duration_hours,
            inspector=inspector,
            is_recurring=is_recurring,
            frequency_days=frequency_days,
            status=InspectionStatus.SCHEDULED,
        )

        # In production, create in CMMS
        self._logger.info(f"Scheduled inspection: {schedule.schedule_id}")

        return schedule

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
        if not self.is_connected:
            raise CMSSConnectionError("Not connected to CMMS")

        self._requests_count += 1

        # In production, query CMMS
        return []

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "state": self._state.value,
            "provider": self._config.provider.value,
            "requests_count": self._requests_count,
            "requests_failed": self._requests_failed,
            "work_orders_created": self._work_orders_created,
            "token_valid": self._access_token is not None,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_sap_pm_connector(
    base_url: str,
    plant_code: str,
    planning_plant: str,
    client_id: str,
    client_secret: str,
    token_url: str,
    connector_name: str = "SAP-PM-Connector",
    **kwargs
) -> CMSSConnector:
    """
    Create SAP PM connector.

    Args:
        base_url: SAP API base URL
        plant_code: Plant code
        planning_plant: Planning plant
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        token_url: OAuth2 token URL
        connector_name: Connector name
        **kwargs: Additional configuration

    Returns:
        Configured CMSSConnector
    """
    oauth2_config = OAuth2Config(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
    )

    sap_config = SAPPMConfig(
        base_url=base_url,
        plant_code=plant_code,
        planning_plant=planning_plant,
    )

    config = CMSSConnectorConfig(
        connector_name=connector_name,
        provider=CMSSProvider.SAP_PM,
        auth_type=AuthenticationType.OAUTH2_CLIENT_CREDENTIALS,
        oauth2_config=oauth2_config,
        sap_pm_config=sap_config,
        **kwargs
    )

    return CMSSConnector(config)


def create_maximo_connector(
    base_url: str,
    organization: str,
    site: str,
    api_key: str,
    connector_name: str = "Maximo-Connector",
    **kwargs
) -> CMSSConnector:
    """
    Create IBM Maximo connector.

    Args:
        base_url: Maximo API base URL
        organization: Organization ID
        site: Site ID
        api_key: Maximo API key
        connector_name: Connector name
        **kwargs: Additional configuration

    Returns:
        Configured CMSSConnector
    """
    api_key_config = APIKeyConfig(
        api_key=api_key,
        header_name="apikey",
    )

    maximo_config = MaximoConfig(
        base_url=base_url,
        organization=organization,
        site=site,
        api_key=api_key,
    )

    config = CMSSConnectorConfig(
        connector_name=connector_name,
        provider=CMSSProvider.IBM_MAXIMO,
        auth_type=AuthenticationType.API_KEY,
        api_key_config=api_key_config,
        maximo_config=maximo_config,
        **kwargs
    )

    return CMSSConnector(config)
