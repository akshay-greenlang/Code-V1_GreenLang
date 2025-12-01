"""
Asset Management Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade integration for asset registry and insulation inventory:
- Equipment master data retrieval and synchronization
- Insulation inventory tracking and material specifications
- Location hierarchy navigation
- Equipment specifications and technical data
- Tag/ID mapping between systems

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
import logging
import uuid

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

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
    AuthenticationType,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class AssetSource(str, Enum):
    """Asset data source systems."""

    SAP_FLOC = "sap_floc"  # SAP Functional Location
    IBM_MAXIMO_LOCATION = "ibm_maximo_location"  # Maximo Location hierarchy
    ORACLE_EAM_LOCATION = "oracle_eam_location"  # Oracle EAM locations
    AVEVA_ASSET = "aveva_asset"  # AVEVA Asset Information Mgt
    HEXAGON_SDX = "hexagon_sdx"  # Hexagon SDx
    CUSTOM_REGISTRY = "custom_registry"  # Custom asset registry


class LocationLevel(str, Enum):
    """Location hierarchy levels."""

    ENTERPRISE = "enterprise"
    SITE = "site"
    PLANT = "plant"
    AREA = "area"
    UNIT = "unit"
    SYSTEM = "system"
    EQUIPMENT = "equipment"
    COMPONENT = "component"


class AssetStatus(str, Enum):
    """Asset operational status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"
    UNDER_CONSTRUCTION = "under_construction"
    PLANNED = "planned"
    MOTHBALLED = "mothballed"


class InsulationClass(str, Enum):
    """Insulation classification."""

    HOT = "hot"  # Hot service (> 60C)
    COLD = "cold"  # Cold service (< 15C)
    CRYOGENIC = "cryogenic"  # Cryogenic (< -40C)
    PERSONNEL_PROTECTION = "personnel_protection"  # Safety insulation
    CONDENSATION_CONTROL = "condensation_control"  # Anti-sweat
    ACOUSTIC = "acoustic"  # Sound attenuation
    FIRE_PROTECTION = "fire_protection"  # Fire-rated


class InsulationMaterial(str, Enum):
    """Types of insulation materials."""

    MINERAL_WOOL = "mineral_wool"
    FIBERGLASS = "fiberglass"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    POLYISOCYANURATE = "polyisocyanurate"
    PHENOLIC_FOAM = "phenolic_foam"
    EXPANDED_POLYSTYRENE = "expanded_polystyrene"
    EXTRUDED_POLYSTYRENE = "extruded_polystyrene"
    POLYURETHANE = "polyurethane"
    CERAMIC_FIBER = "ceramic_fiber"
    VERMICULITE = "vermiculite"


class JacketMaterial(str, Enum):
    """Types of jacketing materials."""

    ALUMINUM = "aluminum"
    STAINLESS_STEEL = "stainless_steel"
    GALVANIZED_STEEL = "galvanized_steel"
    PVC = "pvc"
    MASTICS = "mastics"
    CANVAS = "canvas"
    FIBERGLASS_CLOTH = "fiberglass_cloth"
    METAL_MESH = "metal_mesh"


class ConditionRating(str, Enum):
    """Asset/insulation condition ratings."""

    EXCELLENT = "excellent"  # 90-100% - Like new
    GOOD = "good"  # 70-89% - Minor wear
    FAIR = "fair"  # 50-69% - Moderate wear
    POOR = "poor"  # 30-49% - Significant degradation
    CRITICAL = "critical"  # 10-29% - Requires immediate attention
    FAILED = "failed"  # 0-9% - Non-functional


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class AssetRegistryConfig(BaseModel):
    """Asset registry configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., description="Asset registry API URL")
    api_version: str = Field(default="v1", description="API version")
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization identifier"
    )
    default_site: Optional[str] = Field(
        default=None,
        description="Default site code"
    )


class InsulationInventoryConfig(BaseModel):
    """Insulation inventory configuration."""

    model_config = ConfigDict(extra="forbid")

    inventory_enabled: bool = Field(
        default=True,
        description="Enable inventory tracking"
    )
    warehouse_codes: List[str] = Field(
        default_factory=list,
        description="Warehouse codes for materials"
    )
    auto_sync_interval_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Auto-sync interval in minutes"
    )


class TagMappingConfig(BaseModel):
    """Tag/ID mapping configuration."""

    model_config = ConfigDict(extra="forbid")

    mapping_source: str = Field(
        default="database",
        description="Tag mapping source"
    )
    mapping_table: Optional[str] = Field(
        default=None,
        description="Database table for mappings"
    )
    cache_mappings: bool = Field(
        default=True,
        description="Cache tag mappings"
    )


class AssetManagementConnectorConfig(BaseConnectorConfig):
    """Configuration for asset management connector."""

    model_config = ConfigDict(extra="forbid")

    # Source system
    asset_source: AssetSource = Field(
        default=AssetSource.CUSTOM_REGISTRY,
        description="Asset data source"
    )

    # Authentication
    auth_type: AuthenticationType = Field(
        default=AuthenticationType.API_KEY,
        description="Authentication type"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for basic auth"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for basic auth"
    )

    # Registry configuration
    registry_config: AssetRegistryConfig = Field(
        ...,
        description="Asset registry configuration"
    )

    # Inventory configuration
    inventory_config: InsulationInventoryConfig = Field(
        default_factory=InsulationInventoryConfig,
        description="Inventory configuration"
    )

    # Tag mapping
    tag_mapping_config: TagMappingConfig = Field(
        default_factory=TagMappingConfig,
        description="Tag mapping configuration"
    )

    # Sync settings
    full_sync_on_connect: bool = Field(
        default=False,
        description="Perform full sync on connection"
    )
    incremental_sync_enabled: bool = Field(
        default=True,
        description="Enable incremental sync"
    )

    def __init__(self, **data):
        """Initialize with connector type set."""
        data['connector_type'] = ConnectorType.ASSET_MANAGEMENT
        super().__init__(**data)


# =============================================================================
# Data Models - Location Hierarchy
# =============================================================================


class LocationNode(BaseModel):
    """Node in location hierarchy."""

    model_config = ConfigDict(frozen=False)

    location_id: str = Field(..., description="Location identifier")
    location_code: str = Field(..., description="Location code")
    name: str = Field(..., description="Location name")
    description: Optional[str] = Field(default=None, description="Description")
    level: LocationLevel = Field(..., description="Hierarchy level")

    # Hierarchy
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent location ID"
    )
    path: str = Field(
        default="",
        description="Full path in hierarchy"
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Depth in hierarchy"
    )

    # Geographic info
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    building: Optional[str] = Field(default=None, description="Building name")
    floor: Optional[str] = Field(default=None, description="Floor/level")
    room: Optional[str] = Field(default=None, description="Room")

    # Metadata
    status: AssetStatus = Field(
        default=AssetStatus.ACTIVE,
        description="Location status"
    )
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom attributes"
    )


class LocationHierarchy(BaseModel):
    """Complete location hierarchy."""

    model_config = ConfigDict(frozen=False)

    root_locations: List[LocationNode] = Field(
        default_factory=list,
        description="Root level locations"
    )
    all_locations: Dict[str, LocationNode] = Field(
        default_factory=dict,
        description="All locations indexed by ID"
    )
    total_count: int = Field(default=0, ge=0)
    max_depth: int = Field(default=0, ge=0)
    last_synced: Optional[datetime] = Field(default=None)


# =============================================================================
# Data Models - Equipment
# =============================================================================


class EquipmentSpecification(BaseModel):
    """Technical specifications for insulated equipment."""

    model_config = ConfigDict(frozen=False)

    equipment_id: str = Field(..., description="Equipment ID")

    # Operating conditions
    operating_temperature_min_c: Optional[float] = Field(
        default=None,
        description="Minimum operating temperature"
    )
    operating_temperature_max_c: Optional[float] = Field(
        default=None,
        description="Maximum operating temperature"
    )
    operating_temperature_normal_c: Optional[float] = Field(
        default=None,
        description="Normal operating temperature"
    )
    operating_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Operating pressure in kPa"
    )
    design_temperature_c: Optional[float] = Field(
        default=None,
        description="Design temperature"
    )
    design_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Design pressure"
    )

    # Physical dimensions
    outer_diameter_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Outer diameter in mm"
    )
    length_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Length in meters"
    )
    surface_area_m2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Surface area in m2"
    )
    volume_m3: Optional[float] = Field(
        default=None,
        ge=0,
        description="Volume in m3"
    )

    # Material
    material_of_construction: Optional[str] = Field(
        default=None,
        description="Material of construction"
    )
    wall_thickness_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Wall thickness"
    )

    # Process info
    process_fluid: Optional[str] = Field(
        default=None,
        description="Process fluid"
    )
    process_service: Optional[str] = Field(
        default=None,
        description="Process service description"
    )

    # Safety
    is_corrosion_under_insulation_risk: bool = Field(
        default=False,
        description="CUI risk flag"
    )
    is_personnel_protection_required: bool = Field(
        default=False,
        description="Personnel protection required"
    )
    touch_temperature_limit_c: Optional[float] = Field(
        default=60.0,
        description="Maximum touch temperature"
    )


class InsulationSpecification(BaseModel):
    """Insulation specifications for equipment."""

    model_config = ConfigDict(frozen=False)

    spec_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Specification ID"
    )
    equipment_id: str = Field(..., description="Equipment ID")

    # Classification
    insulation_class: InsulationClass = Field(
        default=InsulationClass.HOT,
        description="Insulation classification"
    )

    # Material specifications
    insulation_material: InsulationMaterial = Field(
        ...,
        description="Insulation material type"
    )
    material_density_kg_m3: Optional[float] = Field(
        default=None,
        ge=0,
        description="Material density"
    )
    thermal_conductivity_w_mk: Optional[float] = Field(
        default=None,
        ge=0,
        description="Thermal conductivity at mean temp"
    )
    thermal_conductivity_reference_temp_c: Optional[float] = Field(
        default=None,
        description="Reference temperature for conductivity"
    )

    # Thickness
    thickness_mm: float = Field(
        ...,
        ge=0,
        description="Insulation thickness in mm"
    )
    thickness_calculated_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Calculated required thickness"
    )
    thickness_installed_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actually installed thickness"
    )
    number_of_layers: int = Field(
        default=1,
        ge=1,
        description="Number of insulation layers"
    )

    # Jacketing
    jacket_material: Optional[JacketMaterial] = Field(
        default=None,
        description="Jacketing material"
    )
    jacket_thickness_mm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Jacketing thickness"
    )

    # Vapor barrier
    vapor_barrier_required: bool = Field(
        default=False,
        description="Vapor barrier required"
    )
    vapor_barrier_type: Optional[str] = Field(
        default=None,
        description="Vapor barrier type"
    )

    # Design basis
    ambient_temperature_c: float = Field(
        default=20.0,
        description="Design ambient temperature"
    )
    wind_speed_m_s: float = Field(
        default=0.0,
        ge=0,
        description="Design wind speed"
    )
    surface_temperature_limit_c: Optional[float] = Field(
        default=None,
        description="Surface temperature limit"
    )
    heat_loss_limit_w_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat loss limit per unit length"
    )

    # Installation
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Installation date"
    )
    manufacturer: Optional[str] = Field(
        default=None,
        description="Insulation manufacturer"
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Product name/model"
    )
    installer: Optional[str] = Field(
        default=None,
        description="Installation contractor"
    )

    # Expected performance
    expected_heat_loss_w_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Expected heat loss"
    )
    expected_surface_temperature_c: Optional[float] = Field(
        default=None,
        description="Expected surface temperature"
    )
    expected_lifespan_years: int = Field(
        default=20,
        ge=1,
        description="Expected service life"
    )


class InsulatedEquipmentAsset(BaseModel):
    """Complete insulated equipment asset record."""

    model_config = ConfigDict(frozen=False)

    # Identification
    asset_id: str = Field(..., description="Asset ID")
    equipment_tag: str = Field(..., description="Equipment tag/number")
    equipment_name: str = Field(..., description="Equipment name")
    description: Optional[str] = Field(default=None)

    # Classification
    equipment_type: str = Field(..., description="Equipment type")
    equipment_subtype: Optional[str] = Field(default=None)
    equipment_class: Optional[str] = Field(default=None)

    # Location
    location_id: str = Field(..., description="Location ID")
    location_path: Optional[str] = Field(default=None)
    plant_code: Optional[str] = Field(default=None)
    area_code: Optional[str] = Field(default=None)

    # Status
    status: AssetStatus = Field(
        default=AssetStatus.ACTIVE,
        description="Asset status"
    )
    condition_rating: ConditionRating = Field(
        default=ConditionRating.GOOD,
        description="Current condition"
    )
    condition_score: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Condition score 0-100"
    )

    # Specifications
    equipment_spec: Optional[EquipmentSpecification] = Field(
        default=None,
        description="Equipment specifications"
    )
    insulation_spec: Optional[InsulationSpecification] = Field(
        default=None,
        description="Insulation specifications"
    )

    # Inspection info
    last_inspection_date: Optional[datetime] = Field(default=None)
    next_inspection_date: Optional[datetime] = Field(default=None)
    inspection_frequency_days: int = Field(
        default=365,
        ge=1,
        description="Inspection frequency"
    )

    # Thermal imaging history
    thermal_images_count: int = Field(default=0, ge=0)
    last_thermal_image_date: Optional[datetime] = Field(default=None)
    hotspots_detected: int = Field(default=0, ge=0)

    # Energy impact
    estimated_heat_loss_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current estimated heat loss"
    )
    annual_energy_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual energy cost from losses"
    )
    annual_co2_emissions_tonnes: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual CO2 emissions"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)

    # Custom
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


# =============================================================================
# Data Models - Inventory
# =============================================================================


class InsulationMaterialStock(BaseModel):
    """Insulation material inventory item."""

    model_config = ConfigDict(frozen=False)

    stock_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stock item ID"
    )
    material_code: str = Field(..., description="Material code")
    material_name: str = Field(..., description="Material name")
    material_type: InsulationMaterial = Field(
        ...,
        description="Material type"
    )

    # Specifications
    thickness_mm: Optional[float] = Field(default=None, ge=0)
    density_kg_m3: Optional[float] = Field(default=None, ge=0)
    thermal_conductivity_w_mk: Optional[float] = Field(default=None, ge=0)
    max_temperature_c: Optional[float] = Field(default=None)

    # Inventory
    warehouse_code: str = Field(..., description="Warehouse code")
    bin_location: Optional[str] = Field(default=None)
    quantity_on_hand: float = Field(default=0, ge=0)
    unit_of_measure: str = Field(default="m2")
    minimum_quantity: float = Field(default=0, ge=0)
    reorder_quantity: float = Field(default=0, ge=0)

    # Cost
    unit_cost: Optional[float] = Field(default=None, ge=0)
    total_value: Optional[float] = Field(default=None, ge=0)

    # Dates
    last_receipt_date: Optional[datetime] = Field(default=None)
    last_issue_date: Optional[datetime] = Field(default=None)

    @property
    def is_below_minimum(self) -> bool:
        """Check if stock is below minimum."""
        return self.quantity_on_hand < self.minimum_quantity


class InsulationInventorySummary(BaseModel):
    """Summary of insulation inventory."""

    model_config = ConfigDict(frozen=True)

    total_items: int = Field(default=0, ge=0)
    items_below_minimum: int = Field(default=0, ge=0)
    total_value: float = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    by_material_type: Dict[str, int] = Field(default_factory=dict)
    by_warehouse: Dict[str, int] = Field(default_factory=dict)


# =============================================================================
# Data Models - Tag Mapping
# =============================================================================


class TagMapping(BaseModel):
    """Mapping between different system tags/IDs."""

    model_config = ConfigDict(frozen=False)

    mapping_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Mapping ID"
    )
    source_system: str = Field(..., description="Source system name")
    source_tag: str = Field(..., description="Tag in source system")
    target_system: str = Field(..., description="Target system name")
    target_tag: str = Field(..., description="Tag in target system")

    # Metadata
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment ID"
    )
    mapping_type: str = Field(
        default="equipment",
        description="Type of mapping"
    )
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None)


# =============================================================================
# Asset Management Connector
# =============================================================================


class AssetManagementConnector(BaseConnector):
    """
    Asset Management Connector for GL-015 INSULSCAN.

    Provides unified interface for asset registry and inventory management
    supporting equipment master data, insulation specifications,
    location hierarchy, and tag mapping.

    Features:
    - Equipment master data retrieval and sync
    - Insulation specification management
    - Location hierarchy navigation
    - Insulation inventory tracking
    - Cross-system tag mapping
    """

    def __init__(self, config: AssetManagementConnectorConfig) -> None:
        """
        Initialize asset management connector.

        Args:
            config: Asset management connector configuration
        """
        super().__init__(config)
        self._asset_config = config

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Cached data
        self._location_hierarchy: Optional[LocationHierarchy] = None
        self._tag_mappings: Dict[str, TagMapping] = {}

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Establish connection to asset registry."""
        self._logger.info(
            f"Connecting to asset registry: {self._asset_config.asset_source.value}"
        )

        try:
            self._state = ConnectionState.CONNECTING

            # Get HTTP session from pool
            self._http_session = await self._pool.get_session()

            # Verify connection
            await self._verify_connection()

            # Load location hierarchy if full sync enabled
            if self._asset_config.full_sync_on_connect:
                await self._load_location_hierarchy()

            # Load tag mappings if caching enabled
            if self._asset_config.tag_mapping_config.cache_mappings:
                await self._load_tag_mappings()

            self._state = ConnectionState.CONNECTED
            self._logger.info("Connected to asset registry")

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to asset registry: {e}")
            raise ConnectionError(f"Asset registry connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from asset registry."""
        self._logger.info("Disconnecting from asset registry")
        self._location_hierarchy = None
        self._tag_mappings.clear()
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on asset registry connection."""
        import time
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Not connected: {self._state.value}"
                )

            # Verify API is responding
            await self._verify_connection()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Asset registry healthy",
                details={
                    "source": self._asset_config.asset_source.value,
                    "locations_cached": self._location_hierarchy is not None,
                    "tag_mappings_cached": len(self._tag_mappings),
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
        if not self._asset_config.registry_config.base_url:
            raise ConfigurationError("Registry base URL is required")

        if self._asset_config.auth_type == AuthenticationType.API_KEY:
            if not self._asset_config.api_key:
                raise ConfigurationError("API key required for API key auth")

        elif self._asset_config.auth_type == AuthenticationType.BASIC:
            if not self._asset_config.username or not self._asset_config.password:
                raise ConfigurationError("Username and password required for basic auth")

        return True

    async def _verify_connection(self) -> None:
        """Verify asset registry connection."""
        config = self._asset_config.registry_config
        url = f"{config.base_url}/api/{config.api_version}/health"

        headers = self._get_auth_headers()

        async with self._http_session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status not in [200, 404]:
                raise ConnectionError(
                    f"Asset registry verification failed: {response.status}"
                )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {"Content-Type": "application/json"}

        if self._asset_config.auth_type == AuthenticationType.API_KEY:
            headers["X-API-Key"] = self._asset_config.api_key

        elif self._asset_config.auth_type == AuthenticationType.BASIC:
            import base64
            credentials = f"{self._asset_config.username}:{self._asset_config.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    # =========================================================================
    # Location Hierarchy
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_location_hierarchy(
        self,
        force_refresh: bool = False
    ) -> LocationHierarchy:
        """
        Get complete location hierarchy.

        Args:
            force_refresh: Force refresh from source

        Returns:
            Location hierarchy
        """
        if self._location_hierarchy and not force_refresh:
            return self._location_hierarchy

        async def _fetch():
            return await self._load_location_hierarchy()

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_location_hierarchy",
            validate_result=False
        )

    async def _load_location_hierarchy(self) -> LocationHierarchy:
        """Load location hierarchy from source."""
        config = self._asset_config.registry_config
        url = f"{config.base_url}/api/{config.api_version}/locations"

        headers = self._get_auth_headers()

        async with self._http_session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                # Return empty hierarchy if endpoint not available
                self._logger.warning(
                    f"Location hierarchy endpoint returned {response.status}"
                )
                return LocationHierarchy()

            data = await response.json()
            locations = data.get("locations", [])

            # Build hierarchy
            all_locations: Dict[str, LocationNode] = {}
            root_locations: List[LocationNode] = []

            for loc_data in locations:
                node = LocationNode(
                    location_id=loc_data.get("id", str(uuid.uuid4())),
                    location_code=loc_data.get("code", ""),
                    name=loc_data.get("name", ""),
                    description=loc_data.get("description"),
                    level=LocationLevel(loc_data.get("level", "equipment")),
                    parent_id=loc_data.get("parent_id"),
                    path=loc_data.get("path", ""),
                    depth=loc_data.get("depth", 0),
                    status=AssetStatus(loc_data.get("status", "active")),
                )
                all_locations[node.location_id] = node

                if not node.parent_id:
                    root_locations.append(node)

            max_depth = max((n.depth for n in all_locations.values()), default=0)

            self._location_hierarchy = LocationHierarchy(
                root_locations=root_locations,
                all_locations=all_locations,
                total_count=len(all_locations),
                max_depth=max_depth,
                last_synced=datetime.utcnow(),
            )

            self._logger.info(
                f"Loaded {len(all_locations)} locations in hierarchy"
            )

            return self._location_hierarchy

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_location(
        self,
        location_id: str
    ) -> Optional[LocationNode]:
        """
        Get location by ID.

        Args:
            location_id: Location identifier

        Returns:
            Location node or None
        """
        if self._location_hierarchy:
            return self._location_hierarchy.all_locations.get(location_id)

        # Fetch from API
        async def _fetch():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/locations/{location_id}"

            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise ConnectorError(f"API error: {response.status}")

                data = await response.json()
                return LocationNode(
                    location_id=data.get("id", location_id),
                    location_code=data.get("code", ""),
                    name=data.get("name", ""),
                    level=LocationLevel(data.get("level", "equipment")),
                    parent_id=data.get("parent_id"),
                    path=data.get("path", ""),
                )

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_location",
            use_cache=True,
            cache_key=f"location_{location_id}"
        )

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_child_locations(
        self,
        parent_id: str
    ) -> List[LocationNode]:
        """
        Get child locations of a parent.

        Args:
            parent_id: Parent location ID

        Returns:
            List of child locations
        """
        if self._location_hierarchy:
            return [
                node for node in self._location_hierarchy.all_locations.values()
                if node.parent_id == parent_id
            ]

        async def _fetch():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/locations"
            params = {"parent_id": parent_id}

            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise ConnectorError(f"API error: {response.status}")

                data = await response.json()
                return [
                    LocationNode(
                        location_id=loc.get("id", ""),
                        location_code=loc.get("code", ""),
                        name=loc.get("name", ""),
                        level=LocationLevel(loc.get("level", "equipment")),
                        parent_id=parent_id,
                    )
                    for loc in data.get("locations", [])
                ]

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_child_locations",
            validate_result=False
        )

    # =========================================================================
    # Equipment Operations
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_equipment(
        self,
        equipment_id: str
    ) -> Optional[InsulatedEquipmentAsset]:
        """
        Get equipment by ID.

        Args:
            equipment_id: Equipment identifier

        Returns:
            Equipment asset or None
        """
        async def _fetch():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/equipment/{equipment_id}"

            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise ConnectorError(f"API error: {response.status}")

                data = await response.json()
                return self._parse_equipment(data)

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_equipment",
            use_cache=True,
            cache_key=f"equipment_{equipment_id}"
        )

    @with_retry(max_retries=3, base_delay=1.0)
    async def list_equipment(
        self,
        location_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
        status: Optional[AssetStatus] = None,
        condition_max: Optional[ConditionRating] = None,
        has_insulation: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[InsulatedEquipmentAsset]:
        """
        List equipment with filters.

        Args:
            location_id: Filter by location
            equipment_type: Filter by type
            status: Filter by status
            condition_max: Filter by maximum condition rating
            has_insulation: Only insulated equipment
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of equipment
        """
        async def _fetch():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/equipment"

            params = {
                "limit": str(limit),
                "offset": str(offset),
            }
            if location_id:
                params["location_id"] = location_id
            if equipment_type:
                params["type"] = equipment_type
            if status:
                params["status"] = status.value
            if has_insulation:
                params["has_insulation"] = "true"

            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    raise ConnectorError(f"API error: {response.status}")

                data = await response.json()
                return [
                    self._parse_equipment(eq)
                    for eq in data.get("equipment", [])
                ]

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="list_equipment",
            validate_result=False
        )

    def _parse_equipment(self, data: Dict[str, Any]) -> InsulatedEquipmentAsset:
        """Parse equipment data from API response."""
        # Parse equipment spec if present
        equipment_spec = None
        if "equipment_spec" in data:
            spec_data = data["equipment_spec"]
            equipment_spec = EquipmentSpecification(
                equipment_id=data.get("id", ""),
                operating_temperature_min_c=spec_data.get("operating_temp_min"),
                operating_temperature_max_c=spec_data.get("operating_temp_max"),
                operating_temperature_normal_c=spec_data.get("operating_temp_normal"),
                outer_diameter_mm=spec_data.get("outer_diameter_mm"),
                length_m=spec_data.get("length_m"),
                surface_area_m2=spec_data.get("surface_area_m2"),
            )

        # Parse insulation spec if present
        insulation_spec = None
        if "insulation_spec" in data:
            ins_data = data["insulation_spec"]
            insulation_spec = InsulationSpecification(
                equipment_id=data.get("id", ""),
                insulation_class=InsulationClass(
                    ins_data.get("class", "hot")
                ),
                insulation_material=InsulationMaterial(
                    ins_data.get("material", "mineral_wool")
                ),
                thickness_mm=ins_data.get("thickness_mm", 50),
                jacket_material=JacketMaterial(
                    ins_data.get("jacket_material", "aluminum")
                ) if ins_data.get("jacket_material") else None,
            )

        return InsulatedEquipmentAsset(
            asset_id=data.get("id", str(uuid.uuid4())),
            equipment_tag=data.get("tag", ""),
            equipment_name=data.get("name", ""),
            description=data.get("description"),
            equipment_type=data.get("type", "other"),
            location_id=data.get("location_id", ""),
            location_path=data.get("location_path"),
            plant_code=data.get("plant_code"),
            status=AssetStatus(data.get("status", "active")),
            condition_rating=ConditionRating(
                data.get("condition_rating", "good")
            ),
            condition_score=data.get("condition_score", 80.0),
            equipment_spec=equipment_spec,
            insulation_spec=insulation_spec,
            last_inspection_date=datetime.fromisoformat(
                data["last_inspection_date"]
            ) if data.get("last_inspection_date") else None,
            next_inspection_date=datetime.fromisoformat(
                data["next_inspection_date"]
            ) if data.get("next_inspection_date") else None,
            thermal_images_count=data.get("thermal_images_count", 0),
            hotspots_detected=data.get("hotspots_detected", 0),
            estimated_heat_loss_kw=data.get("estimated_heat_loss_kw"),
            annual_energy_cost=data.get("annual_energy_cost"),
        )

    @with_retry(max_retries=3, base_delay=1.0)
    async def update_equipment_condition(
        self,
        equipment_id: str,
        condition_rating: ConditionRating,
        condition_score: float,
        notes: Optional[str] = None
    ) -> InsulatedEquipmentAsset:
        """
        Update equipment condition from inspection.

        Args:
            equipment_id: Equipment ID
            condition_rating: New condition rating
            condition_score: Condition score 0-100
            notes: Condition notes

        Returns:
            Updated equipment
        """
        async def _update():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/equipment/{equipment_id}/condition"

            payload = {
                "condition_rating": condition_rating.value,
                "condition_score": condition_score,
                "notes": notes,
                "updated_at": datetime.utcnow().isoformat(),
            }

            async with self._http_session.patch(
                url,
                headers=self._get_auth_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status not in [200, 204]:
                    raise ConnectorError(f"Update failed: {response.status}")

                # Invalidate cache
                if self._cache:
                    await self._cache.delete(f"equipment_{equipment_id}")

                # Return updated equipment
                return await self.get_equipment(equipment_id)

        return await self.execute_with_protection(
            operation=_update,
            operation_name="update_equipment_condition",
            validate_result=False
        )

    # =========================================================================
    # Insulation Inventory
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_inventory_summary(self) -> InsulationInventorySummary:
        """
        Get insulation inventory summary.

        Returns:
            Inventory summary
        """
        if not self._asset_config.inventory_config.inventory_enabled:
            return InsulationInventorySummary()

        async def _fetch():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/inventory/summary"

            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise ConnectorError(f"API error: {response.status}")

                data = await response.json()
                return InsulationInventorySummary(
                    total_items=data.get("total_items", 0),
                    items_below_minimum=data.get("items_below_minimum", 0),
                    total_value=data.get("total_value", 0),
                    by_material_type=data.get("by_material_type", {}),
                    by_warehouse=data.get("by_warehouse", {}),
                )

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_inventory_summary",
            use_cache=True,
            cache_key="inventory_summary"
        )

    @with_retry(max_retries=3, base_delay=1.0)
    async def list_inventory(
        self,
        warehouse_code: Optional[str] = None,
        material_type: Optional[InsulationMaterial] = None,
        below_minimum_only: bool = False,
        limit: int = 100
    ) -> List[InsulationMaterialStock]:
        """
        List inventory items.

        Args:
            warehouse_code: Filter by warehouse
            material_type: Filter by material type
            below_minimum_only: Only items below minimum stock
            limit: Maximum results

        Returns:
            List of inventory items
        """
        if not self._asset_config.inventory_config.inventory_enabled:
            return []

        async def _fetch():
            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/inventory"

            params = {"limit": str(limit)}
            if warehouse_code:
                params["warehouse"] = warehouse_code
            if material_type:
                params["material_type"] = material_type.value
            if below_minimum_only:
                params["below_minimum"] = "true"

            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                params=params,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    raise ConnectorError(f"API error: {response.status}")

                data = await response.json()
                return [
                    InsulationMaterialStock(
                        stock_id=item.get("id", str(uuid.uuid4())),
                        material_code=item.get("code", ""),
                        material_name=item.get("name", ""),
                        material_type=InsulationMaterial(
                            item.get("type", "mineral_wool")
                        ),
                        thickness_mm=item.get("thickness_mm"),
                        warehouse_code=item.get("warehouse", ""),
                        quantity_on_hand=item.get("quantity", 0),
                        unit_of_measure=item.get("uom", "m2"),
                        minimum_quantity=item.get("min_quantity", 0),
                        unit_cost=item.get("unit_cost"),
                    )
                    for item in data.get("items", [])
                ]

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="list_inventory",
            validate_result=False
        )

    # =========================================================================
    # Tag Mapping
    # =========================================================================

    async def _load_tag_mappings(self) -> None:
        """Load tag mappings into cache."""
        config = self._asset_config.registry_config
        url = f"{config.base_url}/api/{config.api_version}/tag-mappings"

        try:
            async with self._http_session.get(
                url,
                headers=self._get_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    self._logger.warning(
                        f"Tag mappings endpoint returned {response.status}"
                    )
                    return

                data = await response.json()
                for mapping in data.get("mappings", []):
                    tm = TagMapping(
                        mapping_id=mapping.get("id", str(uuid.uuid4())),
                        source_system=mapping.get("source_system", ""),
                        source_tag=mapping.get("source_tag", ""),
                        target_system=mapping.get("target_system", ""),
                        target_tag=mapping.get("target_tag", ""),
                        equipment_id=mapping.get("equipment_id"),
                    )
                    key = f"{tm.source_system}:{tm.source_tag}"
                    self._tag_mappings[key] = tm

                self._logger.info(
                    f"Loaded {len(self._tag_mappings)} tag mappings"
                )

        except Exception as e:
            self._logger.warning(f"Failed to load tag mappings: {e}")

    def get_mapped_tag(
        self,
        source_system: str,
        source_tag: str,
        target_system: str
    ) -> Optional[str]:
        """
        Get mapped tag for a source tag.

        Args:
            source_system: Source system name
            source_tag: Tag in source system
            target_system: Target system name

        Returns:
            Mapped tag in target system or None
        """
        key = f"{source_system}:{source_tag}"
        mapping = self._tag_mappings.get(key)

        if mapping and mapping.target_system == target_system:
            return mapping.target_tag

        return None

    @with_retry(max_retries=3, base_delay=1.0)
    async def create_tag_mapping(
        self,
        source_system: str,
        source_tag: str,
        target_system: str,
        target_tag: str,
        equipment_id: Optional[str] = None
    ) -> TagMapping:
        """
        Create a new tag mapping.

        Args:
            source_system: Source system name
            source_tag: Tag in source system
            target_system: Target system name
            target_tag: Tag in target system
            equipment_id: Associated equipment

        Returns:
            Created tag mapping
        """
        async def _create():
            mapping = TagMapping(
                source_system=source_system,
                source_tag=source_tag,
                target_system=target_system,
                target_tag=target_tag,
                equipment_id=equipment_id,
            )

            config = self._asset_config.registry_config
            url = f"{config.base_url}/api/{config.api_version}/tag-mappings"

            async with self._http_session.post(
                url,
                headers=self._get_auth_headers(),
                json={
                    "source_system": source_system,
                    "source_tag": source_tag,
                    "target_system": target_system,
                    "target_tag": target_tag,
                    "equipment_id": equipment_id,
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status not in [200, 201]:
                    raise ConnectorError(f"Create failed: {response.status}")

            # Add to cache
            key = f"{source_system}:{source_tag}"
            self._tag_mappings[key] = mapping

            return mapping

        return await self.execute_with_protection(
            operation=_create,
            operation_name="create_tag_mapping",
            validate_result=False
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_asset_management_connector(
    base_url: str,
    connector_name: str = "AssetManagement",
    asset_source: AssetSource = AssetSource.CUSTOM_REGISTRY,
    **kwargs
) -> AssetManagementConnector:
    """
    Factory function to create asset management connector.

    Args:
        base_url: Asset registry base URL
        connector_name: Connector name
        asset_source: Asset data source
        **kwargs: Additional configuration options

    Returns:
        Configured AssetManagementConnector instance
    """
    registry_config = AssetRegistryConfig(
        base_url=base_url,
        **kwargs.pop("registry_config", {})
    )

    config = AssetManagementConnectorConfig(
        connector_name=connector_name,
        asset_source=asset_source,
        registry_config=registry_config,
        **kwargs
    )
    return AssetManagementConnector(config)
