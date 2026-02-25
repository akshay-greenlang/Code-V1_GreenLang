"""
Upstream Transportation & Distribution Service Setup - AGENT-MRV-017

This module provides the service facade that wires together all 7 engines
for upstream transportation and distribution emissions calculations (Scope 3 Category 4).

The UpstreamTransportationService class provides a high-level API for:
- Distance-based, fuel-based, and spend-based calculations
- Multi-leg transport chain modeling
- Emission factor management with 6+ authoritative sources
- Mode/vehicle classification and optimization
- Compliance checking across 7 regulatory frameworks
- Uncertainty quantification via Monte Carlo simulation
- Aggregations, hot spots, and reporting

Engines:
    1. TransportDatabaseEngine - Data access and persistence
    2. DistanceBasedCalculatorEngine - Distance-based method (Tier 1)
    3. FuelBasedCalculatorEngine - Fuel-based method (Tier 2)
    4. SpendBasedCalculatorEngine - Spend-based method (Tier 1)
    5. MultiLegCalculatorEngine - Multi-leg transport chain orchestrator
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. TransportPipelineEngine - End-to-end calculation pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.upstream_transportation.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate(CalculateRequest(
    ...     tenant_id="acme-corp",
    ...     shipment_id="SHP-2024-001",
    ...     mode="ROAD",
    ...     vehicle_type="DIESEL_TRUCK",
    ...     distance_km=500.0,
    ...     cargo_mass_kg=20000.0
    ... ))
    >>> assert response.success
    >>> assert response.total_co2e_kg > 0

Integration:
    >>> from greenlang.upstream_transportation.setup import configure_upstream_transportation
    >>> app = FastAPI()
    >>> configure_upstream_transportation(app)  # Registers routes, middleware
"""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional['UpstreamTransportationService'] = None

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class CalculateRequest(BaseModel):
    """Request model for single emissions calculation."""

    tenant_id: str = Field(..., description="Tenant identifier")
    shipment_id: str = Field(..., description="Unique shipment identifier")
    calculation_method: str = Field(
        "DISTANCE_BASED",
        description="Calculation method: DISTANCE_BASED, FUEL_BASED, SPEND_BASED"
    )

    # Common fields
    mode: Optional[str] = Field(None, description="Transport mode: ROAD, RAIL, AIR, SEA")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type (e.g., DIESEL_TRUCK, CONTAINER_SHIP)")

    # Distance-based fields
    distance_km: Optional[float] = Field(None, ge=0, description="Distance traveled in km")
    cargo_mass_kg: Optional[float] = Field(None, ge=0, description="Cargo mass in kg")

    # Fuel-based fields
    fuel_type: Optional[str] = Field(None, description="Fuel type (e.g., DIESEL, GASOLINE, JET_FUEL)")
    fuel_amount: Optional[float] = Field(None, ge=0, description="Fuel consumed")
    fuel_unit: Optional[str] = Field(None, description="Fuel unit (LITERS, KG, GALLONS)")

    # Spend-based fields
    transport_spend: Optional[float] = Field(None, ge=0, description="Transport spend in USD")
    spend_category: Optional[str] = Field(None, description="Spend category for EEIO mapping")

    # Optional metadata
    origin: Optional[str] = Field(None, description="Origin location")
    destination: Optional[str] = Field(None, description="Destination location")
    carrier_name: Optional[str] = Field(None, description="Transport carrier name")
    reporting_period: Optional[str] = Field(None, description="Reporting period (YYYY-MM)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('calculation_method')
    def validate_method(cls, v):
        """Validate calculation method."""
        allowed = ['DISTANCE_BASED', 'FUEL_BASED', 'SPEND_BASED']
        if v not in allowed:
            raise ValueError(f"calculation_method must be one of {allowed}")
        return v


class BatchCalculateRequest(BaseModel):
    """Request model for batch calculations."""

    tenant_id: str = Field(..., description="Tenant identifier")
    calculations: List[CalculateRequest] = Field(..., description="List of calculations")
    parallel: bool = Field(True, description="Execute in parallel")


class TransportChainRequest(BaseModel):
    """Request model for creating a multi-leg transport chain."""

    tenant_id: str = Field(..., description="Tenant identifier")
    chain_id: str = Field(..., description="Unique chain identifier")
    legs: List[CalculateRequest] = Field(..., description="Sequential transport legs")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ClassificationRequest(BaseModel):
    """Request model for shipment classification."""

    tenant_id: str = Field(..., description="Tenant identifier")
    shipment_description: Optional[str] = Field(None, description="Free-text shipment description")
    cargo_type: Optional[str] = Field(None, description="Cargo type")
    distance_km: Optional[float] = Field(None, ge=0)
    origin_country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code")
    destination_country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    tenant_id: str = Field(..., description="Tenant identifier")
    calculation_id: str = Field(..., description="Calculation to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description="Frameworks: GHG_PROTOCOL, ISO_14083, GLEC, CSRD, CDP, SBTi, EU_ETS"
    )


class CustomEmissionFactorRequest(BaseModel):
    """Request model for creating custom emission factors."""

    tenant_id: str = Field(..., description="Tenant identifier")
    mode: str = Field(..., description="Transport mode")
    vehicle_type: str = Field(..., description="Vehicle type")
    fuel_type: Optional[str] = Field(None, description="Fuel type")
    co2_factor: float = Field(..., ge=0, description="CO2 emission factor")
    ch4_factor: Optional[float] = Field(None, ge=0, description="CH4 emission factor")
    n2o_factor: Optional[float] = Field(None, ge=0, description="N2O emission factor")
    unit: str = Field(..., description="Factor unit (e.g., kg_CO2e/tkm)")
    source: str = Field(..., description="Data source/authority")
    valid_from: datetime = Field(..., description="Valid from date")
    valid_until: Optional[datetime] = Field(None, description="Valid until date")
    uncertainty_percentage: Optional[float] = Field(None, ge=0, le=100)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============================================================================
# Response Models
# ============================================================================


class CalculationResult(BaseModel):
    """Individual calculation result."""

    calculation_id: str = Field(..., description="Unique calculation identifier")
    shipment_id: str = Field(..., description="Shipment identifier")
    calculation_method: str = Field(..., description="Method used")

    # Emissions results
    co2_kg: float = Field(..., description="CO2 emissions in kg")
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")

    # Calculation details
    mode: str = Field(..., description="Transport mode")
    vehicle_type: str = Field(..., description="Vehicle type")
    distance_km: Optional[float] = Field(None, description="Distance traveled")
    cargo_mass_kg: Optional[float] = Field(None, description="Cargo mass")
    emission_factor: float = Field(..., description="Emission factor used")
    emission_factor_unit: str = Field(..., description="EF unit")
    emission_factor_source: str = Field(..., description="EF source")

    # Metadata
    created_at: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    uncertainty_percentage: Optional[float] = Field(None, description="Uncertainty estimate")


class CalculateResponse(BaseModel):
    """Response model for single calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    result: Optional[CalculationResult] = Field(None, description="Calculation result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculations."""

    success: bool = Field(..., description="Overall success flag")
    total_calculations: int = Field(..., description="Total calculations requested")
    successful_calculations: int = Field(..., description="Successful calculations")
    failed_calculations: int = Field(..., description="Failed calculations")
    results: List[CalculateResponse] = Field(..., description="Individual results")
    processing_time_ms: float = Field(..., description="Total processing time in ms")


class CalculationListResponse(BaseModel):
    """Response model for listing calculations."""

    success: bool = Field(..., description="Success flag")
    total_count: int = Field(..., description="Total calculations")
    calculations: List[CalculationResult] = Field(..., description="Calculation list")
    page: int = Field(1, description="Current page")
    page_size: int = Field(100, description="Page size")


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    success: bool = Field(..., description="Success flag")
    calculation: Optional[CalculationResult] = Field(None, description="Calculation details")
    error: Optional[str] = Field(None, description="Error message")


class DeleteResponse(BaseModel):
    """Response model for deletion operations."""

    success: bool = Field(..., description="Success flag")
    deleted_id: str = Field(..., description="Deleted identifier")
    message: Optional[str] = Field(None, description="Status message")


class TransportChainResult(BaseModel):
    """Multi-leg transport chain result."""

    chain_id: str = Field(..., description="Chain identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    total_legs: int = Field(..., description="Number of legs")
    total_distance_km: float = Field(..., description="Total distance")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions")
    leg_results: List[CalculationResult] = Field(..., description="Individual leg results")
    created_at: datetime = Field(..., description="Creation timestamp")
    provenance_hash: str = Field(..., description="Chain provenance hash")


class TransportChainResponse(BaseModel):
    """Response model for transport chain creation."""

    success: bool = Field(..., description="Success flag")
    chain: Optional[TransportChainResult] = Field(None, description="Chain result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class TransportChainListResponse(BaseModel):
    """Response model for listing transport chains."""

    success: bool = Field(..., description="Success flag")
    total_count: int = Field(..., description="Total chains")
    chains: List[TransportChainResult] = Field(..., description="Chain list")
    page: int = Field(1, description="Current page")
    page_size: int = Field(100, description="Page size")


class TransportChainDetailResponse(BaseModel):
    """Response model for single chain detail."""

    success: bool = Field(..., description="Success flag")
    chain: Optional[TransportChainResult] = Field(None, description="Chain details")
    error: Optional[str] = Field(None, description="Error message")


class EmissionFactorDetail(BaseModel):
    """Emission factor detail."""

    factor_id: str = Field(..., description="Factor identifier")
    mode: str = Field(..., description="Transport mode")
    vehicle_type: str = Field(..., description="Vehicle type")
    fuel_type: Optional[str] = Field(None, description="Fuel type")
    co2_factor: float = Field(..., description="CO2 factor")
    ch4_factor: Optional[float] = Field(None, description="CH4 factor")
    n2o_factor: Optional[float] = Field(None, description="N2O factor")
    unit: str = Field(..., description="Factor unit")
    source: str = Field(..., description="Data source")
    valid_from: datetime = Field(..., description="Valid from")
    valid_until: Optional[datetime] = Field(None, description="Valid until")
    uncertainty_percentage: Optional[float] = Field(None)


class EmissionFactorListResponse(BaseModel):
    """Response model for listing emission factors."""

    success: bool = Field(..., description="Success flag")
    total_count: int = Field(..., description="Total factors")
    factors: List[EmissionFactorDetail] = Field(..., description="Factor list")
    page: int = Field(1, description="Current page")
    page_size: int = Field(100, description="Page size")


class EmissionFactorDetailResponse(BaseModel):
    """Response model for single emission factor."""

    success: bool = Field(..., description="Success flag")
    factor: Optional[EmissionFactorDetail] = Field(None, description="Factor details")
    error: Optional[str] = Field(None, description="Error message")


class EmissionFactorResponse(BaseModel):
    """Response model for creating custom emission factor."""

    success: bool = Field(..., description="Success flag")
    factor_id: str = Field(..., description="Created factor ID")
    factor: Optional[EmissionFactorDetail] = Field(None, description="Factor details")
    error: Optional[str] = Field(None, description="Error message")


class ClassificationResult(BaseModel):
    """Shipment classification result."""

    recommended_mode: str = Field(..., description="Recommended transport mode")
    recommended_vehicle_type: str = Field(..., description="Recommended vehicle type")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")
    alternative_options: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative mode/vehicle options"
    )
    reasoning: Optional[str] = Field(None, description="Classification reasoning")


class ClassificationResponse(BaseModel):
    """Response model for classification."""

    success: bool = Field(..., description="Success flag")
    classification: Optional[ClassificationResult] = Field(None, description="Classification result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class ComplianceResult(BaseModel):
    """Compliance check result for a single framework."""

    framework: str = Field(..., description="Framework name")
    compliant: bool = Field(..., description="Compliance status")
    issues: List[str] = Field(default_factory=list, description="Compliance issues")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    check_id: str = Field(..., description="Compliance check identifier")
    calculation_id: str = Field(..., description="Calculation checked")
    overall_compliant: bool = Field(..., description="Overall compliance status")
    framework_results: List[ComplianceResult] = Field(..., description="Per-framework results")
    checked_at: datetime = Field(..., description="Check timestamp")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class ComplianceDetailResponse(BaseModel):
    """Response model for compliance result detail."""

    success: bool = Field(..., description="Success flag")
    compliance: Optional[ComplianceCheckResponse] = Field(None, description="Compliance details")
    error: Optional[str] = Field(None, description="Error message")


class UncertaintyResult(BaseModel):
    """Uncertainty quantification result."""

    calculation_id: str = Field(..., description="Calculation identifier")
    mean_co2e_kg: float = Field(..., description="Mean CO2e emissions")
    median_co2e_kg: float = Field(..., description="Median CO2e emissions")
    std_dev_co2e_kg: float = Field(..., description="Standard deviation")
    p5_co2e_kg: float = Field(..., description="5th percentile")
    p95_co2e_kg: float = Field(..., description="95th percentile")
    uncertainty_percentage: float = Field(..., description="Uncertainty percentage")
    confidence_interval_95: Dict[str, float] = Field(..., description="95% confidence interval")
    monte_carlo_iterations: int = Field(..., description="MC iterations performed")


class UncertaintyResponse(BaseModel):
    """Response model for uncertainty quantification."""

    success: bool = Field(..., description="Success flag")
    uncertainty: Optional[UncertaintyResult] = Field(None, description="Uncertainty result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class AggregationResult(BaseModel):
    """Aggregation result."""

    group_by: str = Field(..., description="Grouping dimension")
    groups: Dict[str, Dict[str, float]] = Field(..., description="Aggregated data by group")
    total_co2e_kg: float = Field(..., description="Total emissions across all groups")
    calculation_count: int = Field(..., description="Total calculations")


class AggregationResponse(BaseModel):
    """Response model for aggregations."""

    success: bool = Field(..., description="Success flag")
    aggregation: Optional[AggregationResult] = Field(None, description="Aggregation result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class HotSpotResult(BaseModel):
    """Hot spot analysis result."""

    hot_spots: List[Dict[str, Any]] = Field(..., description="Hot spots ranked by emissions")
    pareto_threshold_percentage: float = Field(..., description="Pareto threshold (e.g., 80%)")
    top_contributors: List[Dict[str, Any]] = Field(..., description="Top contributors")
    total_co2e_kg: float = Field(..., description="Total emissions")


class HotSpotResponse(BaseModel):
    """Response model for hot spot analysis."""

    success: bool = Field(..., description="Success flag")
    hot_spots: Optional[HotSpotResult] = Field(None, description="Hot spot result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class ExportResponse(BaseModel):
    """Response model for exports."""

    success: bool = Field(..., description="Success flag")
    export_format: str = Field(..., description="Export format (JSON, CSV, XLSX, PDF)")
    export_url: Optional[str] = Field(None, description="Download URL")
    export_data: Optional[Any] = Field(None, description="Inline export data")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    engines_status: Dict[str, str] = Field(..., description="Per-engine status")
    database_connected: bool = Field(..., description="Database connectivity")
    uptime_seconds: float = Field(..., description="Service uptime")


class StatsResponse(BaseModel):
    """Response model for service statistics."""

    total_calculations: int = Field(..., description="Total calculations")
    total_transport_chains: int = Field(..., description="Total transport chains")
    total_custom_factors: int = Field(..., description="Total custom emission factors")
    calculations_by_method: Dict[str, int] = Field(..., description="Calculations by method")
    calculations_by_mode: Dict[str, int] = Field(..., description="Calculations by mode")
    total_co2e_kg: float = Field(..., description="Total emissions calculated")
    avg_processing_time_ms: float = Field(..., description="Average processing time")


# ============================================================================
# UpstreamTransportationService Class
# ============================================================================


class UpstreamTransportationService:
    """
    Upstream Transportation & Distribution Service Facade.

    This service wires together all 7 engines to provide a complete API
    for upstream transportation and distribution emissions calculations
    (Scope 3 Category 4).

    The service supports:
        - Distance-based calculations (Tier 1)
        - Fuel-based calculations (Tier 2)
        - Spend-based calculations (Tier 1)
        - Multi-leg transport chain modeling
        - Emission factor management (6+ authoritative sources)
        - Shipment classification via LLM/rules
        - Compliance checking (7 regulatory frameworks)
        - Uncertainty quantification (Monte Carlo)
        - Aggregations, hot spots, reporting

    Engines:
        1. TransportDatabaseEngine - Data persistence
        2. DistanceBasedCalculatorEngine - Distance-based method
        3. FuelBasedCalculatorEngine - Fuel-based method
        4. SpendBasedCalculatorEngine - Spend-based method
        5. MultiLegCalculatorEngine - Multi-leg orchestration
        6. ComplianceCheckerEngine - Compliance validation
        7. TransportPipelineEngine - End-to-end pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton instance.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(CalculateRequest(
        ...     tenant_id="acme-corp",
        ...     shipment_id="SHP-001",
        ...     mode="ROAD",
        ...     vehicle_type="DIESEL_TRUCK",
        ...     distance_km=500.0,
        ...     cargo_mass_kg=20000.0
        ... ))
        >>> assert response.success
        >>> print(f"Emissions: {response.result.total_co2e_kg} kg CO2e")

    Attributes:
        config: Service configuration
        metrics: Metrics tracker (OBS-001 integration)
        provenance: Provenance tracker (AGENT-FOUND-008)
        transport_db_engine: Database engine
        distance_calc_engine: Distance-based calculator
        fuel_calc_engine: Fuel-based calculator
        spend_calc_engine: Spend-based calculator
        multileg_calc_engine: Multi-leg calculator
        compliance_engine: Compliance checker
        pipeline_engine: Pipeline orchestrator
    """

    def __init__(self):
        """Initialize UpstreamTransportationService."""
        logger.info("Initializing UpstreamTransportationService")

        self._start_time = datetime.now()
        self._initialized = False

        try:
            # Load configuration
            self.config = self._load_config()
            logger.debug(f"Loaded config: {self.config}")

            # Initialize metrics tracker
            self.metrics = self._initialize_metrics()

            # Initialize provenance tracker
            self.provenance = self._initialize_provenance()

            # Initialize all 7 engines
            self._initialize_engines()

            self._initialized = True
            logger.info("UpstreamTransportationService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize UpstreamTransportationService: {e}", exc_info=True)
            raise

    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        try:
            from greenlang.config import get_config
            config = get_config()
            return config.get('upstream_transportation', {})
        except ImportError:
            logger.warning("Config module not available, using defaults")
            return {
                'database_url': 'postgresql://localhost:5432/greenlang',
                'enable_cache': True,
                'cache_ttl_seconds': 3600,
                'monte_carlo_iterations': 10000,
                'parallel_batch_size': 50,
            }

    def _initialize_metrics(self):
        """Initialize metrics tracker."""
        try:
            from greenlang.observability.metrics import get_metrics
            return get_metrics()
        except ImportError:
            logger.warning("Metrics module not available")
            return None

    def _initialize_provenance(self):
        """Initialize provenance tracker."""
        try:
            from greenlang.provenance import get_provenance_tracker
            return get_provenance_tracker()
        except ImportError:
            logger.warning("Provenance module not available")
            return None

    def _initialize_engines(self):
        """Initialize all 7 engines."""
        try:
            # Engine 1: TransportDatabaseEngine
            from greenlang.upstream_transportation.engines.transport_database_engine import (
                TransportDatabaseEngine
            )
            self.transport_db_engine = TransportDatabaseEngine(self.config)
            logger.info("TransportDatabaseEngine initialized")

            # Engine 2: DistanceBasedCalculatorEngine
            from greenlang.upstream_transportation.engines.distance_based_calculator_engine import (
                DistanceBasedCalculatorEngine
            )
            self.distance_calc_engine = DistanceBasedCalculatorEngine(self.config)
            logger.info("DistanceBasedCalculatorEngine initialized")

            # Engine 3: FuelBasedCalculatorEngine
            from greenlang.upstream_transportation.engines.fuel_based_calculator_engine import (
                FuelBasedCalculatorEngine
            )
            self.fuel_calc_engine = FuelBasedCalculatorEngine(self.config)
            logger.info("FuelBasedCalculatorEngine initialized")

            # Engine 4: SpendBasedCalculatorEngine
            from greenlang.upstream_transportation.engines.spend_based_calculator_engine import (
                SpendBasedCalculatorEngine
            )
            self.spend_calc_engine = SpendBasedCalculatorEngine(self.config)
            logger.info("SpendBasedCalculatorEngine initialized")

            # Engine 5: MultiLegCalculatorEngine
            from greenlang.upstream_transportation.engines.multileg_calculator_engine import (
                MultiLegCalculatorEngine
            )
            self.multileg_calc_engine = MultiLegCalculatorEngine(self.config)
            logger.info("MultiLegCalculatorEngine initialized")

            # Engine 6: ComplianceCheckerEngine
            from greenlang.upstream_transportation.engines.compliance_checker_engine import (
                ComplianceCheckerEngine
            )
            self.compliance_engine = ComplianceCheckerEngine(self.config)
            logger.info("ComplianceCheckerEngine initialized")

            # Engine 7: TransportPipelineEngine
            from greenlang.upstream_transportation.engines.transport_pipeline_engine import (
                TransportPipelineEngine
            )
            self.pipeline_engine = TransportPipelineEngine(
                self.config,
                transport_db_engine=self.transport_db_engine,
                distance_calc_engine=self.distance_calc_engine,
                fuel_calc_engine=self.fuel_calc_engine,
                spend_calc_engine=self.spend_calc_engine,
                multileg_calc_engine=self.multileg_calc_engine,
                compliance_engine=self.compliance_engine
            )
            logger.info("TransportPipelineEngine initialized")

        except ImportError as e:
            logger.error(f"Failed to import engine: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}", exc_info=True)
            raise

    # ========================================================================
    # Public API Methods
    # ========================================================================

    def calculate(self, request: CalculateRequest) -> CalculateResponse:
        """
        Calculate emissions for a single shipment.

        This method routes the calculation to the appropriate engine based on
        the calculation_method field (DISTANCE_BASED, FUEL_BASED, SPEND_BASED).

        Args:
            request: Calculation request with shipment details

        Returns:
            CalculateResponse with calculation result and provenance

        Raises:
            ValueError: If request validation fails
            ProcessingError: If calculation fails

        Example:
            >>> response = service.calculate(CalculateRequest(
            ...     tenant_id="acme-corp",
            ...     shipment_id="SHP-001",
            ...     mode="ROAD",
            ...     vehicle_type="DIESEL_TRUCK",
            ...     distance_km=500.0,
            ...     cargo_mass_kg=20000.0
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating emissions for shipment {request.shipment_id}")

        try:
            # Track metrics
            if self.metrics:
                self.metrics.increment('upstream_transportation.calculations.total')
                self.metrics.increment(f'upstream_transportation.calculations.method.{request.calculation_method.lower()}')

            # Execute calculation via pipeline
            result = self.pipeline_engine.execute_calculation(request)

            # Convert to response model
            calculation_result = CalculationResult(
                calculation_id=result['calculation_id'],
                shipment_id=result['shipment_id'],
                calculation_method=result['calculation_method'],
                co2_kg=result['co2_kg'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                mode=result['mode'],
                vehicle_type=result['vehicle_type'],
                distance_km=result.get('distance_km'),
                cargo_mass_kg=result.get('cargo_mass_kg'),
                emission_factor=result['emission_factor'],
                emission_factor_unit=result['emission_factor_unit'],
                emission_factor_source=result['emission_factor_source'],
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash'],
                uncertainty_percentage=result.get('uncertainty_percentage')
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Track success metrics
            if self.metrics:
                self.metrics.increment('upstream_transportation.calculations.success')
                self.metrics.histogram('upstream_transportation.processing_time_ms', processing_time_ms)

            logger.info(f"Calculation {result['calculation_id']} completed in {processing_time_ms:.2f}ms")

            return CalculateResponse(
                success=True,
                calculation_id=result['calculation_id'],
                result=calculation_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.calculations.error')

            return CalculateResponse(
                success=False,
                calculation_id=str(uuid4()),
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_batch(self, request: BatchCalculateRequest) -> BatchCalculateResponse:
        """
        Calculate emissions for multiple shipments in batch.

        Supports parallel execution for improved performance on large batches.

        Args:
            request: Batch calculation request with list of shipments

        Returns:
            BatchCalculateResponse with individual results and summary

        Example:
            >>> response = service.calculate_batch(BatchCalculateRequest(
            ...     tenant_id="acme-corp",
            ...     calculations=[calc1, calc2, calc3],
            ...     parallel=True
            ... ))
            >>> assert response.successful_calculations == 3
        """
        start_time = datetime.now()
        logger.info(f"Batch calculation: {len(request.calculations)} shipments, parallel={request.parallel}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.batch_calculations.total')

            results = []

            if request.parallel:
                # Parallel execution
                from concurrent.futures import ThreadPoolExecutor, as_completed

                batch_size = self.config.get('parallel_batch_size', 50)
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {
                        executor.submit(self.calculate, calc): calc
                        for calc in request.calculations
                    }

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Batch calculation item failed: {e}")
                            results.append(CalculateResponse(
                                success=False,
                                calculation_id=str(uuid4()),
                                error=str(e),
                                processing_time_ms=0
                            ))
            else:
                # Sequential execution
                for calc in request.calculations:
                    result = self.calculate(calc)
                    results.append(result)

            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(f"Batch calculation completed: {successful} succeeded, {failed} failed")

            if self.metrics:
                self.metrics.increment('upstream_transportation.batch_calculations.success')
                self.metrics.histogram('upstream_transportation.batch_processing_time_ms', processing_time_ms)

            return BatchCalculateResponse(
                success=True,
                total_calculations=len(request.calculations),
                successful_calculations=successful,
                failed_calculations=failed,
                results=results,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Batch calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.batch_calculations.error')

            return BatchCalculateResponse(
                success=False,
                total_calculations=len(request.calculations),
                successful_calculations=0,
                failed_calculations=len(request.calculations),
                results=[],
                processing_time_ms=processing_time_ms
            )

    def list_calculations(
        self,
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 100
    ) -> CalculationListResponse:
        """
        List calculations for a tenant with optional filtering.

        Args:
            tenant_id: Tenant identifier
            filters: Optional filters (e.g., {'mode': 'ROAD', 'period': '2024-01'})
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            CalculationListResponse with paginated results

        Example:
            >>> response = service.list_calculations(
            ...     tenant_id="acme-corp",
            ...     filters={'mode': 'ROAD', 'period': '2024-01'},
            ...     page=1,
            ...     page_size=50
            ... )
            >>> assert len(response.calculations) <= 50
        """
        logger.info(f"Listing calculations for tenant {tenant_id}, page={page}, filters={filters}")

        try:
            results = self.transport_db_engine.list_calculations(
                tenant_id=tenant_id,
                filters=filters or {},
                page=page,
                page_size=page_size
            )

            calculations = [
                CalculationResult(**calc) for calc in results['calculations']
            ]

            return CalculationListResponse(
                success=True,
                total_count=results['total_count'],
                calculations=calculations,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"List calculations failed: {e}", exc_info=True)
            return CalculationListResponse(
                success=False,
                total_count=0,
                calculations=[],
                page=page,
                page_size=page_size
            )

    def get_calculation(self, calculation_id: str) -> CalculationDetailResponse:
        """
        Get a single calculation by ID.

        Args:
            calculation_id: Calculation identifier

        Returns:
            CalculationDetailResponse with calculation details

        Example:
            >>> response = service.get_calculation("calc-001")
            >>> assert response.success
            >>> print(response.calculation.total_co2e_kg)
        """
        logger.info(f"Retrieving calculation {calculation_id}")

        try:
            calc = self.transport_db_engine.get_calculation(calculation_id)

            if calc:
                return CalculationDetailResponse(
                    success=True,
                    calculation=CalculationResult(**calc)
                )
            else:
                return CalculationDetailResponse(
                    success=False,
                    error=f"Calculation {calculation_id} not found"
                )

        except Exception as e:
            logger.error(f"Get calculation failed: {e}", exc_info=True)
            return CalculationDetailResponse(
                success=False,
                error=str(e)
            )

    def delete_calculation(self, calculation_id: str) -> DeleteResponse:
        """
        Delete a calculation by ID.

        Args:
            calculation_id: Calculation identifier

        Returns:
            DeleteResponse with status

        Example:
            >>> response = service.delete_calculation("calc-001")
            >>> assert response.success
        """
        logger.info(f"Deleting calculation {calculation_id}")

        try:
            # Track provenance
            if self.provenance:
                self.provenance.track_deletion(
                    entity_type='calculation',
                    entity_id=calculation_id
                )

            self.transport_db_engine.delete_calculation(calculation_id)

            if self.metrics:
                self.metrics.increment('upstream_transportation.calculations.deleted')

            return DeleteResponse(
                success=True,
                deleted_id=calculation_id,
                message="Calculation deleted successfully"
            )

        except Exception as e:
            logger.error(f"Delete calculation failed: {e}", exc_info=True)
            return DeleteResponse(
                success=False,
                deleted_id=calculation_id,
                message=str(e)
            )

    def create_transport_chain(self, chain_data: TransportChainRequest) -> TransportChainResponse:
        """
        Create a multi-leg transport chain.

        This method orchestrates multiple transport legs sequentially and
        aggregates the total emissions.

        Args:
            chain_data: Transport chain request with sequential legs

        Returns:
            TransportChainResponse with aggregated results

        Example:
            >>> response = service.create_transport_chain(TransportChainRequest(
            ...     tenant_id="acme-corp",
            ...     chain_id="CHAIN-001",
            ...     legs=[leg1, leg2, leg3]
            ... ))
            >>> assert response.chain.total_legs == 3
        """
        start_time = datetime.now()
        logger.info(f"Creating transport chain {chain_data.chain_id} with {len(chain_data.legs)} legs")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.transport_chains.total')

            # Execute multi-leg calculation
            result = self.multileg_calc_engine.calculate_transport_chain(chain_data)

            # Convert to response model
            chain_result = TransportChainResult(
                chain_id=result['chain_id'],
                tenant_id=result['tenant_id'],
                total_legs=result['total_legs'],
                total_distance_km=result['total_distance_km'],
                total_co2e_kg=result['total_co2e_kg'],
                leg_results=[CalculationResult(**leg) for leg in result['leg_results']],
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Track provenance
            if self.provenance:
                self.provenance.track_creation(
                    entity_type='transport_chain',
                    entity_id=result['chain_id'],
                    provenance_hash=result['provenance_hash']
                )

            if self.metrics:
                self.metrics.increment('upstream_transportation.transport_chains.success')

            logger.info(f"Transport chain {result['chain_id']} created in {processing_time_ms:.2f}ms")

            return TransportChainResponse(
                success=True,
                chain=chain_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Create transport chain failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.transport_chains.error')

            return TransportChainResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def list_transport_chains(
        self,
        tenant_id: str,
        page: int = 1,
        page_size: int = 100
    ) -> TransportChainListResponse:
        """
        List transport chains for a tenant.

        Args:
            tenant_id: Tenant identifier
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            TransportChainListResponse with paginated results

        Example:
            >>> response = service.list_transport_chains("acme-corp", page=1, page_size=50)
            >>> assert len(response.chains) <= 50
        """
        logger.info(f"Listing transport chains for tenant {tenant_id}, page={page}")

        try:
            results = self.transport_db_engine.list_transport_chains(
                tenant_id=tenant_id,
                page=page,
                page_size=page_size
            )

            chains = [
                TransportChainResult(**chain) for chain in results['chains']
            ]

            return TransportChainListResponse(
                success=True,
                total_count=results['total_count'],
                chains=chains,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"List transport chains failed: {e}", exc_info=True)
            return TransportChainListResponse(
                success=False,
                total_count=0,
                chains=[],
                page=page,
                page_size=page_size
            )

    def get_transport_chain(self, chain_id: str) -> TransportChainDetailResponse:
        """
        Get a single transport chain by ID.

        Args:
            chain_id: Chain identifier

        Returns:
            TransportChainDetailResponse with chain details

        Example:
            >>> response = service.get_transport_chain("CHAIN-001")
            >>> assert response.success
        """
        logger.info(f"Retrieving transport chain {chain_id}")

        try:
            chain = self.transport_db_engine.get_transport_chain(chain_id)

            if chain:
                return TransportChainDetailResponse(
                    success=True,
                    chain=TransportChainResult(**chain)
                )
            else:
                return TransportChainDetailResponse(
                    success=False,
                    error=f"Transport chain {chain_id} not found"
                )

        except Exception as e:
            logger.error(f"Get transport chain failed: {e}", exc_info=True)
            return TransportChainDetailResponse(
                success=False,
                error=str(e)
            )

    def get_emission_factors(
        self,
        mode: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 100
    ) -> EmissionFactorListResponse:
        """
        List emission factors with optional filtering.

        Args:
            mode: Filter by transport mode (e.g., 'ROAD', 'RAIL', 'AIR', 'SEA')
            vehicle_type: Filter by vehicle type
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            EmissionFactorListResponse with paginated factors

        Example:
            >>> response = service.get_emission_factors(mode='ROAD', page=1, page_size=50)
            >>> assert all(f.mode == 'ROAD' for f in response.factors)
        """
        logger.info(f"Listing emission factors: mode={mode}, vehicle_type={vehicle_type}")

        try:
            filters = {}
            if mode:
                filters['mode'] = mode
            if vehicle_type:
                filters['vehicle_type'] = vehicle_type

            results = self.transport_db_engine.list_emission_factors(
                filters=filters,
                page=page,
                page_size=page_size
            )

            factors = [
                EmissionFactorDetail(**ef) for ef in results['factors']
            ]

            return EmissionFactorListResponse(
                success=True,
                total_count=results['total_count'],
                factors=factors,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"List emission factors failed: {e}", exc_info=True)
            return EmissionFactorListResponse(
                success=False,
                total_count=0,
                factors=[],
                page=page,
                page_size=page_size
            )

    def get_emission_factor(self, factor_id: str) -> EmissionFactorDetailResponse:
        """
        Get a single emission factor by ID.

        Args:
            factor_id: Factor identifier

        Returns:
            EmissionFactorDetailResponse with factor details

        Example:
            >>> response = service.get_emission_factor("EF-001")
            >>> assert response.success
        """
        logger.info(f"Retrieving emission factor {factor_id}")

        try:
            factor = self.transport_db_engine.get_emission_factor(factor_id)

            if factor:
                return EmissionFactorDetailResponse(
                    success=True,
                    factor=EmissionFactorDetail(**factor)
                )
            else:
                return EmissionFactorDetailResponse(
                    success=False,
                    error=f"Emission factor {factor_id} not found"
                )

        except Exception as e:
            logger.error(f"Get emission factor failed: {e}", exc_info=True)
            return EmissionFactorDetailResponse(
                success=False,
                error=str(e)
            )

    def create_custom_emission_factor(
        self,
        ef_data: CustomEmissionFactorRequest
    ) -> EmissionFactorResponse:
        """
        Create a custom emission factor.

        Allows tenants to define custom emission factors when authoritative
        sources do not cover their specific use case.

        Args:
            ef_data: Custom emission factor data

        Returns:
            EmissionFactorResponse with created factor

        Example:
            >>> response = service.create_custom_emission_factor(
            ...     CustomEmissionFactorRequest(
            ...         tenant_id="acme-corp",
            ...         mode="ROAD",
            ...         vehicle_type="ELECTRIC_TRUCK",
            ...         co2_factor=0.05,
            ...         unit="kg_CO2e/tkm",
            ...         source="Internal testing",
            ...         valid_from=datetime.now()
            ...     )
            ... )
            >>> assert response.success
        """
        logger.info(f"Creating custom emission factor for {ef_data.tenant_id}")

        try:
            factor_id = str(uuid4())

            # Track provenance
            if self.provenance:
                provenance_hash = self.provenance.track_creation(
                    entity_type='emission_factor',
                    entity_id=factor_id,
                    data=ef_data.dict()
                )
            else:
                import hashlib
                provenance_hash = hashlib.sha256(
                    ef_data.json().encode()
                ).hexdigest()

            # Store in database
            factor_data = {
                'factor_id': factor_id,
                'tenant_id': ef_data.tenant_id,
                'mode': ef_data.mode,
                'vehicle_type': ef_data.vehicle_type,
                'fuel_type': ef_data.fuel_type,
                'co2_factor': ef_data.co2_factor,
                'ch4_factor': ef_data.ch4_factor,
                'n2o_factor': ef_data.n2o_factor,
                'unit': ef_data.unit,
                'source': ef_data.source,
                'valid_from': ef_data.valid_from,
                'valid_until': ef_data.valid_until,
                'uncertainty_percentage': ef_data.uncertainty_percentage,
                'metadata': ef_data.metadata,
                'provenance_hash': provenance_hash,
                'created_at': datetime.now()
            }

            self.transport_db_engine.create_custom_emission_factor(factor_data)

            if self.metrics:
                self.metrics.increment('upstream_transportation.custom_factors.created')

            logger.info(f"Custom emission factor {factor_id} created")

            return EmissionFactorResponse(
                success=True,
                factor_id=factor_id,
                factor=EmissionFactorDetail(**factor_data)
            )

        except Exception as e:
            logger.error(f"Create custom emission factor failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.custom_factors.error')

            return EmissionFactorResponse(
                success=False,
                factor_id="",
                error=str(e)
            )

    def classify_shipment(self, shipment_data: ClassificationRequest) -> ClassificationResponse:
        """
        Classify a shipment to recommend optimal transport mode and vehicle type.

        Uses rule-based logic and optional LLM classification to recommend
        the most appropriate transport mode and vehicle type based on shipment
        characteristics (cargo type, distance, origin/destination).

        Args:
            shipment_data: Shipment classification request

        Returns:
            ClassificationResponse with recommendations

        Example:
            >>> response = service.classify_shipment(ClassificationRequest(
            ...     tenant_id="acme-corp",
            ...     shipment_description="Bulk grain shipment",
            ...     distance_km=5000,
            ...     origin_country="US",
            ...     destination_country="CN"
            ... ))
            >>> assert response.classification.recommended_mode in ['SEA', 'AIR']
        """
        start_time = datetime.now()
        logger.info(f"Classifying shipment for tenant {shipment_data.tenant_id}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.classifications.total')

            # Use pipeline engine's classification logic
            result = self.pipeline_engine.classify_shipment(shipment_data)

            classification = ClassificationResult(
                recommended_mode=result['recommended_mode'],
                recommended_vehicle_type=result['recommended_vehicle_type'],
                confidence_score=result['confidence_score'],
                alternative_options=result.get('alternative_options', []),
                reasoning=result.get('reasoning')
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('upstream_transportation.classifications.success')

            logger.info(f"Classification completed: {result['recommended_mode']} / {result['recommended_vehicle_type']}")

            return ClassificationResponse(
                success=True,
                classification=classification,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Classification failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.classifications.error')

            return ClassificationResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def check_compliance(
        self,
        calculation_id: str,
        frameworks: List[str]
    ) -> ComplianceCheckResponse:
        """
        Check calculation compliance against regulatory frameworks.

        Validates a calculation against one or more regulatory frameworks:
        - GHG_PROTOCOL
        - ISO_14083
        - GLEC
        - CSRD
        - CDP
        - SBTi
        - EU_ETS

        Args:
            calculation_id: Calculation to validate
            frameworks: List of frameworks to check

        Returns:
            ComplianceCheckResponse with per-framework results

        Example:
            >>> response = service.check_compliance(
            ...     calculation_id="calc-001",
            ...     frameworks=["GHG_PROTOCOL", "CSRD"]
            ... )
            >>> assert response.overall_compliant
        """
        start_time = datetime.now()
        logger.info(f"Checking compliance for {calculation_id} against {frameworks}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.compliance_checks.total')

            # Retrieve calculation
            calc = self.transport_db_engine.get_calculation(calculation_id)
            if not calc:
                raise ValueError(f"Calculation {calculation_id} not found")

            # Run compliance check
            check_id = str(uuid4())
            framework_results = []

            for framework in frameworks:
                result = self.compliance_engine.check_framework(framework, calc)
                framework_results.append(ComplianceResult(
                    framework=framework,
                    compliant=result['compliant'],
                    issues=result.get('issues', []),
                    warnings=result.get('warnings', []),
                    recommendations=result.get('recommendations', [])
                ))

            overall_compliant = all(r.compliant for r in framework_results)
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('upstream_transportation.compliance_checks.success')
                if overall_compliant:
                    self.metrics.increment('upstream_transportation.compliance_checks.compliant')
                else:
                    self.metrics.increment('upstream_transportation.compliance_checks.non_compliant')

            logger.info(f"Compliance check {check_id} completed: overall_compliant={overall_compliant}")

            return ComplianceCheckResponse(
                success=True,
                check_id=check_id,
                calculation_id=calculation_id,
                overall_compliant=overall_compliant,
                framework_results=framework_results,
                checked_at=datetime.now(),
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Compliance check failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.compliance_checks.error')

            # Return minimal response on error
            return ComplianceCheckResponse(
                success=False,
                check_id=str(uuid4()),
                calculation_id=calculation_id,
                overall_compliant=False,
                framework_results=[],
                checked_at=datetime.now(),
                processing_time_ms=processing_time_ms
            )

    def get_compliance_result(self, check_id: str) -> ComplianceDetailResponse:
        """
        Get a compliance check result by ID.

        Args:
            check_id: Compliance check identifier

        Returns:
            ComplianceDetailResponse with check details

        Example:
            >>> response = service.get_compliance_result("check-001")
            >>> assert response.success
        """
        logger.info(f"Retrieving compliance result {check_id}")

        try:
            result = self.transport_db_engine.get_compliance_result(check_id)

            if result:
                return ComplianceDetailResponse(
                    success=True,
                    compliance=ComplianceCheckResponse(**result)
                )
            else:
                return ComplianceDetailResponse(
                    success=False,
                    error=f"Compliance result {check_id} not found"
                )

        except Exception as e:
            logger.error(f"Get compliance result failed: {e}", exc_info=True)
            return ComplianceDetailResponse(
                success=False,
                error=str(e)
            )

    def calculate_uncertainty(
        self,
        calculation_id: str,
        iterations: Optional[int] = None
    ) -> UncertaintyResponse:
        """
        Quantify uncertainty for a calculation using Monte Carlo simulation.

        Performs Monte Carlo simulation to estimate uncertainty in emissions
        calculations by varying input parameters within their uncertainty ranges.

        Args:
            calculation_id: Calculation to analyze
            iterations: Number of Monte Carlo iterations (default: 10000)

        Returns:
            UncertaintyResponse with statistical results

        Example:
            >>> response = service.calculate_uncertainty("calc-001", iterations=10000)
            >>> assert response.uncertainty.uncertainty_percentage > 0
        """
        start_time = datetime.now()
        logger.info(f"Calculating uncertainty for {calculation_id}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.uncertainty.total')

            # Retrieve calculation
            calc = self.transport_db_engine.get_calculation(calculation_id)
            if not calc:
                raise ValueError(f"Calculation {calculation_id} not found")

            # Run Monte Carlo simulation
            mc_iterations = iterations or self.config.get('monte_carlo_iterations', 10000)
            result = self.pipeline_engine.calculate_uncertainty(calc, mc_iterations)

            uncertainty = UncertaintyResult(
                calculation_id=calculation_id,
                mean_co2e_kg=result['mean_co2e_kg'],
                median_co2e_kg=result['median_co2e_kg'],
                std_dev_co2e_kg=result['std_dev_co2e_kg'],
                p5_co2e_kg=result['p5_co2e_kg'],
                p95_co2e_kg=result['p95_co2e_kg'],
                uncertainty_percentage=result['uncertainty_percentage'],
                confidence_interval_95=result['confidence_interval_95'],
                monte_carlo_iterations=mc_iterations
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('upstream_transportation.uncertainty.success')

            logger.info(f"Uncertainty calculation completed: {result['uncertainty_percentage']:.2f}%")

            return UncertaintyResponse(
                success=True,
                uncertainty=uncertainty,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Uncertainty calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.uncertainty.error')

            return UncertaintyResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def get_aggregations(
        self,
        tenant_id: str,
        group_by: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> AggregationResponse:
        """
        Get aggregated emissions by a dimension.

        Aggregates emissions across calculations grouped by a specified dimension:
        - mode: Transport mode (ROAD, RAIL, AIR, SEA)
        - vehicle_type: Vehicle type
        - carrier: Carrier name
        - origin: Origin location
        - destination: Destination location
        - period: Reporting period (YYYY-MM)

        Args:
            tenant_id: Tenant identifier
            group_by: Dimension to group by
            filters: Optional filters to apply before aggregation

        Returns:
            AggregationResponse with aggregated data

        Example:
            >>> response = service.get_aggregations(
            ...     tenant_id="acme-corp",
            ...     group_by="mode",
            ...     filters={'period': '2024-01'}
            ... )
            >>> assert 'ROAD' in response.aggregation.groups
        """
        start_time = datetime.now()
        logger.info(f"Getting aggregations for {tenant_id}, group_by={group_by}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.aggregations.total')

            result = self.transport_db_engine.get_aggregations(
                tenant_id=tenant_id,
                group_by=group_by,
                filters=filters or {}
            )

            aggregation = AggregationResult(
                group_by=group_by,
                groups=result['groups'],
                total_co2e_kg=result['total_co2e_kg'],
                calculation_count=result['calculation_count']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('upstream_transportation.aggregations.success')

            return AggregationResponse(
                success=True,
                aggregation=aggregation,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Aggregation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.aggregations.error')

            return AggregationResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def get_hot_spots(
        self,
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> HotSpotResponse:
        """
        Identify emissions hot spots using Pareto analysis.

        Identifies the top contributors to emissions (e.g., modes, carriers, routes)
        that account for 80% of total emissions (Pareto principle).

        Args:
            tenant_id: Tenant identifier
            filters: Optional filters to apply

        Returns:
            HotSpotResponse with ranked hot spots

        Example:
            >>> response = service.get_hot_spots("acme-corp")
            >>> assert len(response.hot_spots.top_contributors) > 0
        """
        start_time = datetime.now()
        logger.info(f"Analyzing hot spots for {tenant_id}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.hot_spots.total')

            result = self.transport_db_engine.get_hot_spots(
                tenant_id=tenant_id,
                filters=filters or {}
            )

            hot_spots = HotSpotResult(
                hot_spots=result['hot_spots'],
                pareto_threshold_percentage=result['pareto_threshold_percentage'],
                top_contributors=result['top_contributors'],
                total_co2e_kg=result['total_co2e_kg']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('upstream_transportation.hot_spots.success')

            logger.info(f"Hot spot analysis completed: {len(result['top_contributors'])} top contributors")

            return HotSpotResponse(
                success=True,
                hot_spots=hot_spots,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Hot spot analysis failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.hot_spots.error')

            return HotSpotResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def export_report(
        self,
        calculation_ids: List[str],
        export_format: str = "JSON"
    ) -> ExportResponse:
        """
        Export calculation results in various formats.

        Supported formats:
        - JSON: Machine-readable JSON
        - CSV: Tabular CSV
        - XLSX: Excel workbook
        - PDF: Formatted PDF report

        Args:
            calculation_ids: List of calculation IDs to export
            export_format: Export format (JSON, CSV, XLSX, PDF)

        Returns:
            ExportResponse with export data or download URL

        Example:
            >>> response = service.export_report(
            ...     calculation_ids=["calc-001", "calc-002"],
            ...     export_format="CSV"
            ... )
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Exporting {len(calculation_ids)} calculations as {export_format}")

        try:
            if self.metrics:
                self.metrics.increment('upstream_transportation.exports.total')
                self.metrics.increment(f'upstream_transportation.exports.format.{export_format.lower()}')

            # Retrieve calculations
            calculations = [
                self.transport_db_engine.get_calculation(calc_id)
                for calc_id in calculation_ids
            ]
            calculations = [c for c in calculations if c is not None]

            # Generate export
            result = self.pipeline_engine.export_report(calculations, export_format)

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('upstream_transportation.exports.success')

            return ExportResponse(
                success=True,
                export_format=export_format,
                export_url=result.get('export_url'),
                export_data=result.get('export_data'),
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Export failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('upstream_transportation.exports.error')

            return ExportResponse(
                success=False,
                export_format=export_format,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def health_check(self) -> HealthResponse:
        """
        Check service health.

        Returns health status of the service and all engines.

        Returns:
            HealthResponse with service status

        Example:
            >>> response = service.health_check()
            >>> assert response.status in ['healthy', 'degraded', 'unhealthy']
        """
        logger.debug("Running health check")

        try:
            engines_status = {
                'transport_db_engine': self._check_engine_health(self.transport_db_engine),
                'distance_calc_engine': self._check_engine_health(self.distance_calc_engine),
                'fuel_calc_engine': self._check_engine_health(self.fuel_calc_engine),
                'spend_calc_engine': self._check_engine_health(self.spend_calc_engine),
                'multileg_calc_engine': self._check_engine_health(self.multileg_calc_engine),
                'compliance_engine': self._check_engine_health(self.compliance_engine),
                'pipeline_engine': self._check_engine_health(self.pipeline_engine),
            }

            database_connected = self.transport_db_engine.check_connection()

            # Determine overall status
            if all(s == 'healthy' for s in engines_status.values()) and database_connected:
                status = 'healthy'
            elif any(s == 'unhealthy' for s in engines_status.values()) or not database_connected:
                status = 'unhealthy'
            else:
                status = 'degraded'

            uptime = (datetime.now() - self._start_time).total_seconds()

            return HealthResponse(
                status=status,
                version='1.0.0',
                engines_status=engines_status,
                database_connected=database_connected,
                uptime_seconds=uptime
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return HealthResponse(
                status='unhealthy',
                version='1.0.0',
                engines_status={},
                database_connected=False,
                uptime_seconds=0
            )

    def get_stats(self, tenant_id: Optional[str] = None) -> StatsResponse:
        """
        Get service statistics.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            StatsResponse with service statistics

        Example:
            >>> response = service.get_stats(tenant_id="acme-corp")
            >>> print(f"Total calculations: {response.total_calculations}")
        """
        logger.debug(f"Getting stats for tenant {tenant_id}")

        try:
            stats = self.transport_db_engine.get_stats(tenant_id)

            return StatsResponse(
                total_calculations=stats['total_calculations'],
                total_transport_chains=stats['total_transport_chains'],
                total_custom_factors=stats['total_custom_factors'],
                calculations_by_method=stats['calculations_by_method'],
                calculations_by_mode=stats['calculations_by_mode'],
                total_co2e_kg=stats['total_co2e_kg'],
                avg_processing_time_ms=stats['avg_processing_time_ms']
            )

        except Exception as e:
            logger.error(f"Get stats failed: {e}", exc_info=True)
            return StatsResponse(
                total_calculations=0,
                total_transport_chains=0,
                total_custom_factors=0,
                calculations_by_method={},
                calculations_by_mode={},
                total_co2e_kg=0.0,
                avg_processing_time_ms=0.0
            )

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _check_engine_health(self, engine) -> str:
        """
        Check health of a single engine.

        Args:
            engine: Engine instance

        Returns:
            Health status: 'healthy', 'degraded', 'unhealthy'
        """
        try:
            if hasattr(engine, 'health_check'):
                return engine.health_check()
            else:
                return 'healthy' if engine else 'unhealthy'
        except Exception:
            return 'unhealthy'


# ============================================================================
# Module-Level Functions
# ============================================================================


def get_service() -> UpstreamTransportationService:
    """
    Get singleton UpstreamTransportationService instance.

    Thread-safe singleton pattern ensures only one instance exists.

    Returns:
        UpstreamTransportationService instance

    Example:
        >>> service = get_service()
        >>> response = service.health_check()
        >>> assert response.status == 'healthy'
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            # Double-checked locking
            if _service_instance is None:
                _service_instance = UpstreamTransportationService()

    return _service_instance


def get_router():
    """
    Get FastAPI router for upstream transportation API.

    Returns:
        APIRouter instance with all upstream transportation routes

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> router = get_router()
        >>> app.include_router(router, prefix="/api/v1/upstream-transportation")
    """
    try:
        from greenlang.upstream_transportation.api.router import router
        return router
    except ImportError as e:
        logger.error(f"Failed to import router: {e}")
        raise


def configure_upstream_transportation(app) -> None:
    """
    Configure upstream transportation service with FastAPI app.

    Registers routes, middleware, and initializes the service.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> from greenlang.upstream_transportation.setup import configure_upstream_transportation
        >>> app = FastAPI()
        >>> configure_upstream_transportation(app)
    """
    logger.info("Configuring upstream transportation service")

    try:
        # Get router
        router = get_router()

        # Include router with prefix
        app.include_router(
            router,
            prefix="/api/v1/upstream-transportation",
            tags=["upstream-transportation"]
        )

        # Initialize service (singleton)
        service = get_service()

        logger.info(f"Upstream transportation service configured: {service._initialized}")

    except Exception as e:
        logger.error(f"Failed to configure upstream transportation service: {e}", exc_info=True)
        raise


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Service
    'UpstreamTransportationService',
    'get_service',
    'get_router',
    'configure_upstream_transportation',

    # Request Models
    'CalculateRequest',
    'BatchCalculateRequest',
    'TransportChainRequest',
    'ClassificationRequest',
    'ComplianceCheckRequest',
    'CustomEmissionFactorRequest',

    # Response Models
    'CalculateResponse',
    'BatchCalculateResponse',
    'CalculationListResponse',
    'CalculationDetailResponse',
    'DeleteResponse',
    'TransportChainResponse',
    'TransportChainListResponse',
    'TransportChainDetailResponse',
    'EmissionFactorListResponse',
    'EmissionFactorDetailResponse',
    'EmissionFactorResponse',
    'ClassificationResponse',
    'ComplianceCheckResponse',
    'ComplianceDetailResponse',
    'UncertaintyResponse',
    'AggregationResponse',
    'HotSpotResponse',
    'ExportResponse',
    'HealthResponse',
    'StatsResponse',

    # Result Models
    'CalculationResult',
    'TransportChainResult',
    'EmissionFactorDetail',
    'ClassificationResult',
    'ComplianceResult',
    'UncertaintyResult',
    'AggregationResult',
    'HotSpotResult',
]
