"""
Waste Generated in Operations Service Setup - AGENT-MRV-018

This module provides the service facade that wires together all 7 engines
for waste generated in operations emissions calculations (Scope 3 Category 5).

The WasteGeneratedService class provides a high-level API for:
- Landfill, incineration, recycling, composting, and anaerobic digestion calculations
- Wastewater treatment emissions
- Multi-waste type classification (12 waste types)
- Emission factor management with 7+ authoritative sources
- Treatment method optimization and diversion analysis
- Compliance checking across 7 regulatory frameworks
- Uncertainty quantification via Monte Carlo simulation
- Aggregations, diversion metrics, and reporting

Engines:
    1. WasteClassificationDatabaseEngine - Data access and persistence
    2. LandfillEmissionsEngine - Landfill emissions with methane modeling
    3. IncinerationEmissionsEngine - Incineration emissions (with/without energy recovery)
    4. RecyclingCompostingEngine - Recycling, composting, anaerobic digestion
    5. WastewaterEmissionsEngine - Wastewater treatment emissions
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. WasteGeneratedPipelineEngine - End-to-end calculation pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.agents.mrv.waste_generated.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate_landfill(LandfillRequest(
    ...     tenant_id="acme-corp",
    ...     waste_id="WASTE-2024-001",
    ...     waste_type="FOOD_WASTE",
    ...     amount_kg=5000.0,
    ...     landfill_type="MANAGED_MSW",
    ...     has_methane_recovery=False
    ... ))
    >>> assert response.success
    >>> assert response.ch4_emissions_kg > 0

Integration:
    >>> from greenlang.agents.mrv.waste_generated.setup import configure_waste_generated
    >>> app = FastAPI()
    >>> configure_waste_generated(app)  # Registers routes, middleware
"""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional['WasteGeneratedService'] = None

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class WasteCalculationRequest(BaseModel):
    """Request model for single waste emissions calculation."""

    tenant_id: str = Field(..., description="Tenant identifier")
    waste_id: str = Field(..., description="Unique waste identifier")
    waste_type: str = Field(..., description="Waste type (FOOD_WASTE, PAPER, PLASTICS, etc.)")
    treatment_method: str = Field(
        ...,
        description="Treatment method: LANDFILL, INCINERATION, RECYCLING, COMPOSTING, ANAEROBIC_DIGESTION, WASTEWATER"
    )

    # Common fields
    amount_kg: float = Field(..., ge=0, description="Waste amount in kg")

    # Landfill-specific fields
    landfill_type: Optional[str] = Field(None, description="Landfill type (MANAGED_MSW, UNMANAGED, etc.)")
    has_methane_recovery: Optional[bool] = Field(False, description="Methane recovery system present")
    methane_recovery_rate: Optional[float] = Field(None, ge=0, le=1, description="Methane recovery rate (0-1)")

    # Incineration-specific fields
    incineration_type: Optional[str] = Field(None, description="Incineration type (WITH_ENERGY, WITHOUT_ENERGY)")
    energy_recovered_kwh: Optional[float] = Field(None, ge=0, description="Energy recovered in kWh")

    # Wastewater-specific fields
    bod_mg_per_liter: Optional[float] = Field(None, ge=0, description="BOD concentration (mg/L)")
    cod_mg_per_liter: Optional[float] = Field(None, ge=0, description="COD concentration (mg/L)")
    wastewater_volume_liters: Optional[float] = Field(None, ge=0, description="Wastewater volume in liters")
    treatment_system: Optional[str] = Field(None, description="Treatment system type")

    # Optional metadata
    waste_source: Optional[str] = Field(None, description="Source of waste")
    disposal_facility: Optional[str] = Field(None, description="Disposal facility name")
    reporting_period: Optional[str] = Field(None, description="Reporting period (YYYY-MM)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('treatment_method')
    def validate_treatment_method(cls, v):
        """Validate treatment method."""
        allowed = ['LANDFILL', 'INCINERATION', 'RECYCLING', 'COMPOSTING', 'ANAEROBIC_DIGESTION', 'WASTEWATER']
        if v not in allowed:
            raise ValueError(f"treatment_method must be one of {allowed}")
        return v


class BatchWasteCalculationRequest(BaseModel):
    """Request model for batch waste calculations."""

    tenant_id: str = Field(..., description="Tenant identifier")
    calculations: List[WasteCalculationRequest] = Field(..., description="List of waste calculations")
    parallel: bool = Field(True, description="Execute in parallel")


class LandfillRequest(BaseModel):
    """Request model for landfill emissions calculation."""

    tenant_id: str = Field(..., description="Tenant identifier")
    waste_id: str = Field(..., description="Unique waste identifier")
    waste_type: str = Field(..., description="Waste type")
    amount_kg: float = Field(..., ge=0, description="Waste amount in kg")
    landfill_type: str = Field(..., description="Landfill type (MANAGED_MSW, UNMANAGED, DEEP, SHALLOW)")
    has_methane_recovery: bool = Field(False, description="Methane recovery system present")
    methane_recovery_rate: Optional[float] = Field(None, ge=0, le=1, description="Methane recovery rate (0-1)")
    climate_zone: Optional[str] = Field(None, description="Climate zone (BOREAL, TEMPERATE, TROPICAL)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IncinerationRequest(BaseModel):
    """Request model for incineration emissions calculation."""

    tenant_id: str = Field(..., description="Tenant identifier")
    waste_id: str = Field(..., description="Unique waste identifier")
    waste_type: str = Field(..., description="Waste type")
    amount_kg: float = Field(..., ge=0, description="Waste amount in kg")
    incineration_type: str = Field(..., description="Incineration type (WITH_ENERGY, WITHOUT_ENERGY)")
    energy_recovered_kwh: Optional[float] = Field(None, ge=0, description="Energy recovered in kWh")
    facility_efficiency: Optional[float] = Field(None, ge=0, le=1, description="Facility efficiency (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RecyclingCompostingRequest(BaseModel):
    """Request model for recycling/composting/anaerobic digestion emissions."""

    tenant_id: str = Field(..., description="Tenant identifier")
    waste_id: str = Field(..., description="Unique waste identifier")
    waste_type: str = Field(..., description="Waste type")
    amount_kg: float = Field(..., ge=0, description="Waste amount in kg")
    treatment_method: str = Field(..., description="RECYCLING, COMPOSTING, or ANAEROBIC_DIGESTION")
    material_recovery_rate: Optional[float] = Field(None, ge=0, le=1, description="Material recovery rate (0-1)")
    biogas_captured: Optional[bool] = Field(False, description="Biogas captured (for anaerobic digestion)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WastewaterRequest(BaseModel):
    """Request model for wastewater emissions calculation."""

    tenant_id: str = Field(..., description="Tenant identifier")
    waste_id: str = Field(..., description="Unique waste identifier")
    wastewater_volume_liters: float = Field(..., ge=0, description="Wastewater volume in liters")
    bod_mg_per_liter: Optional[float] = Field(None, ge=0, description="BOD concentration (mg/L)")
    cod_mg_per_liter: Optional[float] = Field(None, ge=0, description="COD concentration (mg/L)")
    treatment_system: str = Field(..., description="Treatment system type (AEROBIC, ANAEROBIC, LAGOON, etc.)")
    has_biogas_recovery: bool = Field(False, description="Biogas recovery system present")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    tenant_id: str = Field(..., description="Tenant identifier")
    calculation_id: str = Field(..., description="Calculation to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description="Frameworks: GHG_PROTOCOL, ISO_14064, CSRD, CDP, SBTi, EU_ETS, TCFD"
    )


class DiversionAnalysisRequest(BaseModel):
    """Request model for waste diversion analysis."""

    tenant_id: str = Field(..., description="Tenant identifier")
    reporting_period: str = Field(..., description="Reporting period (YYYY-MM)")
    baseline_period: Optional[str] = Field(None, description="Baseline period for comparison (YYYY-MM)")
    target_diversion_rate: Optional[float] = Field(None, ge=0, le=1, description="Target diversion rate (0-1)")


# ============================================================================
# Response Models
# ============================================================================


class WasteCalculationResult(BaseModel):
    """Individual waste calculation result."""

    calculation_id: str = Field(..., description="Unique calculation identifier")
    waste_id: str = Field(..., description="Waste identifier")
    waste_type: str = Field(..., description="Waste type")
    treatment_method: str = Field(..., description="Treatment method")

    # Emissions results
    co2_kg: float = Field(..., description="CO2 emissions in kg")
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")

    # Calculation details
    amount_kg: float = Field(..., description="Waste amount in kg")
    emission_factor: float = Field(..., description="Emission factor used")
    emission_factor_unit: str = Field(..., description="EF unit")
    emission_factor_source: str = Field(..., description="EF source")

    # Metadata
    created_at: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    uncertainty_percentage: Optional[float] = Field(None, description="Uncertainty estimate")


class WasteCalculateResponse(BaseModel):
    """Response model for single waste calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    result: Optional[WasteCalculationResult] = Field(None, description="Calculation result")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class WasteBatchResponse(BaseModel):
    """Response model for batch waste calculations."""

    success: bool = Field(..., description="Overall success flag")
    total_calculations: int = Field(..., description="Total calculations requested")
    successful_calculations: int = Field(..., description="Successful calculations")
    failed_calculations: int = Field(..., description="Failed calculations")
    results: List[WasteCalculateResponse] = Field(..., description="Individual results")
    processing_time_ms: float = Field(..., description="Total processing time in ms")


class LandfillEmissionsResult(BaseModel):
    """Landfill-specific emissions result."""

    calculation_id: str = Field(..., description="Calculation identifier")
    waste_id: str = Field(..., description="Waste identifier")
    waste_type: str = Field(..., description="Waste type")
    amount_kg: float = Field(..., description="Waste amount in kg")
    landfill_type: str = Field(..., description="Landfill type")

    # Methane modeling results
    ch4_generated_kg: float = Field(..., description="Total CH4 generated in kg")
    ch4_recovered_kg: float = Field(..., description="CH4 recovered in kg")
    ch4_emissions_kg: float = Field(..., description="Net CH4 emissions in kg")
    co2_biogenic_kg: float = Field(..., description="Biogenic CO2 in kg")
    co2_fossil_kg: float = Field(..., description="Fossil CO2 in kg")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")

    # Modeling parameters
    degradable_organic_carbon: float = Field(..., description="DOC content")
    methane_correction_factor: float = Field(..., description="MCF")
    methane_generation_rate: float = Field(..., description="k value")

    created_at: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="Provenance hash")


class LandfillResponse(BaseModel):
    """Response model for landfill calculation."""

    success: bool = Field(..., description="Success flag")
    result: Optional[LandfillEmissionsResult] = Field(None, description="Landfill result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class IncinerationEmissionsResult(BaseModel):
    """Incineration-specific emissions result."""

    calculation_id: str = Field(..., description="Calculation identifier")
    waste_id: str = Field(..., description="Waste identifier")
    waste_type: str = Field(..., description="Waste type")
    amount_kg: float = Field(..., description="Waste amount in kg")
    incineration_type: str = Field(..., description="Incineration type")

    # Emissions results
    co2_fossil_kg: float = Field(..., description="Fossil CO2 in kg")
    co2_biogenic_kg: float = Field(..., description="Biogenic CO2 in kg")
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")

    # Energy recovery
    energy_recovered_kwh: Optional[float] = Field(None, description="Energy recovered in kWh")
    avoided_emissions_kg: Optional[float] = Field(None, description="Avoided emissions from energy recovery")

    created_at: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="Provenance hash")


class IncinerationResponse(BaseModel):
    """Response model for incineration calculation."""

    success: bool = Field(..., description="Success flag")
    result: Optional[IncinerationEmissionsResult] = Field(None, description="Incineration result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class RecyclingCompostingResult(BaseModel):
    """Recycling/composting/anaerobic digestion result."""

    calculation_id: str = Field(..., description="Calculation identifier")
    waste_id: str = Field(..., description="Waste identifier")
    waste_type: str = Field(..., description="Waste type")
    treatment_method: str = Field(..., description="Treatment method")
    amount_kg: float = Field(..., description="Waste amount in kg")

    # Emissions results
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")

    # Benefits
    avoided_emissions_kg: Optional[float] = Field(None, description="Avoided emissions from recycling")
    net_emissions_kg: float = Field(..., description="Net emissions (emissions - avoided)")

    created_at: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="Provenance hash")


class RecyclingCompostingResponse(BaseModel):
    """Response model for recycling/composting calculation."""

    success: bool = Field(..., description="Success flag")
    result: Optional[RecyclingCompostingResult] = Field(None, description="Result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class WastewaterEmissionsResult(BaseModel):
    """Wastewater-specific emissions result."""

    calculation_id: str = Field(..., description="Calculation identifier")
    waste_id: str = Field(..., description="Waste identifier")
    wastewater_volume_liters: float = Field(..., description="Wastewater volume in liters")
    treatment_system: str = Field(..., description="Treatment system type")

    # Emissions results
    ch4_kg: float = Field(..., description="CH4 emissions in kg")
    n2o_kg: float = Field(..., description="N2O emissions in kg")
    total_co2e_kg: float = Field(..., description="Total CO2e emissions in kg")

    # Water quality parameters
    bod_removed_kg: Optional[float] = Field(None, description="BOD removed in kg")
    cod_removed_kg: Optional[float] = Field(None, description="COD removed in kg")

    created_at: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="Provenance hash")


class WastewaterResponse(BaseModel):
    """Response model for wastewater calculation."""

    success: bool = Field(..., description="Success flag")
    result: Optional[WastewaterEmissionsResult] = Field(None, description="Wastewater result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class WasteCalculationListResponse(BaseModel):
    """Response model for listing waste calculations."""

    success: bool = Field(..., description="Success flag")
    total_count: int = Field(..., description="Total calculations")
    calculations: List[WasteCalculationResult] = Field(..., description="Calculation list")
    page: int = Field(1, description="Current page")
    page_size: int = Field(100, description="Page size")


class WasteCalculationDetailResponse(BaseModel):
    """Response model for single waste calculation detail."""

    success: bool = Field(..., description="Success flag")
    calculation: Optional[WasteCalculationResult] = Field(None, description="Calculation details")
    error: Optional[str] = Field(None, description="Error message")


class DeleteResponse(BaseModel):
    """Response model for deletion operations."""

    success: bool = Field(..., description="Success flag")
    deleted_id: str = Field(..., description="Deleted identifier")
    message: Optional[str] = Field(None, description="Status message")


class WasteEmissionFactor(BaseModel):
    """Waste emission factor detail."""

    factor_id: str = Field(..., description="Factor identifier")
    waste_type: str = Field(..., description="Waste type")
    treatment_method: str = Field(..., description="Treatment method")
    co2_factor: float = Field(..., description="CO2 factor")
    ch4_factor: float = Field(..., description="CH4 factor")
    n2o_factor: Optional[float] = Field(None, description="N2O factor")
    unit: str = Field(..., description="Factor unit")
    source: str = Field(..., description="Data source")
    valid_from: datetime = Field(..., description="Valid from")
    valid_until: Optional[datetime] = Field(None, description="Valid until")


class WasteEmissionFactorListResponse(BaseModel):
    """Response model for listing waste emission factors."""

    success: bool = Field(..., description="Success flag")
    total_count: int = Field(..., description="Total factors")
    factors: List[WasteEmissionFactor] = Field(..., description="Factor list")
    page: int = Field(1, description="Current page")
    page_size: int = Field(100, description="Page size")


class WasteEmissionFactorDetailResponse(BaseModel):
    """Response model for single waste emission factor."""

    success: bool = Field(..., description="Success flag")
    factor: Optional[WasteEmissionFactor] = Field(None, description="Factor details")
    error: Optional[str] = Field(None, description="Error message")


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


class WasteAggregationResult(BaseModel):
    """Waste aggregation result."""

    group_by: str = Field(..., description="Grouping dimension")
    groups: Dict[str, Dict[str, float]] = Field(..., description="Aggregated data by group")
    total_co2e_kg: float = Field(..., description="Total emissions across all groups")
    total_waste_kg: float = Field(..., description="Total waste across all groups")
    calculation_count: int = Field(..., description="Total calculations")


class WasteAggregationResponse(BaseModel):
    """Response model for waste aggregations."""

    success: bool = Field(..., description="Success flag")
    aggregation: Optional[WasteAggregationResult] = Field(None, description="Aggregation result")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class WasteDiversionAnalysis(BaseModel):
    """Waste diversion analysis result."""

    reporting_period: str = Field(..., description="Reporting period")
    total_waste_kg: float = Field(..., description="Total waste generated")
    landfilled_kg: float = Field(..., description="Waste sent to landfill")
    incinerated_kg: float = Field(..., description="Waste incinerated")
    recycled_kg: float = Field(..., description="Waste recycled")
    composted_kg: float = Field(..., description="Waste composted")
    diverted_kg: float = Field(..., description="Total waste diverted from landfill")
    diversion_rate: float = Field(..., ge=0, le=1, description="Waste diversion rate (0-1)")
    total_emissions_kg: float = Field(..., description="Total emissions from all waste")
    emissions_per_kg_waste: float = Field(..., description="Emissions intensity (kg CO2e / kg waste)")

    # Comparison with baseline (if provided)
    baseline_diversion_rate: Optional[float] = Field(None, description="Baseline diversion rate")
    diversion_rate_improvement: Optional[float] = Field(None, description="Improvement vs baseline")


class WasteDiversionResponse(BaseModel):
    """Response model for waste diversion analysis."""

    success: bool = Field(..., description="Success flag")
    analysis: Optional[WasteDiversionAnalysis] = Field(None, description="Diversion analysis")
    error: Optional[str] = Field(None, description="Error message")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class ProvenanceChainResult(BaseModel):
    """Provenance chain result."""

    calculation_id: str = Field(..., description="Calculation identifier")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    input_hashes: List[str] = Field(..., description="Input data hashes")
    engine_versions: Dict[str, str] = Field(..., description="Engine versions used")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    reproducible: bool = Field(..., description="Calculation is reproducible")


class ProvenanceResponse(BaseModel):
    """Response model for provenance queries."""

    success: bool = Field(..., description="Success flag")
    provenance: Optional[ProvenanceChainResult] = Field(None, description="Provenance chain")
    error: Optional[str] = Field(None, description="Error message")


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
    calculations_by_treatment: Dict[str, int] = Field(..., description="Calculations by treatment method")
    calculations_by_waste_type: Dict[str, int] = Field(..., description="Calculations by waste type")
    total_waste_kg: float = Field(..., description="Total waste processed")
    total_co2e_kg: float = Field(..., description="Total emissions calculated")
    avg_diversion_rate: float = Field(..., description="Average diversion rate")
    avg_processing_time_ms: float = Field(..., description="Average processing time")


# ============================================================================
# WasteGeneratedService Class
# ============================================================================


class WasteGeneratedService:
    """
    Waste Generated in Operations Service Facade.

    This service wires together all 7 engines to provide a complete API
    for waste generated in operations emissions calculations (Scope 3 Category 5).

    The service supports:
        - Landfill emissions with methane modeling
        - Incineration emissions (with/without energy recovery)
        - Recycling, composting, and anaerobic digestion
        - Wastewater treatment emissions
        - 12 waste types (food, paper, plastics, metals, glass, textiles, etc.)
        - Emission factor management (7+ authoritative sources)
        - Waste diversion analysis and optimization
        - Compliance checking (7 regulatory frameworks)
        - Uncertainty quantification (Monte Carlo)
        - Aggregations, diversion metrics, reporting

    Engines:
        1. WasteClassificationDatabaseEngine - Data persistence
        2. LandfillEmissionsEngine - Landfill emissions with methane modeling
        3. IncinerationEmissionsEngine - Incineration emissions
        4. RecyclingCompostingEngine - Recycling, composting, anaerobic digestion
        5. WastewaterEmissionsEngine - Wastewater treatment emissions
        6. ComplianceCheckerEngine - Compliance validation
        7. WasteGeneratedPipelineEngine - End-to-end pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton instance.

    Example:
        >>> service = get_service()
        >>> response = service.calculate_landfill(LandfillRequest(
        ...     tenant_id="acme-corp",
        ...     waste_id="WASTE-001",
        ...     waste_type="FOOD_WASTE",
        ...     amount_kg=5000.0,
        ...     landfill_type="MANAGED_MSW",
        ...     has_methane_recovery=False
        ... ))
        >>> assert response.success
        >>> print(f"CH4 Emissions: {response.result.ch4_emissions_kg} kg")

    Attributes:
        config: Service configuration
        metrics: Metrics tracker (OBS-001 integration)
        provenance: Provenance tracker (AGENT-FOUND-008)
        waste_db_engine: Database engine
        landfill_engine: Landfill emissions engine
        incineration_engine: Incineration emissions engine
        recycling_composting_engine: Recycling/composting engine
        wastewater_engine: Wastewater emissions engine
        compliance_engine: Compliance checker
        pipeline_engine: Pipeline orchestrator
    """

    def __init__(self):
        """Initialize WasteGeneratedService."""
        logger.info("Initializing WasteGeneratedService")

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
            logger.info("WasteGeneratedService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WasteGeneratedService: {e}", exc_info=True)
            raise

    def _load_config(self) -> Dict[str, Any]:
        """Load service configuration."""
        try:
            from greenlang.config import get_config
            config = get_config()
            return config.get('waste_generated', {})
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
            # Engine 1: WasteClassificationDatabaseEngine
            from greenlang.agents.mrv.waste_generated.engines.waste_classification_database import (
                WasteClassificationDatabaseEngine
            )
            self.waste_db_engine = WasteClassificationDatabaseEngine(self.config)
            logger.info("WasteClassificationDatabaseEngine initialized")

            # Engine 2: LandfillEmissionsEngine
            from greenlang.agents.mrv.waste_generated.engines.landfill_emissions import (
                LandfillEmissionsEngine
            )
            self.landfill_engine = LandfillEmissionsEngine(self.config)
            logger.info("LandfillEmissionsEngine initialized")

            # Engine 3: IncinerationEmissionsEngine
            from greenlang.agents.mrv.waste_generated.engines.incineration_emissions import (
                IncinerationEmissionsEngine
            )
            self.incineration_engine = IncinerationEmissionsEngine(self.config)
            logger.info("IncinerationEmissionsEngine initialized")

            # Engine 4: RecyclingCompostingEngine
            from greenlang.agents.mrv.waste_generated.engines.recycling_composting import (
                RecyclingCompostingEngine
            )
            self.recycling_composting_engine = RecyclingCompostingEngine(self.config)
            logger.info("RecyclingCompostingEngine initialized")

            # Engine 5: WastewaterEmissionsEngine
            from greenlang.agents.mrv.waste_generated.engines.wastewater_emissions import (
                WastewaterEmissionsEngine
            )
            self.wastewater_engine = WastewaterEmissionsEngine(self.config)
            logger.info("WastewaterEmissionsEngine initialized")

            # Engine 6: ComplianceCheckerEngine
            from greenlang.agents.mrv.waste_generated.engines.compliance_checker import (
                ComplianceCheckerEngine
            )
            self.compliance_engine = ComplianceCheckerEngine(self.config)
            logger.info("ComplianceCheckerEngine initialized")

            # Engine 7: WasteGeneratedPipelineEngine
            from greenlang.agents.mrv.waste_generated.engines.waste_generated_pipeline import (
                WasteGeneratedPipelineEngine
            )
            self.pipeline_engine = WasteGeneratedPipelineEngine(
                self.config,
                waste_db_engine=self.waste_db_engine,
                landfill_engine=self.landfill_engine,
                incineration_engine=self.incineration_engine,
                recycling_composting_engine=self.recycling_composting_engine,
                wastewater_engine=self.wastewater_engine,
                compliance_engine=self.compliance_engine
            )
            logger.info("WasteGeneratedPipelineEngine initialized")

        except ImportError as e:
            logger.error(f"Failed to import engine: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}", exc_info=True)
            raise

    # ========================================================================
    # Public API Methods
    # ========================================================================

    def calculate(self, request: WasteCalculationRequest) -> WasteCalculateResponse:
        """
        Calculate emissions for a single waste disposal.

        This method routes the calculation to the appropriate engine based on
        the treatment_method field (LANDFILL, INCINERATION, RECYCLING, etc.).

        Args:
            request: Waste calculation request with disposal details

        Returns:
            WasteCalculateResponse with calculation result and provenance

        Raises:
            ValueError: If request validation fails
            ProcessingError: If calculation fails

        Example:
            >>> response = service.calculate(WasteCalculationRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WASTE-001",
            ...     waste_type="FOOD_WASTE",
            ...     treatment_method="LANDFILL",
            ...     amount_kg=5000.0,
            ...     landfill_type="MANAGED_MSW"
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating waste emissions for {request.waste_id}")

        try:
            # Track metrics
            if self.metrics:
                self.metrics.increment('waste_generated.calculations.total')
                self.metrics.increment(f'waste_generated.calculations.method.{request.treatment_method.lower()}')

            # Execute calculation via pipeline
            result = self.pipeline_engine.execute_calculation(request)

            # Convert to response model
            calculation_result = WasteCalculationResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                waste_type=result['waste_type'],
                treatment_method=result['treatment_method'],
                co2_kg=result['co2_kg'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                amount_kg=result['amount_kg'],
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
                self.metrics.increment('waste_generated.calculations.success')
                self.metrics.histogram('waste_generated.processing_time_ms', processing_time_ms)

            logger.info(f"Calculation {result['calculation_id']} completed in {processing_time_ms:.2f}ms")

            return WasteCalculateResponse(
                success=True,
                calculation_id=result['calculation_id'],
                result=calculation_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Waste calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.calculations.error')

            return WasteCalculateResponse(
                success=False,
                calculation_id=str(uuid4()),
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_batch(self, request: BatchWasteCalculationRequest) -> WasteBatchResponse:
        """
        Calculate emissions for multiple waste disposals in batch.

        Supports parallel execution for improved performance on large batches.

        Args:
            request: Batch calculation request with list of waste disposals

        Returns:
            WasteBatchResponse with individual results and summary

        Example:
            >>> response = service.calculate_batch(BatchWasteCalculationRequest(
            ...     tenant_id="acme-corp",
            ...     calculations=[calc1, calc2, calc3],
            ...     parallel=True
            ... ))
            >>> assert response.successful_calculations == 3
        """
        start_time = datetime.now()
        logger.info(f"Batch waste calculation: {len(request.calculations)} items, parallel={request.parallel}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.batch_calculations.total')

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
                            results.append(WasteCalculateResponse(
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
                self.metrics.increment('waste_generated.batch_calculations.success')
                self.metrics.histogram('waste_generated.batch_processing_time_ms', processing_time_ms)

            return WasteBatchResponse(
                success=True,
                total_calculations=len(request.calculations),
                successful_calculations=successful,
                failed_calculations=failed,
                results=results,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Batch waste calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.batch_calculations.error')

            return WasteBatchResponse(
                success=False,
                total_calculations=len(request.calculations),
                successful_calculations=0,
                failed_calculations=len(request.calculations),
                results=[],
                processing_time_ms=processing_time_ms
            )

    def calculate_landfill(self, request: LandfillRequest) -> LandfillResponse:
        """
        Calculate landfill emissions with methane modeling.

        Uses IPCC methodology with methane generation modeling (first-order decay).
        Accounts for methane recovery systems and climate-specific parameters.

        Args:
            request: Landfill calculation request

        Returns:
            LandfillResponse with detailed methane modeling results

        Example:
            >>> response = service.calculate_landfill(LandfillRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WASTE-001",
            ...     waste_type="FOOD_WASTE",
            ...     amount_kg=5000.0,
            ...     landfill_type="MANAGED_MSW",
            ...     has_methane_recovery=True,
            ...     methane_recovery_rate=0.75
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating landfill emissions for {request.waste_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.landfill_calculations.total')

            # Execute landfill calculation
            result = self.landfill_engine.calculate_landfill_emissions(request)

            # Convert to response model
            landfill_result = LandfillEmissionsResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                waste_type=result['waste_type'],
                amount_kg=result['amount_kg'],
                landfill_type=result['landfill_type'],
                ch4_generated_kg=result['ch4_generated_kg'],
                ch4_recovered_kg=result['ch4_recovered_kg'],
                ch4_emissions_kg=result['ch4_emissions_kg'],
                co2_biogenic_kg=result['co2_biogenic_kg'],
                co2_fossil_kg=result['co2_fossil_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                degradable_organic_carbon=result['degradable_organic_carbon'],
                methane_correction_factor=result['methane_correction_factor'],
                methane_generation_rate=result['methane_generation_rate'],
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.landfill_calculations.success')

            logger.info(f"Landfill calculation completed: {result['ch4_emissions_kg']:.2f} kg CH4")

            return LandfillResponse(
                success=True,
                result=landfill_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Landfill calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.landfill_calculations.error')

            return LandfillResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_incineration(self, request: IncinerationRequest) -> IncinerationResponse:
        """
        Calculate incineration emissions with optional energy recovery.

        Accounts for fossil and biogenic CO2, CH4, and N2O emissions.
        Calculates avoided emissions from energy recovery where applicable.

        Args:
            request: Incineration calculation request

        Returns:
            IncinerationResponse with emissions and energy recovery details

        Example:
            >>> response = service.calculate_incineration(IncinerationRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WASTE-002",
            ...     waste_type="MIXED_MSW",
            ...     amount_kg=10000.0,
            ...     incineration_type="WITH_ENERGY",
            ...     energy_recovered_kwh=5000.0
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating incineration emissions for {request.waste_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.incineration_calculations.total')

            # Execute incineration calculation
            result = self.incineration_engine.calculate_incineration_emissions(request)

            # Convert to response model
            incineration_result = IncinerationEmissionsResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                waste_type=result['waste_type'],
                amount_kg=result['amount_kg'],
                incineration_type=result['incineration_type'],
                co2_fossil_kg=result['co2_fossil_kg'],
                co2_biogenic_kg=result['co2_biogenic_kg'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                energy_recovered_kwh=result.get('energy_recovered_kwh'),
                avoided_emissions_kg=result.get('avoided_emissions_kg'),
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.incineration_calculations.success')

            logger.info(f"Incineration calculation completed: {result['total_co2e_kg']:.2f} kg CO2e")

            return IncinerationResponse(
                success=True,
                result=incineration_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Incineration calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.incineration_calculations.error')

            return IncinerationResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_recycling(self, request: RecyclingCompostingRequest) -> RecyclingCompostingResponse:
        """
        Calculate recycling emissions with avoided emissions benefits.

        Accounts for emissions from recycling processes and avoided emissions
        from virgin material production.

        Args:
            request: Recycling calculation request

        Returns:
            RecyclingCompostingResponse with net emissions

        Example:
            >>> response = service.calculate_recycling(RecyclingCompostingRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WASTE-003",
            ...     waste_type="PAPER",
            ...     amount_kg=2000.0,
            ...     treatment_method="RECYCLING",
            ...     material_recovery_rate=0.85
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating recycling emissions for {request.waste_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.recycling_calculations.total')

            # Execute recycling calculation
            result = self.recycling_composting_engine.calculate_recycling_emissions(request)

            # Convert to response model
            recycling_result = RecyclingCompostingResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                waste_type=result['waste_type'],
                treatment_method=result['treatment_method'],
                amount_kg=result['amount_kg'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                avoided_emissions_kg=result.get('avoided_emissions_kg'),
                net_emissions_kg=result['net_emissions_kg'],
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.recycling_calculations.success')

            logger.info(f"Recycling calculation completed: {result['net_emissions_kg']:.2f} kg CO2e net")

            return RecyclingCompostingResponse(
                success=True,
                result=recycling_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Recycling calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.recycling_calculations.error')

            return RecyclingCompostingResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_composting(self, request: RecyclingCompostingRequest) -> RecyclingCompostingResponse:
        """
        Calculate composting emissions.

        Accounts for CH4 and N2O emissions from aerobic/anaerobic composting.

        Args:
            request: Composting calculation request

        Returns:
            RecyclingCompostingResponse with emissions

        Example:
            >>> response = service.calculate_composting(RecyclingCompostingRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WASTE-004",
            ...     waste_type="FOOD_WASTE",
            ...     amount_kg=1000.0,
            ...     treatment_method="COMPOSTING"
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating composting emissions for {request.waste_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.composting_calculations.total')

            # Execute composting calculation
            result = self.recycling_composting_engine.calculate_composting_emissions(request)

            # Convert to response model
            composting_result = RecyclingCompostingResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                waste_type=result['waste_type'],
                treatment_method=result['treatment_method'],
                amount_kg=result['amount_kg'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                avoided_emissions_kg=result.get('avoided_emissions_kg'),
                net_emissions_kg=result['net_emissions_kg'],
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.composting_calculations.success')

            logger.info(f"Composting calculation completed: {result['total_co2e_kg']:.2f} kg CO2e")

            return RecyclingCompostingResponse(
                success=True,
                result=composting_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Composting calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.composting_calculations.error')

            return RecyclingCompostingResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_anaerobic_digestion(self, request: RecyclingCompostingRequest) -> RecyclingCompostingResponse:
        """
        Calculate anaerobic digestion emissions.

        Accounts for CH4 emissions and biogas capture/utilization benefits.

        Args:
            request: Anaerobic digestion calculation request

        Returns:
            RecyclingCompostingResponse with emissions and benefits

        Example:
            >>> response = service.calculate_anaerobic_digestion(RecyclingCompostingRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WASTE-005",
            ...     waste_type="FOOD_WASTE",
            ...     amount_kg=3000.0,
            ...     treatment_method="ANAEROBIC_DIGESTION",
            ...     biogas_captured=True
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating anaerobic digestion emissions for {request.waste_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.anaerobic_digestion_calculations.total')

            # Execute anaerobic digestion calculation
            result = self.recycling_composting_engine.calculate_anaerobic_digestion_emissions(request)

            # Convert to response model
            ad_result = RecyclingCompostingResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                waste_type=result['waste_type'],
                treatment_method=result['treatment_method'],
                amount_kg=result['amount_kg'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                avoided_emissions_kg=result.get('avoided_emissions_kg'),
                net_emissions_kg=result['net_emissions_kg'],
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.anaerobic_digestion_calculations.success')

            logger.info(f"Anaerobic digestion calculation completed: {result['net_emissions_kg']:.2f} kg CO2e net")

            return RecyclingCompostingResponse(
                success=True,
                result=ad_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Anaerobic digestion calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.anaerobic_digestion_calculations.error')

            return RecyclingCompostingResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def calculate_wastewater(self, request: WastewaterRequest) -> WastewaterResponse:
        """
        Calculate wastewater treatment emissions.

        Accounts for CH4 and N2O emissions from wastewater treatment systems
        based on BOD/COD loading and treatment system type.

        Args:
            request: Wastewater calculation request

        Returns:
            WastewaterResponse with emissions

        Example:
            >>> response = service.calculate_wastewater(WastewaterRequest(
            ...     tenant_id="acme-corp",
            ...     waste_id="WW-001",
            ...     wastewater_volume_liters=100000.0,
            ...     bod_mg_per_liter=250.0,
            ...     treatment_system="AEROBIC",
            ...     has_biogas_recovery=False
            ... ))
            >>> assert response.success
        """
        start_time = datetime.now()
        logger.info(f"Calculating wastewater emissions for {request.waste_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.wastewater_calculations.total')

            # Execute wastewater calculation
            result = self.wastewater_engine.calculate_wastewater_emissions(request)

            # Convert to response model
            wastewater_result = WastewaterEmissionsResult(
                calculation_id=result['calculation_id'],
                waste_id=result['waste_id'],
                wastewater_volume_liters=result['wastewater_volume_liters'],
                treatment_system=result['treatment_system'],
                ch4_kg=result['ch4_kg'],
                n2o_kg=result['n2o_kg'],
                total_co2e_kg=result['total_co2e_kg'],
                bod_removed_kg=result.get('bod_removed_kg'),
                cod_removed_kg=result.get('cod_removed_kg'),
                created_at=result['created_at'],
                provenance_hash=result['provenance_hash']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.wastewater_calculations.success')

            logger.info(f"Wastewater calculation completed: {result['total_co2e_kg']:.2f} kg CO2e")

            return WastewaterResponse(
                success=True,
                result=wastewater_result,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Wastewater calculation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.wastewater_calculations.error')

            return WastewaterResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def get_calculation(self, calculation_id: str) -> WasteCalculationDetailResponse:
        """
        Get a single waste calculation by ID.

        Args:
            calculation_id: Calculation identifier

        Returns:
            WasteCalculationDetailResponse with calculation details

        Example:
            >>> response = service.get_calculation("calc-001")
            >>> assert response.success
            >>> print(response.calculation.total_co2e_kg)
        """
        logger.info(f"Retrieving waste calculation {calculation_id}")

        try:
            calc = self.waste_db_engine.get_calculation(calculation_id)

            if calc:
                return WasteCalculationDetailResponse(
                    success=True,
                    calculation=WasteCalculationResult(**calc)
                )
            else:
                return WasteCalculationDetailResponse(
                    success=False,
                    error=f"Calculation {calculation_id} not found"
                )

        except Exception as e:
            logger.error(f"Get waste calculation failed: {e}", exc_info=True)
            return WasteCalculationDetailResponse(
                success=False,
                error=str(e)
            )

    def list_calculations(
        self,
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 100
    ) -> WasteCalculationListResponse:
        """
        List waste calculations for a tenant with optional filtering.

        Args:
            tenant_id: Tenant identifier
            filters: Optional filters (e.g., {'waste_type': 'FOOD_WASTE', 'period': '2024-01'})
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            WasteCalculationListResponse with paginated results

        Example:
            >>> response = service.list_calculations(
            ...     tenant_id="acme-corp",
            ...     filters={'waste_type': 'FOOD_WASTE', 'period': '2024-01'},
            ...     page=1,
            ...     page_size=50
            ... )
            >>> assert len(response.calculations) <= 50
        """
        logger.info(f"Listing waste calculations for tenant {tenant_id}, page={page}, filters={filters}")

        try:
            results = self.waste_db_engine.list_calculations(
                tenant_id=tenant_id,
                filters=filters or {},
                page=page,
                page_size=page_size
            )

            calculations = [
                WasteCalculationResult(**calc) for calc in results['calculations']
            ]

            return WasteCalculationListResponse(
                success=True,
                total_count=results['total_count'],
                calculations=calculations,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"List waste calculations failed: {e}", exc_info=True)
            return WasteCalculationListResponse(
                success=False,
                total_count=0,
                calculations=[],
                page=page,
                page_size=page_size
            )

    def delete_calculation(self, calculation_id: str) -> DeleteResponse:
        """
        Delete a waste calculation by ID.

        Args:
            calculation_id: Calculation identifier

        Returns:
            DeleteResponse with status

        Example:
            >>> response = service.delete_calculation("calc-001")
            >>> assert response.success
        """
        logger.info(f"Deleting waste calculation {calculation_id}")

        try:
            # Track provenance
            if self.provenance:
                self.provenance.track_deletion(
                    entity_type='waste_calculation',
                    entity_id=calculation_id
                )

            self.waste_db_engine.delete_calculation(calculation_id)

            if self.metrics:
                self.metrics.increment('waste_generated.calculations.deleted')

            return DeleteResponse(
                success=True,
                deleted_id=calculation_id,
                message="Waste calculation deleted successfully"
            )

        except Exception as e:
            logger.error(f"Delete waste calculation failed: {e}", exc_info=True)
            return DeleteResponse(
                success=False,
                deleted_id=calculation_id,
                message=str(e)
            )

    def list_emission_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 100
    ) -> WasteEmissionFactorListResponse:
        """
        List waste emission factors with optional filtering.

        Args:
            filters: Optional filters (e.g., {'waste_type': 'FOOD_WASTE', 'treatment_method': 'LANDFILL'})
            page: Page number (1-indexed)
            page_size: Results per page

        Returns:
            WasteEmissionFactorListResponse with paginated factors

        Example:
            >>> response = service.list_emission_factors(
            ...     filters={'waste_type': 'FOOD_WASTE'},
            ...     page=1,
            ...     page_size=50
            ... )
            >>> assert all(f.waste_type == 'FOOD_WASTE' for f in response.factors)
        """
        logger.info(f"Listing waste emission factors: filters={filters}")

        try:
            results = self.waste_db_engine.list_emission_factors(
                filters=filters or {},
                page=page,
                page_size=page_size
            )

            factors = [
                WasteEmissionFactor(**ef) for ef in results['factors']
            ]

            return WasteEmissionFactorListResponse(
                success=True,
                total_count=results['total_count'],
                factors=factors,
                page=page,
                page_size=page_size
            )

        except Exception as e:
            logger.error(f"List waste emission factors failed: {e}", exc_info=True)
            return WasteEmissionFactorListResponse(
                success=False,
                total_count=0,
                factors=[],
                page=page,
                page_size=page_size
            )

    def get_emission_factor(self, waste_type: str, treatment_method: str) -> WasteEmissionFactorDetailResponse:
        """
        Get emission factor for a specific waste type and treatment method.

        Args:
            waste_type: Waste type (e.g., 'FOOD_WASTE', 'PAPER', 'PLASTICS')
            treatment_method: Treatment method (e.g., 'LANDFILL', 'INCINERATION', 'RECYCLING')

        Returns:
            WasteEmissionFactorDetailResponse with factor details

        Example:
            >>> response = service.get_emission_factor("FOOD_WASTE", "LANDFILL")
            >>> assert response.success
        """
        logger.info(f"Retrieving waste emission factor: {waste_type} / {treatment_method}")

        try:
            factor = self.waste_db_engine.get_emission_factor(waste_type, treatment_method)

            if factor:
                return WasteEmissionFactorDetailResponse(
                    success=True,
                    factor=WasteEmissionFactor(**factor)
                )
            else:
                return WasteEmissionFactorDetailResponse(
                    success=False,
                    error=f"Emission factor not found for {waste_type} / {treatment_method}"
                )

        except Exception as e:
            logger.error(f"Get waste emission factor failed: {e}", exc_info=True)
            return WasteEmissionFactorDetailResponse(
                success=False,
                error=str(e)
            )

    def list_waste_types(self) -> List[Dict[str, Any]]:
        """
        List all supported waste types.

        Returns:
            List of waste type definitions with metadata

        Example:
            >>> waste_types = service.list_waste_types()
            >>> assert any(wt['type'] == 'FOOD_WASTE' for wt in waste_types)
        """
        logger.info("Listing waste types")

        try:
            return self.waste_db_engine.list_waste_types()
        except Exception as e:
            logger.error(f"List waste types failed: {e}", exc_info=True)
            return []

    def list_treatment_methods(self) -> List[Dict[str, Any]]:
        """
        List all supported treatment methods.

        Returns:
            List of treatment method definitions with metadata

        Example:
            >>> methods = service.list_treatment_methods()
            >>> assert any(m['method'] == 'LANDFILL' for m in methods)
        """
        logger.info("Listing treatment methods")

        try:
            return self.waste_db_engine.list_treatment_methods()
        except Exception as e:
            logger.error(f"List treatment methods failed: {e}", exc_info=True)
            return []

    def check_compliance(
        self,
        calculation_id: str,
        frameworks: List[str]
    ) -> ComplianceCheckResponse:
        """
        Check waste calculation compliance against regulatory frameworks.

        Validates a calculation against one or more regulatory frameworks:
        - GHG_PROTOCOL
        - ISO_14064
        - CSRD
        - CDP
        - SBTi
        - EU_ETS
        - TCFD

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
        logger.info(f"Checking compliance for waste calculation {calculation_id} against {frameworks}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.compliance_checks.total')

            # Retrieve calculation
            calc = self.waste_db_engine.get_calculation(calculation_id)
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
                self.metrics.increment('waste_generated.compliance_checks.success')
                if overall_compliant:
                    self.metrics.increment('waste_generated.compliance_checks.compliant')
                else:
                    self.metrics.increment('waste_generated.compliance_checks.non_compliant')

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
                self.metrics.increment('waste_generated.compliance_checks.error')

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

    def analyze_uncertainty(
        self,
        calculation_id: str,
        iterations: Optional[int] = None
    ) -> UncertaintyResponse:
        """
        Quantify uncertainty for a waste calculation using Monte Carlo simulation.

        Performs Monte Carlo simulation to estimate uncertainty in emissions
        calculations by varying input parameters within their uncertainty ranges.

        Args:
            calculation_id: Calculation to analyze
            iterations: Number of Monte Carlo iterations (default: 10000)

        Returns:
            UncertaintyResponse with statistical results

        Example:
            >>> response = service.analyze_uncertainty("calc-001", iterations=10000)
            >>> assert response.uncertainty.uncertainty_percentage > 0
        """
        start_time = datetime.now()
        logger.info(f"Analyzing uncertainty for waste calculation {calculation_id}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.uncertainty.total')

            # Retrieve calculation
            calc = self.waste_db_engine.get_calculation(calculation_id)
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
                self.metrics.increment('waste_generated.uncertainty.success')

            logger.info(f"Uncertainty analysis completed: {result['uncertainty_percentage']:.2f}%")

            return UncertaintyResponse(
                success=True,
                uncertainty=uncertainty,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Uncertainty analysis failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.uncertainty.error')

            return UncertaintyResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def get_aggregation(
        self,
        tenant_id: str,
        group_by: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> WasteAggregationResponse:
        """
        Get aggregated waste emissions by a dimension.

        Aggregates emissions across calculations grouped by a specified dimension:
        - waste_type: Waste type (FOOD_WASTE, PAPER, PLASTICS, etc.)
        - treatment_method: Treatment method (LANDFILL, INCINERATION, RECYCLING, etc.)
        - waste_source: Source of waste
        - period: Reporting period (YYYY-MM)

        Args:
            tenant_id: Tenant identifier
            group_by: Dimension to group by
            filters: Optional filters to apply before aggregation

        Returns:
            WasteAggregationResponse with aggregated data

        Example:
            >>> response = service.get_aggregation(
            ...     tenant_id="acme-corp",
            ...     group_by="waste_type",
            ...     filters={'period': '2024-01'}
            ... )
            >>> assert 'FOOD_WASTE' in response.aggregation.groups
        """
        start_time = datetime.now()
        logger.info(f"Getting waste aggregations for {tenant_id}, group_by={group_by}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.aggregations.total')

            result = self.waste_db_engine.get_aggregations(
                tenant_id=tenant_id,
                group_by=group_by,
                filters=filters or {}
            )

            aggregation = WasteAggregationResult(
                group_by=group_by,
                groups=result['groups'],
                total_co2e_kg=result['total_co2e_kg'],
                total_waste_kg=result['total_waste_kg'],
                calculation_count=result['calculation_count']
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.aggregations.success')

            return WasteAggregationResponse(
                success=True,
                aggregation=aggregation,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Waste aggregation failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.aggregations.error')

            return WasteAggregationResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def analyze_diversion(
        self,
        tenant_id: str,
        reporting_period: str,
        baseline_period: Optional[str] = None
    ) -> WasteDiversionResponse:
        """
        Analyze waste diversion rates and emissions intensity.

        Calculates waste diversion metrics including diversion rate,
        emissions intensity, and comparison with baseline if provided.

        Args:
            tenant_id: Tenant identifier
            reporting_period: Reporting period (YYYY-MM)
            baseline_period: Optional baseline period for comparison (YYYY-MM)

        Returns:
            WasteDiversionResponse with diversion analysis

        Example:
            >>> response = service.analyze_diversion(
            ...     tenant_id="acme-corp",
            ...     reporting_period="2024-01",
            ...     baseline_period="2023-01"
            ... )
            >>> assert response.analysis.diversion_rate >= 0
        """
        start_time = datetime.now()
        logger.info(f"Analyzing waste diversion for {tenant_id}, period={reporting_period}")

        try:
            if self.metrics:
                self.metrics.increment('waste_generated.diversion_analysis.total')

            result = self.waste_db_engine.analyze_diversion(
                tenant_id=tenant_id,
                reporting_period=reporting_period,
                baseline_period=baseline_period
            )

            analysis = WasteDiversionAnalysis(
                reporting_period=result['reporting_period'],
                total_waste_kg=result['total_waste_kg'],
                landfilled_kg=result['landfilled_kg'],
                incinerated_kg=result['incinerated_kg'],
                recycled_kg=result['recycled_kg'],
                composted_kg=result['composted_kg'],
                diverted_kg=result['diverted_kg'],
                diversion_rate=result['diversion_rate'],
                total_emissions_kg=result['total_emissions_kg'],
                emissions_per_kg_waste=result['emissions_per_kg_waste'],
                baseline_diversion_rate=result.get('baseline_diversion_rate'),
                diversion_rate_improvement=result.get('diversion_rate_improvement')
            )

            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            if self.metrics:
                self.metrics.increment('waste_generated.diversion_analysis.success')

            logger.info(f"Diversion analysis completed: {result['diversion_rate']:.2%} diversion rate")

            return WasteDiversionResponse(
                success=True,
                analysis=analysis,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Diversion analysis failed: {e}", exc_info=True)

            if self.metrics:
                self.metrics.increment('waste_generated.diversion_analysis.error')

            return WasteDiversionResponse(
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )

    def get_provenance(self, calculation_id: str) -> ProvenanceResponse:
        """
        Get provenance chain for a waste calculation.

        Returns complete audit trail including input hashes, engine versions,
        and reproducibility status.

        Args:
            calculation_id: Calculation identifier

        Returns:
            ProvenanceResponse with provenance chain

        Example:
            >>> response = service.get_provenance("calc-001")
            >>> assert response.provenance.reproducible
        """
        logger.info(f"Retrieving provenance for waste calculation {calculation_id}")

        try:
            calc = self.waste_db_engine.get_calculation(calculation_id)
            if not calc:
                raise ValueError(f"Calculation {calculation_id} not found")

            provenance_data = self.waste_db_engine.get_provenance(calculation_id)

            provenance = ProvenanceChainResult(
                calculation_id=calculation_id,
                provenance_hash=calc['provenance_hash'],
                input_hashes=provenance_data.get('input_hashes', []),
                engine_versions=provenance_data.get('engine_versions', {}),
                timestamp=calc['created_at'],
                reproducible=provenance_data.get('reproducible', True)
            )

            return ProvenanceResponse(
                success=True,
                provenance=provenance
            )

        except Exception as e:
            logger.error(f"Get provenance failed: {e}", exc_info=True)
            return ProvenanceResponse(
                success=False,
                error=str(e)
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
                'waste_db_engine': self._check_engine_health(self.waste_db_engine),
                'landfill_engine': self._check_engine_health(self.landfill_engine),
                'incineration_engine': self._check_engine_health(self.incineration_engine),
                'recycling_composting_engine': self._check_engine_health(self.recycling_composting_engine),
                'wastewater_engine': self._check_engine_health(self.wastewater_engine),
                'compliance_engine': self._check_engine_health(self.compliance_engine),
                'pipeline_engine': self._check_engine_health(self.pipeline_engine),
            }

            database_connected = self.waste_db_engine.check_connection()

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
            stats = self.waste_db_engine.get_stats(tenant_id)

            return StatsResponse(
                total_calculations=stats['total_calculations'],
                calculations_by_treatment=stats['calculations_by_treatment'],
                calculations_by_waste_type=stats['calculations_by_waste_type'],
                total_waste_kg=stats['total_waste_kg'],
                total_co2e_kg=stats['total_co2e_kg'],
                avg_diversion_rate=stats['avg_diversion_rate'],
                avg_processing_time_ms=stats['avg_processing_time_ms']
            )

        except Exception as e:
            logger.error(f"Get stats failed: {e}", exc_info=True)
            return StatsResponse(
                total_calculations=0,
                calculations_by_treatment={},
                calculations_by_waste_type={},
                total_waste_kg=0.0,
                total_co2e_kg=0.0,
                avg_diversion_rate=0.0,
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


def get_service() -> WasteGeneratedService:
    """
    Get singleton WasteGeneratedService instance.

    Thread-safe singleton pattern ensures only one instance exists.

    Returns:
        WasteGeneratedService instance

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
                _service_instance = WasteGeneratedService()

    return _service_instance


def get_router():
    """
    Get FastAPI router for waste generated API.

    Returns:
        APIRouter instance with all waste generated routes

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> router = get_router()
        >>> app.include_router(router, prefix="/api/v1/waste-generated")
    """
    try:
        from greenlang.agents.mrv.waste_generated.api.router import router
        return router
    except ImportError as e:
        logger.error(f"Failed to import router: {e}")
        raise


def configure_waste_generated(app) -> None:
    """
    Configure waste generated service with FastAPI app.

    Registers routes, middleware, and initializes the service.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.mrv.waste_generated.setup import configure_waste_generated
        >>> app = FastAPI()
        >>> configure_waste_generated(app)
    """
    logger.info("Configuring waste generated service")

    try:
        # Get router
        router = get_router()

        # Include router with prefix
        app.include_router(
            router,
            prefix="/api/v1/waste-generated",
            tags=["waste-generated"]
        )

        # Initialize service (singleton)
        service = get_service()

        logger.info(f"Waste generated service configured: {service._initialized}")

    except Exception as e:
        logger.error(f"Failed to configure waste generated service: {e}", exc_info=True)
        raise


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Service
    'WasteGeneratedService',
    'get_service',
    'get_router',
    'configure_waste_generated',

    # Request Models
    'WasteCalculationRequest',
    'BatchWasteCalculationRequest',
    'LandfillRequest',
    'IncinerationRequest',
    'RecyclingCompostingRequest',
    'WastewaterRequest',
    'ComplianceCheckRequest',
    'DiversionAnalysisRequest',

    # Response Models
    'WasteCalculateResponse',
    'WasteBatchResponse',
    'LandfillResponse',
    'IncinerationResponse',
    'RecyclingCompostingResponse',
    'WastewaterResponse',
    'WasteCalculationListResponse',
    'WasteCalculationDetailResponse',
    'DeleteResponse',
    'WasteEmissionFactorListResponse',
    'WasteEmissionFactorDetailResponse',
    'ComplianceCheckResponse',
    'UncertaintyResponse',
    'WasteAggregationResponse',
    'WasteDiversionResponse',
    'ProvenanceResponse',
    'HealthResponse',
    'StatsResponse',

    # Result Models
    'WasteCalculationResult',
    'LandfillEmissionsResult',
    'IncinerationEmissionsResult',
    'RecyclingCompostingResult',
    'WastewaterEmissionsResult',
    'WasteEmissionFactor',
    'ComplianceResult',
    'UncertaintyResult',
    'WasteAggregationResult',
    'WasteDiversionAnalysis',
    'ProvenanceChainResult',
]
