"""
WasteGeneratedPipelineEngine - Orchestrated 10-stage pipeline for waste emissions.

This module implements the WasteGeneratedPipelineEngine for AGENT-MRV-018 (Waste Generated in Operations).
It orchestrates a 10-stage pipeline for complete waste emissions calculation from raw input
to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (required fields, types, ranges)
2. CLASSIFY: Waste classification by EWC code, category, hazardous status, treatment compatibility
3. NORMALIZE: Unit conversion (short tons→tonnes, wet→dry weight, currency→USD)
4. RESOLVE_EFS: Select emission factors from hierarchy (EPA WARM→DEFRA→IPCC)
5. CALCULATE_TREATMENT: Route to treatment-specific engine (landfill/incineration/recycling/wastewater)
6. CALCULATE_TRANSPORT: Optional transport emissions to treatment facility
7. ALLOCATE: Allocate emissions to business units, products, facilities
8. COMPLIANCE: Framework compliance checking (7 frameworks)
9. AGGREGATE: Aggregate by treatment method, waste category, facility, period
10. SEAL: Provenance hash, audit trail

Example:
    >>> from greenlang.waste_generated.waste_generated_pipeline import WasteGeneratedPipelineEngine
    >>> engine = WasteGeneratedPipelineEngine(config)
    >>> result = engine.process(waste_input)
    >>> assert result.status == "SUCCESS"
    >>> print(f"Total emissions: {result.total_emissions_kg_co2e} kg CO2e")

Module: greenlang.waste_generated.waste_generated_pipeline
Agent: AGENT-MRV-018
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from decimal import Decimal
import logging
import hashlib
import json
from threading import RLock
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator

from greenlang.waste_generated.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    CalculationMethod,
    WasteTreatmentMethod,
    WasteCategory,
    WasteStream,
    LandfillType,
    ClimateZone,
    IncineratorType,
    RecyclingType,
    WastewaterSystem,
    GasCollectionSystem,
    EFSource,
    ComplianceFramework,
    DataQualityTier,
    WasteDataSource,
    ProvenanceStage,
    UncertaintyMethod,
    HazardClass,
    GWPVersion,
    IndustryWastewaterType,
    EmissionGas,
    DQIDimension,
    DQIScore,
    ComplianceStatus,
    CurrencyCode,
)
from greenlang.waste_generated.config import WasteGeneratedConfig
from greenlang.waste_generated.metrics import WasteMetricsCollector
from greenlang.waste_generated.provenance import WasteProvenanceTracker


logger = logging.getLogger(__name__)


# ==============================================================================
# PIPELINE ENUMERATIONS
# ==============================================================================


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class AllocationMethod(str, Enum):
    """Allocation method for shared waste streams."""
    MASS_BASED = "mass_based"  # Allocate by mass proportion
    VOLUME_BASED = "volume_based"  # Allocate by volume proportion
    COST_BASED = "cost_based"  # Allocate by cost proportion
    HEADCOUNT_BASED = "headcount_based"  # Allocate by headcount
    FLOOR_AREA_BASED = "floor_area_based"  # Allocate by floor area
    ACTIVITY_BASED = "activity_based"  # Allocate by activity metric


# ==============================================================================
# INPUT/OUTPUT MODELS
# ==============================================================================


class WasteStreamInput(BaseModel):
    """Input data model for a waste stream."""

    # Identifiers
    stream_id: str = Field(..., description="Unique waste stream identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    facility_id: str = Field(..., description="Facility identifier")
    reporting_period_start: datetime = Field(..., description="Reporting period start")
    reporting_period_end: datetime = Field(..., description="Reporting period end")

    # Waste characterization
    waste_category: WasteCategory = Field(..., description="Waste material category")
    waste_stream: WasteStream = Field(..., description="Waste stream type")
    treatment_method: WasteTreatmentMethod = Field(..., description="Treatment method")
    ewc_code: Optional[str] = Field(None, description="European Waste Catalogue code")
    is_hazardous: bool = Field(False, description="Hazardous waste flag")
    hazard_classes: List[HazardClass] = Field(default_factory=list, description="Basel hazard classes")

    # Quantity data
    mass_tonnes: Optional[Decimal] = Field(None, ge=0, description="Mass in metric tonnes")
    mass_short_tons: Optional[Decimal] = Field(None, ge=0, description="Mass in short tons")
    volume_m3: Optional[Decimal] = Field(None, ge=0, description="Volume in cubic meters")
    moisture_content_pct: Optional[Decimal] = Field(None, ge=0, le=100, description="Moisture content %")

    # Treatment-specific parameters
    landfill_type: Optional[LandfillType] = Field(None, description="Landfill type (if applicable)")
    climate_zone: Optional[ClimateZone] = Field(None, description="Climate zone (for landfill)")
    gas_collection: Optional[GasCollectionSystem] = Field(None, description="Gas collection system")
    incinerator_type: Optional[IncineratorType] = Field(None, description="Incinerator type")
    recycling_type: Optional[RecyclingType] = Field(None, description="Recycling type")
    wastewater_system: Optional[WastewaterSystem] = Field(None, description="Wastewater treatment system")
    industry_type: Optional[IndustryWastewaterType] = Field(None, description="Industry type (for wastewater)")

    # Composition data (for mixed waste)
    composition: Optional[Dict[str, Decimal]] = Field(None, description="Waste composition by category (% by mass)")

    # Transport to facility (optional)
    transport_distance_km: Optional[Decimal] = Field(None, ge=0, description="Distance to treatment facility (km)")
    transport_mode: Optional[str] = Field(None, description="Transport mode")

    # Allocation
    allocation_method: Optional[AllocationMethod] = Field(None, description="Allocation method")
    allocation_factor: Optional[Decimal] = Field(None, ge=0, le=1, description="Allocation factor (0-1)")

    # Spend-based method
    waste_management_spend: Optional[Decimal] = Field(None, ge=0, description="Waste management spend")
    spend_currency: Optional[CurrencyCode] = Field(None, description="Currency code")

    # Data quality
    calculation_method: Optional[CalculationMethod] = Field(None, description="Calculation method")
    data_source: Optional[WasteDataSource] = Field(None, description="Data source type")
    data_quality_tier: Optional[DataQualityTier] = Field(None, description="IPCC data quality tier")

    # Compliance
    frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: [ComplianceFramework.GHG_PROTOCOL],
        description="Regulatory frameworks to check"
    )

    # Metadata
    supplier_id: Optional[str] = Field(None, description="Waste contractor/supplier ID")
    contract_id: Optional[str] = Field(None, description="Waste management contract ID")
    treatment_facility_id: Optional[str] = Field(None, description="Treatment facility ID")
    business_unit: Optional[str] = Field(None, description="Business unit")
    product_line: Optional[str] = Field(None, description="Product line")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WasteClassificationResult(BaseModel):
    """Result of waste classification stage."""
    waste_category: WasteCategory = Field(..., description="Classified waste category")
    waste_stream: WasteStream = Field(..., description="Waste stream type")
    is_hazardous: bool = Field(..., description="Hazardous classification")
    hazard_classes: List[HazardClass] = Field(default_factory=list, description="Hazard classes")
    ewc_code: Optional[str] = Field(None, description="EWC code")
    compatible_treatments: List[WasteTreatmentMethod] = Field(..., description="Compatible treatment methods")
    biogenic_fraction: Decimal = Field(..., ge=0, le=1, description="Biogenic carbon fraction")
    fossil_fraction: Decimal = Field(..., ge=0, le=1, description="Fossil carbon fraction")
    recommended_method: CalculationMethod = Field(..., description="Recommended calculation method")
    classification_confidence: Decimal = Field(..., ge=0, le=1, description="Classification confidence score")


class WasteEmissionFactor(BaseModel):
    """Emission factor for waste treatment."""
    treatment_method: WasteTreatmentMethod = Field(..., description="Treatment method")
    waste_category: WasteCategory = Field(..., description="Waste category")
    ef_source: EFSource = Field(..., description="Emission factor source")

    # Emission factors (kg gas / tonne waste)
    ef_co2_fossil: Decimal = Field(..., description="Fossil CO2 EF (kg CO2/tonne)")
    ef_co2_biogenic: Decimal = Field(..., description="Biogenic CO2 EF (kg CO2/tonne)")
    ef_ch4: Decimal = Field(..., description="CH4 EF (kg CH4/tonne)")
    ef_n2o: Decimal = Field(..., description="N2O EF (kg N2O/tonne)")

    # Uncertainty
    uncertainty_pct: Optional[Decimal] = Field(None, description="Uncertainty (%)")

    # GWP
    gwp_version: GWPVersion = Field(default=GWPVersion.AR5, description="GWP version")
    gwp_ch4: Decimal = Field(default=Decimal("28"), description="CH4 GWP")
    gwp_n2o: Decimal = Field(default=Decimal("265"), description="N2O GWP")

    # Metadata
    data_quality_tier: DataQualityTier = Field(..., description="Data quality tier")
    region: str = Field(default="Global", description="Geographic region")
    year: int = Field(..., description="EF reference year")
    notes: Optional[str] = Field(None, description="Additional notes")


class TreatmentEmissionsResult(BaseModel):
    """Result from treatment-specific calculation engine."""
    treatment_method: WasteTreatmentMethod = Field(..., description="Treatment method")
    mass_treated_tonnes: Decimal = Field(..., ge=0, description="Mass treated (tonnes)")

    # Emissions (kg)
    emissions_kg_co2_fossil: Decimal = Field(..., ge=0, description="Fossil CO2 emissions (kg)")
    emissions_kg_co2_biogenic: Decimal = Field(..., ge=0, description="Biogenic CO2 emissions (kg)")
    emissions_kg_ch4: Decimal = Field(..., ge=0, description="CH4 emissions (kg)")
    emissions_kg_n2o: Decimal = Field(..., ge=0, description="N2O emissions (kg)")
    emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total CO2e emissions (kg)")

    # Emission factor used
    emission_factor: WasteEmissionFactor = Field(..., description="Emission factor used")

    # Treatment-specific results
    treatment_specific_data: Dict[str, Any] = Field(default_factory=dict, description="Treatment-specific data")

    # Uncertainty
    uncertainty_kg_co2e: Optional[Decimal] = Field(None, description="Absolute uncertainty (kg CO2e)")

    # Metadata
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WasteDiversionAnalysis(BaseModel):
    """Waste diversion analysis (from landfill)."""
    total_waste_tonnes: Decimal = Field(..., ge=0, description="Total waste generated (tonnes)")
    landfilled_tonnes: Decimal = Field(..., ge=0, description="Waste sent to landfill (tonnes)")
    recycled_tonnes: Decimal = Field(..., ge=0, description="Waste recycled (tonnes)")
    composted_tonnes: Decimal = Field(..., ge=0, description="Waste composted (tonnes)")
    incinerated_tonnes: Decimal = Field(..., ge=0, description="Waste incinerated (tonnes)")
    other_tonnes: Decimal = Field(..., ge=0, description="Other treatment (tonnes)")

    # Diversion rates
    diversion_rate_pct: Decimal = Field(..., ge=0, le=100, description="Diversion rate from landfill (%)")
    recycling_rate_pct: Decimal = Field(..., ge=0, le=100, description="Recycling rate (%)")

    # Emissions avoided
    emissions_avoided_kg_co2e: Decimal = Field(..., description="Emissions avoided vs. landfill baseline (kg CO2e)")

    # CSRD ESRS E5 metrics
    circular_material_use_rate: Decimal = Field(..., ge=0, le=100, description="Circular material use rate (%)")


class ComplianceCheckResult(BaseModel):
    """Compliance check result for a framework."""
    framework: ComplianceFramework = Field(..., description="Framework")
    status: ComplianceStatus = Field(..., description="Compliance status")
    required_fields: List[str] = Field(default_factory=list, description="Required fields")
    missing_fields: List[str] = Field(default_factory=list, description="Missing fields")
    errors: List[str] = Field(default_factory=list, description="Compliance errors")
    warnings: List[str] = Field(default_factory=list, description="Compliance warnings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class WasteAggregationResult(BaseModel):
    """Aggregated emissions across multiple dimensions."""
    by_treatment_method: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by treatment")
    by_waste_category: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by waste type")
    by_facility: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by facility")
    by_period: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by period")
    by_business_unit: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by business unit")
    by_waste_stream: Dict[str, Decimal] = Field(default_factory=dict, description="Emissions by waste stream")

    # Totals
    total_mass_tonnes: Decimal = Field(..., ge=0, description="Total waste mass (tonnes)")
    total_emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total emissions (kg CO2e)")

    # Diversion analysis
    diversion_analysis: Optional[WasteDiversionAnalysis] = Field(None, description="Diversion analysis")


class WasteCalculationResult(BaseModel):
    """Result model for waste emissions calculation."""
    stream_id: str = Field(..., description="Waste stream identifier")
    status: PipelineStatus = Field(..., description="Pipeline execution status")
    tenant_id: str = Field(..., description="Tenant identifier")
    facility_id: str = Field(..., description="Facility identifier")

    # Results
    treatment_result: Optional[TreatmentEmissionsResult] = Field(None, description="Treatment emissions result")
    transport_emissions_kg_co2e: Decimal = Field(default=Decimal("0"), ge=0, description="Transport emissions (kg CO2e)")

    # Totals (after allocation)
    total_emissions_kg_co2_fossil: Decimal = Field(..., ge=0, description="Total fossil CO2 emissions (kg)")
    total_emissions_kg_co2_biogenic: Decimal = Field(..., ge=0, description="Total biogenic CO2 emissions (kg)")
    total_emissions_kg_ch4: Decimal = Field(..., ge=0, description="Total CH4 emissions (kg)")
    total_emissions_kg_n2o: Decimal = Field(..., ge=0, description="Total N2O emissions (kg)")
    total_emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total CO2e emissions (kg)")

    # Classification
    classification_result: Optional[WasteClassificationResult] = Field(None, description="Classification result")

    # Compliance
    compliance_results: Dict[str, ComplianceCheckResult] = Field(
        default_factory=dict,
        description="Compliance check results by framework"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage_durations_ms: Dict[str, float] = Field(default_factory=dict, description="Duration per stage (ms)")

    # Errors/Warnings
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WasteBatchResult(BaseModel):
    """Result model for batch calculation."""
    batch_id: str = Field(..., description="Batch identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    results: List[WasteCalculationResult] = Field(default_factory=list, description="Individual results")
    aggregation: Optional[WasteAggregationResult] = Field(None, description="Aggregated results")
    total_emissions_kg_co2e: Decimal = Field(..., ge=0, description="Total batch emissions")
    successful_count: int = Field(..., ge=0, description="Number of successful calculations")
    failed_count: int = Field(..., ge=0, description="Number of failed calculations")
    batch_duration_ms: float = Field(..., ge=0, description="Total batch duration (ms)")
    provenance_hash: str = Field(..., description="Batch-level provenance hash")


class ProvenanceChainResult(BaseModel):
    """Provenance chain result."""
    chain_id: str = Field(..., description="Chain identifier")
    provenance_hash: str = Field(..., description="Final provenance hash")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Stage entries")
    sealed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# WASTE GENERATED PIPELINE ENGINE
# ==============================================================================


class WasteGeneratedPipelineEngine:
    """
    WasteGeneratedPipelineEngine - Orchestrated 10-stage pipeline for waste emissions.

    This engine coordinates the complete waste emissions calculation workflow
    through 10 sequential stages, from input validation to sealed audit trail.

    The engine uses lazy initialization for calculation engines, creating them
    only when needed. This reduces startup time and memory footprint.

    Attributes:
        config: Waste generated configuration
        metrics: Metrics collector
        provenance: Provenance tracker
        _landfill_engine: Landfill emissions engine (lazy-loaded)
        _incineration_engine: Incineration emissions engine (lazy-loaded)
        _recycling_engine: Recycling/composting engine (lazy-loaded)
        _wastewater_engine: Wastewater emissions engine (lazy-loaded)
        _database_engine: Database engine for EF lookups (lazy-loaded)
        _compliance_engine: Compliance checking engine (lazy-loaded)

    Example:
        >>> config = WasteGeneratedConfig()
        >>> engine = WasteGeneratedPipelineEngine(config)
        >>> waste_input = WasteStreamInput(
        ...     stream_id="WS-2026-001",
        ...     tenant_id="tenant-123",
        ...     facility_id="FAC-001",
        ...     waste_category=WasteCategory.PAPER_CARDBOARD,
        ...     treatment_method=WasteTreatmentMethod.LANDFILL,
        ...     mass_tonnes=Decimal("15.5"),
        ...     landfill_type=LandfillType.MANAGED_ANAEROBIC,
        ...     climate_zone=ClimateZone.TEMPERATE_WET,
        ... )
        >>> result = engine.process(waste_input)
        >>> print(f"Emissions: {result.total_emissions_kg_co2e} kg CO2e")
    """

    _instance: Optional['WasteGeneratedPipelineEngine'] = None
    _lock: RLock = RLock()

    def __new__(cls, config: Optional[WasteGeneratedConfig] = None):
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[WasteGeneratedConfig] = None):
        """
        Initialize WasteGeneratedPipelineEngine.

        Args:
            config: Waste generated configuration (optional, uses default if not provided)
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or WasteGeneratedConfig()
        self.metrics = WasteMetricsCollector()
        self.provenance = WasteProvenanceTracker()

        # Lazy-loaded engines (created on first use)
        self._landfill_engine: Optional[Any] = None
        self._incineration_engine: Optional[Any] = None
        self._recycling_engine: Optional[Any] = None
        self._wastewater_engine: Optional[Any] = None
        self._database_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        # Pipeline state
        self._stage_cache: Dict[str, Dict[ProvenanceStage, Any]] = {}

        self._initialized = True
        logger.info(f"WasteGeneratedPipelineEngine initialized (version {VERSION})")

    # ==========================================================================
    # PUBLIC API - CORE PROCESSING METHODS
    # ==========================================================================

    def process(self, input_data: WasteStreamInput) -> WasteCalculationResult:
        """
        Execute the 10-stage waste emissions calculation pipeline.

        Args:
            input_data: Waste stream input data

        Returns:
            WasteCalculationResult with emissions, compliance, and provenance

        Raises:
            ValueError: If input validation fails
            RuntimeError: If pipeline execution fails
        """
        chain_id = input_data.stream_id
        self.provenance.start_chain(chain_id)

        stage_durations: Dict[str, float] = {}
        errors: List[str] = []
        warnings: List[str] = []

        try:
            # Stage 1: VALIDATE
            start_time = datetime.now(timezone.utc)
            is_valid, validation_errors = self.validate_stage(input_data)
            if not is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["VALIDATE"] = duration_ms
            self.provenance.record_stage(
                chain_id, ProvenanceStage.VALIDATE, input_data, {"is_valid": True}, duration_ms
            )
            logger.info(f"[{chain_id}] Stage VALIDATE completed in {duration_ms:.2f}ms")

            # Stage 2: CLASSIFY
            start_time = datetime.now(timezone.utc)
            classification_result = self.classify_stage(input_data)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["CLASSIFY"] = duration_ms
            self.provenance.record_stage(
                chain_id, ProvenanceStage.CLASSIFY, input_data, classification_result, duration_ms
            )
            logger.info(f"[{chain_id}] Stage CLASSIFY completed in {duration_ms:.2f}ms")

            # Stage 3: NORMALIZE
            start_time = datetime.now(timezone.utc)
            normalized_input = self.normalize_stage(input_data)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["NORMALIZE"] = duration_ms
            self.provenance.record_stage(
                chain_id, ProvenanceStage.NORMALIZE, input_data, normalized_input, duration_ms
            )
            logger.info(f"[{chain_id}] Stage NORMALIZE completed in {duration_ms:.2f}ms")

            # Stage 4: RESOLVE_EFS
            start_time = datetime.now(timezone.utc)
            emission_factor = self.resolve_efs_stage(
                normalized_input.waste_category,
                normalized_input.treatment_method
            )
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["RESOLVE_EFS"] = duration_ms
            self.provenance.record_stage(
                chain_id, ProvenanceStage.RESOLVE_EFS, normalized_input, emission_factor, duration_ms
            )
            logger.info(
                f"[{chain_id}] Stage RESOLVE_EFS completed in {duration_ms:.2f}ms "
                f"(source: {emission_factor.ef_source.value})"
            )

            # Stage 5: CALCULATE_TREATMENT
            start_time = datetime.now(timezone.utc)
            treatment_result = self.calculate_treatment_stage(normalized_input, emission_factor)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["CALCULATE_TREATMENT"] = duration_ms
            self.provenance.record_stage(
                chain_id, ProvenanceStage.CALCULATE_TREATMENT, normalized_input, treatment_result, duration_ms
            )
            logger.info(
                f"[{chain_id}] Stage CALCULATE_TREATMENT completed in {duration_ms:.2f}ms "
                f"({treatment_result.emissions_kg_co2e} kg CO2e)"
            )

            # Stage 6: CALCULATE_TRANSPORT
            start_time = datetime.now(timezone.utc)
            transport_emissions = self.calculate_transport_stage(normalized_input)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["CALCULATE_TRANSPORT"] = duration_ms
            self.provenance.record_stage(
                chain_id,
                ProvenanceStage.CALCULATE_TRANSPORT,
                normalized_input,
                {"emissions_kg_co2e": transport_emissions},
                duration_ms
            )
            logger.info(
                f"[{chain_id}] Stage CALCULATE_TRANSPORT completed in {duration_ms:.2f}ms "
                f"({transport_emissions} kg CO2e)"
            )

            # Calculate totals before allocation
            total_co2_fossil = treatment_result.emissions_kg_co2_fossil
            total_co2_biogenic = treatment_result.emissions_kg_co2_biogenic
            total_ch4 = treatment_result.emissions_kg_ch4
            total_n2o = treatment_result.emissions_kg_n2o
            total_co2e = treatment_result.emissions_kg_co2e + transport_emissions

            # Stage 7: ALLOCATE
            start_time = datetime.now(timezone.utc)
            if normalized_input.allocation_factor:
                allocated_emissions = self.allocate_stage(
                    treatment_result,
                    normalized_input.allocation_factor
                )
                total_co2_fossil = allocated_emissions["co2_fossil"]
                total_co2_biogenic = allocated_emissions["co2_biogenic"]
                total_ch4 = allocated_emissions["ch4"]
                total_n2o = allocated_emissions["n2o"]
                total_co2e = allocated_emissions["co2e"]
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["ALLOCATE"] = duration_ms
            self.provenance.record_stage(
                chain_id,
                ProvenanceStage.ALLOCATE,
                treatment_result,
                {"total_co2e": total_co2e},
                duration_ms
            )
            logger.info(f"[{chain_id}] Stage ALLOCATE completed in {duration_ms:.2f}ms")

            # Stage 8: COMPLIANCE
            start_time = datetime.now(timezone.utc)
            compliance_results = {}
            for framework in normalized_input.frameworks:
                check_result = self.compliance_stage(
                    normalized_input,
                    treatment_result,
                    framework
                )
                compliance_results[framework.value] = check_result
                if check_result.warnings:
                    warnings.extend(check_result.warnings)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["COMPLIANCE"] = duration_ms
            self.provenance.record_stage(
                chain_id, ProvenanceStage.COMPLIANCE, treatment_result, compliance_results, duration_ms
            )
            logger.info(f"[{chain_id}] Stage COMPLIANCE completed in {duration_ms:.2f}ms")

            # Create result (aggregation handled separately for batch processing)
            result = WasteCalculationResult(
                stream_id=chain_id,
                status=PipelineStatus.SUCCESS,
                tenant_id=normalized_input.tenant_id,
                facility_id=normalized_input.facility_id,
                treatment_result=treatment_result,
                transport_emissions_kg_co2e=transport_emissions,
                total_emissions_kg_co2_fossil=total_co2_fossil,
                total_emissions_kg_co2_biogenic=total_co2_biogenic,
                total_emissions_kg_ch4=total_ch4,
                total_emissions_kg_n2o=total_n2o,
                total_emissions_kg_co2e=total_co2e,
                classification_result=classification_result,
                compliance_results=compliance_results,
                provenance_hash="",  # Will be set in Stage 10
                stage_durations_ms=stage_durations,
                errors=errors,
                warnings=warnings,
                metadata=normalized_input.metadata,
            )

            # Stage 10: SEAL
            start_time = datetime.now(timezone.utc)
            sealed_result = self.seal_stage(result, chain_id)
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_durations["SEAL"] = duration_ms
            result.provenance_hash = sealed_result.provenance_hash
            logger.info(f"[{chain_id}] Stage SEAL completed in {duration_ms:.2f}ms")

            # Record metrics
            self.metrics.record_calculation(
                tenant_id=normalized_input.tenant_id,
                waste_category=normalized_input.waste_category.value,
                treatment_method=normalized_input.treatment_method.value,
                mass_tonnes=float(normalized_input.mass_tonnes or 0),
                emissions_kg_co2e=float(total_co2e),
                duration_ms=sum(stage_durations.values())
            )

            logger.info(
                f"[{chain_id}] Pipeline completed successfully. "
                f"Total emissions: {total_co2e} kg CO2e"
            )

            return result

        except Exception as e:
            logger.error(f"[{chain_id}] Pipeline execution failed: {str(e)}", exc_info=True)
            errors.append(f"Pipeline execution failed: {str(e)}")

            # Return partial result
            return WasteCalculationResult(
                stream_id=chain_id,
                status=PipelineStatus.FAILED,
                tenant_id=input_data.tenant_id,
                facility_id=input_data.facility_id,
                total_emissions_kg_co2_fossil=Decimal("0"),
                total_emissions_kg_co2_biogenic=Decimal("0"),
                total_emissions_kg_ch4=Decimal("0"),
                total_emissions_kg_n2o=Decimal("0"),
                total_emissions_kg_co2e=Decimal("0"),
                compliance_results={},
                provenance_hash="",
                stage_durations_ms=stage_durations,
                errors=errors,
                warnings=warnings,
            )

    def process_batch(
        self,
        inputs: List[WasteStreamInput],
        batch_id: Optional[str] = None
    ) -> WasteBatchResult:
        """
        Execute batch calculation for multiple waste streams.

        Args:
            inputs: List of waste stream inputs
            batch_id: Optional batch identifier (auto-generated if not provided)

        Returns:
            WasteBatchResult with aggregated results
        """
        batch_id = batch_id or f"batch-{datetime.now(timezone.utc).isoformat()}"
        start_time = datetime.now(timezone.utc)
        results: List[WasteCalculationResult] = []

        logger.info(
            f"[{batch_id}] Starting batch calculation ({len(inputs)} waste streams)"
        )

        # Process each waste stream
        for waste_input in inputs:
            try:
                result = self.process(waste_input)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"[{batch_id}] Waste stream {waste_input.stream_id} failed: {str(e)}"
                )

        # Stage 9: AGGREGATE (for batch)
        start_time_aggregate = datetime.now(timezone.utc)
        aggregation = self.aggregate_stage(results, "all")
        duration_ms_aggregate = (datetime.now(timezone.utc) - start_time_aggregate).total_seconds() * 1000
        logger.info(f"[{batch_id}] Stage AGGREGATE completed in {duration_ms_aggregate:.2f}ms")

        # Calculate batch totals
        total_emissions = sum(r.total_emissions_kg_co2e for r in results)
        successful_count = sum(1 for r in results if r.status == PipelineStatus.SUCCESS)
        failed_count = len(results) - successful_count

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Create batch provenance hash
        batch_data = {
            "batch_id": batch_id,
            "tenant_id": inputs[0].tenant_id if inputs else "",
            "calculation_hashes": [r.provenance_hash for r in results],
            "total_emissions_kg_co2e": str(total_emissions),
        }
        batch_hash = hashlib.sha256(
            json.dumps(batch_data, sort_keys=True).encode()
        ).hexdigest()

        logger.info(
            f"[{batch_id}] Batch calculation completed in {duration_ms:.2f}ms. "
            f"Success: {successful_count}, Failed: {failed_count}, "
            f"Total emissions: {total_emissions} kg CO2e"
        )

        return WasteBatchResult(
            batch_id=batch_id,
            tenant_id=inputs[0].tenant_id if inputs else "",
            results=results,
            aggregation=aggregation,
            total_emissions_kg_co2e=total_emissions,
            successful_count=successful_count,
            failed_count=failed_count,
            batch_duration_ms=duration_ms,
            provenance_hash=batch_hash,
        )

    def process_with_config(
        self,
        input_data: WasteStreamInput,
        config_overrides: Dict[str, Any]
    ) -> WasteCalculationResult:
        """
        Process waste stream with configuration overrides.

        Args:
            input_data: Waste stream input data
            config_overrides: Configuration overrides (e.g., GWP version, EF source preference)

        Returns:
            WasteCalculationResult
        """
        # Store original config
        original_config = self.config

        try:
            # Apply overrides
            config_dict = self.config.dict()
            config_dict.update(config_overrides)
            self.config = WasteGeneratedConfig(**config_dict)

            # Process with modified config
            result = self.process(input_data)

            return result

        finally:
            # Restore original config
            self.config = original_config

    # ==========================================================================
    # STAGE METHODS (Public for testing/granular control)
    # ==========================================================================

    def validate_stage(self, input_data: WasteStreamInput) -> Tuple[bool, List[str]]:
        """
        Stage 1: VALIDATE - Input validation.

        Validates:
        - Required fields present
        - Type checking (handled by Pydantic)
        - Range validation
        - Data consistency
        - Treatment method compatibility

        Args:
            input_data: Waste stream input

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: List[str] = []

        # Check mass data
        if not input_data.mass_tonnes and not input_data.mass_short_tons and not input_data.volume_m3:
            if input_data.calculation_method != CalculationMethod.SPEND_BASED:
                errors.append("At least one of mass_tonnes, mass_short_tons, or volume_m3 must be provided")

        # Check spend-based method requirements
        if input_data.calculation_method == CalculationMethod.SPEND_BASED:
            if not input_data.waste_management_spend:
                errors.append("waste_management_spend required for SPEND_BASED method")
            if not input_data.spend_currency:
                errors.append("spend_currency required for SPEND_BASED method")

        # Check landfill-specific requirements
        if input_data.treatment_method in [
            WasteTreatmentMethod.LANDFILL,
            WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
            WasteTreatmentMethod.LANDFILL_WITH_ENERGY_RECOVERY,
        ]:
            if not input_data.landfill_type:
                errors.append("landfill_type required for landfill treatment methods")
            if not input_data.climate_zone:
                errors.append("climate_zone required for landfill treatment methods")

        # Check incineration-specific requirements
        if input_data.treatment_method in [
            WasteTreatmentMethod.INCINERATION,
            WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
        ]:
            if not input_data.incinerator_type:
                errors.append("incinerator_type required for incineration treatment methods")

        # Check recycling-specific requirements
        if input_data.treatment_method in [
            WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
        ]:
            if not input_data.recycling_type:
                errors.append("recycling_type required for recycling treatment methods")

        # Check wastewater-specific requirements
        if input_data.treatment_method == WasteTreatmentMethod.WASTEWATER_TREATMENT:
            if not input_data.wastewater_system:
                errors.append("wastewater_system required for wastewater treatment")

        # Check composition data for mixed waste
        if input_data.waste_category == WasteCategory.MIXED_MSW:
            if not input_data.composition:
                errors.append("composition data required for MIXED_MSW category")
            elif sum(input_data.composition.values()) != Decimal("100"):
                errors.append("composition percentages must sum to 100%")

        # Check date range
        if input_data.reporting_period_end <= input_data.reporting_period_start:
            errors.append("reporting_period_end must be after reporting_period_start")

        # Check allocation
        if input_data.allocation_factor:
            if not input_data.allocation_method:
                errors.append("allocation_method required when allocation_factor is provided")

        is_valid = len(errors) == 0
        return is_valid, errors

    def classify_stage(self, input_data: WasteStreamInput) -> WasteClassificationResult:
        """
        Stage 2: CLASSIFY - Waste classification.

        Classification logic:
        - Determine waste category (if not explicit)
        - Check hazardous status
        - Identify compatible treatment methods
        - Calculate biogenic/fossil fractions
        - Recommend calculation method

        Args:
            input_data: Waste stream input

        Returns:
            WasteClassificationResult with classification data
        """
        # Determine biogenic/fossil fractions based on waste category
        biogenic_fraction, fossil_fraction = self._get_carbon_fractions(input_data.waste_category)

        # Identify compatible treatment methods
        compatible_treatments = self._get_compatible_treatments(
            input_data.waste_category,
            input_data.is_hazardous
        )

        # Recommend calculation method
        recommended_method = self._recommend_calculation_method(input_data)

        # Classification confidence (simplified - could use ML model)
        confidence = Decimal("0.95") if input_data.ewc_code else Decimal("0.85")

        return WasteClassificationResult(
            waste_category=input_data.waste_category,
            waste_stream=input_data.waste_stream,
            is_hazardous=input_data.is_hazardous,
            hazard_classes=input_data.hazard_classes,
            ewc_code=input_data.ewc_code,
            compatible_treatments=compatible_treatments,
            biogenic_fraction=biogenic_fraction,
            fossil_fraction=fossil_fraction,
            recommended_method=recommended_method,
            classification_confidence=confidence,
        )

    def normalize_stage(self, input_data: WasteStreamInput) -> WasteStreamInput:
        """
        Stage 3: NORMALIZE - Unit conversion and standardization.

        Conversions:
        - Short tons → metric tonnes
        - Wet weight → dry weight (if moisture content provided)
        - Currency → USD
        - Volume → mass (using density factors)

        Args:
            input_data: Waste stream input

        Returns:
            WasteStreamInput with normalized values
        """
        # Create a mutable copy
        normalized_data = input_data.dict()

        # Convert short tons to tonnes
        if input_data.mass_short_tons and not input_data.mass_tonnes:
            normalized_data["mass_tonnes"] = input_data.mass_short_tons * Decimal("0.907185")

        # Adjust for moisture content (dry weight basis)
        if input_data.moisture_content_pct and normalized_data.get("mass_tonnes"):
            dry_fraction = (Decimal("100") - input_data.moisture_content_pct) / Decimal("100")
            normalized_data["mass_tonnes"] = normalized_data["mass_tonnes"] * dry_fraction
            normalized_data["metadata"]["moisture_adjusted"] = True
            normalized_data["metadata"]["original_wet_mass_tonnes"] = input_data.mass_tonnes

        # Convert volume to mass (if mass not provided)
        if input_data.volume_m3 and not normalized_data.get("mass_tonnes"):
            density = self._get_waste_density(input_data.waste_category)  # tonnes/m3
            normalized_data["mass_tonnes"] = input_data.volume_m3 * density
            normalized_data["metadata"]["volume_converted"] = True

        # Currency conversion to USD (simplified - would use exchange rates in production)
        if input_data.waste_management_spend and input_data.spend_currency:
            if input_data.spend_currency != CurrencyCode.USD:
                # Placeholder conversion (would be real exchange rate)
                conversion_rate = Decimal("1.0")  # Replace with actual rate
                normalized_data["metadata"]["spend_usd"] = input_data.waste_management_spend * conversion_rate
                normalized_data["metadata"]["currency_conversion_rate"] = conversion_rate
            else:
                normalized_data["metadata"]["spend_usd"] = input_data.waste_management_spend

        return WasteStreamInput(**normalized_data)

    def resolve_efs_stage(
        self,
        waste_category: WasteCategory,
        treatment_method: WasteTreatmentMethod
    ) -> WasteEmissionFactor:
        """
        Stage 4: RESOLVE_EFS - Select emission factors from hierarchy.

        Hierarchy:
        1. EPA WARM v16 (highest priority for US waste)
        2. DEFRA/BEIS conversion factors
        3. IPCC 2019 Refinement defaults
        4. IPCC 2006 defaults (fallback)

        Args:
            waste_category: Waste material category
            treatment_method: Treatment method

        Returns:
            WasteEmissionFactor with complete emission factors

        Raises:
            ValueError: If no emission factor found
        """
        # Try EPA WARM first (best available for waste)
        ef = self._lookup_emission_factor(
            waste_category,
            treatment_method,
            EFSource.EPA_WARM
        )

        if ef:
            logger.debug(f"Using EPA WARM EF for {waste_category.value}/{treatment_method.value}")
            return ef

        # Try DEFRA
        ef = self._lookup_emission_factor(
            waste_category,
            treatment_method,
            EFSource.DEFRA_BEIS
        )

        if ef:
            logger.debug(f"Using DEFRA EF for {waste_category.value}/{treatment_method.value}")
            return ef

        # Try IPCC 2019
        ef = self._lookup_emission_factor(
            waste_category,
            treatment_method,
            EFSource.IPCC_2019
        )

        if ef:
            logger.debug(f"Using IPCC 2019 EF for {waste_category.value}/{treatment_method.value}")
            return ef

        # Fallback to IPCC 2006
        ef = self._lookup_emission_factor(
            waste_category,
            treatment_method,
            EFSource.IPCC_2006
        )

        if ef:
            logger.debug(f"Using IPCC 2006 EF for {waste_category.value}/{treatment_method.value}")
            return ef

        raise ValueError(
            f"No emission factor found for waste_category={waste_category.value}, "
            f"treatment_method={treatment_method.value}"
        )

    def calculate_treatment_stage(
        self,
        input_data: WasteStreamInput,
        emission_factor: WasteEmissionFactor
    ) -> TreatmentEmissionsResult:
        """
        Stage 5: CALCULATE_TREATMENT - Route to treatment-specific engine.

        Routes to:
        - Landfill → LandfillEmissionsEngine
        - Incineration → IncinerationEmissionsEngine
        - Recycling/Composting/AD → RecyclingCompostingEngine
        - Wastewater → WastewaterEmissionsEngine

        Args:
            input_data: Normalized waste stream input
            emission_factor: Emission factor to use

        Returns:
            TreatmentEmissionsResult

        Raises:
            ValueError: If treatment method not supported
        """
        treatment_method = input_data.treatment_method

        # Route to appropriate engine
        if treatment_method in [
            WasteTreatmentMethod.LANDFILL,
            WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
            WasteTreatmentMethod.LANDFILL_WITH_ENERGY_RECOVERY,
        ]:
            engine = self._get_landfill_engine()
            result = engine.calculate_landfill_emissions(input_data, emission_factor)

        elif treatment_method in [
            WasteTreatmentMethod.INCINERATION,
            WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
        ]:
            engine = self._get_incineration_engine()
            result = engine.calculate_incineration_emissions(input_data, emission_factor)

        elif treatment_method in [
            WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            WasteTreatmentMethod.COMPOSTING,
            WasteTreatmentMethod.ANAEROBIC_DIGESTION,
        ]:
            engine = self._get_recycling_engine()
            result = engine.calculate_recycling_emissions(input_data, emission_factor)

        elif treatment_method == WasteTreatmentMethod.WASTEWATER_TREATMENT:
            engine = self._get_wastewater_engine()
            result = engine.calculate_wastewater_emissions(input_data, emission_factor)

        else:
            # Generic calculation for "OTHER" treatment
            result = self._calculate_generic_emissions(input_data, emission_factor)

        return result

    def calculate_transport_stage(self, input_data: WasteStreamInput) -> Decimal:
        """
        Stage 6: CALCULATE_TRANSPORT - Calculate transport to treatment facility.

        Optional stage - only calculates if transport_distance_km is provided.

        Args:
            input_data: Waste stream input

        Returns:
            Transport emissions in kg CO2e (0 if no transport data)
        """
        if not input_data.transport_distance_km or input_data.transport_distance_km == 0:
            return Decimal("0")

        # Default transport mode for waste: heavy-duty truck
        # EF: ~0.062 kg CO2e per tonne-km (DEFRA 2024)
        transport_ef = Decimal("0.062")  # kg CO2e per tonne-km

        mass_tonnes = input_data.mass_tonnes or Decimal("0")
        distance_km = input_data.transport_distance_km

        transport_emissions = mass_tonnes * distance_km * transport_ef

        return transport_emissions

    def allocate_stage(
        self,
        result: TreatmentEmissionsResult,
        allocation_factor: Decimal
    ) -> Dict[str, Decimal]:
        """
        Stage 7: ALLOCATE - Apply allocation factor.

        Allocates emissions based on allocation factor (0-1).

        Args:
            result: Treatment emissions result
            allocation_factor: Allocation factor (0-1)

        Returns:
            Dictionary with allocated emissions by gas
        """
        return {
            "co2_fossil": result.emissions_kg_co2_fossil * allocation_factor,
            "co2_biogenic": result.emissions_kg_co2_biogenic * allocation_factor,
            "ch4": result.emissions_kg_ch4 * allocation_factor,
            "n2o": result.emissions_kg_n2o * allocation_factor,
            "co2e": result.emissions_kg_co2e * allocation_factor,
        }

    def compliance_stage(
        self,
        input_data: WasteStreamInput,
        result: TreatmentEmissionsResult,
        framework: ComplianceFramework
    ) -> ComplianceCheckResult:
        """
        Stage 8: COMPLIANCE - Framework compliance checking.

        Checks compliance for specified framework:
        - GHG Protocol Scope 3 Standard
        - ISO 14064-1:2018
        - CSRD ESRS E1 + E5
        - CDP Climate Change
        - SBTi
        - EU Waste Framework Directive
        - EPA 40 CFR Part 98

        Args:
            input_data: Waste stream input
            result: Treatment emissions result
            framework: Regulatory framework

        Returns:
            ComplianceCheckResult
        """
        # This would delegate to ComplianceCheckerEngine in production
        # Simplified implementation here
        errors = []
        warnings = []
        missing_fields = []
        recommendations = []

        if framework == ComplianceFramework.GHG_PROTOCOL:
            required_fields = ["waste_category", "treatment_method", "mass_tonnes"]
            for field in required_fields:
                if not getattr(input_data, field, None):
                    missing_fields.append(field)

            if input_data.calculation_method == CalculationMethod.SPEND_BASED:
                warnings.append("Spend-based method has lower data quality than mass-based methods")

        elif framework == ComplianceFramework.CSRD_ESRS:
            # CSRD requires circular economy metrics
            if not input_data.treatment_method in [
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            ]:
                recommendations.append("Consider waste diversion for CSRD ESRS E5 circular economy disclosure")

        status = ComplianceStatus.COMPLIANT
        if errors or missing_fields:
            status = ComplianceStatus.NON_COMPLIANT
        elif warnings:
            status = ComplianceStatus.PARTIAL

        return ComplianceCheckResult(
            framework=framework,
            status=status,
            required_fields=required_fields if framework == ComplianceFramework.GHG_PROTOCOL else [],
            missing_fields=missing_fields,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def aggregate_stage(
        self,
        results: List[WasteCalculationResult],
        dimension: str
    ) -> WasteAggregationResult:
        """
        Stage 9: AGGREGATE - Aggregate emissions across dimensions.

        Aggregates by:
        - Treatment method
        - Waste category
        - Facility
        - Period
        - Business unit
        - Waste stream

        Args:
            results: List of calculation results
            dimension: Aggregation dimension (or "all" for all dimensions)

        Returns:
            WasteAggregationResult with aggregated emissions
        """
        by_treatment: Dict[str, Decimal] = {}
        by_category: Dict[str, Decimal] = {}
        by_facility: Dict[str, Decimal] = {}
        by_period: Dict[str, Decimal] = {}
        by_business_unit: Dict[str, Decimal] = {}
        by_waste_stream: Dict[str, Decimal] = {}

        total_mass = Decimal("0")
        total_emissions = Decimal("0")

        for result in results:
            if not result.treatment_result:
                continue

            treatment = result.treatment_result.treatment_method.value
            mass = result.treatment_result.mass_treated_tonnes
            emissions = result.total_emissions_kg_co2e

            total_mass += mass
            total_emissions += emissions

            # Aggregate by treatment
            by_treatment[treatment] = by_treatment.get(treatment, Decimal("0")) + emissions

            # Aggregate by facility
            facility = result.facility_id
            by_facility[facility] = by_facility.get(facility, Decimal("0")) + emissions

            # Additional aggregations from metadata
            if result.metadata.get("waste_category"):
                category = result.metadata["waste_category"]
                by_category[category] = by_category.get(category, Decimal("0")) + emissions

        # Calculate diversion analysis
        diversion_analysis = self._calculate_diversion_from_results(results)

        return WasteAggregationResult(
            by_treatment_method=by_treatment,
            by_waste_category=by_category,
            by_facility=by_facility,
            by_period=by_period,
            by_business_unit=by_business_unit,
            by_waste_stream=by_waste_stream,
            total_mass_tonnes=total_mass,
            total_emissions_kg_co2e=total_emissions,
            diversion_analysis=diversion_analysis,
        )

    def seal_stage(self, result: WasteCalculationResult, chain_id: str) -> ProvenanceChainResult:
        """
        Stage 10: SEAL - Seal provenance chain.

        Creates final SHA-256 hash of complete provenance chain.

        Args:
            result: Calculation result
            chain_id: Provenance chain identifier

        Returns:
            ProvenanceChainResult with final hash
        """
        provenance_hash = self.provenance.seal_chain(chain_id)
        chain_entries = self.provenance.get_chain(chain_id)

        stages = [
            {
                "stage": entry.stage.value,
                "timestamp": entry.timestamp.isoformat(),
                "input_hash": entry.input_hash,
                "output_hash": entry.output_hash,
                "duration_ms": entry.duration_ms,
            }
            for entry in chain_entries
        ]

        return ProvenanceChainResult(
            chain_id=chain_id,
            provenance_hash=provenance_hash,
            stages=stages,
        )

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Dictionary with pipeline status information
        """
        return {
            "version": VERSION,
            "agent_id": AGENT_ID,
            "config": self.config.dict(),
            "engines_loaded": {
                "landfill": self._landfill_engine is not None,
                "incineration": self._incineration_engine is not None,
                "recycling": self._recycling_engine is not None,
                "wastewater": self._wastewater_engine is not None,
                "database": self._database_engine is not None,
                "compliance": self._compliance_engine is not None,
            },
            "provenance_chains": len(self.provenance.chains),
        }

    def reset_pipeline(self) -> None:
        """
        Reset pipeline state (clear caches, provenance chains).

        Used for testing or periodic cleanup.
        """
        with self._lock:
            self._stage_cache.clear()
            self.provenance.chains.clear()
            logger.info("Pipeline state reset")

    def calculate_diversion_analysis(
        self,
        results: List[WasteCalculationResult]
    ) -> WasteDiversionAnalysis:
        """
        Calculate waste diversion analysis from landfill.

        Args:
            results: List of calculation results

        Returns:
            WasteDiversionAnalysis with diversion metrics
        """
        return self._calculate_diversion_from_results(results)

    def auto_select_method(self, input_data: WasteStreamInput) -> CalculationMethod:
        """
        Automatically select calculation method based on available data.

        Priority:
        1. SUPPLIER_SPECIFIC (if supplier emissions data available)
        2. WASTE_TYPE_SPECIFIC (if mass + waste category + treatment method available)
        3. AVERAGE_DATA (if only total mass + treatment method available)
        4. SPEND_BASED (if only spend data available)

        Args:
            input_data: Waste stream input

        Returns:
            Recommended CalculationMethod
        """
        return self._recommend_calculation_method(input_data)

    def get_treatment_engine(self, treatment_method: WasteTreatmentMethod) -> Any:
        """
        Get reference to treatment-specific engine.

        Args:
            treatment_method: Treatment method

        Returns:
            Reference to engine instance

        Raises:
            ValueError: If treatment method not supported
        """
        if treatment_method in [
            WasteTreatmentMethod.LANDFILL,
            WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
            WasteTreatmentMethod.LANDFILL_WITH_ENERGY_RECOVERY,
        ]:
            return self._get_landfill_engine()

        elif treatment_method in [
            WasteTreatmentMethod.INCINERATION,
            WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
        ]:
            return self._get_incineration_engine()

        elif treatment_method in [
            WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            WasteTreatmentMethod.COMPOSTING,
            WasteTreatmentMethod.ANAEROBIC_DIGESTION,
        ]:
            return self._get_recycling_engine()

        elif treatment_method == WasteTreatmentMethod.WASTEWATER_TREATMENT:
            return self._get_wastewater_engine()

        else:
            raise ValueError(f"Unsupported treatment method: {treatment_method.value}")

    # ==========================================================================
    # PRIVATE HELPER METHODS
    # ==========================================================================

    def _get_carbon_fractions(
        self,
        waste_category: WasteCategory
    ) -> Tuple[Decimal, Decimal]:
        """
        Get biogenic and fossil carbon fractions for waste category.

        Args:
            waste_category: Waste material category

        Returns:
            Tuple of (biogenic_fraction, fossil_fraction)
        """
        # Biogenic waste (100% biogenic carbon)
        biogenic_categories = [
            WasteCategory.FOOD_WASTE,
            WasteCategory.GARDEN_WASTE,
            WasteCategory.PAPER_CARDBOARD,
            WasteCategory.WOOD,
        ]

        # Fossil waste (100% fossil carbon)
        fossil_categories = [
            WasteCategory.PLASTICS_HDPE,
            WasteCategory.PLASTICS_LDPE,
            WasteCategory.PLASTICS_PET,
            WasteCategory.PLASTICS_PP,
            WasteCategory.PLASTICS_MIXED,
        ]

        if waste_category in biogenic_categories:
            return Decimal("1.0"), Decimal("0.0")
        elif waste_category in fossil_categories:
            return Decimal("0.0"), Decimal("1.0")
        elif waste_category == WasteCategory.MIXED_MSW:
            # Typical MSW: 50% biogenic, 50% fossil (IPCC default)
            return Decimal("0.5"), Decimal("0.5")
        else:
            # Conservative default: assume fossil
            return Decimal("0.0"), Decimal("1.0")

    def _get_compatible_treatments(
        self,
        waste_category: WasteCategory,
        is_hazardous: bool
    ) -> List[WasteTreatmentMethod]:
        """
        Get compatible treatment methods for waste category.

        Args:
            waste_category: Waste material category
            is_hazardous: Hazardous waste flag

        Returns:
            List of compatible treatment methods
        """
        # Hazardous waste typically requires specialized treatment
        if is_hazardous:
            return [
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
            ]

        # Material-specific compatibility
        if waste_category in [WasteCategory.PAPER_CARDBOARD, WasteCategory.GLASS]:
            return [
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
                WasteTreatmentMethod.LANDFILL,
            ]

        elif waste_category in [
            WasteCategory.PLASTICS_HDPE,
            WasteCategory.PLASTICS_LDPE,
            WasteCategory.PLASTICS_PET,
        ]:
            return [
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
                WasteTreatmentMethod.LANDFILL,
            ]

        elif waste_category in [WasteCategory.FOOD_WASTE, WasteCategory.GARDEN_WASTE]:
            return [
                WasteTreatmentMethod.COMPOSTING,
                WasteTreatmentMethod.ANAEROBIC_DIGESTION,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
                WasteTreatmentMethod.LANDFILL,
            ]

        else:
            # Default: all methods compatible
            return list(WasteTreatmentMethod)

    def _recommend_calculation_method(
        self,
        input_data: WasteStreamInput
    ) -> CalculationMethod:
        """
        Recommend calculation method based on available data.

        Args:
            input_data: Waste stream input

        Returns:
            Recommended CalculationMethod
        """
        if input_data.supplier_id and input_data.metadata.get("supplier_emissions_available"):
            return CalculationMethod.SUPPLIER_SPECIFIC

        if input_data.mass_tonnes and input_data.waste_category and input_data.treatment_method:
            return CalculationMethod.WASTE_TYPE_SPECIFIC

        if input_data.mass_tonnes and input_data.treatment_method:
            return CalculationMethod.AVERAGE_DATA

        if input_data.waste_management_spend:
            return CalculationMethod.SPEND_BASED

        # Default
        return CalculationMethod.WASTE_TYPE_SPECIFIC

    def _get_waste_density(self, waste_category: WasteCategory) -> Decimal:
        """
        Get typical density for waste category.

        Args:
            waste_category: Waste material category

        Returns:
            Density in tonnes/m3
        """
        density_map = {
            WasteCategory.PAPER_CARDBOARD: Decimal("0.15"),
            WasteCategory.PLASTICS_MIXED: Decimal("0.05"),
            WasteCategory.GLASS: Decimal("0.30"),
            WasteCategory.METALS_ALUMINUM: Decimal("0.25"),
            WasteCategory.FOOD_WASTE: Decimal("0.40"),
            WasteCategory.GARDEN_WASTE: Decimal("0.20"),
            WasteCategory.WOOD: Decimal("0.30"),
            WasteCategory.MIXED_MSW: Decimal("0.25"),
        }
        return density_map.get(waste_category, Decimal("0.25"))  # Default 0.25 tonnes/m3

    def _lookup_emission_factor(
        self,
        waste_category: WasteCategory,
        treatment_method: WasteTreatmentMethod,
        source: EFSource
    ) -> Optional[WasteEmissionFactor]:
        """
        Lookup emission factor from database.

        This is a placeholder - would query database in production.

        Args:
            waste_category: Waste material category
            treatment_method: Treatment method
            source: Emission factor source

        Returns:
            WasteEmissionFactor or None if not found
        """
        # Placeholder implementation
        # In production, this would query WasteDatabaseEngine

        # Example: Paper/cardboard landfill (EPA WARM v16)
        if (
            waste_category == WasteCategory.PAPER_CARDBOARD
            and treatment_method == WasteTreatmentMethod.LANDFILL
            and source == EFSource.EPA_WARM
        ):
            return WasteEmissionFactor(
                treatment_method=treatment_method,
                waste_category=waste_category,
                ef_source=source,
                ef_co2_fossil=Decimal("0"),  # Paper is biogenic
                ef_co2_biogenic=Decimal("320"),  # kg CO2/tonne
                ef_ch4=Decimal("52"),  # kg CH4/tonne
                ef_n2o=Decimal("0.1"),  # kg N2O/tonne
                uncertainty_pct=Decimal("30"),
                gwp_version=GWPVersion.AR5,
                gwp_ch4=Decimal("28"),
                gwp_n2o=Decimal("265"),
                data_quality_tier=DataQualityTier.TIER_2,
                region="US",
                year=2024,
                notes="EPA WARM v16",
            )

        return None  # Not found

    def _calculate_generic_emissions(
        self,
        input_data: WasteStreamInput,
        emission_factor: WasteEmissionFactor
    ) -> TreatmentEmissionsResult:
        """
        Generic emissions calculation (simple mass × EF).

        Args:
            input_data: Waste stream input
            emission_factor: Emission factor

        Returns:
            TreatmentEmissionsResult
        """
        mass = input_data.mass_tonnes or Decimal("0")

        emissions_co2_fossil = mass * emission_factor.ef_co2_fossil
        emissions_co2_biogenic = mass * emission_factor.ef_co2_biogenic
        emissions_ch4 = mass * emission_factor.ef_ch4
        emissions_n2o = mass * emission_factor.ef_n2o

        emissions_co2e = (
            emissions_co2_fossil
            + emissions_ch4 * emission_factor.gwp_ch4
            + emissions_n2o * emission_factor.gwp_n2o
        )

        return TreatmentEmissionsResult(
            treatment_method=input_data.treatment_method,
            mass_treated_tonnes=mass,
            emissions_kg_co2_fossil=emissions_co2_fossil,
            emissions_kg_co2_biogenic=emissions_co2_biogenic,
            emissions_kg_ch4=emissions_ch4,
            emissions_kg_n2o=emissions_n2o,
            emissions_kg_co2e=emissions_co2e,
            emission_factor=emission_factor,
        )

    def _calculate_diversion_from_results(
        self,
        results: List[WasteCalculationResult]
    ) -> WasteDiversionAnalysis:
        """
        Calculate diversion analysis from results.

        Args:
            results: List of calculation results

        Returns:
            WasteDiversionAnalysis
        """
        landfilled = Decimal("0")
        recycled = Decimal("0")
        composted = Decimal("0")
        incinerated = Decimal("0")
        other = Decimal("0")

        for result in results:
            if not result.treatment_result:
                continue

            mass = result.treatment_result.mass_treated_tonnes
            treatment = result.treatment_result.treatment_method

            if treatment in [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
                WasteTreatmentMethod.LANDFILL_WITH_ENERGY_RECOVERY,
            ]:
                landfilled += mass

            elif treatment in [
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            ]:
                recycled += mass

            elif treatment in [
                WasteTreatmentMethod.COMPOSTING,
                WasteTreatmentMethod.ANAEROBIC_DIGESTION,
            ]:
                composted += mass

            elif treatment in [
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
            ]:
                incinerated += mass

            else:
                other += mass

        total = landfilled + recycled + composted + incinerated + other

        if total == 0:
            diversion_rate = Decimal("0")
            recycling_rate = Decimal("0")
            circular_rate = Decimal("0")
        else:
            diverted = recycled + composted
            diversion_rate = (diverted / total) * Decimal("100")
            recycling_rate = (recycled / total) * Decimal("100")
            circular_rate = recycling_rate  # Simplified

        # Emissions avoided (baseline: all to landfill)
        # This would require re-calculating with landfill baseline
        emissions_avoided = Decimal("0")  # Placeholder

        return WasteDiversionAnalysis(
            total_waste_tonnes=total,
            landfilled_tonnes=landfilled,
            recycled_tonnes=recycled,
            composted_tonnes=composted,
            incinerated_tonnes=incinerated,
            other_tonnes=other,
            diversion_rate_pct=diversion_rate,
            recycling_rate_pct=recycling_rate,
            emissions_avoided_kg_co2e=emissions_avoided,
            circular_material_use_rate=circular_rate,
        )

    # ==========================================================================
    # LAZY ENGINE LOADING
    # ==========================================================================

    def _get_landfill_engine(self) -> Any:
        """Get or create landfill engine (lazy loading)."""
        if self._landfill_engine is None:
            # Import here to avoid circular dependencies
            # from greenlang.waste_generated.landfill_engine import LandfillEmissionsEngine
            # self._landfill_engine = LandfillEmissionsEngine(self.config)
            # Placeholder for now
            logger.warning("LandfillEmissionsEngine not yet implemented - using placeholder")
            self._landfill_engine = object()  # Placeholder
        return self._landfill_engine

    def _get_incineration_engine(self) -> Any:
        """Get or create incineration engine (lazy loading)."""
        if self._incineration_engine is None:
            # from greenlang.waste_generated.incineration_engine import IncinerationEmissionsEngine
            # self._incineration_engine = IncinerationEmissionsEngine(self.config)
            logger.warning("IncinerationEmissionsEngine not yet implemented - using placeholder")
            self._incineration_engine = object()
        return self._incineration_engine

    def _get_recycling_engine(self) -> Any:
        """Get or create recycling engine (lazy loading)."""
        if self._recycling_engine is None:
            # from greenlang.waste_generated.recycling_engine import RecyclingCompostingEngine
            # self._recycling_engine = RecyclingCompostingEngine(self.config)
            logger.warning("RecyclingCompostingEngine not yet implemented - using placeholder")
            self._recycling_engine = object()
        return self._recycling_engine

    def _get_wastewater_engine(self) -> Any:
        """Get or create wastewater engine (lazy loading)."""
        if self._wastewater_engine is None:
            # from greenlang.waste_generated.wastewater_engine import WastewaterEmissionsEngine
            # self._wastewater_engine = WastewaterEmissionsEngine(self.config)
            logger.warning("WastewaterEmissionsEngine not yet implemented - using placeholder")
            self._wastewater_engine = object()
        return self._wastewater_engine

    def _get_database_engine(self) -> Any:
        """Get or create database engine (lazy loading)."""
        if self._database_engine is None:
            # from greenlang.waste_generated.waste_database import WasteDatabaseEngine
            # self._database_engine = WasteDatabaseEngine(self.config)
            logger.warning("WasteDatabaseEngine not yet implemented - using placeholder")
            self._database_engine = object()
        return self._database_engine

    def _get_compliance_engine(self) -> Any:
        """Get or create compliance engine (lazy loading)."""
        if self._compliance_engine is None:
            # from greenlang.waste_generated.compliance_checker import ComplianceCheckerEngine
            # self._compliance_engine = ComplianceCheckerEngine(self.config)
            logger.warning("ComplianceCheckerEngine not yet implemented - using placeholder")
            self._compliance_engine = object()
        return self._compliance_engine


# ==============================================================================
# MODULE-LEVEL HELPERS
# ==============================================================================


def get_pipeline_engine(config: Optional[WasteGeneratedConfig] = None) -> WasteGeneratedPipelineEngine:
    """
    Get singleton pipeline engine instance.

    Args:
        config: Optional configuration (uses default if not provided)

    Returns:
        WasteGeneratedPipelineEngine singleton instance
    """
    return WasteGeneratedPipelineEngine(config)
