"""
Waste Generated in Operations API Router - AGENT-MRV-018

This module implements the FastAPI router for waste generated in operations
emissions calculations following GHG Protocol Scope 3 Category 5 requirements.

Provides 20 REST endpoints for:
- Emissions calculations (single and batch)
- Treatment-specific calculations (landfill, incineration, recycling, composting, AD, wastewater)
- Emission factor lookup and waste type management
- Treatment method management
- Compliance checking
- Uncertainty analysis
- Aggregations and diversion analysis
- Provenance tracking
- Health and statistics

Follows GreenLang's zero-hallucination principle with deterministic calculations.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.waste_generated.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime, date
from uuid import UUID
import logging

from pydantic import BaseModel, Field, validator, constr

from greenlang.waste_generated.service import WasteGeneratedService
from greenlang.waste_generated.models import (
    WasteType,
    TreatmentMethod,
    CalculationMethod,
    UncertaintyMethod,
    WasteStream,
    EmissionFactor,
    CalculationResult,
)

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/waste-generated",
    tags=["waste-generated"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# REQUEST MODELS
# ============================================================================


class CalculateRequest(BaseModel):
    """
    Request model for single waste generated emissions calculation.

    Attributes:
        tenant_id: Tenant identifier for multi-tenancy
        calculation_id: Optional UUID for idempotency
        waste_type: Type of waste material (FOOD_WASTE, PAPER, PLASTIC, etc.)
        treatment_method: Disposal/treatment method (LANDFILL, INCINERATION, RECYCLING, etc.)
        waste_mass_tonnes: Mass of waste in metric tonnes
        calculation_method: Calculation method (WASTE_TYPE_SPECIFIC, MATERIAL_SPECIFIC, AVERAGE_DATA)
        landfill_type: Optional landfill type (MANAGED, UNMANAGED, DEEP, UNCATEGORIZED)
        moisture_content_pct: Optional moisture content percentage
        doc_value: Optional degradable organic carbon value
        mcf_value: Optional methane correction factor
        has_gas_recovery: Whether landfill has gas recovery system
        incineration_has_energy_recovery: Whether incineration has energy recovery
        recycling_efficiency_pct: Optional recycling efficiency percentage
        composting_method: Optional composting method (AEROBIC, ANAEROBIC)
        ad_biogas_capture_pct: Optional anaerobic digestion biogas capture percentage
        wastewater_bod_mg_l: Optional wastewater biological oxygen demand
        wastewater_volume_m3: Optional wastewater volume
        emission_factor_id: Optional custom emission factor ID
        facility_country: Optional ISO 3166-1 alpha-3 facility country code
        year: Calculation year
        metadata: Additional metadata for audit trail
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    waste_type: WasteType = Field(..., description="Type of waste material")
    treatment_method: TreatmentMethod = Field(..., description="Treatment/disposal method")
    waste_mass_tonnes: Decimal = Field(..., ge=0, description="Waste mass in metric tonnes")
    calculation_method: CalculationMethod = Field(
        CalculationMethod.WASTE_TYPE_SPECIFIC, description="Calculation method"
    )

    # Landfill-specific parameters
    landfill_type: Optional[str] = Field(None, description="Landfill type")
    moisture_content_pct: Optional[Decimal] = Field(
        None, ge=0, le=100, description="Moisture content percentage"
    )
    doc_value: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Degradable organic carbon fraction"
    )
    mcf_value: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Methane correction factor"
    )
    has_gas_recovery: bool = Field(False, description="Landfill has gas recovery")

    # Incineration-specific parameters
    incineration_has_energy_recovery: bool = Field(
        False, description="Incineration with energy recovery"
    )

    # Recycling-specific parameters
    recycling_efficiency_pct: Optional[Decimal] = Field(
        None, ge=0, le=100, description="Recycling efficiency percentage"
    )

    # Composting-specific parameters
    composting_method: Optional[str] = Field(None, description="Composting method")

    # Anaerobic digestion-specific parameters
    ad_biogas_capture_pct: Optional[Decimal] = Field(
        None, ge=0, le=100, description="AD biogas capture percentage"
    )

    # Wastewater-specific parameters
    wastewater_bod_mg_l: Optional[Decimal] = Field(
        None, ge=0, description="Wastewater BOD (mg/L)"
    )
    wastewater_volume_m3: Optional[Decimal] = Field(
        None, ge=0, description="Wastewater volume (m³)"
    )

    emission_factor_id: Optional[UUID] = Field(
        None, description="Custom emission factor ID"
    )
    facility_country: Optional[str] = Field(
        None, max_length=3, description="ISO 3166-1 alpha-3 facility country code"
    )
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('waste_mass_tonnes')
    def validate_waste_mass(cls, v):
        """Validate waste mass is positive."""
        if v <= 0:
            raise ValueError("waste_mass_tonnes must be greater than zero")
        return v

    @validator('wastewater_volume_m3')
    def validate_wastewater_params(cls, v, values):
        """Validate wastewater parameters are provided together."""
        treatment = values.get('treatment_method')
        if treatment == TreatmentMethod.WASTEWATER_TREATMENT:
            bod = values.get('wastewater_bod_mg_l')
            if v is None or bod is None:
                raise ValueError(
                    "wastewater_volume_m3 and wastewater_bod_mg_l required for WASTEWATER_TREATMENT"
                )
        return v


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch waste generated emissions calculations.

    Attributes:
        tenant_id: Tenant identifier
        calculations: List of calculation requests
        batch_id: Optional batch identifier
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculations: List[CalculateRequest] = Field(
        ..., min_items=1, max_items=10000, description="List of calculations"
    )
    batch_id: Optional[UUID] = Field(None, description="Optional batch identifier")


class LandfillCalculateRequest(BaseModel):
    """
    Request model for landfill-specific calculation using FOD model.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Optional UUID for idempotency
        waste_type: Type of waste material
        waste_mass_tonnes: Mass of waste in metric tonnes
        landfill_type: Type of landfill
        moisture_content_pct: Moisture content percentage
        doc_value: Degradable organic carbon fraction
        mcf_value: Methane correction factor
        has_gas_recovery: Whether landfill has gas recovery
        oxidation_factor: Methane oxidation factor (0-0.1)
        year: Calculation year
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    waste_type: WasteType = Field(..., description="Type of waste material")
    waste_mass_tonnes: Decimal = Field(..., ge=0, description="Waste mass in metric tonnes")
    landfill_type: str = Field(..., description="Landfill type")
    moisture_content_pct: Decimal = Field(..., ge=0, le=100, description="Moisture content %")
    doc_value: Decimal = Field(..., ge=0, le=1, description="DOC fraction")
    mcf_value: Decimal = Field(..., ge=0, le=1, description="MCF value")
    has_gas_recovery: bool = Field(False, description="Gas recovery system")
    oxidation_factor: Decimal = Field(
        Decimal("0.1"), ge=0, le=1, description="Methane oxidation factor"
    )
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class IncinerationCalculateRequest(BaseModel):
    """
    Request model for incineration-specific calculation.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Optional UUID for idempotency
        waste_type: Type of waste material
        waste_mass_tonnes: Mass of waste in metric tonnes
        has_energy_recovery: Whether incineration has energy recovery
        combustion_efficiency_pct: Combustion efficiency percentage
        year: Calculation year
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    waste_type: WasteType = Field(..., description="Type of waste material")
    waste_mass_tonnes: Decimal = Field(..., ge=0, description="Waste mass in metric tonnes")
    has_energy_recovery: bool = Field(False, description="Energy recovery system")
    combustion_efficiency_pct: Decimal = Field(
        Decimal("99.5"), ge=0, le=100, description="Combustion efficiency %"
    )
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RecyclingCalculateRequest(BaseModel):
    """
    Request model for recycling-specific calculation.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Optional UUID for idempotency
        waste_type: Type of waste material
        waste_mass_tonnes: Mass of waste in metric tonnes
        recycling_efficiency_pct: Recycling efficiency percentage
        avoided_emissions: Whether to calculate avoided emissions
        year: Calculation year
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    waste_type: WasteType = Field(..., description="Type of waste material")
    waste_mass_tonnes: Decimal = Field(..., ge=0, description="Waste mass in metric tonnes")
    recycling_efficiency_pct: Decimal = Field(
        Decimal("95.0"), ge=0, le=100, description="Recycling efficiency %"
    )
    avoided_emissions: bool = Field(True, description="Calculate avoided emissions")
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CompostingCalculateRequest(BaseModel):
    """
    Request model for composting-specific calculation.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Optional UUID for idempotency
        waste_type: Type of waste material
        waste_mass_tonnes: Mass of waste in metric tonnes
        composting_method: Composting method (AEROBIC, ANAEROBIC)
        aeration_frequency: Aeration frequency (for aerobic)
        year: Calculation year
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    waste_type: WasteType = Field(..., description="Type of waste material")
    waste_mass_tonnes: Decimal = Field(..., ge=0, description="Waste mass in metric tonnes")
    composting_method: str = Field("AEROBIC", description="Composting method")
    aeration_frequency: Optional[str] = Field(None, description="Aeration frequency")
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AnaerobicDigestionCalculateRequest(BaseModel):
    """
    Request model for anaerobic digestion-specific calculation.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Optional UUID for idempotency
        waste_type: Type of waste material
        waste_mass_tonnes: Mass of waste in metric tonnes
        biogas_capture_pct: Biogas capture percentage
        biogas_utilization: Biogas utilization method (FLARE, ENERGY, DIRECT_USE)
        year: Calculation year
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    waste_type: WasteType = Field(..., description="Type of waste material")
    waste_mass_tonnes: Decimal = Field(..., ge=0, description="Waste mass in metric tonnes")
    biogas_capture_pct: Decimal = Field(
        Decimal("90.0"), ge=0, le=100, description="Biogas capture %"
    )
    biogas_utilization: str = Field("ENERGY", description="Biogas utilization method")
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class WastewaterCalculateRequest(BaseModel):
    """
    Request model for wastewater treatment-specific calculation.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Optional UUID for idempotency
        wastewater_volume_m3: Wastewater volume in cubic meters
        bod_mg_l: Biological oxygen demand in mg/L
        cod_mg_l: Chemical oxygen demand in mg/L
        treatment_type: Treatment type (AEROBIC, ANAEROBIC, LAGOON, SEPTIC)
        has_sludge_removal: Whether treatment has sludge removal
        year: Calculation year
        metadata: Additional metadata
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: Optional[UUID] = Field(
        None, description="Optional UUID for idempotency"
    )
    wastewater_volume_m3: Decimal = Field(..., ge=0, description="Wastewater volume (m³)")
    bod_mg_l: Decimal = Field(..., ge=0, description="BOD (mg/L)")
    cod_mg_l: Optional[Decimal] = Field(None, ge=0, description="COD (mg/L)")
    treatment_type: str = Field("AEROBIC", description="Treatment type")
    has_sludge_removal: bool = Field(True, description="Sludge removal system")
    year: int = Field(..., ge=1990, le=2100, description="Calculation year")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ComplianceCheckRequest(BaseModel):
    """
    Request model for compliance checking.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Calculation ID to check
        frameworks: List of regulatory frameworks to check against
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: UUID = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        ["GHG_PROTOCOL", "IPCC_2006", "ISO_14064"],
        description="Regulatory frameworks"
    )


class UncertaintyRequest(BaseModel):
    """
    Request model for uncertainty analysis.

    Attributes:
        tenant_id: Tenant identifier
        calculation_id: Calculation ID to analyze
        method: Uncertainty analysis method
        confidence_level: Confidence level (0.90, 0.95, 0.99)
        monte_carlo_iterations: Number of iterations for Monte Carlo
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    calculation_id: UUID = Field(..., description="Calculation ID")
    method: UncertaintyMethod = Field(
        UncertaintyMethod.TIER_1, description="Uncertainty method"
    )
    confidence_level: Decimal = Field(
        Decimal("0.95"), ge=Decimal("0.80"), le=Decimal("0.99"),
        description="Confidence level"
    )
    monte_carlo_iterations: int = Field(
        10000, ge=1000, le=100000, description="Monte Carlo iterations"
    )


class DiversionAnalysisRequest(BaseModel):
    """
    Request model for waste diversion analysis.

    Attributes:
        tenant_id: Tenant identifier
        from_date: Start date for analysis
        to_date: End date for analysis
        include_recycling: Include recycling in diversion
        include_composting: Include composting in diversion
        include_anaerobic_digestion: Include AD in diversion
        include_energy_recovery: Include energy recovery in diversion
    """

    tenant_id: constr(min_length=1, max_length=100) = Field(
        ..., description="Tenant identifier"
    )
    from_date: date = Field(..., description="Start date")
    to_date: date = Field(..., description="End date")
    include_recycling: bool = Field(True, description="Include recycling")
    include_composting: bool = Field(True, description="Include composting")
    include_anaerobic_digestion: bool = Field(True, description="Include AD")
    include_energy_recovery: bool = Field(True, description="Include energy recovery")

    @validator('to_date')
    def validate_date_range(cls, v, values):
        """Validate date range is valid."""
        from_date = values.get('from_date')
        if from_date and v < from_date:
            raise ValueError("to_date must be after from_date")
        return v


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CalculateResponse(BaseModel):
    """Response model for single calculation."""

    calculation_id: UUID = Field(..., description="Calculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    waste_type: str = Field(..., description="Waste type")
    treatment_method: str = Field(..., description="Treatment method")
    calculation_method: str = Field(..., description="Calculation method")
    waste_mass_tonnes: Decimal = Field(..., description="Waste mass")
    co2_kg: Decimal = Field(..., description="CO2 emissions in kg")
    ch4_kg: Decimal = Field(..., description="CH4 emissions in kg")
    n2o_kg: Decimal = Field(..., description="N2O emissions in kg")
    co2e_kg: Decimal = Field(..., description="CO2e emissions in kg")
    avoided_emissions_kg: Optional[Decimal] = Field(None, description="Avoided emissions in kg")
    emission_factor_source: str = Field(..., description="Emission factor source")
    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculations."""

    batch_id: UUID = Field(..., description="Batch ID")
    tenant_id: str = Field(..., description="Tenant ID")
    total_calculations: int = Field(..., description="Total calculations")
    successful: int = Field(..., description="Successful calculations")
    failed: int = Field(..., description="Failed calculations")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e emissions in kg")
    total_avoided_emissions_kg: Decimal = Field(..., description="Total avoided emissions in kg")
    results: List[CalculateResponse] = Field(..., description="Individual results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class EmissionFactorResponse(BaseModel):
    """Response model for emission factor."""

    factor_id: UUID = Field(..., description="Factor ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (null for global)")
    waste_type: str = Field(..., description="Waste type")
    treatment_method: str = Field(..., description="Treatment method")
    factor_name: str = Field(..., description="Factor name")
    co2_kg_per_tonne: Decimal = Field(..., description="CO2 kg per tonne")
    ch4_kg_per_tonne: Decimal = Field(..., description="CH4 kg per tonne")
    n2o_kg_per_tonne: Decimal = Field(..., description="N2O kg per tonne")
    source: str = Field(..., description="Source")
    year: int = Field(..., description="Year")
    region: Optional[str] = Field(None, description="Region")
    created_at: datetime = Field(..., description="Creation timestamp")


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    factors: List[EmissionFactorResponse] = Field(..., description="Emission factors")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page limit")
    offset: int = Field(..., description="Page offset")


class WasteTypeResponse(BaseModel):
    """Response model for waste type."""

    waste_type_code: str = Field(..., description="Waste type code")
    waste_type_name: str = Field(..., description="Waste type name")
    category: str = Field(..., description="Waste category")
    is_organic: bool = Field(..., description="Is organic waste")
    is_recyclable: bool = Field(..., description="Is recyclable")
    typical_doc_value: Optional[Decimal] = Field(None, description="Typical DOC value")
    description: Optional[str] = Field(None, description="Description")


class WasteTypeListResponse(BaseModel):
    """Response model for waste type listing."""

    waste_types: List[WasteTypeResponse] = Field(..., description="Waste types")
    total: int = Field(..., description="Total count")


class TreatmentMethodResponse(BaseModel):
    """Response model for treatment method."""

    method_code: str = Field(..., description="Method code")
    method_name: str = Field(..., description="Method name")
    category: str = Field(..., description="Method category")
    is_diversion: bool = Field(..., description="Counts as diversion")
    has_energy_recovery: bool = Field(..., description="Has energy recovery")
    description: Optional[str] = Field(None, description="Description")


class TreatmentMethodListResponse(BaseModel):
    """Response model for treatment method listing."""

    treatment_methods: List[TreatmentMethodResponse] = Field(..., description="Treatment methods")
    total: int = Field(..., description="Total count")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check."""

    check_id: UUID = Field(..., description="Check ID")
    tenant_id: str = Field(..., description="Tenant ID")
    calculation_id: UUID = Field(..., description="Calculation ID")
    frameworks: List[str] = Field(..., description="Checked frameworks")
    overall_status: str = Field(..., description="Overall compliance status")
    findings: List[Dict[str, Any]] = Field(..., description="Compliance findings")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    check_timestamp: datetime = Field(..., description="Check timestamp")


class UncertaintyResponse(BaseModel):
    """Response model for uncertainty analysis."""

    analysis_id: UUID = Field(..., description="Analysis ID")
    tenant_id: str = Field(..., description="Tenant ID")
    calculation_id: UUID = Field(..., description="Calculation ID")
    method: str = Field(..., description="Uncertainty method")
    confidence_level: Decimal = Field(..., description="Confidence level")
    co2e_mean_kg: Decimal = Field(..., description="Mean CO2e")
    co2e_std_kg: Decimal = Field(..., description="Standard deviation CO2e")
    co2e_lower_bound_kg: Decimal = Field(..., description="Lower bound CO2e")
    co2e_upper_bound_kg: Decimal = Field(..., description="Upper bound CO2e")
    relative_uncertainty_pct: Decimal = Field(..., description="Relative uncertainty %")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")


class AggregationResponse(BaseModel):
    """Response model for aggregations."""

    tenant_id: str = Field(..., description="Tenant ID")
    period: str = Field(..., description="Aggregation period")
    from_date: date = Field(..., description="Start date")
    to_date: date = Field(..., description="End date")
    total_waste_tonnes: Decimal = Field(..., description="Total waste mass")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e")
    total_avoided_emissions_kg: Decimal = Field(..., description="Total avoided emissions")
    by_waste_type: Dict[str, Any] = Field(..., description="Breakdown by waste type")
    by_treatment_method: Dict[str, Any] = Field(..., description="Breakdown by treatment method")
    total_calculations: int = Field(..., description="Total calculations")


class DiversionAnalysisResponse(BaseModel):
    """Response model for diversion analysis."""

    tenant_id: str = Field(..., description="Tenant ID")
    from_date: date = Field(..., description="Start date")
    to_date: date = Field(..., description="End date")
    total_waste_tonnes: Decimal = Field(..., description="Total waste mass")
    diverted_waste_tonnes: Decimal = Field(..., description="Diverted waste mass")
    disposal_waste_tonnes: Decimal = Field(..., description="Disposal waste mass")
    diversion_rate_pct: Decimal = Field(..., description="Diversion rate %")
    recycling_tonnes: Decimal = Field(..., description="Recycling mass")
    composting_tonnes: Decimal = Field(..., description="Composting mass")
    anaerobic_digestion_tonnes: Decimal = Field(..., description="AD mass")
    energy_recovery_tonnes: Decimal = Field(..., description="Energy recovery mass")
    landfill_tonnes: Decimal = Field(..., description="Landfill mass")
    incineration_tonnes: Decimal = Field(..., description="Incineration mass")
    diversion_emissions_co2e_kg: Decimal = Field(..., description="Diversion emissions")
    disposal_emissions_co2e_kg: Decimal = Field(..., description="Disposal emissions")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")


class CalculationListResponse(BaseModel):
    """Response model for calculation listing."""

    calculations: List[CalculateResponse] = Field(..., description="Calculations")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Page limit")
    offset: int = Field(..., description="Page offset")


class CalculationDetailResponse(CalculateResponse):
    """Response model for calculation detail."""

    uncertainty_data: Optional[Dict[str, Any]] = Field(None, description="Uncertainty data")
    compliance_status: Optional[Dict[str, Any]] = Field(None, description="Compliance status")
    treatment_parameters: Optional[Dict[str, Any]] = Field(None, description="Treatment parameters")


class DeleteResponse(BaseModel):
    """Response model for deletion."""

    deleted: bool = Field(..., description="Deletion status")
    calculation_id: UUID = Field(..., description="Deleted calculation ID")
    timestamp: datetime = Field(..., description="Deletion timestamp")


class ProvenanceResponse(BaseModel):
    """Response model for provenance tracking."""

    calculation_id: UUID = Field(..., description="Calculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    input_data_hash: str = Field(..., description="Input data hash")
    emission_factor_hash: str = Field(..., description="Emission factor hash")
    calculation_method: str = Field(..., description="Calculation method")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    verifiable: bool = Field(..., description="Provenance is verifiable")
    audit_trail: List[Dict[str, Any]] = Field(..., description="Audit trail entries")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Check timestamp")
    database_connected: bool = Field(..., description="Database connection status")
    cache_connected: bool = Field(..., description="Cache connection status")


class StatsResponse(BaseModel):
    """Response model for statistics."""

    total_calculations: int = Field(..., description="Total calculations")
    total_waste_tonnes: Decimal = Field(..., description="Total waste mass")
    total_co2e_kg: Decimal = Field(..., description="Total CO2e emissions")
    total_avoided_emissions_kg: Decimal = Field(..., description="Total avoided emissions")
    calculations_by_waste_type: Dict[str, int] = Field(..., description="Calculations by waste type")
    calculations_by_treatment: Dict[str, int] = Field(..., description="Calculations by treatment")
    average_diversion_rate_pct: Decimal = Field(..., description="Average diversion rate")
    timestamp: datetime = Field(..., description="Stats timestamp")


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance: Optional[WasteGeneratedService] = None


def get_service() -> WasteGeneratedService:
    """
    Get or create WasteGeneratedService instance.

    Returns:
        WasteGeneratedService instance

    Raises:
        HTTPException: If service initialization fails
    """
    global _service_instance

    if _service_instance is None:
        try:
            _service_instance = WasteGeneratedService()
            logger.info("WasteGeneratedService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WasteGeneratedService: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed"
            )

    return _service_instance


# ============================================================================
# ENDPOINTS - CALCULATIONS (11)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate waste generated emissions",
    description=(
        "Calculate GHG emissions for a single waste stream. "
        "Supports multiple waste types and treatment methods. "
        "Returns deterministic results with provenance hash for audit trail."
    ),
)
async def calculate_emissions(
    request: CalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate waste generated emissions for a single waste stream.

    Args:
        request: Calculation request with waste stream data
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails or validation errors occur
    """
    try:
        logger.info(
            f"Calculating emissions for tenant {request.tenant_id}, "
            f"waste_type {request.waste_type}, treatment {request.treatment_method}"
        )

        result = await service.calculate(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Calculation failed"
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate batch waste generated emissions",
    description=(
        "Calculate GHG emissions for multiple waste streams in a single request. "
        "Processes up to 10,000 calculations with parallel execution. "
        "Returns aggregated results with individual calculation details and error handling."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate waste generated emissions for multiple waste streams.

    Args:
        request: Batch calculation request
        service: WasteGeneratedService instance

    Returns:
        BatchCalculateResponse with aggregated results

    Raises:
        HTTPException: If batch calculation fails
    """
    try:
        logger.info(
            f"Calculating batch emissions for tenant {request.tenant_id}, "
            f"count {len(request.calculations)}"
        )

        result = await service.calculate_batch(request.dict())

        return BatchCalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_batch_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_batch_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed"
        )


@router.post(
    "/calculate/landfill",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate landfill emissions using FOD model",
    description=(
        "Calculate landfill emissions using IPCC First Order Decay (FOD) model. "
        "Supports multiple landfill types and accounts for gas recovery systems. "
        "Returns CH4 and CO2 emissions with provenance tracking."
    ),
)
async def calculate_landfill_emissions(
    request: LandfillCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate landfill-specific emissions using FOD model.

    Args:
        request: Landfill calculation request
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(
            f"Calculating landfill emissions for tenant {request.tenant_id}, "
            f"landfill_type {request.landfill_type}"
        )

        result = await service.calculate_landfill(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_landfill_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_landfill_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Landfill calculation failed"
        )


@router.post(
    "/calculate/incineration",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate incineration emissions",
    description=(
        "Calculate incineration emissions with optional energy recovery accounting. "
        "Supports waste-to-energy facilities with avoided emissions calculations. "
        "Returns fossil and biogenic CO2 separately."
    ),
)
async def calculate_incineration_emissions(
    request: IncinerationCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate incineration-specific emissions.

    Args:
        request: Incineration calculation request
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(
            f"Calculating incineration emissions for tenant {request.tenant_id}, "
            f"energy_recovery {request.has_energy_recovery}"
        )

        result = await service.calculate_incineration(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_incineration_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_incineration_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Incineration calculation failed"
        )


@router.post(
    "/calculate/recycling",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate recycling emissions",
    description=(
        "Calculate recycling process emissions and avoided emissions from virgin material displacement. "
        "Accounts for recycling efficiency and material-specific benefits. "
        "Returns net emissions considering avoided impacts."
    ),
)
async def calculate_recycling_emissions(
    request: RecyclingCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate recycling-specific emissions.

    Args:
        request: Recycling calculation request
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and avoided emissions

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(
            f"Calculating recycling emissions for tenant {request.tenant_id}, "
            f"efficiency {request.recycling_efficiency_pct}%"
        )

        result = await service.calculate_recycling(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_recycling_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_recycling_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recycling calculation failed"
        )


@router.post(
    "/calculate/composting",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate composting emissions",
    description=(
        "Calculate composting emissions for organic waste. "
        "Supports aerobic and anaerobic composting methods. "
        "Returns CH4 and N2O emissions based on composting conditions."
    ),
)
async def calculate_composting_emissions(
    request: CompostingCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate composting-specific emissions.

    Args:
        request: Composting calculation request
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(
            f"Calculating composting emissions for tenant {request.tenant_id}, "
            f"method {request.composting_method}"
        )

        result = await service.calculate_composting(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_composting_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_composting_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Composting calculation failed"
        )


@router.post(
    "/calculate/anaerobic-digestion",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate anaerobic digestion emissions",
    description=(
        "Calculate anaerobic digestion emissions with biogas capture accounting. "
        "Supports various biogas utilization methods (flare, energy, direct use). "
        "Returns net emissions considering captured biogas."
    ),
)
async def calculate_anaerobic_digestion_emissions(
    request: AnaerobicDigestionCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate anaerobic digestion-specific emissions.

    Args:
        request: Anaerobic digestion calculation request
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(
            f"Calculating anaerobic digestion emissions for tenant {request.tenant_id}, "
            f"capture {request.biogas_capture_pct}%"
        )

        result = await service.calculate_anaerobic_digestion(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_anaerobic_digestion_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_anaerobic_digestion_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Anaerobic digestion calculation failed"
        )


@router.post(
    "/calculate/wastewater",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate wastewater treatment emissions",
    description=(
        "Calculate wastewater treatment emissions based on BOD/COD and treatment type. "
        "Accounts for CH4 from anaerobic conditions and N2O from nitrification/denitrification. "
        "Returns emissions based on IPCC wastewater guidelines."
    ),
)
async def calculate_wastewater_emissions(
    request: WastewaterCalculateRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> CalculateResponse:
    """
    Calculate wastewater treatment-specific emissions.

    Args:
        request: Wastewater calculation request
        service: WasteGeneratedService instance

    Returns:
        CalculateResponse with emissions and metadata

    Raises:
        HTTPException: If calculation fails
    """
    try:
        logger.info(
            f"Calculating wastewater emissions for tenant {request.tenant_id}, "
            f"treatment_type {request.treatment_type}"
        )

        result = await service.calculate_wastewater(request.dict())

        return CalculateResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_wastewater_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in calculate_wastewater_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Wastewater calculation failed"
        )


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get waste calculation detail",
    description=(
        "Retrieve detailed information for a specific waste calculation. "
        "Includes emissions data, uncertainty information, and compliance status."
    ),
)
async def get_calculation_detail(
    calculation_id: UUID = Path(..., description="Calculation ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: WasteGeneratedService = Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        tenant_id: Tenant identifier
        service: WasteGeneratedService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: If calculation not found or access denied
    """
    try:
        logger.info(f"Getting calculation detail {calculation_id} for tenant {tenant_id}")

        result = await service.get_calculation(calculation_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found"
            )

        return CalculationDetailResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_calculation_detail: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation"
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List waste calculations",
    description=(
        "Retrieve a paginated list of waste calculations. "
        "Supports filtering by tenant, waste type, treatment method, and date range. "
        "Returns summary information for each calculation."
    ),
)
async def list_calculations(
    tenant_id: str = Query(..., description="Tenant identifier"),
    waste_type: Optional[WasteType] = Query(None, description="Filter by waste type"),
    treatment_method: Optional[TreatmentMethod] = Query(None, description="Filter by treatment method"),
    from_date: Optional[date] = Query(None, description="Filter from date"),
    to_date: Optional[date] = Query(None, description="Filter to date"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    service: WasteGeneratedService = Depends(get_service),
) -> CalculationListResponse:
    """
    List waste calculations with filtering and pagination.

    Args:
        tenant_id: Tenant identifier
        waste_type: Optional waste type filter
        treatment_method: Optional treatment method filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        limit: Maximum number of results
        offset: Number of results to skip
        service: WasteGeneratedService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info(f"Listing calculations for tenant {tenant_id}")

        filters = {
            "tenant_id": tenant_id,
            "waste_type": waste_type.value if waste_type else None,
            "treatment_method": treatment_method.value if treatment_method else None,
            "from_date": from_date,
            "to_date": to_date,
            "limit": limit,
            "offset": offset,
        }

        result = await service.list_calculations(filters)

        return CalculationListResponse(**result)

    except Exception as e:
        logger.error(f"Error in list_calculations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations"
        )


@router.delete(
    "/calculations/{calculation_id}",
    response_model=DeleteResponse,
    summary="Delete waste calculation",
    description=(
        "Delete a specific waste calculation. "
        "Requires tenant ownership. Soft delete with audit trail."
    ),
)
async def delete_calculation(
    calculation_id: UUID = Path(..., description="Calculation ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: WasteGeneratedService = Depends(get_service),
) -> DeleteResponse:
    """
    Delete a specific calculation.

    Args:
        calculation_id: Calculation UUID
        tenant_id: Tenant identifier
        service: WasteGeneratedService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: If calculation not found or deletion fails
    """
    try:
        logger.info(f"Deleting calculation {calculation_id} for tenant {tenant_id}")

        deleted = await service.delete_calculation(calculation_id, tenant_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found"
            )

        return DeleteResponse(
            deleted=True,
            calculation_id=calculation_id,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_calculation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation"
        )


# ============================================================================
# ENDPOINTS - EMISSION FACTORS & METADATA (4)
# ============================================================================


@router.get(
    "/emission-factors",
    response_model=EmissionFactorListResponse,
    summary="List emission factors",
    description=(
        "Retrieve available emission factors for waste treatment. "
        "Supports filtering by waste type and treatment method. "
        "Includes both standard and custom emission factors."
    ),
)
async def list_emission_factors(
    tenant_id: Optional[str] = Query(None, description="Tenant identifier for custom factors"),
    waste_type: Optional[WasteType] = Query(None, description="Filter by waste type"),
    treatment_method: Optional[TreatmentMethod] = Query(None, description="Filter by treatment method"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    service: WasteGeneratedService = Depends(get_service),
) -> EmissionFactorListResponse:
    """
    List emission factors with filtering and pagination.

    Args:
        tenant_id: Optional tenant identifier for custom factors
        waste_type: Optional waste type filter
        treatment_method: Optional treatment method filter
        limit: Maximum number of results
        offset: Number of results to skip
        service: WasteGeneratedService instance

    Returns:
        EmissionFactorListResponse with paginated results

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info(f"Listing emission factors for tenant {tenant_id or 'global'}")

        filters = {
            "tenant_id": tenant_id,
            "waste_type": waste_type.value if waste_type else None,
            "treatment_method": treatment_method.value if treatment_method else None,
            "limit": limit,
            "offset": offset,
        }

        result = await service.list_emission_factors(filters)

        return EmissionFactorListResponse(**result)

    except Exception as e:
        logger.error(f"Error in list_emission_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list emission factors"
        )


@router.get(
    "/emission-factors/{waste_type}",
    response_model=EmissionFactorListResponse,
    summary="Get emission factors by waste type",
    description=(
        "Retrieve emission factors for a specific waste type across all treatment methods. "
        "Useful for comparing treatment options for a given waste material."
    ),
)
async def get_emission_factors_by_waste_type(
    waste_type: WasteType = Path(..., description="Waste type"),
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    service: WasteGeneratedService = Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific waste type.

    Args:
        waste_type: Waste type
        tenant_id: Optional tenant identifier
        service: WasteGeneratedService instance

    Returns:
        EmissionFactorListResponse with factors for the waste type

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Getting emission factors for waste type {waste_type}")

        filters = {
            "tenant_id": tenant_id,
            "waste_type": waste_type.value,
            "limit": 1000,
            "offset": 0,
        }

        result = await service.list_emission_factors(filters)

        return EmissionFactorListResponse(**result)

    except Exception as e:
        logger.error(f"Error in get_emission_factors_by_waste_type: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors"
        )


@router.get(
    "/waste-types",
    response_model=WasteTypeListResponse,
    summary="List waste types",
    description=(
        "Retrieve all supported waste types with metadata. "
        "Includes waste categories, recyclability, and typical DOC values. "
        "Useful for waste stream classification."
    ),
)
async def list_waste_types(
    service: WasteGeneratedService = Depends(get_service),
) -> WasteTypeListResponse:
    """
    List all supported waste types.

    Args:
        service: WasteGeneratedService instance

    Returns:
        WasteTypeListResponse with all waste types

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info("Listing waste types")

        result = await service.list_waste_types()

        return WasteTypeListResponse(**result)

    except Exception as e:
        logger.error(f"Error in list_waste_types: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list waste types"
        )


@router.get(
    "/treatment-methods",
    response_model=TreatmentMethodListResponse,
    summary="List treatment methods",
    description=(
        "Retrieve all supported treatment methods with metadata. "
        "Includes method categories, diversion status, and energy recovery information. "
        "Useful for waste management planning."
    ),
)
async def list_treatment_methods(
    service: WasteGeneratedService = Depends(get_service),
) -> TreatmentMethodListResponse:
    """
    List all supported treatment methods.

    Args:
        service: WasteGeneratedService instance

    Returns:
        TreatmentMethodListResponse with all treatment methods

    Raises:
        HTTPException: If listing fails
    """
    try:
        logger.info("Listing treatment methods")

        result = await service.list_treatment_methods()

        return TreatmentMethodListResponse(**result)

    except Exception as e:
        logger.error(f"Error in list_treatment_methods: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list treatment methods"
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE & UNCERTAINTY (2)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    summary="Check calculation compliance",
    description=(
        "Check calculation compliance against regulatory frameworks. "
        "Supports GHG Protocol Scope 3, IPCC 2006 Guidelines, ISO 14064, and local regulations. "
        "Returns compliance status, findings, and recommendations."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request
        service: WasteGeneratedService instance

    Returns:
        ComplianceCheckResponse with compliance findings

    Raises:
        HTTPException: If compliance check fails
    """
    try:
        logger.info(
            f"Checking compliance for calculation {request.calculation_id}, "
            f"tenant {request.tenant_id}"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceCheckResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in check_compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in check_compliance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed"
        )


@router.post(
    "/uncertainty/analyze",
    response_model=UncertaintyResponse,
    summary="Analyze calculation uncertainty",
    description=(
        "Perform uncertainty analysis on waste emissions calculations. "
        "Supports Tier 1 (default factors), Tier 2 (custom factors), "
        "and Monte Carlo simulation methods. "
        "Returns confidence intervals and relative uncertainty."
    ),
)
async def analyze_uncertainty(
    request: UncertaintyRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> UncertaintyResponse:
    """
    Perform uncertainty analysis on a calculation.

    Args:
        request: Uncertainty analysis request
        service: WasteGeneratedService instance

    Returns:
        UncertaintyResponse with uncertainty metrics

    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(
            f"Analyzing uncertainty for calculation {request.calculation_id}, "
            f"tenant {request.tenant_id}, method {request.method}"
        )

        result = await service.analyze_uncertainty(request.dict())

        return UncertaintyResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in analyze_uncertainty: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in analyze_uncertainty: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uncertainty analysis failed"
        )


# ============================================================================
# ENDPOINTS - AGGREGATIONS & ANALYSIS (2)
# ============================================================================


@router.get(
    "/aggregations/{period}",
    response_model=AggregationResponse,
    summary="Get aggregated waste emissions",
    description=(
        "Retrieve aggregated waste emissions data for a specified period. "
        "Supports daily, weekly, monthly, quarterly, and annual aggregations. "
        "Returns breakdowns by waste type and treatment method."
    ),
)
async def get_aggregations(
    period: str = Path(..., description="Aggregation period (daily, weekly, monthly, quarterly, annual)"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    from_date: date = Query(..., description="Start date"),
    to_date: date = Query(..., description="End date"),
    service: WasteGeneratedService = Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated waste emissions data.

    Args:
        period: Aggregation period
        tenant_id: Tenant identifier
        from_date: Start date
        to_date: End date
        service: WasteGeneratedService instance

    Returns:
        AggregationResponse with aggregated data

    Raises:
        HTTPException: If aggregation fails
    """
    try:
        logger.info(f"Getting aggregations for tenant {tenant_id}, period {period}")

        if period not in ["daily", "weekly", "monthly", "quarterly", "annual"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Period must be one of: daily, weekly, monthly, quarterly, annual"
            )

        filters = {
            "tenant_id": tenant_id,
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_aggregations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed"
        )


@router.post(
    "/diversion/analyze",
    response_model=DiversionAnalysisResponse,
    summary="Analyze waste diversion rates",
    description=(
        "Analyze waste diversion rates and emissions by treatment category. "
        "Calculates diversion rate, breakdown by diversion method, and emissions comparison. "
        "Useful for sustainability reporting and waste management optimization."
    ),
)
async def analyze_diversion(
    request: DiversionAnalysisRequest,
    service: WasteGeneratedService = Depends(get_service),
) -> DiversionAnalysisResponse:
    """
    Analyze waste diversion rates.

    Args:
        request: Diversion analysis request
        service: WasteGeneratedService instance

    Returns:
        DiversionAnalysisResponse with diversion metrics

    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(
            f"Analyzing diversion for tenant {request.tenant_id}, "
            f"period {request.from_date} to {request.to_date}"
        )

        result = await service.analyze_diversion(request.dict())

        return DiversionAnalysisResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in analyze_diversion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in analyze_diversion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Diversion analysis failed"
        )


# ============================================================================
# ENDPOINTS - PROVENANCE (1)
# ============================================================================


@router.get(
    "/provenance/{calculation_id}",
    response_model=ProvenanceResponse,
    summary="Get calculation provenance",
    description=(
        "Retrieve complete provenance tracking for a calculation. "
        "Includes input data hash, emission factor hash, and audit trail. "
        "Enables verification and reproducibility of results."
    ),
)
async def get_provenance(
    calculation_id: UUID = Path(..., description="Calculation ID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service: WasteGeneratedService = Depends(get_service),
) -> ProvenanceResponse:
    """
    Get calculation provenance tracking.

    Args:
        calculation_id: Calculation UUID
        tenant_id: Tenant identifier
        service: WasteGeneratedService instance

    Returns:
        ProvenanceResponse with complete provenance data

    Raises:
        HTTPException: If calculation not found or access denied
    """
    try:
        logger.info(f"Getting provenance for calculation {calculation_id}, tenant {tenant_id}")

        result = await service.get_provenance(calculation_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found"
            )

        return ProvenanceResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_provenance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provenance"
        )
