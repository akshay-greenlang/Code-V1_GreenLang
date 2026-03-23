"""
Downstream Transportation & Distribution API Router - AGENT-MRV-022

This module implements the FastAPI router for downstream transportation and
distribution emissions calculations following GHG Protocol Scope 3 Category 9
requirements.

Provides 22 REST endpoints for:
- Emissions calculations (distance, spend, average-data, warehouse, last-mile,
  supplier-specific, batch, portfolio, and full pipeline)
- Compliance checking across 7 regulatory frameworks
- Calculation CRUD (get, list, delete)
- Emission factor lookup by transport mode
- Warehouse benchmark reference data
- Last-mile delivery factor reference data
- Incoterm-based Cat 4 vs Cat 9 classification rules
- Aggregated emissions by dimension
- Provenance chain verification
- Health monitoring
- Uncertainty analysis
- Portfolio-level analysis

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Sub-Activities per GHG Protocol:
    9a - Outbound transportation (post-sale, Incoterm-driven boundary)
    9b - Outbound distribution (DC / warehouse operations)
    9c - Retail storage (third-party retail energy consumption)
    9d - Last-mile delivery (final delivery to end consumer)

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.downstream_transportation.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, date
from uuid import UUID
import logging
import uuid as _uuid

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/api/v1/downstream-transportation",
    tags=["Downstream Transportation"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create DownstreamTransportService singleton instance.

    Uses a lazy-initialization singleton pattern to avoid creating the service
    until the first request. The service wires together all 7 downstream
    transportation engines.

    Returns:
        DownstreamTransportService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.agents.mrv.downstream_transportation.setup import (
                DownstreamTransportService,
            )
            _service_instance = DownstreamTransportService()
            logger.info("DownstreamTransportService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DownstreamTransportService: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS (12)
# ============================================================================


class CalculateRequest(BaseModel):
    """
    Request model for full pipeline downstream transportation emissions calculation.

    Routes the request through the 10-stage DownstreamTransportPipelineEngine
    including validation, method selection, EF resolution, calculation,
    warehouse allocation, last-mile, compliance, and provenance sealing.

    Attributes:
        tenant_id: Tenant identifier for multi-tenancy
        shipment_id: Unique shipment or consignment identifier
        calculation_method: Method to use (distance, spend, average_data, supplier_specific)
        mode: Transport mode (road, rail, air, sea, pipeline, multimodal)
        sub_activity: Sub-activity classification (9a, 9b, 9c, 9d)
        distance_km: Transportation distance in kilometres
        mass_tonnes: Cargo mass in metric tonnes
        spend_amount: Spend amount for spend-based method
        spend_currency: ISO 4217 currency code for spend
        carrier_name: Carrier or logistics provider name
        incoterm: Incoterm code for boundary determination (e.g. EXW, FOB, CIF, DDP)
        year: Reporting year
        metadata: Additional metadata for audit trail
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    shipment_id: Optional[str] = Field(
        None,
        max_length=200,
        description="Unique shipment identifier",
    )
    calculation_method: str = Field(
        "distance_based",
        description=(
            "Calculation method: distance_based, spend_based, "
            "average_data, supplier_specific"
        ),
    )
    mode: Optional[str] = Field(
        None,
        description="Transport mode (road, rail, air, sea, pipeline, multimodal)",
    )
    sub_activity: str = Field(
        "9a",
        description="Sub-activity: 9a (outbound transport), 9b (distribution), 9c (retail), 9d (last-mile)",
    )
    distance_km: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Distance in kilometres",
    )
    mass_tonnes: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Cargo mass in metric tonnes",
    )
    spend_amount: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Spend amount for spend-based method",
    )
    spend_currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code",
    )
    carrier_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Carrier or logistics provider name",
    )
    incoterm: Optional[str] = Field(
        None,
        max_length=10,
        description="Incoterm code (EXW, FOB, CIF, DAP, DDP, etc.)",
    )
    year: int = Field(
        2024,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for audit trail",
    )

    @validator("calculation_method")
    def validate_calculation_method(cls, v: str) -> str:
        """Validate calculation method is supported."""
        allowed = ["distance_based", "spend_based", "average_data", "supplier_specific"]
        if v not in allowed:
            raise ValueError(f"calculation_method must be one of {allowed}")
        return v

    @validator("sub_activity")
    def validate_sub_activity(cls, v: str) -> str:
        """Validate sub-activity code."""
        allowed = ["9a", "9b", "9c", "9d"]
        if v not in allowed:
            raise ValueError(f"sub_activity must be one of {allowed}")
        return v


class DistanceCalcRequest(BaseModel):
    """
    Request model for distance-based downstream transportation calculation.

    Uses tonne-km emission factors from DEFRA, EPA SmartWay, GLEC, or ICAO
    depending on transport mode and region.

    Attributes:
        tenant_id: Tenant identifier
        mode: Transport mode (road, rail, air, sea)
        vehicle_type: Specific vehicle type for EF selection
        distance_km: Transportation distance in kilometres
        mass_tonnes: Cargo mass in metric tonnes
        return_trip: Whether to include return journey (empty or laden)
        return_load_factor: Load factor for return leg (0.0 = empty return)
        year: Reporting year
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    mode: str = Field(
        ...,
        description="Transport mode (road, rail, air, sea)",
    )
    vehicle_type: Optional[str] = Field(
        None,
        description="Vehicle type (e.g. rigid_truck, articulated_truck, container_ship)",
    )
    distance_km: Decimal = Field(
        ...,
        gt=0,
        description="Distance in kilometres",
    )
    mass_tonnes: Decimal = Field(
        ...,
        gt=0,
        description="Cargo mass in metric tonnes",
    )
    return_trip: bool = Field(
        False,
        description="Include return journey",
    )
    return_load_factor: Decimal = Field(
        Decimal("0.0"),
        ge=0,
        le=1,
        description="Load factor for return leg (0.0 = empty return)",
    )
    year: int = Field(
        2024,
        ge=1990,
        le=2100,
        description="Reporting year",
    )


class SpendCalcRequest(BaseModel):
    """
    Request model for spend-based downstream transportation calculation.

    Uses EEIO factors with CPI deflation and currency conversion.
    Applies margin removal to convert purchaser price to producer price.

    Attributes:
        tenant_id: Tenant identifier
        spend_amount: Transport spend amount
        currency: ISO 4217 currency code
        naics_code: NAICS code for EEIO factor selection
        reporting_year: Year for CPI deflation adjustment
        margin_rate: Margin rate to remove from spend (e.g. 0.20 for 20%)
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    spend_amount: Decimal = Field(
        ...,
        gt=0,
        description="Transport spend amount",
    )
    currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code",
    )
    naics_code: Optional[str] = Field(
        None,
        description="NAICS code for EEIO factor (default: 484 trucking)",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year for CPI deflation",
    )
    margin_rate: Decimal = Field(
        Decimal("0.0"),
        ge=0,
        le=1,
        description="Margin rate to remove (0.0 = no margin removal)",
    )


class AverageDataCalcRequest(BaseModel):
    """
    Request model for average-data downstream transportation calculation.

    Uses industry-average emission intensities per product unit, revenue,
    or weight by distribution channel type.

    Attributes:
        tenant_id: Tenant identifier
        distribution_channel: Channel type (e-commerce, retail, wholesale, direct)
        product_category: Product category for average EF lookup
        quantity: Quantity of product units
        quantity_unit: Unit of quantity (units, kg, m3, pallets)
        revenue: Revenue for revenue-intensity method
        revenue_currency: Currency for revenue
        year: Reporting year
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    distribution_channel: str = Field(
        ...,
        description="Channel: e_commerce, retail, wholesale, direct",
    )
    product_category: Optional[str] = Field(
        None,
        description="Product category for average EF lookup",
    )
    quantity: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Product quantity",
    )
    quantity_unit: str = Field(
        "units",
        description="Unit of quantity (units, kg, m3, pallets)",
    )
    revenue: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Revenue for revenue-intensity method",
    )
    revenue_currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
        description="Revenue currency code",
    )
    year: int = Field(
        2024,
        ge=1990,
        le=2100,
        description="Reporting year",
    )


class WarehouseCalcRequest(BaseModel):
    """
    Request model for warehouse and distribution centre emissions (sub-activity 9b).

    Calculates emissions from energy consumption at third-party warehouses,
    distribution centres, and retail storage facilities.

    Attributes:
        tenant_id: Tenant identifier
        warehouse_type: Type of warehouse (ambient, cold_chain, frozen, refrigerated)
        floor_area_m2: Warehouse floor area in square metres
        storage_duration_days: Duration goods are stored
        energy_source: Primary energy source (grid, gas, diesel_generator)
        grid_region: Grid region for electricity EF (e.g. eGRID subregion or country code)
        throughput_tonnes: Throughput in tonnes for activity-based allocation
        utilization_pct: Warehouse space utilization percentage
        year: Reporting year
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    warehouse_type: str = Field(
        "ambient",
        description="Type: ambient, cold_chain, frozen, refrigerated",
    )
    floor_area_m2: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Warehouse floor area in m2",
    )
    storage_duration_days: Decimal = Field(
        ...,
        gt=0,
        description="Duration goods are stored in days",
    )
    energy_source: str = Field(
        "grid",
        description="Primary energy source (grid, gas, diesel_generator)",
    )
    grid_region: Optional[str] = Field(
        None,
        description="Grid region for electricity EF (e.g. US-CAMX, GB, DE)",
    )
    throughput_tonnes: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Throughput in tonnes for allocation",
    )
    utilization_pct: Decimal = Field(
        Decimal("80.0"),
        ge=0,
        le=100,
        description="Warehouse utilization percentage",
    )
    year: int = Field(
        2024,
        ge=1990,
        le=2100,
        description="Reporting year",
    )


class LastMileCalcRequest(BaseModel):
    """
    Request model for last-mile delivery emissions (sub-activity 9d).

    Calculates emissions from final delivery to end consumer including
    parcel delivery, courier, and e-commerce fulfilment.

    Attributes:
        tenant_id: Tenant identifier
        delivery_type: Type of last-mile delivery (parcel, courier, grocery, bulky)
        vehicle_type: Delivery vehicle type (van, cargo_bike, ev_van, motorcycle)
        parcels: Number of parcels or deliveries
        average_distance_km: Average delivery distance per stop
        stops_per_route: Number of stops per delivery route
        failed_delivery_rate: Rate of failed first deliveries requiring re-delivery
        year: Reporting year
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    delivery_type: str = Field(
        "parcel",
        description="Delivery type: parcel, courier, grocery, bulky",
    )
    vehicle_type: str = Field(
        "van",
        description="Vehicle type: van, cargo_bike, ev_van, motorcycle, drone",
    )
    parcels: int = Field(
        ...,
        ge=1,
        description="Number of parcels or deliveries",
    )
    average_distance_km: Decimal = Field(
        ...,
        gt=0,
        description="Average delivery distance per stop in km",
    )
    stops_per_route: int = Field(
        20,
        ge=1,
        le=500,
        description="Number of stops per delivery route",
    )
    failed_delivery_rate: Decimal = Field(
        Decimal("0.05"),
        ge=0,
        le=1,
        description="Rate of failed first deliveries (default 5%)",
    )
    year: int = Field(
        2024,
        ge=1990,
        le=2100,
        description="Reporting year",
    )


class SupplierSpecificRequest(BaseModel):
    """
    Request model for supplier/carrier-specific downstream transportation calculation.

    Uses primary data provided by the carrier or logistics provider (e.g. from
    carrier carbon reports, SmartWay data, or GLEC-accredited tools).

    Attributes:
        tenant_id: Tenant identifier
        carrier_id: Carrier or logistics provider identifier
        carrier_name: Carrier display name
        co2_kg: CO2 emissions reported by carrier
        ch4_kg: CH4 emissions reported by carrier
        n2o_kg: N2O emissions reported by carrier
        co2e_kg: Total CO2e emissions (if pre-calculated by carrier)
        data_source: Source of the carrier data (e.g. SmartWay, GLEC, EPD, carrier_report)
        verification_status: Data verification status (verified, unverified, third_party)
        year: Reporting year
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    carrier_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Carrier identifier",
    )
    carrier_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Carrier display name",
    )
    co2_kg: Decimal = Field(
        ...,
        ge=0,
        description="CO2 emissions in kg (carrier-reported)",
    )
    ch4_kg: Decimal = Field(
        Decimal("0"),
        ge=0,
        description="CH4 emissions in kg",
    )
    n2o_kg: Decimal = Field(
        Decimal("0"),
        ge=0,
        description="N2O emissions in kg",
    )
    co2e_kg: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Total CO2e if pre-calculated by carrier",
    )
    data_source: str = Field(
        "carrier_report",
        description="Data source: smartway, glec, epd, carrier_report",
    )
    verification_status: str = Field(
        "unverified",
        description="Verification status: verified, unverified, third_party",
    )
    year: int = Field(
        2024,
        ge=1990,
        le=2100,
        description="Reporting year",
    )


class BatchCalcRequest(BaseModel):
    """
    Request model for batch downstream transportation calculations.

    Processes multiple shipments in a single request with per-item error
    isolation and aggregated totals.

    Attributes:
        tenant_id: Tenant identifier
        calculations: List of individual calculation requests
        parallel: Whether to execute calculations in parallel
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    calculations: List[CalculateRequest] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of individual calculation requests",
    )
    parallel: bool = Field(
        True,
        description="Execute calculations in parallel",
    )


class PortfolioCalcRequest(BaseModel):
    """
    Request model for full downstream transportation portfolio calculation.

    Processes all downstream transportation and distribution activities for
    a reporting period including outbound transport, warehousing, retail
    storage, and last-mile delivery.

    Attributes:
        tenant_id: Tenant identifier
        reporting_period: Reporting period identifier (e.g. '2024-Q4')
        shipments: List of outbound shipment calculation requests
        warehouses: List of warehouse calculation requests
        last_mile: List of last-mile delivery requests
        include_retail_storage: Whether to include retail storage estimates
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period (e.g. '2024-Q4', '2024')",
    )
    shipments: List[CalculateRequest] = Field(
        default_factory=list,
        description="Outbound shipment calculations (sub-activity 9a)",
    )
    warehouses: List[WarehouseCalcRequest] = Field(
        default_factory=list,
        description="Warehouse calculations (sub-activity 9b)",
    )
    last_mile: List[LastMileCalcRequest] = Field(
        default_factory=list,
        description="Last-mile delivery calculations (sub-activity 9d)",
    )
    include_retail_storage: bool = Field(
        False,
        description="Include estimated retail storage emissions (9c)",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Checks downstream transportation calculations against regulatory
    frameworks for boundary correctness, Incoterm classification,
    completeness of sub-activity reporting, and DQI requirements.

    Attributes:
        tenant_id: Tenant identifier
        calculation_ids: List of calculation IDs to check
        frameworks: List of frameworks to check against
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    calculation_ids: List[str] = Field(
        ...,
        min_items=1,
        description="Calculation IDs to check",
    )
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description=(
            "Frameworks: GHG_PROTOCOL, ISO_14083, CSRD, CDP, SBTI, SB_253, GRI"
        ),
    )


class UncertaintyRequest(BaseModel):
    """
    Request model for uncertainty analysis.

    Supports Monte Carlo simulation, analytical error propagation,
    and IPCC Tier 2 default uncertainty ranges.

    Attributes:
        tenant_id: Tenant identifier
        calculation_ids: Calculation IDs to analyze
        method: Uncertainty method (monte_carlo, analytical, ipcc_tier_2)
        iterations: Monte Carlo iterations
        confidence_level: Confidence level (0.80 to 0.99)
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    calculation_ids: List[str] = Field(
        ...,
        min_items=1,
        description="Calculation IDs to analyze",
    )
    method: str = Field(
        "monte_carlo",
        description="Uncertainty method: monte_carlo, analytical, ipcc_tier_2",
    )
    iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Monte Carlo iterations",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence level for interval",
    )


class PortfolioAnalysisRequest(BaseModel):
    """
    Request model for portfolio-level analysis of downstream transportation.

    Provides hot-spot identification, mode-shift opportunities, and
    reduction pathway analysis across all downstream transport activities.

    Attributes:
        tenant_id: Tenant identifier
        reporting_period: Reporting period
        group_by: Dimensions to group by (mode, sub_activity, carrier, channel)
        top_n: Number of top contributors to return
    """

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period identifier",
    )
    group_by: List[str] = Field(
        default_factory=lambda: ["mode"],
        description="Dimensions to group by: mode, sub_activity, carrier, channel",
    )
    top_n: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of top contributors to return",
    )


# ============================================================================
# RESPONSE MODELS (14)
# ============================================================================


class CalculationResponse(BaseModel):
    """Response model for a single downstream transportation calculation."""

    calculation_id: str = Field(
        ..., description="Unique calculation UUID"
    )
    tenant_id: str = Field(
        ..., description="Tenant identifier"
    )
    shipment_id: Optional[str] = Field(
        None, description="Shipment identifier"
    )
    calculation_method: str = Field(
        ..., description="Method used"
    )
    sub_activity: str = Field(
        ..., description="Sub-activity (9a/9b/9c/9d)"
    )
    mode: Optional[str] = Field(
        None, description="Transport mode"
    )
    distance_km: Optional[str] = Field(
        None, description="Distance in km (Decimal as string)"
    )
    mass_tonnes: Optional[str] = Field(
        None, description="Mass in tonnes (Decimal as string)"
    )
    co2_kg: str = Field(
        ..., description="CO2 emissions in kg (Decimal as string)"
    )
    ch4_kg: str = Field(
        ..., description="CH4 emissions in kg (Decimal as string)"
    )
    n2o_kg: str = Field(
        ..., description="N2O emissions in kg (Decimal as string)"
    )
    co2e_kg: str = Field(
        ..., description="Total CO2e in kg (Decimal as string)"
    )
    wtt_co2e_kg: str = Field(
        "0", description="Well-to-tank CO2e in kg"
    )
    ef_source: str = Field(
        ..., description="Emission factor source"
    )
    ef_value: Optional[str] = Field(
        None, description="Emission factor value used"
    )
    dqi_score: Optional[float] = Field(
        None, description="Data quality indicator score (1-5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )


class ShipmentResponse(BaseModel):
    """Response model for distance-based shipment calculation."""

    calculation_id: str = Field(
        ..., description="Unique calculation UUID"
    )
    mode: str = Field(
        ..., description="Transport mode"
    )
    vehicle_type: Optional[str] = Field(
        None, description="Vehicle type used"
    )
    distance_km: str = Field(
        ..., description="Distance in km"
    )
    mass_tonnes: str = Field(
        ..., description="Mass in tonnes"
    )
    tonne_km: str = Field(
        ..., description="Tonne-kilometres"
    )
    co2e_kg: str = Field(
        ..., description="Total CO2e in kg"
    )
    wtt_co2e_kg: str = Field(
        "0", description="Well-to-tank CO2e in kg"
    )
    ef_source: str = Field(
        ..., description="Emission factor source"
    )
    ef_value: str = Field(
        ..., description="EF value (kgCO2e/tkm)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class WarehouseResponse(BaseModel):
    """Response model for warehouse storage emissions calculation."""

    calculation_id: str = Field(
        ..., description="Unique calculation UUID"
    )
    warehouse_type: str = Field(
        ..., description="Warehouse type"
    )
    floor_area_m2: Optional[str] = Field(
        None, description="Floor area in m2"
    )
    storage_duration_days: str = Field(
        ..., description="Storage duration in days"
    )
    energy_kwh: str = Field(
        ..., description="Estimated energy consumption in kWh"
    )
    co2e_kg: str = Field(
        ..., description="Total CO2e in kg"
    )
    ef_source: str = Field(
        ..., description="Emission factor source (grid region)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class LastMileResponse(BaseModel):
    """Response model for last-mile delivery emissions calculation."""

    calculation_id: str = Field(
        ..., description="Unique calculation UUID"
    )
    delivery_type: str = Field(
        ..., description="Delivery type"
    )
    vehicle_type: str = Field(
        ..., description="Delivery vehicle type"
    )
    parcels: int = Field(
        ..., description="Number of parcels"
    )
    total_distance_km: str = Field(
        ..., description="Total delivery distance in km"
    )
    co2e_kg: str = Field(
        ..., description="Total CO2e in kg"
    )
    co2e_per_parcel_kg: str = Field(
        ..., description="CO2e per parcel in kg"
    )
    failed_delivery_co2e_kg: str = Field(
        "0", description="Additional CO2e from failed deliveries"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class BatchResponse(BaseModel):
    """Response model for batch calculations."""

    batch_id: str = Field(
        ..., description="Unique batch UUID"
    )
    tenant_id: str = Field(
        ..., description="Tenant identifier"
    )
    total_calculations: int = Field(
        ..., description="Total calculations requested"
    )
    successful: int = Field(
        ..., description="Successful calculations"
    )
    failed: int = Field(
        ..., description="Failed calculations"
    )
    total_co2e_kg: str = Field(
        ..., description="Total CO2e across all calculations"
    )
    results: List[CalculationResponse] = Field(
        ..., description="Individual calculation results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-item error details"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )


class ComplianceResponse(BaseModel):
    """Response model for compliance checking."""

    check_id: str = Field(
        ..., description="Unique compliance check UUID"
    )
    tenant_id: str = Field(
        ..., description="Tenant identifier"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status: PASS, WARNING, FAIL"
    )
    framework_results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    incoterm_classification: Optional[Dict[str, Any]] = Field(
        None, description="Incoterm boundary classification result"
    )
    checked_at: str = Field(
        ..., description="ISO 8601 check timestamp"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


class AggregationResponse(BaseModel):
    """Response model for aggregated emissions."""

    tenant_id: str = Field(
        ..., description="Tenant identifier"
    )
    group_by: List[str] = Field(
        ..., description="Dimensions grouped by"
    )
    from_date: Optional[str] = Field(
        None, description="Start date filter"
    )
    to_date: Optional[str] = Field(
        None, description="End date filter"
    )
    aggregations: List[Dict[str, Any]] = Field(
        ..., description="Aggregated data by group"
    )
    total_co2e_kg: str = Field(
        ..., description="Total CO2e across all groups"
    )
    total_calculations: int = Field(
        ..., description="Total calculations aggregated"
    )


class EmissionFactorResponse(BaseModel):
    """Response model for emission factor lookup by transport mode."""

    mode: str = Field(
        ..., description="Transport mode"
    )
    factors: List[Dict[str, Any]] = Field(
        ..., description="List of emission factors for the mode"
    )
    total_count: int = Field(
        ..., description="Number of factors returned"
    )
    source: str = Field(
        ..., description="Primary EF source (DEFRA, EPA, GLEC, etc.)"
    )


class WarehouseBenchmarkResponse(BaseModel):
    """Response model for warehouse energy benchmark data."""

    benchmarks: List[Dict[str, Any]] = Field(
        ..., description="Warehouse benchmarks by type and region"
    )
    total_count: int = Field(
        ..., description="Number of benchmarks returned"
    )
    source: str = Field(
        ..., description="Benchmark data source"
    )


class LastMileFactorResponse(BaseModel):
    """Response model for last-mile delivery emission factors."""

    factors: List[Dict[str, Any]] = Field(
        ..., description="Last-mile delivery factors by vehicle type"
    )
    total_count: int = Field(
        ..., description="Number of factors returned"
    )
    source: str = Field(
        ..., description="Factor data source"
    )


class IncotermResponse(BaseModel):
    """Response model for Incoterm classification rules."""

    incoterms: List[Dict[str, Any]] = Field(
        ..., description="Incoterm rules for Cat 4 vs Cat 9 boundary"
    )
    total_count: int = Field(
        ..., description="Number of Incoterm rules"
    )
    source: str = Field(
        ..., description="Classification source (GHG Protocol)"
    )


class ProvenanceResponse(BaseModel):
    """Response model for provenance chain verification."""

    calculation_id: str = Field(
        ..., description="Calculation UUID"
    )
    chain: List[Dict[str, Any]] = Field(
        ..., description="Ordered list of provenance stage records"
    )
    is_valid: bool = Field(
        ..., description="Whether the provenance chain is intact"
    )
    root_hash: str = Field(
        ..., description="Root SHA-256 hash of the chain"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        ..., description="Service status: healthy, degraded, unhealthy"
    )
    agent_id: str = Field(
        ..., description="Agent identifier (GL-MRV-S3-009)"
    )
    version: str = Field(
        ..., description="Service version"
    )
    engines_status: Dict[str, bool] = Field(
        ..., description="Per-engine availability status"
    )
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )


class UncertaintyResponse(BaseModel):
    """Response model for uncertainty analysis."""

    analysis_id: str = Field(
        ..., description="Unique analysis UUID"
    )
    tenant_id: str = Field(
        ..., description="Tenant identifier"
    )
    method: str = Field(
        ..., description="Uncertainty method used"
    )
    confidence_level: float = Field(
        ..., description="Confidence level"
    )
    mean_co2e_kg: str = Field(
        ..., description="Mean CO2e in kg"
    )
    std_dev_co2e_kg: str = Field(
        ..., description="Standard deviation CO2e in kg"
    )
    ci_lower_kg: str = Field(
        ..., description="Confidence interval lower bound in kg"
    )
    ci_upper_kg: str = Field(
        ..., description="Confidence interval upper bound in kg"
    )
    relative_uncertainty_pct: str = Field(
        ..., description="Relative uncertainty percentage"
    )
    iterations: int = Field(
        ..., description="Monte Carlo iterations performed"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINTS - CORE CALCULATIONS (9)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate downstream transportation emissions (full pipeline)",
    description=(
        "Calculate GHG emissions for a downstream transportation activity "
        "through the full 10-stage pipeline. Supports all transport modes, "
        "4 calculation methods, and all sub-activities (9a-9d). Routes to the "
        "appropriate engine based on method and sub-activity. Returns "
        "deterministic results with SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: CalculateRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate downstream transportation emissions through the full pipeline.

    Args:
        request: Calculation request with activity data.
        service: DownstreamTransportService instance.

    Returns:
        CalculationResponse with emissions and provenance hash.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Calculating emissions for tenant={request.tenant_id}, "
            f"method={request.calculation_method}, sub_activity={request.sub_activity}"
        )

        result = await service.calculate(request.dict())

        return CalculationResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_emissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_emissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Calculation failed",
        )


@router.post(
    "/calculate/distance",
    response_model=ShipmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate distance-based downstream transport emissions",
    description=(
        "Calculate GHG emissions using the distance-based method "
        "(tonne-km x mode-specific emission factor). Supports road, rail, "
        "air, sea, and pipeline modes with DEFRA, EPA SmartWay, GLEC, "
        "and ICAO emission factors. Includes optional return trip with "
        "configurable load factor."
    ),
)
async def calculate_distance_based(
    request: DistanceCalcRequest,
    service=Depends(get_service),
) -> ShipmentResponse:
    """
    Calculate distance-based downstream transportation emissions.

    Args:
        request: Distance calculation request with mode, distance, and mass.
        service: DownstreamTransportService instance.

    Returns:
        ShipmentResponse with tonne-km, emissions, and EF details.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Calculating distance-based emissions: mode={request.mode}, "
            f"distance={request.distance_km}km, mass={request.mass_tonnes}t"
        )

        result = await service.calculate_distance(request.dict())

        return ShipmentResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_distance_based: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_distance_based: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Distance-based calculation failed",
        )


@router.post(
    "/calculate/spend",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate spend-based downstream transport emissions",
    description=(
        "Calculate GHG emissions using the spend-based method with EEIO "
        "factors. Applies CPI deflation, currency conversion, and optional "
        "margin removal. Maps NAICS industry codes to emission intensities "
        "for downstream transport and distribution activities."
    ),
)
async def calculate_spend_based(
    request: SpendCalcRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate spend-based downstream transportation emissions.

    Args:
        request: Spend calculation request with amount and currency.
        service: DownstreamTransportService instance.

    Returns:
        CalculationResponse with spend-based emissions.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Calculating spend-based emissions: amount={request.spend_amount} "
            f"{request.currency}, naics={request.naics_code}"
        )

        result = await service.calculate_spend(request.dict())

        return CalculationResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_spend_based: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_spend_based: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Spend-based calculation failed",
        )


@router.post(
    "/calculate/average-data",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate average-data downstream transport emissions",
    description=(
        "Calculate GHG emissions using industry-average emission intensities "
        "per product unit, revenue, or weight by distribution channel. "
        "Suitable when shipment-level data is unavailable. Uses channel "
        "defaults for e-commerce, retail, wholesale, and direct distribution."
    ),
)
async def calculate_average_data(
    request: AverageDataCalcRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate average-data downstream transportation emissions.

    Args:
        request: Average-data request with distribution channel and quantity.
        service: DownstreamTransportService instance.

    Returns:
        CalculationResponse with average-data emissions.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Calculating average-data emissions: channel={request.distribution_channel}, "
            f"quantity={request.quantity} {request.quantity_unit}"
        )

        result = await service.calculate_average_data(request.dict())

        return CalculationResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_average_data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_average_data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Average-data calculation failed",
        )


@router.post(
    "/calculate/warehouse",
    response_model=WarehouseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate warehouse and distribution centre emissions",
    description=(
        "Calculate GHG emissions from energy consumption at third-party "
        "warehouses, distribution centres, and cold storage facilities "
        "(sub-activity 9b). Supports ambient, cold chain, frozen, and "
        "refrigerated warehouse types with region-specific grid electricity "
        "emission factors."
    ),
)
async def calculate_warehouse(
    request: WarehouseCalcRequest,
    service=Depends(get_service),
) -> WarehouseResponse:
    """
    Calculate warehouse storage emissions.

    Args:
        request: Warehouse calculation request with type and area.
        service: DownstreamTransportService instance.

    Returns:
        WarehouseResponse with energy and emissions data.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Calculating warehouse emissions: type={request.warehouse_type}, "
            f"area={request.floor_area_m2}m2, duration={request.storage_duration_days}d"
        )

        result = await service.calculate_warehouse(request.dict())

        return WarehouseResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_warehouse: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_warehouse: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Warehouse calculation failed",
        )


@router.post(
    "/calculate/last-mile",
    response_model=LastMileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate last-mile delivery emissions",
    description=(
        "Calculate GHG emissions from last-mile delivery to end consumers "
        "(sub-activity 9d). Supports parcel delivery, courier, grocery, and "
        "bulky item delivery with van, cargo bike, EV van, motorcycle, and "
        "drone vehicle types. Includes failed delivery re-delivery emissions."
    ),
)
async def calculate_last_mile(
    request: LastMileCalcRequest,
    service=Depends(get_service),
) -> LastMileResponse:
    """
    Calculate last-mile delivery emissions.

    Args:
        request: Last-mile request with delivery type and vehicle.
        service: DownstreamTransportService instance.

    Returns:
        LastMileResponse with per-parcel and total emissions.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Calculating last-mile emissions: type={request.delivery_type}, "
            f"vehicle={request.vehicle_type}, parcels={request.parcels}"
        )

        result = await service.calculate_last_mile(request.dict())

        return LastMileResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_last_mile: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_last_mile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Last-mile calculation failed",
        )


@router.post(
    "/calculate/supplier-specific",
    response_model=CalculationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate supplier/carrier-specific emissions",
    description=(
        "Record and validate emissions data provided directly by the "
        "carrier or logistics provider. Supports SmartWay, GLEC-accredited "
        "tools, EPD, and carrier report data sources. Validates gas-level "
        "data and applies GWP conversions."
    ),
)
async def calculate_supplier_specific(
    request: SupplierSpecificRequest,
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Calculate emissions from carrier-specific primary data.

    Args:
        request: Supplier-specific request with carrier-reported emissions.
        service: DownstreamTransportService instance.

    Returns:
        CalculationResponse with validated carrier emissions.

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures.
    """
    try:
        logger.info(
            f"Processing supplier-specific emissions: carrier={request.carrier_name}, "
            f"source={request.data_source}"
        )

        result = await service.calculate_supplier_specific(request.dict())

        return CalculationResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_supplier_specific: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_supplier_specific: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supplier-specific calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate downstream transport emissions",
    description=(
        "Calculate GHG emissions for multiple downstream transportation "
        "activities in a single request. Processes up to 10,000 calculations "
        "with optional parallel execution and per-item error isolation. "
        "Returns aggregated totals with individual results."
    ),
)
async def calculate_batch(
    request: BatchCalcRequest,
    service=Depends(get_service),
) -> BatchResponse:
    """
    Calculate batch downstream transportation emissions.

    Args:
        request: Batch request with list of calculations.
        service: DownstreamTransportService instance.

    Returns:
        BatchResponse with aggregated and individual results.

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures.
    """
    try:
        logger.info(
            f"Calculating batch emissions: {len(request.calculations)} items, "
            f"parallel={request.parallel}"
        )

        result = await service.calculate_batch(request.dict())

        return BatchResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed",
        )


@router.post(
    "/calculate/portfolio",
    response_model=BatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate full downstream transportation portfolio",
    description=(
        "Calculate GHG emissions for a complete downstream transportation "
        "portfolio covering all sub-activities: outbound transport (9a), "
        "warehouse/DC operations (9b), retail storage (9c), and last-mile "
        "delivery (9d). Provides total category 9 footprint for reporting."
    ),
)
async def calculate_portfolio(
    request: PortfolioCalcRequest,
    service=Depends(get_service),
) -> BatchResponse:
    """
    Calculate full downstream transportation portfolio emissions.

    Args:
        request: Portfolio request with shipments, warehouses, and last-mile.
        service: DownstreamTransportService instance.

    Returns:
        BatchResponse with portfolio-level results.

    Raises:
        HTTPException: 400 for validation errors, 500 for portfolio failures.
    """
    try:
        logger.info(
            f"Calculating portfolio: {len(request.shipments)} shipments, "
            f"{len(request.warehouses)} warehouses, "
            f"{len(request.last_mile)} last-mile"
        )

        result = await service.calculate_portfolio(request.dict())

        return BatchResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in calculate_portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_portfolio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio calculation failed",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE (1)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check downstream transport compliance",
    description=(
        "Check downstream transportation calculations against regulatory "
        "frameworks. Validates boundary correctness per Incoterms, "
        "completeness of sub-activity reporting (9a-9d), data quality, "
        "and disclosure requirements. Supports GHG Protocol, ISO 14083, "
        "CSRD, CDP, SBTi, SB 253, and GRI."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceResponse:
    """
    Check compliance against regulatory frameworks.

    Args:
        request: Compliance check request with calculation IDs and frameworks.
        service: DownstreamTransportService instance.

    Returns:
        ComplianceResponse with per-framework findings.

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures.
    """
    try:
        logger.info(
            f"Checking compliance for {len(request.calculation_ids)} calculations, "
            f"frameworks={request.frameworks}"
        )

        result = await service.check_compliance(request.dict())

        return ComplianceResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in check_compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in check_compliance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed",
        )


# ============================================================================
# ENDPOINTS - CALCULATION CRUD (3)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific downstream "
        "transportation calculation including full input/output payload, "
        "provenance hash, and calculation metadata."
    ),
)
async def get_calculation_detail(
    calculation_id: str = Path(..., description="Calculation UUID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service=Depends(get_service),
) -> CalculationResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calculation_id: Calculation UUID.
        tenant_id: Tenant identifier.
        service: DownstreamTransportService instance.

    Returns:
        CalculationResponse with full calculation data.

    Raises:
        HTTPException: 404 if not found, 500 for failures.
    """
    try:
        logger.info(f"Getting calculation detail: {calculation_id}")

        result = await service.get_calculation(calculation_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return CalculationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_calculation_detail: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation",
        )


@router.get(
    "/calculations",
    response_model=Dict[str, Any],
    summary="List downstream transportation calculations",
    description=(
        "Retrieve a paginated list of downstream transportation calculations. "
        "Supports filtering by tenant, mode, sub-activity, method, and date range."
    ),
)
async def list_calculations(
    tenant_id: str = Query(..., description="Tenant identifier"),
    mode: Optional[str] = Query(None, description="Filter by transport mode"),
    sub_activity: Optional[str] = Query(None, description="Filter by sub-activity (9a/9b/9c/9d)"),
    method: Optional[str] = Query(None, description="Filter by calculation method"),
    from_date: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    to_date: Optional[str] = Query(None, description="End date (ISO 8601)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Results per page"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    List downstream transportation calculations with filters and pagination.

    Args:
        tenant_id: Tenant identifier.
        mode: Optional transport mode filter.
        sub_activity: Optional sub-activity filter.
        method: Optional calculation method filter.
        from_date: Optional start date.
        to_date: Optional end date.
        page: Page number (1-indexed).
        page_size: Results per page.
        service: DownstreamTransportService instance.

    Returns:
        Dictionary with calculations list, total count, page, and page_size.

    Raises:
        HTTPException: 500 for listing failures.
    """
    try:
        logger.info(
            f"Listing calculations: tenant={tenant_id}, mode={mode}, "
            f"sub_activity={sub_activity}, page={page}"
        )

        filters = {
            "tenant_id": tenant_id,
            "mode": mode,
            "sub_activity": sub_activity,
            "method": method,
            "from_date": from_date,
            "to_date": to_date,
            "page": page,
            "page_size": page_size,
        }

        result = await service.list_calculations(filters)

        return result

    except Exception as e:
        logger.error(f"Error in list_calculations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations",
        )


@router.delete(
    "/calculations/{calculation_id}",
    response_model=Dict[str, Any],
    summary="Delete downstream transportation calculation",
    description=(
        "Soft-delete a specific downstream transportation calculation. "
        "Marks the calculation as deleted with audit trail; data is "
        "retained for regulatory compliance."
    ),
)
async def delete_calculation(
    calculation_id: str = Path(..., description="Calculation UUID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Soft-delete a specific calculation.

    Args:
        calculation_id: Calculation UUID.
        tenant_id: Tenant identifier.
        service: DownstreamTransportService instance.

    Returns:
        Dictionary with deletion status.

    Raises:
        HTTPException: 404 if not found, 500 for failures.
    """
    try:
        logger.info(f"Deleting calculation: {calculation_id}")

        result = await service.delete_calculation(calculation_id, tenant_id)

        if not result.get("deleted", False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_calculation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation",
        )


# ============================================================================
# ENDPOINTS - REFERENCE DATA (4)
# ============================================================================


@router.get(
    "/emission-factors/{mode}",
    response_model=EmissionFactorResponse,
    summary="Get emission factors by transport mode",
    description=(
        "Retrieve downstream transportation emission factors for a specific "
        "transport mode. Returns vehicle/vessel types with their kgCO2e/tkm "
        "factors from DEFRA, EPA SmartWay, GLEC, ICAO, and IMO sources."
    ),
)
async def get_emission_factors_by_mode(
    mode: str = Path(
        ...,
        description="Transport mode (road, rail, air, sea, pipeline)",
    ),
    year: Optional[int] = Query(None, ge=1990, le=2100, description="Factor year"),
    service=Depends(get_service),
) -> EmissionFactorResponse:
    """
    Get emission factors for a specific transport mode.

    Args:
        mode: Transport mode identifier.
        year: Optional year filter for factor vintage.
        service: DownstreamTransportService instance.

    Returns:
        EmissionFactorResponse with mode-specific factors.

    Raises:
        HTTPException: 400 for invalid mode, 500 for retrieval failures.
    """
    try:
        logger.info(f"Getting emission factors for mode: {mode}, year: {year}")

        result = await service.get_emission_factors(mode, year)

        return EmissionFactorResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in get_emission_factors_by_mode: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in get_emission_factors_by_mode: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


@router.get(
    "/warehouse-benchmarks",
    response_model=WarehouseBenchmarkResponse,
    summary="Get warehouse energy benchmarks",
    description=(
        "Retrieve warehouse energy consumption benchmarks by type "
        "(ambient, cold chain, frozen, refrigerated) and region. "
        "Used for estimating warehouse emissions when primary energy "
        "data is unavailable."
    ),
)
async def get_warehouse_benchmarks(
    warehouse_type: Optional[str] = Query(
        None,
        description="Filter by warehouse type (ambient, cold_chain, frozen, refrigerated)",
    ),
    region: Optional[str] = Query(
        None,
        description="Filter by region code (e.g. US, EU, GB, GLOBAL)",
    ),
    service=Depends(get_service),
) -> WarehouseBenchmarkResponse:
    """
    Get warehouse energy consumption benchmarks.

    Args:
        warehouse_type: Optional warehouse type filter.
        region: Optional region filter.
        service: DownstreamTransportService instance.

    Returns:
        WarehouseBenchmarkResponse with benchmark data.

    Raises:
        HTTPException: 500 for retrieval failures.
    """
    try:
        logger.info(
            f"Getting warehouse benchmarks: type={warehouse_type}, region={region}"
        )

        result = await service.get_warehouse_benchmarks(warehouse_type, region)

        return WarehouseBenchmarkResponse(**result)

    except Exception as e:
        logger.error(f"Error in get_warehouse_benchmarks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve warehouse benchmarks",
        )


@router.get(
    "/last-mile-factors",
    response_model=LastMileFactorResponse,
    summary="Get last-mile delivery emission factors",
    description=(
        "Retrieve emission factors for last-mile delivery vehicles "
        "including vans, cargo bikes, EV vans, motorcycles, and drones. "
        "Returns kgCO2e per delivery-km and per parcel factors."
    ),
)
async def get_last_mile_factors(
    vehicle_type: Optional[str] = Query(
        None,
        description="Filter by vehicle type (van, cargo_bike, ev_van, motorcycle, drone)",
    ),
    service=Depends(get_service),
) -> LastMileFactorResponse:
    """
    Get last-mile delivery emission factors.

    Args:
        vehicle_type: Optional vehicle type filter.
        service: DownstreamTransportService instance.

    Returns:
        LastMileFactorResponse with delivery factors.

    Raises:
        HTTPException: 500 for retrieval failures.
    """
    try:
        logger.info(f"Getting last-mile factors: vehicle_type={vehicle_type}")

        result = await service.get_last_mile_factors(vehicle_type)

        return LastMileFactorResponse(**result)

    except Exception as e:
        logger.error(f"Error in get_last_mile_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve last-mile factors",
        )


@router.get(
    "/incoterm-classification",
    response_model=IncotermResponse,
    summary="Get Incoterm classification rules",
    description=(
        "Retrieve Incoterm rules for classifying transportation as Category 4 "
        "(upstream, paid by reporter) vs Category 9 (downstream, not paid by "
        "reporter). Based on GHG Protocol Scope 3 guidance for transfer of "
        "control and cost responsibility."
    ),
)
async def get_incoterm_classification(
    incoterm: Optional[str] = Query(
        None,
        description="Filter by specific Incoterm code (EXW, FOB, CIF, DAP, DDP, etc.)",
    ),
    service=Depends(get_service),
) -> IncotermResponse:
    """
    Get Incoterm classification rules for Cat 4 vs Cat 9 boundary.

    Args:
        incoterm: Optional specific Incoterm filter.
        service: DownstreamTransportService instance.

    Returns:
        IncotermResponse with classification rules.

    Raises:
        HTTPException: 500 for retrieval failures.
    """
    try:
        logger.info(f"Getting Incoterm classification: incoterm={incoterm}")

        result = await service.get_incoterm_rules(incoterm)

        return IncotermResponse(**result)

    except Exception as e:
        logger.error(f"Error in get_incoterm_classification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve Incoterm classification",
        )


# ============================================================================
# ENDPOINTS - AGGREGATION (1)
# ============================================================================


@router.get(
    "/aggregations",
    response_model=AggregationResponse,
    summary="Get aggregated downstream transport emissions",
    description=(
        "Retrieve aggregated downstream transportation emissions grouped "
        "by specified dimensions (mode, sub_activity, carrier, channel). "
        "Supports date range filtering. Useful for Scope 3 Category 9 "
        "reporting and analysis."
    ),
)
async def get_aggregations(
    tenant_id: str = Query(..., description="Tenant identifier"),
    group_by: List[str] = Query(
        ["mode"],
        description="Dimensions to group by: mode, sub_activity, carrier, channel",
    ),
    from_date: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    to_date: Optional[str] = Query(None, description="End date (ISO 8601)"),
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions data grouped by specified dimensions.

    Args:
        tenant_id: Tenant identifier.
        group_by: Dimensions to group by.
        from_date: Optional start date filter.
        to_date: Optional end date filter.
        service: DownstreamTransportService instance.

    Returns:
        AggregationResponse with grouped emissions data.

    Raises:
        HTTPException: 500 for aggregation failures.
    """
    try:
        logger.info(
            f"Getting aggregations: tenant={tenant_id}, group_by={group_by}"
        )

        filters = {
            "tenant_id": tenant_id,
            "group_by": group_by,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.get_aggregations(filters)

        return AggregationResponse(**result)

    except Exception as e:
        logger.error(f"Error in get_aggregations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed",
        )


# ============================================================================
# ENDPOINTS - PROVENANCE (1)
# ============================================================================


@router.get(
    "/provenance/{calculation_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a downstream "
        "transportation calculation. Includes all 10 pipeline stages "
        "(validate, classify_incoterm, resolve_efs, calculate, warehouse_alloc, "
        "last_mile, compliance, aggregate, seal, persist) with per-stage "
        "hashes and verification."
    ),
)
async def get_provenance(
    calculation_id: str = Path(..., description="Calculation UUID"),
    tenant_id: str = Query(..., description="Tenant identifier"),
    service=Depends(get_service),
) -> ProvenanceResponse:
    """
    Get provenance chain for a specific calculation.

    Args:
        calculation_id: Calculation UUID.
        tenant_id: Tenant identifier.
        service: DownstreamTransportService instance.

    Returns:
        ProvenanceResponse with chain stages and verification status.

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures.
    """
    try:
        logger.info(f"Getting provenance for calculation: {calculation_id}")

        result = await service.get_provenance(calculation_id, tenant_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provenance for {calculation_id} not found",
            )

        return ProvenanceResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_provenance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provenance",
        )


# ============================================================================
# ENDPOINTS - HEALTH (1)
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Health check endpoint for the Downstream Transportation Agent "
        "(GL-MRV-S3-009). Returns service status, agent identifier, "
        "version, per-engine availability, and uptime. No authentication "
        "required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no authentication required).

    Returns:
        HealthResponse with service status and per-engine health.
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        # Attempt to check engine status via service
        try:
            service = get_service()
            health_data = await service.get_health()
            return HealthResponse(
                status=health_data.get("status", "healthy"),
                agent_id="GL-MRV-S3-009",
                version="1.0.0",
                engines_status=health_data.get("engines_status", {}),
                uptime_seconds=round(uptime, 2),
            )
        except Exception:
            # Service not initialized yet; return minimal healthy response
            return HealthResponse(
                status="healthy",
                agent_id="GL-MRV-S3-009",
                version="1.0.0",
                engines_status={},
                uptime_seconds=round(uptime, 2),
            )

    except Exception as e:
        logger.error(f"Error in health_check: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-009",
            version="1.0.0",
            engines_status={},
            uptime_seconds=0.0,
        )


# ============================================================================
# ENDPOINTS - UNCERTAINTY (1)
# ============================================================================


@router.post(
    "/uncertainty/analyze",
    response_model=UncertaintyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze calculation uncertainty",
    description=(
        "Perform uncertainty analysis on downstream transportation emissions "
        "calculations. Supports Monte Carlo simulation, analytical error "
        "propagation, and IPCC Tier 2 default ranges. Returns mean, "
        "standard deviation, confidence intervals, and relative uncertainty."
    ),
)
async def analyze_uncertainty(
    request: UncertaintyRequest,
    service=Depends(get_service),
) -> UncertaintyResponse:
    """
    Perform uncertainty analysis on downstream transport calculations.

    Args:
        request: Uncertainty analysis request.
        service: DownstreamTransportService instance.

    Returns:
        UncertaintyResponse with statistical uncertainty metrics.

    Raises:
        HTTPException: 400 for invalid method, 500 for analysis failures.
    """
    try:
        logger.info(
            f"Analyzing uncertainty: method={request.method}, "
            f"iterations={request.iterations}, "
            f"confidence={request.confidence_level}"
        )

        result = await service.analyze_uncertainty(request.dict())

        return UncertaintyResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error in analyze_uncertainty: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in analyze_uncertainty: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uncertainty analysis failed",
        )


# ============================================================================
# ENDPOINTS - PORTFOLIO ANALYSIS (1)
# ============================================================================


@router.post(
    "/portfolio/analyze",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Analyze downstream transport portfolio",
    description=(
        "Perform portfolio-level analysis of downstream transportation "
        "emissions including hot-spot identification by mode, sub-activity, "
        "and carrier. Provides mode-shift opportunities, warehouse "
        "optimization recommendations, and last-mile improvement paths."
    ),
)
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Analyze downstream transportation portfolio for insights.

    Args:
        request: Portfolio analysis request with grouping and period.
        service: DownstreamTransportService instance.

    Returns:
        Dictionary with hot-spots, opportunities, and recommendations.

    Raises:
        HTTPException: 400 for invalid input, 500 for analysis failures.
    """
    try:
        logger.info(
            f"Analyzing portfolio: tenant={request.tenant_id}, "
            f"period={request.reporting_period}, group_by={request.group_by}"
        )

        result = await service.analyze_portfolio(request.dict())

        return result

    except ValueError as e:
        logger.error(f"Validation error in analyze_portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed",
        )
