"""
End-of-Life Treatment of Sold Products API Router - AGENT-MRV-025

This module implements the FastAPI router for end-of-life treatment of sold
products emissions calculations following GHG Protocol Scope 3 Category 12
requirements.

Provides 22 REST endpoints for:
- Full pipeline emissions calculations (single and batch)
- Treatment-specific calculations (landfill FOD, incineration/WtE, recycling cut-off)
- Waste-type-specific, average-data, producer-specific, and hybrid methods
- Portfolio analysis with circularity scoring
- Emission factor lookup and product composition management
- Regional treatment mix profiles
- Avoided emissions tracking (recycling credits, energy recovery)
- Compliance checking across 7 regulatory frameworks
- Uncertainty analysis (Monte Carlo, analytical)
- Aggregations by period, treatment, and material
- Provenance tracking with SHA-256 chain verification
- Health monitoring

Key Differentiator from Category 5 (Waste Generated in Operations):
    Category 5 covers waste from the reporting company's own operations.
    Category 12 covers end-of-life treatment of products SOLD by the company,
    disposed of by downstream consumers and third parties.

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.end_of_life_treatment.api.router import router
    >>> app = FastAPI()
    >>> app.include_router(router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal
import json
import logging
import uuid
from datetime import datetime, date

logger = logging.getLogger(__name__)


# ============================================================================
# DECIMAL JSON ENCODER
# ============================================================================


class DecimalEncoder(json.JSONEncoder):
    """
    JSON encoder that handles Decimal values.

    Converts Decimal instances to float for JSON serialization
    while preserving precision for API responses.
    """

    def default(self, o: Any) -> Any:
        """Encode Decimal as float, delegate others to parent."""
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================


router = APIRouter(
    prefix="/api/v1/end-of-life-treatment",
    tags=["End-of-Life Treatment"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create EndOfLifeTreatmentService singleton instance.

    Returns:
        EndOfLifeTreatmentService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.end_of_life_treatment.service import (
                EndOfLifeTreatmentService,
            )
            _service_instance = EndOfLifeTreatmentService()
            logger.info("EndOfLifeTreatmentService initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize EndOfLifeTreatmentService: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS (12)
# ============================================================================


class MaterialComposition(BaseModel):
    """
    Single material entry within a product bill of materials (BOM).

    Attributes:
        material_type: Material identifier (e.g. HDPE, PET, STEEL, GLASS)
        weight_kg: Mass of material in kilograms per unit of product
        fraction: Mass fraction (0-1) relative to total product weight
        treatment_method: Optional override treatment (LANDFILL, INCINERATION, etc.)
    """

    material_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Material type code (e.g. HDPE, PET, STEEL, GLASS, CARDBOARD)",
    )
    weight_kg: Optional[float] = Field(
        None,
        ge=0,
        description="Mass of material per unit product (kg)",
    )
    fraction: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Mass fraction of total product weight (0.0 - 1.0)",
    )
    treatment_method: Optional[str] = Field(
        None,
        description="Override treatment method for this material",
    )

    @validator("fraction")
    def validate_fraction_or_weight(cls, v: Optional[float], values: Dict) -> Optional[float]:
        """Ensure at least one of weight_kg or fraction is provided."""
        if v is None and values.get("weight_kg") is None:
            raise ValueError(
                "At least one of weight_kg or fraction must be provided"
            )
        return v


class TreatmentScenario(BaseModel):
    """
    Treatment scenario specifying disposal pathway fractions.

    Attributes:
        landfill_fraction: Fraction of product going to landfill
        incineration_fraction: Fraction going to incineration/WtE
        recycling_fraction: Fraction going to recycling
        composting_fraction: Fraction going to composting/AD
        open_burning_fraction: Fraction going to open burning
        wastewater_fraction: Fraction going to wastewater treatment
    """

    landfill_fraction: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Fraction of product sent to landfill",
    )
    incineration_fraction: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Fraction sent to incineration/WtE",
    )
    recycling_fraction: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Fraction sent to recycling",
    )
    composting_fraction: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Fraction sent to composting/anaerobic digestion",
    )
    open_burning_fraction: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Fraction sent to open burning (developing regions)",
    )
    wastewater_fraction: float = Field(
        0.0,
        ge=0,
        le=1,
        description="Fraction treated as wastewater",
    )

    @validator("wastewater_fraction")
    def validate_fractions_sum(cls, v: float, values: Dict) -> float:
        """Validate that treatment fractions sum to approximately 1.0."""
        total = (
            values.get("landfill_fraction", 0.0)
            + values.get("incineration_fraction", 0.0)
            + values.get("recycling_fraction", 0.0)
            + values.get("composting_fraction", 0.0)
            + values.get("open_burning_fraction", 0.0)
            + v
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Treatment fractions must sum to 1.0 (got {total:.4f})"
            )
        return v


class ProductEntry(BaseModel):
    """
    A single product entry for end-of-life treatment calculation.

    Attributes:
        product_id: Unique product identifier
        product_category: Product category for average-data lookups
        product_name: Human-readable product name
        units_sold: Number of units sold in the reporting year
        weight_per_unit_kg: Weight of a single product unit in kg
        materials: Optional bill of materials (BOM)
        treatment_scenario: Optional treatment scenario override
        region: Region/country where product is disposed
    """

    product_id: Optional[str] = Field(
        None,
        description="Unique product identifier or SKU",
    )
    product_category: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Product category (e.g. CONSUMER_ELECTRONICS, PACKAGING, TEXTILES)",
    )
    product_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Human-readable product name",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        description="Units sold in the reporting year",
    )
    weight_per_unit_kg: Optional[float] = Field(
        None,
        gt=0,
        description="Weight per unit (kg); uses default if omitted",
    )
    materials: Optional[List[MaterialComposition]] = Field(
        None,
        description="Bill of materials (BOM) for waste-type-specific method",
    )
    treatment_scenario: Optional[TreatmentScenario] = Field(
        None,
        description="Custom treatment scenario; uses regional default if omitted",
    )
    region: Optional[str] = Field(
        None,
        max_length=50,
        description="ISO country code or region (US, EU, CN, GLOBAL)",
    )


class FullPipelineRequest(BaseModel):
    """
    Request model for full end-of-life treatment pipeline calculation.

    Runs the complete 10-stage pipeline: validate -> classify -> normalize ->
    resolve_efs -> calculate -> allocate -> aggregate -> compliance ->
    provenance -> seal.

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        region: Default region for treatment mixes
        products: List of products to calculate EoL emissions
        include_avoided_emissions: Include recycling/energy credits
        gwp_version: GWP version for CO2e conversion
        metadata: Additional metadata for audit trail
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Default region for treatment mix lookup (US, EU, CN, GLOBAL)",
    )
    products: List[ProductEntry] = Field(
        ...,
        min_items=1,
        max_items=5000,
        description="List of products sold in the reporting year",
    )
    include_avoided_emissions: bool = Field(
        True,
        description="Include avoided emissions from recycling/energy recovery",
    )
    gwp_version: str = Field(
        "AR5",
        description="IPCC GWP version (AR4, AR5, AR6)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for audit trail",
    )


class WasteTypeSpecificRequest(BaseModel):
    """
    Request model for waste-type-specific (Method A) calculation.

    Uses bill of materials (BOM) to determine material-specific treatment
    and emission factors per kg of material.

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        product_category: Product category identifier
        units_sold: Number of units sold
        weight_per_unit_kg: Weight per unit in kg
        materials: Bill of materials (BOM)
        treatment_scenario: Treatment pathway fractions
        region: Region for treatment mix defaults
        include_avoided_emissions: Include recycling/energy credits
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    product_category: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Product category",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        description="Units sold in reporting year",
    )
    weight_per_unit_kg: float = Field(
        ...,
        gt=0,
        description="Product weight per unit (kg)",
    )
    materials: List[MaterialComposition] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Bill of materials (BOM)",
    )
    treatment_scenario: TreatmentScenario = Field(
        ...,
        description="Treatment pathway fractions",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Region for EF lookup",
    )
    include_avoided_emissions: bool = Field(
        True,
        description="Include recycling/energy recovery credits",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class LandfillRequest(BaseModel):
    """
    Request model for landfill-specific calculation using IPCC FOD model.

    Calculates methane generation using first-order decay kinetics with
    climate-zone-specific decay rates, gas collection, and oxidation.

    CH4_emitted = (CH4_generated - CH4_recovered) * (1 - OX)

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        material_type: Material type for FOD parameter lookup
        mass_tonnes: Mass of material sent to landfill (tonnes)
        climate_zone: IPCC climate zone (TROPICAL_WET, TROPICAL_DRY, TEMPERATE, BOREAL)
        landfill_type: Landfill type (MANAGED_ANAEROBIC, MANAGED_SEMI, UNMANAGED_DEEP, UNMANAGED_SHALLOW)
        gas_collection: Whether landfill has gas collection system
        collection_efficiency: Gas collection efficiency (0-1)
        flare_efficiency: Flare destruction efficiency (0-1)
        projection_years: Number of years to project FOD decay
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    material_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Material type for FOD parameter lookup",
    )
    mass_tonnes: float = Field(
        ...,
        gt=0,
        description="Mass sent to landfill (tonnes)",
    )
    climate_zone: str = Field(
        "TEMPERATE",
        description="IPCC climate zone (TROPICAL_WET, TROPICAL_DRY, TEMPERATE, BOREAL)",
    )
    landfill_type: str = Field(
        "MANAGED_ANAEROBIC",
        description="Landfill type (MANAGED_ANAEROBIC, MANAGED_SEMI, UNMANAGED_DEEP, UNMANAGED_SHALLOW)",
    )
    gas_collection: bool = Field(
        True,
        description="Whether landfill has gas collection system",
    )
    collection_efficiency: float = Field(
        0.75,
        ge=0,
        le=1,
        description="Gas collection efficiency fraction (0.0 - 1.0)",
    )
    flare_efficiency: float = Field(
        0.99,
        ge=0,
        le=1,
        description="Flare destruction efficiency fraction (0.0 - 1.0)",
    )
    projection_years: int = Field(
        100,
        ge=1,
        le=500,
        description="Years to project FOD decay (default 100 per IPCC)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class IncinerationRequest(BaseModel):
    """
    Request model for incineration/waste-to-energy calculation.

    Calculates fossil CO2 from combustion using material carbon content
    and fossil carbon fraction. Biogenic CO2 is reported separately.
    Energy recovery credits are optional.

    CO2_fossil = mass * dry_matter * carbon * fossil_fraction * oxidation * (44/12)

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        material_type: Material type for combustion parameter lookup
        mass_tonnes: Mass incinerated (tonnes)
        energy_recovery: Whether facility has energy recovery (WtE)
        wte_efficiency: Waste-to-energy conversion efficiency
        include_biogenic: Whether to report biogenic CO2 separately
        region: Region for displaced grid EF (energy recovery credits)
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    material_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Material type for combustion parameters",
    )
    mass_tonnes: float = Field(
        ...,
        gt=0,
        description="Mass incinerated (tonnes)",
    )
    energy_recovery: bool = Field(
        False,
        description="Whether facility has energy recovery (WtE)",
    )
    wte_efficiency: float = Field(
        0.25,
        ge=0,
        le=1,
        description="Waste-to-energy conversion efficiency (0.0 - 1.0)",
    )
    include_biogenic: bool = Field(
        True,
        description="Report biogenic CO2 as a memo item",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Region for displaced grid EF when energy recovery applies",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class RecyclingRequest(BaseModel):
    """
    Request model for recycling calculation using cut-off approach.

    Accounts for transport to MRF, MRF processing emissions, and
    optionally calculates avoided emissions from virgin material displacement.

    Net = transport_ef + mrf_ef - avoided_ef (if include_avoided)

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        material_type: Material type for recycling factor lookup
        mass_tonnes: Mass sent to recycling (tonnes)
        include_transport: Include collection/transport emissions
        transport_distance_km: Average transport distance to MRF (km)
        include_mrf_processing: Include MRF sorting/processing
        include_avoided_emissions: Include avoided virgin material credits
        recycling_rate: Effective recycling rate after MRF losses
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    material_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Material type for recycling factors",
    )
    mass_tonnes: float = Field(
        ...,
        gt=0,
        description="Mass sent to recycling (tonnes)",
    )
    include_transport: bool = Field(
        True,
        description="Include transport emissions to MRF",
    )
    transport_distance_km: float = Field(
        50.0,
        ge=0,
        description="Average transport distance to MRF (km)",
    )
    include_mrf_processing: bool = Field(
        True,
        description="Include MRF sorting/processing emissions",
    )
    include_avoided_emissions: bool = Field(
        True,
        description="Include avoided emissions from virgin material displacement",
    )
    recycling_rate: float = Field(
        0.85,
        ge=0,
        le=1,
        description="Effective recycling rate after MRF losses (0.0 - 1.0)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class AverageDataRequest(BaseModel):
    """
    Request model for average-data (Method B) calculation.

    Uses product-category-level composite emission factors that combine
    default BOM and regional treatment mixes.

    Emissions = units_sold * weight_per_unit * composite_ef

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        product_category: Product category for composite EF lookup
        units_sold: Number of units sold
        weight_per_unit_kg: Optional weight override (uses default if omitted)
        region: Region for treatment mix
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    product_category: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Product category for composite EF lookup",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        description="Number of units sold in reporting year",
    )
    weight_per_unit_kg: Optional[float] = Field(
        None,
        gt=0,
        description="Weight per unit (kg); uses product category default if omitted",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Region for treatment mix lookup",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class ProducerSpecificRequest(BaseModel):
    """
    Request model for producer-specific (Method D) calculation.

    Uses Environmental Product Declarations (EPDs) or Product Carbon
    Footprints (PCFs) that include end-of-life stage data.

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        product_category: Product category
        units_sold: Number of units sold
        epd_eol_co2e_per_unit: EoL-stage CO2e from EPD (kg CO2e/unit)
        epd_source: EPD or PCF source identifier
        epd_verification_status: Verification status (VERIFIED, SELF_DECLARED, THIRD_PARTY)
        epd_expiry_date: EPD expiry date for validity checks
        include_module_c: Include EPD Module C (disposal)
        include_module_d: Include EPD Module D (reuse/recycling credits)
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    product_category: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Product category",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        description="Number of units sold",
    )
    epd_eol_co2e_per_unit: float = Field(
        ...,
        description="EoL-stage CO2e from EPD/PCF (kg CO2e per unit)",
    )
    epd_source: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="EPD/PCF source identifier (e.g. EPD-ACME-2024-001)",
    )
    epd_verification_status: str = Field(
        "SELF_DECLARED",
        description="Verification status (VERIFIED, SELF_DECLARED, THIRD_PARTY)",
    )
    epd_expiry_date: Optional[str] = Field(
        None,
        description="EPD expiry date (ISO 8601) for validity check",
    )
    include_module_c: bool = Field(
        True,
        description="Include EPD Module C (disposal stage)",
    )
    include_module_d: bool = Field(
        False,
        description="Include EPD Module D (credits beyond system boundary)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class HybridRequest(BaseModel):
    """
    Request model for hybrid (Method E) multi-method calculation.

    Combines results from multiple methods using a waterfall hierarchy:
    producer-specific > waste-type-specific > average-data.
    Higher-quality methods take priority with gap-filling from lower tiers.

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        products: List of products with method assignments
        method_waterfall: Priority order of methods
        region: Default region
        include_avoided_emissions: Include recycling/energy credits
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    products: List[ProductEntry] = Field(
        ...,
        min_items=1,
        max_items=5000,
        description="Products with optional method assignments",
    )
    method_waterfall: List[str] = Field(
        ["producer_specific", "waste_type_specific", "average_data"],
        description="Priority order of calculation methods",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Default region for treatment mixes",
    )
    include_avoided_emissions: bool = Field(
        True,
        description="Include recycling/energy recovery credits",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch end-of-life treatment calculations.

    Processes up to 10,000 products in a single request with parallel
    execution and per-product error isolation.

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        products: List of product entries
        method: Default calculation method
        region: Default region
        include_avoided_emissions: Include recycling/energy credits
        batch_id: Optional batch identifier for idempotency
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    products: List[ProductEntry] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="Products for batch calculation",
    )
    method: str = Field(
        "average_data",
        description="Default method (waste_type_specific, average_data, producer_specific, hybrid)",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Default region",
    )
    include_avoided_emissions: bool = Field(
        True,
        description="Include recycling/energy credits",
    )
    batch_id: Optional[str] = Field(
        None,
        description="Optional batch UUID for idempotency",
    )


class PortfolioAnalysisRequest(BaseModel):
    """
    Request model for portfolio-level analysis with circularity scoring.

    Analyzes the full product portfolio for end-of-life emissions,
    circularity metrics, and hot-spot identification.

    Attributes:
        org_id: Organization identifier
        year: Reporting year
        products: Complete product portfolio
        region: Default region
        include_circularity: Calculate circularity metrics
        include_hotspot: Identify top emitting product categories
        top_n_hotspots: Number of hot-spots to return
        metadata: Additional metadata
    """

    org_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Organization identifier",
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2100,
        description="Reporting year",
    )
    products: List[ProductEntry] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="Complete product portfolio",
    )
    region: str = Field(
        "GLOBAL",
        max_length=50,
        description="Default region for treatment mixes",
    )
    include_circularity: bool = Field(
        True,
        description="Calculate circularity metrics (recycling rate, material recovery)",
    )
    include_hotspot: bool = Field(
        True,
        description="Identify top emitting product categories",
    )
    top_n_hotspots: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of top hot-spot categories to return",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Validates calculation results against 7 regulatory frameworks for
    completeness, boundary correctness, method hierarchy, and disclosure.

    Attributes:
        calculation_id: Calculation ID to check
        frameworks: List of framework identifiers
        include_recommendations: Include improvement recommendations
    """

    calculation_id: str = Field(
        ...,
        description="Calculation UUID to check",
    )
    frameworks: List[str] = Field(
        [
            "ghg_protocol",
            "iso_14064",
            "csrd_esrs",
            "cdp",
            "sbti",
            "sb_253",
            "gri",
        ],
        min_items=1,
        description=(
            "Frameworks to check: ghg_protocol, iso_14064, csrd_esrs, "
            "cdp, sbti, sb_253, gri"
        ),
    )
    include_recommendations: bool = Field(
        True,
        description="Include improvement recommendations per finding",
    )


class UncertaintyAnalysisRequest(BaseModel):
    """
    Request model for uncertainty analysis.

    Supports Monte Carlo simulation, analytical error propagation,
    and IPCC Tier 2 default uncertainty ranges.

    Attributes:
        calculation_id: Calculation ID to analyze
        method: Uncertainty analysis method
        iterations: Monte Carlo iterations (ignored for analytical)
        confidence_level: Confidence interval level (0.80 - 0.99)
        parameter_distributions: Optional custom parameter distributions
    """

    calculation_id: str = Field(
        ...,
        description="Calculation UUID to analyze",
    )
    method: str = Field(
        "monte_carlo",
        description="Uncertainty method (monte_carlo, analytical, ipcc_tier_2)",
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
        description="Confidence interval level",
    )
    parameter_distributions: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom parameter distributions for Monte Carlo",
    )


# ============================================================================
# RESPONSE MODELS (14)
# ============================================================================


class CalculateResponse(BaseModel):
    """Response model for full pipeline calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    org_id: str = Field(..., description="Organization identifier")
    year: int = Field(..., description="Reporting year")
    method: str = Field(..., description="Primary calculation method used")
    total_products: int = Field(
        ..., description="Total products in calculation"
    )
    total_mass_tonnes: float = Field(
        ..., description="Total product mass (tonnes)"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Gross emissions before avoided credits (tCO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Avoided emissions from recycling/energy (tCO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Net emissions = gross - avoided (tCO2e)"
    )
    by_treatment: Dict[str, float] = Field(
        ..., description="Emissions breakdown by treatment method (tCO2e)"
    )
    by_material: Dict[str, float] = Field(
        ..., description="Emissions breakdown by material type (tCO2e)"
    )
    dqi_score: Optional[float] = Field(
        None, description="Data quality indicator score (1-5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class WasteTypeResponse(BaseModel):
    """Response model for waste-type-specific calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    org_id: str = Field(..., description="Organization identifier")
    method: str = Field(
        "waste_type_specific", description="Calculation method"
    )
    product_category: str = Field(..., description="Product category")
    total_mass_tonnes: float = Field(
        ..., description="Total product mass (tonnes)"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Gross emissions (tCO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Avoided emissions (tCO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Net emissions (tCO2e)"
    )
    material_results: List[Dict[str, Any]] = Field(
        ..., description="Per-material breakdown"
    )
    treatment_results: List[Dict[str, Any]] = Field(
        ..., description="Per-treatment breakdown"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class LandfillResponse(BaseModel):
    """Response model for landfill FOD calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    material_type: str = Field(..., description="Material type")
    mass_tonnes: float = Field(
        ..., description="Mass sent to landfill (tonnes)"
    )
    ch4_generated_tonnes: float = Field(
        ..., description="Total CH4 generated (tonnes)"
    )
    ch4_recovered_tonnes: float = Field(
        ..., description="CH4 recovered by gas collection (tonnes)"
    )
    ch4_oxidized_tonnes: float = Field(
        ..., description="CH4 oxidized in cover soil (tonnes)"
    )
    ch4_emitted_tonnes: float = Field(
        ..., description="Net CH4 emitted to atmosphere (tonnes)"
    )
    co2e_tonnes: float = Field(
        ..., description="Total CO2e from landfill (tonnes)"
    )
    fod_parameters: Dict[str, float] = Field(
        ..., description="FOD parameters used (DOC, DOCf, MCF, k, OX)"
    )
    projection_years: int = Field(
        ..., description="Years of FOD projection"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class IncinerationResponse(BaseModel):
    """Response model for incineration/WtE calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    material_type: str = Field(..., description="Material type")
    mass_tonnes: float = Field(
        ..., description="Mass incinerated (tonnes)"
    )
    co2_fossil_tonnes: float = Field(
        ..., description="Fossil CO2 emissions (tonnes)"
    )
    co2_biogenic_tonnes: float = Field(
        ..., description="Biogenic CO2 (memo item, tonnes)"
    )
    ch4_tonnes: float = Field(
        ..., description="CH4 emissions (tonnes)"
    )
    n2o_tonnes: float = Field(
        ..., description="N2O emissions (tonnes)"
    )
    co2e_gross_tonnes: float = Field(
        ..., description="Gross CO2e before energy credits (tonnes)"
    )
    energy_recovered_mwh: Optional[float] = Field(
        None, description="Energy recovered (MWh)"
    )
    avoided_co2e_tonnes: Optional[float] = Field(
        None, description="Avoided emissions from energy recovery (tonnes)"
    )
    co2e_net_tonnes: float = Field(
        ..., description="Net CO2e after energy credits (tonnes)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class RecyclingResponse(BaseModel):
    """Response model for recycling cut-off calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    material_type: str = Field(..., description="Material type")
    mass_tonnes: float = Field(
        ..., description="Mass sent to recycling (tonnes)"
    )
    transport_co2e_tonnes: float = Field(
        ..., description="Transport to MRF emissions (tonnes CO2e)"
    )
    mrf_processing_co2e_tonnes: float = Field(
        ..., description="MRF sorting/processing emissions (tonnes CO2e)"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Gross recycling process emissions (tonnes CO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Avoided virgin material emissions (tonnes CO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Net emissions = gross - avoided (tonnes CO2e)"
    )
    effective_recycling_rate: float = Field(
        ..., description="Effective recycling rate after MRF losses"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class AverageDataResponse(BaseModel):
    """Response model for average-data calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    product_category: str = Field(..., description="Product category")
    units_sold: int = Field(..., description="Units sold")
    weight_per_unit_kg: float = Field(
        ..., description="Weight per unit (kg)"
    )
    total_mass_tonnes: float = Field(
        ..., description="Total mass (tonnes)"
    )
    composite_ef_kgco2e_per_kg: float = Field(
        ..., description="Composite emission factor (kgCO2e/kg)"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Gross emissions (tCO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Avoided emissions (tCO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Net emissions (tCO2e)"
    )
    treatment_mix_used: Dict[str, float] = Field(
        ..., description="Treatment mix fractions applied"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class ProducerSpecificResponse(BaseModel):
    """Response model for producer-specific (EPD) calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    product_category: str = Field(..., description="Product category")
    units_sold: int = Field(..., description="Units sold")
    epd_source: str = Field(..., description="EPD/PCF source identifier")
    epd_verification_status: str = Field(
        ..., description="Verification status"
    )
    module_c_co2e_tonnes: float = Field(
        ..., description="Module C (disposal) emissions (tCO2e)"
    )
    module_d_co2e_tonnes: Optional[float] = Field(
        None, description="Module D (credits) emissions (tCO2e)"
    )
    total_co2e_tonnes: float = Field(
        ..., description="Total EoL emissions from EPD (tCO2e)"
    )
    dqi_score: float = Field(
        ..., description="Data quality score (1-5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculation."""

    batch_id: str = Field(..., description="Unique batch UUID")
    org_id: str = Field(..., description="Organization identifier")
    total_products: int = Field(
        ..., description="Total products processed"
    )
    successful: int = Field(
        ..., description="Successful calculations"
    )
    failed: int = Field(
        ..., description="Failed calculations"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Total gross emissions (tCO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Total avoided emissions (tCO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Total net emissions (tCO2e)"
    )
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual product results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-product error details"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time (ms)"
    )


class PortfolioAnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""

    analysis_id: str = Field(..., description="Unique analysis UUID")
    org_id: str = Field(..., description="Organization identifier")
    year: int = Field(..., description="Reporting year")
    total_products: int = Field(
        ..., description="Total products in portfolio"
    )
    total_mass_tonnes: float = Field(
        ..., description="Total product mass (tonnes)"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Gross emissions (tCO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Avoided emissions (tCO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Net emissions (tCO2e)"
    )
    by_treatment: Dict[str, float] = Field(
        ..., description="Breakdown by treatment (tCO2e)"
    )
    by_category: Dict[str, float] = Field(
        ..., description="Breakdown by product category (tCO2e)"
    )
    circularity_score: Optional[float] = Field(
        None, description="Portfolio circularity score (0-100)"
    )
    recycling_rate: Optional[float] = Field(
        None, description="Portfolio-level recycling rate (0.0 - 1.0)"
    )
    material_recovery_rate: Optional[float] = Field(
        None, description="Total material recovery rate (0.0 - 1.0)"
    )
    hotspots: Optional[List[Dict[str, Any]]] = Field(
        None, description="Top emitting product categories"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check."""

    check_id: str = Field(..., description="Unique check UUID")
    calculation_id: str = Field(..., description="Calculation UUID checked")
    frameworks_checked: List[str] = Field(
        ..., description="Frameworks validated"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (PASS, FAIL, WARNING)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0 - 1.0)"
    )
    results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    checked_at: str = Field(
        ..., description="ISO 8601 check timestamp"
    )


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    org_id: str = Field(..., description="Organization identifier")
    year: int = Field(..., description="Reporting year")
    method: str = Field(..., description="Calculation method")
    status: str = Field(
        ..., description="Calculation status (COMPLETED, FAILED, PENDING)"
    )
    gross_co2e_tonnes: float = Field(
        ..., description="Gross emissions (tCO2e)"
    )
    avoided_co2e_tonnes: float = Field(
        ..., description="Avoided emissions (tCO2e)"
    )
    net_co2e_tonnes: float = Field(
        ..., description="Net emissions (tCO2e)"
    )
    details: Dict[str, Any] = Field(
        ..., description="Full calculation detail payload"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance chain hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 calculation timestamp"
    )


class CalculationListResponse(BaseModel):
    """Response model for paginated calculation listing."""

    calculations: List[Dict[str, Any]] = Field(
        ..., description="Calculation summaries"
    )
    total: int = Field(..., description="Total matching calculations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class DeleteResponse(BaseModel):
    """Response model for soft deletion."""

    calculation_id: str = Field(
        ..., description="Deleted calculation UUID"
    )
    deleted: bool = Field(
        ..., description="Whether deletion succeeded"
    )
    message: str = Field(
        ..., description="Human-readable status message"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )
    database_connected: bool = Field(
        ..., description="Database connection status"
    )
    cache_connected: bool = Field(
        ..., description="Cache connection status"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINTS - CALCULATIONS (10)
# ============================================================================


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate end-of-life treatment emissions (full pipeline)",
    description=(
        "Calculate GHG emissions from end-of-life treatment of sold products "
        "through the full 10-stage pipeline. Accepts a list of products with "
        "optional BOM and treatment scenarios. Returns gross, avoided, and net "
        "emissions with breakdowns by treatment method and material type. "
        "Deterministic results with SHA-256 provenance hash for audit trail."
    ),
)
async def calculate_emissions(
    request: FullPipelineRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate end-of-life treatment emissions through the full pipeline.

    Args:
        request: Full pipeline request with products and parameters
        service: EndOfLifeTreatmentService instance

    Returns:
        CalculateResponse with gross/avoided/net emissions

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating EoL emissions for org={request.org_id}, "
            f"year={request.year}, products={len(request.products)}"
        )

        result = await service.calculate(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            org_id=result.get("org_id", request.org_id),
            year=result.get("year", request.year),
            method=result.get("method", "full_pipeline"),
            total_products=result.get("total_products", len(request.products)),
            total_mass_tonnes=result.get("total_mass_tonnes", 0.0),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            by_treatment=result.get("by_treatment", {}),
            by_material=result.get("by_material", {}),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

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
    "/calculate/waste-type-specific",
    response_model=WasteTypeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate using waste-type-specific method (Method A)",
    description=(
        "Calculate end-of-life emissions using waste-type-specific method "
        "(Method A). Requires bill of materials (BOM) and treatment scenario. "
        "Applies material-specific emission factors per kg for each treatment "
        "pathway. Highest data quality method when BOM is available."
    ),
)
async def calculate_waste_type_specific(
    request: WasteTypeSpecificRequest,
    service=Depends(get_service),
) -> WasteTypeResponse:
    """
    Calculate emissions using waste-type-specific method.

    Args:
        request: Waste-type-specific request with BOM and treatment scenario
        service: EndOfLifeTreatmentService instance

    Returns:
        WasteTypeResponse with per-material and per-treatment breakdowns

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating waste-type-specific EoL for org={request.org_id}, "
            f"category={request.product_category}, "
            f"materials={len(request.materials)}"
        )

        result = await service.calculate_waste_type_specific(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return WasteTypeResponse(
            calculation_id=calculation_id,
            org_id=result.get("org_id", request.org_id),
            method="waste_type_specific",
            product_category=result.get(
                "product_category", request.product_category
            ),
            total_mass_tonnes=result.get("total_mass_tonnes", 0.0),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            material_results=result.get("material_results", []),
            treatment_results=result.get("treatment_results", []),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_waste_type_specific: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_waste_type_specific: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Waste-type-specific calculation failed",
        )


@router.post(
    "/calculate/waste-type-specific/landfill",
    response_model=LandfillResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate landfill emissions using IPCC FOD model",
    description=(
        "Calculate landfill methane emissions using the IPCC First Order "
        "Decay (FOD) model. Accounts for climate-zone-specific decay rates, "
        "gas collection systems, flare efficiency, and methane oxidation in "
        "cover soil. Returns CH4 generated, recovered, oxidized, and emitted."
    ),
)
async def calculate_landfill(
    request: LandfillRequest,
    service=Depends(get_service),
) -> LandfillResponse:
    """
    Calculate landfill-specific emissions using IPCC FOD model.

    Args:
        request: Landfill request with material, mass, and FOD parameters
        service: EndOfLifeTreatmentService instance

    Returns:
        LandfillResponse with CH4 mass balance and CO2e

    Raises:
        HTTPException: 400 for invalid parameters, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating landfill FOD for org={request.org_id}, "
            f"material={request.material_type}, "
            f"mass={request.mass_tonnes}t, "
            f"climate={request.climate_zone}"
        )

        result = await service.calculate_landfill(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return LandfillResponse(
            calculation_id=calculation_id,
            material_type=result.get(
                "material_type", request.material_type
            ),
            mass_tonnes=result.get("mass_tonnes", request.mass_tonnes),
            ch4_generated_tonnes=result.get("ch4_generated_tonnes", 0.0),
            ch4_recovered_tonnes=result.get("ch4_recovered_tonnes", 0.0),
            ch4_oxidized_tonnes=result.get("ch4_oxidized_tonnes", 0.0),
            ch4_emitted_tonnes=result.get("ch4_emitted_tonnes", 0.0),
            co2e_tonnes=result.get("co2e_tonnes", 0.0),
            fod_parameters=result.get("fod_parameters", {}),
            projection_years=result.get(
                "projection_years", request.projection_years
            ),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_landfill: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_landfill: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Landfill calculation failed",
        )


@router.post(
    "/calculate/waste-type-specific/incineration",
    response_model=IncinerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate incineration/WtE emissions",
    description=(
        "Calculate incineration emissions from combustion of waste materials. "
        "Separates fossil and biogenic CO2. Optional energy recovery credits "
        "from waste-to-energy (WtE) facilities based on displaced grid "
        "electricity. Returns gross, energy credit, and net emissions."
    ),
)
async def calculate_incineration(
    request: IncinerationRequest,
    service=Depends(get_service),
) -> IncinerationResponse:
    """
    Calculate incineration/WtE emissions.

    Args:
        request: Incineration request with material and combustion parameters
        service: EndOfLifeTreatmentService instance

    Returns:
        IncinerationResponse with fossil/biogenic CO2 and energy credits

    Raises:
        HTTPException: 400 for invalid parameters, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating incineration for org={request.org_id}, "
            f"material={request.material_type}, "
            f"mass={request.mass_tonnes}t, "
            f"energy_recovery={request.energy_recovery}"
        )

        result = await service.calculate_incineration(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return IncinerationResponse(
            calculation_id=calculation_id,
            material_type=result.get(
                "material_type", request.material_type
            ),
            mass_tonnes=result.get("mass_tonnes", request.mass_tonnes),
            co2_fossil_tonnes=result.get("co2_fossil_tonnes", 0.0),
            co2_biogenic_tonnes=result.get("co2_biogenic_tonnes", 0.0),
            ch4_tonnes=result.get("ch4_tonnes", 0.0),
            n2o_tonnes=result.get("n2o_tonnes", 0.0),
            co2e_gross_tonnes=result.get("co2e_gross_tonnes", 0.0),
            energy_recovered_mwh=result.get("energy_recovered_mwh"),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes"),
            co2e_net_tonnes=result.get("co2e_net_tonnes", 0.0),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_incineration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_incineration: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Incineration calculation failed",
        )


@router.post(
    "/calculate/waste-type-specific/recycling",
    response_model=RecyclingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate recycling emissions (cut-off approach)",
    description=(
        "Calculate recycling process emissions using the cut-off approach. "
        "Includes transport to MRF, MRF sorting/processing, and optional "
        "avoided emissions from virgin material displacement. Returns gross, "
        "avoided, and net emissions with effective recycling rate."
    ),
)
async def calculate_recycling(
    request: RecyclingRequest,
    service=Depends(get_service),
) -> RecyclingResponse:
    """
    Calculate recycling emissions with cut-off approach.

    Args:
        request: Recycling request with material and processing parameters
        service: EndOfLifeTreatmentService instance

    Returns:
        RecyclingResponse with process, avoided, and net emissions

    Raises:
        HTTPException: 400 for invalid parameters, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating recycling for org={request.org_id}, "
            f"material={request.material_type}, "
            f"mass={request.mass_tonnes}t, "
            f"avoided={request.include_avoided_emissions}"
        )

        result = await service.calculate_recycling(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return RecyclingResponse(
            calculation_id=calculation_id,
            material_type=result.get(
                "material_type", request.material_type
            ),
            mass_tonnes=result.get("mass_tonnes", request.mass_tonnes),
            transport_co2e_tonnes=result.get("transport_co2e_tonnes", 0.0),
            mrf_processing_co2e_tonnes=result.get(
                "mrf_processing_co2e_tonnes", 0.0
            ),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            effective_recycling_rate=result.get(
                "effective_recycling_rate", request.recycling_rate
            ),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_recycling: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_recycling: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recycling calculation failed",
        )


@router.post(
    "/calculate/average-data",
    response_model=AverageDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate using average-data method (Method B)",
    description=(
        "Calculate end-of-life emissions using average-data method (Method B). "
        "Uses product-category-level composite emission factors that combine "
        "default BOM and regional treatment mixes. Suitable when detailed "
        "BOM is not available."
    ),
)
async def calculate_average_data(
    request: AverageDataRequest,
    service=Depends(get_service),
) -> AverageDataResponse:
    """
    Calculate emissions using average-data method.

    Args:
        request: Average-data request with product category and units
        service: EndOfLifeTreatmentService instance

    Returns:
        AverageDataResponse with composite EF and treatment mix applied

    Raises:
        HTTPException: 400 for unknown category, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating average-data EoL for org={request.org_id}, "
            f"category={request.product_category}, "
            f"units={request.units_sold}"
        )

        result = await service.calculate_average_data(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return AverageDataResponse(
            calculation_id=calculation_id,
            product_category=result.get(
                "product_category", request.product_category
            ),
            units_sold=result.get("units_sold", request.units_sold),
            weight_per_unit_kg=result.get(
                "weight_per_unit_kg",
                request.weight_per_unit_kg or 0.0,
            ),
            total_mass_tonnes=result.get("total_mass_tonnes", 0.0),
            composite_ef_kgco2e_per_kg=result.get(
                "composite_ef_kgco2e_per_kg", 0.0
            ),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            treatment_mix_used=result.get("treatment_mix_used", {}),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_average_data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_average_data: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Average-data calculation failed",
        )


@router.post(
    "/calculate/producer-specific",
    response_model=ProducerSpecificResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate using producer-specific method (Method D)",
    description=(
        "Calculate end-of-life emissions using producer-specific method "
        "(Method D). Uses Environmental Product Declarations (EPDs) or "
        "Product Carbon Footprints (PCFs) with Module C (disposal) and "
        "optional Module D (credits). Highest data quality when EPDs exist."
    ),
)
async def calculate_producer_specific(
    request: ProducerSpecificRequest,
    service=Depends(get_service),
) -> ProducerSpecificResponse:
    """
    Calculate emissions using producer-specific EPD/PCF data.

    Args:
        request: Producer-specific request with EPD data
        service: EndOfLifeTreatmentService instance

    Returns:
        ProducerSpecificResponse with Module C/D emissions

    Raises:
        HTTPException: 400 for invalid EPD data, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating producer-specific EoL for org={request.org_id}, "
            f"epd={request.epd_source}, units={request.units_sold}"
        )

        result = await service.calculate_producer_specific(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return ProducerSpecificResponse(
            calculation_id=calculation_id,
            product_category=result.get(
                "product_category", request.product_category
            ),
            units_sold=result.get("units_sold", request.units_sold),
            epd_source=result.get("epd_source", request.epd_source),
            epd_verification_status=result.get(
                "epd_verification_status",
                request.epd_verification_status,
            ),
            module_c_co2e_tonnes=result.get("module_c_co2e_tonnes", 0.0),
            module_d_co2e_tonnes=result.get("module_d_co2e_tonnes"),
            total_co2e_tonnes=result.get("total_co2e_tonnes", 0.0),
            dqi_score=result.get("dqi_score", 3.0),
            provenance_hash=result.get("provenance_hash", ""),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_producer_specific: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_producer_specific: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Producer-specific calculation failed",
        )


@router.post(
    "/calculate/hybrid",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate using hybrid method (Method E)",
    description=(
        "Calculate end-of-life emissions using hybrid method (Method E). "
        "Combines multiple methods with a waterfall hierarchy: "
        "producer-specific > waste-type-specific > average-data. "
        "Higher-quality methods take priority with gap-filling from lower tiers."
    ),
)
async def calculate_hybrid(
    request: HybridRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate emissions using hybrid multi-method approach.

    Args:
        request: Hybrid request with products and method waterfall
        service: EndOfLifeTreatmentService instance

    Returns:
        CalculateResponse with blended multi-method results

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating hybrid EoL for org={request.org_id}, "
            f"products={len(request.products)}, "
            f"waterfall={request.method_waterfall}"
        )

        result = await service.calculate_hybrid(request.dict())
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            org_id=result.get("org_id", request.org_id),
            year=result.get("year", request.year),
            method=result.get("method", "hybrid"),
            total_products=result.get(
                "total_products", len(request.products)
            ),
            total_mass_tonnes=result.get("total_mass_tonnes", 0.0),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            by_treatment=result.get("by_treatment", {}),
            by_material=result.get("by_material", {}),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(f"Validation error in calculate_hybrid: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in calculate_hybrid: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid calculation failed",
        )


@router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate end-of-life treatment emissions",
    description=(
        "Calculate end-of-life treatment emissions for up to 10,000 products "
        "in a single request. Processes with parallel execution and per-product "
        "error isolation. Returns aggregated totals with individual results."
    ),
)
async def calculate_batch(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate batch end-of-life treatment emissions.

    Args:
        request: Batch request with product list
        service: EndOfLifeTreatmentService instance

    Returns:
        BatchCalculateResponse with aggregated and per-product results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch EoL for org={request.org_id}, "
            f"products={len(request.products)}, method={request.method}"
        )

        result = await service.calculate_batch(request.dict())
        batch_id = result.get(
            "batch_id", request.batch_id or str(uuid.uuid4())
        )

        return BatchCalculateResponse(
            batch_id=batch_id,
            org_id=result.get("org_id", request.org_id),
            total_products=result.get(
                "total_products", len(request.products)
            ),
            successful=result.get("successful", 0),
            failed=result.get("failed", 0),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            results=result.get("results", []),
            errors=result.get("errors", []),
            processing_time_ms=result.get("processing_time_ms", 0.0),
        )

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
    response_model=PortfolioAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Portfolio-level analysis with circularity scoring",
    description=(
        "Analyze the full product portfolio for end-of-life emissions, "
        "circularity metrics (recycling rate, material recovery), and "
        "hot-spot identification. Returns portfolio-level aggregations "
        "with per-category breakdowns and reduction opportunities."
    ),
)
async def calculate_portfolio(
    request: PortfolioAnalysisRequest,
    service=Depends(get_service),
) -> PortfolioAnalysisResponse:
    """
    Portfolio-level end-of-life analysis with circularity scoring.

    Args:
        request: Portfolio analysis request with product list
        service: EndOfLifeTreatmentService instance

    Returns:
        PortfolioAnalysisResponse with circularity metrics and hot-spots

    Raises:
        HTTPException: 400 for validation errors, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing portfolio EoL for org={request.org_id}, "
            f"products={len(request.products)}, "
            f"circularity={request.include_circularity}"
        )

        result = await service.analyze_portfolio(request.dict())
        analysis_id = result.get("analysis_id", str(uuid.uuid4()))

        return PortfolioAnalysisResponse(
            analysis_id=analysis_id,
            org_id=result.get("org_id", request.org_id),
            year=result.get("year", request.year),
            total_products=result.get(
                "total_products", len(request.products)
            ),
            total_mass_tonnes=result.get("total_mass_tonnes", 0.0),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            by_treatment=result.get("by_treatment", {}),
            by_category=result.get("by_category", {}),
            circularity_score=result.get("circularity_score"),
            recycling_rate=result.get("recycling_rate"),
            material_recovery_rate=result.get("material_recovery_rate"),
            hotspots=result.get("hotspots"),
            provenance_hash=result.get("provenance_hash", ""),
        )

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
            detail="Portfolio analysis failed",
        )


# ============================================================================
# ENDPOINTS - COMPLIANCE & UNCERTAINTY (2)
# ============================================================================


@router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check multi-framework compliance",
    description=(
        "Check end-of-life treatment calculation results against one or more "
        "regulatory frameworks. Validates method hierarchy, boundary "
        "correctness, avoided emissions treatment, biogenic CO2 disclosure, "
        "and data quality requirements. Supports GHG Protocol, ISO 14064, "
        "CSRD ESRS E1/E5, CDP, SBTi, SB 253, and GRI 305/306."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request with frameworks
        service: EndOfLifeTreatmentService instance

    Returns:
        ComplianceCheckResponse with per-framework findings

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures
    """
    try:
        logger.info(
            f"Checking compliance for calc={request.calculation_id}, "
            f"frameworks={len(request.frameworks)}"
        )

        result = await service.check_compliance(request.dict())
        check_id = result.get("check_id", str(uuid.uuid4()))

        return ComplianceCheckResponse(
            check_id=check_id,
            calculation_id=result.get(
                "calculation_id", request.calculation_id
            ),
            frameworks_checked=result.get(
                "frameworks_checked", request.frameworks
            ),
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
            results=result.get("results", []),
            recommendations=result.get("recommendations", []),
            checked_at=result.get(
                "checked_at", datetime.utcnow().isoformat()
            ),
        )

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


@router.post(
    "/uncertainty/analyze",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Analyze calculation uncertainty",
    description=(
        "Perform uncertainty analysis on end-of-life treatment emissions "
        "calculations. Supports Monte Carlo simulation, analytical error "
        "propagation, and IPCC Tier 2 default ranges. Returns mean, "
        "standard deviation, confidence intervals, and sensitivity indices."
    ),
)
async def analyze_uncertainty(
    request: UncertaintyAnalysisRequest,
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Perform uncertainty analysis on calculation results.

    Args:
        request: Uncertainty analysis request
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with uncertainty metrics and confidence intervals

    Raises:
        HTTPException: 400 for invalid method, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing uncertainty for calc={request.calculation_id}, "
            f"method={request.method}, "
            f"iterations={request.iterations}"
        )

        result = await service.analyze_uncertainty(request.dict())

        return {
            "analysis_id": result.get("analysis_id", str(uuid.uuid4())),
            "calculation_id": result.get(
                "calculation_id", request.calculation_id
            ),
            "method": result.get("method", request.method),
            "iterations": result.get("iterations", request.iterations),
            "confidence_level": result.get(
                "confidence_level", request.confidence_level
            ),
            "mean_co2e_tonnes": result.get("mean_co2e_tonnes", 0.0),
            "std_dev_co2e_tonnes": result.get("std_dev_co2e_tonnes", 0.0),
            "ci_lower_co2e_tonnes": result.get("ci_lower_co2e_tonnes", 0.0),
            "ci_upper_co2e_tonnes": result.get("ci_upper_co2e_tonnes", 0.0),
            "relative_uncertainty_pct": result.get(
                "relative_uncertainty_pct", 0.0
            ),
            "sensitivity_indices": result.get("sensitivity_indices", {}),
            "analyzed_at": result.get(
                "analyzed_at", datetime.utcnow().isoformat()
            ),
        }

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
# ENDPOINTS - CALCULATION CRUD (3)
# ============================================================================


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific end-of-life treatment "
        "calculation including full input/output payload, provenance hash, "
        "and calculation metadata."
    ),
)
async def get_calculation_detail(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> CalculationDetailResponse:
    """
    Get detailed information for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: EndOfLifeTreatmentService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(f"Getting calculation detail: {calculation_id}")

        result = await service.get_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return CalculationDetailResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            org_id=result.get("org_id", ""),
            year=result.get("year", 0),
            method=result.get("method", ""),
            status=result.get("status", "COMPLETED"),
            gross_co2e_tonnes=result.get("gross_co2e_tonnes", 0.0),
            avoided_co2e_tonnes=result.get("avoided_co2e_tonnes", 0.0),
            net_co2e_tonnes=result.get("net_co2e_tonnes", 0.0),
            details=result.get("details", {}),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_calculation_detail: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve calculation",
        )


@router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List calculations",
    description=(
        "Retrieve a paginated list of end-of-life treatment calculations. "
        "Supports filtering by org_id, method, product category, year, "
        "and date range."
    ),
)
async def list_calculations(
    org_id: Optional[str] = Query(
        None, description="Filter by organization"
    ),
    method: Optional[str] = Query(
        None, description="Filter by calculation method"
    ),
    product_category: Optional[str] = Query(
        None, description="Filter by product category"
    ),
    year: Optional[int] = Query(
        None, ge=1990, le=2100, description="Filter by reporting year"
    ),
    from_date: Optional[str] = Query(
        None, description="Filter from date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="Filter to date (ISO 8601)"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        50, ge=1, le=500, description="Results per page"
    ),
    service=Depends(get_service),
) -> CalculationListResponse:
    """
    List end-of-life treatment calculations with filtering.

    Args:
        org_id: Optional organization filter
        method: Optional method filter
        product_category: Optional product category filter
        year: Optional reporting year filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        page: Page number (1-indexed)
        page_size: Results per page
        service: EndOfLifeTreatmentService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing calculations: org={org_id}, page={page}, "
            f"size={page_size}"
        )

        filters = {
            "org_id": org_id,
            "method": method,
            "product_category": product_category,
            "year": year,
            "from_date": from_date,
            "to_date": to_date,
            "page": page,
            "page_size": page_size,
        }

        result = await service.list_calculations(filters)

        return CalculationListResponse(
            calculations=result.get("calculations", []),
            total=result.get("total", 0),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Error in list_calculations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations",
        )


@router.delete(
    "/calculations/{calculation_id}",
    response_model=DeleteResponse,
    summary="Delete calculation (soft delete)",
    description=(
        "Soft-delete a specific end-of-life treatment calculation. "
        "Marks the calculation as deleted with audit trail; data is "
        "retained for regulatory compliance."
    ),
)
async def delete_calculation(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> DeleteResponse:
    """
    Soft-delete a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: EndOfLifeTreatmentService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info(f"Deleting calculation: {calculation_id}")

        deleted = await service.delete_calculation(calculation_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        return DeleteResponse(
            calculation_id=calculation_id,
            deleted=True,
            message=f"Calculation {calculation_id} soft-deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_calculation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation",
        )


# ============================================================================
# ENDPOINTS - REFERENCE DATA & LOOKUPS (6)
# ============================================================================


@router.get(
    "/emission-factors/{material}",
    response_model=Dict[str, Any],
    summary="Get emission factors for a material",
    description=(
        "Retrieve emission factors for a specific material across all "
        "treatment methods. Returns kgCO2e/kg values with gas-by-gas "
        "breakdown (CO2 fossil, CO2 biogenic, CH4, N2O) and EF source."
    ),
)
async def get_emission_factors(
    material: str = Path(
        ..., description="Material type (e.g. HDPE, PET, STEEL, GLASS)"
    ),
    region: Optional[str] = Query(
        None, description="Region filter (US, EU, GLOBAL)"
    ),
    year: Optional[int] = Query(
        None, ge=1990, le=2100, description="Year filter"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get emission factors for a specific material type.

    Args:
        material: Material type identifier
        region: Optional region filter
        year: Optional year filter
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with material EFs by treatment method

    Raises:
        HTTPException: 404 if material not found, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting EFs for material={material}, "
            f"region={region}, year={year}"
        )

        result = await service.get_emission_factors(
            material, region=region, year=year
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No emission factors found for material '{material}'",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_emission_factors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


@router.get(
    "/product-compositions",
    response_model=Dict[str, Any],
    summary="Get default product compositions (BOM)",
    description=(
        "Retrieve default bill of materials (BOM) for product categories. "
        "Used as fallback when company-specific BOM is not available. "
        "Includes material fractions and default weights per unit."
    ),
)
async def get_product_compositions(
    product_category: Optional[str] = Query(
        None, description="Filter by product category"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get default product compositions (BOM).

    Args:
        product_category: Optional product category filter
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with product compositions

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting product compositions: category={product_category}"
        )

        result = await service.get_product_compositions(
            product_category=product_category
        )

        return {
            "compositions": result.get("compositions", []),
            "total": result.get("total", 0),
        }

    except Exception as e:
        logger.error(
            f"Error in get_product_compositions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve product compositions",
        )


@router.get(
    "/treatment-mixes",
    response_model=Dict[str, Any],
    summary="Get regional treatment mix profiles",
    description=(
        "Retrieve regional waste treatment mix profiles that define the "
        "fraction of waste going to each treatment pathway (landfill, "
        "incineration, recycling, composting) by region and year."
    ),
)
async def get_treatment_mixes(
    region: Optional[str] = Query(
        None, description="Filter by region (US, EU, CN, JP, GLOBAL)"
    ),
    year: Optional[int] = Query(
        None, ge=1990, le=2100, description="Filter by year"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get regional treatment mix profiles.

    Args:
        region: Optional region filter
        year: Optional year filter
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with treatment mix data

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting treatment mixes: region={region}, year={year}"
        )

        result = await service.get_treatment_mixes(
            region=region, year=year
        )

        return {
            "treatment_mixes": result.get("treatment_mixes", []),
            "total": result.get("total", 0),
        }

    except Exception as e:
        logger.error(f"Error in get_treatment_mixes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve treatment mixes",
        )


@router.get(
    "/avoided-emissions/{calculation_id}",
    response_model=Dict[str, Any],
    summary="Get avoided emissions detail",
    description=(
        "Retrieve detailed breakdown of avoided emissions for a calculation. "
        "Includes recycling credits (virgin material displacement) and energy "
        "recovery credits (displaced grid electricity). Reported separately "
        "from gross emissions per GHG Protocol guidance."
    ),
)
async def get_avoided_emissions(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get avoided emissions breakdown for a calculation.

    Args:
        calculation_id: Calculation UUID
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with recycling and energy recovery credits

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting avoided emissions for calc={calculation_id}"
        )

        result = await service.get_avoided_emissions(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Avoided emissions for {calculation_id} not found",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_avoided_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve avoided emissions",
        )


@router.get(
    "/circularity-score/{calculation_id}",
    response_model=Dict[str, Any],
    summary="Get circularity metrics",
    description=(
        "Retrieve circularity metrics for a calculation including recycling "
        "rate, material recovery rate, and circularity score. Based on "
        "Ellen MacArthur Foundation Material Circularity Indicator (MCI) "
        "methodology."
    ),
)
async def get_circularity_score(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get circularity metrics for a calculation.

    Args:
        calculation_id: Calculation UUID
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with circularity score and metrics

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting circularity score for calc={calculation_id}"
        )

        result = await service.get_circularity_score(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Circularity data for {calculation_id} not found",
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_circularity_score: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve circularity score",
        )


@router.get(
    "/aggregations",
    response_model=Dict[str, Any],
    summary="Get aggregated results",
    description=(
        "Retrieve aggregated end-of-life treatment emissions for a period. "
        "Returns totals with breakdowns by treatment method, material type, "
        "and product category. Supports daily, monthly, quarterly, and "
        "annual aggregation."
    ),
)
async def get_aggregations(
    org_id: Optional[str] = Query(
        None, description="Filter by organization"
    ),
    period: str = Query(
        "monthly",
        description="Aggregation period (daily, monthly, quarterly, annual)",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get aggregated end-of-life treatment emissions.

    Args:
        org_id: Optional organization filter
        period: Aggregation period
        from_date: Optional start date
        to_date: Optional end date
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for aggregation failures
    """
    try:
        logger.info(
            f"Getting aggregations: org={org_id}, period={period}"
        )

        valid_periods = {"daily", "monthly", "quarterly", "annual"}
        if period not in valid_periods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(valid_periods))}"
                ),
            )

        filters = {
            "org_id": org_id,
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.get_aggregations(filters)

        return {
            "period": period,
            "org_id": org_id,
            "total_mass_tonnes": result.get("total_mass_tonnes", 0.0),
            "gross_co2e_tonnes": result.get("gross_co2e_tonnes", 0.0),
            "avoided_co2e_tonnes": result.get("avoided_co2e_tonnes", 0.0),
            "net_co2e_tonnes": result.get("net_co2e_tonnes", 0.0),
            "by_treatment": result.get("by_treatment", {}),
            "by_material": result.get("by_material", {}),
            "by_category": result.get("by_category", {}),
            "calculation_count": result.get("calculation_count", 0),
        }

    except HTTPException:
        raise
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
    response_model=Dict[str, Any],
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a calculation. "
        "Includes all 10 pipeline stages (validate, classify, normalize, "
        "resolve_efs, calculate, allocate, aggregate, compliance, provenance, "
        "seal) with per-stage hashes and verification status."
    ),
)
async def get_provenance(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> Dict[str, Any]:
    """
    Get provenance chain for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: EndOfLifeTreatmentService instance

    Returns:
        Dictionary with provenance chain stages and verification

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting provenance for calculation: {calculation_id}"
        )

        result = await service.get_provenance(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Provenance for calculation {calculation_id} not found"
                ),
            )

        return {
            "calculation_id": result.get(
                "calculation_id", calculation_id
            ),
            "chain": result.get("chain", []),
            "is_valid": result.get("is_valid", False),
            "root_hash": result.get("root_hash", ""),
            "stages": result.get("stages", []),
            "verified_at": result.get(
                "verified_at", datetime.utcnow().isoformat()
            ),
        }

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
        "Health check endpoint for the End-of-Life Treatment Agent. "
        "Returns service status, agent identifier, version, uptime, "
        "and connection status. No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status, uptime, and connectivity
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        # Attempt lightweight service connectivity check
        db_connected = True
        cache_connected = True
        try:
            svc = get_service()
            if hasattr(svc, "health_check"):
                health = await svc.health_check()
                db_connected = health.get("database_connected", True)
                cache_connected = health.get("cache_connected", True)
        except Exception:
            db_connected = False
            cache_connected = False

        return HealthResponse(
            status="healthy" if db_connected else "degraded",
            agent_id="GL-MRV-S3-012",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            database_connected=db_connected,
            cache_connected=cache_connected,
        )

    except Exception as e:
        logger.error(f"Error in health_check: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-012",
            version="1.0.0",
            uptime_seconds=0.0,
            database_connected=False,
            cache_connected=False,
        )
