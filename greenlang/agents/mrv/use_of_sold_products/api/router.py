"""
Use of Sold Products Agent API Router - AGENT-MRV-024

This module implements the FastAPI router for use-of-sold-products emissions
calculations following GHG Protocol Scope 3 Category 11 requirements.

Provides 22 REST endpoints for:
- Full pipeline calculation with 10-stage processing
- Direct use-phase emissions (fuel combustion, refrigerant leakage, chemical release)
- Indirect use-phase emissions (electricity consumption, heating fuel, steam/cooling)
- Fuels and feedstocks sold (end-user combustion/oxidation)
- Batch processing (up to 10,000 products per request)
- Portfolio analysis across product categories
- Compliance checking across 7 regulatory frameworks
- Product energy profile lookup (24 product types)
- Refrigerant GWP reference table (AR5/AR6)
- Fuel combustion emission factors (15 fuel types)
- Product lifetime estimates (by category with adjustment)
- Aggregated emissions by period and category
- Provenance tracking with SHA-256 chain verification
- Health check monitoring

Follows GreenLang's zero-hallucination principle with deterministic calculations.
All numeric outputs use deterministic formulas; no LLM calls in the calculation path.

Emission Types:
    Direct: Fuel combustion (vehicles, generators), refrigerant leakage (HVAC),
            chemical release (consumer products)
    Indirect: Electricity consumption (appliances, IT), heating fuel (furnaces),
              steam/cooling (district energy)
    Fuels: Combustion of fuels/feedstocks sold to end users

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.use_of_sold_products.api.router import usp_router
    >>> app = FastAPI()
    >>> app.include_router(usp_router)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from decimal import Decimal
import logging
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM JSON ENCODER FOR DECIMAL SERIALIZATION
# ============================================================================


class DecimalEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles Decimal serialization.

    Converts Decimal values to float for JSON compatibility while
    preserving precision during internal calculations.
    """

    def default(self, obj: Any) -> Any:
        """Encode Decimal as float, delegate all else to parent."""
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def serialize_decimal(value: Any) -> Any:
    """
    Recursively convert Decimal values to float for API responses.

    Args:
        value: Any value that may contain Decimal instances

    Returns:
        The same structure with Decimal values converted to float
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: serialize_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_decimal(item) for item in value]
    return value


# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================


usp_router = APIRouter(
    prefix="/api/v1/use-of-sold-products",
    tags=["Use of Sold Products"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# SERVICE DEPENDENCY
# ============================================================================


_service_instance = None


def get_service():
    """
    Get or create UseOfSoldProductsService singleton instance.

    Returns:
        UseOfSoldProductsService instance

    Raises:
        HTTPException: If service initialization fails (503)
    """
    global _service_instance

    if _service_instance is None:
        try:
            from greenlang.agents.mrv.use_of_sold_products.service import (
                UseOfSoldProductsService,
            )
            _service_instance = UseOfSoldProductsService()
            logger.info("UseOfSoldProductsService initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize UseOfSoldProductsService: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service initialization failed",
            )

    return _service_instance


# ============================================================================
# REQUEST MODELS (12)
# ============================================================================


class FullPipelineRequest(BaseModel):
    """
    Request model for full pipeline use-of-sold-products emissions calculation.

    Supports all product categories and emission types with automatic routing
    to the appropriate calculation engine based on product classification.

    Attributes:
        product_name: Product identifier or name
        product_category: Product category for routing to calculation engine
        emission_type: Emission type (direct, indirect, fuels_feedstocks)
        units_sold: Number of units sold in the reporting period
        product_lifetime_years: Expected useful life of the product in years
        annual_energy_kwh: Annual electricity consumption per unit (indirect)
        annual_fuel_consumption: Annual fuel consumption per unit (direct)
        fuel_type: Fuel type for direct combustion calculations
        refrigerant_type: Refrigerant type for leakage calculations
        refrigerant_charge_kg: Initial refrigerant charge per unit (kg)
        annual_leak_rate: Annual refrigerant leak rate as fraction (0-1)
        grid_region: Grid region for electricity emission factor lookup
        gwp_version: GWP assessment report version (AR5 or AR6)
        reporting_year: Reporting year for the calculation
        metadata: Additional product-specific metadata
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    product_category: str = Field(
        ...,
        description=(
            "Product category (VEHICLES, APPLIANCES, HVAC, LIGHTING, "
            "IT_EQUIPMENT, INDUSTRIAL_EQUIPMENT, FUELS_FEEDSTOCKS, "
            "BUILDING_PRODUCTS, CONSUMER_PRODUCTS, MEDICAL_DEVICES)"
        ),
    )
    emission_type: str = Field(
        "direct",
        description="Emission type (direct, indirect, fuels_feedstocks)",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in the reporting period",
    )
    product_lifetime_years: Optional[float] = Field(
        None,
        gt=0,
        le=100,
        description="Expected useful life of the product in years",
    )
    annual_energy_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual electricity consumption per unit (kWh)",
    )
    annual_fuel_consumption: Optional[float] = Field(
        None,
        ge=0,
        description="Annual fuel consumption per unit (litres or kg)",
    )
    fuel_type: Optional[str] = Field(
        None,
        description="Fuel type for direct combustion (gasoline, diesel, natural_gas, etc.)",
    )
    refrigerant_type: Optional[str] = Field(
        None,
        description="Refrigerant type (R-410A, R-134a, R-32, etc.)",
    )
    refrigerant_charge_kg: Optional[float] = Field(
        None,
        ge=0,
        description="Initial refrigerant charge per unit (kg)",
    )
    annual_leak_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1.0,
        description="Annual refrigerant leak rate as fraction (0.0-1.0)",
    )
    grid_region: Optional[str] = Field(
        None,
        description="Grid region for electricity EF (US_AVERAGE, EU_AVERAGE, etc.)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP assessment report version (AR5 or AR6)",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year for the calculation",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional product-specific metadata",
    )

    @validator("product_category")
    def validate_product_category(cls, v: str) -> str:
        """Validate product_category is a recognized category."""
        valid_categories = {
            "VEHICLES", "APPLIANCES", "HVAC", "LIGHTING",
            "IT_EQUIPMENT", "INDUSTRIAL_EQUIPMENT", "FUELS_FEEDSTOCKS",
            "BUILDING_PRODUCTS", "CONSUMER_PRODUCTS", "MEDICAL_DEVICES",
        }
        upper_v = v.upper()
        if upper_v not in valid_categories:
            raise ValueError(
                f"Invalid product_category '{v}'. "
                f"Must be one of: {', '.join(sorted(valid_categories))}"
            )
        return upper_v

    @validator("emission_type")
    def validate_emission_type(cls, v: str) -> str:
        """Validate emission_type is a recognized type."""
        valid_types = {"direct", "indirect", "fuels_feedstocks"}
        lower_v = v.lower()
        if lower_v not in valid_types:
            raise ValueError(
                f"Invalid emission_type '{v}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}"
            )
        return lower_v

    @validator("gwp_version")
    def validate_gwp_version(cls, v: str) -> str:
        """Validate GWP version is AR5 or AR6."""
        upper_v = v.upper()
        if upper_v not in {"AR5", "AR6"}:
            raise ValueError(
                f"Invalid gwp_version '{v}'. Must be AR5 or AR6."
            )
        return upper_v


class DirectFuelRequest(BaseModel):
    """
    Request model for direct fuel combustion emissions from product use.

    Calculates emissions from the combustion of fuel during product use
    (e.g., gasoline in vehicles, natural gas in generators).

    Formula: emissions = units_sold * lifetime * annual_fuel * EF

    Attributes:
        product_name: Product identifier
        product_category: Product category
        units_sold: Number of units sold
        product_lifetime_years: Product lifetime in years
        annual_fuel_consumption: Annual fuel consumption per unit
        fuel_type: Type of fuel consumed
        fuel_unit: Unit of fuel measurement
        degradation_rate: Annual efficiency degradation rate
        gwp_version: GWP version (AR5 or AR6)
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    product_category: str = Field(
        "VEHICLES",
        description="Product category (VEHICLES, INDUSTRIAL_EQUIPMENT)",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in reporting period",
    )
    product_lifetime_years: float = Field(
        ...,
        gt=0,
        le=100,
        description="Expected useful life in years",
    )
    annual_fuel_consumption: float = Field(
        ...,
        gt=0,
        description="Annual fuel consumption per unit (litres or m3)",
    )
    fuel_type: str = Field(
        ...,
        description=(
            "Fuel type (gasoline, diesel, natural_gas, lpg, ethanol, "
            "biodiesel, jet_fuel, kerosene, fuel_oil, propane, "
            "butane, cng, lng, hydrogen, wood_pellets)"
        ),
    )
    fuel_unit: str = Field(
        "litres",
        description="Fuel unit (litres, kg, m3, gallons)",
    )
    degradation_rate: float = Field(
        0.0,
        ge=0,
        le=0.2,
        description="Annual efficiency degradation rate (0.0-0.2)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class DirectRefrigerantRequest(BaseModel):
    """
    Request model for direct refrigerant leakage emissions from product use.

    Calculates emissions from refrigerant leakage during the use phase
    of sold HVAC equipment, refrigerators, and cooling systems.

    Formula: emissions = units_sold * charge_kg * leak_rate * lifetime * GWP

    Attributes:
        product_name: Product identifier
        product_category: Product category
        units_sold: Number of units sold
        product_lifetime_years: Product lifetime in years
        refrigerant_type: Refrigerant identifier
        refrigerant_charge_kg: Initial charge per unit (kg)
        annual_leak_rate: Annual leak rate as fraction
        gwp_version: GWP version (AR5 or AR6)
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    product_category: str = Field(
        "HVAC",
        description="Product category (HVAC, APPLIANCES)",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in reporting period",
    )
    product_lifetime_years: float = Field(
        ...,
        gt=0,
        le=50,
        description="Expected useful life in years",
    )
    refrigerant_type: str = Field(
        ...,
        description=(
            "Refrigerant type (R-134a, R-410A, R-32, R-404A, R-407C, "
            "R-507A, R-22, R-290, R-600a, R-1234yf)"
        ),
    )
    refrigerant_charge_kg: float = Field(
        ...,
        gt=0,
        le=500,
        description="Initial refrigerant charge per unit (kg)",
    )
    annual_leak_rate: float = Field(
        ...,
        gt=0,
        le=1.0,
        description="Annual refrigerant leak rate as fraction (0.0-1.0)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class DirectChemicalRequest(BaseModel):
    """
    Request model for direct chemical release emissions from product use.

    Calculates emissions from the release of GHGs contained in or
    generated by consumer products (aerosols, solvents, fertilizers).

    Formula: emissions = units_sold * ghg_content_kg * release_fraction * GWP

    Attributes:
        product_name: Product identifier
        units_sold: Number of units sold
        ghg_content_kg: GHG content per unit (kg)
        release_fraction: Fraction released during use (0-1)
        ghg_type: Type of GHG released
        gwp_version: GWP version (AR5 or AR6)
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in reporting period",
    )
    ghg_content_kg: float = Field(
        ...,
        gt=0,
        description="GHG content per product unit (kg)",
    )
    release_fraction: float = Field(
        ...,
        gt=0,
        le=1.0,
        description="Fraction released during use phase (0.0-1.0)",
    )
    ghg_type: str = Field(
        "CO2",
        description="GHG type (CO2, CH4, N2O, HFC, SF6)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class IndirectElectricityRequest(BaseModel):
    """
    Request model for indirect electricity consumption emissions.

    Calculates emissions from electricity consumed by sold products
    during their use phase (appliances, IT, lighting, medical devices).

    Formula: emissions = units_sold * lifetime * annual_kwh * grid_EF * (1 + degradation)^year

    Attributes:
        product_name: Product identifier
        product_category: Product category
        units_sold: Number of units sold
        product_lifetime_years: Product lifetime in years
        annual_energy_kwh: Annual electricity consumption per unit (kWh)
        grid_region: Electricity grid region for EF lookup
        degradation_rate: Annual efficiency degradation rate
        usage_adjustment: Usage adjustment factor name
        gwp_version: GWP version (AR5 or AR6)
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    product_category: str = Field(
        "APPLIANCES",
        description=(
            "Product category (APPLIANCES, IT_EQUIPMENT, LIGHTING, "
            "MEDICAL_DEVICES, BUILDING_PRODUCTS)"
        ),
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in reporting period",
    )
    product_lifetime_years: float = Field(
        ...,
        gt=0,
        le=100,
        description="Expected useful life in years",
    )
    annual_energy_kwh: float = Field(
        ...,
        gt=0,
        description="Annual electricity consumption per unit (kWh)",
    )
    grid_region: str = Field(
        "US_AVERAGE",
        description=(
            "Grid region (US_AVERAGE, EU_AVERAGE, UK_GRID, CN_GRID, "
            "IN_GRID, JP_GRID, etc.)"
        ),
    )
    degradation_rate: float = Field(
        0.0,
        ge=0,
        le=0.2,
        description="Annual efficiency degradation rate (0.0-0.2)",
    )
    usage_adjustment: Optional[str] = Field(
        None,
        description="Usage adjustment factor (climate_hot, climate_cold, etc.)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class IndirectHeatingRequest(BaseModel):
    """
    Request model for indirect heating fuel consumption emissions.

    Calculates emissions from heating fuel consumed by sold products
    (furnaces, boilers, water heaters).

    Formula: emissions = units_sold * lifetime * annual_fuel * EF

    Attributes:
        product_name: Product identifier
        product_category: Product category
        units_sold: Number of units sold
        product_lifetime_years: Product lifetime in years
        annual_fuel_consumption: Annual fuel consumption per unit
        fuel_type: Heating fuel type
        fuel_unit: Fuel measurement unit
        efficiency: Product thermal efficiency
        gwp_version: GWP version (AR5 or AR6)
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    product_category: str = Field(
        "HVAC",
        description="Product category (HVAC, INDUSTRIAL_EQUIPMENT)",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in reporting period",
    )
    product_lifetime_years: float = Field(
        ...,
        gt=0,
        le=50,
        description="Expected useful life in years",
    )
    annual_fuel_consumption: float = Field(
        ...,
        gt=0,
        description="Annual fuel consumption per unit (litres, m3, or kg)",
    )
    fuel_type: str = Field(
        ...,
        description="Fuel type (natural_gas, fuel_oil, propane, kerosene, wood_pellets)",
    )
    fuel_unit: str = Field(
        "m3",
        description="Fuel unit (litres, m3, kg, therms)",
    )
    efficiency: float = Field(
        0.90,
        gt=0,
        le=1.0,
        description="Product thermal efficiency (0.0-1.0)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class IndirectSteamRequest(BaseModel):
    """
    Request model for indirect steam and cooling consumption emissions.

    Calculates emissions from purchased steam or cooling consumed by
    sold products that use district energy systems.

    Formula: emissions = units_sold * lifetime * annual_consumption * steam_EF

    Attributes:
        product_name: Product identifier
        units_sold: Number of units sold
        product_lifetime_years: Product lifetime in years
        annual_consumption_kwh: Annual steam/cooling consumption per unit (kWh)
        energy_type: Type of district energy
        steam_factor_source: Source of steam/cooling emission factor
        gwp_version: GWP version (AR5 or AR6)
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product identifier or name",
    )
    units_sold: int = Field(
        ...,
        ge=1,
        le=100000000,
        description="Number of units sold in reporting period",
    )
    product_lifetime_years: float = Field(
        ...,
        gt=0,
        le=50,
        description="Expected useful life in years",
    )
    annual_consumption_kwh: float = Field(
        ...,
        gt=0,
        description="Annual steam/cooling consumption per unit (kWh)",
    )
    energy_type: str = Field(
        "steam",
        description="District energy type (steam, hot_water, chilled_water, cooling)",
    )
    steam_factor_source: str = Field(
        "DEFAULT",
        description="Emission factor source (DEFAULT, EPA, IEA, SUPPLIER)",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class FuelsSoldRequest(BaseModel):
    """
    Request model for fuels and feedstocks sold to end users.

    Calculates downstream emissions from the combustion or oxidation
    of fuels and feedstocks sold by the reporting company.

    Formula: emissions = quantity_sold * EF (combustion or oxidation)

    Attributes:
        fuel_name: Fuel or feedstock product name
        fuel_type: Fuel type identifier
        quantity_sold: Quantity sold in reporting period
        quantity_unit: Unit of measurement
        is_feedstock: Whether sold as feedstock (partial oxidation)
        feedstock_oxidation_fraction: Fraction oxidized during use
        reporting_year: Reporting year
        gwp_version: GWP version (AR5 or AR6)
    """

    fuel_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Fuel or feedstock product name",
    )
    fuel_type: str = Field(
        ...,
        description=(
            "Fuel type (gasoline, diesel, natural_gas, lpg, jet_fuel, "
            "kerosene, fuel_oil, propane, butane, coal_bituminous, "
            "coal_sub_bituminous, coal_lignite, coke, petroleum_coke, "
            "wood_pellets)"
        ),
    )
    quantity_sold: float = Field(
        ...,
        gt=0,
        description="Quantity sold in reporting period",
    )
    quantity_unit: str = Field(
        "litres",
        description="Unit of measurement (litres, kg, m3, tonnes, gallons, MMBtu)",
    )
    is_feedstock: bool = Field(
        False,
        description="Whether sold as feedstock (assumes partial oxidation)",
    )
    feedstock_oxidation_fraction: float = Field(
        1.0,
        ge=0,
        le=1.0,
        description="Fraction oxidized during use (feedstock only, 0.0-1.0)",
    )
    reporting_year: int = Field(
        2024,
        ge=2000,
        le=2100,
        description="Reporting year",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version (AR5 or AR6)",
    )


class BatchCalculateRequest(BaseModel):
    """
    Request model for batch use-of-sold-products emissions calculations.

    Processes multiple products in a single request with parallel execution
    and per-product error isolation.

    Attributes:
        products: List of product calculation dictionaries
        reporting_period: Reporting period identifier
        gwp_version: GWP version (AR5 or AR6)
    """

    products: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="List of product calculation dictionaries",
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period identifier (e.g. '2024-Q4')",
    )
    gwp_version: str = Field(
        "AR5",
        description="GWP version for all calculations (AR5 or AR6)",
    )


class UncertaintyAnalysisRequest(BaseModel):
    """
    Request model for uncertainty analysis of use-of-sold-products calculations.

    Supports Monte Carlo simulation, analytical error propagation,
    and IPCC Tier 2 default uncertainty ranges. Accounts for product
    lifetime uncertainty and emission factor uncertainty.

    Attributes:
        method: Uncertainty analysis method
        iterations: Monte Carlo iterations (if applicable)
        confidence_level: Confidence interval level (0.90, 0.95, 0.99)
        calculation_results: Calculation results to analyze
        include_lifetime_uncertainty: Whether to include lifetime uncertainty
    """

    method: str = Field(
        "monte_carlo",
        description="Uncertainty method (monte_carlo, analytical, ipcc_tier_2)",
    )
    iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Monte Carlo iterations (ignored for analytical/ipcc_tier_2)",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence interval level",
    )
    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to analyze for uncertainty",
    )
    include_lifetime_uncertainty: bool = Field(
        True,
        description="Whether to include product lifetime uncertainty",
    )


class PortfolioAnalysisRequest(BaseModel):
    """
    Request model for portfolio-level analysis across product categories.

    Performs aggregated analysis of emissions across the full product
    portfolio, identifying hot-spots and reduction opportunities.

    Attributes:
        calculation_ids: List of existing calculation IDs to analyze
        product_results: Alternatively, provide product results directly
        group_by: Grouping dimension for analysis
        top_n: Number of top contributors to return
    """

    calculation_ids: Optional[List[str]] = Field(
        None,
        description="List of existing calculation UUIDs to include in analysis",
    )
    product_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of product calculation result dicts",
    )
    group_by: str = Field(
        "product_category",
        description="Grouping dimension (product_category, emission_type, fuel_type)",
    )
    top_n: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of top contributors to return",
    )


class ComplianceCheckRequest(BaseModel):
    """
    Request model for multi-framework compliance checking.

    Checks calculation results against selected regulatory frameworks
    for completeness, boundary correctness, and disclosure requirements
    specific to Scope 3 Category 11.

    Attributes:
        frameworks: List of framework identifiers to check against
        calculation_results: List of calculation result dicts to check
        lifetime_disclosed: Whether product lifetimes have been disclosed
        methodology_documented: Whether calculation methodology is documented
        category_breakdown_provided: Whether per-category breakdown is provided
    """

    frameworks: List[str] = Field(
        ...,
        min_items=1,
        description=(
            "Frameworks to check (ghg_protocol, iso_14064, csrd_esrs, "
            "cdp, sbti, sb_253, gri)"
        ),
    )
    calculation_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Calculation results to check for compliance",
    )
    lifetime_disclosed: bool = Field(
        False,
        description="Whether product lifetimes have been disclosed",
    )
    methodology_documented: bool = Field(
        False,
        description="Whether calculation methodology is documented",
    )
    category_breakdown_provided: bool = Field(
        False,
        description="Whether per-category breakdown is provided",
    )


# ============================================================================
# RESPONSE MODELS (14)
# ============================================================================


class CalculateResponse(BaseModel):
    """Response model for single pipeline calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    product_name: str = Field(..., description="Product name or identifier")
    product_category: str = Field(..., description="Product category")
    emission_type: str = Field(
        ..., description="Emission type (direct, indirect, fuels_feedstocks)"
    )
    method: str = Field(..., description="Calculation method applied")
    total_co2e_kg: float = Field(
        ..., description="Total lifetime CO2e emissions in kg"
    )
    annual_co2e_kg: float = Field(
        ..., description="Annual CO2e emissions per unit in kg"
    )
    units_sold: int = Field(
        ..., description="Number of units sold"
    )
    product_lifetime_years: float = Field(
        ..., description="Product lifetime used in calculation"
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


class DirectEmissionResponse(BaseModel):
    """Response model for direct use-phase emission calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    product_name: str = Field(..., description="Product name")
    emission_source: str = Field(
        ..., description="Source (fuel_combustion, refrigerant_leakage, chemical_release)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total lifetime CO2e emissions (kg)"
    )
    annual_co2e_per_unit_kg: float = Field(
        ..., description="Annual CO2e per unit (kg)"
    )
    lifetime_co2e_per_unit_kg: float = Field(
        ..., description="Lifetime CO2e per unit (kg)"
    )
    units_sold: int = Field(..., description="Units sold")
    product_lifetime_years: float = Field(
        ..., description="Product lifetime (years)"
    )
    ef_value: float = Field(
        ..., description="Emission factor value applied"
    )
    ef_unit: str = Field(
        ..., description="Emission factor unit"
    )
    ef_source: str = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 timestamp"
    )


class IndirectEmissionResponse(BaseModel):
    """Response model for indirect use-phase emission calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    product_name: str = Field(..., description="Product name")
    energy_source: str = Field(
        ..., description="Energy source (electricity, heating_fuel, steam, cooling)"
    )
    total_co2e_kg: float = Field(
        ..., description="Total lifetime CO2e emissions (kg)"
    )
    annual_co2e_per_unit_kg: float = Field(
        ..., description="Annual CO2e per unit (kg)"
    )
    lifetime_co2e_per_unit_kg: float = Field(
        ..., description="Lifetime CO2e per unit (kg)"
    )
    units_sold: int = Field(..., description="Units sold")
    product_lifetime_years: float = Field(
        ..., description="Product lifetime (years)"
    )
    annual_consumption: float = Field(
        ..., description="Annual energy consumption per unit"
    )
    consumption_unit: str = Field(
        ..., description="Consumption unit (kWh, litres, m3)"
    )
    grid_ef: float = Field(
        ..., description="Grid/fuel emission factor applied"
    )
    grid_region: Optional[str] = Field(
        None, description="Grid region (electricity only)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 timestamp"
    )


class FuelsSoldResponse(BaseModel):
    """Response model for fuels and feedstocks sold calculation."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    fuel_name: str = Field(..., description="Fuel or feedstock name")
    fuel_type: str = Field(..., description="Fuel type identifier")
    total_co2e_kg: float = Field(
        ..., description="Total CO2e from sold fuel combustion/oxidation (kg)"
    )
    quantity_sold: float = Field(..., description="Quantity sold")
    quantity_unit: str = Field(..., description="Unit of measurement")
    is_feedstock: bool = Field(
        ..., description="Whether sold as feedstock"
    )
    ef_value: float = Field(
        ..., description="Combustion emission factor applied"
    )
    ef_unit: str = Field(
        ..., description="Emission factor unit"
    )
    ncv: Optional[float] = Field(
        None, description="Net calorific value (MJ/unit)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    calculated_at: str = Field(
        ..., description="ISO 8601 timestamp"
    )


class BatchCalculateResponse(BaseModel):
    """Response model for batch calculation."""

    batch_id: str = Field(..., description="Unique batch UUID")
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual product calculation results"
    )
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for all products in batch"
    )
    count: int = Field(..., description="Number of successful calculations")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-product error details"
    )
    reporting_period: str = Field(
        ..., description="Reporting period identifier"
    )


class PortfolioAnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""

    total_co2e_kg: float = Field(
        ..., description="Total portfolio CO2e (kg)"
    )
    product_count: int = Field(
        ..., description="Number of products analyzed"
    )
    by_category: Dict[str, float] = Field(
        ..., description="CO2e breakdown by product category"
    )
    by_emission_type: Dict[str, float] = Field(
        ..., description="CO2e breakdown by emission type"
    )
    top_contributors: List[Dict[str, Any]] = Field(
        ..., description="Top N contributing products"
    )
    reduction_opportunities: List[Dict[str, Any]] = Field(
        ..., description="Identified reduction opportunities"
    )


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check."""

    results: List[Dict[str, Any]] = Field(
        ..., description="Per-framework compliance results"
    )
    overall_status: str = Field(
        ..., description="Overall compliance status (pass, fail, warning)"
    )
    overall_score: float = Field(
        ..., description="Overall compliance score (0.0-1.0)"
    )


class CalculationDetailResponse(BaseModel):
    """Response model for single calculation detail."""

    calculation_id: str = Field(..., description="Unique calculation UUID")
    product_name: str = Field(..., description="Product name")
    product_category: str = Field(..., description="Product category")
    emission_type: str = Field(..., description="Emission type")
    method: str = Field(..., description="Calculation method")
    total_co2e_kg: float = Field(..., description="Total CO2e (kg)")
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
    count: int = Field(..., description="Total matching calculations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


class DeleteResponse(BaseModel):
    """Response model for soft deletion."""

    calculation_id: str = Field(..., description="Deleted calculation UUID")
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Human-readable status message")


class EmissionFactorListResponse(BaseModel):
    """Response model for emission factor listing."""

    factors: List[Dict[str, Any]] = Field(
        ..., description="List of emission factors"
    )
    count: int = Field(..., description="Total factor count returned")
    category: Optional[str] = Field(
        None, description="Emission factor category filter applied"
    )


class AggregationResponse(BaseModel):
    """Response model for aggregated emissions."""

    period: str = Field(..., description="Aggregation period identifier")
    total_co2e_kg: float = Field(
        ..., description="Total CO2e for the period (kg)"
    )
    by_category: Dict[str, float] = Field(
        ..., description="CO2e breakdown by product category"
    )
    by_emission_type: Dict[str, float] = Field(
        ..., description="CO2e breakdown by emission type (direct/indirect/fuels)"
    )
    product_count: int = Field(
        ..., description="Total number of product calculations"
    )


class ProvenanceResponse(BaseModel):
    """Response model for provenance chain verification."""

    calculation_id: str = Field(..., description="Calculation UUID")
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

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Agent version")
    uptime_seconds: float = Field(
        ..., description="Seconds since service start"
    )


# ============================================================================
# MODULE-LEVEL TRACKING
# ============================================================================

_start_time: datetime = datetime.utcnow()


# ============================================================================
# ENDPOINT 1: POST /calculate - Full Pipeline
# ============================================================================


@usp_router.post(
    "/calculate",
    response_model=CalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate use-of-sold-products emissions",
    description=(
        "Calculate GHG emissions for the use phase of sold products through "
        "the full 10-stage pipeline. Supports all product categories and "
        "emission types (direct, indirect, fuels/feedstocks). Routes to the "
        "appropriate calculation engine based on product classification. "
        "Returns deterministic results with SHA-256 provenance hash."
    ),
)
async def calculate_emissions(
    request: FullPipelineRequest,
    service=Depends(get_service),
) -> CalculateResponse:
    """
    Calculate use-of-sold-products emissions through the full pipeline.

    Args:
        request: Full pipeline calculation request
        service: UseOfSoldProductsService instance

    Returns:
        CalculateResponse with lifetime emissions and provenance hash

    Raises:
        HTTPException: 400 for validation errors, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating use-phase emissions: product={request.product_name}, "
            f"category={request.product_category}, type={request.emission_type}"
        )

        result = await service.calculate(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return CalculateResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            product_category=result.get(
                "product_category", request.product_category
            ),
            emission_type=result.get("emission_type", request.emission_type),
            method=result.get("method", "full_pipeline"),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_kg=result.get("annual_co2e_kg", 0.0),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years",
                request.product_lifetime_years or 10.0,
            ),
            dqi_score=result.get("dqi_score"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in calculate_emissions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Calculation failed",
        )


# ============================================================================
# ENDPOINT 2: POST /calculate/direct/fuel - Direct Fuel Combustion
# ============================================================================


@usp_router.post(
    "/calculate/direct/fuel",
    response_model=DirectEmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate direct fuel combustion emissions",
    description=(
        "Calculate GHG emissions from direct fuel combustion during product "
        "use. Applies to vehicles, generators, and other fuel-consuming "
        "products. Uses DEFRA/EPA fuel-specific emission factors with "
        "optional efficiency degradation modeling."
    ),
)
async def calculate_direct_fuel(
    request: DirectFuelRequest,
    service=Depends(get_service),
) -> DirectEmissionResponse:
    """
    Calculate direct fuel combustion emissions from product use.

    Args:
        request: Direct fuel combustion request
        service: UseOfSoldProductsService instance

    Returns:
        DirectEmissionResponse with fuel combustion emissions

    Raises:
        HTTPException: 400 for invalid fuel type, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating direct fuel emissions: product={request.product_name}, "
            f"fuel={request.fuel_type}, units={request.units_sold}"
        )

        result = await service.calculate_direct_fuel(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return DirectEmissionResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            emission_source="fuel_combustion",
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_per_unit_kg=result.get("annual_co2e_per_unit_kg", 0.0),
            lifetime_co2e_per_unit_kg=result.get(
                "lifetime_co2e_per_unit_kg", 0.0
            ),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years", request.product_lifetime_years
            ),
            ef_value=result.get("ef_value", 0.0),
            ef_unit=result.get("ef_unit", "kgCO2e/litre"),
            ef_source=result.get("ef_source", "DEFRA_2024"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_direct_fuel: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_direct_fuel: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Direct fuel calculation failed",
        )


# ============================================================================
# ENDPOINT 3: POST /calculate/direct/refrigerant - Refrigerant Leakage
# ============================================================================


@usp_router.post(
    "/calculate/direct/refrigerant",
    response_model=DirectEmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate direct refrigerant leakage emissions",
    description=(
        "Calculate GHG emissions from refrigerant leakage during product "
        "use phase. Applies to HVAC systems, refrigerators, and cooling "
        "equipment. Uses IPCC AR5/AR6 GWP values for 10 common refrigerants."
    ),
)
async def calculate_direct_refrigerant(
    request: DirectRefrigerantRequest,
    service=Depends(get_service),
) -> DirectEmissionResponse:
    """
    Calculate direct refrigerant leakage emissions from product use.

    Args:
        request: Refrigerant leakage request
        service: UseOfSoldProductsService instance

    Returns:
        DirectEmissionResponse with refrigerant leakage emissions

    Raises:
        HTTPException: 400 for invalid refrigerant, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating refrigerant leakage emissions: "
            f"product={request.product_name}, "
            f"refrigerant={request.refrigerant_type}, "
            f"charge={request.refrigerant_charge_kg}kg"
        )

        result = await service.calculate_direct_refrigerant(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return DirectEmissionResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            emission_source="refrigerant_leakage",
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_per_unit_kg=result.get("annual_co2e_per_unit_kg", 0.0),
            lifetime_co2e_per_unit_kg=result.get(
                "lifetime_co2e_per_unit_kg", 0.0
            ),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years", request.product_lifetime_years
            ),
            ef_value=result.get("ef_value", 0.0),
            ef_unit=result.get("ef_unit", "kgCO2e/kg-refrigerant"),
            ef_source=result.get("ef_source", "IPCC_AR5"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_direct_refrigerant: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_direct_refrigerant: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Refrigerant leakage calculation failed",
        )


# ============================================================================
# ENDPOINT 4: POST /calculate/direct/chemical - Chemical Release
# ============================================================================


@usp_router.post(
    "/calculate/direct/chemical",
    response_model=DirectEmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate direct chemical release emissions",
    description=(
        "Calculate GHG emissions from chemical release during product use. "
        "Applies to consumer products containing GHGs (aerosols, solvents, "
        "fertilizers). Uses product-specific GHG content and release fractions."
    ),
)
async def calculate_direct_chemical(
    request: DirectChemicalRequest,
    service=Depends(get_service),
) -> DirectEmissionResponse:
    """
    Calculate direct chemical release emissions from product use.

    Args:
        request: Chemical release request
        service: UseOfSoldProductsService instance

    Returns:
        DirectEmissionResponse with chemical release emissions

    Raises:
        HTTPException: 400 for invalid GHG type, 500 for processing failures
    """
    try:
        logger.info(
            f"Calculating chemical release emissions: "
            f"product={request.product_name}, "
            f"ghg={request.ghg_type}, units={request.units_sold}"
        )

        result = await service.calculate_direct_chemical(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return DirectEmissionResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            emission_source="chemical_release",
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_per_unit_kg=result.get("annual_co2e_per_unit_kg", 0.0),
            lifetime_co2e_per_unit_kg=result.get(
                "lifetime_co2e_per_unit_kg", 0.0
            ),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years", 1.0
            ),
            ef_value=result.get("ef_value", 0.0),
            ef_unit=result.get("ef_unit", "kgCO2e/kg-product"),
            ef_source=result.get("ef_source", "IPCC_AR5"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_direct_chemical: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_direct_chemical: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chemical release calculation failed",
        )


# ============================================================================
# ENDPOINT 5: POST /calculate/indirect/electricity - Electricity Consumption
# ============================================================================


@usp_router.post(
    "/calculate/indirect/electricity",
    response_model=IndirectEmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate indirect electricity consumption emissions",
    description=(
        "Calculate GHG emissions from electricity consumed by sold products "
        "during their use phase. Applies to appliances, IT equipment, "
        "lighting, and medical devices. Uses region-specific grid emission "
        "factors (16 regions) with optional degradation modeling."
    ),
)
async def calculate_indirect_electricity(
    request: IndirectElectricityRequest,
    service=Depends(get_service),
) -> IndirectEmissionResponse:
    """
    Calculate indirect electricity consumption emissions.

    Args:
        request: Indirect electricity request
        service: UseOfSoldProductsService instance

    Returns:
        IndirectEmissionResponse with electricity-related emissions

    Raises:
        HTTPException: 400 for invalid grid region, 500 for failures
    """
    try:
        logger.info(
            f"Calculating indirect electricity emissions: "
            f"product={request.product_name}, "
            f"kwh={request.annual_energy_kwh}, region={request.grid_region}"
        )

        result = await service.calculate_indirect_electricity(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return IndirectEmissionResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            energy_source="electricity",
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_per_unit_kg=result.get("annual_co2e_per_unit_kg", 0.0),
            lifetime_co2e_per_unit_kg=result.get(
                "lifetime_co2e_per_unit_kg", 0.0
            ),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years", request.product_lifetime_years
            ),
            annual_consumption=result.get(
                "annual_consumption", request.annual_energy_kwh
            ),
            consumption_unit="kWh",
            grid_ef=result.get("grid_ef", 0.0),
            grid_region=result.get("grid_region", request.grid_region),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_indirect_electricity: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_indirect_electricity: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indirect electricity calculation failed",
        )


# ============================================================================
# ENDPOINT 6: POST /calculate/indirect/heating - Heating Fuel
# ============================================================================


@usp_router.post(
    "/calculate/indirect/heating",
    response_model=IndirectEmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate indirect heating fuel emissions",
    description=(
        "Calculate GHG emissions from heating fuel consumed by sold products "
        "during their use phase. Applies to furnaces, boilers, and water "
        "heaters. Uses fuel-specific emission factors with thermal "
        "efficiency adjustments."
    ),
)
async def calculate_indirect_heating(
    request: IndirectHeatingRequest,
    service=Depends(get_service),
) -> IndirectEmissionResponse:
    """
    Calculate indirect heating fuel emissions from product use.

    Args:
        request: Indirect heating request
        service: UseOfSoldProductsService instance

    Returns:
        IndirectEmissionResponse with heating-related emissions

    Raises:
        HTTPException: 400 for invalid fuel type, 500 for failures
    """
    try:
        logger.info(
            f"Calculating indirect heating emissions: "
            f"product={request.product_name}, "
            f"fuel={request.fuel_type}, "
            f"consumption={request.annual_fuel_consumption}"
        )

        result = await service.calculate_indirect_heating(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return IndirectEmissionResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            energy_source="heating_fuel",
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_per_unit_kg=result.get("annual_co2e_per_unit_kg", 0.0),
            lifetime_co2e_per_unit_kg=result.get(
                "lifetime_co2e_per_unit_kg", 0.0
            ),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years", request.product_lifetime_years
            ),
            annual_consumption=result.get(
                "annual_consumption", request.annual_fuel_consumption
            ),
            consumption_unit=result.get("consumption_unit", request.fuel_unit),
            grid_ef=result.get("fuel_ef", 0.0),
            grid_region=None,
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_indirect_heating: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_indirect_heating: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indirect heating calculation failed",
        )


# ============================================================================
# ENDPOINT 7: POST /calculate/indirect/steam - Steam/Cooling
# ============================================================================


@usp_router.post(
    "/calculate/indirect/steam",
    response_model=IndirectEmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate indirect steam and cooling emissions",
    description=(
        "Calculate GHG emissions from purchased steam or cooling consumed "
        "by sold products using district energy systems. Uses supplier-specific "
        "or default emission factors for steam and chilled water."
    ),
)
async def calculate_indirect_steam(
    request: IndirectSteamRequest,
    service=Depends(get_service),
) -> IndirectEmissionResponse:
    """
    Calculate indirect steam and cooling emissions from product use.

    Args:
        request: Indirect steam/cooling request
        service: UseOfSoldProductsService instance

    Returns:
        IndirectEmissionResponse with steam/cooling-related emissions

    Raises:
        HTTPException: 400 for invalid energy type, 500 for failures
    """
    try:
        logger.info(
            f"Calculating indirect steam/cooling emissions: "
            f"product={request.product_name}, "
            f"type={request.energy_type}, "
            f"kwh={request.annual_consumption_kwh}"
        )

        result = await service.calculate_indirect_steam(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return IndirectEmissionResponse(
            calculation_id=calculation_id,
            product_name=result.get("product_name", request.product_name),
            energy_source=result.get("energy_source", request.energy_type),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            annual_co2e_per_unit_kg=result.get("annual_co2e_per_unit_kg", 0.0),
            lifetime_co2e_per_unit_kg=result.get(
                "lifetime_co2e_per_unit_kg", 0.0
            ),
            units_sold=result.get("units_sold", request.units_sold),
            product_lifetime_years=result.get(
                "product_lifetime_years", request.product_lifetime_years
            ),
            annual_consumption=result.get(
                "annual_consumption", request.annual_consumption_kwh
            ),
            consumption_unit="kWh",
            grid_ef=result.get("steam_ef", 0.0),
            grid_region=None,
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error(
            f"Validation error in calculate_indirect_steam: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_indirect_steam: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Indirect steam/cooling calculation failed",
        )


# ============================================================================
# ENDPOINT 8: POST /calculate/fuels - Fuels & Feedstocks Sold
# ============================================================================


@usp_router.post(
    "/calculate/fuels",
    response_model=FuelsSoldResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calculate fuels and feedstocks sold emissions",
    description=(
        "Calculate downstream GHG emissions from the combustion or oxidation "
        "of fuels and feedstocks sold by the reporting company. Supports "
        "15 fuel types with DEFRA/EPA/IPCC emission factors and net calorific "
        "values. Feedstocks use partial oxidation fractions."
    ),
)
async def calculate_fuels_sold(
    request: FuelsSoldRequest,
    service=Depends(get_service),
) -> FuelsSoldResponse:
    """
    Calculate emissions from fuels and feedstocks sold to end users.

    Args:
        request: Fuels sold request
        service: UseOfSoldProductsService instance

    Returns:
        FuelsSoldResponse with downstream combustion emissions

    Raises:
        HTTPException: 400 for invalid fuel type, 500 for failures
    """
    try:
        logger.info(
            f"Calculating fuels sold emissions: fuel={request.fuel_type}, "
            f"qty={request.quantity_sold} {request.quantity_unit}, "
            f"feedstock={request.is_feedstock}"
        )

        result = await service.calculate_fuels_sold(request.dict())
        result = serialize_decimal(result)
        calculation_id = result.get("calculation_id", str(uuid.uuid4()))

        return FuelsSoldResponse(
            calculation_id=calculation_id,
            fuel_name=result.get("fuel_name", request.fuel_name),
            fuel_type=result.get("fuel_type", request.fuel_type),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            quantity_sold=result.get("quantity_sold", request.quantity_sold),
            quantity_unit=result.get("quantity_unit", request.quantity_unit),
            is_feedstock=result.get("is_feedstock", request.is_feedstock),
            ef_value=result.get("ef_value", 0.0),
            ef_unit=result.get("ef_unit", "kgCO2e/litre"),
            ncv=result.get("ncv"),
            provenance_hash=result.get("provenance_hash", ""),
            calculated_at=result.get(
                "calculated_at", datetime.utcnow().isoformat()
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_fuels_sold: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_fuels_sold: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fuels sold calculation failed",
        )


# ============================================================================
# ENDPOINT 9: POST /calculate/batch - Batch Processing
# ============================================================================


@usp_router.post(
    "/calculate/batch",
    response_model=BatchCalculateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch calculate use-of-sold-products emissions",
    description=(
        "Calculate GHG emissions for multiple sold products in a single "
        "request. Processes up to 10,000 products with parallel execution "
        "and per-product error isolation. Returns aggregated totals with "
        "individual results and any per-product errors."
    ),
)
async def calculate_batch_emissions(
    request: BatchCalculateRequest,
    service=Depends(get_service),
) -> BatchCalculateResponse:
    """
    Calculate batch use-of-sold-products emissions.

    Args:
        request: Batch calculation request with product list
        service: UseOfSoldProductsService instance

    Returns:
        BatchCalculateResponse with aggregated and per-product results

    Raises:
        HTTPException: 400 for validation errors, 500 for batch failures
    """
    try:
        logger.info(
            f"Calculating batch emissions for {len(request.products)} products, "
            f"period={request.reporting_period}"
        )

        result = await service.calculate_batch(request.dict())
        result = serialize_decimal(result)
        batch_id = result.get("batch_id", str(uuid.uuid4()))

        return BatchCalculateResponse(
            batch_id=batch_id,
            results=result.get("results", []),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            count=result.get("count", 0),
            errors=result.get("errors", []),
            reporting_period=result.get(
                "reporting_period", request.reporting_period
            ),
        )

    except ValueError as e:
        logger.error("Validation error in calculate_batch_emissions: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Error in calculate_batch_emissions: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch calculation failed",
        )


# ============================================================================
# ENDPOINT 10: POST /calculate/portfolio - Portfolio Analysis
# ============================================================================


@usp_router.post(
    "/calculate/portfolio",
    response_model=PortfolioAnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze product portfolio emissions",
    description=(
        "Perform portfolio-level analysis of use-of-sold-products emissions "
        "across all product categories. Identifies top contributors and "
        "reduction opportunities using Pareto analysis. Groups results by "
        "product category, emission type, or fuel type."
    ),
)
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    service=Depends(get_service),
) -> PortfolioAnalysisResponse:
    """
    Analyze product portfolio emissions.

    Args:
        request: Portfolio analysis request
        service: UseOfSoldProductsService instance

    Returns:
        PortfolioAnalysisResponse with category breakdowns and hot-spots

    Raises:
        HTTPException: 400 for invalid input, 500 for analysis failures
    """
    try:
        logger.info(
            f"Analyzing portfolio: group_by={request.group_by}, "
            f"top_n={request.top_n}"
        )

        result = await service.analyze_portfolio(request.dict())
        result = serialize_decimal(result)

        return PortfolioAnalysisResponse(
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            product_count=result.get("product_count", 0),
            by_category=result.get("by_category", {}),
            by_emission_type=result.get("by_emission_type", {}),
            top_contributors=result.get("top_contributors", []),
            reduction_opportunities=result.get(
                "reduction_opportunities", []
            ),
        )

    except ValueError as e:
        logger.error("Validation error in analyze_portfolio: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in analyze_portfolio: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed",
        )


# ============================================================================
# ENDPOINT 11: POST /compliance/check - Compliance Validation
# ============================================================================


@usp_router.post(
    "/compliance/check",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check multi-framework compliance",
    description=(
        "Check use-of-sold-products calculation results against one or more "
        "regulatory frameworks. Validates completeness, boundary correctness, "
        "lifetime disclosure, methodology documentation, and category "
        "breakdown requirements specific to Scope 3 Category 11. Supports "
        "GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305."
    ),
)
async def check_compliance(
    request: ComplianceCheckRequest,
    service=Depends(get_service),
) -> ComplianceCheckResponse:
    """
    Check calculation compliance against regulatory frameworks.

    Args:
        request: Compliance check request with frameworks and results
        service: UseOfSoldProductsService instance

    Returns:
        ComplianceCheckResponse with per-framework findings

    Raises:
        HTTPException: 400 for invalid frameworks, 500 for check failures
    """
    try:
        logger.info(
            f"Checking compliance for {len(request.frameworks)} frameworks, "
            f"{len(request.calculation_results)} results"
        )

        result = await service.check_compliance(request.dict())
        result = serialize_decimal(result)

        return ComplianceCheckResponse(
            results=result.get("results", []),
            overall_status=result.get("overall_status", "unknown"),
            overall_score=result.get("overall_score", 0.0),
        )

    except ValueError as e:
        logger.error("Validation error in check_compliance: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Error in check_compliance: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed",
        )


# ============================================================================
# ENDPOINT 12: GET /calculations/{id} - Get Calculation by ID
# ============================================================================


@usp_router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationDetailResponse,
    summary="Get calculation detail",
    description=(
        "Retrieve detailed information for a specific use-of-sold-products "
        "calculation including full input/output payload, provenance hash, "
        "product details, and calculation metadata."
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
        service: UseOfSoldProductsService instance

    Returns:
        CalculationDetailResponse with full calculation data

    Raises:
        HTTPException: 404 if calculation not found, 500 for failures
    """
    try:
        logger.info("Getting calculation detail: %s", calculation_id)

        result = await service.get_calculation(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calculation {calculation_id} not found",
            )

        result = serialize_decimal(result)

        return CalculationDetailResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            product_name=result.get("product_name", ""),
            product_category=result.get("product_category", ""),
            emission_type=result.get("emission_type", ""),
            method=result.get("method", ""),
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
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


# ============================================================================
# ENDPOINT 13: GET /calculations - List with Pagination
# ============================================================================


@usp_router.get(
    "/calculations",
    response_model=CalculationListResponse,
    summary="List calculations",
    description=(
        "Retrieve a paginated list of use-of-sold-products calculations. "
        "Supports filtering by product category, emission type, and date "
        "range. Returns summary information for each calculation."
    ),
)
async def list_calculations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    product_category: Optional[str] = Query(
        None, description="Filter by product category"
    ),
    emission_type: Optional[str] = Query(
        None, description="Filter by emission type (direct, indirect, fuels_feedstocks)"
    ),
    from_date: Optional[str] = Query(
        None, description="Filter from date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="Filter to date (ISO 8601)"
    ),
    service=Depends(get_service),
) -> CalculationListResponse:
    """
    List use-of-sold-products calculations with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        product_category: Optional product category filter
        emission_type: Optional emission type filter
        from_date: Optional start date filter
        to_date: Optional end date filter
        service: UseOfSoldProductsService instance

    Returns:
        CalculationListResponse with paginated results

    Raises:
        HTTPException: 500 for listing failures
    """
    try:
        logger.info(
            f"Listing calculations: page={page}, size={page_size}, "
            f"category={product_category}, type={emission_type}"
        )

        filters = {
            "page": page,
            "page_size": page_size,
            "product_category": product_category,
            "emission_type": emission_type,
            "from_date": from_date,
            "to_date": to_date,
        }

        result = await service.list_calculations(filters)
        result = serialize_decimal(result)

        return CalculationListResponse(
            calculations=result.get("calculations", []),
            count=result.get("count", 0),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error("Error in list_calculations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list calculations",
        )


# ============================================================================
# ENDPOINT 14: DELETE /calculations/{id} - Delete Calculation
# ============================================================================


@usp_router.delete(
    "/calculations/{calculation_id}",
    response_model=DeleteResponse,
    summary="Delete calculation",
    description=(
        "Soft-delete a specific use-of-sold-products calculation. "
        "Marks the calculation as deleted with audit trail; "
        "data is retained for regulatory compliance."
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
        service: UseOfSoldProductsService instance

    Returns:
        DeleteResponse with deletion confirmation

    Raises:
        HTTPException: 404 if not found, 500 for deletion failures
    """
    try:
        logger.info("Deleting calculation: %s", calculation_id)

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
        logger.error("Error in delete_calculation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete calculation",
        )


# ============================================================================
# ENDPOINT 15: GET /emission-factors/{category} - EFs by Category
# ============================================================================


@usp_router.get(
    "/emission-factors/{category}",
    response_model=EmissionFactorListResponse,
    summary="Get emission factors by category",
    description=(
        "Retrieve emission factors for a specific category. Categories "
        "include fuel, grid, refrigerant, steam, and chemical. Returns "
        "all factors for the specified category with source and unit details."
    ),
)
async def get_emission_factors_by_category(
    category: str = Path(
        ...,
        description="EF category (fuel, grid, refrigerant, steam, chemical)",
    ),
    source: Optional[str] = Query(
        None, description="Filter by EF source (DEFRA, EPA, IPCC, IEA)"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get emission factors for a specific category.

    Args:
        category: Emission factor category
        source: Optional source filter
        service: UseOfSoldProductsService instance

    Returns:
        EmissionFactorListResponse with category-specific factors

    Raises:
        HTTPException: 400 for invalid category, 500 for retrieval failures
    """
    try:
        logger.info(
            f"Getting emission factors for category: {category}, "
            f"source={source}"
        )

        valid_categories = {"fuel", "grid", "refrigerant", "steam", "chemical"}
        if category.lower() not in valid_categories:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid category '{category}'. "
                    f"Must be one of: {', '.join(sorted(valid_categories))}"
                ),
            )

        filters = {"category": category.lower(), "source": source}
        result = await service.get_emission_factors(filters)
        result = serialize_decimal(result)

        return EmissionFactorListResponse(
            factors=result.get("factors", []),
            count=result.get("count", 0),
            category=category.lower(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in get_emission_factors_by_category: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve emission factors",
        )


# ============================================================================
# ENDPOINT 16: GET /energy-profiles - Product Energy Profiles
# ============================================================================


@usp_router.get(
    "/energy-profiles",
    response_model=EmissionFactorListResponse,
    summary="Get product energy profiles",
    description=(
        "Retrieve product energy profiles with default lifetime, annual "
        "energy consumption, and energy unit for 24 product types across "
        "10 categories. Used as default parameters when product-specific "
        "data is unavailable."
    ),
)
async def get_energy_profiles(
    category: Optional[str] = Query(
        None, description="Filter by product category"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get product energy profiles.

    Args:
        category: Optional product category filter
        service: UseOfSoldProductsService instance

    Returns:
        EmissionFactorListResponse with product energy profiles

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting product energy profiles: category=%s", category)

        filters = {"category": category}
        result = await service.get_energy_profiles(filters)
        result = serialize_decimal(result)

        return EmissionFactorListResponse(
            factors=result.get("profiles", []),
            count=result.get("count", 0),
            category=category,
        )

    except Exception as e:
        logger.error(
            f"Error in get_energy_profiles: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve energy profiles",
        )


# ============================================================================
# ENDPOINT 17: GET /refrigerant-gwps - Refrigerant GWP Table
# ============================================================================


@usp_router.get(
    "/refrigerant-gwps",
    response_model=EmissionFactorListResponse,
    summary="Get refrigerant GWP table",
    description=(
        "Retrieve Global Warming Potential (GWP) values for 10 common "
        "refrigerants with both AR5 and AR6 values. Includes typical "
        "charge size and annual leak rate for each refrigerant type."
    ),
)
async def get_refrigerant_gwps(
    gwp_version: Optional[str] = Query(
        None, description="Filter by GWP version (AR5, AR6)"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get refrigerant GWP reference table.

    Args:
        gwp_version: Optional GWP version filter
        service: UseOfSoldProductsService instance

    Returns:
        EmissionFactorListResponse with refrigerant GWP data

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting refrigerant GWPs: version=%s", gwp_version)

        filters = {"gwp_version": gwp_version}
        result = await service.get_refrigerant_gwps(filters)
        result = serialize_decimal(result)

        return EmissionFactorListResponse(
            factors=result.get("refrigerants", []),
            count=result.get("count", 0),
            category="refrigerant",
        )

    except Exception as e:
        logger.error(
            f"Error in get_refrigerant_gwps: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve refrigerant GWPs",
        )


# ============================================================================
# ENDPOINT 18: GET /fuel-factors - Fuel Combustion EFs
# ============================================================================


@usp_router.get(
    "/fuel-factors",
    response_model=EmissionFactorListResponse,
    summary="Get fuel combustion emission factors",
    description=(
        "Retrieve combustion emission factors for 15 fuel types including "
        "gasoline, diesel, natural gas, LPG, and solid fuels. Returns "
        "emission factor, net calorific value, and factor unit."
    ),
)
async def get_fuel_factors(
    fuel_type: Optional[str] = Query(
        None, description="Filter by fuel type"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get fuel combustion emission factors.

    Args:
        fuel_type: Optional fuel type filter
        service: UseOfSoldProductsService instance

    Returns:
        EmissionFactorListResponse with fuel emission factors

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting fuel factors: type=%s", fuel_type)

        filters = {"fuel_type": fuel_type}
        result = await service.get_fuel_factors(filters)
        result = serialize_decimal(result)

        return EmissionFactorListResponse(
            factors=result.get("factors", []),
            count=result.get("count", 0),
            category="fuel",
        )

    except Exception as e:
        logger.error("Error in get_fuel_factors: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fuel factors",
        )


# ============================================================================
# ENDPOINT 19: GET /lifetime-estimates - Lifetime Tables
# ============================================================================


@usp_router.get(
    "/lifetime-estimates",
    response_model=EmissionFactorListResponse,
    summary="Get product lifetime estimates",
    description=(
        "Retrieve expected product lifetime estimates by category with "
        "adjustment factors. Used for lifetime modeling when "
        "product-specific data is unavailable."
    ),
)
async def get_lifetime_estimates(
    category: Optional[str] = Query(
        None, description="Filter by product category"
    ),
    service=Depends(get_service),
) -> EmissionFactorListResponse:
    """
    Get product lifetime estimates.

    Args:
        category: Optional product category filter
        service: UseOfSoldProductsService instance

    Returns:
        EmissionFactorListResponse with lifetime estimates

    Raises:
        HTTPException: 500 for retrieval failures
    """
    try:
        logger.info("Getting lifetime estimates: category=%s", category)

        filters = {"category": category}
        result = await service.get_lifetime_estimates(filters)
        result = serialize_decimal(result)

        return EmissionFactorListResponse(
            factors=result.get("lifetimes", []),
            count=result.get("count", 0),
            category=category,
        )

    except Exception as e:
        logger.error(
            f"Error in get_lifetime_estimates: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve lifetime estimates",
        )


# ============================================================================
# ENDPOINT 20: GET /aggregations - Aggregated Results
# ============================================================================


@usp_router.get(
    "/aggregations",
    response_model=AggregationResponse,
    summary="Get aggregated emissions",
    description=(
        "Retrieve aggregated use-of-sold-products emissions for a specified "
        "period. Returns totals with breakdowns by product category and "
        "emission type. Supports daily, weekly, monthly, quarterly, and "
        "annual aggregation periods."
    ),
)
async def get_aggregations(
    period: str = Query(
        "monthly",
        description="Aggregation period (daily, weekly, monthly, quarterly, annual)",
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO 8601)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO 8601)"
    ),
    product_category: Optional[str] = Query(
        None, description="Filter by product category"
    ),
    service=Depends(get_service),
) -> AggregationResponse:
    """
    Get aggregated emissions for a specified period.

    Args:
        period: Aggregation period identifier
        from_date: Optional start date filter
        to_date: Optional end date filter
        product_category: Optional product category filter
        service: UseOfSoldProductsService instance

    Returns:
        AggregationResponse with aggregated emissions data

    Raises:
        HTTPException: 400 for invalid period, 500 for aggregation failures
    """
    try:
        logger.info(
            f"Getting aggregations: period={period}, "
            f"from={from_date}, to={to_date}, category={product_category}"
        )

        valid_periods = {
            "daily", "weekly", "monthly", "quarterly", "annual",
        }
        if period not in valid_periods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(valid_periods))}"
                ),
            )

        filters = {
            "period": period,
            "from_date": from_date,
            "to_date": to_date,
            "product_category": product_category,
        }

        result = await service.get_aggregations(filters)
        result = serialize_decimal(result)

        return AggregationResponse(
            period=period,
            total_co2e_kg=result.get("total_co2e_kg", 0.0),
            by_category=result.get("by_category", {}),
            by_emission_type=result.get("by_emission_type", {}),
            product_count=result.get("product_count", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_aggregations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aggregation failed",
        )


# ============================================================================
# ENDPOINT 21: GET /provenance/{id} - Provenance Chain
# ============================================================================


@usp_router.get(
    "/provenance/{calculation_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance chain",
    description=(
        "Retrieve the complete SHA-256 provenance chain for a calculation. "
        "Includes all 10 pipeline stages (validate, classify, normalize, "
        "resolve_efs, calculate, lifetime, aggregate, compliance, "
        "provenance, seal) with per-stage hashes and verification."
    ),
)
async def get_provenance(
    calculation_id: str = Path(..., description="Calculation UUID"),
    service=Depends(get_service),
) -> ProvenanceResponse:
    """
    Get provenance chain for a specific calculation.

    Args:
        calculation_id: Calculation UUID
        service: UseOfSoldProductsService instance

    Returns:
        ProvenanceResponse with chain stages and verification status

    Raises:
        HTTPException: 404 if not found, 500 for retrieval failures
    """
    try:
        logger.info("Getting provenance for calculation: %s", calculation_id)

        result = await service.get_provenance(calculation_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Provenance for calculation {calculation_id} not found"
                ),
            )

        result = serialize_decimal(result)

        return ProvenanceResponse(
            calculation_id=result.get("calculation_id", calculation_id),
            chain=result.get("chain", []),
            is_valid=result.get("is_valid", False),
            root_hash=result.get("root_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in get_provenance: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provenance",
        )


# ============================================================================
# ENDPOINT 22: GET /health - Health Check
# ============================================================================


@usp_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Health check endpoint for the Use of Sold Products Agent. "
        "Returns service status, agent identifier, version, and uptime. "
        "No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Returns:
        HealthResponse with service status and uptime
    """
    try:
        uptime = (datetime.utcnow() - _start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            agent_id="GL-MRV-S3-011",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
        )

    except Exception as e:
        logger.error("Error in health_check: %s", e, exc_info=True)
        return HealthResponse(
            status="unhealthy",
            agent_id="GL-MRV-S3-011",
            version="1.0.0",
            uptime_seconds=0.0,
        )
