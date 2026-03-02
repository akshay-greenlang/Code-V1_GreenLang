# -*- coding: utf-8 -*-
"""
Use of Sold Products Service Setup - AGENT-MRV-024

This module provides the service facade that wires together all 7 engines
for Use of Sold Products emissions calculations (Scope 3 Category 11).

The UseOfSoldProductsService class provides a high-level API for:
- Direct use-phase emissions (fuel combustion, refrigerant leakage, chemical release)
- Indirect use-phase emissions (electricity, heating fuel, steam/cooling)
- Fuels and feedstocks sold to end users
- Product lifetime modeling (degradation, total lifetime emissions)
- Compliance checking across 7 regulatory frameworks
- Double-counting prevention (8 rules: DC-USP-001 through DC-USP-008)
- Multi-dimensional aggregation (by category, emission type, method)
- Provenance tracking with SHA-256 audit trail
- Portfolio-level analysis and hot-spot identification

Engines:
    1. ProductUseDatabaseEngine - Emission factor data and persistence
    2. DirectEmissionsCalculatorEngine - Fuel combustion, refrigerant, chemical
    3. IndirectEmissionsCalculatorEngine - Electricity, heating, steam/cooling
    4. FuelsAndFeedstocksCalculatorEngine - Fuels sold, feedstocks sold
    5. LifetimeModelingEngine - Product lifetime and degradation
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. UseOfSoldProductsPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.use_of_sold_products.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate(CalculationRequest(
    ...     products=[{"product_category": "vehicles", "fuel_type": "petrol",
    ...                "fuel_consumption_litres_per_year": 1500,
    ...                "units_sold": 10000, "lifetime_years": 15}],
    ...     org_id="ORG-001",
    ...     reporting_year=2024,
    ... ))
    >>> assert response.success

Integration:
    >>> from greenlang.use_of_sold_products.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/use-of-sold-products")

Module: greenlang.use_of_sold_products.setup
Agent: AGENT-MRV-024
Version: 1.0.0
"""

import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional["UseOfSoldProductsService"] = None

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

SERVICE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"
API_PREFIX: str = "/api/v1/use-of-sold-products"
TABLE_PREFIX: str = "gl_usp_"
METRICS_PREFIX: str = "gl_usp_"


# ============================================================================
# Request Models
# ============================================================================


class ProductInput(BaseModel):
    """Input model for a single product in Category 11 calculation."""

    product_category: str = Field(
        ...,
        description=(
            "Product use category: vehicles, appliances, hvac, lighting, "
            "it_equipment, industrial_equipment, fuels_feedstocks, "
            "building_products, consumer_products, medical_devices"
        ),
    )
    units_sold: int = Field(..., gt=0, description="Number of units sold in reporting period")
    product_name: Optional[str] = Field(None, description="Product name for reporting")

    # Direct fuel combustion fields
    fuel_type: Optional[str] = Field(None, description="Fuel type: petrol, diesel, natural_gas, lpg, etc.")
    fuel_consumption_litres_per_year: Optional[float] = Field(None, gt=0, description="Annual fuel consumption per unit (litres)")
    fuel_consumption_gallons_per_year: Optional[float] = Field(None, gt=0, description="Annual fuel consumption per unit (gallons)")

    # Direct refrigerant leakage fields
    refrigerant_type: Optional[str] = Field(None, description="Refrigerant type: R-134a, R-410A, R-32, etc.")
    refrigerant_charge_kg: Optional[float] = Field(None, gt=0, description="Refrigerant charge per unit (kg)")
    annual_leak_rate: Optional[float] = Field(None, ge=0, le=1, description="Annual leak rate (0-1 fraction)")

    # Direct chemical release fields
    chemical_type: Optional[str] = Field(None, description="Chemical type for direct release")
    chemical_content_kg: Optional[float] = Field(None, gt=0, description="Chemical content per unit (kg)")
    release_fraction: Optional[float] = Field(None, ge=0, le=1, description="Fraction released during use (0-1)")
    chemical_gwp: Optional[float] = Field(None, ge=0, description="GWP of the chemical")

    # Indirect electricity fields
    electricity_kwh_per_year: Optional[float] = Field(None, gt=0, description="Annual electricity per unit (kWh)")
    electricity_wh_per_year: Optional[float] = Field(None, gt=0, description="Annual electricity per unit (Wh)")
    electricity_mwh_per_year: Optional[float] = Field(None, gt=0, description="Annual electricity per unit (MWh)")
    energy_btu_per_year: Optional[float] = Field(None, gt=0, description="Annual energy per unit (BTU)")
    country_code: Optional[str] = Field("GLOBAL", description="ISO country code or eGRID subregion")
    egrid_subregion: Optional[str] = Field(None, description="US eGRID subregion code")

    # Indirect heating fuel fields
    heating_fuel_type: Optional[str] = Field(None, description="Heating fuel type: natural_gas, heating_oil, etc.")
    heating_fuel_consumption: Optional[float] = Field(None, gt=0, description="Annual heating fuel per unit")

    # Indirect steam/cooling fields
    steam_kwh_per_year: Optional[float] = Field(None, gt=0, description="Annual steam consumption per unit (kWh)")
    cooling_kwh_per_year: Optional[float] = Field(None, gt=0, description="Annual cooling consumption per unit (kWh)")
    steam_ef: Optional[float] = Field(None, gt=0, description="Steam emission factor (kgCO2e/kWh)")

    # Fuels/feedstocks sold fields
    quantity_sold: Optional[float] = Field(None, gt=0, description="Total quantity of fuel/feedstock sold")
    quantity_unit: Optional[str] = Field("litres", description="Unit: litres, m3, kg")
    is_feedstock: Optional[bool] = Field(False, description="True if feedstock (not fuel)")
    carbon_content: Optional[float] = Field(None, ge=0, le=1, description="Carbon content fraction")
    oxidation_factor: Optional[float] = Field(None, ge=0, le=1, description="Oxidation factor")

    # Lifetime fields
    lifetime_years: Optional[int] = Field(None, gt=0, le=100, description="Product lifetime in years")
    degradation_rate: Optional[float] = Field(None, ge=0, le=0.5, description="Annual efficiency degradation rate")

    # Boundary / DC prevention fields
    sold_to_self: Optional[bool] = Field(False, description="Sold to reporting company itself")
    own_use: Optional[bool] = Field(False, description="Used by reporting company")
    is_intermediate: Optional[bool] = Field(False, description="Intermediate product for further processing")
    is_leased_downstream: Optional[bool] = Field(False, description="Leased to downstream user")
    includes_end_of_life: Optional[bool] = Field(False, description="Includes end-of-life emissions")
    is_fuel_product: Optional[bool] = Field(False, description="Product is a fuel")
    fuel_used_internally: Optional[bool] = Field(False, description="Fuel also consumed by reporting company")

    # Override
    calculation_method: Optional[str] = Field(None, description="Explicit calculation method override")

    @validator("product_category")
    def validate_category(cls, v: str) -> str:
        """Validate product use category."""
        allowed = [
            "vehicles", "appliances", "hvac", "lighting", "it_equipment",
            "industrial_equipment", "fuels_feedstocks", "building_products",
            "consumer_products", "medical_devices",
        ]
        if v.lower() not in allowed:
            raise ValueError(f"product_category must be one of {allowed}")
        return v.lower()


class CalculationRequest(BaseModel):
    """Request model for full Category 11 calculation."""

    products: List[ProductInput] = Field(..., min_length=1, description="List of products")
    org_id: str = Field("", description="Organization identifier")
    reporting_year: int = Field(2024, ge=2015, le=2035, description="Reporting year")
    method: Optional[str] = Field(None, description="Explicit method override for all products")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class DirectFuelRequest(BaseModel):
    """Request model for direct fuel combustion calculation."""

    product_category: str = Field("vehicles", description="Product category")
    fuel_type: str = Field(..., description="Fuel type: petrol, diesel, lpg, etc.")
    fuel_consumption_litres_per_year: float = Field(..., gt=0, description="Annual fuel per unit (litres)")
    units_sold: int = Field(..., gt=0, description="Units sold")
    lifetime_years: Optional[int] = Field(None, gt=0, description="Lifetime years")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class DirectRefrigerantRequest(BaseModel):
    """Request model for direct refrigerant leakage calculation."""

    product_category: str = Field("hvac", description="Product category")
    refrigerant_type: str = Field(..., description="Refrigerant: R-134a, R-410A, R-32, etc.")
    refrigerant_charge_kg: float = Field(..., gt=0, description="Charge per unit (kg)")
    annual_leak_rate: float = Field(0.02, ge=0, le=1, description="Annual leak rate")
    units_sold: int = Field(..., gt=0, description="Units sold")
    lifetime_years: Optional[int] = Field(None, gt=0, description="Lifetime years")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class DirectChemicalRequest(BaseModel):
    """Request model for direct chemical release calculation."""

    product_category: str = Field("consumer_products", description="Product category")
    chemical_type: str = Field(..., description="Chemical type")
    chemical_content_kg: float = Field(..., gt=0, description="Chemical content per unit (kg)")
    release_fraction: float = Field(1.0, ge=0, le=1, description="Release fraction")
    chemical_gwp: float = Field(1.0, ge=0, description="Chemical GWP")
    units_sold: int = Field(..., gt=0, description="Units sold")
    lifetime_years: Optional[int] = Field(None, gt=0, description="Lifetime years")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class IndirectElectricityRequest(BaseModel):
    """Request model for indirect electricity calculation."""

    product_category: str = Field("appliances", description="Product category")
    electricity_kwh_per_year: float = Field(..., gt=0, description="Annual electricity per unit (kWh)")
    units_sold: int = Field(..., gt=0, description="Units sold")
    country_code: str = Field("GLOBAL", description="ISO country code")
    lifetime_years: Optional[int] = Field(None, gt=0, description="Lifetime years")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class IndirectHeatingRequest(BaseModel):
    """Request model for indirect heating fuel calculation."""

    product_category: str = Field("hvac", description="Product category")
    heating_fuel_type: str = Field(..., description="Fuel type: natural_gas, heating_oil, etc.")
    heating_fuel_consumption: float = Field(..., gt=0, description="Annual fuel per unit")
    units_sold: int = Field(..., gt=0, description="Units sold")
    lifetime_years: Optional[int] = Field(None, gt=0, description="Lifetime years")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class IndirectSteamRequest(BaseModel):
    """Request model for indirect steam/cooling calculation."""

    product_category: str = Field("building_products", description="Product category")
    steam_kwh_per_year: Optional[float] = Field(None, gt=0, description="Annual steam (kWh)")
    cooling_kwh_per_year: Optional[float] = Field(None, gt=0, description="Annual cooling (kWh)")
    steam_ef: Optional[float] = Field(None, gt=0, description="Steam EF (kgCO2e/kWh)")
    units_sold: int = Field(..., gt=0, description="Units sold")
    country_code: str = Field("GLOBAL", description="ISO country code")
    lifetime_years: Optional[int] = Field(None, gt=0, description="Lifetime years")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class FuelsSoldRequest(BaseModel):
    """Request model for fuels sold calculation."""

    fuel_type: str = Field(..., description="Fuel type: petrol, diesel, natural_gas, etc.")
    quantity_sold: float = Field(..., gt=0, description="Total fuel quantity sold")
    quantity_unit: str = Field("litres", description="Unit: litres, m3, kg")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class BatchCalculationRequest(BaseModel):
    """Request model for batch portfolio calculation."""

    portfolios: List[CalculationRequest] = Field(..., min_length=1, description="List of portfolios")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio-level analysis."""

    products: List[ProductInput] = Field(..., min_length=1, description="Products to analyze")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    calculation_id: str = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["ghg_protocol"],
        description="Frameworks: ghg_protocol, iso_14064, csrd_esrs, cdp, sbti, sb_253, gri",
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ProductProfileRequest(BaseModel):
    """Request model for product profile lookup."""

    product_category: str = Field(..., description="Product category")
    product_name: Optional[str] = Field(None, description="Specific product name")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class AggregationRequest(BaseModel):
    """Request model for aggregation queries."""

    reporting_year: int = Field(2024, ge=2015, le=2035, description="Reporting year")
    group_by: Optional[str] = Field("category", description="Group by: category, emission_type, method")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


# ============================================================================
# Response Models
# ============================================================================


class CalculationResponse(BaseModel):
    """Response model for Category 11 calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    total_co2e_kg: float = Field(0.0, description="Total lifetime CO2e in kg")
    direct_co2e_kg: float = Field(0.0, description="Direct use-phase CO2e in kg")
    indirect_co2e_kg: float = Field(0.0, description="Indirect use-phase CO2e in kg")
    wtt_co2e_kg: float = Field(0.0, description="Well-to-tank CO2e in kg")
    by_category: Dict[str, float] = Field(default_factory=dict, description="Emissions by product category")
    by_emission_type: Dict[str, float] = Field(default_factory=dict, description="Emissions by type")
    by_method: Dict[str, float] = Field(default_factory=dict, description="Emissions by method")
    product_count: int = Field(0, description="Number of products processed")
    total_units_sold: int = Field(0, description="Total units sold")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    reporting_year: int = Field(2024, description="Reporting year")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")
    detail: Dict[str, Any] = Field(default_factory=dict, description="Full pipeline result")


class BatchResponse(BaseModel):
    """Response model for batch calculation."""

    success: bool = Field(..., description="Overall success flag")
    total_portfolios: int = Field(0, description="Total portfolios processed")
    successful: int = Field(0, description="Successful portfolios")
    failed: int = Field(0, description="Failed portfolios")
    total_co2e_kg: float = Field(0.0, description="Total CO2e across all portfolios")
    results: List[CalculationResponse] = Field(default_factory=list, description="Individual results")
    errors: List[dict] = Field(default_factory=list, description="Error details")
    processing_time_ms: float = Field(0.0, description="Total processing time")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation checked")
    overall_status: str = Field(..., description="Overall status: PASS, WARNING, FAIL")
    overall_score: float = Field(0.0, description="Overall compliance score (0-100)")
    framework_results: Dict[str, Any] = Field(default_factory=dict, description="Per-framework results")
    total_findings: int = Field(0, description="Total compliance findings")
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Check timestamp")
    processing_time_ms: float = Field(0.0, description="Processing time")


class ProductProfileResponse(BaseModel):
    """Response model for product profile lookup."""

    success: bool = Field(..., description="Success flag")
    product_category: str = Field(..., description="Product category")
    default_lifetime_years: int = Field(0, description="Default lifetime")
    default_degradation_rate: float = Field(0.0, description="Default degradation rate")
    emission_type: str = Field("", description="direct, indirect, or both")
    default_energy_kwh: Optional[float] = Field(None, description="Default annual energy")


class FuelEFResponse(BaseModel):
    """Response model for fuel emission factor lookup."""

    success: bool = Field(..., description="Success flag")
    fuel_type: str = Field(..., description="Fuel type")
    emission_factors: Dict[str, float] = Field(default_factory=dict, description="EF values")
    source: str = Field("DEFRA_2024", description="EF source")


class RefrigerantGWPResponse(BaseModel):
    """Response model for refrigerant GWP lookup."""

    success: bool = Field(..., description="Success flag")
    refrigerant_type: str = Field(..., description="Refrigerant type")
    gwp: int = Field(0, description="GWP value (AR5 100-year)")
    source: str = Field("IPCC_AR5", description="GWP source")


class GridEFResponse(BaseModel):
    """Response model for grid emission factor lookup."""

    success: bool = Field(..., description="Success flag")
    country_code: str = Field(..., description="Country/region code")
    grid_ef_kg_co2e_per_kwh: float = Field(0.0, description="Grid EF")
    source: str = Field("IEA_eGRID_2024", description="EF source")


class LifetimeEstimateResponse(BaseModel):
    """Response model for lifetime estimate."""

    success: bool = Field(..., description="Success flag")
    product_category: str = Field(..., description="Product category")
    lifetime_years: int = Field(0, description="Estimated lifetime")
    degradation_rate: float = Field(0.0, description="Annual degradation rate")
    source: str = Field("industry_average", description="Estimate source")


class DegradationRateResponse(BaseModel):
    """Response model for degradation rate lookup."""

    success: bool = Field(..., description="Success flag")
    product_category: str = Field(..., description="Product category")
    degradation_rate: float = Field(0.0, description="Annual degradation rate")


class CategoryListResponse(BaseModel):
    """Response model for listing all product categories."""

    success: bool = Field(..., description="Success flag")
    categories: List[Dict[str, Any]] = Field(default_factory=list, description="Category list")
    total_count: int = Field(0, description="Total categories")


class EnergyProfilesResponse(BaseModel):
    """Response model for energy profiles."""

    success: bool = Field(..., description="Success flag")
    profiles: Dict[str, float] = Field(default_factory=dict, description="Product -> kWh/year")
    total_count: int = Field(0, description="Total profiles")


class ProvenanceResponse(BaseModel):
    """Response model for provenance queries."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation ID")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    chain: List[dict] = Field(default_factory=list, description="Provenance chain entries")
    is_valid: bool = Field(False, description="Chain integrity verified")


class AggregationResponse(BaseModel):
    """Response model for aggregation queries."""

    success: bool = Field(..., description="Success flag")
    total_co2e_kg: float = Field(0.0, description="Total CO2e")
    by_category: Dict[str, float] = Field(default_factory=dict, description="By category")
    by_emission_type: Dict[str, float] = Field(default_factory=dict, description="By emission type")
    by_method: Dict[str, float] = Field(default_factory=dict, description="By method")
    reporting_year: int = Field(2024, description="Reporting year")
    processing_time_ms: float = Field(0.0, description="Processing time")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    engines_status: Dict[str, bool] = Field(default_factory=dict, description="Per-engine status")
    uptime_seconds: float = Field(0.0, description="Service uptime")


class VersionResponse(BaseModel):
    """Response model for version info."""

    version: str = Field(..., description="Service version")
    agent_id: str = Field(..., description="Agent identifier")
    agent_component: str = Field(..., description="Agent component")
    api_prefix: str = Field(..., description="API route prefix")


class ConfigResponse(BaseModel):
    """Response model for configuration."""

    success: bool = Field(..., description="Success flag")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration values")


# ============================================================================
# UseOfSoldProductsService Class
# ============================================================================


class UseOfSoldProductsService:
    """
    Use of Sold Products Service Facade.

    This service wires together all 7 engines to provide a complete API
    for Use of Sold Products emissions calculations (Scope 3 Category 11).

    The service supports:
        - Direct use-phase emissions (fuel, refrigerant, chemical)
        - Indirect use-phase emissions (electricity, heating, steam/cooling)
        - Fuels and feedstocks sold to end users
        - Product lifetime modeling with degradation
        - Compliance checking (7 regulatory frameworks)
        - Double-counting prevention (8 rules)
        - Multi-dimensional aggregation and reporting
        - Portfolio-level analysis and hot-spot identification

    Engines:
        1. ProductUseDatabaseEngine - Data persistence
        2. DirectEmissionsCalculatorEngine - Direct emissions
        3. IndirectEmissionsCalculatorEngine - Indirect emissions
        4. FuelsAndFeedstocksCalculatorEngine - Fuels/feedstocks
        5. LifetimeModelingEngine - Lifetime modeling
        6. ComplianceCheckerEngine - Compliance validation
        7. UseOfSoldProductsPipelineEngine - End-to-end pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(CalculationRequest(
        ...     products=[ProductInput(
        ...         product_category="vehicles",
        ...         fuel_type="petrol",
        ...         fuel_consumption_litres_per_year=1500,
        ...         units_sold=10000,
        ...         lifetime_years=15,
        ...     )],
        ...     org_id="ORG-001",
        ...     reporting_year=2024,
        ... ))
        >>> assert response.success
    """

    def __init__(self) -> None:
        """Initialize UseOfSoldProductsService with all 7 engines."""
        logger.info("Initializing UseOfSoldProductsService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        # Initialize engines with graceful fallback
        self._database_engine = self._init_engine(
            "greenlang.use_of_sold_products.product_use_database",
            "ProductUseDatabaseEngine",
        )
        self._direct_engine = self._init_engine(
            "greenlang.use_of_sold_products.direct_emissions_calculator",
            "DirectEmissionsCalculatorEngine",
        )
        self._indirect_engine = self._init_engine(
            "greenlang.use_of_sold_products.indirect_emissions_calculator",
            "IndirectEmissionsCalculatorEngine",
        )
        self._fuels_engine = self._init_engine(
            "greenlang.use_of_sold_products.fuels_feedstocks_calculator",
            "FuelsAndFeedstocksCalculatorEngine",
        )
        self._lifetime_engine = self._init_engine(
            "greenlang.use_of_sold_products.lifetime_modeling",
            "LifetimeModelingEngine",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.use_of_sold_products.compliance_checker",
            "ComplianceCheckerEngine",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.use_of_sold_products.use_of_sold_products_pipeline",
            "UseOfSoldProductsPipelineEngine",
        )

        # In-memory calculation store (for dev/testing; production uses DB)
        self._calculations: Dict[str, dict] = {}

        self._initialized = True
        logger.info("UseOfSoldProductsService initialized successfully")

    @staticmethod
    def _init_engine(module_path: str, class_name: str) -> Optional[Any]:
        """
        Initialize an engine with graceful ImportError handling.

        Args:
            module_path: Fully qualified module path.
            class_name: Class name within the module.

        Returns:
            Engine instance or None if import fails.
        """
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Check for singleton pattern
            if hasattr(cls, "get_instance"):
                instance = cls.get_instance()
            else:
                instance = cls()
            logger.info("%s initialized", class_name)
            return instance
        except ImportError:
            logger.warning("%s not available (ImportError)", class_name)
            return None
        except Exception as e:
            logger.warning("%s initialization failed: %s", class_name, e)
            return None

    # ========================================================================
    # Public API Methods - Core Calculations
    # ========================================================================

    def calculate(
        self, request: CalculationRequest
    ) -> CalculationResponse:
        """
        Calculate lifetime emissions for sold products (full pipeline).

        Delegates to the UseOfSoldProductsPipelineEngine for full
        10-stage processing with provenance tracking.

        Args:
            request: Calculation request with products and metadata.

        Returns:
            CalculationResponse with lifetime emissions and provenance.
        """
        start_time = time.monotonic()
        calc_id = f"usp-{uuid4().hex[:12]}"

        try:
            # Convert ProductInput models to dicts for pipeline
            product_dicts = [p.dict(exclude_none=True) for p in request.products]

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.run_pipeline(
                    product_dicts,
                    org_id=request.org_id,
                    year=request.reporting_year,
                )
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            response = CalculationResponse(
                success=True,
                calculation_id=calc_id,
                total_co2e_kg=float(result.get("total_co2e", 0)),
                direct_co2e_kg=float(result.get("direct_co2e", 0)),
                indirect_co2e_kg=float(result.get("indirect_co2e", 0)),
                wtt_co2e_kg=float(result.get("total_wtt_co2e", 0) if "total_wtt_co2e" in result else 0),
                by_category={
                    k: float(v) for k, v in result.get("by_category", {}).items()
                },
                by_emission_type={
                    k: float(v) for k, v in result.get("by_emission_type", {}).items()
                },
                by_method={
                    k: float(v) for k, v in result.get("by_method", {}).items()
                },
                product_count=result.get("product_count", 0),
                total_units_sold=result.get("total_units_sold", 0),
                provenance_hash=result.get("provenance_hash", ""),
                reporting_year=request.reporting_year,
                processing_time_ms=elapsed,
                detail=result,
            )

            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Calculation %s failed: %s", calc_id, e, exc_info=True)
            return CalculationResponse(
                success=False,
                calculation_id=calc_id,
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed,
            )

    def calculate_direct_fuel(
        self, request: DirectFuelRequest
    ) -> CalculationResponse:
        """
        Calculate direct fuel combustion emissions for sold products.

        Args:
            request: Direct fuel combustion request.

        Returns:
            CalculationResponse with fuel combustion emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category=request.product_category,
                fuel_type=request.fuel_type,
                fuel_consumption_litres_per_year=request.fuel_consumption_litres_per_year,
                units_sold=request.units_sold,
                lifetime_years=request.lifetime_years,
            )],
            org_id="direct_fuel",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_direct_refrigerant(
        self, request: DirectRefrigerantRequest
    ) -> CalculationResponse:
        """
        Calculate direct refrigerant leakage emissions for sold products.

        Args:
            request: Direct refrigerant leakage request.

        Returns:
            CalculationResponse with refrigerant leakage emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category=request.product_category,
                refrigerant_type=request.refrigerant_type,
                refrigerant_charge_kg=request.refrigerant_charge_kg,
                annual_leak_rate=request.annual_leak_rate,
                units_sold=request.units_sold,
                lifetime_years=request.lifetime_years,
            )],
            org_id="direct_refrigerant",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_direct_chemical(
        self, request: DirectChemicalRequest
    ) -> CalculationResponse:
        """
        Calculate direct chemical release emissions for sold products.

        Args:
            request: Direct chemical release request.

        Returns:
            CalculationResponse with chemical release emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category=request.product_category,
                chemical_type=request.chemical_type,
                chemical_content_kg=request.chemical_content_kg,
                release_fraction=request.release_fraction,
                chemical_gwp=request.chemical_gwp,
                units_sold=request.units_sold,
                lifetime_years=request.lifetime_years,
            )],
            org_id="direct_chemical",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_indirect_electricity(
        self, request: IndirectElectricityRequest
    ) -> CalculationResponse:
        """
        Calculate indirect electricity consumption emissions for sold products.

        Args:
            request: Indirect electricity request.

        Returns:
            CalculationResponse with electricity use-phase emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category=request.product_category,
                electricity_kwh_per_year=request.electricity_kwh_per_year,
                units_sold=request.units_sold,
                country_code=request.country_code,
                lifetime_years=request.lifetime_years,
            )],
            org_id="indirect_electricity",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_indirect_heating(
        self, request: IndirectHeatingRequest
    ) -> CalculationResponse:
        """
        Calculate indirect heating fuel emissions for sold products.

        Args:
            request: Indirect heating fuel request.

        Returns:
            CalculationResponse with heating fuel use-phase emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category=request.product_category,
                heating_fuel_type=request.heating_fuel_type,
                heating_fuel_consumption=request.heating_fuel_consumption,
                units_sold=request.units_sold,
                lifetime_years=request.lifetime_years,
            )],
            org_id="indirect_heating",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_indirect_steam(
        self, request: IndirectSteamRequest
    ) -> CalculationResponse:
        """
        Calculate indirect steam/cooling emissions for sold products.

        Args:
            request: Indirect steam/cooling request.

        Returns:
            CalculationResponse with steam/cooling use-phase emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category=request.product_category,
                steam_kwh_per_year=request.steam_kwh_per_year,
                cooling_kwh_per_year=request.cooling_kwh_per_year,
                steam_ef=request.steam_ef,
                units_sold=request.units_sold,
                country_code=request.country_code,
                lifetime_years=request.lifetime_years,
            )],
            org_id="indirect_steam",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_fuels_sold(
        self, request: FuelsSoldRequest
    ) -> CalculationResponse:
        """
        Calculate emissions from fuels sold to end users.

        Args:
            request: Fuels sold request.

        Returns:
            CalculationResponse with fuel combustion emissions.
        """
        calc_request = CalculationRequest(
            products=[ProductInput(
                product_category="fuels_feedstocks",
                fuel_type=request.fuel_type,
                quantity_sold=request.quantity_sold,
                quantity_unit=request.quantity_unit,
                units_sold=1,
                is_fuel_product=True,
            )],
            org_id="fuels_sold",
            reporting_year=2024,
            tenant_id=request.tenant_id,
        )
        return self.calculate(calc_request)

    def calculate_batch(
        self, request: BatchCalculationRequest
    ) -> BatchResponse:
        """
        Process multiple portfolios in a single batch.

        Args:
            request: Batch request with multiple portfolios.

        Returns:
            BatchResponse with individual results and totals.
        """
        start_time = time.monotonic()
        results: List[CalculationResponse] = []
        errors: List[dict] = []
        total_co2e = 0.0

        for idx, portfolio in enumerate(request.portfolios):
            resp = self.calculate(portfolio)
            results.append(resp)
            if resp.success:
                total_co2e += resp.total_co2e_kg
            else:
                errors.append({
                    "index": idx,
                    "org_id": portfolio.org_id,
                    "error": resp.error,
                })

        elapsed = (time.monotonic() - start_time) * 1000.0

        return BatchResponse(
            success=len(errors) == 0,
            total_portfolios=len(request.portfolios),
            successful=len(results) - len(errors),
            failed=len(errors),
            total_co2e_kg=total_co2e,
            results=results,
            errors=errors,
            processing_time_ms=elapsed,
        )

    def calculate_portfolio(
        self, request: PortfolioAnalysisRequest
    ) -> CalculationResponse:
        """
        Run portfolio-level analysis with hot-spot identification.

        Args:
            request: Portfolio analysis request.

        Returns:
            CalculationResponse with portfolio analysis detail.
        """
        start_time = time.monotonic()
        calc_id = f"usp-portfolio-{uuid4().hex[:12]}"

        try:
            product_dicts = [p.dict(exclude_none=True) for p in request.products]

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.run_portfolio_analysis(product_dicts)
            else:
                raise RuntimeError("Pipeline engine not available")

            pipeline_result = result.get("pipeline_result", {})
            elapsed = (time.monotonic() - start_time) * 1000.0

            response = CalculationResponse(
                success=True,
                calculation_id=calc_id,
                total_co2e_kg=float(result.get("total_co2e", 0)),
                direct_co2e_kg=float(pipeline_result.get("direct_co2e", 0)),
                indirect_co2e_kg=float(pipeline_result.get("indirect_co2e", 0)),
                by_category={
                    k: float(v)
                    for k, v in pipeline_result.get("by_category", {}).items()
                },
                product_count=pipeline_result.get("product_count", 0),
                total_units_sold=result.get("total_units_sold", 0),
                provenance_hash=pipeline_result.get("provenance_hash", ""),
                processing_time_ms=elapsed,
                detail={
                    "hot_spots": result.get("hot_spots", []),
                    "intensity_per_unit_kg_co2e": result.get("intensity_per_unit_kg_co2e"),
                    "category_count": result.get("category_count"),
                },
            )

            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Portfolio analysis %s failed: %s", calc_id, e, exc_info=True)
            return CalculationResponse(
                success=False,
                calculation_id=calc_id,
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed,
            )

    # ========================================================================
    # Public API Methods - Compliance & Analysis
    # ========================================================================

    def check_compliance(
        self, request: ComplianceCheckRequest
    ) -> ComplianceCheckResponse:
        """
        Run compliance checks against specified frameworks.

        Args:
            request: Compliance check request.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start_time = time.monotonic()

        calc_data = self._calculations.get(request.calculation_id)
        if not calc_data:
            elapsed = (time.monotonic() - start_time) * 1000.0
            return ComplianceCheckResponse(
                success=False,
                calculation_id=request.calculation_id,
                overall_status="FAIL",
                overall_score=0.0,
                total_findings=1,
                processing_time_ms=elapsed,
            )

        # Delegate to compliance engine
        if self._compliance_engine is not None:
            try:
                results = self._compliance_engine.check_all(calc_data)
                report = self._compliance_engine.generate_report(results)

                elapsed = (time.monotonic() - start_time) * 1000.0
                return ComplianceCheckResponse(
                    success=True,
                    calculation_id=request.calculation_id,
                    overall_status=report.get("overall_status", "FAIL"),
                    overall_score=report.get("overall_score", 0.0),
                    framework_results=report.get("framework_scores", {}),
                    total_findings=report.get("total_findings", 0),
                    processing_time_ms=elapsed,
                )
            except Exception as e:
                logger.error("Compliance check failed: %s", e, exc_info=True)

        elapsed = (time.monotonic() - start_time) * 1000.0
        return ComplianceCheckResponse(
            success=False,
            calculation_id=request.calculation_id,
            overall_status="FAIL",
            overall_score=0.0,
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Public API Methods - Data Lookups
    # ========================================================================

    def get_product_profile(
        self, request: ProductProfileRequest
    ) -> ProductProfileResponse:
        """
        Get product profile with default parameters.

        Args:
            request: Product profile request.

        Returns:
            ProductProfileResponse with default lifetime, degradation, etc.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            DEFAULT_PRODUCT_LIFETIMES,
            DEFAULT_DEGRADATION_RATES,
            DEFAULT_ENERGY_PROFILES,
            CATEGORY_EMISSION_TYPE,
        )

        category = request.product_category.lower()
        lifetime = DEFAULT_PRODUCT_LIFETIMES.get(category, 10)
        degradation = DEFAULT_DEGRADATION_RATES.get(category, Decimal("0.00"))
        emission_type = CATEGORY_EMISSION_TYPE.get(category)
        energy = DEFAULT_ENERGY_PROFILES.get(
            request.product_name.lower() if request.product_name else "",
        )

        return ProductProfileResponse(
            success=True,
            product_category=category,
            default_lifetime_years=lifetime,
            default_degradation_rate=float(degradation),
            emission_type=emission_type.value if emission_type else "indirect",
            default_energy_kwh=float(energy) if energy else None,
        )

    def get_fuel_ef(self, fuel_type: str) -> FuelEFResponse:
        """
        Get emission factors for a fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            FuelEFResponse with emission factor values.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            FUEL_EMISSION_FACTORS,
        )

        ef_data = FUEL_EMISSION_FACTORS.get(fuel_type.lower())
        if ef_data:
            return FuelEFResponse(
                success=True,
                fuel_type=fuel_type,
                emission_factors={k: float(v) for k, v in ef_data.items()},
                source="DEFRA_2024",
            )
        return FuelEFResponse(
            success=False,
            fuel_type=fuel_type,
            source="not_found",
        )

    def get_refrigerant_gwp(self, refrigerant_type: str) -> RefrigerantGWPResponse:
        """
        Get GWP for a refrigerant type.

        Args:
            refrigerant_type: Refrigerant identifier (e.g., R-134a).

        Returns:
            RefrigerantGWPResponse with GWP value.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            REFRIGERANT_GWP,
        )

        gwp = REFRIGERANT_GWP.get(refrigerant_type)
        if gwp is not None:
            return RefrigerantGWPResponse(
                success=True,
                refrigerant_type=refrigerant_type,
                gwp=gwp,
                source="IPCC_AR5",
            )
        return RefrigerantGWPResponse(
            success=False,
            refrigerant_type=refrigerant_type,
            source="not_found",
        )

    def get_grid_ef(self, country_code: str) -> GridEFResponse:
        """
        Get grid electricity emission factor for a country/region.

        Args:
            country_code: ISO country code or eGRID subregion.

        Returns:
            GridEFResponse with grid emission factor.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            GRID_EMISSION_FACTORS,
        )

        ef = GRID_EMISSION_FACTORS.get(country_code.upper())
        if ef is not None:
            return GridEFResponse(
                success=True,
                country_code=country_code.upper(),
                grid_ef_kg_co2e_per_kwh=float(ef),
                source="IEA_eGRID_2024",
            )
        return GridEFResponse(
            success=False,
            country_code=country_code,
            source="not_found",
        )

    def get_lifetime_estimate(
        self, product_category: str
    ) -> LifetimeEstimateResponse:
        """
        Get default lifetime estimate for a product category.

        Args:
            product_category: Product category identifier.

        Returns:
            LifetimeEstimateResponse with lifetime and degradation rate.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            DEFAULT_PRODUCT_LIFETIMES,
            DEFAULT_DEGRADATION_RATES,
        )

        category = product_category.lower()
        lifetime = DEFAULT_PRODUCT_LIFETIMES.get(category)
        degradation = DEFAULT_DEGRADATION_RATES.get(category)

        if lifetime is not None:
            return LifetimeEstimateResponse(
                success=True,
                product_category=category,
                lifetime_years=lifetime,
                degradation_rate=float(degradation or 0),
                source="industry_average",
            )
        return LifetimeEstimateResponse(
            success=False,
            product_category=category,
        )

    def get_degradation_rate(
        self, product_category: str
    ) -> DegradationRateResponse:
        """
        Get default degradation rate for a product category.

        Args:
            product_category: Product category identifier.

        Returns:
            DegradationRateResponse with annual degradation rate.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            DEFAULT_DEGRADATION_RATES,
        )

        category = product_category.lower()
        rate = DEFAULT_DEGRADATION_RATES.get(category)

        return DegradationRateResponse(
            success=rate is not None,
            product_category=category,
            degradation_rate=float(rate) if rate is not None else 0.0,
        )

    def get_all_categories(self) -> CategoryListResponse:
        """
        Get all supported product use categories with metadata.

        Returns:
            CategoryListResponse with category list.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            DEFAULT_PRODUCT_LIFETIMES,
            DEFAULT_DEGRADATION_RATES,
            CATEGORY_EMISSION_TYPE,
            ProductUseCategory,
        )

        categories: List[Dict[str, Any]] = []
        for cat in ProductUseCategory:
            categories.append({
                "category": cat.value,
                "display_name": cat.value.replace("_", " ").title(),
                "default_lifetime_years": DEFAULT_PRODUCT_LIFETIMES.get(cat.value, 10),
                "default_degradation_rate": float(
                    DEFAULT_DEGRADATION_RATES.get(cat.value, Decimal("0"))
                ),
                "emission_type": CATEGORY_EMISSION_TYPE.get(
                    cat.value, "indirect"
                ).value if hasattr(
                    CATEGORY_EMISSION_TYPE.get(cat.value, "indirect"), "value"
                ) else str(CATEGORY_EMISSION_TYPE.get(cat.value, "indirect")),
            })

        return CategoryListResponse(
            success=True,
            categories=categories,
            total_count=len(categories),
        )

    def get_energy_profiles(self) -> EnergyProfilesResponse:
        """
        Get default energy consumption profiles for common products.

        Returns:
            EnergyProfilesResponse with product -> kWh/year mapping.
        """
        from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
            DEFAULT_ENERGY_PROFILES,
        )

        return EnergyProfilesResponse(
            success=True,
            profiles={k: float(v) for k, v in DEFAULT_ENERGY_PROFILES.items()},
            total_count=len(DEFAULT_ENERGY_PROFILES),
        )

    def get_provenance(self, calculation_id: str) -> ProvenanceResponse:
        """
        Get provenance chain for a calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            ProvenanceResponse with chain entries and integrity status.
        """
        calc = self._calculations.get(calculation_id)
        if calc:
            detail = calc.get("detail", {})
            return ProvenanceResponse(
                success=True,
                calculation_id=calculation_id,
                provenance_hash=calc.get("provenance_hash", ""),
                chain=detail.get("provenance_chain", []) if isinstance(detail, dict) else [],
                is_valid=True,
            )
        return ProvenanceResponse(
            success=False,
            calculation_id=calculation_id,
            provenance_hash="",
            chain=[],
            is_valid=False,
        )

    def get_aggregations(
        self, request: AggregationRequest
    ) -> AggregationResponse:
        """
        Get aggregated emissions for a reporting year.

        Args:
            request: Aggregation request with year and group_by.

        Returns:
            AggregationResponse with multi-dimensional breakdown.
        """
        start_time = time.monotonic()

        by_category: Dict[str, float] = {}
        by_emission_type: Dict[str, float] = {}
        by_method: Dict[str, float] = {}
        total = 0.0

        for calc in self._calculations.values():
            co2e = calc.get("total_co2e_kg", 0.0)
            total += co2e

            for cat, val in calc.get("by_category", {}).items():
                by_category[cat] = by_category.get(cat, 0.0) + float(val)

            for et, val in calc.get("by_emission_type", {}).items():
                by_emission_type[et] = by_emission_type.get(et, 0.0) + float(val)

            for m, val in calc.get("by_method", {}).items():
                by_method[m] = by_method.get(m, 0.0) + float(val)

        elapsed = (time.monotonic() - start_time) * 1000.0

        return AggregationResponse(
            success=True,
            total_co2e_kg=total,
            by_category=by_category,
            by_emission_type=by_emission_type,
            by_method=by_method,
            reporting_year=request.reporting_year,
            processing_time_ms=elapsed,
        )

    # ========================================================================
    # Health, Version, and Config
    # ========================================================================

    def health_check(self) -> HealthResponse:
        """
        Perform service health check.

        Returns:
            HealthResponse with engine statuses and uptime.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = {
            "database": self._database_engine is not None,
            "direct": self._direct_engine is not None,
            "indirect": self._indirect_engine is not None,
            "fuels": self._fuels_engine is not None,
            "lifetime": self._lifetime_engine is not None,
            "compliance": self._compliance_engine is not None,
            "pipeline": self._pipeline_engine is not None,
        }

        all_healthy = all(engines_status.values())
        any_healthy = any(engines_status.values())

        if all_healthy:
            status = "healthy"
        elif any_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            version=SERVICE_VERSION,
            engines_status=engines_status,
            uptime_seconds=uptime,
        )

    def get_version(self) -> VersionResponse:
        """
        Get service version information.

        Returns:
            VersionResponse with version, agent ID, and API prefix.
        """
        return VersionResponse(
            version=SERVICE_VERSION,
            agent_id=AGENT_ID,
            agent_component=AGENT_COMPONENT,
            api_prefix=API_PREFIX,
        )

    def get_config(self) -> ConfigResponse:
        """
        Get service configuration.

        Returns:
            ConfigResponse with configuration values.
        """
        return ConfigResponse(
            success=True,
            config={
                "service_version": SERVICE_VERSION,
                "agent_id": AGENT_ID,
                "agent_component": AGENT_COMPONENT,
                "api_prefix": API_PREFIX,
                "table_prefix": TABLE_PREFIX,
                "metrics_prefix": METRICS_PREFIX,
                "engines": {
                    "database": self._database_engine is not None,
                    "direct": self._direct_engine is not None,
                    "indirect": self._indirect_engine is not None,
                    "fuels": self._fuels_engine is not None,
                    "lifetime": self._lifetime_engine is not None,
                    "compliance": self._compliance_engine is not None,
                    "pipeline": self._pipeline_engine is not None,
                },
                "product_categories": [
                    "vehicles", "appliances", "hvac", "lighting",
                    "it_equipment", "industrial_equipment",
                    "fuels_feedstocks", "building_products",
                    "consumer_products", "medical_devices",
                ],
                "calculation_methods": [
                    "direct_fuel_combustion",
                    "direct_refrigerant_leakage",
                    "direct_chemical_release",
                    "indirect_electricity",
                    "indirect_heating_fuel",
                    "indirect_steam_cooling",
                    "fuels_sold",
                    "feedstocks_sold",
                ],
                "compliance_frameworks": [
                    "ghg_protocol", "iso_14064", "csrd_esrs",
                    "cdp", "sbti", "sb_253", "gri",
                ],
                "double_counting_rules": [
                    "DC-USP-001", "DC-USP-002", "DC-USP-003",
                    "DC-USP-004", "DC-USP-005", "DC-USP-006",
                    "DC-USP-007", "DC-USP-008",
                ],
            },
        )


# ============================================================================
# Module-Level Helpers
# ============================================================================


def get_service() -> UseOfSoldProductsService:
    """
    Get singleton UseOfSoldProductsService instance.

    Thread-safe via double-checked locking.

    Returns:
        UseOfSoldProductsService singleton instance.

    Example:
        >>> service = get_service()
        >>> response = service.health_check()
        >>> response.status
        'healthy'
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = UseOfSoldProductsService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for use-of-sold-products endpoints.

    Returns:
        FastAPI APIRouter instance.

    Example:
        >>> router = get_router()
        >>> app.include_router(router, prefix="/api/v1/use-of-sold-products")
    """
    from greenlang.use_of_sold_products.api.router import router
    return router


def reset_service() -> None:
    """
    Reset the service singleton (for testing only).

    Thread-safe via lock.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("UseOfSoldProductsService singleton reset")
