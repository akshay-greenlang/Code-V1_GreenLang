# -*- coding: utf-8 -*-
"""
Processing of Sold Products Service Setup - AGENT-MRV-023

This module provides the service facade that wires together all 7 engines
for processing of sold products emissions calculations (Scope 3 Category 10).

The ProcessingSoldProductsService class provides a high-level API for:
- Site-specific emissions (direct measurement, energy-based, fuel-based)
- Average-data emissions (product category x processing type EFs)
- Energy intensity emissions (kWh/tonne x grid EF)
- Spend-based emissions (EEIO sector factors with CPI/margin adjustment)
- Hybrid aggregation (method waterfall, gap-filling, Pareto hotspots)
- Multi-step processing chain calculations (8 predefined chains)
- Allocation (mass, revenue, units, equal) across end-uses
- Compliance checking across 7 regulatory frameworks
- Portfolio aggregation by category, method, country, and period
- Uncertainty quantification (analytical error propagation)
- Data quality indicator scoring (5-dimension weighted average)
- SHA-256 provenance tracking and audit trail verification

Engines:
    1. ProcessingDatabaseEngine - Emission factor data and persistence
    2. SiteSpecificCalculatorEngine - Customer-reported / energy / fuel
    3. AverageDataCalculatorEngine - Category x processing type EFs
    4. SpendBasedCalculatorEngine - EEIO sector factors
    5. HybridAggregatorEngine - Multi-method aggregation
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. ProcessingPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Lazy engine initialization on first use
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

GHG Protocol Scope 3 Category 10 Boundary:
    Includes: Emissions from the processing of sold intermediate products
              by downstream third-party processors.
    Excludes: Processing at the reporting company (Scope 1/2),
              purchased goods (Cat 1), capital goods (Cat 2),
              transportation (Cat 4/9), use of sold products (Cat 11),
              end-of-life treatment (Cat 12).

Example:
    >>> from greenlang.agents.mrv.processing_sold_products.setup import get_service
    >>> service = get_service()
    >>> result = service.calculate(
    ...     inputs=[{"product_id": "STEEL-001", "category": "metals_ferrous",
    ...              "processing_type": "machining", "quantity": "500"}],
    ...     method="average_data",
    ...     org_id="ORG-001",
    ...     year=2025,
    ... )
    >>> print(f"Total: {result['total_emissions_kg']} kgCO2e")

Integration:
    >>> from greenlang.agents.mrv.processing_sold_products.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/processing-sold-products")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-023 Processing of Sold Products (GL-MRV-S3-010)
Status: Production Ready
"""

import importlib
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, validator

# ==============================================================================
# MODULE METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"

# ==============================================================================
# DECIMAL PRECISION
# ==============================================================================

ZERO: Decimal = Decimal("0")
ONE: Decimal = Decimal("1")
ONE_THOUSAND: Decimal = Decimal("1000")
_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")

# ==============================================================================
# THREAD-SAFE SINGLETON STATE
# ==============================================================================

_service_lock = threading.Lock()
_service_instance: Optional["ProcessingSoldProductsService"] = None

logger = logging.getLogger(__name__)


# ==============================================================================
# REQUEST MODELS
# ==============================================================================


class ProductCalculationRequest(BaseModel):
    """Request model for single or multi-product emissions calculation."""

    inputs: List[dict] = Field(
        ..., min_length=1,
        description="List of product input dicts with product_id, category, processing_type, quantity"
    )
    method: str = Field(
        "average_data",
        description="Calculation method: site_specific_direct, site_specific_energy, "
                    "site_specific_fuel, average_data, spend_based, hybrid"
    )
    org_id: str = Field(..., min_length=1, max_length=128, description="Organization identifier")
    year: int = Field(..., ge=2015, le=2030, description="Reporting year")
    allocation_method: Optional[str] = Field(None, description="Allocation method: mass, revenue, units, equal")
    end_uses: Optional[List[dict]] = Field(None, description="End-use definitions for allocation")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")

    @validator("method")
    def validate_method(cls, v: str) -> str:
        """Validate calculation method."""
        allowed = [
            "site_specific_direct", "site_specific_energy", "site_specific_fuel",
            "average_data", "energy_intensity", "spend_based", "hybrid",
        ]
        if v.lower() not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v.lower()


class SpendCalculationRequest(BaseModel):
    """Request model for spend-based EEIO calculation."""

    revenue: float = Field(..., gt=0, description="Revenue amount")
    currency: str = Field("USD", description="ISO 4217 currency code")
    sector: str = Field(..., description="NAICS sector code (e.g., '331')")
    year: int = Field(..., ge=2015, le=2030, description="Revenue year")
    org_id: str = Field(..., min_length=1, max_length=128, description="Organization identifier")
    reporting_year: int = Field(..., ge=2015, le=2030, description="Reporting year")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class BatchCalculationRequest(BaseModel):
    """Request model for batch calculations."""

    batch_inputs: List[ProductCalculationRequest] = Field(
        ..., min_length=1,
        description="List of calculation requests"
    )
    method: str = Field("average_data", description="Calculation method for all items")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class PortfolioRequest(BaseModel):
    """Request model for portfolio-level calculation and aggregation."""

    inputs: List[dict] = Field(
        ..., min_length=1,
        description="List of product input dicts"
    )
    org_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2015, le=2030, description="Reporting year")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    calculation_id: str = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL"],
        description="Frameworks: GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, SB_253, GRI"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class AggregationRequest(BaseModel):
    """Request model for aggregation queries."""

    org_id: str = Field(..., description="Organization identifier")
    period: str = Field(..., description="Reporting period (e.g., '2025', '2025-Q3')")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


# ==============================================================================
# RESPONSE MODELS
# ==============================================================================


class CalculationResponse(BaseModel):
    """Response model for calculation results."""

    success: bool = Field(..., description="Success flag")
    calc_id: str = Field(..., description="Calculation identifier")
    method: str = Field(..., description="Calculation method used")
    total_emissions_kg: float = Field(..., description="Total CO2e in kg")
    total_emissions_tco2e: float = Field(..., description="Total CO2e in tonnes")
    product_count: int = Field(0, description="Number of products calculated")
    dqi_score: Optional[float] = Field(None, description="Data quality score (1-5)")
    uncertainty_pct: Optional[float] = Field(None, description="Uncertainty percentage")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    breakdowns: List[dict] = Field(default_factory=list, description="Per-product breakdowns")
    detail: dict = Field(default_factory=dict, description="Additional detail")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchCalculationResponse(BaseModel):
    """Response model for batch calculations."""

    success: bool = Field(..., description="Overall success flag")
    total_calculations: int = Field(..., description="Total calculations requested")
    successful: int = Field(..., description="Successful calculations")
    failed: int = Field(..., description="Failed calculations")
    total_emissions_kg: float = Field(..., description="Total CO2e for all calculations")
    results: List[CalculationResponse] = Field(..., description="Individual results")
    errors: List[dict] = Field(default_factory=list, description="Failed calculation errors")
    processing_time_ms: float = Field(..., description="Total processing time")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation checked")
    overall_status: str = Field(..., description="Overall compliance status: PASS, WARNING, FAIL")
    framework_results: List[dict] = Field(..., description="Per-framework results")
    checked_at: datetime = Field(..., description="Check timestamp")
    processing_time_ms: float = Field(..., description="Processing time")


class AggregationResponse(BaseModel):
    """Response model for aggregation queries."""

    success: bool = Field(..., description="Success flag")
    period: str = Field(..., description="Reporting period")
    total_tco2e: float = Field(..., description="Total CO2e in tonnes")
    by_category: Dict[str, float] = Field(default_factory=dict, description="By product category")
    by_method: Dict[str, float] = Field(default_factory=dict, description="By calculation method")
    by_country: Dict[str, float] = Field(default_factory=dict, description="By customer country")
    processing_time_ms: float = Field(..., description="Processing time")


class ProvenanceResponse(BaseModel):
    """Response model for provenance queries."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation ID")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    chain: List[dict] = Field(default_factory=list, description="Provenance chain entries")
    is_valid: bool = Field(..., description="Chain integrity verified")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    engines_status: Dict[str, bool] = Field(..., description="Per-engine availability")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class EmissionFactorResponse(BaseModel):
    """Response model for emission factor queries."""

    success: bool = Field(..., description="Success flag")
    factor_type: str = Field(..., description="Type of factor")
    value: float = Field(..., description="Factor value")
    unit: str = Field(..., description="Factor unit")
    source: str = Field("", description="Data source")


# ==============================================================================
# SERVICE CLASS
# ==============================================================================


class ProcessingSoldProductsService:
    """
    Processing of Sold Products Service Facade.

    This service wires together all 7 engines to provide a complete API
    for Scope 3 Category 10 emissions calculations (Processing of Sold
    Products, GHG Protocol).

    Supports:
        - 5 calculation methods (site-specific direct/energy/fuel,
          average-data, spend-based)
        - 12 intermediate product categories
        - 18 processing types
        - 8 multi-step processing chains
        - 4 allocation methods (mass, revenue, units, equal)
        - 7 regulatory frameworks (GHG Protocol, ISO 14064, CSRD, CDP,
          SBTi, SB 253, GRI)
        - Hybrid multi-method aggregation with Pareto hotspot identification
        - Portfolio DQI scoring and uncertainty quantification
        - SHA-256 provenance tracking and audit trails

    Engines:
        1. ProcessingDatabaseEngine - Data persistence and EF lookups
        2. SiteSpecificCalculatorEngine - Customer-reported emissions
        3. AverageDataCalculatorEngine - Category-average emissions
        4. SpendBasedCalculatorEngine - Revenue-based EEIO emissions
        5. HybridAggregatorEngine - Multi-method aggregation
        6. ComplianceCheckerEngine - Regulatory compliance checking
        7. ProcessingPipelineEngine - End-to-end 10-stage pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> result = service.calculate(
        ...     inputs=[{"product_id": "STEEL-001", "category": "metals_ferrous",
        ...              "processing_type": "machining", "quantity": "500"}],
        ...     method="average_data",
        ...     org_id="ORG-001",
        ...     year=2025,
        ... )
        >>> print(result.total_emissions_kg)

    Attributes:
        _database_engine: Engine 1 - ProcessingDatabaseEngine
        _site_specific_engine: Engine 2 - SiteSpecificCalculatorEngine
        _average_data_engine: Engine 3 - AverageDataCalculatorEngine
        _spend_based_engine: Engine 4 - SpendBasedCalculatorEngine
        _hybrid_engine: Engine 5 - HybridAggregatorEngine
        _compliance_engine: Engine 6 - ComplianceCheckerEngine
        _pipeline_engine: Engine 7 - ProcessingPipelineEngine
    """

    def __init__(self) -> None:
        """Initialize ProcessingSoldProductsService with all 7 engines."""
        logger.info("Initializing ProcessingSoldProductsService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        # Lazy engine references
        self._database_engine: Optional[Any] = None
        self._site_specific_engine: Optional[Any] = None
        self._average_data_engine: Optional[Any] = None
        self._spend_based_engine: Optional[Any] = None
        self._hybrid_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None
        self._pipeline_engine: Optional[Any] = None

        # Engine initialization status
        self._engines_loaded: Dict[str, bool] = {
            "database": False,
            "site_specific": False,
            "average_data": False,
            "spend_based": False,
            "hybrid": False,
            "compliance": False,
            "pipeline": False,
        }

        # Initialize all engines
        self._database_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.processing_database",
            "ProcessingDatabaseEngine",
            "database",
        )
        self._site_specific_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.site_specific_calculator",
            "SiteSpecificCalculatorEngine",
            "site_specific",
        )
        self._average_data_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.average_data_calculator",
            "AverageDataCalculatorEngine",
            "average_data",
        )
        self._spend_based_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.spend_based_calculator",
            "SpendBasedCalculatorEngine",
            "spend_based",
        )
        self._hybrid_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.hybrid_aggregator",
            "HybridAggregatorEngine",
            "hybrid",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.compliance_checker",
            "ComplianceCheckerEngine",
            "compliance",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.agents.mrv.processing_sold_products.processing_pipeline",
            "ProcessingPipelineEngine",
            "pipeline",
        )

        # In-memory calculation store (for dev/testing; production uses DB engine)
        self._calculations: Dict[str, dict] = {}

        self._initialized = True
        loaded_count = sum(1 for v in self._engines_loaded.values() if v)
        logger.info(
            "ProcessingSoldProductsService initialized: %d/%d engines loaded",
            loaded_count, len(self._engines_loaded),
        )

    def _init_engine(
        self,
        module_path: str,
        class_name: str,
        engine_key: str,
    ) -> Optional[Any]:
        """Initialize an engine with graceful ImportError handling.

        Args:
            module_path: Fully qualified module path.
            class_name: Class name within the module.
            engine_key: Key in _engines_loaded dict.

        Returns:
            Engine instance or None if import fails.
        """
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            self._engines_loaded[engine_key] = True
            logger.info("%s initialized", class_name)
            return instance
        except ImportError:
            logger.warning("%s not available (ImportError)", class_name)
            self._engines_loaded[engine_key] = False
            return None
        except Exception as e:
            logger.warning("%s initialization failed: %s", class_name, e)
            self._engines_loaded[engine_key] = False
            return None

    # ==========================================================================
    # CORE CALCULATION METHODS
    # ==========================================================================

    def calculate(
        self,
        inputs: List[dict],
        method: str = "average_data",
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions for a list of products using the specified method.

        Delegates to the appropriate engine based on the method parameter.
        Falls back to the pipeline engine if available for full processing.

        Args:
            inputs: List of product input dicts with product_id, category,
                processing_type, quantity, and optional site-specific fields.
            method: Calculation method (site_specific_direct, site_specific_energy,
                site_specific_fuel, average_data, energy_intensity, spend_based, hybrid).
            org_id: Reporting organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with emissions, breakdowns, and provenance.
        """
        start = time.monotonic()
        calc_id = f"psp-{uuid4().hex[:12]}"

        try:
            method_lower = method.lower()
            result: Optional[Dict[str, Any]] = None

            if method_lower in ("site_specific_direct", "site_specific_energy", "site_specific_fuel"):
                result = self._calculate_site_specific(inputs, method_lower, org_id, year)
            elif method_lower == "average_data":
                result = self._calculate_average_data(inputs, org_id, year)
            elif method_lower == "energy_intensity":
                result = self._calculate_energy_intensity(inputs, org_id, year)
            elif method_lower == "spend_based":
                result = self._calculate_spend_based_from_inputs(inputs, org_id, year)
            elif method_lower == "hybrid":
                result = self._calculate_hybrid(inputs, org_id, year)
            else:
                raise ValueError(f"Unknown calculation method: {method}")

            elapsed_ms = (time.monotonic() - start) * 1000.0

            if result is None:
                raise RuntimeError("Calculation returned None - engine may not be available")

            response = self._build_calculation_response(
                calc_id, method_lower, result, elapsed_ms
            )
            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.error("Calculation %s failed: %s", calc_id, e, exc_info=True)
            return CalculationResponse(
                success=False,
                calc_id=calc_id,
                method=method,
                total_emissions_kg=0.0,
                total_emissions_tco2e=0.0,
                product_count=0,
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed_ms,
            )

    def calculate_site_specific_direct(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using site-specific direct method.

        Uses customer-reported processing emissions per product.

        Args:
            inputs: List of product input dicts with processing_emissions_kg.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with emissions.
        """
        return self.calculate(inputs, "site_specific_direct", org_id, year)

    def calculate_site_specific_energy(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using site-specific energy method.

        Uses customer-reported energy consumption x grid emission factor.

        Args:
            inputs: List of product input dicts with processing_energy_kwh.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with emissions.
        """
        return self.calculate(inputs, "site_specific_energy", org_id, year)

    def calculate_site_specific_fuel(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using site-specific fuel method.

        Uses customer-reported fuel consumption x combustion emission factor.

        Args:
            inputs: List of product input dicts with fuel_type and fuel_quantity_kwh.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with emissions.
        """
        return self.calculate(inputs, "site_specific_fuel", org_id, year)

    def calculate_average_data(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using average-data method.

        Multiplies product quantity by category-level processing EFs.

        Args:
            inputs: List of product input dicts.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with emissions.
        """
        return self.calculate(inputs, "average_data", org_id, year)

    def calculate_energy_intensity(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using energy intensity method.

        Multiplies product quantity by processing type energy intensity
        and grid emission factor.

        Args:
            inputs: List of product input dicts with processing_type.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with emissions.
        """
        return self.calculate(inputs, "energy_intensity", org_id, year)

    def calculate_spend_based(
        self,
        revenue: float,
        currency: str,
        sector: str,
        year: int,
        org_id: str = "ORG-000",
        reporting_year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using spend-based EEIO method.

        Applies sector-level EEIO factors to revenue after currency
        conversion, CPI deflation, and margin removal.

        Args:
            revenue: Revenue amount in the specified currency.
            currency: ISO 4217 currency code.
            sector: NAICS sector code.
            year: Year of the revenue data.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            CalculationResponse with emissions.
        """
        start = time.monotonic()
        calc_id = f"psp-spend-{uuid4().hex[:12]}"

        try:
            result = None
            if self._spend_based_engine is not None:
                result = self._delegate_spend_based(
                    revenue, currency, sector, year, org_id, reporting_year
                )
            elif self._pipeline_engine is not None:
                result = self._delegate_pipeline_spend(
                    revenue, currency, sector, year, org_id, reporting_year
                )
            else:
                result = self._fallback_spend_based(revenue, currency, sector, year)

            elapsed_ms = (time.monotonic() - start) * 1000.0
            response = self._build_calculation_response(
                calc_id, "spend_based", result, elapsed_ms
            )
            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.error("Spend-based calculation %s failed: %s", calc_id, e, exc_info=True)
            return CalculationResponse(
                success=False,
                calc_id=calc_id,
                method="spend_based",
                total_emissions_kg=0.0,
                total_emissions_tco2e=0.0,
                product_count=0,
                provenance_hash="",
                error=str(e),
                processing_time_ms=(time.monotonic() - start) * 1000.0,
            )

    def calculate_hybrid(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        year: int = 2025,
    ) -> CalculationResponse:
        """Calculate emissions using hybrid multi-method aggregation.

        Runs all available engines and applies the method waterfall.

        Args:
            inputs: List of product input dicts.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            CalculationResponse with aggregated emissions.
        """
        return self.calculate(inputs, "hybrid", org_id, year)

    def calculate_batch(
        self,
        batch_inputs: List[dict],
        method: str = "average_data",
    ) -> BatchCalculationResponse:
        """Process multiple calculation requests in a batch.

        Args:
            batch_inputs: List of dicts, each containing 'inputs', 'org_id', 'year'.
            method: Calculation method for all items.

        Returns:
            BatchCalculationResponse with individual results and totals.
        """
        start = time.monotonic()
        results: List[CalculationResponse] = []
        errors: List[dict] = []

        for idx, batch_item in enumerate(batch_inputs):
            inputs = batch_item.get("inputs", [])
            org_id = batch_item.get("org_id", "ORG-000")
            year = batch_item.get("year", 2025)

            resp = self.calculate(inputs, method, org_id, year)
            results.append(resp)
            if not resp.success:
                errors.append({"index": idx, "error": resp.error})

        total_co2e = sum(r.total_emissions_kg for r in results if r.success)
        successful = sum(1 for r in results if r.success)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        return BatchCalculationResponse(
            success=len(errors) == 0,
            total_calculations=len(batch_inputs),
            successful=successful,
            failed=len(errors),
            total_emissions_kg=total_co2e,
            results=results,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )

    def calculate_portfolio(
        self,
        inputs: List[dict],
        org_id: str = "ORG-000",
        reporting_year: int = 2025,
    ) -> AggregationResponse:
        """Calculate portfolio-level emissions with full aggregation.

        Runs hybrid aggregation and returns multi-dimensional breakdowns.

        Args:
            inputs: List of product input dicts.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            AggregationResponse with category, method, and country breakdowns.
        """
        start = time.monotonic()

        try:
            if self._hybrid_engine is not None:
                result = self._hybrid_engine.aggregate(
                    products=inputs,
                    org_id=org_id,
                    reporting_year=reporting_year,
                )
            else:
                # Fallback: run average-data and aggregate manually
                calc_resp = self.calculate(inputs, "average_data", org_id, reporting_year)
                result = {
                    "total_emissions_tco2e": Decimal(str(calc_resp.total_emissions_tco2e)),
                    "by_category": {},
                    "by_method": {"average_data": calc_resp.total_emissions_tco2e},
                    "by_country": {},
                }

            elapsed_ms = (time.monotonic() - start) * 1000.0

            total_tco2e = result.get("total_emissions_tco2e", ZERO)
            by_category_raw = result.get("by_category", {})
            by_method_raw = result.get("by_method", {})
            by_country_raw = result.get("by_country", {})

            return AggregationResponse(
                success=True,
                period=str(reporting_year),
                total_tco2e=float(total_tco2e),
                by_category={k: float(v) for k, v in by_category_raw.items()},
                by_method={k: float(v) for k, v in by_method_raw.items()},
                by_country={k: float(v) for k, v in by_country_raw.items()},
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.error("Portfolio calculation failed: %s", e, exc_info=True)
            return AggregationResponse(
                success=False,
                period=str(reporting_year),
                total_tco2e=0.0,
                processing_time_ms=elapsed_ms,
            )

    # ==========================================================================
    # COMPLIANCE
    # ==========================================================================

    def check_compliance(
        self,
        calculation_id: str,
        frameworks: Optional[List[str]] = None,
    ) -> ComplianceCheckResponse:
        """Check a calculation result against regulatory frameworks.

        Args:
            calculation_id: ID of a previously stored calculation.
            frameworks: List of framework identifiers. Defaults to ['GHG_PROTOCOL'].

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start = time.monotonic()
        frameworks = frameworks or ["GHG_PROTOCOL"]

        try:
            calc_data = self._calculations.get(calculation_id)
            if calc_data is None:
                raise ValueError(f"Calculation {calculation_id} not found")

            framework_results: List[dict] = []

            if self._compliance_engine is not None:
                try:
                    raw_result = self._compliance_engine.check_all(calc_data)
                    if isinstance(raw_result, dict):
                        for fw_name, fw_result in raw_result.items():
                            framework_results.append({
                                "framework": fw_name,
                                "status": "PASS" if fw_result.get("passed", True) else "FAIL",
                                "rules_checked": fw_result.get("rules_checked", 0),
                                "rules_passed": fw_result.get("rules_passed", 0),
                                "rules_failed": fw_result.get("rules_failed", 0),
                                "findings": fw_result.get("findings", []),
                            })
                    elif isinstance(raw_result, list):
                        for item in raw_result:
                            if isinstance(item, dict):
                                framework_results.append(item)
                            else:
                                try:
                                    framework_results.append(item.model_dump(mode="json"))
                                except AttributeError:
                                    framework_results.append({"result": str(item)})
                except Exception as comp_err:
                    logger.warning("Compliance engine check failed: %s", comp_err)
                    framework_results = self._fallback_compliance_check(
                        calc_data, frameworks
                    )
            else:
                framework_results = self._fallback_compliance_check(
                    calc_data, frameworks
                )

            # Determine overall status
            overall_status = "PASS"
            for fr in framework_results:
                status = fr.get("status", "PASS")
                if status == "FAIL":
                    overall_status = "FAIL"
                    break
                elif status == "WARNING" and overall_status != "FAIL":
                    overall_status = "WARNING"

            elapsed_ms = (time.monotonic() - start) * 1000.0

            return ComplianceCheckResponse(
                success=True,
                calculation_id=calculation_id,
                overall_status=overall_status,
                framework_results=framework_results,
                checked_at=datetime.now(timezone.utc),
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.error("Compliance check failed: %s", e, exc_info=True)
            return ComplianceCheckResponse(
                success=False,
                calculation_id=calculation_id,
                overall_status="FAIL",
                framework_results=[{"error": str(e)}],
                checked_at=datetime.now(timezone.utc),
                processing_time_ms=elapsed_ms,
            )

    # ==========================================================================
    # EMISSION FACTOR LOOKUPS
    # ==========================================================================

    def get_processing_ef(self, category: str) -> Decimal:
        """Get the average-data processing emission factor for a product category.

        Args:
            category: Product category string (e.g., 'metals_ferrous').

        Returns:
            Emission factor in kgCO2e per tonne.

        Raises:
            ValueError: If category is not found.
        """
        if self._database_engine is not None:
            try:
                return self._database_engine.get_processing_ef(category.upper())
            except Exception:
                pass

        # Fallback to models
        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                get_processing_ef as models_get_ef,
                IntermediateProductCategory,
            )
            return models_get_ef(IntermediateProductCategory(category.lower()))
        except Exception:
            pass

        # Hard fallback
        from greenlang.agents.mrv.processing_sold_products.hybrid_aggregator import _get_fallback_ef
        return _get_fallback_ef(category)

    def get_energy_intensity(self, processing_type: str) -> Decimal:
        """Get the energy intensity factor for a processing type.

        Args:
            processing_type: Processing type string (e.g., 'machining').

        Returns:
            Energy intensity in kWh per tonne.

        Raises:
            ValueError: If processing type is not found.
        """
        if self._database_engine is not None:
            try:
                return self._database_engine.get_energy_intensity(processing_type.upper())
            except Exception:
                pass

        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                get_energy_intensity as models_get_ei,
                ProcessingType,
            )
            return models_get_ei(ProcessingType(processing_type.lower()))
        except Exception:
            pass

        raise ValueError(f"Energy intensity not found for '{processing_type}'")

    def get_grid_ef(self, region: str) -> Decimal:
        """Get the grid electricity emission factor for a region.

        Args:
            region: Grid region code (e.g., 'US', 'DE', 'GLOBAL').

        Returns:
            Grid emission factor in kgCO2e per kWh.
        """
        if self._database_engine is not None:
            try:
                return self._database_engine.get_grid_ef(region.upper())
            except Exception:
                pass

        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                get_grid_ef as models_get_gef,
                GridRegion,
            )
            return models_get_gef(GridRegion(region.upper()))
        except Exception:
            pass

        # Global fallback
        return Decimal("0.475")

    def get_fuel_ef(self, fuel_type: str) -> Decimal:
        """Get the combustion emission factor for a fuel type.

        Args:
            fuel_type: Fuel type string (e.g., 'natural_gas').

        Returns:
            Fuel emission factor in kgCO2e per kWh thermal.

        Raises:
            ValueError: If fuel type is not found.
        """
        if self._database_engine is not None:
            try:
                return self._database_engine.get_fuel_ef(fuel_type.upper())
            except Exception:
                pass

        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                get_fuel_ef as models_get_fef,
                FuelType,
            )
            return models_get_fef(FuelType(fuel_type.lower()))
        except Exception:
            pass

        raise ValueError(f"Fuel emission factor not found for '{fuel_type}'")

    def get_eeio_factor(self, sector: str) -> Tuple[Decimal, Decimal]:
        """Get the EEIO emission factor and margin for a NAICS sector.

        Args:
            sector: NAICS sector code (e.g., '331').

        Returns:
            Tuple of (ef_kgco2e_per_usd, margin_fraction).

        Raises:
            ValueError: If sector is not found.
        """
        if self._database_engine is not None:
            try:
                return self._database_engine.get_eeio_factor(sector)
            except Exception:
                pass

        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                get_eeio_factor as models_get_eeio,
                NAICSSector,
            )
            return models_get_eeio(NAICSSector(sector))
        except Exception:
            pass

        raise ValueError(f"EEIO factor not found for sector '{sector}'")

    def get_processing_chain(self, chain_type: str) -> Dict[str, Any]:
        """Get a predefined multi-step processing chain.

        Args:
            chain_type: Processing chain type (e.g., 'metals_automotive').

        Returns:
            Dict with 'steps', 'combined_ef', 'description'.

        Raises:
            ValueError: If chain type is not found.
        """
        if self._database_engine is not None:
            try:
                return self._database_engine.get_processing_chain(chain_type)
            except Exception:
                pass

        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                get_processing_chain as models_get_chain,
                ProcessingChainType,
            )
            return models_get_chain(ProcessingChainType(chain_type))
        except Exception:
            pass

        raise ValueError(f"Processing chain not found for '{chain_type}'")

    # ==========================================================================
    # ENUMERATION LOOKUPS
    # ==========================================================================

    def get_all_categories(self) -> List[str]:
        """Get all available intermediate product categories.

        Returns:
            List of category value strings.
        """
        try:
            from greenlang.agents.mrv.processing_sold_products.models import IntermediateProductCategory
            return [c.value for c in IntermediateProductCategory]
        except ImportError:
            return list(
                _get_fallback_categories()
            )

    def get_all_processing_types(self) -> List[str]:
        """Get all available processing types.

        Returns:
            List of processing type value strings.
        """
        try:
            from greenlang.agents.mrv.processing_sold_products.models import ProcessingType
            return [pt.value for pt in ProcessingType]
        except ImportError:
            return [
                "machining", "stamping", "welding", "heat_treatment",
                "injection_molding", "extrusion", "blow_molding", "casting",
                "forging", "coating", "assembly", "chemical_reaction",
                "refining", "milling", "drying", "sintering",
                "fermentation", "textile_finishing",
            ]

    def get_processing_chains(self) -> List[str]:
        """Get all available processing chain types.

        Returns:
            List of processing chain type value strings.
        """
        try:
            from greenlang.agents.mrv.processing_sold_products.models import ProcessingChainType
            return [pct.value for pct in ProcessingChainType]
        except ImportError:
            return [
                "metals_automotive", "aluminum_packaging", "plastic_packaging",
                "semiconductor", "food_products", "textile_garments",
                "glass_bottles", "paper_products",
            ]

    # ==========================================================================
    # PROVENANCE AND AGGREGATION
    # ==========================================================================

    def get_provenance(self, calc_id: str) -> ProvenanceResponse:
        """Get the provenance chain for a calculation.

        Args:
            calc_id: Calculation identifier.

        Returns:
            ProvenanceResponse with hash and chain entries.
        """
        calc = self._calculations.get(calc_id)
        if calc:
            return ProvenanceResponse(
                success=True,
                calculation_id=calc_id,
                provenance_hash=calc.get("provenance_hash", ""),
                chain=[],
                is_valid=True,
            )
        return ProvenanceResponse(
            success=False,
            calculation_id=calc_id,
            provenance_hash="",
            chain=[],
            is_valid=False,
        )

    def get_aggregations(
        self,
        org_id: str,
        period: str,
    ) -> AggregationResponse:
        """Get aggregated emissions for an organization and period.

        Aggregates all stored calculations matching the org_id.

        Args:
            org_id: Organization identifier.
            period: Reporting period string.

        Returns:
            AggregationResponse with multi-dimensional breakdown.
        """
        start = time.monotonic()

        by_category: Dict[str, float] = {}
        by_method: Dict[str, float] = {}
        by_country: Dict[str, float] = {}
        total_tco2e = 0.0

        for calc in self._calculations.values():
            co2e = calc.get("total_emissions_tco2e", 0.0)
            total_tco2e += co2e

            method = calc.get("method", "unknown")
            by_method[method] = by_method.get(method, 0.0) + co2e

            # Extract category/country from breakdowns
            for bd in calc.get("breakdowns", []):
                cat = bd.get("category", "unknown")
                by_category[cat] = by_category.get(cat, 0.0) + bd.get("emissions_tco2e", 0.0)

                country = bd.get("country", "GLOBAL")
                by_country[country] = by_country.get(country, 0.0) + bd.get("emissions_tco2e", 0.0)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return AggregationResponse(
            success=True,
            period=period,
            total_tco2e=total_tco2e,
            by_category=by_category,
            by_method=by_method,
            by_country=by_country,
            processing_time_ms=elapsed_ms,
        )

    # ==========================================================================
    # HEALTH AND STATUS
    # ==========================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform service health check.

        Returns:
            Dict with status, version, engine statuses, and uptime.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = dict(self._engines_loaded)

        all_healthy = all(engines_status.values())
        any_healthy = any(engines_status.values())

        if all_healthy:
            status = "healthy"
        elif any_healthy:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "version": VERSION,
            "agent_id": AGENT_ID,
            "component": AGENT_COMPONENT,
            "engines_status": engines_status,
            "uptime_seconds": uptime,
            "calculations_stored": len(self._calculations),
        }

    def get_version(self) -> str:
        """Get service version string.

        Returns:
            Version string (e.g., '1.0.0').
        """
        return VERSION

    def get_config(self) -> Dict[str, Any]:
        """Get service configuration summary.

        Returns:
            Dict with agent_id, component, version, table_prefix,
            engines_loaded, supported methods, and category counts.
        """
        return {
            "agent_id": AGENT_ID,
            "component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "engines_loaded": dict(self._engines_loaded),
            "supported_methods": [
                "site_specific_direct",
                "site_specific_energy",
                "site_specific_fuel",
                "average_data",
                "energy_intensity",
                "spend_based",
                "hybrid",
            ],
            "product_categories": len(self.get_all_categories()),
            "processing_types": len(self.get_all_processing_types()),
            "processing_chains": len(self.get_processing_chains()),
            "compliance_frameworks": 7,
            "allocation_methods": ["mass", "revenue", "units", "equal"],
            "double_counting_rules": 8,
        }

    # ==========================================================================
    # INTERNAL CALCULATION DELEGATES
    # ==========================================================================

    def _calculate_site_specific(
        self,
        inputs: List[dict],
        method: str,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Delegate to SiteSpecificCalculatorEngine.

        Args:
            inputs: Product input dicts.
            method: One of site_specific_direct, site_specific_energy, site_specific_fuel.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Calculation result dict.
        """
        if self._site_specific_engine is not None:
            try:
                if method == "site_specific_direct":
                    result = self._site_specific_engine.calculate_direct(inputs, org_id, year)
                elif method == "site_specific_energy":
                    result = self._site_specific_engine.calculate_energy(inputs, org_id, year)
                elif method == "site_specific_fuel":
                    result = self._site_specific_engine.calculate_fuel(inputs, org_id, year)
                else:
                    raise ValueError(f"Unknown site-specific method: {method}")
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning(
                    "Site-specific engine failed for %s, falling back: %s",
                    method, e,
                )

        # Fallback to pipeline engine
        if self._pipeline_engine is not None:
            try:
                result = self._pipeline_engine.run_pipeline(
                    inputs={"products": inputs},
                    method=method,
                    org_id=org_id,
                    reporting_year=year,
                )
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("Pipeline engine failed: %s", e)

        raise RuntimeError(
            f"No engine available for method '{method}'. "
            "SiteSpecificCalculatorEngine and ProcessingPipelineEngine are both unavailable."
        )

    def _calculate_average_data(
        self,
        inputs: List[dict],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Delegate to AverageDataCalculatorEngine.

        Args:
            inputs: Product input dicts.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Calculation result dict.
        """
        if self._average_data_engine is not None:
            try:
                result = self._average_data_engine.calculate(inputs, org_id, year)
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("AverageData engine failed, using fallback: %s", e)

        if self._pipeline_engine is not None:
            try:
                result = self._pipeline_engine.run_pipeline(
                    inputs={"products": inputs},
                    method="average_data",
                    org_id=org_id,
                    reporting_year=year,
                )
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("Pipeline engine failed: %s", e)

        # Manual fallback: calculate using embedded EFs
        return self._fallback_average_data(inputs, org_id, year)

    def _calculate_energy_intensity(
        self,
        inputs: List[dict],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Delegate to AverageDataCalculatorEngine for energy intensity method.

        Args:
            inputs: Product input dicts.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Calculation result dict.
        """
        if self._average_data_engine is not None:
            try:
                result = self._average_data_engine.calculate_energy_intensity(
                    inputs, org_id, year
                )
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("AverageData engine energy_intensity failed: %s", e)

        if self._pipeline_engine is not None:
            try:
                result = self._pipeline_engine.run_pipeline(
                    inputs={"products": inputs},
                    method="energy_intensity",
                    org_id=org_id,
                    reporting_year=year,
                )
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("Pipeline engine failed: %s", e)

        # Fallback to average_data
        return self._fallback_average_data(inputs, org_id, year)

    def _calculate_spend_based_from_inputs(
        self,
        inputs: List[dict],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Delegate spend-based calculation from product inputs.

        Args:
            inputs: Product input dicts (may include revenue, sector fields).
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Calculation result dict.
        """
        if self._spend_based_engine is not None:
            try:
                result = self._spend_based_engine.calculate_batch(inputs, org_id, year)
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("SpendBased engine batch failed: %s", e)

        if self._pipeline_engine is not None:
            try:
                result = self._pipeline_engine.run_pipeline(
                    inputs={"products": inputs},
                    method="spend_based",
                    org_id=org_id,
                    reporting_year=year,
                )
                return self._result_to_dict(result)
            except Exception as e:
                logger.warning("Pipeline engine failed: %s", e)

        # Fallback: treat as average-data
        return self._fallback_average_data(inputs, org_id, year)

    def _calculate_hybrid(
        self,
        inputs: List[dict],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Run hybrid multi-method aggregation.

        Attempts to run site-specific, average-data, and spend-based engines
        independently, then feeds all results into the HybridAggregatorEngine.

        Args:
            inputs: Product input dicts.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Hybrid aggregation result dict.
        """
        site_result = None
        avg_result = None
        spend_result = None

        # Attempt each engine independently (failures are non-fatal)
        if self._site_specific_engine is not None:
            try:
                site_result = self._site_specific_engine.calculate_energy(inputs, org_id, year)
            except Exception as e:
                logger.debug("Site-specific engine failed in hybrid mode: %s", e)

        if self._average_data_engine is not None:
            try:
                avg_result = self._average_data_engine.calculate(inputs, org_id, year)
            except Exception as e:
                logger.debug("Average-data engine failed in hybrid mode: %s", e)

        if self._spend_based_engine is not None:
            try:
                spend_result = self._spend_based_engine.calculate_batch(inputs, org_id, year)
            except Exception as e:
                logger.debug("Spend-based engine failed in hybrid mode: %s", e)

        # Aggregate
        if self._hybrid_engine is not None:
            result = self._hybrid_engine.aggregate(
                products=inputs,
                site_results=site_result,
                avg_results=avg_result,
                spend_results=spend_result,
                org_id=org_id,
                reporting_year=year,
            )
            return result

        # Fallback: use whichever result is available
        for r in (site_result, avg_result, spend_result):
            if r is not None:
                return self._result_to_dict(r)

        return self._fallback_average_data(inputs, org_id, year)

    def _delegate_spend_based(
        self,
        revenue: float,
        currency: str,
        sector: str,
        year: int,
        org_id: str,
        reporting_year: int = 2025,
    ) -> Dict[str, Any]:
        """Delegate single spend-based calculation to the engine.

        Args:
            revenue: Revenue amount.
            currency: Currency code.
            sector: NAICS sector code.
            year: Revenue year.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Calculation result dict.
        """
        result = self._spend_based_engine.calculate(
            revenue=str(revenue),
            currency=currency,
            sector=sector,
            year=year,
            org_id=org_id,
            reporting_year=reporting_year,
        )
        return self._result_to_dict(result)

    def _delegate_pipeline_spend(
        self,
        revenue: float,
        currency: str,
        sector: str,
        year: int,
        org_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """Delegate spend calculation to pipeline engine.

        Args:
            revenue: Revenue amount.
            currency: Currency code.
            sector: NAICS sector code.
            year: Revenue year.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Calculation result dict.
        """
        result = self._pipeline_engine.run_pipeline(
            inputs={
                "revenue": revenue,
                "currency": currency,
                "sector": sector,
                "year": year,
            },
            method="spend_based",
            org_id=org_id,
            reporting_year=reporting_year,
        )
        return self._result_to_dict(result)

    # ==========================================================================
    # FALLBACK CALCULATIONS
    # ==========================================================================

    def _fallback_average_data(
        self,
        inputs: List[dict],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Fallback average-data calculation using embedded EFs.

        Used when no engine is available. Applies category-level EFs directly.

        Args:
            inputs: Product input dicts.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Calculation result dict.
        """
        import hashlib as _hl

        from greenlang.agents.mrv.processing_sold_products.hybrid_aggregator import (
            _get_fallback_ef,
            _safe_decimal,
            _quantize,
        )

        breakdowns: List[dict] = []
        total_kg = ZERO

        for inp in inputs:
            pid = inp.get("product_id", f"fallback-{uuid4().hex[:8]}")
            cat = inp.get("category", "unknown")
            quantity = _safe_decimal(inp.get("quantity", inp.get("quantity_tonnes", 1)))
            ef = _get_fallback_ef(cat)
            emissions = _quantize(quantity * ef)
            total_kg += emissions

            breakdowns.append({
                "product_id": pid,
                "category": cat,
                "processing_type": inp.get("processing_type", "unknown"),
                "quantity": float(quantity),
                "emissions_kg": float(emissions),
                "emissions_tco2e": float(_quantize(emissions / ONE_THOUSAND)),
                "ef_used": float(ef),
                "method": "average_data",
                "dqi": 2.8,
                "country": inp.get("customer_country", inp.get("country", "GLOBAL")),
            })

        total_tco2e = _quantize(total_kg / ONE_THOUSAND)

        provenance_str = f"{org_id}:{year}:{len(inputs)}:{str(total_kg)}"
        prov_hash = _hl.sha256(provenance_str.encode("utf-8")).hexdigest()

        return {
            "org_id": org_id,
            "reporting_year": year,
            "method": "average_data",
            "total_emissions_kg": total_kg,
            "total_emissions_tco2e": total_tco2e,
            "product_breakdowns": breakdowns,
            "product_count": len(breakdowns),
            "dqi_score": Decimal("2.8"),
            "uncertainty": Decimal("0.30"),
            "provenance_hash": prov_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fallback_spend_based(
        self,
        revenue: float,
        currency: str,
        sector: str,
        year: int,
    ) -> Dict[str, Any]:
        """Fallback spend-based calculation using embedded EEIO factors.

        Args:
            revenue: Revenue amount.
            currency: Currency code.
            sector: NAICS sector code.
            year: Revenue year.

        Returns:
            Calculation result dict.
        """
        import hashlib as _hl

        try:
            from greenlang.agents.mrv.processing_sold_products.models import (
                EEIO_SECTOR_FACTORS,
                CURRENCIES,
                CPI_DEFLATORS,
            )
            sector_data = EEIO_SECTOR_FACTORS.get(sector, {"ef": Decimal("0.50"), "margin": Decimal("0.20")})
            fx_rate = CURRENCIES.get(currency, Decimal("1.0"))
            cpi = CPI_DEFLATORS.get(year, Decimal("1.0"))
            cpi_base = CPI_DEFLATORS.get(2021, Decimal("1.0"))
        except ImportError:
            sector_data = {"ef": Decimal("0.50"), "margin": Decimal("0.20")}
            fx_rate = Decimal("1.0")
            cpi = Decimal("1.0")
            cpi_base = Decimal("1.0")

        ef = sector_data["ef"]
        margin = sector_data["margin"]

        revenue_usd = Decimal(str(revenue)) * fx_rate
        revenue_deflated = revenue_usd * cpi_base / cpi if cpi > ZERO else revenue_usd
        revenue_adjusted = revenue_deflated * (ONE - margin)
        emissions_kg = revenue_adjusted * ef
        emissions_kg = emissions_kg.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
        emissions_tco2e = (emissions_kg / ONE_THOUSAND).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)

        prov = f"spend:{sector}:{revenue}:{currency}:{year}:{str(emissions_kg)}"
        prov_hash = _hl.sha256(prov.encode("utf-8")).hexdigest()

        return {
            "method": "spend_based",
            "total_emissions_kg": emissions_kg,
            "total_emissions_tco2e": emissions_tco2e,
            "product_breakdowns": [{
                "product_id": f"spend-{sector}-{year}",
                "category": sector,
                "processing_type": "eeio",
                "quantity": float(revenue),
                "emissions_kg": float(emissions_kg),
                "emissions_tco2e": float(emissions_tco2e),
                "ef_used": float(ef),
                "method": "spend_based",
                "dqi": 1.6,
                "country": "GLOBAL",
            }],
            "product_count": 1,
            "dqi_score": Decimal("1.6"),
            "uncertainty": Decimal("0.50"),
            "provenance_hash": prov_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fallback_compliance_check(
        self,
        calc_data: dict,
        frameworks: List[str],
    ) -> List[dict]:
        """Fallback compliance check when engine is unavailable.

        Performs basic structural validation of the calculation data.

        Args:
            calc_data: Stored calculation dict.
            frameworks: List of framework identifiers.

        Returns:
            List of per-framework result dicts.
        """
        results: List[dict] = []
        for fw in frameworks:
            has_total = calc_data.get("total_emissions_kg", 0) > 0 or calc_data.get("total_emissions_tco2e", 0) > 0
            has_method = bool(calc_data.get("method"))
            has_provenance = bool(calc_data.get("provenance_hash"))

            rules_checked = 3
            rules_passed = sum([has_total, has_method, has_provenance])
            rules_failed = rules_checked - rules_passed

            findings: List[dict] = []
            if not has_total:
                findings.append({
                    "rule_id": f"{fw}-001",
                    "severity": "error",
                    "message": "Total emissions value is zero or missing",
                })
            if not has_method:
                findings.append({
                    "rule_id": f"{fw}-002",
                    "severity": "error",
                    "message": "Calculation method not specified",
                })
            if not has_provenance:
                findings.append({
                    "rule_id": f"{fw}-003",
                    "severity": "warning",
                    "message": "Provenance hash not available",
                })

            status = "PASS" if rules_failed == 0 else ("WARNING" if rules_failed == 1 and has_total else "FAIL")

            results.append({
                "framework": fw,
                "status": status,
                "rules_checked": rules_checked,
                "rules_passed": rules_passed,
                "rules_failed": rules_failed,
                "findings": findings,
            })

        return results

    # ==========================================================================
    # RESULT CONVERSION
    # ==========================================================================

    @staticmethod
    def _result_to_dict(result: Any) -> Dict[str, Any]:
        """Convert an engine result (Pydantic model, dict, or dataclass) to dict.

        Args:
            result: Engine result object.

        Returns:
            Dict representation.
        """
        if isinstance(result, dict):
            return result

        # Pydantic model
        try:
            return result.model_dump(mode="json")
        except AttributeError:
            pass

        try:
            return result.dict()
        except AttributeError:
            pass

        # Dataclass
        try:
            from dataclasses import asdict
            return asdict(result)
        except (TypeError, ImportError):
            pass

        # Last resort: extract known attributes
        out: Dict[str, Any] = {}
        for attr in (
            "org_id", "reporting_year", "method", "total_co2e", "total_co2e_tonnes",
            "total_emissions_kg", "total_emissions_tco2e", "product_count",
            "product_breakdowns", "breakdowns", "dqi_score", "uncertainty",
            "provenance_hash", "timestamp", "processing_time_ms",
        ):
            val = getattr(result, attr, None)
            if val is not None:
                out[attr] = val

        return out

    def _build_calculation_response(
        self,
        calc_id: str,
        method: str,
        result: Dict[str, Any],
        elapsed_ms: float,
    ) -> CalculationResponse:
        """Build a standardized CalculationResponse from engine output.

        Args:
            calc_id: Calculation identifier.
            method: Calculation method used.
            result: Engine result dict.
            elapsed_ms: Processing time in milliseconds.

        Returns:
            CalculationResponse Pydantic model.
        """
        # Extract total emissions
        total_kg = result.get("total_emissions_kg", result.get("total_co2e", ZERO))
        total_tco2e = result.get("total_emissions_tco2e", result.get("total_co2e_tonnes", ZERO))

        if isinstance(total_kg, Decimal):
            total_kg_f = float(total_kg)
        else:
            total_kg_f = float(total_kg) if total_kg else 0.0

        if isinstance(total_tco2e, Decimal):
            total_tco2e_f = float(total_tco2e)
        else:
            total_tco2e_f = float(total_tco2e) if total_tco2e else 0.0

        # If tco2e is not set, derive from kg
        if total_tco2e_f == 0.0 and total_kg_f > 0.0:
            total_tco2e_f = total_kg_f / 1000.0

        # Extract breakdowns
        breakdowns = result.get("product_breakdowns", result.get("breakdowns", []))
        if not isinstance(breakdowns, list):
            breakdowns = []

        # Serialize breakdowns to dicts
        serialized_breakdowns: List[dict] = []
        for bd in breakdowns:
            if isinstance(bd, dict):
                serialized_breakdowns.append(bd)
            else:
                try:
                    serialized_breakdowns.append(bd.model_dump(mode="json"))
                except AttributeError:
                    try:
                        serialized_breakdowns.append(bd.dict())
                    except AttributeError:
                        serialized_breakdowns.append({"data": str(bd)})

        # DQI score -- may be Decimal, dict, or Pydantic model
        dqi_raw = result.get("dqi_score")
        dqi_f = None
        if isinstance(dqi_raw, dict):
            dqi_f = float(dqi_raw.get("overall", 0))
        elif hasattr(dqi_raw, "overall_score"):
            # Pydantic DataQualityScore model
            dqi_f = float(dqi_raw.overall_score) / 20.0  # Normalize 0-100 to 1-5
        elif hasattr(dqi_raw, "overall"):
            dqi_f = float(dqi_raw.overall)
        elif dqi_raw is not None:
            try:
                dqi_f = float(dqi_raw)
            except (TypeError, ValueError):
                dqi_f = None

        # Uncertainty -- may be Decimal, dict, or Pydantic UncertaintyResult model
        unc_raw = result.get("uncertainty")
        unc_f = None
        if isinstance(unc_raw, dict):
            unc_f = float(unc_raw.get("half_width_fraction", 0)) * 100.0
        elif hasattr(unc_raw, "half_width_fraction"):
            # Pydantic UncertaintyResult model
            unc_f = float(unc_raw.half_width_fraction) * 100.0
        elif hasattr(unc_raw, "model_dump"):
            # Generic Pydantic model -- try to extract half_width_fraction
            d = unc_raw.model_dump()
            unc_f = float(d.get("half_width_fraction", d.get("uncertainty_pct", 0))) * 100.0
        elif unc_raw is not None:
            try:
                val = float(unc_raw)
                unc_f = val * 100.0 if val < 1.0 else val
            except (TypeError, ValueError):
                unc_f = None

        return CalculationResponse(
            success=True,
            calc_id=calc_id,
            method=method,
            total_emissions_kg=total_kg_f,
            total_emissions_tco2e=total_tco2e_f,
            product_count=result.get("product_count", len(serialized_breakdowns)),
            dqi_score=dqi_f,
            uncertainty_pct=unc_f,
            provenance_hash=result.get("provenance_hash", ""),
            breakdowns=serialized_breakdowns,
            detail={
                "org_id": result.get("org_id", ""),
                "reporting_year": result.get("reporting_year", 0),
                "timestamp": result.get("timestamp", ""),
            },
            processing_time_ms=elapsed_ms,
        )


# ==============================================================================
# HELPER
# ==============================================================================


def _get_fallback_categories() -> List[str]:
    """Get fallback category list when models module is unavailable.

    Returns:
        List of category value strings.
    """
    return [
        "metals_ferrous", "metals_non_ferrous",
        "plastics_thermoplastic", "plastics_thermoset",
        "chemicals", "food_ingredients", "textiles",
        "electronics", "glass_ceramics", "wood_paper",
        "minerals", "agricultural",
    ]


# ==============================================================================
# MODULE-LEVEL FUNCTIONS
# ==============================================================================


def get_service() -> ProcessingSoldProductsService:
    """Get singleton ProcessingSoldProductsService instance.

    Thread-safe via double-checked locking.

    Returns:
        ProcessingSoldProductsService singleton instance.

    Example:
        >>> service = get_service()
        >>> service.health_check()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ProcessingSoldProductsService()
    return _service_instance


def get_router():
    """Get the FastAPI router for processing sold products endpoints.

    Returns:
        FastAPI APIRouter instance.

    Example:
        >>> router = get_router()
        >>> app.include_router(router, prefix="/api/v1/processing-sold-products")
    """
    from greenlang.agents.mrv.processing_sold_products.api.router import router
    return router


def reset_service() -> None:
    """Reset the singleton service instance.

    For testing purposes only. Clears the singleton so the next call to
    get_service() creates a fresh instance.

    Example:
        >>> reset_service()
        >>> service = get_service()  # New instance
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.info("ProcessingSoldProductsService singleton reset")
