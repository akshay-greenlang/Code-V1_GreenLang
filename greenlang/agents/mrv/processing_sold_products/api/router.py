# -*- coding: utf-8 -*-
"""
Processing of Sold Products REST API Router - AGENT-MRV-023
=============================================================

20 REST endpoints for the Processing of Sold Products Agent (GL-MRV-S3-010).

GHG Protocol Scope 3, Category 10: Processing of Sold Products.
Calculates emissions from downstream processing of intermediate products
sold by the reporting company, where such processing is not controlled
by the reporting company.

Prefix: ``/api/v1/processing-sold-products``

Endpoints:
     1. POST   /calculate                         - Full pipeline calculation
     2. POST   /calculate/site-specific            - Site-specific direct method
     3. POST   /calculate/site-specific/energy     - Energy-based site-specific
     4. POST   /calculate/site-specific/fuel       - Fuel-based site-specific
     5. POST   /calculate/average-data             - Average-data method
     6. POST   /calculate/average-data/energy-intensity - Energy intensity method
     7. POST   /calculate/spend                    - Spend-based EEIO
     8. POST   /calculate/hybrid                   - Hybrid aggregation
     9. POST   /calculate/batch                    - Batch calculation
    10. POST   /calculate/portfolio                - Portfolio analysis
    11. POST   /compliance/check                   - Compliance validation
    12. GET    /calculations/{calculation_id}      - Get calculation by ID
    13. GET    /calculations                       - List calculations with pagination
    14. DELETE /calculations/{calculation_id}      - Delete calculation
    15. GET    /emission-factors/{category}        - Get EFs by product category
    16. GET    /processing-types                   - List processing types & energy intensities
    17. GET    /processing-chains                  - Get multi-step chain definitions
    18. GET    /aggregations                       - Get aggregated results
    19. GET    /provenance/{calculation_id}        - Get provenance chain
    20. GET    /health                             - Health check

Calculation Methods:
    - Site-specific direct: Customer-reported processing emissions
    - Site-specific energy: Energy consumption x grid/fuel EF
    - Site-specific fuel: Fuel consumption x combustion EF
    - Average-data: Product category x processing type EF
    - Spend-based: Revenue x EEIO sector factor
    - Hybrid: Multi-method aggregation with weighting

Intermediate Product Categories:
    - Metals (ferrous / non-ferrous)
    - Plastics (thermoplastic / thermoset)
    - Chemicals, food ingredients, textiles
    - Electronics components, glass/ceramics
    - Wood/paper pulp, minerals, agricultural commodities

Processing Types (18):
    - Machining, stamping, welding, heat treatment
    - Injection molding, extrusion, blow molding
    - Casting, forging, coating, assembly
    - Chemical reaction, refining, milling
    - Drying, sintering, fermentation, textile finishing

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-023 Processing of Sold Products (GL-MRV-S3-010)
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Path, Depends, status
    from pydantic import BaseModel, Field, validator, constr
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; router unavailable")

# ---------------------------------------------------------------------------
# Graceful engine imports with _AVAILABLE flags
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.processing_sold_products.processing_database import (
        ProcessingDatabaseEngine,
    )
    PROCESSING_DB_AVAILABLE = True
except ImportError:
    ProcessingDatabaseEngine = None  # type: ignore[assignment,misc]
    PROCESSING_DB_AVAILABLE = False
    logger.debug("ProcessingDatabaseEngine not available")

try:
    from greenlang.agents.mrv.processing_sold_products.site_specific_calculator import (
        SiteSpecificCalculatorEngine,
    )
    SITE_SPECIFIC_AVAILABLE = True
except ImportError:
    SiteSpecificCalculatorEngine = None  # type: ignore[assignment,misc]
    SITE_SPECIFIC_AVAILABLE = False
    logger.debug("SiteSpecificCalculatorEngine not available")

try:
    from greenlang.agents.mrv.processing_sold_products.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
    AVERAGE_DATA_AVAILABLE = True
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]
    AVERAGE_DATA_AVAILABLE = False
    logger.debug("AverageDataCalculatorEngine not available")

try:
    from greenlang.agents.mrv.processing_sold_products.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
    SPEND_BASED_AVAILABLE = True
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]
    SPEND_BASED_AVAILABLE = False
    logger.debug("SpendBasedCalculatorEngine not available")

try:
    from greenlang.agents.mrv.processing_sold_products.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]
    HYBRID_AVAILABLE = False
    logger.debug("HybridAggregatorEngine not available")

try:
    from greenlang.agents.mrv.processing_sold_products.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]
    COMPLIANCE_AVAILABLE = False
    logger.debug("ComplianceCheckerEngine not available")

try:
    from greenlang.agents.mrv.processing_sold_products.processing_pipeline import (
        ProcessingPipelineEngine,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    ProcessingPipelineEngine = None  # type: ignore[assignment,misc]
    PIPELINE_AVAILABLE = False
    logger.debug("ProcessingPipelineEngine not available")


# ===================================================================
# Pydantic JSON configuration for Decimal serialization
# ===================================================================

if FASTAPI_AVAILABLE:

    class _DecimalModel(BaseModel):
        """Base model with Decimal JSON serialization support."""

        class Config:
            json_encoders = {Decimal: lambda v: float(v)}
            arbitrary_types_allowed = True

    # ===================================================================
    # REQUEST MODELS (12)
    # ===================================================================

    class FullPipelineCalculateRequest(_DecimalModel):
        """
        Request body for full-pipeline processing of sold products calculation.

        The pipeline routes to the best available method based on input data,
        running all 10 stages (VALIDATE -> CLASSIFY -> NORMALIZE ->
        RESOLVE_EFS -> CALCULATE -> ALLOCATE -> AGGREGATE -> COMPLIANCE ->
        PROVENANCE -> SEAL).

        Attributes:
            tenant_id: Tenant identifier for multi-tenancy isolation.
            org_id: Organisation identifier within the tenant.
            reporting_year: The reporting year for which emissions are calculated.
            product_category: Intermediate product category (e.g., METALS_FERROUS).
            processing_type: Downstream processing type (e.g., MACHINING).
            quantity: Quantity of intermediate product sold (in product_unit).
            product_unit: Unit of the quantity field (kg, tonne, m3, unit).
            method: Preferred calculation method (auto-selected if omitted).
            customer_name: Optional name of the downstream customer.
            customer_country: Optional ISO 3166-1 alpha-2 country code.
            energy_kwh: Energy consumed during processing (for energy method).
            fuel_litres: Fuel consumed during processing (for fuel method).
            fuel_type: Fuel type (for fuel method).
            grid_region: Grid region for electricity EF lookup.
            direct_co2e_kg: Direct customer-reported CO2e emissions in kg.
            revenue: Revenue from the product sale (for spend-based).
            currency: Currency code (ISO 4217) for the revenue field.
            sector_code: NAICS/NACE sector code for EEIO lookup.
            allocation_pct: Percentage of customer processing attributable to this product.
            metadata: Additional metadata for audit trail.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        product_category: str = Field(
            ..., description="Intermediate product category (e.g., METALS_FERROUS, PLASTICS_THERMOPLASTIC)"
        )
        processing_type: str = Field(
            ..., description="Processing type applied by customer (e.g., MACHINING, INJECTION_MOLDING)"
        )
        quantity: Decimal = Field(
            ..., ge=0, description="Quantity of intermediate product sold"
        )
        product_unit: str = Field(
            "tonne", description="Unit of quantity (kg, tonne, m3, unit, litre)"
        )
        method: Optional[str] = Field(
            None,
            description=(
                "Calculation method: SITE_SPECIFIC_DIRECT, SITE_SPECIFIC_ENERGY, "
                "SITE_SPECIFIC_FUEL, AVERAGE_DATA, ENERGY_INTENSITY, SPEND_BASED, HYBRID"
            ),
        )
        customer_name: Optional[str] = Field(None, max_length=500, description="Customer name")
        customer_country: Optional[str] = Field(
            None, max_length=2, description="Customer ISO 3166-1 alpha-2 country"
        )
        energy_kwh: Optional[Decimal] = Field(
            None, ge=0, description="Energy consumed in processing (kWh)"
        )
        fuel_litres: Optional[Decimal] = Field(
            None, ge=0, description="Fuel consumed in processing (litres)"
        )
        fuel_type: Optional[str] = Field(
            None, description="Fuel type (natural_gas, diesel, lpg, heavy_fuel_oil, coal, biomass)"
        )
        grid_region: Optional[str] = Field(
            None, max_length=50, description="Grid region code for electricity EF"
        )
        direct_co2e_kg: Optional[Decimal] = Field(
            None, ge=0, description="Direct customer-reported CO2e (kg)"
        )
        revenue: Optional[Decimal] = Field(
            None, ge=0, description="Revenue from product sale"
        )
        currency: Optional[str] = Field(
            None, max_length=3, description="ISO 4217 currency code"
        )
        sector_code: Optional[str] = Field(
            None, max_length=20, description="NAICS/NACE sector code for EEIO"
        )
        allocation_pct: Optional[Decimal] = Field(
            None, ge=0, le=100, description="Allocation percentage (0-100)"
        )
        metadata: Optional[Dict[str, Any]] = Field(
            default_factory=dict, description="Additional metadata for audit trail"
        )

        @validator("quantity")
        def validate_quantity_positive(cls, v: Decimal) -> Decimal:
            """Validate that quantity is strictly positive."""
            if v <= 0:
                raise ValueError("quantity must be greater than zero")
            return v

        @validator("method")
        def validate_method(cls, v: Optional[str]) -> Optional[str]:
            """Validate calculation method is recognized."""
            if v is None:
                return v
            allowed = {
                "SITE_SPECIFIC_DIRECT", "SITE_SPECIFIC_ENERGY",
                "SITE_SPECIFIC_FUEL", "AVERAGE_DATA",
                "ENERGY_INTENSITY", "SPEND_BASED", "HYBRID",
            }
            upper = v.upper()
            if upper not in allowed:
                raise ValueError(f"method must be one of {sorted(allowed)}")
            return upper

    class SiteSpecificDirectRequest(_DecimalModel):
        """
        Request body for site-specific direct calculation.

        The customer provides actual measured processing emissions
        (e.g., from a lifecycle assessment or EPD).

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            product_category: Intermediate product category.
            processing_type: Processing type.
            quantity: Quantity of product sold.
            product_unit: Unit of quantity.
            direct_co2e_kg: Customer-reported CO2e in kg.
            data_source: Source of the customer data (e.g., EPD, LCA, direct measurement).
            data_quality_score: Data quality indicator score (1=best, 5=worst).
            allocation_pct: Allocation percentage.
            customer_name: Customer name.
            customer_country: Customer country code.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        product_category: str = Field(..., description="Intermediate product category")
        processing_type: str = Field(..., description="Processing type")
        quantity: Decimal = Field(..., gt=0, description="Quantity of product sold")
        product_unit: str = Field("tonne", description="Unit of quantity")
        direct_co2e_kg: Decimal = Field(
            ..., ge=0, description="Customer-reported CO2e emissions in kg"
        )
        data_source: Optional[str] = Field(
            None, description="Data source (EPD, LCA, DIRECT_MEASUREMENT, CUSTOMER_REPORT)"
        )
        data_quality_score: Optional[int] = Field(
            None, ge=1, le=5, description="Data quality indicator (1=best, 5=worst)"
        )
        allocation_pct: Decimal = Field(
            Decimal("100"), ge=0, le=100, description="Allocation percentage"
        )
        customer_name: Optional[str] = Field(None, max_length=500, description="Customer name")
        customer_country: Optional[str] = Field(None, max_length=2, description="Customer country")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class SiteSpecificEnergyRequest(_DecimalModel):
        """
        Request body for site-specific energy-based calculation.

        Uses customer-reported energy consumption data and applies
        grid/fuel emission factors to derive processing emissions.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            product_category: Intermediate product category.
            processing_type: Processing type.
            quantity: Quantity of product sold.
            product_unit: Unit of quantity.
            energy_kwh: Total energy consumed in processing (kWh).
            grid_region: Grid region for electricity emission factor lookup.
            renewable_pct: Percentage of energy from renewable sources.
            allocation_pct: Allocation percentage.
            customer_name: Customer name.
            customer_country: Customer country code.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        product_category: str = Field(..., description="Intermediate product category")
        processing_type: str = Field(..., description="Processing type")
        quantity: Decimal = Field(..., gt=0, description="Quantity of product sold")
        product_unit: str = Field("tonne", description="Unit of quantity")
        energy_kwh: Decimal = Field(
            ..., ge=0, description="Total energy consumed in processing (kWh)"
        )
        grid_region: str = Field(
            ..., max_length=50, description="Grid region code for electricity EF"
        )
        renewable_pct: Decimal = Field(
            Decimal("0"), ge=0, le=100,
            description="Percentage of energy from renewable sources (0-100)"
        )
        allocation_pct: Decimal = Field(
            Decimal("100"), ge=0, le=100, description="Allocation percentage"
        )
        customer_name: Optional[str] = Field(None, max_length=500, description="Customer name")
        customer_country: Optional[str] = Field(None, max_length=2, description="Customer country")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class SiteSpecificFuelRequest(_DecimalModel):
        """
        Request body for site-specific fuel-based calculation.

        Uses customer-reported fuel consumption data and applies
        combustion emission factors.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            product_category: Intermediate product category.
            processing_type: Processing type.
            quantity: Quantity of product sold.
            product_unit: Unit of quantity.
            fuel_litres: Total fuel consumed in processing (litres).
            fuel_type: Fuel type used.
            allocation_pct: Allocation percentage.
            customer_name: Customer name.
            customer_country: Customer country code.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        product_category: str = Field(..., description="Intermediate product category")
        processing_type: str = Field(..., description="Processing type")
        quantity: Decimal = Field(..., gt=0, description="Quantity of product sold")
        product_unit: str = Field("tonne", description="Unit of quantity")
        fuel_litres: Decimal = Field(
            ..., ge=0, description="Total fuel consumed in processing (litres)"
        )
        fuel_type: str = Field(
            ..., description="Fuel type (natural_gas, diesel, lpg, heavy_fuel_oil, coal, biomass)"
        )
        allocation_pct: Decimal = Field(
            Decimal("100"), ge=0, le=100, description="Allocation percentage"
        )
        customer_name: Optional[str] = Field(None, max_length=500, description="Customer name")
        customer_country: Optional[str] = Field(None, max_length=2, description="Customer country")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

        @validator("fuel_type")
        def validate_fuel_type(cls, v: str) -> str:
            """Validate fuel type is recognized."""
            allowed = {
                "natural_gas", "diesel", "lpg", "heavy_fuel_oil",
                "coal", "biomass",
            }
            lower = v.lower()
            if lower not in allowed:
                raise ValueError(f"fuel_type must be one of {sorted(allowed)}")
            return lower

    class AverageDataCalculateRequest(_DecimalModel):
        """
        Request body for average-data calculation.

        Uses product category and processing type emission factors
        to estimate processing emissions.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            items: List of product items with category, processing type, and quantity.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        items: List[Dict[str, Any]] = Field(
            ...,
            min_items=1,
            max_items=10000,
            description=(
                "List of items. Each dict must contain: product_category (str), "
                "processing_type (str), quantity (number), product_unit (str). "
                "Optional: customer_name, customer_country, allocation_pct."
            ),
        )
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class EnergyIntensityCalculateRequest(_DecimalModel):
        """
        Request body for energy-intensity average-data calculation.

        Uses energy intensity factors (kWh per unit of product) for each
        processing type, combined with grid emission factors.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            items: List of product items with processing type, quantity, and grid region.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        items: List[Dict[str, Any]] = Field(
            ...,
            min_items=1,
            max_items=10000,
            description=(
                "List of items. Each dict must contain: product_category (str), "
                "processing_type (str), quantity (number), product_unit (str), "
                "grid_region (str). Optional: customer_name, allocation_pct."
            ),
        )
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class SpendBasedCalculateRequest(_DecimalModel):
        """
        Request body for spend-based EEIO calculation.

        Uses revenue data and EEIO sector factors to estimate
        processing emissions (least precise method).

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            revenue: Total revenue from sold intermediate products.
            currency: ISO 4217 currency code.
            sector_code: NAICS sector code for EEIO factor lookup.
            eeio_base_year: Base year for EEIO factors (for CPI deflation).
            items: Optional list of revenue breakdown by product.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        revenue: Decimal = Field(
            ..., gt=0, description="Total revenue from sold intermediate products"
        )
        currency: str = Field("USD", max_length=3, description="ISO 4217 currency code")
        sector_code: str = Field(
            ..., max_length=20, description="NAICS sector code for EEIO factor lookup"
        )
        eeio_base_year: Optional[int] = Field(
            None, ge=1990, le=2030, description="Base year for EEIO factors"
        )
        items: Optional[List[Dict[str, Any]]] = Field(
            None, description="Optional revenue breakdown by product"
        )
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class HybridCalculateRequest(_DecimalModel):
        """
        Request body for hybrid (multi-method) aggregation.

        Combines results from multiple calculation methods using
        quality-weighted averaging.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            items: List of product items with heterogeneous data availability.
            weighting_strategy: How to weight different methods (QUALITY, EQUAL, CUSTOM).
            custom_weights: Custom method weights (only when weighting_strategy=CUSTOM).
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        items: List[Dict[str, Any]] = Field(
            ...,
            min_items=1,
            max_items=10000,
            description="List of product items with heterogeneous data",
        )
        weighting_strategy: str = Field(
            "QUALITY",
            description="Weighting strategy: QUALITY, EQUAL, CUSTOM",
        )
        custom_weights: Optional[Dict[str, float]] = Field(
            None, description="Custom method weights (when strategy=CUSTOM)"
        )
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

        @validator("weighting_strategy")
        def validate_weighting(cls, v: str) -> str:
            """Validate weighting strategy."""
            allowed = {"QUALITY", "EQUAL", "CUSTOM"}
            upper = v.upper()
            if upper not in allowed:
                raise ValueError(f"weighting_strategy must be one of {sorted(allowed)}")
            return upper

    class BatchCalculateRequest(_DecimalModel):
        """
        Request body for batch calculation.

        Processes multiple product-customer combinations in a single request.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            calculations: List of individual calculation requests.
            method: Default method for all calculations (can be overridden per item).
            parallel: Whether to execute calculations in parallel.
            batch_id: Optional batch identifier for idempotency.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        calculations: List[Dict[str, Any]] = Field(
            ...,
            min_items=1,
            max_items=10000,
            description="List of individual calculation input dicts",
        )
        method: Optional[str] = Field(
            None, description="Default method for all items (overridable per item)"
        )
        parallel: bool = Field(True, description="Execute in parallel")
        batch_id: Optional[str] = Field(None, description="Optional batch identifier")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class PortfolioCalculateRequest(_DecimalModel):
        """
        Request body for portfolio-level analysis.

        Analyses all sold intermediate products across multiple customers
        and produces aggregate metrics with hot-spot identification.

        Attributes:
            tenant_id: Tenant identifier.
            org_id: Organisation identifier.
            reporting_year: Reporting year.
            items: List of all product-customer records for the period.
            group_by: Dimensions to aggregate by.
            include_hotspots: Whether to produce Pareto hot-spot analysis.
            include_trends: Whether to include year-over-year trends.
            metadata: Additional metadata.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        org_id: Optional[str] = Field(None, max_length=200, description="Organisation ID")
        reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
        items: List[Dict[str, Any]] = Field(
            ...,
            min_items=1,
            max_items=50000,
            description="Product-customer records for the period",
        )
        group_by: List[str] = Field(
            default_factory=lambda: ["product_category", "processing_type", "customer_country"],
            description="Dimensions to aggregate by",
        )
        include_hotspots: bool = Field(True, description="Include Pareto hot-spot analysis")
        include_trends: bool = Field(False, description="Include year-over-year trends")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")

    class ComplianceCheckRequest(_DecimalModel):
        """
        Request body for compliance checking.

        Validates a calculation result against selected regulatory frameworks.

        Attributes:
            tenant_id: Tenant identifier.
            calculation_id: Calculation to validate.
            frameworks: List of regulatory frameworks to check.
            include_recommendations: Whether to include remediation recommendations.
        """

        tenant_id: constr(min_length=1, max_length=100) = Field(  # type: ignore[valid-type]
            ..., description="Tenant identifier"
        )
        calculation_id: str = Field(..., description="Calculation ID to check")
        frameworks: List[str] = Field(
            default_factory=lambda: [
                "GHG_PROTOCOL", "ISO_14064", "CSRD_E1",
                "CDP", "SBTi", "SB253", "GRI_305",
            ],
            description="Regulatory frameworks to check against",
        )
        include_recommendations: bool = Field(
            True, description="Include remediation recommendations"
        )

    # ===================================================================
    # RESPONSE MODELS (14)
    # ===================================================================

    class CalculationResultResponse(_DecimalModel):
        """
        Response model for a single processing-of-sold-products calculation.

        Contains emissions breakdown, method details, data quality,
        and provenance hash for audit trail.
        """

        calculation_id: str = Field(..., description="Unique calculation identifier")
        tenant_id: str = Field(..., description="Tenant identifier")
        org_id: Optional[str] = Field(None, description="Organisation identifier")
        reporting_year: int = Field(..., description="Reporting year")
        product_category: str = Field(..., description="Product category")
        processing_type: str = Field(..., description="Processing type")
        method: str = Field(..., description="Calculation method used")
        quantity: Decimal = Field(..., description="Quantity of product")
        product_unit: str = Field(..., description="Unit of quantity")

        # Emissions breakdown
        co2_kg: Decimal = Field(..., description="CO2 emissions in kg")
        ch4_kg: Decimal = Field(..., description="CH4 emissions in kg")
        n2o_kg: Decimal = Field(..., description="N2O emissions in kg")
        total_co2e_kg: Decimal = Field(..., description="Total CO2-equivalent emissions in kg")
        co2e_per_unit: Decimal = Field(..., description="Emission intensity per product unit")

        # Factors used
        emission_factor_value: Optional[Decimal] = Field(
            None, description="Emission factor value applied"
        )
        emission_factor_unit: Optional[str] = Field(
            None, description="Emission factor unit"
        )
        emission_factor_source: Optional[str] = Field(
            None, description="Emission factor source"
        )

        # Quality and allocation
        allocation_pct: Decimal = Field(
            Decimal("100"), description="Allocation percentage applied"
        )
        data_quality_score: Optional[Decimal] = Field(
            None, description="Data quality indicator (1=best, 5=worst)"
        )
        uncertainty_pct: Optional[Decimal] = Field(
            None, description="Uncertainty estimate as percentage"
        )

        # Customer
        customer_name: Optional[str] = Field(None, description="Customer name")
        customer_country: Optional[str] = Field(None, description="Customer country")

        # Audit
        calculation_timestamp: datetime = Field(..., description="When calculation was performed")
        provenance_hash: str = Field(..., description="SHA-256 provenance hash")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class BatchCalculationResponse(_DecimalModel):
        """Response model for batch calculation results."""

        batch_id: str = Field(..., description="Batch identifier")
        tenant_id: str = Field(..., description="Tenant identifier")
        total_calculations: int = Field(..., description="Total calculations requested")
        successful: int = Field(..., description="Successful calculations")
        failed: int = Field(..., description="Failed calculations")
        total_co2e_kg: Decimal = Field(..., description="Total CO2e across all calculations")
        results: List[CalculationResultResponse] = Field(
            ..., description="Individual calculation results"
        )
        errors: List[Dict[str, Any]] = Field(
            default_factory=list, description="Error details for failed calculations"
        )
        processing_time_ms: float = Field(..., description="Total processing time in ms")

    class PortfolioAnalysisResponse(_DecimalModel):
        """Response model for portfolio analysis."""

        tenant_id: str = Field(..., description="Tenant identifier")
        org_id: Optional[str] = Field(None, description="Organisation identifier")
        reporting_year: int = Field(..., description="Reporting year")
        total_co2e_kg: Decimal = Field(..., description="Total CO2e across portfolio")
        total_products: int = Field(..., description="Total product records analysed")
        by_category: Dict[str, Any] = Field(
            ..., description="Emissions breakdown by product category"
        )
        by_processing_type: Dict[str, Any] = Field(
            ..., description="Emissions breakdown by processing type"
        )
        by_customer_country: Dict[str, Any] = Field(
            ..., description="Emissions breakdown by customer country"
        )
        by_method: Dict[str, Any] = Field(
            ..., description="Emissions breakdown by calculation method"
        )
        hotspots: Optional[List[Dict[str, Any]]] = Field(
            None, description="Pareto hot-spot analysis (top contributors)"
        )
        trends: Optional[Dict[str, Any]] = Field(
            None, description="Year-over-year trend data"
        )
        avg_data_quality: Optional[Decimal] = Field(
            None, description="Portfolio-level average data quality score"
        )
        analysis_timestamp: datetime = Field(..., description="Analysis timestamp")
        provenance_hash: str = Field(..., description="Portfolio provenance hash")

    class ComplianceCheckResponse(_DecimalModel):
        """Response model for compliance check results."""

        check_id: str = Field(..., description="Compliance check identifier")
        tenant_id: str = Field(..., description="Tenant identifier")
        calculation_id: str = Field(..., description="Calculation checked")
        overall_status: str = Field(
            ..., description="Overall status: PASS, FAIL, WARNING, NOT_APPLICABLE"
        )
        frameworks_checked: List[str] = Field(..., description="Frameworks checked")
        framework_results: List[Dict[str, Any]] = Field(
            ..., description="Per-framework results with findings"
        )
        recommendations: List[str] = Field(
            default_factory=list, description="Remediation recommendations"
        )
        check_timestamp: datetime = Field(..., description="Check timestamp")

    class EmissionFactorResponse(_DecimalModel):
        """Response model for emission factor lookup."""

        factor_id: str = Field(..., description="Factor identifier")
        category: str = Field(..., description="Product category")
        processing_type: str = Field(..., description="Processing type")
        ef_value: Decimal = Field(..., description="Emission factor value")
        ef_unit: str = Field(..., description="Emission factor unit")
        source: str = Field(..., description="Data source")
        region: Optional[str] = Field(None, description="Regional applicability")
        year: int = Field(..., description="Factor year")
        uncertainty_pct: Optional[Decimal] = Field(None, description="Uncertainty %")

    class EmissionFactorListResponse(_DecimalModel):
        """Response model for emission factor listing."""

        factors: List[EmissionFactorResponse] = Field(..., description="Emission factors")
        total: int = Field(..., description="Total count")
        category: str = Field(..., description="Product category queried")

    class ProcessingTypeResponse(_DecimalModel):
        """Response model for a single processing type."""

        processing_type: str = Field(..., description="Processing type code")
        display_name: str = Field(..., description="Human-readable name")
        category: str = Field(..., description="Processing category")
        energy_intensity_kwh_per_tonne: Optional[Decimal] = Field(
            None, description="Typical energy intensity (kWh/tonne)"
        )
        applicable_product_categories: List[str] = Field(
            ..., description="Product categories this processing applies to"
        )
        description: Optional[str] = Field(None, description="Description")

    class ProcessingTypeListResponse(_DecimalModel):
        """Response model for processing type listing."""

        processing_types: List[ProcessingTypeResponse] = Field(
            ..., description="Processing types"
        )
        total: int = Field(..., description="Total count")

    class ProcessingChainResponse(_DecimalModel):
        """Response model for a processing chain definition."""

        chain_id: str = Field(..., description="Chain identifier")
        chain_type: str = Field(..., description="Chain type")
        display_name: str = Field(..., description="Human-readable name")
        steps: List[Dict[str, Any]] = Field(
            ..., description="Ordered list of processing steps"
        )
        total_energy_intensity_kwh_per_tonne: Optional[Decimal] = Field(
            None, description="Cumulative energy intensity"
        )
        description: Optional[str] = Field(None, description="Description")

    class ProcessingChainListResponse(_DecimalModel):
        """Response model for processing chain listing."""

        chains: List[ProcessingChainResponse] = Field(..., description="Processing chains")
        total: int = Field(..., description="Total count")

    class AggregationResponse(_DecimalModel):
        """Response model for aggregated results."""

        tenant_id: str = Field(..., description="Tenant identifier")
        org_id: Optional[str] = Field(None, description="Organisation identifier")
        period: str = Field(..., description="Aggregation period")
        total_co2e_kg: Decimal = Field(..., description="Total CO2e emissions")
        total_quantity: Decimal = Field(..., description="Total product quantity")
        calculation_count: int = Field(..., description="Number of calculations")
        by_category: Dict[str, Any] = Field(
            ..., description="Breakdown by product category"
        )
        by_processing_type: Dict[str, Any] = Field(
            ..., description="Breakdown by processing type"
        )
        by_method: Dict[str, Any] = Field(
            ..., description="Breakdown by calculation method"
        )
        avg_data_quality: Optional[Decimal] = Field(
            None, description="Average data quality score"
        )
        aggregation_timestamp: datetime = Field(..., description="Aggregation timestamp")

    class ProvenanceResponse(_DecimalModel):
        """Response model for provenance chain."""

        calculation_id: str = Field(..., description="Calculation identifier")
        tenant_id: str = Field(..., description="Tenant identifier")
        provenance_hash: str = Field(..., description="Final SHA-256 provenance hash")
        input_hash: str = Field(..., description="Input data hash")
        ef_hash: str = Field(..., description="Emission factor hash")
        output_hash: str = Field(..., description="Output data hash")
        chain: List[Dict[str, Any]] = Field(
            ..., description="Ordered pipeline stage hashes"
        )
        is_verifiable: bool = Field(..., description="Chain integrity verified")
        calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
        engine_versions: Dict[str, str] = Field(
            ..., description="Engine versions used"
        )

    class CalculationListResponse(_DecimalModel):
        """Response model for paginated calculation listing."""

        calculations: List[CalculationResultResponse] = Field(
            ..., description="Calculations"
        )
        total: int = Field(..., description="Total count")
        limit: int = Field(..., description="Page limit")
        offset: int = Field(..., description="Page offset")

    class DeleteResponse(_DecimalModel):
        """Response model for deletion."""

        deleted: bool = Field(..., description="Whether deletion succeeded")
        calculation_id: str = Field(..., description="Deleted calculation ID")
        timestamp: datetime = Field(..., description="Deletion timestamp")

    class HealthResponse(_DecimalModel):
        """Response model for health check."""

        status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
        service: str = Field(
            "processing-sold-products", description="Service name"
        )
        version: str = Field(..., description="Service version")
        agent_id: str = Field("GL-MRV-S3-010", description="Agent identifier")
        timestamp: datetime = Field(..., description="Check timestamp")
        engines: Dict[str, str] = Field(..., description="Per-engine availability")
        database_connected: bool = Field(..., description="Database connectivity")
        cache_connected: bool = Field(..., description="Cache connectivity")

    # ===================================================================
    # ROUTER
    # ===================================================================

    psp_router = APIRouter(
        prefix="/api/v1/processing-sold-products",
        tags=["Processing of Sold Products"],
        responses={
            400: {"description": "Validation error"},
            404: {"description": "Not found"},
            500: {"description": "Internal server error"},
            503: {"description": "Service unavailable"},
        },
    )

    # ---------------------------------------------------------------
    # Service dependency
    # ---------------------------------------------------------------

    _service_instance = None

    def _get_service():
        """
        Get or create the ProcessingSoldProductsService singleton.

        Returns:
            ProcessingSoldProductsService instance.

        Raises:
            HTTPException 503 if the service cannot be initialised.
        """
        global _service_instance
        if _service_instance is None:
            try:
                from greenlang.agents.mrv.processing_sold_products.setup import get_service
                _service_instance = get_service()
                logger.info("ProcessingSoldProductsService injected into router")
            except Exception as exc:
                logger.error(
                    "Failed to initialise ProcessingSoldProductsService: %s",
                    exc, exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Processing of Sold Products service is unavailable",
                )
        return _service_instance

    # ===================================================================
    # ENDPOINT 1: POST /calculate
    # ===================================================================

    @psp_router.post(
        "/calculate",
        response_model=CalculationResultResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Full pipeline calculation",
        description=(
            "Run the full 10-stage pipeline for a single intermediate product. "
            "Automatically selects the best method based on available data. "
            "Returns deterministic emissions with SHA-256 provenance hash."
        ),
    )
    async def calculate_full_pipeline(
        request: FullPipelineCalculateRequest,
        service=Depends(_get_service),
    ) -> CalculationResultResponse:
        """
        Full pipeline calculation for processing of sold products.

        Args:
            request: Full pipeline calculation request.
            service: Injected service facade.

        Returns:
            CalculationResultResponse with emissions and provenance.

        Raises:
            HTTPException 400: Validation error.
            HTTPException 500: Calculation failure.
        """
        try:
            logger.info(
                "Full pipeline calculation: tenant=%s, category=%s, processing=%s",
                request.tenant_id, request.product_category, request.processing_type,
            )
            result = service.calculate(
                inputs=request.dict(),
                method=request.method,
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return CalculationResultResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in calculate: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("calculate failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Calculation failed",
            )

    # ===================================================================
    # ENDPOINT 2: POST /calculate/site-specific
    # ===================================================================

    @psp_router.post(
        "/calculate/site-specific",
        response_model=CalculationResultResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Site-specific direct calculation",
        description=(
            "Calculate using customer-reported processing emissions. "
            "Highest accuracy method (GHG Protocol preferred). "
            "Requires direct CO2e data from the downstream customer."
        ),
    )
    async def calculate_site_specific_direct(
        request: SiteSpecificDirectRequest,
        service=Depends(_get_service),
    ) -> CalculationResultResponse:
        """
        Site-specific direct calculation using customer-reported data.

        Args:
            request: Site-specific direct request with customer CO2e data.
            service: Injected service facade.

        Returns:
            CalculationResultResponse with emissions and provenance.
        """
        try:
            logger.info(
                "Site-specific direct: tenant=%s, category=%s, co2e=%.2f kg",
                request.tenant_id, request.product_category, request.direct_co2e_kg,
            )
            result = service.calculate_site_specific(
                inputs=request.dict(),
                method="SITE_SPECIFIC_DIRECT",
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return CalculationResultResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in site-specific direct: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("site-specific direct failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Site-specific direct calculation failed",
            )

    # ===================================================================
    # ENDPOINT 3: POST /calculate/site-specific/energy
    # ===================================================================

    @psp_router.post(
        "/calculate/site-specific/energy",
        response_model=CalculationResultResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Site-specific energy-based calculation",
        description=(
            "Calculate using customer-reported energy consumption. "
            "Applies grid emission factors by region. "
            "Supports renewable energy deduction."
        ),
    )
    async def calculate_site_specific_energy(
        request: SiteSpecificEnergyRequest,
        service=Depends(_get_service),
    ) -> CalculationResultResponse:
        """
        Site-specific energy-based calculation.

        Args:
            request: Energy-based request with kWh and grid region.
            service: Injected service facade.

        Returns:
            CalculationResultResponse with emissions and provenance.
        """
        try:
            logger.info(
                "Site-specific energy: tenant=%s, energy=%.2f kWh, region=%s",
                request.tenant_id, request.energy_kwh, request.grid_region,
            )
            result = service.calculate_site_specific(
                inputs=request.dict(),
                method="SITE_SPECIFIC_ENERGY",
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return CalculationResultResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in site-specific energy: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("site-specific energy failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Site-specific energy calculation failed",
            )

    # ===================================================================
    # ENDPOINT 4: POST /calculate/site-specific/fuel
    # ===================================================================

    @psp_router.post(
        "/calculate/site-specific/fuel",
        response_model=CalculationResultResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Site-specific fuel-based calculation",
        description=(
            "Calculate using customer-reported fuel consumption. "
            "Applies combustion emission factors by fuel type. "
            "Supports 6 fuel types (natural gas, diesel, LPG, HFO, coal, biomass)."
        ),
    )
    async def calculate_site_specific_fuel(
        request: SiteSpecificFuelRequest,
        service=Depends(_get_service),
    ) -> CalculationResultResponse:
        """
        Site-specific fuel-based calculation.

        Args:
            request: Fuel-based request with litres and fuel type.
            service: Injected service facade.

        Returns:
            CalculationResultResponse with emissions and provenance.
        """
        try:
            logger.info(
                "Site-specific fuel: tenant=%s, fuel=%.2f L (%s)",
                request.tenant_id, request.fuel_litres, request.fuel_type,
            )
            result = service.calculate_site_specific(
                inputs=request.dict(),
                method="SITE_SPECIFIC_FUEL",
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return CalculationResultResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in site-specific fuel: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("site-specific fuel failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Site-specific fuel calculation failed",
            )

    # ===================================================================
    # ENDPOINT 5: POST /calculate/average-data
    # ===================================================================

    @psp_router.post(
        "/calculate/average-data",
        response_model=BatchCalculationResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Average-data calculation",
        description=(
            "Calculate using product category and processing type emission factors. "
            "Supports up to 10,000 items per request. "
            "Formula: SUM(quantity x processing_EF) for each product-processing pair."
        ),
    )
    async def calculate_average_data(
        request: AverageDataCalculateRequest,
        service=Depends(_get_service),
    ) -> BatchCalculationResponse:
        """
        Average-data method calculation.

        Args:
            request: Average-data request with product items.
            service: Injected service facade.

        Returns:
            BatchCalculationResponse with per-item results.
        """
        try:
            logger.info(
                "Average-data calc: tenant=%s, items=%d",
                request.tenant_id, len(request.items),
            )
            result = service.calculate_average_data(
                inputs=request.dict(),
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return BatchCalculationResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in average-data: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("average-data failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Average-data calculation failed",
            )

    # ===================================================================
    # ENDPOINT 6: POST /calculate/average-data/energy-intensity
    # ===================================================================

    @psp_router.post(
        "/calculate/average-data/energy-intensity",
        response_model=BatchCalculationResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Energy intensity average-data calculation",
        description=(
            "Calculate using energy intensity factors (kWh/tonne) for each "
            "processing type, combined with regional grid emission factors. "
            "Formula: SUM(quantity x energy_intensity x grid_EF) per item."
        ),
    )
    async def calculate_energy_intensity(
        request: EnergyIntensityCalculateRequest,
        service=Depends(_get_service),
    ) -> BatchCalculationResponse:
        """
        Energy intensity average-data calculation.

        Args:
            request: Energy intensity request with items and grid regions.
            service: Injected service facade.

        Returns:
            BatchCalculationResponse with per-item results.
        """
        try:
            logger.info(
                "Energy intensity calc: tenant=%s, items=%d",
                request.tenant_id, len(request.items),
            )
            result = service.calculate_average_data(
                inputs={**request.dict(), "sub_method": "ENERGY_INTENSITY"},
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return BatchCalculationResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in energy-intensity: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("energy-intensity failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Energy intensity calculation failed",
            )

    # ===================================================================
    # ENDPOINT 7: POST /calculate/spend
    # ===================================================================

    @psp_router.post(
        "/calculate/spend",
        response_model=CalculationResultResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Spend-based EEIO calculation",
        description=(
            "Calculate using revenue data and EEIO sector factors. "
            "Least precise method per GHG Protocol hierarchy. "
            "Supports CPI deflation and 20 currencies."
        ),
    )
    async def calculate_spend_based(
        request: SpendBasedCalculateRequest,
        service=Depends(_get_service),
    ) -> CalculationResultResponse:
        """
        Spend-based EEIO calculation.

        Args:
            request: Spend-based request with revenue and sector code.
            service: Injected service facade.

        Returns:
            CalculationResultResponse with emissions and provenance.
        """
        try:
            logger.info(
                "Spend-based calc: tenant=%s, revenue=%.2f %s, sector=%s",
                request.tenant_id, request.revenue,
                request.currency, request.sector_code,
            )
            result = service.calculate_spend_based(
                revenue=request.revenue,
                currency=request.currency,
                sector=request.sector_code,
                year=request.eeio_base_year,
                org_id=request.org_id,
                reporting_year=request.reporting_year,
            )
            return CalculationResultResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in spend-based: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("spend-based failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Spend-based calculation failed",
            )

    # ===================================================================
    # ENDPOINT 8: POST /calculate/hybrid
    # ===================================================================

    @psp_router.post(
        "/calculate/hybrid",
        response_model=PortfolioAnalysisResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Hybrid multi-method aggregation",
        description=(
            "Combine results from multiple calculation methods using "
            "quality-weighted averaging. Supports QUALITY, EQUAL, "
            "and CUSTOM weighting strategies."
        ),
    )
    async def calculate_hybrid(
        request: HybridCalculateRequest,
        service=Depends(_get_service),
    ) -> PortfolioAnalysisResponse:
        """
        Hybrid multi-method aggregation.

        Args:
            request: Hybrid request with mixed-method items.
            service: Injected service facade.

        Returns:
            PortfolioAnalysisResponse with aggregated emissions.
        """
        try:
            logger.info(
                "Hybrid calc: tenant=%s, items=%d, weighting=%s",
                request.tenant_id, len(request.items), request.weighting_strategy,
            )
            result = service.calculate_hybrid(
                inputs=request.dict(),
                org_id=request.org_id,
                year=request.reporting_year,
            )
            return PortfolioAnalysisResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in hybrid: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("hybrid failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Hybrid calculation failed",
            )

    # ===================================================================
    # ENDPOINT 9: POST /calculate/batch
    # ===================================================================

    @psp_router.post(
        "/calculate/batch",
        response_model=BatchCalculationResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Batch calculation",
        description=(
            "Process up to 10,000 product-customer calculations in a single "
            "request. Supports parallel execution. Each item can specify "
            "its own method or inherit the batch default."
        ),
    )
    async def calculate_batch(
        request: BatchCalculateRequest,
        service=Depends(_get_service),
    ) -> BatchCalculationResponse:
        """
        Batch calculation for multiple items.

        Args:
            request: Batch request with multiple calculation dicts.
            service: Injected service facade.

        Returns:
            BatchCalculationResponse with per-item results and summary.
        """
        try:
            logger.info(
                "Batch calc: tenant=%s, count=%d, parallel=%s",
                request.tenant_id, len(request.calculations), request.parallel,
            )
            results = service.calculate_batch(
                batch_inputs=request.dict(),
                method=request.method,
            )
            return BatchCalculationResponse(**results)
        except ValueError as exc:
            logger.warning("Validation error in batch: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("batch failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch calculation failed",
            )

    # ===================================================================
    # ENDPOINT 10: POST /calculate/portfolio
    # ===================================================================

    @psp_router.post(
        "/calculate/portfolio",
        response_model=PortfolioAnalysisResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Portfolio analysis",
        description=(
            "Analyse all sold intermediate products across customers. "
            "Produces aggregate metrics with hot-spot identification. "
            "Supports grouping by category, processing type, country, and method."
        ),
    )
    async def calculate_portfolio(
        request: PortfolioCalculateRequest,
        service=Depends(_get_service),
    ) -> PortfolioAnalysisResponse:
        """
        Portfolio-level analysis.

        Args:
            request: Portfolio request with all product records.
            service: Injected service facade.

        Returns:
            PortfolioAnalysisResponse with aggregations and hot-spots.
        """
        try:
            logger.info(
                "Portfolio analysis: tenant=%s, items=%d, hotspots=%s",
                request.tenant_id, len(request.items), request.include_hotspots,
            )
            result = service.calculate_portfolio(inputs=request.dict())
            return PortfolioAnalysisResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in portfolio: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("portfolio failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Portfolio analysis failed",
            )

    # ===================================================================
    # ENDPOINT 11: POST /compliance/check
    # ===================================================================

    @psp_router.post(
        "/compliance/check",
        response_model=ComplianceCheckResponse,
        status_code=status.HTTP_200_OK,
        summary="Compliance validation",
        description=(
            "Validate a calculation result against 7 regulatory frameworks: "
            "GHG Protocol, ISO 14064, CSRD E1, CDP, SBTi, SB 253, GRI 305. "
            "Returns per-framework findings and recommendations."
        ),
    )
    async def check_compliance(
        request: ComplianceCheckRequest,
        service=Depends(_get_service),
    ) -> ComplianceCheckResponse:
        """
        Run compliance checks against regulatory frameworks.

        Args:
            request: Compliance check request with calculation ID and frameworks.
            service: Injected service facade.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        try:
            logger.info(
                "Compliance check: tenant=%s, calc=%s, frameworks=%s",
                request.tenant_id, request.calculation_id, request.frameworks,
            )
            result = service.check_compliance(result=request.dict())
            return ComplianceCheckResponse(**result)
        except ValueError as exc:
            logger.warning("Validation error in compliance: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("compliance check failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Compliance check failed",
            )

    # ===================================================================
    # ENDPOINT 12: GET /calculations/{calculation_id}
    # ===================================================================

    @psp_router.get(
        "/calculations/{calculation_id}",
        response_model=CalculationResultResponse,
        summary="Get calculation by ID",
        description="Retrieve a single calculation result by its unique identifier.",
    )
    async def get_calculation(
        calculation_id: str = Path(
            ..., description="Calculation identifier"
        ),
        tenant_id: str = Query(
            ..., description="Tenant identifier"
        ),
        service=Depends(_get_service),
    ) -> CalculationResultResponse:
        """
        Retrieve a calculation result by ID.

        Args:
            calculation_id: Unique calculation identifier.
            tenant_id: Tenant identifier for isolation.
            service: Injected service facade.

        Returns:
            CalculationResultResponse with full details.

        Raises:
            HTTPException 404: Calculation not found.
        """
        try:
            logger.info(
                "Get calculation: id=%s, tenant=%s",
                calculation_id, tenant_id,
            )
            result = service.get_calculation(
                calculation_id=calculation_id,
                tenant_id=tenant_id,
            )
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Calculation {calculation_id} not found",
                )
            return CalculationResultResponse(**result)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("get_calculation failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve calculation",
            )

    # ===================================================================
    # ENDPOINT 13: GET /calculations
    # ===================================================================

    @psp_router.get(
        "/calculations",
        response_model=CalculationListResponse,
        summary="List calculations",
        description=(
            "List calculations with pagination, filtering by tenant, "
            "org, year, category, processing type, and method."
        ),
    )
    async def list_calculations(
        tenant_id: str = Query(..., description="Tenant identifier"),
        org_id: Optional[str] = Query(None, description="Organisation ID"),
        reporting_year: Optional[int] = Query(None, ge=1990, le=2100, description="Reporting year"),
        product_category: Optional[str] = Query(None, description="Product category filter"),
        processing_type: Optional[str] = Query(None, description="Processing type filter"),
        method: Optional[str] = Query(None, description="Calculation method filter"),
        limit: int = Query(100, ge=1, le=10000, description="Page size"),
        offset: int = Query(0, ge=0, description="Page offset"),
        service=Depends(_get_service),
    ) -> CalculationListResponse:
        """
        List calculations with pagination and filtering.

        Args:
            tenant_id: Tenant identifier.
            org_id: Optional organisation filter.
            reporting_year: Optional year filter.
            product_category: Optional category filter.
            processing_type: Optional processing type filter.
            method: Optional method filter.
            limit: Page size (default 100, max 10000).
            offset: Page offset (default 0).
            service: Injected service facade.

        Returns:
            CalculationListResponse with paginated results.
        """
        try:
            logger.info(
                "List calculations: tenant=%s, limit=%d, offset=%d",
                tenant_id, limit, offset,
            )
            result = service.list_calculations(
                tenant_id=tenant_id,
                org_id=org_id,
                reporting_year=reporting_year,
                product_category=product_category,
                processing_type=processing_type,
                method=method,
                limit=limit,
                offset=offset,
            )
            return CalculationListResponse(**result)
        except Exception as exc:
            logger.error("list_calculations failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list calculations",
            )

    # ===================================================================
    # ENDPOINT 14: DELETE /calculations/{calculation_id}
    # ===================================================================

    @psp_router.delete(
        "/calculations/{calculation_id}",
        response_model=DeleteResponse,
        summary="Delete calculation",
        description="Soft-delete a calculation by its identifier.",
    )
    async def delete_calculation(
        calculation_id: str = Path(
            ..., description="Calculation identifier"
        ),
        tenant_id: str = Query(
            ..., description="Tenant identifier"
        ),
        service=Depends(_get_service),
    ) -> DeleteResponse:
        """
        Delete a calculation by ID.

        Args:
            calculation_id: Calculation identifier.
            tenant_id: Tenant identifier for isolation.
            service: Injected service facade.

        Returns:
            DeleteResponse with deletion status.

        Raises:
            HTTPException 404: Calculation not found.
        """
        try:
            logger.info(
                "Delete calculation: id=%s, tenant=%s",
                calculation_id, tenant_id,
            )
            deleted = service.delete_calculation(
                calculation_id=calculation_id,
                tenant_id=tenant_id,
            )
            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Calculation {calculation_id} not found",
                )
            return DeleteResponse(
                deleted=True,
                calculation_id=calculation_id,
                timestamp=datetime.utcnow(),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("delete_calculation failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete calculation",
            )

    # ===================================================================
    # ENDPOINT 15: GET /emission-factors/{category}
    # ===================================================================

    @psp_router.get(
        "/emission-factors/{category}",
        response_model=EmissionFactorListResponse,
        summary="Get emission factors by product category",
        description=(
            "Retrieve processing emission factors for a given product "
            "category. Returns all available processing types and their "
            "emission factors from authoritative sources."
        ),
    )
    async def get_emission_factors(
        category: str = Path(
            ..., description="Product category (e.g., METALS_FERROUS, PLASTICS_THERMOPLASTIC)"
        ),
        source: Optional[str] = Query(None, description="Filter by EF source"),
        region: Optional[str] = Query(None, description="Filter by region"),
        year: Optional[int] = Query(None, ge=1990, le=2100, description="Filter by year"),
        service=Depends(_get_service),
    ) -> EmissionFactorListResponse:
        """
        Get emission factors for a product category.

        Args:
            category: Product category code.
            source: Optional source filter.
            region: Optional region filter.
            year: Optional year filter.
            service: Injected service facade.

        Returns:
            EmissionFactorListResponse with matching factors.
        """
        try:
            logger.info(
                "Get EFs: category=%s, source=%s, region=%s",
                category, source, region,
            )
            factors = service.get_emission_factors(
                category=category,
                source=source,
                region=region,
                year=year,
            )
            return EmissionFactorListResponse(**factors)
        except ValueError as exc:
            logger.warning("Invalid category: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error("get_emission_factors failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve emission factors",
            )

    # ===================================================================
    # ENDPOINT 16: GET /processing-types
    # ===================================================================

    @psp_router.get(
        "/processing-types",
        response_model=ProcessingTypeListResponse,
        summary="List processing types",
        description=(
            "List all 18 supported processing types with their energy "
            "intensities and applicable product categories."
        ),
    )
    async def list_processing_types(
        category: Optional[str] = Query(
            None, description="Filter by product category"
        ),
        service=Depends(_get_service),
    ) -> ProcessingTypeListResponse:
        """
        List processing types and energy intensities.

        Args:
            category: Optional product category filter.
            service: Injected service facade.

        Returns:
            ProcessingTypeListResponse with processing types.
        """
        try:
            logger.info("List processing types: category=%s", category)
            result = service.get_all_processing_types(category=category)
            return ProcessingTypeListResponse(**result)
        except Exception as exc:
            logger.error("list_processing_types failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list processing types",
            )

    # ===================================================================
    # ENDPOINT 17: GET /processing-chains
    # ===================================================================

    @psp_router.get(
        "/processing-chains",
        response_model=ProcessingChainListResponse,
        summary="Get processing chain definitions",
        description=(
            "Retrieve multi-step processing chain definitions. "
            "Each chain describes an ordered sequence of processing "
            "steps applied to an intermediate product."
        ),
    )
    async def list_processing_chains(
        chain_type: Optional[str] = Query(
            None, description="Filter by chain type"
        ),
        product_category: Optional[str] = Query(
            None, description="Filter by product category"
        ),
        service=Depends(_get_service),
    ) -> ProcessingChainListResponse:
        """
        List processing chain definitions.

        Args:
            chain_type: Optional chain type filter.
            product_category: Optional category filter.
            service: Injected service facade.

        Returns:
            ProcessingChainListResponse with chain definitions.
        """
        try:
            logger.info(
                "List processing chains: type=%s, category=%s",
                chain_type, product_category,
            )
            result = service.get_processing_chains(
                chain_type=chain_type,
                product_category=product_category,
            )
            return ProcessingChainListResponse(**result)
        except Exception as exc:
            logger.error("list_processing_chains failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list processing chains",
            )

    # ===================================================================
    # ENDPOINT 18: GET /aggregations
    # ===================================================================

    @psp_router.get(
        "/aggregations",
        response_model=AggregationResponse,
        summary="Get aggregated results",
        description=(
            "Retrieve aggregated emission results for a tenant, "
            "optionally filtered by organisation and period."
        ),
    )
    async def get_aggregations(
        tenant_id: str = Query(..., description="Tenant identifier"),
        org_id: Optional[str] = Query(None, description="Organisation ID"),
        period: str = Query(
            ..., description="Aggregation period (e.g., '2025', '2025-Q1', '2025-01')"
        ),
        service=Depends(_get_service),
    ) -> AggregationResponse:
        """
        Get aggregated results.

        Args:
            tenant_id: Tenant identifier.
            org_id: Optional organisation filter.
            period: Aggregation period.
            service: Injected service facade.

        Returns:
            AggregationResponse with aggregated emissions.
        """
        try:
            logger.info(
                "Get aggregations: tenant=%s, org=%s, period=%s",
                tenant_id, org_id, period,
            )
            result = service.get_aggregations(
                org_id=org_id or tenant_id,
                period=period,
            )
            return AggregationResponse(**result)
        except Exception as exc:
            logger.error("get_aggregations failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve aggregations",
            )

    # ===================================================================
    # ENDPOINT 19: GET /provenance/{calculation_id}
    # ===================================================================

    @psp_router.get(
        "/provenance/{calculation_id}",
        response_model=ProvenanceResponse,
        summary="Get provenance chain",
        description=(
            "Retrieve the SHA-256 provenance hash chain for a calculation. "
            "Includes per-stage hashes for full audit trail verification."
        ),
    )
    async def get_provenance(
        calculation_id: str = Path(
            ..., description="Calculation identifier"
        ),
        tenant_id: str = Query(
            ..., description="Tenant identifier"
        ),
        service=Depends(_get_service),
    ) -> ProvenanceResponse:
        """
        Get provenance chain for a calculation.

        Args:
            calculation_id: Calculation identifier.
            tenant_id: Tenant identifier for isolation.
            service: Injected service facade.

        Returns:
            ProvenanceResponse with hash chain.

        Raises:
            HTTPException 404: Provenance not found.
        """
        try:
            logger.info(
                "Get provenance: calc=%s, tenant=%s",
                calculation_id, tenant_id,
            )
            result = service.get_provenance(calc_id=calculation_id)
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Provenance for {calculation_id} not found",
                )
            return ProvenanceResponse(**result)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("get_provenance failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve provenance",
            )

    # ===================================================================
    # ENDPOINT 20: GET /health
    # ===================================================================

    @psp_router.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description=(
            "Check the health of the Processing of Sold Products service. "
            "Reports engine availability, database connectivity, and version."
        ),
    )
    async def health_check(
        service=Depends(_get_service),
    ) -> HealthResponse:
        """
        Service health check.

        Args:
            service: Injected service facade.

        Returns:
            HealthResponse with engine status and connectivity.
        """
        try:
            health = service.health_check()
            return HealthResponse(**health)
        except Exception as exc:
            logger.error("health_check failed: %s", exc, exc_info=True)
            return HealthResponse(
                status="unhealthy",
                version="1.0.0",
                timestamp=datetime.utcnow(),
                engines={
                    "ProcessingDatabaseEngine": "unavailable",
                    "SiteSpecificCalculatorEngine": "unavailable",
                    "AverageDataCalculatorEngine": "unavailable",
                    "SpendBasedCalculatorEngine": "unavailable",
                    "HybridAggregatorEngine": "unavailable",
                    "ComplianceCheckerEngine": "unavailable",
                    "ProcessingPipelineEngine": "unavailable",
                },
                database_connected=False,
                cache_connected=False,
            )


# ---------------------------------------------------------------------------
# Module-level router export (for use when FastAPI IS available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = psp_router
else:
    router = None  # type: ignore[assignment]
    psp_router = None  # type: ignore[assignment]
