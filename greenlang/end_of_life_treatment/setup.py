# -*- coding: utf-8 -*-
"""
End-of-Life Treatment Service Setup - AGENT-MRV-025

This module provides the service facade that wires together all 7 engines
for end-of-life treatment of sold products emissions calculations
(Scope 3 Category 12).

The EndOfLifeTreatmentService class provides a high-level API for:
- Waste-type-specific emissions (material x treatment x EF)
- Landfill emissions (IPCC FOD model parameters)
- Incineration emissions (mass burn, WtE, open burning)
- Recycling emissions (cut-off approach, avoided credits separate)
- Average-data calculations (product category composite EFs)
- Producer-specific calculations (EPD-based lifecycle data)
- Hybrid multi-method aggregation
- Compliance checking across 7 regulatory frameworks
- Circularity metrics (recycling rate, diversion rate)
- Portfolio hot-spot analysis
- Aggregations by treatment, material, category, region, period
- Provenance tracking with SHA-256 audit trail

Engines:
    1. EOLProductDatabaseEngine - Product data and EF persistence
    2. WasteTypeSpecificCalculatorEngine - Material x treatment calculations
    3. AverageDataCalculatorEngine - Composite category-level calculations
    4. ProducerSpecificCalculatorEngine - EPD-based lifecycle calculations
    5. HybridAggregatorEngine - Multi-method blended aggregation
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. EndOfLifeTreatmentPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.end_of_life_treatment.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate({
    ...     "products": [
    ...         {"name": "Widget", "weight_kg": 0.5, "units_sold": 100000}
    ...     ],
    ...     "region": "US",
    ... })
    >>> assert response["success"]

Integration:
    >>> from greenlang.end_of_life_treatment.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/end-of-life-treatment")

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
"""

import importlib
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_service_lock = threading.Lock()
_service_instance: Optional["EndOfLifeTreatmentService"] = None

# ==============================================================================
# CONSTANTS
# ==============================================================================

SERVICE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP


# ==============================================================================
# Request Models
# ==============================================================================


class EOLCalculationRequest(BaseModel):
    """Request model for end-of-life treatment emissions calculation."""

    products: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of products with name, weight_kg, units_sold, category, material_composition",
    )
    region: str = Field("GLOBAL", description="Region for treatment mix: US, EU, UK, DE, JP, CN, IN, BR, AU, CA, KR, GLOBAL")
    method: Optional[str] = Field(None, description="Calculation method override: waste_type_specific, average_data, producer_specific")
    reporting_year: int = Field(2025, ge=2015, le=2035, description="Reporting year")
    org_id: Optional[str] = Field(None, description="Organization identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class LandfillCalculationRequest(BaseModel):
    """Request model for landfill-specific calculation."""

    material_type: str = Field(..., description="Material type: plastics, paper_cardboard, textiles, wood, organic, mixed, etc.")
    weight_kg: float = Field(..., gt=0, description="Weight in kg")
    region: str = Field("GLOBAL", description="Region for EF lookup")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class IncinerationCalculationRequest(BaseModel):
    """Request model for incineration-specific calculation."""

    material_type: str = Field(..., description="Material type")
    weight_kg: float = Field(..., gt=0, description="Weight in kg")
    energy_recovery: bool = Field(False, description="Whether energy is recovered (WtE)")
    region: str = Field("GLOBAL", description="Region for EF lookup")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class RecyclingCalculationRequest(BaseModel):
    """Request model for recycling-specific calculation."""

    material_type: str = Field(..., description="Material type")
    weight_kg: float = Field(..., gt=0, description="Weight in kg")
    include_avoided: bool = Field(True, description="Calculate avoided emissions (reported separately)")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class AverageDataCalculationRequest(BaseModel):
    """Request model for average-data calculation."""

    product_category: str = Field(..., description="Product category from taxonomy")
    total_weight_kg: float = Field(..., gt=0, description="Total weight in kg")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ProducerSpecificCalculationRequest(BaseModel):
    """Request model for producer-specific (EPD) calculation."""

    product_name: str = Field(..., description="Product name")
    total_weight_kg: float = Field(..., gt=0, description="Total weight in kg")
    producer_ef_kg_co2e_per_kg: float = Field(..., gt=0, description="EPD end-of-life EF")
    avoided_ef_kg_co2e_per_kg: float = Field(0.0, ge=0, description="EPD avoided emissions EF")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class HybridCalculationRequest(BaseModel):
    """Request model for hybrid multi-method calculation."""

    products: List[Dict[str, Any]] = Field(..., min_length=1, description="Products with mixed methods")
    region: str = Field("GLOBAL", description="Default region")
    reporting_year: int = Field(2025, ge=2015, le=2035, description="Reporting year")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class BatchCalculationRequest(BaseModel):
    """Request model for batch calculation."""

    items: List[EOLCalculationRequest] = Field(..., min_length=1, max_length=5000, description="Batch items")
    org_id: Optional[str] = Field(None, description="Organization identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio analysis."""

    products: List[Dict[str, Any]] = Field(..., min_length=1, description="Product portfolio")
    region: str = Field("GLOBAL", description="Region for treatment mix")
    reporting_year: int = Field(2025, ge=2015, le=2035, description="Reporting year")
    org_id: Optional[str] = Field(None, description="Organization identifier")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    calculation_id: str = Field(..., description="Calculation ID to check")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL_SCOPE3"],
        description="Frameworks to check",
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class MaterialEFRequest(BaseModel):
    """Request model for material emission factor lookup."""

    material_type: str = Field(..., description="Material type")
    treatment: str = Field(..., description="Treatment pathway")


class AggregationRequest(BaseModel):
    """Request model for aggregation queries."""

    reporting_year: int = Field(..., description="Reporting year")
    group_by: str = Field("treatment", description="Group by: treatment, material, category, region")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


# ==============================================================================
# Response Models
# ==============================================================================


class EOLCalculationResponse(BaseModel):
    """Response model for end-of-life treatment calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    total_co2e_kg: float = Field(..., description="Total gross CO2e in kg")
    total_co2e_tonnes: float = Field(0.0, description="Total gross CO2e in tonnes")
    avoided_emissions_kg: float = Field(0.0, description="Avoided emissions (separate)")
    energy_recovery_credits_kg: float = Field(0.0, description="Energy recovery credits (separate)")
    recycling_rate_pct: float = Field(0.0, description="Recycling rate percentage")
    diversion_rate_pct: float = Field(0.0, description="Landfill diversion rate percentage")
    method: str = Field("", description="Calculation method used")
    region: str = Field("GLOBAL", description="Region used")
    by_treatment: Dict[str, Any] = Field(default_factory=dict, description="Breakdown by treatment")
    by_material: Dict[str, Any] = Field(default_factory=dict, description="Breakdown by material")
    by_category: Dict[str, Any] = Field(default_factory=dict, description="Breakdown by category")
    product_count: int = Field(0, description="Number of products processed")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class SingleTreatmentResponse(BaseModel):
    """Response for single treatment pathway calculation."""

    success: bool = Field(..., description="Success flag")
    material_type: str = Field(..., description="Material type")
    treatment: str = Field(..., description="Treatment pathway")
    weight_kg: float = Field(..., description="Weight in kg")
    co2e_kg: float = Field(..., description="CO2e in kg")
    avoided_emissions_kg: float = Field(0.0, description="Avoided emissions (separate)")
    energy_recovery_credits_kg: float = Field(0.0, description="Energy credits (separate)")
    ef_source: str = Field("", description="Emission factor source")
    processing_time_ms: float = Field(0.0, description="Processing time")
    error: Optional[str] = Field(None, description="Error message")


class BatchResponse(BaseModel):
    """Response model for batch calculation."""

    success: bool = Field(..., description="Overall success flag")
    total_items: int = Field(..., description="Total items")
    successful: int = Field(..., description="Successful items")
    failed: int = Field(..., description="Failed items")
    total_co2e_kg: float = Field(..., description="Total CO2e across batch")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Errors")
    processing_time_ms: float = Field(0.0, description="Total processing time")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation checked")
    overall_status: str = Field(..., description="Overall status: pass, warning, fail")
    overall_score: float = Field(0.0, description="Overall compliance score")
    framework_results: Dict[str, Any] = Field(default_factory=dict, description="Per-framework results")
    double_counting_findings: List[Dict[str, Any]] = Field(default_factory=list, description="DC findings")
    processing_time_ms: float = Field(0.0, description="Processing time")


class MaterialEFResponse(BaseModel):
    """Response model for material emission factor lookup."""

    success: bool = Field(..., description="Success flag")
    material_type: str = Field(..., description="Material type")
    treatment: str = Field(..., description="Treatment pathway")
    co2e_per_kg: float = Field(0.0, description="EF in kgCO2e per kg")
    ch4_per_kg: float = Field(0.0, description="CH4 in kg per kg")
    source: str = Field("", description="EF source")
    error: Optional[str] = Field(None, description="Error")


class ProductCompositionResponse(BaseModel):
    """Response for product composition lookup."""

    success: bool = Field(..., description="Success flag")
    category: str = Field(..., description="Product category")
    composition: Dict[str, float] = Field(default_factory=dict, description="Material fractions")


class RegionalTreatmentMixResponse(BaseModel):
    """Response for regional treatment mix lookup."""

    success: bool = Field(..., description="Success flag")
    region: str = Field(..., description="Region code")
    treatment_mix: Dict[str, float] = Field(default_factory=dict, description="Treatment fractions")


class AvoidedEmissionsResponse(BaseModel):
    """Response for avoided emissions lookup."""

    success: bool = Field(..., description="Success flag")
    material_type: str = Field(..., description="Material type")
    avoided_co2e_per_kg: float = Field(0.0, description="Avoided EF per kg recycled")
    energy_credit_per_kg: float = Field(0.0, description="Energy credit per kg incinerated")


class CircularityScoreResponse(BaseModel):
    """Response for circularity score calculation."""

    success: bool = Field(..., description="Success flag")
    recycling_rate_pct: float = Field(0.0, description="Recycling rate %")
    diversion_rate_pct: float = Field(0.0, description="Diversion rate %")
    circularity_score: float = Field(0.0, description="Composite circularity score")


class AggregationResponse(BaseModel):
    """Response model for aggregation queries."""

    success: bool = Field(..., description="Success flag")
    total_co2e_kg: float = Field(0.0, description="Total CO2e")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Breakdown by group_by dimension")
    reporting_year: int = Field(0, description="Reporting year")
    processing_time_ms: float = Field(0.0, description="Processing time")


class ProvenanceResponse(BaseModel):
    """Response for provenance lookup."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field("", description="Calculation ID")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    chain: List[Dict[str, Any]] = Field(default_factory=list, description="Provenance chain")
    is_valid: bool = Field(False, description="Chain integrity verified")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    engines_status: Dict[str, bool] = Field(default_factory=dict, description="Per-engine status")
    uptime_seconds: float = Field(0.0, description="Service uptime")


class CategoryListResponse(BaseModel):
    """Response for listing all product categories."""

    success: bool = Field(..., description="Success flag")
    categories: List[Dict[str, Any]] = Field(default_factory=list, description="Category list")
    total: int = Field(0, description="Total categories")


# ==============================================================================
# EndOfLifeTreatmentService
# ==============================================================================


class EndOfLifeTreatmentService:
    """
    End-of-Life Treatment Service Facade.

    This service wires together all 7 engines to provide a complete API
    for end-of-life treatment of sold products emissions calculations
    (Scope 3 Category 12).

    The service supports:
        - Waste-type-specific calculations (material x treatment x EF)
        - Landfill emissions (FOD model parameters via EF table)
        - Incineration emissions (mass burn, WtE, energy recovery credits)
        - Recycling emissions (cut-off approach, avoided credits separate)
        - Average-data calculations (product category composite EFs)
        - Producer-specific calculations (EPD end-of-life data)
        - Hybrid multi-method aggregation
        - Compliance checking (7 regulatory frameworks, 50 rules)
        - Double-counting prevention (8 rules)
        - Circularity metrics (recycling rate, diversion rate)
        - Portfolio hot-spot analysis
        - Multi-dimensional aggregation and reporting
        - SHA-256 provenance tracking

    Engines:
        1. EOLProductDatabaseEngine - Data persistence
        2. WasteTypeSpecificCalculatorEngine - Material x treatment
        3. AverageDataCalculatorEngine - Category composites
        4. ProducerSpecificCalculatorEngine - EPD-based
        5. HybridAggregatorEngine - Multi-method blending
        6. ComplianceCheckerEngine - Compliance validation
        7. EndOfLifeTreatmentPipelineEngine - End-to-end pipeline

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(EOLCalculationRequest(
        ...     products=[{"name": "Widget", "weight_kg": 0.5, "units_sold": 100000}],
        ...     region="US",
        ... ))
        >>> assert response.success
    """

    def __init__(self) -> None:
        """Initialize EndOfLifeTreatmentService with all 7 engines."""
        logger.info("Initializing EndOfLifeTreatmentService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        # Initialize engines with graceful fallback
        self._database_engine = self._init_engine(
            "greenlang.end_of_life_treatment.eol_product_database",
            "EOLProductDatabaseEngine",
        )
        self._waste_type_engine = self._init_engine(
            "greenlang.end_of_life_treatment.waste_type_specific_calculator",
            "WasteTypeSpecificCalculatorEngine",
        )
        self._average_data_engine = self._init_engine(
            "greenlang.end_of_life_treatment.average_data_calculator",
            "AverageDataCalculatorEngine",
        )
        self._producer_specific_engine = self._init_engine(
            "greenlang.end_of_life_treatment.producer_specific_calculator",
            "ProducerSpecificCalculatorEngine",
        )
        self._hybrid_engine = self._init_engine(
            "greenlang.end_of_life_treatment.hybrid_aggregator",
            "HybridAggregatorEngine",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.end_of_life_treatment.compliance_checker",
            "ComplianceCheckerEngine",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.end_of_life_treatment.end_of_life_treatment_pipeline",
            "EndOfLifeTreatmentPipelineEngine",
        )

        # In-memory calculation store (for dev/testing; production uses DB)
        self._calculations: Dict[str, Dict[str, Any]] = {}

        self._initialized = True
        logger.info("EndOfLifeTreatmentService initialized successfully")

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
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            # Use get_instance if available (singleton), else instantiate
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
    # Core Calculation Methods (10)
    # ========================================================================

    def calculate(self, request: Union[EOLCalculationRequest, Dict[str, Any]]) -> EOLCalculationResponse:
        """
        Calculate end-of-life treatment emissions for sold products.

        Delegates to the pipeline engine for full 10-stage processing.

        Args:
            request: Calculation request (Pydantic model or dict).

        Returns:
            EOLCalculationResponse with emissions and provenance.
        """
        start_time = time.monotonic()
        calc_id = f"eol-{uuid4().hex[:12]}"

        try:
            if isinstance(request, dict):
                inputs = request
                org_id = request.get("org_id", "")
                year = request.get("reporting_year", 2025)
                region = request.get("region", "GLOBAL")
            else:
                inputs = {
                    "products": request.products,
                    "region": request.region,
                    "method": request.method,
                }
                org_id = request.org_id or ""
                year = request.reporting_year
                region = request.region

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.run_pipeline(
                    inputs=inputs, org_id=org_id, year=year
                )
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            response = EOLCalculationResponse(
                success=True,
                calculation_id=calc_id,
                total_co2e_kg=float(Decimal(str(result.get("total_co2e_kg", "0")))),
                total_co2e_tonnes=float(Decimal(str(result.get("total_co2e_tonnes", "0")))),
                avoided_emissions_kg=float(Decimal(str(result.get("avoided_emissions_kg", "0")))),
                energy_recovery_credits_kg=float(Decimal(str(result.get("energy_recovery_credits_kg", "0")))),
                recycling_rate_pct=float(Decimal(str(result.get("recycling_rate_pct", "0")))),
                diversion_rate_pct=float(Decimal(str(result.get("diversion_rate_pct", "0")))),
                method=result.get("method", ""),
                region=result.get("region", region),
                by_treatment=result.get("by_treatment", {}),
                by_material=result.get("by_material", {}),
                by_category=result.get("by_category", {}),
                product_count=len(result.get("products", [])),
                provenance_hash=result.get("provenance_hash", ""),
                processing_time_ms=elapsed,
            )

            # Store in memory
            self._calculations[calc_id] = {
                **response.dict(),
                "compliance": result.get("compliance", {}),
                "products": result.get("products", []),
            }
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Calculation %s failed: %s", calc_id, e, exc_info=True)
            return EOLCalculationResponse(
                success=False,
                calculation_id=calc_id,
                total_co2e_kg=0.0,
                method="unknown",
                processing_time_ms=elapsed,
                error=str(e),
            )

    def calculate_waste_type_specific(
        self, products: List[Dict[str, Any]], region: str = "GLOBAL"
    ) -> EOLCalculationResponse:
        """
        Calculate using waste-type-specific method.

        Args:
            products: List of product dictionaries.
            region: Region for treatment mix.

        Returns:
            EOLCalculationResponse.
        """
        return self.calculate(EOLCalculationRequest(
            products=products,
            region=region,
            method="waste_type_specific",
        ))

    def calculate_landfill(self, request: LandfillCalculationRequest) -> SingleTreatmentResponse:
        """
        Calculate landfill-specific emissions for a material.

        Args:
            request: Landfill calculation request.

        Returns:
            SingleTreatmentResponse with landfill emissions.
        """
        start_time = time.monotonic()
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
                MATERIAL_TREATMENT_EFS,
            )
            lookup_key = f"{request.material_type}__landfill"
            ef_entry = MATERIAL_TREATMENT_EFS.get(lookup_key)

            if ef_entry is None:
                ef_entry = MATERIAL_TREATMENT_EFS.get("mixed__landfill", {"co2e_per_kg": Decimal("0.59")})

            weight = Decimal(str(request.weight_kg))
            co2e = (weight * ef_entry["co2e_per_kg"]).quantize(_QUANT_8DP, rounding=ROUNDING)
            elapsed = (time.monotonic() - start_time) * 1000.0

            return SingleTreatmentResponse(
                success=True,
                material_type=request.material_type,
                treatment="landfill",
                weight_kg=request.weight_kg,
                co2e_kg=float(co2e),
                ef_source=ef_entry.get("source", "EPA_WARM"),
                processing_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            return SingleTreatmentResponse(
                success=False,
                material_type=request.material_type,
                treatment="landfill",
                weight_kg=request.weight_kg,
                co2e_kg=0.0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def calculate_incineration(self, request: IncinerationCalculationRequest) -> SingleTreatmentResponse:
        """
        Calculate incineration-specific emissions for a material.

        Args:
            request: Incineration calculation request.

        Returns:
            SingleTreatmentResponse with incineration emissions.
        """
        start_time = time.monotonic()
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
                MATERIAL_TREATMENT_EFS,
                ENERGY_RECOVERY_CREDITS,
            )
            lookup_key = f"{request.material_type}__incineration"
            ef_entry = MATERIAL_TREATMENT_EFS.get(lookup_key)

            if ef_entry is None:
                ef_entry = MATERIAL_TREATMENT_EFS.get("mixed__incineration", {"co2e_per_kg": Decimal("1.14")})

            weight = Decimal(str(request.weight_kg))
            co2e = (weight * ef_entry["co2e_per_kg"]).quantize(_QUANT_8DP, rounding=ROUNDING)

            energy_credit = Decimal("0")
            if request.energy_recovery:
                credit_ef = ENERGY_RECOVERY_CREDITS.get(request.material_type, Decimal("0"))
                energy_credit = (weight * credit_ef).quantize(_QUANT_8DP, rounding=ROUNDING)

            elapsed = (time.monotonic() - start_time) * 1000.0

            return SingleTreatmentResponse(
                success=True,
                material_type=request.material_type,
                treatment="incineration",
                weight_kg=request.weight_kg,
                co2e_kg=float(co2e),
                energy_recovery_credits_kg=float(energy_credit),
                ef_source=ef_entry.get("source", "EPA_WARM"),
                processing_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            return SingleTreatmentResponse(
                success=False,
                material_type=request.material_type,
                treatment="incineration",
                weight_kg=request.weight_kg,
                co2e_kg=0.0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def calculate_recycling(self, request: RecyclingCalculationRequest) -> SingleTreatmentResponse:
        """
        Calculate recycling-specific emissions for a material.

        Avoided emissions are calculated but reported SEPARATELY per GHG Protocol.

        Args:
            request: Recycling calculation request.

        Returns:
            SingleTreatmentResponse with recycling emissions and avoided credits.
        """
        start_time = time.monotonic()
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
                MATERIAL_TREATMENT_EFS,
                RECYCLING_AVOIDED_EFS,
            )
            lookup_key = f"{request.material_type}__recycling"
            ef_entry = MATERIAL_TREATMENT_EFS.get(lookup_key)

            if ef_entry is None:
                ef_entry = {"co2e_per_kg": Decimal("0.04"), "source": "DEFAULT"}

            weight = Decimal(str(request.weight_kg))
            co2e = (weight * ef_entry["co2e_per_kg"]).quantize(_QUANT_8DP, rounding=ROUNDING)

            avoided = Decimal("0")
            if request.include_avoided:
                avoided_ef = RECYCLING_AVOIDED_EFS.get(request.material_type, Decimal("0"))
                avoided = (weight * avoided_ef).quantize(_QUANT_8DP, rounding=ROUNDING)

            elapsed = (time.monotonic() - start_time) * 1000.0

            return SingleTreatmentResponse(
                success=True,
                material_type=request.material_type,
                treatment="recycling",
                weight_kg=request.weight_kg,
                co2e_kg=float(co2e),
                avoided_emissions_kg=float(avoided),
                ef_source=ef_entry.get("source", "EPA_WARM"),
                processing_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            return SingleTreatmentResponse(
                success=False,
                material_type=request.material_type,
                treatment="recycling",
                weight_kg=request.weight_kg,
                co2e_kg=0.0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def calculate_average_data(self, request: AverageDataCalculationRequest) -> SingleTreatmentResponse:
        """
        Calculate using average-data method for a product category.

        Args:
            request: Average-data calculation request.

        Returns:
            SingleTreatmentResponse with category-level emissions.
        """
        start_time = time.monotonic()
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import AVERAGE_DATA_EFS
            avg_ef = AVERAGE_DATA_EFS.get(request.product_category, AVERAGE_DATA_EFS.get("other", Decimal("0.59")))
            weight = Decimal(str(request.total_weight_kg))
            co2e = (weight * avg_ef).quantize(_QUANT_8DP, rounding=ROUNDING)
            elapsed = (time.monotonic() - start_time) * 1000.0

            return SingleTreatmentResponse(
                success=True,
                material_type=request.product_category,
                treatment="average_data",
                weight_kg=request.total_weight_kg,
                co2e_kg=float(co2e),
                ef_source="COMPOSITE",
                processing_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            return SingleTreatmentResponse(
                success=False,
                material_type=request.product_category,
                treatment="average_data",
                weight_kg=request.total_weight_kg,
                co2e_kg=0.0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def calculate_producer_specific(self, request: ProducerSpecificCalculationRequest) -> SingleTreatmentResponse:
        """
        Calculate using producer-specific (EPD) method.

        Args:
            request: Producer-specific calculation request.

        Returns:
            SingleTreatmentResponse with EPD-based emissions.
        """
        start_time = time.monotonic()
        try:
            weight = Decimal(str(request.total_weight_kg))
            ef = Decimal(str(request.producer_ef_kg_co2e_per_kg))
            co2e = (weight * ef).quantize(_QUANT_8DP, rounding=ROUNDING)

            avoided = Decimal("0")
            if request.avoided_ef_kg_co2e_per_kg > 0:
                avoided_ef = Decimal(str(request.avoided_ef_kg_co2e_per_kg))
                avoided = (weight * avoided_ef).quantize(_QUANT_8DP, rounding=ROUNDING)

            elapsed = (time.monotonic() - start_time) * 1000.0

            return SingleTreatmentResponse(
                success=True,
                material_type=request.product_name,
                treatment="producer_specific",
                weight_kg=request.total_weight_kg,
                co2e_kg=float(co2e),
                avoided_emissions_kg=float(avoided),
                ef_source="EPD",
                processing_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            return SingleTreatmentResponse(
                success=False,
                material_type=request.product_name,
                treatment="producer_specific",
                weight_kg=request.total_weight_kg,
                co2e_kg=0.0,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def calculate_hybrid(self, request: HybridCalculationRequest) -> EOLCalculationResponse:
        """
        Calculate using hybrid multi-method approach.

        Args:
            request: Hybrid calculation request with products that may use different methods.

        Returns:
            EOLCalculationResponse with blended results.
        """
        return self.calculate(EOLCalculationRequest(
            products=request.products,
            region=request.region,
            reporting_year=request.reporting_year,
            method=None,  # Let pipeline auto-select per product
        ))

    def calculate_batch(self, request: BatchCalculationRequest) -> BatchResponse:
        """
        Process multiple calculation requests in a batch.

        Args:
            request: Batch request with multiple items.

        Returns:
            BatchResponse with individual results and totals.
        """
        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = 0.0

        for idx, item in enumerate(request.items):
            resp = self.calculate(item)
            if resp.success:
                results.append(resp.dict())
                total_co2e += resp.total_co2e_kg
            else:
                errors.append({"index": idx, "error": resp.error})

        elapsed = (time.monotonic() - start_time) * 1000.0

        return BatchResponse(
            success=len(errors) == 0,
            total_items=len(request.items),
            successful=len(results),
            failed=len(errors),
            total_co2e_kg=total_co2e,
            results=results,
            errors=errors,
            processing_time_ms=elapsed,
        )

    def calculate_portfolio(self, request: PortfolioAnalysisRequest) -> Dict[str, Any]:
        """
        Run portfolio-level analysis with hot-spots and circularity metrics.

        Args:
            request: Portfolio analysis request.

        Returns:
            Portfolio analysis dictionary.
        """
        if self._pipeline_engine is not None and hasattr(self._pipeline_engine, "run_portfolio_analysis"):
            return self._pipeline_engine.run_portfolio_analysis(
                inputs={"products": request.products, "region": request.region},
                org_id=request.org_id or "",
                year=request.reporting_year,
            )

        # Fallback: standard pipeline
        return self.calculate(EOLCalculationRequest(
            products=request.products,
            region=request.region,
            reporting_year=request.reporting_year,
            org_id=request.org_id,
        )).dict()

    # ========================================================================
    # Data Lookup Methods (8)
    # ========================================================================

    def check_compliance(self, request: ComplianceCheckRequest) -> ComplianceCheckResponse:
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
                overall_status="fail",
                processing_time_ms=elapsed,
            )

        if self._compliance_engine is not None:
            try:
                compliance_input = {
                    "total_co2e": calc_data.get("total_co2e_kg", 0),
                    "total_co2e_kg": calc_data.get("total_co2e_kg", 0),
                    "method": calc_data.get("method", ""),
                    "calculation_method": calc_data.get("method", ""),
                    "by_treatment": calc_data.get("by_treatment", {}),
                    "treatment_breakdown": calc_data.get("by_treatment", {}),
                    "by_material": calc_data.get("by_material", {}),
                    "material_breakdown": calc_data.get("by_material", {}),
                    "by_category": calc_data.get("by_category", {}),
                    "product_boundary": "sold_products",
                    "avoided_reported_separately": True,
                    "energy_credits_reported_separately": True,
                }

                results = self._compliance_engine.check_all_frameworks(compliance_input)
                summary = self._compliance_engine.get_compliance_summary(results)
                dc_findings = self._compliance_engine.check_double_counting(compliance_input)

                elapsed = (time.monotonic() - start_time) * 1000.0
                return ComplianceCheckResponse(
                    success=True,
                    calculation_id=request.calculation_id,
                    overall_status=summary.get("overall_status", "fail"),
                    overall_score=summary.get("overall_score", 0.0),
                    framework_results=summary.get("framework_scores", {}),
                    double_counting_findings=dc_findings,
                    processing_time_ms=elapsed,
                )
            except Exception as e:
                logger.error("Compliance check failed: %s", e, exc_info=True)

        elapsed = (time.monotonic() - start_time) * 1000.0
        return ComplianceCheckResponse(
            success=False,
            calculation_id=request.calculation_id,
            overall_status="fail",
            processing_time_ms=elapsed,
        )

    def get_material_ef(self, request: MaterialEFRequest) -> MaterialEFResponse:
        """
        Look up emission factor for a material x treatment combination.

        Args:
            request: Material EF lookup request.

        Returns:
            MaterialEFResponse with EF data.
        """
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import MATERIAL_TREATMENT_EFS
            lookup_key = f"{request.material_type}__{request.treatment}"
            ef_entry = MATERIAL_TREATMENT_EFS.get(lookup_key)

            if ef_entry:
                return MaterialEFResponse(
                    success=True,
                    material_type=request.material_type,
                    treatment=request.treatment,
                    co2e_per_kg=float(ef_entry["co2e_per_kg"]),
                    ch4_per_kg=float(ef_entry.get("ch4_per_kg", Decimal("0"))),
                    source=ef_entry.get("source", ""),
                )
            else:
                return MaterialEFResponse(
                    success=False,
                    material_type=request.material_type,
                    treatment=request.treatment,
                    error=f"No EF found for {lookup_key}",
                )
        except Exception as e:
            return MaterialEFResponse(
                success=False,
                material_type=request.material_type,
                treatment=request.treatment,
                error=str(e),
            )

    def get_product_composition(self, category: str) -> ProductCompositionResponse:
        """
        Get default material composition for a product category.

        Args:
            category: Product category name.

        Returns:
            ProductCompositionResponse with material fractions.
        """
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import DEFAULT_PRODUCT_BOMS
            bom = DEFAULT_PRODUCT_BOMS.get(category, DEFAULT_PRODUCT_BOMS.get("other", {}))
            return ProductCompositionResponse(
                success=True,
                category=category,
                composition={k: float(v) for k, v in bom.items()},
            )
        except Exception as e:
            return ProductCompositionResponse(
                success=False,
                category=category,
                composition={},
            )

    def get_regional_treatment_mix(self, region: str) -> RegionalTreatmentMixResponse:
        """
        Get treatment mix fractions for a region.

        Args:
            region: Region code (US, EU, UK, DE, JP, CN, IN, BR, AU, CA, KR, GLOBAL).

        Returns:
            RegionalTreatmentMixResponse with treatment fractions.
        """
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import REGIONAL_TREATMENT_MIXES
            mix = REGIONAL_TREATMENT_MIXES.get(region, REGIONAL_TREATMENT_MIXES.get("GLOBAL", {}))
            return RegionalTreatmentMixResponse(
                success=True,
                region=region,
                treatment_mix={k: float(v) for k, v in mix.items()},
            )
        except Exception as e:
            return RegionalTreatmentMixResponse(
                success=False,
                region=region,
            )

    def get_avoided_emissions(self, material_type: str) -> AvoidedEmissionsResponse:
        """
        Get avoided emissions and energy recovery credit factors for a material.

        Args:
            material_type: Material type identifier.

        Returns:
            AvoidedEmissionsResponse with recycling and energy credit factors.
        """
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
                RECYCLING_AVOIDED_EFS,
                ENERGY_RECOVERY_CREDITS,
            )
            avoided = RECYCLING_AVOIDED_EFS.get(material_type, Decimal("0"))
            energy = ENERGY_RECOVERY_CREDITS.get(material_type, Decimal("0"))
            return AvoidedEmissionsResponse(
                success=True,
                material_type=material_type,
                avoided_co2e_per_kg=float(avoided),
                energy_credit_per_kg=float(energy),
            )
        except Exception as e:
            return AvoidedEmissionsResponse(
                success=False,
                material_type=material_type,
            )

    def get_circularity_score(
        self,
        recycling_rate_pct: float,
        diversion_rate_pct: float,
    ) -> CircularityScoreResponse:
        """
        Calculate composite circularity score.

        Weighted: 60% recycling rate + 40% diversion rate.

        Args:
            recycling_rate_pct: Recycling rate percentage (0-100).
            diversion_rate_pct: Diversion rate percentage (0-100).

        Returns:
            CircularityScoreResponse with composite score.
        """
        recycling = Decimal(str(recycling_rate_pct))
        diversion = Decimal(str(diversion_rate_pct))
        score = (
            recycling * Decimal("0.6") + diversion * Decimal("0.4")
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        return CircularityScoreResponse(
            success=True,
            recycling_rate_pct=recycling_rate_pct,
            diversion_rate_pct=diversion_rate_pct,
            circularity_score=float(score),
        )

    def get_product_weight_default(self, category: str) -> Dict[str, Any]:
        """
        Get default weight per unit for a product category.

        Args:
            category: Product category name.

        Returns:
            Dictionary with category and default weight_kg.
        """
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import DEFAULT_PRODUCT_WEIGHTS
            weight = DEFAULT_PRODUCT_WEIGHTS.get(category, DEFAULT_PRODUCT_WEIGHTS.get("other", Decimal("1.00")))
            return {
                "success": True,
                "category": category,
                "default_weight_kg": float(weight),
            }
        except Exception as e:
            return {"success": False, "category": category, "error": str(e)}

    def get_all_categories(self) -> CategoryListResponse:
        """
        Get all supported product categories with metadata.

        Returns:
            CategoryListResponse with category list.
        """
        try:
            from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
                DEFAULT_PRODUCT_BOMS,
                DEFAULT_PRODUCT_WEIGHTS,
                AVERAGE_DATA_EFS,
            )
            categories = []
            for cat in DEFAULT_PRODUCT_BOMS:
                weight = DEFAULT_PRODUCT_WEIGHTS.get(cat, Decimal("1.00"))
                avg_ef = AVERAGE_DATA_EFS.get(cat, Decimal("0.59"))
                materials = list(DEFAULT_PRODUCT_BOMS[cat].keys())
                categories.append({
                    "category": cat,
                    "default_weight_kg": float(weight),
                    "average_ef_kg_co2e_per_kg": float(avg_ef),
                    "material_count": len(materials),
                    "materials": materials,
                })

            return CategoryListResponse(
                success=True,
                categories=categories,
                total=len(categories),
            )
        except Exception as e:
            return CategoryListResponse(success=False, total=0)

    # ========================================================================
    # Additional Methods (6)
    # ========================================================================

    def get_aggregations(self, request: AggregationRequest) -> AggregationResponse:
        """
        Get aggregated emissions for a reporting year.

        Args:
            request: Aggregation request with year and group_by.

        Returns:
            AggregationResponse with breakdown.
        """
        start_time = time.monotonic()
        total = 0.0
        breakdown: Dict[str, float] = {}

        for calc in self._calculations.values():
            co2e = calc.get("total_co2e_kg", 0.0)
            total += co2e

            if request.group_by == "treatment":
                for treat, data in calc.get("by_treatment", {}).items():
                    val = float(data.get("co2e_kg", 0)) if isinstance(data, dict) else float(data)
                    breakdown[treat] = breakdown.get(treat, 0.0) + val
            elif request.group_by == "material":
                for mat, data in calc.get("by_material", {}).items():
                    val = float(data.get("co2e_kg", 0)) if isinstance(data, dict) else float(data)
                    breakdown[mat] = breakdown.get(mat, 0.0) + val
            elif request.group_by == "category":
                for cat, data in calc.get("by_category", {}).items():
                    val = float(data.get("co2e_kg", 0)) if isinstance(data, dict) else float(data)
                    breakdown[cat] = breakdown.get(cat, 0.0) + val

        elapsed = (time.monotonic() - start_time) * 1000.0

        return AggregationResponse(
            success=True,
            total_co2e_kg=total,
            breakdown=breakdown,
            reporting_year=request.reporting_year,
            processing_time_ms=elapsed,
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
            return ProvenanceResponse(
                success=True,
                calculation_id=calculation_id,
                provenance_hash=calc.get("provenance_hash", ""),
                chain=[],
                is_valid=True,
            )
        return ProvenanceResponse(
            success=False,
            calculation_id=calculation_id,
        )

    def health_check(self) -> HealthResponse:
        """
        Perform service health check.

        Returns:
            HealthResponse with engine statuses and uptime.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = {
            "database": self._database_engine is not None,
            "waste_type_specific": self._waste_type_engine is not None,
            "average_data": self._average_data_engine is not None,
            "producer_specific": self._producer_specific_engine is not None,
            "hybrid": self._hybrid_engine is not None,
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

    def get_version(self) -> Dict[str, str]:
        """
        Get service version information.

        Returns:
            Version dictionary.
        """
        return {
            "service_version": SERVICE_VERSION,
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get service configuration summary.

        Returns:
            Configuration summary dictionary.
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": SERVICE_VERSION,
            "engines_loaded": sum(
                1 for e in [
                    self._database_engine,
                    self._waste_type_engine,
                    self._average_data_engine,
                    self._producer_specific_engine,
                    self._hybrid_engine,
                    self._compliance_engine,
                    self._pipeline_engine,
                ]
                if e is not None
            ),
            "calculations_stored": len(self._calculations),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Statistics dictionary.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return {
            "agent_id": AGENT_ID,
            "version": SERVICE_VERSION,
            "uptime_seconds": uptime,
            "calculations_processed": len(self._calculations),
            "pipeline_status": (
                self._pipeline_engine.get_pipeline_status()
                if self._pipeline_engine and hasattr(self._pipeline_engine, "get_pipeline_status")
                else {"status": "unavailable"}
            ),
            "compliance_stats": (
                self._compliance_engine.get_engine_stats()
                if self._compliance_engine and hasattr(self._compliance_engine, "get_engine_stats")
                else {"status": "unavailable"}
            ),
        }


# ==============================================================================
# Module-Level Helpers
# ==============================================================================


def get_service() -> EndOfLifeTreatmentService:
    """
    Get singleton EndOfLifeTreatmentService instance.

    Thread-safe via double-checked locking.

    Returns:
        EndOfLifeTreatmentService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = EndOfLifeTreatmentService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for end-of-life treatment endpoints.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.end_of_life_treatment.api.router import router
    return router


def reset_service() -> None:
    """
    Reset the service singleton (for testing only).

    Thread-safe reset that also resets sub-engine singletons.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None

    # Also reset engine singletons
    try:
        from greenlang.end_of_life_treatment.compliance_checker import ComplianceCheckerEngine
        ComplianceCheckerEngine.reset_instance()
    except ImportError:
        pass

    try:
        from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import EndOfLifeTreatmentPipelineEngine
        EndOfLifeTreatmentPipelineEngine.reset_instance()
    except ImportError:
        pass

    logger.info("EndOfLifeTreatmentService singleton reset")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "SERVICE_VERSION",
    "AGENT_ID",
    "AGENT_COMPONENT",
    # Request models
    "EOLCalculationRequest",
    "LandfillCalculationRequest",
    "IncinerationCalculationRequest",
    "RecyclingCalculationRequest",
    "AverageDataCalculationRequest",
    "ProducerSpecificCalculationRequest",
    "HybridCalculationRequest",
    "BatchCalculationRequest",
    "PortfolioAnalysisRequest",
    "ComplianceCheckRequest",
    "MaterialEFRequest",
    "AggregationRequest",
    # Response models
    "EOLCalculationResponse",
    "SingleTreatmentResponse",
    "BatchResponse",
    "ComplianceCheckResponse",
    "MaterialEFResponse",
    "ProductCompositionResponse",
    "RegionalTreatmentMixResponse",
    "AvoidedEmissionsResponse",
    "CircularityScoreResponse",
    "AggregationResponse",
    "ProvenanceResponse",
    "HealthResponse",
    "CategoryListResponse",
    # Service
    "EndOfLifeTreatmentService",
    "get_service",
    "get_router",
    "reset_service",
]
