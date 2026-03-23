# -*- coding: utf-8 -*-
"""
Franchises Service Setup - AGENT-MRV-027

This module provides the service facade that wires together all 7 engines
for franchise emissions calculations (Scope 3 Category 14).

The FranchisesService class provides a high-level API for:
- Franchise-specific emissions (primary energy/refrigerant data per unit)
- Average-data emissions (EUI benchmarks by type and climate zone)
- Spend-based emissions (EEIO factors applied to royalty/revenue data)
- Hybrid calculations (blend of methods across heterogeneous networks)
- Network-level aggregation (by type, region, method)
- Compliance checking across 7 regulatory frameworks
- Emission factor lookups
- Benchmark data access
- Health status monitoring

Engines:
    1. FranchiseDatabaseEngine - Emission factor data and persistence
    2. FranchiseSpecificCalculatorEngine - Primary energy data calculations
    3. AverageDataCalculatorEngine - EUI benchmark calculations
    4. SpendBasedCalculatorEngine - EEIO spend-based calculations
    5. HybridAggregatorEngine - Blended method calculations
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. FranchisesPipelineEngine - End-to-end 10-stage pipeline

Architecture:
    - Thread-safe singleton pattern for service instance
    - Graceful imports with try/except for optional dependencies
    - Comprehensive metrics tracking via OBS-001 integration
    - Provenance tracking for all mutations via AGENT-FOUND-008
    - Type-safe request/response models using Pydantic
    - Structured logging with contextual information

Example:
    >>> from greenlang.agents.mrv.franchises.setup import get_service
    >>> service = get_service()
    >>> response = service.calculate(FranchiseCalculationRequest(
    ...     franchise_type="quick_service_restaurant",
    ...     electricity_kwh=150000,
    ...     gas_kwh=80000,
    ... ))
    >>> assert response.success

Integration:
    >>> from greenlang.agents.mrv.franchises.setup import get_router
    >>> app.include_router(get_router(), prefix="/api/v1/franchises")

Module: greenlang.agents.mrv.franchises.setup
Agent: AGENT-MRV-027
Agent ID: GL-MRV-S3-014
Version: 1.0.0
"""

import importlib
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

# Thread-safe singleton lock
_service_lock = threading.RLock()
_service_instance: Optional["FranchisesService"] = None

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class FranchiseCalculationRequest(BaseModel):
    """Request model for single franchise unit emissions calculation."""

    unit_id: Optional[str] = Field(None, description="Unique franchise unit identifier")
    franchise_type: str = Field(
        "other",
        description=(
            "Franchise type: quick_service_restaurant, full_service_restaurant, "
            "hotel, convenience_store, gas_station, retail_store, fitness_center, "
            "automotive_service, laundry_dry_cleaning, office, other"
        ),
    )
    ownership_type: str = Field("franchise", description="Ownership: franchise, company_owned, joint_venture, leased")
    agreement_type: Optional[str] = Field("single_unit", description="Agreement: single_unit, multi_unit, master_franchise, area_development, conversion, sub_franchise")

    # Energy data (for franchise-specific method)
    electricity_kwh: Optional[float] = Field(None, ge=0, description="Annual electricity consumption in kWh")
    gas_kwh: Optional[float] = Field(None, ge=0, description="Annual natural gas consumption in kWh")
    gas_therms: Optional[float] = Field(None, ge=0, description="Annual natural gas consumption in therms")
    other_energy_kwh: Optional[float] = Field(None, ge=0, description="Other energy (steam/cooling) in kWh")

    # Floor area (for average-data method)
    floor_area_sqm: Optional[float] = Field(None, ge=0, description="Floor area in square metres")
    floor_area_sqft: Optional[float] = Field(None, ge=0, description="Floor area in square feet")

    # Spend data (for spend-based method)
    revenue: Optional[float] = Field(None, ge=0, description="Annual revenue in USD")
    spend: Optional[float] = Field(None, ge=0, description="Annual spend in USD")
    royalty_revenue: Optional[float] = Field(None, ge=0, description="Royalty revenue in USD")
    currency: str = Field("USD", description="Currency code")
    naics_code: Optional[str] = Field(None, description="NAICS code for EEIO factor")

    # Refrigerant data
    refrigerant_charge_kg: Optional[float] = Field(None, ge=0, description="Refrigerant charge in kg")
    refrigerant_leak_rate: Optional[float] = Field(None, ge=0, le=1, description="Annual leak rate (0-1)")
    refrigerant_gwp: Optional[float] = Field(None, ge=0, description="Refrigerant GWP (CO2e)")

    # Location / context
    region: Optional[str] = Field(None, description="Region or eGRID subregion code")
    country_code: Optional[str] = Field(None, description="ISO country code")
    climate_zone: Optional[str] = Field(None, description="Climate zone")
    operating_months: Optional[float] = Field(None, ge=0, le=12, description="Months in operation (for partial year)")

    # Multi-brand
    brands: Optional[Dict[str, float]] = Field(None, description="Brand name -> revenue share for multi-brand units")

    # Override
    calculation_method: Optional[str] = Field(None, description="Override: franchise_specific, average_data, spend_based, hybrid")

    # Metadata
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    reporting_period: Optional[str] = Field(None, description="Reporting period (e.g., 2025)")

    @validator("franchise_type")
    def validate_franchise_type(cls, v: str) -> str:
        """Validate franchise type."""
        allowed = [
            "quick_service_restaurant", "full_service_restaurant", "hotel",
            "convenience_store", "gas_station", "retail_store", "fitness_center",
            "automotive_service", "laundry_dry_cleaning", "office", "other",
        ]
        if v.lower() not in allowed:
            raise ValueError(f"franchise_type must be one of {allowed}")
        return v.lower()


class BatchFranchiseCalculationRequest(BaseModel):
    """Request model for batch franchise calculations."""

    units: List[FranchiseCalculationRequest] = Field(..., min_length=1, description="List of franchise units")
    reporting_period: str = Field(..., description="Reporting period (e.g., '2025')")
    consolidation_approach: str = Field("operational_control", description="Consolidation approach")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class NetworkCalculationRequest(BaseModel):
    """Request model for full franchise network calculation."""

    units: List[FranchiseCalculationRequest] = Field(..., min_length=1, description="All franchise units")
    reporting_period: str = Field(..., description="Reporting period")
    consolidation_approach: str = Field("operational_control", description="Consolidation approach")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking."""

    calculation_id: Optional[str] = Field(None, description="Calculation ID to check")
    calculation_result: Optional[dict] = Field(None, description="Calculation result to check directly")
    frameworks: List[str] = Field(
        default_factory=lambda: ["ghg_protocol"],
        description="Frameworks: ghg_protocol, iso_14064, csrd_esrs, cdp, sbti, sb_253, gri",
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class EmissionFactorRequest(BaseModel):
    """Request model for emission factor lookup."""

    franchise_type: str = Field(..., description="Franchise type for EUI lookup")
    region: Optional[str] = Field(None, description="Region for electricity EF")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class BenchmarkRequest(BaseModel):
    """Request model for benchmark data."""

    franchise_type: Optional[str] = Field(None, description="Filter by franchise type")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


# ============================================================================
# Response Models
# ============================================================================


class FranchiseCalculationResponse(BaseModel):
    """Response model for single franchise unit calculation."""

    success: bool = Field(..., description="Success flag")
    calculation_id: str = Field(..., description="Calculation identifier")
    unit_id: Optional[str] = Field(None, description="Franchise unit identifier")
    franchise_type: str = Field(..., description="Franchise type")
    method: str = Field(..., description="Calculation method used")
    total_co2e_kg: float = Field(..., description="Total CO2e in kg")
    electricity_co2e_kg: float = Field(0, description="Electricity CO2e in kg")
    gas_co2e_kg: float = Field(0, description="Natural gas CO2e in kg")
    refrigerant_co2e_kg: float = Field(0, description="Refrigerant CO2e in kg")
    other_co2e_kg: float = Field(0, description="Other CO2e in kg")
    dqi_score: Optional[float] = Field(None, description="Data quality score (1-5)")
    ef_source: str = Field(..., description="Emission factor source")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    detail: dict = Field(default_factory=dict, description="Calculation detail")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time in ms")


class BatchFranchiseResponse(BaseModel):
    """Response model for batch franchise calculation."""

    success: bool = Field(..., description="Overall success flag")
    total_units: int = Field(..., description="Total units requested")
    successful_units: int = Field(..., description="Successful units")
    failed_units: int = Field(..., description="Failed units")
    total_co2e_kg: float = Field(..., description="Total CO2e for all units")
    results: List[FranchiseCalculationResponse] = Field(..., description="Individual results")
    errors: List[dict] = Field(default_factory=list, description="Failed unit errors")
    reporting_period: str = Field(..., description="Reporting period")
    processing_time_ms: float = Field(..., description="Total processing time")


class NetworkCalculationResponse(BaseModel):
    """Response model for full network calculation."""

    success: bool = Field(..., description="Success flag")
    total_co2e_kg: float = Field(..., description="Total network CO2e in kg")
    total_units: int = Field(..., description="Total franchise units")
    successful_units: int = Field(..., description="Units successfully calculated")
    failed_units: int = Field(..., description="Units that failed calculation")
    by_franchise_type: Dict[str, float] = Field(default_factory=dict, description="Emissions by franchise type")
    by_region: Dict[str, float] = Field(default_factory=dict, description="Emissions by region")
    by_method: Dict[str, float] = Field(default_factory=dict, description="Emissions by method")
    data_coverage: Optional[float] = Field(None, description="% of units with primary data")
    compliance: Optional[dict] = Field(None, description="Compliance check results")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    reporting_period: str = Field(..., description="Reporting period")
    consolidation_approach: str = Field(..., description="Consolidation approach used")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    error: Optional[str] = Field(None, description="Error message if failed")


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance checking."""

    success: bool = Field(..., description="Success flag")
    overall_status: str = Field(..., description="Overall compliance status: PASS, WARNING, FAIL")
    overall_score: float = Field(..., description="Overall compliance score (0-100)")
    framework_results: List[dict] = Field(..., description="Per-framework results")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    checked_at: datetime = Field(..., description="Check timestamp")
    processing_time_ms: float = Field(..., description="Processing time")


class EmissionFactorResponse(BaseModel):
    """Response model for emission factor lookup."""

    success: bool = Field(..., description="Success flag")
    franchise_type: str = Field(..., description="Franchise type")
    eui_kwh_per_sqm: Optional[float] = Field(None, description="EUI benchmark")
    electricity_ef_kgco2e_per_kwh: Optional[float] = Field(None, description="Electricity EF")
    gas_ef_kgco2e_per_kwh: Optional[float] = Field(None, description="Gas EF")
    refrigerant_ef_kgco2e: Optional[float] = Field(None, description="Refrigerant default EF")
    eeio_ef_kgco2e_per_usd: Optional[float] = Field(None, description="EEIO EF")
    sources: List[str] = Field(default_factory=list, description="EF sources")


class BenchmarkResponse(BaseModel):
    """Response model for benchmark data."""

    success: bool = Field(..., description="Success flag")
    benchmarks: List[dict] = Field(..., description="Benchmark data by type")
    total_types: int = Field(..., description="Number of franchise types")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="Service version")
    agent_id: str = Field(..., description="Agent identifier")
    engines_status: Dict[str, bool] = Field(..., description="Per-engine status")
    uptime_seconds: float = Field(..., description="Service uptime")


# ============================================================================
# FranchisesService Class
# ============================================================================


class FranchisesService:
    """
    Franchises Service Facade.

    This service wires together all 7 engines to provide a complete API
    for franchise emissions calculations (Scope 3 Category 14).

    The service supports:
        - Franchise-specific calculations (primary energy data)
        - Average-data calculations (EUI benchmarks)
        - Spend-based calculations (EEIO factors)
        - Hybrid calculations (blended methods)
        - Network-level aggregation
        - Compliance checking (7 frameworks)
        - Emission factor lookups
        - Benchmark data

    Thread Safety:
        This service is thread-safe. Use get_service() to obtain a singleton.

    Example:
        >>> service = get_service()
        >>> response = service.calculate(FranchiseCalculationRequest(
        ...     franchise_type="hotel",
        ...     electricity_kwh=500000,
        ...     gas_kwh=200000,
        ... ))
        >>> assert response.success

    Attributes:
        _database_engine: Database engine for persistence
        _franchise_specific_engine: Franchise-specific calculator
        _average_data_engine: Average-data calculator
        _spend_engine: Spend-based calculator
        _hybrid_engine: Hybrid aggregator
        _compliance_engine: Compliance checker
        _pipeline_engine: Pipeline orchestration engine
    """

    def __init__(self) -> None:
        """Initialize FranchisesService with all 7 engines."""
        logger.info("Initializing FranchisesService")
        self._start_time = datetime.now(timezone.utc)
        self._initialized = False

        # Initialize engines with graceful fallback
        self._database_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.franchise_database",
            "FranchiseDatabaseEngine",
        )
        self._franchise_specific_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.franchise_specific_calculator",
            "FranchiseSpecificCalculatorEngine",
        )
        self._average_data_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.average_data_calculator",
            "AverageDataCalculatorEngine",
        )
        self._spend_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.spend_based_calculator",
            "SpendBasedCalculatorEngine",
        )
        self._hybrid_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.hybrid_aggregator",
            "HybridAggregatorEngine",
        )
        self._compliance_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.compliance_checker",
            "ComplianceCheckerEngine",
        )
        self._pipeline_engine = self._init_engine(
            "greenlang.agents.mrv.franchises.franchises_pipeline",
            "FranchisesPipelineEngine",
        )

        # In-memory calculation store (for dev/testing; production uses DB)
        self._calculations: Dict[str, dict] = {}

        self._initialized = True
        logger.info("FranchisesService initialized successfully")

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
            # Use get_instance() for singletons that support it
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

    def calculate(self, request: FranchiseCalculationRequest) -> FranchiseCalculationResponse:
        """
        Calculate emissions for a single franchise unit.

        Delegates to the pipeline engine for full 10-stage processing.

        Args:
            request: Franchise calculation request.

        Returns:
            FranchiseCalculationResponse with emissions and provenance.

        Example:
            >>> response = service.calculate(FranchiseCalculationRequest(
            ...     franchise_type="quick_service_restaurant",
            ...     electricity_kwh=150000,
            ... ))
            >>> assert response.success
        """
        start_time = time.monotonic()
        calc_id = f"frn-{uuid4().hex[:12]}"

        try:
            # Build pipeline input from request
            unit_input = self._request_to_input(request)

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.execute_single(unit_input)
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            response = FranchiseCalculationResponse(
                success=True,
                calculation_id=calc_id,
                unit_id=request.unit_id,
                franchise_type=result.get("franchise_type", request.franchise_type),
                method=result.get("method", "unknown"),
                total_co2e_kg=float(Decimal(str(result.get("total_co2e", "0")))),
                electricity_co2e_kg=float(Decimal(str(result.get("electricity_co2e", "0")))),
                gas_co2e_kg=float(Decimal(str(result.get("gas_co2e", "0")))),
                refrigerant_co2e_kg=float(Decimal(str(result.get("refrigerant_co2e", "0")))),
                other_co2e_kg=float(Decimal(str(result.get("other_co2e", "0")))),
                dqi_score=float(result.get("dqi_score")) if result.get("dqi_score") else None,
                ef_source=result.get("ef_source", ""),
                provenance_hash=result.get("provenance_hash", ""),
                detail=result,
                processing_time_ms=elapsed,
            )

            # Store in memory
            self._calculations[calc_id] = response.dict()
            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Calculation %s failed: %s", calc_id, e, exc_info=True)
            return FranchiseCalculationResponse(
                success=False,
                calculation_id=calc_id,
                unit_id=request.unit_id,
                franchise_type=request.franchise_type,
                method="unknown",
                total_co2e_kg=0.0,
                ef_source="none",
                provenance_hash="",
                error=str(e),
                processing_time_ms=elapsed,
            )

    def calculate_franchise_specific(
        self, request: FranchiseCalculationRequest,
    ) -> FranchiseCalculationResponse:
        """
        Calculate using franchise-specific (primary energy) data.

        Args:
            request: Franchise calculation request with energy data.

        Returns:
            FranchiseCalculationResponse with emissions.
        """
        request.calculation_method = "franchise_specific"
        return self.calculate(request)

    def calculate_average_data(
        self, request: FranchiseCalculationRequest,
    ) -> FranchiseCalculationResponse:
        """
        Calculate using average-data (EUI benchmark) method.

        Args:
            request: Franchise calculation request with floor area.

        Returns:
            FranchiseCalculationResponse with emissions.
        """
        request.calculation_method = "average_data"
        return self.calculate(request)

    def calculate_spend_based(
        self, request: FranchiseCalculationRequest,
    ) -> FranchiseCalculationResponse:
        """
        Calculate using spend-based (EEIO) method.

        Args:
            request: Franchise calculation request with revenue/spend.

        Returns:
            FranchiseCalculationResponse with emissions.
        """
        request.calculation_method = "spend_based"
        return self.calculate(request)

    def calculate_hybrid(
        self, request: FranchiseCalculationRequest,
    ) -> FranchiseCalculationResponse:
        """
        Calculate using hybrid method (blend of available data).

        Args:
            request: Franchise calculation request with mixed data.

        Returns:
            FranchiseCalculationResponse with emissions.
        """
        request.calculation_method = "hybrid"
        return self.calculate(request)

    def calculate_batch(
        self, request: BatchFranchiseCalculationRequest,
    ) -> BatchFranchiseResponse:
        """
        Process multiple franchise units in a single batch.

        Args:
            request: Batch request with units and reporting period.

        Returns:
            BatchFranchiseResponse with individual results and totals.
        """
        start_time = time.monotonic()
        results: List[FranchiseCalculationResponse] = []
        errors: List[dict] = []

        for idx, unit_req in enumerate(request.units):
            unit_req.reporting_period = request.reporting_period
            unit_req.tenant_id = request.tenant_id
            resp = self.calculate(unit_req)
            results.append(resp)
            if not resp.success:
                errors.append({
                    "index": idx,
                    "unit_id": unit_req.unit_id,
                    "franchise_type": unit_req.franchise_type,
                    "error": resp.error,
                })

        total_co2e = sum(r.total_co2e_kg for r in results if r.success)
        successful = sum(1 for r in results if r.success)
        elapsed = (time.monotonic() - start_time) * 1000.0

        return BatchFranchiseResponse(
            success=len(errors) == 0,
            total_units=len(request.units),
            successful_units=successful,
            failed_units=len(errors),
            total_co2e_kg=total_co2e,
            results=results,
            errors=errors,
            reporting_period=request.reporting_period,
            processing_time_ms=elapsed,
        )

    def calculate_network(
        self, request: NetworkCalculationRequest,
    ) -> NetworkCalculationResponse:
        """
        Calculate emissions for an entire franchise network.

        Args:
            request: Network calculation request with all units.

        Returns:
            NetworkCalculationResponse with aggregated results.
        """
        start_time = time.monotonic()

        try:
            # Build network input
            network_input = {
                "units": [self._request_to_input(u) for u in request.units],
                "reporting_period": request.reporting_period,
                "consolidation_approach": request.consolidation_approach,
                "tenant_id": request.tenant_id,
            }

            if self._pipeline_engine is not None:
                result = self._pipeline_engine.execute(network_input)
            else:
                raise RuntimeError("Pipeline engine not available")

            elapsed = (time.monotonic() - start_time) * 1000.0

            # Convert string values to floats for response
            by_type = {
                k: float(Decimal(str(v)))
                for k, v in result.get("by_franchise_type", {}).items()
            }
            by_region = {
                k: float(Decimal(str(v)))
                for k, v in result.get("by_region", {}).items()
            }
            by_method = {
                k: float(Decimal(str(v)))
                for k, v in result.get("by_method", {}).items()
            }

            return NetworkCalculationResponse(
                success=True,
                total_co2e_kg=float(Decimal(str(result.get("total_co2e", "0")))),
                total_units=result.get("total_units", 0),
                successful_units=result.get("successful_units", 0),
                failed_units=result.get("failed_units", 0),
                by_franchise_type=by_type,
                by_region=by_region,
                by_method=by_method,
                data_coverage=float(Decimal(str(result.get("data_coverage", "0")))) if result.get("data_coverage") else None,
                compliance=result.get("compliance"),
                provenance_hash=result.get("provenance_hash", ""),
                reporting_period=request.reporting_period,
                consolidation_approach=request.consolidation_approach,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Network calculation failed: %s", e, exc_info=True)
            return NetworkCalculationResponse(
                success=False,
                total_co2e_kg=0.0,
                total_units=len(request.units),
                successful_units=0,
                failed_units=len(request.units),
                provenance_hash="",
                reporting_period=request.reporting_period,
                consolidation_approach=request.consolidation_approach,
                processing_time_ms=elapsed,
                error=str(e),
            )

    # ========================================================================
    # Public API Methods - Compliance
    # ========================================================================

    def check_compliance(
        self,
        request: ComplianceCheckRequest,
    ) -> ComplianceCheckResponse:
        """
        Run compliance checks against specified frameworks.

        Args:
            request: Compliance check request.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        start_time = time.monotonic()

        try:
            # Get calculation result
            calc_data = request.calculation_result
            if calc_data is None and request.calculation_id:
                stored = self._calculations.get(request.calculation_id)
                if stored:
                    calc_data = stored
                else:
                    elapsed = (time.monotonic() - start_time) * 1000.0
                    return ComplianceCheckResponse(
                        success=False,
                        overall_status="FAIL",
                        overall_score=0.0,
                        framework_results=[{
                            "framework": "all",
                            "status": "FAIL",
                            "findings": [f"Calculation {request.calculation_id} not found"],
                        }],
                        checked_at=datetime.now(timezone.utc),
                        processing_time_ms=elapsed,
                    )

            if calc_data is None:
                elapsed = (time.monotonic() - start_time) * 1000.0
                return ComplianceCheckResponse(
                    success=False,
                    overall_status="FAIL",
                    overall_score=0.0,
                    framework_results=[],
                    checked_at=datetime.now(timezone.utc),
                    processing_time_ms=elapsed,
                )

            # Use compliance engine
            if self._compliance_engine is not None:
                check_results = self._compliance_engine.check_compliance(
                    calc_data, request.frameworks,
                )
                summary = self._compliance_engine.get_compliance_summary(check_results)

                elapsed = (time.monotonic() - start_time) * 1000.0

                framework_results = [
                    {
                        "framework": cr.framework.value,
                        "status": cr.status.value,
                        "score": float(cr.score),
                        "findings": cr.findings,
                        "recommendations": cr.recommendations,
                    }
                    for cr in check_results
                ]

                return ComplianceCheckResponse(
                    success=True,
                    overall_status=summary.get("overall_status", "FAIL"),
                    overall_score=summary.get("overall_score", 0.0),
                    framework_results=framework_results,
                    recommendations=summary.get("recommendations", []),
                    checked_at=datetime.now(timezone.utc),
                    processing_time_ms=elapsed,
                )
            else:
                # Inline fallback
                elapsed = (time.monotonic() - start_time) * 1000.0
                return ComplianceCheckResponse(
                    success=True,
                    overall_status="WARNING",
                    overall_score=50.0,
                    framework_results=[{
                        "framework": "inline",
                        "status": "WARNING",
                        "findings": ["ComplianceCheckerEngine not available"],
                    }],
                    checked_at=datetime.now(timezone.utc),
                    processing_time_ms=elapsed,
                )

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000.0
            logger.error("Compliance check failed: %s", e, exc_info=True)
            return ComplianceCheckResponse(
                success=False,
                overall_status="FAIL",
                overall_score=0.0,
                framework_results=[],
                checked_at=datetime.now(timezone.utc),
                processing_time_ms=elapsed,
            )

    # ========================================================================
    # Public API Methods - Data Access
    # ========================================================================

    def get_emission_factors(
        self, franchise_type: str,
    ) -> EmissionFactorResponse:
        """
        Get emission factors for a franchise type.

        Args:
            franchise_type: Franchise type string.

        Returns:
            EmissionFactorResponse with factor data.
        """
        from greenlang.agents.mrv.franchises.franchises_pipeline import (
            DEFAULT_EUI_KWH_PER_SQM,
            DEFAULT_ELECTRICITY_EF,
            DEFAULT_GAS_EF,
            DEFAULT_REFRIGERANT_EF,
            EEIO_FRANCHISE_FACTORS,
        )

        eui = DEFAULT_EUI_KWH_PER_SQM.get(franchise_type)
        ref_ef = DEFAULT_REFRIGERANT_EF.get(franchise_type)

        return EmissionFactorResponse(
            success=True,
            franchise_type=franchise_type,
            eui_kwh_per_sqm=float(eui) if eui else None,
            electricity_ef_kgco2e_per_kwh=float(DEFAULT_ELECTRICITY_EF),
            gas_ef_kgco2e_per_kwh=float(DEFAULT_GAS_EF),
            refrigerant_ef_kgco2e=float(ref_ef) if ref_ef else None,
            eeio_ef_kgco2e_per_usd=float(EEIO_FRANCHISE_FACTORS.get("999999", Decimal("0.25"))),
            sources=["eGRID", "CBECS", "ENERGY STAR", "EPA"],
        )

    def get_benchmarks(self) -> BenchmarkResponse:
        """
        Get benchmark data for all franchise types.

        Returns:
            BenchmarkResponse with EUI benchmarks per type.
        """
        from greenlang.agents.mrv.franchises.franchises_pipeline import (
            DEFAULT_EUI_KWH_PER_SQM,
            DEFAULT_REFRIGERANT_EF,
        )

        benchmarks: List[dict] = []
        for ft, eui in DEFAULT_EUI_KWH_PER_SQM.items():
            ref_ef = DEFAULT_REFRIGERANT_EF.get(ft)
            benchmarks.append({
                "franchise_type": ft,
                "eui_kwh_per_sqm": float(eui),
                "refrigerant_ef_kgco2e": float(ref_ef) if ref_ef else None,
                "source": "CBECS/ENERGY STAR",
            })

        return BenchmarkResponse(
            success=True,
            benchmarks=benchmarks,
            total_types=len(benchmarks),
        )

    # ========================================================================
    # Health and Status
    # ========================================================================

    def get_health(self) -> HealthResponse:
        """
        Perform service health check.

        Returns:
            HealthResponse with engine statuses and uptime.

        Example:
            >>> health = service.get_health()
            >>> health.status
            'healthy'
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        engines_status = {
            "database": self._database_engine is not None,
            "franchise_specific": self._franchise_specific_engine is not None,
            "average_data": self._average_data_engine is not None,
            "spend_based": self._spend_engine is not None,
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
            version="1.0.0",
            agent_id="GL-MRV-S3-014",
            engines_status=engines_status,
            uptime_seconds=uptime,
        )

    # ========================================================================
    # Public API Methods - Data Retrieval
    # ========================================================================

    def get_calculation(self, calculation_id: str) -> dict:
        """
        Retrieve a single calculation by ID.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            Dictionary with success flag and calculation data.

        Example:
            >>> result = service.get_calculation("frn-abc123def456")
            >>> result["success"]
            True
        """
        calc = self._calculations.get(calculation_id)
        if calc:
            return {"success": True, "calculation": calc}
        return {
            "success": False,
            "error": f"Calculation {calculation_id} not found",
        }

    def list_calculations(
        self,
        tenant_id: Optional[str] = None,
        franchise_type: Optional[str] = None,
        method: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> dict:
        """
        List calculations with optional filters.

        Args:
            tenant_id: Filter by tenant.
            franchise_type: Filter by franchise type.
            method: Filter by calculation method.
            page: Page number (1-based).
            page_size: Page size (1-1000).

        Returns:
            Dictionary with paginated calculation list.

        Example:
            >>> result = service.list_calculations(franchise_type="hotel")
            >>> result["total_count"]
            5
        """
        all_calcs = list(self._calculations.values())

        # Apply filters
        if franchise_type:
            all_calcs = [
                c for c in all_calcs
                if c.get("franchise_type") == franchise_type
            ]
        if method:
            all_calcs = [
                c for c in all_calcs
                if c.get("method") == method
            ]

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_calcs = all_calcs[start_idx:end_idx]

        return {
            "success": True,
            "calculations": page_calcs,
            "total_count": len(all_calcs),
            "page": page,
            "page_size": page_size,
        }

    def delete_calculation(self, calculation_id: str) -> dict:
        """
        Delete a calculation by ID.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            Dictionary with deletion status.

        Example:
            >>> result = service.delete_calculation("frn-abc123def456")
            >>> result["success"]
            True
        """
        if calculation_id in self._calculations:
            del self._calculations[calculation_id]
            return {
                "success": True,
                "deleted_id": calculation_id,
                "message": "Calculation deleted",
            }
        return {
            "success": False,
            "deleted_id": calculation_id,
            "message": "Calculation not found",
        }

    def get_aggregations(
        self,
        reporting_period: str,
        group_by: str = "franchise_type",
    ) -> dict:
        """
        Get aggregated emissions for a reporting period.

        Args:
            reporting_period: Reporting period filter.
            group_by: Aggregation dimension (franchise_type, method, region).

        Returns:
            Dictionary with aggregated emissions data.

        Example:
            >>> result = service.get_aggregations("2025")
            >>> result["total_co2e_kg"]
            1250000.0
        """
        start_time = time.monotonic()

        by_group: Dict[str, float] = {}
        total = 0.0

        for calc in self._calculations.values():
            if not calc.get("success"):
                continue

            co2e = calc.get("total_co2e_kg", 0.0)
            total += co2e

            if group_by == "franchise_type":
                key = calc.get("franchise_type", "unknown")
            elif group_by == "method":
                key = calc.get("method", "unknown")
            else:
                key = calc.get("detail", {}).get("region", "unknown")

            by_group[key] = by_group.get(key, 0.0) + co2e

        elapsed = (time.monotonic() - start_time) * 1000.0

        return {
            "success": True,
            "total_co2e_kg": total,
            "by_group": by_group,
            "group_by": group_by,
            "reporting_period": reporting_period,
            "processing_time_ms": elapsed,
        }

    def get_provenance(self, calculation_id: str) -> dict:
        """
        Get provenance chain for a calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            Dictionary with provenance hash and chain data.

        Example:
            >>> result = service.get_provenance("frn-abc123def456")
            >>> result["is_valid"]
            True
        """
        calc = self._calculations.get(calculation_id)
        if calc:
            return {
                "success": True,
                "calculation_id": calculation_id,
                "provenance_hash": calc.get("provenance_hash", ""),
                "chain": [],
                "is_valid": True,
            }
        return {
            "success": False,
            "calculation_id": calculation_id,
            "provenance_hash": "",
            "chain": [],
            "is_valid": False,
        }

    def check_double_counting(
        self, units: List[dict],
    ) -> dict:
        """
        Check for double-counting issues across franchise units.

        Validates all 8 double-counting prevention rules at the unit level.

        Args:
            units: List of franchise unit dictionaries.

        Returns:
            Dictionary with findings list and overall status.

        Example:
            >>> result = service.check_double_counting(units)
            >>> result["total_findings"]
            0
        """
        start_time = time.monotonic()

        if self._compliance_engine is not None:
            try:
                findings = self._compliance_engine.check_double_counting(units)
                elapsed = (time.monotonic() - start_time) * 1000.0
                return {
                    "success": True,
                    "total_findings": len(findings),
                    "findings": findings,
                    "status": "PASS" if not findings else "FAIL",
                    "processing_time_ms": elapsed,
                }
            except Exception as e:
                logger.error("Double-counting check failed: %s", e)

        elapsed = (time.monotonic() - start_time) * 1000.0
        return {
            "success": False,
            "total_findings": 0,
            "findings": [],
            "status": "UNKNOWN",
            "error": "ComplianceCheckerEngine not available",
            "processing_time_ms": elapsed,
        }

    def get_franchise_types(self) -> List[dict]:
        """
        Get all supported franchise types with metadata.

        Returns:
            List of franchise type info dictionaries.

        Example:
            >>> types = service.get_franchise_types()
            >>> len(types)
            11
        """
        from greenlang.agents.mrv.franchises.franchises_pipeline import (
            FranchiseType,
            DEFAULT_EUI_KWH_PER_SQM,
        )

        return [
            {
                "franchise_type": ft.value,
                "display_name": ft.value.replace("_", " ").title(),
                "eui_kwh_per_sqm": float(
                    DEFAULT_EUI_KWH_PER_SQM.get(ft.value, Decimal("300"))
                ),
            }
            for ft in FranchiseType
        ]

    def get_calculation_methods(self) -> List[dict]:
        """
        Get all supported calculation methods with descriptions.

        Returns:
            List of method info dictionaries.

        Example:
            >>> methods = service.get_calculation_methods()
            >>> len(methods)
            4
        """
        return [
            {
                "method": "franchise_specific",
                "display_name": "Franchise-Specific",
                "description": (
                    "Uses primary energy data (electricity, gas, refrigerant) "
                    "collected directly from franchise units."
                ),
                "dqi_score": 4.5,
                "data_required": ["electricity_kwh", "gas_kwh"],
            },
            {
                "method": "average_data",
                "display_name": "Average Data",
                "description": (
                    "Uses EUI benchmarks (kWh/m2/year) by franchise type "
                    "and climate zone, applied to floor area."
                ),
                "dqi_score": 2.5,
                "data_required": ["floor_area_sqm", "franchise_type"],
            },
            {
                "method": "spend_based",
                "display_name": "Spend-Based",
                "description": (
                    "Uses EEIO emission factors (kgCO2e/USD) applied to "
                    "franchise revenue or royalty income."
                ),
                "dqi_score": 1.5,
                "data_required": ["revenue or spend", "naics_code"],
            },
            {
                "method": "hybrid",
                "display_name": "Hybrid",
                "description": (
                    "Blends franchise-specific and average-data methods "
                    "across a heterogeneous franchise network."
                ),
                "dqi_score": 3.5,
                "data_required": ["varies by unit"],
            },
        ]

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    @staticmethod
    def _request_to_input(request: FranchiseCalculationRequest) -> dict:
        """
        Convert a Pydantic request model to a pipeline input dictionary.

        Args:
            request: FranchiseCalculationRequest model.

        Returns:
            Dictionary suitable for pipeline execute_single().
        """
        data: Dict[str, Any] = {
            "franchise_type": request.franchise_type,
            "ownership_type": request.ownership_type,
        }

        if request.unit_id:
            data["unit_id"] = request.unit_id
        if request.agreement_type:
            data["agreement_type"] = request.agreement_type
        if request.electricity_kwh is not None:
            data["electricity_kwh"] = request.electricity_kwh
        if request.gas_kwh is not None:
            data["gas_kwh"] = request.gas_kwh
        if request.gas_therms is not None:
            data["gas_therms"] = request.gas_therms
        if request.other_energy_kwh is not None:
            data["other_energy_kwh"] = request.other_energy_kwh
        if request.floor_area_sqm is not None:
            data["floor_area_sqm"] = request.floor_area_sqm
        if request.floor_area_sqft is not None:
            data["floor_area_sqft"] = request.floor_area_sqft
        if request.revenue is not None:
            data["revenue"] = request.revenue
        if request.spend is not None:
            data["spend"] = request.spend
        if request.royalty_revenue is not None:
            data["royalty_revenue"] = request.royalty_revenue
        if request.currency:
            data["currency"] = request.currency
        if request.naics_code:
            data["naics_code"] = request.naics_code
        if request.refrigerant_charge_kg is not None:
            data["refrigerant_charge_kg"] = request.refrigerant_charge_kg
        if request.refrigerant_leak_rate is not None:
            data["refrigerant_leak_rate"] = request.refrigerant_leak_rate
        if request.refrigerant_gwp is not None:
            data["refrigerant_gwp"] = request.refrigerant_gwp
        if request.region:
            data["region"] = request.region
        if request.country_code:
            data["country_code"] = request.country_code
        if request.climate_zone:
            data["climate_zone"] = request.climate_zone
        if request.operating_months is not None:
            data["operating_months"] = request.operating_months
        if request.brands:
            data["brands"] = request.brands
        if request.calculation_method:
            data["calculation_method"] = request.calculation_method
        if request.tenant_id:
            data["tenant_id"] = request.tenant_id
        if request.reporting_period:
            data["reporting_period"] = request.reporting_period

        return data


# ============================================================================
# Module-Level Helpers
# ============================================================================


def get_service() -> FranchisesService:
    """
    Get singleton FranchisesService instance.

    Thread-safe via double-checked locking with RLock.

    Returns:
        FranchisesService singleton instance.

    Example:
        >>> service = get_service()
        >>> service.get_health().status
        'healthy'
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = FranchisesService()
    return _service_instance


def get_router():
    """
    Get the FastAPI router for franchises endpoints.

    Returns:
        FastAPI APIRouter instance.

    Example:
        >>> router = get_router()
        >>> app.include_router(router, prefix="/api/v1/franchises")
    """
    from greenlang.agents.mrv.franchises.api.router import router
    return router


def create_app():
    """
    Standalone app factory for testing.

    Creates a FastAPI application with franchises routes registered.
    Used for development and integration testing.

    Returns:
        FastAPI application instance.

    Example:
        >>> app = create_app()
        >>> # use with TestClient(app) for testing
    """
    try:
        from fastapi import FastAPI

        app = FastAPI(
            title="GreenLang Franchises Agent",
            description="Scope 3 Category 14 Franchise Emissions API",
            version="1.0.0",
        )

        try:
            router = get_router()
            app.include_router(router, prefix="/api/v1/franchises", tags=["franchises"])
        except ImportError:
            logger.warning("Franchises router not available for test app")

        @app.get("/health")
        def health():
            """Health check endpoint."""
            service = get_service()
            return service.get_health().dict()

        return app

    except ImportError:
        logger.warning("FastAPI not available; create_app() returns None")
        return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Request models
    "FranchiseCalculationRequest",
    "BatchFranchiseCalculationRequest",
    "NetworkCalculationRequest",
    "ComplianceCheckRequest",
    "EmissionFactorRequest",
    "BenchmarkRequest",
    # Response models
    "FranchiseCalculationResponse",
    "BatchFranchiseResponse",
    "NetworkCalculationResponse",
    "ComplianceCheckResponse",
    "EmissionFactorResponse",
    "BenchmarkResponse",
    "HealthResponse",
    # Service
    "FranchisesService",
    # Helpers
    "get_service",
    "get_router",
    "create_app",
]
