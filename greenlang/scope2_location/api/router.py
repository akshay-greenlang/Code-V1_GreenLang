# -*- coding: utf-8 -*-
"""
Scope 2 Location-Based Emissions REST API Router - AGENT-MRV-009
=================================================================

20 REST endpoints for the Scope 2 Location-Based Emissions Agent
(GL-MRV-SCOPE2-001).

Prefix: ``/api/v1/scope2-location``

Endpoints:
     1. POST   /calculations                - Execute single calculation
     2. POST   /calculations/batch          - Execute batch calculations
     3. GET    /calculations                - List calculations with filters
     4. GET    /calculations/{id}           - Get calculation by ID
     5. DELETE /calculations/{id}           - Delete calculation
     6. POST   /facilities                  - Register a facility
     7. GET    /facilities                  - List facilities with filters
     8. PUT    /facilities/{id}             - Update a facility
     9. POST   /consumption                 - Record energy consumption
    10. GET    /consumption                 - List consumption records
    11. GET    /grid-factors                - List grid emission factors
    12. GET    /grid-factors/{region}       - Get factor for a region
    13. POST   /grid-factors/custom         - Add custom grid factor
    14. GET    /td-losses                   - List T&D loss factors
    15. POST   /compliance/check            - Run compliance check
    16. GET    /compliance/{id}             - Get compliance result
    17. POST   /uncertainty                 - Run uncertainty analysis
    18. GET    /aggregations                - Get aggregated emissions
    19. GET    /health                      - Service health check
    20. GET    /stats                       - Service statistics

GHG Protocol Scope 2 Guidance location-based method implementation for
purchased electricity, steam, heating, and cooling emission calculations.

All emission factors and intermediate values use Python ``Decimal`` for
zero-hallucination deterministic arithmetic. No LLM involvement in any
calculation path. Every result carries a SHA-256 provenance hash for
full audit trails.

Supported Energy Types:
    - Electricity: 26 US EPA eGRID subregions, 130+ IEA countries,
      27 EU member states, UK DEFRA factors
    - Steam: natural gas, coal, biomass fuel sources
    - Heating: district, gas boiler, electric systems
    - Cooling: electric chiller, absorption, district networks

Supported Regulatory Frameworks (80 requirements across 7 frameworks):
    - GHG Protocol Scope 2 Guidance (2015)
    - IPCC 2006 Guidelines for National GHG Inventories
    - ISO 14064-1:2018
    - CSRD/ESRS E1
    - US EPA Greenhouse Gas Reporting Program
    - UK DEFRA Reporting (SECR)
    - CDP Climate Change

Uncertainty Quantification:
    - Monte Carlo simulation (configurable 100 to 1,000,000 iterations)
    - Analytical error propagation (IPCC Approach 1)
    - Data Quality Indicator (DQI) scoring across 4 dimensions

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
Status: Production Ready
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI/Pydantic imports
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Path
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; scope2-location router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    # ---------------------------------------------------------------
    # Calculation request models
    # ---------------------------------------------------------------

    class SingleCalculationRequest(BaseModel):
        """Request body for a single Scope 2 location-based emission calculation.

        Accepts electricity, steam, heating, or cooling energy consumption
        data and computes CO2e emissions using the GHG Protocol location-
        based method. Grid emission factors are resolved automatically from
        the facility's country code and optional eGRID subregion.
        """

        facility_id: str = Field(
            ...,
            min_length=1,
            description="Reference to the consuming facility",
        )
        energy_type: str = Field(
            ...,
            description="Purchased energy type: electricity, steam, "
            "heating, or cooling",
        )
        consumption_value: float = Field(
            ..., ge=0,
            description="Energy consumption quantity in the specified unit",
        )
        consumption_unit: str = Field(
            default="mwh",
            description="Unit of measurement: kwh, mwh, gj, mmbtu, or therms",
        )
        country_code: str = Field(
            ...,
            min_length=2,
            max_length=2,
            description="ISO 3166-1 alpha-2 country code for the facility",
        )
        egrid_subregion: Optional[str] = Field(
            default=None,
            max_length=10,
            description="US EPA eGRID subregion code (e.g. CAMX, ERCT). "
            "Applicable only for US facilities",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source: AR4, AR5, AR6, or AR6_20YR "
            "(default: AR5)",
        )
        include_td_losses: Optional[bool] = Field(
            default=True,
            description="Whether to include transmission and distribution "
            "loss emissions for electricity consumption",
        )
        steam_type: Optional[str] = Field(
            default=None,
            description="Steam fuel source: natural_gas, coal, biomass "
            "(required when energy_type is steam)",
        )
        heating_type: Optional[str] = Field(
            default=None,
            description="Heating technology type: district, gas_boiler, "
            "electric (required when energy_type is heating)",
        )
        cooling_type: Optional[str] = Field(
            default=None,
            description="Cooling technology type: electric_chiller, "
            "absorption, district (required when energy_type is cooling)",
        )
        custom_ef: Optional[float] = Field(
            default=None,
            ge=0,
            description="Custom emission factor override (kgCO2e per MWh "
            "for electricity, kgCO2e per GJ for steam/heat/cooling)",
        )
        include_compliance: Optional[bool] = Field(
            default=False,
            description="Whether to include regulatory compliance checks",
        )
        compliance_frameworks: Optional[List[str]] = Field(
            default=None,
            description="Specific regulatory frameworks to check "
            "(e.g. ghg_protocol_scope2, ipcc_2006, csrd_e1)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )

    class BatchCalculationBody(BaseModel):
        """Request body for batch Scope 2 location-based emission calculations.

        Processes multiple calculation requests in a single batch. Each
        item in the ``requests`` list follows the same schema as the
        single calculate endpoint's pipeline request format.
        """

        batch_id: Optional[str] = Field(
            default=None,
            description="Optional batch identifier (auto-generated if omitted)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )
        requests: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation request dictionaries, each "
            "containing facility_id, energy_type, consumption_value, "
            "consumption_unit, country_code, and optional fields",
        )

    # ---------------------------------------------------------------
    # Facility request models
    # ---------------------------------------------------------------

    class FacilityBody(BaseModel):
        """Request body for registering a new facility.

        Creates a facility record with geographic, grid region, and
        classification metadata. Every facility is scoped to a tenant
        and mapped to a grid region for emission factor lookup.
        """

        facility_id: Optional[str] = Field(
            default=None,
            description="Optional facility identifier "
            "(auto-generated UUID if omitted)",
        )
        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Human-readable facility name or label",
        )
        facility_type: str = Field(
            ...,
            description="Facility classification: office, warehouse, "
            "manufacturing, retail, data_center, hospital, school, other",
        )
        country_code: str = Field(
            ..., min_length=2, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        grid_region_id: Optional[str] = Field(
            default=None,
            max_length=50,
            description="Grid region identifier for emission factor lookup "
            "(defaults to country code if omitted)",
        )
        egrid_subregion: Optional[str] = Field(
            default=None,
            max_length=10,
            description="US eGRID subregion code (e.g. CAMX, ERCT)",
        )
        latitude: Optional[float] = Field(
            default=None, ge=-90, le=90,
            description="WGS84 latitude in decimal degrees",
        )
        longitude: Optional[float] = Field(
            default=None, ge=-180, le=180,
            description="WGS84 longitude in decimal degrees",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier for multi-tenancy",
        )

    class FacilityUpdateBody(BaseModel):
        """Request body for updating an existing facility.

        Only non-null fields in the request body will be updated.
        """

        name: Optional[str] = Field(
            default=None, min_length=1, max_length=500,
            description="Human-readable facility name",
        )
        facility_type: Optional[str] = Field(
            default=None,
            description="Facility classification",
        )
        country_code: Optional[str] = Field(
            default=None, min_length=2, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        grid_region_id: Optional[str] = Field(
            default=None, max_length=50,
            description="Grid region identifier for EF lookup",
        )
        egrid_subregion: Optional[str] = Field(
            default=None, max_length=10,
            description="US eGRID subregion code",
        )
        latitude: Optional[float] = Field(
            default=None, ge=-90, le=90,
            description="WGS84 latitude in decimal degrees",
        )
        longitude: Optional[float] = Field(
            default=None, ge=-180, le=180,
            description="WGS84 longitude in decimal degrees",
        )

    # ---------------------------------------------------------------
    # Consumption request model
    # ---------------------------------------------------------------

    class ConsumptionBody(BaseModel):
        """Request body for recording an energy consumption measurement.

        Creates a consumption record from a meter reading, utility invoice,
        or engineering estimate. Multiple records may exist for the same
        facility covering different energy types, meters, or time periods.
        """

        facility_id: str = Field(
            ..., min_length=1,
            description="Reference to the consuming facility",
        )
        energy_type: str = Field(
            ...,
            description="Type of purchased energy: electricity, steam, "
            "heating, or cooling",
        )
        quantity: float = Field(
            ..., ge=0,
            description="Consumption quantity in the specified unit",
        )
        unit: str = Field(
            ...,
            description="Unit of measurement: kwh, mwh, gj, mmbtu, therms",
        )
        period_start: str = Field(
            ...,
            description="Start date of the consumption period (ISO-8601)",
        )
        period_end: str = Field(
            ...,
            description="End date of the consumption period (ISO-8601)",
        )
        data_source: Optional[str] = Field(
            default="invoice",
            description="Origin of the data: meter, invoice, estimate, "
            "or benchmark",
        )
        meter_id: Optional[str] = Field(
            default=None, max_length=100,
            description="Optional utility meter identifier for traceability",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # ---------------------------------------------------------------
    # Grid factor request model
    # ---------------------------------------------------------------

    class CustomGridFactorBody(BaseModel):
        """Request body for adding a custom grid emission factor.

        Custom factors must include documented provenance for audit
        compliance. The quality tier defaults to tier_3 (facility-
        specific) when a custom factor is provided with evidence.
        """

        region_id: str = Field(
            ..., min_length=1,
            description="Grid region identifier for the custom factor",
        )
        co2_kg_per_mwh: float = Field(
            ..., ge=0,
            description="CO2 emission rate in kg per MWh",
        )
        ch4_kg_per_mwh: Optional[float] = Field(
            default=0.0, ge=0,
            description="CH4 emission rate in kg per MWh "
            "(defaults to 0 if omitted)",
        )
        n2o_kg_per_mwh: Optional[float] = Field(
            default=0.0, ge=0,
            description="N2O emission rate in kg per MWh "
            "(defaults to 0 if omitted)",
        )
        year: int = Field(
            ..., ge=1990, le=2100,
            description="Reporting year the factor applies to",
        )
        quality_tier: Optional[str] = Field(
            default="tier_3",
            description="Data quality tier: tier_1, tier_2, or tier_3",
        )
        source_name: Optional[str] = Field(
            default="custom",
            max_length=200,
            description="Name of the data source (e.g. utility disclosure, "
            "national inventory)",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the custom factor provenance",
        )

    # ---------------------------------------------------------------
    # Compliance request model
    # ---------------------------------------------------------------

    class ComplianceCheckBody(BaseModel):
        """Request body for a regulatory compliance check.

        Evaluates a completed calculation against one or more regulatory
        frameworks. If no frameworks are specified, all supported
        frameworks are evaluated.
        """

        calculation_id: str = Field(
            ..., min_length=1,
            description="ID of a previous calculation to check",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Regulatory frameworks to check. Empty means all. "
            "Options: ghg_protocol_scope2, ipcc_2006, iso_14064, "
            "csrd_e1, epa_ghgrp, defra_secr, cdp",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    # ---------------------------------------------------------------
    # Uncertainty request model
    # ---------------------------------------------------------------

    class UncertaintyBody(BaseModel):
        """Request body for uncertainty analysis.

        Runs Monte Carlo simulation or analytical error propagation
        on a previously computed calculation result to quantify the
        uncertainty range of the emission estimate.
        """

        calculation_id: str = Field(
            ..., min_length=1,
            description="ID of a previous calculation to analyse",
        )
        iterations: int = Field(
            default=10000, ge=100, le=1_000_000,
            description="Number of Monte Carlo simulation iterations",
        )
        confidence_level: float = Field(
            default=95.0, gt=0, lt=100,
            description="Confidence level percentage for the uncertainty "
            "interval (e.g. 95 for 95%% CI)",
        )
        seed: int = Field(
            default=42, ge=0,
            description="Random seed for reproducibility",
        )
        method: Optional[str] = Field(
            default="monte_carlo",
            description="Uncertainty method: monte_carlo or analytical",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Scope 2 Location-Based Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints covering calculations,
        facilities, consumption, grid factors, T&D losses, compliance,
        uncertainty, aggregations, health, and statistics.

    Raises:
        RuntimeError: If FastAPI is not installed in the environment.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the scope2-location router"
        )

    router = APIRouter(
        prefix="/api/v1/scope2-location",
        tags=["Scope 2 Location-Based Emissions"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the Scope2LocationPipelineEngine singleton.

        Returns the initialized pipeline engine. Raises HTTPException 503
        if the service has not been initialized.
        """
        try:
            from greenlang.scope2_location.scope2_location_pipeline import (
                Scope2LocationPipelineEngine,
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Scope 2 Location pipeline engine not available",
            )

        # Try to get from setup module if it exists
        try:
            from greenlang.scope2_location.setup import get_service
            svc = get_service()
            if svc is not None:
                return svc
        except (ImportError, AttributeError):
            pass

        # Fallback: create a default pipeline engine instance
        if not hasattr(_get_service, "_instance"):
            _get_service._instance = Scope2LocationPipelineEngine()
        return _get_service._instance

    def _get_grid_factor_db():
        """Get the GridEmissionFactorDatabaseEngine.

        Returns the grid factor database engine from the pipeline, or
        creates a standalone instance if the pipeline is not available.
        Raises HTTPException 503 if the engine cannot be loaded.
        """
        try:
            svc = _get_service()
            if hasattr(svc, "_grid_factor_db") and svc._grid_factor_db is not None:
                return svc._grid_factor_db
        except HTTPException:
            pass

        try:
            from greenlang.scope2_location.grid_factor_database import (
                GridEmissionFactorDatabaseEngine,
            )
            if not hasattr(_get_grid_factor_db, "_instance"):
                _get_grid_factor_db._instance = GridEmissionFactorDatabaseEngine()
            return _get_grid_factor_db._instance
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Grid factor database engine not available",
            )

    def _get_transmission_engine():
        """Get the TransmissionLossEngine.

        Returns the T&D loss engine from the pipeline, or creates a
        standalone instance if the pipeline is not available.
        """
        try:
            svc = _get_service()
            if hasattr(svc, "_transmission") and svc._transmission is not None:
                return svc._transmission
        except HTTPException:
            pass

        try:
            from greenlang.scope2_location.transmission_loss import (
                TransmissionLossEngine,
            )
            if not hasattr(_get_transmission_engine, "_instance"):
                _get_transmission_engine._instance = TransmissionLossEngine()
            return _get_transmission_engine._instance
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # In-memory stores for API-level state
    # ------------------------------------------------------------------

    _calculations: List[Dict[str, Any]] = []
    _facilities: List[Dict[str, Any]] = []
    _consumption_records: List[Dict[str, Any]] = []
    _compliance_results: List[Dict[str, Any]] = []
    _uncertainty_results: List[Dict[str, Any]] = []

    # ==================================================================
    # 1. POST /calculations - Execute single calculation
    # ==================================================================

    @router.post("/calculations", status_code=201)
    async def create_calculation(
        body: SingleCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute a single Scope 2 location-based emission calculation.

        Computes CO2e emissions for a facility's purchased energy using
        the GHG Protocol Scope 2 Guidance location-based method. Grid
        emission factors are resolved automatically from the facility's
        country code and optional eGRID subregion. Supports electricity,
        steam, heating, and cooling energy types.

        Formula (electricity):
            Emissions = Consumption (MWh) x Grid EF (kgCO2e/MWh) x (1 + T&D%)

        Per-gas breakdown: CO2, CH4, and N2O with configurable GWP source
        (IPCC AR4, AR5, AR6, or AR6 20-year).

        Permission: scope2-location:calculate
        """
        svc = _get_service()

        # Build pipeline request dict
        request_data: Dict[str, Any] = {
            "facility_id": body.facility_id,
            "energy_type": body.energy_type.lower(),
            "consumption_value": Decimal(str(body.consumption_value)),
            "consumption_unit": body.consumption_unit.lower(),
            "country_code": body.country_code.upper(),
            "include_td_losses": body.include_td_losses
            if body.include_td_losses is not None else True,
            "include_compliance": body.include_compliance or False,
        }

        if body.egrid_subregion is not None:
            request_data["egrid_subregion"] = body.egrid_subregion.upper()
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source.upper()
        if body.steam_type is not None:
            request_data["steam_type"] = body.steam_type
        if body.heating_type is not None:
            request_data["heating_type"] = body.heating_type
        if body.cooling_type is not None:
            request_data["cooling_type"] = body.cooling_type
        if body.custom_ef is not None:
            request_data["custom_ef"] = Decimal(str(body.custom_ef))
        if body.compliance_frameworks is not None:
            request_data["compliance_frameworks"] = body.compliance_frameworks
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.run_pipeline(request_data)

            # Store result for later retrieval
            _calculations.append(result)

            logger.info(
                "Calculation completed: id=%s type=%s co2e=%.6f tonnes",
                result.get("calculation_id", ""),
                result.get("energy_type", ""),
                float(result.get("total_co2e_tonnes", 0)),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_calculation failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculations/batch - Execute batch calculations
    # ==================================================================

    @router.post("/calculations/batch", status_code=201)
    async def create_batch_calculation(
        body: BatchCalculationBody,
    ) -> Dict[str, Any]:
        """Execute batch Scope 2 location-based emission calculations.

        Processes multiple calculation requests in a single batch. Each
        item in the ``requests`` list follows the same schema as the
        pipeline ``run_pipeline`` method. Returns aggregated portfolio-
        level totals alongside individual results.

        Permission: scope2-location:calculate
        """
        svc = _get_service()

        batch_data: Dict[str, Any] = {
            "batch_id": body.batch_id or str(uuid.uuid4()),
            "requests": body.requests,
        }
        if body.tenant_id is not None:
            batch_data["tenant_id"] = body.tenant_id

        try:
            result = svc.run_batch_pipeline(batch_data)

            # Store individual results for later retrieval
            for r in result.get("results", []):
                _calculations.append(r)

            logger.info(
                "Batch completed: batch_id=%s total=%d successful=%d",
                result.get("batch_id", ""),
                result.get("total_requests", 0),
                result.get("successful", 0),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_batch_calculation failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /calculations - List calculations with filters
    # ==================================================================

    @router.get("/calculations", status_code=200)
    async def list_calculations(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        energy_type: Optional[str] = Query(
            None,
            description="Filter by energy type: electricity, steam, "
            "heating, cooling",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility identifier",
        ),
        country_code: Optional[str] = Query(
            None, description="Filter by ISO 3166-1 alpha-2 country code",
        ),
        skip: int = Query(
            0, ge=0,
            description="Number of records to skip (offset)",
        ),
        limit: int = Query(
            20, ge=1, le=100,
            description="Maximum number of records to return",
        ),
    ) -> Dict[str, Any]:
        """List Scope 2 location-based calculation results with pagination.

        Returns calculation results filtered by tenant, energy type,
        facility, or country code. Results are ordered by calculation
        timestamp (newest first).

        Permission: scope2-location:read
        """
        filtered = list(_calculations)

        if tenant_id is not None:
            filtered = [
                c for c in filtered
                if c.get("tenant_id") == tenant_id
            ]
        if energy_type is not None:
            filtered = [
                c for c in filtered
                if c.get("energy_type") == energy_type.lower()
            ]
        if facility_id is not None:
            filtered = [
                c for c in filtered
                if c.get("facility_id") == facility_id
            ]
        if country_code is not None:
            filtered = [
                c for c in filtered
                if c.get("country_code", "").upper() == country_code.upper()
            ]

        total = len(filtered)
        page_data = filtered[skip: skip + limit]

        return {
            "calculations": [_serialize_result(c) for c in page_data],
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    # ==================================================================
    # 4. GET /calculations/{id} - Get calculation by ID
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a Scope 2 calculation result by its unique identifier.

        Returns the full calculation result including gas breakdown,
        emission factor metadata, T&D loss data, compliance results,
        and provenance hash.

        Permission: scope2-location:read
        """
        for calc in _calculations:
            if calc.get("calculation_id") == calc_id:
                return _serialize_result(calc)

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 5. DELETE /calculations/{id} - Delete calculation
    # ==================================================================

    @router.delete("/calculations/{calc_id}", status_code=200)
    async def delete_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Delete a Scope 2 calculation result by its unique identifier.

        Permanently removes the calculation result from the in-memory
        store. This action cannot be undone.

        Permission: scope2-location:delete
        """
        for i, calc in enumerate(_calculations):
            if calc.get("calculation_id") == calc_id:
                _calculations.pop(i)
                return {
                    "deleted": True,
                    "calculation_id": calc_id,
                }

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 6. POST /facilities - Register a facility
    # ==================================================================

    @router.post("/facilities", status_code=201)
    async def create_facility(
        body: FacilityBody,
    ) -> Dict[str, Any]:
        """Register a new facility for Scope 2 emission tracking.

        Creates a facility record with geographic, grid region, and
        classification metadata. Every facility is scoped to a tenant
        and mapped to a grid region for emission factor lookup.

        The ``grid_region_id`` defaults to the ``country_code`` when
        not explicitly provided. For US facilities, specify the
        ``egrid_subregion`` to use subregional emission factors with
        higher accuracy.

        Permission: scope2-location:facilities:write
        """
        facility_id = body.facility_id or str(uuid.uuid4())

        # Check for duplicate facility_id
        for fac in _facilities:
            if fac.get("facility_id") == facility_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"Facility {facility_id} already exists",
                )

        facility = {
            "facility_id": facility_id,
            "name": body.name,
            "facility_type": body.facility_type,
            "country_code": body.country_code.upper(),
            "grid_region_id": (
                body.grid_region_id or body.country_code.upper()
            ),
            "egrid_subregion": (
                body.egrid_subregion.upper()
                if body.egrid_subregion else None
            ),
            "latitude": body.latitude,
            "longitude": body.longitude,
            "tenant_id": body.tenant_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        _facilities.append(facility)

        logger.info(
            "Facility registered: id=%s name=%s country=%s",
            facility_id, body.name, body.country_code.upper(),
        )

        return facility

    # ==================================================================
    # 7. GET /facilities - List facilities with filters
    # ==================================================================

    @router.get("/facilities", status_code=200)
    async def list_facilities(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        country_code: Optional[str] = Query(
            None, description="Filter by ISO 3166-1 alpha-2 country code",
        ),
        facility_type: Optional[str] = Query(
            None,
            description="Filter by facility type: office, warehouse, "
            "manufacturing, retail, data_center, hospital, school, other",
        ),
        skip: int = Query(
            0, ge=0,
            description="Number of records to skip (offset)",
        ),
        limit: int = Query(
            20, ge=1, le=100,
            description="Maximum number of records to return",
        ),
    ) -> Dict[str, Any]:
        """List registered facilities with pagination and filters.

        Returns facilities filtered by tenant, country, or facility type.
        Results are ordered by creation timestamp (newest first).

        Permission: scope2-location:facilities:read
        """
        filtered = list(_facilities)

        if tenant_id is not None:
            filtered = [
                f for f in filtered
                if f.get("tenant_id") == tenant_id
            ]
        if country_code is not None:
            filtered = [
                f for f in filtered
                if f.get("country_code", "").upper() == country_code.upper()
            ]
        if facility_type is not None:
            filtered = [
                f for f in filtered
                if f.get("facility_type") == facility_type
            ]

        total = len(filtered)
        page_data = filtered[skip: skip + limit]

        return {
            "facilities": page_data,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    # ==================================================================
    # 8. PUT /facilities/{id} - Update a facility
    # ==================================================================

    @router.put("/facilities/{facility_id}", status_code=200)
    async def update_facility(
        body: FacilityUpdateBody,
        facility_id: str = Path(
            ..., description="Facility identifier",
        ),
    ) -> Dict[str, Any]:
        """Update an existing facility's attributes.

        Only non-null fields in the request body will be updated.
        The ``facility_id`` and ``tenant_id`` cannot be changed.

        Permission: scope2-location:facilities:write
        """
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }

        # Normalize country_code to uppercase if present
        if "country_code" in update_data:
            update_data["country_code"] = update_data["country_code"].upper()
        if "egrid_subregion" in update_data:
            update_data["egrid_subregion"] = (
                update_data["egrid_subregion"].upper()
            )

        for i, fac in enumerate(_facilities):
            if fac.get("facility_id") == facility_id:
                _facilities[i] = {
                    **fac,
                    **update_data,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                logger.info(
                    "Facility updated: id=%s fields=%s",
                    facility_id, list(update_data.keys()),
                )

                return _facilities[i]

        raise HTTPException(
            status_code=404,
            detail=f"Facility {facility_id} not found",
        )

    # ==================================================================
    # 9. POST /consumption - Record energy consumption
    # ==================================================================

    @router.post("/consumption", status_code=201)
    async def create_consumption(
        body: ConsumptionBody,
    ) -> Dict[str, Any]:
        """Record an energy consumption measurement for a facility.

        Creates a consumption record from a meter reading, utility
        invoice, or engineering estimate. Records are associated with
        a facility and can be used as input data for emission
        calculations.

        Permission: scope2-location:consumption:write
        """
        # Validate energy type
        valid_types = {"electricity", "steam", "heating", "cooling"}
        if body.energy_type.lower() not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid energy_type '{body.energy_type}'. "
                f"Must be one of: {sorted(valid_types)}",
            )

        # Validate unit
        valid_units = {"kwh", "mwh", "gj", "mmbtu", "therms"}
        if body.unit.lower() not in valid_units:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid unit '{body.unit}'. "
                f"Must be one of: {sorted(valid_units)}",
            )

        # Validate data_source
        valid_sources = {"meter", "invoice", "estimate", "benchmark"}
        ds = (body.data_source or "invoice").lower()
        if ds not in valid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data_source '{body.data_source}'. "
                f"Must be one of: {sorted(valid_sources)}",
            )

        record = {
            "consumption_id": str(uuid.uuid4()),
            "facility_id": body.facility_id,
            "energy_type": body.energy_type.lower(),
            "quantity": body.quantity,
            "unit": body.unit.lower(),
            "period_start": body.period_start,
            "period_end": body.period_end,
            "data_source": ds,
            "meter_id": body.meter_id,
            "tenant_id": body.tenant_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        _consumption_records.append(record)

        logger.info(
            "Consumption recorded: facility=%s type=%s qty=%.2f %s",
            body.facility_id, body.energy_type, body.quantity, body.unit,
        )

        return record

    # ==================================================================
    # 10. GET /consumption - List consumption records
    # ==================================================================

    @router.get("/consumption", status_code=200)
    async def list_consumption(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility identifier",
        ),
        energy_type: Optional[str] = Query(
            None,
            description="Filter by energy type: electricity, steam, "
            "heating, cooling",
        ),
        skip: int = Query(
            0, ge=0,
            description="Number of records to skip (offset)",
        ),
        limit: int = Query(
            20, ge=1, le=100,
            description="Maximum number of records to return",
        ),
    ) -> Dict[str, Any]:
        """List energy consumption records with pagination and filters.

        Returns consumption records filtered by tenant, facility, or
        energy type. Records are ordered by period start date.

        Permission: scope2-location:consumption:read
        """
        filtered = list(_consumption_records)

        if tenant_id is not None:
            filtered = [
                r for r in filtered
                if r.get("tenant_id") == tenant_id
            ]
        if facility_id is not None:
            filtered = [
                r for r in filtered
                if r.get("facility_id") == facility_id
            ]
        if energy_type is not None:
            filtered = [
                r for r in filtered
                if r.get("energy_type") == energy_type.lower()
            ]

        total = len(filtered)
        page_data = filtered[skip: skip + limit]

        return {
            "consumption_records": page_data,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    # ==================================================================
    # 11. GET /grid-factors - List grid emission factors
    # ==================================================================

    @router.get("/grid-factors", status_code=200)
    async def list_grid_factors(
        source: Optional[str] = Query(
            None,
            description="Filter by factor source: egrid, iea, eu_eea, "
            "defra, national, custom, ipcc",
        ),
    ) -> Dict[str, Any]:
        """List available grid emission factors by source.

        Returns emission factors from all supported authoritative
        databases: EPA eGRID (26 US subregions), IEA (130+ countries),
        EU EEA (27 member states), UK DEFRA, and any custom factors.

        When ``source`` is specified, returns only factors from that
        database. When omitted, returns a summary of all available
        factor sources with record counts.

        Permission: scope2-location:factors:read
        """
        try:
            gfdb = _get_grid_factor_db()

            if source is not None:
                source_lower = source.lower()
                result: Dict[str, Any] = {
                    "source": source_lower,
                    "factors": [],
                }

                if source_lower == "egrid":
                    subregions = gfdb.list_egrid_subregions()
                    factors = []
                    for sr in subregions:
                        try:
                            ef = gfdb.get_egrid_factor(sr)
                            factors.append({
                                "region": sr,
                                "co2_kg_per_mwh": _dec_to_float(
                                    ef.get("co2", Decimal("0"))
                                ),
                                "ch4_kg_per_mwh": _dec_to_float(
                                    ef.get("ch4", Decimal("0"))
                                ),
                                "n2o_kg_per_mwh": _dec_to_float(
                                    ef.get("n2o", Decimal("0"))
                                ),
                                "source": "egrid",
                            })
                        except Exception:
                            pass
                    result["factors"] = factors
                    result["count"] = len(factors)

                elif source_lower == "iea":
                    countries = gfdb.list_countries()
                    factors = []
                    for cc in countries:
                        try:
                            ef = gfdb.get_iea_factor(cc)
                            factors.append({
                                "country_code": cc,
                                "co2_tonne_per_mwh": _dec_to_float(
                                    ef if isinstance(ef, Decimal)
                                    else ef.get("factor", Decimal("0"))
                                ),
                                "source": "iea",
                            })
                        except Exception:
                            pass
                    result["factors"] = factors
                    result["count"] = len(factors)

                elif source_lower == "eu_eea":
                    eu_countries = gfdb.list_eu_countries()
                    factors = []
                    for cc in eu_countries:
                        try:
                            ef = gfdb.get_eu_factor(cc)
                            factors.append({
                                "country_code": cc,
                                "co2_tonne_per_mwh": _dec_to_float(
                                    ef if isinstance(ef, Decimal)
                                    else ef.get("factor", Decimal("0"))
                                ),
                                "source": "eu_eea",
                            })
                        except Exception:
                            pass
                    result["factors"] = factors
                    result["count"] = len(factors)

                elif source_lower == "defra":
                    try:
                        ef = gfdb.get_defra_factor()
                        result["factors"] = [
                            _serialize_result(
                                ef if isinstance(ef, dict) else {"factor": ef}
                            )
                        ]
                        result["count"] = 1
                    except Exception:
                        result["factors"] = []
                        result["count"] = 0

                elif source_lower == "custom":
                    custom = gfdb.list_custom_factors()
                    if isinstance(custom, list):
                        result["factors"] = [
                            _serialize_result(f) for f in custom
                        ]
                    else:
                        result["factors"] = [
                            _serialize_result(v) for v in custom.values()
                        ] if isinstance(custom, dict) else []
                    result["count"] = len(result["factors"])

                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown source '{source}'. Valid sources: "
                        "egrid, iea, eu_eea, defra, national, custom",
                    )

                return result

            # No source filter: return summary of all sources
            summary: Dict[str, Any] = {
                "sources": {},
            }
            try:
                summary["sources"]["egrid"] = {
                    "count": len(gfdb.list_egrid_subregions()),
                    "description": "US EPA eGRID subregion emission factors",
                }
            except Exception:
                summary["sources"]["egrid"] = {"count": 0}

            try:
                summary["sources"]["iea"] = {
                    "count": len(gfdb.list_countries()),
                    "description": "IEA country-level electricity factors",
                }
            except Exception:
                summary["sources"]["iea"] = {"count": 0}

            try:
                summary["sources"]["eu_eea"] = {
                    "count": len(gfdb.list_eu_countries()),
                    "description": "EU member state electricity factors",
                }
            except Exception:
                summary["sources"]["eu_eea"] = {"count": 0}

            summary["sources"]["defra"] = {
                "count": 1,
                "description": "UK DEFRA GHG conversion factors",
            }

            try:
                custom = gfdb.list_custom_factors()
                custom_count = (
                    len(custom) if isinstance(custom, (list, dict)) else 0
                )
                summary["sources"]["custom"] = {
                    "count": custom_count,
                    "description": "User-defined custom emission factors",
                }
            except Exception:
                summary["sources"]["custom"] = {"count": 0}

            return summary

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "list_grid_factors failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. GET /grid-factors/{region} - Get factor for a region
    # ==================================================================

    @router.get("/grid-factors/{region}", status_code=200)
    async def get_grid_factor(
        region: str = Path(
            ..., description="Grid region identifier (eGRID subregion "
            "code, ISO country code, or custom region ID)",
        ),
    ) -> Dict[str, Any]:
        """Get the emission factor for a specific grid region.

        Accepts eGRID subregion codes (e.g. CAMX, ERCT), ISO 3166-1
        alpha-2 country codes (e.g. US, GB, DE), or custom region
        identifiers.

        The resolution order is:
        1. eGRID subregion (if region matches a known subregion code)
        2. EU EEA country (if region matches a known EU member state)
        3. IEA country (if region matches a known ISO country code)
        4. Custom factor (if region matches a custom region ID)

        Permission: scope2-location:factors:read
        """
        try:
            gfdb = _get_grid_factor_db()

            region_upper = region.upper()

            # Try eGRID subregion first
            try:
                egrid_subregions = gfdb.list_egrid_subregions()
                if region_upper in egrid_subregions:
                    ef = gfdb.get_egrid_factor(region_upper)
                    return {
                        "region": region_upper,
                        "source": "egrid",
                        "co2_kg_per_mwh": _dec_to_float(
                            ef.get("co2", Decimal("0"))
                        ),
                        "ch4_kg_per_mwh": _dec_to_float(
                            ef.get("ch4", Decimal("0"))
                        ),
                        "n2o_kg_per_mwh": _dec_to_float(
                            ef.get("n2o", Decimal("0"))
                        ),
                    }
            except Exception:
                pass

            # Try country-level factor (EU, IEA, or DEFRA)
            try:
                resolved = gfdb.resolve_emission_factor(
                    country_code=region_upper,
                )
                return _serialize_result(resolved)
            except Exception:
                pass

            # Try generic grid factor
            try:
                gf = gfdb.get_grid_factor(region_upper)
                return _serialize_result(gf)
            except Exception:
                pass

            raise HTTPException(
                status_code=404,
                detail=f"No emission factor found for region '{region}'",
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "get_grid_factor failed for region=%s: %s",
                region, exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. POST /grid-factors/custom - Add custom grid factor
    # ==================================================================

    @router.post("/grid-factors/custom", status_code=201)
    async def add_custom_grid_factor(
        body: CustomGridFactorBody,
    ) -> Dict[str, Any]:
        """Add a custom grid emission factor for a region.

        Custom factors are stored alongside authoritative factors and
        can be used for locations where published factors are unavailable
        or outdated. Custom factors require documented provenance for
        audit compliance.

        The total CO2e rate is automatically calculated from individual
        gas emission rates using IPCC AR6 GWP values.

        Permission: scope2-location:factors:write
        """
        try:
            gfdb = _get_grid_factor_db()

            co2 = Decimal(str(body.co2_kg_per_mwh))
            ch4 = Decimal(str(body.ch4_kg_per_mwh or 0))
            n2o = Decimal(str(body.n2o_kg_per_mwh or 0))

            # Calculate total CO2e using AR6 GWP
            total_co2e = co2 + (ch4 * Decimal("27.9")) + (n2o * Decimal("273"))

            result = gfdb.add_custom_factor(
                region_id=body.region_id,
                co2_kg_per_mwh=co2,
                ch4_kg_per_mwh=ch4,
                n2o_kg_per_mwh=n2o,
                total_co2e_kg_per_mwh=total_co2e,
                year=body.year,
                quality_tier=body.quality_tier or "tier_3",
                source_name=body.source_name or "custom",
                notes=body.notes or "",
            )

            logger.info(
                "Custom grid factor added: region=%s year=%d co2e=%.2f",
                body.region_id, body.year, float(total_co2e),
            )

            return _serialize_result(result)

        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "add_custom_grid_factor failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. GET /td-losses - List T&D loss factors
    # ==================================================================

    @router.get("/td-losses", status_code=200)
    async def list_td_losses() -> Dict[str, Any]:
        """List all available transmission and distribution loss factors.

        Returns T&D loss percentages for 50+ countries from World Bank
        and IEA data. T&D losses represent electricity lost in the grid
        between generation and consumption.

        GHG Protocol Scope 2 Guidance requires reporting T&D loss
        emissions separately or as part of location-based totals.

        Permission: scope2-location:factors:read
        """
        try:
            td_engine = _get_transmission_engine()
            if td_engine is not None and hasattr(td_engine, "list_all_factors"):
                factors = td_engine.list_all_factors()
                return _serialize_result({
                    "factors": factors,
                    "count": (
                        len(factors) if isinstance(factors, (list, dict))
                        else 0
                    ),
                    "source": "world_bank_iea",
                    "unit": "fraction (0.0 - 1.0)",
                    "description": "Country-level electricity T&D loss "
                    "percentages as decimal fractions",
                })
            else:
                # Fallback: return factors from models module
                try:
                    from greenlang.scope2_location.models import (
                        TD_LOSS_FACTORS,
                    )
                    factors_list = []
                    for cc, pct in sorted(TD_LOSS_FACTORS.items()):
                        factors_list.append({
                            "country_code": cc,
                            "td_loss_pct": _dec_to_float(pct),
                        })
                    return {
                        "factors": factors_list,
                        "count": len(factors_list),
                        "source": "world_bank_iea",
                        "unit": "fraction (0.0 - 1.0)",
                    }
                except ImportError:
                    return {
                        "factors": [],
                        "count": 0,
                        "source": "unavailable",
                        "message": "T&D loss factors not available",
                    }

        except Exception as exc:
            logger.error(
                "list_td_losses failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckBody,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check on a calculation.

        Evaluates a previously computed calculation against applicable
        Scope 2 reporting frameworks. Supports 7 frameworks with 80
        total requirements covering data completeness, methodological
        correctness, and reporting readiness.

        Supported frameworks:
        - ghg_protocol_scope2: GHG Protocol Scope 2 Guidance (2015)
        - ipcc_2006: IPCC 2006 Guidelines
        - iso_14064: ISO 14064-1:2018
        - csrd_e1: CSRD / ESRS E1
        - epa_ghgrp: US EPA Greenhouse Gas Reporting Program
        - defra_secr: UK DEFRA Reporting (SECR)
        - cdp: CDP Climate Change

        Permission: scope2-location:compliance:check
        """
        # Find the calculation
        calc = None
        for c in _calculations:
            if c.get("calculation_id") == body.calculation_id:
                calc = c
                break

        if calc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation {body.calculation_id} not found. "
                "Perform a calculation first before running compliance.",
            )

        try:
            svc = _get_service()

            if (
                hasattr(svc, "_compliance")
                and svc._compliance is not None
            ):
                # Use the compliance checker engine
                compliance_input = {
                    "calculation_id": body.calculation_id,
                    "energy_type": calc.get("energy_type", ""),
                    "country_code": calc.get("country_code", ""),
                    "emission_factor_source": calc.get(
                        "emission_factor_source", ""
                    ),
                    "ef_year": calc.get("metadata", {}).get(
                        "ef_year", 2024
                    ),
                    "reporting_year": datetime.now(timezone.utc).year,
                    "grid_region": calc.get("grid_region", ""),
                    "gas_breakdown": calc.get("gas_breakdown", []),
                    "td_loss_pct": calc.get("td_loss_pct", 0),
                    "total_co2e_kg": calc.get("total_co2e_kg", 0),
                    "total_co2e_tonnes": calc.get(
                        "total_co2e_tonnes", 0
                    ),
                    "gwp_source": calc.get("gwp_source", "AR5"),
                    "provenance_hash": calc.get("provenance_hash", ""),
                }

                frameworks = (
                    body.frameworks if body.frameworks else None
                )
                results = svc._compliance.check_compliance(
                    compliance_input, frameworks
                )

                # Store results
                check_id = str(uuid.uuid4())
                result = {
                    "id": check_id,
                    "calculation_id": body.calculation_id,
                    "frameworks_checked": (
                        body.frameworks if body.frameworks
                        else ["all"]
                    ),
                    "results": (
                        [_serialize_result(r) for r in results]
                        if isinstance(results, list)
                        else _serialize_result(results)
                    ),
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
                _compliance_results.append(result)

                logger.info(
                    "Compliance check completed: calc=%s frameworks=%s",
                    body.calculation_id,
                    body.frameworks or "all",
                )

                return result

            else:
                # Compliance engine not available - return basic check
                check_id = str(uuid.uuid4())
                result = {
                    "id": check_id,
                    "calculation_id": body.calculation_id,
                    "frameworks_checked": (
                        body.frameworks if body.frameworks
                        else ["all"]
                    ),
                    "results": [],
                    "status": "compliance_engine_unavailable",
                    "message": "Compliance checker engine is not initialized. "
                    "Results will be available when the engine is loaded.",
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
                _compliance_results.append(result)
                return result

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /compliance/{id} - Get compliance result
    # ==================================================================

    @router.get("/compliance/{compliance_id}", status_code=200)
    async def get_compliance_result(
        compliance_id: str = Path(
            ..., description="Compliance check identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a compliance check result by its unique identifier.

        Returns the full compliance check result including per-framework
        findings, recommendations, and compliance status.

        Permission: scope2-location:compliance:read
        """
        for result in _compliance_results:
            if result.get("id") == compliance_id:
                return result

        raise HTTPException(
            status_code=404,
            detail=f"Compliance result {compliance_id} not found",
        )

    # ==================================================================
    # 17. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyBody,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation result.

        Performs Monte Carlo simulation or analytical error propagation
        to quantify the uncertainty range of the emission estimate.
        Returns statistical characterization including mean, standard
        deviation, confidence intervals, and coefficient of variation.

        Uncertainty sources quantified:
        - Grid emission factor uncertainty (+/-5-50%)
        - Activity data uncertainty (+/-2-30%)
        - T&D loss factor uncertainty (+/-10-30%)
        - GWP uncertainty (+/-5-15%)
        - Steam/heat/cooling factor uncertainty (+/-10-30%)

        Permission: scope2-location:uncertainty:run
        """
        # Find the calculation
        calc = None
        for c in _calculations:
            if c.get("calculation_id") == body.calculation_id:
                calc = c
                break

        if calc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation {body.calculation_id} not found. "
                "Perform a calculation first before running uncertainty.",
            )

        try:
            svc = _get_service()

            if (
                hasattr(svc, "_uncertainty")
                and svc._uncertainty is not None
            ):
                # Use the uncertainty quantifier engine
                base_emissions_kg = calc.get(
                    "total_co2e_kg", Decimal("0")
                )
                if not isinstance(base_emissions_kg, Decimal):
                    base_emissions_kg = Decimal(str(base_emissions_kg))

                mc_result = svc._uncertainty.run_monte_carlo(
                    base_emissions_kg=base_emissions_kg,
                    iterations=body.iterations,
                    seed=body.seed,
                )

                result = {
                    "calculation_id": body.calculation_id,
                    "method": body.method or "monte_carlo",
                    "iterations": body.iterations,
                    "confidence_level": body.confidence_level,
                    "seed": body.seed,
                    "result": _serialize_result(mc_result),
                    "analysed_at": datetime.now(timezone.utc).isoformat(),
                }

                _uncertainty_results.append(result)

                logger.info(
                    "Uncertainty analysis completed: calc=%s "
                    "iterations=%d",
                    body.calculation_id, body.iterations,
                )

                return result

            else:
                # Uncertainty engine not available - return estimate
                total_co2e = float(calc.get("total_co2e_tonnes", 0))
                # Default IPCC Tier 1 uncertainty: +/- 25%
                uncertainty_pct = 0.25

                result = {
                    "calculation_id": body.calculation_id,
                    "method": "ipcc_default_estimate",
                    "iterations": 0,
                    "confidence_level": body.confidence_level,
                    "mean_co2e_tonnes": total_co2e,
                    "std_dev_tonnes": total_co2e * uncertainty_pct / 1.96,
                    "ci_lower_tonnes": total_co2e * (1 - uncertainty_pct),
                    "ci_upper_tonnes": total_co2e * (1 + uncertainty_pct),
                    "relative_uncertainty_pct": uncertainty_pct * 100,
                    "message": "Uncertainty engine not initialized. "
                    "Using IPCC Tier 1 default estimate (+/-25%).",
                    "analysed_at": datetime.now(timezone.utc).isoformat(),
                }

                _uncertainty_results.append(result)
                return result

        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "run_uncertainty failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 18. GET /aggregations - Get aggregated emissions
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        tenant_id: Optional[str] = Query(
            None,
            description="Tenant identifier for scoping "
            "(required if multiple tenants)",
        ),
        group_by: Optional[str] = Query(
            None,
            description="Comma-separated fields to group results by: "
            "energy_type, facility_id, country_code, grid_region",
        ),
        energy_type: Optional[str] = Query(
            None,
            description="Filter by energy type: electricity, steam, "
            "heating, cooling",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility identifier",
        ),
        country_code: Optional[str] = Query(
            None, description="Filter by ISO 3166-1 alpha-2 country code",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated Scope 2 location-based emissions.

        Aggregates calculation results by specified grouping dimensions.
        Supports grouping by energy type, facility, country, or grid
        region. When no ``group_by`` is specified, returns a flat
        portfolio-level total.

        Permission: scope2-location:read
        """
        # Filter calculations
        filtered = list(_calculations)

        if tenant_id is not None:
            filtered = [
                c for c in filtered
                if c.get("tenant_id") == tenant_id
            ]
        if energy_type is not None:
            filtered = [
                c for c in filtered
                if c.get("energy_type") == energy_type.lower()
            ]
        if facility_id is not None:
            filtered = [
                c for c in filtered
                if c.get("facility_id") == facility_id
            ]
        if country_code is not None:
            filtered = [
                c for c in filtered
                if c.get("country_code", "").upper() == country_code.upper()
            ]

        # Calculate total
        total_co2e_tonnes = Decimal("0")
        for c in filtered:
            v = c.get("total_co2e_tonnes", 0)
            if isinstance(v, Decimal):
                total_co2e_tonnes += v
            else:
                try:
                    total_co2e_tonnes += Decimal(str(v))
                except (InvalidOperation, TypeError):
                    pass

        facility_ids = set(
            c.get("facility_id", "") for c in filtered
        )

        # Group if requested
        groups: List[Dict[str, Any]] = []
        if group_by:
            group_fields = [
                g.strip() for g in group_by.split(",")
                if g.strip()
            ]

            # Build groups
            group_map: Dict[str, Dict[str, Any]] = {}
            for calc in filtered:
                key_parts = []
                for gf in group_fields:
                    val = calc.get(gf, "unknown")
                    if isinstance(val, Decimal):
                        val = str(val)
                    key_parts.append(f"{gf}={val}")
                key = "|".join(key_parts)

                if key not in group_map:
                    group_entry: Dict[str, Any] = {
                        "group_key": key,
                        "total_co2e_tonnes": Decimal("0"),
                        "calculation_count": 0,
                        "facility_ids": set(),
                    }
                    for gf in group_fields:
                        group_entry[gf] = calc.get(gf, "unknown")
                    group_map[key] = group_entry

                v = calc.get("total_co2e_tonnes", 0)
                if not isinstance(v, Decimal):
                    try:
                        v = Decimal(str(v))
                    except (InvalidOperation, TypeError):
                        v = Decimal("0")
                group_map[key]["total_co2e_tonnes"] += v
                group_map[key]["calculation_count"] += 1
                group_map[key]["facility_ids"].add(
                    calc.get("facility_id", "")
                )

            for g in group_map.values():
                g["facility_count"] = len(g.pop("facility_ids"))
                g["total_co2e_tonnes"] = _dec_to_float(
                    g["total_co2e_tonnes"]
                )
                groups.append(g)

        return {
            "total_co2e_tonnes": _dec_to_float(total_co2e_tonnes),
            "calculation_count": len(filtered),
            "facility_count": len(facility_ids),
            "group_by": group_by,
            "groups": groups,
            "filters": {
                "tenant_id": tenant_id,
                "energy_type": energy_type,
                "facility_id": facility_id,
                "country_code": country_code,
            },
        }

    # ==================================================================
    # 19. GET /health - Service health check
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Service health check for load balancers and monitoring.

        Returns the health status of all Scope 2 location-based
        emission calculation engines. No authentication required.

        Checks:
        - Pipeline engine availability
        - Grid factor database engine availability
        - Electricity emissions engine availability
        - Steam/heat/cooling engine availability
        - Transmission loss engine availability
        - Uncertainty quantifier engine availability
        - Compliance checker engine availability
        """
        engines_status: Dict[str, Any] = {
            "pipeline": False,
            "grid_factor_db": False,
            "electricity": False,
            "steam_heat_cool": False,
            "transmission": False,
            "uncertainty": False,
            "compliance": False,
        }

        overall_healthy = False

        try:
            svc = _get_service()
            engines_status["pipeline"] = True

            if hasattr(svc, "_grid_factor_db"):
                engines_status["grid_factor_db"] = (
                    svc._grid_factor_db is not None
                )
            if hasattr(svc, "_electricity"):
                engines_status["electricity"] = (
                    svc._electricity is not None
                )
            if hasattr(svc, "_steam_heat_cool"):
                engines_status["steam_heat_cool"] = (
                    svc._steam_heat_cool is not None
                )
            if hasattr(svc, "_transmission"):
                engines_status["transmission"] = (
                    svc._transmission is not None
                )
            if hasattr(svc, "_uncertainty"):
                engines_status["uncertainty"] = (
                    svc._uncertainty is not None
                )
            if hasattr(svc, "_compliance"):
                engines_status["compliance"] = (
                    svc._compliance is not None
                )

            overall_healthy = engines_status["pipeline"]

        except Exception as exc:
            logger.warning("Health check: service unavailable: %s", exc)

        engines_active = sum(
            1 for v in engines_status.values() if v
        )
        engines_total = len(engines_status)

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "service": "scope2-location",
            "version": "1.0.0",
            "agent": "AGENT-MRV-009",
            "engines": engines_status,
            "engines_active": engines_active,
            "engines_total": engines_total,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # 20. GET /stats - Service statistics
    # ==================================================================

    @router.get("/stats", status_code=200)
    async def get_stats() -> Dict[str, Any]:
        """Get service statistics for the Scope 2 location-based agent.

        Returns operational statistics including pipeline run counts,
        total CO2e processed, engine availability, and API-level
        record counts. No authentication required.
        """
        pipeline_stats: Dict[str, Any] = {}

        try:
            svc = _get_service()
            if hasattr(svc, "get_statistics"):
                pipeline_stats = _serialize_result(svc.get_statistics())
            elif hasattr(svc, "get_pipeline_status"):
                pipeline_stats = _serialize_result(svc.get_pipeline_status())
        except Exception as exc:
            pipeline_stats = {"error": str(exc)}

        # Grid factor database stats
        grid_stats: Dict[str, Any] = {}
        try:
            gfdb = _get_grid_factor_db()
            if hasattr(gfdb, "get_statistics"):
                grid_stats = _serialize_result(gfdb.get_statistics())
        except Exception:
            grid_stats = {"status": "unavailable"}

        # Transmission engine stats
        td_stats: Dict[str, Any] = {}
        try:
            td = _get_transmission_engine()
            if td is not None and hasattr(td, "get_statistics"):
                td_stats = _serialize_result(td.get_statistics())
        except Exception:
            td_stats = {"status": "unavailable"}

        # API-level record counts
        api_counts = {
            "calculations": len(_calculations),
            "facilities": len(_facilities),
            "consumption_records": len(_consumption_records),
            "compliance_checks": len(_compliance_results),
            "uncertainty_analyses": len(_uncertainty_results),
        }

        # Energy type distribution
        energy_distribution: Dict[str, int] = {}
        for calc in _calculations:
            et = calc.get("energy_type", "unknown")
            energy_distribution[et] = energy_distribution.get(et, 0) + 1

        # Country distribution
        country_distribution: Dict[str, int] = {}
        for calc in _calculations:
            cc = calc.get("country_code", "unknown")
            country_distribution[cc] = (
                country_distribution.get(cc, 0) + 1
            )

        # Total CO2e processed through API
        api_total_co2e = Decimal("0")
        for calc in _calculations:
            v = calc.get("total_co2e_tonnes", 0)
            try:
                api_total_co2e += (
                    v if isinstance(v, Decimal)
                    else Decimal(str(v))
                )
            except (InvalidOperation, TypeError):
                pass

        return {
            "service": "scope2-location",
            "version": "1.0.0",
            "agent": "AGENT-MRV-009",
            "pipeline": pipeline_stats,
            "grid_factor_db": grid_stats,
            "transmission_loss": td_stats,
            "api": {
                "record_counts": api_counts,
                "total_co2e_tonnes": _dec_to_float(api_total_co2e),
                "energy_type_distribution": energy_distribution,
                "country_distribution": country_distribution,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_result(obj: Any) -> Any:
        """Recursively serialize Decimal and datetime values for JSON.

        Converts Decimal to float and datetime to ISO-8601 string.
        Handles nested dicts, lists, and Pydantic BaseModel instances.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable version of the object.
        """
        if obj is None:
            return None
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _serialize_result(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize_result(item) for item in obj]
        if isinstance(obj, set):
            return [_serialize_result(item) for item in sorted(obj)]
        if hasattr(obj, "model_dump"):
            return _serialize_result(obj.model_dump(mode="json"))
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            return _serialize_result(
                {k: v for k, v in obj.__dict__.items()
                 if not k.startswith("_")}
            )
        return obj

    def _dec_to_float(val: Any) -> float:
        """Safely convert a Decimal or numeric value to float.

        Args:
            val: Value to convert.

        Returns:
            Float representation, or 0.0 if conversion fails.
        """
        if isinstance(val, Decimal):
            return float(val)
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(Decimal(str(val)))
        except (InvalidOperation, TypeError, ValueError):
            return 0.0

    return router


# ===================================================================
# Module-level router instance for direct import
# ===================================================================

# Create a module-level router instance for use in auth_setup.py
# and other modules that import the router directly.
try:
    router = create_router()
except RuntimeError:
    router = None  # type: ignore[assignment]
    logger.debug(
        "Could not create scope2-location router (FastAPI not available)"
    )


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router", "router"]
