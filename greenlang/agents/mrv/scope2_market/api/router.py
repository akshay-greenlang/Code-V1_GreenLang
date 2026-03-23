# -*- coding: utf-8 -*-
"""
Scope 2 Market-Based Emissions REST API Router - AGENT-MRV-010
================================================================

20 REST endpoints for the Scope 2 Market-Based Emissions Agent
(GL-MRV-SCOPE2-002).

Prefix: ``/api/v1/scope2-market``

Endpoints:
     1. POST   /calculations                - Execute single market-based calculation
     2. POST   /calculations/batch          - Execute batch calculations
     3. GET    /calculations                - List calculations with filters
     4. GET    /calculations/{id}           - Get calculation by ID
     5. DELETE /calculations/{id}           - Delete calculation
     6. POST   /facilities                  - Register a facility
     7. GET    /facilities                  - List facilities with filters
     8. PUT    /facilities/{id}             - Update a facility
     9. POST   /instruments                 - Register contractual instrument
    10. GET    /instruments                 - List instruments with filters
    11. POST   /instruments/{id}/retire     - Retire instrument
    12. POST   /compliance/check            - Run compliance check
    13. GET    /compliance/{id}             - Get compliance result
    14. POST   /uncertainty                 - Run uncertainty analysis
    15. POST   /dual-report                 - Generate dual report
    16. GET    /aggregations                - Get aggregated emissions
    17. GET    /coverage/{facility_id}      - Get coverage analysis
    18. GET    /health                      - Service health check
    19. GET    /stats                       - Service statistics
    20. GET    /engines                     - Engine availability status

GHG Protocol Scope 2 Guidance (2015) market-based method implementation for
purchased electricity, steam, heating, and cooling emission calculations
using contractual instruments (RECs, GOs, PPAs, I-RECs, supplier-specific
emission factors) and residual mix factors.

The market-based method applies contractual instruments in a quality
hierarchy to energy purchases. Consumption covered by instruments uses
the instrument's emission factor; uncovered consumption uses the grid
residual mix factor. This produces emission totals that reflect an
organisation's electricity procurement decisions.

All emission factors and intermediate values use Python ``Decimal`` for
zero-hallucination deterministic arithmetic. No LLM involvement in any
calculation path. Every result carries a SHA-256 provenance hash for
full audit trails.

Supported Contractual Instruments (10 types):
    - PPA: Power Purchase Agreement (physical or virtual)
    - REC: Renewable Energy Certificate (US)
    - GO: Guarantee of Origin (EU)
    - REGO: Renewable Energy Guarantee of Origin (UK)
    - I-REC: International Renewable Energy Certificate
    - T-REC: Tradeable Renewable Energy Certificate (Asia)
    - J-Credit: J-Credit (Japan)
    - LGC: Large-scale Generation Certificate (Australia)
    - Green Tariff: Green electricity tariff
    - Supplier-Specific: Supplier-specific emission factor

Supported Regulatory Frameworks:
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
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
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
    logger.debug("FastAPI not installed; scope2-market router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    # ---------------------------------------------------------------
    # Calculation request models
    # ---------------------------------------------------------------

    class EnergyPurchaseItem(BaseModel):
        """A single energy purchase entry within a calculation request.

        Represents a purchased electricity, steam, heating, or cooling
        quantity that may be partially or fully covered by contractual
        instruments. When instruments are attached, the market-based
        method applies instrument emission factors to covered MWh and
        residual mix factors to uncovered MWh.
        """

        purchase_id: Optional[str] = Field(
            default=None,
            description="Optional purchase identifier "
            "(auto-generated UUID if omitted)",
        )
        quantity: float = Field(
            ..., ge=0,
            description="Energy quantity in the specified unit",
        )
        unit: str = Field(
            default="mwh",
            description="Unit of measurement: kwh, mwh, gj, mmbtu, or therms",
        )
        energy_type: str = Field(
            default="electricity",
            description="Energy type: electricity, steam, heating, or cooling",
        )
        region: Optional[str] = Field(
            default=None,
            description="Grid region for residual mix lookup "
            "(e.g. US-CAMX, EU-DE, UK, GLOBAL)",
        )
        instruments: Optional[List[Dict[str, Any]]] = Field(
            default=None,
            description="List of contractual instruments applied to this "
            "purchase. Each dict should contain instrument_id, "
            "instrument_type, mwh, and optional emission_factor, "
            "vintage_year, energy_source, tracking_system fields",
        )

    class CalculateRequest(BaseModel):
        """Request body for a single Scope 2 market-based emission calculation.

        Accepts energy purchases and associated contractual instruments,
        computes covered and uncovered emissions using the GHG Protocol
        market-based method, and returns a full result with per-gas
        breakdown, instrument allocation, coverage analysis, and
        provenance hash.

        Formula (electricity):
            Covered Emissions   = sum(instrument_mwh_i x instrument_ef_i)
            Uncovered Emissions = uncovered_mwh x residual_mix_ef
            Total Emissions     = Covered + Uncovered
        """

        facility_id: str = Field(
            ...,
            min_length=1,
            description="Reference to the consuming facility",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )
        energy_purchases: List[EnergyPurchaseItem] = Field(
            ..., min_length=1,
            description="List of energy purchase entries, each with quantity, "
            "unit, energy type, optional region, and optional instruments",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source: AR4, AR5, AR6, or AR6_20YR "
            "(default: AR5)",
        )
        region: Optional[str] = Field(
            default=None,
            description="Default grid region for residual mix lookup. "
            "Applied to purchases that do not specify a region",
        )
        compliance_frameworks: Optional[List[str]] = Field(
            default=None,
            description="Regulatory frameworks to check on completion "
            "(e.g. ghg_protocol_scope2, ipcc_2006, csrd_e1)",
        )
        include_compliance: Optional[bool] = Field(
            default=False,
            description="Whether to include regulatory compliance checks "
            "in the calculation result",
        )

    class BatchCalculationBody(BaseModel):
        """Request body for batch Scope 2 market-based emission calculations.

        Processes multiple calculation requests in a single batch. Each
        item in the ``requests`` list follows the same schema as the
        pipeline ``run_pipeline`` method's request format.
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
            "containing facility_id, energy_purchases, region, "
            "and optional instrument and compliance fields",
        )

    # ---------------------------------------------------------------
    # Facility request models
    # ---------------------------------------------------------------

    class FacilityBody(BaseModel):
        """Request body for registering a new facility.

        Creates a facility record with geographic, grid region, and
        classification metadata. Every facility is scoped to a tenant
        and mapped to a grid region for residual mix factor lookup.
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
            description="Grid region identifier for residual mix lookup "
            "(defaults to country code if omitted)",
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
            description="Grid region identifier for residual mix lookup",
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
    # Instrument request models
    # ---------------------------------------------------------------

    class InstrumentRequest(BaseModel):
        """Request body for registering a contractual instrument.

        Contractual instruments convey emission factor information from
        an electricity generator to the consumer under the GHG Protocol
        Scope 2 market-based method. Each instrument has a type, energy
        source, emission factor, vintage year, and tracking system.
        """

        instrument_id: Optional[str] = Field(
            default=None,
            description="Optional instrument identifier "
            "(auto-generated UUID if omitted)",
        )
        type: str = Field(
            ...,
            description="Instrument type: ppa, rec, go, rego, i_rec, "
            "t_rec, j_credit, lgc, green_tariff, supplier_specific",
        )
        quantity_mwh: float = Field(
            ..., ge=0,
            description="Instrument quantity in MWh",
        )
        energy_source: Optional[str] = Field(
            default=None,
            description="Generation energy source: solar, wind, hydro, "
            "nuclear, biomass, geothermal, natural_gas_ccgt, "
            "natural_gas_ocgt, coal, oil, mixed",
        )
        emission_factor: Optional[float] = Field(
            default=None, ge=0,
            description="Instrument emission factor in tCO2e/MWh. "
            "For renewable instruments this is typically 0.0",
        )
        vintage_year: Optional[int] = Field(
            default=None, ge=1990, le=2100,
            description="Generation vintage year for the instrument",
        )
        tracking_system: Optional[str] = Field(
            default=None,
            description="Tracking system registry: green_e, aib_eecs, "
            "ofgem, i_rec_standard, m_rets, nar, wregis, custom",
        )
        certificate_id: Optional[str] = Field(
            default=None, max_length=200,
            description="Certificate serial number or reference ID "
            "from the tracking system",
        )
        region: Optional[str] = Field(
            default=None,
            description="Geographic region where the instrument was "
            "generated (e.g. US-CAMX, EU-DE, UK)",
        )
        supplier_id: Optional[str] = Field(
            default=None,
            description="Supplier identifier for supplier-specific "
            "emission factors",
        )
        is_renewable: Optional[bool] = Field(
            default=None,
            description="Whether the instrument represents renewable "
            "electricity (auto-detected from energy_source if omitted)",
        )
        contract_start: Optional[str] = Field(
            default=None,
            description="Contract start date (ISO-8601)",
        )
        contract_end: Optional[str] = Field(
            default=None,
            description="Contract end date (ISO-8601)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # ---------------------------------------------------------------
    # Dual report request model
    # ---------------------------------------------------------------

    class DualReportRequest(BaseModel):
        """Request body for generating a dual Scope 2 report.

        GHG Protocol Scope 2 Guidance mandates reporting both location-
        and market-based totals side by side. This endpoint compares a
        location-based result (from AGENT-MRV-009) with a market-based
        result (from AGENT-MRV-010) to produce a unified dual report.
        """

        location_result: Dict[str, Any] = Field(
            ...,
            description="Location-based calculation result dict containing "
            "at minimum total_co2e_tonnes and facility_id",
        )
        market_result: Dict[str, Any] = Field(
            ...,
            description="Market-based calculation result dict containing "
            "at minimum total_co2e_tonnes and facility_id",
        )
        facility_id: Optional[str] = Field(
            default=None,
            description="Facility identifier (overrides facility_id "
            "in the result dicts if provided)",
        )
        reporting_period: Optional[str] = Field(
            default=None,
            description="Reporting period label (e.g. 2025, Q1-2025)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # ---------------------------------------------------------------
    # Compliance request model
    # ---------------------------------------------------------------

    class ComplianceCheckBody(BaseModel):
        """Request body for a regulatory compliance check.

        Evaluates a completed market-based calculation against one or
        more regulatory frameworks. Accepts either a calculation_id
        (to look up a stored result) or a calculation_result dict
        (inline result data).
        """

        calculation_id: Optional[str] = Field(
            default=None,
            description="ID of a previous calculation to check. "
            "Mutually exclusive with calculation_result",
        )
        calculation_result: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Inline calculation result dict. "
            "Mutually exclusive with calculation_id",
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
        on a previously computed market-based calculation result to
        quantify the uncertainty range of the emission estimate.
        """

        calculation_id: str = Field(
            ..., min_length=1,
            description="ID of a previous calculation to analyse",
        )
        method: Optional[str] = Field(
            default="monte_carlo",
            description="Uncertainty method: monte_carlo or analytical",
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


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Scope 2 Market-Based Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints covering calculations,
        facilities, instruments, compliance, uncertainty, dual reporting,
        aggregations, coverage, health, statistics, and engine status.

    Raises:
        RuntimeError: If FastAPI is not installed in the environment.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the scope2-market router"
        )

    router = APIRouter(
        prefix="/api/v1/scope2-market",
        tags=["scope2-market"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the Scope2MarketPipelineEngine singleton.

        Returns the initialized pipeline engine. Raises HTTPException 503
        if the service has not been initialized.
        """
        try:
            from greenlang.agents.mrv.scope2_market.scope2_market_pipeline import (
                Scope2MarketPipelineEngine,
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Scope 2 Market pipeline engine not available",
            )

        # Try to get from setup module if it exists
        try:
            from greenlang.agents.mrv.scope2_market.setup import get_service
            svc = get_service()
            if svc is not None:
                return svc
        except (ImportError, AttributeError):
            pass

        # Fallback: create a default pipeline engine instance
        if not hasattr(_get_service, "_instance"):
            _get_service._instance = Scope2MarketPipelineEngine()
        return _get_service._instance

    def _get_instrument_db():
        """Get the ContractualInstrumentDatabaseEngine.

        Returns the instrument database engine from the pipeline, or
        creates a standalone instance if the pipeline is not available.
        Raises HTTPException 503 if the engine cannot be loaded.
        """
        try:
            svc = _get_service()
            if hasattr(svc, "_instrument_db") and svc._instrument_db is not None:
                return svc._instrument_db
        except HTTPException:
            pass

        try:
            from greenlang.agents.mrv.scope2_market.contractual_instrument_database import (
                ContractualInstrumentDatabaseEngine,
            )
            if not hasattr(_get_instrument_db, "_instance"):
                _get_instrument_db._instance = ContractualInstrumentDatabaseEngine()
            return _get_instrument_db._instance
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Contractual instrument database engine not available",
            )

    def _get_dual_reporting_engine():
        """Get the DualReportingEngine.

        Returns the dual reporting engine from the pipeline, or creates
        a standalone instance if the pipeline is not available.
        """
        try:
            svc = _get_service()
            if hasattr(svc, "_dual_reporting") and svc._dual_reporting is not None:
                return svc._dual_reporting
        except HTTPException:
            pass

        try:
            from greenlang.agents.mrv.scope2_market.dual_reporting import (
                DualReportingEngine,
            )
            if not hasattr(_get_dual_reporting_engine, "_instance"):
                _get_dual_reporting_engine._instance = DualReportingEngine()
            return _get_dual_reporting_engine._instance
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # In-memory stores for API-level state
    # ------------------------------------------------------------------

    _calculations: List[Dict[str, Any]] = []
    _facilities: List[Dict[str, Any]] = []
    _instruments: List[Dict[str, Any]] = []
    _compliance_results: List[Dict[str, Any]] = []
    _uncertainty_results: List[Dict[str, Any]] = []
    _dual_reports: List[Dict[str, Any]] = []

    # ==================================================================
    # 1. POST /calculations - Execute single market-based calculation
    # ==================================================================

    @router.post("/calculations", status_code=201)
    async def create_calculation(
        body: CalculateRequest,
    ) -> Dict[str, Any]:
        """Execute a single Scope 2 market-based emission calculation.

        Computes CO2e emissions for a facility's purchased energy using
        the GHG Protocol Scope 2 Guidance market-based method. Instruments
        are allocated to purchases following the quality hierarchy; any
        uncovered consumption uses the grid residual mix factor.

        Formula:
            Covered   = sum(instrument_mwh_i x instrument_ef_i)
            Uncovered = uncovered_mwh x residual_mix_ef
            Total     = Covered + Uncovered

        Per-gas breakdown: CO2, CH4, and N2O with configurable GWP source
        (IPCC AR4, AR5, AR6, or AR6 20-year).

        Permission: scope2-market:calculate
        """
        svc = _get_service()

        # Build pipeline request dict from the Pydantic model
        purchases = []
        for ep in body.energy_purchases:
            purchase_dict: Dict[str, Any] = {
                "purchase_id": ep.purchase_id or str(uuid.uuid4()),
                "mwh": Decimal(str(ep.quantity)),
                "energy_type": ep.energy_type.lower(),
                "unit": ep.unit.lower(),
            }
            if ep.region is not None:
                purchase_dict["region"] = ep.region
            if ep.instruments is not None:
                purchase_dict["instruments"] = ep.instruments
            purchases.append(purchase_dict)

        request_data: Dict[str, Any] = {
            "facility_id": body.facility_id,
            "region": body.region or "GLOBAL",
            "purchases": purchases,
            "include_compliance": body.include_compliance or False,
        }

        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id
        if body.gwp_source is not None:
            request_data["gwp_source"] = body.gwp_source.upper()
        if body.compliance_frameworks is not None:
            request_data["compliance_frameworks"] = body.compliance_frameworks

        # Flatten purchase-level instruments into top-level instruments list
        all_instruments: List[Dict[str, Any]] = []
        for p in purchases:
            if "instruments" in p and p["instruments"]:
                all_instruments.extend(p["instruments"])
        if all_instruments:
            request_data["instruments"] = all_instruments

        try:
            result = svc.run_pipeline(request_data)

            # Store result for later retrieval
            _calculations.append(result)

            logger.info(
                "Market-based calculation completed: id=%s facility=%s "
                "co2e=%.6f tonnes",
                result.get("calculation_id", ""),
                result.get("facility_id", ""),
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
        """Execute batch Scope 2 market-based emission calculations.

        Processes multiple calculation requests in a single batch. Each
        item in the ``requests`` list follows the same schema as the
        pipeline ``run_pipeline`` method. Returns aggregated portfolio-
        level totals alongside individual results.

        Permission: scope2-market:calculate
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
        facility_id: Optional[str] = Query(
            None, description="Filter by facility identifier",
        ),
        region: Optional[str] = Query(
            None, description="Filter by grid region (e.g. US-CAMX, EU-DE)",
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
        """List Scope 2 market-based calculation results with pagination.

        Returns calculation results filtered by tenant, facility, or
        region. Results are ordered by calculation timestamp (newest first).

        Permission: scope2-market:read
        """
        filtered = list(_calculations)

        if tenant_id is not None:
            filtered = [
                c for c in filtered
                if c.get("tenant_id") == tenant_id
            ]
        if facility_id is not None:
            filtered = [
                c for c in filtered
                if c.get("facility_id") == facility_id
            ]
        if region is not None:
            filtered = [
                c for c in filtered
                if c.get("region", "").upper() == region.upper()
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
        """Get a Scope 2 market-based calculation result by its unique ID.

        Returns the full calculation result including instrument allocation,
        covered and uncovered emissions, gas breakdown, compliance results,
        coverage analysis, and provenance hash.

        Permission: scope2-market:read
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
        """Delete a Scope 2 market-based calculation result by its ID.

        Permanently removes the calculation result from the in-memory
        store. This action cannot be undone.

        Permission: scope2-market:delete
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
        """Register a new facility for Scope 2 market-based emission tracking.

        Creates a facility record with geographic, grid region, and
        classification metadata. Every facility is scoped to a tenant
        and mapped to a grid region for residual mix factor lookup.

        The ``grid_region_id`` defaults to the ``country_code`` when
        not explicitly provided.

        Permission: scope2-market:facilities:write
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

        Permission: scope2-market:facilities:read
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

        Permission: scope2-market:facilities:write
        """
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }

        # Normalize country_code to uppercase if present
        if "country_code" in update_data:
            update_data["country_code"] = update_data["country_code"].upper()

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
    # 9. POST /instruments - Register contractual instrument
    # ==================================================================

    @router.post("/instruments", status_code=201)
    async def create_instrument(
        body: InstrumentRequest,
    ) -> Dict[str, Any]:
        """Register a new contractual instrument for market-based reporting.

        Creates an instrument record representing an energy attribute
        certificate (REC, GO, I-REC, REGO, LGC, T-REC), power purchase
        agreement (PPA), green tariff, J-Credit, or supplier-specific
        emission factor. Instruments are allocated to energy purchases
        during market-based calculations.

        The instrument is created in ACTIVE status. Use the retire
        endpoint to permanently retire the instrument against a
        specific calculation or reporting period.

        Permission: scope2-market:instruments:write
        """
        # Validate instrument type
        valid_types = {
            "ppa", "rec", "go", "rego", "i_rec", "t_rec",
            "j_credit", "lgc", "green_tariff", "supplier_specific",
        }
        instrument_type = body.type.lower()
        if instrument_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid instrument type '{body.type}'. "
                f"Must be one of: {sorted(valid_types)}",
            )

        instrument_id = body.instrument_id or str(uuid.uuid4())

        # Check for duplicate instrument_id
        for inst in _instruments:
            if inst.get("instrument_id") == instrument_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"Instrument {instrument_id} already exists",
                )

        # Determine renewable status
        renewable_sources = {
            "solar", "wind", "hydro", "nuclear", "biomass", "geothermal",
        }
        is_renewable = body.is_renewable
        if is_renewable is None and body.energy_source:
            is_renewable = body.energy_source.lower() in renewable_sources

        instrument = {
            "instrument_id": instrument_id,
            "type": instrument_type,
            "quantity_mwh": body.quantity_mwh,
            "energy_source": (
                body.energy_source.lower() if body.energy_source else None
            ),
            "emission_factor": body.emission_factor,
            "vintage_year": body.vintage_year,
            "tracking_system": body.tracking_system,
            "certificate_id": body.certificate_id,
            "region": body.region,
            "supplier_id": body.supplier_id,
            "is_renewable": is_renewable,
            "contract_start": body.contract_start,
            "contract_end": body.contract_end,
            "tenant_id": body.tenant_id,
            "status": "active",
            "remaining_mwh": body.quantity_mwh,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        _instruments.append(instrument)

        logger.info(
            "Instrument registered: id=%s type=%s mwh=%.2f source=%s",
            instrument_id, instrument_type, body.quantity_mwh,
            body.energy_source or "unspecified",
        )

        return instrument

    # ==================================================================
    # 10. GET /instruments - List instruments with filters
    # ==================================================================

    @router.get("/instruments", status_code=200)
    async def list_instruments(
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        instrument_type: Optional[str] = Query(
            None,
            description="Filter by instrument type: ppa, rec, go, rego, "
            "i_rec, t_rec, j_credit, lgc, green_tariff, supplier_specific",
        ),
        status: Optional[str] = Query(
            None,
            description="Filter by status: active, retired, expired, "
            "cancelled, pending",
        ),
        region: Optional[str] = Query(
            None, description="Filter by region (e.g. US-CAMX, EU-DE)",
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
        """List registered contractual instruments with pagination and filters.

        Returns instruments filtered by tenant, type, status, or region.
        Results are ordered by creation timestamp (newest first).

        Permission: scope2-market:instruments:read
        """
        filtered = list(_instruments)

        if tenant_id is not None:
            filtered = [
                inst for inst in filtered
                if inst.get("tenant_id") == tenant_id
            ]
        if instrument_type is not None:
            filtered = [
                inst for inst in filtered
                if inst.get("type") == instrument_type.lower()
            ]
        if status is not None:
            filtered = [
                inst for inst in filtered
                if inst.get("status") == status.lower()
            ]
        if region is not None:
            filtered = [
                inst for inst in filtered
                if (inst.get("region") or "").upper() == region.upper()
            ]

        total = len(filtered)
        page_data = filtered[skip: skip + limit]

        return {
            "instruments": page_data,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    # ==================================================================
    # 11. POST /instruments/{id}/retire - Retire instrument
    # ==================================================================

    @router.post("/instruments/{instrument_id}/retire", status_code=200)
    async def retire_instrument(
        instrument_id: str = Path(
            ..., description="Instrument identifier to retire",
        ),
        mwh: Optional[float] = Query(
            None, ge=0,
            description="MWh quantity to retire. If omitted, retires "
            "the full remaining quantity",
        ),
        calculation_id: Optional[str] = Query(
            None,
            description="Calculation ID to associate with the retirement",
        ),
        reporting_period: Optional[str] = Query(
            None,
            description="Reporting period for the retirement (e.g. 2025)",
        ),
    ) -> Dict[str, Any]:
        """Retire a contractual instrument (fully or partially).

        Permanently marks the instrument as retired, preventing it from
        being re-used or double-counted in future calculations. Partial
        retirement is supported by specifying the ``mwh`` quantity to
        retire.

        Once an instrument is fully retired (remaining_mwh reaches 0),
        its status changes to ``retired`` and it cannot be allocated
        to any further calculations.

        Permission: scope2-market:instruments:write
        """
        for i, inst in enumerate(_instruments):
            if inst.get("instrument_id") == instrument_id:
                current_status = inst.get("status", "active")
                if current_status == "retired":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Instrument {instrument_id} is already "
                        "fully retired",
                    )
                if current_status in ("expired", "cancelled"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Instrument {instrument_id} has status "
                        f"'{current_status}' and cannot be retired",
                    )

                remaining = float(inst.get("remaining_mwh", 0))
                retire_qty = mwh if mwh is not None else remaining

                if retire_qty > remaining:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Requested retirement of {retire_qty} MWh "
                        f"exceeds remaining {remaining} MWh",
                    )

                new_remaining = remaining - retire_qty
                new_status = "retired" if new_remaining <= 0 else "active"

                _instruments[i] = {
                    **inst,
                    "remaining_mwh": new_remaining,
                    "status": new_status,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                retirement_record = {
                    "instrument_id": instrument_id,
                    "retired_mwh": retire_qty,
                    "remaining_mwh": new_remaining,
                    "new_status": new_status,
                    "calculation_id": calculation_id,
                    "reporting_period": reporting_period,
                    "retired_at": datetime.now(timezone.utc).isoformat(),
                }

                logger.info(
                    "Instrument retired: id=%s mwh=%.2f remaining=%.2f "
                    "status=%s",
                    instrument_id, retire_qty, new_remaining, new_status,
                )

                return retirement_record

        raise HTTPException(
            status_code=404,
            detail=f"Instrument {instrument_id} not found",
        )

    # ==================================================================
    # 12. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckBody,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check on a market-based calculation.

        Evaluates a previously computed calculation against applicable
        Scope 2 reporting frameworks. Supports 7 frameworks covering
        data completeness, methodological correctness, instrument
        quality, and reporting readiness.

        Accepts either ``calculation_id`` (to look up a stored result)
        or ``calculation_result`` (inline data). At least one must be
        provided.

        Supported frameworks:
        - ghg_protocol_scope2: GHG Protocol Scope 2 Guidance (2015)
        - ipcc_2006: IPCC 2006 Guidelines
        - iso_14064: ISO 14064-1:2018
        - csrd_e1: CSRD / ESRS E1
        - epa_ghgrp: US EPA Greenhouse Gas Reporting Program
        - defra_secr: UK DEFRA Reporting (SECR)
        - cdp: CDP Climate Change

        Permission: scope2-market:compliance:check
        """
        # Resolve the calculation data
        calc = body.calculation_result
        if calc is None and body.calculation_id is not None:
            for c in _calculations:
                if c.get("calculation_id") == body.calculation_id:
                    calc = c
                    break
            if calc is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Calculation {body.calculation_id} not found. "
                    "Perform a calculation first or provide "
                    "calculation_result inline.",
                )
        elif calc is None:
            raise HTTPException(
                status_code=400,
                detail="Either calculation_id or calculation_result "
                "must be provided",
            )

        try:
            svc = _get_service()

            if (
                hasattr(svc, "_compliance")
                and svc._compliance is not None
            ):
                frameworks = (
                    body.frameworks if body.frameworks else None
                )
                results = svc._compliance.check_compliance(
                    calc, frameworks
                )

                check_id = str(uuid.uuid4())
                result = {
                    "id": check_id,
                    "calculation_id": (
                        body.calculation_id
                        or calc.get("calculation_id", "inline")
                    ),
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
                    body.calculation_id or "inline",
                    body.frameworks or "all",
                )

                return result

            else:
                # Compliance engine not available - return placeholder
                check_id = str(uuid.uuid4())
                result = {
                    "id": check_id,
                    "calculation_id": (
                        body.calculation_id
                        or calc.get("calculation_id", "inline")
                    ),
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
    # 13. GET /compliance/{id} - Get compliance result
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

        Permission: scope2-market:compliance:read
        """
        for result in _compliance_results:
            if result.get("id") == compliance_id:
                return result

        raise HTTPException(
            status_code=404,
            detail=f"Compliance result {compliance_id} not found",
        )

    # ==================================================================
    # 14. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyBody,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a market-based calculation result.

        Performs Monte Carlo simulation or analytical error propagation
        to quantify the uncertainty range of the emission estimate.
        Returns statistical characterization including mean, standard
        deviation, confidence intervals, and coefficient of variation.

        Uncertainty sources quantified:
        - Instrument emission factor uncertainty (+/-2-10%)
        - Supplier-specific EF uncertainty (+/-5-15%)
        - Residual mix factor uncertainty (+/-10-20%)
        - Activity data uncertainty (+/-2-30%)
        - Coverage allocation uncertainty (+/-1-5%)

        Permission: scope2-market:uncertainty:run
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
                    "method=%s iterations=%d",
                    body.calculation_id, body.method, body.iterations,
                )

                return result

            else:
                # Uncertainty engine not available - return IPCC estimate
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
    # 15. POST /dual-report - Generate dual report
    # ==================================================================

    @router.post("/dual-report", status_code=201)
    async def generate_dual_report(
        body: DualReportRequest,
    ) -> Dict[str, Any]:
        """Generate a GHG Protocol dual Scope 2 report.

        Combines a location-based result (from AGENT-MRV-009) and a
        market-based result (from AGENT-MRV-010) into a unified dual
        report as mandated by the GHG Protocol Scope 2 Guidance (2015).

        The dual report includes:
        - Side-by-side location-based vs. market-based totals
        - Procurement impact analysis (renewable reduction)
        - Coverage gap analysis
        - RE100 progress tracking
        - GHG Protocol Table 6.1 format output

        Permission: scope2-market:dual-report:write
        """
        try:
            dual_engine = _get_dual_reporting_engine()

            facility_id = body.facility_id or (
                body.market_result.get("facility_id")
                or body.location_result.get("facility_id")
                or "unknown"
            )

            if dual_engine is not None:
                report = dual_engine.generate_dual_report(
                    body.location_result, body.market_result,
                )

                # Enrich with request metadata
                report_result = _serialize_result(report)
                report_result["facility_id"] = facility_id
                report_result["reporting_period"] = body.reporting_period
                report_result["tenant_id"] = body.tenant_id
                report_result["report_id"] = str(uuid.uuid4())
                report_result["generated_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )

                _dual_reports.append(report_result)

                logger.info(
                    "Dual report generated: facility=%s period=%s",
                    facility_id, body.reporting_period,
                )

                return report_result

            else:
                # Dual reporting engine not available - compute basic report
                loc_co2e = _safe_decimal(
                    body.location_result.get("total_co2e_tonnes", 0)
                )
                mkt_co2e = _safe_decimal(
                    body.market_result.get("total_co2e_tonnes", 0)
                )
                difference = mkt_co2e - loc_co2e
                difference_pct = (
                    (difference / loc_co2e * Decimal("100"))
                    if loc_co2e > Decimal("0")
                    else Decimal("0")
                )

                report_result = {
                    "report_id": str(uuid.uuid4()),
                    "facility_id": facility_id,
                    "reporting_period": body.reporting_period,
                    "tenant_id": body.tenant_id,
                    "location_based_tco2e": _dec_to_float(loc_co2e),
                    "market_based_tco2e": _dec_to_float(mkt_co2e),
                    "difference_tco2e": _dec_to_float(difference),
                    "difference_pct": _dec_to_float(difference_pct),
                    "lower_method": (
                        "market" if mkt_co2e < loc_co2e
                        else "location" if loc_co2e < mkt_co2e
                        else "equal"
                    ),
                    "message": "Dual reporting engine not initialized. "
                    "Using basic comparison.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

                _dual_reports.append(report_result)

                logger.info(
                    "Dual report (basic) generated: facility=%s "
                    "loc=%.2f mkt=%.2f",
                    facility_id, float(loc_co2e), float(mkt_co2e),
                )

                return report_result

        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "generate_dual_report failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. GET /aggregations - Get aggregated emissions
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
            "facility_id, region, instrument_type, energy_type",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility identifier",
        ),
        region: Optional[str] = Query(
            None, description="Filter by grid region",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated Scope 2 market-based emissions.

        Aggregates calculation results by specified grouping dimensions.
        Supports grouping by facility, region, instrument type, or
        energy type. When no ``group_by`` is specified, returns a flat
        portfolio-level total.

        Permission: scope2-market:read
        """
        # Filter calculations
        filtered = list(_calculations)

        if tenant_id is not None:
            filtered = [
                c for c in filtered
                if c.get("tenant_id") == tenant_id
            ]
        if facility_id is not None:
            filtered = [
                c for c in filtered
                if c.get("facility_id") == facility_id
            ]
        if region is not None:
            filtered = [
                c for c in filtered
                if c.get("region", "").upper() == region.upper()
            ]

        # Calculate total
        total_co2e_tonnes = Decimal("0")
        total_covered_mwh = Decimal("0")
        total_uncovered_mwh = Decimal("0")
        for c in filtered:
            total_co2e_tonnes += _safe_decimal(
                c.get("total_co2e_tonnes", 0)
            )
            total_covered_mwh += _safe_decimal(
                c.get("covered_mwh", 0)
            )
            total_uncovered_mwh += _safe_decimal(
                c.get("uncovered_mwh", 0)
            )

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
                        "covered_mwh": Decimal("0"),
                        "uncovered_mwh": Decimal("0"),
                        "calculation_count": 0,
                        "facility_ids": set(),
                    }
                    for gf in group_fields:
                        group_entry[gf] = calc.get(gf, "unknown")
                    group_map[key] = group_entry

                group_map[key]["total_co2e_tonnes"] += _safe_decimal(
                    calc.get("total_co2e_tonnes", 0)
                )
                group_map[key]["covered_mwh"] += _safe_decimal(
                    calc.get("covered_mwh", 0)
                )
                group_map[key]["uncovered_mwh"] += _safe_decimal(
                    calc.get("uncovered_mwh", 0)
                )
                group_map[key]["calculation_count"] += 1
                group_map[key]["facility_ids"].add(
                    calc.get("facility_id", "")
                )

            for g in group_map.values():
                g["facility_count"] = len(g.pop("facility_ids"))
                g["total_co2e_tonnes"] = _dec_to_float(
                    g["total_co2e_tonnes"]
                )
                g["covered_mwh"] = _dec_to_float(g["covered_mwh"])
                g["uncovered_mwh"] = _dec_to_float(g["uncovered_mwh"])
                groups.append(g)

        total_mwh = total_covered_mwh + total_uncovered_mwh
        coverage_pct = (
            _dec_to_float(
                total_covered_mwh / total_mwh * Decimal("100")
            )
            if total_mwh > Decimal("0")
            else 0.0
        )

        return {
            "total_co2e_tonnes": _dec_to_float(total_co2e_tonnes),
            "total_covered_mwh": _dec_to_float(total_covered_mwh),
            "total_uncovered_mwh": _dec_to_float(total_uncovered_mwh),
            "coverage_pct": coverage_pct,
            "calculation_count": len(filtered),
            "facility_count": len(facility_ids),
            "group_by": group_by,
            "groups": groups,
            "filters": {
                "tenant_id": tenant_id,
                "facility_id": facility_id,
                "region": region,
            },
        }

    # ==================================================================
    # 17. GET /coverage/{facility_id} - Get coverage analysis
    # ==================================================================

    @router.get("/coverage/{facility_id}", status_code=200)
    async def get_coverage_analysis(
        facility_id: str = Path(
            ..., description="Facility identifier",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Tenant identifier for scoping",
        ),
    ) -> Dict[str, Any]:
        """Get instrument coverage analysis for a facility.

        Analyses the contractual instrument coverage for a specific
        facility's energy purchases across all stored calculations.
        Returns coverage metrics including total MWh, covered MWh,
        uncovered MWh, coverage percentage, and instrument breakdown.

        Coverage statuses:
        - FULLY_COVERED: 100% of MWh covered by instruments
        - PARTIALLY_COVERED: 1-99% of MWh covered
        - UNCOVERED: 0% covered (uses residual mix only)

        Permission: scope2-market:coverage:read
        """
        # Filter calculations for the facility
        facility_calcs = [
            c for c in _calculations
            if c.get("facility_id") == facility_id
        ]

        if tenant_id is not None:
            facility_calcs = [
                c for c in facility_calcs
                if c.get("tenant_id") == tenant_id
            ]

        if not facility_calcs:
            raise HTTPException(
                status_code=404,
                detail=f"No calculations found for facility {facility_id}",
            )

        # Aggregate coverage metrics
        total_mwh = Decimal("0")
        covered_mwh = Decimal("0")
        uncovered_mwh = Decimal("0")
        instrument_types_used: Dict[str, float] = {}

        for calc in facility_calcs:
            total_mwh += _safe_decimal(
                calc.get("total_mwh", 0)
            )
            covered_mwh += _safe_decimal(
                calc.get("covered_mwh", 0)
            )
            uncovered_mwh += _safe_decimal(
                calc.get("uncovered_mwh", 0)
            )

            # Track instrument types used in allocations
            allocations = calc.get("allocations", [])
            if isinstance(allocations, list):
                for alloc in allocations:
                    itype = alloc.get("instrument_type", "unknown")
                    alloc_mwh = float(
                        _safe_decimal(alloc.get("mwh", 0))
                    )
                    instrument_types_used[itype] = (
                        instrument_types_used.get(itype, 0.0) + alloc_mwh
                    )

        coverage_pct = (
            _dec_to_float(
                covered_mwh / total_mwh * Decimal("100")
            )
            if total_mwh > Decimal("0")
            else 0.0
        )

        # Determine coverage status
        if coverage_pct >= 100.0:
            coverage_status = "FULLY_COVERED"
        elif coverage_pct > 0.0:
            coverage_status = "PARTIALLY_COVERED"
        else:
            coverage_status = "UNCOVERED"

        # Facility instruments (registered but not necessarily allocated)
        facility_instruments = [
            inst for inst in _instruments
            if inst.get("tenant_id") in (
                tenant_id,
                facility_calcs[0].get("tenant_id") if facility_calcs else None,
            )
        ]
        active_instruments = [
            inst for inst in facility_instruments
            if inst.get("status") == "active"
        ]
        total_available_mwh = sum(
            float(inst.get("remaining_mwh", 0))
            for inst in active_instruments
        )

        return {
            "facility_id": facility_id,
            "coverage_status": coverage_status,
            "coverage_pct": coverage_pct,
            "total_mwh": _dec_to_float(total_mwh),
            "covered_mwh": _dec_to_float(covered_mwh),
            "uncovered_mwh": _dec_to_float(uncovered_mwh),
            "calculation_count": len(facility_calcs),
            "instrument_types_used": instrument_types_used,
            "available_instruments": len(active_instruments),
            "available_instrument_mwh": total_available_mwh,
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # 18. GET /health - Service health check
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Service health check for load balancers and monitoring.

        Returns the health status of all Scope 2 market-based
        emission calculation engines. No authentication required.

        Checks:
        - Pipeline engine availability
        - Instrument database engine availability
        - Instrument allocation engine availability
        - Covered emissions engine availability
        - Residual mix engine availability
        - Uncertainty quantifier engine availability
        - Compliance checker engine availability
        """
        engines_status: Dict[str, Any] = {
            "pipeline": False,
            "instrument_db": False,
            "allocation": False,
            "covered": False,
            "residual_mix": False,
            "uncertainty": False,
            "compliance": False,
        }

        overall_healthy = False

        try:
            svc = _get_service()
            engines_status["pipeline"] = True

            if hasattr(svc, "_instrument_db"):
                engines_status["instrument_db"] = (
                    svc._instrument_db is not None
                )
            if hasattr(svc, "_allocation"):
                engines_status["allocation"] = (
                    svc._allocation is not None
                )
            if hasattr(svc, "_covered"):
                engines_status["covered"] = (
                    svc._covered is not None
                )
            if hasattr(svc, "_residual_mix"):
                engines_status["residual_mix"] = (
                    svc._residual_mix is not None
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
            "service": "scope2-market",
            "version": "1.0.0",
            "agent": "AGENT-MRV-010",
            "engines": engines_status,
            "engines_active": engines_active,
            "engines_total": engines_total,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # 19. GET /stats - Service statistics
    # ==================================================================

    @router.get("/stats", status_code=200)
    async def get_stats() -> Dict[str, Any]:
        """Get service statistics for the Scope 2 market-based agent.

        Returns operational statistics including pipeline run counts,
        total CO2e processed, instrument counts, coverage metrics,
        and API-level record counts. No authentication required.
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

        # Instrument database stats
        instrument_db_stats: Dict[str, Any] = {}
        try:
            idb = _get_instrument_db()
            if hasattr(idb, "get_statistics"):
                instrument_db_stats = _serialize_result(idb.get_statistics())
        except Exception:
            instrument_db_stats = {"status": "unavailable"}

        # API-level record counts
        api_counts = {
            "calculations": len(_calculations),
            "facilities": len(_facilities),
            "instruments": len(_instruments),
            "compliance_checks": len(_compliance_results),
            "uncertainty_analyses": len(_uncertainty_results),
            "dual_reports": len(_dual_reports),
        }

        # Instrument type distribution
        instrument_type_dist: Dict[str, int] = {}
        for inst in _instruments:
            itype = inst.get("type", "unknown")
            instrument_type_dist[itype] = (
                instrument_type_dist.get(itype, 0) + 1
            )

        # Instrument status distribution
        instrument_status_dist: Dict[str, int] = {}
        for inst in _instruments:
            istatus = inst.get("status", "unknown")
            instrument_status_dist[istatus] = (
                instrument_status_dist.get(istatus, 0) + 1
            )

        # Region distribution of calculations
        region_distribution: Dict[str, int] = {}
        for calc in _calculations:
            rgn = calc.get("region", "unknown")
            region_distribution[rgn] = (
                region_distribution.get(rgn, 0) + 1
            )

        # Total CO2e processed through API
        api_total_co2e = Decimal("0")
        api_total_covered_mwh = Decimal("0")
        api_total_uncovered_mwh = Decimal("0")
        for calc in _calculations:
            api_total_co2e += _safe_decimal(
                calc.get("total_co2e_tonnes", 0)
            )
            api_total_covered_mwh += _safe_decimal(
                calc.get("covered_mwh", 0)
            )
            api_total_uncovered_mwh += _safe_decimal(
                calc.get("uncovered_mwh", 0)
            )

        return {
            "service": "scope2-market",
            "version": "1.0.0",
            "agent": "AGENT-MRV-010",
            "pipeline": pipeline_stats,
            "instrument_db": instrument_db_stats,
            "api": {
                "record_counts": api_counts,
                "total_co2e_tonnes": _dec_to_float(api_total_co2e),
                "total_covered_mwh": _dec_to_float(api_total_covered_mwh),
                "total_uncovered_mwh": _dec_to_float(
                    api_total_uncovered_mwh
                ),
                "instrument_type_distribution": instrument_type_dist,
                "instrument_status_distribution": instrument_status_dist,
                "region_distribution": region_distribution,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ==================================================================
    # 20. GET /engines - Engine availability status
    # ==================================================================

    @router.get("/engines", status_code=200)
    async def get_engines() -> Dict[str, Any]:
        """Get detailed engine availability and version information.

        Returns the status and capabilities of each engine in the
        Scope 2 Market-Based Emissions Agent pipeline. Useful for
        debugging and operational monitoring.

        Engines (7):
        1. ContractualInstrumentDatabaseEngine - Instrument EF lookups
        2. InstrumentAllocationEngine - GHG Protocol hierarchy allocation
        3. CoveredEmissionsEngine - Covered emissions calculations
        4. ResidualMixEngine - Residual mix factor resolution
        5. UncertaintyQuantifierEngine - Monte Carlo uncertainty
        6. ComplianceCheckerEngine - Multi-framework compliance
        7. Scope2MarketPipelineEngine - 8-stage orchestrated pipeline

        No authentication required.
        """
        engines: List[Dict[str, Any]] = []

        engine_defs = [
            ("pipeline", "Scope2MarketPipelineEngine",
             "8-stage orchestrated market-based calculation pipeline"),
            ("instrument_db", "ContractualInstrumentDatabaseEngine",
             "Instrument metadata, residual mix, and EF storage"),
            ("allocation", "InstrumentAllocationEngine",
             "GHG Protocol hierarchy allocation of instruments"),
            ("covered", "CoveredEmissionsEngine",
             "Covered emissions calculation from allocated instruments"),
            ("residual_mix", "ResidualMixEngine",
             "Residual mix factor resolution by region"),
            ("uncertainty", "UncertaintyQuantifierEngine",
             "Monte Carlo and analytical uncertainty quantification"),
            ("compliance", "ComplianceCheckerEngine",
             "Multi-framework regulatory compliance checking"),
        ]

        try:
            svc = _get_service()
            for attr_name, class_name, description in engine_defs:
                engine_info: Dict[str, Any] = {
                    "name": class_name,
                    "attribute": attr_name,
                    "description": description,
                    "available": False,
                    "version": None,
                }

                if attr_name == "pipeline":
                    engine_info["available"] = True
                    engine_info["version"] = "1.0.0"
                elif hasattr(svc, f"_{attr_name}"):
                    engine_inst = getattr(svc, f"_{attr_name}")
                    engine_info["available"] = engine_inst is not None
                    if engine_inst is not None and hasattr(engine_inst, "version"):
                        engine_info["version"] = engine_inst.version

                engines.append(engine_info)

        except Exception as exc:
            logger.warning("Engine status check failed: %s", exc)
            for _, class_name, description in engine_defs:
                engines.append({
                    "name": class_name,
                    "description": description,
                    "available": False,
                    "error": str(exc),
                })

        engines_available = sum(
            1 for e in engines if e.get("available", False)
        )

        return {
            "service": "scope2-market",
            "agent": "AGENT-MRV-010",
            "engines": engines,
            "engines_available": engines_available,
            "engines_total": len(engines),
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

    def _safe_decimal(val: Any) -> Decimal:
        """Safely convert a value to Decimal.

        Args:
            val: Value to convert.

        Returns:
            Decimal representation, or Decimal("0") if conversion fails.
        """
        if isinstance(val, Decimal):
            return val
        try:
            return Decimal(str(val))
        except (InvalidOperation, TypeError, ValueError):
            return Decimal("0")

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
        "Could not create scope2-market router (FastAPI not available)"
    )


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router", "router"]
