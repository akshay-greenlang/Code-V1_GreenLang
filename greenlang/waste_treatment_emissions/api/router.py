# -*- coding: utf-8 -*-
"""
Waste Treatment Emissions REST API Router - AGENT-MRV-008
==========================================================

20 REST endpoints for the On-site Waste Treatment Emissions Agent
(GL-MRV-SCOPE1-008).

Prefix: ``/api/v1/waste-treatment-emissions``

Endpoints:
     1. POST   /calculations                         - Execute single calculation
     2. POST   /calculations/batch                   - Execute batch calculations
     3. GET    /calculations/{id}                    - Get calculation by ID
     4. GET    /calculations                         - List calculations with filters
     5. DELETE /calculations/{id}                    - Delete calculation
     6. POST   /facilities                           - Register treatment facility
     7. GET    /facilities                           - List facilities
     8. PUT    /facilities/{id}                      - Update facility metadata
     9. POST   /waste-streams                        - Register waste stream
    10. GET    /waste-streams                        - List waste streams
    11. PUT    /waste-streams/{id}                   - Update stream composition
    12. POST   /treatment-events                     - Record treatment event
    13. GET    /treatment-events                     - List events with filters
    14. POST   /methane-recovery                     - Record methane recovery
    15. GET    /methane-recovery/{facility_id}       - Get recovery history
    16. POST   /compliance/check                     - Run compliance check
    17. GET    /compliance/{id}                      - Get compliance result
    18. POST   /uncertainty                          - Run Monte Carlo analysis
    19. GET    /aggregations                         - Get aggregated emissions
    20. GET    /health                               - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query, Path
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not installed; router unavailable")


# ===================================================================
# Request body models (Pydantic)
# ===================================================================

if FASTAPI_AVAILABLE:

    # --------------------------------------------------------------
    # Calculation models
    # --------------------------------------------------------------

    class SingleCalculationRequest(BaseModel):
        """Request body for a single waste treatment emission calculation.

        Supports biological treatment (composting, anaerobic digestion),
        thermal treatment (incineration, open burning), wastewater
        treatment, and landfill-based on-site disposal calculations
        using IPCC Tier 1/2/3 methodologies.
        """

        tenant_id: str = Field(
            default="default",
            description="Owning tenant identifier",
        )
        facility_id: Optional[str] = Field(
            default=None,
            description="Reference to the treatment facility",
        )
        treatment_method: str = Field(
            ...,
            description="Waste treatment method "
            "(composting, incineration, anaerobic_digestion, "
            "open_burning, landfill, mechanical_biological, "
            "autoclaving, pyrolysis, gasification, "
            "wastewater_treatment)",
        )
        waste_category: str = Field(
            ...,
            description="Waste category "
            "(food_waste, garden_waste, paper_cardboard, "
            "wood_waste, textiles, rubber_leather, plastics, "
            "msw, industrial_solid, clinical, hazardous, "
            "construction_demolition, sludge, agricultural)",
        )
        waste_tonnes: float = Field(
            ..., gt=0,
            description="Mass of waste treated in metric tonnes",
        )
        calculation_method: str = Field(
            default="ipcc_tier_2",
            description="IPCC calculation tier "
            "(ipcc_tier_1, ipcc_tier_2, ipcc_tier_3)",
        )
        gwp_source: str = Field(
            default="AR6",
            description="GWP source (AR4, AR5, AR6, AR6_GTP)",
        )

        # Biological treatment parameters (composting / anaerobic digestion)
        composting_type: Optional[str] = Field(
            default=None,
            description="Composting type "
            "(windrow, in_vessel, static_pile, vermicomposting)",
        )
        management_quality: Optional[str] = Field(
            default=None,
            description="Composting management quality "
            "(well_managed, poorly_managed)",
        )
        biofilter_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Biofilter CH4 removal efficiency (0.0-1.0)",
        )
        volatile_solids_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of volatile solids in waste (0.0-1.0)",
        )
        bmp: Optional[float] = Field(
            default=None, ge=0.0,
            description="Biochemical methane potential "
            "(m3 CH4 / kg VS)",
        )
        digestion_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Anaerobic digestion efficiency (0.0-1.0)",
        )
        methane_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of CH4 in biogas (0.0-1.0)",
        )

        # Thermal treatment parameters (incineration / open burning)
        incinerator_type: Optional[str] = Field(
            default=None,
            description="Incinerator type "
            "(mass_burn, modular, fluidized_bed, rotary_kiln, "
            "moving_grate, stoker)",
        )
        oxidation_factor: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of carbon oxidized during combustion",
        )
        electric_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Electrical energy recovery efficiency",
        )
        thermal_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Thermal energy recovery efficiency",
        )
        grid_ef_electric: Optional[float] = Field(
            default=None, ge=0.0,
            description="Grid emission factor for displaced "
            "electricity (tCO2e/MWh)",
        )

        # Wastewater treatment parameters
        tow_kg_yr: Optional[float] = Field(
            default=None, ge=0.0,
            description="Total organically degradable material in "
            "wastewater (kg BOD/yr or kg COD/yr)",
        )
        bod_or_cod: Optional[str] = Field(
            default=None,
            description="Organic loading metric (BOD or COD)",
        )
        wastewater_system: Optional[str] = Field(
            default=None,
            description="Wastewater treatment system type "
            "(aerobic_centralized, aerobic_shallow, "
            "anaerobic_reactor, anaerobic_lagoon, "
            "septic_tank, latrine_dry, latrine_wet, "
            "sea_river_discharge, stagnant_sewer)",
        )
        flow_m3_yr: Optional[float] = Field(
            default=None, ge=0.0,
            description="Wastewater flow rate (m3/yr)",
        )
        n_influent: Optional[float] = Field(
            default=None, ge=0.0,
            description="Nitrogen in wastewater influent (kg N/yr)",
        )
        n_effluent: Optional[float] = Field(
            default=None, ge=0.0,
            description="Nitrogen in wastewater effluent (kg N/yr)",
        )

        # Methane recovery parameters
        capture_efficiency: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Methane capture efficiency (0.0-1.0)",
        )
        flare_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of captured methane sent to flare",
        )
        utilize_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fraction of captured methane utilized "
            "for energy",
        )
        flare_dre: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Flare destruction and removal efficiency",
        )

        # Compliance
        compliance_frameworks: Optional[List[str]] = Field(
            default=None,
            description="Regulatory frameworks for compliance "
            "(GHG_PROTOCOL, IPCC, CSRD, EU_ETS, "
            "US_EPA, UK_SECR, UNFCCC)",
        )

        # Metadata
        metadata: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Arbitrary key-value metadata attached "
            "to this calculation",
        )

    class BatchCalculationBody(BaseModel):
        """Request body for batch waste treatment emission calculations."""

        calculations: List[Dict[str, Any]] = Field(
            ..., min_length=1, max_length=10000,
            description="List of calculation request dictionaries "
            "(max 10,000 per batch)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source applied to all calculations "
            "(AR4, AR5, AR6)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier applied to all "
            "calculations in the batch",
        )

    # --------------------------------------------------------------
    # Facility models
    # --------------------------------------------------------------

    class FacilityBody(BaseModel):
        """Request body for registering a waste treatment facility."""

        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Human-readable facility name",
        )
        facility_type: str = Field(
            ...,
            description="Treatment facility type "
            "(composting_plant, incinerator, anaerobic_digester, "
            "mbt_plant, landfill, wastewater_treatment_plant, "
            "pyrolysis_plant, gasification_plant, "
            "autoclave_facility)",
        )
        capacity_tonnes_yr: float = Field(
            ..., gt=0,
            description="Annual treatment capacity in metric tonnes",
        )
        latitude: float = Field(
            ..., ge=-90, le=90,
            description="WGS84 latitude in decimal degrees",
        )
        longitude: float = Field(
            ..., ge=-180, le=180,
            description="WGS84 longitude in decimal degrees",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )
        country_code: Optional[str] = Field(
            default=None, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        operating_since: Optional[str] = Field(
            default=None,
            description="Date facility began operations (ISO-8601)",
        )
        technology_details: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Technology-specific configuration details",
        )
        emission_controls: Optional[List[str]] = Field(
            default=None,
            description="Emission control technologies installed "
            "(scrubber, esp, baghouse, scr, sncr, "
            "activated_carbon, biofilter)",
        )
        permits: Optional[List[str]] = Field(
            default=None,
            description="Environmental permit references",
        )

    class FacilityUpdateBody(BaseModel):
        """Request body for updating a waste treatment facility."""

        name: Optional[str] = Field(
            default=None, min_length=1, max_length=500,
            description="Human-readable facility name",
        )
        facility_type: Optional[str] = Field(
            default=None,
            description="Treatment facility type",
        )
        capacity_tonnes_yr: Optional[float] = Field(
            default=None, gt=0,
            description="Annual treatment capacity in metric tonnes",
        )
        technology_details: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Technology-specific configuration details",
        )
        emission_controls: Optional[List[str]] = Field(
            default=None,
            description="Emission control technologies installed",
        )
        permits: Optional[List[str]] = Field(
            default=None,
            description="Environmental permit references",
        )
        country_code: Optional[str] = Field(
            default=None, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )

    # --------------------------------------------------------------
    # Waste stream models
    # --------------------------------------------------------------

    class WasteStreamBody(BaseModel):
        """Request body for registering a waste stream."""

        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Human-readable waste stream name",
        )
        waste_category: str = Field(
            ...,
            description="Primary waste category "
            "(food_waste, garden_waste, paper_cardboard, "
            "wood_waste, textiles, rubber_leather, plastics, "
            "msw, industrial_solid, clinical, hazardous, "
            "construction_demolition, sludge, agricultural)",
        )
        source_type: str = Field(
            ...,
            description="Source of waste "
            "(residential, commercial, industrial, "
            "institutional, agricultural, construction)",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier",
        )
        facility_id: Optional[str] = Field(
            default=None,
            description="Facility handling this waste stream",
        )
        composition: Optional[Dict[str, float]] = Field(
            default=None,
            description="Waste composition fractions by material "
            "(keys: organic, paper, plastic, glass, metal, "
            "textile, other; values sum to 1.0)",
        )
        moisture_content: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Moisture content fraction (0.0-1.0)",
        )
        carbon_content: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Total carbon content fraction (dry basis)",
        )
        fossil_carbon_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fossil carbon fraction of total carbon",
        )
        doc_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Degradable organic carbon fraction",
        )
        annual_tonnes: Optional[float] = Field(
            default=None, gt=0,
            description="Expected annual throughput in tonnes",
        )

    class WasteStreamUpdateBody(BaseModel):
        """Request body for updating a waste stream."""

        name: Optional[str] = Field(
            default=None, min_length=1, max_length=500,
            description="Human-readable waste stream name",
        )
        waste_category: Optional[str] = Field(
            default=None,
            description="Primary waste category",
        )
        source_type: Optional[str] = Field(
            default=None,
            description="Source of waste",
        )
        facility_id: Optional[str] = Field(
            default=None,
            description="Facility handling this waste stream",
        )
        composition: Optional[Dict[str, float]] = Field(
            default=None,
            description="Waste composition fractions by material",
        )
        moisture_content: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Moisture content fraction (0.0-1.0)",
        )
        carbon_content: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Total carbon content fraction (dry basis)",
        )
        fossil_carbon_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Fossil carbon fraction of total carbon",
        )
        doc_fraction: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Degradable organic carbon fraction",
        )
        annual_tonnes: Optional[float] = Field(
            default=None, gt=0,
            description="Expected annual throughput in tonnes",
        )

    # --------------------------------------------------------------
    # Treatment event models
    # --------------------------------------------------------------

    class TreatmentEventBody(BaseModel):
        """Request body for recording a treatment event."""

        facility_id: str = Field(
            ..., min_length=1,
            description="Reference to the treatment facility",
        )
        waste_stream_id: Optional[str] = Field(
            default=None,
            description="Reference to the waste stream",
        )
        treatment_method: str = Field(
            ...,
            description="Treatment method applied "
            "(composting, incineration, anaerobic_digestion, "
            "open_burning, landfill, mechanical_biological, "
            "autoclaving, pyrolysis, gasification, "
            "wastewater_treatment)",
        )
        waste_category: str = Field(
            ...,
            description="Waste category treated",
        )
        waste_tonnes: float = Field(
            ..., gt=0,
            description="Mass of waste treated in metric tonnes",
        )
        event_date: str = Field(
            ...,
            description="Date of the treatment event (ISO-8601)",
        )
        event_duration_hours: Optional[float] = Field(
            default=None, gt=0,
            description="Duration of the treatment event in hours",
        )
        operating_temperature_c: Optional[float] = Field(
            default=None,
            description="Operating temperature in degrees Celsius",
        )
        energy_recovered_mwh: Optional[float] = Field(
            default=None, ge=0,
            description="Energy recovered during treatment (MWh)",
        )
        residue_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Mass of residue produced in metric tonnes",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the treatment event",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # --------------------------------------------------------------
    # Methane recovery model
    # --------------------------------------------------------------

    class MethaneRecoveryBody(BaseModel):
        """Request body for recording a methane recovery event."""

        facility_id: str = Field(
            ..., min_length=1,
            description="Reference to the treatment facility",
        )
        recovery_date: str = Field(
            ...,
            description="Date of methane recovery event (ISO-8601)",
        )
        methane_captured_tonnes: float = Field(
            ..., ge=0,
            description="Methane captured in metric tonnes CH4",
        )
        methane_flared_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Methane destroyed by flaring (tonnes CH4)",
        )
        methane_utilized_tonnes: Optional[float] = Field(
            default=None, ge=0,
            description="Methane utilized for energy (tonnes CH4)",
        )
        flare_dre: Optional[float] = Field(
            default=None, ge=0.0, le=1.0,
            description="Flare destruction and removal efficiency",
        )
        electricity_generated_mwh: Optional[float] = Field(
            default=None, ge=0,
            description="Electricity generated from captured "
            "methane (MWh)",
        )
        heat_generated_mwh: Optional[float] = Field(
            default=None, ge=0,
            description="Heat generated from captured methane (MWh)",
        )
        capture_system: Optional[str] = Field(
            default=None,
            description="Capture system type "
            "(enclosed_flare, open_flare, gas_engine, "
            "gas_turbine, boiler, fuel_cell, cng_upgrade)",
        )
        notes: Optional[str] = Field(
            default=None, max_length=2000,
            description="Optional notes about the recovery event",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier",
        )

    # --------------------------------------------------------------
    # Compliance & uncertainty models
    # --------------------------------------------------------------

    class ComplianceCheckBody(BaseModel):
        """Request body for a compliance check."""

        calculation_id: str = Field(
            default="",
            description="ID of a previous calculation (optional)",
        )
        frameworks: List[str] = Field(
            default_factory=list,
            description="Frameworks to check "
            "(empty = all frameworks). "
            "Options: GHG_PROTOCOL, IPCC, CSRD, "
            "EU_ETS, US_EPA, UK_SECR, UNFCCC",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    class UncertaintyBody(BaseModel):
        """Request body for Monte Carlo uncertainty analysis."""

        calculation_id: str = Field(
            ..., description="ID of a previous calculation",
        )
        iterations: int = Field(
            default=5000, gt=0, le=1_000_000,
            description="Monte Carlo iterations",
        )
        seed: int = Field(
            default=42, ge=0,
            description="Random seed for reproducibility",
        )
        confidence_level: float = Field(
            default=95.0, gt=0, lt=100,
            description="Confidence level percentage",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Waste Treatment Emissions FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints.

    Raises:
        RuntimeError: If FastAPI is not installed.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the waste treatment emissions router"
        )

    router = APIRouter(
        prefix="/api/v1/waste-treatment-emissions",
        tags=["Waste Treatment Emissions"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the WasteTreatmentEmissionsService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.waste_treatment_emissions.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Waste Treatment Emissions service "
                "not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculations - Execute single calculation
    # ==================================================================

    @router.post("/calculations", status_code=201)
    async def create_calculation(
        body: SingleCalculationRequest,
    ) -> Dict[str, Any]:
        """Execute a single waste treatment emission calculation.

        Computes GHG emissions (CO2, CH4, N2O) for a waste treatment
        operation using the specified IPCC tier methodology. Supports
        biological treatment, thermal treatment, and wastewater
        treatment pathways with optional methane recovery credits.
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "tenant_id": body.tenant_id,
            "treatment_method": body.treatment_method,
            "waste_category": body.waste_category,
            "waste_tonnes": body.waste_tonnes,
            "calculation_method": body.calculation_method,
            "gwp_source": body.gwp_source,
        }

        # Optional facility reference
        if body.facility_id is not None:
            request_data["facility_id"] = body.facility_id

        # Biological treatment parameters
        if body.composting_type is not None:
            request_data["composting_type"] = body.composting_type
        if body.management_quality is not None:
            request_data["management_quality"] = body.management_quality
        if body.biofilter_efficiency is not None:
            request_data["biofilter_efficiency"] = (
                body.biofilter_efficiency
            )
        if body.volatile_solids_fraction is not None:
            request_data["volatile_solids_fraction"] = (
                body.volatile_solids_fraction
            )
        if body.bmp is not None:
            request_data["bmp"] = body.bmp
        if body.digestion_efficiency is not None:
            request_data["digestion_efficiency"] = (
                body.digestion_efficiency
            )
        if body.methane_fraction is not None:
            request_data["methane_fraction"] = body.methane_fraction

        # Thermal treatment parameters
        if body.incinerator_type is not None:
            request_data["incinerator_type"] = body.incinerator_type
        if body.oxidation_factor is not None:
            request_data["oxidation_factor"] = body.oxidation_factor
        if body.electric_efficiency is not None:
            request_data["electric_efficiency"] = (
                body.electric_efficiency
            )
        if body.thermal_efficiency is not None:
            request_data["thermal_efficiency"] = (
                body.thermal_efficiency
            )
        if body.grid_ef_electric is not None:
            request_data["grid_ef_electric"] = body.grid_ef_electric

        # Wastewater treatment parameters
        if body.tow_kg_yr is not None:
            request_data["tow_kg_yr"] = body.tow_kg_yr
        if body.bod_or_cod is not None:
            request_data["bod_or_cod"] = body.bod_or_cod
        if body.wastewater_system is not None:
            request_data["wastewater_system"] = body.wastewater_system
        if body.flow_m3_yr is not None:
            request_data["flow_m3_yr"] = body.flow_m3_yr
        if body.n_influent is not None:
            request_data["n_influent"] = body.n_influent
        if body.n_effluent is not None:
            request_data["n_effluent"] = body.n_effluent

        # Methane recovery parameters
        if body.capture_efficiency is not None:
            request_data["capture_efficiency"] = (
                body.capture_efficiency
            )
        if body.flare_fraction is not None:
            request_data["flare_fraction"] = body.flare_fraction
        if body.utilize_fraction is not None:
            request_data["utilize_fraction"] = body.utilize_fraction
        if body.flare_dre is not None:
            request_data["flare_dre"] = body.flare_dre

        # Compliance frameworks
        if body.compliance_frameworks is not None:
            request_data["compliance_frameworks"] = (
                body.compliance_frameworks
            )

        # Metadata
        if body.metadata is not None:
            request_data["metadata"] = body.metadata

        try:
            result = svc.calculate(request_data)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
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
        """Execute batch waste treatment emission calculations.

        Processes up to 10,000 calculation requests in a single batch.
        Each item in the ``calculations`` list follows the same schema
        as the single calculate endpoint. Optionally applies a shared
        ``gwp_source`` and ``tenant_id`` to all items.
        """
        svc = _get_service()

        if len(body.calculations) > 10000:
            raise HTTPException(
                status_code=400,
                detail="Batch size exceeds maximum of 10,000 "
                "calculations",
            )

        try:
            result = svc.calculate_batch(
                body.calculations,
                gwp_source=body.gwp_source,
                tenant_id=body.tenant_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_batch_calculation failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. GET /calculations/{id} - Get calculation by ID
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a waste treatment calculation result by its unique ID.

        Returns the full calculation result including emissions
        breakdown by gas (CO2, CH4, N2O), treatment method details,
        methane recovery credits, and provenance hash.
        """
        svc = _get_service()

        for calc in svc._calculations:
            if calc.get("calculation_id") == calc_id:
                return calc

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 4. GET /calculations - List calculations with filters
    # ==================================================================

    @router.get("/calculations", status_code=200)
    async def list_calculations(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        treatment_method: Optional[str] = Query(
            None,
            description="Filter by treatment method "
            "(composting, incineration, anaerobic_digestion, "
            "wastewater_treatment, etc.)",
        ),
        waste_category: Optional[str] = Query(
            None,
            description="Filter by waste category "
            "(food_waste, msw, plastics, sludge, etc.)",
        ),
        from_date: Optional[str] = Query(
            None,
            description="Filter calculations from this date "
            "(ISO-8601)",
        ),
        to_date: Optional[str] = Query(
            None,
            description="Filter calculations up to this date "
            "(ISO-8601)",
        ),
    ) -> Dict[str, Any]:
        """List waste treatment calculation results with pagination.

        Supports filtering by tenant, treatment method, waste category,
        and date range. Returns paginated results with total count.
        """
        svc = _get_service()
        all_calcs = list(svc._calculations)

        # Apply filters
        if tenant_id is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("tenant_id") == tenant_id
            ]
        if treatment_method is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("treatment_method") == treatment_method
            ]
        if waste_category is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("waste_category") == waste_category
            ]
        if from_date is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("calculated_at", "") >= from_date
            ]
        if to_date is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("calculated_at", "") <= to_date
            ]

        total = len(all_calcs)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_calcs[start:end]

        return {
            "calculations": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # 5. DELETE /calculations/{id} - Delete calculation
    # ==================================================================

    @router.delete("/calculations/{calc_id}", status_code=200)
    async def delete_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Delete a waste treatment calculation result by its unique ID.

        Permanently removes the calculation record. This operation
        cannot be undone.
        """
        svc = _get_service()

        for i, calc in enumerate(svc._calculations):
            if calc.get("calculation_id") == calc_id:
                svc._calculations.pop(i)
                svc._total_calculations = max(
                    0, svc._total_calculations - 1,
                )
                return {
                    "deleted": True,
                    "calculation_id": calc_id,
                }

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 6. POST /facilities - Register treatment facility
    # ==================================================================

    @router.post("/facilities", status_code=201)
    async def create_facility(
        body: FacilityBody,
    ) -> Dict[str, Any]:
        """Register a new waste treatment facility.

        Creates a facility record with geographic location, treatment
        technology type, capacity, and emission control details.
        Every facility is scoped to a tenant.
        """
        svc = _get_service()
        try:
            result = svc.register_facility(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_facility failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 7. GET /facilities - List facilities
    # ==================================================================

    @router.get("/facilities", status_code=200)
    async def list_facilities(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        facility_type: Optional[str] = Query(
            None,
            description="Filter by facility type "
            "(composting_plant, incinerator, "
            "anaerobic_digester, mbt_plant, etc.)",
        ),
        country_code: Optional[str] = Query(
            None,
            description="Filter by ISO 3166-1 alpha-2 country code",
        ),
    ) -> Dict[str, Any]:
        """List registered waste treatment facilities with pagination.

        Returns facilities filtered by tenant, facility type, and
        country code.
        """
        svc = _get_service()
        try:
            result = svc.list_facilities(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                facility_type=facility_type,
                country_code=country_code,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_facilities failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 8. PUT /facilities/{id} - Update facility metadata
    # ==================================================================

    @router.put("/facilities/{facility_id}", status_code=200)
    async def update_facility(
        body: FacilityUpdateBody,
        facility_id: str = Path(
            ..., description="Facility identifier",
        ),
    ) -> Dict[str, Any]:
        """Update an existing waste treatment facility's attributes.

        Only non-null fields in the request body will be updated.
        Immutable fields (tenant_id, latitude, longitude) cannot be
        changed through this endpoint.
        """
        svc = _get_service()
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }
        try:
            result = svc.update_facility(facility_id, update_data)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Facility {facility_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_facility failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 9. POST /waste-streams - Register waste stream
    # ==================================================================

    @router.post("/waste-streams", status_code=201)
    async def create_waste_stream(
        body: WasteStreamBody,
    ) -> Dict[str, Any]:
        """Register a new waste stream.

        Creates a waste stream record with category, source type,
        composition, and carbon content characteristics. Waste streams
        are linked to facilities and used for emission calculations.
        """
        svc = _get_service()
        try:
            result = svc.register_waste_stream(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_waste_stream failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 10. GET /waste-streams - List waste streams
    # ==================================================================

    @router.get("/waste-streams", status_code=200)
    async def list_waste_streams(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
        waste_category: Optional[str] = Query(
            None,
            description="Filter by waste category "
            "(food_waste, msw, plastics, sludge, etc.)",
        ),
        source_type: Optional[str] = Query(
            None,
            description="Filter by source type "
            "(residential, commercial, industrial, etc.)",
        ),
        facility_id: Optional[str] = Query(
            None,
            description="Filter by facility identifier",
        ),
    ) -> Dict[str, Any]:
        """List registered waste streams with pagination and filters.

        Returns waste streams filtered by tenant, waste category,
        source type, and linked facility.
        """
        svc = _get_service()
        try:
            result = svc.list_waste_streams(
                page=page,
                page_size=page_size,
                tenant_id=tenant_id,
                waste_category=waste_category,
                source_type=source_type,
                facility_id=facility_id,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_waste_streams failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 11. PUT /waste-streams/{id} - Update stream composition
    # ==================================================================

    @router.put("/waste-streams/{stream_id}", status_code=200)
    async def update_waste_stream(
        body: WasteStreamUpdateBody,
        stream_id: str = Path(
            ..., description="Waste stream identifier",
        ),
    ) -> Dict[str, Any]:
        """Update an existing waste stream's attributes.

        Only non-null fields in the request body will be updated.
        Allows updating composition, carbon content, and other
        characterization parameters as better data becomes available.
        """
        svc = _get_service()
        update_data = {
            k: v for k, v in body.model_dump().items()
            if v is not None
        }
        try:
            result = svc.update_waste_stream(stream_id, update_data)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Waste stream {stream_id} not found",
                )
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "update_waste_stream failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 12. POST /treatment-events - Record treatment event
    # ==================================================================

    @router.post("/treatment-events", status_code=201)
    async def create_treatment_event(
        body: TreatmentEventBody,
    ) -> Dict[str, Any]:
        """Record a waste treatment event.

        Captures an individual treatment operation at a facility,
        including the waste category, mass treated, method applied,
        energy recovered, and residue produced. Treatment events
        provide the activity data for emission calculations.
        """
        svc = _get_service()
        try:
            result = svc.record_treatment_event(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_treatment_event failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 13. GET /treatment-events - List events with filters
    # ==================================================================

    @router.get("/treatment-events", status_code=200)
    async def list_treatment_events(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        facility_id: Optional[str] = Query(
            None, description="Filter by facility identifier",
        ),
        treatment_method: Optional[str] = Query(
            None,
            description="Filter by treatment method",
        ),
        waste_category: Optional[str] = Query(
            None,
            description="Filter by waste category",
        ),
        from_date: Optional[str] = Query(
            None,
            description="Filter events from this date (ISO-8601)",
        ),
        to_date: Optional[str] = Query(
            None,
            description="Filter events up to this date (ISO-8601)",
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant identifier",
        ),
    ) -> Dict[str, Any]:
        """List treatment events with pagination and filters.

        Returns treatment events filtered by facility, treatment
        method, waste category, date range, and tenant.
        """
        svc = _get_service()
        try:
            result = svc.list_treatment_events(
                page=page,
                page_size=page_size,
                facility_id=facility_id,
                treatment_method=treatment_method,
                waste_category=waste_category,
                from_date=from_date,
                to_date=to_date,
                tenant_id=tenant_id,
            )
            return result
        except Exception as exc:
            logger.error(
                "list_treatment_events failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 14. POST /methane-recovery - Record methane recovery
    # ==================================================================

    @router.post("/methane-recovery", status_code=201)
    async def create_methane_recovery(
        body: MethaneRecoveryBody,
    ) -> Dict[str, Any]:
        """Record a methane recovery event.

        Captures methane capture, flaring, and utilization data for a
        facility. Methane recovery reduces net emissions and is
        credited in the GHG inventory. Supports recording of
        electricity and heat generation from captured biogas.
        """
        svc = _get_service()
        try:
            result = svc.record_methane_recovery(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "create_methane_recovery failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 15. GET /methane-recovery/{facility_id} - Get recovery history
    # ==================================================================

    @router.get(
        "/methane-recovery/{facility_id}", status_code=200,
    )
    async def get_methane_recovery_history(
        facility_id: str = Path(
            ..., description="Facility identifier",
        ),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20, ge=1, le=100, description="Items per page",
        ),
        from_date: Optional[str] = Query(
            None,
            description="Filter from this date (ISO-8601)",
        ),
        to_date: Optional[str] = Query(
            None,
            description="Filter up to this date (ISO-8601)",
        ),
    ) -> Dict[str, Any]:
        """Get methane recovery history for a facility.

        Returns all methane recovery events for the specified facility,
        ordered by recovery date. Includes totals for captured, flared,
        and utilized methane, as well as energy generation summaries.
        """
        svc = _get_service()
        try:
            all_records = [
                r for r in svc._methane_recovery_records
                if r.get("facility_id") == facility_id
            ]

            # Apply date filters
            if from_date is not None:
                all_records = [
                    r for r in all_records
                    if r.get("recovery_date", "") >= from_date
                ]
            if to_date is not None:
                all_records = [
                    r for r in all_records
                    if r.get("recovery_date", "") <= to_date
                ]

            total = len(all_records)
            start = (page - 1) * page_size
            end = start + page_size
            page_data = all_records[start:end]

            # Compute summary totals
            total_captured = sum(
                float(r.get("methane_captured_tonnes", 0))
                for r in all_records
            )
            total_flared = sum(
                float(r.get("methane_flared_tonnes", 0))
                for r in all_records
            )
            total_utilized = sum(
                float(r.get("methane_utilized_tonnes", 0))
                for r in all_records
            )
            total_electricity = sum(
                float(r.get("electricity_generated_mwh", 0))
                for r in all_records
            )
            total_heat = sum(
                float(r.get("heat_generated_mwh", 0))
                for r in all_records
            )

            return {
                "facility_id": facility_id,
                "records": page_data,
                "total": total,
                "page": page,
                "page_size": page_size,
                "summary": {
                    "total_captured_tonnes": total_captured,
                    "total_flared_tonnes": total_flared,
                    "total_utilized_tonnes": total_utilized,
                    "total_electricity_mwh": total_electricity,
                    "total_heat_mwh": total_heat,
                },
            }
        except Exception as exc:
            logger.error(
                "get_methane_recovery_history failed: %s", exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 16. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckBody,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check against multiple frameworks.

        Evaluates the waste treatment calculation against applicable
        frameworks: GHG Protocol Corporate Standard, IPCC 2006
        Guidelines (Vol 5, Ch 4-6), CSRD/ESRS E1, EU ETS MRV
        Regulation, US EPA Subpart HH/TT, UK SECR, and UNFCCC
        reporting requirements.
        """
        svc = _get_service()
        try:
            result = svc.check_compliance(body.model_dump())
            return result
        except Exception as exc:
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 17. GET /compliance/{id} - Get compliance result
    # ==================================================================

    @router.get("/compliance/{compliance_id}", status_code=200)
    async def get_compliance_result(
        compliance_id: str = Path(
            ..., description="Compliance check identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a compliance check result by its unique identifier.

        Returns the full compliance assessment including per-framework
        status, rule violations, recommendations, and overall
        compliance score.
        """
        svc = _get_service()

        for result in svc._compliance_results:
            if result.get("id") == compliance_id:
                return result

        raise HTTPException(
            status_code=404,
            detail=f"Compliance result {compliance_id} not found",
        )

    # ==================================================================
    # 18. POST /uncertainty - Run Monte Carlo analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyBody,
    ) -> Dict[str, Any]:
        """Run Monte Carlo uncertainty analysis on a calculation.

        Requires a previous ``calculation_id``. Performs Monte Carlo
        simulation by sampling emission factor distributions, activity
        data uncertainty, and parameter uncertainty. Returns
        statistical characterization including mean, standard
        deviation, confidence intervals, percentiles, and coefficient
        of variation.
        """
        svc = _get_service()
        try:
            result = svc.run_uncertainty(body.model_dump())
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error(
                "run_uncertainty failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 19. GET /aggregations - Get aggregated emissions
    # ==================================================================

    @router.get("/aggregations", status_code=200)
    async def get_aggregations(
        tenant_id: str = Query(
            ..., min_length=1,
            description="Tenant identifier for scoping",
        ),
        period: Optional[str] = Query(
            None,
            description="Reporting period "
            "(monthly, quarterly, annual, custom)",
        ),
        group_by: Optional[str] = Query(
            None,
            description="Comma-separated fields to group by "
            "(treatment_method, waste_category, facility_id)",
        ),
        date_from: Optional[str] = Query(
            None, description="Start date (ISO-8601)",
        ),
        date_to: Optional[str] = Query(
            None, description="End date (ISO-8601)",
        ),
        treatment_methods: Optional[str] = Query(
            None,
            description="Comma-separated treatment method filter",
        ),
        waste_categories: Optional[str] = Query(
            None,
            description="Comma-separated waste category filter",
        ),
        facility_ids: Optional[str] = Query(
            None,
            description="Comma-separated facility ID filter",
        ),
    ) -> Dict[str, Any]:
        """Get aggregated waste treatment emissions.

        Aggregates calculation results by specified grouping fields
        and reporting period. Supports filtering by tenant, treatment
        method, waste category, facility, and date range. Returns
        totals for CO2, CH4, N2O, and total CO2e, along with methane
        recovery credits.
        """
        svc = _get_service()
        try:
            agg_data: Dict[str, Any] = {
                "tenant_id": tenant_id,
            }
            if period is not None:
                agg_data["period"] = period
            if group_by is not None:
                agg_data["group_by"] = [
                    g.strip() for g in group_by.split(",")
                    if g.strip()
                ]
            if date_from is not None:
                agg_data["date_from"] = date_from
            if date_to is not None:
                agg_data["date_to"] = date_to
            if treatment_methods is not None:
                agg_data["treatment_methods"] = [
                    m.strip()
                    for m in treatment_methods.split(",")
                    if m.strip()
                ]
            if waste_categories is not None:
                agg_data["waste_categories"] = [
                    c.strip()
                    for c in waste_categories.split(",")
                    if c.strip()
                ]
            if facility_ids is not None:
                agg_data["facility_ids"] = [
                    f.strip()
                    for f in facility_ids.split(",")
                    if f.strip()
                ]

            result = svc.aggregate(agg_data)
            return result
        except Exception as exc:
            logger.error(
                "get_aggregations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 20. GET /health - Health check
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Health check for the Waste Treatment Emissions service.

        Returns service status, version, uptime, and summary
        statistics about registered facilities, waste streams,
        treatment events, and calculations.
        """
        try:
            svc = _get_service()
            now = datetime.now(timezone.utc).isoformat()

            return {
                "status": "healthy",
                "service": "waste-treatment-emissions",
                "version": "1.0.0",
                "timestamp": now,
                "stats": {
                    "total_calculations": getattr(
                        svc, "_total_calculations", 0,
                    ),
                    "total_facilities": len(
                        getattr(svc, "_facilities", []),
                    ),
                    "total_waste_streams": len(
                        getattr(svc, "_waste_streams", []),
                    ),
                    "total_treatment_events": len(
                        getattr(svc, "_treatment_events", []),
                    ),
                    "total_methane_recovery_records": len(
                        getattr(
                            svc, "_methane_recovery_records", [],
                        ),
                    ),
                    "total_compliance_checks": len(
                        getattr(svc, "_compliance_results", []),
                    ),
                },
            }
        except HTTPException:
            # Service not initialized but endpoint is reachable
            now = datetime.now(timezone.utc).isoformat()
            return {
                "status": "degraded",
                "service": "waste-treatment-emissions",
                "version": "1.0.0",
                "timestamp": now,
                "detail": "Service not yet initialized",
                "stats": {},
            }
        except Exception as exc:
            logger.error(
                "health_check failed: %s", exc, exc_info=True,
            )
            now = datetime.now(timezone.utc).isoformat()
            return {
                "status": "unhealthy",
                "service": "waste-treatment-emissions",
                "version": "1.0.0",
                "timestamp": now,
                "detail": str(exc),
                "stats": {},
            }

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router"]
