# -*- coding: utf-8 -*-
"""
Scope 2 Steam/Heat Purchase REST API Router - AGENT-MRV-011
=============================================================

20 REST endpoints for the Scope 2 Steam/Heat Purchase Agent
(GL-MRV-X-022).

Prefix: ``/api/v1/steam-heat-purchase``

Endpoints:
     1. POST   /calculate/steam              - Calculate steam emissions
     2. POST   /calculate/heating             - Calculate district heating emissions
     3. POST   /calculate/cooling             - Calculate district cooling emissions
     4. POST   /calculate/chp                 - Calculate CHP-allocated emissions
     5. POST   /calculate/batch               - Batch calculation
     6. GET    /factors/fuels                 - List all fuel emission factors
     7. GET    /factors/fuels/{fuel_type}     - Get specific fuel emission factor
     8. GET    /factors/heating/{region}      - Get district heating network factor
     9. GET    /factors/cooling/{technology}  - Get cooling system COP
    10. GET    /factors/chp-defaults          - Get CHP default parameters
    11. POST   /facilities                    - Register a facility
    12. GET    /facilities/{facility_id}      - Get facility
    13. POST   /suppliers                     - Register steam/heat supplier
    14. GET    /suppliers/{supplier_id}       - Get supplier
    15. POST   /uncertainty                   - Run uncertainty analysis
    16. POST   /compliance/check              - Run compliance check
    17. GET    /compliance/frameworks         - List available frameworks
    18. POST   /aggregate                     - Aggregate calculation results
    19. GET    /calculations/{calc_id}        - Get calculation result
    20. GET    /health                        - Service health check

GHG Protocol Scope 2 Guidance (2015) implementation for purchased steam,
district heating, and district cooling emission calculations with four
calculation methodologies:

    1. **Direct Emission Factor** - Applies a composite kgCO2e/GJ factor
       directly to metered consumption. Simplest approach for supplier-
       disclosed emission factors.
    2. **Fuel-Based** - Calculates per-gas emissions (CO2, CH4, N2O) from
       the fuel type, fuel quantity, and boiler efficiency used to generate
       steam. Supports 14 fuel types including biomass with separate
       biogenic CO2 reporting.
    3. **COP-Based** - Converts cooling output to energy input using the
       coefficient of performance of the cooling technology (9 technologies
       from centrifugal chillers to free cooling).
    4. **CHP Allocation** - Allocates total CHP/cogeneration fuel emissions
       between electrical and thermal outputs using efficiency, energy, or
       exergy methods per GHG Protocol guidance.

Supported Regulatory Frameworks (7):
    - GHG Protocol Scope 2 Guidance (2015)
    - IPCC 2006 Guidelines for National GHG Inventories
    - ISO 14064-1:2018
    - CSRD / ESRS E1
    - US EPA Greenhouse Gas Reporting Program
    - UK DEFRA Reporting (SECR)
    - CDP Climate Change

Emission Factor Data:
    - 14 fuel types with per-gas emission factors (IPCC 2006 Vol 2)
    - 13 regional district heating factors (kgCO2e/GJ delivered heat)
    - 9 cooling system technologies with COP ranges
    - 5 CHP fuel types with electrical, thermal, and overall efficiencies

Uncertainty Quantification:
    - Monte Carlo simulation (configurable 100 to 1,000,000 iterations)
    - Analytical error propagation (IPCC Approach 1)
    - Per-parameter uncertainty: activity data, emission factor, efficiency

All emission factors and intermediate values use Python ``Decimal`` for
zero-hallucination deterministic arithmetic. No LLM involvement in any
calculation path. Every result carries a SHA-256 provenance hash for
full audit trails.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
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
    logger.debug("FastAPI not installed; steam-heat-purchase router unavailable")


# ---------------------------------------------------------------------------
# Optional model imports from AGENT-MRV-011 models module
# ---------------------------------------------------------------------------

try:
    from greenlang.steam_heat_purchase.models import (
        FUEL_EMISSION_FACTORS,
        DISTRICT_HEATING_FACTORS,
        COOLING_SYSTEM_FACTORS,
        COOLING_ENERGY_SOURCE,
        CHP_DEFAULT_EFFICIENCIES,
        UNIT_CONVERSIONS,
        GWP_VALUES,
        VERSION,
        FuelType,
        CoolingTechnology,
        EnergyType,
        NetworkType,
        FacilityType,
        GWPSource,
        CHPAllocMethod,
        DataQualityTier,
        SteamPressure,
        SteamQuality,
        ReportingPeriod,
        AggregationType,
        ComplianceStatus,
        FuelEmissionFactor,
        DistrictHeatingFactor,
        CoolingSystemFactor,
        CHPParameters,
        FacilityInfo,
        SteamSupplier,
        SteamCalculationRequest,
        HeatingCalculationRequest,
        CoolingCalculationRequest,
        CHPAllocationRequest,
        GasEmissionDetail,
        CalculationResult,
        CHPAllocationResult,
        BatchCalculationRequest,
        BatchCalculationResult,
        UncertaintyRequest,
        UncertaintyResult,
        ComplianceCheckResult,
        AggregationRequest,
        AggregationResult,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.debug(
        "steam_heat_purchase.models not available; "
        "router will use inline request/response models"
    )
    # Provide stub constants for the router to compile
    FUEL_EMISSION_FACTORS = {}
    DISTRICT_HEATING_FACTORS = {}
    COOLING_SYSTEM_FACTORS = {}
    COOLING_ENERGY_SOURCE = {}
    CHP_DEFAULT_EFFICIENCIES = {}
    UNIT_CONVERSIONS = {}
    GWP_VALUES = {}
    VERSION = "1.0.0"


# ===================================================================
# Request body models (Pydantic) - inline for when models.py is not
# on the import path or to provide simpler API-facing schemas
# ===================================================================

if FASTAPI_AVAILABLE:

    # ---------------------------------------------------------------
    # Steam calculation request
    # ---------------------------------------------------------------

    class SteamCalcBody(BaseModel):
        """Request body for a purchased steam emission calculation.

        Accepts steam consumption in GJ at the building meter, fuel type,
        boiler efficiency, supplier reference, and optional condensate
        return percentage. Dispatches to the fuel-based or direct EF
        calculation method depending on available data.

        Formula (fuel-based):
            fuel_input_gj = consumption_gj / boiler_efficiency
            co2_kg = fuel_input_gj * co2_ef_per_gj
            ch4_kg = fuel_input_gj * ch4_ef_per_gj
            n2o_kg = fuel_input_gj * n2o_ef_per_gj
            co2e_kg = co2_kg + (ch4_kg * gwp_ch4) + (n2o_kg * gwp_n2o)
        """

        facility_id: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Reference to the consuming facility",
        )
        consumption_gj: float = Field(
            ..., gt=0,
            description="Steam consumption in GJ at the building meter",
        )
        fuel_type: Optional[str] = Field(
            default=None,
            description="Fuel type: natural_gas, fuel_oil_2, fuel_oil_6, "
            "coal_bituminous, coal_subbituminous, coal_lignite, lpg, "
            "biomass_wood, biomass_biogas, municipal_waste, waste_heat, "
            "geothermal, solar_thermal, electric",
        )
        boiler_efficiency: Optional[float] = Field(
            default=None, gt=0, le=1,
            description="Boiler thermal efficiency override (0-1). "
            "If omitted, default efficiency for the fuel type is used",
        )
        supplier_id: Optional[str] = Field(
            default=None,
            max_length=200,
            description="Reference to the steam supplier for supplier-"
            "specific emission factor or fuel mix lookup",
        )
        steam_pressure: Optional[str] = Field(
            default=None,
            description="Steam pressure classification: low, medium, "
            "high, or very_high",
        )
        steam_quality: Optional[str] = Field(
            default=None,
            description="Steam quality classification: saturated, "
            "superheated, or wet",
        )
        condensate_return_pct: float = Field(
            default=0.0, ge=0, le=100,
            description="Condensate return percentage (0-100). Reduces "
            "effective steam consumption by recovered energy",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source: AR4, AR5, AR6, or AR6_20YR "
            "(default: AR6)",
        )
        data_quality_tier: Optional[str] = Field(
            default=None,
            description="Data quality tier: tier_1, tier_2, or tier_3 "
            "(default: tier_1)",
        )
        reporting_period: Optional[str] = Field(
            default=None,
            description="Reporting period: monthly, quarterly, or annual "
            "(default: annual)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )

    # ---------------------------------------------------------------
    # Heating calculation request
    # ---------------------------------------------------------------

    class HeatingCalcBody(BaseModel):
        """Request body for a district heating emission calculation.

        Accepts heating consumption in GJ, geographic region for emission
        factor lookup, and optional network type and supplier-specific
        emission factor overrides.

        Formula:
            adjusted_consumption = consumption_gj * (1 + distribution_loss_pct)
            co2e_kg = adjusted_consumption * ef_kgco2e_per_gj
        """

        facility_id: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Reference to the consuming facility",
        )
        consumption_gj: float = Field(
            ..., gt=0,
            description="District heating consumption in GJ at the "
            "building meter",
        )
        region: str = Field(
            ...,
            min_length=1,
            max_length=100,
            description="Geographic region for factor lookup: denmark, "
            "sweden, finland, germany, poland, netherlands, france, "
            "uk, us, china, japan, south_korea, or global_default",
        )
        network_type: Optional[str] = Field(
            default=None,
            description="Network type: municipal, industrial, campus, "
            "or mixed (default: municipal)",
        )
        supplier_ef_kgco2e_per_gj: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier-specific emission factor override "
            "in kgCO2e per GJ. Overrides regional default",
        )
        distribution_loss_pct: Optional[float] = Field(
            default=None, ge=0, le=1,
            description="Distribution loss fraction override (0-1). "
            "Overrides regional default",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source: AR4, AR5, AR6, or AR6_20YR "
            "(default: AR6)",
        )
        data_quality_tier: Optional[str] = Field(
            default=None,
            description="Data quality tier: tier_1, tier_2, or tier_3",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )

    # ---------------------------------------------------------------
    # Cooling calculation request
    # ---------------------------------------------------------------

    class CoolingCalcBody(BaseModel):
        """Request body for a district cooling emission calculation.

        Accepts cooling output in GJ, cooling technology type, and
        optional measured COP. Converts cooling output to energy input
        using the COP, then applies the appropriate emission factor.

        Formula:
            energy_input_gj = cooling_output_gj / cop
            co2e_kg = energy_input_gj * energy_source_ef
        """

        facility_id: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Reference to the consuming facility",
        )
        cooling_output_gj: float = Field(
            ..., gt=0,
            description="Cooling output delivered to the facility in GJ",
        )
        technology: str = Field(
            ...,
            description="Cooling technology: centrifugal_chiller, "
            "screw_chiller, reciprocating_chiller, absorption_single, "
            "absorption_double, absorption_triple, free_cooling, "
            "ice_storage, or thermal_storage",
        )
        cop: Optional[float] = Field(
            default=None, gt=0,
            description="Measured COP override. If omitted, default "
            "COP for the technology is used",
        )
        grid_ef_kgco2e_per_kwh: Optional[float] = Field(
            default=None, ge=0,
            description="Grid electricity emission factor in kgCO2e per kWh. "
            "Required for electric chillers when not using default",
        )
        heat_source_ef_kgco2e_per_gj: Optional[float] = Field(
            default=None, ge=0,
            description="Heat source emission factor in kgCO2e per GJ. "
            "Required for absorption chillers",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source: AR4, AR5, AR6, or AR6_20YR "
            "(default: AR6)",
        )
        data_quality_tier: Optional[str] = Field(
            default=None,
            description="Data quality tier: tier_1, tier_2, or tier_3",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )

    # ---------------------------------------------------------------
    # CHP allocation request
    # ---------------------------------------------------------------

    class CHPAllocBody(BaseModel):
        """Request body for CHP emission allocation calculation.

        Allocates total CHP plant fuel emissions between electrical and
        thermal outputs using one of three GHG Protocol methods.

        Efficiency-based formula:
            heat_share = (Q_heat / eta_thermal) /
                ((Q_heat / eta_thermal) + (Q_elec / eta_elec))
            heat_emissions = total_fuel_emissions * heat_share

        Energy-based formula:
            heat_share = Q_heat / (Q_heat + Q_elec)

        Exergy-based formula:
            carnot = 1 - (T_ambient_K / T_steam_K)
            heat_share = (Q_heat * carnot) / ((Q_heat * carnot) + Q_elec)
        """

        facility_id: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Reference to the consuming facility receiving "
            "the thermal output",
        )
        total_fuel_gj: float = Field(
            ..., gt=0,
            description="Total fuel input to the CHP plant in GJ for "
            "the reporting period",
        )
        fuel_type: str = Field(
            ...,
            description="Primary fuel type used by the CHP plant: "
            "natural_gas, fuel_oil_2, fuel_oil_6, coal_bituminous, "
            "coal_subbituminous, coal_lignite, lpg, biomass_wood, "
            "biomass_biogas, municipal_waste, waste_heat, geothermal, "
            "solar_thermal, or electric",
        )
        heat_output_gj: float = Field(
            ..., gt=0,
            description="Useful thermal (heat) output in GJ",
        )
        power_output_gj: float = Field(
            ..., gt=0,
            description="Electrical power output in GJ",
        )
        cooling_output_gj: float = Field(
            default=0.0, ge=0,
            description="Cooling output from absorption chiller driven "
            "by CHP heat, in GJ (0 if no cooling)",
        )
        method: Optional[str] = Field(
            default=None,
            description="Allocation method: efficiency, energy, or exergy "
            "(default: efficiency)",
        )
        electrical_efficiency: Optional[float] = Field(
            default=None, gt=0, lt=1,
            description="Electrical efficiency override (0-1). "
            "If omitted, CHP default for fuel type is used",
        )
        thermal_efficiency: Optional[float] = Field(
            default=None, gt=0, lt=1,
            description="Thermal efficiency override (0-1). "
            "If omitted, CHP default for fuel type is used",
        )
        steam_temperature_c: Optional[float] = Field(
            default=None,
            description="Steam supply temperature in degrees Celsius. "
            "Required for exergy-based allocation (Carnot factor)",
        )
        ambient_temperature_c: float = Field(
            default=25.0,
            description="Ambient reference temperature in degrees Celsius "
            "(default: 25 C / 298.15 K)",
        )
        gwp_source: Optional[str] = Field(
            default=None,
            description="GWP source: AR4, AR5, AR6, or AR6_20YR "
            "(default: AR6)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )

    # ---------------------------------------------------------------
    # Batch calculation request
    # ---------------------------------------------------------------

    class BatchCalcBody(BaseModel):
        """Request body for batch thermal energy emission calculations.

        Processes multiple calculation requests (steam, heating, cooling,
        or CHP) in a single batch. Returns portfolio-level totals
        alongside individual results.
        """

        batch_id: Optional[str] = Field(
            default=None,
            description="Optional batch identifier (auto-generated "
            "UUID if omitted)",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )
        requests: List[Dict[str, Any]] = Field(
            ..., min_length=1,
            description="List of calculation request dictionaries. Each "
            "must contain 'energy_type' (steam, district_heating, "
            "district_cooling, or chp) and the corresponding fields "
            "from the individual calculation request schemas",
        )

    # ---------------------------------------------------------------
    # Facility request model
    # ---------------------------------------------------------------

    class FacilityBody(BaseModel):
        """Request body for registering a new facility.

        Creates a facility record with geographic, supplier, and
        network connection metadata for thermal energy tracking.
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
            description="Facility classification: industrial, commercial, "
            "institutional, residential, data_center, or campus",
        )
        country: str = Field(
            ..., min_length=2, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        region: str = Field(
            ..., min_length=1, max_length=100,
            description="Geographic region for heating factor lookup "
            "(e.g. germany, sweden, global_default)",
        )
        latitude: Optional[float] = Field(
            default=None, ge=-90, le=90,
            description="WGS84 latitude in decimal degrees",
        )
        longitude: Optional[float] = Field(
            default=None, ge=-180, le=180,
            description="WGS84 longitude in decimal degrees",
        )
        steam_suppliers: Optional[List[str]] = Field(
            default=None,
            description="List of steam supplier IDs connected "
            "to this facility",
        )
        heating_network: Optional[str] = Field(
            default=None,
            max_length=200,
            description="District heating network identifier",
        )
        cooling_system: Optional[str] = Field(
            default=None,
            max_length=200,
            description="District cooling system identifier",
        )
        tenant_id: str = Field(
            ..., min_length=1,
            description="Owning tenant identifier for multi-tenancy",
        )

    # ---------------------------------------------------------------
    # Supplier request model
    # ---------------------------------------------------------------

    class SupplierBody(BaseModel):
        """Request body for registering a steam/heat supplier.

        Creates a supplier profile with fuel mix, boiler efficiency,
        and optionally a verified composite emission factor.
        """

        supplier_id: Optional[str] = Field(
            default=None,
            description="Optional supplier identifier "
            "(auto-generated UUID if omitted)",
        )
        name: str = Field(
            ..., min_length=1, max_length=500,
            description="Human-readable supplier name",
        )
        fuel_mix: Optional[Dict[str, float]] = Field(
            default=None,
            description="Fuel type proportions in supplier generation "
            "mix (keys: fuel type, values: fraction 0-1, should sum to 1)",
        )
        boiler_efficiency: Optional[float] = Field(
            default=None, gt=0, le=1,
            description="Overall boiler thermal efficiency (0-1)",
        )
        supplier_ef_kgco2e_per_gj: Optional[float] = Field(
            default=None, ge=0,
            description="Supplier composite emission factor in kgCO2e/GJ",
        )
        country: str = Field(
            ..., min_length=2, max_length=2,
            description="ISO 3166-1 alpha-2 country code",
        )
        region: Optional[str] = Field(
            default=None,
            max_length=100,
            description="Geographic region for factor lookup",
        )
        verified: bool = Field(
            default=False,
            description="Whether independently verified by a third party",
        )
        data_quality_tier: Optional[str] = Field(
            default=None,
            description="Data quality tier: tier_1, tier_2, or tier_3",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )

    # ---------------------------------------------------------------
    # Uncertainty request model
    # ---------------------------------------------------------------

    class UncertaintyBody(BaseModel):
        """Request body for uncertainty analysis on a calculation result.

        Runs Monte Carlo simulation or analytical error propagation
        on a previously computed thermal energy calculation to quantify
        the uncertainty range of the emission estimate.
        """

        calc_id: str = Field(
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
            description="Confidence level percentage for the "
            "uncertainty interval (e.g. 95 for 95%% CI)",
        )
        activity_data_uncertainty_pct: float = Field(
            default=5.0, ge=0, le=100,
            description="Activity data uncertainty percentage (default: 5%%)",
        )
        emission_factor_uncertainty_pct: float = Field(
            default=10.0, ge=0, le=100,
            description="Emission factor uncertainty percentage "
            "(default: 10%%)",
        )
        efficiency_uncertainty_pct: float = Field(
            default=5.0, ge=0, le=100,
            description="Boiler/CHP efficiency uncertainty percentage "
            "(default: 5%%)",
        )
        seed: int = Field(
            default=42, ge=0,
            description="Random seed for reproducibility",
        )

    # ---------------------------------------------------------------
    # Compliance check request model
    # ---------------------------------------------------------------

    class ComplianceCheckBody(BaseModel):
        """Request body for a regulatory compliance check.

        Evaluates a completed thermal energy calculation against one or
        more regulatory frameworks.
        """

        calc_result: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Inline calculation result dict. "
            "Mutually exclusive with calc_id",
        )
        calc_id: Optional[str] = Field(
            default=None,
            description="ID of a previous calculation to check. "
            "Mutually exclusive with calc_result",
        )
        frameworks: Optional[List[str]] = Field(
            default=None,
            description="Regulatory frameworks to check. Empty means all. "
            "Options: ghg_protocol_scope2, ipcc_2006, iso_14064, "
            "csrd_e1, epa_ghgrp, defra_secr, cdp",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Tenant identifier for scoping",
        )

    # ---------------------------------------------------------------
    # Aggregation request model
    # ---------------------------------------------------------------

    class AggregationBody(BaseModel):
        """Request body for aggregating multiple calculation results.

        Aggregates specified calculations by facility, fuel type,
        energy type, supplier, or time period.
        """

        calc_ids: List[str] = Field(
            ..., min_length=1,
            description="List of calculation result IDs to aggregate",
        )
        aggregation_type: str = Field(
            ...,
            description="Aggregation dimension: by_facility, by_fuel, "
            "by_energy_type, by_supplier, or by_period",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Owning tenant identifier for multi-tenancy",
        )


# ===================================================================
# Router factory
# ===================================================================


def create_router() -> "APIRouter":
    """Create and return the Steam/Heat Purchase FastAPI APIRouter.

    Returns:
        Configured APIRouter with 20 endpoints covering steam, heating,
        cooling, and CHP emission calculations; emission factor lookups;
        facility and supplier registration; uncertainty analysis;
        compliance checking; aggregation; and health status.

    Raises:
        RuntimeError: If FastAPI is not installed in the environment.
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is required for the steam-heat-purchase router"
        )

    router = APIRouter(
        prefix="/api/v1/steam-heat-purchase",
        tags=["steam-heat-purchase"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------

    def _get_service():
        """Get the SteamHeatPipelineEngine singleton.

        Returns the initialized pipeline engine. Raises HTTPException 503
        if the service has not been initialized.
        """
        try:
            from greenlang.steam_heat_purchase.steam_heat_pipeline import (
                SteamHeatPipelineEngine,
            )
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Steam/Heat Purchase pipeline engine not available",
            )

        # Try to get from setup module if it exists
        try:
            from greenlang.steam_heat_purchase.setup import get_service
            svc = get_service()
            if svc is not None:
                return svc
        except (ImportError, AttributeError):
            pass

        # Fallback: create a default pipeline engine instance
        if not hasattr(_get_service, "_instance"):
            _get_service._instance = SteamHeatPipelineEngine()
        return _get_service._instance

    def _get_db_engine():
        """Get the SteamHeatDatabaseEngine.

        Returns the database engine from the pipeline, or creates a
        standalone instance. Raises HTTPException 503 if unavailable.
        """
        try:
            svc = _get_service()
            if hasattr(svc, "_db_engine") and svc._db_engine is not None:
                return svc._db_engine
        except HTTPException:
            pass

        try:
            from greenlang.steam_heat_purchase.steam_heat_database import (
                SteamHeatDatabaseEngine,
            )
            if not hasattr(_get_db_engine, "_instance"):
                _get_db_engine._instance = SteamHeatDatabaseEngine()
            return _get_db_engine._instance
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Steam/Heat database engine not available",
            )

    def _get_uncertainty_engine():
        """Get the UncertaintyQuantifierEngine.

        Returns the uncertainty engine from the pipeline, or creates a
        standalone instance.
        """
        try:
            svc = _get_service()
            if (
                hasattr(svc, "_uncertainty_engine")
                and svc._uncertainty_engine is not None
            ):
                return svc._uncertainty_engine
        except HTTPException:
            pass

        try:
            from greenlang.steam_heat_purchase.uncertainty_quantifier import (
                UncertaintyQuantifierEngine,
            )
            if not hasattr(_get_uncertainty_engine, "_instance"):
                _get_uncertainty_engine._instance = (
                    UncertaintyQuantifierEngine()
                )
            return _get_uncertainty_engine._instance
        except ImportError:
            return None

    def _get_compliance_engine():
        """Get the ComplianceCheckerEngine.

        Returns the compliance engine from the pipeline, or creates a
        standalone instance.
        """
        try:
            svc = _get_service()
            if (
                hasattr(svc, "_compliance_engine")
                and svc._compliance_engine is not None
            ):
                return svc._compliance_engine
        except HTTPException:
            pass

        try:
            from greenlang.steam_heat_purchase.compliance_checker import (
                ComplianceCheckerEngine,
            )
            if not hasattr(_get_compliance_engine, "_instance"):
                _get_compliance_engine._instance = ComplianceCheckerEngine()
            return _get_compliance_engine._instance
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # In-memory stores for API-level state
    # ------------------------------------------------------------------

    _calculations: List[Dict[str, Any]] = []
    _facilities: List[Dict[str, Any]] = []
    _suppliers: List[Dict[str, Any]] = []
    _compliance_results: List[Dict[str, Any]] = []
    _uncertainty_results: List[Dict[str, Any]] = []

    # ==================================================================
    # 1. POST /calculate/steam - Calculate steam emissions
    # ==================================================================

    @router.post("/calculate/steam", status_code=201)
    async def calculate_steam(
        body: SteamCalcBody,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from purchased steam consumption.

        Computes per-gas (CO2, CH4, N2O) and total CO2e emissions for
        purchased steam using the fuel-based calculation method. Applies
        the fuel emission factor divided by boiler efficiency to the
        metered consumption in GJ, adjusted for condensate return.

        Formula (fuel-based):
            effective_consumption = consumption_gj * (1 - condensate_return_pct/100)
            fuel_input_gj = effective_consumption / boiler_efficiency
            emissions_per_gas = fuel_input_gj * gas_ef_per_gj
            co2e = sum(emissions_per_gas * gwp)

        Supports 14 fuel types with IPCC 2006 emission factors.
        Biogenic CO2 from biomass fuels reported separately.

        Permission: steam-heat-purchase:calculate
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "energy_type": "steam",
            "facility_id": body.facility_id,
            "consumption_gj": body.consumption_gj,
            "condensate_return_pct": body.condensate_return_pct,
            "gwp_source": (body.gwp_source or "AR6").upper(),
        }

        if body.fuel_type is not None:
            request_data["fuel_type"] = body.fuel_type.lower()
        if body.boiler_efficiency is not None:
            request_data["boiler_efficiency"] = body.boiler_efficiency
        if body.supplier_id is not None:
            request_data["supplier_id"] = body.supplier_id
        if body.steam_pressure is not None:
            request_data["steam_pressure"] = body.steam_pressure.lower()
        if body.steam_quality is not None:
            request_data["steam_quality"] = body.steam_quality.lower()
        if body.data_quality_tier is not None:
            request_data["data_quality_tier"] = body.data_quality_tier
        if body.reporting_period is not None:
            request_data["reporting_period"] = body.reporting_period
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.run_pipeline(request_data)

            _calculations.append(result)

            logger.info(
                "Steam calculation completed: calc_id=%s facility=%s "
                "co2e=%.6f kg",
                result.get("calc_id", ""),
                result.get("facility_id", ""),
                float(result.get("total_co2e_kg", 0)),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_steam failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 2. POST /calculate/heating - Calculate district heating emissions
    # ==================================================================

    @router.post("/calculate/heating", status_code=201)
    async def calculate_heating(
        body: HeatingCalcBody,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from district heating consumption.

        Computes CO2e emissions for district heating using regional
        emission factors (kgCO2e per GJ) adjusted for distribution
        network losses. Supports 13 regional factors covering
        Scandinavia, Western Europe, Eastern Europe, Americas, and Asia.

        Formula:
            adjusted_gj = consumption_gj * (1 + distribution_loss_pct)
            co2e_kg = adjusted_gj * ef_kgco2e_per_gj

        Supplier-specific emission factors take precedence over regional
        defaults when provided (Tier 2 data quality).

        Permission: steam-heat-purchase:calculate
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "energy_type": "district_heating",
            "facility_id": body.facility_id,
            "consumption_gj": body.consumption_gj,
            "region": body.region.strip().lower(),
            "gwp_source": (body.gwp_source or "AR6").upper(),
        }

        if body.network_type is not None:
            request_data["network_type"] = body.network_type.lower()
        if body.supplier_ef_kgco2e_per_gj is not None:
            request_data["supplier_ef_kgco2e_per_gj"] = (
                body.supplier_ef_kgco2e_per_gj
            )
        if body.distribution_loss_pct is not None:
            request_data["distribution_loss_pct"] = (
                body.distribution_loss_pct
            )
        if body.data_quality_tier is not None:
            request_data["data_quality_tier"] = body.data_quality_tier
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.run_pipeline(request_data)

            _calculations.append(result)

            logger.info(
                "Heating calculation completed: calc_id=%s facility=%s "
                "region=%s co2e=%.6f kg",
                result.get("calc_id", ""),
                result.get("facility_id", ""),
                body.region,
                float(result.get("total_co2e_kg", 0)),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_heating failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 3. POST /calculate/cooling - Calculate district cooling emissions
    # ==================================================================

    @router.post("/calculate/cooling", status_code=201)
    async def calculate_cooling(
        body: CoolingCalcBody,
    ) -> Dict[str, Any]:
        """Calculate Scope 2 emissions from district cooling consumption.

        Computes CO2e emissions for district cooling using the COP-based
        method. Converts cooling output (GJ) to energy input using the
        coefficient of performance for the cooling technology, then
        applies the appropriate emission factor (grid electricity EF
        for electric chillers, heat source EF for absorption chillers).

        Formula:
            energy_input_gj = cooling_output_gj / cop
            co2e_kg = energy_input_gj * energy_source_ef_per_gj

        Supports 9 cooling technologies:
        - Electric: centrifugal, screw, reciprocating chillers
        - Absorption: single, double, triple-effect
        - Storage: ice storage, thermal storage
        - Natural: free cooling

        Permission: steam-heat-purchase:calculate
        """
        svc = _get_service()

        # Validate technology
        valid_technologies = {
            "centrifugal_chiller", "screw_chiller",
            "reciprocating_chiller", "absorption_single",
            "absorption_double", "absorption_triple",
            "free_cooling", "ice_storage", "thermal_storage",
        }
        tech = body.technology.lower()
        if tech not in valid_technologies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid technology '{body.technology}'. "
                f"Must be one of: {sorted(valid_technologies)}",
            )

        request_data: Dict[str, Any] = {
            "energy_type": "district_cooling",
            "facility_id": body.facility_id,
            "cooling_output_gj": body.cooling_output_gj,
            "technology": tech,
            "gwp_source": (body.gwp_source or "AR6").upper(),
        }

        if body.cop is not None:
            request_data["cop"] = body.cop
        if body.grid_ef_kgco2e_per_kwh is not None:
            request_data["grid_ef_kgco2e_per_kwh"] = (
                body.grid_ef_kgco2e_per_kwh
            )
        if body.heat_source_ef_kgco2e_per_gj is not None:
            request_data["heat_source_ef_kgco2e_per_gj"] = (
                body.heat_source_ef_kgco2e_per_gj
            )
        if body.data_quality_tier is not None:
            request_data["data_quality_tier"] = body.data_quality_tier
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.run_pipeline(request_data)

            _calculations.append(result)

            logger.info(
                "Cooling calculation completed: calc_id=%s facility=%s "
                "technology=%s co2e=%.6f kg",
                result.get("calc_id", ""),
                result.get("facility_id", ""),
                tech,
                float(result.get("total_co2e_kg", 0)),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_cooling failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 4. POST /calculate/chp - Calculate CHP-allocated emissions
    # ==================================================================

    @router.post("/calculate/chp", status_code=201)
    async def calculate_chp(
        body: CHPAllocBody,
    ) -> Dict[str, Any]:
        """Calculate CHP/cogeneration emission allocation.

        Allocates total CHP plant fuel emissions between electrical and
        thermal outputs using one of three GHG Protocol methods:

        1. **Efficiency method** (default): Allocates proportionally to
           energy output divided by respective conversion efficiency.
        2. **Energy method**: Allocates proportionally to energy content
           (GJ) of each output without efficiency adjustment.
        3. **Exergy method**: Allocates proportionally to exergy content
           using a Carnot factor based on steam temperature.

        Returns heat_share, power_share, cooling_share (fractions 0-1),
        allocated emissions per output (kgCO2e), total fuel emissions,
        and primary energy savings percentage.

        Permission: steam-heat-purchase:calculate
        """
        svc = _get_service()

        request_data: Dict[str, Any] = {
            "energy_type": "chp",
            "facility_id": body.facility_id,
            "total_fuel_gj": body.total_fuel_gj,
            "fuel_type": body.fuel_type.lower(),
            "heat_output_gj": body.heat_output_gj,
            "power_output_gj": body.power_output_gj,
            "cooling_output_gj": body.cooling_output_gj,
            "method": (body.method or "efficiency").lower(),
            "ambient_temperature_c": body.ambient_temperature_c,
            "gwp_source": (body.gwp_source or "AR6").upper(),
        }

        if body.electrical_efficiency is not None:
            request_data["electrical_efficiency"] = body.electrical_efficiency
        if body.thermal_efficiency is not None:
            request_data["thermal_efficiency"] = body.thermal_efficiency
        if body.steam_temperature_c is not None:
            request_data["steam_temperature_c"] = body.steam_temperature_c
        if body.tenant_id is not None:
            request_data["tenant_id"] = body.tenant_id

        try:
            result = svc.run_pipeline(request_data)

            _calculations.append(result)

            logger.info(
                "CHP allocation completed: calc_id=%s facility=%s "
                "method=%s heat_share=%.4f",
                result.get("calc_id", result.get("allocation_id", "")),
                result.get("facility_id", ""),
                body.method or "efficiency",
                float(result.get("heat_share", 0)),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_chp failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 5. POST /calculate/batch - Batch calculation
    # ==================================================================

    @router.post("/calculate/batch", status_code=201)
    async def calculate_batch(
        body: BatchCalcBody,
    ) -> Dict[str, Any]:
        """Execute batch thermal energy emission calculations.

        Processes multiple calculation requests (steam, heating, cooling,
        or CHP) in a single batch. Each item in the ``requests`` list
        must contain an ``energy_type`` field plus the corresponding
        parameters. Returns portfolio-level totals alongside individual
        results.

        Supported energy types:
        - steam: Purchased steam (fuel-based or direct EF)
        - district_heating: District heating (regional factors)
        - district_cooling: District cooling (COP-based)
        - chp: CHP/cogeneration (allocation method)

        Permission: steam-heat-purchase:calculate
        """
        svc = _get_service()

        batch_id = body.batch_id or str(uuid.uuid4())

        try:
            result = svc.run_batch(body.requests)

            # Enrich with batch metadata
            result["batch_id"] = batch_id
            if body.tenant_id is not None:
                result["tenant_id"] = body.tenant_id

            # Store individual results for later retrieval
            for r in result.get("results", []):
                _calculations.append(r)

            logger.info(
                "Batch completed: batch_id=%s total=%d successful=%d",
                batch_id,
                result.get("total_requests", 0),
                result.get("success_count", 0),
            )

            return _serialize_result(result)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "calculate_batch failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ==================================================================
    # 6. GET /factors/fuels - List all fuel emission factors
    # ==================================================================

    @router.get("/factors/fuels", status_code=200)
    async def list_fuel_factors() -> Dict[str, Any]:
        """List all fuel emission factors for steam and heat generation.

        Returns emission factors (CO2, CH4, N2O per GJ of fuel input)
        for all 14 supported fuel types. Factors are sourced from IPCC
        2006 Guidelines Volume 2, Chapter 2 and US EPA AP-42.

        Each entry includes:
        - co2_ef: kgCO2 per GJ fuel input (HHV basis)
        - ch4_ef: kgCH4 per GJ fuel input
        - n2o_ef: kgN2O per GJ fuel input
        - default_efficiency: Typical boiler efficiency (fraction 0-1)
        - is_biogenic: Whether the fuel is biogenic (biomass)

        No authentication required.
        """
        try:
            db = _get_db_engine()
            if hasattr(db, "get_all_fuel_factors"):
                factors = db.get_all_fuel_factors()
                return _serialize_result({
                    "fuel_factors": factors,
                    "count": len(factors),
                    "source": "database",
                })
        except HTTPException:
            pass
        except Exception as exc:
            logger.warning(
                "Database fuel factor lookup failed, using fallback: %s",
                exc,
            )

        # Fallback to constant table
        factors_dict = {}
        for fuel_name, ef_data in FUEL_EMISSION_FACTORS.items():
            factors_dict[fuel_name] = {
                "co2_ef": float(ef_data.get("co2_ef", 0)),
                "ch4_ef": float(ef_data.get("ch4_ef", 0)),
                "n2o_ef": float(ef_data.get("n2o_ef", 0)),
                "default_efficiency": float(
                    ef_data.get("default_efficiency", 0)
                ),
                "is_biogenic": bool(
                    int(ef_data.get("is_biogenic", 0))
                ),
            }

        return {
            "fuel_factors": factors_dict,
            "count": len(factors_dict),
            "source": "constant_table",
        }

    # ==================================================================
    # 7. GET /factors/fuels/{fuel_type} - Get specific fuel EF
    # ==================================================================

    @router.get("/factors/fuels/{fuel_type}", status_code=200)
    async def get_fuel_factor(
        fuel_type: str = Path(
            ...,
            description="Fuel type identifier: natural_gas, fuel_oil_2, "
            "fuel_oil_6, coal_bituminous, coal_subbituminous, "
            "coal_lignite, lpg, biomass_wood, biomass_biogas, "
            "municipal_waste, waste_heat, geothermal, solar_thermal, "
            "electric",
        ),
    ) -> Dict[str, Any]:
        """Get emission factor for a specific fuel type.

        Returns the CO2, CH4, and N2O emission factors per GJ of fuel
        input, default boiler efficiency, and biogenic flag for the
        specified fuel type.

        No authentication required.
        """
        fuel_key = fuel_type.strip().lower()

        # Try database first
        try:
            db = _get_db_engine()
            if hasattr(db, "get_fuel_factor"):
                factor = db.get_fuel_factor(fuel_key)
                if factor is not None:
                    return _serialize_result({
                        "fuel_type": fuel_key,
                        "factor": factor,
                        "source": "database",
                    })
        except HTTPException:
            pass
        except Exception as exc:
            logger.warning(
                "Database fuel factor lookup for %s failed: %s",
                fuel_key, exc,
            )

        # Fallback to constant table
        if fuel_key not in FUEL_EMISSION_FACTORS:
            raise HTTPException(
                status_code=404,
                detail=f"Fuel type '{fuel_type}' not found. "
                f"Available types: {sorted(FUEL_EMISSION_FACTORS.keys())}",
            )

        ef_data = FUEL_EMISSION_FACTORS[fuel_key]
        return {
            "fuel_type": fuel_key,
            "factor": {
                "co2_ef_per_gj": float(ef_data.get("co2_ef", 0)),
                "ch4_ef_per_gj": float(ef_data.get("ch4_ef", 0)),
                "n2o_ef_per_gj": float(ef_data.get("n2o_ef", 0)),
                "default_efficiency": float(
                    ef_data.get("default_efficiency", 0)
                ),
                "is_biogenic": bool(
                    int(ef_data.get("is_biogenic", 0))
                ),
            },
            "source": "constant_table",
        }

    # ==================================================================
    # 8. GET /factors/heating/{region} - Get DH network EF
    # ==================================================================

    @router.get("/factors/heating/{region}", status_code=200)
    async def get_heating_factor(
        region: str = Path(
            ...,
            description="Region identifier: denmark, sweden, finland, "
            "germany, poland, netherlands, france, uk, us, china, "
            "japan, south_korea, or global_default",
        ),
    ) -> Dict[str, Any]:
        """Get district heating emission factor for a region.

        Returns the composite emission factor (kgCO2e per GJ of
        delivered heat) and distribution loss percentage for the
        specified region.

        No authentication required.
        """
        region_key = region.strip().lower()

        # Try database first
        try:
            db = _get_db_engine()
            if hasattr(db, "get_heating_factor"):
                factor = db.get_heating_factor(region_key)
                if factor is not None:
                    return _serialize_result({
                        "region": region_key,
                        "factor": factor,
                        "source": "database",
                    })
        except HTTPException:
            pass
        except Exception as exc:
            logger.warning(
                "Database heating factor lookup for %s failed: %s",
                region_key, exc,
            )

        # Fallback to constant table
        if region_key not in DISTRICT_HEATING_FACTORS:
            raise HTTPException(
                status_code=404,
                detail=f"Region '{region}' not found. "
                f"Available regions: "
                f"{sorted(DISTRICT_HEATING_FACTORS.keys())}",
            )

        dh_data = DISTRICT_HEATING_FACTORS[region_key]
        return {
            "region": region_key,
            "factor": {
                "ef_kgco2e_per_gj": float(
                    dh_data.get("ef_kgco2e_per_gj", 0)
                ),
                "distribution_loss_pct": float(
                    dh_data.get("distribution_loss_pct", 0)
                ),
            },
            "source": "constant_table",
        }

    # ==================================================================
    # 9. GET /factors/cooling/{technology} - Get cooling system COP
    # ==================================================================

    @router.get("/factors/cooling/{technology}", status_code=200)
    async def get_cooling_factor(
        technology: str = Path(
            ...,
            description="Cooling technology: centrifugal_chiller, "
            "screw_chiller, reciprocating_chiller, absorption_single, "
            "absorption_double, absorption_triple, free_cooling, "
            "ice_storage, or thermal_storage",
        ),
    ) -> Dict[str, Any]:
        """Get cooling system performance parameters for a technology.

        Returns the COP range (min, max, default) and primary energy
        source (electricity or heat) for the specified cooling
        technology.

        No authentication required.
        """
        tech_key = technology.strip().lower()

        # Try database first
        try:
            db = _get_db_engine()
            if hasattr(db, "get_cooling_factor"):
                factor = db.get_cooling_factor(tech_key)
                if factor is not None:
                    return _serialize_result({
                        "technology": tech_key,
                        "factor": factor,
                        "source": "database",
                    })
        except HTTPException:
            pass
        except Exception as exc:
            logger.warning(
                "Database cooling factor lookup for %s failed: %s",
                tech_key, exc,
            )

        # Fallback to constant table
        if tech_key not in COOLING_SYSTEM_FACTORS:
            raise HTTPException(
                status_code=404,
                detail=f"Cooling technology '{technology}' not found. "
                f"Available technologies: "
                f"{sorted(COOLING_SYSTEM_FACTORS.keys())}",
            )

        cool_data = COOLING_SYSTEM_FACTORS[tech_key]
        energy_source = COOLING_ENERGY_SOURCE.get(tech_key, "electricity")

        return {
            "technology": tech_key,
            "factor": {
                "cop_min": float(cool_data.get("cop_min", 0)),
                "cop_max": float(cool_data.get("cop_max", 0)),
                "cop_default": float(cool_data.get("cop_default", 0)),
                "energy_source": energy_source,
            },
            "source": "constant_table",
        }

    # ==================================================================
    # 10. GET /factors/chp-defaults - Get CHP default parameters
    # ==================================================================

    @router.get("/factors/chp-defaults", status_code=200)
    async def get_chp_defaults() -> Dict[str, Any]:
        """Get default CHP/cogeneration efficiency parameters.

        Returns the default electrical efficiency, thermal efficiency,
        and overall efficiency for all 5 supported CHP fuel types.
        These defaults represent typical mid-size industrial CHP plants.

        Sources: US EPA CHP Partnership, COGEN Europe, IEA CHP assessments.

        No authentication required.
        """
        try:
            db = _get_db_engine()
            if hasattr(db, "get_chp_defaults"):
                defaults = db.get_chp_defaults()
                if defaults is not None:
                    return _serialize_result({
                        "chp_defaults": defaults,
                        "count": len(defaults),
                        "source": "database",
                    })
        except HTTPException:
            pass
        except Exception as exc:
            logger.warning(
                "Database CHP defaults lookup failed: %s", exc,
            )

        # Fallback to constant table
        defaults_dict = {}
        for fuel_name, eff_data in CHP_DEFAULT_EFFICIENCIES.items():
            defaults_dict[fuel_name] = {
                "electrical_efficiency": float(
                    eff_data.get("electrical_efficiency", 0)
                ),
                "thermal_efficiency": float(
                    eff_data.get("thermal_efficiency", 0)
                ),
                "overall_efficiency": float(
                    eff_data.get("overall_efficiency", 0)
                ),
            }

        return {
            "chp_defaults": defaults_dict,
            "count": len(defaults_dict),
            "source": "constant_table",
        }

    # ==================================================================
    # 11. POST /facilities - Register facility
    # ==================================================================

    @router.post("/facilities", status_code=201)
    async def create_facility(
        body: FacilityBody,
    ) -> Dict[str, Any]:
        """Register a new facility for Scope 2 thermal energy tracking.

        Creates a facility record with geographic, supplier connection,
        and network metadata. Every facility is scoped to a tenant and
        mapped to a region for district heating factor lookup.

        Permission: steam-heat-purchase:facilities:write
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
            "facility_type": body.facility_type.lower(),
            "country": body.country.upper(),
            "region": body.region.strip().lower(),
            "latitude": body.latitude,
            "longitude": body.longitude,
            "steam_suppliers": body.steam_suppliers or [],
            "heating_network": body.heating_network,
            "cooling_system": body.cooling_system,
            "tenant_id": body.tenant_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        _facilities.append(facility)

        logger.info(
            "Facility registered: id=%s name=%s country=%s region=%s",
            facility_id, body.name, body.country.upper(),
            body.region.strip().lower(),
        )

        return facility

    # ==================================================================
    # 12. GET /facilities/{facility_id} - Get facility
    # ==================================================================

    @router.get("/facilities/{facility_id}", status_code=200)
    async def get_facility(
        facility_id: str = Path(
            ..., description="Facility identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a registered facility by its unique identifier.

        Returns the full facility record including geographic location,
        supplier connections, network mappings, and tenant scoping.

        Permission: steam-heat-purchase:facilities:read
        """
        for fac in _facilities:
            if fac.get("facility_id") == facility_id:
                return fac

        raise HTTPException(
            status_code=404,
            detail=f"Facility {facility_id} not found",
        )

    # ==================================================================
    # 13. POST /suppliers - Register steam/heat supplier
    # ==================================================================

    @router.post("/suppliers", status_code=201)
    async def create_supplier(
        body: SupplierBody,
    ) -> Dict[str, Any]:
        """Register a new steam or heat supplier.

        Creates a supplier profile with fuel mix, boiler efficiency,
        and optionally a verified composite emission factor. Supplier
        data provides Tier 2 data quality for emission calculations.

        Permission: steam-heat-purchase:suppliers:write
        """
        supplier_id = body.supplier_id or str(uuid.uuid4())

        # Check for duplicate supplier_id
        for sup in _suppliers:
            if sup.get("supplier_id") == supplier_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"Supplier {supplier_id} already exists",
                )

        # Validate fuel mix fractions if provided
        if body.fuel_mix:
            for fuel, fraction in body.fuel_mix.items():
                if fraction < 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fuel mix fraction for '{fuel}' must "
                        f"be >= 0, got {fraction}",
                    )

        supplier = {
            "supplier_id": supplier_id,
            "name": body.name,
            "fuel_mix": body.fuel_mix or {},
            "boiler_efficiency": body.boiler_efficiency,
            "supplier_ef_kgco2e_per_gj": body.supplier_ef_kgco2e_per_gj,
            "country": body.country.upper(),
            "region": body.region.strip().lower() if body.region else None,
            "verified": body.verified,
            "data_quality_tier": body.data_quality_tier or "tier_1",
            "tenant_id": body.tenant_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        _suppliers.append(supplier)

        logger.info(
            "Supplier registered: id=%s name=%s country=%s verified=%s",
            supplier_id, body.name, body.country.upper(), body.verified,
        )

        return supplier

    # ==================================================================
    # 14. GET /suppliers/{supplier_id} - Get supplier
    # ==================================================================

    @router.get("/suppliers/{supplier_id}", status_code=200)
    async def get_supplier(
        supplier_id: str = Path(
            ..., description="Supplier identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a registered steam/heat supplier by its unique identifier.

        Returns the full supplier profile including fuel mix, boiler
        efficiency, composite emission factor, verification status,
        and data quality tier.

        Permission: steam-heat-purchase:suppliers:read
        """
        for sup in _suppliers:
            if sup.get("supplier_id") == supplier_id:
                return sup

        raise HTTPException(
            status_code=404,
            detail=f"Supplier {supplier_id} not found",
        )

    # ==================================================================
    # 15. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty", status_code=200)
    async def run_uncertainty(
        body: UncertaintyBody,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a thermal energy calculation result.

        Performs Monte Carlo simulation or analytical error propagation
        to quantify the uncertainty range of the emission estimate.
        Returns statistical characterisation including mean, standard
        deviation, confidence intervals, and coefficient of variation.

        Uncertainty sources quantified:
        - Activity data (metered consumption) uncertainty
        - Emission factor uncertainty (fuel-specific or regional)
        - Boiler/CHP efficiency uncertainty
        - Distribution loss uncertainty (district heating)
        - COP uncertainty (district cooling)

        Permission: steam-heat-purchase:uncertainty:run
        """
        # Find the calculation
        calc = None
        for c in _calculations:
            cid = c.get("calc_id") or c.get("calculation_id", "")
            if cid == body.calc_id:
                calc = c
                break

        if calc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation {body.calc_id} not found. "
                "Perform a calculation first before running uncertainty.",
            )

        try:
            unc_engine = _get_uncertainty_engine()

            if unc_engine is not None and hasattr(
                unc_engine, "run_monte_carlo"
            ):
                base_emissions_kg = calc.get(
                    "total_co2e_kg", Decimal("0")
                )
                if not isinstance(base_emissions_kg, Decimal):
                    base_emissions_kg = Decimal(str(base_emissions_kg))

                mc_result = unc_engine.run_monte_carlo(
                    base_emissions_kg=base_emissions_kg,
                    iterations=body.iterations,
                    seed=body.seed,
                )

                result = {
                    "calc_id": body.calc_id,
                    "method": body.method or "monte_carlo",
                    "iterations": body.iterations,
                    "confidence_level": body.confidence_level,
                    "seed": body.seed,
                    "activity_data_uncertainty_pct": (
                        body.activity_data_uncertainty_pct
                    ),
                    "emission_factor_uncertainty_pct": (
                        body.emission_factor_uncertainty_pct
                    ),
                    "efficiency_uncertainty_pct": (
                        body.efficiency_uncertainty_pct
                    ),
                    "result": _serialize_result(mc_result),
                    "analysed_at": datetime.now(timezone.utc).isoformat(),
                }

                _uncertainty_results.append(result)

                logger.info(
                    "Uncertainty analysis completed: calc=%s "
                    "method=%s iterations=%d",
                    body.calc_id, body.method, body.iterations,
                )

                return result

            else:
                # Uncertainty engine not available - return IPCC estimate
                total_co2e = float(calc.get("total_co2e_kg", 0))
                # Default IPCC Tier 1 uncertainty: +/- 25%
                uncertainty_pct = 0.25

                result = {
                    "calc_id": body.calc_id,
                    "method": "ipcc_default_estimate",
                    "iterations": 0,
                    "confidence_level": body.confidence_level,
                    "mean_co2e_kg": total_co2e,
                    "std_dev_kg": total_co2e * uncertainty_pct / 1.96,
                    "ci_lower_kg": total_co2e * (1 - uncertainty_pct),
                    "ci_upper_kg": total_co2e * (1 + uncertainty_pct),
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
    # 16. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check", status_code=200)
    async def check_compliance(
        body: ComplianceCheckBody,
    ) -> Dict[str, Any]:
        """Run regulatory compliance check on a thermal energy calculation.

        Evaluates a previously computed calculation against applicable
        Scope 2 reporting frameworks. Supports 7 frameworks covering
        data completeness, methodological correctness, emission factor
        quality, and reporting readiness.

        Accepts either ``calc_id`` (to look up a stored result) or
        ``calc_result`` (inline data). At least one must be provided.

        Supported frameworks:
        - ghg_protocol_scope2: GHG Protocol Scope 2 Guidance (2015)
        - ipcc_2006: IPCC 2006 Guidelines
        - iso_14064: ISO 14064-1:2018
        - csrd_e1: CSRD / ESRS E1
        - epa_ghgrp: US EPA Greenhouse Gas Reporting Program
        - defra_secr: UK DEFRA Reporting (SECR)
        - cdp: CDP Climate Change

        Permission: steam-heat-purchase:compliance:check
        """
        # Resolve the calculation data
        calc = body.calc_result
        if calc is None and body.calc_id is not None:
            for c in _calculations:
                cid = c.get("calc_id") or c.get("calculation_id", "")
                if cid == body.calc_id:
                    calc = c
                    break
            if calc is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Calculation {body.calc_id} not found. "
                    "Perform a calculation first or provide "
                    "calc_result inline.",
                )
        elif calc is None:
            raise HTTPException(
                status_code=400,
                detail="Either calc_id or calc_result must be provided",
            )

        try:
            comp_engine = _get_compliance_engine()

            if comp_engine is not None and hasattr(
                comp_engine, "check_compliance"
            ):
                frameworks = (
                    body.frameworks if body.frameworks else None
                )
                results = comp_engine.check_compliance(
                    calc, frameworks
                )

                check_id = str(uuid.uuid4())
                result = {
                    "id": check_id,
                    "calc_id": (
                        body.calc_id
                        or calc.get("calc_id", "inline")
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
                    "checked_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
                _compliance_results.append(result)

                logger.info(
                    "Compliance check completed: calc=%s frameworks=%s",
                    body.calc_id or "inline",
                    body.frameworks or "all",
                )

                return result

            else:
                # Compliance engine not available - return placeholder
                check_id = str(uuid.uuid4())
                result = {
                    "id": check_id,
                    "calc_id": (
                        body.calc_id
                        or calc.get("calc_id", "inline")
                    ),
                    "frameworks_checked": (
                        body.frameworks if body.frameworks
                        else ["all"]
                    ),
                    "results": [],
                    "status": "compliance_engine_unavailable",
                    "message": "Compliance checker engine is not "
                    "initialized. Results will be available when "
                    "the engine is loaded.",
                    "checked_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
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
    # 17. GET /compliance/frameworks - List available frameworks
    # ==================================================================

    @router.get("/compliance/frameworks", status_code=200)
    async def list_compliance_frameworks() -> Dict[str, Any]:
        """List all available regulatory compliance frameworks.

        Returns the 7 supported regulatory frameworks for thermal energy
        emission compliance checking, each with an identifier, display
        name, and brief description.

        No authentication required.
        """
        frameworks = [
            {
                "id": "ghg_protocol_scope2",
                "name": "GHG Protocol Scope 2 Guidance",
                "version": "2015",
                "description": "Corporate Value Chain (Scope 2) "
                "Accounting and Reporting Standard. Defines "
                "location-based and market-based methods for "
                "reporting Scope 2 emissions from purchased energy.",
            },
            {
                "id": "ipcc_2006",
                "name": "IPCC 2006 Guidelines",
                "version": "2006 / 2019 Refinement",
                "description": "IPCC Guidelines for National Greenhouse "
                "Gas Inventories. Volume 2 Energy, Chapter 2 provides "
                "stationary combustion emission factors used for "
                "steam and heat calculations.",
            },
            {
                "id": "iso_14064",
                "name": "ISO 14064-1:2018",
                "version": "2018",
                "description": "Specification with guidance for "
                "quantification and reporting of greenhouse gas "
                "emissions and removals at the organisation level.",
            },
            {
                "id": "csrd_e1",
                "name": "CSRD / ESRS E1",
                "version": "2024",
                "description": "European Sustainability Reporting "
                "Standards E1: Climate Change. Requires disclosure "
                "of Scope 1, 2, and 3 emissions using location-based "
                "and market-based methods.",
            },
            {
                "id": "epa_ghgrp",
                "name": "US EPA GHGRP",
                "version": "40 CFR Part 98",
                "description": "US Environmental Protection Agency "
                "Greenhouse Gas Reporting Program. Mandatory reporting "
                "for facilities emitting 25,000 tCO2e/yr or more.",
            },
            {
                "id": "defra_secr",
                "name": "UK DEFRA SECR",
                "version": "2019",
                "description": "UK Streamlined Energy and Carbon "
                "Reporting. Mandatory for large UK companies to report "
                "energy use and GHG emissions in annual reports.",
            },
            {
                "id": "cdp",
                "name": "CDP Climate Change",
                "version": "2024",
                "description": "Carbon Disclosure Project Climate Change "
                "questionnaire. Requests detailed Scope 1, 2, and 3 "
                "emissions data with breakdowns by energy type, "
                "method, and verification status.",
            },
        ]

        return {
            "frameworks": frameworks,
            "count": len(frameworks),
        }

    # ==================================================================
    # 18. POST /aggregate - Aggregate results
    # ==================================================================

    @router.post("/aggregate", status_code=200)
    async def aggregate_results(
        body: AggregationBody,
    ) -> Dict[str, Any]:
        """Aggregate multiple thermal energy calculation results.

        Groups and sums emissions from specified calculations by the
        requested aggregation dimension (facility, fuel, energy type,
        supplier, or time period). Returns portfolio-level totals and
        per-group breakdowns.

        Aggregation dimensions:
        - by_facility: Group by facility_id
        - by_fuel: Group by fuel_type
        - by_energy_type: Group by energy_type (steam/heating/cooling)
        - by_supplier: Group by supplier_id
        - by_period: Group by reporting_period

        Permission: steam-heat-purchase:read
        """
        # Validate aggregation type
        valid_agg_types = {
            "by_facility", "by_fuel", "by_energy_type",
            "by_supplier", "by_period",
        }
        agg_type = body.aggregation_type.lower()
        if agg_type not in valid_agg_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aggregation_type '{body.aggregation_type}'. "
                f"Must be one of: {sorted(valid_agg_types)}",
            )

        # Map aggregation type to grouping field
        agg_field_map = {
            "by_facility": "facility_id",
            "by_fuel": "fuel_type",
            "by_energy_type": "energy_type",
            "by_supplier": "supplier_id",
            "by_period": "reporting_period",
        }
        group_field = agg_field_map[agg_type]

        # Find matching calculations
        matched: List[Dict[str, Any]] = []
        for calc in _calculations:
            cid = calc.get("calc_id") or calc.get("calculation_id", "")
            if cid in body.calc_ids:
                matched.append(calc)

        if not matched:
            raise HTTPException(
                status_code=404,
                detail="No matching calculations found for the "
                "provided calc_ids",
            )

        # Aggregate totals
        total_co2e_kg = Decimal("0")
        total_fossil_co2e_kg = Decimal("0")
        total_biogenic_co2_kg = Decimal("0")
        breakdown: Dict[str, Decimal] = {}

        for calc in matched:
            co2e = _safe_decimal(calc.get("total_co2e_kg", 0))
            fossil = _safe_decimal(calc.get("fossil_co2e_kg", co2e))
            biogenic = _safe_decimal(calc.get("biogenic_co2_kg", 0))

            total_co2e_kg += co2e
            total_fossil_co2e_kg += fossil
            total_biogenic_co2_kg += biogenic

            group_key = str(calc.get(group_field, "unknown"))
            breakdown[group_key] = (
                breakdown.get(group_key, Decimal("0")) + co2e
            )

        aggregation_id = str(uuid.uuid4())

        result = {
            "aggregation_id": aggregation_id,
            "aggregation_type": agg_type,
            "total_co2e_kg": _dec_to_float(total_co2e_kg),
            "total_fossil_co2e_kg": _dec_to_float(total_fossil_co2e_kg),
            "total_biogenic_co2_kg": _dec_to_float(total_biogenic_co2_kg),
            "breakdown": {
                k: _dec_to_float(v) for k, v in breakdown.items()
            },
            "count": len(matched),
            "calc_ids_requested": len(body.calc_ids),
            "calc_ids_matched": len(matched),
            "tenant_id": body.tenant_id,
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Aggregation completed: id=%s type=%s count=%d "
            "total_co2e=%.2f kg",
            aggregation_id, agg_type, len(matched),
            float(total_co2e_kg),
        )

        return result

    # ==================================================================
    # 19. GET /calculations/{calc_id} - Get calculation result
    # ==================================================================

    @router.get("/calculations/{calc_id}", status_code=200)
    async def get_calculation(
        calc_id: str = Path(
            ..., description="Calculation identifier",
        ),
    ) -> Dict[str, Any]:
        """Get a thermal energy calculation result by its unique ID.

        Returns the full calculation result including total CO2e, fossil
        CO2e, biogenic CO2, per-gas breakdown, effective emission factor,
        calculation method, trace steps, and provenance hash.

        Permission: steam-heat-purchase:read
        """
        for calc in _calculations:
            cid = calc.get("calc_id") or calc.get("calculation_id", "")
            if cid == calc_id:
                return _serialize_result(calc)

        raise HTTPException(
            status_code=404,
            detail=f"Calculation {calc_id} not found",
        )

    # ==================================================================
    # 20. GET /health - Health check
    # ==================================================================

    @router.get("/health", status_code=200)
    async def health_check() -> Dict[str, Any]:
        """Service health check for load balancers and monitoring.

        Returns the health status of all Steam/Heat Purchase Agent
        engines. No authentication required.

        Checks:
        - Pipeline engine availability
        - Database engine (EF lookups, fuel data, regional factors)
        - Steam emissions calculator engine
        - Heat/cooling calculator engine
        - CHP allocation engine
        - Uncertainty quantifier engine
        - Compliance checker engine
        """
        engines_status: Dict[str, Any] = {
            "pipeline": False,
            "database": False,
            "steam_calculator": False,
            "heat_cooling_calculator": False,
            "chp_allocation": False,
            "uncertainty": False,
            "compliance": False,
        }

        overall_healthy = False

        try:
            svc = _get_service()
            engines_status["pipeline"] = True

            if hasattr(svc, "_db_engine"):
                engines_status["database"] = (
                    svc._db_engine is not None
                )
            if hasattr(svc, "_steam_engine"):
                engines_status["steam_calculator"] = (
                    svc._steam_engine is not None
                )
            if hasattr(svc, "_heat_cool_engine"):
                engines_status["heat_cooling_calculator"] = (
                    svc._heat_cool_engine is not None
                )
            if hasattr(svc, "_chp_engine"):
                engines_status["chp_allocation"] = (
                    svc._chp_engine is not None
                )
            if hasattr(svc, "_uncertainty_engine"):
                engines_status["uncertainty"] = (
                    svc._uncertainty_engine is not None
                )
            if hasattr(svc, "_compliance_engine"):
                engines_status["compliance"] = (
                    svc._compliance_engine is not None
                )

            overall_healthy = engines_status["pipeline"]

        except Exception as exc:
            logger.warning("Health check: service unavailable: %s", exc)

        engines_active = sum(
            1 for v in engines_status.values() if v
        )
        engines_total = len(engines_status)

        # Pipeline statistics
        pipeline_stats: Dict[str, Any] = {}
        try:
            svc = _get_service()
            if hasattr(svc, "get_pipeline_status"):
                pipeline_stats = _serialize_result(
                    svc.get_pipeline_status()
                )
        except Exception:
            pass

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "service": "steam-heat-purchase",
            "version": VERSION,
            "agent": "AGENT-MRV-011",
            "engines": engines_status,
            "engines_active": engines_active,
            "engines_total": engines_total,
            "pipeline_stats": pipeline_stats,
            "api_record_counts": {
                "calculations": len(_calculations),
                "facilities": len(_facilities),
                "suppliers": len(_suppliers),
                "compliance_checks": len(_compliance_results),
                "uncertainty_analyses": len(_uncertainty_results),
            },
            "supported_energy_types": [
                "steam",
                "district_heating",
                "district_cooling",
                "chp",
            ],
            "supported_fuel_types": sorted(FUEL_EMISSION_FACTORS.keys()),
            "supported_regions": sorted(DISTRICT_HEATING_FACTORS.keys()),
            "supported_cooling_technologies": sorted(
                COOLING_SYSTEM_FACTORS.keys()
            ),
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
        "Could not create steam-heat-purchase router "
        "(FastAPI not available)"
    )


# ===================================================================
# Public API
# ===================================================================

__all__ = ["create_router", "router"]
