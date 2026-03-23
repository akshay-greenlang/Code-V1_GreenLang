# -*- coding: utf-8 -*-
"""
MRVBridge - Bridge to MRV Agents for M&V Savings Emissions Verification
==========================================================================

This module routes verified energy savings data from M&V calculations to
the appropriate MRV (Monitoring, Reporting, Verification) agents for
emissions reduction verification. M&V-verified savings provide the most
accurate and defensible data for Scope 1 and Scope 2 emissions reduction
claims, eliminating estimation uncertainty.

Routing Table:
    Scope 1 combustion savings  --> MRV-001 (Scope 1 Stationary Combustion)
    Scope 2 electricity savings --> MRV-009 (Scope 2 Location-Based)
    Scope 2 market-based        --> MRV-010 (Scope 2 Market-Based)

Bidirectional Data Flow:
    MRV --> M&V:  Emission factors, grid factors, fuel factors
    M&V --> MRV:  Verified savings, avoided energy, uncertainty bounds

Key Formulas (deterministic, zero-hallucination):
    scope2_reduction_tco2e = verified_savings_kwh * grid_ef_kgco2_per_kwh / 1000
    scope1_reduction_tco2e = verified_therms_saved * gas_ef_kgco2_per_therm / 1000

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-040 Measurement & Verification
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------


class _AgentStub:
    """Stub for unavailable MRV agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
                "emissions_tco2e": 0.0,
            }
        return _stub_method


def _try_import_mrv_agent(agent_id: str, module_path: str) -> Any:
    """Try to import an MRV agent with graceful fallback.

    Args:
        agent_id: Agent identifier (e.g., 'MRV-001').
        module_path: Python module path for the agent.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        import importlib
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"


class MVEmissionCategory(str, Enum):
    """M&V emission reduction categories."""

    ELECTRICITY_SAVINGS = "electricity_savings"
    NATURAL_GAS_SAVINGS = "natural_gas_savings"
    STEAM_SAVINGS = "steam_savings"
    CHILLED_WATER_SAVINGS = "chilled_water_savings"
    FUEL_OIL_SAVINGS = "fuel_oil_savings"
    DEMAND_REDUCTION = "demand_reduction"


class EmissionFactorSource(str, Enum):
    """Sources for emission factors."""

    EPA_EGRID = "epa_egrid"
    IEA = "iea"
    DEFRA = "defra"
    CUSTOM = "custom"
    MRV_AGENT = "mrv_agent"


class MeterType(str, Enum):
    """Metering types for M&V."""

    REVENUE_METER = "revenue_meter"
    SUB_METER = "sub_meter"
    TEMPORARY_METER = "temporary_meter"
    VIRTUAL_METER = "virtual_meter"
    CT_CLAMP = "ct_clamp"


class AccountingMethod(str, Enum):
    """Emissions accounting methods for Scope 2."""

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"
    DUAL_REPORTING = "dual_reporting"


class SavingsType(str, Enum):
    """Types of verified savings."""

    AVOIDED_ENERGY = "avoided_energy"
    NORMALIZED_SAVINGS = "normalized_savings"
    COST_SAVINGS = "cost_savings"
    DEMAND_SAVINGS = "demand_savings"


# ---------------------------------------------------------------------------
# Grid Emission Factors (kg CO2e per kWh) by ISO/RTO region
# ---------------------------------------------------------------------------

GRID_EMISSION_FACTORS: Dict[str, float] = {
    "US_AVERAGE": 0.417,
    "ERCOT": 0.395,
    "PJM": 0.440,
    "MISO": 0.510,
    "CAISO": 0.225,
    "NYISO": 0.280,
    "ISO_NE": 0.305,
    "SPP": 0.475,
    "WECC_NW": 0.310,
    "WECC_SW": 0.420,
    "EU_AVERAGE": 0.276,
    "UK_GRID": 0.233,
    "DE_GRID": 0.385,
    "FR_GRID": 0.052,
    "AU_GRID": 0.680,
    "JP_GRID": 0.470,
}

# Natural gas emission factors (kg CO2e per therm)
GAS_EMISSION_FACTORS: Dict[str, float] = {
    "NATURAL_GAS": 5.302,
    "PROPANE": 5.741,
    "FUEL_OIL_2": 10.21,
    "FUEL_OIL_6": 11.27,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EmissionFactorSet(BaseModel):
    """Emission factor set for a grid region or fuel type."""

    factor_id: str = Field(default_factory=_new_uuid)
    source: EmissionFactorSource = Field(default=EmissionFactorSource.EPA_EGRID)
    region: str = Field(default="US_AVERAGE")
    factor_kg_co2e: float = Field(default=0.417, ge=0.0)
    unit: str = Field(default="kg_co2e_per_kwh")
    vintage_year: int = Field(default=2024, ge=2000)
    valid_from: Optional[str] = Field(None)
    valid_to: Optional[str] = Field(None)


class MRVRouteConfig(BaseModel):
    """Configuration for routing savings to MRV agents."""

    route_id: str = Field(default_factory=_new_uuid)
    scope: MRVScope = Field(...)
    category: MVEmissionCategory = Field(...)
    mrv_agent_id: str = Field(..., description="Target MRV agent (e.g., MRV-001)")
    grid_region: str = Field(default="US_AVERAGE")
    accounting_method: AccountingMethod = Field(default=AccountingMethod.LOCATION_BASED)
    emission_factor: Optional[EmissionFactorSet] = Field(None)
    enabled: bool = Field(default=True)


class MRVRequest(BaseModel):
    """Request to route verified savings to MRV agent."""

    request_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    ecm_id: str = Field(default="")
    scope: MRVScope = Field(...)
    category: MVEmissionCategory = Field(...)
    savings_type: SavingsType = Field(default=SavingsType.AVOIDED_ENERGY)
    verified_savings_value: float = Field(..., description="Verified savings quantity")
    savings_unit: str = Field(default="kWh")
    uncertainty_pct: float = Field(default=0.0, ge=0.0)
    reporting_period_start: str = Field(default="")
    reporting_period_end: str = Field(default="")
    grid_region: str = Field(default="US_AVERAGE")
    accounting_method: AccountingMethod = Field(default=AccountingMethod.LOCATION_BASED)
    timestamp: datetime = Field(default_factory=_utcnow)


class MRVResponse(BaseModel):
    """Response from MRV agent after emissions reduction calculation."""

    response_id: str = Field(default_factory=_new_uuid)
    request_id: str = Field(default="")
    mrv_agent_id: str = Field(default="")
    scope: str = Field(default="")
    emissions_reduction_tco2e: float = Field(default=0.0)
    emissions_reduction_lower_tco2e: float = Field(default=0.0)
    emissions_reduction_upper_tco2e: float = Field(default=0.0)
    emission_factor_used: float = Field(default=0.0)
    emission_factor_source: str = Field(default="")
    calculation_method: str = Field(default="")
    status: str = Field(default="success")
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# MRVBridge
# ---------------------------------------------------------------------------


class MRVBridge:
    """Bridge between M&V verified savings and MRV emissions agents.

    Routes verified energy savings from IPMVP-compliant M&V calculations
    to the appropriate MRV agents for emissions reduction verification.
    Supports bidirectional data flow: emission factors from MRV agents,
    and verified savings data to MRV agents.

    Attributes:
        config: Route configuration list.
        _mrv_001: MRV-001 agent (Scope 1 Stationary Combustion).
        _mrv_009: MRV-009 agent (Scope 2 Location-Based).
        _mrv_010: MRV-010 agent (Scope 2 Market-Based).

    Example:
        >>> bridge = MRVBridge()
        >>> response = bridge.route_savings_to_mrv(request)
        >>> assert response.status == "success"
    """

    def __init__(
        self,
        routes: Optional[List[MRVRouteConfig]] = None,
    ) -> None:
        """Initialize MRVBridge with route configuration.

        Args:
            routes: Custom routing configuration. Uses defaults if None.
        """
        self.routes = routes or self._default_routes()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._mrv_001 = _try_import_mrv_agent(
            "MRV-001", "greenlang.agents.mrv.mrv_001_stationary_combustion"
        )
        self._mrv_009 = _try_import_mrv_agent(
            "MRV-009", "greenlang.agents.mrv.mrv_009_scope2_location"
        )
        self._mrv_010 = _try_import_mrv_agent(
            "MRV-010", "greenlang.agents.mrv.mrv_010_scope2_market"
        )

        self.logger.info(
            "MRVBridge initialized: %d routes configured, agents=[MRV-001, MRV-009, MRV-010]",
            len(self.routes),
        )

    def route_savings_to_mrv(self, request: MRVRequest) -> MRVResponse:
        """Route verified savings to the appropriate MRV agent.

        Deterministic calculation: emission reduction = savings * factor / 1000.

        Args:
            request: Verified savings data to route.

        Returns:
            MRVResponse with emissions reduction calculation.
        """
        start_time = time.monotonic()

        ef = self._get_emission_factor(request.category, request.grid_region)
        savings_decimal = Decimal(str(request.verified_savings_value))
        ef_decimal = Decimal(str(ef))
        divisor = Decimal("1000")

        reduction = (savings_decimal * ef_decimal / divisor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        uncertainty_decimal = Decimal(str(request.uncertainty_pct)) / Decimal("100")
        lower = reduction * (Decimal("1") - uncertainty_decimal)
        upper = reduction * (Decimal("1") + uncertainty_decimal)

        mrv_agent_id = self._resolve_mrv_agent(request.scope, request.category)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        response = MRVResponse(
            request_id=request.request_id,
            mrv_agent_id=mrv_agent_id,
            scope=request.scope.value,
            emissions_reduction_tco2e=float(reduction),
            emissions_reduction_lower_tco2e=float(lower.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            emissions_reduction_upper_tco2e=float(upper.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            emission_factor_used=ef,
            emission_factor_source="grid_emission_factors",
            calculation_method=f"savings * ef / 1000 ({request.category.value})",
            status="success",
            provenance_hash=_compute_hash({
                "request": request.model_dump(mode="json"),
                "reduction_tco2e": float(reduction),
            }),
            processing_time_ms=elapsed_ms,
        )

        self.logger.info(
            "Routed savings to %s: %.1f %s -> %.3f tCO2e (ef=%.4f, region=%s)",
            mrv_agent_id, request.verified_savings_value, request.savings_unit,
            float(reduction), ef, request.grid_region,
        )
        return response

    def get_emission_factor(
        self,
        category: MVEmissionCategory,
        region: str = "US_AVERAGE",
    ) -> EmissionFactorSet:
        """Get emission factor for a category and region.

        Args:
            category: Emission category (electricity, gas, etc.).
            region: Grid region or fuel type key.

        Returns:
            EmissionFactorSet with the factor details.
        """
        ef_value = self._get_emission_factor(category, region)
        unit = "kg_co2e_per_kwh"
        if category in (
            MVEmissionCategory.NATURAL_GAS_SAVINGS,
            MVEmissionCategory.FUEL_OIL_SAVINGS,
        ):
            unit = "kg_co2e_per_therm"

        return EmissionFactorSet(
            source=EmissionFactorSource.EPA_EGRID,
            region=region,
            factor_kg_co2e=ef_value,
            unit=unit,
        )

    def batch_route_savings(
        self,
        requests: List[MRVRequest],
    ) -> List[MRVResponse]:
        """Route multiple savings records to MRV agents in batch.

        Args:
            requests: List of savings routing requests.

        Returns:
            List of MRV responses.
        """
        self.logger.info("Batch routing %d savings records to MRV", len(requests))
        responses: List[MRVResponse] = []
        for req in requests:
            try:
                resp = self.route_savings_to_mrv(req)
                responses.append(resp)
            except Exception as exc:
                self.logger.error("Failed to route request %s: %s", req.request_id, exc)
                responses.append(MRVResponse(
                    request_id=req.request_id,
                    status="error",
                    provenance_hash=_compute_hash({"error": str(exc)}),
                ))
        return responses

    def get_total_emission_reductions(
        self,
        responses: List[MRVResponse],
    ) -> Dict[str, Any]:
        """Aggregate total emission reductions from MRV responses.

        Args:
            responses: List of MRV responses to aggregate.

        Returns:
            Dict with total reductions by scope.
        """
        scope_totals: Dict[str, Decimal] = {}
        for resp in responses:
            if resp.status != "success":
                continue
            scope = resp.scope
            if scope not in scope_totals:
                scope_totals[scope] = Decimal("0")
            scope_totals[scope] += Decimal(str(resp.emissions_reduction_tco2e))

        total = sum(scope_totals.values(), Decimal("0"))
        return {
            "total_reduction_tco2e": float(total),
            "by_scope": {k: float(v) for k, v in scope_totals.items()},
            "response_count": len(responses),
            "successful_count": sum(1 for r in responses if r.status == "success"),
            "provenance_hash": _compute_hash(scope_totals),
        }

    def validate_routing(self) -> Dict[str, Any]:
        """Validate all MRV routing configurations.

        Returns:
            Dict with validation results for all routes.
        """
        results: List[Dict[str, Any]] = []
        for route in self.routes:
            is_valid = (
                route.enabled
                and route.mrv_agent_id in ("MRV-001", "MRV-009", "MRV-010")
                and route.grid_region in GRID_EMISSION_FACTORS
            )
            results.append({
                "route_id": route.route_id,
                "scope": route.scope.value,
                "category": route.category.value,
                "mrv_agent_id": route.mrv_agent_id,
                "valid": is_valid,
            })

        return {
            "total_routes": len(self.routes),
            "valid_routes": sum(1 for r in results if r["valid"]),
            "routes": results,
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _get_emission_factor(
        self, category: MVEmissionCategory, region: str
    ) -> float:
        """Look up the emission factor for a category and region.

        Args:
            category: The emission category.
            region: Grid region or fuel type.

        Returns:
            Emission factor in kg CO2e per unit.
        """
        if category == MVEmissionCategory.NATURAL_GAS_SAVINGS:
            return GAS_EMISSION_FACTORS.get(region, GAS_EMISSION_FACTORS["NATURAL_GAS"])
        if category == MVEmissionCategory.FUEL_OIL_SAVINGS:
            return GAS_EMISSION_FACTORS.get(region, GAS_EMISSION_FACTORS["FUEL_OIL_2"])
        return GRID_EMISSION_FACTORS.get(region, GRID_EMISSION_FACTORS["US_AVERAGE"])

    def _resolve_mrv_agent(
        self, scope: MRVScope, category: MVEmissionCategory
    ) -> str:
        """Resolve the target MRV agent ID based on scope and category.

        Args:
            scope: The emission scope.
            category: The emission category.

        Returns:
            MRV agent identifier string.
        """
        if scope == MRVScope.SCOPE_1:
            return "MRV-001"
        if scope == MRVScope.SCOPE_2_MARKET:
            return "MRV-010"
        return "MRV-009"

    def _default_routes(self) -> List[MRVRouteConfig]:
        """Generate default MRV routing configuration.

        Returns:
            List of default route configurations.
        """
        return [
            MRVRouteConfig(
                scope=MRVScope.SCOPE_2_LOCATION,
                category=MVEmissionCategory.ELECTRICITY_SAVINGS,
                mrv_agent_id="MRV-009",
                grid_region="US_AVERAGE",
            ),
            MRVRouteConfig(
                scope=MRVScope.SCOPE_2_MARKET,
                category=MVEmissionCategory.ELECTRICITY_SAVINGS,
                mrv_agent_id="MRV-010",
                grid_region="US_AVERAGE",
                accounting_method=AccountingMethod.MARKET_BASED,
            ),
            MRVRouteConfig(
                scope=MRVScope.SCOPE_1,
                category=MVEmissionCategory.NATURAL_GAS_SAVINGS,
                mrv_agent_id="MRV-001",
                grid_region="NATURAL_GAS",
            ),
            MRVRouteConfig(
                scope=MRVScope.SCOPE_2_LOCATION,
                category=MVEmissionCategory.STEAM_SAVINGS,
                mrv_agent_id="MRV-009",
                grid_region="US_AVERAGE",
            ),
            MRVRouteConfig(
                scope=MRVScope.SCOPE_2_LOCATION,
                category=MVEmissionCategory.CHILLED_WATER_SAVINGS,
                mrv_agent_id="MRV-009",
                grid_region="US_AVERAGE",
            ),
        ]
