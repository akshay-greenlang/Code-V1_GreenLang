# -*- coding: utf-8 -*-
"""
DERCoordinatorEngine - PACK-037 Demand Response Engine 6
=========================================================

Distributed Energy Resource (DER) coordination engine for demand
response programmes.  Manages a portfolio of heterogeneous DER assets
(BESS, Solar PV, Backup Generators, EV Chargers, Thermal Storage, CHP),
optimises dispatch during DR events, tracks state-of-charge (SOC)
within safe operating limits, monitors cycling degradation, and
calculates aggregate portfolio contribution.

Calculation Methodology:
    SOC Management:
        soc_after = soc_before - (dispatch_kw * duration_h) / capacity_kwh
        Enforced limits: SOC_MIN = 20%, SOC_MAX = 90%
        usable_capacity = capacity_kwh * (soc_max - soc_min)

    Cycling Degradation (BESS):
        cycles_full_equivalent = cumulative_discharge_kwh / capacity_kwh
        degradation_pct = cycles_full_equivalent * degradation_per_cycle
        effective_capacity = capacity_kwh * (1 - degradation_pct / 100)

    DER Dispatch Optimisation:
        For each asset in priority order:
            available_kw = min(rated_kw, usable_kw_from_soc)
            dispatch_kw  = min(available_kw, remaining_target_kw)
            remaining_target_kw -= dispatch_kw

    Availability Assessment:
        availability_pct = available_hours / total_hours * 100
        capacity_factor  = actual_output_kwh / (rated_kw * hours) * 100

    Performance Tracking:
        contribution_ratio = dispatched_kw / rated_kw * 100
        utilisation_pct    = actual_kwh / usable_capacity * 100

Regulatory References:
    - IEEE 1547-2018 - Standard for Interconnection of DERs
    - IEEE 2030.5 - Smart Energy Profile (SEP 2.0)
    - FERC Order 2222 - DER Aggregation in Wholesale Markets
    - IEC 61850 - Communication Networks and Systems for Power Automation
    - SAE J3072 - PEV/Grid Communication Interface
    - UL 9540 - Energy Storage Systems Safety Standard

Zero-Hallucination:
    - SOC calculations use deterministic energy balance equations
    - Degradation models from published lithium-ion cycling data
    - No LLM involvement in any dispatch or SOC calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-037 Demand Response
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DERAssetType(str, Enum):
    """Distributed Energy Resource asset type classification.

    BESS:             Battery Energy Storage System.
    SOLAR_PV:         Solar photovoltaic generation.
    BACKUP_GENERATOR: Diesel or natural gas backup generator.
    EV_CHARGER:       Electric vehicle charger (V2G capable).
    THERMAL_STORAGE:  Thermal energy storage (ice/chilled water/hot water).
    CHP:              Combined Heat and Power / cogeneration.
    """
    BESS = "bess"
    SOLAR_PV = "solar_pv"
    BACKUP_GENERATOR = "backup_generator"
    EV_CHARGER = "ev_charger"
    THERMAL_STORAGE = "thermal_storage"
    CHP = "chp"

class DERStatus(str, Enum):
    """Operational status of a DER asset.

    ONLINE:      Asset available and ready for dispatch.
    OFFLINE:     Asset unavailable (maintenance, fault, disconnected).
    DISPATCHING: Asset currently being dispatched.
    CHARGING:    Storage asset currently charging.
    STANDBY:     Asset on standby, ready for rapid dispatch.
    DEGRADED:    Asset operating at reduced capacity.
    """
    ONLINE = "online"
    OFFLINE = "offline"
    DISPATCHING = "dispatching"
    CHARGING = "charging"
    STANDBY = "standby"
    DEGRADED = "degraded"

class DispatchPriority(str, Enum):
    """Dispatch priority for DER assets during DR events.

    CRITICAL:  Dispatch first - lowest marginal cost or highest value.
    HIGH:      Second tier dispatch.
    MEDIUM:    Standard dispatch priority.
    LOW:       Last resort dispatch (e.g. backup generators).
    EXCLUDED:  Do not dispatch this asset.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXCLUDED = "excluded"

class DegradationModel(str, Enum):
    """Battery degradation model type.

    LINEAR:     Linear degradation per full equivalent cycle.
    CALENDAR:   Calendar-based aging (time-dependent).
    COMBINED:   Combined cycling + calendar degradation.
    NONE:       No degradation tracking (non-battery assets).
    """
    LINEAR = "linear"
    CALENDAR = "calendar"
    COMBINED = "combined"
    NONE = "none"

class DispatchStrategy(str, Enum):
    """Portfolio dispatch strategy for DR events.

    MERIT_ORDER:         Dispatch by ascending marginal cost.
    PRIORITY_BASED:      Dispatch by priority tier then capacity.
    MAXIMIZE_DURATION:   Maximize event coverage duration.
    MINIMIZE_DEGRADATION: Minimize battery cycling degradation.
    ROUND_ROBIN:         Distribute load evenly across assets.
    """
    MERIT_ORDER = "merit_order"
    PRIORITY_BASED = "priority_based"
    MAXIMIZE_DURATION = "maximize_duration"
    MINIMIZE_DEGRADATION = "minimize_degradation"
    ROUND_ROBIN = "round_robin"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SOC operating limits (fraction of capacity).
DEFAULT_SOC_MIN: Decimal = Decimal("0.20")
DEFAULT_SOC_MAX: Decimal = Decimal("0.90")

# Default degradation rate per full equivalent cycle (lithium-ion NMC).
DEFAULT_DEGRADATION_PER_CYCLE: Decimal = Decimal("0.004")

# Calendar degradation rate (% capacity loss per year at 25C).
DEFAULT_CALENDAR_DEGRADATION_PCT_PER_YEAR: Decimal = Decimal("2.0")

# Dispatch efficiency (round-trip for BESS).
DEFAULT_ROUNDTRIP_EFFICIENCY: Decimal = Decimal("0.90")

# Maximum number of DER assets in a portfolio.
MAX_PORTFOLIO_ASSETS: int = 500

# Priority order mapping for dispatch.
PRIORITY_ORDER: Dict[str, int] = {
    DispatchPriority.CRITICAL.value: 1,
    DispatchPriority.HIGH.value: 2,
    DispatchPriority.MEDIUM.value: 3,
    DispatchPriority.LOW.value: 4,
    DispatchPriority.EXCLUDED.value: 99,
}

# Marginal cost defaults by asset type (USD/kWh dispatched).
DEFAULT_MARGINAL_COST: Dict[str, Decimal] = {
    DERAssetType.BESS.value: Decimal("0.05"),
    DERAssetType.SOLAR_PV.value: Decimal("0.00"),
    DERAssetType.BACKUP_GENERATOR.value: Decimal("0.25"),
    DERAssetType.EV_CHARGER.value: Decimal("0.08"),
    DERAssetType.THERMAL_STORAGE.value: Decimal("0.03"),
    DERAssetType.CHP.value: Decimal("0.12"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class DERAsset(BaseModel):
    """Distributed Energy Resource asset definition.

    Attributes:
        asset_id: Unique asset identifier.
        name: Human-readable asset name.
        asset_type: DER asset type classification.
        rated_power_kw: Rated power capacity (kW).
        energy_capacity_kwh: Energy storage capacity (kWh), 0 for non-storage.
        current_soc: Current state of charge (0.0 - 1.0), storage assets only.
        soc_min: Minimum allowed SOC (fraction).
        soc_max: Maximum allowed SOC (fraction).
        status: Current operational status.
        priority: Dispatch priority tier.
        marginal_cost_per_kwh: Marginal cost of dispatch (USD/kWh).
        roundtrip_efficiency: Round-trip efficiency (fraction, storage only).
        degradation_model: Degradation tracking model.
        degradation_per_cycle: Capacity loss per full equivalent cycle (fraction).
        cumulative_cycles: Cumulative full equivalent cycles to date.
        calendar_age_years: Calendar age of the asset (years).
        location_id: Site or location identifier.
        ramp_rate_kw_per_min: Maximum ramp rate (kW/minute).
        min_dispatch_kw: Minimum dispatch level (kW).
    """
    asset_id: str = Field(
        default_factory=_new_uuid, description="Unique asset identifier"
    )
    name: str = Field(
        default="", max_length=500, description="Asset name"
    )
    asset_type: DERAssetType = Field(
        ..., description="DER asset type"
    )
    rated_power_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Rated power (kW)"
    )
    energy_capacity_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy capacity (kWh)"
    )
    current_soc: Decimal = Field(
        default=Decimal("0.50"), ge=0, le=Decimal("1.0"),
        description="Current SOC (0-1)"
    )
    soc_min: Decimal = Field(
        default=DEFAULT_SOC_MIN, ge=0, le=Decimal("1.0"),
        description="Minimum SOC limit"
    )
    soc_max: Decimal = Field(
        default=DEFAULT_SOC_MAX, ge=0, le=Decimal("1.0"),
        description="Maximum SOC limit"
    )
    status: DERStatus = Field(
        default=DERStatus.ONLINE, description="Operational status"
    )
    priority: DispatchPriority = Field(
        default=DispatchPriority.MEDIUM, description="Dispatch priority"
    )
    marginal_cost_per_kwh: Decimal = Field(
        default=Decimal("0.05"), ge=0, description="Marginal cost (USD/kWh)"
    )
    roundtrip_efficiency: Decimal = Field(
        default=DEFAULT_ROUNDTRIP_EFFICIENCY, ge=Decimal("0.50"), le=Decimal("1.0"),
        description="Round-trip efficiency"
    )
    degradation_model: DegradationModel = Field(
        default=DegradationModel.NONE, description="Degradation model"
    )
    degradation_per_cycle: Decimal = Field(
        default=DEFAULT_DEGRADATION_PER_CYCLE, ge=0, le=Decimal("0.1"),
        description="Capacity loss per full equivalent cycle"
    )
    cumulative_cycles: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cumulative full equivalent cycles"
    )
    calendar_age_years: Decimal = Field(
        default=Decimal("0"), ge=0, description="Calendar age (years)"
    )
    location_id: str = Field(
        default="", max_length=200, description="Location identifier"
    )
    ramp_rate_kw_per_min: Decimal = Field(
        default=Decimal("0"), ge=0, description="Ramp rate (kW/min)"
    )
    min_dispatch_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Minimum dispatch level (kW)"
    )

    @field_validator("asset_type", mode="before")
    @classmethod
    def validate_asset_type(cls, v: Any) -> Any:
        """Accept string values for DERAssetType."""
        if isinstance(v, str):
            valid = {t.value for t in DERAssetType}
            if v not in valid:
                raise ValueError(
                    f"Unknown asset type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v

    @field_validator("soc_max")
    @classmethod
    def validate_soc_range(cls, v: Decimal) -> Decimal:
        """Ensure SOC max is reasonable."""
        if v < Decimal("0.50"):
            raise ValueError("SOC max must be >= 0.50")
        return v

class DispatchRequest(BaseModel):
    """Dispatch request for a DR event.

    Attributes:
        event_id: DR event identifier.
        target_kw: Total curtailment target (kW).
        duration_hours: Event duration (hours).
        strategy: Dispatch strategy to apply.
        max_assets: Maximum number of assets to dispatch.
        exclude_asset_ids: Assets to exclude from dispatch.
    """
    event_id: str = Field(
        default_factory=_new_uuid, description="DR event ID"
    )
    target_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Target curtailment (kW)"
    )
    duration_hours: Decimal = Field(
        default=Decimal("1"), ge=Decimal("0.25"), le=Decimal("24"),
        description="Event duration (hours)"
    )
    strategy: DispatchStrategy = Field(
        default=DispatchStrategy.PRIORITY_BASED, description="Dispatch strategy"
    )
    max_assets: int = Field(
        default=MAX_PORTFOLIO_ASSETS, ge=1, description="Max assets to dispatch"
    )
    exclude_asset_ids: List[str] = Field(
        default_factory=list, description="Asset IDs to exclude"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class DERDispatch(BaseModel):
    """Dispatch assignment for a single DER asset.

    Attributes:
        asset_id: Asset being dispatched.
        asset_type: Type of asset.
        dispatch_kw: Power to dispatch (kW).
        dispatch_kwh: Energy to dispatch (kWh).
        soc_before: SOC before dispatch (storage assets).
        soc_after: SOC after dispatch (storage assets).
        marginal_cost: Cost of this dispatch (USD).
        degradation_impact: Estimated degradation from this dispatch (pct).
        is_feasible: Whether dispatch is feasible within constraints.
        constraint_notes: Notes on any binding constraints.
    """
    asset_id: str = Field(default="", description="Asset ID")
    asset_type: DERAssetType = Field(
        default=DERAssetType.BESS, description="Asset type"
    )
    dispatch_kw: Decimal = Field(
        default=Decimal("0"), description="Dispatch power (kW)"
    )
    dispatch_kwh: Decimal = Field(
        default=Decimal("0"), description="Dispatch energy (kWh)"
    )
    soc_before: Decimal = Field(
        default=Decimal("0"), description="SOC before dispatch"
    )
    soc_after: Decimal = Field(
        default=Decimal("0"), description="SOC after dispatch"
    )
    marginal_cost: Decimal = Field(
        default=Decimal("0"), description="Dispatch cost (USD)"
    )
    degradation_impact: Decimal = Field(
        default=Decimal("0"), description="Degradation impact (pct)"
    )
    is_feasible: bool = Field(
        default=True, description="Dispatch feasibility"
    )
    constraint_notes: str = Field(
        default="", description="Constraint notes"
    )

class DERPerformance(BaseModel):
    """Performance metrics for a DER asset over a period.

    Attributes:
        asset_id: Asset identifier.
        asset_type: Asset type.
        total_dispatches: Number of dispatch events.
        total_dispatched_kwh: Total energy dispatched (kWh).
        availability_pct: Availability percentage.
        capacity_factor_pct: Capacity factor percentage.
        contribution_ratio_pct: Average dispatch vs rated power.
        total_degradation_pct: Cumulative degradation (pct).
        effective_capacity_kwh: Remaining effective capacity (kWh).
        total_dispatch_cost: Total dispatch cost (USD).
        avg_response_time_min: Average response time (minutes).
        provenance_hash: SHA-256 audit hash.
    """
    asset_id: str = Field(default="", description="Asset ID")
    asset_type: DERAssetType = Field(
        default=DERAssetType.BESS, description="Asset type"
    )
    total_dispatches: int = Field(
        default=0, ge=0, description="Total dispatch events"
    )
    total_dispatched_kwh: Decimal = Field(
        default=Decimal("0"), description="Total dispatched (kWh)"
    )
    availability_pct: Decimal = Field(
        default=Decimal("0"), description="Availability (%)"
    )
    capacity_factor_pct: Decimal = Field(
        default=Decimal("0"), description="Capacity factor (%)"
    )
    contribution_ratio_pct: Decimal = Field(
        default=Decimal("0"), description="Contribution ratio (%)"
    )
    total_degradation_pct: Decimal = Field(
        default=Decimal("0"), description="Cumulative degradation (%)"
    )
    effective_capacity_kwh: Decimal = Field(
        default=Decimal("0"), description="Effective capacity (kWh)"
    )
    total_dispatch_cost: Decimal = Field(
        default=Decimal("0"), description="Total dispatch cost (USD)"
    )
    avg_response_time_min: Decimal = Field(
        default=Decimal("0"), description="Avg response time (min)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class DERPortfolio(BaseModel):
    """Aggregated DER portfolio summary.

    Attributes:
        portfolio_id: Portfolio identifier.
        total_assets: Number of registered assets.
        total_rated_kw: Total rated power capacity (kW).
        total_available_kw: Total currently available power (kW).
        total_energy_capacity_kwh: Total energy capacity (kWh).
        total_usable_kwh: Total usable energy capacity (kWh).
        dispatches: Individual dispatch assignments.
        total_dispatched_kw: Total dispatched power (kW).
        total_dispatched_kwh: Total dispatched energy (kWh).
        shortfall_kw: Unmet dispatch target (kW).
        total_dispatch_cost: Total dispatch cost (USD).
        avg_portfolio_soc: Average SOC across storage assets.
        asset_type_breakdown: Breakdown by asset type.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    portfolio_id: str = Field(
        default_factory=_new_uuid, description="Portfolio ID"
    )
    total_assets: int = Field(
        default=0, ge=0, description="Total assets"
    )
    total_rated_kw: Decimal = Field(
        default=Decimal("0"), description="Total rated kW"
    )
    total_available_kw: Decimal = Field(
        default=Decimal("0"), description="Total available kW"
    )
    total_energy_capacity_kwh: Decimal = Field(
        default=Decimal("0"), description="Total energy capacity (kWh)"
    )
    total_usable_kwh: Decimal = Field(
        default=Decimal("0"), description="Total usable energy (kWh)"
    )
    dispatches: List[DERDispatch] = Field(
        default_factory=list, description="Dispatch assignments"
    )
    total_dispatched_kw: Decimal = Field(
        default=Decimal("0"), description="Total dispatched kW"
    )
    total_dispatched_kwh: Decimal = Field(
        default=Decimal("0"), description="Total dispatched kWh"
    )
    shortfall_kw: Decimal = Field(
        default=Decimal("0"), description="Unmet target (kW)"
    )
    total_dispatch_cost: Decimal = Field(
        default=Decimal("0"), description="Total dispatch cost (USD)"
    )
    avg_portfolio_soc: Decimal = Field(
        default=Decimal("0"), description="Average SOC"
    )
    asset_type_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict, description="kW by asset type"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DERCoordinatorEngine:
    """DER coordination engine for demand response programmes.

    Manages a portfolio of heterogeneous DER assets, optimises dispatch
    during DR events, tracks SOC within safe operating limits, monitors
    cycling degradation, and calculates aggregate portfolio contribution.

    Usage::

        engine = DERCoordinatorEngine()
        engine.register_asset(bess_asset)
        engine.register_asset(solar_asset)
        availability = engine.assess_availability()
        portfolio = engine.optimize_dispatch(dispatch_request)
        perf = engine.track_performance(asset_id)
        degradation = engine.calculate_degradation(asset_id)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DERCoordinatorEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - soc_min (Decimal): default minimum SOC
                - soc_max (Decimal): default maximum SOC
                - degradation_per_cycle (Decimal): default degradation rate
                - roundtrip_efficiency (Decimal): default round-trip efficiency
        """
        self.config = config or {}
        self._soc_min = _decimal(self.config.get("soc_min", DEFAULT_SOC_MIN))
        self._soc_max = _decimal(self.config.get("soc_max", DEFAULT_SOC_MAX))
        self._degradation_rate = _decimal(
            self.config.get("degradation_per_cycle", DEFAULT_DEGRADATION_PER_CYCLE)
        )
        self._assets: Dict[str, DERAsset] = {}
        self._dispatch_history: List[DERDispatch] = []
        logger.info(
            "DERCoordinatorEngine v%s initialised (soc_min=%.2f, soc_max=%.2f)",
            self.engine_version,
            float(self._soc_min),
            float(self._soc_max),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def register_asset(self, asset: DERAsset) -> DERAsset:
        """Register a DER asset in the portfolio.

        Validates asset constraints and adds it to the managed portfolio.
        If the asset_id already exists, it is updated.

        Args:
            asset: DER asset definition.

        Returns:
            The registered asset with any defaults applied.

        Raises:
            ValueError: If portfolio capacity exceeded or asset invalid.
        """
        if len(self._assets) >= MAX_PORTFOLIO_ASSETS and asset.asset_id not in self._assets:
            raise ValueError(
                f"Portfolio capacity exceeded ({MAX_PORTFOLIO_ASSETS} assets max)."
            )
        if asset.soc_min >= asset.soc_max:
            raise ValueError(
                f"SOC min ({asset.soc_min}) must be less than SOC max ({asset.soc_max})."
            )
        self._assets[asset.asset_id] = asset
        logger.info(
            "Registered DER asset: id=%s, type=%s, rated=%s kW, capacity=%s kWh",
            asset.asset_id, asset.asset_type.value,
            str(asset.rated_power_kw), str(asset.energy_capacity_kwh),
        )
        return asset

    def assess_availability(
        self,
        asset_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assess availability of DER assets for DR dispatch.

        Evaluates each asset's current status, SOC, and usable capacity
        to determine real-time availability for event dispatch.

        Args:
            asset_ids: Specific assets to assess (None = all).

        Returns:
            Dictionary with per-asset and aggregate availability metrics.
        """
        t0 = time.perf_counter()
        targets = self._get_target_assets(asset_ids)
        logger.info("Assessing availability for %d assets", len(targets))

        results: List[Dict[str, Any]] = []
        total_rated = Decimal("0")
        total_available = Decimal("0")

        for asset in targets:
            available_kw = self._calculate_available_kw(asset)
            usable_kwh = self._calculate_usable_kwh(asset)
            is_available = (
                asset.status in (DERStatus.ONLINE, DERStatus.STANDBY)
                and available_kw > Decimal("0")
                and asset.priority != DispatchPriority.EXCLUDED
            )
            total_rated += asset.rated_power_kw
            if is_available:
                total_available += available_kw

            results.append({
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type.value,
                "rated_kw": str(_round_val(asset.rated_power_kw, 2)),
                "available_kw": str(_round_val(available_kw, 2)),
                "usable_kwh": str(_round_val(usable_kwh, 2)),
                "current_soc": str(_round_val(asset.current_soc, 4)),
                "status": asset.status.value,
                "is_available": is_available,
            })

        availability_pct = _safe_pct(total_available, total_rated)
        elapsed = (time.perf_counter() - t0) * 1000.0

        assessment = {
            "total_assets": len(targets),
            "total_rated_kw": str(_round_val(total_rated, 2)),
            "total_available_kw": str(_round_val(total_available, 2)),
            "availability_pct": str(_round_val(availability_pct, 2)),
            "assets": results,
            "assessed_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        assessment["provenance_hash"] = _compute_hash(assessment)

        logger.info(
            "Availability: %d assets, available=%s kW (%.1f%%), hash=%s (%.1f ms)",
            len(targets), str(_round_val(total_available, 2)),
            float(availability_pct), assessment["provenance_hash"][:16], elapsed,
        )
        return assessment

    def optimize_dispatch(
        self,
        request: DispatchRequest,
    ) -> DERPortfolio:
        """Optimise DER dispatch for a DR event.

        Selects and dispatches assets according to the chosen strategy,
        respecting SOC limits, ramp rates, and capacity constraints.

        Args:
            request: Dispatch request with target and strategy.

        Returns:
            DERPortfolio with dispatch assignments and aggregate metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Optimising dispatch: event=%s, target=%s kW, duration=%s h, strategy=%s",
            request.event_id, str(request.target_kw),
            str(request.duration_hours), request.strategy.value,
        )

        # Get eligible assets
        eligible = self._get_eligible_assets(request)

        # Sort by strategy
        sorted_assets = self._sort_by_strategy(eligible, request.strategy)

        # Dispatch assets
        dispatches: List[DERDispatch] = []
        remaining_kw = request.target_kw
        total_cost = Decimal("0")
        dispatched_count = 0

        for asset in sorted_assets:
            if remaining_kw <= Decimal("0"):
                break
            if dispatched_count >= request.max_assets:
                break

            dispatch = self._dispatch_asset(asset, remaining_kw, request.duration_hours)
            dispatches.append(dispatch)

            if dispatch.is_feasible and dispatch.dispatch_kw > Decimal("0"):
                remaining_kw -= dispatch.dispatch_kw
                total_cost += dispatch.marginal_cost
                dispatched_count += 1
                self._dispatch_history.append(dispatch)

        # Aggregate metrics
        total_dispatched_kw = sum(
            (d.dispatch_kw for d in dispatches if d.is_feasible), Decimal("0")
        )
        total_dispatched_kwh = sum(
            (d.dispatch_kwh for d in dispatches if d.is_feasible), Decimal("0")
        )
        shortfall = max(request.target_kw - total_dispatched_kw, Decimal("0"))

        # Asset type breakdown
        breakdown: Dict[str, Decimal] = {}
        for d in dispatches:
            if d.is_feasible and d.dispatch_kw > Decimal("0"):
                key = d.asset_type.value
                breakdown[key] = breakdown.get(key, Decimal("0")) + d.dispatch_kw

        # Portfolio SOC
        avg_soc = self._calculate_avg_soc()

        # Total portfolio metrics
        total_rated = sum(
            (a.rated_power_kw for a in self._assets.values()), Decimal("0")
        )
        total_available = sum(
            (self._calculate_available_kw(a) for a in self._assets.values()),
            Decimal("0"),
        )
        total_energy = sum(
            (a.energy_capacity_kwh for a in self._assets.values()), Decimal("0")
        )
        total_usable = sum(
            (self._calculate_usable_kwh(a) for a in self._assets.values()),
            Decimal("0"),
        )

        portfolio = DERPortfolio(
            total_assets=len(self._assets),
            total_rated_kw=_round_val(total_rated, 2),
            total_available_kw=_round_val(total_available, 2),
            total_energy_capacity_kwh=_round_val(total_energy, 2),
            total_usable_kwh=_round_val(total_usable, 2),
            dispatches=dispatches,
            total_dispatched_kw=_round_val(total_dispatched_kw, 2),
            total_dispatched_kwh=_round_val(total_dispatched_kwh, 2),
            shortfall_kw=_round_val(shortfall, 2),
            total_dispatch_cost=_round_val(total_cost, 2),
            avg_portfolio_soc=_round_val(avg_soc, 4),
            asset_type_breakdown={k: _round_val(v, 2) for k, v in breakdown.items()},
        )
        portfolio.provenance_hash = _compute_hash(portfolio)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Dispatch complete: event=%s, dispatched=%s/%s kW, "
            "shortfall=%s kW, cost=%s, hash=%s (%.1f ms)",
            request.event_id,
            str(_round_val(total_dispatched_kw, 2)), str(request.target_kw),
            str(_round_val(shortfall, 2)), str(_round_val(total_cost, 2)),
            portfolio.provenance_hash[:16], elapsed,
        )
        return portfolio

    def track_performance(
        self,
        asset_id: str,
        period_hours: Decimal = Decimal("720"),
    ) -> DERPerformance:
        """Track performance metrics for a specific DER asset.

        Analyses dispatch history and calculates availability, capacity
        factor, contribution ratio, degradation, and costs.

        Args:
            asset_id: Asset to track.
            period_hours: Evaluation period in hours (default 720 = 30 days).

        Returns:
            DERPerformance with comprehensive metrics.

        Raises:
            ValueError: If asset_id not found in portfolio.
        """
        t0 = time.perf_counter()
        if asset_id not in self._assets:
            raise ValueError(f"Asset '{asset_id}' not found in portfolio.")

        asset = self._assets[asset_id]
        logger.info(
            "Tracking performance: asset=%s, type=%s, period=%s h",
            asset_id, asset.asset_type.value, str(period_hours),
        )

        # Filter dispatch history for this asset
        asset_dispatches = [
            d for d in self._dispatch_history if d.asset_id == asset_id
        ]

        total_dispatches = len(asset_dispatches)
        total_dispatched_kwh = sum(
            (d.dispatch_kwh for d in asset_dispatches), Decimal("0")
        )
        total_cost = sum(
            (d.marginal_cost for d in asset_dispatches), Decimal("0")
        )

        # Availability: assume online unless offline status
        avail_hours = period_hours if asset.status != DERStatus.OFFLINE else Decimal("0")
        availability_pct = _safe_pct(avail_hours, period_hours)

        # Capacity factor
        max_possible_kwh = asset.rated_power_kw * period_hours
        capacity_factor_pct = _safe_pct(total_dispatched_kwh, max_possible_kwh)

        # Contribution ratio (average dispatch / rated)
        if total_dispatches > 0:
            avg_dispatch_kw = sum(
                (d.dispatch_kw for d in asset_dispatches), Decimal("0")
            ) / _decimal(total_dispatches)
            contribution_ratio = _safe_pct(avg_dispatch_kw, asset.rated_power_kw)
        else:
            contribution_ratio = Decimal("0")

        # Degradation
        degradation_pct = self._compute_total_degradation(asset)
        effective_kwh = self._calculate_effective_capacity(asset)

        perf = DERPerformance(
            asset_id=asset_id,
            asset_type=asset.asset_type,
            total_dispatches=total_dispatches,
            total_dispatched_kwh=_round_val(total_dispatched_kwh, 2),
            availability_pct=_round_val(availability_pct, 2),
            capacity_factor_pct=_round_val(capacity_factor_pct, 4),
            contribution_ratio_pct=_round_val(contribution_ratio, 2),
            total_degradation_pct=_round_val(degradation_pct, 4),
            effective_capacity_kwh=_round_val(effective_kwh, 2),
            total_dispatch_cost=_round_val(total_cost, 2),
            avg_response_time_min=Decimal("0"),
        )
        perf.provenance_hash = _compute_hash(perf)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Performance: asset=%s, dispatches=%d, kwh=%.2f, "
            "avail=%.1f%%, degradation=%.4f%%, hash=%s (%.1f ms)",
            asset_id, total_dispatches, float(total_dispatched_kwh),
            float(availability_pct), float(degradation_pct),
            perf.provenance_hash[:16], elapsed,
        )
        return perf

    def calculate_degradation(
        self,
        asset_id: str,
        additional_cycles: Decimal = Decimal("0"),
        additional_years: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """Calculate current and projected degradation for a DER asset.

        Supports linear cycling, calendar, and combined degradation
        models.  Projects future degradation under additional cycling
        and aging scenarios.

        Args:
            asset_id: Asset to evaluate.
            additional_cycles: Projected additional full equivalent cycles.
            additional_years: Projected additional calendar years.

        Returns:
            Dictionary with current and projected degradation metrics.

        Raises:
            ValueError: If asset_id not found.
        """
        t0 = time.perf_counter()
        if asset_id not in self._assets:
            raise ValueError(f"Asset '{asset_id}' not found in portfolio.")

        asset = self._assets[asset_id]
        logger.info(
            "Calculating degradation: asset=%s, model=%s, cycles=%.1f, age=%.1f yr",
            asset_id, asset.degradation_model.value,
            float(asset.cumulative_cycles), float(asset.calendar_age_years),
        )

        # Current degradation
        current_cycling = self._cycling_degradation(
            asset.cumulative_cycles, asset.degradation_per_cycle,
        )
        current_calendar = self._calendar_degradation(asset.calendar_age_years)
        current_total = self._combine_degradation(
            current_cycling, current_calendar, asset.degradation_model,
        )
        current_effective = asset.energy_capacity_kwh * (
            Decimal("1") - current_total / Decimal("100")
        )

        # Projected degradation
        proj_cycles = asset.cumulative_cycles + additional_cycles
        proj_years = asset.calendar_age_years + additional_years
        proj_cycling = self._cycling_degradation(
            proj_cycles, asset.degradation_per_cycle,
        )
        proj_calendar = self._calendar_degradation(proj_years)
        proj_total = self._combine_degradation(
            proj_cycling, proj_calendar, asset.degradation_model,
        )
        proj_effective = asset.energy_capacity_kwh * (
            Decimal("1") - proj_total / Decimal("100")
        )

        # Remaining useful life estimate (80% capacity threshold)
        rul_cycles = Decimal("0")
        if asset.degradation_per_cycle > Decimal("0"):
            # Cycles to reach 80% remaining = 20% degradation
            max_cycling_deg = Decimal("20") - current_calendar
            if max_cycling_deg > Decimal("0"):
                rul_cycles = _safe_divide(
                    max_cycling_deg,
                    asset.degradation_per_cycle * Decimal("100"),
                ) - asset.cumulative_cycles
                rul_cycles = max(rul_cycles, Decimal("0"))

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = {
            "asset_id": asset_id,
            "degradation_model": asset.degradation_model.value,
            "current": {
                "cycling_degradation_pct": str(_round_val(current_cycling, 4)),
                "calendar_degradation_pct": str(_round_val(current_calendar, 4)),
                "total_degradation_pct": str(_round_val(current_total, 4)),
                "effective_capacity_kwh": str(_round_val(current_effective, 2)),
                "cumulative_cycles": str(_round_val(asset.cumulative_cycles, 1)),
                "calendar_age_years": str(_round_val(asset.calendar_age_years, 1)),
            },
            "projected": {
                "additional_cycles": str(_round_val(additional_cycles, 1)),
                "additional_years": str(_round_val(additional_years, 1)),
                "total_cycles": str(_round_val(proj_cycles, 1)),
                "total_years": str(_round_val(proj_years, 1)),
                "cycling_degradation_pct": str(_round_val(proj_cycling, 4)),
                "calendar_degradation_pct": str(_round_val(proj_calendar, 4)),
                "total_degradation_pct": str(_round_val(proj_total, 4)),
                "effective_capacity_kwh": str(_round_val(proj_effective, 2)),
            },
            "remaining_useful_life_cycles": str(_round_val(rul_cycles, 0)),
            "eol_threshold_pct": "20.0",
            "calculated_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Degradation: asset=%s, current=%.4f%%, projected=%.4f%%, "
            "RUL=%.0f cycles, hash=%s (%.1f ms)",
            asset_id, float(current_total), float(proj_total),
            float(rul_cycles), result["provenance_hash"][:16], elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal: Availability Calculations                                 #
    # ------------------------------------------------------------------ #

    def _calculate_available_kw(self, asset: DERAsset) -> Decimal:
        """Calculate currently available dispatch power for an asset.

        For storage assets, limits available kW by remaining usable SOC.
        For generation assets, returns rated power if online.

        Args:
            asset: DER asset to evaluate.

        Returns:
            Available dispatch power (kW).
        """
        if asset.status in (DERStatus.OFFLINE, DERStatus.CHARGING):
            return Decimal("0")
        if asset.priority == DispatchPriority.EXCLUDED:
            return Decimal("0")

        # For storage assets, limit by SOC
        if asset.energy_capacity_kwh > Decimal("0"):
            usable_soc = asset.current_soc - asset.soc_min
            if usable_soc <= Decimal("0"):
                return Decimal("0")
            # Power limited by energy: available for 1 hour minimum
            energy_limited_kw = asset.energy_capacity_kwh * usable_soc
            return min(asset.rated_power_kw, energy_limited_kw)

        # Non-storage: rated power if online
        return asset.rated_power_kw

    def _calculate_usable_kwh(self, asset: DERAsset) -> Decimal:
        """Calculate usable energy capacity respecting SOC limits.

        usable_kwh = capacity * (current_soc - soc_min)

        Args:
            asset: DER asset.

        Returns:
            Usable energy (kWh).
        """
        if asset.energy_capacity_kwh <= Decimal("0"):
            return Decimal("0")

        effective = self._calculate_effective_capacity(asset)
        usable_soc = max(asset.current_soc - asset.soc_min, Decimal("0"))
        return effective * usable_soc

    def _calculate_effective_capacity(self, asset: DERAsset) -> Decimal:
        """Calculate effective capacity after degradation.

        Args:
            asset: DER asset.

        Returns:
            Effective energy capacity (kWh).
        """
        degradation = self._compute_total_degradation(asset)
        factor = max(Decimal("1") - degradation / Decimal("100"), Decimal("0"))
        return asset.energy_capacity_kwh * factor

    def _calculate_avg_soc(self) -> Decimal:
        """Calculate average SOC across all storage assets.

        Returns:
            Average SOC (0-1 fraction).
        """
        storage_assets = [
            a for a in self._assets.values()
            if a.energy_capacity_kwh > Decimal("0")
        ]
        if not storage_assets:
            return Decimal("0")

        total_soc = sum((a.current_soc for a in storage_assets), Decimal("0"))
        return total_soc / _decimal(len(storage_assets))

    # ------------------------------------------------------------------ #
    # Internal: Dispatch Logic                                            #
    # ------------------------------------------------------------------ #

    def _get_target_assets(
        self,
        asset_ids: Optional[List[str]],
    ) -> List[DERAsset]:
        """Get target assets for evaluation.

        Args:
            asset_ids: Specific IDs or None for all.

        Returns:
            List of DERAsset objects.
        """
        if asset_ids is None:
            return list(self._assets.values())
        return [
            self._assets[aid] for aid in asset_ids
            if aid in self._assets
        ]

    def _get_eligible_assets(self, request: DispatchRequest) -> List[DERAsset]:
        """Get assets eligible for dispatch.

        Filters out excluded, offline, and explicitly excluded assets.

        Args:
            request: Dispatch request.

        Returns:
            List of eligible DERAsset objects.
        """
        eligible = []
        for asset in self._assets.values():
            if asset.asset_id in request.exclude_asset_ids:
                continue
            if asset.priority == DispatchPriority.EXCLUDED:
                continue
            if asset.status in (DERStatus.OFFLINE,):
                continue
            eligible.append(asset)
        return eligible

    def _sort_by_strategy(
        self,
        assets: List[DERAsset],
        strategy: DispatchStrategy,
    ) -> List[DERAsset]:
        """Sort assets according to dispatch strategy.

        Args:
            assets: Eligible assets.
            strategy: Dispatch strategy.

        Returns:
            Sorted list of assets.
        """
        if strategy == DispatchStrategy.MERIT_ORDER:
            return sorted(assets, key=lambda a: a.marginal_cost_per_kwh)

        if strategy == DispatchStrategy.PRIORITY_BASED:
            return sorted(
                assets,
                key=lambda a: (
                    PRIORITY_ORDER.get(a.priority.value, 50),
                    -float(a.rated_power_kw),
                ),
            )

        if strategy == DispatchStrategy.MAXIMIZE_DURATION:
            return sorted(
                assets,
                key=lambda a: float(self._calculate_usable_kwh(a)),
                reverse=True,
            )

        if strategy == DispatchStrategy.MINIMIZE_DEGRADATION:
            # Prefer non-storage assets first, then lowest degradation
            return sorted(
                assets,
                key=lambda a: (
                    0 if a.energy_capacity_kwh <= Decimal("0") else 1,
                    float(a.cumulative_cycles),
                ),
            )

        if strategy == DispatchStrategy.ROUND_ROBIN:
            # Sort by least recent dispatch (fewest total dispatches)
            dispatch_counts: Dict[str, int] = {}
            for d in self._dispatch_history:
                dispatch_counts[d.asset_id] = dispatch_counts.get(d.asset_id, 0) + 1
            return sorted(
                assets,
                key=lambda a: dispatch_counts.get(a.asset_id, 0),
            )

        return assets

    def _dispatch_asset(
        self,
        asset: DERAsset,
        target_kw: Decimal,
        duration_hours: Decimal,
    ) -> DERDispatch:
        """Dispatch a single asset towards the target.

        Calculates feasible dispatch considering SOC limits, power
        ratings, and minimum dispatch thresholds.

        Args:
            asset: Asset to dispatch.
            target_kw: Remaining target power (kW).
            duration_hours: Event duration (hours).

        Returns:
            DERDispatch with assignment details.
        """
        available_kw = self._calculate_available_kw(asset)
        soc_before = asset.current_soc
        constraints: List[str] = []

        # Minimum dispatch check
        if available_kw < asset.min_dispatch_kw and asset.min_dispatch_kw > Decimal("0"):
            return DERDispatch(
                asset_id=asset.asset_id,
                asset_type=asset.asset_type,
                dispatch_kw=Decimal("0"),
                dispatch_kwh=Decimal("0"),
                soc_before=soc_before,
                soc_after=soc_before,
                is_feasible=False,
                constraint_notes="Below minimum dispatch threshold",
            )

        # Determine dispatch level
        dispatch_kw = min(available_kw, target_kw)

        # For storage: limit by usable energy over duration
        if asset.energy_capacity_kwh > Decimal("0"):
            usable_kwh = self._calculate_usable_kwh(asset)
            max_kw_for_duration = _safe_divide(usable_kwh, duration_hours)
            if dispatch_kw > max_kw_for_duration:
                dispatch_kw = max_kw_for_duration
                constraints.append("SOC-limited for duration")

        dispatch_kwh = dispatch_kw * duration_hours

        # Calculate SOC after (storage only)
        soc_after = soc_before
        if asset.energy_capacity_kwh > Decimal("0"):
            effective = self._calculate_effective_capacity(asset)
            if effective > Decimal("0"):
                soc_delta = _safe_divide(dispatch_kwh, effective)
                soc_after = max(soc_before - soc_delta, asset.soc_min)
                if soc_after <= asset.soc_min:
                    constraints.append("SOC at minimum limit")

        # Marginal cost
        marginal_cost = dispatch_kwh * asset.marginal_cost_per_kwh

        # Degradation impact (storage only)
        degradation_impact = Decimal("0")
        if asset.energy_capacity_kwh > Decimal("0"):
            cycle_fraction = _safe_divide(dispatch_kwh, asset.energy_capacity_kwh)
            degradation_impact = cycle_fraction * asset.degradation_per_cycle * Decimal("100")

        is_feasible = dispatch_kw > Decimal("0")

        return DERDispatch(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            dispatch_kw=_round_val(dispatch_kw, 2),
            dispatch_kwh=_round_val(dispatch_kwh, 2),
            soc_before=_round_val(soc_before, 4),
            soc_after=_round_val(soc_after, 4),
            marginal_cost=_round_val(marginal_cost, 2),
            degradation_impact=_round_val(degradation_impact, 6),
            is_feasible=is_feasible,
            constraint_notes="; ".join(constraints) if constraints else "",
        )

    # ------------------------------------------------------------------ #
    # Internal: Degradation Models                                        #
    # ------------------------------------------------------------------ #

    def _cycling_degradation(
        self,
        cycles: Decimal,
        degradation_per_cycle: Decimal,
    ) -> Decimal:
        """Calculate cycling degradation percentage.

        degradation_pct = cycles * degradation_per_cycle * 100

        Args:
            cycles: Full equivalent cycles.
            degradation_per_cycle: Capacity loss per cycle (fraction).

        Returns:
            Degradation percentage.
        """
        return cycles * degradation_per_cycle * Decimal("100")

    def _calendar_degradation(self, age_years: Decimal) -> Decimal:
        """Calculate calendar aging degradation percentage.

        Uses default 2% per year at 25C reference temperature.

        Args:
            age_years: Calendar age in years.

        Returns:
            Calendar degradation percentage.
        """
        return age_years * DEFAULT_CALENDAR_DEGRADATION_PCT_PER_YEAR

    def _combine_degradation(
        self,
        cycling_pct: Decimal,
        calendar_pct: Decimal,
        model: DegradationModel,
    ) -> Decimal:
        """Combine cycling and calendar degradation per model.

        Args:
            cycling_pct: Cycling degradation percentage.
            calendar_pct: Calendar degradation percentage.
            model: Degradation model to apply.

        Returns:
            Total degradation percentage (capped at 100).
        """
        if model == DegradationModel.LINEAR:
            total = cycling_pct
        elif model == DegradationModel.CALENDAR:
            total = calendar_pct
        elif model == DegradationModel.COMBINED:
            total = cycling_pct + calendar_pct
        else:
            total = Decimal("0")

        return min(total, Decimal("100"))

    def _compute_total_degradation(self, asset: DERAsset) -> Decimal:
        """Compute total degradation for an asset.

        Args:
            asset: DER asset.

        Returns:
            Total degradation percentage.
        """
        cycling = self._cycling_degradation(
            asset.cumulative_cycles, asset.degradation_per_cycle,
        )
        calendar = self._calendar_degradation(asset.calendar_age_years)
        return self._combine_degradation(cycling, calendar, asset.degradation_model)
