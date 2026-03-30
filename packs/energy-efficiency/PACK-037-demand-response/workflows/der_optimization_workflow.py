# -*- coding: utf-8 -*-
"""
DER Optimization Workflow
===================================

3-phase workflow for optimizing distributed energy resource (DER)
dispatch during demand response events within PACK-037 Demand Response Pack.

Phases:
    1. AssetInventory        -- Catalog on-site DER assets (BESS, solar, gen, EV)
    2. DispatchStrategy      -- Optimize DER dispatch sequence and schedule
    3. CoordinatedResponse   -- Coordinate DER + load curtailment for max response

The workflow follows GreenLang zero-hallucination principles: DER
dispatch optimization uses deterministic priority-based scheduling
with published asset performance curves. SHA-256 provenance hashes
guarantee auditability.

Regulatory references:
    - FERC Order 2222 (DER aggregation in wholesale markets)
    - IEEE 1547-2018 (DER interconnection)
    - ISO/RTO behind-the-meter DER participation rules

Schedule: event-triggered / on-demand
Estimated duration: 10 minutes

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class DERType(str, Enum):
    """Type of distributed energy resource."""

    BATTERY = "battery"
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    BACKUP_GENERATOR = "backup_generator"
    EV_FLEET = "ev_fleet"
    CHP = "chp"
    FUEL_CELL = "fuel_cell"
    THERMAL_STORAGE = "thermal_storage"

class DispatchMode(str, Enum):
    """DER dispatch mode during DR event."""

    DISCHARGE = "discharge"
    EXPORT = "export"
    ISLAND = "island"
    CURTAIL_EXPORT = "curtail_export"
    V2G = "v2g"
    LOAD_SHIFT = "load_shift"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# DER asset performance parameters
DER_PERFORMANCE_SPECS: Dict[str, Dict[str, Any]] = {
    "battery": {
        "description": "Battery Energy Storage System (BESS)",
        "round_trip_efficiency": Decimal("0.90"),
        "max_discharge_rate_pct": Decimal("100"),
        "ramp_time_min": 1,
        "dispatch_priority": 1,
        "dispatch_mode": "discharge",
        "min_soc_pct": Decimal("10"),
        "degradation_per_cycle_pct": Decimal("0.01"),
    },
    "solar_pv": {
        "description": "Solar Photovoltaic System",
        "capacity_factor_summer": Decimal("0.25"),
        "capacity_factor_winter": Decimal("0.10"),
        "ramp_time_min": 0,
        "dispatch_priority": 2,
        "dispatch_mode": "export",
        "intermittent": True,
        "curtailable": True,
    },
    "backup_generator": {
        "description": "Diesel/Gas Backup Generator",
        "efficiency": Decimal("0.35"),
        "ramp_time_min": 5,
        "dispatch_priority": 4,
        "dispatch_mode": "island",
        "emissions_kg_co2_per_kwh": Decimal("0.70"),
        "max_runtime_hours": 8,
        "fuel_cost_per_kwh": Decimal("0.18"),
    },
    "ev_fleet": {
        "description": "Electric Vehicle Fleet (V2G capable)",
        "round_trip_efficiency": Decimal("0.85"),
        "max_discharge_rate_per_vehicle_kw": Decimal("7.0"),
        "ramp_time_min": 2,
        "dispatch_priority": 3,
        "dispatch_mode": "v2g",
        "min_soc_pct": Decimal("20"),
        "availability_pct": Decimal("60"),
    },
    "thermal_storage": {
        "description": "Ice/Chilled Water Thermal Storage",
        "storage_efficiency": Decimal("0.95"),
        "discharge_duration_hours": Decimal("6"),
        "ramp_time_min": 5,
        "dispatch_priority": 2,
        "dispatch_mode": "load_shift",
    },
    "chp": {
        "description": "Combined Heat and Power",
        "electrical_efficiency": Decimal("0.35"),
        "thermal_efficiency": Decimal("0.45"),
        "ramp_time_min": 15,
        "dispatch_priority": 5,
        "dispatch_mode": "island",
        "emissions_kg_co2_per_kwh": Decimal("0.45"),
    },
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class DERAsset(BaseModel):
    """A distributed energy resource asset."""

    asset_id: str = Field(default_factory=lambda: f"der-{uuid.uuid4().hex[:8]}")
    der_type: str = Field(default="battery", description="DER asset type")
    name: str = Field(default="", description="Asset display name")
    capacity_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Rated power kW")
    energy_kwh: Decimal = Field(default=Decimal("0"), ge=0, description="Energy capacity kWh")
    current_soc_pct: Decimal = Field(default=Decimal("50"), ge=0, le=100)
    available: bool = Field(default=True, description="Asset availability")
    dispatch_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Dispatched power kW")
    dispatch_mode: str = Field(default="", description="Active dispatch mode")
    dispatch_duration_hours: Decimal = Field(default=Decimal("0"), ge=0)

class DERDispatchPlan(BaseModel):
    """Dispatch plan for a single DER asset."""

    asset_id: str = Field(default="", description="Asset identifier")
    der_type: str = Field(default="", description="DER type")
    dispatch_mode: str = Field(default="", description="Dispatch mode")
    dispatch_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Dispatched kW")
    dispatch_duration_hours: Decimal = Field(default=Decimal("0"), ge=0)
    dispatch_energy_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    priority: int = Field(default=0, ge=0, description="Dispatch priority order")
    emissions_kg_co2: Decimal = Field(default=Decimal("0"), ge=0)

class DEROptimizationInput(BaseModel):
    """Input data model for DEROptimizationWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    event_id: str = Field(default="", description="DR event identifier")
    target_response_kw: Decimal = Field(..., gt=0, description="Target DR response kW")
    event_duration_hours: Decimal = Field(default=Decimal("4"), gt=0)
    load_curtailment_kw: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="kW already achieved via load curtailment",
    )
    der_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of DER asset data dicts",
    )
    season: str = Field(default="summer", description="summer|winter|shoulder")
    minimize_emissions: bool = Field(default=True, description="Prefer low-emission DERs")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class DEROptimizationResult(BaseModel):
    """Complete result from DER optimization workflow."""

    optimization_id: str = Field(..., description="Unique optimization ID")
    facility_id: str = Field(default="", description="Facility identifier")
    event_id: str = Field(default="", description="DR event identifier")
    assets_inventoried: int = Field(default=0, ge=0)
    assets_dispatched: int = Field(default=0, ge=0)
    der_assets: List[DERAsset] = Field(default_factory=list)
    dispatch_plans: List[DERDispatchPlan] = Field(default_factory=list)
    total_der_response_kw: Decimal = Field(default=Decimal("0"), ge=0)
    total_load_curtailment_kw: Decimal = Field(default=Decimal("0"), ge=0)
    combined_response_kw: Decimal = Field(default=Decimal("0"), ge=0)
    target_kw: Decimal = Field(default=Decimal("0"), ge=0)
    target_met: bool = Field(default=False)
    total_emissions_kg_co2: Decimal = Field(default=Decimal("0"), ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class DEROptimizationWorkflow:
    """
    3-phase DER optimization workflow for demand response events.

    Catalogs on-site DER assets, optimizes dispatch sequence and
    schedule, and coordinates DER response with load curtailment
    for maximum combined response.

    Zero-hallucination: dispatch optimization uses priority-based
    greedy scheduling with published round-trip efficiency and
    capacity factors. No LLM calls in the numeric computation path.

    Attributes:
        optimization_id: Unique optimization execution identifier.
        _assets: Cataloged DER assets.
        _dispatch_plans: Optimized dispatch plans.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = DEROptimizationWorkflow()
        >>> inp = DEROptimizationInput(
        ...     target_response_kw=Decimal("500"),
        ...     der_assets=[{"der_type": "battery", "capacity_kw": 250, "energy_kwh": 1000}],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.total_der_response_kw > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DEROptimizationWorkflow."""
        self.optimization_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._assets: List[DERAsset] = []
        self._dispatch_plans: List[DERDispatchPlan] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: DEROptimizationInput) -> DEROptimizationResult:
        """
        Execute the 3-phase DER optimization workflow.

        Args:
            input_data: Validated DER optimization input.

        Returns:
            DEROptimizationResult with dispatch plans and combined response.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        self.logger.info(
            "Starting DER optimization workflow %s target=%.0f kW",
            self.optimization_id, float(input_data.target_response_kw),
        )

        self._phase_results = []
        self._assets = []
        self._dispatch_plans = []

        try:
            # Phase 1: Asset Inventory
            phase1 = self._phase_asset_inventory(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Dispatch Strategy
            phase2 = self._phase_dispatch_strategy(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Coordinated Response
            phase3 = self._phase_coordinated_response(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "DER optimization workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        total_der = sum(p.dispatch_kw for p in self._dispatch_plans)
        load_curtail = input_data.load_curtailment_kw
        combined = total_der + load_curtail
        target_met = combined >= input_data.target_response_kw
        total_emissions = sum(p.emissions_kg_co2 for p in self._dispatch_plans)
        dispatched_count = sum(1 for p in self._dispatch_plans if p.dispatch_kw > 0)
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = DEROptimizationResult(
            optimization_id=self.optimization_id,
            facility_id=input_data.facility_id,
            event_id=input_data.event_id,
            assets_inventoried=len(self._assets),
            assets_dispatched=dispatched_count,
            der_assets=self._assets,
            dispatch_plans=self._dispatch_plans,
            total_der_response_kw=total_der,
            total_load_curtailment_kw=load_curtail,
            combined_response_kw=combined,
            target_kw=input_data.target_response_kw,
            target_met=target_met,
            total_emissions_kg_co2=total_emissions,
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "DER optimization workflow %s completed in %dms "
            "der=%.0f load=%.0f combined=%.0f target_met=%s",
            self.optimization_id, int(elapsed_ms), float(total_der),
            float(load_curtail), float(combined), target_met,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Asset Inventory
    # -------------------------------------------------------------------------

    def _phase_asset_inventory(
        self, input_data: DEROptimizationInput
    ) -> PhaseResult:
        """Catalog on-site DER assets with current status."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for asset_dict in input_data.der_assets:
            der_type = asset_dict.get("der_type", "battery")
            specs = DER_PERFORMANCE_SPECS.get(der_type, {})

            asset = DERAsset(
                asset_id=asset_dict.get("asset_id", f"der-{_new_uuid()[:8]}"),
                der_type=der_type,
                name=asset_dict.get("name", specs.get("description", der_type)),
                capacity_kw=Decimal(str(asset_dict.get("capacity_kw", 0))),
                energy_kwh=Decimal(str(asset_dict.get("energy_kwh", 0))),
                current_soc_pct=Decimal(str(asset_dict.get("current_soc_pct", 50))),
                available=asset_dict.get("available", True),
            )

            if not asset.available:
                warnings.append(f"Asset {asset.asset_id} ({der_type}) not available")

            self._assets.append(asset)

        available_count = sum(1 for a in self._assets if a.available)
        total_capacity = sum(a.capacity_kw for a in self._assets if a.available)

        outputs["assets_cataloged"] = len(self._assets)
        outputs["assets_available"] = available_count
        outputs["total_available_capacity_kw"] = str(total_capacity)
        outputs["asset_types"] = list(set(a.der_type for a in self._assets))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 AssetInventory: %d assets, %d available, %.0f kW capacity",
            len(self._assets), available_count, float(total_capacity),
        )
        return PhaseResult(
            phase_name="asset_inventory", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Dispatch Strategy
    # -------------------------------------------------------------------------

    def _phase_dispatch_strategy(
        self, input_data: DEROptimizationInput
    ) -> PhaseResult:
        """Optimize DER dispatch sequence and schedule."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        remaining_target = input_data.target_response_kw - input_data.load_curtailment_kw
        event_hours = input_data.event_duration_hours
        season = input_data.season

        # Sort assets by dispatch priority (and emissions if minimizing)
        available_assets = [a for a in self._assets if a.available and a.capacity_kw > 0]

        if input_data.minimize_emissions:
            available_assets.sort(key=lambda a: (
                DER_PERFORMANCE_SPECS.get(a.der_type, {}).get(
                    "emissions_kg_co2_per_kwh", Decimal("0")
                ),
                DER_PERFORMANCE_SPECS.get(a.der_type, {}).get("dispatch_priority", 99),
            ))
        else:
            available_assets.sort(key=lambda a: (
                DER_PERFORMANCE_SPECS.get(a.der_type, {}).get("dispatch_priority", 99),
            ))

        cumulative_kw = Decimal("0")
        priority_counter = 0

        for asset in available_assets:
            if cumulative_kw >= remaining_target:
                break

            specs = DER_PERFORMANCE_SPECS.get(asset.der_type, {})
            dispatch_mode = specs.get("dispatch_mode", "discharge")

            # Calculate available dispatch power
            dispatch_kw = self._calculate_dispatch_kw(asset, specs, season, event_hours)
            if dispatch_kw <= 0:
                continue

            # Calculate dispatch duration
            dispatch_duration = self._calculate_dispatch_duration(
                asset, specs, dispatch_kw, event_hours
            )

            # Calculate dispatch energy
            dispatch_energy = (dispatch_kw * dispatch_duration).quantize(Decimal("0.1"))

            # Calculate emissions
            emissions_factor = specs.get("emissions_kg_co2_per_kwh", Decimal("0"))
            emissions = (dispatch_energy * emissions_factor).quantize(Decimal("0.01"))

            priority_counter += 1
            cumulative_kw += dispatch_kw

            plan = DERDispatchPlan(
                asset_id=asset.asset_id,
                der_type=asset.der_type,
                dispatch_mode=dispatch_mode,
                dispatch_kw=dispatch_kw,
                dispatch_duration_hours=dispatch_duration,
                dispatch_energy_kwh=dispatch_energy,
                priority=priority_counter,
                emissions_kg_co2=emissions,
            )
            self._dispatch_plans.append(plan)

            # Update asset state
            asset.dispatch_kw = dispatch_kw
            asset.dispatch_mode = dispatch_mode
            asset.dispatch_duration_hours = dispatch_duration

        total_dispatched = sum(p.dispatch_kw for p in self._dispatch_plans)
        shortfall = max(Decimal("0"), remaining_target - total_dispatched)
        if shortfall > 0:
            warnings.append(
                f"DER shortfall of {shortfall} kW; "
                f"dispatched {total_dispatched} kW < remaining target {remaining_target} kW"
            )

        outputs["assets_dispatched"] = len(self._dispatch_plans)
        outputs["total_dispatched_kw"] = str(total_dispatched)
        outputs["remaining_target_kw"] = str(remaining_target)
        outputs["shortfall_kw"] = str(shortfall)
        outputs["dispatch_priority_order"] = [
            p.asset_id for p in self._dispatch_plans
        ]

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 DispatchStrategy: %d assets dispatched, total=%.0f kW",
            len(self._dispatch_plans), float(total_dispatched),
        )
        return PhaseResult(
            phase_name="dispatch_strategy", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_dispatch_kw(
        self,
        asset: DERAsset,
        specs: Dict[str, Any],
        season: str,
        event_hours: Decimal,
    ) -> Decimal:
        """Calculate available dispatch power for a DER asset."""
        der_type = asset.der_type

        if der_type == "battery":
            efficiency = specs.get("round_trip_efficiency", Decimal("0.90"))
            min_soc = specs.get("min_soc_pct", Decimal("10"))
            usable_soc = max(Decimal("0"), asset.current_soc_pct - min_soc) / Decimal("100")
            usable_kwh = asset.energy_kwh * usable_soc * efficiency
            max_kw = asset.capacity_kw
            kw_from_energy = (usable_kwh / event_hours).quantize(Decimal("0.1")) if event_hours > 0 else Decimal("0")
            return min(max_kw, kw_from_energy)

        if der_type == "solar_pv":
            cf_key = f"capacity_factor_{season}" if season in ("summer", "winter") else "capacity_factor_summer"
            cf = specs.get(cf_key, Decimal("0.20"))
            return (asset.capacity_kw * cf).quantize(Decimal("0.1"))

        if der_type == "ev_fleet":
            availability = specs.get("availability_pct", Decimal("60")) / Decimal("100")
            per_vehicle_kw = specs.get("max_discharge_rate_per_vehicle_kw", Decimal("7"))
            vehicle_count = (
                asset.capacity_kw / per_vehicle_kw
            ).quantize(Decimal("1")) if per_vehicle_kw > 0 else Decimal("0")
            return (vehicle_count * per_vehicle_kw * availability).quantize(Decimal("0.1"))

        if der_type == "thermal_storage":
            duration = specs.get("discharge_duration_hours", Decimal("6"))
            if duration > 0:
                return (asset.energy_kwh / duration).quantize(Decimal("0.1"))
            return Decimal("0")

        # Default: use rated capacity
        return asset.capacity_kw

    def _calculate_dispatch_duration(
        self,
        asset: DERAsset,
        specs: Dict[str, Any],
        dispatch_kw: Decimal,
        event_hours: Decimal,
    ) -> Decimal:
        """Calculate how long the asset can sustain dispatch."""
        if asset.energy_kwh > 0 and dispatch_kw > 0:
            max_duration = (asset.energy_kwh / dispatch_kw).quantize(Decimal("0.1"))
            max_runtime = Decimal(str(specs.get("max_runtime_hours", 999)))
            return min(event_hours, max_duration, max_runtime)
        return event_hours

    # -------------------------------------------------------------------------
    # Phase 3: Coordinated Response
    # -------------------------------------------------------------------------

    def _phase_coordinated_response(
        self, input_data: DEROptimizationInput
    ) -> PhaseResult:
        """Coordinate DER dispatch with load curtailment for combined response."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_der = sum(p.dispatch_kw for p in self._dispatch_plans)
        load_curtail = input_data.load_curtailment_kw
        combined = total_der + load_curtail
        target = input_data.target_response_kw
        target_met = combined >= target

        # Calculate response breakdown
        der_pct = (
            float(total_der) / float(combined) * 100.0
            if combined > 0 else 0.0
        )
        load_pct = 100.0 - der_pct

        # Calculate total emissions
        total_emissions = sum(p.emissions_kg_co2 for p in self._dispatch_plans)
        total_energy = sum(p.dispatch_energy_kwh for p in self._dispatch_plans)

        # Build coordination timeline
        timeline: List[Dict[str, Any]] = []
        for plan in sorted(self._dispatch_plans, key=lambda p: p.priority):
            specs = DER_PERFORMANCE_SPECS.get(plan.der_type, {})
            ramp_time = specs.get("ramp_time_min", 5)
            timeline.append({
                "priority": plan.priority,
                "asset_id": plan.asset_id,
                "der_type": plan.der_type,
                "dispatch_kw": str(plan.dispatch_kw),
                "ramp_time_min": ramp_time,
                "mode": plan.dispatch_mode,
            })

        outputs["combined_response_kw"] = str(combined)
        outputs["der_response_kw"] = str(total_der)
        outputs["load_curtailment_kw"] = str(load_curtail)
        outputs["der_pct"] = round(der_pct, 1)
        outputs["load_pct"] = round(load_pct, 1)
        outputs["target_kw"] = str(target)
        outputs["target_met"] = target_met
        outputs["total_emissions_kg_co2"] = str(total_emissions)
        outputs["total_dispatch_energy_kwh"] = str(total_energy)
        outputs["coordination_timeline"] = timeline

        if not target_met:
            shortfall = target - combined
            warnings.append(
                f"Combined response {combined} kW < target {target} kW "
                f"(shortfall {shortfall} kW)"
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 CoordinatedResponse: combined=%.0f kW "
            "(DER=%.0f + Load=%.0f) target_met=%s",
            float(combined), float(total_der), float(load_curtail), target_met,
        )
        return PhaseResult(
            phase_name="coordinated_response", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: DEROptimizationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
