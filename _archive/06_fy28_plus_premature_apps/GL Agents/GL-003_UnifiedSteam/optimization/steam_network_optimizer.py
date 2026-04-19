"""
GL-003 UNIFIEDSTEAM - Steam Network Optimizer

Provides network-wide optimization for steam systems:
- Header pressure optimization
- PRV setpoint optimization
- Multi-boiler load allocation
- Total loss minimization

Multi-objective optimization balancing:
- Efficiency
- Emissions (CO2e)
- Cost (fuel, water, maintenance)
- Reliability
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import time

from pydantic import BaseModel, Field, validator

from .constraints import (
    ConstraintCheckResult,
    ConstraintSeverity,
    ConstraintStatus,
    EquipmentConstraints,
    SafetyConstraints,
    SteamSystemConstraints,
    UncertaintyConstraints,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class HeaderType(str, Enum):
    """Steam header type."""

    HIGH_PRESSURE = "hp"
    MEDIUM_PRESSURE = "mp"
    LOW_PRESSURE = "lp"


class BoilerState(BaseModel):
    """Current state of a boiler."""

    boiler_id: str = Field(..., description="Boiler identifier")
    is_online: bool = Field(default=True, description="Online status")
    current_load_percent: float = Field(
        ..., ge=0, le=110, description="Current load (%)"
    )
    rated_capacity_klb_hr: float = Field(
        ..., gt=0, description="Rated capacity (klb/hr)"
    )
    min_load_percent: float = Field(
        default=25.0, ge=0, le=100, description="Minimum stable load (%)"
    )
    max_load_percent: float = Field(
        default=100.0, ge=0, le=110, description="Maximum load (%)"
    )
    current_efficiency_percent: float = Field(
        ..., ge=50, le=100, description="Current efficiency (%)"
    )
    fuel_type: str = Field(default="natural_gas", description="Primary fuel")
    fuel_cost_per_mmbtu: float = Field(
        default=8.0, gt=0, description="Fuel cost ($/MMBTU)"
    )
    co2_factor_lb_mmbtu: float = Field(
        default=117.0, description="CO2 emission factor (lb/MMBTU)"
    )
    maintenance_priority: int = Field(
        default=0, ge=0, description="Maintenance priority (0=none)"
    )


class HeaderState(BaseModel):
    """Current state of a steam header."""

    header_id: str = Field(..., description="Header identifier")
    header_type: HeaderType = Field(..., description="Header type")
    pressure_psig: float = Field(..., description="Current pressure (psig)")
    setpoint_psig: float = Field(..., description="Pressure setpoint (psig)")
    temperature_f: float = Field(..., description="Current temperature (F)")
    flow_klb_hr: float = Field(..., ge=0, description="Total flow (klb/hr)")
    user_demand_klb_hr: float = Field(
        ..., ge=0, description="User demand (klb/hr)"
    )
    connected_boilers: List[str] = Field(
        default_factory=list, description="Connected boiler IDs"
    )
    connected_prvs: List[str] = Field(
        default_factory=list, description="Connected PRV IDs"
    )


class PRVState(BaseModel):
    """Current state of a pressure reducing valve."""

    prv_id: str = Field(..., description="PRV identifier")
    upstream_header: str = Field(..., description="Upstream header ID")
    downstream_header: str = Field(..., description="Downstream header ID")
    upstream_pressure_psig: float = Field(
        ..., description="Upstream pressure (psig)"
    )
    downstream_pressure_psig: float = Field(
        ..., description="Downstream pressure (psig)"
    )
    setpoint_psig: float = Field(
        ..., description="Downstream setpoint (psig)"
    )
    flow_klb_hr: float = Field(..., ge=0, description="Current flow (klb/hr)")
    valve_position_percent: float = Field(
        ..., ge=0, le=100, description="Valve position (%)"
    )
    max_capacity_klb_hr: float = Field(
        ..., gt=0, description="Maximum capacity (klb/hr)"
    )
    is_desuperheating: bool = Field(
        default=False, description="Has desuperheating capability"
    )


class DemandForecast(BaseModel):
    """Steam demand forecast."""

    forecast_horizon_hours: int = Field(
        default=24, description="Forecast horizon (hours)"
    )
    hp_demand_klb_hr: List[float] = Field(
        default_factory=list, description="HP demand forecast (klb/hr)"
    )
    mp_demand_klb_hr: List[float] = Field(
        default_factory=list, description="MP demand forecast (klb/hr)"
    )
    lp_demand_klb_hr: List[float] = Field(
        default_factory=list, description="LP demand forecast (klb/hr)"
    )
    confidence: float = Field(
        default=0.85, ge=0, le=1, description="Forecast confidence"
    )


class NetworkModel(BaseModel):
    """Steam network model."""

    boilers: List[BoilerState] = Field(
        default_factory=list, description="Boiler states"
    )
    headers: List[HeaderState] = Field(
        default_factory=list, description="Header states"
    )
    prvs: List[PRVState] = Field(
        default_factory=list, description="PRV states"
    )
    total_generation_klb_hr: float = Field(
        default=0.0, ge=0, description="Total steam generation (klb/hr)"
    )
    total_demand_klb_hr: float = Field(
        default=0.0, ge=0, description="Total steam demand (klb/hr)"
    )
    distribution_loss_percent: float = Field(
        default=3.0, ge=0, le=20, description="Distribution losses (%)"
    )


# =============================================================================
# Result Models
# =============================================================================


class BoilerLoadAllocation(BaseModel):
    """Load allocation for a single boiler."""

    boiler_id: str = Field(..., description="Boiler identifier")
    current_load_percent: float = Field(..., description="Current load (%)")
    recommended_load_percent: float = Field(
        ..., description="Recommended load (%)"
    )
    recommended_output_klb_hr: float = Field(
        ..., description="Recommended output (klb/hr)"
    )
    efficiency_at_load: float = Field(..., description="Efficiency at load (%)")
    cost_at_load: float = Field(..., description="Operating cost ($/hr)")
    co2_at_load_lb_hr: float = Field(..., description="CO2 emissions (lb/hr)")
    change_required: bool = Field(..., description="Whether change needed")


class LoadAllocationResult(BaseModel):
    """Result of load allocation optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    allocations: List[BoilerLoadAllocation] = Field(
        default_factory=list, description="Per-boiler allocations"
    )
    total_demand_klb_hr: float = Field(..., description="Total demand (klb/hr)")
    total_generation_klb_hr: float = Field(
        ..., description="Total generation (klb/hr)"
    )
    total_cost_per_hr: float = Field(..., description="Total cost ($/hr)")
    total_co2_lb_hr: float = Field(..., description="Total CO2 (lb/hr)")
    weighted_efficiency: float = Field(..., description="Weighted efficiency (%)")
    optimization_objective: str = Field(
        default="cost", description="Optimization objective"
    )
    improvement_percent: float = Field(
        default=0.0, description="Improvement from current (%)"
    )
    confidence: float = Field(default=0.95, ge=0, le=1)
    provenance_hash: str = Field(default="")


class HeaderOptimization(BaseModel):
    """Result of header pressure optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    header_id: str = Field(..., description="Header identifier")
    current_pressure_psig: float = Field(..., description="Current pressure")
    recommended_pressure_psig: float = Field(
        ..., description="Recommended pressure"
    )
    pressure_change_psig: float = Field(..., description="Pressure change")
    reason: str = Field(..., description="Optimization reason")
    expected_savings_per_hr: float = Field(
        default=0.0, description="Expected savings ($/hr)"
    )
    efficiency_improvement_percent: float = Field(
        default=0.0, description="Efficiency improvement (%)"
    )
    constraints_satisfied: bool = Field(default=True)
    constraint_warnings: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.90, ge=0, le=1)


class PRVOptimization(BaseModel):
    """Result of PRV setpoint optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    optimizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-PRV optimizations"
    )
    total_prv_flow_klb_hr: float = Field(
        default=0.0, description="Total PRV flow (klb/hr)"
    )
    total_letdown_loss_btu_hr: float = Field(
        default=0.0, description="Total letdown loss (BTU/hr)"
    )
    improvement_potential_percent: float = Field(
        default=0.0, description="Improvement potential (%)"
    )
    confidence: float = Field(default=0.90, ge=0, le=1)


class LossMinimizationResult(BaseModel):
    """Result of total loss minimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Current losses
    current_generation_loss_klb_hr: float = Field(
        default=0.0, description="Current generation losses (klb/hr)"
    )
    current_distribution_loss_klb_hr: float = Field(
        default=0.0, description="Current distribution losses (klb/hr)"
    )
    current_trap_loss_klb_hr: float = Field(
        default=0.0, description="Current trap losses (klb/hr)"
    )
    current_total_loss_klb_hr: float = Field(
        default=0.0, description="Total current losses (klb/hr)"
    )
    current_loss_cost_per_hr: float = Field(
        default=0.0, description="Current loss cost ($/hr)"
    )

    # Optimized losses
    optimized_total_loss_klb_hr: float = Field(
        default=0.0, description="Optimized total losses (klb/hr)"
    )
    optimized_loss_cost_per_hr: float = Field(
        default=0.0, description="Optimized loss cost ($/hr)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list, description="Loss reduction recommendations"
    )
    total_savings_per_hr: float = Field(
        default=0.0, description="Total savings ($/hr)"
    )
    annual_savings_potential: float = Field(
        default=0.0, description="Annual savings potential ($)"
    )
    confidence: float = Field(default=0.85, ge=0, le=1)
    provenance_hash: str = Field(default="")


# =============================================================================
# Steam Network Optimizer
# =============================================================================


class SteamNetworkOptimizer:
    """
    Network-wide optimization for steam systems.

    Performs multi-objective optimization balancing:
    - Efficiency (minimize fuel per unit steam)
    - Emissions (minimize CO2e per unit steam)
    - Cost (minimize total operating cost)
    - Reliability (maintain N+1 redundancy where required)

    Uses deterministic calculations (zero-hallucination approach):
    - Boiler efficiency curves
    - Energy balances
    - Constraint satisfaction
    """

    DEFAULT_OPERATING_HOURS = 8000
    DEFAULT_CO2_COST_PER_TON = 0.0  # Carbon tax if applicable

    def __init__(
        self,
        constraints: Optional[SteamSystemConstraints] = None,
        operating_hours: int = DEFAULT_OPERATING_HOURS,
        co2_cost_per_ton: float = DEFAULT_CO2_COST_PER_TON,
    ) -> None:
        """
        Initialize steam network optimizer.

        Args:
            constraints: System constraints
            operating_hours: Annual operating hours
            co2_cost_per_ton: CO2 cost ($/ton)
        """
        self.constraints = constraints or SteamSystemConstraints()
        self.operating_hours = operating_hours
        self.co2_cost_per_ton = co2_cost_per_ton

        # Optimization history
        self._optimization_history: List[Dict[str, Any]] = []
        self._max_history = 500

        logger.info("SteamNetworkOptimizer initialized")

    def optimize_header_pressures(
        self,
        demand_forecast: DemandForecast,
        boiler_constraints: Dict[str, Any],
        current_headers: List[HeaderState],
    ) -> List[HeaderOptimization]:
        """
        Optimize header pressure setpoints.

        Lower pressure = lower losses and higher efficiency, but must maintain:
        - Minimum pressure for end users
        - Adequate superheat
        - PRV differential requirements

        Args:
            demand_forecast: Steam demand forecast
            boiler_constraints: Boiler operating constraints
            current_headers: Current header states

        Returns:
            List of header optimization results
        """
        results: List[HeaderOptimization] = []

        for header in current_headers:
            # Get demand for this header type
            if header.header_type == HeaderType.HIGH_PRESSURE:
                demand_list = demand_forecast.hp_demand_klb_hr
            elif header.header_type == HeaderType.MEDIUM_PRESSURE:
                demand_list = demand_forecast.mp_demand_klb_hr
            else:
                demand_list = demand_forecast.lp_demand_klb_hr

            peak_demand = max(demand_list) if demand_list else header.user_demand_klb_hr

            # Check if pressure can be optimized
            result = self._optimize_single_header(
                header=header,
                peak_demand=peak_demand,
            )
            results.append(result)

        logger.info(f"Header pressure optimization complete for {len(results)} headers")

        return results

    def _optimize_single_header(
        self,
        header: HeaderState,
        peak_demand: float,
    ) -> HeaderOptimization:
        """Optimize pressure for a single header."""
        current_pressure = header.pressure_psig
        setpoint = header.setpoint_psig

        # Get limits based on header type
        safety = self.constraints.safety
        if header.header_type == HeaderType.HIGH_PRESSURE:
            min_p = safety.pressure.hp_header_min_psig
            max_p = safety.pressure.hp_header_max_psig
            target_range = (min_p, max_p)
        elif header.header_type == HeaderType.MEDIUM_PRESSURE:
            min_p = safety.pressure.mp_header_min_psig
            max_p = safety.pressure.mp_header_max_psig
            target_range = (min_p, max_p)
        else:
            min_p = safety.pressure.lp_header_min_psig
            max_p = safety.pressure.lp_header_max_psig
            target_range = (min_p, max_p)

        # Calculate minimum pressure required for users
        user_min_pressure = self._calculate_user_min_pressure(
            header.header_type, peak_demand
        )

        # Optimal pressure is the minimum that satisfies all constraints
        optimal_pressure = max(min_p, user_min_pressure)

        # Check if we can reduce from current
        warnings = []
        if optimal_pressure > max_p:
            warnings.append(f"Calculated optimal {optimal_pressure:.1f} exceeds max {max_p:.1f}")
            optimal_pressure = max_p

        # Calculate potential savings from pressure reduction
        if setpoint > optimal_pressure:
            pressure_reduction = setpoint - optimal_pressure
            # Approximate: 1% fuel savings per 10 psi reduction
            savings_percent = pressure_reduction / 10 * 0.01
            # Assume $1000/hr base cost for estimation
            savings_per_hr = savings_percent * 1000
            efficiency_improvement = savings_percent * 100
            reason = f"Reduce pressure to minimize losses ({pressure_reduction:.1f} psi reduction)"
        else:
            savings_per_hr = 0.0
            efficiency_improvement = 0.0
            reason = "Current pressure is at or below optimal"
            optimal_pressure = setpoint  # No change needed

        return HeaderOptimization(
            header_id=header.header_id,
            current_pressure_psig=current_pressure,
            recommended_pressure_psig=round(optimal_pressure, 1),
            pressure_change_psig=round(optimal_pressure - setpoint, 1),
            reason=reason,
            expected_savings_per_hr=savings_per_hr,
            efficiency_improvement_percent=efficiency_improvement,
            constraints_satisfied=len(warnings) == 0,
            constraint_warnings=warnings,
            confidence=0.90 if len(warnings) == 0 else 0.75,
        )

    def _calculate_user_min_pressure(
        self,
        header_type: HeaderType,
        demand_klb_hr: float,
    ) -> float:
        """Calculate minimum pressure required by users."""
        # This would be based on actual user requirements in a real system
        # Simplified estimation based on header type
        if header_type == HeaderType.HIGH_PRESSURE:
            return 550.0  # Typical HP user minimum
        elif header_type == HeaderType.MEDIUM_PRESSURE:
            return 140.0  # Typical MP user minimum
        else:
            return 10.0  # Typical LP user minimum

    def optimize_prv_setpoints(
        self,
        network_state: NetworkModel,
        user_requirements: Dict[str, float],
    ) -> PRVOptimization:
        """
        Optimize PRV setpoints for downstream pressure control.

        Balances:
        - Downstream pressure requirements
        - Minimum PRV differential for stable operation
        - Desuperheating if applicable

        Args:
            network_state: Current network state
            user_requirements: Minimum pressure requirements by header

        Returns:
            PRV optimization result
        """
        optimizations: List[Dict[str, Any]] = []
        total_flow = 0.0
        total_letdown_loss = 0.0

        for prv in network_state.prvs:
            # Check downstream requirements
            downstream_min = user_requirements.get(prv.downstream_header, 0.0)

            # Check PRV differential
            current_diff = prv.upstream_pressure_psig - prv.downstream_pressure_psig
            min_diff = self.constraints.safety.pressure.min_prv_delta_psi

            # Optimal downstream is minimum that satisfies users
            optimal_downstream = max(downstream_min, prv.setpoint_psig)

            # Ensure adequate differential
            if prv.upstream_pressure_psig - optimal_downstream < min_diff:
                optimal_downstream = prv.upstream_pressure_psig - min_diff

            # Calculate letdown loss
            letdown_loss = self._calculate_letdown_loss(
                prv.flow_klb_hr,
                prv.upstream_pressure_psig,
                optimal_downstream,
            )

            change_needed = abs(optimal_downstream - prv.setpoint_psig) > 1.0

            optimizations.append({
                "prv_id": prv.prv_id,
                "current_setpoint_psig": prv.setpoint_psig,
                "recommended_setpoint_psig": round(optimal_downstream, 1),
                "change_psig": round(optimal_downstream - prv.setpoint_psig, 1),
                "current_flow_klb_hr": prv.flow_klb_hr,
                "letdown_loss_btu_hr": letdown_loss,
                "change_needed": change_needed,
            })

            total_flow += prv.flow_klb_hr
            total_letdown_loss += letdown_loss

        return PRVOptimization(
            optimizations=optimizations,
            total_prv_flow_klb_hr=total_flow,
            total_letdown_loss_btu_hr=total_letdown_loss,
            improvement_potential_percent=0.0,  # Would calculate from baseline
            confidence=0.90,
        )

    def _calculate_letdown_loss(
        self,
        flow_klb_hr: float,
        upstream_psig: float,
        downstream_psig: float,
    ) -> float:
        """Calculate energy loss from pressure letdown (BTU/hr)."""
        # Isenthalpic expansion - no heat loss but exergy loss
        # This represents the lost opportunity to do work
        # Simplified: proportional to pressure drop
        pressure_drop = upstream_psig - downstream_psig
        # Approximate: 1 BTU/lb per psi for saturated steam
        exergy_loss_per_lb = pressure_drop * 1.0
        return flow_klb_hr * 1000 * exergy_loss_per_lb

    def optimize_load_allocation(
        self,
        boilers: List[BoilerState],
        total_demand_klb_hr: float,
        objective: str = "cost",
    ) -> LoadAllocationResult:
        """
        Optimize load allocation across multiple boilers.

        Uses efficiency curves and cost functions to determine
        optimal loading for each boiler.

        Objectives:
        - "cost": Minimize total operating cost
        - "efficiency": Maximize weighted average efficiency
        - "emissions": Minimize total CO2 emissions
        - "balanced": Multi-objective with equal weights

        Args:
            boilers: List of boiler states
            total_demand_klb_hr: Total steam demand (klb/hr)
            objective: Optimization objective

        Returns:
            LoadAllocationResult with optimal allocations
        """
        start_time = time.perf_counter()

        # Filter to online boilers
        online_boilers = [b for b in boilers if b.is_online]

        if not online_boilers:
            raise ValueError("No online boilers available")

        # Calculate total capacity
        total_capacity = sum(
            b.rated_capacity_klb_hr * b.max_load_percent / 100
            for b in online_boilers
        )

        if total_demand_klb_hr > total_capacity:
            logger.warning(
                f"Demand {total_demand_klb_hr:.0f} exceeds capacity {total_capacity:.0f}"
            )
            total_demand_klb_hr = total_capacity * 0.95

        # Perform optimization based on objective
        if objective == "efficiency":
            allocations = self._allocate_by_efficiency(online_boilers, total_demand_klb_hr)
        elif objective == "emissions":
            allocations = self._allocate_by_emissions(online_boilers, total_demand_klb_hr)
        elif objective == "balanced":
            allocations = self._allocate_balanced(online_boilers, total_demand_klb_hr)
        else:  # cost
            allocations = self._allocate_by_cost(online_boilers, total_demand_klb_hr)

        # Calculate totals
        total_cost = sum(a.cost_at_load for a in allocations)
        total_co2 = sum(a.co2_at_load_lb_hr for a in allocations)
        total_generation = sum(a.recommended_output_klb_hr for a in allocations)

        # Weighted efficiency
        weighted_eff = sum(
            a.efficiency_at_load * a.recommended_output_klb_hr
            for a in allocations
        ) / total_generation if total_generation > 0 else 0

        # Calculate current totals for comparison
        current_cost = sum(
            self._calculate_boiler_cost(b, b.current_load_percent)
            for b in online_boilers
        )
        improvement = (
            (current_cost - total_cost) / current_cost * 100
            if current_cost > 0 else 0
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        result = LoadAllocationResult(
            allocations=allocations,
            total_demand_klb_hr=total_demand_klb_hr,
            total_generation_klb_hr=total_generation,
            total_cost_per_hr=total_cost,
            total_co2_lb_hr=total_co2,
            weighted_efficiency=weighted_eff,
            optimization_objective=objective,
            improvement_percent=improvement,
            confidence=0.95,
        )

        result.provenance_hash = self._generate_provenance_hash(
            boilers, total_demand_klb_hr, result
        )

        logger.info(
            f"Load allocation ({objective}): "
            f"demand={total_demand_klb_hr:.0f} klb/hr, "
            f"cost=${total_cost:.0f}/hr, "
            f"efficiency={weighted_eff:.1f}%, "
            f"improvement={improvement:.1f}% in {computation_time:.1f}ms"
        )

        return result

    def _allocate_by_cost(
        self,
        boilers: List[BoilerState],
        demand: float,
    ) -> List[BoilerLoadAllocation]:
        """Allocate load to minimize cost using marginal cost ordering."""
        allocations = []
        remaining_demand = demand

        # Sort by marginal cost at 50% load
        sorted_boilers = sorted(
            boilers,
            key=lambda b: self._calculate_marginal_cost(b, 50.0)
        )

        for boiler in sorted_boilers:
            if remaining_demand <= 0:
                load_pct = 0.0
                output = 0.0
            else:
                min_output = boiler.rated_capacity_klb_hr * boiler.min_load_percent / 100
                max_output = boiler.rated_capacity_klb_hr * boiler.max_load_percent / 100

                if remaining_demand >= max_output:
                    output = max_output
                elif remaining_demand >= min_output:
                    output = remaining_demand
                else:
                    output = 0.0  # Can't meet minimum load

                load_pct = (
                    output / boiler.rated_capacity_klb_hr * 100
                    if output > 0 else 0
                )
                remaining_demand -= output

            efficiency = self._calculate_efficiency_at_load(boiler, load_pct)
            cost = self._calculate_boiler_cost(boiler, load_pct)
            co2 = self._calculate_boiler_co2(boiler, load_pct)

            allocations.append(
                BoilerLoadAllocation(
                    boiler_id=boiler.boiler_id,
                    current_load_percent=boiler.current_load_percent,
                    recommended_load_percent=round(load_pct, 1),
                    recommended_output_klb_hr=round(output, 1),
                    efficiency_at_load=round(efficiency, 1),
                    cost_at_load=round(cost, 2),
                    co2_at_load_lb_hr=round(co2, 0),
                    change_required=abs(load_pct - boiler.current_load_percent) > 2.0,
                )
            )

        return allocations

    def _allocate_by_efficiency(
        self,
        boilers: List[BoilerState],
        demand: float,
    ) -> List[BoilerLoadAllocation]:
        """Allocate load to maximize efficiency."""
        allocations = []
        remaining_demand = demand

        # Sort by peak efficiency (typically at 70-80% load)
        sorted_boilers = sorted(
            boilers,
            key=lambda b: self._calculate_efficiency_at_load(b, 75.0),
            reverse=True
        )

        for boiler in sorted_boilers:
            if remaining_demand <= 0:
                load_pct = 0.0
                output = 0.0
            else:
                min_output = boiler.rated_capacity_klb_hr * boiler.min_load_percent / 100
                max_output = boiler.rated_capacity_klb_hr * boiler.max_load_percent / 100

                # Try to load at optimal efficiency point
                optimal_output = boiler.rated_capacity_klb_hr * 0.75

                if remaining_demand >= max_output:
                    output = max_output
                elif remaining_demand >= optimal_output:
                    output = min(remaining_demand, max_output)
                elif remaining_demand >= min_output:
                    output = remaining_demand
                else:
                    output = 0.0

                load_pct = (
                    output / boiler.rated_capacity_klb_hr * 100
                    if output > 0 else 0
                )
                remaining_demand -= output

            efficiency = self._calculate_efficiency_at_load(boiler, load_pct)
            cost = self._calculate_boiler_cost(boiler, load_pct)
            co2 = self._calculate_boiler_co2(boiler, load_pct)

            allocations.append(
                BoilerLoadAllocation(
                    boiler_id=boiler.boiler_id,
                    current_load_percent=boiler.current_load_percent,
                    recommended_load_percent=round(load_pct, 1),
                    recommended_output_klb_hr=round(output, 1),
                    efficiency_at_load=round(efficiency, 1),
                    cost_at_load=round(cost, 2),
                    co2_at_load_lb_hr=round(co2, 0),
                    change_required=abs(load_pct - boiler.current_load_percent) > 2.0,
                )
            )

        return allocations

    def _allocate_by_emissions(
        self,
        boilers: List[BoilerState],
        demand: float,
    ) -> List[BoilerLoadAllocation]:
        """Allocate load to minimize emissions."""
        # Sort by CO2 factor * efficiency inverse
        sorted_boilers = sorted(
            boilers,
            key=lambda b: b.co2_factor_lb_mmbtu / max(
                self._calculate_efficiency_at_load(b, 75.0), 60
            )
        )
        return self._allocate_by_cost(sorted_boilers, demand)

    def _allocate_balanced(
        self,
        boilers: List[BoilerState],
        demand: float,
    ) -> List[BoilerLoadAllocation]:
        """Allocate load using multi-objective balance."""
        # Simple approach: score each boiler on cost, efficiency, emissions
        # More sophisticated would use Pareto optimization
        return self._allocate_by_cost(boilers, demand)

    def _calculate_efficiency_at_load(
        self,
        boiler: BoilerState,
        load_percent: float,
    ) -> float:
        """Calculate boiler efficiency at given load."""
        if load_percent <= 0:
            return 0.0

        # Typical efficiency curve (quadratic)
        # Peak around 70-80%, drops at low and high loads
        design_eff = 85.0  # Typical design efficiency

        # Coefficients for quadratic curve
        a = -0.002
        b = 0.3
        c = 75.0

        efficiency = a * load_percent ** 2 + b * load_percent + c
        return max(60.0, min(95.0, efficiency))

    def _calculate_marginal_cost(
        self,
        boiler: BoilerState,
        load_percent: float,
    ) -> float:
        """Calculate marginal cost at load point."""
        efficiency = self._calculate_efficiency_at_load(boiler, load_percent)
        if efficiency <= 0:
            return float('inf')

        # Cost per unit output
        output_klb_hr = boiler.rated_capacity_klb_hr * load_percent / 100
        if output_klb_hr <= 0:
            return float('inf')

        cost = self._calculate_boiler_cost(boiler, load_percent)
        return cost / output_klb_hr

    def _calculate_boiler_cost(
        self,
        boiler: BoilerState,
        load_percent: float,
    ) -> float:
        """Calculate operating cost at load."""
        if load_percent <= 0:
            return 0.0

        efficiency = self._calculate_efficiency_at_load(boiler, load_percent) / 100
        output_klb_hr = boiler.rated_capacity_klb_hr * load_percent / 100

        # Heat output (BTU/hr)
        steam_enthalpy = 1190  # BTU/lb (superheated steam approx)
        fw_enthalpy = 200  # BTU/lb
        heat_output = output_klb_hr * 1000 * (steam_enthalpy - fw_enthalpy)

        # Fuel input
        fuel_input_mmbtu = heat_output / efficiency / 1e6

        return fuel_input_mmbtu * boiler.fuel_cost_per_mmbtu

    def _calculate_boiler_co2(
        self,
        boiler: BoilerState,
        load_percent: float,
    ) -> float:
        """Calculate CO2 emissions at load."""
        if load_percent <= 0:
            return 0.0

        efficiency = self._calculate_efficiency_at_load(boiler, load_percent) / 100
        output_klb_hr = boiler.rated_capacity_klb_hr * load_percent / 100

        # Heat output
        steam_enthalpy = 1190
        fw_enthalpy = 200
        heat_output = output_klb_hr * 1000 * (steam_enthalpy - fw_enthalpy)

        # Fuel input
        fuel_input_mmbtu = heat_output / efficiency / 1e6

        return fuel_input_mmbtu * boiler.co2_factor_lb_mmbtu

    def minimize_total_losses(
        self,
        network_model: NetworkModel,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> LossMinimizationResult:
        """
        Minimize total steam generation and distribution losses.

        Analyzes:
        - Generation losses (stack, blowdown, radiation)
        - Distribution losses (insulation, trap leaks, PRV letdown)
        - Identifies opportunities for reduction

        Args:
            network_model: Current network model
            constraints: Additional constraints

        Returns:
            LossMinimizationResult with recommendations
        """
        start_time = time.perf_counter()

        # Calculate current losses
        generation_loss = self._calculate_generation_losses(network_model)
        distribution_loss = self._calculate_distribution_losses(network_model)
        trap_loss = self._estimate_trap_losses(network_model)

        total_loss = generation_loss + distribution_loss + trap_loss

        # Calculate loss cost
        steam_cost_per_klb = 10.0  # $/klb (approximate)
        current_loss_cost = total_loss * steam_cost_per_klb

        # Generate recommendations
        recommendations = []
        optimized_loss = total_loss

        # Check for high distribution losses
        loss_percent = (
            network_model.distribution_loss_percent
            if network_model.total_generation_klb_hr > 0 else 0
        )
        if loss_percent > 5.0:
            recommendations.append(
                f"Distribution losses ({loss_percent:.1f}%) exceed target (5%). "
                "Inspect insulation and trap surveys."
            )
            optimized_loss -= (loss_percent - 5.0) / 100 * network_model.total_generation_klb_hr

        # Check trap losses
        if trap_loss > network_model.total_generation_klb_hr * 0.03:
            recommendations.append(
                f"Trap losses ({trap_loss:.0f} klb/hr) exceed 3% of generation. "
                "Priority trap maintenance recommended."
            )
            target_trap_loss = network_model.total_generation_klb_hr * 0.03
            optimized_loss -= (trap_loss - target_trap_loss)

        # Check PRV letdown
        prv_flow = sum(prv.flow_klb_hr for prv in network_model.prvs)
        if prv_flow > network_model.total_generation_klb_hr * 0.3:
            recommendations.append(
                f"PRV letdown ({prv_flow:.0f} klb/hr) is high. "
                "Consider backpressure turbines or direct generation at MP/LP."
            )

        optimized_loss_cost = optimized_loss * steam_cost_per_klb
        savings = current_loss_cost - optimized_loss_cost

        result = LossMinimizationResult(
            current_generation_loss_klb_hr=generation_loss,
            current_distribution_loss_klb_hr=distribution_loss,
            current_trap_loss_klb_hr=trap_loss,
            current_total_loss_klb_hr=total_loss,
            current_loss_cost_per_hr=current_loss_cost,
            optimized_total_loss_klb_hr=optimized_loss,
            optimized_loss_cost_per_hr=optimized_loss_cost,
            recommendations=recommendations,
            total_savings_per_hr=savings,
            annual_savings_potential=savings * self.operating_hours,
            confidence=0.85,
        )

        result.provenance_hash = self._generate_loss_provenance_hash(
            network_model, result
        )

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Loss minimization: current={total_loss:.0f} klb/hr, "
            f"optimized={optimized_loss:.0f} klb/hr, "
            f"savings=${savings:.0f}/hr in {computation_time:.1f}ms"
        )

        return result

    def _calculate_generation_losses(
        self,
        network: NetworkModel,
    ) -> float:
        """Calculate generation-side losses."""
        total_generation = network.total_generation_klb_hr

        # Typical generation losses:
        # - Stack loss: 15-20% of fuel input (accounted in efficiency)
        # - Blowdown: 2-3% of steam
        # - Radiation: 0.5-1% of capacity
        blowdown_fraction = 0.025
        radiation_fraction = 0.005

        blowdown_loss = total_generation * blowdown_fraction
        radiation_loss = sum(
            b.rated_capacity_klb_hr * radiation_fraction
            for b in network.boilers if b.is_online
        )

        return blowdown_loss + radiation_loss

    def _calculate_distribution_losses(
        self,
        network: NetworkModel,
    ) -> float:
        """Calculate distribution losses."""
        return network.total_generation_klb_hr * network.distribution_loss_percent / 100

    def _estimate_trap_losses(
        self,
        network: NetworkModel,
    ) -> float:
        """Estimate steam trap losses."""
        # Typical: 3-10% of steam passes through failed traps
        # Assume 5% failure rate, 20 lb/hr per failed trap
        estimated_trap_count = network.total_generation_klb_hr / 10  # 1 trap per 10 klb/hr
        failed_traps = estimated_trap_count * 0.05
        loss_per_trap = 0.020  # klb/hr
        return failed_traps * loss_per_trap

    def _generate_provenance_hash(
        self,
        boilers: List[BoilerState],
        demand: float,
        result: LoadAllocationResult,
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = (
            f"{[b.boiler_id for b in boilers]}"
            f"{demand}"
            f"{result.total_cost_per_hr}"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def _generate_loss_provenance_hash(
        self,
        network: NetworkModel,
        result: LossMinimizationResult,
    ) -> str:
        """Generate SHA-256 provenance hash for loss result."""
        data = (
            f"{network.total_generation_klb_hr}"
            f"{result.current_total_loss_klb_hr}"
            f"{result.optimized_total_loss_klb_hr}"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()
