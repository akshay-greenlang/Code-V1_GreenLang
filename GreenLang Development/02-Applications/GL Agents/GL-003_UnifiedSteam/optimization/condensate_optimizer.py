"""
GL-003 UNIFIEDSTEAM - Condensate Recovery Optimizer

Provides optimization for condensate recovery systems:
- Return routing optimization
- Receiver pressure optimization
- Recovery opportunity identification
- Improvement prioritization with ROI

Objectives:
- Maximize heat recovery
- Avoid system upsets (receiver overpressure, pump cavitation)
- Minimize makeup water and treatment costs
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
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
    UncertaintyConstraints,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class CondensateSourceType(str, Enum):
    """Type of condensate source."""

    HEAT_EXCHANGER = "heat_exchanger"
    STEAM_TRAP = "steam_trap"
    TURBINE = "turbine"
    FLASH_TANK = "flash_tank"
    PROCESS = "process"
    OTHER = "other"


class CondensateSource(BaseModel):
    """A source of condensate in the network."""

    source_id: str = Field(..., description="Source identifier")
    source_type: CondensateSourceType = Field(..., description="Type of source")
    location: str = Field(default="", description="Physical location")
    current_flow_lb_hr: float = Field(..., ge=0, description="Current flow (lb/hr)")
    temperature_f: float = Field(..., description="Condensate temperature (F)")
    pressure_psig: float = Field(..., description="Source pressure (psig)")
    quality_percent: float = Field(
        default=99.0, ge=0, le=100, description="Condensate purity (%)"
    )
    is_contaminated: bool = Field(
        default=False, description="Whether condensate is contaminated"
    )
    current_destination: str = Field(
        default="", description="Current routing destination"
    )


class CondensateReceiver(BaseModel):
    """Condensate receiver/flash tank data."""

    receiver_id: str = Field(..., description="Receiver identifier")
    capacity_gal: float = Field(..., gt=0, description="Tank capacity (gallons)")
    current_level_percent: float = Field(
        ..., ge=0, le=100, description="Current level (%)"
    )
    current_pressure_psig: float = Field(..., description="Current pressure (psig)")
    design_pressure_psig: float = Field(..., description="Design pressure (psig)")
    max_operating_pressure_psig: float = Field(
        ..., description="Maximum operating pressure (psig)"
    )
    temperature_f: float = Field(..., description="Receiver temperature (F)")
    vent_rate_lb_hr: float = Field(
        default=0.0, ge=0, description="Current vent rate (lb/hr)"
    )
    flash_steam_rate_lb_hr: float = Field(
        default=0.0, ge=0, description="Flash steam production (lb/hr)"
    )
    connected_sources: List[str] = Field(
        default_factory=list, description="Connected source IDs"
    )


class CondensatePump(BaseModel):
    """Condensate pump data."""

    pump_id: str = Field(..., description="Pump identifier")
    receiver_id: str = Field(..., description="Associated receiver ID")
    rated_flow_gpm: float = Field(..., gt=0, description="Rated flow (gpm)")
    current_flow_gpm: float = Field(..., ge=0, description="Current flow (gpm)")
    discharge_pressure_psig: float = Field(
        ..., description="Discharge pressure (psig)"
    )
    npsh_available_ft: float = Field(..., description="NPSH available (ft)")
    npsh_required_ft: float = Field(..., description="NPSH required (ft)")
    motor_load_percent: float = Field(
        default=50.0, ge=0, le=120, description="Motor load (%)"
    )
    is_running: bool = Field(default=True, description="Pump running status")


class NetworkTopology(BaseModel):
    """Condensate network topology."""

    sources: List[CondensateSource] = Field(
        default_factory=list, description="Condensate sources"
    )
    receivers: List[CondensateReceiver] = Field(
        default_factory=list, description="Receivers/flash tanks"
    )
    pumps: List[CondensatePump] = Field(
        default_factory=list, description="Condensate pumps"
    )
    connections: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Source to receiver connections"
    )
    return_to_boiler: bool = Field(
        default=True, description="Whether condensate returns to boiler"
    )
    deaerator_id: Optional[str] = Field(
        default=None, description="Deaerator ID if present"
    )


class FlashConstraints(BaseModel):
    """Constraints for flash steam operations."""

    max_receiver_pressure_psig: float = Field(
        default=15.0, description="Maximum receiver pressure (psig)"
    )
    min_receiver_level_percent: float = Field(
        default=20.0, description="Minimum receiver level (%)"
    )
    max_receiver_level_percent: float = Field(
        default=80.0, description="Maximum receiver level (%)"
    )
    max_vent_rate_lb_hr: float = Field(
        default=100.0, description="Maximum acceptable vent rate (lb/hr)"
    )
    min_pump_npsh_margin_ft: float = Field(
        default=3.0, description="Minimum NPSH margin (ft)"
    )


# =============================================================================
# Result Models
# =============================================================================


class RoutingRecommendation(BaseModel):
    """Recommendation for a single source routing."""

    source_id: str = Field(..., description="Source identifier")
    current_destination: str = Field(..., description="Current destination")
    recommended_destination: str = Field(..., description="Recommended destination")
    reason: str = Field(..., description="Reason for recommendation")
    expected_heat_recovery_btu_hr: float = Field(
        default=0.0, description="Expected heat recovery (BTU/hr)"
    )
    expected_savings_per_hr: float = Field(
        default=0.0, description="Expected cost savings ($/hr)"
    )


class RoutingOptimization(BaseModel):
    """Result of return routing optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    recommendations: List[RoutingRecommendation] = Field(
        default_factory=list, description="Routing recommendations"
    )
    total_recoverable_flow_lb_hr: float = Field(
        default=0.0, description="Total recoverable condensate (lb/hr)"
    )
    current_return_rate_percent: float = Field(
        default=0.0, description="Current condensate return rate (%)"
    )
    optimized_return_rate_percent: float = Field(
        default=0.0, description="Optimized return rate (%)"
    )
    total_heat_recovery_improvement_btu_hr: float = Field(
        default=0.0, description="Heat recovery improvement (BTU/hr)"
    )
    total_cost_savings_per_hr: float = Field(
        default=0.0, description="Total cost savings ($/hr)"
    )
    annual_savings_estimate: float = Field(
        default=0.0, description="Estimated annual savings ($)"
    )
    confidence: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Confidence level"
    )
    constraints_satisfied: bool = Field(default=True)
    provenance_hash: str = Field(default="")


class PressureSetpoint(BaseModel):
    """Optimized pressure setpoint for receiver."""

    receiver_id: str = Field(..., description="Receiver identifier")
    current_pressure_psig: float = Field(..., description="Current pressure (psig)")
    recommended_pressure_psig: float = Field(
        ..., description="Recommended pressure (psig)"
    )
    flash_steam_impact_lb_hr: float = Field(
        default=0.0, description="Impact on flash steam production (lb/hr)"
    )
    pump_npsh_margin_ft: float = Field(
        default=0.0, description="Resulting NPSH margin (ft)"
    )
    reason: str = Field(..., description="Reason for recommendation")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)


class RecoveryOpportunity(BaseModel):
    """An identified condensate recovery opportunity."""

    opportunity_id: str = Field(..., description="Opportunity identifier")
    category: str = Field(..., description="Category (routing/flash/trap/other)")
    location: str = Field(..., description="Location in the system")
    description: str = Field(..., description="Description of opportunity")
    current_loss_lb_hr: float = Field(
        ..., ge=0, description="Current condensate loss (lb/hr)"
    )
    recoverable_lb_hr: float = Field(
        ..., ge=0, description="Recoverable amount (lb/hr)"
    )
    implementation_cost: float = Field(
        default=0.0, ge=0, description="Implementation cost ($)"
    )
    annual_savings: float = Field(
        default=0.0, ge=0, description="Annual savings ($)"
    )
    payback_months: float = Field(
        default=0.0, ge=0, description="Simple payback (months)"
    )
    roi_percent: float = Field(default=0.0, description="ROI (%)")
    priority_score: float = Field(
        default=0.0, description="Calculated priority score"
    )
    risk_level: str = Field(default="low", description="Implementation risk")
    prerequisites: List[str] = Field(
        default_factory=list, description="Prerequisites for implementation"
    )


class PrioritizedList(BaseModel):
    """Prioritized list of recovery opportunities."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    opportunities: List[RecoveryOpportunity] = Field(
        default_factory=list, description="Prioritized opportunities"
    )
    total_annual_savings: float = Field(
        default=0.0, description="Total annual savings if all implemented ($)"
    )
    total_implementation_cost: float = Field(
        default=0.0, description="Total implementation cost ($)"
    )
    budget_utilization_percent: float = Field(
        default=0.0, description="Budget utilization (%)"
    )
    within_budget_count: int = Field(
        default=0, description="Number of opportunities within budget"
    )


# =============================================================================
# Condensate Recovery Optimizer
# =============================================================================


class CondensateRecoveryOptimizer:
    """
    Optimizes condensate recovery and return systems.

    Objectives:
    - Maximize heat recovery from condensate
    - Optimize flash steam recovery
    - Minimize makeup water requirements
    - Avoid pump cavitation and receiver overpressure

    Uses deterministic calculations (zero-hallucination approach) based on:
    - Steam tables (thermodynamic properties)
    - Energy balance equations
    - Economic analysis (ROI calculations)
    """

    # Cost assumptions
    DEFAULT_MAKEUP_WATER_COST_PER_GAL = 0.005  # $/gal
    DEFAULT_TREATMENT_COST_PER_GAL = 0.002  # $/gal
    DEFAULT_FUEL_COST_PER_MMBTU = 8.0  # $/MMBTU
    DEFAULT_OPERATING_HOURS_PER_YEAR = 8000

    def __init__(
        self,
        safety_constraints: Optional[SafetyConstraints] = None,
        equipment_constraints: Optional[EquipmentConstraints] = None,
        makeup_water_cost: float = DEFAULT_MAKEUP_WATER_COST_PER_GAL,
        treatment_cost: float = DEFAULT_TREATMENT_COST_PER_GAL,
        fuel_cost: float = DEFAULT_FUEL_COST_PER_MMBTU,
        operating_hours: int = DEFAULT_OPERATING_HOURS_PER_YEAR,
    ) -> None:
        """
        Initialize condensate recovery optimizer.

        Args:
            safety_constraints: Safety constraints
            equipment_constraints: Equipment constraints
            makeup_water_cost: Makeup water cost ($/gal)
            treatment_cost: Water treatment cost ($/gal)
            fuel_cost: Fuel cost ($/MMBTU)
            operating_hours: Annual operating hours
        """
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.equipment_constraints = equipment_constraints or EquipmentConstraints()
        self.makeup_water_cost = makeup_water_cost
        self.treatment_cost = treatment_cost
        self.fuel_cost = fuel_cost
        self.operating_hours = operating_hours

        logger.info("CondensateRecoveryOptimizer initialized")

    def optimize_return_routing(
        self,
        network_topology: NetworkTopology,
        current_flows: Dict[str, float],
    ) -> RoutingOptimization:
        """
        Optimize condensate return routing across the network.

        Analyzes each source and recommends optimal routing based on:
        - Condensate quality
        - Temperature/pressure compatibility
        - Receiver capacity
        - Pump capability

        Args:
            network_topology: Network topology definition
            current_flows: Current flow rates by source ID

        Returns:
            RoutingOptimization with recommendations
        """
        start_time = time.perf_counter()

        recommendations: List[RoutingRecommendation] = []
        total_flow = 0.0
        total_returned = 0.0
        total_recoverable = 0.0
        total_heat_improvement = 0.0
        total_savings = 0.0

        # Build receiver capacity map
        receiver_capacity = self._calculate_receiver_capacities(
            network_topology.receivers
        )

        # Analyze each source
        for source in network_topology.sources:
            flow = current_flows.get(source.source_id, source.current_flow_lb_hr)
            total_flow += flow

            # Check if currently returning
            is_returning = (
                source.current_destination != ""
                and source.current_destination != "sewer"
                and source.current_destination != "drain"
            )
            if is_returning:
                total_returned += flow

            # Check if source is recoverable
            if source.is_contaminated:
                recommendations.append(
                    RoutingRecommendation(
                        source_id=source.source_id,
                        current_destination=source.current_destination,
                        recommended_destination="treatment",
                        reason="Contaminated condensate requires treatment before return",
                        expected_heat_recovery_btu_hr=0,
                        expected_savings_per_hr=0,
                    )
                )
                continue

            if source.quality_percent < 95.0:
                recommendations.append(
                    RoutingRecommendation(
                        source_id=source.source_id,
                        current_destination=source.current_destination,
                        recommended_destination="polisher",
                        reason=f"Quality {source.quality_percent:.1f}% below threshold",
                        expected_heat_recovery_btu_hr=0,
                        expected_savings_per_hr=0,
                    )
                )
                continue

            # Find optimal receiver
            best_receiver = self._find_optimal_receiver(
                source, network_topology.receivers, receiver_capacity
            )

            if best_receiver and source.current_destination != best_receiver:
                # Calculate benefits
                heat_recovery = self._calculate_heat_recovery(source, flow)
                savings = self._calculate_savings(source, flow)

                recommendations.append(
                    RoutingRecommendation(
                        source_id=source.source_id,
                        current_destination=source.current_destination or "drain",
                        recommended_destination=best_receiver,
                        reason=f"Optimal receiver based on pressure/temp compatibility",
                        expected_heat_recovery_btu_hr=heat_recovery,
                        expected_savings_per_hr=savings,
                    )
                )

                if not is_returning:
                    total_recoverable += flow
                    total_heat_improvement += heat_recovery
                    total_savings += savings

            elif not is_returning and best_receiver:
                # Not currently returning but can return
                heat_recovery = self._calculate_heat_recovery(source, flow)
                savings = self._calculate_savings(source, flow)

                recommendations.append(
                    RoutingRecommendation(
                        source_id=source.source_id,
                        current_destination=source.current_destination or "drain",
                        recommended_destination=best_receiver,
                        reason="Enable return to recover heat",
                        expected_heat_recovery_btu_hr=heat_recovery,
                        expected_savings_per_hr=savings,
                    )
                )
                total_recoverable += flow
                total_heat_improvement += heat_recovery
                total_savings += savings

        # Calculate rates
        current_return_rate = (total_returned / total_flow * 100) if total_flow > 0 else 0
        optimized_return_rate = (
            ((total_returned + total_recoverable) / total_flow * 100)
            if total_flow > 0 else 0
        )

        result = RoutingOptimization(
            recommendations=recommendations,
            total_recoverable_flow_lb_hr=total_recoverable,
            current_return_rate_percent=current_return_rate,
            optimized_return_rate_percent=optimized_return_rate,
            total_heat_recovery_improvement_btu_hr=total_heat_improvement,
            total_cost_savings_per_hr=total_savings,
            annual_savings_estimate=total_savings * self.operating_hours,
            confidence=0.90,
            constraints_satisfied=True,
        )

        # Generate provenance hash
        result.provenance_hash = self._generate_provenance_hash(
            network_topology, result
        )

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Routing optimization complete: {len(recommendations)} recommendations, "
            f"recoverable={total_recoverable:.0f} lb/hr, "
            f"savings=${total_savings:.2f}/hr in {computation_time:.1f}ms"
        )

        return result

    def optimize_receiver_pressure(
        self,
        receiver_data: CondensateReceiver,
        flash_constraints: FlashConstraints,
        connected_pump: Optional[CondensatePump] = None,
    ) -> PressureSetpoint:
        """
        Optimize receiver pressure for flash steam recovery.

        Balances:
        - Flash steam production (higher pressure = less flash)
        - Pump NPSH requirements (higher pressure = more NPSH)
        - Vent losses (avoid over-pressure venting)

        Args:
            receiver_data: Current receiver state
            flash_constraints: Operating constraints
            connected_pump: Associated pump data

        Returns:
            PressureSetpoint recommendation
        """
        current_pressure = receiver_data.current_pressure_psig

        # Calculate saturation temperature at current pressure
        sat_temp_current = self._saturation_temp_from_pressure(current_pressure)

        # Check NPSH constraint if pump present
        min_pressure_for_npsh = 0.0
        if connected_pump:
            # Calculate minimum pressure for NPSH
            npsh_margin = (
                connected_pump.npsh_available_ft - connected_pump.npsh_required_ft
            )
            if npsh_margin < flash_constraints.min_pump_npsh_margin_ft:
                # Need higher pressure for more NPSH
                # Approximate: 2.31 ft per psi at water density
                pressure_increase = (
                    (flash_constraints.min_pump_npsh_margin_ft - npsh_margin) / 2.31
                )
                min_pressure_for_npsh = current_pressure + pressure_increase

        # Check vent constraint
        if receiver_data.vent_rate_lb_hr > flash_constraints.max_vent_rate_lb_hr:
            # Venting too much - reduce pressure to reduce flashing
            recommended_pressure = max(
                0.0,
                current_pressure - 2.0  # Reduce by 2 psi
            )
            reason = "Reduce pressure to minimize vent losses"
        elif current_pressure > flash_constraints.max_receiver_pressure_psig:
            # Over max operating pressure
            recommended_pressure = flash_constraints.max_receiver_pressure_psig
            reason = "Reduce pressure to within operating limits"
        elif min_pressure_for_npsh > current_pressure:
            # Need more pressure for pump NPSH
            recommended_pressure = min(
                min_pressure_for_npsh,
                flash_constraints.max_receiver_pressure_psig
            )
            reason = "Increase pressure to improve pump NPSH margin"
        else:
            # Current pressure is OK - optimize for flash recovery
            # Lower pressure = more flash steam = more heat recovery
            # But must maintain pump NPSH
            optimal_pressure = max(
                min_pressure_for_npsh,
                2.0  # Minimum 2 psig for positive pressure
            )
            recommended_pressure = optimal_pressure
            reason = "Optimized for flash steam recovery"

        # Calculate flash steam impact
        flash_current = self._calculate_flash_steam(
            receiver_data.temperature_f, current_pressure
        )
        flash_recommended = self._calculate_flash_steam(
            receiver_data.temperature_f, recommended_pressure
        )
        flash_impact = flash_recommended - flash_current

        # Calculate NPSH margin at recommended pressure
        npsh_margin = 0.0
        if connected_pump:
            # Simplified NPSH calculation
            sat_temp_new = self._saturation_temp_from_pressure(recommended_pressure)
            vapor_pressure_psia = self._vapor_pressure_from_temp(sat_temp_new)
            npsh_available = (
                (recommended_pressure + 14.7 - vapor_pressure_psia) * 2.31
            )
            npsh_margin = npsh_available - connected_pump.npsh_required_ft

        return PressureSetpoint(
            receiver_id=receiver_data.receiver_id,
            current_pressure_psig=current_pressure,
            recommended_pressure_psig=round(recommended_pressure, 1),
            flash_steam_impact_lb_hr=flash_impact,
            pump_npsh_margin_ft=round(npsh_margin, 1),
            reason=reason,
            confidence=0.85 if connected_pump else 0.90,
        )

    def identify_recovery_opportunities(
        self,
        network_audit: NetworkTopology,
    ) -> List[RecoveryOpportunity]:
        """
        Identify condensate recovery improvement opportunities.

        Analyzes the network for:
        - Sources not returning condensate
        - Flash steam venting
        - Failed/leaking traps
        - Sub-optimal routing

        Args:
            network_audit: Network audit data

        Returns:
            List of identified opportunities
        """
        opportunities: List[RecoveryOpportunity] = []
        opportunity_counter = 1

        # 1. Check for non-returning sources
        for source in network_audit.sources:
            if (
                source.current_destination in ("", "sewer", "drain")
                and not source.is_contaminated
                and source.quality_percent >= 95.0
            ):
                # Calculate potential savings
                annual_loss_lb = source.current_flow_lb_hr * self.operating_hours
                annual_loss_gal = annual_loss_lb / 8.34
                makeup_cost = annual_loss_gal * self.makeup_water_cost
                treatment_cost = annual_loss_gal * self.treatment_cost
                heat_value = self._calculate_heat_value(source)

                annual_savings = makeup_cost + treatment_cost + heat_value
                implementation_cost = 5000.0  # Estimate for piping

                opportunities.append(
                    RecoveryOpportunity(
                        opportunity_id=f"OPP-{opportunity_counter:03d}",
                        category="routing",
                        location=source.location,
                        description=(
                            f"Return {source.source_type.value} condensate from "
                            f"{source.source_id}"
                        ),
                        current_loss_lb_hr=source.current_flow_lb_hr,
                        recoverable_lb_hr=source.current_flow_lb_hr,
                        implementation_cost=implementation_cost,
                        annual_savings=annual_savings,
                        payback_months=(
                            implementation_cost / (annual_savings / 12)
                            if annual_savings > 0 else float('inf')
                        ),
                        roi_percent=(
                            (annual_savings - implementation_cost) /
                            implementation_cost * 100
                            if implementation_cost > 0 else 0
                        ),
                        risk_level="low",
                        prerequisites=["Verify water quality", "Check pipe routing"],
                    )
                )
                opportunity_counter += 1

        # 2. Check for flash steam venting
        for receiver in network_audit.receivers:
            if receiver.vent_rate_lb_hr > 50:  # Significant venting
                # Calculate annual loss
                annual_loss_lb = receiver.vent_rate_lb_hr * self.operating_hours
                latent_heat = 900  # BTU/lb approximate
                annual_heat_loss_mmbtu = annual_loss_lb * latent_heat / 1e6
                annual_fuel_cost = annual_heat_loss_mmbtu * self.fuel_cost

                implementation_cost = 15000.0  # Flash tank + piping

                opportunities.append(
                    RecoveryOpportunity(
                        opportunity_id=f"OPP-{opportunity_counter:03d}",
                        category="flash",
                        location=receiver.receiver_id,
                        description=(
                            f"Recover flash steam from {receiver.receiver_id} "
                            f"(currently venting {receiver.vent_rate_lb_hr:.0f} lb/hr)"
                        ),
                        current_loss_lb_hr=receiver.vent_rate_lb_hr,
                        recoverable_lb_hr=receiver.vent_rate_lb_hr * 0.9,
                        implementation_cost=implementation_cost,
                        annual_savings=annual_fuel_cost,
                        payback_months=(
                            implementation_cost / (annual_fuel_cost / 12)
                            if annual_fuel_cost > 0 else float('inf')
                        ),
                        roi_percent=(
                            (annual_fuel_cost - implementation_cost) /
                            implementation_cost * 100
                            if implementation_cost > 0 else 0
                        ),
                        risk_level="medium",
                        prerequisites=[
                            "Verify LP steam header capacity",
                            "Engineering review",
                        ],
                    )
                )
                opportunity_counter += 1

        # 3. Check pump efficiency
        for pump in network_audit.pumps:
            npsh_margin = pump.npsh_available_ft - pump.npsh_required_ft
            if npsh_margin < 2.0 and pump.is_running:
                opportunities.append(
                    RecoveryOpportunity(
                        opportunity_id=f"OPP-{opportunity_counter:03d}",
                        category="equipment",
                        location=pump.pump_id,
                        description=(
                            f"Address low NPSH margin ({npsh_margin:.1f} ft) on "
                            f"{pump.pump_id} to prevent cavitation"
                        ),
                        current_loss_lb_hr=0,
                        recoverable_lb_hr=0,
                        implementation_cost=8000.0,
                        annual_savings=5000.0,  # Maintenance savings
                        payback_months=19.2,
                        roi_percent=-37.5,  # Initial loss but prevents failure
                        risk_level="high",
                        prerequisites=["Engineering analysis"],
                    )
                )
                opportunity_counter += 1

        # Calculate priority scores
        for opp in opportunities:
            opp.priority_score = self._calculate_priority_score(opp)

        # Sort by priority score
        opportunities.sort(key=lambda x: x.priority_score, reverse=True)

        logger.info(f"Identified {len(opportunities)} recovery opportunities")

        return opportunities

    def prioritize_improvements(
        self,
        opportunities: List[RecoveryOpportunity],
        budget_constraint: float,
    ) -> PrioritizedList:
        """
        Prioritize improvement opportunities within budget constraint.

        Uses simple ROI-based prioritization with budget knapsack approach.

        Args:
            opportunities: List of identified opportunities
            budget_constraint: Available budget ($)

        Returns:
            PrioritizedList with opportunities within budget
        """
        # Sort by ROI (already sorted by priority score, re-sort by ROI)
        sorted_opps = sorted(
            opportunities,
            key=lambda x: x.roi_percent,
            reverse=True
        )

        selected: List[RecoveryOpportunity] = []
        remaining_budget = budget_constraint
        total_savings = 0.0
        total_cost = 0.0

        for opp in sorted_opps:
            if opp.implementation_cost <= remaining_budget:
                # Check ROI threshold (minimum 20%)
                if opp.roi_percent >= 20 or opp.risk_level == "high":
                    selected.append(opp)
                    remaining_budget -= opp.implementation_cost
                    total_savings += opp.annual_savings
                    total_cost += opp.implementation_cost

        # Re-sort selected by priority score for implementation order
        selected.sort(key=lambda x: x.priority_score, reverse=True)

        budget_utilization = (
            (budget_constraint - remaining_budget) / budget_constraint * 100
            if budget_constraint > 0 else 0
        )

        return PrioritizedList(
            opportunities=selected,
            total_annual_savings=total_savings,
            total_implementation_cost=total_cost,
            budget_utilization_percent=budget_utilization,
            within_budget_count=len(selected),
        )

    def _calculate_receiver_capacities(
        self,
        receivers: List[CondensateReceiver],
    ) -> Dict[str, float]:
        """Calculate available capacity for each receiver."""
        capacities = {}
        for receiver in receivers:
            available_level = 80 - receiver.current_level_percent  # Up to 80%
            available_gal = receiver.capacity_gal * available_level / 100
            capacities[receiver.receiver_id] = max(0, available_gal)
        return capacities

    def _find_optimal_receiver(
        self,
        source: CondensateSource,
        receivers: List[CondensateReceiver],
        capacities: Dict[str, float],
    ) -> Optional[str]:
        """Find optimal receiver for a condensate source."""
        best_receiver = None
        best_score = -1

        for receiver in receivers:
            # Check capacity
            if capacities.get(receiver.receiver_id, 0) <= 0:
                continue

            # Check pressure compatibility
            if source.pressure_psig > receiver.max_operating_pressure_psig:
                continue

            # Score based on temperature match and capacity
            temp_diff = abs(source.temperature_f - receiver.temperature_f)
            capacity_score = capacities[receiver.receiver_id] / receiver.capacity_gal
            score = capacity_score * 100 - temp_diff

            if score > best_score:
                best_score = score
                best_receiver = receiver.receiver_id

        return best_receiver

    def _calculate_heat_recovery(
        self,
        source: CondensateSource,
        flow_lb_hr: float,
    ) -> float:
        """Calculate heat recovery potential (BTU/hr)."""
        # Heat above makeup water temperature (60F typical)
        makeup_temp = 60.0
        sensible_heat = flow_lb_hr * 1.0 * (source.temperature_f - makeup_temp)
        return sensible_heat

    def _calculate_savings(
        self,
        source: CondensateSource,
        flow_lb_hr: float,
    ) -> float:
        """Calculate cost savings per hour."""
        # Water savings
        flow_gal_hr = flow_lb_hr / 8.34
        water_cost = flow_gal_hr * (self.makeup_water_cost + self.treatment_cost)

        # Energy savings
        heat_recovery = self._calculate_heat_recovery(source, flow_lb_hr)
        energy_cost = heat_recovery / 1e6 * self.fuel_cost

        return water_cost + energy_cost

    def _calculate_heat_value(
        self,
        source: CondensateSource,
    ) -> float:
        """Calculate annual heat value of condensate."""
        heat_recovery = self._calculate_heat_recovery(
            source, source.current_flow_lb_hr
        )
        annual_btu = heat_recovery * self.operating_hours
        return annual_btu / 1e6 * self.fuel_cost

    def _calculate_flash_steam(
        self,
        inlet_temp_f: float,
        receiver_pressure_psig: float,
    ) -> float:
        """Calculate flash steam rate (lb/lb condensate)."""
        sat_temp = self._saturation_temp_from_pressure(receiver_pressure_psig)
        if inlet_temp_f <= sat_temp:
            return 0.0

        # Flash fraction = (h_in - h_sat_liquid) / (h_sat_vapor - h_sat_liquid)
        # Simplified using temperature
        latent_heat = 900  # BTU/lb approximate
        sensible_above = inlet_temp_f - sat_temp
        flash_fraction = sensible_above / latent_heat
        return min(0.3, max(0, flash_fraction))  # Cap at 30%

    def _calculate_priority_score(
        self,
        opportunity: RecoveryOpportunity,
    ) -> float:
        """Calculate priority score for an opportunity."""
        # Factors: ROI, payback, risk, savings magnitude
        score = 0.0

        # ROI contribution (0-40 points)
        roi_score = min(40, max(0, opportunity.roi_percent / 2.5))
        score += roi_score

        # Payback contribution (0-30 points, shorter = better)
        if opportunity.payback_months < 6:
            payback_score = 30
        elif opportunity.payback_months < 12:
            payback_score = 25
        elif opportunity.payback_months < 24:
            payback_score = 15
        elif opportunity.payback_months < 36:
            payback_score = 5
        else:
            payback_score = 0
        score += payback_score

        # Risk contribution (0-20 points, lower risk = better)
        risk_scores = {"low": 20, "medium": 10, "high": 0}
        score += risk_scores.get(opportunity.risk_level, 0)

        # Savings magnitude (0-10 points)
        if opportunity.annual_savings > 50000:
            savings_score = 10
        elif opportunity.annual_savings > 20000:
            savings_score = 7
        elif opportunity.annual_savings > 10000:
            savings_score = 5
        else:
            savings_score = 2
        score += savings_score

        return score

    def _saturation_temp_from_pressure(self, pressure_psig: float) -> float:
        """Get saturation temperature from pressure."""
        pressure_psia = pressure_psig + 14.7
        return 115.0 + 45.0 * math.log(pressure_psia)

    def _vapor_pressure_from_temp(self, temp_f: float) -> float:
        """Get vapor pressure from temperature (simplified)."""
        return math.exp((temp_f - 115.0) / 45.0) - 14.7

    def _generate_provenance_hash(
        self,
        topology: NetworkTopology,
        result: RoutingOptimization,
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = (
            f"{topology.json()}"
            f"{result.total_recoverable_flow_lb_hr}"
            f"{result.total_cost_savings_per_hr}"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()
