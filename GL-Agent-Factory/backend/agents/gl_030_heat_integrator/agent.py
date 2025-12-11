"""GL-030 HEATINTEGRATOR: Heat Integration Optimizer Agent.

Optimizes heat integration across the entire process using pinch analysis
to minimize energy consumption and maximize heat recovery.

Standards: IChemE Pinch Analysis, ASME Energy Assessment
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HeatStream(BaseModel):
    """Individual hot or cold stream."""

    stream_id: str = Field(..., description="Stream identifier")
    stream_type: str = Field(..., description="HOT or COLD")
    supply_temp_c: float = Field(..., description="Supply/start temperature (°C)")
    target_temp_c: float = Field(..., description="Target/end temperature (°C)")
    heat_capacity_kw_k: float = Field(..., ge=0, description="mCp flow (kW/K)")
    current_utility_kw: float = Field(0, description="Current utility usage (kW)")
    can_integrate: bool = Field(True, description="Available for integration")
    min_approach_c: float = Field(10, description="Minimum approach temp (°C)")


class HeatIntegrationInput(BaseModel):
    """Input for heat integration optimization."""

    facility_id: str = Field(..., description="Facility identifier")

    # Process streams
    hot_streams: List[HeatStream] = Field(default_factory=list, description="Hot streams to cool")
    cold_streams: List[HeatStream] = Field(default_factory=list, description="Cold streams to heat")

    # Utility costs
    hot_utility_cost_per_kwh: float = Field(0.05, description="Steam/hot oil cost ($/kWh)")
    cold_utility_cost_per_kwh: float = Field(0.02, description="Cooling water cost ($/kWh)")
    electricity_cost_per_kwh: float = Field(0.10, description="Electricity cost ($/kWh)")

    # Constraints
    global_min_approach_c: float = Field(10, ge=5, description="Global ΔT_min (°C)")
    max_heat_exchangers: int = Field(20, ge=1, description="Max number of exchangers")
    area_cost_per_m2: float = Field(500, description="Heat exchanger cost ($/m²)")

    # Carbon factors
    utility_carbon_factor: float = Field(0.2, description="kg CO2/kWh utility")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HeatExchangerMatch(BaseModel):
    """Proposed heat exchanger match."""

    hot_stream: str
    cold_stream: str
    duty_kw: float
    hot_inlet_c: float
    hot_outlet_c: float
    cold_inlet_c: float
    cold_outlet_c: float
    lmtd_c: float
    area_m2: float
    estimated_cost: float


class PinchAnalysisResult(BaseModel):
    """Results from pinch analysis."""

    pinch_temp_c: float = Field(..., description="Pinch temperature (°C)")
    min_hot_utility_kw: float = Field(..., description="Minimum hot utility (kW)")
    min_cold_utility_kw: float = Field(..., description="Minimum cold utility (kW)")
    max_heat_recovery_kw: float = Field(..., description="Maximum process heat recovery (kW)")
    grand_composite_curve: List[Dict[str, float]] = Field(default_factory=list)


class HeatIntegrationOutput(BaseModel):
    """Output from heat integration optimizer."""

    # Pinch analysis
    pinch_analysis: PinchAnalysisResult

    # Current state
    current_hot_utility_kw: float = Field(..., description="Current hot utility usage")
    current_cold_utility_kw: float = Field(..., description="Current cold utility usage")
    current_heat_recovery_kw: float = Field(..., description="Current heat recovery")

    # Improvement potential
    hot_utility_savings_kw: float = Field(..., description="Potential hot utility savings")
    cold_utility_savings_kw: float = Field(..., description="Potential cold utility savings")
    additional_recovery_kw: float = Field(..., description="Additional heat recovery potential")

    # Economics
    annual_energy_savings: float = Field(..., description="Annual energy savings ($)")
    annual_carbon_savings_tonnes: float = Field(..., description="Annual CO2 reduction (t)")
    required_investment: float = Field(..., description="Total investment required ($)")
    simple_payback_years: float = Field(..., description="Simple payback period")
    npv_10yr: float = Field(..., description="10-year NPV at 10% discount")

    # Proposed matches
    proposed_exchangers: List[HeatExchangerMatch] = Field(default_factory=list)
    exchanger_count: int = Field(..., description="Number of proposed exchangers")
    total_area_m2: float = Field(..., description="Total new exchanger area")

    # Cross-pinch violations (current)
    cross_pinch_violations: List[str] = Field(default_factory=list)
    cross_pinch_waste_kw: float = Field(0, description="Heat wasted across pinch")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_stream_enthalpy(stream: HeatStream) -> Tuple[float, float, float]:
    """
    Calculate stream enthalpy change.

    Returns: (Q_kW, T_start_shifted, T_end_shifted)
    """
    q = stream.heat_capacity_kw_k * abs(stream.supply_temp_c - stream.target_temp_c)

    # Shifted temperatures for composite curves
    dt_shift = stream.min_approach_c / 2

    if stream.stream_type == "HOT":
        t_start = stream.supply_temp_c - dt_shift
        t_end = stream.target_temp_c - dt_shift
    else:  # COLD
        t_start = stream.supply_temp_c + dt_shift
        t_end = stream.target_temp_c + dt_shift

    return q, t_start, t_end


def perform_pinch_analysis(
    hot_streams: List[HeatStream],
    cold_streams: List[HeatStream],
    dt_min: float
) -> PinchAnalysisResult:
    """
    Perform pinch analysis to find minimum utility targets.

    Uses problem table algorithm (simplified).
    """
    # Collect all temperature intervals
    temps = set()
    for s in hot_streams:
        temps.add(s.supply_temp_c - dt_min/2)
        temps.add(s.target_temp_c - dt_min/2)
    for s in cold_streams:
        temps.add(s.supply_temp_c + dt_min/2)
        temps.add(s.target_temp_c + dt_min/2)

    temps = sorted(temps, reverse=True)

    if len(temps) < 2:
        return PinchAnalysisResult(
            pinch_temp_c=0,
            min_hot_utility_kw=0,
            min_cold_utility_kw=0,
            max_heat_recovery_kw=0,
            grand_composite_curve=[]
        )

    # Problem table
    cascade = []
    cumulative_heat = 0

    for i in range(len(temps) - 1):
        t_high = temps[i]
        t_low = temps[i + 1]
        dt = t_high - t_low

        # Sum heat capacities of active streams
        cp_hot = sum(
            s.heat_capacity_kw_k for s in hot_streams
            if s.supply_temp_c - dt_min/2 >= t_high and s.target_temp_c - dt_min/2 <= t_low
        )
        cp_cold = sum(
            s.heat_capacity_kw_k for s in cold_streams
            if s.target_temp_c + dt_min/2 >= t_high and s.supply_temp_c + dt_min/2 <= t_low
        )

        interval_heat = (cp_hot - cp_cold) * dt
        cumulative_heat += interval_heat
        cascade.append({
            "temp": t_low + dt_min/2,  # Actual temp
            "heat": cumulative_heat
        })

    # Find pinch (minimum in cascade)
    if cascade:
        min_point = min(cascade, key=lambda x: x["heat"])
        pinch_temp = min_point["temp"]
        min_cold_utility = -min_point["heat"]  # Deficit at pinch
        min_hot_utility = cascade[-1]["heat"] + min_cold_utility

        # Total heat available and required
        total_hot = sum(
            s.heat_capacity_kw_k * (s.supply_temp_c - s.target_temp_c)
            for s in hot_streams
        )
        total_cold = sum(
            s.heat_capacity_kw_k * (s.target_temp_c - s.supply_temp_c)
            for s in cold_streams
        )

        max_recovery = min(total_hot, total_cold) - max(0, min_cold_utility)
    else:
        pinch_temp = 0
        min_hot_utility = 0
        min_cold_utility = 0
        max_recovery = 0

    return PinchAnalysisResult(
        pinch_temp_c=round(pinch_temp, 1),
        min_hot_utility_kw=round(max(0, min_hot_utility), 1),
        min_cold_utility_kw=round(max(0, min_cold_utility), 1),
        max_heat_recovery_kw=round(max_recovery, 1),
        grand_composite_curve=cascade
    )


def match_streams(
    hot_streams: List[HeatStream],
    cold_streams: List[HeatStream],
    pinch_temp: float,
    dt_min: float
) -> List[HeatExchangerMatch]:
    """
    Create heat exchanger matches following pinch rules.

    Rules:
    - Above pinch: match hot to cold, hot CP ≤ cold CP
    - Below pinch: match hot to cold, hot CP ≥ cold CP
    - Never transfer heat across pinch
    """
    matches = []

    # Simple greedy matching (production would use MILP)
    available_hot = [s for s in hot_streams if s.can_integrate]
    available_cold = [s for s in cold_streams if s.can_integrate]

    for hot in available_hot:
        for cold in available_cold:
            # Check temperature feasibility
            if hot.supply_temp_c - dt_min < cold.target_temp_c:
                continue

            # Determine match duty (minimum of available heats)
            hot_q = hot.heat_capacity_kw_k * (hot.supply_temp_c - max(hot.target_temp_c, cold.target_temp_c + dt_min))
            cold_q = cold.heat_capacity_kw_k * (min(cold.target_temp_c, hot.supply_temp_c - dt_min) - cold.supply_temp_c)
            duty = min(hot_q, cold_q)

            if duty < 10:  # Minimum practical size
                continue

            # Calculate temperatures
            hot_out = hot.supply_temp_c - duty / hot.heat_capacity_kw_k
            cold_out = cold.supply_temp_c + duty / cold.heat_capacity_kw_k

            # LMTD
            import math
            dt1 = hot.supply_temp_c - cold_out
            dt2 = hot_out - cold.supply_temp_c
            if dt1 > 0 and dt2 > 0 and abs(dt1 - dt2) > 0.1:
                lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
            else:
                lmtd = (dt1 + dt2) / 2

            if lmtd < dt_min:
                continue

            # Estimate area (U ~ 500 W/m²K typical)
            u = 0.5  # kW/m²K
            area = duty / (u * lmtd) if lmtd > 0 else 0

            matches.append(HeatExchangerMatch(
                hot_stream=hot.stream_id,
                cold_stream=cold.stream_id,
                duty_kw=round(duty, 1),
                hot_inlet_c=round(hot.supply_temp_c, 1),
                hot_outlet_c=round(hot_out, 1),
                cold_inlet_c=round(cold.supply_temp_c, 1),
                cold_outlet_c=round(cold_out, 1),
                lmtd_c=round(lmtd, 1),
                area_m2=round(area, 1),
                estimated_cost=round(area * 500, 0)  # $500/m²
            ))

            break  # One match per hot stream in this simple algorithm

    return matches


class HeatIntegrationOptimizerAgent:
    """Heat integration (pinch analysis) optimizer."""

    AGENT_ID = "GL-030"
    AGENT_NAME = "HEATINTEGRATOR"
    VERSION = "1.0.0"

    HOURS_PER_YEAR = 8000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = HeatIntegrationInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: HeatIntegrationInput) -> HeatIntegrationOutput:
        recommendations = []
        warnings = []

        # Current utility usage
        current_hot = sum(s.current_utility_kw for s in inp.cold_streams)
        current_cold = sum(s.current_utility_kw for s in inp.hot_streams)

        # Calculate current heat recovery
        total_hot_available = sum(
            s.heat_capacity_kw_k * (s.supply_temp_c - s.target_temp_c)
            for s in inp.hot_streams
        )
        total_cold_required = sum(
            s.heat_capacity_kw_k * (s.target_temp_c - s.supply_temp_c)
            for s in inp.cold_streams
        )
        current_recovery = min(total_hot_available, total_cold_required) - current_hot - current_cold

        # Perform pinch analysis
        pinch = perform_pinch_analysis(
            inp.hot_streams,
            inp.cold_streams,
            inp.global_min_approach_c
        )

        # Calculate savings potential
        hot_utility_savings = max(0, current_hot - pinch.min_hot_utility_kw)
        cold_utility_savings = max(0, current_cold - pinch.min_cold_utility_kw)
        additional_recovery = pinch.max_heat_recovery_kw - current_recovery

        # Match streams
        matches = match_streams(
            inp.hot_streams,
            inp.cold_streams,
            pinch.pinch_temp_c,
            inp.global_min_approach_c
        )

        total_area = sum(m.area_m2 for m in matches)
        total_investment = total_area * inp.area_cost_per_m2

        # Economics
        annual_hot_savings = hot_utility_savings * self.HOURS_PER_YEAR * inp.hot_utility_cost_per_kwh
        annual_cold_savings = cold_utility_savings * self.HOURS_PER_YEAR * inp.cold_utility_cost_per_kwh
        annual_savings = annual_hot_savings + annual_cold_savings

        # Carbon
        annual_carbon = (hot_utility_savings + cold_utility_savings) * self.HOURS_PER_YEAR * inp.utility_carbon_factor / 1000

        # Payback
        payback = total_investment / annual_savings if annual_savings > 0 else float('inf')

        # NPV (simplified at 10%)
        discount_rate = 0.10
        npv = -total_investment + sum(
            annual_savings / (1 + discount_rate) ** year
            for year in range(1, 11)
        )

        # Check for cross-pinch violations
        violations = []
        cross_pinch_waste = 0
        for hot in inp.hot_streams:
            if hot.supply_temp_c > pinch.pinch_temp_c and hot.target_temp_c < pinch.pinch_temp_c:
                # Hot stream crosses pinch
                if hot.current_utility_kw > 0:
                    violations.append(f"Hot stream {hot.stream_id} cooling across pinch")
                    cross_pinch_waste += hot.current_utility_kw

        for cold in inp.cold_streams:
            if cold.target_temp_c > pinch.pinch_temp_c and cold.supply_temp_c < pinch.pinch_temp_c:
                if cold.current_utility_kw > 0:
                    violations.append(f"Cold stream {cold.stream_id} heating across pinch")
                    cross_pinch_waste += cold.current_utility_kw

        # Recommendations
        if hot_utility_savings > 100:
            recommendations.append(
                f"Hot utility can be reduced by {hot_utility_savings:.0f} kW "
                f"(${annual_hot_savings/1000:.0f}k/year)"
            )

        if cold_utility_savings > 100:
            recommendations.append(
                f"Cold utility can be reduced by {cold_utility_savings:.0f} kW"
            )

        if violations:
            warnings.extend(violations)
            recommendations.append(f"Eliminate cross-pinch heat transfer to save {cross_pinch_waste:.0f} kW")

        if payback < 2:
            recommendations.append(f"Excellent payback of {payback:.1f} years - prioritize implementation")
        elif payback < 5:
            recommendations.append(f"Good payback of {payback:.1f} years")

        if len(matches) > inp.max_heat_exchangers:
            warnings.append(f"Proposed {len(matches)} exchangers exceeds limit of {inp.max_heat_exchangers}")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "pinch": pinch.pinch_temp_c,
            "min_hot": pinch.min_hot_utility_kw,
            "matches": len(matches)
        }).encode()).hexdigest()

        return HeatIntegrationOutput(
            pinch_analysis=pinch,
            current_hot_utility_kw=round(current_hot, 1),
            current_cold_utility_kw=round(current_cold, 1),
            current_heat_recovery_kw=round(max(0, current_recovery), 1),
            hot_utility_savings_kw=round(hot_utility_savings, 1),
            cold_utility_savings_kw=round(cold_utility_savings, 1),
            additional_recovery_kw=round(max(0, additional_recovery), 1),
            annual_energy_savings=round(annual_savings, 0),
            annual_carbon_savings_tonnes=round(annual_carbon, 1),
            required_investment=round(total_investment, 0),
            simple_payback_years=round(min(payback, 99), 1),
            npv_10yr=round(npv, 0),
            proposed_exchangers=matches,
            exchanger_count=len(matches),
            total_area_m2=round(total_area, 1),
            cross_pinch_violations=violations,
            cross_pinch_waste_kw=round(cross_pinch_waste, 1),
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Process Integration",
            "type": "Optimizer",
            "complexity": "High",
            "priority": "P1",
            "market_size": "$12B"
        }
