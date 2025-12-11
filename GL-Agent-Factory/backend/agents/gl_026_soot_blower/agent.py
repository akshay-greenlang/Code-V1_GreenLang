"""GL-026 SOOTBLAST: Soot Blower Controller Agent.

Optimizes soot blowing to maintain heat transfer while minimizing
steam consumption and tube erosion.

Standards: ASME PTC 4, EPRI Guidelines
"""
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SootBlowerInput(BaseModel):
    """Input for soot blower optimization."""

    # Furnace/boiler conditions
    equipment_id: str = Field(..., description="Boiler/furnace ID")
    boiler_load_pct: float = Field(..., ge=0, le=110)
    fuel_type: str = Field("coal", description="coal, oil, gas, biomass")
    fuel_ash_pct: float = Field(10.0, ge=0, le=50, description="Fuel ash content (%)")

    # Heat transfer monitoring
    furnace_exit_gas_temp_c: float = Field(..., description="FEGT (°C)")
    design_fegt_c: float = Field(..., description="Design FEGT (°C)")
    economizer_gas_temp_out_c: float = Field(..., description="Economizer exit temp (°C)")
    steam_temp_deviation_c: float = Field(0.0, description="Steam temp from setpoint (°C)")

    # Cleanliness factors (1.0 = clean)
    superheater_cleanliness: float = Field(1.0, ge=0.5, le=1.0)
    reheater_cleanliness: float = Field(1.0, ge=0.5, le=1.0)
    economizer_cleanliness: float = Field(1.0, ge=0.5, le=1.0)
    air_preheater_cleanliness: float = Field(1.0, ge=0.5, le=1.0)

    # Soot blower status
    blower_zones: List[str] = Field(default=["furnace", "superheater", "reheater", "economizer", "aph"])
    hours_since_last_blow: Dict[str, float] = Field(default_factory=dict)
    steam_pressure_bar: float = Field(10.0, description="Soot blowing steam pressure")
    steam_cost_per_ton: float = Field(15.0, description="Steam cost ($/ton)")

    # Operating constraints
    min_interval_hours: float = Field(2.0, description="Minimum between blows")
    max_blowers_simultaneous: int = Field(2, description="Max simultaneous operation")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BlowingAction(BaseModel):
    """Soot blowing action for a zone."""
    zone: str
    action: str  # BLOW_NOW, SCHEDULE, SKIP
    priority: int  # 1=highest
    reason: str
    estimated_steam_kg: float
    estimated_cost: float


class SootBlowerOutput(BaseModel):
    """Output from soot blower controller."""

    # Recommended actions
    actions: List[BlowingAction] = Field(default_factory=list)
    immediate_blow_zones: List[str] = Field(default_factory=list)
    scheduled_blow_zones: List[str] = Field(default_factory=list)

    # Fouling analysis
    overall_fouling_index: float = Field(..., description="0-100 fouling severity")
    heat_rate_penalty_pct: float = Field(..., description="Efficiency loss from fouling")
    estimated_fuel_waste_per_hour: float = Field(..., description="$/hr from fouling")

    # Optimization metrics
    optimal_blow_interval_hours: float = Field(..., description="Recommended interval")
    steam_saved_vs_fixed_schedule_pct: float = Field(..., description="Steam savings")
    estimated_daily_steam_usage_kg: float = Field(..., description="Daily steam use")
    estimated_daily_cost: float = Field(..., description="Daily soot blowing cost")

    # Tube erosion monitoring
    erosion_risk: str = Field(..., description="LOW, MEDIUM, HIGH")
    recommended_pressure_bar: float = Field(..., description="Optimal steam pressure")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_fouling_index(
    cleanliness_factors: Dict[str, float],
    fegt_actual: float,
    fegt_design: float
) -> float:
    """
    Calculate overall fouling index (0-100).

    Index considers:
    - Individual surface cleanliness
    - FEGT deviation (higher = more fouling)
    """
    avg_cleanliness = sum(cleanliness_factors.values()) / len(cleanliness_factors)

    # FEGT contribution (higher temp = more fouling blocking heat transfer)
    fegt_deviation = max(0, fegt_actual - fegt_design)
    fegt_factor = min(1.0, fegt_deviation / 50)  # 50°C deviation = max contribution

    # Fouling index: 0 = clean, 100 = severely fouled
    fouling_index = (1 - avg_cleanliness) * 70 + fegt_factor * 30

    return round(fouling_index, 1)


def calculate_heat_rate_penalty(fouling_index: float) -> float:
    """
    Estimate heat rate penalty from fouling.

    Typical: 1% heat rate increase per 10-point fouling index
    """
    penalty = fouling_index * 0.1
    return round(penalty, 2)


def determine_blow_priority(
    zone: str,
    cleanliness: float,
    hours_since_blow: float,
    fuel_ash_pct: float
) -> tuple:
    """
    Determine soot blowing priority for a zone.

    Returns: (priority 1-5, action, reason)
    """
    # Ash factor - more ash = more frequent blowing needed
    ash_factor = 1.0 + fuel_ash_pct / 20  # 20% ash = 2x more frequent

    # Time factor - longer since blow = higher priority
    max_interval = 8.0 / ash_factor  # Base 8 hours, adjusted for ash

    if cleanliness < 0.7:
        return 1, "BLOW_NOW", f"Cleanliness {cleanliness:.2f} critically low"
    elif cleanliness < 0.8 or hours_since_blow > max_interval:
        return 2, "BLOW_NOW", f"Cleanliness degraded or interval exceeded"
    elif cleanliness < 0.85:
        return 3, "SCHEDULE", f"Cleanliness approaching threshold"
    elif hours_since_blow > max_interval * 0.8:
        return 4, "SCHEDULE", f"Approaching max interval"
    else:
        return 5, "SKIP", f"Zone clean (CF={cleanliness:.2f})"


class SootBlowerControllerAgent:
    """Soot blower optimization agent."""

    AGENT_ID = "GL-026"
    AGENT_NAME = "SOOTBLAST"
    VERSION = "1.0.0"

    # Steam consumption per zone (typical kg per blow cycle)
    STEAM_PER_BLOW = {
        "furnace": 500,
        "superheater": 300,
        "reheater": 300,
        "economizer": 200,
        "aph": 400
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = SootBlowerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: SootBlowerInput) -> SootBlowerOutput:
        recommendations = []
        warnings = []
        actions = []

        # Build cleanliness dict
        cleanliness = {
            "furnace": 0.85,  # Default if not monitored
            "superheater": inp.superheater_cleanliness,
            "reheater": inp.reheater_cleanliness,
            "economizer": inp.economizer_cleanliness,
            "aph": inp.air_preheater_cleanliness
        }

        # Calculate fouling index
        fouling_index = calculate_fouling_index(
            cleanliness,
            inp.furnace_exit_gas_temp_c,
            inp.design_fegt_c
        )

        heat_rate_penalty = calculate_heat_rate_penalty(fouling_index)

        # Estimate fuel waste (assume $50/MWh fuel, 500 MW boiler)
        boiler_size_mw = 500  # Assumed
        fuel_waste = boiler_size_mw * (heat_rate_penalty / 100) * 50  # $/hr

        # Determine actions for each zone
        immediate_zones = []
        scheduled_zones = []

        for zone in inp.blower_zones:
            cf = cleanliness.get(zone, 0.9)
            hours = inp.hours_since_last_blow.get(zone, 4.0)

            priority, action, reason = determine_blow_priority(
                zone, cf, hours, inp.fuel_ash_pct
            )

            steam_kg = self.STEAM_PER_BLOW.get(zone, 300)
            cost = steam_kg / 1000 * inp.steam_cost_per_ton

            actions.append(BlowingAction(
                zone=zone,
                action=action,
                priority=priority,
                reason=reason,
                estimated_steam_kg=steam_kg,
                estimated_cost=round(cost, 2)
            ))

            if action == "BLOW_NOW":
                immediate_zones.append(zone)
            elif action == "SCHEDULE":
                scheduled_zones.append(zone)

        # Sort by priority
        actions.sort(key=lambda x: x.priority)

        # Check simultaneous blowing constraint
        if len(immediate_zones) > inp.max_blowers_simultaneous:
            warnings.append(
                f"{len(immediate_zones)} zones need immediate blowing but max is "
                f"{inp.max_blowers_simultaneous}"
            )

        # Calculate optimal interval based on fouling rate
        if fouling_index > 50:
            optimal_interval = 2.0
        elif fouling_index > 30:
            optimal_interval = 4.0
        elif fouling_index > 15:
            optimal_interval = 6.0
        else:
            optimal_interval = 8.0

        # Adjust for ash content
        optimal_interval /= (1 + inp.fuel_ash_pct / 30)

        # Estimate daily usage
        blows_per_day = 24 / optimal_interval
        zones_count = len(inp.blower_zones)
        avg_steam = sum(self.STEAM_PER_BLOW.values()) / len(self.STEAM_PER_BLOW)
        daily_steam = blows_per_day * zones_count * avg_steam
        daily_cost = daily_steam / 1000 * inp.steam_cost_per_ton

        # Compare to fixed 4-hour schedule
        fixed_daily_steam = 6 * zones_count * avg_steam
        steam_savings = (fixed_daily_steam - daily_steam) / fixed_daily_steam * 100

        # Erosion risk based on steam pressure and frequency
        if inp.steam_pressure_bar > 15 and blows_per_day > 8:
            erosion_risk = "HIGH"
            warnings.append("High erosion risk - consider reducing pressure or frequency")
            recommended_pressure = 12.0
        elif inp.steam_pressure_bar > 12 or blows_per_day > 6:
            erosion_risk = "MEDIUM"
            recommended_pressure = inp.steam_pressure_bar
        else:
            erosion_risk = "LOW"
            recommended_pressure = inp.steam_pressure_bar

        # Recommendations
        if fouling_index > 40:
            recommendations.append(f"High fouling (index={fouling_index:.0f}) - increase blowing frequency")

        if heat_rate_penalty > 1.5:
            recommendations.append(f"Fouling causing {heat_rate_penalty:.1f}% efficiency loss")

        if steam_savings > 10:
            recommendations.append(f"Intelligent blowing saves {steam_savings:.0f}% steam vs fixed schedule")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "fouling": fouling_index,
            "immediate": immediate_zones,
            "optimal_interval": optimal_interval
        }).encode()).hexdigest()

        return SootBlowerOutput(
            actions=actions,
            immediate_blow_zones=immediate_zones,
            scheduled_blow_zones=scheduled_zones,
            overall_fouling_index=fouling_index,
            heat_rate_penalty_pct=heat_rate_penalty,
            estimated_fuel_waste_per_hour=round(fuel_waste, 2),
            optimal_blow_interval_hours=round(optimal_interval, 1),
            steam_saved_vs_fixed_schedule_pct=round(steam_savings, 1),
            estimated_daily_steam_usage_kg=round(daily_steam, 0),
            estimated_daily_cost=round(daily_cost, 2),
            erosion_risk=erosion_risk,
            recommended_pressure_bar=recommended_pressure,
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
            "category": "Combustion",
            "type": "Controller"
        }
