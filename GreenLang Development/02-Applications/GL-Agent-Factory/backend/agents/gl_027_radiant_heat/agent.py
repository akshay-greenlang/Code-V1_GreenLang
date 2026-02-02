"""GL-027 RADIANT-OPT: Radiant Heat Optimizer Agent.

Optimizes radiant section heat transfer in process furnaces to prevent
coking, tube damage, and maximize heat absorption efficiency.

Standards: API 560, ASME B31.3
"""
import hashlib
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RadiantSectionInput(BaseModel):
    """Input for radiant section optimization."""

    equipment_id: str = Field(..., description="Furnace ID")
    furnace_type: str = Field("fired_heater", description="fired_heater, reformer, cracker")

    # Process conditions
    process_fluid: str = Field("crude", description="crude, vacuum_residue, naphtha, gas")
    inlet_temp_c: float = Field(..., description="Process inlet temperature (°C)")
    outlet_temp_c: float = Field(..., description="Process outlet temperature (°C)")
    process_flow_kg_s: float = Field(..., ge=0, description="Process flow rate (kg/s)")
    operating_pressure_bar: float = Field(..., ge=1, description="Operating pressure (bar)")

    # Tube conditions
    tube_count: int = Field(..., ge=1, description="Number of radiant tubes")
    tube_od_mm: float = Field(114.3, description="Tube OD (mm)")
    tube_wall_mm: float = Field(6.0, description="Tube wall thickness (mm)")
    tube_material: str = Field("9Cr-1Mo", description="Tube metallurgy")
    max_tube_metal_temp_c: float = Field(600, description="Design max TMT (°C)")

    # Measured tube temperatures (representative)
    tube_metal_temps_c: List[float] = Field(default_factory=list, description="TMT readings")
    max_measured_tmt_c: Optional[float] = Field(None, description="Maximum measured TMT")
    avg_measured_tmt_c: Optional[float] = Field(None, description="Average measured TMT")

    # Firing conditions
    heat_duty_mw: float = Field(..., ge=0, description="Radiant heat duty (MW)")
    burner_count: int = Field(..., ge=1, description="Number of burners")
    burner_firing_rates: List[float] = Field(default_factory=list, description="Individual burner rates (%)")
    flame_impingement_detected: bool = Field(False, description="Flame impingement alarm")

    # Coking indicators
    pressure_drop_bar: float = Field(..., ge=0, description="Process side ΔP (bar)")
    design_pressure_drop_bar: float = Field(..., ge=0, description="Design ΔP (bar)")
    run_length_days: int = Field(0, description="Days since last decoking")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RadiantSectionOutput(BaseModel):
    """Output from radiant section optimizer."""

    # Temperature analysis
    max_tmt_c: float = Field(..., description="Maximum tube metal temp")
    tmt_margin_c: float = Field(..., description="Margin to design limit")
    tmt_uniformity_index: float = Field(..., description="TMT uniformity (0-1)")
    hot_spot_locations: List[int] = Field(default_factory=list, description="Hot tube indices")

    # Heat flux analysis
    avg_heat_flux_kw_m2: float = Field(..., description="Average heat flux")
    max_heat_flux_kw_m2: float = Field(..., description="Peak heat flux estimate")
    heat_flux_limit_kw_m2: float = Field(..., description="Allowable heat flux")
    flux_margin_pct: float = Field(..., description="Margin below limit (%)")

    # Coking assessment
    coking_index: float = Field(..., description="Coking severity (0-100)")
    estimated_coke_thickness_mm: float = Field(..., description="Estimated coke layer")
    remaining_run_length_days: int = Field(..., description="Estimated days to decoke")

    # Firing optimization
    recommended_duty_mw: float = Field(..., description="Recommended heat duty")
    burner_balance_index: float = Field(..., description="Burner balance (0-1)")
    burner_adjustments: Dict[str, float] = Field(default_factory=dict)

    # Efficiency
    radiant_efficiency_pct: float = Field(..., description="Radiant section efficiency")
    excess_fuel_pct: float = Field(..., description="Fuel excess due to coking")

    # Safety status
    safety_status: str = Field(..., description="SAFE, CAUTION, WARNING, CRITICAL")
    derating_required_pct: float = Field(0.0, description="Required capacity derate")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_heat_flux(
    heat_duty_kw: float,
    tube_count: int,
    tube_od_m: float,
    tube_length_m: float = 10.0
) -> tuple:
    """
    Calculate average and estimated peak heat flux.

    q_avg = Q / A_total
    q_peak ≈ 1.5 × q_avg (typical for floor-fired heaters)
    """
    area_per_tube = math.pi * tube_od_m * tube_length_m
    total_area = area_per_tube * tube_count

    if total_area <= 0:
        return 0.0, 0.0

    q_avg = heat_duty_kw / total_area  # kW/m²
    q_peak = q_avg * 1.5  # Peak/average ratio

    return round(q_avg, 1), round(q_peak, 1)


def estimate_allowable_flux(
    tube_material: str,
    process_fluid: str,
    max_tmt_c: float
) -> float:
    """
    Estimate allowable heat flux based on tube material and process.

    Based on API 560 guidelines and industry practice.
    """
    # Base allowable flux by material (kW/m²)
    material_base = {
        "carbon_steel": 35,
        "5Cr-0.5Mo": 40,
        "9Cr-1Mo": 45,
        "stainless_304": 40,
        "stainless_316": 42,
        "Incoloy_800": 50
    }

    base = material_base.get(tube_material, 40)

    # Process factor (coking tendency reduces allowable)
    process_factor = {
        "gas": 1.2,
        "naphtha": 1.0,
        "crude": 0.85,
        "vacuum_residue": 0.7
    }

    factor = process_factor.get(process_fluid, 0.9)

    return round(base * factor, 1)


def calculate_coking_index(
    pressure_drop_actual: float,
    pressure_drop_design: float,
    run_length_days: int,
    tmt_increase_c: float = 0
) -> tuple:
    """
    Calculate coking severity index and estimated thickness.

    Index based on:
    - Pressure drop increase (primary indicator)
    - Run length
    - TMT increase from baseline
    """
    # Pressure drop ratio contribution
    if pressure_drop_design <= 0:
        dp_ratio = 1.0
    else:
        dp_ratio = pressure_drop_actual / pressure_drop_design

    dp_index = (dp_ratio - 1.0) * 50  # 2x ΔP = 50 points

    # Run length contribution (assuming typical 365 day run)
    run_index = (run_length_days / 365) * 30

    # TMT increase contribution
    tmt_index = tmt_increase_c / 2  # 20°C increase = 10 points

    coking_index = min(100, max(0, dp_index + run_index + tmt_index))

    # Estimate coke thickness (simplified)
    # Typical: 0.1 mm per 10 points of coking index
    coke_mm = coking_index * 0.01

    return round(coking_index, 1), round(coke_mm, 2)


def calculate_remaining_run(
    coking_index: float,
    max_coking_index: float = 80
) -> int:
    """Estimate remaining run length before decoking needed."""
    if coking_index >= max_coking_index:
        return 0

    # Linear extrapolation (simplified)
    remaining_fraction = (max_coking_index - coking_index) / max_coking_index
    typical_run = 365
    return int(remaining_fraction * typical_run)


def calculate_burner_balance(firing_rates: List[float]) -> tuple:
    """
    Calculate burner balance index and recommended adjustments.

    Perfect balance = 1.0, poor balance approaches 0.
    """
    if not firing_rates or len(firing_rates) < 2:
        return 1.0, {}

    avg_rate = sum(firing_rates) / len(firing_rates)
    if avg_rate <= 0:
        return 1.0, {}

    # Standard deviation as fraction of average
    variance = sum((r - avg_rate) ** 2 for r in firing_rates) / len(firing_rates)
    std_dev = math.sqrt(variance)
    cv = std_dev / avg_rate  # Coefficient of variation

    balance_index = max(0, 1.0 - cv * 2)  # CV of 0.5 = balance of 0

    # Calculate adjustments
    adjustments = {}
    for i, rate in enumerate(firing_rates):
        adjustment = avg_rate - rate
        if abs(adjustment) > 2:  # Only significant adjustments
            adjustments[f"burner_{i+1}"] = round(adjustment, 1)

    return round(balance_index, 3), adjustments


class RadiantHeatOptimizerAgent:
    """Radiant section heat transfer optimizer."""

    AGENT_ID = "GL-027"
    AGENT_NAME = "RADIANT-OPT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = RadiantSectionInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: RadiantSectionInput) -> RadiantSectionOutput:
        recommendations = []
        warnings = []

        # Tube metal temperature analysis
        if inp.tube_metal_temps_c:
            max_tmt = max(inp.tube_metal_temps_c)
            avg_tmt = sum(inp.tube_metal_temps_c) / len(inp.tube_metal_temps_c)
            min_tmt = min(inp.tube_metal_temps_c)
            uniformity = 1.0 - (max_tmt - min_tmt) / max_tmt if max_tmt > 0 else 1.0

            # Find hot spots (>10% above average)
            hot_spots = [i for i, t in enumerate(inp.tube_metal_temps_c) if t > avg_tmt * 1.1]
        else:
            max_tmt = inp.max_measured_tmt_c or inp.outlet_temp_c + 50
            avg_tmt = inp.avg_measured_tmt_c or inp.outlet_temp_c + 30
            uniformity = 0.95
            hot_spots = []

        tmt_margin = inp.max_tube_metal_temp_c - max_tmt

        # Heat flux calculation
        duty_kw = inp.heat_duty_mw * 1000
        tube_od_m = inp.tube_od_mm / 1000
        q_avg, q_peak = calculate_heat_flux(duty_kw, inp.tube_count, tube_od_m)
        q_allowable = estimate_allowable_flux(inp.tube_material, inp.process_fluid, max_tmt)
        flux_margin = (q_allowable - q_peak) / q_allowable * 100 if q_allowable > 0 else 0

        # Coking assessment
        coking_index, coke_thickness = calculate_coking_index(
            inp.pressure_drop_bar,
            inp.design_pressure_drop_bar,
            inp.run_length_days,
            max_tmt - (inp.outlet_temp_c + 30)  # TMT increase estimate
        )
        remaining_run = calculate_remaining_run(coking_index)

        # Burner balance
        burner_balance, burner_adjustments = calculate_burner_balance(inp.burner_firing_rates)

        # Safety assessment
        if tmt_margin < 20 or inp.flame_impingement_detected:
            safety_status = "CRITICAL"
            derating = 15
            warnings.append(f"CRITICAL: TMT margin only {tmt_margin:.0f}°C!")
            recommendations.append("Immediately reduce firing rate by 15%")
        elif tmt_margin < 40 or flux_margin < 10:
            safety_status = "WARNING"
            derating = 10
            warnings.append("Approaching temperature/flux limits")
        elif tmt_margin < 60 or coking_index > 60:
            safety_status = "CAUTION"
            derating = 5
        else:
            safety_status = "SAFE"
            derating = 0

        # Efficiency estimate
        radiant_efficiency = 45 - coking_index * 0.1  # Baseline ~45%, drops with coking
        excess_fuel = coking_index * 0.15  # ~15% excess fuel at coking index 100

        # Recommended duty
        if derating > 0:
            recommended_duty = inp.heat_duty_mw * (1 - derating / 100)
        else:
            recommended_duty = inp.heat_duty_mw

        # Additional recommendations
        if coking_index > 40:
            recommendations.append(f"Schedule decoking - index at {coking_index:.0f}")

        if burner_balance < 0.9:
            recommendations.append("Rebalance burners for more uniform heat distribution")
            for burner, adj in burner_adjustments.items():
                if adj > 0:
                    recommendations.append(f"Increase {burner} by {adj:.1f}%")
                else:
                    recommendations.append(f"Decrease {burner} by {-adj:.1f}%")

        if uniformity < 0.9 and hot_spots:
            recommendations.append(f"Hot spots detected at tubes: {hot_spots}")

        if remaining_run < 30:
            warnings.append(f"Only ~{remaining_run} days remaining before decoking required")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "max_tmt": max_tmt,
            "coking_index": coking_index,
            "safety": safety_status
        }).encode()).hexdigest()

        return RadiantSectionOutput(
            max_tmt_c=round(max_tmt, 1),
            tmt_margin_c=round(tmt_margin, 1),
            tmt_uniformity_index=round(uniformity, 3),
            hot_spot_locations=hot_spots,
            avg_heat_flux_kw_m2=q_avg,
            max_heat_flux_kw_m2=q_peak,
            heat_flux_limit_kw_m2=q_allowable,
            flux_margin_pct=round(flux_margin, 1),
            coking_index=coking_index,
            estimated_coke_thickness_mm=coke_thickness,
            remaining_run_length_days=remaining_run,
            recommended_duty_mw=round(recommended_duty, 3),
            burner_balance_index=burner_balance,
            burner_adjustments=burner_adjustments,
            radiant_efficiency_pct=round(radiant_efficiency, 1),
            excess_fuel_pct=round(excess_fuel, 1),
            safety_status=safety_status,
            derating_required_pct=derating,
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
            "category": "Furnaces",
            "type": "Optimizer"
        }
