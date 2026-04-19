"""GL-024 AIRPREHEATER: Air Preheater Optimizer Agent.

Optimizes air preheater operation to maximize heat recovery from flue gas
while preventing cold-end corrosion and maintaining efficiency.

Standards: ASME PTC 4.3, API 560
"""
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============== Models ==============

class AirPreheaterInput(BaseModel):
    """Input parameters for air preheater optimization."""

    # Flue gas conditions
    flue_gas_inlet_temp_c: float = Field(..., ge=100, le=600, description="Flue gas inlet temperature (°C)")
    flue_gas_outlet_temp_c: float = Field(..., ge=50, le=300, description="Flue gas outlet temperature (°C)")
    flue_gas_flow_kg_s: float = Field(..., ge=0, description="Flue gas mass flow (kg/s)")

    # Air conditions
    air_inlet_temp_c: float = Field(..., ge=-40, le=60, description="Ambient air temperature (°C)")
    air_outlet_temp_c: float = Field(..., ge=0, le=400, description="Preheated air temperature (°C)")
    air_flow_kg_s: float = Field(..., ge=0, description="Air mass flow (kg/s)")

    # Fuel properties
    fuel_type: str = Field("natural_gas", description="Fuel type")
    fuel_sulfur_pct: float = Field(0.0, ge=0, le=5, description="Fuel sulfur content (%)")
    fuel_moisture_pct: float = Field(0.0, ge=0, le=50, description="Fuel moisture content (%)")

    # Equipment parameters
    aph_type: str = Field("rotary", description="APH type: rotary, tubular, plate")
    design_effectiveness: float = Field(0.7, ge=0.3, le=0.9, description="Design effectiveness")
    leakage_pct: float = Field(5.0, ge=0, le=20, description="Air-to-gas leakage (%)")

    # Operating conditions
    boiler_load_pct: float = Field(100, ge=20, le=110, description="Boiler load (%)")
    excess_o2_pct: float = Field(3.0, ge=0.5, le=10, description="Excess O2 in flue gas (%)")

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AirPreheaterOutput(BaseModel):
    """Output from air preheater optimizer."""

    # Performance metrics
    heat_recovery_kw: float = Field(..., description="Heat recovered (kW)")
    current_effectiveness: float = Field(..., description="Current effectiveness (0-1)")
    effectiveness_degradation_pct: float = Field(..., description="Degradation from design (%)")

    # Temperature analysis
    log_mean_temp_diff_c: float = Field(..., description="LMTD (°C)")
    cold_end_temp_c: float = Field(..., description="Cold end temperature (°C)")
    acid_dew_point_c: float = Field(..., description="Acid dew point temperature (°C)")
    dew_point_margin_c: float = Field(..., description="Margin above dew point (°C)")

    # Efficiency impact
    boiler_efficiency_gain_pct: float = Field(..., description="Efficiency gain from APH (%)")
    fuel_savings_pct: float = Field(..., description="Fuel savings from heat recovery (%)")

    # Leakage analysis
    actual_leakage_pct: float = Field(..., description="Estimated actual leakage (%)")
    leakage_heat_loss_kw: float = Field(..., description="Heat loss from leakage (kW)")

    # Fouling/corrosion status
    fouling_indicator: str = Field(..., description="CLEAN, LIGHT, MODERATE, HEAVY")
    corrosion_risk: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")

    # Recommendations
    optimal_exit_gas_temp_c: float = Field(..., description="Optimal flue gas exit temp")
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Provenance
    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


# ============== Formulas ==============

def calculate_acid_dew_point(
    sulfur_pct: float,
    moisture_pct: float,
    excess_o2_pct: float
) -> float:
    """
    Calculate acid dew point using Verhoff-Banchero correlation.

    T_dp = 1000 / (2.276 - 0.0294*ln(P_H2O) - 0.0858*ln(P_SO3) + 0.0062*ln(P_H2O)*ln(P_SO3))

    Simplified for industrial use:
    T_dp ≈ 125 + 18*S^0.5 for natural gas (low sulfur)
    T_dp ≈ 115 + 30*S^0.5 for coal/oil (higher sulfur)
    """
    if sulfur_pct <= 0.01:
        # Very low sulfur - use water dew point
        return 45.0 + moisture_pct * 0.5

    # Verhoff-Banchero approximation
    t_dp = 125 + 18 * math.sqrt(sulfur_pct)

    # Adjust for moisture
    t_dp += moisture_pct * 0.3

    # Adjust for excess O2 (higher O2 = more SO3 = higher dew point)
    t_dp += (excess_o2_pct - 3.0) * 2

    return round(t_dp, 1)


def calculate_lmtd(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float
) -> float:
    """
    Calculate Log Mean Temperature Difference for counterflow heat exchanger.

    LMTD = (ΔT1 - ΔT2) / ln(ΔT1/ΔT2)

    Where:
    ΔT1 = T_hot_in - T_cold_out (hot end)
    ΔT2 = T_hot_out - T_cold_in (cold end)
    """
    dt1 = t_hot_in - t_cold_out  # Hot end
    dt2 = t_hot_out - t_cold_in  # Cold end

    if dt1 <= 0 or dt2 <= 0:
        return 0.0

    if abs(dt1 - dt2) < 0.1:
        # Nearly equal - use arithmetic mean
        return (dt1 + dt2) / 2

    lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
    return round(lmtd, 2)


def calculate_effectiveness(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    c_hot: float,
    c_cold: float
) -> float:
    """
    Calculate heat exchanger effectiveness.

    ε = Q_actual / Q_max
    Q_actual = C_min * (T_hot_in - T_hot_out) or C_cold * (T_cold_out - T_cold_in)
    Q_max = C_min * (T_hot_in - T_cold_in)
    """
    c_min = min(c_hot, c_cold)
    c_max = max(c_hot, c_cold)

    if c_min <= 0:
        return 0.0

    q_max = c_min * (t_hot_in - t_cold_in)
    if q_max <= 0:
        return 0.0

    q_actual = c_cold * (t_cold_out - t_cold_in)
    effectiveness = q_actual / q_max

    return round(min(1.0, max(0.0, effectiveness)), 4)


def calculate_heat_recovery(
    air_flow_kg_s: float,
    air_inlet_temp_c: float,
    air_outlet_temp_c: float,
    cp_air: float = 1.005
) -> float:
    """
    Calculate heat recovered by air preheater.

    Q = m_air * cp * ΔT (kW)
    """
    delta_t = air_outlet_temp_c - air_inlet_temp_c
    q_kw = air_flow_kg_s * cp_air * delta_t
    return round(q_kw, 2)


def calculate_boiler_efficiency_gain(
    air_preheat_temp_rise_c: float,
    flue_gas_temp_drop_c: float,
    baseline_efficiency_pct: float = 85.0
) -> float:
    """
    Calculate boiler efficiency improvement from air preheating.

    Rule of thumb: Each 20°C reduction in exit gas temp ≈ 1% efficiency gain
    Each 40°C air preheat ≈ 1% efficiency gain
    """
    gain_from_air = air_preheat_temp_rise_c / 40.0
    gain_from_gas = flue_gas_temp_drop_c / 20.0

    # Take the heat recovery approach (lower of two)
    efficiency_gain = min(gain_from_air, gain_from_gas)
    return round(efficiency_gain, 2)


def estimate_leakage(
    air_flow_in: float,
    air_flow_out: float,
    design_leakage_pct: float
) -> float:
    """
    Estimate actual air-to-gas leakage percentage.

    Leakage = (Air_flow_at_APH_exit - Air_flow_at_APH_inlet) / Air_flow_at_APH_inlet * 100
    """
    if air_flow_in <= 0:
        return design_leakage_pct

    # Simple model - actual would use O2 rise across APH
    return design_leakage_pct  # Placeholder - would need O2 data


# ============== Agent ==============

class AirPreheaterOptimizerAgent:
    """Air preheater optimization agent."""

    AGENT_ID = "GL-024"
    AGENT_NAME = "AIRPREHEATER"
    VERSION = "1.0.0"

    # Safety margins
    MIN_DEW_POINT_MARGIN_C = 15.0
    WARNING_DEW_POINT_MARGIN_C = 25.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = AirPreheaterInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: AirPreheaterInput) -> AirPreheaterOutput:
        recommendations = []
        warnings = []

        # Calculate acid dew point
        acid_dew_point = calculate_acid_dew_point(
            inp.fuel_sulfur_pct,
            inp.fuel_moisture_pct,
            inp.excess_o2_pct
        )

        # Cold end temperature (approximation)
        cold_end_temp = inp.flue_gas_outlet_temp_c

        # Dew point margin
        dew_point_margin = cold_end_temp - acid_dew_point

        # Corrosion risk assessment
        if dew_point_margin < self.MIN_DEW_POINT_MARGIN_C:
            corrosion_risk = "CRITICAL"
            warnings.append(f"CRITICAL: Operating {abs(dew_point_margin):.1f}°C below acid dew point!")
            recommendations.append("Immediately increase exit gas temperature")
        elif dew_point_margin < self.WARNING_DEW_POINT_MARGIN_C:
            corrosion_risk = "HIGH"
            warnings.append(f"Cold-end corrosion risk - only {dew_point_margin:.1f}°C margin")
        elif dew_point_margin < 40:
            corrosion_risk = "MEDIUM"
        else:
            corrosion_risk = "LOW"

        # Calculate LMTD
        lmtd = calculate_lmtd(
            inp.flue_gas_inlet_temp_c,
            inp.flue_gas_outlet_temp_c,
            inp.air_inlet_temp_c,
            inp.air_outlet_temp_c
        )

        # Heat capacity rates (approx)
        cp_air = 1.005  # kJ/kg·K
        cp_gas = 1.1    # kJ/kg·K
        c_air = inp.air_flow_kg_s * cp_air
        c_gas = inp.flue_gas_flow_kg_s * cp_gas

        # Effectiveness
        effectiveness = calculate_effectiveness(
            inp.flue_gas_inlet_temp_c,
            inp.flue_gas_outlet_temp_c,
            inp.air_inlet_temp_c,
            inp.air_outlet_temp_c,
            c_gas, c_air
        )

        effectiveness_degradation = (inp.design_effectiveness - effectiveness) / inp.design_effectiveness * 100

        # Fouling indicator
        if effectiveness_degradation < 5:
            fouling = "CLEAN"
        elif effectiveness_degradation < 15:
            fouling = "LIGHT"
        elif effectiveness_degradation < 25:
            fouling = "MODERATE"
            recommendations.append("Schedule cleaning at next opportunity")
        else:
            fouling = "HEAVY"
            warnings.append(f"Severe fouling detected - {effectiveness_degradation:.1f}% degradation")
            recommendations.append("Priority cleaning required")

        # Heat recovery
        heat_recovery = calculate_heat_recovery(
            inp.air_flow_kg_s,
            inp.air_inlet_temp_c,
            inp.air_outlet_temp_c
        )

        # Efficiency gain
        air_temp_rise = inp.air_outlet_temp_c - inp.air_inlet_temp_c
        gas_temp_drop = inp.flue_gas_inlet_temp_c - inp.flue_gas_outlet_temp_c
        efficiency_gain = calculate_boiler_efficiency_gain(air_temp_rise, gas_temp_drop)

        # Fuel savings
        fuel_savings = efficiency_gain  # Approximately equal

        # Leakage
        actual_leakage = inp.leakage_pct  # Simplified
        leakage_heat_loss = heat_recovery * (actual_leakage / 100) * 0.5

        if actual_leakage > 10:
            warnings.append(f"High leakage rate: {actual_leakage:.1f}%")
            recommendations.append("Inspect and repair seals")

        # Optimal exit gas temp
        optimal_exit = acid_dew_point + 30  # 30°C margin

        if inp.flue_gas_outlet_temp_c > optimal_exit + 20:
            recommendations.append(
                f"Exit gas temp {inp.flue_gas_outlet_temp_c:.0f}°C high - "
                f"potential to recover {(inp.flue_gas_outlet_temp_c - optimal_exit) * 0.05:.1f}% more efficiency"
            )

        # Provenance hash
        calc_hash = hashlib.sha256(json.dumps({
            "inputs": inp.model_dump(),
            "effectiveness": effectiveness,
            "heat_recovery": heat_recovery
        }, sort_keys=True, default=str).encode()).hexdigest()

        return AirPreheaterOutput(
            heat_recovery_kw=heat_recovery,
            current_effectiveness=effectiveness,
            effectiveness_degradation_pct=round(effectiveness_degradation, 2),
            log_mean_temp_diff_c=lmtd,
            cold_end_temp_c=cold_end_temp,
            acid_dew_point_c=acid_dew_point,
            dew_point_margin_c=round(dew_point_margin, 1),
            boiler_efficiency_gain_pct=efficiency_gain,
            fuel_savings_pct=fuel_savings,
            actual_leakage_pct=actual_leakage,
            leakage_heat_loss_kw=round(leakage_heat_loss, 2),
            fouling_indicator=fouling,
            corrosion_risk=corrosion_risk,
            optimal_exit_gas_temp_c=round(optimal_exit, 1),
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
            "category": "Heat Recovery",
            "type": "Optimizer"
        }
