"""GL-038: Solar Thermal Agent (SOLAR-THERMAL).

Optimizes solar thermal systems for process heat.

Standards: ISO 9806, ASHRAE 93
"""
import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CollectorType(str, Enum):
    FLAT_PLATE = "FLAT_PLATE"
    EVACUATED_TUBE = "EVACUATED_TUBE"
    PARABOLIC_TROUGH = "PARABOLIC_TROUGH"
    LINEAR_FRESNEL = "LINEAR_FRESNEL"


class SolarThermalInput(BaseModel):
    equipment_id: str
    collector_type: CollectorType = Field(default=CollectorType.FLAT_PLATE)
    collector_area_m2: float = Field(..., gt=0)
    solar_irradiance_w_m2: float = Field(default=800, ge=0, le=1400)
    ambient_temp_c: float = Field(default=25)
    inlet_temp_c: float = Field(default=40)
    target_temp_c: float = Field(default=80)
    flow_rate_kg_s: float = Field(default=0.02, gt=0)
    optical_efficiency: float = Field(default=0.75, ge=0, le=1)
    heat_loss_coeff_w_m2k: float = Field(default=4, ge=0)
    latitude_deg: float = Field(default=35, ge=-90, le=90)
    annual_solar_hours: int = Field(default=2000)
    fuel_cost_kwh: float = Field(default=0.05, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SolarThermalOutput(BaseModel):
    equipment_id: str
    collector_efficiency_pct: float
    useful_heat_kw: float
    outlet_temp_c: float
    annual_heat_kwh: float
    annual_fuel_savings_usd: float
    solar_fraction_pct: float
    optimal_flow_rate_kg_s: float
    optimal_tilt_angle_deg: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class SolarThermalAgent:
    AGENT_ID = "GL-038"
    AGENT_NAME = "SOLAR-THERMAL"
    VERSION = "1.0.0"

    # Specific heat of water (kJ/kg·K)
    CP_WATER = 4.186

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"SolarThermalAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = SolarThermalInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: SolarThermalInput) -> SolarThermalOutput:
        recommendations = []

        # Average collector temperature
        avg_temp = (inp.inlet_temp_c + inp.target_temp_c) / 2
        delta_t = avg_temp - inp.ambient_temp_c

        # Collector efficiency: η = η₀ - (U * ΔT / G)
        if inp.solar_irradiance_w_m2 > 0:
            efficiency = inp.optical_efficiency - (inp.heat_loss_coeff_w_m2k * delta_t / inp.solar_irradiance_w_m2)
            efficiency = max(0, min(1, efficiency))
        else:
            efficiency = 0

        # Useful heat gain
        q_incident = inp.solar_irradiance_w_m2 * inp.collector_area_m2 / 1000  # kW
        useful_heat = q_incident * efficiency

        # Outlet temperature: Q = m * Cp * ΔT
        if inp.flow_rate_kg_s > 0:
            delta_t_fluid = useful_heat / (inp.flow_rate_kg_s * self.CP_WATER)
            outlet_temp = inp.inlet_temp_c + delta_t_fluid
        else:
            outlet_temp = inp.inlet_temp_c

        # Annual production
        annual_heat = useful_heat * inp.annual_solar_hours
        annual_savings = annual_heat * inp.fuel_cost_kwh

        # Solar fraction (assuming constant load)
        load_kw = inp.flow_rate_kg_s * self.CP_WATER * (inp.target_temp_c - inp.inlet_temp_c)
        solar_fraction = (useful_heat / load_kw * 100) if load_kw > 0 else 0
        solar_fraction = min(100, solar_fraction)

        # Optimal flow rate (balance between efficiency and temperature rise)
        optimal_flow = useful_heat / (self.CP_WATER * (inp.target_temp_c - inp.inlet_temp_c)) if (inp.target_temp_c - inp.inlet_temp_c) > 0 else 0.02

        # Optimal tilt angle (latitude + 15° for winter heating)
        optimal_tilt = abs(inp.latitude_deg) + 15

        # Recommendations
        if efficiency < 0.4:
            recommendations.append(f"Low efficiency ({efficiency*100:.1f}%) - consider evacuated tubes")
        if outlet_temp < inp.target_temp_c * 0.9:
            recommendations.append(f"Outlet temp {outlet_temp:.1f}°C below target - increase collector area")
        if solar_fraction < 30:
            recommendations.append(f"Low solar fraction ({solar_fraction:.1f}%) - add storage tank")
        if inp.collector_type == CollectorType.FLAT_PLATE and inp.target_temp_c > 80:
            recommendations.append("Target temp >80°C - upgrade to evacuated tube collectors")
        if inp.heat_loss_coeff_w_m2k > 6:
            recommendations.append("High heat loss coefficient - check insulation")

        calc_hash = hashlib.sha256(json.dumps({
            "equipment": inp.equipment_id,
            "efficiency": round(efficiency, 3),
            "useful_heat": round(useful_heat, 2)
        }).encode()).hexdigest()

        return SolarThermalOutput(
            equipment_id=inp.equipment_id,
            collector_efficiency_pct=round(efficiency * 100, 1),
            useful_heat_kw=round(useful_heat, 2),
            outlet_temp_c=round(outlet_temp, 1),
            annual_heat_kwh=round(annual_heat, 0),
            annual_fuel_savings_usd=round(annual_savings, 2),
            solar_fraction_pct=round(solar_fraction, 1),
            optimal_flow_rate_kg_s=round(optimal_flow, 4),
            optimal_tilt_angle_deg=round(optimal_tilt, 0),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-038", "name": "SOLAR-THERMAL", "version": "1.0.0",
    "summary": "Solar thermal system optimization for process heat",
    "standards": [{"ref": "ISO 9806"}, {"ref": "ASHRAE 93"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
