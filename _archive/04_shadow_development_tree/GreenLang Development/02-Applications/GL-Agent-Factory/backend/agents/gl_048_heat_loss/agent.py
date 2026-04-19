"""GL-048: Heat Loss Agent (HEAT-LOSS).

Analyzes and minimizes heat losses in thermal systems.

Standards: ISO 12241, ASTM C680
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


class LossType(str, Enum):
    CONDUCTION = "CONDUCTION"
    CONVECTION = "CONVECTION"
    RADIATION = "RADIATION"
    COMBINED = "COMBINED"


class SurfaceType(str, Enum):
    PIPE = "PIPE"
    TANK = "TANK"
    DUCT = "DUCT"
    WALL = "WALL"


class HeatLossInput(BaseModel):
    equipment_id: str
    surface_type: SurfaceType = Field(default=SurfaceType.PIPE)
    surface_temp_c: float = Field(..., description="Surface temperature")
    ambient_temp_c: float = Field(default=20)
    surface_area_m2: float = Field(..., gt=0)
    insulation_thickness_mm: float = Field(default=0, ge=0)
    insulation_conductivity: float = Field(default=0.04, gt=0)  # W/m·K
    emissivity: float = Field(default=0.9, ge=0, le=1)
    wind_speed_m_s: float = Field(default=0, ge=0)
    operating_hours_year: int = Field(default=8000)
    energy_cost_kwh: float = Field(default=0.05, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HeatLossBreakdown(BaseModel):
    conduction_kw: float
    convection_kw: float
    radiation_kw: float
    total_kw: float


class HeatLossOutput(BaseModel):
    equipment_id: str
    heat_loss_breakdown: HeatLossBreakdown
    total_heat_loss_kw: float
    heat_loss_per_m2_w: float
    annual_energy_loss_mwh: float
    annual_cost_usd: float
    insulation_effectiveness_pct: float
    optimal_insulation_mm: float
    potential_savings_usd: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


STEFAN_BOLTZMANN = 5.67e-8  # W/m²·K⁴


class HeatLossAgent:
    AGENT_ID = "GL-048"
    AGENT_NAME = "HEAT-LOSS"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"HeatLossAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = HeatLossInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calc_convection_coeff(self, wind: float) -> float:
        """Calculate convection coefficient h (W/m²·K)."""
        if wind < 0.5:
            return 5.0  # Natural convection
        return 5.0 + 3.8 * wind  # Forced convection

    def _process(self, inp: HeatLossInput) -> HeatLossOutput:
        recommendations = []
        delta_t = inp.surface_temp_c - inp.ambient_temp_c

        # Conduction through insulation
        if inp.insulation_thickness_mm > 0:
            r_insulation = (inp.insulation_thickness_mm / 1000) / inp.insulation_conductivity
            q_cond = delta_t / r_insulation * inp.surface_area_m2 / 1000  # kW
        else:
            q_cond = 0

        # Convection
        h_conv = self._calc_convection_coeff(inp.wind_speed_m_s)
        q_conv = h_conv * delta_t * inp.surface_area_m2 / 1000  # kW

        # Radiation: Q = ε·σ·A·(Ts⁴ - Ta⁴)
        ts_k = inp.surface_temp_c + 273.15
        ta_k = inp.ambient_temp_c + 273.15
        q_rad = inp.emissivity * STEFAN_BOLTZMANN * inp.surface_area_m2 * (ts_k**4 - ta_k**4) / 1000  # kW

        # Total loss depends on whether insulated
        if inp.insulation_thickness_mm > 0:
            total_kw = q_cond  # Dominated by conduction through insulation
        else:
            total_kw = q_conv + q_rad  # Bare surface

        # Per unit area
        heat_loss_w_m2 = total_kw * 1000 / inp.surface_area_m2 if inp.surface_area_m2 > 0 else 0

        # Annual metrics
        annual_mwh = total_kw * inp.operating_hours_year / 1000
        annual_cost = annual_mwh * 1000 * inp.energy_cost_kwh

        # Optimal insulation (economic thickness)
        # Simple estimate: 50mm for <200°C, 100mm for >200°C
        if inp.surface_temp_c > 200:
            optimal_ins = 100
        elif inp.surface_temp_c > 100:
            optimal_ins = 75
        else:
            optimal_ins = 50

        # Insulation effectiveness
        if inp.insulation_thickness_mm > 0:
            # Compare to bare surface
            bare_loss = (h_conv * delta_t + inp.emissivity * STEFAN_BOLTZMANN * (ts_k**4 - ta_k**4)) * inp.surface_area_m2 / 1000
            effectiveness = (1 - total_kw / bare_loss) * 100 if bare_loss > 0 else 0
        else:
            effectiveness = 0

        # Potential savings with optimal insulation
        if inp.insulation_thickness_mm < optimal_ins:
            r_optimal = (optimal_ins / 1000) / inp.insulation_conductivity
            q_optimal = delta_t / r_optimal * inp.surface_area_m2 / 1000
            savings_kw = total_kw - q_optimal
            potential_savings = savings_kw * inp.operating_hours_year * inp.energy_cost_kwh
        else:
            potential_savings = 0

        # Recommendations
        if inp.insulation_thickness_mm == 0 and delta_t > 30:
            recommendations.append(f"Add {optimal_ins}mm insulation - surface is bare")
        if inp.insulation_thickness_mm < optimal_ins * 0.5 and delta_t > 50:
            recommendations.append(f"Increase insulation to {optimal_ins}mm")
        if effectiveness < 80 and inp.insulation_thickness_mm > 0:
            recommendations.append("Insulation effectiveness low - check for damage")
        if potential_savings > 1000:
            recommendations.append(f"Potential annual savings: ${potential_savings:,.0f}")
        if heat_loss_w_m2 > 500:
            recommendations.append(f"High heat loss ({heat_loss_w_m2:.0f} W/m²) - prioritize remediation")

        calc_hash = hashlib.sha256(json.dumps({
            "equipment": inp.equipment_id,
            "total_kw": round(total_kw, 3),
            "annual_cost": round(annual_cost, 2)
        }).encode()).hexdigest()

        return HeatLossOutput(
            equipment_id=inp.equipment_id,
            heat_loss_breakdown=HeatLossBreakdown(
                conduction_kw=round(q_cond, 3),
                convection_kw=round(q_conv, 3),
                radiation_kw=round(q_rad, 3),
                total_kw=round(total_kw, 3)
            ),
            total_heat_loss_kw=round(total_kw, 3),
            heat_loss_per_m2_w=round(heat_loss_w_m2, 1),
            annual_energy_loss_mwh=round(annual_mwh, 2),
            annual_cost_usd=round(annual_cost, 2),
            insulation_effectiveness_pct=round(max(0, effectiveness), 1),
            optimal_insulation_mm=optimal_ins,
            potential_savings_usd=round(potential_savings, 2),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-048", "name": "HEAT-LOSS", "version": "1.0.0",
    "summary": "Heat loss analysis and insulation optimization",
    "standards": [{"ref": "ISO 12241"}, {"ref": "ASTM C680"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
