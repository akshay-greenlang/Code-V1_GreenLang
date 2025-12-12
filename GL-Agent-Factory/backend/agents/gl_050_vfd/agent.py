"""GL-050: VFD Agent (VFD-OPTIMIZER).

Optimizes Variable Frequency Drive operation for energy savings.

Standards: IEC 61800, NEMA MG1
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LoadType(str, Enum):
    FAN = "FAN"
    PUMP = "PUMP"
    COMPRESSOR = "COMPRESSOR"
    CONVEYOR = "CONVEYOR"


class VFDInput(BaseModel):
    equipment_id: str
    equipment_name: str = Field(default="Motor")
    load_type: LoadType = Field(default=LoadType.FAN)
    motor_power_kw: float = Field(..., gt=0)
    rated_speed_rpm: float = Field(default=1800, gt=0)
    current_speed_rpm: float = Field(..., gt=0)
    current_load_pct: float = Field(..., ge=0, le=100)
    has_vfd: bool = Field(default=True)
    vfd_efficiency_pct: float = Field(default=97, ge=0, le=100)
    motor_efficiency_pct: float = Field(default=93, ge=0, le=100)
    operating_hours_year: int = Field(default=8000)
    electricity_price_kwh: float = Field(default=0.10, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VFDOutput(BaseModel):
    equipment_id: str
    equipment_name: str
    current_power_kw: float
    theoretical_min_power_kw: float
    energy_savings_potential_pct: float
    optimal_speed_rpm: float
    annual_energy_kwh: float
    annual_cost_usd: float
    potential_savings_kwh: float
    potential_savings_usd: float
    vfd_payback_years: Optional[float]
    affinity_law_applied: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class VFDAgent:
    """GL-050: VFD Agent - Variable Frequency Drive optimization."""

    AGENT_ID = "GL-050"
    AGENT_NAME = "VFD-OPTIMIZER"
    VERSION = "1.0.0"

    # VFD cost per kW (typical)
    VFD_COST_PER_KW = 150

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"VFDAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = VFDInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: VFDInput) -> VFDOutput:
        recommendations = []

        # Speed ratio
        speed_ratio = inp.current_speed_rpm / inp.rated_speed_rpm

        # Apply affinity laws based on load type
        # Fan/Pump: Power ∝ Speed³
        # Compressor: Power ∝ Speed² (approximately)
        if inp.load_type in [LoadType.FAN, LoadType.PUMP]:
            power_ratio = speed_ratio ** 3
            affinity = "Cubic (P ∝ N³)"
        else:
            power_ratio = speed_ratio ** 2
            affinity = "Quadratic (P ∝ N²)"

        # Current power
        current_power = inp.motor_power_kw * (inp.current_load_pct / 100)

        # Theoretical minimum with VFD at current speed
        if inp.has_vfd:
            theoretical_power = inp.motor_power_kw * power_ratio * (inp.vfd_efficiency_pct / 100)
        else:
            # Without VFD, throttling/dampers used - very inefficient
            theoretical_power = inp.motor_power_kw * power_ratio

        # Optimal speed for current load
        optimal_speed = inp.rated_speed_rpm * (inp.current_load_pct / 100) ** (1/3 if inp.load_type in [LoadType.FAN, LoadType.PUMP] else 0.5)

        # Energy savings potential
        if current_power > 0:
            savings_pct = max(0, (current_power - theoretical_power) / current_power * 100)
        else:
            savings_pct = 0

        # Annual metrics
        annual_energy = current_power * inp.operating_hours_year
        annual_cost = annual_energy * inp.electricity_price_kwh

        potential_savings_kwh = annual_energy * (savings_pct / 100)
        potential_savings_usd = potential_savings_kwh * inp.electricity_price_kwh

        # VFD payback (if not installed)
        if not inp.has_vfd and potential_savings_usd > 0:
            vfd_cost = inp.motor_power_kw * self.VFD_COST_PER_KW
            payback = vfd_cost / potential_savings_usd
        else:
            payback = None

        # Recommendations
        if not inp.has_vfd:
            recommendations.append(f"Install VFD - estimated payback {payback:.1f} years" if payback else "Install VFD for speed control")

        if inp.has_vfd and inp.current_speed_rpm > optimal_speed * 1.1:
            recommendations.append(f"Reduce speed from {inp.current_speed_rpm:.0f} to {optimal_speed:.0f} RPM")

        if savings_pct > 20:
            recommendations.append(f"High savings potential ({savings_pct:.1f}%) - optimize control strategy")

        if inp.motor_efficiency_pct < 90:
            recommendations.append("Consider high-efficiency motor upgrade")

        if inp.load_type == LoadType.PUMP and inp.current_load_pct < 50:
            recommendations.append("Low load factor - consider impeller trimming")

        calc_hash = hashlib.sha256(json.dumps({
            "equipment": inp.equipment_id,
            "current_power": round(current_power, 2),
            "savings_pct": round(savings_pct, 1)
        }).encode()).hexdigest()

        return VFDOutput(
            equipment_id=inp.equipment_id,
            equipment_name=inp.equipment_name,
            current_power_kw=round(current_power, 2),
            theoretical_min_power_kw=round(theoretical_power, 2),
            energy_savings_potential_pct=round(savings_pct, 1),
            optimal_speed_rpm=round(optimal_speed, 0),
            annual_energy_kwh=round(annual_energy, 0),
            annual_cost_usd=round(annual_cost, 2),
            potential_savings_kwh=round(potential_savings_kwh, 0),
            potential_savings_usd=round(potential_savings_usd, 2),
            vfd_payback_years=round(payback, 1) if payback else None,
            affinity_law_applied=affinity,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-050", "name": "VFD-OPTIMIZER", "version": "1.0.0",
    "summary": "Variable Frequency Drive optimization using affinity laws",
    "standards": [{"ref": "IEC 61800"}, {"ref": "NEMA MG1"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
