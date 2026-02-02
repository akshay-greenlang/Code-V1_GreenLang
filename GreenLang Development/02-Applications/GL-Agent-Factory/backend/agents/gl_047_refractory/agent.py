"""GL-047: Refractory Agent (REFRACTORY).

Monitors and optimizes refractory performance.

Standards: ASTM C155, ISO 836
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RefractoryType(str, Enum):
    FIREBRICK = "FIREBRICK"
    CASTABLE = "CASTABLE"
    CERAMIC_FIBER = "CERAMIC_FIBER"
    INSULATING = "INSULATING"


class RefractoryStatus(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class RefractoryInput(BaseModel):
    equipment_id: str
    refractory_type: RefractoryType = Field(default=RefractoryType.FIREBRICK)
    thickness_mm: float = Field(..., gt=0)
    hot_face_temp_c: float = Field(..., ge=0)
    cold_face_temp_c: float = Field(..., ge=0)
    design_thickness_mm: float = Field(..., gt=0)
    age_years: float = Field(default=0, ge=0)
    thermal_conductivity: float = Field(default=1.5, gt=0)
    max_service_temp_c: float = Field(default=1400, gt=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RefractoryOutput(BaseModel):
    equipment_id: str
    heat_loss_kw_m2: float
    wear_rate_mm_year: float
    remaining_life_years: float
    thickness_loss_pct: float
    status: RefractoryStatus
    efficiency_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class RefractoryAgent:
    AGENT_ID = "GL-047"
    AGENT_NAME = "REFRACTORY"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"RefractoryAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = RefractoryInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: RefractoryInput) -> RefractoryOutput:
        recommendations = []

        # Heat loss: Q = k * (Th - Tc) / L
        delta_t = inp.hot_face_temp_c - inp.cold_face_temp_c
        heat_loss = inp.thermal_conductivity * delta_t / (inp.thickness_mm / 1000)  # kW/m2

        # Thickness loss
        thickness_loss = inp.design_thickness_mm - inp.thickness_mm
        thickness_loss_pct = (thickness_loss / inp.design_thickness_mm) * 100 if inp.design_thickness_mm > 0 else 0

        # Wear rate
        wear_rate = thickness_loss / inp.age_years if inp.age_years > 0 else 0

        # Remaining life
        if wear_rate > 0:
            remaining_thickness = inp.thickness_mm - (inp.design_thickness_mm * 0.3)  # 30% min
            remaining_life = remaining_thickness / wear_rate if remaining_thickness > 0 else 0
        else:
            remaining_life = 10  # Default

        # Efficiency
        if inp.hot_face_temp_c <= inp.max_service_temp_c:
            efficiency = 100 - thickness_loss_pct
        else:
            efficiency = max(0, 100 - thickness_loss_pct - 20)

        # Status
        if thickness_loss_pct > 50 or remaining_life < 1:
            status = RefractoryStatus.CRITICAL
        elif thickness_loss_pct > 30 or remaining_life < 3:
            status = RefractoryStatus.DEGRADED
        elif thickness_loss_pct > 15:
            status = RefractoryStatus.ACCEPTABLE
        else:
            status = RefractoryStatus.GOOD

        # Recommendations
        if status == RefractoryStatus.CRITICAL:
            recommendations.append("URGENT: Plan refractory replacement")
        if thickness_loss_pct > 20:
            recommendations.append(f"Thickness loss at {thickness_loss_pct:.1f}% - monitor closely")
        if heat_loss > 5:
            recommendations.append(f"High heat loss ({heat_loss:.1f} kW/m²) - consider repair")
        if inp.hot_face_temp_c > inp.max_service_temp_c * 0.9:
            recommendations.append("Operating near max temperature - reduce thermal stress")

        calc_hash = hashlib.sha256(json.dumps({
            "equipment": inp.equipment_id,
            "heat_loss": round(heat_loss, 2),
            "status": status.value
        }).encode()).hexdigest()

        return RefractoryOutput(
            equipment_id=inp.equipment_id,
            heat_loss_kw_m2=round(heat_loss, 2),
            wear_rate_mm_year=round(wear_rate, 2),
            remaining_life_years=round(remaining_life, 1),
            thickness_loss_pct=round(thickness_loss_pct, 1),
            status=status,
            efficiency_pct=round(efficiency, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-047", "name": "REFRACTORY", "version": "1.0.0",
    "summary": "Refractory performance monitoring and optimization",
    "standards": [{"ref": "ASTM C155"}, {"ref": "ISO 836"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
