"""GL-046: Draft Control Agent (DRAFT-CONTROL).

Manages furnace draft and pressure control with stack effect calculation,
damper optimization, and pressure balance analysis.

Standards: NFPA 86, API 560
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


class DraftType(str, Enum):
    NATURAL = "NATURAL"
    INDUCED = "INDUCED"
    FORCED = "FORCED"
    BALANCED = "BALANCED"


class DraftStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    ACCEPTABLE = "ACCEPTABLE"
    HIGH = "HIGH"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


class DraftControlInput(BaseModel):
    """Input for draft control analysis."""
    furnace_id: str = Field(..., description="Furnace identifier")
    draft_type: DraftType = Field(default=DraftType.BALANCED)
    stack_temperature_c: float = Field(..., ge=0, description="Stack gas temperature")
    ambient_temperature_c: float = Field(..., description="Ambient temperature")
    stack_height_m: float = Field(..., gt=0, description="Stack height in meters")
    furnace_pressure_inwc: float = Field(..., description="Furnace pressure in inWC")
    target_pressure_inwc: float = Field(default=-0.1, description="Target pressure")
    damper_position_pct: float = Field(default=50, ge=0, le=100)
    flue_gas_flow_kg_s: float = Field(default=10, ge=0)
    excess_air_pct: float = Field(default=15, ge=0, le=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DraftControlOutput(BaseModel):
    """Output from draft control analysis."""
    furnace_id: str
    stack_effect_inwc: float = Field(..., description="Calculated stack effect")
    stack_effect_pa: float = Field(..., description="Stack effect in Pascals")
    theoretical_draft_inwc: float
    overall_status: DraftStatus
    pressure_deviation_inwc: float
    optimal_damper_position_pct: float
    draft_loss_pct: float
    energy_impact_kw: float
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_stack_effect_inwc(stack_temp_c: float, ambient_temp_c: float, height_m: float) -> float:
    """Calculate stack effect using temperature differential.

    Formula: SE = 7.64 * H * (1/Ta - 1/Ts) where T in Kelvin
    Reference: API 560 Section 8
    """
    stack_k = stack_temp_c + 273.15
    ambient_k = ambient_temp_c + 273.15
    stack_effect = 7.64 * height_m * (1/ambient_k - 1/stack_k)
    return round(stack_effect, 4)


def calculate_optimal_damper(required: float, available: float, current: float) -> float:
    """Calculate optimal damper position for required draft."""
    if available <= 0:
        return current
    ratio = abs(required) / abs(available)
    optimal = math.sqrt(ratio) * 100
    return round(max(0, min(100, optimal)), 1)


class DraftControlAgent:
    """GL-046: Draft Control Agent - Furnace draft and pressure control."""

    AGENT_ID = "GL-046"
    AGENT_NAME = "DRAFT-CONTROL"
    VERSION = "1.0.0"

    MIN_DRAFT_INWC = -0.5
    MAX_DRAFT_INWC = 0.05
    TARGET_DRAFT_INWC = -0.1

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"DraftControlAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = DraftControlInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: DraftControlInput) -> DraftControlOutput:
        recommendations = []
        warnings = []

        # Calculate stack effect
        stack_effect = calculate_stack_effect_inwc(
            inp.stack_temperature_c,
            inp.ambient_temperature_c,
            inp.stack_height_m
        )
        stack_effect_pa = round(stack_effect * 249.089, 2)

        # Pressure deviation
        deviation = inp.furnace_pressure_inwc - inp.target_pressure_inwc

        # Determine status
        if inp.furnace_pressure_inwc >= self.MAX_DRAFT_INWC:
            status = DraftStatus.CRITICAL
            warnings.append(f"CRITICAL: Positive pressure ({inp.furnace_pressure_inwc:.3f} inWC) - flue gas escape risk")
        elif inp.furnace_pressure_inwc <= self.MIN_DRAFT_INWC:
            status = DraftStatus.LOW
            warnings.append(f"Excessive draft ({inp.furnace_pressure_inwc:.3f} inWC) - air infiltration risk")
        elif abs(deviation) <= 0.02:
            status = DraftStatus.OPTIMAL
        elif abs(deviation) <= 0.1:
            status = DraftStatus.ACCEPTABLE
        else:
            status = DraftStatus.HIGH if deviation > 0 else DraftStatus.LOW

        # Optimal damper position
        optimal_damper = calculate_optimal_damper(
            abs(inp.target_pressure_inwc),
            stack_effect,
            inp.damper_position_pct
        )

        # Draft loss
        draft_loss = 0.0
        if stack_effect > 0:
            loss = stack_effect - abs(inp.furnace_pressure_inwc)
            draft_loss = round(max(0, loss / stack_effect * 100), 1)

        # Energy impact estimation
        pressure_drop_pa = (draft_loss / 100) * 249.0
        mean_temp_k = (inp.stack_temperature_c + inp.ambient_temperature_c) / 2 + 273.15
        density = 1.293 * (273.15 / mean_temp_k)
        volume_flow = inp.flue_gas_flow_kg_s / density
        energy_impact = round((volume_flow * pressure_drop_pa) / (1000 * 0.7), 2)

        # Recommendations
        if status == DraftStatus.CRITICAL:
            recommendations.append("URGENT: Increase stack damper opening to restore negative pressure")
        elif status == DraftStatus.LOW:
            recommendations.append("Reduce stack damper opening to decrease excessive draft")

        if draft_loss > 30:
            recommendations.append(f"Draft loss at {draft_loss:.1f}% - inspect ductwork")

        if inp.excess_air_pct > 25:
            recommendations.append(f"Excess air at {inp.excess_air_pct:.1f}% - optimize combustion")

        damper_diff = abs(optimal_damper - inp.damper_position_pct)
        if damper_diff > 10:
            recommendations.append(f"Adjust damper from {inp.damper_position_pct:.0f}% to {optimal_damper:.0f}%")

        # Provenance hash
        calc_hash = hashlib.sha256(json.dumps({
            "stack_effect": stack_effect,
            "status": status.value,
            "furnace_id": inp.furnace_id
        }).encode()).hexdigest()

        return DraftControlOutput(
            furnace_id=inp.furnace_id,
            stack_effect_inwc=stack_effect,
            stack_effect_pa=stack_effect_pa,
            theoretical_draft_inwc=stack_effect,
            overall_status=status,
            pressure_deviation_inwc=round(deviation, 4),
            optimal_damper_position_pct=optimal_damper,
            draft_loss_pct=draft_loss,
            energy_impact_kw=energy_impact,
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
            "category": "Process Heat",
            "type": "Control",
            "standards": ["NFPA 86", "API 560"]
        }


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-046",
    "name": "DRAFT-CONTROL",
    "version": "1.0.0",
    "summary": "Furnace draft and pressure control with stack effect calculation",
    "tags": ["furnace", "draft-control", "NFPA-86", "API-560"],
    "standards": [
        {"ref": "NFPA 86", "description": "Standard for Ovens and Furnaces"},
        {"ref": "API 560", "description": "Fired Heaters for General Refinery Service"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
