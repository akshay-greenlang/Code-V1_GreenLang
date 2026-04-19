"""GL-077: Life Cycle Assessment Agent (LCA).

Performs life cycle assessment for energy systems.

Standards: ISO 14040, ISO 14044
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LifeCyclePhase(str, Enum):
    RAW_MATERIALS = "RAW_MATERIALS"
    MANUFACTURING = "MANUFACTURING"
    TRANSPORTATION = "TRANSPORTATION"
    INSTALLATION = "INSTALLATION"
    OPERATION = "OPERATION"
    MAINTENANCE = "MAINTENANCE"
    END_OF_LIFE = "END_OF_LIFE"


class ImpactCategory(str, Enum):
    GLOBAL_WARMING = "GLOBAL_WARMING"
    ACIDIFICATION = "ACIDIFICATION"
    EUTROPHICATION = "EUTROPHICATION"
    OZONE_DEPLETION = "OZONE_DEPLETION"
    RESOURCE_DEPLETION = "RESOURCE_DEPLETION"


class PhaseInput(BaseModel):
    phase: LifeCyclePhase
    energy_kwh: float = Field(default=0, ge=0)
    materials_kg: float = Field(default=0, ge=0)
    transport_km: float = Field(default=0, ge=0)
    waste_kg: float = Field(default=0, ge=0)
    water_m3: float = Field(default=0, ge=0)


class LCAInput(BaseModel):
    product_id: str
    product_name: str = Field(default="Product")
    functional_unit: str = Field(default="1 kWh delivered")
    lifetime_years: int = Field(default=20, ge=1)
    annual_output_kwh: float = Field(default=100000, ge=0)
    phases: List[PhaseInput] = Field(default_factory=list)
    electricity_emission_factor: float = Field(default=0.5, ge=0)  # kg CO2/kWh
    transport_emission_factor: float = Field(default=0.1, ge=0)  # kg CO2/km/kg
    material_emission_factor: float = Field(default=2.0, ge=0)  # kg CO2/kg
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PhaseImpact(BaseModel):
    phase: str
    co2_kg: float
    share_pct: float
    energy_kwh: float


class LCAOutput(BaseModel):
    product_id: str
    product_name: str
    total_co2_kg: float
    co2_per_kwh_g: float
    phase_impacts: List[PhaseImpact]
    dominant_phase: str
    dominant_share_pct: float
    lifetime_energy_kwh: float
    energy_payback_years: float
    carbon_payback_years: float
    improvement_potential_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class LCAAgent:
    AGENT_ID = "GL-077"
    AGENT_NAME = "LCA"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"LCAAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = LCAInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_phase_impact(self, phase: PhaseInput, inp: LCAInput) -> float:
        """Calculate CO2 impact for a phase."""
        co2 = 0

        # Energy-related emissions
        co2 += phase.energy_kwh * inp.electricity_emission_factor

        # Material-related emissions
        co2 += phase.materials_kg * inp.material_emission_factor

        # Transport-related emissions
        co2 += phase.transport_km * phase.materials_kg * inp.transport_emission_factor / 1000

        # Waste treatment (simplified)
        co2 += phase.waste_kg * 0.5

        return co2

    def _process(self, inp: LCAInput) -> LCAOutput:
        recommendations = []

        # Calculate impacts per phase
        phase_impacts = []
        total_co2 = 0
        total_energy = 0

        for phase in inp.phases:
            co2 = self._calculate_phase_impact(phase, inp)
            total_co2 += co2
            total_energy += phase.energy_kwh
            phase_impacts.append({
                "phase": phase.phase.value,
                "co2_kg": co2,
                "energy_kwh": phase.energy_kwh
            })

        # Calculate shares
        impacts = []
        for pi in phase_impacts:
            share = (pi["co2_kg"] / total_co2 * 100) if total_co2 > 0 else 0
            impacts.append(PhaseImpact(
                phase=pi["phase"],
                co2_kg=round(pi["co2_kg"], 2),
                share_pct=round(share, 1),
                energy_kwh=round(pi["energy_kwh"], 2)
            ))

        # Dominant phase
        if impacts:
            dominant = max(impacts, key=lambda x: x.share_pct)
            dominant_phase = dominant.phase
            dominant_share = dominant.share_pct
        else:
            dominant_phase = "UNKNOWN"
            dominant_share = 0

        # Lifetime energy output
        lifetime_energy = inp.annual_output_kwh * inp.lifetime_years

        # CO2 per kWh delivered
        co2_per_kwh = (total_co2 / lifetime_energy * 1000) if lifetime_energy > 0 else 0

        # Energy payback (years to recover embodied energy)
        embodied_energy = total_energy
        if inp.annual_output_kwh > 0:
            energy_payback = embodied_energy / inp.annual_output_kwh
        else:
            energy_payback = inp.lifetime_years

        # Carbon payback (years to offset vs grid)
        grid_avoided_annual = inp.annual_output_kwh * inp.electricity_emission_factor
        if grid_avoided_annual > 0:
            carbon_payback = total_co2 / grid_avoided_annual
        else:
            carbon_payback = inp.lifetime_years

        # Improvement potential (operational phase reduction)
        op_phase = next((p for p in impacts if p.phase == "OPERATION"), None)
        improvement = op_phase.share_pct * 0.3 if op_phase else 10

        # Recommendations
        if dominant_phase == "OPERATION":
            recommendations.append(f"Operation phase dominates ({dominant_share:.0f}%) - focus on efficiency improvements")
        elif dominant_phase == "MANUFACTURING":
            recommendations.append(f"Manufacturing phase significant ({dominant_share:.0f}%) - consider recycled materials")
        elif dominant_phase == "RAW_MATERIALS":
            recommendations.append(f"Raw materials impact high ({dominant_share:.0f}%) - evaluate alternative materials")

        if energy_payback > 3:
            recommendations.append(f"Energy payback {energy_payback:.1f} years - optimize embodied energy")
        if carbon_payback > 5:
            recommendations.append(f"Carbon payback {carbon_payback:.1f} years - consider lower-carbon manufacturing")
        if co2_per_kwh > 100:
            recommendations.append(f"High lifecycle emissions ({co2_per_kwh:.0f} g/kWh) - below grid average but improvement possible")

        calc_hash = hashlib.sha256(json.dumps({
            "product": inp.product_id,
            "total_co2": round(total_co2, 2),
            "co2_per_kwh": round(co2_per_kwh, 2)
        }).encode()).hexdigest()

        return LCAOutput(
            product_id=inp.product_id,
            product_name=inp.product_name,
            total_co2_kg=round(total_co2, 2),
            co2_per_kwh_g=round(co2_per_kwh, 2),
            phase_impacts=impacts,
            dominant_phase=dominant_phase,
            dominant_share_pct=round(dominant_share, 1),
            lifetime_energy_kwh=round(lifetime_energy, 0),
            energy_payback_years=round(energy_payback, 2),
            carbon_payback_years=round(carbon_payback, 2),
            improvement_potential_pct=round(improvement, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-077", "name": "LCA", "version": "1.0.0",
    "summary": "Life cycle assessment for energy systems",
    "standards": [{"ref": "ISO 14040"}, {"ref": "ISO 14044"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
