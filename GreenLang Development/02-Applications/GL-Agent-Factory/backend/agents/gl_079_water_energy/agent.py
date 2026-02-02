"""GL-079: Water-Energy Nexus Agent (WATER-ENERGY).

Optimizes the water-energy nexus for industrial facilities.

Standards: ISO 46001, Alliance for Water Stewardship
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WaterUseType(str, Enum):
    COOLING = "COOLING"
    PROCESS = "PROCESS"
    BOILER = "BOILER"
    SANITARY = "SANITARY"
    LANDSCAPE = "LANDSCAPE"


class WaterSource(str, Enum):
    MUNICIPAL = "MUNICIPAL"
    GROUNDWATER = "GROUNDWATER"
    SURFACE = "SURFACE"
    RECYCLED = "RECYCLED"
    RAINWATER = "RAINWATER"


class WaterStream(BaseModel):
    stream_id: str
    use_type: WaterUseType
    source: WaterSource
    consumption_m3_day: float = Field(ge=0)
    energy_for_pumping_kwh_m3: float = Field(default=0.5, ge=0)
    energy_for_treatment_kwh_m3: float = Field(default=0.3, ge=0)
    temperature_in_c: float = Field(default=15)
    temperature_out_c: float = Field(default=25)


class WaterEnergyInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    water_streams: List[WaterStream] = Field(default_factory=list)
    water_cost_m3: float = Field(default=2.0, ge=0)
    electricity_cost_kwh: float = Field(default=0.10, ge=0)
    wastewater_cost_m3: float = Field(default=3.0, ge=0)
    operating_days_year: int = Field(default=300, ge=1)
    cooling_tower_efficiency: float = Field(default=0.70, ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NexusOpportunity(BaseModel):
    stream_id: str
    description: str
    water_savings_m3_day: float
    energy_savings_kwh_day: float
    annual_cost_savings_usd: float
    implementation_cost_usd: float
    payback_years: float


class WaterEnergyOutput(BaseModel):
    facility_id: str
    total_water_m3_day: float
    total_energy_for_water_kwh_day: float
    water_intensity_m3_per_mwh: float
    energy_intensity_kwh_per_m3: float
    annual_water_cost_usd: float
    annual_energy_for_water_cost_usd: float
    heat_recovery_potential_kwh_day: float
    recycling_potential_m3_day: float
    opportunities: List[NexusOpportunity]
    total_savings_potential_usd: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class WaterEnergyAgent:
    AGENT_ID = "GL-079"
    AGENT_NAME = "WATER-ENERGY"
    VERSION = "1.0.0"

    # Specific heat of water (kWh/m³·°C)
    WATER_SPECIFIC_HEAT = 1.163

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"WaterEnergyAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = WaterEnergyInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _identify_opportunities(self, streams: List[WaterStream], inp: WaterEnergyInput) -> List[NexusOpportunity]:
        """Identify water-energy nexus opportunities."""
        opportunities = []

        for stream in streams:
            # Cooling water recycling
            if stream.use_type == WaterUseType.COOLING:
                recycling_potential = stream.consumption_m3_day * 0.3  # 30% recyclable
                water_savings = recycling_potential
                energy_savings = water_savings * stream.energy_for_pumping_kwh_m3

                annual_savings = (
                    water_savings * inp.water_cost_m3 +
                    energy_savings * inp.electricity_cost_kwh +
                    water_savings * inp.wastewater_cost_m3
                ) * inp.operating_days_year

                impl_cost = water_savings * 100  # Rough estimate

                if annual_savings > 0:
                    opportunities.append(NexusOpportunity(
                        stream_id=stream.stream_id,
                        description="Implement cooling water recycling",
                        water_savings_m3_day=round(water_savings, 2),
                        energy_savings_kwh_day=round(energy_savings, 2),
                        annual_cost_savings_usd=round(annual_savings, 2),
                        implementation_cost_usd=round(impl_cost, 2),
                        payback_years=round(impl_cost / annual_savings, 2) if annual_savings > 0 else 99
                    ))

            # Heat recovery from warm water
            delta_t = stream.temperature_out_c - stream.temperature_in_c
            if delta_t > 5:
                heat_energy = stream.consumption_m3_day * self.WATER_SPECIFIC_HEAT * delta_t * 0.5  # 50% recovery
                energy_value = heat_energy * inp.electricity_cost_kwh * inp.operating_days_year * 0.3  # COP factor

                if energy_value > 1000:
                    opportunities.append(NexusOpportunity(
                        stream_id=stream.stream_id,
                        description=f"Heat recovery from {stream.use_type.value} water",
                        water_savings_m3_day=0,
                        energy_savings_kwh_day=round(heat_energy, 2),
                        annual_cost_savings_usd=round(energy_value, 2),
                        implementation_cost_usd=round(energy_value * 2, 2),
                        payback_years=2.0
                    ))

        return opportunities

    def _process(self, inp: WaterEnergyInput) -> WaterEnergyOutput:
        recommendations = []

        # Calculate totals
        total_water = sum(s.consumption_m3_day for s in inp.water_streams)
        total_energy = sum(
            s.consumption_m3_day * (s.energy_for_pumping_kwh_m3 + s.energy_for_treatment_kwh_m3)
            for s in inp.water_streams
        )

        # Intensities
        energy_intensity = total_energy / total_water if total_water > 0 else 0
        # Assume 1 MWh production per 10 m³ water for water intensity
        water_intensity = total_water / (total_energy / 1000) if total_energy > 0 else 0

        # Annual costs
        annual_water = total_water * inp.water_cost_m3 * inp.operating_days_year
        annual_energy = total_energy * inp.electricity_cost_kwh * inp.operating_days_year

        # Heat recovery potential
        heat_recovery = sum(
            s.consumption_m3_day * self.WATER_SPECIFIC_HEAT * (s.temperature_out_c - s.temperature_in_c)
            for s in inp.water_streams if s.temperature_out_c > s.temperature_in_c + 5
        )

        # Recycling potential
        recycling = sum(
            s.consumption_m3_day * 0.3 for s in inp.water_streams
            if s.use_type in [WaterUseType.COOLING, WaterUseType.PROCESS]
        )

        # Identify opportunities
        opportunities = self._identify_opportunities(inp.water_streams, inp)
        opportunities.sort(key=lambda x: x.payback_years)

        total_savings = sum(o.annual_cost_savings_usd for o in opportunities)

        # Recommendations
        if energy_intensity > 1.0:
            recommendations.append(f"High energy intensity ({energy_intensity:.2f} kWh/m³) - optimize pumping")
        if water_intensity > 5:
            recommendations.append(f"High water intensity ({water_intensity:.1f} m³/MWh) - implement conservation")

        recycled_streams = [s for s in inp.water_streams if s.source == WaterSource.RECYCLED]
        if not recycled_streams:
            recommendations.append("No recycled water usage - evaluate recycling opportunities")

        cooling_streams = [s for s in inp.water_streams if s.use_type == WaterUseType.COOLING]
        if cooling_streams:
            cooling_pct = sum(s.consumption_m3_day for s in cooling_streams) / total_water * 100
            if cooling_pct > 50:
                recommendations.append(f"Cooling dominates water use ({cooling_pct:.0f}%) - optimize cooling tower")

        if heat_recovery > 100:
            recommendations.append(f"Heat recovery potential {heat_recovery:.0f} kWh/day from warm water")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "total_water": round(total_water, 2),
            "total_energy": round(total_energy, 2)
        }).encode()).hexdigest()

        return WaterEnergyOutput(
            facility_id=inp.facility_id,
            total_water_m3_day=round(total_water, 2),
            total_energy_for_water_kwh_day=round(total_energy, 2),
            water_intensity_m3_per_mwh=round(water_intensity, 2),
            energy_intensity_kwh_per_m3=round(energy_intensity, 2),
            annual_water_cost_usd=round(annual_water, 2),
            annual_energy_for_water_cost_usd=round(annual_energy, 2),
            heat_recovery_potential_kwh_day=round(heat_recovery, 2),
            recycling_potential_m3_day=round(recycling, 2),
            opportunities=opportunities,
            total_savings_potential_usd=round(total_savings, 2),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-079", "name": "WATER-ENERGY", "version": "1.0.0",
    "summary": "Water-energy nexus optimization",
    "standards": [{"ref": "ISO 46001"}, {"ref": "Alliance for Water Stewardship"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
