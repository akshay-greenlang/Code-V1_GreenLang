"""GL-037: Biomass Combustion Agent (BIOMASS).

Optimizes biomass combustion for renewable heat generation.

Standards: EN 303-5, EPA 40 CFR 60
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BiomassType(str, Enum):
    WOOD_CHIPS = "WOOD_CHIPS"
    WOOD_PELLETS = "WOOD_PELLETS"
    AGRICULTURAL = "AGRICULTURAL"
    BAGASSE = "BAGASSE"


class BiomassCombustionInput(BaseModel):
    equipment_id: str
    biomass_type: BiomassType = Field(default=BiomassType.WOOD_CHIPS)
    fuel_rate_kg_hr: float = Field(..., gt=0)
    moisture_content_pct: float = Field(default=20, ge=0, le=80)
    ash_content_pct: float = Field(default=2, ge=0, le=30)
    lower_heating_value_mj_kg: float = Field(default=15, gt=0)
    combustion_air_temp_c: float = Field(default=25, ge=-20)
    excess_air_pct: float = Field(default=30, ge=0, le=200)
    flue_gas_temp_c: float = Field(default=180, gt=0)
    operating_hours_year: int = Field(default=6000)
    carbon_credit_per_tco2: float = Field(default=50, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BiomassCombustionOutput(BaseModel):
    equipment_id: str
    thermal_output_kw: float
    combustion_efficiency_pct: float
    boiler_efficiency_pct: float
    co2_avoided_kg_hr: float
    annual_co2_avoided_tonnes: float
    carbon_credit_value_usd: float
    optimal_moisture_pct: float
    optimal_excess_air_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class BiomassCombustionAgent:
    AGENT_ID = "GL-037"
    AGENT_NAME = "BIOMASS"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"BiomassCombustionAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = BiomassCombustionInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: BiomassCombustionInput) -> BiomassCombustionOutput:
        recommendations = []

        # Adjust LHV for moisture: LHV_wet = LHV_dry * (1 - M) - 2.44 * M
        lhv_wet = inp.lower_heating_value_mj_kg * (1 - inp.moisture_content_pct/100) - 2.44 * (inp.moisture_content_pct/100)

        # Heat input
        heat_input_kw = inp.fuel_rate_kg_hr * lhv_wet * 1000 / 3600

        # Combustion efficiency (affected by excess air and moisture)
        comb_eff = 100 - (inp.excess_air_pct * 0.05) - (inp.moisture_content_pct * 0.3)
        comb_eff = max(70, min(99, comb_eff))

        # Flue gas loss
        flue_loss = (inp.flue_gas_temp_c - inp.combustion_air_temp_c) * 0.045 * (1 + inp.excess_air_pct/100)

        # Boiler efficiency
        boiler_eff = comb_eff - flue_loss - (inp.ash_content_pct * 0.5)
        boiler_eff = max(60, min(95, boiler_eff))

        # Thermal output
        thermal_output = heat_input_kw * (boiler_eff / 100)

        # CO2 avoided vs natural gas (56 kg CO2/GJ for gas, biomass ~neutral)
        ng_emission_factor = 56  # kg CO2/GJ
        heat_gj_hr = thermal_output * 3600 / 1e6
        co2_avoided_hr = heat_gj_hr * ng_emission_factor
        annual_co2_tonnes = co2_avoided_hr * inp.operating_hours_year / 1000
        carbon_value = annual_co2_tonnes * inp.carbon_credit_per_tco2

        # Optimal parameters
        optimal_moisture = 15 if inp.biomass_type == BiomassType.WOOD_PELLETS else 25
        optimal_excess_air = 25 if inp.biomass_type == BiomassType.WOOD_PELLETS else 35

        # Recommendations
        if inp.moisture_content_pct > 40:
            recommendations.append(f"High moisture ({inp.moisture_content_pct}%) - pre-dry fuel")
        if inp.excess_air_pct > 50:
            recommendations.append(f"Reduce excess air from {inp.excess_air_pct}% to {optimal_excess_air}%")
        if inp.flue_gas_temp_c > 200:
            recommendations.append("Add economizer to recover flue gas heat")
        if boiler_eff < 80:
            recommendations.append(f"Low efficiency ({boiler_eff:.1f}%) - tune combustion")
        if inp.ash_content_pct > 5:
            recommendations.append("High ash content - increase ash removal frequency")

        calc_hash = hashlib.sha256(json.dumps({
            "equipment": inp.equipment_id,
            "thermal_kw": round(thermal_output, 1),
            "efficiency": round(boiler_eff, 1)
        }).encode()).hexdigest()

        return BiomassCombustionOutput(
            equipment_id=inp.equipment_id,
            thermal_output_kw=round(thermal_output, 1),
            combustion_efficiency_pct=round(comb_eff, 1),
            boiler_efficiency_pct=round(boiler_eff, 1),
            co2_avoided_kg_hr=round(co2_avoided_hr, 1),
            annual_co2_avoided_tonnes=round(annual_co2_tonnes, 1),
            carbon_credit_value_usd=round(carbon_value, 2),
            optimal_moisture_pct=optimal_moisture,
            optimal_excess_air_pct=optimal_excess_air,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-037", "name": "BIOMASS", "version": "1.0.0",
    "summary": "Biomass combustion optimization for renewable heat",
    "standards": [{"ref": "EN 303-5"}, {"ref": "EPA 40 CFR 60"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
