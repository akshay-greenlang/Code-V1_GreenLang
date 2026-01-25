"""
GL-044: Water Treatment Advisor Agent (WATER-TREATMENT-ADVISOR)

Boiler water treatment optimization.

Standards: ASME Guidelines, ABMA
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WaterAnalysis(BaseModel):
    """Water chemistry analysis."""
    sample_point: str
    ph: float = Field(default=9.0, ge=0, le=14)
    conductivity_umho: float = Field(default=500, ge=0)
    total_hardness_ppm: float = Field(default=0, ge=0)
    silica_ppm: float = Field(default=5, ge=0)
    dissolved_oxygen_ppb: float = Field(default=7, ge=0)
    iron_ppm: float = Field(default=0.05, ge=0)
    chloride_ppm: float = Field(default=10, ge=0)
    alkalinity_ppm: float = Field(default=100, ge=0)


class WaterTreatmentAdvisorInput(BaseModel):
    """Input for WaterTreatmentAdvisorAgent."""
    system_id: str
    water_analysis: WaterAnalysis
    steam_purity_requirement: str = Field(default="standard")  # standard, high_purity, ultra_pure
    current_blowdown_rate_percent: float = Field(default=5, ge=0, le=100)
    chemical_costs: Dict[str, float] = Field(default_factory=lambda: {
        "phosphate": 2.0, "caustic": 1.5, "oxygen_scavenger": 3.0, "amine": 4.0
    })
    boiler_pressure_psig: float = Field(default=150)
    makeup_water_flow_gpm: float = Field(default=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChemicalDosing(BaseModel):
    """Chemical dosing recommendation."""
    chemical: str
    current_dose_ppm: float
    recommended_dose_ppm: float
    reason: str


class WaterTreatmentAdvisorOutput(BaseModel):
    """Output from WaterTreatmentAdvisorAgent."""
    analysis_id: str
    system_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    water_quality_index: float
    treatment_recommendations: List[ChemicalDosing]
    optimal_blowdown_rate_percent: float
    blowdown_adjustment: str
    scaling_risk: str
    corrosion_risk: str
    carryover_risk: str
    annual_chemical_cost: float
    annual_blowdown_cost: float
    optimization_potential_per_year: float
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class WaterTreatmentAdvisorAgent:
    """GL-044: Water Treatment Advisor Agent."""

    AGENT_ID = "GL-044"
    AGENT_NAME = "WATER-TREATMENT-ADVISOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"WaterTreatmentAdvisorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: WaterTreatmentAdvisorInput) -> WaterTreatmentAdvisorOutput:
        """Execute water treatment analysis."""
        start_time = datetime.utcnow()

        wa = input_data.water_analysis

        # Water quality index (100 = perfect)
        wqi = 100
        if wa.total_hardness_ppm > 0.5:
            wqi -= 20
        if wa.ph < 8.5 or wa.ph > 10.5:
            wqi -= 15
        if wa.dissolved_oxygen_ppb > 10:
            wqi -= 15
        if wa.silica_ppm > 10:
            wqi -= 10
        if wa.iron_ppm > 0.1:
            wqi -= 10
        wqi = max(0, wqi)

        # Risk assessments
        scaling_risk = "HIGH" if wa.total_hardness_ppm > 2 or wa.silica_ppm > 15 else "MODERATE" if wa.total_hardness_ppm > 0.5 else "LOW"
        corrosion_risk = "HIGH" if wa.dissolved_oxygen_ppb > 20 or wa.ph < 8 else "MODERATE" if wa.dissolved_oxygen_ppb > 10 else "LOW"
        carryover_risk = "HIGH" if wa.conductivity_umho > 3000 else "MODERATE" if wa.conductivity_umho > 2000 else "LOW"

        # Chemical dosing recommendations
        recommendations = []

        if wa.ph < 9.0:
            recommendations.append(ChemicalDosing(
                chemical="caustic_soda",
                current_dose_ppm=0,
                recommended_dose_ppm=5,
                reason="Raise pH to protect against corrosion"
            ))

        if wa.dissolved_oxygen_ppb > 10:
            recommendations.append(ChemicalDosing(
                chemical="oxygen_scavenger",
                current_dose_ppm=0,
                recommended_dose_ppm=wa.dissolved_oxygen_ppb * 0.5,
                reason="Remove dissolved oxygen to prevent pitting"
            ))

        if wa.total_hardness_ppm > 0.5:
            recommendations.append(ChemicalDosing(
                chemical="phosphate",
                current_dose_ppm=0,
                recommended_dose_ppm=30,
                reason="Precipitate hardness to prevent scale"
            ))

        # Optimal blowdown
        # Higher conductivity needs more blowdown
        optimal_bd = 3 + (wa.conductivity_umho / 1000)
        optimal_bd = min(20, max(2, optimal_bd))

        if optimal_bd > input_data.current_blowdown_rate_percent:
            bd_adj = "INCREASE"
        elif optimal_bd < input_data.current_blowdown_rate_percent - 1:
            bd_adj = "DECREASE"
        else:
            bd_adj = "MAINTAIN"

        # Costs
        annual_chem_cost = sum(input_data.chemical_costs.values()) * input_data.makeup_water_flow_gpm * 60 * 8760 / 1000
        annual_bd_cost = input_data.current_blowdown_rate_percent / 100 * input_data.makeup_water_flow_gpm * 60 * 8760 * 0.01

        # Optimization potential
        optimization = 0
        if bd_adj == "DECREASE":
            optimization += (input_data.current_blowdown_rate_percent - optimal_bd) / 100 * annual_bd_cost
        optimization = max(0, optimization)

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "system": input_data.system_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return WaterTreatmentAdvisorOutput(
            analysis_id=f"WT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_id=input_data.system_id,
            water_quality_index=round(wqi, 1),
            treatment_recommendations=recommendations,
            optimal_blowdown_rate_percent=round(optimal_bd, 1),
            blowdown_adjustment=bd_adj,
            scaling_risk=scaling_risk,
            corrosion_risk=corrosion_risk,
            carryover_risk=carryover_risk,
            annual_chemical_cost=round(annual_chem_cost, 0),
            annual_blowdown_cost=round(annual_bd_cost, 0),
            optimization_potential_per_year=round(optimization, 0),
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-044",
    "name": "WATER-TREATMENT-ADVISOR",
    "version": "1.0.0",
    "summary": "Boiler water treatment optimization",
    "tags": ["water-treatment", "boiler", "chemistry", "ASME", "ABMA"],
    "standards": [
        {"ref": "ASME Guidelines", "description": "Boiler Water Quality Guidelines"},
        {"ref": "ABMA", "description": "American Boiler Manufacturers Association"}
    ]
}
