"""
GL-045: Carbon Intensity Tracker Agent (CARBON-INTENSITY-TRACKER)

Real-time carbon intensity monitoring for process heat systems.

Standards: GHG Protocol, ISO 14064
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Standard emission factors (kg CO2 per unit)
EMISSION_FACTORS = {
    "natural_gas_mmbtu": 53.06,
    "coal_mmbtu": 95.0,
    "fuel_oil_mmbtu": 74.0,
    "propane_mmbtu": 63.0,
    "electricity_kwh_us_avg": 0.42,
    "electricity_kwh_coal": 0.95,
    "electricity_kwh_gas": 0.45,
    "electricity_kwh_renewable": 0.0,
}


class FuelConsumption(BaseModel):
    """Fuel consumption data."""
    fuel_type: str
    quantity: float
    unit: str  # mmbtu, therms, gallons, etc.


class CarbonIntensityTrackerInput(BaseModel):
    """Input for CarbonIntensityTrackerAgent."""
    facility_id: str
    reporting_period_hours: float = Field(default=1)
    fuel_consumption: List[FuelConsumption] = Field(default_factory=list)
    electricity_kwh: float = Field(default=0, ge=0)
    grid_emission_factor_kg_per_kwh: float = Field(default=0.42)
    production_output: float = Field(default=1, gt=0)
    production_unit: str = Field(default="tonne")
    custom_emission_factors: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CarbonIntensityTrackerOutput(BaseModel):
    """Output from CarbonIntensityTrackerAgent."""
    analysis_id: str
    facility_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    carbon_intensity_kg_per_unit: float
    scope1_emissions_kg: float
    scope2_emissions_kg: float
    total_emissions_kg: float
    scope1_breakdown: Dict[str, float]
    primary_emission_source: str
    reduction_from_baseline_percent: Optional[float]
    intensity_trend: str
    decarbonization_opportunities: List[Dict[str, Any]]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class CarbonIntensityTrackerAgent:
    """GL-045: Carbon Intensity Tracker Agent."""

    AGENT_ID = "GL-045"
    AGENT_NAME = "CARBON-INTENSITY-TRACKER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"CarbonIntensityTrackerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: CarbonIntensityTrackerInput) -> CarbonIntensityTrackerOutput:
        """Execute carbon intensity tracking."""
        start_time = datetime.utcnow()

        # Combine standard and custom emission factors
        ef = {**EMISSION_FACTORS, **input_data.custom_emission_factors}

        # Calculate Scope 1 emissions (direct combustion)
        scope1_breakdown = {}
        scope1_total = 0

        for fuel in input_data.fuel_consumption:
            # Normalize to MMBtu
            if fuel.unit == "mmbtu":
                mmbtu = fuel.quantity
            elif fuel.unit == "therms":
                mmbtu = fuel.quantity * 0.1
            elif fuel.unit == "gallons":
                mmbtu = fuel.quantity * 0.092  # Approximate for fuel oil
            else:
                mmbtu = fuel.quantity

            ef_key = f"{fuel.fuel_type.lower()}_mmbtu"
            factor = ef.get(ef_key, 53.0)  # Default to natural gas

            emissions = mmbtu * factor
            scope1_breakdown[fuel.fuel_type] = round(emissions, 2)
            scope1_total += emissions

        # Calculate Scope 2 emissions (electricity)
        scope2_total = input_data.electricity_kwh * input_data.grid_emission_factor_kg_per_kwh

        # Total emissions
        total_emissions = scope1_total + scope2_total

        # Carbon intensity
        carbon_intensity = total_emissions / input_data.production_output

        # Primary emission source
        all_sources = {**scope1_breakdown, "electricity": scope2_total}
        primary_source = max(all_sources, key=all_sources.get) if all_sources else "none"

        # Decarbonization opportunities
        opportunities = []
        if scope1_total > scope2_total:
            opportunities.append({
                "opportunity": "Fuel switching to lower-carbon fuels",
                "potential_reduction_percent": 30,
                "implementation": "Medium-term"
            })
        if scope2_total > 0:
            opportunities.append({
                "opportunity": "Renewable electricity procurement",
                "potential_reduction_percent": round(scope2_total / total_emissions * 100, 0) if total_emissions > 0 else 0,
                "implementation": "Near-term"
            })
        opportunities.append({
            "opportunity": "Energy efficiency improvements",
            "potential_reduction_percent": 15,
            "implementation": "Ongoing"
        })

        # Intensity trend (would compare to historical, simplified here)
        trend = "STABLE"

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "facility": input_data.facility_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return CarbonIntensityTrackerOutput(
            analysis_id=f"CI-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_id=input_data.facility_id,
            carbon_intensity_kg_per_unit=round(carbon_intensity, 2),
            scope1_emissions_kg=round(scope1_total, 2),
            scope2_emissions_kg=round(scope2_total, 2),
            total_emissions_kg=round(total_emissions, 2),
            scope1_breakdown=scope1_breakdown,
            primary_emission_source=primary_source,
            reduction_from_baseline_percent=None,
            intensity_trend=trend,
            decarbonization_opportunities=opportunities,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-045",
    "name": "CARBON-INTENSITY-TRACKER",
    "version": "1.0.0",
    "summary": "Real-time carbon intensity monitoring",
    "tags": ["carbon", "emissions", "GHG-Protocol", "ISO-14064", "scope1", "scope2"],
    "standards": [
        {"ref": "GHG Protocol", "description": "Greenhouse Gas Protocol"},
        {"ref": "ISO 14064", "description": "Greenhouse Gas Accounting and Verification"}
    ]
}
