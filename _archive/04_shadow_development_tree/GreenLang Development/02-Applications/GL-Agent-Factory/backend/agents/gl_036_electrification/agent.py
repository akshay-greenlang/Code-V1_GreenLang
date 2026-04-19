"""GL-036: Electrification Agent (ELECTRIFICATION).

Analyzes industrial electrification opportunities for decarbonization.

Standards: IEC, IEEE, NFPA 70
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


class FuelType(str, Enum):
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    DIESEL = "diesel"


class ElecTechnology(str, Enum):
    HEAT_PUMP = "heat_pump"
    ELECTRIC_BOILER = "electric_boiler"
    INDUCTION = "induction"
    INFRARED = "infrared"
    RESISTANCE = "resistance"


class Feasibility(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    CHALLENGING = "challenging"


class HeatLoad(BaseModel):
    """Process heat load specification."""
    load_id: str
    name: str
    thermal_demand_kw: float = Field(..., ge=0)
    temperature_c: float
    operating_hours_year: int = Field(default=8000, ge=0, le=8760)
    current_fuel: FuelType
    current_efficiency_pct: float = Field(default=80, ge=0, le=100)


class ElectrificationInput(BaseModel):
    """Input for electrification analysis."""
    facility_name: str = Field(default="Facility")
    heat_loads: List[HeatLoad] = Field(..., min_length=1)
    grid_carbon_intensity_gco2_kwh: float = Field(default=400, ge=0)
    electricity_price_kwh: float = Field(default=0.10, ge=0)
    natural_gas_price_mmbtu: float = Field(default=5.0, ge=0)
    carbon_price_tco2: float = Field(default=50.0, ge=0)
    analysis_years: int = Field(default=20, ge=1, le=50)
    discount_rate_pct: float = Field(default=8, ge=0, le=30)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoadAnalysis(BaseModel):
    """Analysis result for a single load."""
    load_id: str
    load_name: str
    recommended_tech: ElecTechnology
    feasibility: Feasibility
    electrical_demand_kw: float
    annual_electricity_kwh: float
    current_annual_cost_usd: float
    electrified_annual_cost_usd: float
    annual_savings_usd: float
    capital_cost_usd: float
    payback_years: Optional[float]
    npv_usd: float
    co2_reduction_tpa: float
    co2_reduction_pct: float


class ElectrificationOutput(BaseModel):
    """Output from electrification analysis."""
    facility_name: str
    total_loads: int
    feasible_loads: int
    electrification_potential_pct: float
    total_co2_reduction_tpa: float
    total_savings_usd: float
    total_capital_usd: float
    overall_payback_years: Optional[float]
    overall_npv_usd: float
    load_analyses: List[LoadAnalysis]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


# Emission factors kg CO2/kWh thermal
FUEL_EMISSIONS = {
    FuelType.NATURAL_GAS: 0.184,
    FuelType.LPG: 0.214,
    FuelType.FUEL_OIL: 0.267,
    FuelType.COAL: 0.341,
    FuelType.DIESEL: 0.253,
}

# Technology specs: max_temp, COP/efficiency, capex $/kW
TECH_SPECS = {
    ElecTechnology.HEAT_PUMP: (150, 3.0, 800),
    ElecTechnology.ELECTRIC_BOILER: (200, 0.98, 150),
    ElecTechnology.INDUCTION: (1600, 0.90, 500),
    ElecTechnology.INFRARED: (600, 0.85, 300),
    ElecTechnology.RESISTANCE: (1200, 0.97, 200),
}


class ElectrificationAgent:
    """GL-036: Electrification Agent - Industrial electrification analysis."""

    AGENT_ID = "GL-036"
    AGENT_NAME = "ELECTRIFICATION"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"ElectrificationAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ElectrificationInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _select_technology(self, temp_c: float) -> ElecTechnology:
        """Select best electrification technology for temperature."""
        if temp_c <= 100:
            return ElecTechnology.HEAT_PUMP
        elif temp_c <= 200:
            return ElecTechnology.ELECTRIC_BOILER
        elif temp_c <= 600:
            return ElecTechnology.INFRARED
        elif temp_c <= 1200:
            return ElecTechnology.RESISTANCE
        else:
            return ElecTechnology.INDUCTION

    def _calculate_npv(self, capex: float, annual_savings: float, years: int, rate: float) -> float:
        """Calculate NPV."""
        r = rate / 100
        npv = -capex
        for y in range(1, years + 1):
            npv += annual_savings / ((1 + r) ** y)
        return round(npv, 2)

    def _process(self, inp: ElectrificationInput) -> ElectrificationOutput:
        analyses = []
        recommendations = []
        total_co2_red = 0
        total_savings = 0
        total_capex = 0
        feasible_count = 0

        for load in inp.heat_loads:
            tech = self._select_technology(load.temperature_c)
            max_temp, cop_eff, capex_kw = TECH_SPECS[tech]

            # Electrical demand
            if cop_eff > 1:  # Heat pump
                elec_demand = load.thermal_demand_kw / cop_eff
            else:
                elec_demand = load.thermal_demand_kw / cop_eff

            # Annual consumption
            annual_kwh = elec_demand * load.operating_hours_year * 0.7  # 70% load factor

            # Current fuel cost (natural gas equivalent)
            fuel_kwh = (load.thermal_demand_kw * load.operating_hours_year * 0.7) / (load.current_efficiency_pct / 100)
            fuel_mmbtu = fuel_kwh / 293.07
            current_cost = fuel_mmbtu * inp.natural_gas_price_mmbtu

            # Electrified cost
            elec_cost = annual_kwh * inp.electricity_price_kwh

            # Carbon costs
            current_emissions = fuel_kwh * FUEL_EMISSIONS.get(load.current_fuel, 0.2) / 1000  # tCO2
            elec_emissions = annual_kwh * inp.grid_carbon_intensity_gco2_kwh / 1_000_000  # tCO2

            current_cost += current_emissions * inp.carbon_price_tco2
            elec_cost += elec_emissions * inp.carbon_price_tco2

            savings = current_cost - elec_cost
            capex = elec_demand * capex_kw
            co2_red = current_emissions - elec_emissions
            co2_red_pct = (co2_red / current_emissions * 100) if current_emissions > 0 else 0

            payback = capex / savings if savings > 0 else None
            npv = self._calculate_npv(capex, savings, inp.analysis_years, inp.discount_rate_pct)

            # Feasibility
            if payback and payback <= 5:
                feasibility = Feasibility.EXCELLENT
            elif payback and payback <= 8:
                feasibility = Feasibility.GOOD
            elif payback and payback <= 12:
                feasibility = Feasibility.MODERATE
            else:
                feasibility = Feasibility.CHALLENGING

            if feasibility in [Feasibility.EXCELLENT, Feasibility.GOOD, Feasibility.MODERATE]:
                feasible_count += 1

            analyses.append(LoadAnalysis(
                load_id=load.load_id,
                load_name=load.name,
                recommended_tech=tech,
                feasibility=feasibility,
                electrical_demand_kw=round(elec_demand, 2),
                annual_electricity_kwh=round(annual_kwh, 2),
                current_annual_cost_usd=round(current_cost, 2),
                electrified_annual_cost_usd=round(elec_cost, 2),
                annual_savings_usd=round(savings, 2),
                capital_cost_usd=round(capex, 2),
                payback_years=round(payback, 1) if payback else None,
                npv_usd=npv,
                co2_reduction_tpa=round(co2_red, 2),
                co2_reduction_pct=round(co2_red_pct, 1)
            ))

            total_co2_red += co2_red
            total_savings += savings
            total_capex += capex

        # Overall metrics
        total_loads = len(inp.heat_loads)
        elec_potential = (feasible_count / total_loads * 100) if total_loads > 0 else 0
        overall_payback = total_capex / total_savings if total_savings > 0 else None
        overall_npv = self._calculate_npv(total_capex, total_savings, inp.analysis_years, inp.discount_rate_pct)

        # Recommendations
        if elec_potential >= 70:
            recommendations.append("Strong electrification potential - proceed with phased implementation")
        elif elec_potential >= 40:
            recommendations.append("Moderate electrification potential - prioritize high-feasibility loads")
        else:
            recommendations.append("Limited electrification potential - consider hybrid approaches")

        excellent = [a for a in analyses if a.feasibility == Feasibility.EXCELLENT]
        if excellent:
            recommendations.append(f"{len(excellent)} loads with excellent payback - implement first")

        if inp.grid_carbon_intensity_gco2_kwh > 400:
            recommendations.append("Grid carbon intensity is high - consider onsite renewables")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_name,
            "loads": total_loads,
            "co2_red": round(total_co2_red, 2)
        }).encode()).hexdigest()

        return ElectrificationOutput(
            facility_name=inp.facility_name,
            total_loads=total_loads,
            feasible_loads=feasible_count,
            electrification_potential_pct=round(elec_potential, 1),
            total_co2_reduction_tpa=round(total_co2_red, 2),
            total_savings_usd=round(total_savings, 2),
            total_capital_usd=round(total_capex, 2),
            overall_payback_years=round(overall_payback, 1) if overall_payback else None,
            overall_npv_usd=overall_npv,
            load_analyses=analyses,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Decarbonization",
            "type": "Analysis",
            "standards": ["IEC", "IEEE", "NFPA 70"]
        }


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-036",
    "name": "ELECTRIFICATION",
    "version": "1.0.0",
    "summary": "Industrial electrification analysis for decarbonization",
    "tags": ["electrification", "decarbonization", "heat-pump", "electric-boiler"],
    "standards": [
        {"ref": "IEC 60034", "description": "Rotating Electrical Machines"},
        {"ref": "IEEE 519", "description": "Harmonic Control"},
        {"ref": "NFPA 70", "description": "National Electrical Code"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
