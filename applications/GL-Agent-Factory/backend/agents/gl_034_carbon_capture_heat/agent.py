"""GL-034 CARBONCAPTURE-HEAT: Carbon Capture Heat Integration Agent.

Optimizes heat integration with carbon capture systems to minimize
energy penalty and maximize capture efficiency.

Standards: ISO 27913, IEA CCS Guidelines
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CCSHeatInput(BaseModel):
    """Input for CCS heat integration."""

    facility_id: str = Field(..., description="Facility identifier")
    ccs_technology: str = Field("amine", description="amine, membrane, cryogenic, oxyfuel")

    # Capture parameters
    flue_gas_flow_kg_s: float = Field(..., ge=0, description="Flue gas flow (kg/s)")
    co2_concentration_pct: float = Field(..., ge=0, le=100, description="CO2 in flue gas (%)")
    target_capture_rate_pct: float = Field(90, description="Target capture rate (%)")
    current_capture_rate_pct: float = Field(..., description="Current capture rate (%)")

    # Heat requirements (amine absorption)
    reboiler_duty_mw: float = Field(..., ge=0, description="Reboiler duty (MW)")
    reboiler_temp_c: float = Field(120, description="Reboiler temperature (°C)")
    lean_solvent_temp_c: float = Field(40, description="Lean solvent temp (°C)")
    rich_solvent_temp_c: float = Field(50, description="Rich solvent temp (°C)")

    # Available waste heat sources
    waste_heat_sources: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Available sources: {id, temp_c, available_mw}"
    )

    # Process steam
    steam_available_mw: float = Field(..., ge=0, description="LP steam available (MW)")
    steam_temp_c: float = Field(150, description="Steam temperature (°C)")
    steam_cost_per_mwh: float = Field(30, description="Steam cost ($/MWh)")

    # Compression
    co2_compression_power_mw: float = Field(..., ge=0, description="CO2 compressor power (MW)")
    target_co2_pressure_bar: float = Field(150, description="CO2 delivery pressure (bar)")

    # Economics
    carbon_price_per_ton: float = Field(50, description="Carbon price/credit ($/ton)")
    electricity_cost_per_mwh: float = Field(80, description="Electricity cost ($/MWh)")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CCSHeatOutput(BaseModel):
    """Output from CCS heat integration agent."""

    # Capture performance
    co2_captured_tonnes_hr: float = Field(..., description="CO2 captured (t/hr)")
    specific_heat_mj_kg_co2: float = Field(..., description="Specific heat (MJ/kg CO2)")
    capture_efficiency_pct: float = Field(..., description="Capture efficiency")

    # Heat integration
    waste_heat_utilized_mw: float = Field(..., description="Waste heat used (MW)")
    waste_heat_sources_used: List[str] = Field(default_factory=list)
    steam_consumed_mw: float = Field(..., description="Steam consumed (MW)")
    total_heat_supplied_mw: float = Field(..., description="Total heat to CCS (MW)")

    # Energy penalty
    heat_energy_penalty_pct: float = Field(..., description="Heat penalty on plant (%)")
    electrical_penalty_pct: float = Field(..., description="Electrical penalty (%)")
    total_energy_penalty_pct: float = Field(..., description="Total energy penalty (%)")

    # Integration recommendations
    additional_integration_potential_mw: float = Field(..., description="More integration possible")
    optimized_steam_consumption_mw: float = Field(..., description="Optimized steam use")
    steam_savings_pct: float = Field(..., description="Potential steam savings (%)")

    # Economics
    hourly_operating_cost: float = Field(..., description="CCS operating cost ($/hr)")
    cost_per_tonne_co2: float = Field(..., description="Cost per tonne captured ($)")
    carbon_revenue_per_hour: float = Field(..., description="Carbon credit revenue ($/hr)")
    net_cost_per_hour: float = Field(..., description="Net cost ($/hr)")

    # Optimization status
    heat_integration_status: str = Field(..., description="OPTIMAL, SUBOPTIMAL, POOR")
    efficiency_improvement_potential_pct: float = Field(..., description="Improvement potential")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_co2_captured(
    flue_gas_kg_s: float,
    co2_pct: float,
    capture_rate_pct: float
) -> float:
    """Calculate CO2 captured in tonnes per hour."""
    co2_in_flue = flue_gas_kg_s * (co2_pct / 100)
    captured = co2_in_flue * (capture_rate_pct / 100)
    return round(captured * 3.6, 2)  # Convert kg/s to t/hr


def calculate_specific_heat(
    reboiler_duty_mw: float,
    co2_captured_t_hr: float
) -> float:
    """Calculate specific regeneration heat (MJ/kg CO2)."""
    if co2_captured_t_hr <= 0:
        return 0.0
    # Convert MW to MJ/hr, divide by tonnes to kg
    specific = (reboiler_duty_mw * 3600) / (co2_captured_t_hr * 1000)
    return round(specific, 2)


def match_waste_heat(
    sources: List[Dict[str, float]],
    required_temp_c: float,
    required_duty_mw: float
) -> tuple:
    """Match waste heat sources to CCS requirements."""
    usable_sources = []
    total_usable = 0.0

    for source in sources:
        if source.get("temp_c", 0) >= required_temp_c + 10:  # 10°C approach
            usable = min(source.get("available_mw", 0), required_duty_mw - total_usable)
            if usable > 0:
                usable_sources.append(source.get("id", "unknown"))
                total_usable += usable

        if total_usable >= required_duty_mw:
            break

    return usable_sources, round(total_usable, 2)


class CarbonCaptureHeatAgent:
    """Carbon capture heat integration agent."""

    AGENT_ID = "GL-034"
    AGENT_NAME = "CARBONCAPTURE-HEAT"
    VERSION = "1.0.0"

    # Benchmarks for amine CCS
    BEST_SPECIFIC_HEAT = 2.5  # MJ/kg CO2 (advanced amine)
    TYPICAL_SPECIFIC_HEAT = 3.5  # MJ/kg CO2 (standard MEA)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = CCSHeatInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: CCSHeatInput) -> CCSHeatOutput:
        recommendations = []
        warnings = []

        # Calculate CO2 captured
        co2_captured = calculate_co2_captured(
            inp.flue_gas_flow_kg_s,
            inp.co2_concentration_pct,
            inp.current_capture_rate_pct
        )

        # Specific heat consumption
        specific_heat = calculate_specific_heat(inp.reboiler_duty_mw, co2_captured)

        # Match waste heat sources
        waste_sources_used, waste_heat_utilized = match_waste_heat(
            inp.waste_heat_sources,
            inp.reboiler_temp_c,
            inp.reboiler_duty_mw
        )

        # Steam consumption
        steam_consumed = max(0, inp.reboiler_duty_mw - waste_heat_utilized)
        steam_consumed = min(steam_consumed, inp.steam_available_mw)

        total_heat = waste_heat_utilized + steam_consumed

        # Check if heat requirements are met
        heat_shortfall = max(0, inp.reboiler_duty_mw - total_heat)
        if heat_shortfall > 0:
            warnings.append(f"Heat shortfall of {heat_shortfall:.1f} MW - capture rate may decrease")

        # Energy penalties (relative to typical 500 MW power plant)
        plant_capacity_mw = 500  # Assumed
        heat_penalty = (steam_consumed / plant_capacity_mw) * 100
        electrical_penalty = (inp.co2_compression_power_mw / plant_capacity_mw) * 100
        total_penalty = heat_penalty + electrical_penalty

        # Additional integration potential
        unused_waste_heat = sum(
            s.get("available_mw", 0) for s in inp.waste_heat_sources
            if s.get("temp_c", 0) >= inp.reboiler_temp_c + 10
        ) - waste_heat_utilized
        additional_potential = max(0, min(unused_waste_heat, steam_consumed))

        # Optimized scenario
        optimized_steam = steam_consumed - additional_potential
        steam_savings = (additional_potential / steam_consumed * 100) if steam_consumed > 0 else 0

        # Economics
        hourly_steam_cost = steam_consumed * inp.steam_cost_per_mwh
        hourly_power_cost = inp.co2_compression_power_mw * inp.electricity_cost_per_mwh
        hourly_operating_cost = hourly_steam_cost + hourly_power_cost

        cost_per_tonne = hourly_operating_cost / co2_captured if co2_captured > 0 else 0
        carbon_revenue = co2_captured * inp.carbon_price_per_ton
        net_cost = hourly_operating_cost - carbon_revenue

        # Integration status
        if specific_heat <= self.BEST_SPECIFIC_HEAT:
            integration_status = "OPTIMAL"
            improvement_potential = 0
        elif specific_heat <= self.TYPICAL_SPECIFIC_HEAT:
            integration_status = "SUBOPTIMAL"
            improvement_potential = ((specific_heat - self.BEST_SPECIFIC_HEAT) / specific_heat) * 100
        else:
            integration_status = "POOR"
            improvement_potential = ((specific_heat - self.TYPICAL_SPECIFIC_HEAT) / specific_heat) * 100

        # Recommendations
        if additional_potential > 1:
            recommendations.append(
                f"Integrate {additional_potential:.1f} MW additional waste heat to reduce steam by {steam_savings:.0f}%"
            )

        if specific_heat > self.TYPICAL_SPECIFIC_HEAT:
            recommendations.append(
                f"Specific heat {specific_heat:.2f} MJ/kg above benchmark - check lean solvent loading and stripper design"
            )

        if waste_heat_utilized < inp.reboiler_duty_mw * 0.3:
            recommendations.append("Less than 30% of reboiler duty from waste heat - explore heat pump integration")

        if net_cost > 0:
            recommendations.append(f"CCS operating at net cost ${net_cost:.0f}/hr - carbon price of ${cost_per_tonne:.0f}/t needed for breakeven")
        else:
            recommendations.append(f"CCS generating net revenue of ${-net_cost:.0f}/hr at current carbon price")

        if inp.current_capture_rate_pct < inp.target_capture_rate_pct:
            warnings.append(f"Capture rate {inp.current_capture_rate_pct:.0f}% below target {inp.target_capture_rate_pct:.0f}%")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "captured": co2_captured,
            "specific_heat": specific_heat,
            "penalty": total_penalty
        }).encode()).hexdigest()

        return CCSHeatOutput(
            co2_captured_tonnes_hr=co2_captured,
            specific_heat_mj_kg_co2=specific_heat,
            capture_efficiency_pct=round(inp.current_capture_rate_pct, 1),
            waste_heat_utilized_mw=waste_heat_utilized,
            waste_heat_sources_used=waste_sources_used,
            steam_consumed_mw=round(steam_consumed, 2),
            total_heat_supplied_mw=round(total_heat, 2),
            heat_energy_penalty_pct=round(heat_penalty, 2),
            electrical_penalty_pct=round(electrical_penalty, 2),
            total_energy_penalty_pct=round(total_penalty, 2),
            additional_integration_potential_mw=round(additional_potential, 2),
            optimized_steam_consumption_mw=round(optimized_steam, 2),
            steam_savings_pct=round(steam_savings, 1),
            hourly_operating_cost=round(hourly_operating_cost, 2),
            cost_per_tonne_co2=round(cost_per_tonne, 2),
            carbon_revenue_per_hour=round(carbon_revenue, 2),
            net_cost_per_hour=round(net_cost, 2),
            heat_integration_status=integration_status,
            efficiency_improvement_potential_pct=round(improvement_potential, 1),
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
            "category": "Decarbonization",
            "type": "Optimizer",
            "priority": "P0",
            "market_size": "$18B"
        }
