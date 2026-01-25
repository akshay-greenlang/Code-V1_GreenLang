"""GL-025 COGENMAX: Cogeneration Optimizer Agent.

Optimizes combined heat and power (CHP) systems for maximum efficiency
and cost savings through intelligent dispatch.

Standards: IEEE 1547, ASHRAE 90.1, ISO 50001
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============== Models ==============

class CHPSystemInput(BaseModel):
    """Input parameters for CHP optimization."""

    # System configuration
    chp_type: str = Field("gas_turbine", description="gas_turbine, gas_engine, steam_turbine")
    rated_power_mw: float = Field(..., ge=0.1, description="Rated electrical output (MW)")
    rated_heat_mw: float = Field(..., ge=0.1, description="Rated thermal output (MW)")

    # Current operation
    current_power_mw: float = Field(..., ge=0, description="Current power output (MW)")
    current_heat_mw: float = Field(..., ge=0, description="Current heat output (MW)")
    fuel_input_mw: float = Field(..., ge=0, description="Fuel input (MW)")

    # Demands
    site_power_demand_mw: float = Field(..., ge=0, description="Site electrical demand (MW)")
    site_heat_demand_mw: float = Field(..., ge=0, description="Site thermal demand (MW)")

    # Performance curves (normalized 0-1 load)
    min_load_pct: float = Field(30, ge=0, le=100, description="Minimum stable load (%)")
    electrical_efficiency_at_rated: float = Field(0.35, ge=0.2, le=0.6)
    thermal_efficiency_at_rated: float = Field(0.45, ge=0.3, le=0.6)

    # Economic inputs
    electricity_buy_price: float = Field(..., description="Grid import price ($/MWh)")
    electricity_sell_price: float = Field(0.0, description="Grid export price ($/MWh)")
    fuel_price_per_mwh: float = Field(..., description="Fuel price ($/MWh)")
    carbon_price_per_ton: float = Field(0.0, description="Carbon price ($/ton)")

    # Grid constraints
    can_export_power: bool = Field(False, description="Can export to grid")
    max_import_mw: float = Field(100, description="Max grid import (MW)")
    max_export_mw: float = Field(0, description="Max grid export (MW)")

    # Auxiliary systems
    auxiliary_boiler_efficiency: float = Field(0.85, description="Aux boiler efficiency")
    auxiliary_boiler_capacity_mw: float = Field(10, description="Aux boiler capacity (MW)")
    chiller_cop: float = Field(5.0, description="Absorption chiller COP")

    # Emissions
    fuel_emissions_factor: float = Field(0.2, description="kg CO2/kWh fuel")
    grid_emissions_factor: float = Field(0.4, description="kg CO2/kWh grid")

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CHPOutput(BaseModel):
    """Output from CHP optimizer."""

    # Optimal operating point
    optimal_power_mw: float = Field(..., description="Optimal power output (MW)")
    optimal_heat_mw: float = Field(..., description="Optimal heat output (MW)")
    optimal_fuel_mw: float = Field(..., description="Optimal fuel input (MW)")
    operating_mode: str = Field(..., description="ELECTRICAL_FOLLOWING, THERMAL_FOLLOWING, OPTIMAL")

    # Energy balance
    grid_import_mw: float = Field(..., description="Grid import required (MW)")
    grid_export_mw: float = Field(..., description="Grid export available (MW)")
    auxiliary_heat_mw: float = Field(..., description="Aux boiler heat required (MW)")
    heat_dump_mw: float = Field(..., description="Excess heat to dump (MW)")

    # Efficiency metrics
    chp_electrical_efficiency: float = Field(..., description="CHP electrical efficiency")
    chp_thermal_efficiency: float = Field(..., description="CHP thermal efficiency")
    overall_chp_efficiency: float = Field(..., description="Overall CHP efficiency")
    primary_energy_savings_pct: float = Field(..., description="PES vs separate production (%)")

    # Economic metrics
    hourly_fuel_cost: float = Field(..., description="Fuel cost ($/hr)")
    hourly_power_cost_savings: float = Field(..., description="Power cost savings ($/hr)")
    hourly_export_revenue: float = Field(0.0, description="Export revenue ($/hr)")
    total_hourly_benefit: float = Field(..., description="Net hourly benefit ($/hr)")

    # Environmental metrics
    chp_emissions_kg_hr: float = Field(..., description="CHP CO2 emissions (kg/hr)")
    avoided_grid_emissions_kg_hr: float = Field(..., description="Avoided grid emissions (kg/hr)")
    net_emissions_impact_kg_hr: float = Field(..., description="Net emissions change (kg/hr)")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Provenance
    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


# ============== Formulas ==============

def calculate_chp_efficiency(
    power_output_mw: float,
    heat_output_mw: float,
    fuel_input_mw: float
) -> tuple:
    """
    Calculate CHP efficiencies.

    η_e = P_elec / Q_fuel (electrical)
    η_th = Q_heat / Q_fuel (thermal)
    η_overall = (P_elec + Q_heat) / Q_fuel
    """
    if fuel_input_mw <= 0:
        return 0.0, 0.0, 0.0

    eta_e = power_output_mw / fuel_input_mw
    eta_th = heat_output_mw / fuel_input_mw
    eta_overall = (power_output_mw + heat_output_mw) / fuel_input_mw

    return round(eta_e, 4), round(eta_th, 4), round(eta_overall, 4)


def calculate_primary_energy_savings(
    chp_fuel_mw: float,
    power_generated_mw: float,
    heat_generated_mw: float,
    ref_power_efficiency: float = 0.40,
    ref_heat_efficiency: float = 0.90
) -> float:
    """
    Calculate Primary Energy Savings (PES) per EU CHP Directive.

    PES = 1 - 1 / (η_e/η_ref_e + η_th/η_ref_th)

    Where η_ref are reference efficiencies for separate production.
    CHP qualifies as "high efficiency" if PES > 10%.
    """
    if chp_fuel_mw <= 0:
        return 0.0

    eta_e = power_generated_mw / chp_fuel_mw
    eta_th = heat_generated_mw / chp_fuel_mw

    if eta_e <= 0 and eta_th <= 0:
        return 0.0

    denominator = 0
    if eta_e > 0 and ref_power_efficiency > 0:
        denominator += eta_e / ref_power_efficiency
    if eta_th > 0 and ref_heat_efficiency > 0:
        denominator += eta_th / ref_heat_efficiency

    if denominator <= 0:
        return 0.0

    pes = (1 - 1 / denominator) * 100
    return round(pes, 2)


def calculate_heat_to_power_ratio(
    chp_type: str,
    load_fraction: float
) -> float:
    """
    Calculate heat-to-power ratio based on CHP type and load.

    Gas turbine: H/P ≈ 1.5-2.0
    Gas engine: H/P ≈ 1.0-1.5
    Steam turbine: H/P ≈ 3.0-8.0
    """
    ratios = {
        "gas_turbine": 1.8,
        "gas_engine": 1.2,
        "steam_turbine": 5.0
    }
    base_ratio = ratios.get(chp_type, 1.5)

    # Ratio increases at part load
    load_factor = 1.0 + 0.2 * (1.0 - load_fraction)
    return base_ratio * load_factor


def optimize_dispatch(
    power_demand: float,
    heat_demand: float,
    rated_power: float,
    rated_heat: float,
    min_load_pct: float,
    power_efficiency: float,
    thermal_efficiency: float,
    fuel_price: float,
    electricity_buy: float,
    electricity_sell: float,
    can_export: bool
) -> Dict[str, float]:
    """
    Optimize CHP dispatch based on demands and prices.

    Strategies:
    1. Electrical-following: Match power demand, use aux heat
    2. Thermal-following: Match heat demand, import/export power
    3. Economic optimal: Maximize net benefit
    """
    min_power = rated_power * min_load_pct / 100
    h_p_ratio = rated_heat / rated_power if rated_power > 0 else 1.5

    results = {}

    # Strategy 1: Electrical following
    p_elec_follow = min(power_demand, rated_power)
    p_elec_follow = max(p_elec_follow, min_power) if p_elec_follow > 0 else 0
    h_elec_follow = p_elec_follow * h_p_ratio

    # Strategy 2: Thermal following
    h_thermal_follow = min(heat_demand, rated_heat)
    p_thermal_follow = h_thermal_follow / h_p_ratio if h_p_ratio > 0 else 0
    p_thermal_follow = min(p_thermal_follow, rated_power)
    p_thermal_follow = max(p_thermal_follow, min_power) if p_thermal_follow > 0 else 0
    h_thermal_follow = p_thermal_follow * h_p_ratio

    # Calculate costs for each strategy
    def calc_cost(p_gen, h_gen):
        fuel_mw = p_gen / power_efficiency if power_efficiency > 0 else 0
        fuel_cost = fuel_mw * fuel_price

        power_shortfall = max(0, power_demand - p_gen)
        power_surplus = max(0, p_gen - power_demand) if can_export else 0

        power_import_cost = power_shortfall * electricity_buy
        power_export_rev = power_surplus * electricity_sell

        # Aux heat for shortfall
        heat_shortfall = max(0, heat_demand - h_gen)
        aux_fuel = heat_shortfall / 0.85
        aux_cost = aux_fuel * fuel_price * 0.9  # Assume cheaper fuel for boiler

        total_cost = fuel_cost + power_import_cost + aux_cost - power_export_rev
        return total_cost, fuel_mw

    cost_elec, fuel_elec = calc_cost(p_elec_follow, h_elec_follow)
    cost_thermal, fuel_thermal = calc_cost(p_thermal_follow, h_thermal_follow)

    # Choose best strategy
    if cost_elec <= cost_thermal:
        return {
            "power": p_elec_follow,
            "heat": h_elec_follow,
            "fuel": fuel_elec,
            "mode": "ELECTRICAL_FOLLOWING"
        }
    else:
        return {
            "power": p_thermal_follow,
            "heat": h_thermal_follow,
            "fuel": fuel_thermal,
            "mode": "THERMAL_FOLLOWING"
        }


# ============== Agent ==============

class CogenerationOptimizerAgent:
    """Cogeneration (CHP) optimization agent."""

    AGENT_ID = "GL-025"
    AGENT_NAME = "COGENMAX"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = CHPSystemInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: CHPSystemInput) -> CHPOutput:
        recommendations = []
        warnings = []

        # Optimize dispatch
        dispatch = optimize_dispatch(
            power_demand=inp.site_power_demand_mw,
            heat_demand=inp.site_heat_demand_mw,
            rated_power=inp.rated_power_mw,
            rated_heat=inp.rated_heat_mw,
            min_load_pct=inp.min_load_pct,
            power_efficiency=inp.electrical_efficiency_at_rated,
            thermal_efficiency=inp.thermal_efficiency_at_rated,
            fuel_price=inp.fuel_price_per_mwh,
            electricity_buy=inp.electricity_buy_price,
            electricity_sell=inp.electricity_sell_price,
            can_export=inp.can_export_power
        )

        optimal_power = dispatch["power"]
        optimal_heat = dispatch["heat"]
        optimal_fuel = dispatch["fuel"]
        mode = dispatch["mode"]

        # Calculate energy balance
        grid_import = max(0, inp.site_power_demand_mw - optimal_power)
        grid_export = max(0, optimal_power - inp.site_power_demand_mw) if inp.can_export_power else 0
        aux_heat = max(0, inp.site_heat_demand_mw - optimal_heat)
        heat_dump = max(0, optimal_heat - inp.site_heat_demand_mw)

        # Calculate efficiencies
        eta_e, eta_th, eta_overall = calculate_chp_efficiency(
            optimal_power, optimal_heat, optimal_fuel
        )

        # Primary Energy Savings
        pes = calculate_primary_energy_savings(
            optimal_fuel, optimal_power, optimal_heat
        )

        if pes > 10:
            recommendations.append(f"CHP qualifies as high-efficiency (PES={pes:.1f}%)")

        # Economic calculations
        hourly_fuel_cost = optimal_fuel * inp.fuel_price_per_mwh
        power_cost_avoided = optimal_power * inp.electricity_buy_price
        export_revenue = grid_export * inp.electricity_sell_price
        aux_fuel_cost = aux_heat / inp.auxiliary_boiler_efficiency * inp.fuel_price_per_mwh

        # Compare to all-grid scenario
        grid_only_power_cost = inp.site_power_demand_mw * inp.electricity_buy_price
        grid_only_heat_cost = inp.site_heat_demand_mw / 0.85 * inp.fuel_price_per_mwh

        total_benefit = (grid_only_power_cost + grid_only_heat_cost) - \
                       (hourly_fuel_cost + grid_import * inp.electricity_buy_price + aux_fuel_cost)

        # Emissions
        chp_emissions = optimal_fuel * inp.fuel_emissions_factor * 1000  # kg/hr
        avoided_grid = optimal_power * inp.grid_emissions_factor * 1000  # kg/hr
        net_emissions = chp_emissions - avoided_grid

        # Warnings and recommendations
        if grid_import > inp.max_import_mw:
            warnings.append(f"Grid import {grid_import:.2f} MW exceeds limit")

        if heat_dump > 0:
            warnings.append(f"Dumping {heat_dump:.2f} MW of heat")
            recommendations.append("Consider absorption chiller for cooling load")

        load_pct = optimal_power / inp.rated_power_mw * 100 if inp.rated_power_mw > 0 else 0
        if load_pct < 50:
            recommendations.append(f"Low CHP utilization ({load_pct:.0f}%) - consider heat storage")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "power": optimal_power,
            "heat": optimal_heat,
            "mode": mode
        }, default=str).encode()).hexdigest()

        return CHPOutput(
            optimal_power_mw=round(optimal_power, 3),
            optimal_heat_mw=round(optimal_heat, 3),
            optimal_fuel_mw=round(optimal_fuel, 3),
            operating_mode=mode,
            grid_import_mw=round(grid_import, 3),
            grid_export_mw=round(grid_export, 3),
            auxiliary_heat_mw=round(aux_heat, 3),
            heat_dump_mw=round(heat_dump, 3),
            chp_electrical_efficiency=eta_e,
            chp_thermal_efficiency=eta_th,
            overall_chp_efficiency=eta_overall,
            primary_energy_savings_pct=pes,
            hourly_fuel_cost=round(hourly_fuel_cost, 2),
            hourly_power_cost_savings=round(power_cost_avoided, 2),
            hourly_export_revenue=round(export_revenue, 2),
            total_hourly_benefit=round(total_benefit, 2),
            chp_emissions_kg_hr=round(chp_emissions, 2),
            avoided_grid_emissions_kg_hr=round(avoided_grid, 2),
            net_emissions_impact_kg_hr=round(net_emissions, 2),
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
            "category": "Cogeneration",
            "type": "Optimizer",
            "complexity": "High",
            "priority": "P0",
            "market_size": "$15B"
        }
