"""GL-039 HEATPUMP-PRO: Industrial Heat Pump Optimizer Agent.

Optimizes industrial heat pumps for low-grade heat recovery and
process heating electrification.

Standards: EN 14511, ASHRAE 90.1, IEA Heat Pump Centre
"""
import hashlib
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HeatPumpInput(BaseModel):
    """Input for heat pump optimization."""

    equipment_id: str = Field(..., description="Heat pump ID")
    hp_type: str = Field("compression", description="compression, absorption, hybrid")
    refrigerant: str = Field("R717", description="Refrigerant type")

    # Source (low temperature)
    source_temp_c: float = Field(..., description="Heat source temperature (°C)")
    source_flow_kg_s: float = Field(..., ge=0, description="Source flow rate (kg/s)")
    source_type: str = Field("waste_heat", description="waste_heat, ambient, water")

    # Sink (high temperature)
    sink_temp_required_c: float = Field(..., description="Required delivery temp (°C)")
    heat_demand_kw: float = Field(..., ge=0, description="Heat demand (kW)")
    sink_flow_kg_s: float = Field(..., ge=0, description="Sink flow rate (kg/s)")

    # Current operation
    current_cop: float = Field(..., ge=1, description="Current COP")
    compressor_power_kw: float = Field(..., ge=0, description="Compressor power (kW)")
    heat_output_kw: float = Field(..., ge=0, description="Current heat output (kW)")

    # Design parameters
    rated_capacity_kw: float = Field(..., ge=0, description="Rated heat capacity (kW)")
    rated_cop: float = Field(..., ge=1, description="Rated COP")
    max_lift_c: float = Field(60, description="Maximum temperature lift (°C)")

    # Economics
    electricity_price_per_kwh: float = Field(0.10, description="Electricity price ($/kWh)")
    alternative_heat_price_per_kwh: float = Field(0.05, description="Gas/steam price ($/kWh)")
    carbon_intensity_grid_kg_kwh: float = Field(0.4, description="Grid carbon (kg/kWh)")
    carbon_intensity_gas_kg_kwh: float = Field(0.2, description="Gas carbon (kg/kWh)")

    # Operating constraints
    min_source_temp_c: float = Field(10, description="Minimum source temp (°C)")
    max_sink_temp_c: float = Field(90, description="Maximum achievable sink temp (°C)")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HeatPumpOutput(BaseModel):
    """Output from heat pump optimizer."""

    # Performance metrics
    actual_cop: float = Field(..., description="Actual COP")
    carnot_cop: float = Field(..., description="Theoretical Carnot COP")
    second_law_efficiency_pct: float = Field(..., description="2nd law efficiency (%)")
    temperature_lift_c: float = Field(..., description="Temperature lift (°C)")

    # Operating point
    optimal_heat_output_kw: float = Field(..., description="Optimal heat output")
    optimal_compressor_power_kw: float = Field(..., description="Optimal power input")
    source_heat_extracted_kw: float = Field(..., description="Source heat extracted")
    capacity_utilization_pct: float = Field(..., description="Capacity utilization (%)")

    # Economics
    hourly_electricity_cost: float = Field(..., description="Electricity cost ($/hr)")
    alternative_heat_cost: float = Field(..., description="Gas/steam equivalent ($/hr)")
    hourly_savings: float = Field(..., description="Savings vs alternative ($/hr)")
    annual_savings_estimate: float = Field(..., description="Annual savings ($)")

    # Environmental
    hp_emissions_kg_hr: float = Field(..., description="HP CO2 emissions (kg/hr)")
    alternative_emissions_kg_hr: float = Field(..., description="Alternative CO2 (kg/hr)")
    emissions_reduction_kg_hr: float = Field(..., description="CO2 reduction (kg/hr)")
    emissions_reduction_pct: float = Field(..., description="CO2 reduction (%)")

    # Operating recommendations
    optimal_source_temp_c: Optional[float] = Field(None, description="Optimal source temp if adjustable")
    optimal_sink_temp_c: Optional[float] = Field(None, description="Optimal sink temp")
    efficiency_improvement_potential_pct: float = Field(..., description="COP improvement potential")

    # Diagnostics
    performance_status: str = Field(..., description="OPTIMAL, DEGRADED, POOR")
    refrigerant_charge_status: str = Field(..., description="OK, LOW, HIGH")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_carnot_cop(t_hot_c: float, t_cold_c: float) -> float:
    """
    Calculate Carnot COP (theoretical maximum).

    COP_Carnot = T_hot / (T_hot - T_cold) [Kelvin]
    """
    t_hot_k = t_hot_c + 273.15
    t_cold_k = t_cold_c + 273.15

    if t_hot_k <= t_cold_k:
        return float('inf')  # No lift needed

    return round(t_hot_k / (t_hot_k - t_cold_k), 2)


def calculate_actual_cop(heat_output_kw: float, power_input_kw: float) -> float:
    """Calculate actual COP from measured values."""
    if power_input_kw <= 0:
        return 0.0
    return round(heat_output_kw / power_input_kw, 2)


def estimate_cop_at_conditions(
    source_temp_c: float,
    sink_temp_c: float,
    rated_cop: float,
    rated_lift_c: float = 40
) -> float:
    """
    Estimate COP at operating conditions.

    COP decreases roughly 2-3% per °C additional lift.
    """
    lift = sink_temp_c - source_temp_c
    if lift <= 0:
        return rated_cop * 1.2  # No lift - very efficient

    # Degradation from rated conditions
    degradation_per_c = 0.025  # 2.5% per °C
    extra_lift = lift - rated_lift_c
    factor = 1 - degradation_per_c * max(0, extra_lift)

    return round(rated_cop * max(0.3, factor), 2)


class HeatPumpOptimizerAgent:
    """Industrial heat pump optimization agent."""

    AGENT_ID = "GL-039"
    AGENT_NAME = "HEATPUMP-PRO"
    VERSION = "1.0.0"

    HOURS_PER_YEAR = 8000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = HeatPumpInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: HeatPumpInput) -> HeatPumpOutput:
        recommendations = []
        warnings = []

        # Temperature lift
        lift = inp.sink_temp_required_c - inp.source_temp_c

        if lift > inp.max_lift_c:
            warnings.append(f"Temperature lift {lift:.0f}°C exceeds max {inp.max_lift_c:.0f}°C")

        # Carnot COP
        carnot = calculate_carnot_cop(inp.sink_temp_required_c, inp.source_temp_c)

        # Actual COP
        actual_cop = calculate_actual_cop(inp.heat_output_kw, inp.compressor_power_kw)

        # Second law efficiency
        second_law_eff = (actual_cop / carnot * 100) if carnot > 0 and carnot < 100 else 0

        # Expected COP at conditions
        expected_cop = estimate_cop_at_conditions(
            inp.source_temp_c,
            inp.sink_temp_required_c,
            inp.rated_cop
        )

        # Performance status
        if actual_cop >= expected_cop * 0.95:
            performance_status = "OPTIMAL"
            refrigerant_status = "OK"
        elif actual_cop >= expected_cop * 0.85:
            performance_status = "DEGRADED"
            refrigerant_status = "OK"
            recommendations.append("Performance below expected - check evaporator/condenser fouling")
        else:
            performance_status = "POOR"
            refrigerant_status = "LOW"  # Likely cause
            warnings.append(f"COP {actual_cop:.1f} significantly below expected {expected_cop:.1f}")
            recommendations.append("Check refrigerant charge and compressor health")

        # Optimal operating point
        optimal_heat = min(inp.heat_demand_kw, inp.rated_capacity_kw)
        optimal_power = optimal_heat / expected_cop if expected_cop > 0 else 0
        source_extracted = optimal_heat - optimal_power

        capacity_util = (optimal_heat / inp.rated_capacity_kw * 100) if inp.rated_capacity_kw > 0 else 0

        # Economics
        hourly_elec_cost = optimal_power / 1000 * inp.electricity_price_per_kwh
        alternative_cost = optimal_heat / 1000 * inp.alternative_heat_price_per_kwh
        hourly_savings = alternative_cost - hourly_elec_cost
        annual_savings = hourly_savings * self.HOURS_PER_YEAR

        # Emissions
        hp_emissions = (optimal_power / 1000) * inp.carbon_intensity_grid_kg_kwh
        alt_emissions = (optimal_heat / 1000) * inp.carbon_intensity_gas_kg_kwh
        emissions_reduction = alt_emissions - hp_emissions
        reduction_pct = (emissions_reduction / alt_emissions * 100) if alt_emissions > 0 else 0

        # Improvement potential
        improvement = ((expected_cop - actual_cop) / actual_cop * 100) if actual_cop > 0 else 0

        # Recommendations
        if actual_cop < expected_cop:
            recommendations.append(f"COP improvement potential: {improvement:.0f}%")

        if lift > 50:
            recommendations.append("High temperature lift - consider cascade heat pump for better efficiency")

        if inp.source_temp_c < inp.min_source_temp_c + 5:
            warnings.append(f"Source temp approaching minimum - risk of poor performance")

        if second_law_eff < 40:
            recommendations.append(f"Second law efficiency {second_law_eff:.0f}% is low - review system design")

        if hourly_savings > 0:
            recommendations.append(f"Heat pump saving ${hourly_savings:.2f}/hr vs conventional heating")
        else:
            warnings.append("Heat pump more expensive than alternative - review economics")

        if emissions_reduction > 0:
            recommendations.append(f"Reducing emissions by {emissions_reduction:.1f} kg CO2/hr ({reduction_pct:.0f}%)")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "cop": actual_cop,
            "carnot": carnot,
            "savings": hourly_savings
        }).encode()).hexdigest()

        return HeatPumpOutput(
            actual_cop=actual_cop,
            carnot_cop=carnot,
            second_law_efficiency_pct=round(second_law_eff, 1),
            temperature_lift_c=round(lift, 1),
            optimal_heat_output_kw=round(optimal_heat, 1),
            optimal_compressor_power_kw=round(optimal_power, 1),
            source_heat_extracted_kw=round(source_extracted, 1),
            capacity_utilization_pct=round(capacity_util, 1),
            hourly_electricity_cost=round(hourly_elec_cost, 2),
            alternative_heat_cost=round(alternative_cost, 2),
            hourly_savings=round(hourly_savings, 2),
            annual_savings_estimate=round(annual_savings, 0),
            hp_emissions_kg_hr=round(hp_emissions, 2),
            alternative_emissions_kg_hr=round(alt_emissions, 2),
            emissions_reduction_kg_hr=round(emissions_reduction, 2),
            emissions_reduction_pct=round(reduction_pct, 1),
            optimal_source_temp_c=None,
            optimal_sink_temp_c=None,
            efficiency_improvement_potential_pct=round(max(0, improvement), 1),
            performance_status=performance_status,
            refrigerant_charge_status=refrigerant_status,
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
            "category": "Heat Pumps",
            "type": "Optimizer",
            "complexity": "High",
            "priority": "P1",
            "market_size": "$11B"
        }
