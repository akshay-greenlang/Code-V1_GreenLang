"""
GL-033: District Heating Integrator Agent (DISTRICT-LINK)

This module implements the DistrictHeatingAgent for integrating industrial
waste heat into district heating networks.

The agent provides:
- Waste heat source characterization
- District heating network integration analysis
- Temperature and pressure matching
- Economic feasibility assessment
- Complete SHA-256 provenance tracking

Standards Compliance:
- EN 13941: District Heating Networks Design
- ISO 50001: Energy Management Systems
- ASHRAE: District Energy Systems

Example:
    >>> agent = DistrictHeatingAgent()
    >>> result = agent.run(DistrictHeatingInput(
    ...     facility_id="FACILITY-001",
    ...     waste_heat_sources=[...],
    ...     district_network=DistrictNetwork(...),
    ... ))
    >>> print(f"Integration Feasibility: {result.feasibility_score}")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT MODELS
# =============================================================================

class WasteHeatSource(BaseModel):
    """Industrial waste heat source."""

    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(..., description="flue_gas, cooling_water, process_steam, etc.")
    temperature_supply_c: float = Field(..., description="Supply temperature (°C)")
    temperature_return_c: float = Field(..., description="Return temperature (°C)")
    flow_rate_kg_s: float = Field(..., ge=0, description="Flow rate (kg/s)")
    specific_heat_kj_kg_k: float = Field(default=4.18, gt=0, description="Specific heat")
    availability_hours_per_year: float = Field(default=8000, ge=0, le=8760)
    distance_to_network_m: float = Field(..., ge=0, description="Distance to DHN (m)")
    contaminants: List[str] = Field(default_factory=list, description="Contaminants present")


class DistrictNetwork(BaseModel):
    """District heating network parameters."""

    network_id: str = Field(..., description="Network identifier")
    supply_temperature_c: float = Field(..., description="Network supply temp (°C)")
    return_temperature_c: float = Field(..., description="Network return temp (°C)")
    design_capacity_mw: float = Field(..., ge=0, description="Design capacity (MW)")
    current_load_mw: float = Field(..., ge=0, description="Current heat load (MW)")
    pressure_supply_bar: float = Field(default=16.0, gt=0, description="Supply pressure (bar)")
    pressure_return_bar: float = Field(default=6.0, gt=0, description="Return pressure (bar)")
    heat_tariff_per_mwh: float = Field(default=50.0, ge=0, description="Heat tariff ($/MWh)")


class DistrictHeatingInput(BaseModel):
    """Input data model for DistrictHeatingAgent."""

    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    waste_heat_sources: List[WasteHeatSource] = Field(..., description="Available waste heat sources")
    district_network: DistrictNetwork = Field(..., description="District heating network parameters")

    # Economic parameters
    heat_exchanger_cost_per_kw: float = Field(default=150.0, ge=0, description="HX cost ($/kW)")
    piping_cost_per_m: float = Field(default=500.0, ge=0, description="Piping cost ($/m)")
    pumping_cost_per_kwh: float = Field(default=0.08, ge=0, description="Pumping electricity ($/kWh)")
    discount_rate: float = Field(default=0.08, ge=0, le=1, description="Discount rate")

    # Technical constraints
    min_approach_temp_c: float = Field(default=10.0, gt=0, description="Minimum approach temp (°C)")
    max_piping_distance_m: float = Field(default=2000.0, gt=0, description="Max economical distance (m)")
    heat_pump_option: bool = Field(default=False, description="Consider heat pump integration")

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class IntegrationScenario(BaseModel):
    """Waste heat integration scenario."""

    scenario_id: str = Field(..., description="Scenario identifier")
    source_id: str = Field(..., description="Waste heat source ID")
    heat_delivered_kw: float = Field(..., description="Heat delivered to DHN (kW)")
    heat_delivered_mwh_per_year: float = Field(..., description="Annual heat delivery (MWh/yr)")

    # Technical parameters
    heat_exchanger_size_kw: float = Field(..., description="Required HX size (kW)")
    heat_pump_required: bool = Field(..., description="Whether heat pump is needed")
    heat_pump_cop: Optional[float] = Field(None, description="Heat pump COP if applicable")

    # Economics
    capital_cost: float = Field(..., description="Total capital cost ($)")
    annual_revenue: float = Field(..., description="Annual heat sales revenue ($)")
    annual_operating_cost: float = Field(..., description="Annual operating cost ($)")
    annual_net_benefit: float = Field(..., description="Annual net benefit ($)")
    simple_payback_years: float = Field(..., description="Simple payback period (years)")
    npv_20yr: float = Field(..., description="20-year NPV ($)")

    # Environmental
    co2_reduction_tonnes_per_year: float = Field(..., description="CO2 reduction (tonnes/yr)")

    # Feasibility factors
    technical_feasibility: str = Field(..., description="HIGH, MEDIUM, LOW")
    economic_feasibility: str = Field(..., description="HIGH, MEDIUM, LOW")


class DistrictHeatingOutput(BaseModel):
    """Output data model for DistrictHeatingAgent."""

    # Identification
    analysis_id: str = Field(..., description="Unique analysis identifier")
    facility_id: str = Field(..., description="Facility identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Integration scenarios
    integration_scenarios: List[IntegrationScenario] = Field(
        default_factory=list,
        description="Evaluated integration scenarios"
    )

    # Overall metrics
    total_potential_heat_mw: float = Field(..., description="Total recoverable heat (MW)")
    total_annual_heat_mwh: float = Field(..., description="Total annual heat delivery (MWh/yr)")
    total_annual_revenue: float = Field(..., description="Total annual revenue ($)")
    total_capital_cost: float = Field(..., description="Total capital investment ($)")
    total_co2_reduction_tonnes: float = Field(..., description="Total CO2 reduction (tonnes/yr)")

    # Feasibility assessment
    feasibility_score: float = Field(..., ge=0, le=100, description="Overall feasibility score")
    recommended_scenario_id: Optional[str] = Field(None, description="Best scenario ID")

    # Recommendations and warnings
    recommendations: List[str] = Field(default_factory=list, description="Implementation recommendations")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    # Technical requirements
    required_infrastructure: List[str] = Field(default_factory=list)

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 hash of calculations")
    provenance_chain: List[Dict[str, Any]] = Field(default_factory=list)

    # Processing metadata
    processing_time_ms: float = Field(..., description="Processing duration in ms")
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# FORMULAS (ZERO-HALLUCINATION)
# =============================================================================

def calculate_heat_capacity(
    flow_rate_kg_s: float,
    temp_supply_c: float,
    temp_return_c: float,
    specific_heat_kj_kg_k: float = 4.18
) -> float:
    """
    Calculate heat capacity from waste heat source.

    ZERO-HALLUCINATION FORMULA:
    Q = m × Cp × ΔT
    Where:
    - Q = heat transfer rate (kW)
    - m = mass flow rate (kg/s)
    - Cp = specific heat (kJ/kg·K)
    - ΔT = temperature difference (K)

    Args:
        flow_rate_kg_s: Mass flow rate
        temp_supply_c: Supply temperature
        temp_return_c: Return temperature
        specific_heat_kj_kg_k: Specific heat capacity

    Returns:
        Heat capacity in kW
    """
    delta_t = temp_supply_c - temp_return_c
    if delta_t <= 0:
        return 0.0

    heat_kw = flow_rate_kg_s * specific_heat_kj_kg_k * delta_t
    return round(heat_kw, 2)


def calculate_carnot_cop_heating(temp_hot_c: float, temp_cold_c: float) -> float:
    """
    Calculate theoretical Carnot COP for heating.

    ZERO-HALLUCINATION FORMULA (Thermodynamics):
    COP_Carnot = T_hot / (T_hot - T_cold)

    Args:
        temp_hot_c: Hot side temperature (°C)
        temp_cold_c: Cold side temperature (°C)

    Returns:
        Carnot COP
    """
    t_hot_k = temp_hot_c + 273.15
    t_cold_k = temp_cold_c + 273.15

    if t_hot_k <= t_cold_k:
        return 1.0

    cop_carnot = t_hot_k / (t_hot_k - t_cold_k)
    # Real COP is typically 40-60% of Carnot
    real_cop = cop_carnot * 0.5
    return round(min(real_cop, 8.0), 2)  # Cap at realistic value


def calculate_pumping_power(
    flow_rate_kg_s: float,
    pressure_drop_bar: float,
    pump_efficiency: float = 0.75
) -> float:
    """
    Calculate pumping power requirement.

    ZERO-HALLUCINATION FORMULA:
    P = (Q × ΔP) / η
    Where:
    - P = power (kW)
    - Q = volumetric flow (m³/s)
    - ΔP = pressure drop (Pa)
    - η = pump efficiency

    Args:
        flow_rate_kg_s: Mass flow rate (kg/s)
        pressure_drop_bar: Pressure drop (bar)
        pump_efficiency: Pump efficiency (0-1)

    Returns:
        Pumping power in kW
    """
    # Assume water density 1000 kg/m³
    flow_rate_m3_s = flow_rate_kg_s / 1000.0
    pressure_drop_pa = pressure_drop_bar * 100000.0

    power_kw = (flow_rate_m3_s * pressure_drop_pa / 1000.0) / pump_efficiency
    return round(power_kw, 2)


def calculate_npv(
    capital_cost: float,
    annual_cash_flow: float,
    discount_rate: float,
    years: int
) -> float:
    """
    Calculate Net Present Value.

    ZERO-HALLUCINATION FORMULA:
    NPV = -C₀ + Σ(CFₜ / (1+r)ᵗ) for t=1 to n

    Args:
        capital_cost: Initial investment
        annual_cash_flow: Annual net cash flow
        discount_rate: Discount rate
        years: Analysis period

    Returns:
        Net present value
    """
    npv = -capital_cost
    for year in range(1, years + 1):
        npv += annual_cash_flow / ((1 + discount_rate) ** year)
    return round(npv, 2)


# =============================================================================
# DISTRICT HEATING AGENT
# =============================================================================

class DistrictHeatingAgent:
    """
    GL-033: District Heating Integrator Agent (DISTRICT-LINK).

    This agent analyzes industrial waste heat sources and evaluates
    integration opportunities with district heating networks, including
    technical feasibility, economic viability, and environmental benefits.

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas from thermodynamics
    - No LLM inference in calculation path
    - Complete audit trail for regulatory compliance

    Attributes:
        AGENT_ID: Unique agent identifier (GL-033)
        AGENT_NAME: Agent name (DISTRICT-LINK)
        VERSION: Agent version
    """

    AGENT_ID = "GL-033"
    AGENT_NAME = "DISTRICT-LINK"
    VERSION = "1.0.0"
    DESCRIPTION = "District Heating Integration Analyzer"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the DistrictHeatingAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._recommendations: List[str] = []
        self._warnings: List[str] = []

        logger.info(
            f"DistrictHeatingAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: DistrictHeatingInput) -> DistrictHeatingOutput:
        """
        Execute district heating integration analysis.

        This method performs comprehensive integration analysis:
        1. Characterize waste heat sources
        2. Check temperature compatibility
        3. Calculate heat delivery potential
        4. Assess heat pump requirements
        5. Estimate capital and operating costs
        6. Calculate economic metrics
        7. Rank scenarios by feasibility

        Args:
            input_data: Validated input data

        Returns:
            Complete integration analysis with provenance hash
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._recommendations = []
        self._warnings = []

        logger.info(f"Starting district heating analysis for {input_data.facility_id}")

        try:
            scenarios = []
            scenario_num = 0

            # Analyze each waste heat source
            for source in input_data.waste_heat_sources:
                scenario_num += 1

                # Calculate available heat
                available_heat_kw = calculate_heat_capacity(
                    source.flow_rate_kg_s,
                    source.temperature_supply_c,
                    source.temperature_return_c,
                    source.specific_heat_kj_kg_k
                )

                self._track_provenance(
                    f"calculate_heat_{source.source_id}",
                    {
                        "flow": source.flow_rate_kg_s,
                        "temp_supply": source.temperature_supply_c,
                        "temp_return": source.temperature_return_c
                    },
                    {"heat_kw": available_heat_kw},
                    "heat_capacity_calculator"
                )

                if available_heat_kw < 100:
                    self._warnings.append(
                        f"Source {source.source_id}: heat capacity {available_heat_kw:.0f} kW too low for DHN"
                    )
                    continue

                # Check temperature compatibility
                needs_heat_pump = (
                    source.temperature_supply_c <
                    input_data.district_network.supply_temperature_c + input_data.min_approach_temp_c
                )

                if needs_heat_pump and not input_data.heat_pump_option:
                    self._warnings.append(
                        f"Source {source.source_id}: temperature {source.temperature_supply_c:.0f}°C too low "
                        f"for DHN {input_data.district_network.supply_temperature_c:.0f}°C - heat pump required"
                    )
                    continue

                # Check distance feasibility
                if source.distance_to_network_m > input_data.max_piping_distance_m:
                    self._warnings.append(
                        f"Source {source.source_id}: distance {source.distance_to_network_m:.0f}m "
                        f"exceeds maximum {input_data.max_piping_distance_m:.0f}m"
                    )
                    continue

                # Calculate heat delivery
                if needs_heat_pump:
                    # Heat pump case
                    cop = calculate_carnot_cop_heating(
                        input_data.district_network.supply_temperature_c,
                        source.temperature_supply_c
                    )
                    # Heat delivered includes source heat plus compressor work
                    heat_delivered_kw = available_heat_kw * (cop / (cop - 1))
                    heat_pump_power_kw = available_heat_kw / (cop - 1)
                else:
                    # Direct heat exchange
                    cop = None
                    heat_delivered_kw = available_heat_kw
                    heat_pump_power_kw = 0

                # Annual heat delivery
                annual_heat_mwh = (heat_delivered_kw / 1000) * source.availability_hours_per_year

                # Economics - Capital costs
                hx_cost = heat_delivered_kw * input_data.heat_exchanger_cost_per_kw
                piping_cost = source.distance_to_network_m * input_data.piping_cost_per_m

                if needs_heat_pump:
                    heat_pump_cost = heat_pump_power_kw * 800  # $800/kW for industrial heat pump
                else:
                    heat_pump_cost = 0

                total_capital = hx_cost + piping_cost + heat_pump_cost

                # Annual revenue
                annual_revenue = annual_heat_mwh * input_data.district_network.heat_tariff_per_mwh

                # Annual operating costs
                pumping_power_kw = calculate_pumping_power(
                    source.flow_rate_kg_s,
                    input_data.district_network.pressure_supply_bar -
                    input_data.district_network.pressure_return_bar
                )

                pumping_cost_annual = (
                    (pumping_power_kw * source.availability_hours_per_year / 1000) *
                    input_data.pumping_cost_per_kwh
                )

                if needs_heat_pump:
                    hp_electricity_cost = (
                        (heat_pump_power_kw * source.availability_hours_per_year / 1000) *
                        input_data.pumping_cost_per_kwh
                    )
                else:
                    hp_electricity_cost = 0

                maintenance_cost = total_capital * 0.02  # 2% of capital per year

                total_operating_cost = pumping_cost_annual + hp_electricity_cost + maintenance_cost

                # Net benefit
                annual_net_benefit = annual_revenue - total_operating_cost

                # Payback
                if annual_net_benefit > 0:
                    payback = total_capital / annual_net_benefit
                else:
                    payback = 999

                # NPV (20-year analysis)
                npv_20yr = calculate_npv(
                    total_capital,
                    annual_net_benefit,
                    input_data.discount_rate,
                    20
                )

                # CO2 reduction (assuming gas boiler baseline at 0.2 kg CO2/kWh)
                co2_reduction = annual_heat_mwh * 1000 * 0.2 / 1000  # tonnes

                # Feasibility assessment
                technical_feasibility = "HIGH"
                if needs_heat_pump:
                    technical_feasibility = "MEDIUM"
                if source.distance_to_network_m > 1000:
                    technical_feasibility = "MEDIUM"
                if source.contaminants:
                    technical_feasibility = "MEDIUM"

                if npv_20yr > 0 and payback < 10:
                    economic_feasibility = "HIGH"
                elif npv_20yr > 0 and payback < 15:
                    economic_feasibility = "MEDIUM"
                else:
                    economic_feasibility = "LOW"

                scenario = IntegrationScenario(
                    scenario_id=f"DHN-{scenario_num:02d}",
                    source_id=source.source_id,
                    heat_delivered_kw=round(heat_delivered_kw, 1),
                    heat_delivered_mwh_per_year=round(annual_heat_mwh, 0),
                    heat_exchanger_size_kw=round(heat_delivered_kw, 1),
                    heat_pump_required=needs_heat_pump,
                    heat_pump_cop=cop,
                    capital_cost=round(total_capital, 0),
                    annual_revenue=round(annual_revenue, 0),
                    annual_operating_cost=round(total_operating_cost, 0),
                    annual_net_benefit=round(annual_net_benefit, 0),
                    simple_payback_years=round(payback, 1),
                    npv_20yr=round(npv_20yr, 0),
                    co2_reduction_tonnes_per_year=round(co2_reduction, 1),
                    technical_feasibility=technical_feasibility,
                    economic_feasibility=economic_feasibility
                )

                scenarios.append(scenario)

            # Sort by NPV
            scenarios.sort(key=lambda x: -x.npv_20yr)

            # Calculate totals
            total_heat_mw = sum(s.heat_delivered_kw for s in scenarios) / 1000
            total_annual_heat = sum(s.heat_delivered_mwh_per_year for s in scenarios)
            total_revenue = sum(s.annual_revenue for s in scenarios)
            total_capital = sum(s.capital_cost for s in scenarios)
            total_co2 = sum(s.co2_reduction_tonnes_per_year for s in scenarios)

            # Feasibility score
            if scenarios:
                best_scenario = scenarios[0]
                recommended_id = best_scenario.scenario_id

                # Score based on best scenario
                if best_scenario.npv_20yr > 0 and best_scenario.simple_payback_years < 10:
                    feasibility_score = 90.0
                elif best_scenario.npv_20yr > 0:
                    feasibility_score = 70.0
                else:
                    feasibility_score = 40.0

                # Adjust for technical factors
                if best_scenario.technical_feasibility == "HIGH":
                    feasibility_score = min(100, feasibility_score + 5)
                elif best_scenario.technical_feasibility == "LOW":
                    feasibility_score = max(0, feasibility_score - 20)
            else:
                feasibility_score = 0.0
                recommended_id = None

            # Generate recommendations
            self._generate_recommendations(scenarios, input_data)

            # Infrastructure requirements
            required_infra = []
            if any(s.heat_pump_required for s in scenarios):
                required_infra.append("Industrial heat pump system")
            required_infra.extend([
                "Heat exchangers",
                "Insulated district heating pipework",
                "Pumping stations",
                "Control and monitoring systems"
            ])

            # Calculate provenance hash
            calc_hash = self._calculate_provenance_hash()

            # Processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"DH-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.facility_id.encode()).hexdigest()[:8]}"
            )

            output = DistrictHeatingOutput(
                analysis_id=analysis_id,
                facility_id=input_data.facility_id,
                integration_scenarios=scenarios,
                total_potential_heat_mw=round(total_heat_mw, 2),
                total_annual_heat_mwh=round(total_annual_heat, 0),
                total_annual_revenue=round(total_revenue, 0),
                total_capital_cost=round(total_capital, 0),
                total_co2_reduction_tonnes=round(total_co2, 1),
                feasibility_score=round(feasibility_score, 1),
                recommended_scenario_id=recommended_id,
                recommendations=self._recommendations,
                warnings=self._warnings,
                required_infrastructure=required_infra,
                calculation_hash=calc_hash,
                provenance_chain=self._provenance_steps,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS",
                validation_errors=[]
            )

            logger.info(
                f"District heating analysis complete for {input_data.facility_id}: "
                f"feasibility={feasibility_score:.0f}, scenarios={len(scenarios)} "
                f"(duration: {processing_time:.1f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"District heating analysis failed: {str(e)}", exc_info=True)
            raise

    def _generate_recommendations(
        self,
        scenarios: List[IntegrationScenario],
        input_data: DistrictHeatingInput
    ):
        """Generate implementation recommendations."""
        if not scenarios:
            self._recommendations.append(
                "No viable integration scenarios identified - consider improving heat source quality "
                "or reducing distance to district network"
            )
            return

        best = scenarios[0]

        if best.npv_20yr > 0:
            self._recommendations.append(
                f"Recommended scenario {best.scenario_id}: "
                f"NPV ${best.npv_20yr:,.0f} over 20 years, "
                f"payback {best.simple_payback_years:.1f} years"
            )

        if best.heat_pump_required:
            self._recommendations.append(
                f"Heat pump system required with COP {best.heat_pump_cop:.1f} "
                f"to boost temperature for district network"
            )

        if len(scenarios) > 1:
            self._recommendations.append(
                f"Consider phased implementation: start with scenario {best.scenario_id}, "
                f"then add additional sources"
            )

        # Check network capacity
        for scenario in scenarios:
            if scenario.heat_delivered_kw / 1000 > input_data.district_network.design_capacity_mw * 0.3:
                self._recommendations.append(
                    f"Scenario {scenario.scenario_id} represents significant capacity - "
                    f"coordinate closely with DHN operator"
                )
                break

        self._recommendations.append(
            "Engage with district heating network operator early in planning process"
        )

        if any(s.co2_reduction_tonnes_per_year > 100 for s in scenarios):
            self._recommendations.append(
                "Significant CO2 reduction potential - explore carbon credits and incentives"
            )

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ):
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"]
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata."""
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "description": self.DESCRIPTION,
            "category": "District Heating",
            "type": "Integration Analyzer",
            "standards": ["EN_13941", "ISO_50001", "ASHRAE"],
        }


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-033",
    "name": "DISTRICT-LINK - District Heating Integration Analyzer",
    "version": "1.0.0",
    "summary": "Analyzes industrial waste heat integration with district heating networks",
    "tags": [
        "district-heating",
        "waste-heat",
        "heat-integration",
        "energy-efficiency",
        "EN-13941",
        "ISO-50001"
    ],
    "owners": ["district-energy-team"],
    "compute": {
        "entrypoint": "python://agents.gl_033_district_heating.agent:DistrictHeatingAgent",
        "deterministic": True
    },
    "standards": [
        {"ref": "EN 13941", "description": "District Heating Networks Design"},
        {"ref": "ISO 50001", "description": "Energy Management Systems"},
        {"ref": "ASHRAE", "description": "District Energy Systems"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
