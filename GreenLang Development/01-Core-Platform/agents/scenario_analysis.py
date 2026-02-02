# -*- coding: utf-8 -*-
"""
greenlang/agents/scenario_analysis.py

Scenario Analysis for FuelAgentAI v2

OBJECTIVE:
Enable "what-if" analysis for emissions reduction strategies

FEATURES:
- Fuel switching scenarios (diesel → biodiesel, coal → natural gas)
- Efficiency improvement scenarios (boiler upgrade 80% → 95%)
- Renewable offset scenarios (0% → 50% renewables)
- Side-by-side comparison of multiple scenarios
- Cost-benefit analysis (emissions reduction vs cost)
- Sensitivity analysis (how changes affect outcomes)

USE CASES:
- Corporate decarbonization planning
- Technology investment decisions
- Compliance strategy optimization
- Stakeholder reporting (what-if scenarios)

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ScenarioType(str, Enum):
    """Type of reduction scenario"""
    FUEL_SWITCH = "fuel_switch"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    RENEWABLE_OFFSET = "renewable_offset"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    COMBINED = "combined"


@dataclass
class Scenario:
    """
    Represents a single emissions reduction scenario.
    """
    name: str
    description: str
    scenario_type: ScenarioType
    baseline_payload: Dict[str, Any]
    modified_payload: Dict[str, Any]
    expected_reduction_pct: Optional[float] = None
    implementation_cost_usd: Optional[float] = None
    annual_savings_usd: Optional[float] = None
    payback_years: Optional[float] = None


@dataclass
class ScenarioResult:
    """
    Results from analyzing a scenario.
    """
    scenario_name: str
    baseline_emissions_kg: float
    scenario_emissions_kg: float
    reduction_kg: float
    reduction_pct: float
    implementation_cost_usd: Optional[float] = None
    annual_savings_usd: Optional[float] = None
    payback_years: Optional[float] = None
    cost_per_ton_co2e_avoided: Optional[float] = None
    baseline_data: Dict[str, Any] = field(default_factory=dict)
    scenario_data: Dict[str, Any] = field(default_factory=dict)


class ScenarioAnalysis:
    """
    Scenario analysis engine for emissions reduction strategies.

    Enables "what-if" analysis by comparing baseline emissions to
    alternative scenarios (fuel switching, efficiency improvements, etc.)
    """

    def __init__(self, agent):
        """
        Initialize scenario analysis with FuelAgentAI instance.

        Args:
            agent: FuelAgentAI_v2 instance
        """
        self.agent = agent

    def analyze_scenario(
        self,
        scenario: Scenario,
        response_format: str = "enhanced"
    ) -> ScenarioResult:
        """
        Analyze a single scenario.

        Args:
            scenario: Scenario definition
            response_format: Output format (legacy, enhanced, compact)

        Returns:
            ScenarioResult with emissions comparison
        """
        # Run baseline
        baseline_payload = {**scenario.baseline_payload, "response_format": response_format}
        baseline_result = self.agent.run(baseline_payload)

        if not baseline_result["success"]:
            raise ValueError(f"Baseline calculation failed: {baseline_result['error']}")

        baseline_emissions = baseline_result["data"]["co2e_emissions_kg"]

        # Run scenario
        scenario_payload = {**scenario.modified_payload, "response_format": response_format}
        scenario_result = self.agent.run(scenario_payload)

        if not scenario_result["success"]:
            raise ValueError(f"Scenario calculation failed: {scenario_result['error']}")

        scenario_emissions = scenario_result["data"]["co2e_emissions_kg"]

        # Calculate reduction
        reduction_kg = baseline_emissions - scenario_emissions
        reduction_pct = (reduction_kg / baseline_emissions) * 100 if baseline_emissions > 0 else 0

        # Calculate cost-effectiveness
        cost_per_ton = None
        if scenario.implementation_cost_usd and reduction_kg > 0:
            reduction_tons = reduction_kg / 1000  # kg to tonnes
            cost_per_ton = scenario.implementation_cost_usd / reduction_tons

        return ScenarioResult(
            scenario_name=scenario.name,
            baseline_emissions_kg=baseline_emissions,
            scenario_emissions_kg=scenario_emissions,
            reduction_kg=reduction_kg,
            reduction_pct=reduction_pct,
            implementation_cost_usd=scenario.implementation_cost_usd,
            annual_savings_usd=scenario.annual_savings_usd,
            payback_years=scenario.payback_years,
            cost_per_ton_co2e_avoided=cost_per_ton,
            baseline_data=baseline_result["data"],
            scenario_data=scenario_result["data"],
        )

    def compare_scenarios(
        self,
        scenarios: List[Scenario],
        response_format: str = "enhanced"
    ) -> List[ScenarioResult]:
        """
        Compare multiple scenarios side-by-side.

        Args:
            scenarios: List of scenarios to compare
            response_format: Output format

        Returns:
            List of ScenarioResult sorted by reduction_pct (descending)
        """
        results = []

        for scenario in scenarios:
            result = self.analyze_scenario(scenario, response_format)
            results.append(result)

        # Sort by reduction percentage (descending)
        results.sort(key=lambda x: x.reduction_pct, reverse=True)

        return results

    def generate_fuel_switch_scenario(
        self,
        baseline_fuel: str,
        baseline_amount: float,
        baseline_unit: str,
        target_fuel: str,
        target_amount: Optional[float] = None,
        target_unit: Optional[str] = None,
        country: str = "US",
        implementation_cost: Optional[float] = None,
    ) -> Scenario:
        """
        Generate a fuel switching scenario.

        Args:
            baseline_fuel: Current fuel type
            baseline_amount: Current consumption amount
            baseline_unit: Current unit
            target_fuel: Target fuel type
            target_amount: Target consumption (if different)
            target_unit: Target unit (if different)
            country: Country code
            implementation_cost: Cost to implement switch (USD)

        Returns:
            Scenario object
        """
        # Use same amount/unit if not specified
        if target_amount is None:
            target_amount = baseline_amount
        if target_unit is None:
            target_unit = baseline_unit

        baseline_payload = {
            "fuel_type": baseline_fuel,
            "amount": baseline_amount,
            "unit": baseline_unit,
            "country": country,
        }

        modified_payload = {
            "fuel_type": target_fuel,
            "amount": target_amount,
            "unit": target_unit,
            "country": country,
        }

        return Scenario(
            name=f"Fuel Switch: {baseline_fuel} → {target_fuel}",
            description=f"Switch from {baseline_fuel} to {target_fuel}",
            scenario_type=ScenarioType.FUEL_SWITCH,
            baseline_payload=baseline_payload,
            modified_payload=modified_payload,
            implementation_cost_usd=implementation_cost,
        )

    def generate_efficiency_scenario(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        baseline_efficiency: float,
        target_efficiency: float,
        country: str = "US",
        implementation_cost: Optional[float] = None,
    ) -> Scenario:
        """
        Generate an efficiency improvement scenario.

        Args:
            fuel_type: Fuel type
            amount: Consumption amount
            unit: Unit
            baseline_efficiency: Current efficiency (0-1)
            target_efficiency: Target efficiency (0-1)
            country: Country code
            implementation_cost: Cost to upgrade (USD)

        Returns:
            Scenario object
        """
        baseline_payload = {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "country": country,
            "efficiency": baseline_efficiency,
        }

        modified_payload = {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "country": country,
            "efficiency": target_efficiency,
        }

        improvement_pct = ((target_efficiency - baseline_efficiency) / baseline_efficiency) * 100

        return Scenario(
            name=f"Efficiency Upgrade: {baseline_efficiency*100:.0f}% → {target_efficiency*100:.0f}%",
            description=f"Improve equipment efficiency by {improvement_pct:.1f}%",
            scenario_type=ScenarioType.EFFICIENCY_IMPROVEMENT,
            baseline_payload=baseline_payload,
            modified_payload=modified_payload,
            implementation_cost_usd=implementation_cost,
        )

    def generate_renewable_scenario(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        baseline_renewable_pct: float,
        target_renewable_pct: float,
        country: str = "US",
        implementation_cost: Optional[float] = None,
    ) -> Scenario:
        """
        Generate a renewable offset scenario.

        Args:
            fuel_type: Fuel type (typically "electricity")
            amount: Consumption amount
            unit: Unit
            baseline_renewable_pct: Current renewable % (0-100)
            target_renewable_pct: Target renewable % (0-100)
            country: Country code
            implementation_cost: Cost to add renewables (USD)

        Returns:
            Scenario object
        """
        baseline_payload = {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "country": country,
            "renewable_percentage": baseline_renewable_pct,
        }

        modified_payload = {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "country": country,
            "renewable_percentage": target_renewable_pct,
        }

        increase_pct = target_renewable_pct - baseline_renewable_pct

        return Scenario(
            name=f"Renewable Increase: {baseline_renewable_pct:.0f}% → {target_renewable_pct:.0f}%",
            description=f"Add {increase_pct:.0f}% renewable energy",
            scenario_type=ScenarioType.RENEWABLE_OFFSET,
            baseline_payload=baseline_payload,
            modified_payload=modified_payload,
            implementation_cost_usd=implementation_cost,
        )

    def generate_common_scenarios(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        country: str = "US"
    ) -> List[Scenario]:
        """
        Generate common reduction scenarios for a fuel type.

        Args:
            fuel_type: Fuel type
            amount: Consumption amount
            unit: Unit
            country: Country code

        Returns:
            List of common scenarios for the fuel type
        """
        scenarios = []

        # Diesel scenarios
        if fuel_type == "diesel":
            scenarios.extend([
                self.generate_fuel_switch_scenario(
                    "diesel", amount, unit, "biodiesel", country=country,
                    implementation_cost=5000
                ),
                self.generate_efficiency_scenario(
                    "diesel", amount, unit, 0.85, 0.95, country=country,
                    implementation_cost=15000
                ),
            ])

        # Natural gas scenarios
        elif fuel_type == "natural_gas":
            scenarios.extend([
                self.generate_efficiency_scenario(
                    "natural_gas", amount, unit, 0.80, 0.90, country=country,
                    implementation_cost=20000
                ),
                self.generate_efficiency_scenario(
                    "natural_gas", amount, unit, 0.80, 0.95, country=country,
                    implementation_cost=35000
                ),
            ])

        # Electricity scenarios
        elif fuel_type == "electricity":
            scenarios.extend([
                self.generate_renewable_scenario(
                    "electricity", amount, unit, 0, 25, country=country,
                    implementation_cost=10000
                ),
                self.generate_renewable_scenario(
                    "electricity", amount, unit, 0, 50, country=country,
                    implementation_cost=25000
                ),
                self.generate_renewable_scenario(
                    "electricity", amount, unit, 0, 100, country=country,
                    implementation_cost=60000
                ),
            ])

        # Coal scenarios
        elif fuel_type == "coal":
            scenarios.extend([
                self.generate_fuel_switch_scenario(
                    "coal", amount, "tons", "natural_gas", amount * 2, "therms",
                    country=country, implementation_cost=100000
                ),
            ])

        return scenarios

    def sensitivity_analysis(
        self,
        base_payload: Dict[str, Any],
        parameter: str,
        values: List[Any],
        response_format: str = "enhanced"
    ) -> List[Dict[str, Any]]:
        """
        Perform sensitivity analysis on a parameter.

        Args:
            base_payload: Base calculation payload
            parameter: Parameter to vary (e.g., "efficiency", "renewable_percentage")
            values: List of values to test
            response_format: Output format

        Returns:
            List of results for each parameter value
        """
        results = []

        for value in values:
            payload = {**base_payload, parameter: value, "response_format": response_format}
            result = self.agent.run(payload)

            if result["success"]:
                results.append({
                    parameter: value,
                    "emissions_kg": result["data"]["co2e_emissions_kg"],
                    "data": result["data"]
                })

        return results
