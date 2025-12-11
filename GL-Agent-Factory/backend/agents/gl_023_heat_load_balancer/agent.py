"""GL-023 LOADBALANCER: Heat Load Balancer Agent.

Optimizes heat load distribution across multiple boilers and furnaces
using economic dispatch principles for cost, efficiency, and emissions.

Standards: IEEE 1547, NERC, Industrial best practices
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    LoadBalancerInput,
    LoadBalancerOutput,
    LoadAllocation,
    EquipmentUnit,
)
from .formulas import (
    calculate_efficiency_at_load,
    calculate_fuel_consumption,
    calculate_hourly_cost,
    calculate_emissions,
    economic_dispatch_merit_order,
    calculate_fleet_efficiency,
    calculate_equal_loading,
    generate_calculation_hash,
)

logger = logging.getLogger(__name__)


class HeatLoadBalancerAgent:
    """
    Heat load balancing agent using economic dispatch optimization.

    Features:
    - Merit order economic dispatch
    - Multi-objective optimization (cost/efficiency/emissions)
    - Spinning reserve management
    - Unit commitment recommendations
    - Zero-hallucination deterministic calculations
    """

    AGENT_ID = "GL-023"
    AGENT_NAME = "LOADBALANCER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the heat load balancer agent."""
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous entry point."""
        validated_input = LoadBalancerInput(**input_data)
        output = self._process(validated_input)
        return output.model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async entry point."""
        return self.run(input_data)

    def _process(self, input_data: LoadBalancerInput) -> LoadBalancerOutput:
        """Process input and optimize load allocation."""
        recommendations = []
        warnings = []
        constraint_violations = []

        # Convert equipment to dict format for formulas
        units = [unit.model_dump() for unit in input_data.equipment]

        # Calculate total available capacity
        available_units = [u for u in units if u['is_available']]
        total_capacity = sum(u['max_load_mw'] for u in available_units)
        total_min_load = sum(u['min_load_mw'] for u in available_units if u['is_running'])

        # Check if demand can be met
        if input_data.total_heat_demand_mw > total_capacity:
            warnings.append(
                f"Demand {input_data.total_heat_demand_mw:.1f} MW exceeds capacity "
                f"{total_capacity:.1f} MW"
            )
            constraint_violations.append("CAPACITY_EXCEEDED")

        # Perform economic dispatch
        allocations = economic_dispatch_merit_order(
            units=units,
            total_demand_mw=input_data.total_heat_demand_mw,
            carbon_price=input_data.carbon_price_per_ton
        )

        # Build detailed allocation results
        allocation_results = []
        total_allocated = 0.0
        total_hourly_cost = 0.0
        total_hourly_emissions = 0.0
        units_running = 0
        units_starting = 0
        units_stopping = 0

        unit_lookup = {u['unit_id']: u for u in units}

        for unit_id, target_load in allocations:
            unit = unit_lookup.get(unit_id)
            if not unit:
                continue

            current_load = unit['current_load_mw']
            load_change = target_load - current_load

            # Determine action
            if target_load > 0 and current_load == 0:
                action = "START"
                units_starting += 1
            elif target_load == 0 and current_load > 0:
                action = "STOP"
                units_stopping += 1
            elif load_change > 0.1:
                action = "INCREASE"
            elif load_change < -0.1:
                action = "DECREASE"
            else:
                action = "MAINTAIN"

            if target_load > 0:
                units_running += 1

            # Calculate operating parameters
            efficiency = calculate_efficiency_at_load(
                target_load,
                unit['min_load_mw'],
                unit['max_load_mw'],
                unit.get('efficiency_curve_a', 0),
                unit.get('efficiency_curve_b', 0),
                unit.get('efficiency_curve_c', 0),
                unit.get('current_efficiency_pct', 80)
            )

            fuel_consumption = calculate_fuel_consumption(target_load, efficiency)
            hourly_cost = calculate_hourly_cost(
                fuel_consumption,
                unit['fuel_cost_per_mwh'],
                unit.get('maintenance_cost_per_mwh', 0),
                target_load
            )
            hourly_emissions = calculate_emissions(
                fuel_consumption,
                unit.get('emissions_factor_kg_co2_mwh', 200)
            )

            allocation_results.append(LoadAllocation(
                unit_id=unit_id,
                target_load_mw=round(target_load, 3),
                load_change_mw=round(load_change, 3),
                action=action,
                efficiency_at_load_pct=efficiency,
                fuel_consumption_mw=fuel_consumption,
                hourly_cost=hourly_cost,
                hourly_emissions_kg_co2=hourly_emissions
            ))

            total_allocated += target_load
            total_hourly_cost += hourly_cost
            total_hourly_emissions += hourly_emissions

        # Calculate spinning reserve
        spinning_reserve = total_capacity - total_allocated
        spinning_reserve_pct = (spinning_reserve / total_capacity * 100) if total_capacity > 0 else 0

        # Check spinning reserve constraint
        if spinning_reserve_pct < input_data.min_spinning_reserve_pct:
            warnings.append(
                f"Spinning reserve {spinning_reserve_pct:.1f}% below minimum "
                f"{input_data.min_spinning_reserve_pct:.1f}%"
            )
            constraint_violations.append("SPINNING_RESERVE_LOW")
            recommendations.append("Consider starting additional standby units")

        # Check max startups constraint
        if units_starting > input_data.max_units_starting:
            warnings.append(
                f"Starting {units_starting} units exceeds limit {input_data.max_units_starting}"
            )
            constraint_violations.append("MAX_STARTUPS_EXCEEDED")

        # Calculate fleet efficiency
        fleet_efficiency = calculate_fleet_efficiency(allocations, units)

        # Calculate baseline (equal loading) for comparison
        baseline_efficiency, baseline_cost = calculate_equal_loading(
            units, input_data.total_heat_demand_mw
        )

        efficiency_improvement = fleet_efficiency - baseline_efficiency if baseline_efficiency > 0 else 0
        cost_savings_pct = ((baseline_cost - total_hourly_cost) / baseline_cost * 100) if baseline_cost > 0 else 0

        # Cost per MWh and emissions intensity
        cost_per_mwh = total_hourly_cost / total_allocated if total_allocated > 0 else 0
        emissions_intensity = total_hourly_emissions / total_allocated if total_allocated > 0 else 0

        # Generate recommendations
        if efficiency_improvement > 1:
            recommendations.append(
                f"Optimized loading improves efficiency by {efficiency_improvement:.1f}% "
                f"vs equal loading"
            )

        if cost_savings_pct > 0:
            recommendations.append(
                f"Economic dispatch saves ${baseline_cost - total_hourly_cost:.0f}/hr "
                f"({cost_savings_pct:.1f}%)"
            )

        # Check for units at limits
        for alloc in allocation_results:
            unit = unit_lookup.get(alloc.unit_id)
            if unit and alloc.target_load_mw >= unit['max_load_mw'] * 0.95:
                recommendations.append(
                    f"Unit {alloc.unit_id} near maximum - consider additional capacity"
                )

        # Forecast warning
        if input_data.demand_forecast_1hr_mw:
            if input_data.demand_forecast_1hr_mw > total_capacity * 0.9:
                warnings.append(
                    f"1-hr forecast {input_data.demand_forecast_1hr_mw:.1f} MW approaching capacity"
                )

        constraints_satisfied = len(constraint_violations) == 0

        # Generate provenance hash
        calc_hash = generate_calculation_hash(
            inputs={"demand": input_data.total_heat_demand_mw, "units": len(units)},
            outputs={"allocated": total_allocated, "cost": total_hourly_cost}
        )

        return LoadBalancerOutput(
            allocations=allocation_results,
            total_capacity_mw=round(total_capacity, 3),
            total_allocated_mw=round(total_allocated, 3),
            spinning_reserve_mw=round(spinning_reserve, 3),
            spinning_reserve_pct=round(spinning_reserve_pct, 2),
            fleet_efficiency_pct=fleet_efficiency,
            efficiency_vs_equal_load_pct=round(efficiency_improvement, 2),
            total_hourly_cost=round(total_hourly_cost, 2),
            cost_per_mwh=round(cost_per_mwh, 2),
            cost_savings_vs_equal_pct=round(cost_savings_pct, 2),
            total_hourly_emissions_kg=round(total_hourly_emissions, 2),
            emissions_intensity_kg_mwh=round(emissions_intensity, 2),
            units_running=units_running,
            units_starting=units_starting,
            units_stopping=units_stopping,
            constraints_satisfied=constraints_satisfied,
            constraint_violations=constraint_violations,
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            optimization_method="MERIT_ORDER_DISPATCH",
            calculation_timestamp=datetime.utcnow(),
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Optimization",
            "type": "Optimizer",
            "complexity": "High",
            "description": "Balances heat loads across multiple boilers and furnaces",
            "input_schema": LoadBalancerInput.model_json_schema(),
            "output_schema": LoadBalancerOutput.model_json_schema()
        }
