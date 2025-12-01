# -*- coding: utf-8 -*-
"""
Cost Optimization Calculator for GL-011 FUELCRAFT.

Provides deterministic cost minimization algorithms for fuel selection
considering market prices, inventory levels, and delivery costs.

Zero-hallucination: All calculations are deterministic.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CostOptimizationInput:
    """Input for cost optimization."""
    energy_demand_mw: float
    available_fuels: List[str]
    fuel_properties: Dict[str, Dict[str, Any]]
    market_prices: Dict[str, float]
    fuel_inventories: Dict[str, float]
    delivery_costs: Dict[str, float]
    constraints: Dict[str, Any]


@dataclass
class CostOptimizationOutput:
    """Output of cost optimization."""
    optimal_fuel: str
    optimal_quantity_kg: float
    total_cost_usd: float
    cost_breakdown: Dict[str, float]
    alternatives: List[Dict[str, Any]]
    savings_potential_usd: float
    inventory_sufficient: bool
    procurement_needed: float
    provenance_hash: str


class CostOptimizationCalculator:
    """
    Deterministic cost optimization calculator.

    Finds the minimum cost fuel option considering:
    - Fuel prices
    - Delivery costs
    - Inventory availability
    - Minimum order quantities
    """

    MJ_TO_GJ = 0.001

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        self.config = config or {}
        self.calculation_count = 0

    def optimize(self, input_data: CostOptimizationInput) -> CostOptimizationOutput:
        """
        Find optimal fuel for minimum cost.

        Args:
            input_data: Cost optimization parameters

        Returns:
            Cost-optimized fuel selection
        """
        self.calculation_count += 1

        # Calculate energy requirement in GJ/hr
        energy_gj = input_data.energy_demand_mw * 3.6

        # Calculate cost for each fuel option
        fuel_costs = []

        for fuel in input_data.available_fuels:
            props = input_data.fuel_properties.get(fuel, {})
            price = input_data.market_prices.get(fuel, 0.05)
            inventory = input_data.fuel_inventories.get(fuel, 0)
            delivery = input_data.delivery_costs.get(fuel, 0)
            heating_value_gj = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ

            # Calculate required quantity
            qty_kg = energy_gj / heating_value_gj if heating_value_gj > 0 else 0

            # Calculate costs
            fuel_cost = qty_kg * price
            delivery_cost = qty_kg * delivery / 1000  # Delivery per tonne
            total_cost = fuel_cost + delivery_cost

            # Check inventory
            inventory_sufficient = inventory >= qty_kg
            procurement_needed = max(0, qty_kg - inventory)

            fuel_costs.append({
                'fuel': fuel,
                'quantity_kg': qty_kg,
                'fuel_cost_usd': fuel_cost,
                'delivery_cost_usd': delivery_cost,
                'total_cost_usd': total_cost,
                'cost_per_gj': total_cost / energy_gj if energy_gj > 0 else 0,
                'inventory_sufficient': inventory_sufficient,
                'procurement_needed': procurement_needed,
                'price_per_kg': price
            })

        # Sort by total cost
        fuel_costs.sort(key=lambda x: x['total_cost_usd'])

        # Select optimal (cheapest)
        optimal = fuel_costs[0] if fuel_costs else {
            'fuel': None,
            'quantity_kg': 0,
            'total_cost_usd': 0
        }

        # Calculate savings vs alternatives
        savings = 0
        if len(fuel_costs) > 1:
            savings = fuel_costs[-1]['total_cost_usd'] - optimal['total_cost_usd']

        # Provenance hash
        provenance_hash = self._calculate_provenance(input_data, optimal)

        return CostOptimizationOutput(
            optimal_fuel=optimal['fuel'],
            optimal_quantity_kg=round(optimal['quantity_kg'], 2),
            total_cost_usd=round(optimal['total_cost_usd'], 2),
            cost_breakdown={
                'fuel_cost': round(optimal.get('fuel_cost_usd', 0), 2),
                'delivery_cost': round(optimal.get('delivery_cost_usd', 0), 2)
            },
            alternatives=fuel_costs[1:4],
            savings_potential_usd=round(savings, 2),
            inventory_sufficient=optimal.get('inventory_sufficient', False),
            procurement_needed=round(optimal.get('procurement_needed', 0), 2),
            provenance_hash=provenance_hash
        )

    def calculate_levelized_cost(
        self,
        fuel: str,
        properties: Dict[str, Any],
        price: float,
        delivery_cost: float,
        carbon_price: float = 0
    ) -> float:
        """
        Calculate levelized cost of energy for a fuel.

        Args:
            fuel: Fuel type
            properties: Fuel properties
            price: Fuel price per kg
            delivery_cost: Delivery cost per tonne
            carbon_price: Carbon price per tonne CO2

        Returns:
            Levelized cost in USD/GJ
        """
        heating_value_gj = properties.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ
        co2_factor = properties.get('emission_factor_co2_kg_gj', 60)

        # Fuel cost per GJ
        fuel_cost_per_gj = price / heating_value_gj if heating_value_gj > 0 else 0

        # Delivery cost per GJ
        delivery_per_gj = (delivery_cost / 1000) / heating_value_gj if heating_value_gj > 0 else 0

        # Carbon cost per GJ
        carbon_cost_per_gj = co2_factor * carbon_price / 1000

        return fuel_cost_per_gj + delivery_per_gj + carbon_cost_per_gj

    def _calculate_provenance(
        self,
        input_data: CostOptimizationInput,
        result: Dict[str, Any]
    ) -> str:
        """Calculate provenance hash."""
        data = {
            'demand': input_data.energy_demand_mw,
            'fuels': sorted(input_data.available_fuels),
            'prices': input_data.market_prices,
            'optimal_fuel': result.get('fuel'),
            'cost': result.get('total_cost_usd')
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
