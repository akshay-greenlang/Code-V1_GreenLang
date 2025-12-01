# -*- coding: utf-8 -*-
"""
Procurement Optimizer Calculator for GL-011 FUELCRAFT.

Provides deterministic algorithms for fuel sourcing, inventory optimization,
and procurement scheduling.

Zero-hallucination: All calculations are deterministic.
"""

import hashlib
import json
import logging
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ProcurementInput:
    """Input for procurement optimization."""
    fuel_inventories: Dict[str, float]  # fuel -> kg
    daily_consumption: Dict[str, float]  # fuel -> kg/day
    market_prices: Dict[str, float]  # fuel -> USD/kg
    delivery_costs: Dict[str, float]  # fuel -> USD/tonne
    lead_times: Dict[str, int]  # fuel -> days
    storage_capacities: Dict[str, float]  # fuel -> kg
    safety_stock_days: int = 7
    planning_horizon_days: int = 30
    minimum_orders: Dict[str, float] = None  # fuel -> kg


@dataclass
class ProcurementOutput:
    """Output of procurement optimization."""
    recommendations: List[Dict[str, Any]]
    total_procurement_cost_usd: float
    fuels_requiring_reorder: int
    immediate_actions: int
    days_of_supply: Dict[str, float]
    reorder_schedule: List[Dict[str, Any]]
    economic_order_quantities: Dict[str, float]
    provenance_hash: str


class ProcurementOptimizer:
    """
    Deterministic procurement optimization calculator.

    Implements EOQ (Economic Order Quantity) and reorder point
    calculations for fuel inventory management.

    Features:
    - Reorder point calculation
    - Economic order quantity
    - Safety stock optimization
    - Lead time consideration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        self.config = config or {}
        self.holding_cost_rate = self.config.get('holding_cost_rate', 0.15)  # 15% annually
        self.calculation_count = 0

    def optimize(self, input_data: ProcurementInput) -> ProcurementOutput:
        """
        Optimize procurement for all fuels.

        Args:
            input_data: Procurement parameters

        Returns:
            Procurement recommendations
        """
        self.calculation_count += 1

        recommendations = []
        total_cost = 0.0
        reorder_schedule = []
        days_of_supply = {}
        eoq_results = {}
        immediate_count = 0
        reorder_count = 0

        minimum_orders = input_data.minimum_orders or {}

        for fuel, inventory in input_data.fuel_inventories.items():
            consumption = input_data.daily_consumption.get(fuel, 0)
            price = input_data.market_prices.get(fuel, 0.05)
            delivery = input_data.delivery_costs.get(fuel, 0)
            lead_time = input_data.lead_times.get(fuel, 7)
            capacity = input_data.storage_capacities.get(fuel, float('inf'))
            min_order = minimum_orders.get(fuel, 1000)

            if consumption <= 0:
                days_of_supply[fuel] = float('inf')
                recommendations.append({
                    'fuel': fuel,
                    'action': 'monitor',
                    'priority': 'low',
                    'current_inventory_kg': round(inventory, 2),
                    'days_of_supply': 'N/A',
                    'reason': 'No consumption'
                })
                continue

            # Calculate days of supply
            dos = inventory / consumption
            days_of_supply[fuel] = dos

            # Calculate reorder point
            safety_stock = consumption * input_data.safety_stock_days
            reorder_point = consumption * lead_time + safety_stock

            # Calculate EOQ
            annual_demand = consumption * 365
            order_cost = delivery / 1000 * 100  # Assume ordering cost proportional to delivery
            holding_cost = price * self.holding_cost_rate

            eoq = self._calculate_eoq(annual_demand, order_cost, holding_cost)
            eoq = max(eoq, min_order)  # Apply minimum order
            eoq = min(eoq, capacity - safety_stock)  # Respect capacity
            eoq_results[fuel] = eoq

            # Determine action
            if inventory <= safety_stock:
                priority = 'critical'
                action = 'reorder_urgent'
                immediate_count += 1
                reorder_count += 1
            elif inventory <= reorder_point:
                priority = 'high'
                action = 'reorder'
                reorder_count += 1
            elif dos < input_data.planning_horizon_days:
                priority = 'medium'
                action = 'plan_order'
            else:
                priority = 'low'
                action = 'monitor'

            # Calculate order quantity
            target_inventory = consumption * input_data.planning_horizon_days
            order_qty = max(0, min(target_inventory - inventory, eoq))

            # Calculate cost
            order_cost_usd = order_qty * price + (order_qty / 1000) * delivery
            if order_qty > 0:
                total_cost += order_cost_usd

            rec = {
                'fuel': fuel,
                'action': action,
                'priority': priority,
                'current_inventory_kg': round(inventory, 2),
                'days_of_supply': round(dos, 1),
                'reorder_point_kg': round(reorder_point, 2),
                'safety_stock_kg': round(safety_stock, 2),
                'recommended_order_kg': round(order_qty, 2),
                'estimated_cost_usd': round(order_cost_usd, 2),
                'lead_time_days': lead_time,
                'current_price_usd_kg': round(price, 4)
            }

            recommendations.append(rec)

            # Add to reorder schedule if needed
            if action in ['reorder', 'reorder_urgent']:
                reorder_date = datetime.now() + timedelta(days=max(0, dos - lead_time - 2))
                reorder_schedule.append({
                    'fuel': fuel,
                    'order_date': reorder_date.strftime('%Y-%m-%d'),
                    'expected_delivery': (reorder_date + timedelta(days=lead_time)).strftime('%Y-%m-%d'),
                    'quantity_kg': round(order_qty, 2),
                    'priority': priority
                })

        # Sort recommendations by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

        # Sort reorder schedule by date
        reorder_schedule.sort(key=lambda x: x['order_date'])

        provenance = self._calculate_provenance(input_data, total_cost)

        return ProcurementOutput(
            recommendations=recommendations,
            total_procurement_cost_usd=round(total_cost, 2),
            fuels_requiring_reorder=reorder_count,
            immediate_actions=immediate_count,
            days_of_supply={k: round(v, 1) for k, v in days_of_supply.items()},
            reorder_schedule=reorder_schedule,
            economic_order_quantities={k: round(v, 2) for k, v in eoq_results.items()},
            provenance_hash=provenance
        )

    def _calculate_eoq(
        self,
        annual_demand: float,
        order_cost: float,
        holding_cost: float
    ) -> float:
        """
        Calculate Economic Order Quantity.

        EOQ = sqrt(2 * D * S / H)
        where D = annual demand, S = order cost, H = holding cost per unit

        Args:
            annual_demand: Annual demand in units
            order_cost: Cost per order
            holding_cost: Holding cost per unit per year

        Returns:
            Economic order quantity
        """
        if holding_cost <= 0:
            return annual_demand / 12  # Monthly order as fallback

        eoq = math.sqrt(2 * annual_demand * order_cost / holding_cost)
        return eoq

    def calculate_total_cost(
        self,
        order_qty: float,
        annual_demand: float,
        order_cost: float,
        holding_cost: float,
        unit_price: float
    ) -> Dict[str, float]:
        """
        Calculate total inventory cost.

        Args:
            order_qty: Order quantity
            annual_demand: Annual demand
            order_cost: Cost per order
            holding_cost: Holding cost rate
            unit_price: Unit price

        Returns:
            Cost breakdown
        """
        num_orders = annual_demand / order_qty if order_qty > 0 else 0
        ordering_cost = num_orders * order_cost
        avg_inventory = order_qty / 2
        holding_cost_total = avg_inventory * unit_price * holding_cost
        purchase_cost = annual_demand * unit_price

        return {
            'ordering_cost': round(ordering_cost, 2),
            'holding_cost': round(holding_cost_total, 2),
            'purchase_cost': round(purchase_cost, 2),
            'total_cost': round(ordering_cost + holding_cost_total + purchase_cost, 2)
        }

    def _calculate_provenance(
        self,
        input_data: ProcurementInput,
        result: float
    ) -> str:
        """Calculate provenance hash."""
        data = {
            'inventories': input_data.fuel_inventories,
            'consumption': input_data.daily_consumption,
            'horizon': input_data.planning_horizon_days,
            'result': result
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
