# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

class PartCriticality(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class StockStatus(Enum):
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    ON_ORDER = "on_order"

@dataclass
class SparePart:
    part_id: str
    part_number: str
    description: str
    criticality: PartCriticality
    unit_cost: float
    lead_time_days: int
    minimum_order_quantity: int = 1
    compatible_assets: List[str] = field(default_factory=list)
    supplier_ids: List[str] = field(default_factory=list)

@dataclass
class InventoryLevel:
    part_id: str
    quantity_on_hand: int
    quantity_reserved: int
    quantity_on_order: int
    reorder_point: int
    safety_stock: int
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def available_quantity(self) -> int:
        return self.quantity_on_hand - self.quantity_reserved
    
    @property
    def status(self) -> StockStatus:
        if self.available_quantity <= 0:
            return StockStatus.OUT_OF_STOCK
        elif self.available_quantity <= self.safety_stock:
            return StockStatus.LOW_STOCK
        return StockStatus.IN_STOCK

@dataclass
class ReplenishmentOrder:
    order_id: str
    part_id: str
    quantity: int
    unit_cost: float
    supplier_id: str
    order_date: datetime
    expected_delivery: datetime
    status: str = "pending"
    
    @property
    def total_cost(self) -> float:
        return self.quantity * self.unit_cost

@dataclass
class InventoryPlanResult:
    recommendations: List[ReplenishmentOrder]
    total_investment: float
    service_level_impact: float
    risk_mitigation: float
    provenance_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class InventoryConfig:
    target_service_level: float = 0.95
    holding_cost_rate: float = 0.25
    stockout_cost_multiplier: float = 10.0
    planning_horizon_days: int = 90
    review_period_days: int = 7

class InventoryPlanner:
    def __init__(self, config: Optional[InventoryConfig] = None):
        self.config = config or InventoryConfig()
        self._parts: Dict[str, SparePart] = {}
        self._inventory: Dict[str, InventoryLevel] = {}
        
    def register_part(self, part: SparePart) -> None:
        self._parts[part.part_id] = part
        
    def update_inventory(self, level: InventoryLevel) -> None:
        self._inventory[level.part_id] = level
        
    def calculate_safety_stock(self, part_id: str, demand_std: float = 10.0) -> int:
        part = self._parts.get(part_id)
        if not part:
            return 0
        z = 1.65  # 95% service level
        lead_time_factor = np.sqrt(part.lead_time_days / self.config.review_period_days)
        return max(1, int(np.ceil(z * demand_std * lead_time_factor)))
    
    def calculate_reorder_point(self, part_id: str, avg_daily_demand: float = 1.0) -> int:
        part = self._parts.get(part_id)
        if not part:
            return 0
        lead_time_demand = avg_daily_demand * part.lead_time_days
        safety_stock = self.calculate_safety_stock(part_id)
        return int(np.ceil(lead_time_demand + safety_stock))
    
    def calculate_eoq(self, part_id: str, annual_demand: float) -> int:
        part = self._parts.get(part_id)
        if not part:
            return 1
        ordering_cost = 50.0
        holding_cost = part.unit_cost * self.config.holding_cost_rate
        if holding_cost <= 0:
            return part.minimum_order_quantity
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
        return max(part.minimum_order_quantity, int(np.ceil(eoq)))
    
    def generate_replenishment_plan(self, part_demands: Dict[str, int]) -> InventoryPlanResult:
        recommendations = []
        total_investment = 0.0
        
        for part_id, demand in part_demands.items():
            inventory = self._inventory.get(part_id)
            part = self._parts.get(part_id)
            if not part:
                continue
            if not inventory:
                order_qty = max(demand, part.minimum_order_quantity)
            else:
                reorder_point = self.calculate_reorder_point(part_id)
                projected = inventory.available_quantity - demand
                if projected < reorder_point:
                    order_qty = self.calculate_eoq(part_id, demand * 12)
                else:
                    continue
            
            order = ReplenishmentOrder(
                order_id=hashlib.sha256(f"{part_id}{datetime.utcnow()}".encode()).hexdigest()[:12],
                part_id=part_id,
                quantity=order_qty,
                unit_cost=part.unit_cost,
                supplier_id=part.supplier_ids[0] if part.supplier_ids else "default",
                order_date=datetime.utcnow(),
                expected_delivery=datetime.utcnow() + timedelta(days=part.lead_time_days),
            )
            recommendations.append(order)
            total_investment += order.total_cost
        
        provenance = hashlib.sha256(f"{len(recommendations)}:{total_investment}".encode()).hexdigest()
        
        return InventoryPlanResult(
            recommendations=recommendations,
            total_investment=total_investment,
            service_level_impact=self.config.target_service_level if recommendations else 0.0,
            risk_mitigation=len(recommendations) / max(1, len(part_demands)),
            provenance_hash=provenance,
        )
    
    def get_critical_alerts(self) -> List[Dict]:
        alerts = []
        for part_id, inventory in self._inventory.items():
            part = self._parts.get(part_id)
            if not part:
                continue
            if inventory.status == StockStatus.OUT_OF_STOCK:
                alerts.append({"part_id": part_id, "severity": "critical", "criticality": part.criticality.name})
            elif inventory.status == StockStatus.LOW_STOCK and part.criticality in [PartCriticality.CRITICAL, PartCriticality.HIGH]:
                alerts.append({"part_id": part_id, "severity": "warning", "available": inventory.available_quantity})
        return alerts
