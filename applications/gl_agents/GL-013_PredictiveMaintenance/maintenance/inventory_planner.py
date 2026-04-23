"""GL-013 PredictiveMaintenance - Inventory Planning Module"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib, logging, math
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PartCategory(str, Enum):
    BEARING = "bearing"
    SEAL = "seal"
    MOTOR = "motor"
    PUMP = "pump"
    FILTER = "filter"
    BELT = "belt"
    COUPLING = "coupling"
    ELECTRICAL = "electrical"
    LUBRICATION = "lubrication"
    INSTRUMENTATION = "instrumentation"

class BOMMapping(BaseModel):
    failure_mode: str
    part_numbers: List[str]
    part_descriptions: List[str]
    quantities: List[int]
    part_categories: List[PartCategory]
    lead_time_days: Dict[str, int] = Field(default_factory=dict)
    unit_costs_usd: Dict[str, float] = Field(default_factory=dict)

class PartsDemandForecast(BaseModel):
    part_number: str
    part_description: str
    category: PartCategory
    forecast_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    forecast_horizon_days: int = Field(default=90)
    expected_demand: float = Field(default=0.0, ge=0.0)
    demand_p10: float = Field(default=0.0, ge=0.0)
    demand_p50: float = Field(default=0.0, ge=0.0)
    demand_p90: float = Field(default=0.0, ge=0.0)
    current_stock: int = Field(default=0, ge=0)
    reorder_point: int = Field(default=0, ge=0)
    safety_stock: int = Field(default=0, ge=0)
    lead_time_days: int = Field(default=14, ge=0)

class ReorderProposal(BaseModel):
    proposal_id: str
    part_number: str
    part_description: str
    category: PartCategory
    proposed_quantity: int = Field(..., ge=1)
    urgency: str = Field(default="normal")
    reason: str
    expected_stockout_date: Optional[datetime] = None
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    lead_time_days: int = Field(default=14)
    order_by_date: Optional[datetime] = None
    provenance_hash: str

class InventoryPlannerConfig(BaseModel):
    forecast_horizon_days: int = Field(default=90)
    safety_stock_days: int = Field(default=14)
    service_level_target: float = Field(default=0.95)
    reorder_lead_time_buffer_days: int = Field(default=7)
    min_order_quantity: int = Field(default=1)

class InventoryPlanner:
    """Parts and inventory planner with demand forecasting from RUL distributions."""

    def __init__(self, config: Optional[InventoryPlannerConfig] = None):
        self.config = config or InventoryPlannerConfig()
        self._proposal_count = 0
        self._bom_mappings: Dict[str, BOMMapping] = {}
        logger.info("InventoryPlanner initialized")

    def register_bom_mapping(self, failure_mode: str, mapping: BOMMapping) -> None:
        """Register failure mode to BOM mapping."""
        self._bom_mappings[failure_mode] = mapping

    def forecast_parts_demand(self, rul_distributions: List[Dict], failure_modes: List[str]) -> List[PartsDemandForecast]:
        """Forecast parts demand from RUL distributions and failure modes."""
        forecasts = []
        for fm in failure_modes:
            if fm in self._bom_mappings:
                bom = self._bom_mappings[fm]
                for i, pn in enumerate(bom.part_numbers):
                    demand = self._calculate_demand_from_rul(rul_distributions, bom.quantities[i])
                    lt = bom.lead_time_days.get(pn, 14)
                    forecasts.append(PartsDemandForecast(part_number=pn, part_description=bom.part_descriptions[i], category=bom.part_categories[i], expected_demand=demand[0], demand_p10=demand[1], demand_p50=demand[2], demand_p90=demand[3], lead_time_days=lt))
        return forecasts

    def _calculate_demand_from_rul(self, rul_distributions: List[Dict], qty: int) -> Tuple[float, float, float, float]:
        """Calculate expected demand from RUL distributions."""
        if not rul_distributions:
            return (0.0, 0.0, 0.0, 0.0)
        total_prob = 0.0
        for rul in rul_distributions:
            p10 = rul.get("rul_p10_days", 30)
            prob_failure = 1.0 - math.exp(-self.config.forecast_horizon_days / max(1, p10))
            total_prob += prob_failure
        expected = total_prob * qty
        return (expected, expected * 0.5, expected, expected * 1.5)

    def calculate_safety_stock(self, lead_time_days: int, demand_std: float) -> int:
        """Calculate safety stock based on lead time and demand variability."""
        z_score = 1.65 if self.config.service_level_target >= 0.95 else 1.28
        safety_stock = z_score * demand_std * math.sqrt(lead_time_days / 30.0)
        return max(1, int(math.ceil(safety_stock)))

    def calculate_reorder_point(self, lead_time_days: int, daily_demand: float, safety_stock: int) -> int:
        """Calculate reorder point = lead time demand + safety stock."""
        lt_demand = daily_demand * lead_time_days
        return max(1, int(math.ceil(lt_demand + safety_stock)))

    def generate_reorder_proposals(self, forecasts: List[PartsDemandForecast], current_inventory: Dict[str, int]) -> List[ReorderProposal]:
        """Generate reorder proposals based on forecasted demand and current inventory."""
        proposals = []
        now = datetime.now(timezone.utc)
        for fc in forecasts:
            current_stock = current_inventory.get(fc.part_number, 0)
            daily_demand = fc.expected_demand / self.config.forecast_horizon_days
            ss = self.calculate_safety_stock(fc.lead_time_days, fc.demand_p90 - fc.demand_p10)
            rop = self.calculate_reorder_point(fc.lead_time_days, daily_demand, ss)
            if current_stock <= rop:
                qty = max(self.config.min_order_quantity, int(fc.expected_demand) - current_stock + ss)
                days_until_stockout = int(current_stock / max(0.1, daily_demand))
                urgency = "critical" if days_until_stockout < fc.lead_time_days else "normal"
                self._proposal_count += 1
                order_by = now + timedelta(days=max(0, days_until_stockout - fc.lead_time_days - self.config.reorder_lead_time_buffer_days))
                prov = hashlib.sha256(f"{fc.part_number}|{qty}|{now.isoformat()}".encode()).hexdigest()
                proposals.append(ReorderProposal(proposal_id=f"REORDER-{self._proposal_count:06d}", part_number=fc.part_number, part_description=fc.part_description, category=fc.category, proposed_quantity=qty, urgency=urgency, reason=f"Stock {current_stock} <= ROP {rop}", expected_stockout_date=now + timedelta(days=days_until_stockout), lead_time_days=fc.lead_time_days, order_by_date=order_by, provenance_hash=prov))
        return proposals
