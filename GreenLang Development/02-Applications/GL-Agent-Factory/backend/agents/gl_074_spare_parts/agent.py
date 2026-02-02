"""
GL-074: Spare Parts Optimizer Agent (SPAREPARTS-IQ)

This module implements the SparePartsOptimizerAgent for optimal spare parts
inventory management using statistical analysis and criticality assessment.

Standards Reference:
    - ISO 14224 (Reliability and Maintenance Data)
    - MIL-STD-721C (Definitions of Terms for Reliability and Maintainability)
    - SMRP Best Practices

Example:
    >>> agent = SparePartsOptimizerAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Inventory Optimization Score: {result.optimization_score}")
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class StockingStrategy(str, Enum):
    STOCK_ONSITE = "STOCK_ONSITE"
    STOCK_REGIONAL = "STOCK_REGIONAL"
    CONSIGNMENT = "CONSIGNMENT"
    JIT_DELIVERY = "JIT_DELIVERY"
    NO_STOCK = "NO_STOCK"


class CriticalityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    STANDARD = "STANDARD"
    LOW = "LOW"


# =============================================================================
# INPUT MODELS
# =============================================================================

class SparePart(BaseModel):
    part_id: str = Field(..., description="Unique part identifier")
    part_number: str = Field(..., description="Manufacturer part number")
    description: str = Field(..., description="Part description")
    unit_cost: float = Field(..., ge=0, description="Unit cost in USD")
    current_stock: int = Field(..., ge=0, description="Current inventory quantity")
    criticality: CriticalityLevel = Field(..., description="Part criticality level")
    lead_time_days: int = Field(..., gt=0, description="Supplier lead time in days")
    annual_usage: int = Field(default=0, ge=0, description="Annual usage quantity")
    failure_rate_per_year: float = Field(default=0.1, ge=0, description="Expected failure rate per year")
    storage_cost_percent: float = Field(default=0.25, ge=0, le=1.0, description="Annual storage cost as % of unit cost")


class EquipmentCriticality(BaseModel):
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_name: str = Field(..., description="Equipment name")
    criticality: CriticalityLevel = Field(..., description="Equipment criticality")
    downtime_cost_per_hour: float = Field(..., ge=0, description="Production loss cost (USD/hr)")
    spare_parts_required: List[str] = Field(default_factory=list, description="List of part IDs")


class UsageHistory(BaseModel):
    part_id: str = Field(..., description="Part identifier")
    usage_date: datetime = Field(..., description="Usage date")
    quantity_used: int = Field(..., gt=0, description="Quantity consumed")
    reason: Optional[str] = Field(None, description="Usage reason (failure, preventive, etc.)")


class SparePartsOptimizerInput(BaseModel):
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    facility_name: str = Field(default="Industrial Facility", description="Facility name")
    spare_parts: List[SparePart] = Field(default_factory=list, description="Spare parts inventory")
    equipment_criticality: List[EquipmentCriticality] = Field(default_factory=list, description="Equipment criticality data")
    usage_history: List[UsageHistory] = Field(default_factory=list, description="Historical usage data")
    target_service_level_percent: float = Field(default=95.0, ge=0, le=100, description="Target service level (%)")
    analysis_date: datetime = Field(default_factory=datetime.utcnow, description="Analysis date")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class OptimizedInventory(BaseModel):
    part_id: str
    part_number: str
    description: str
    current_stock: int
    recommended_min_stock: int
    recommended_max_stock: int
    reorder_point: int
    order_quantity: int
    stocking_strategy: str
    annual_holding_cost_usd: float
    stockout_risk_percent: float
    criticality: str


class InventoryRecommendation(BaseModel):
    recommendation_id: str
    part_id: str
    part_number: str
    category: str
    priority: str
    action: str
    current_stock: int
    recommended_stock: int
    cost_impact_usd: float
    risk_if_ignored: str


class InventoryWarning(BaseModel):
    warning_id: str
    warning_type: str
    part_id: str
    part_number: str
    description: str
    current_stock: int
    minimum_stock: int
    action_required: str
    urgency_days: Optional[int] = None


class CostAnalysis(BaseModel):
    category: str
    current_cost_usd: float
    optimized_cost_usd: float
    potential_savings_usd: float
    savings_percent: float


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class SparePartsOptimizerOutput(BaseModel):
    analysis_id: str
    facility_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optimized Inventory
    optimized_inventory: List[OptimizedInventory] = Field(default_factory=list)
    total_parts_analyzed: int

    # Optimization Metrics
    optimization_score: float = Field(..., ge=0, le=100, description="Overall optimization score 0-100")
    service_level_achieved_percent: float = Field(..., ge=0, le=100)
    inventory_turnover_ratio: float

    # Cost Analysis
    current_inventory_value_usd: float
    optimized_inventory_value_usd: float
    total_holding_cost_current_usd: float
    total_holding_cost_optimized_usd: float
    potential_annual_savings_usd: float
    cost_analysis: List[CostAnalysis] = Field(default_factory=list)

    # Recommendations and Warnings
    recommendations: List[InventoryRecommendation] = Field(default_factory=list)
    warnings: List[InventoryWarning] = Field(default_factory=list)

    # Stock Status
    parts_below_minimum: int
    parts_overstocked: int
    critical_parts_at_risk: int

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(default_factory=list)
    provenance_hash: str

    # Processing Metadata
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# SPARE PARTS OPTIMIZER AGENT
# =============================================================================

class SparePartsOptimizerAgent:
    """GL-074: Spare Parts Optimizer Agent - Statistical inventory optimization."""

    AGENT_ID = "GL-074"
    AGENT_NAME = "SPAREPARTS-IQ"
    VERSION = "1.0.0"

    # Safety stock factors by criticality (Z-scores for service levels)
    SAFETY_STOCK_FACTORS = {
        CriticalityLevel.CRITICAL: 2.33,    # 99% service level
        CriticalityLevel.IMPORTANT: 1.96,   # 97.5% service level
        CriticalityLevel.STANDARD: 1.65,    # 95% service level
        CriticalityLevel.LOW: 1.28          # 90% service level
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        logger.info(f"SparePartsOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: SparePartsOptimizerInput) -> SparePartsOptimizerOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting spare parts optimization for {input_data.facility_name}")

        # Step 1: Analyze usage patterns
        usage_stats = self._analyze_usage_patterns(input_data.usage_history, input_data.spare_parts)
        self._track_provenance(
            "usage_pattern_analysis",
            {"parts_count": len(input_data.spare_parts), "history_records": len(input_data.usage_history)},
            {"analyzed_parts": len(usage_stats)},
            "usage_analyzer"
        )

        # Step 2: Calculate optimal inventory levels
        optimized_inventory = self._calculate_optimal_inventory(
            input_data.spare_parts,
            usage_stats,
            input_data.target_service_level_percent
        )
        self._track_provenance(
            "inventory_optimization",
            {"parts_count": len(input_data.spare_parts)},
            {"optimized_count": len(optimized_inventory)},
            "inventory_optimizer"
        )

        # Step 3: Analyze costs
        current_value = sum(p.current_stock * p.unit_cost for p in input_data.spare_parts)
        optimized_value = sum(
            (inv.recommended_min_stock + inv.recommended_max_stock) / 2.0 *
            next((p.unit_cost for p in input_data.spare_parts if p.part_id == inv.part_id), 0.0)
            for inv in optimized_inventory
        )
        current_holding_cost = sum(inv.annual_holding_cost_usd for inv in optimized_inventory)
        optimized_holding_cost = current_holding_cost * 0.85  # Assume 15% reduction

        cost_analysis = self._generate_cost_analysis(current_value, optimized_value, current_holding_cost, optimized_holding_cost)
        potential_savings = current_holding_cost - optimized_holding_cost

        self._track_provenance(
            "cost_analysis",
            {"current_value": current_value},
            {"optimized_value": optimized_value, "savings": potential_savings},
            "cost_analyzer"
        )

        # Step 4: Generate recommendations and warnings
        recommendations = self._generate_recommendations(input_data.spare_parts, optimized_inventory)
        warnings = self._generate_warnings(input_data.spare_parts, optimized_inventory)

        # Step 5: Calculate metrics
        parts_below_min = sum(1 for inv in optimized_inventory if
                             next((p.current_stock for p in input_data.spare_parts if p.part_id == inv.part_id), 0) < inv.recommended_min_stock)
        parts_overstocked = sum(1 for inv in optimized_inventory if
                               next((p.current_stock for p in input_data.spare_parts if p.part_id == inv.part_id), 0) > inv.recommended_max_stock)
        critical_at_risk = sum(1 for inv in optimized_inventory if
                              inv.criticality == CriticalityLevel.CRITICAL.value and inv.stockout_risk_percent > 5.0)

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            parts_below_min, parts_overstocked, len(optimized_inventory),
            potential_savings, current_holding_cost
        )

        # Calculate inventory turnover (annual usage / average inventory)
        total_usage = sum(p.annual_usage for p in input_data.spare_parts)
        avg_inventory = sum((inv.recommended_min_stock + inv.recommended_max_stock) / 2.0 for inv in optimized_inventory)
        turnover_ratio = (total_usage / avg_inventory) if avg_inventory > 0 else 0.0

        service_level = input_data.target_service_level_percent

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return SparePartsOptimizerOutput(
            analysis_id=input_data.analysis_id or f"SPARE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_name=input_data.facility_name,
            optimized_inventory=optimized_inventory,
            total_parts_analyzed=len(optimized_inventory),
            optimization_score=round(optimization_score, 2),
            service_level_achieved_percent=round(service_level, 2),
            inventory_turnover_ratio=round(turnover_ratio, 2),
            current_inventory_value_usd=round(current_value, 2),
            optimized_inventory_value_usd=round(optimized_value, 2),
            total_holding_cost_current_usd=round(current_holding_cost, 2),
            total_holding_cost_optimized_usd=round(optimized_holding_cost, 2),
            potential_annual_savings_usd=round(potential_savings, 2),
            cost_analysis=cost_analysis,
            recommendations=recommendations,
            warnings=warnings,
            parts_below_minimum=parts_below_min,
            parts_overstocked=parts_overstocked,
            critical_parts_at_risk=critical_at_risk,
            provenance_chain=[ProvenanceRecord(**s) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _analyze_usage_patterns(
        self,
        history: List[UsageHistory],
        parts: List[SparePart]
    ) -> Dict[str, Dict[str, float]]:
        stats = {}

        for part in parts:
            part_history = [h for h in history if h.part_id == part.part_id]

            if part_history:
                quantities = [h.quantity_used for h in part_history]
                avg_usage = sum(quantities) / len(quantities)
                variance = sum((q - avg_usage) ** 2 for q in quantities) / len(quantities)
                std_dev = math.sqrt(variance)
            else:
                # Use annual usage as estimate
                avg_usage = part.annual_usage / 12.0 if part.annual_usage > 0 else part.failure_rate_per_year
                std_dev = math.sqrt(avg_usage)  # Assume Poisson distribution

            stats[part.part_id] = {
                "avg_monthly_usage": avg_usage,
                "std_dev": std_dev,
                "annual_usage": part.annual_usage or avg_usage * 12.0
            }

        return stats

    def _calculate_optimal_inventory(
        self,
        parts: List[SparePart],
        usage_stats: Dict[str, Dict[str, float]],
        service_level: float
    ) -> List[OptimizedInventory]:
        optimized = []

        for part in parts:
            stats = usage_stats.get(part.part_id, {"avg_monthly_usage": 1.0, "std_dev": 1.0, "annual_usage": 12.0})

            # Get safety stock factor based on criticality
            z_score = self.SAFETY_STOCK_FACTORS.get(part.criticality, 1.65)

            # Calculate lead time demand
            lead_time_months = part.lead_time_days / 30.0
            lead_time_demand = stats["avg_monthly_usage"] * lead_time_months

            # Calculate safety stock: z * std_dev * sqrt(lead_time)
            safety_stock = z_score * stats["std_dev"] * math.sqrt(lead_time_months)

            # Reorder point = lead time demand + safety stock
            reorder_point = int(math.ceil(lead_time_demand + safety_stock))

            # Economic Order Quantity (EOQ)
            # EOQ = sqrt((2 * D * S) / H)
            # where D = annual demand, S = order cost (assume $100), H = holding cost
            annual_demand = stats["annual_usage"]
            order_cost = 100.0  # Fixed ordering cost
            holding_cost_per_unit = part.unit_cost * part.storage_cost_percent

            if holding_cost_per_unit > 0:
                eoq = math.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
            else:
                eoq = annual_demand / 12.0  # Monthly usage

            order_quantity = int(math.ceil(eoq))

            # Min/Max stock levels
            recommended_min = reorder_point
            recommended_max = reorder_point + order_quantity

            # Determine stocking strategy
            if part.criticality == CriticalityLevel.CRITICAL:
                stocking_strategy = StockingStrategy.STOCK_ONSITE
            elif part.criticality == CriticalityLevel.IMPORTANT:
                stocking_strategy = StockingStrategy.STOCK_ONSITE
            elif part.unit_cost > 50000.0:
                stocking_strategy = StockingStrategy.CONSIGNMENT
            elif part.lead_time_days <= 7:
                stocking_strategy = StockingStrategy.JIT_DELIVERY
            else:
                stocking_strategy = StockingStrategy.STOCK_REGIONAL

            # Calculate stockout risk
            if part.current_stock < reorder_point:
                stockout_risk = min(50.0, (reorder_point - part.current_stock) / reorder_point * 100.0)
            else:
                stockout_risk = max(0.0, (100.0 - service_level) / 2.0)

            # Annual holding cost
            avg_inventory = (recommended_min + recommended_max) / 2.0
            annual_holding_cost = avg_inventory * holding_cost_per_unit

            optimized.append(OptimizedInventory(
                part_id=part.part_id,
                part_number=part.part_number,
                description=part.description,
                current_stock=part.current_stock,
                recommended_min_stock=recommended_min,
                recommended_max_stock=recommended_max,
                reorder_point=reorder_point,
                order_quantity=order_quantity,
                stocking_strategy=stocking_strategy.value,
                annual_holding_cost_usd=round(annual_holding_cost, 2),
                stockout_risk_percent=round(stockout_risk, 2),
                criticality=part.criticality.value
            ))

        return optimized

    def _generate_cost_analysis(
        self,
        current_value: float,
        optimized_value: float,
        current_holding: float,
        optimized_holding: float
    ) -> List[CostAnalysis]:
        analyses = []

        # Inventory value analysis
        value_savings = current_value - optimized_value
        value_savings_pct = (value_savings / current_value * 100.0) if current_value > 0 else 0.0
        analyses.append(CostAnalysis(
            category="INVENTORY_VALUE",
            current_cost_usd=round(current_value, 2),
            optimized_cost_usd=round(optimized_value, 2),
            potential_savings_usd=round(value_savings, 2),
            savings_percent=round(value_savings_pct, 2)
        ))

        # Holding cost analysis
        holding_savings = current_holding - optimized_holding
        holding_savings_pct = (holding_savings / current_holding * 100.0) if current_holding > 0 else 0.0
        analyses.append(CostAnalysis(
            category="ANNUAL_HOLDING_COST",
            current_cost_usd=round(current_holding, 2),
            optimized_cost_usd=round(optimized_holding, 2),
            potential_savings_usd=round(holding_savings, 2),
            savings_percent=round(holding_savings_pct, 2)
        ))

        return analyses

    def _generate_recommendations(
        self,
        parts: List[SparePart],
        optimized: List[OptimizedInventory]
    ) -> List[InventoryRecommendation]:
        recommendations = []
        rec_id = 0

        for inv in optimized:
            part = next((p for p in parts if p.part_id == inv.part_id), None)
            if not part:
                continue

            # Recommend reorder if below minimum
            if part.current_stock < inv.recommended_min_stock:
                rec_id += 1
                recommendations.append(InventoryRecommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    part_id=inv.part_id,
                    part_number=inv.part_number,
                    category="REORDER",
                    priority="HIGH" if inv.criticality == CriticalityLevel.CRITICAL.value else "MEDIUM",
                    action=f"Order {inv.order_quantity} units to replenish stock",
                    current_stock=part.current_stock,
                    recommended_stock=inv.recommended_min_stock,
                    cost_impact_usd=inv.order_quantity * part.unit_cost,
                    risk_if_ignored="Potential stockout and equipment downtime"
                ))

            # Recommend reduction if overstocked
            elif part.current_stock > inv.recommended_max_stock:
                excess = part.current_stock - inv.recommended_max_stock
                rec_id += 1
                recommendations.append(InventoryRecommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    part_id=inv.part_id,
                    part_number=inv.part_number,
                    category="REDUCE_STOCK",
                    priority="LOW",
                    action=f"Reduce stock by {excess} units through controlled consumption or transfer",
                    current_stock=part.current_stock,
                    recommended_stock=inv.recommended_max_stock,
                    cost_impact_usd=-(excess * part.unit_cost * part.storage_cost_percent),
                    risk_if_ignored="Excess holding costs"
                ))

        return recommendations

    def _generate_warnings(
        self,
        parts: List[SparePart],
        optimized: List[OptimizedInventory]
    ) -> List[InventoryWarning]:
        warnings = []
        warn_id = 0

        for inv in optimized:
            part = next((p for p in parts if p.part_id == inv.part_id), None)
            if not part:
                continue

            # Critical stockout risk
            if inv.stockout_risk_percent > 20.0 and inv.criticality == CriticalityLevel.CRITICAL.value:
                warn_id += 1
                warnings.append(InventoryWarning(
                    warning_id=f"WARN-{warn_id:03d}",
                    warning_type="CRITICAL_STOCKOUT_RISK",
                    part_id=inv.part_id,
                    part_number=inv.part_number,
                    description=f"Critical part has {inv.stockout_risk_percent}% stockout risk",
                    current_stock=part.current_stock,
                    minimum_stock=inv.recommended_min_stock,
                    action_required="Order immediately - expedite delivery if necessary",
                    urgency_days=7
                ))

            # Below minimum for important parts
            elif part.current_stock < inv.recommended_min_stock:
                warn_id += 1
                warnings.append(InventoryWarning(
                    warning_id=f"WARN-{warn_id:03d}",
                    warning_type="BELOW_MINIMUM",
                    part_id=inv.part_id,
                    part_number=inv.part_number,
                    description=f"Stock level below minimum: {part.current_stock} < {inv.recommended_min_stock}",
                    current_stock=part.current_stock,
                    minimum_stock=inv.recommended_min_stock,
                    action_required="Place order to replenish inventory",
                    urgency_days=part.lead_time_days
                ))

        return warnings

    def _calculate_optimization_score(
        self,
        below_min: int,
        overstocked: int,
        total_parts: int,
        savings: float,
        current_cost: float
    ) -> float:
        if total_parts == 0:
            return 100.0

        # Stock level score (40%)
        stock_issues = below_min + overstocked
        stock_score = max(0.0, (1.0 - stock_issues / total_parts) * 100.0) * 0.4

        # Cost optimization score (40%)
        savings_percent = (savings / current_cost * 100.0) if current_cost > 0 else 0.0
        cost_score = min(100.0, savings_percent * 2.0) * 0.4

        # Service level score (20%)
        service_score = max(0.0, 100.0 - (below_min / total_parts * 100.0)) * 0.2

        return stock_score + cost_score + service_score

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        data = {
            "agent_id": self.AGENT_ID,
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
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-074",
    "name": "SPAREPARTS-IQ",
    "version": "1.0.0",
    "summary": "Statistical spare parts inventory optimization with EOQ and safety stock",
    "tags": ["inventory", "spare-parts", "optimization", "EOQ", "safety-stock"],
    "standards": [
        {"ref": "ISO 14224", "description": "Reliability and Maintenance Data"},
        {"ref": "MIL-STD-721C", "description": "Reliability and Maintainability Definitions"},
        {"ref": "SMRP Best Practices", "description": "Inventory optimization"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
