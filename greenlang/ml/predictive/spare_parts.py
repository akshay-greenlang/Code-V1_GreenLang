# -*- coding: utf-8 -*-
"""
Spare Parts Optimization Module (TASK-109)

Zero-Hallucination Inventory Optimization for Maintenance Parts

This module implements deterministic spare parts inventory optimization
for process heat equipment including:
- Inventory level optimization
- Lead time prediction
- Demand forecasting
- Criticality-based stocking strategy
- Economic Order Quantity (EOQ) calculations
- Safety stock calculations

Integrates with GL-013 PredictiveMaint Module for maintenance scheduling
and parts demand forecasting.

References:
    - MIL-STD-1629A: Failure Mode and Effects Analysis
    - IEC 60812: Analysis techniques for system reliability
    - Harris (1913): Economic Order Quantity model
    - Silver-Meal heuristic for dynamic lot sizing

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging
import math

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class PartCriticality(str, Enum):
    """Part criticality classification based on operational impact."""
    CRITICAL = "critical"      # Immediate production stop if unavailable
    ESSENTIAL = "essential"    # Production impacted within hours
    IMPORTANT = "important"    # Production impacted within days
    STANDARD = "standard"      # No immediate production impact
    NON_CRITICAL = "non_critical"  # Convenience items


class EquipmentType(str, Enum):
    """Equipment types for process heat systems."""
    BOILER = "boiler"
    HEAT_EXCHANGER = "heat_exchanger"
    PUMP = "pump"
    FAN = "fan"
    VALVE = "valve"
    BURNER = "burner"
    CONTROL_SYSTEM = "control_system"
    INSTRUMENTATION = "instrumentation"
    REFRACTORY = "refractory"
    PIPING = "piping"


class StockingStrategy(str, Enum):
    """Inventory stocking strategies."""
    STOCK = "stock"              # Always keep on hand
    NON_STOCK = "non_stock"      # Order when needed
    CONSIGNMENT = "consignment"  # Vendor-managed inventory
    INSURANCE = "insurance"      # Low probability, high impact spares


class DemandPattern(str, Enum):
    """Demand pattern classifications."""
    STEADY = "steady"        # Consistent demand
    SEASONAL = "seasonal"    # Seasonal variation
    INTERMITTENT = "intermittent"  # Sporadic demand (lumpy)
    SLOW_MOVING = "slow_moving"    # Very infrequent demand
    OBSOLETE = "obsolete"    # Declining/ending demand


# ============================================================================
# Pydantic Models
# ============================================================================

class PartUsageRecord(BaseModel):
    """Historical part usage record."""

    date: datetime = Field(..., description="Usage date")
    quantity: int = Field(..., ge=0, description="Quantity used")
    reason: str = Field(default="maintenance", description="Usage reason")
    equipment_id: Optional[str] = Field(None, description="Equipment identifier")
    work_order: Optional[str] = Field(None, description="Work order reference")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SparePartInput(BaseModel):
    """Input data for spare part optimization."""

    part_id: str = Field(..., min_length=1, description="Part identifier")
    part_name: str = Field(..., description="Part description")
    equipment_type: EquipmentType = Field(..., description="Associated equipment type")
    criticality: PartCriticality = Field(
        default=PartCriticality.STANDARD,
        description="Part criticality level"
    )

    # Cost parameters
    unit_cost: float = Field(..., gt=0, description="Unit cost ($)")
    ordering_cost: float = Field(default=50.0, gt=0, description="Cost per order ($)")
    holding_cost_rate: float = Field(
        default=0.25,
        ge=0.05,
        le=0.50,
        description="Annual holding cost as fraction of unit cost"
    )
    stockout_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Cost per stockout event ($)"
    )

    # Inventory parameters
    current_stock: int = Field(default=0, ge=0, description="Current inventory level")
    reorder_point: Optional[int] = Field(None, ge=0, description="Current reorder point")
    max_stock: Optional[int] = Field(None, ge=0, description="Maximum stock level")

    # Lead time parameters
    lead_time_days: float = Field(default=14.0, gt=0, description="Average lead time (days)")
    lead_time_std: Optional[float] = Field(None, ge=0, description="Lead time std dev (days)")

    # Usage history
    usage_history: List[PartUsageRecord] = Field(
        default_factory=list,
        description="Historical usage records"
    )
    annual_usage: Optional[float] = Field(None, ge=0, description="Annual usage if known")

    # Equipment context
    equipment_count: int = Field(default=1, ge=1, description="Number of units using this part")
    mtbf_hours: Optional[float] = Field(None, gt=0, description="Mean Time Between Failures")

    @validator('stockout_cost', pre=True, always=True)
    def set_default_stockout_cost(cls, v, values):
        """Set default stockout cost based on criticality."""
        if v is not None:
            return v
        criticality = values.get('criticality', PartCriticality.STANDARD)
        unit_cost = values.get('unit_cost', 100)
        multipliers = {
            PartCriticality.CRITICAL: 100,
            PartCriticality.ESSENTIAL: 50,
            PartCriticality.IMPORTANT: 20,
            PartCriticality.STANDARD: 5,
            PartCriticality.NON_CRITICAL: 1
        }
        return unit_cost * multipliers.get(criticality, 5)


class EOQResult(BaseModel):
    """Economic Order Quantity calculation result."""

    eoq: int = Field(..., ge=1, description="Economic Order Quantity")
    orders_per_year: float = Field(..., ge=0, description="Number of orders per year")
    order_cycle_days: float = Field(..., ge=0, description="Days between orders")
    total_annual_cost: float = Field(..., ge=0, description="Total annual inventory cost")
    ordering_cost_annual: float = Field(..., ge=0, description="Annual ordering cost")
    holding_cost_annual: float = Field(..., ge=0, description="Annual holding cost")


class SafetyStockResult(BaseModel):
    """Safety stock calculation result."""

    safety_stock: int = Field(..., ge=0, description="Safety stock units")
    reorder_point: int = Field(..., ge=0, description="Reorder point")
    service_level: float = Field(..., ge=0, le=1, description="Target service level")
    stockout_risk: float = Field(..., ge=0, le=1, description="Probability of stockout")
    days_of_cover: float = Field(..., ge=0, description="Days of safety stock cover")


class DemandForecastResult(BaseModel):
    """Demand forecast result."""

    demand_pattern: DemandPattern = Field(..., description="Identified demand pattern")
    avg_daily_demand: float = Field(..., ge=0, description="Average daily demand")
    avg_monthly_demand: float = Field(..., ge=0, description="Average monthly demand")
    annual_demand: float = Field(..., ge=0, description="Projected annual demand")
    demand_std: float = Field(..., ge=0, description="Demand standard deviation")
    cv: float = Field(..., ge=0, description="Coefficient of variation")
    forecast_next_30d: float = Field(..., ge=0, description="30-day demand forecast")
    forecast_next_90d: float = Field(..., ge=0, description="90-day demand forecast")
    forecast_confidence: float = Field(..., ge=0, le=1, description="Forecast confidence")


class SparePartOptimizationOutput(BaseModel):
    """Output from spare parts optimization."""

    part_id: str = Field(..., description="Part identifier")
    part_name: str = Field(..., description="Part description")
    equipment_type: str = Field(..., description="Equipment type")
    criticality: str = Field(..., description="Criticality level")

    # Recommended strategy
    stocking_strategy: StockingStrategy = Field(..., description="Recommended stocking strategy")
    strategy_rationale: str = Field(..., description="Strategy rationale")

    # Inventory parameters
    eoq_result: EOQResult = Field(..., description="EOQ calculation")
    safety_stock_result: SafetyStockResult = Field(..., description="Safety stock calculation")

    # Current vs recommended
    current_stock: int = Field(..., description="Current stock level")
    recommended_stock: int = Field(..., description="Recommended stock level")
    stock_adjustment: int = Field(..., description="Adjustment needed (+/-)")

    # Demand analysis
    demand_forecast: DemandForecastResult = Field(..., description="Demand forecast")

    # Cost analysis
    current_annual_cost: float = Field(..., description="Current annual inventory cost")
    optimized_annual_cost: float = Field(..., description="Optimized annual cost")
    annual_savings: float = Field(..., description="Potential annual savings")
    savings_pct: float = Field(..., description="Savings percentage")

    # Lead time analysis
    lead_time_days: float = Field(..., description="Lead time in days")
    lead_time_variability: str = Field(..., description="Lead time variability assessment")

    # Risk analysis
    stockout_risk_current: float = Field(..., description="Current stockout probability")
    stockout_risk_optimized: float = Field(..., description="Optimized stockout probability")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# Core Optimization Engine
# ============================================================================

class SparePartsOptimizer:
    """
    Spare Parts Inventory Optimization Engine.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic mathematical formulas
    - No LLM involvement in optimization calculations
    - Complete provenance tracking
    - Based on established inventory theory (Harris EOQ, etc.)

    Capabilities:
    - Economic Order Quantity (EOQ) calculation
    - Safety stock optimization
    - Demand pattern analysis
    - Lead time prediction
    - Criticality-based stocking strategies
    - Total cost optimization

    Integration:
    - GL-013 PredictiveMaint: Failure predictions drive demand forecasts
    - GL-011 Fuel Module: Energy cost of stockouts

    Example:
        >>> optimizer = SparePartsOptimizer()
        >>> result = optimizer.optimize(input_data)
        >>> print(f"EOQ: {result.eoq_result.eoq} units")
        >>> print(f"Annual savings: ${result.annual_savings:.2f}")
    """

    # Service level z-values (normal distribution)
    Z_VALUES = {
        0.90: 1.282,
        0.95: 1.645,
        0.99: 2.326,
        0.999: 3.090
    }

    # Criticality to service level mapping
    CRITICALITY_SERVICE_LEVELS = {
        PartCriticality.CRITICAL: 0.99,
        PartCriticality.ESSENTIAL: 0.95,
        PartCriticality.IMPORTANT: 0.90,
        PartCriticality.STANDARD: 0.85,
        PartCriticality.NON_CRITICAL: 0.80
    }

    # CV thresholds for demand pattern classification
    CV_THRESHOLDS = {
        "steady": 0.3,
        "seasonal": 0.5,
        "intermittent": 1.0
    }

    def __init__(self, precision: int = 4):
        """
        Initialize spare parts optimizer.

        Args:
            precision: Decimal precision for outputs
        """
        self.precision = precision
        logger.info("SparePartsOptimizer initialized")

    def _apply_precision(self, value: float) -> float:
        """Apply precision rounding."""
        if self.precision == 0:
            return round(value)
        return round(value, self.precision)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "SparePartsOptimizer",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _get_z_value(self, service_level: float) -> float:
        """Get z-value for service level."""
        if service_level >= 0.999:
            return 3.090
        elif service_level >= 0.99:
            return 2.326
        elif service_level >= 0.95:
            return 1.645
        elif service_level >= 0.90:
            return 1.282
        elif service_level >= 0.85:
            return 1.036
        elif service_level >= 0.80:
            return 0.842
        else:
            return 0.524

    def _analyze_demand(
        self,
        usage_history: List[PartUsageRecord],
        annual_usage: Optional[float],
        mtbf_hours: Optional[float],
        equipment_count: int
    ) -> DemandForecastResult:
        """
        Analyze demand patterns and generate forecast.

        ZERO-HALLUCINATION: Statistical demand analysis.

        Args:
            usage_history: Historical usage records
            annual_usage: Known annual usage if available
            mtbf_hours: Mean time between failures
            equipment_count: Number of equipment units

        Returns:
            DemandForecastResult with analysis
        """
        if annual_usage is not None and annual_usage > 0:
            # Use provided annual usage
            daily_demand = annual_usage / 365
            monthly_demand = annual_usage / 12
            demand_std = daily_demand * 0.5  # Assume 50% CV
            cv = 0.5
            pattern = DemandPattern.STEADY
            confidence = 0.9

        elif len(usage_history) >= 12:
            # Calculate from history
            monthly_usage = {}
            for record in usage_history:
                key = (record.date.year, record.date.month)
                monthly_usage[key] = monthly_usage.get(key, 0) + record.quantity

            if monthly_usage:
                values = list(monthly_usage.values())
                monthly_demand = sum(values) / len(values)
                daily_demand = monthly_demand / 30

                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((v - mean)**2 for v in values) / len(values)
                    demand_std = math.sqrt(variance) / 30  # Daily std
                    cv = math.sqrt(variance) / mean if mean > 0 else 0
                else:
                    demand_std = daily_demand * 0.5
                    cv = 0.5
            else:
                daily_demand = 0.01
                monthly_demand = 0.3
                demand_std = 0.01
                cv = 1.0

            # Classify demand pattern
            if cv < self.CV_THRESHOLDS["steady"]:
                pattern = DemandPattern.STEADY
            elif cv < self.CV_THRESHOLDS["seasonal"]:
                pattern = DemandPattern.SEASONAL
            elif cv < self.CV_THRESHOLDS["intermittent"]:
                pattern = DemandPattern.INTERMITTENT
            else:
                pattern = DemandPattern.SLOW_MOVING

            confidence = min(0.95, 0.5 + 0.05 * len(values))

        elif mtbf_hours is not None and mtbf_hours > 0:
            # Estimate from MTBF
            annual_failures = (8760 / mtbf_hours) * equipment_count
            daily_demand = annual_failures / 365
            monthly_demand = annual_failures / 12
            demand_std = daily_demand * 0.7  # Higher uncertainty
            cv = 0.7
            pattern = DemandPattern.INTERMITTENT
            confidence = 0.6

        else:
            # Default low demand
            daily_demand = 0.01
            monthly_demand = 0.3
            demand_std = 0.02
            cv = 1.0
            pattern = DemandPattern.SLOW_MOVING
            confidence = 0.3

        annual_demand = daily_demand * 365

        return DemandForecastResult(
            demand_pattern=pattern,
            avg_daily_demand=self._apply_precision(daily_demand),
            avg_monthly_demand=self._apply_precision(monthly_demand),
            annual_demand=self._apply_precision(annual_demand),
            demand_std=self._apply_precision(demand_std),
            cv=self._apply_precision(cv),
            forecast_next_30d=self._apply_precision(daily_demand * 30),
            forecast_next_90d=self._apply_precision(daily_demand * 90),
            forecast_confidence=self._apply_precision(confidence)
        )

    def _calculate_eoq(
        self,
        annual_demand: float,
        unit_cost: float,
        ordering_cost: float,
        holding_cost_rate: float
    ) -> EOQResult:
        """
        Calculate Economic Order Quantity.

        ZERO-HALLUCINATION: Harris EOQ Formula (1913).

        EOQ = sqrt(2 * D * S / H)

        Where:
        - D = Annual demand
        - S = Ordering cost per order
        - H = Annual holding cost per unit = unit_cost * holding_rate

        Args:
            annual_demand: Annual demand in units
            unit_cost: Cost per unit
            ordering_cost: Cost per order
            holding_cost_rate: Holding cost as fraction of unit cost

        Returns:
            EOQResult with calculations
        """
        if annual_demand <= 0:
            # No demand - minimum order
            return EOQResult(
                eoq=1,
                orders_per_year=0,
                order_cycle_days=365,
                total_annual_cost=0,
                ordering_cost_annual=0,
                holding_cost_annual=0
            )

        holding_cost = unit_cost * holding_cost_rate

        # EOQ formula
        eoq_raw = math.sqrt(2 * annual_demand * ordering_cost / holding_cost)
        eoq = max(1, round(eoq_raw))

        # Orders per year
        orders_per_year = annual_demand / eoq

        # Order cycle
        order_cycle_days = 365 / orders_per_year if orders_per_year > 0 else 365

        # Costs
        ordering_cost_annual = orders_per_year * ordering_cost
        holding_cost_annual = (eoq / 2) * holding_cost
        total_annual_cost = ordering_cost_annual + holding_cost_annual

        return EOQResult(
            eoq=eoq,
            orders_per_year=self._apply_precision(orders_per_year),
            order_cycle_days=self._apply_precision(order_cycle_days),
            total_annual_cost=self._apply_precision(total_annual_cost),
            ordering_cost_annual=self._apply_precision(ordering_cost_annual),
            holding_cost_annual=self._apply_precision(holding_cost_annual)
        )

    def _calculate_safety_stock(
        self,
        daily_demand: float,
        demand_std: float,
        lead_time_days: float,
        lead_time_std: Optional[float],
        service_level: float
    ) -> SafetyStockResult:
        """
        Calculate safety stock and reorder point.

        ZERO-HALLUCINATION: Statistical safety stock calculation.

        SS = z * sqrt(LT * sigma_d^2 + d^2 * sigma_LT^2)

        Where:
        - z = Service level z-value
        - LT = Lead time
        - sigma_d = Demand std dev
        - d = Average demand
        - sigma_LT = Lead time std dev

        Args:
            daily_demand: Average daily demand
            demand_std: Daily demand standard deviation
            lead_time_days: Average lead time
            lead_time_std: Lead time standard deviation
            service_level: Target service level

        Returns:
            SafetyStockResult
        """
        z = self._get_z_value(service_level)

        # Lead time std (default to 20% of mean if not provided)
        lt_std = lead_time_std if lead_time_std is not None else lead_time_days * 0.2

        # Combined standard deviation during lead time
        demand_variance_lt = lead_time_days * (demand_std ** 2)
        lt_variance_demand = (daily_demand ** 2) * (lt_std ** 2)
        combined_std = math.sqrt(demand_variance_lt + lt_variance_demand)

        # Safety stock
        safety_stock_raw = z * combined_std
        safety_stock = max(0, math.ceil(safety_stock_raw))

        # Reorder point = Lead time demand + Safety stock
        lead_time_demand = daily_demand * lead_time_days
        reorder_point = max(0, math.ceil(lead_time_demand + safety_stock))

        # Stockout risk (complement of service level)
        stockout_risk = 1 - service_level

        # Days of cover
        days_of_cover = safety_stock / daily_demand if daily_demand > 0 else 365

        return SafetyStockResult(
            safety_stock=safety_stock,
            reorder_point=reorder_point,
            service_level=service_level,
            stockout_risk=self._apply_precision(stockout_risk),
            days_of_cover=self._apply_precision(days_of_cover)
        )

    def _determine_stocking_strategy(
        self,
        criticality: PartCriticality,
        demand_pattern: DemandPattern,
        unit_cost: float,
        annual_demand: float,
        lead_time_days: float
    ) -> Tuple[StockingStrategy, str]:
        """
        Determine optimal stocking strategy.

        Based on ABC-VED matrix analysis.

        Args:
            criticality: Part criticality
            demand_pattern: Demand pattern
            unit_cost: Unit cost
            annual_demand: Annual demand
            lead_time_days: Lead time

        Returns:
            Tuple of (strategy, rationale)
        """
        annual_value = unit_cost * annual_demand

        # Critical items - always stock
        if criticality == PartCriticality.CRITICAL:
            return (
                StockingStrategy.STOCK,
                "Critical part - production impact justifies holding inventory"
            )

        # High value, low demand - insurance spares
        if annual_value > 10000 and annual_demand < 1:
            if criticality in [PartCriticality.CRITICAL, PartCriticality.ESSENTIAL]:
                return (
                    StockingStrategy.INSURANCE,
                    "High-value insurance spare - critical for risk mitigation"
                )
            else:
                return (
                    StockingStrategy.NON_STOCK,
                    "High-value, low-demand, non-critical - order when needed"
                )

        # Long lead time items
        if lead_time_days > 60 and criticality != PartCriticality.NON_CRITICAL:
            return (
                StockingStrategy.STOCK,
                f"Long lead time ({lead_time_days} days) requires stocking"
            )

        # High turnover items - consider consignment
        if annual_demand > 50 and unit_cost > 100:
            return (
                StockingStrategy.CONSIGNMENT,
                "High-volume, high-value - consignment reduces holding cost"
            )

        # Slow moving non-critical
        if demand_pattern in [DemandPattern.SLOW_MOVING, DemandPattern.OBSOLETE]:
            if criticality in [PartCriticality.STANDARD, PartCriticality.NON_CRITICAL]:
                return (
                    StockingStrategy.NON_STOCK,
                    "Slow-moving non-critical item - order as needed"
                )

        # Default - stock based on criticality
        if criticality in [PartCriticality.CRITICAL, PartCriticality.ESSENTIAL, PartCriticality.IMPORTANT]:
            return (
                StockingStrategy.STOCK,
                f"Stock based on {criticality.value} criticality classification"
            )

        return (
            StockingStrategy.NON_STOCK,
            "Standard/non-critical with adequate lead time - order when needed"
        )

    def _calculate_current_cost(
        self,
        current_stock: int,
        unit_cost: float,
        holding_cost_rate: float,
        annual_demand: float,
        ordering_cost: float
    ) -> float:
        """Calculate current annual inventory cost."""
        # Assume 4 orders per year if no EOQ
        assumed_orders = max(1, annual_demand / max(1, current_stock / 4))

        holding_cost = current_stock * unit_cost * holding_cost_rate
        order_cost = assumed_orders * ordering_cost

        return holding_cost + order_cost

    def _generate_recommendations(
        self,
        input_data: SparePartInput,
        strategy: StockingStrategy,
        eoq: EOQResult,
        safety: SafetyStockResult,
        demand: DemandForecastResult,
        stock_adjustment: int
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Stock level recommendations
        if stock_adjustment > 0:
            recommendations.append(
                f"INCREASE stock by {stock_adjustment} units to reach optimal level of "
                f"{input_data.current_stock + stock_adjustment}"
            )
        elif stock_adjustment < 0:
            recommendations.append(
                f"REDUCE excess stock by {abs(stock_adjustment)} units - consider "
                f"using before reordering"
            )

        # Reorder point
        recommendations.append(
            f"Set reorder point to {safety.reorder_point} units "
            f"(triggers order when stock reaches this level)"
        )

        # Order quantity
        recommendations.append(
            f"Order {eoq.eoq} units per order (EOQ) - expected "
            f"{eoq.orders_per_year:.1f} orders/year"
        )

        # Strategy-specific recommendations
        if strategy == StockingStrategy.CONSIGNMENT:
            recommendations.append(
                "Consider vendor-managed inventory (consignment) to reduce holding costs"
            )
        elif strategy == StockingStrategy.INSURANCE:
            recommendations.append(
                "Insurance spare - review annually and consider equipment upgrade economics"
            )
        elif strategy == StockingStrategy.NON_STOCK:
            recommendations.append(
                "Non-stock item - establish blanket PO or framework agreement with supplier"
            )

        # Lead time recommendations
        if input_data.lead_time_days > 30:
            recommendations.append(
                f"Long lead time ({input_data.lead_time_days} days) - "
                f"consider alternative suppliers or increasing safety stock"
            )

        # Demand pattern recommendations
        if demand.demand_pattern == DemandPattern.INTERMITTENT:
            recommendations.append(
                "Intermittent demand - monitor actual usage vs. forecast closely"
            )
        elif demand.demand_pattern == DemandPattern.SEASONAL:
            recommendations.append(
                "Seasonal demand pattern - adjust stock levels before peak periods"
            )

        # Criticality recommendations
        if input_data.criticality == PartCriticality.CRITICAL:
            recommendations.append(
                "Critical part - ensure backup supplier identified and qualified"
            )

        return recommendations

    def optimize(self, input_data: SparePartInput) -> SparePartOptimizationOutput:
        """
        Optimize spare part inventory.

        ZERO-HALLUCINATION: All calculations use deterministic
        inventory optimization formulas.

        Args:
            input_data: Validated part data and parameters

        Returns:
            SparePartOptimizationOutput with recommendations

        Example:
            >>> optimizer = SparePartsOptimizer()
            >>> result = optimizer.optimize(input_data)
            >>> print(f"Recommended stock: {result.recommended_stock}")
        """
        start_time = datetime.now()

        logger.info(
            f"Optimizing spare part: {input_data.part_id} ({input_data.part_name}), "
            f"criticality: {input_data.criticality.value}"
        )

        # Step 1: Analyze demand
        demand = self._analyze_demand(
            input_data.usage_history,
            input_data.annual_usage,
            input_data.mtbf_hours,
            input_data.equipment_count
        )

        # Step 2: Calculate EOQ
        eoq = self._calculate_eoq(
            demand.annual_demand,
            input_data.unit_cost,
            input_data.ordering_cost,
            input_data.holding_cost_rate
        )

        # Step 3: Calculate safety stock
        service_level = self.CRITICALITY_SERVICE_LEVELS.get(
            input_data.criticality, 0.90
        )
        safety = self._calculate_safety_stock(
            demand.avg_daily_demand,
            demand.demand_std,
            input_data.lead_time_days,
            input_data.lead_time_std,
            service_level
        )

        # Step 4: Determine stocking strategy
        strategy, rationale = self._determine_stocking_strategy(
            input_data.criticality,
            demand.demand_pattern,
            input_data.unit_cost,
            demand.annual_demand,
            input_data.lead_time_days
        )

        # Step 5: Calculate recommended stock level
        if strategy in [StockingStrategy.STOCK, StockingStrategy.INSURANCE]:
            # Target = Safety stock + (EOQ / 2) for average position
            recommended_stock = safety.safety_stock + (eoq.eoq // 2)
        elif strategy == StockingStrategy.CONSIGNMENT:
            # Consignment - keep minimal stock
            recommended_stock = max(1, safety.safety_stock // 2)
        else:
            # Non-stock
            recommended_stock = 0

        stock_adjustment = recommended_stock - input_data.current_stock

        # Step 6: Calculate costs
        current_cost = self._calculate_current_cost(
            input_data.current_stock,
            input_data.unit_cost,
            input_data.holding_cost_rate,
            demand.annual_demand,
            input_data.ordering_cost
        )

        optimized_cost = eoq.total_annual_cost
        annual_savings = current_cost - optimized_cost
        savings_pct = (annual_savings / current_cost * 100) if current_cost > 0 else 0

        # Step 7: Lead time variability assessment
        if input_data.lead_time_std is not None:
            lt_cv = input_data.lead_time_std / input_data.lead_time_days if input_data.lead_time_days > 0 else 0
            if lt_cv < 0.1:
                lt_variability = "low"
            elif lt_cv < 0.3:
                lt_variability = "moderate"
            else:
                lt_variability = "high"
        else:
            lt_variability = "unknown - estimate used"

        # Step 8: Stockout risk analysis
        # Current risk (simplified)
        current_cover_days = input_data.current_stock / demand.avg_daily_demand if demand.avg_daily_demand > 0 else 365
        if current_cover_days < input_data.lead_time_days:
            stockout_risk_current = min(0.9, 1 - (current_cover_days / input_data.lead_time_days))
        else:
            stockout_risk_current = 0.05

        stockout_risk_optimized = safety.stockout_risk

        # Step 9: Generate recommendations
        recommendations = self._generate_recommendations(
            input_data, strategy, eoq, safety, demand, stock_adjustment
        )

        # Calculate provenance
        provenance_inputs = {
            "part_id": input_data.part_id,
            "criticality": input_data.criticality.value,
            "unit_cost": input_data.unit_cost,
            "lead_time_days": input_data.lead_time_days
        }
        provenance_outputs = {
            "eoq": eoq.eoq,
            "safety_stock": safety.safety_stock,
            "recommended_stock": recommended_stock,
            "strategy": strategy.value
        }
        provenance_hash = self._calculate_provenance(provenance_inputs, provenance_outputs)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Optimization complete: strategy={strategy.value}, "
            f"recommended_stock={recommended_stock}, savings=${annual_savings:.2f}"
        )

        return SparePartOptimizationOutput(
            part_id=input_data.part_id,
            part_name=input_data.part_name,
            equipment_type=input_data.equipment_type.value,
            criticality=input_data.criticality.value,
            stocking_strategy=strategy,
            strategy_rationale=rationale,
            eoq_result=eoq,
            safety_stock_result=safety,
            current_stock=input_data.current_stock,
            recommended_stock=recommended_stock,
            stock_adjustment=stock_adjustment,
            demand_forecast=demand,
            current_annual_cost=self._apply_precision(current_cost),
            optimized_annual_cost=self._apply_precision(optimized_cost),
            annual_savings=self._apply_precision(annual_savings),
            savings_pct=self._apply_precision(savings_pct),
            lead_time_days=input_data.lead_time_days,
            lead_time_variability=lt_variability,
            stockout_risk_current=self._apply_precision(stockout_risk_current),
            stockout_risk_optimized=self._apply_precision(stockout_risk_optimized),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    def optimize_portfolio(
        self,
        parts: List[SparePartInput]
    ) -> Dict[str, Any]:
        """
        Optimize entire spare parts portfolio.

        Args:
            parts: List of spare part inputs

        Returns:
            Portfolio optimization summary
        """
        results = []
        total_current_cost = 0
        total_optimized_cost = 0
        total_current_value = 0

        for part in parts:
            result = self.optimize(part)
            results.append(result)
            total_current_cost += result.current_annual_cost
            total_optimized_cost += result.optimized_annual_cost
            total_current_value += part.current_stock * part.unit_cost

        total_savings = total_current_cost - total_optimized_cost

        # Categorize by strategy
        by_strategy = {}
        for r in results:
            strat = r.stocking_strategy.value
            if strat not in by_strategy:
                by_strategy[strat] = 0
            by_strategy[strat] += 1

        # Categorize by criticality
        by_criticality = {}
        for r in results:
            crit = r.criticality
            if crit not in by_criticality:
                by_criticality[crit] = 0
            by_criticality[crit] += 1

        return {
            "total_parts": len(parts),
            "total_current_inventory_value": self._apply_precision(total_current_value),
            "total_current_annual_cost": self._apply_precision(total_current_cost),
            "total_optimized_annual_cost": self._apply_precision(total_optimized_cost),
            "total_annual_savings": self._apply_precision(total_savings),
            "savings_percentage": self._apply_precision(
                (total_savings / total_current_cost * 100) if total_current_cost > 0 else 0
            ),
            "parts_by_strategy": by_strategy,
            "parts_by_criticality": by_criticality,
            "optimization_results": results
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_eoq(
    annual_demand: float,
    unit_cost: float,
    ordering_cost: float = 50.0,
    holding_rate: float = 0.25
) -> EOQResult:
    """
    Quick EOQ calculation.

    Example:
        >>> result = calculate_eoq(annual_demand=120, unit_cost=100)
        >>> print(f"EOQ: {result.eoq} units")
    """
    optimizer = SparePartsOptimizer()
    return optimizer._calculate_eoq(
        annual_demand, unit_cost, ordering_cost, holding_rate
    )


def calculate_safety_stock(
    daily_demand: float,
    demand_std: float,
    lead_time_days: float,
    service_level: float = 0.95
) -> SafetyStockResult:
    """
    Quick safety stock calculation.

    Example:
        >>> result = calculate_safety_stock(
        ...     daily_demand=0.5,
        ...     demand_std=0.2,
        ...     lead_time_days=14,
        ...     service_level=0.95
        ... )
        >>> print(f"Safety stock: {result.safety_stock} units")
    """
    optimizer = SparePartsOptimizer()
    return optimizer._calculate_safety_stock(
        daily_demand, demand_std, lead_time_days, None, service_level
    )


def recommend_stocking_strategy(
    criticality: str,
    unit_cost: float,
    annual_demand: float,
    lead_time_days: float
) -> Tuple[str, str]:
    """
    Quick stocking strategy recommendation.

    Returns:
        Tuple of (strategy, rationale)
    """
    optimizer = SparePartsOptimizer()

    crit = PartCriticality(criticality) if criticality in [e.value for e in PartCriticality] else PartCriticality.STANDARD

    # Determine demand pattern based on annual demand
    if annual_demand > 24:
        pattern = DemandPattern.STEADY
    elif annual_demand > 6:
        pattern = DemandPattern.SEASONAL
    elif annual_demand > 1:
        pattern = DemandPattern.INTERMITTENT
    else:
        pattern = DemandPattern.SLOW_MOVING

    strategy, rationale = optimizer._determine_stocking_strategy(
        crit, pattern, unit_cost, annual_demand, lead_time_days
    )

    return strategy.value, rationale


# ============================================================================
# Unit Test Stubs
# ============================================================================

class TestSparePartsOptimizer:
    """Unit tests for SparePartsOptimizer."""

    def test_init(self):
        """Test initialization."""
        optimizer = SparePartsOptimizer()
        assert optimizer.precision == 4

    def test_eoq_calculation(self):
        """Test EOQ formula."""
        optimizer = SparePartsOptimizer()

        # Known case: D=1000, S=50, H=5 -> EOQ = sqrt(2*1000*50/5) = 141.4
        result = optimizer._calculate_eoq(
            annual_demand=1000,
            unit_cost=100,
            ordering_cost=50,
            holding_cost_rate=0.05
        )

        assert result.eoq == 141 or result.eoq == 142  # Rounding
        assert result.orders_per_year > 0
        assert result.total_annual_cost > 0

    def test_safety_stock_calculation(self):
        """Test safety stock formula."""
        optimizer = SparePartsOptimizer()

        result = optimizer._calculate_safety_stock(
            daily_demand=1.0,
            demand_std=0.3,
            lead_time_days=14,
            lead_time_std=2,
            service_level=0.95
        )

        assert result.safety_stock >= 0
        assert result.reorder_point > result.safety_stock  # ROP includes LT demand
        assert result.service_level == 0.95

    def test_stocking_strategy_critical(self):
        """Test critical parts always stocked."""
        optimizer = SparePartsOptimizer()

        strategy, _ = optimizer._determine_stocking_strategy(
            criticality=PartCriticality.CRITICAL,
            demand_pattern=DemandPattern.SLOW_MOVING,
            unit_cost=1000,
            annual_demand=0.1,
            lead_time_days=7
        )

        assert strategy == StockingStrategy.STOCK

    def test_demand_analysis(self):
        """Test demand pattern analysis."""
        optimizer = SparePartsOptimizer()

        # Steady demand scenario
        result = optimizer._analyze_demand(
            usage_history=[],
            annual_usage=120,  # 10 per month
            mtbf_hours=None,
            equipment_count=1
        )

        assert result.annual_demand == 120
        assert result.avg_monthly_demand == 10
        assert result.forecast_confidence > 0

    def test_full_optimization(self):
        """Test full optimization pipeline."""
        optimizer = SparePartsOptimizer()

        input_data = SparePartInput(
            part_id="PUMP-001",
            part_name="Mechanical Seal",
            equipment_type=EquipmentType.PUMP,
            criticality=PartCriticality.ESSENTIAL,
            unit_cost=500,
            ordering_cost=75,
            holding_cost_rate=0.25,
            current_stock=5,
            lead_time_days=21,
            annual_usage=12
        )

        result = optimizer.optimize(input_data)

        assert result.part_id == "PUMP-001"
        assert result.eoq_result.eoq >= 1
        assert result.safety_stock_result.safety_stock >= 0
        assert result.provenance_hash is not None
        assert len(result.recommendations) > 0

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        optimizer = SparePartsOptimizer()

        inputs = {"part_id": "TEST"}
        outputs = {"eoq": 10}

        hash1 = optimizer._calculate_provenance(inputs, outputs)
        hash2 = optimizer._calculate_provenance(inputs, outputs)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256
