"""
GL-013 PREDICTMAINT - Spare Parts Inventory Calculator

This module implements spare parts inventory optimization for
predictive maintenance using probabilistic demand modeling.

Key Features:
- Poisson demand modeling
- Service level optimization
- Economic Order Quantity (EOQ)
- Safety stock calculation
- Lead time consideration
- Critical spares identification

Reference Standards:
- ISO 55001: Asset Management
- SMRP Best Practice 5.6: Spare Parts Management
- APICS Supply Chain Management

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import math

from .constants import (
    DEFAULT_DECIMAL_PRECISION,
    Z_SCORES,
)
from .provenance import (
    ProvenanceBuilder,
    ProvenanceRecord,
    CalculationType,
    store_provenance,
)


# =============================================================================
# ENUMS
# =============================================================================

class PartCriticality(Enum):
    """Spare part criticality classification."""
    CRITICAL = auto()      # Production stops without this part
    ESSENTIAL = auto()     # Significant impact on operations
    IMPORTANT = auto()     # Moderate impact, workaround possible
    STANDARD = auto()      # Low impact, easily substituted
    CONSUMABLE = auto()    # Regular consumables


class DemandPattern(Enum):
    """Demand pattern classification."""
    FAST_MOVING = auto()   # High, steady demand
    SLOW_MOVING = auto()   # Low but regular demand
    LUMPY = auto()         # Irregular, unpredictable demand
    SEASONAL = auto()      # Predictable seasonal patterns
    SPORADIC = auto()      # Very infrequent demand


class ReplenishmentPolicy(Enum):
    """Inventory replenishment policies."""
    FIXED_ORDER_QUANTITY = auto()  # Order Q when stock hits s (s,Q)
    FIXED_ORDER_INTERVAL = auto()  # Order up to S every T periods (T,S)
    MIN_MAX = auto()               # Order to max when stock hits min
    JUST_IN_TIME = auto()          # Order as needed (for non-critical)


# Service level z-scores
SERVICE_LEVEL_Z: Dict[str, Decimal] = {
    "90%": Decimal("1.282"),
    "95%": Decimal("1.645"),
    "97%": Decimal("1.881"),
    "98%": Decimal("2.054"),
    "99%": Decimal("2.326"),
    "99.5%": Decimal("2.576"),
    "99.9%": Decimal("3.090"),
}


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class InventoryOptimizationResult:
    """
    Result of inventory optimization calculation.

    Attributes:
        reorder_point: Stock level at which to reorder
        order_quantity: Quantity to order (EOQ)
        safety_stock: Safety stock level
        cycle_stock: Average cycle stock
        total_annual_cost: Total expected annual cost
        holding_cost: Annual holding cost
        ordering_cost: Annual ordering cost
        stockout_cost: Expected annual stockout cost
        service_level: Achieved service level
        turns_per_year: Inventory turns
        provenance_hash: SHA-256 hash
    """
    reorder_point: Decimal
    order_quantity: Decimal
    safety_stock: Decimal
    cycle_stock: Decimal
    total_annual_cost: Decimal
    holding_cost: Decimal
    ordering_cost: Decimal
    stockout_cost: Decimal
    service_level: str
    turns_per_year: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reorder_point": str(self.reorder_point),
            "order_quantity": str(self.order_quantity),
            "safety_stock": str(self.safety_stock),
            "cycle_stock": str(self.cycle_stock),
            "total_annual_cost": str(self.total_annual_cost),
            "holding_cost": str(self.holding_cost),
            "ordering_cost": str(self.ordering_cost),
            "stockout_cost": str(self.stockout_cost),
            "service_level": self.service_level,
            "turns_per_year": str(self.turns_per_year),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class EOQResult:
    """Result of Economic Order Quantity calculation."""
    eoq: Decimal
    annual_orders: Decimal
    order_cycle_days: Decimal
    total_cost: Decimal
    holding_cost_component: Decimal
    ordering_cost_component: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class SafetyStockResult:
    """Result of safety stock calculation."""
    safety_stock: Decimal
    reorder_point: Decimal
    service_level: str
    demand_variability: Decimal
    lead_time_variability: Decimal
    combined_variability: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class CriticalSpareResult:
    """Result of critical spare analysis."""
    part_id: str
    criticality: PartCriticality
    criticality_score: Decimal
    recommended_stock: int
    current_stock: int
    stock_status: str
    mean_time_between_usage: Decimal
    lead_time_days: Decimal
    unit_cost: Decimal
    annual_holding_cost: Decimal
    stockout_risk: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class PoissonDemandResult:
    """Result of Poisson demand analysis."""
    mean_demand: Decimal
    variance: Decimal
    probability_zero: Decimal
    probability_table: Dict[int, Decimal]
    confidence_quantile: int
    service_level: str
    provenance_hash: str = ""


# =============================================================================
# SPARE PARTS CALCULATOR
# =============================================================================

class SparePartsCalculator:
    """
    Spare parts inventory optimization calculator.

    Implements probabilistic inventory models for maintenance
    spare parts, considering:
    - Stochastic demand (Poisson for slow-moving parts)
    - Lead time uncertainty
    - Service level requirements
    - Holding and ordering costs

    Reference: Silver, Pyke & Peterson (1998), Inventory Management

    Example:
        >>> calc = SparePartsCalculator()
        >>> result = calc.calculate_eoq(
        ...     annual_demand=100,
        ...     order_cost=50,
        ...     unit_cost=200,
        ...     holding_rate=0.25
        ... )
        >>> print(f"EOQ: {result.eoq} units")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        store_provenance_records: bool = True
    ):
        """
        Initialize Spare Parts Calculator.

        Args:
            precision: Decimal precision for calculations
            store_provenance_records: Whether to store provenance
        """
        self._precision = precision
        self._store_provenance = store_provenance_records

    # =========================================================================
    # ECONOMIC ORDER QUANTITY (EOQ)
    # =========================================================================

    def calculate_eoq(
        self,
        annual_demand: Union[Decimal, float, int, str],
        order_cost: Union[Decimal, float, str],
        unit_cost: Union[Decimal, float, str],
        holding_rate: Union[Decimal, float, str] = "0.25"
    ) -> EOQResult:
        """
        Calculate Economic Order Quantity (EOQ).

        The classic EOQ model minimizes total inventory cost:
            TC = D*C + (D/Q)*S + (Q/2)*H

        Where:
            D = Annual demand
            C = Unit cost
            Q = Order quantity
            S = Order/setup cost per order
            H = Holding cost per unit per year

        Optimal Q (EOQ):
            Q* = sqrt(2*D*S / H)

        Args:
            annual_demand: Annual demand in units
            order_cost: Cost per order ($)
            unit_cost: Cost per unit ($)
            holding_rate: Annual holding rate as fraction of unit cost

        Returns:
            EOQResult

        Reference:
            Harris, F.W. (1913). "How Many Parts to Make at Once"
            Factory, The Magazine of Management, 10(2), 135-136.

        Example:
            >>> calc = SparePartsCalculator()
            >>> result = calc.calculate_eoq(
            ...     annual_demand=1000,
            ...     order_cost=100,
            ...     unit_cost=50,
            ...     holding_rate=0.20
            ... )
            >>> print(f"EOQ: {result.eoq}")
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        # Convert inputs
        D = self._to_decimal(annual_demand)
        S = self._to_decimal(order_cost)
        C = self._to_decimal(unit_cost)
        r = self._to_decimal(holding_rate)

        # Calculate holding cost per unit
        H = C * r

        builder.add_input("annual_demand", D)
        builder.add_input("order_cost", S)
        builder.add_input("unit_cost", C)
        builder.add_input("holding_rate", r)

        # Step 1: Validate inputs
        if D <= Decimal("0"):
            raise ValueError("Annual demand must be positive")
        if S <= Decimal("0"):
            raise ValueError("Order cost must be positive")
        if H <= Decimal("0"):
            raise ValueError("Holding cost must be positive")

        # Step 2: Calculate EOQ
        # Q* = sqrt(2*D*S / H)
        eoq_squared = (Decimal("2") * D * S) / H
        eoq = self._sqrt(eoq_squared)

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate Economic Order Quantity",
            inputs={"D": D, "S": S, "H": H},
            output_name="eoq",
            output_value=eoq,
            formula="Q* = sqrt(2*D*S / H)",
            reference="Harris (1913)"
        )

        # Step 3: Calculate annual orders
        annual_orders = D / eoq

        builder.add_step(
            step_number=2,
            operation="divide",
            description="Calculate number of orders per year",
            inputs={"D": D, "eoq": eoq},
            output_name="annual_orders",
            output_value=annual_orders,
            formula="N = D / Q"
        )

        # Step 4: Calculate order cycle time
        # Cycle = 365 / annual_orders (in days)
        order_cycle_days = Decimal("365") / annual_orders

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate order cycle time",
            inputs={"annual_orders": annual_orders},
            output_name="order_cycle_days",
            output_value=order_cycle_days,
            formula="Cycle = 365 / N"
        )

        # Step 5: Calculate total cost components
        # Ordering cost = (D/Q) * S
        ordering_cost = annual_orders * S

        # Holding cost = (Q/2) * H
        holding_cost = (eoq / Decimal("2")) * H

        # Total cost (excluding purchase cost)
        total_cost = ordering_cost + holding_cost

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate total inventory cost",
            inputs={"ordering": ordering_cost, "holding": holding_cost},
            output_name="total_cost",
            output_value=total_cost,
            formula="TC = (D/Q)*S + (Q/2)*H"
        )

        builder.add_output("eoq", eoq)
        builder.add_output("total_cost", total_cost)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return EOQResult(
            eoq=self._apply_precision(eoq, 0),
            annual_orders=self._apply_precision(annual_orders, 2),
            order_cycle_days=self._apply_precision(order_cycle_days, 1),
            total_cost=self._apply_precision(total_cost, 2),
            holding_cost_component=self._apply_precision(holding_cost, 2),
            ordering_cost_component=self._apply_precision(ordering_cost, 2),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # SAFETY STOCK CALCULATION
    # =========================================================================

    def calculate_safety_stock(
        self,
        average_demand_per_day: Union[Decimal, float, str],
        demand_std_dev: Union[Decimal, float, str],
        lead_time_days: Union[Decimal, float, str],
        lead_time_std_dev: Union[Decimal, float, str] = "0",
        service_level: str = "95%"
    ) -> SafetyStockResult:
        """
        Calculate safety stock for target service level.

        Uses the combined demand-lead time variability formula:

            SS = z * sqrt(LT * sigma_d^2 + d^2 * sigma_LT^2)

        Where:
            z = Service level z-score
            LT = Average lead time
            sigma_d = Demand standard deviation
            d = Average demand
            sigma_LT = Lead time standard deviation

        Args:
            average_demand_per_day: Mean daily demand
            demand_std_dev: Standard deviation of daily demand
            lead_time_days: Average lead time in days
            lead_time_std_dev: Standard deviation of lead time
            service_level: Target service level (e.g., "95%")

        Returns:
            SafetyStockResult

        Reference:
            Silver, Pyke & Peterson (1998), Chapter 7

        Example:
            >>> calc = SparePartsCalculator()
            >>> result = calc.calculate_safety_stock(
            ...     average_demand_per_day=5,
            ...     demand_std_dev=2,
            ...     lead_time_days=7,
            ...     service_level="99%"
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        # Convert inputs
        d = self._to_decimal(average_demand_per_day)
        sigma_d = self._to_decimal(demand_std_dev)
        LT = self._to_decimal(lead_time_days)
        sigma_LT = self._to_decimal(lead_time_std_dev)

        # Get z-score for service level
        if service_level not in SERVICE_LEVEL_Z:
            raise ValueError(f"Unsupported service level: {service_level}")
        z = SERVICE_LEVEL_Z[service_level]

        builder.add_input("average_demand_per_day", d)
        builder.add_input("demand_std_dev", sigma_d)
        builder.add_input("lead_time_days", LT)
        builder.add_input("lead_time_std_dev", sigma_LT)
        builder.add_input("service_level", service_level)
        builder.add_input("z_score", z)

        # Step 1: Calculate demand during lead time variability
        # Variance during LT from demand variability: LT * sigma_d^2
        demand_var_component = LT * sigma_d * sigma_d

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate demand variability component",
            inputs={"LT": LT, "sigma_d": sigma_d},
            output_name="demand_var_component",
            output_value=demand_var_component,
            formula="LT * sigma_d^2"
        )

        # Step 2: Calculate lead time variability component
        # Variance from lead time variability: d^2 * sigma_LT^2
        lt_var_component = d * d * sigma_LT * sigma_LT

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate lead time variability component",
            inputs={"d": d, "sigma_LT": sigma_LT},
            output_name="lt_var_component",
            output_value=lt_var_component,
            formula="d^2 * sigma_LT^2"
        )

        # Step 3: Calculate combined standard deviation
        combined_variance = demand_var_component + lt_var_component
        combined_std_dev = self._sqrt(combined_variance)

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate combined standard deviation",
            inputs={"combined_variance": combined_variance},
            output_name="combined_std_dev",
            output_value=combined_std_dev,
            formula="sqrt(LT*sigma_d^2 + d^2*sigma_LT^2)"
        )

        # Step 4: Calculate safety stock
        safety_stock = z * combined_std_dev

        builder.add_step(
            step_number=4,
            operation="multiply",
            description="Calculate safety stock",
            inputs={"z": z, "combined_std_dev": combined_std_dev},
            output_name="safety_stock",
            output_value=safety_stock,
            formula="SS = z * sigma_combined",
            reference="Silver, Pyke & Peterson (1998)"
        )

        # Step 5: Calculate reorder point
        # ROP = demand during lead time + safety stock
        demand_during_lt = d * LT
        reorder_point = demand_during_lt + safety_stock

        builder.add_step(
            step_number=5,
            operation="add",
            description="Calculate reorder point",
            inputs={"demand_during_lt": demand_during_lt, "safety_stock": safety_stock},
            output_name="reorder_point",
            output_value=reorder_point,
            formula="ROP = d*LT + SS"
        )

        builder.add_output("safety_stock", safety_stock)
        builder.add_output("reorder_point", reorder_point)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return SafetyStockResult(
            safety_stock=self._apply_precision(safety_stock, 0, round_up=True),
            reorder_point=self._apply_precision(reorder_point, 0, round_up=True),
            service_level=service_level,
            demand_variability=self._apply_precision(sigma_d, 4),
            lead_time_variability=self._apply_precision(sigma_LT, 4),
            combined_variability=self._apply_precision(combined_std_dev, 4),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # POISSON DEMAND MODELING
    # =========================================================================

    def calculate_poisson_demand(
        self,
        mean_demand_per_period: Union[Decimal, float, str],
        service_level: str = "95%",
        max_quantity: int = 20
    ) -> PoissonDemandResult:
        """
        Model demand using Poisson distribution.

        For slow-moving spare parts, demand often follows Poisson:
            P(X = k) = (lambda^k * e^(-lambda)) / k!

        The service level quantile gives the stock level needed
        to achieve that probability of not stocking out.

        Args:
            mean_demand_per_period: Average demand (lambda)
            service_level: Target service level
            max_quantity: Maximum quantity to calculate probabilities

        Returns:
            PoissonDemandResult

        Reference:
            ISO 13379-1:2012, Appendix on spare parts

        Example:
            >>> calc = SparePartsCalculator()
            >>> result = calc.calculate_poisson_demand(
            ...     mean_demand_per_period=2.5,
            ...     service_level="99%"
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        lam = self._to_decimal(mean_demand_per_period)

        if service_level not in SERVICE_LEVEL_Z:
            # Try direct percentage
            try:
                target_prob = self._to_decimal(service_level.replace("%", "")) / Decimal("100")
            except:
                raise ValueError(f"Unsupported service level: {service_level}")
        else:
            # Convert z-score to probability (approximately)
            z = SERVICE_LEVEL_Z[service_level]
            target_prob = self._normal_cdf(z)

        builder.add_input("mean_demand", lam)
        builder.add_input("service_level", service_level)
        builder.add_input("target_probability", target_prob)

        # Step 1: Calculate Poisson probabilities
        probabilities = {}
        cumulative = Decimal("0")
        quantile = 0

        for k in range(max_quantity + 1):
            # P(X = k) = (lambda^k * e^(-lambda)) / k!
            prob = self._poisson_pmf(k, lam)
            probabilities[k] = prob
            cumulative += prob

            if cumulative >= target_prob and quantile == 0:
                quantile = k

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate Poisson probability distribution",
            inputs={"lambda": lam, "max_k": max_quantity},
            output_name="probabilities",
            output_value={str(k): str(v) for k, v in list(probabilities.items())[:5]}
        )

        # Step 2: Calculate probability of zero demand
        prob_zero = self._exp(-lam)

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate P(X=0)",
            inputs={"lambda": lam},
            output_name="prob_zero",
            output_value=prob_zero,
            formula="P(X=0) = e^(-lambda)"
        )

        # Step 3: Find service level quantile
        builder.add_step(
            step_number=3,
            operation="find",
            description="Find quantile for service level",
            inputs={"target_probability": target_prob},
            output_name="quantile",
            output_value=quantile,
            formula="Smallest k where P(X <= k) >= target"
        )

        # For Poisson, variance = mean
        variance = lam

        builder.add_output("quantile", quantile)
        builder.add_output("variance", variance)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return PoissonDemandResult(
            mean_demand=lam,
            variance=lam,
            probability_zero=self._apply_precision(prob_zero, 6),
            probability_table={k: self._apply_precision(v, 6) for k, v in probabilities.items()},
            confidence_quantile=quantile,
            service_level=service_level,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # CRITICAL SPARES ANALYSIS
    # =========================================================================

    def analyze_critical_spares(
        self,
        parts_data: List[Dict[str, Any]]
    ) -> List[CriticalSpareResult]:
        """
        Analyze spare parts for criticality and stock recommendations.

        Uses VED (Vital, Essential, Desirable) and ABC analysis
        combined with reliability data.

        Criticality factors:
        - Equipment criticality (production impact)
        - Lead time (supply risk)
        - Cost (financial impact)
        - Failure rate (demand predictability)

        Args:
            parts_data: List of parts with fields:
                - part_id: Part identifier
                - equipment_criticality: 1-5 scale
                - lead_time_days: Supplier lead time
                - unit_cost: Cost per unit
                - mean_time_between_usage: Average MTBU (hours)
                - current_stock: Current inventory level
                - annual_demand: Historical annual demand

        Returns:
            List of CriticalSpareResult sorted by criticality

        Example:
            >>> calc = SparePartsCalculator()
            >>> parts = [
            ...     {"part_id": "SEAL-001", "equipment_criticality": 5,
            ...      "lead_time_days": 30, "unit_cost": 50,
            ...      "mean_time_between_usage": 2000, "current_stock": 2,
            ...      "annual_demand": 4},
            ... ]
            >>> results = calc.analyze_critical_spares(parts)
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        builder.add_input("num_parts", len(parts_data))

        results = []

        for part in parts_data:
            part_id = part.get("part_id", "UNKNOWN")
            eq_crit = self._to_decimal(part.get("equipment_criticality", "3"))
            lead_time = self._to_decimal(part.get("lead_time_days", "14"))
            unit_cost = self._to_decimal(part.get("unit_cost", "100"))
            mtbu = self._to_decimal(part.get("mean_time_between_usage", "8760"))
            current_stock = int(part.get("current_stock", 0))
            annual_demand = self._to_decimal(part.get("annual_demand", "1"))

            # Calculate criticality score (0-100)
            # Factors: equipment importance, lead time risk, cost impact
            eq_crit_norm = eq_crit / Decimal("5") * Decimal("40")  # 40% weight
            lead_time_norm = min(lead_time / Decimal("90"), Decimal("1")) * Decimal("30")  # 30% weight
            cost_norm = min(unit_cost / Decimal("10000"), Decimal("1")) * Decimal("15")  # 15% weight
            demand_norm = min(annual_demand / Decimal("12"), Decimal("1")) * Decimal("15")  # 15% weight

            criticality_score = eq_crit_norm + lead_time_norm + cost_norm + demand_norm

            # Determine criticality class
            if criticality_score >= Decimal("70") or eq_crit >= Decimal("5"):
                criticality = PartCriticality.CRITICAL
            elif criticality_score >= Decimal("50"):
                criticality = PartCriticality.ESSENTIAL
            elif criticality_score >= Decimal("30"):
                criticality = PartCriticality.IMPORTANT
            elif unit_cost < Decimal("50"):
                criticality = PartCriticality.CONSUMABLE
            else:
                criticality = PartCriticality.STANDARD

            # Calculate recommended stock level
            # Based on Poisson model with lead time coverage
            daily_demand = annual_demand / Decimal("365")
            demand_during_lt = daily_demand * lead_time

            # Use Poisson quantile for 99% service on critical, 95% on standard
            if criticality in [PartCriticality.CRITICAL, PartCriticality.ESSENTIAL]:
                service_level = "99%"
            else:
                service_level = "95%"

            poisson_result = self.calculate_poisson_demand(
                demand_during_lt, service_level
            )
            recommended_stock = max(1, poisson_result.confidence_quantile)

            # Determine stock status
            if current_stock >= recommended_stock:
                stock_status = "Adequate"
            elif current_stock > 0:
                stock_status = "Low"
            else:
                stock_status = "Stockout Risk"

            # Calculate holding cost and stockout risk
            holding_rate = Decimal("0.25")  # 25% per year
            annual_holding = unit_cost * holding_rate * Decimal(str(current_stock))

            if current_stock > 0:
                # Approximate stockout probability
                stockout_risk = self._exp(-Decimal(str(current_stock)) / max(demand_during_lt, Decimal("0.1")))
            else:
                stockout_risk = Decimal("1")

            results.append(CriticalSpareResult(
                part_id=part_id,
                criticality=criticality,
                criticality_score=self._apply_precision(criticality_score, 1),
                recommended_stock=recommended_stock,
                current_stock=current_stock,
                stock_status=stock_status,
                mean_time_between_usage=mtbu,
                lead_time_days=lead_time,
                unit_cost=unit_cost,
                annual_holding_cost=self._apply_precision(annual_holding, 2),
                stockout_risk=self._apply_precision(min(stockout_risk, Decimal("1")), 4),
                provenance_hash=""
            ))

        # Sort by criticality (most critical first)
        results.sort(key=lambda x: (-x.criticality_score, x.part_id))

        builder.add_step(
            step_number=1,
            operation="analyze",
            description="Analyze part criticality",
            inputs={"num_parts": len(parts_data)},
            output_name="critical_count",
            output_value=sum(1 for r in results if r.criticality == PartCriticality.CRITICAL)
        )

        builder.add_output("num_critical", sum(1 for r in results if r.criticality == PartCriticality.CRITICAL))
        builder.add_output("num_stockout_risk", sum(1 for r in results if r.stock_status == "Stockout Risk"))

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return results

    # =========================================================================
    # INVENTORY OPTIMIZATION
    # =========================================================================

    def optimize_inventory(
        self,
        annual_demand: Union[Decimal, float, int, str],
        unit_cost: Union[Decimal, float, str],
        order_cost: Union[Decimal, float, str],
        holding_rate: Union[Decimal, float, str] = "0.25",
        stockout_cost_per_unit: Union[Decimal, float, str] = "0",
        lead_time_days: Union[Decimal, float, str] = "14",
        demand_std_dev: Union[Decimal, float, str] = "0",
        lead_time_std_dev: Union[Decimal, float, str] = "0",
        service_level: str = "95%"
    ) -> InventoryOptimizationResult:
        """
        Comprehensive inventory optimization.

        Combines EOQ for order quantity with safety stock
        calculation for complete inventory policy.

        Args:
            annual_demand: Annual demand in units
            unit_cost: Cost per unit
            order_cost: Cost per order
            holding_rate: Annual holding rate (fraction of unit cost)
            stockout_cost_per_unit: Cost per unit of stockout
            lead_time_days: Average lead time
            demand_std_dev: Standard deviation of daily demand
            lead_time_std_dev: Standard deviation of lead time
            service_level: Target service level

        Returns:
            InventoryOptimizationResult

        Example:
            >>> calc = SparePartsCalculator()
            >>> result = calc.optimize_inventory(
            ...     annual_demand=100,
            ...     unit_cost=500,
            ...     order_cost=75,
            ...     lead_time_days=21,
            ...     demand_std_dev=0.5,
            ...     service_level="99%"
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        # Convert inputs
        D = self._to_decimal(annual_demand)
        C = self._to_decimal(unit_cost)
        S = self._to_decimal(order_cost)
        r = self._to_decimal(holding_rate)
        B = self._to_decimal(stockout_cost_per_unit)
        LT = self._to_decimal(lead_time_days)
        sigma_d = self._to_decimal(demand_std_dev)
        sigma_LT = self._to_decimal(lead_time_std_dev)

        H = C * r  # Holding cost per unit per year

        builder.add_input("annual_demand", D)
        builder.add_input("unit_cost", C)
        builder.add_input("holding_rate", r)
        builder.add_input("lead_time_days", LT)
        builder.add_input("service_level", service_level)

        # Step 1: Calculate EOQ
        eoq_result = self.calculate_eoq(D, S, C, r)
        Q = eoq_result.eoq

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate EOQ",
            inputs={"D": D, "S": S, "H": H},
            output_name="eoq",
            output_value=Q,
            reference="Harris (1913)"
        )

        # Step 2: Calculate safety stock
        daily_demand = D / Decimal("365")

        if sigma_d > Decimal("0") or sigma_LT > Decimal("0"):
            ss_result = self.calculate_safety_stock(
                daily_demand, sigma_d, LT, sigma_LT, service_level
            )
            safety_stock = ss_result.safety_stock
            reorder_point = ss_result.reorder_point
        else:
            # No variability data - use simple formula
            z = SERVICE_LEVEL_Z.get(service_level, Decimal("1.645"))
            # Assume coefficient of variation of 0.5
            cv = Decimal("0.5")
            sigma_dlt = daily_demand * LT * cv
            safety_stock = z * sigma_dlt
            reorder_point = daily_demand * LT + safety_stock

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate safety stock",
            inputs={"service_level": service_level, "LT": LT},
            output_name="safety_stock",
            output_value=safety_stock
        )

        # Step 3: Calculate cycle stock
        cycle_stock = Q / Decimal("2")

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate average cycle stock",
            inputs={"Q": Q},
            output_name="cycle_stock",
            output_value=cycle_stock,
            formula="Cycle stock = Q/2"
        )

        # Step 4: Calculate costs
        # Ordering cost
        ordering_cost = (D / Q) * S

        # Holding cost (cycle stock + safety stock)
        average_inventory = cycle_stock + safety_stock
        holding_cost = average_inventory * H

        # Stockout cost (approximate based on service level)
        service_prob = Decimal(service_level.replace("%", "")) / Decimal("100")
        expected_stockouts = D * (Decimal("1") - service_prob)
        stockout_cost = expected_stockouts * B if B > Decimal("0") else Decimal("0")

        total_cost = ordering_cost + holding_cost + stockout_cost

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate total annual costs",
            inputs={
                "ordering_cost": ordering_cost,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost
            },
            output_name="total_cost",
            output_value=total_cost
        )

        # Step 5: Calculate inventory turns
        if average_inventory > Decimal("0"):
            turns = D / average_inventory
        else:
            turns = Decimal("0")

        # Step 6: Calculate availability/service level achieved
        availability = service_prob  # Simplified

        builder.add_output("reorder_point", reorder_point)
        builder.add_output("order_quantity", Q)
        builder.add_output("total_cost", total_cost)
        builder.add_output("turns_per_year", turns)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return InventoryOptimizationResult(
            reorder_point=self._apply_precision(reorder_point, 0, round_up=True),
            order_quantity=self._apply_precision(Q, 0),
            safety_stock=self._apply_precision(safety_stock, 0, round_up=True),
            cycle_stock=self._apply_precision(cycle_stock, 1),
            total_annual_cost=self._apply_precision(total_cost, 2),
            holding_cost=self._apply_precision(holding_cost, 2),
            ordering_cost=self._apply_precision(ordering_cost, 2),
            stockout_cost=self._apply_precision(stockout_cost, 2),
            service_level=service_level,
            turns_per_year=self._apply_precision(turns, 2),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None,
        round_up: bool = False
    ) -> Decimal:
        """Apply precision rounding."""
        prec = precision if precision is not None else self._precision
        rounding = ROUND_CEILING if round_up else ROUND_HALF_UP
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=rounding)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=rounding)

    def _sqrt(self, x: Decimal) -> Decimal:
        """Calculate square root."""
        if x < Decimal("0"):
            raise ValueError("Cannot take square root of negative number")
        if x == Decimal("0"):
            return Decimal("0")
        return Decimal(str(math.sqrt(float(x))))

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate e^x."""
        if x == Decimal("0"):
            return Decimal("1")
        if x < Decimal("-700"):
            return Decimal("0")
        return Decimal(str(math.exp(float(x))))

    def _factorial(self, n: int) -> Decimal:
        """Calculate n!"""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if n <= 1:
            return Decimal("1")
        result = Decimal("1")
        for i in range(2, n + 1):
            result *= Decimal(str(i))
        return result

    def _poisson_pmf(self, k: int, lam: Decimal) -> Decimal:
        """Calculate Poisson probability mass function."""
        if k < 0:
            return Decimal("0")
        if lam == Decimal("0"):
            return Decimal("1") if k == 0 else Decimal("0")
        # P(X=k) = (lambda^k * e^(-lambda)) / k!
        return (self._power(lam, Decimal(str(k))) * self._exp(-lam)) / self._factorial(k)

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate base^exponent."""
        if base == Decimal("0"):
            return Decimal("0") if exponent > Decimal("0") else Decimal("1")
        if exponent == Decimal("0"):
            return Decimal("1")
        return Decimal(str(math.pow(float(base), float(exponent))))

    def _normal_cdf(self, z: Decimal) -> Decimal:
        """Calculate standard normal CDF."""
        z_float = float(z)
        result = 0.5 * (1 + math.erf(z_float / math.sqrt(2)))
        return Decimal(str(result))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "PartCriticality",
    "DemandPattern",
    "ReplenishmentPolicy",

    # Constants
    "SERVICE_LEVEL_Z",

    # Data classes
    "InventoryOptimizationResult",
    "EOQResult",
    "SafetyStockResult",
    "CriticalSpareResult",
    "PoissonDemandResult",

    # Main class
    "SparePartsCalculator",
]
