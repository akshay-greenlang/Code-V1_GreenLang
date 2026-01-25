# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft - Cost Model

Deterministic cost calculation engine for fuel procurement optimization.

Cost Components:
- PurchaseCost: Spot pricing + contract pricing (take-or-pay)
- LogisticsCost: Transportation, delivery fees, port charges
- StorageCost: Holding costs, inventory carrying costs
- LossCost: Boil-off, evaporation, handling losses
- PenaltyCost: Contract shortfall penalties, delivery failures
- CarbonCost: Emissions x carbon price (ETS, carbon tax)
- RiskCost: Uncertainty premiums, hedging costs

Standards:
- ISO 14064 (GHG Quantification)
- GHG Protocol (Scope 1, 2, 3)
- TCFD (Climate Financial Disclosure)

Zero-Hallucination Approach:
- All calculations use Decimal arithmetic for precision
- No LLM calls in any cost calculation path
- All cost formulas are deterministic and auditable
- Full provenance tracking with SHA-256 hashing

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of cost components."""
    PURCHASE = "purchase"
    LOGISTICS = "logistics"
    STORAGE = "storage"
    LOSS = "loss"
    PENALTY = "penalty"
    CARBON = "carbon"
    RISK = "risk"


class PricingType(Enum):
    """Type of pricing mechanism."""
    SPOT = "spot"
    CONTRACT = "contract"
    INDEX_LINKED = "index_linked"
    FORMULA = "formula"


class LogisticsMode(Enum):
    """Transportation mode."""
    PIPELINE = "pipeline"
    TRUCK = "truck"
    RAIL = "rail"
    BARGE = "barge"
    SHIP = "ship"


class CarbonScheme(Enum):
    """Carbon pricing scheme."""
    EU_ETS = "eu_ets"
    UK_ETS = "uk_ets"
    CARBON_TAX = "carbon_tax"
    CORSIA = "corsia"
    CCA = "cca"  # Climate Change Agreement
    INTERNAL = "internal"  # Internal carbon price


@dataclass(frozen=True)
class PurchaseCostParams:
    """
    Parameters for purchase cost calculation.

    Attributes:
        fuel_id: Fuel identifier
        pricing_type: Type of pricing mechanism
        spot_price_per_mj: Spot market price ($/MJ)
        contract_price_per_mj: Contract price ($/MJ)
        index_adjustment: Index adjustment factor
        volume_discount_threshold_mj: Volume for discount eligibility
        volume_discount_pct: Discount percentage for large volumes
    """
    fuel_id: str
    pricing_type: PricingType
    spot_price_per_mj: Decimal
    contract_price_per_mj: Optional[Decimal] = None
    index_adjustment: Decimal = Decimal("1.0")
    volume_discount_threshold_mj: Optional[Decimal] = None
    volume_discount_pct: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fuel_id": self.fuel_id,
            "pricing_type": self.pricing_type.value,
            "spot_price_per_mj": str(self.spot_price_per_mj),
            "contract_price_per_mj": str(self.contract_price_per_mj) if self.contract_price_per_mj else None,
            "index_adjustment": str(self.index_adjustment),
            "volume_discount_threshold_mj": str(self.volume_discount_threshold_mj) if self.volume_discount_threshold_mj else None,
            "volume_discount_pct": str(self.volume_discount_pct)
        }


@dataclass(frozen=True)
class LogisticsCostParams:
    """
    Parameters for logistics cost calculation.

    Attributes:
        mode: Transportation mode
        base_rate_per_mj: Base transportation rate ($/MJ)
        distance_km: Distance for transportation
        distance_rate_per_mj_km: Rate per MJ per km
        fixed_delivery_fee: Fixed fee per delivery
        port_charges_per_delivery: Port/terminal charges
        minimum_order_mj: Minimum order quantity
    """
    mode: LogisticsMode
    base_rate_per_mj: Decimal
    distance_km: Decimal = Decimal("0")
    distance_rate_per_mj_km: Decimal = Decimal("0")
    fixed_delivery_fee: Decimal = Decimal("0")
    port_charges_per_delivery: Decimal = Decimal("0")
    minimum_order_mj: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "base_rate_per_mj": str(self.base_rate_per_mj),
            "distance_km": str(self.distance_km),
            "distance_rate_per_mj_km": str(self.distance_rate_per_mj_km),
            "fixed_delivery_fee": str(self.fixed_delivery_fee),
            "port_charges_per_delivery": str(self.port_charges_per_delivery),
            "minimum_order_mj": str(self.minimum_order_mj)
        }


@dataclass(frozen=True)
class StorageCostParams:
    """
    Parameters for storage cost calculation.

    Attributes:
        tank_id: Tank identifier
        holding_cost_per_mj_day: Daily holding cost ($/MJ/day)
        insurance_rate_annual_pct: Annual insurance rate (%)
        capital_cost_per_mj_day: Capital carrying cost ($/MJ/day)
        loss_rate_per_day_pct: Daily loss rate (%)
    """
    tank_id: str
    holding_cost_per_mj_day: Decimal
    insurance_rate_annual_pct: Decimal = Decimal("0.1")
    capital_cost_per_mj_day: Decimal = Decimal("0")
    loss_rate_per_day_pct: Decimal = Decimal("0.01")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tank_id": self.tank_id,
            "holding_cost_per_mj_day": str(self.holding_cost_per_mj_day),
            "insurance_rate_annual_pct": str(self.insurance_rate_annual_pct),
            "capital_cost_per_mj_day": str(self.capital_cost_per_mj_day),
            "loss_rate_per_day_pct": str(self.loss_rate_per_day_pct)
        }


@dataclass(frozen=True)
class ContractPenaltyParams:
    """
    Parameters for contract penalty calculation.

    Attributes:
        contract_id: Contract identifier
        fuel_id: Associated fuel
        min_take_mj: Minimum take-or-pay quantity (MJ)
        max_take_mj: Maximum contract quantity (MJ)
        shortfall_penalty_per_mj: Penalty for shortfall ($/MJ)
        excess_penalty_per_mj: Penalty for exceeding contract ($/MJ)
    """
    contract_id: str
    fuel_id: str
    min_take_mj: Decimal
    max_take_mj: Decimal
    shortfall_penalty_per_mj: Decimal
    excess_penalty_per_mj: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "contract_id": self.contract_id,
            "fuel_id": self.fuel_id,
            "min_take_mj": str(self.min_take_mj),
            "max_take_mj": str(self.max_take_mj),
            "shortfall_penalty_per_mj": str(self.shortfall_penalty_per_mj),
            "excess_penalty_per_mj": str(self.excess_penalty_per_mj)
        }


@dataclass(frozen=True)
class CarbonCostParams:
    """
    Parameters for carbon cost calculation.

    Attributes:
        scheme: Carbon pricing scheme
        carbon_price_per_kg_co2e: Carbon price ($/kgCO2e)
        carbon_intensity_kg_co2e_mj: Carbon intensity (kgCO2e/MJ)
        free_allowance_kg_co2e: Free emission allowance
        compliance_factor: Compliance obligation factor (0-1)
    """
    scheme: CarbonScheme
    carbon_price_per_kg_co2e: Decimal
    carbon_intensity_kg_co2e_mj: Decimal
    free_allowance_kg_co2e: Decimal = Decimal("0")
    compliance_factor: Decimal = Decimal("1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scheme": self.scheme.value,
            "carbon_price_per_kg_co2e": str(self.carbon_price_per_kg_co2e),
            "carbon_intensity_kg_co2e_mj": str(self.carbon_intensity_kg_co2e_mj),
            "free_allowance_kg_co2e": str(self.free_allowance_kg_co2e),
            "compliance_factor": str(self.compliance_factor)
        }


@dataclass(frozen=True)
class RiskCostParams:
    """
    Parameters for risk cost calculation.

    Attributes:
        volatility_premium_pct: Premium for price volatility (%)
        supply_risk_premium_pct: Premium for supply risk (%)
        counterparty_risk_pct: Counterparty credit risk (%)
        hedging_cost_pct: Cost of hedging instruments (%)
    """
    volatility_premium_pct: Decimal = Decimal("0")
    supply_risk_premium_pct: Decimal = Decimal("0")
    counterparty_risk_pct: Decimal = Decimal("0")
    hedging_cost_pct: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "volatility_premium_pct": str(self.volatility_premium_pct),
            "supply_risk_premium_pct": str(self.supply_risk_premium_pct),
            "counterparty_risk_pct": str(self.counterparty_risk_pct),
            "hedging_cost_pct": str(self.hedging_cost_pct)
        }


@dataclass
class CostComponent:
    """
    Single cost component with breakdown.

    Represents one line item in the cost breakdown with full
    calculation details for audit trail.
    """
    category: CostCategory
    name: str
    amount: Decimal
    unit: str
    quantity: Decimal
    rate: Decimal
    description: str
    calculation_formula: str
    input_values: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "name": self.name,
            "amount": str(self.amount),
            "unit": self.unit,
            "quantity": str(self.quantity),
            "rate": str(self.rate),
            "description": self.description,
            "calculation_formula": self.calculation_formula,
            "input_values": self.input_values
        }


@dataclass
class CostBreakdown:
    """
    Complete cost breakdown with all components.

    Provides detailed cost analysis for a fuel procurement plan
    with full audit trail and provenance tracking.
    """
    # Cost totals by category
    purchase_cost: Decimal
    logistics_cost: Decimal
    storage_cost: Decimal
    loss_cost: Decimal
    penalty_cost: Decimal
    carbon_cost: Decimal
    risk_cost: Decimal

    # Grand total
    total_cost: Decimal

    # Detailed components
    components: List[CostComponent]

    # By fuel breakdown
    cost_by_fuel: Dict[str, Decimal]

    # By period breakdown
    cost_by_period: Dict[int, Decimal]

    # Metadata
    currency: str = "USD"
    period_count: int = 0
    total_energy_mj: Decimal = Decimal("0")
    average_cost_per_mj: Decimal = Decimal("0")

    # Carbon metrics
    total_emissions_kg_co2e: Decimal = Decimal("0")
    carbon_intensity_kg_co2e_mj: Decimal = Decimal("0")

    # Provenance
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization to compute derived fields."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()
        if self.total_energy_mj > Decimal("0"):
            self.average_cost_per_mj = self.total_cost / self.total_energy_mj

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "purchase_cost": str(self.purchase_cost),
            "logistics_cost": str(self.logistics_cost),
            "storage_cost": str(self.storage_cost),
            "loss_cost": str(self.loss_cost),
            "penalty_cost": str(self.penalty_cost),
            "carbon_cost": str(self.carbon_cost),
            "risk_cost": str(self.risk_cost),
            "total_cost": str(self.total_cost),
            "total_energy_mj": str(self.total_energy_mj),
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "purchase_cost": str(self.purchase_cost),
            "logistics_cost": str(self.logistics_cost),
            "storage_cost": str(self.storage_cost),
            "loss_cost": str(self.loss_cost),
            "penalty_cost": str(self.penalty_cost),
            "carbon_cost": str(self.carbon_cost),
            "risk_cost": str(self.risk_cost),
            "total_cost": str(self.total_cost),
            "components": [c.to_dict() for c in self.components],
            "cost_by_fuel": {k: str(v) for k, v in self.cost_by_fuel.items()},
            "cost_by_period": {str(k): str(v) for k, v in self.cost_by_period.items()},
            "currency": self.currency,
            "period_count": self.period_count,
            "total_energy_mj": str(self.total_energy_mj),
            "average_cost_per_mj": str(self.average_cost_per_mj),
            "total_emissions_kg_co2e": str(self.total_emissions_kg_co2e),
            "carbon_intensity_kg_co2e_mj": str(self.carbon_intensity_kg_co2e_mj),
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat()
        }

    def get_cost_summary(self) -> str:
        """Generate human-readable cost summary."""
        lines = [
            "=" * 50,
            "COST BREAKDOWN SUMMARY",
            "=" * 50,
            f"Purchase Cost:    ${self.purchase_cost:>15,.2f}",
            f"Logistics Cost:   ${self.logistics_cost:>15,.2f}",
            f"Storage Cost:     ${self.storage_cost:>15,.2f}",
            f"Loss Cost:        ${self.loss_cost:>15,.2f}",
            f"Penalty Cost:     ${self.penalty_cost:>15,.2f}",
            f"Carbon Cost:      ${self.carbon_cost:>15,.2f}",
            f"Risk Cost:        ${self.risk_cost:>15,.2f}",
            "-" * 50,
            f"TOTAL COST:       ${self.total_cost:>15,.2f}",
            "=" * 50,
            f"Total Energy:     {self.total_energy_mj:>15,.0f} MJ",
            f"Avg Cost/MJ:      ${self.average_cost_per_mj:>15,.6f}",
            f"Total Emissions:  {self.total_emissions_kg_co2e:>15,.0f} kgCO2e",
            f"Carbon Intensity: {self.carbon_intensity_kg_co2e_mj:>15,.6f} kgCO2e/MJ",
            "=" * 50,
        ]
        return "\n".join(lines)


class CostModel:
    """
    Deterministic cost calculation engine for fuel optimization.

    Provides ZERO-HALLUCINATION cost calculations for:
    - Purchase costs (spot, contract, indexed)
    - Logistics costs (transportation, delivery, port charges)
    - Storage costs (holding, insurance, capital)
    - Loss costs (boil-off, evaporation, handling)
    - Penalty costs (contract shortfall, delivery failure)
    - Carbon costs (ETS, carbon tax, internal pricing)
    - Risk costs (volatility, supply, counterparty, hedging)

    All calculations use Decimal arithmetic for precision and
    include full provenance tracking for audit compliance.

    Example:
        >>> model = CostModel()
        >>> purchase_params = PurchaseCostParams(
        ...     fuel_id="diesel",
        ...     pricing_type=PricingType.SPOT,
        ...     spot_price_per_mj=Decimal("0.025")
        ... )
        >>> cost = model.calculate_purchase_cost(
        ...     quantity_mj=Decimal("100000"),
        ...     params=purchase_params
        ... )
        >>> print(f"Purchase cost: ${cost}")
    """

    NAME: str = "CostModel"
    VERSION: str = "1.0.0"

    PRECISION: int = 6
    DAYS_PER_YEAR: Decimal = Decimal("365")

    def __init__(self, currency: str = "USD"):
        """
        Initialize cost model.

        Args:
            currency: Currency code for cost reporting (default: USD)
        """
        self._currency = currency
        self._calculation_steps: List[Dict[str, Any]] = []

        logger.info(f"CostModel initialized with currency={currency}")

    def calculate_purchase_cost(
        self,
        quantity_mj: Decimal,
        params: PurchaseCostParams,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent]:
        """
        Calculate purchase cost for fuel procurement - DETERMINISTIC.

        Formula:
        - Spot: cost = quantity * spot_price * index_adjustment
        - Contract: cost = quantity * contract_price
        - With volume discount: cost = cost * (1 - discount_pct/100)

        Args:
            quantity_mj: Quantity to purchase (MJ)
            params: Purchase cost parameters
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component)
        """
        step = {"operation": "calculate_purchase_cost", "fuel_id": params.fuel_id}

        # Determine base price based on pricing type
        if params.pricing_type == PricingType.SPOT:
            base_price = params.spot_price_per_mj * params.index_adjustment
            formula = "quantity * spot_price * index_adjustment"
        elif params.pricing_type == PricingType.CONTRACT:
            base_price = params.contract_price_per_mj or params.spot_price_per_mj
            formula = "quantity * contract_price"
        else:
            base_price = params.spot_price_per_mj * params.index_adjustment
            formula = "quantity * spot_price * index_adjustment"

        # Calculate base cost
        cost = quantity_mj * base_price

        # Apply volume discount if applicable
        discount_applied = Decimal("0")
        if (params.volume_discount_threshold_mj and
            quantity_mj >= params.volume_discount_threshold_mj and
            params.volume_discount_pct > Decimal("0")):
            discount_applied = cost * params.volume_discount_pct / Decimal("100")
            cost = cost - discount_applied
            formula += " * (1 - volume_discount_pct/100)"

        # Round to precision
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["quantity_mj"] = str(quantity_mj)
        step["base_price"] = str(base_price)
        step["discount_applied"] = str(discount_applied)
        step["cost"] = str(cost)
        self._calculation_steps.append(step)

        component = CostComponent(
            category=CostCategory.PURCHASE,
            name=f"Purchase_{params.fuel_id}",
            amount=cost,
            unit=self._currency,
            quantity=quantity_mj,
            rate=base_price,
            description=f"Purchase cost for {params.fuel_id} ({params.pricing_type.value} pricing)",
            calculation_formula=formula,
            input_values={
                "quantity_mj": str(quantity_mj),
                "base_price_per_mj": str(base_price),
                "discount_pct": str(params.volume_discount_pct),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Purchase cost calculated: {cost} for {quantity_mj} MJ of {params.fuel_id}")
        return cost, component

    def calculate_logistics_cost(
        self,
        quantity_mj: Decimal,
        params: LogisticsCostParams,
        num_deliveries: int = 1,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent]:
        """
        Calculate logistics cost for fuel delivery - DETERMINISTIC.

        Formula:
        cost = (quantity * base_rate) + (quantity * distance * distance_rate)
               + (num_deliveries * fixed_fee) + (num_deliveries * port_charges)

        Args:
            quantity_mj: Quantity to deliver (MJ)
            params: Logistics cost parameters
            num_deliveries: Number of deliveries
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component)
        """
        step = {"operation": "calculate_logistics_cost", "mode": params.mode.value}

        # Base transportation cost
        transport_cost = quantity_mj * params.base_rate_per_mj

        # Distance-based cost
        distance_cost = quantity_mj * params.distance_km * params.distance_rate_per_mj_km

        # Fixed delivery fees
        delivery_fees = Decimal(str(num_deliveries)) * params.fixed_delivery_fee

        # Port/terminal charges
        port_charges = Decimal(str(num_deliveries)) * params.port_charges_per_delivery

        # Total logistics cost
        cost = transport_cost + distance_cost + delivery_fees + port_charges
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["transport_cost"] = str(transport_cost)
        step["distance_cost"] = str(distance_cost)
        step["delivery_fees"] = str(delivery_fees)
        step["port_charges"] = str(port_charges)
        step["total_cost"] = str(cost)
        self._calculation_steps.append(step)

        component = CostComponent(
            category=CostCategory.LOGISTICS,
            name=f"Logistics_{params.mode.value}",
            amount=cost,
            unit=self._currency,
            quantity=quantity_mj,
            rate=params.base_rate_per_mj,
            description=f"Logistics cost via {params.mode.value} ({num_deliveries} deliveries)",
            calculation_formula="(qty * base_rate) + (qty * dist * dist_rate) + (del * fixed_fee) + (del * port)",
            input_values={
                "quantity_mj": str(quantity_mj),
                "distance_km": str(params.distance_km),
                "num_deliveries": str(num_deliveries),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Logistics cost calculated: {cost} for {quantity_mj} MJ")
        return cost, component

    def calculate_storage_cost(
        self,
        average_inventory_mj: Decimal,
        days: int,
        params: StorageCostParams,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent]:
        """
        Calculate storage cost for inventory holding - DETERMINISTIC.

        Formula:
        cost = inventory * days * (holding_rate + capital_rate + insurance_daily)

        Args:
            average_inventory_mj: Average inventory level (MJ)
            days: Number of days in period
            params: Storage cost parameters
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component)
        """
        step = {"operation": "calculate_storage_cost", "tank_id": params.tank_id}

        # Daily insurance rate from annual
        insurance_daily = (params.insurance_rate_annual_pct / Decimal("100")) / self.DAYS_PER_YEAR

        # Total daily rate
        daily_rate = params.holding_cost_per_mj_day + params.capital_cost_per_mj_day + insurance_daily

        # Total storage cost
        cost = average_inventory_mj * Decimal(str(days)) * daily_rate
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["average_inventory_mj"] = str(average_inventory_mj)
        step["days"] = days
        step["daily_rate"] = str(daily_rate)
        step["cost"] = str(cost)
        self._calculation_steps.append(step)

        component = CostComponent(
            category=CostCategory.STORAGE,
            name=f"Storage_{params.tank_id}",
            amount=cost,
            unit=self._currency,
            quantity=average_inventory_mj,
            rate=daily_rate,
            description=f"Storage cost for tank {params.tank_id} over {days} days",
            calculation_formula="avg_inventory * days * daily_rate",
            input_values={
                "average_inventory_mj": str(average_inventory_mj),
                "days": str(days),
                "daily_rate": str(daily_rate),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Storage cost calculated: {cost} for {average_inventory_mj} MJ over {days} days")
        return cost, component

    def calculate_loss_cost(
        self,
        inventory_mj: Decimal,
        days: int,
        params: StorageCostParams,
        fuel_price_per_mj: Decimal,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent]:
        """
        Calculate loss cost from inventory shrinkage - DETERMINISTIC.

        Formula:
        loss_quantity = inventory * (1 - (1 - daily_loss_rate)^days)
        cost = loss_quantity * fuel_price

        Args:
            inventory_mj: Starting inventory (MJ)
            days: Number of days in period
            params: Storage cost parameters (includes loss rate)
            fuel_price_per_mj: Fuel value ($/MJ)
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component)
        """
        step = {"operation": "calculate_loss_cost", "tank_id": params.tank_id}

        # Calculate compound loss over period
        daily_rate = params.loss_rate_per_day_pct / Decimal("100")
        retention_factor = (Decimal("1") - daily_rate) ** days
        loss_factor = Decimal("1") - retention_factor

        # Loss quantity
        loss_quantity_mj = inventory_mj * loss_factor

        # Loss cost (value of lost product)
        cost = loss_quantity_mj * fuel_price_per_mj
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["inventory_mj"] = str(inventory_mj)
        step["days"] = days
        step["loss_factor"] = str(loss_factor)
        step["loss_quantity_mj"] = str(loss_quantity_mj)
        step["cost"] = str(cost)
        self._calculation_steps.append(step)

        component = CostComponent(
            category=CostCategory.LOSS,
            name=f"Loss_{params.tank_id}",
            amount=cost,
            unit=self._currency,
            quantity=loss_quantity_mj,
            rate=fuel_price_per_mj,
            description=f"Product loss in tank {params.tank_id} over {days} days",
            calculation_formula="loss_quantity * fuel_price where loss = inv * (1 - (1-rate)^days)",
            input_values={
                "inventory_mj": str(inventory_mj),
                "days": str(days),
                "loss_rate_daily_pct": str(params.loss_rate_per_day_pct),
                "fuel_price_per_mj": str(fuel_price_per_mj),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Loss cost calculated: {cost} ({loss_quantity_mj} MJ lost)")
        return cost, component

    def calculate_penalty_cost(
        self,
        actual_take_mj: Decimal,
        params: ContractPenaltyParams,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent]:
        """
        Calculate contract penalty cost - DETERMINISTIC.

        Formula:
        shortfall_penalty = max(0, min_take - actual) * shortfall_rate
        excess_penalty = max(0, actual - max_take) * excess_rate
        cost = shortfall_penalty + excess_penalty

        Args:
            actual_take_mj: Actual quantity taken (MJ)
            params: Contract penalty parameters
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component)
        """
        step = {"operation": "calculate_penalty_cost", "contract_id": params.contract_id}

        # Calculate shortfall
        shortfall_mj = max(Decimal("0"), params.min_take_mj - actual_take_mj)
        shortfall_penalty = shortfall_mj * params.shortfall_penalty_per_mj

        # Calculate excess
        excess_mj = max(Decimal("0"), actual_take_mj - params.max_take_mj)
        excess_penalty = excess_mj * params.excess_penalty_per_mj

        # Total penalty
        cost = shortfall_penalty + excess_penalty
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["actual_take_mj"] = str(actual_take_mj)
        step["shortfall_mj"] = str(shortfall_mj)
        step["shortfall_penalty"] = str(shortfall_penalty)
        step["excess_mj"] = str(excess_mj)
        step["excess_penalty"] = str(excess_penalty)
        step["cost"] = str(cost)
        self._calculation_steps.append(step)

        penalty_type = "shortfall" if shortfall_mj > 0 else ("excess" if excess_mj > 0 else "none")

        component = CostComponent(
            category=CostCategory.PENALTY,
            name=f"Penalty_{params.contract_id}",
            amount=cost,
            unit=self._currency,
            quantity=shortfall_mj + excess_mj,
            rate=params.shortfall_penalty_per_mj if shortfall_mj > 0 else params.excess_penalty_per_mj,
            description=f"Contract penalty for {params.contract_id} ({penalty_type})",
            calculation_formula="max(0, min_take - actual) * shortfall_rate + max(0, actual - max_take) * excess_rate",
            input_values={
                "actual_take_mj": str(actual_take_mj),
                "min_take_mj": str(params.min_take_mj),
                "max_take_mj": str(params.max_take_mj),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Penalty cost calculated: {cost} for contract {params.contract_id}")
        return cost, component

    def calculate_carbon_cost(
        self,
        energy_mj: Decimal,
        params: CarbonCostParams,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent, Decimal]:
        """
        Calculate carbon cost from emissions - DETERMINISTIC.

        Formula:
        emissions = energy * carbon_intensity
        taxable_emissions = max(0, emissions - free_allowance) * compliance_factor
        cost = taxable_emissions * carbon_price

        Args:
            energy_mj: Energy consumed (MJ)
            params: Carbon cost parameters
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component, emissions_kg_co2e)
        """
        step = {"operation": "calculate_carbon_cost", "scheme": params.scheme.value}

        # Calculate total emissions
        total_emissions = energy_mj * params.carbon_intensity_kg_co2e_mj

        # Apply free allowance
        net_emissions = max(Decimal("0"), total_emissions - params.free_allowance_kg_co2e)

        # Apply compliance factor
        taxable_emissions = net_emissions * params.compliance_factor

        # Calculate carbon cost
        cost = taxable_emissions * params.carbon_price_per_kg_co2e
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["energy_mj"] = str(energy_mj)
        step["total_emissions_kg"] = str(total_emissions)
        step["net_emissions_kg"] = str(net_emissions)
        step["taxable_emissions_kg"] = str(taxable_emissions)
        step["cost"] = str(cost)
        self._calculation_steps.append(step)

        component = CostComponent(
            category=CostCategory.CARBON,
            name=f"Carbon_{params.scheme.value}",
            amount=cost,
            unit=self._currency,
            quantity=taxable_emissions,
            rate=params.carbon_price_per_kg_co2e,
            description=f"Carbon cost under {params.scheme.value} scheme",
            calculation_formula="max(0, emissions - free_allowance) * compliance_factor * carbon_price",
            input_values={
                "energy_mj": str(energy_mj),
                "carbon_intensity": str(params.carbon_intensity_kg_co2e_mj),
                "carbon_price": str(params.carbon_price_per_kg_co2e),
                "free_allowance_kg": str(params.free_allowance_kg_co2e),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Carbon cost calculated: {cost} for {total_emissions} kgCO2e")
        return cost, component, total_emissions

    def calculate_risk_cost(
        self,
        base_cost: Decimal,
        params: RiskCostParams,
        period: Optional[int] = None
    ) -> Tuple[Decimal, CostComponent]:
        """
        Calculate risk premium cost - DETERMINISTIC.

        Formula:
        risk_premium = base_cost * (volatility_pct + supply_risk_pct +
                       counterparty_risk_pct + hedging_cost_pct) / 100

        Args:
            base_cost: Base cost to apply premium to
            params: Risk cost parameters
            period: Optional period identifier

        Returns:
            Tuple of (cost_amount, cost_component)
        """
        step = {"operation": "calculate_risk_cost"}

        # Calculate total risk premium percentage
        total_risk_pct = (
            params.volatility_premium_pct +
            params.supply_risk_premium_pct +
            params.counterparty_risk_pct +
            params.hedging_cost_pct
        )

        # Calculate risk cost
        cost = base_cost * total_risk_pct / Decimal("100")
        cost = cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        step["base_cost"] = str(base_cost)
        step["total_risk_pct"] = str(total_risk_pct)
        step["cost"] = str(cost)
        self._calculation_steps.append(step)

        component = CostComponent(
            category=CostCategory.RISK,
            name="Risk_Premium",
            amount=cost,
            unit=self._currency,
            quantity=base_cost,
            rate=total_risk_pct / Decimal("100"),
            description="Risk premium on base procurement cost",
            calculation_formula="base_cost * (vol + supply + cpty + hedge) / 100",
            input_values={
                "base_cost": str(base_cost),
                "volatility_pct": str(params.volatility_premium_pct),
                "supply_risk_pct": str(params.supply_risk_premium_pct),
                "counterparty_risk_pct": str(params.counterparty_risk_pct),
                "hedging_cost_pct": str(params.hedging_cost_pct),
                "period": str(period) if period else "N/A"
            }
        )

        logger.debug(f"Risk cost calculated: {cost} ({total_risk_pct}% premium)")
        return cost, component

    def calculate_total_cost(
        self,
        purchase_components: List[Tuple[Decimal, CostComponent]],
        logistics_components: List[Tuple[Decimal, CostComponent]],
        storage_components: List[Tuple[Decimal, CostComponent]],
        loss_components: List[Tuple[Decimal, CostComponent]],
        penalty_components: List[Tuple[Decimal, CostComponent]],
        carbon_components: List[Tuple[Decimal, CostComponent, Decimal]],
        risk_components: List[Tuple[Decimal, CostComponent]],
        total_energy_mj: Decimal,
        period_count: int = 1
    ) -> CostBreakdown:
        """
        Calculate total cost breakdown from all components - DETERMINISTIC.

        Aggregates all cost components into a complete cost breakdown
        with full provenance tracking.

        Args:
            purchase_components: List of purchase cost tuples
            logistics_components: List of logistics cost tuples
            storage_components: List of storage cost tuples
            loss_components: List of loss cost tuples
            penalty_components: List of penalty cost tuples
            carbon_components: List of carbon cost tuples (includes emissions)
            risk_components: List of risk cost tuples
            total_energy_mj: Total energy for average cost calculation
            period_count: Number of time periods

        Returns:
            Complete CostBreakdown with all details
        """
        logger.info("Calculating total cost breakdown")

        # Sum by category
        purchase_total = sum(c[0] for c in purchase_components)
        logistics_total = sum(c[0] for c in logistics_components)
        storage_total = sum(c[0] for c in storage_components)
        loss_total = sum(c[0] for c in loss_components)
        penalty_total = sum(c[0] for c in penalty_components)
        carbon_total = sum(c[0] for c in carbon_components)
        risk_total = sum(c[0] for c in risk_components)

        # Grand total
        total = (purchase_total + logistics_total + storage_total +
                 loss_total + penalty_total + carbon_total + risk_total)

        # Collect all components
        all_components: List[CostComponent] = []
        all_components.extend([c[1] for c in purchase_components])
        all_components.extend([c[1] for c in logistics_components])
        all_components.extend([c[1] for c in storage_components])
        all_components.extend([c[1] for c in loss_components])
        all_components.extend([c[1] for c in penalty_components])
        all_components.extend([c[1] for c in carbon_components])
        all_components.extend([c[1] for c in risk_components])

        # Aggregate by fuel
        cost_by_fuel: Dict[str, Decimal] = {}
        for comp in all_components:
            if comp.category == CostCategory.PURCHASE:
                fuel_id = comp.name.replace("Purchase_", "")
                cost_by_fuel[fuel_id] = cost_by_fuel.get(fuel_id, Decimal("0")) + comp.amount

        # Aggregate by period
        cost_by_period: Dict[int, Decimal] = {}
        for comp in all_components:
            period_str = comp.input_values.get("period", "N/A")
            if period_str != "N/A":
                period = int(period_str)
                cost_by_period[period] = cost_by_period.get(period, Decimal("0")) + comp.amount

        # Total emissions
        total_emissions = sum(c[2] for c in carbon_components)
        carbon_intensity = total_emissions / total_energy_mj if total_energy_mj > 0 else Decimal("0")

        breakdown = CostBreakdown(
            purchase_cost=purchase_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            logistics_cost=logistics_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            storage_cost=storage_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            loss_cost=loss_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            penalty_cost=penalty_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            carbon_cost=carbon_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            risk_cost=risk_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_cost=total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            components=all_components,
            cost_by_fuel=cost_by_fuel,
            cost_by_period=cost_by_period,
            currency=self._currency,
            period_count=period_count,
            total_energy_mj=total_energy_mj,
            total_emissions_kg_co2e=total_emissions.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            carbon_intensity_kg_co2e_mj=carbon_intensity.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            calculation_steps=self._calculation_steps.copy()
        )

        logger.info(f"Total cost calculated: {total} for {total_energy_mj} MJ")
        return breakdown

    def clear_steps(self) -> None:
        """Clear calculation steps for new run."""
        self._calculation_steps = []

    def get_calculation_steps(self) -> List[Dict[str, Any]]:
        """Get recorded calculation steps for audit."""
        return self._calculation_steps.copy()
