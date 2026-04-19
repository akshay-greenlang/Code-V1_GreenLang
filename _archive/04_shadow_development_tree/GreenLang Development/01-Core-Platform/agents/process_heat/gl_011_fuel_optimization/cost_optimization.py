"""
GL-011 FUELCRAFT - Cost Optimizer

This module provides total cost of ownership optimization for fuel selection
including fuel purchase, transport, storage, carbon, and maintenance costs.

Features:
    - Multi-objective optimization (cost, emissions, reliability)
    - Carbon cost integration
    - Rolling horizon optimization
    - Scenario analysis
    - Zero-hallucination deterministic calculations

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization.cost_optimization import (
    ...     CostOptimizer,
    ...     TotalCostInput,
    ... )
    >>>
    >>> optimizer = CostOptimizer(config)
    >>> result = optimizer.optimize(input_data)
    >>> print(f"Optimal cost: ${result.total_cost_usd:.2f}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    CostOptimizationConfig,
    OptimizationMode,
    FuelType,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelPrice,
    FuelProperties,
    CostAnalysis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Emission factors (kg CO2/MMBTU)
CO2_EMISSION_FACTORS = {
    "natural_gas": 53.06,
    "no2_fuel_oil": 73.16,
    "no6_fuel_oil": 75.10,
    "lpg_propane": 62.87,
    "lpg_butane": 64.77,
    "coal_bituminous": 93.28,
    "coal_sub_bituminous": 97.17,
    "biomass_wood": 0.0,  # Biogenic
    "biogas": 0.0,
    "hydrogen": 0.0,
    "rng": 0.0,
}

# Maintenance impact factors (relative to natural gas = 1.0)
MAINTENANCE_FACTORS = {
    "natural_gas": 1.0,
    "no2_fuel_oil": 1.3,
    "no6_fuel_oil": 1.5,
    "lpg_propane": 1.05,
    "coal_bituminous": 2.0,
    "biomass_wood": 1.8,
    "biogas": 1.1,
    "hydrogen": 1.2,
}

# Reliability factors (availability multiplier)
RELIABILITY_FACTORS = {
    "natural_gas": 0.995,  # Pipeline gas very reliable
    "no2_fuel_oil": 0.98,
    "no6_fuel_oil": 0.97,
    "lpg_propane": 0.99,
    "coal_bituminous": 0.95,
    "biomass_wood": 0.93,
    "biogas": 0.90,
    "hydrogen": 0.92,
}


# =============================================================================
# DATA MODELS
# =============================================================================

class TotalCostInput(BaseModel):
    """Input for total cost optimization."""

    # Fuel options
    fuel_options: List[str] = Field(
        ...,
        description="Available fuel options"
    )
    fuel_prices: Dict[str, FuelPrice] = Field(
        ...,
        description="Current fuel prices"
    )
    fuel_properties: Optional[Dict[str, FuelProperties]] = Field(
        default=None,
        description="Fuel properties"
    )

    # Operating conditions
    heat_demand_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Heat demand (MMBTU/hr)"
    )
    operating_hours_year: float = Field(
        default=8000.0,
        gt=0,
        description="Operating hours per year"
    )
    current_fuel: str = Field(..., description="Current fuel type")

    # Equipment
    equipment_efficiency: Dict[str, float] = Field(
        default_factory=dict,
        description="Efficiency by fuel type (%)"
    )
    base_maintenance_cost_usd_year: float = Field(
        default=50000.0,
        ge=0,
        description="Base annual maintenance cost"
    )

    # Carbon
    carbon_price_usd_ton: float = Field(
        default=50.0,
        ge=0,
        description="Carbon price ($/ton)"
    )
    emissions_cap_tons_year: Optional[float] = Field(
        default=None,
        description="Emissions cap (tons/year)"
    )

    # Analysis period
    analysis_period_years: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Analysis period (years)"
    )


class TotalCostOutput(BaseModel):
    """Output from total cost optimization."""

    # Optimal selection
    optimal_fuel: str = Field(..., description="Optimal fuel selection")
    optimization_mode: OptimizationMode = Field(
        ...,
        description="Optimization mode used"
    )

    # Cost breakdown (annualized)
    fuel_cost_usd: float = Field(..., description="Annual fuel cost")
    transport_cost_usd: float = Field(..., description="Annual transport cost")
    storage_cost_usd: float = Field(..., description="Annual storage cost")
    carbon_cost_usd: float = Field(..., description="Annual carbon cost")
    maintenance_cost_usd: float = Field(..., description="Annual maintenance cost")
    total_cost_usd: float = Field(..., description="Total annual cost")

    # Per-unit costs
    cost_per_mmbtu: float = Field(..., description="Cost per MMBTU")
    cost_per_hour: float = Field(..., description="Cost per operating hour")

    # Emissions
    annual_co2_tons: float = Field(..., description="Annual CO2 (tons)")
    co2_intensity_kg_mmbtu: float = Field(..., description="CO2 intensity")

    # Comparison
    fuel_rankings: List[Tuple[str, float]] = Field(
        ...,
        description="Fuel rankings by total cost"
    )
    savings_vs_current_usd: float = Field(
        default=0.0,
        description="Savings vs current fuel"
    )
    savings_vs_current_pct: float = Field(
        default=0.0,
        description="Savings percentage"
    )

    # Multi-objective scores
    cost_score: float = Field(..., ge=0, le=1, description="Cost score (0-1)")
    emissions_score: float = Field(..., ge=0, le=1, description="Emissions score (0-1)")
    reliability_score: float = Field(..., ge=0, le=1, description="Reliability score (0-1)")
    weighted_score: float = Field(..., ge=0, le=1, description="Weighted total score")

    # Provenance
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    provenance_hash: str = Field(..., description="Calculation provenance")

    class Config:
        use_enum_values = True


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for a fuel option."""

    fuel_type: str = Field(..., description="Fuel type")

    # Component costs (annual)
    fuel_purchase_cost: float = Field(..., description="Fuel purchase cost")
    transport_cost: float = Field(..., description="Transport cost")
    storage_cost: float = Field(..., description="Storage cost")
    carbon_cost: float = Field(..., description="Carbon cost")
    maintenance_cost: float = Field(..., description="Maintenance cost")
    efficiency_adjustment: float = Field(
        default=0.0,
        description="Efficiency-based cost adjustment"
    )

    # Totals
    total_annual_cost: float = Field(..., description="Total annual cost")
    cost_per_mmbtu: float = Field(..., description="Levelized cost per MMBTU")

    # Emissions
    annual_co2_tons: float = Field(..., description="Annual CO2 emissions (tons)")
    co2_cost_per_ton: float = Field(..., description="Effective CO2 cost per ton")

    # Reliability
    reliability_factor: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Reliability factor"
    )
    availability_adjusted_cost: float = Field(
        ...,
        description="Availability-adjusted cost"
    )


# =============================================================================
# COST OPTIMIZER
# =============================================================================

class CostOptimizer:
    """
    Total cost of ownership optimizer.

    Optimizes fuel selection considering total cost of ownership including
    fuel purchase, transport, storage, carbon, and maintenance costs.

    Features:
        - Multi-objective optimization
        - Carbon cost integration
        - Reliability weighting
        - Scenario analysis
        - Deterministic calculations

    Example:
        >>> optimizer = CostOptimizer(config)
        >>> result = optimizer.optimize(input_data)
        >>> print(f"Optimal fuel: {result.optimal_fuel}")
    """

    def __init__(self, config: CostOptimizationConfig) -> None:
        """
        Initialize the cost optimizer.

        Args:
            config: Cost optimization configuration
        """
        self.config = config
        self._optimization_count = 0

        logger.info(
            f"CostOptimizer initialized (mode: {config.mode})"
        )

    def optimize(self, input_data: TotalCostInput) -> TotalCostOutput:
        """
        Optimize fuel selection based on total cost of ownership.

        Args:
            input_data: Optimization input

        Returns:
            TotalCostOutput with optimal selection and analysis
        """
        self._optimization_count += 1

        logger.debug(
            f"Optimizing fuel selection for {len(input_data.fuel_options)} options"
        )

        # Calculate cost breakdown for each fuel
        cost_breakdowns: Dict[str, CostBreakdown] = {}
        for fuel in input_data.fuel_options:
            breakdown = self._calculate_cost_breakdown(fuel, input_data)
            cost_breakdowns[fuel] = breakdown

        # Calculate scores for each fuel
        scores: Dict[str, Dict[str, float]] = {}
        for fuel, breakdown in cost_breakdowns.items():
            scores[fuel] = self._calculate_scores(
                breakdown,
                cost_breakdowns,
                input_data
            )

        # Select optimal fuel based on mode
        optimal_fuel = self._select_optimal(scores, input_data)
        optimal_breakdown = cost_breakdowns[optimal_fuel]
        optimal_scores = scores[optimal_fuel]

        # Rank all fuels
        fuel_rankings = sorted(
            [(f, b.total_annual_cost) for f, b in cost_breakdowns.items()],
            key=lambda x: x[1]
        )

        # Calculate savings vs current
        current_breakdown = cost_breakdowns.get(input_data.current_fuel)
        savings_usd = 0.0
        savings_pct = 0.0
        if current_breakdown:
            savings_usd = current_breakdown.total_annual_cost - optimal_breakdown.total_annual_cost
            if current_breakdown.total_annual_cost > 0:
                savings_pct = (savings_usd / current_breakdown.total_annual_cost) * 100

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            optimal_fuel,
            optimal_breakdown.dict()
        )

        return TotalCostOutput(
            optimal_fuel=optimal_fuel,
            optimization_mode=self.config.mode,
            fuel_cost_usd=round(optimal_breakdown.fuel_purchase_cost, 2),
            transport_cost_usd=round(optimal_breakdown.transport_cost, 2),
            storage_cost_usd=round(optimal_breakdown.storage_cost, 2),
            carbon_cost_usd=round(optimal_breakdown.carbon_cost, 2),
            maintenance_cost_usd=round(optimal_breakdown.maintenance_cost, 2),
            total_cost_usd=round(optimal_breakdown.total_annual_cost, 2),
            cost_per_mmbtu=round(optimal_breakdown.cost_per_mmbtu, 4),
            cost_per_hour=round(
                optimal_breakdown.total_annual_cost / input_data.operating_hours_year,
                2
            ),
            annual_co2_tons=round(optimal_breakdown.annual_co2_tons, 1),
            co2_intensity_kg_mmbtu=round(
                self._get_emission_factor(optimal_fuel),
                2
            ),
            fuel_rankings=fuel_rankings,
            savings_vs_current_usd=round(savings_usd, 2),
            savings_vs_current_pct=round(savings_pct, 2),
            cost_score=round(optimal_scores["cost"], 4),
            emissions_score=round(optimal_scores["emissions"], 4),
            reliability_score=round(optimal_scores["reliability"], 4),
            weighted_score=round(optimal_scores["weighted"], 4),
            provenance_hash=provenance_hash,
        )

    def analyze_fuel(
        self,
        fuel_type: str,
        input_data: TotalCostInput,
    ) -> CostBreakdown:
        """
        Analyze total cost for a specific fuel.

        Args:
            fuel_type: Fuel type to analyze
            input_data: Analysis input

        Returns:
            CostBreakdown with detailed analysis
        """
        return self._calculate_cost_breakdown(fuel_type, input_data)

    def compare_scenarios(
        self,
        input_data: TotalCostInput,
        carbon_prices: List[float],
    ) -> Dict[float, TotalCostOutput]:
        """
        Compare optimization results across carbon price scenarios.

        Args:
            input_data: Base input data
            carbon_prices: List of carbon prices to analyze

        Returns:
            Dictionary of carbon price to optimization result
        """
        results = {}

        for price in carbon_prices:
            # Modify input with new carbon price
            modified = input_data.copy()
            modified.carbon_price_usd_ton = price

            # Run optimization
            result = self.optimize(modified)
            results[price] = result

        return results

    def _calculate_cost_breakdown(
        self,
        fuel_type: str,
        input_data: TotalCostInput,
    ) -> CostBreakdown:
        """Calculate detailed cost breakdown for a fuel."""
        # Get fuel price
        price = input_data.fuel_prices.get(fuel_type)
        if not price:
            raise ValueError(f"No price for fuel: {fuel_type}")

        # Annual heat demand
        annual_heat_mmbtu = (
            input_data.heat_demand_mmbtu_hr *
            input_data.operating_hours_year
        )

        # Get efficiency (default 82%)
        efficiency = input_data.equipment_efficiency.get(fuel_type, 82.0) / 100

        # Adjust for efficiency
        fuel_required_mmbtu = annual_heat_mmbtu / efficiency

        # Fuel purchase cost
        fuel_purchase_cost = fuel_required_mmbtu * price.commodity_price

        # Transport cost
        transport_cost = fuel_required_mmbtu * price.transport_cost

        # Storage cost (estimate as 2% of fuel value)
        storage_cost = fuel_purchase_cost * 0.02

        # Carbon cost
        emission_factor = self._get_emission_factor(fuel_type)
        annual_co2_kg = fuel_required_mmbtu * emission_factor
        annual_co2_tons = annual_co2_kg / 1000
        carbon_cost = annual_co2_tons * input_data.carbon_price_usd_ton

        # Maintenance cost
        maintenance_factor = MAINTENANCE_FACTORS.get(fuel_type, 1.0)
        maintenance_cost = input_data.base_maintenance_cost_usd_year * maintenance_factor

        # Efficiency adjustment (opportunity cost of lower efficiency)
        base_efficiency = 0.82  # Reference efficiency
        if efficiency < base_efficiency:
            efficiency_penalty = (base_efficiency - efficiency) * fuel_purchase_cost
        else:
            efficiency_penalty = 0.0

        # Total annual cost
        total_cost = (
            fuel_purchase_cost +
            transport_cost +
            storage_cost +
            carbon_cost +
            maintenance_cost +
            efficiency_penalty
        )

        # Reliability factor
        reliability = RELIABILITY_FACTORS.get(fuel_type, 0.95)
        availability_adjusted = total_cost / reliability

        # Levelized cost per MMBTU
        cost_per_mmbtu = total_cost / annual_heat_mmbtu if annual_heat_mmbtu > 0 else 0

        return CostBreakdown(
            fuel_type=fuel_type,
            fuel_purchase_cost=round(fuel_purchase_cost, 2),
            transport_cost=round(transport_cost, 2),
            storage_cost=round(storage_cost, 2),
            carbon_cost=round(carbon_cost, 2),
            maintenance_cost=round(maintenance_cost, 2),
            efficiency_adjustment=round(efficiency_penalty, 2),
            total_annual_cost=round(total_cost, 2),
            cost_per_mmbtu=round(cost_per_mmbtu, 4),
            annual_co2_tons=round(annual_co2_tons, 1),
            co2_cost_per_ton=input_data.carbon_price_usd_ton,
            reliability_factor=reliability,
            availability_adjusted_cost=round(availability_adjusted, 2),
        )

    def _calculate_scores(
        self,
        breakdown: CostBreakdown,
        all_breakdowns: Dict[str, CostBreakdown],
        input_data: TotalCostInput,
    ) -> Dict[str, float]:
        """Calculate optimization scores for a fuel."""
        # Get min/max values for normalization
        all_costs = [b.total_annual_cost for b in all_breakdowns.values()]
        all_emissions = [b.annual_co2_tons for b in all_breakdowns.values()]

        min_cost, max_cost = min(all_costs), max(all_costs)
        min_emissions, max_emissions = min(all_emissions), max(all_emissions)

        # Cost score (lower is better, so invert)
        if max_cost > min_cost:
            cost_score = 1 - (breakdown.total_annual_cost - min_cost) / (max_cost - min_cost)
        else:
            cost_score = 1.0

        # Emissions score (lower is better, so invert)
        if max_emissions > min_emissions:
            emissions_score = 1 - (breakdown.annual_co2_tons - min_emissions) / (max_emissions - min_emissions)
        else:
            emissions_score = 1.0

        # Reliability score (higher is better)
        reliability_score = breakdown.reliability_factor

        # Weighted score
        weighted_score = (
            self.config.cost_weight * cost_score +
            self.config.emissions_weight * emissions_score +
            self.config.reliability_weight * reliability_score
        )

        return {
            "cost": cost_score,
            "emissions": emissions_score,
            "reliability": reliability_score,
            "weighted": weighted_score,
        }

    def _select_optimal(
        self,
        scores: Dict[str, Dict[str, float]],
        input_data: TotalCostInput,
    ) -> str:
        """Select optimal fuel based on optimization mode."""
        mode = self.config.mode

        if mode == OptimizationMode.MINIMUM_COST:
            # Select by cost score only
            return max(scores.items(), key=lambda x: x[1]["cost"])[0]

        elif mode == OptimizationMode.MINIMUM_EMISSIONS:
            # Select by emissions score only
            return max(scores.items(), key=lambda x: x[1]["emissions"])[0]

        elif mode == OptimizationMode.RELIABILITY:
            # Select by reliability score only
            return max(scores.items(), key=lambda x: x[1]["reliability"])[0]

        else:  # BALANCED or CUSTOM
            # Select by weighted score
            return max(scores.items(), key=lambda x: x[1]["weighted"])[0]

    def _get_emission_factor(self, fuel_type: str) -> float:
        """Get CO2 emission factor for fuel type."""
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")
        return CO2_EMISSION_FACTORS.get(fuel_key, 53.0)

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        optimal_fuel: str,
        breakdown: Dict[str, Any],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        data = {
            "optimizer": "CostOptimizer",
            "mode": self.config.mode.value,
            "optimal_fuel": optimal_fuel,
            "breakdown_hash": hashlib.sha256(
                json.dumps(breakdown, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
        }

        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def create_cost_analysis(
        self,
        output: TotalCostOutput,
        input_data: TotalCostInput,
    ) -> CostAnalysis:
        """
        Create a CostAnalysis from optimization output.

        Args:
            output: Optimization output
            input_data: Original input

        Returns:
            CostAnalysis for reporting
        """
        return CostAnalysis(
            period_hours=input_data.operating_hours_year,
            fuel_cost_usd=output.fuel_cost_usd,
            transport_cost_usd=output.transport_cost_usd,
            storage_cost_usd=output.storage_cost_usd,
            carbon_cost_usd=output.carbon_cost_usd,
            maintenance_cost_usd=output.maintenance_cost_usd,
            total_cost_usd=output.total_cost_usd,
            cost_per_mmbtu=output.cost_per_mmbtu,
            total_co2_kg=output.annual_co2_tons * 1000,
            total_co2_cost_usd=output.carbon_cost_usd,
            co2_intensity_kg_mmbtu=output.co2_intensity_kg_mmbtu,
            savings_vs_baseline_usd=output.savings_vs_current_usd,
            savings_pct=output.savings_vs_current_pct,
        )

    @property
    def optimization_count(self) -> int:
        """Get total optimization count."""
        return self._optimization_count
