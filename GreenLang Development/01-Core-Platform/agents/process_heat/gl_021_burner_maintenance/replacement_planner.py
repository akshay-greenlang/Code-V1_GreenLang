# -*- coding: utf-8 -*-
"""
GL-021 BurnerSentry Agent - Replacement Planner Module

This module implements optimal burner replacement scheduling with comprehensive
economic analysis. It provides NPV/IRR calculations, optimal timing analysis,
group replacement strategies, and spare parts inventory optimization.

Key Capabilities:
    - Economic replacement analysis (NPV, IRR, payback period)
    - Optimal replacement timing using cost minimization
    - Weibull-based failure probability integration
    - Monte Carlo simulation for uncertainty quantification
    - Group replacement strategies for economies of scale
    - Spare parts inventory optimization (ABC analysis)
    - Turnaround/outage coordination

Reference Standards:
    - API 560 Fired Heaters for General Refinery Service
    - NFPA 86 Standard for Ovens and Furnaces
    - ISO 14224 Petroleum/Natural Gas Industries Reliability Data

ZERO HALLUCINATION: All calculations use deterministic formulas with
full provenance tracking. No ML in calculation path.

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.replacement_planner import (
    ...     ReplacementPlanner, BurnerAsset, EconomicParameters
    ... )
    >>> planner = ReplacementPlanner(config)
    >>> result = planner.analyze_replacement(burner, economic_params)
    >>> print(f"Optimal replacement: {result.optimal_replacement_date}")
    >>> print(f"NPV of replacement: ${result.npv_replacement:,.0f}")

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math
import random
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class BurnerType(str, Enum):
    """Burner types for industrial furnaces."""
    PREMIX = "premix"
    NOZZLE_MIX = "nozzle_mix"
    RAW_GAS = "raw_gas"
    FLAT_FLAME = "flat_flame"
    RADIANT_WALL = "radiant_wall"
    LOW_NOX = "low_nox"
    ULTRA_LOW_NOX = "ultra_low_nox"
    STAGED_AIR = "staged_air"
    STAGED_FUEL = "staged_fuel"


class ReplacementStrategy(str, Enum):
    """Replacement strategy options."""
    RUN_TO_FAILURE = "run_to_failure"
    AGE_BASED = "age_based"
    CONDITION_BASED = "condition_based"
    GROUP_REPLACEMENT = "group_replacement"
    BLOCK_REPLACEMENT = "block_replacement"
    OPPORTUNISTIC = "opportunistic"


class CriticalityLevel(str, Enum):
    """Equipment criticality levels."""
    CRITICAL = "critical"      # Production-critical, safety-critical
    HIGH = "high"              # Major production impact
    MEDIUM = "medium"          # Moderate impact
    LOW = "low"                # Minimal impact


class SparePartCategory(str, Enum):
    """ABC analysis categories for spare parts."""
    A_CRITICAL = "A"           # Critical, high value (70-80% of value)
    B_IMPORTANT = "B"          # Important (15-25% of value)
    C_ROUTINE = "C"            # Routine (5-10% of value)


class OutageType(str, Enum):
    """Types of maintenance outages."""
    UNPLANNED = "unplanned"
    PLANNED_MINOR = "planned_minor"
    PLANNED_MAJOR = "planned_major"
    TURNAROUND = "turnaround"


# =============================================================================
# DATA MODELS
# =============================================================================

class BurnerAsset(BaseModel):
    """Burner asset information for replacement planning."""

    asset_id: str = Field(..., description="Unique asset identifier")
    tag_number: str = Field(..., description="Plant tag number")
    burner_type: BurnerType = Field(..., description="Burner type")
    manufacturer: str = Field(default="", description="Manufacturer")
    model: str = Field(default="", description="Model number")

    # Installation and age
    installation_date: datetime = Field(..., description="Installation date")
    current_age_hours: float = Field(..., ge=0, description="Current operating hours")
    expected_life_hours: float = Field(
        default=50000, gt=0, description="Expected design life (hours)"
    )

    # Capacity and performance
    heat_input_mmbtu_hr: float = Field(
        ..., gt=0, description="Design heat input (MMBtu/hr)"
    )
    current_efficiency_pct: float = Field(
        default=85.0, ge=0, le=100, description="Current thermal efficiency (%)"
    )
    design_efficiency_pct: float = Field(
        default=90.0, ge=0, le=100, description="Design thermal efficiency (%)"
    )

    # Costs
    original_cost: float = Field(..., ge=0, description="Original purchase cost ($)")
    replacement_cost: float = Field(..., ge=0, description="Current replacement cost ($)")
    installation_cost: float = Field(default=0, ge=0, description="Installation cost ($)")

    # Criticality
    criticality: CriticalityLevel = Field(
        default=CriticalityLevel.MEDIUM, description="Equipment criticality"
    )
    process_unit: str = Field(default="", description="Process unit identifier")

    # Weibull parameters (from historical data)
    weibull_beta: float = Field(default=2.5, gt=0, description="Weibull shape parameter")
    weibull_eta: float = Field(
        default=50000, gt=0, description="Weibull scale parameter (hours)"
    )

    # Emissions
    nox_emissions_lb_mmbtu: float = Field(
        default=0.1, ge=0, description="Current NOx emissions (lb/MMBtu)"
    )
    permit_nox_limit_lb_mmbtu: float = Field(
        default=0.15, ge=0, description="Permit NOx limit (lb/MMBtu)"
    )

    @property
    def efficiency_degradation_pct(self) -> float:
        """Calculate efficiency degradation from design."""
        return self.design_efficiency_pct - self.current_efficiency_pct

    @property
    def age_factor(self) -> float:
        """Calculate age as fraction of expected life."""
        return self.current_age_hours / self.expected_life_hours if self.expected_life_hours > 0 else 1.0


class EconomicParameters(BaseModel):
    """Economic parameters for replacement analysis."""

    discount_rate: float = Field(
        default=0.10, ge=0, le=0.50, description="Discount rate (10% = 0.10)"
    )
    analysis_horizon_years: int = Field(
        default=10, ge=1, le=30, description="Analysis time horizon (years)"
    )
    operating_hours_per_year: float = Field(
        default=8000, gt=0, le=8760, description="Operating hours per year"
    )

    # Fuel costs
    fuel_cost_per_mmbtu: float = Field(
        default=5.0, ge=0, description="Fuel cost ($/MMBtu)"
    )
    fuel_cost_escalation_rate: float = Field(
        default=0.02, ge=-0.1, le=0.2, description="Annual fuel cost escalation"
    )

    # Maintenance costs
    preventive_maint_cost_per_year: float = Field(
        default=5000, ge=0, description="Annual PM cost ($)"
    )
    maint_cost_escalation_factor: float = Field(
        default=1.5, ge=1.0, le=5.0, description="Maintenance escalation with age"
    )

    # Failure costs
    unplanned_failure_cost: float = Field(
        default=50000, ge=0, description="Cost per unplanned failure ($)"
    )
    lost_production_cost_per_hour: float = Field(
        default=10000, ge=0, description="Lost production cost ($/hour)"
    )
    avg_repair_time_hours: float = Field(
        default=24, ge=0, description="Average repair time (hours)"
    )

    # Emissions costs
    nox_emission_cost_per_ton: float = Field(
        default=20000, ge=0, description="NOx emission cost ($/ton)"
    )
    carbon_cost_per_ton: float = Field(
        default=50, ge=0, description="Carbon cost ($/ton CO2)"
    )

    # Efficiency improvement
    new_efficiency_pct: float = Field(
        default=92.0, ge=0, le=100, description="New burner efficiency (%)"
    )
    new_nox_lb_mmbtu: float = Field(
        default=0.03, ge=0, description="New burner NOx (lb/MMBtu)"
    )

    # Tax and depreciation
    tax_rate: float = Field(
        default=0.25, ge=0, le=0.5, description="Corporate tax rate"
    )
    depreciation_years: int = Field(
        default=7, ge=1, le=20, description="Depreciation period (years)"
    )


class FailureCostModel(BaseModel):
    """Model for failure cost estimation."""

    direct_repair_cost: float = Field(..., ge=0, description="Direct repair cost ($)")
    lost_production_cost: float = Field(..., ge=0, description="Lost production ($)")
    environmental_penalty: float = Field(default=0, ge=0, description="Environmental penalty ($)")
    safety_investigation_cost: float = Field(default=0, ge=0, description="Safety investigation ($)")
    total_failure_cost: float = Field(..., ge=0, description="Total failure cost ($)")

    @validator("total_failure_cost", always=True)
    def calculate_total(cls, v, values):
        """Calculate total failure cost."""
        return (
            values.get("direct_repair_cost", 0) +
            values.get("lost_production_cost", 0) +
            values.get("environmental_penalty", 0) +
            values.get("safety_investigation_cost", 0)
        )


class ReplacementAnalysisResult(BaseModel):
    """Result of replacement analysis."""

    asset_id: str = Field(..., description="Asset identifier")
    analysis_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Economic metrics
    npv_continue: float = Field(..., description="NPV of continuing operation ($)")
    npv_replacement: float = Field(..., description="NPV of replacement ($)")
    npv_advantage: float = Field(..., description="NPV advantage of replacement ($)")
    irr_replacement: Optional[float] = Field(
        default=None, description="IRR of replacement (%)"
    )
    payback_period_years: Optional[float] = Field(
        default=None, description="Simple payback period (years)"
    )

    # Optimal timing
    optimal_replacement_date: datetime = Field(
        ..., description="Optimal replacement date"
    )
    optimal_replacement_hours: float = Field(
        ..., description="Optimal replacement age (hours)"
    )

    # Cost breakdown
    annual_savings: float = Field(..., description="Annual savings from replacement ($)")
    fuel_savings_annual: float = Field(..., description="Annual fuel savings ($)")
    maintenance_savings_annual: float = Field(..., description="Annual maintenance savings ($)")
    emissions_savings_annual: float = Field(..., description="Annual emissions savings ($)")
    avoided_failure_cost_annual: float = Field(..., description="Annual avoided failure cost ($)")

    # Failure risk
    current_failure_probability: float = Field(
        ..., ge=0, le=1, description="Current failure probability"
    )
    expected_failures_next_year: float = Field(
        ..., ge=0, description="Expected failures in next year"
    )

    # Recommendation
    recommendation: str = Field(..., description="Replacement recommendation")
    confidence_level: float = Field(
        ..., ge=0, le=1, description="Analysis confidence"
    )

    # Sensitivity analysis
    sensitivity_results: Dict[str, float] = Field(
        default_factory=dict, description="Sensitivity analysis results"
    )

    # Monte Carlo results
    monte_carlo_npv_mean: Optional[float] = Field(
        default=None, description="Monte Carlo mean NPV"
    )
    monte_carlo_npv_p10: Optional[float] = Field(
        default=None, description="Monte Carlo P10 NPV"
    )
    monte_carlo_npv_p90: Optional[float] = Field(
        default=None, description="Monte Carlo P90 NPV"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class GroupReplacementResult(BaseModel):
    """Result of group replacement analysis."""

    group_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Group identifier"
    )
    analysis_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Group members
    asset_ids: List[str] = Field(..., description="Assets in group")
    total_assets: int = Field(..., description="Total number of assets")

    # Economics
    individual_replacement_cost: float = Field(
        ..., description="Cost of individual replacements ($)"
    )
    group_replacement_cost: float = Field(
        ..., description="Cost of group replacement ($)"
    )
    group_savings: float = Field(
        ..., description="Savings from group replacement ($)"
    )
    savings_percentage: float = Field(
        ..., description="Percentage savings from grouping"
    )

    # Timing
    recommended_replacement_date: datetime = Field(
        ..., description="Recommended group replacement date"
    )

    # Strategy
    strategy: ReplacementStrategy = Field(
        ..., description="Recommended replacement strategy"
    )

    recommendation: str = Field(..., description="Group replacement recommendation")


class SparePartRecommendation(BaseModel):
    """Spare part inventory recommendation."""

    part_number: str = Field(..., description="Part number")
    description: str = Field(..., description="Part description")
    category: SparePartCategory = Field(..., description="ABC category")

    # Quantity recommendations
    current_quantity: int = Field(default=0, ge=0, description="Current stock")
    recommended_quantity: int = Field(..., ge=0, description="Recommended stock")
    reorder_point: int = Field(..., ge=0, description="Reorder point")
    order_quantity: int = Field(..., ge=0, description="Order quantity")

    # Cost analysis
    unit_cost: float = Field(..., ge=0, description="Unit cost ($)")
    annual_holding_cost: float = Field(..., ge=0, description="Annual holding cost ($)")
    stockout_risk: float = Field(..., ge=0, le=1, description="Stockout risk")

    # Lead time
    lead_time_days: int = Field(default=30, ge=0, description="Lead time (days)")
    criticality: CriticalityLevel = Field(
        default=CriticalityLevel.MEDIUM, description="Part criticality"
    )


class InventoryOptimizationResult(BaseModel):
    """Result of inventory optimization."""

    analysis_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Recommendations
    recommendations: List[SparePartRecommendation] = Field(
        default_factory=list, description="Part recommendations"
    )

    # Summary
    total_current_inventory_value: float = Field(
        ..., ge=0, description="Current inventory value ($)"
    )
    total_recommended_inventory_value: float = Field(
        ..., ge=0, description="Recommended inventory value ($)"
    )
    inventory_value_change: float = Field(
        ..., description="Change in inventory value ($)"
    )

    # Risk metrics
    current_service_level: float = Field(
        ..., ge=0, le=1, description="Current service level"
    )
    target_service_level: float = Field(
        default=0.95, ge=0, le=1, description="Target service level"
    )


# =============================================================================
# ECONOMIC REPLACEMENT MODEL
# =============================================================================

class EconomicReplacementModel:
    """
    Economic analysis for burner replacement decisions.

    Implements NPV, IRR, and payback calculations for replace vs. continue
    operating analysis with full consideration of:
    - Fuel cost savings from efficiency improvement
    - Maintenance cost reduction
    - Failure risk reduction
    - Emissions cost reduction

    All calculations are DETERMINISTIC with full provenance tracking.

    Example:
        >>> model = EconomicReplacementModel()
        >>> npv = model.calculate_npv_replacement(burner, econ_params)
        >>> print(f"NPV: ${npv:,.0f}")
    """

    def __init__(self) -> None:
        """Initialize economic replacement model."""
        logger.info("EconomicReplacementModel initialized")

    def calculate_npv_replacement(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate NPV of replacement vs. continuing operation.

        Args:
            asset: Burner asset information
            params: Economic parameters

        Returns:
            Tuple of (net_npv, cost_breakdown)
        """
        logger.info(f"Calculating NPV for {asset.asset_id}")

        # Calculate annual benefits of replacement
        fuel_savings = self._calculate_fuel_savings(asset, params)
        maint_savings = self._calculate_maintenance_savings(asset, params)
        emissions_savings = self._calculate_emissions_savings(asset, params)
        failure_cost_avoided = self._calculate_avoided_failure_cost(asset, params)

        total_annual_savings = (
            fuel_savings + maint_savings + emissions_savings + failure_cost_avoided
        )

        # Calculate total replacement cost
        total_replacement_cost = asset.replacement_cost + asset.installation_cost

        # Calculate NPV of benefits
        npv_benefits = 0.0
        for year in range(1, params.analysis_horizon_years + 1):
            # Escalate fuel savings
            escalated_fuel = fuel_savings * (
                (1 + params.fuel_cost_escalation_rate) ** year
            )
            # Keep other savings constant (conservative)
            annual_benefit = (
                escalated_fuel + maint_savings + emissions_savings + failure_cost_avoided
            )
            # Discount to present value
            discount_factor = 1 / ((1 + params.discount_rate) ** year)
            npv_benefits += annual_benefit * discount_factor

        # Calculate depreciation tax shield
        annual_depreciation = total_replacement_cost / params.depreciation_years
        npv_tax_shield = 0.0
        for year in range(1, min(params.depreciation_years, params.analysis_horizon_years) + 1):
            tax_benefit = annual_depreciation * params.tax_rate
            discount_factor = 1 / ((1 + params.discount_rate) ** year)
            npv_tax_shield += tax_benefit * discount_factor

        # Net NPV
        net_npv = npv_benefits + npv_tax_shield - total_replacement_cost

        cost_breakdown = {
            "fuel_savings_annual": fuel_savings,
            "maintenance_savings_annual": maint_savings,
            "emissions_savings_annual": emissions_savings,
            "avoided_failure_cost_annual": failure_cost_avoided,
            "total_annual_savings": total_annual_savings,
            "total_replacement_cost": total_replacement_cost,
            "npv_benefits": npv_benefits,
            "npv_tax_shield": npv_tax_shield,
            "net_npv": net_npv,
        }

        logger.info(f"NPV calculation complete: ${net_npv:,.0f}")
        return net_npv, cost_breakdown

    def calculate_irr(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        max_iterations: int = 100,
        tolerance: float = 0.0001,
    ) -> Optional[float]:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        Args:
            asset: Burner asset information
            params: Economic parameters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            IRR as decimal (0.15 = 15%) or None if not converging
        """
        # Calculate cash flows
        total_cost = asset.replacement_cost + asset.installation_cost
        fuel_savings = self._calculate_fuel_savings(asset, params)
        maint_savings = self._calculate_maintenance_savings(asset, params)
        emissions_savings = self._calculate_emissions_savings(asset, params)
        failure_avoided = self._calculate_avoided_failure_cost(asset, params)

        annual_savings = fuel_savings + maint_savings + emissions_savings + failure_avoided

        # Cash flows: Year 0 = -cost, Years 1-N = savings
        cash_flows = [-total_cost]
        for year in range(1, params.analysis_horizon_years + 1):
            # Escalate fuel savings only
            escalated_fuel = fuel_savings * (
                (1 + params.fuel_cost_escalation_rate) ** year
            )
            annual = escalated_fuel + maint_savings + emissions_savings + failure_avoided
            cash_flows.append(annual)

        # Newton-Raphson for IRR
        irr_guess = 0.10  # Start at 10%

        for _ in range(max_iterations):
            npv = sum(
                cf / ((1 + irr_guess) ** i) for i, cf in enumerate(cash_flows)
            )

            # Derivative of NPV w.r.t. rate
            npv_derivative = sum(
                -i * cf / ((1 + irr_guess) ** (i + 1))
                for i, cf in enumerate(cash_flows) if i > 0
            )

            if abs(npv_derivative) < 1e-10:
                break

            irr_new = irr_guess - npv / npv_derivative

            if abs(irr_new - irr_guess) < tolerance:
                return irr_new

            irr_guess = irr_new

            # Guard against divergence
            if irr_guess < -0.9 or irr_guess > 10:
                return None

        return irr_guess if abs(npv) < 1000 else None

    def calculate_payback_period(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> Optional[float]:
        """
        Calculate simple payback period in years.

        Args:
            asset: Burner asset information
            params: Economic parameters

        Returns:
            Payback period in years or None if savings <= 0
        """
        total_cost = asset.replacement_cost + asset.installation_cost

        fuel_savings = self._calculate_fuel_savings(asset, params)
        maint_savings = self._calculate_maintenance_savings(asset, params)
        emissions_savings = self._calculate_emissions_savings(asset, params)
        failure_avoided = self._calculate_avoided_failure_cost(asset, params)

        annual_savings = fuel_savings + maint_savings + emissions_savings + failure_avoided

        if annual_savings <= 0:
            return None

        return total_cost / annual_savings

    def _calculate_fuel_savings(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate annual fuel savings from efficiency improvement."""
        # Efficiency improvement
        eff_current = asset.current_efficiency_pct / 100
        eff_new = params.new_efficiency_pct / 100

        if eff_current <= 0:
            return 0.0

        # Annual fuel consumption reduction
        # Fuel_new / Fuel_old = eff_old / eff_new
        fuel_reduction_factor = 1 - (eff_current / eff_new)

        # Annual fuel consumption at current efficiency
        annual_heat_input = asset.heat_input_mmbtu_hr * params.operating_hours_per_year
        annual_fuel_mmbtu = annual_heat_input / eff_current

        # Fuel savings
        fuel_savings = annual_fuel_mmbtu * fuel_reduction_factor * params.fuel_cost_per_mmbtu

        return max(0, fuel_savings)

    def _calculate_maintenance_savings(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate maintenance cost savings."""
        # Current maintenance cost (escalated with age)
        age_factor = asset.age_factor
        current_maint = params.preventive_maint_cost_per_year * (
            1 + (params.maint_cost_escalation_factor - 1) * age_factor
        )

        # New equipment maintenance (baseline)
        new_maint = params.preventive_maint_cost_per_year

        return max(0, current_maint - new_maint)

    def _calculate_emissions_savings(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate emissions cost savings (NOx + CO2)."""
        annual_heat = asset.heat_input_mmbtu_hr * params.operating_hours_per_year

        # NOx savings
        nox_reduction_lb_mmbtu = asset.nox_emissions_lb_mmbtu - params.new_nox_lb_mmbtu
        nox_reduction_tons = (nox_reduction_lb_mmbtu * annual_heat) / 2000
        nox_savings = nox_reduction_tons * params.nox_emission_cost_per_ton

        # CO2 savings from reduced fuel use
        eff_current = asset.current_efficiency_pct / 100
        eff_new = params.new_efficiency_pct / 100

        if eff_current > 0:
            fuel_reduction_factor = 1 - (eff_current / eff_new)
            # Approximate CO2 factor for natural gas: 117 lb/MMBtu
            co2_reduction_lb = (annual_heat / eff_current) * fuel_reduction_factor * 117
            co2_reduction_tons = co2_reduction_lb / 2000
            co2_savings = co2_reduction_tons * params.carbon_cost_per_ton
        else:
            co2_savings = 0.0

        return max(0, nox_savings + co2_savings)

    def _calculate_avoided_failure_cost(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate expected avoided failure cost."""
        # Calculate failure probability using Weibull
        failure_prob = self._weibull_failure_probability(
            asset.current_age_hours,
            asset.weibull_beta,
            asset.weibull_eta
        )

        # Expected failures per year
        expected_failures = failure_prob * (
            params.operating_hours_per_year / asset.expected_life_hours
        )

        # Cost per failure
        failure_cost = (
            params.unplanned_failure_cost +
            params.lost_production_cost_per_hour * params.avg_repair_time_hours
        )

        # Expected annual avoided cost (new equipment has much lower failure rate)
        new_failure_rate = 0.02  # Assume 2% failure rate for new equipment
        avoided_failures = expected_failures - new_failure_rate

        return max(0, avoided_failures * failure_cost)

    def _weibull_failure_probability(
        self,
        t: float,
        beta: float,
        eta: float,
    ) -> float:
        """Calculate Weibull cumulative failure probability."""
        if t <= 0 or eta <= 0:
            return 0.0
        return 1 - math.exp(-((t / eta) ** beta))


# =============================================================================
# OPTIMAL TIMING CALCULATOR
# =============================================================================

class OptimalTimingCalculator:
    """
    Calculate optimal replacement timing using cost minimization.

    Minimizes: C_total = C_maintenance + C_failure_risk + C_efficiency_loss

    Uses Weibull-based failure probability and deterministic calculations.

    Example:
        >>> calculator = OptimalTimingCalculator()
        >>> result = calculator.find_optimal_time(asset, params)
        >>> print(f"Optimal age: {result['optimal_hours']:,.0f} hours")
    """

    def __init__(self) -> None:
        """Initialize optimal timing calculator."""
        logger.info("OptimalTimingCalculator initialized")

    def find_optimal_replacement_time(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        time_step_hours: float = 1000,
        max_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal replacement time minimizing total cost.

        Args:
            asset: Burner asset
            params: Economic parameters
            time_step_hours: Time step for analysis
            max_hours: Maximum hours to analyze

        Returns:
            Dictionary with optimal timing results
        """
        logger.info(f"Finding optimal replacement time for {asset.asset_id}")

        max_hours = max_hours or asset.expected_life_hours * 2
        current_hours = asset.current_age_hours

        # Search for minimum total cost
        min_total_cost = float('inf')
        optimal_hours = current_hours
        cost_curve = []

        for t in range(
            int(current_hours),
            int(max_hours),
            int(time_step_hours)
        ):
            if t < current_hours:
                continue

            # Calculate costs at this time
            maint_cost = self._maintenance_cost_to_time(
                t, current_hours, asset, params
            )
            failure_cost = self._expected_failure_cost_to_time(
                t, current_hours, asset, params
            )
            efficiency_cost = self._efficiency_loss_cost_to_time(
                t, current_hours, asset, params
            )

            total_cost = maint_cost + failure_cost + efficiency_cost

            cost_curve.append({
                "hours": t,
                "maintenance_cost": maint_cost,
                "failure_cost": failure_cost,
                "efficiency_cost": efficiency_cost,
                "total_cost": total_cost,
            })

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                optimal_hours = t

        # Calculate optimal date
        hours_until_optimal = optimal_hours - current_hours
        days_until_optimal = hours_until_optimal / (params.operating_hours_per_year / 365)
        optimal_date = datetime.now(timezone.utc) + timedelta(days=days_until_optimal)

        result = {
            "optimal_hours": optimal_hours,
            "optimal_date": optimal_date,
            "hours_until_replacement": hours_until_optimal,
            "minimum_total_cost": min_total_cost,
            "cost_curve": cost_curve,
            "current_age_hours": current_hours,
        }

        logger.info(f"Optimal replacement: {optimal_hours:,.0f} hours")
        return result

    def _maintenance_cost_to_time(
        self,
        t: float,
        t_current: float,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate cumulative maintenance cost from current to time t."""
        if t <= t_current:
            return 0.0

        years = (t - t_current) / params.operating_hours_per_year

        # Integrate maintenance cost with escalation
        total_cost = 0.0
        for year in range(1, int(years) + 1):
            age_at_year = t_current + year * params.operating_hours_per_year
            age_factor = age_at_year / asset.expected_life_hours
            annual_cost = params.preventive_maint_cost_per_year * (
                1 + (params.maint_cost_escalation_factor - 1) * min(age_factor, 2.0)
            )
            # Discount to present value
            discount = 1 / ((1 + params.discount_rate) ** year)
            total_cost += annual_cost * discount

        return total_cost

    def _expected_failure_cost_to_time(
        self,
        t: float,
        t_current: float,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate expected failure cost from current to time t."""
        if t <= t_current:
            return 0.0

        # Failure cost
        cost_per_failure = (
            params.unplanned_failure_cost +
            params.lost_production_cost_per_hour * params.avg_repair_time_hours
        )

        # Expected number of failures using Weibull hazard rate integral
        # Simplified: use average hazard rate
        p_current = 1 - math.exp(-((t_current / asset.weibull_eta) ** asset.weibull_beta))
        p_t = 1 - math.exp(-((t / asset.weibull_eta) ** asset.weibull_beta))

        # Conditional probability of failure
        if p_current >= 1:
            return cost_per_failure * 2  # High cost if already past life

        p_fail_given_survived = (p_t - p_current) / (1 - p_current)

        # Expected failures (can be > 1 if past expected life)
        expected_failures = p_fail_given_survived

        # Discount to midpoint
        years = (t - t_current) / params.operating_hours_per_year / 2
        discount = 1 / ((1 + params.discount_rate) ** years)

        return expected_failures * cost_per_failure * discount

    def _efficiency_loss_cost_to_time(
        self,
        t: float,
        t_current: float,
        asset: BurnerAsset,
        params: EconomicParameters,
    ) -> float:
        """Calculate efficiency degradation cost from current to time t."""
        if t <= t_current:
            return 0.0

        years = (t - t_current) / params.operating_hours_per_year

        # Assume efficiency degrades linearly with age
        # Current degradation rate
        current_deg_rate = (
            asset.design_efficiency_pct - asset.current_efficiency_pct
        ) / (asset.current_age_hours / params.operating_hours_per_year)

        if current_deg_rate <= 0:
            current_deg_rate = 0.5  # Default 0.5% per year

        # Annual fuel consumption at design efficiency
        annual_heat = asset.heat_input_mmbtu_hr * params.operating_hours_per_year
        design_fuel = annual_heat / (asset.design_efficiency_pct / 100)

        total_cost = 0.0
        for year in range(1, int(years) + 1):
            # Efficiency at year
            eff_at_year = asset.current_efficiency_pct - current_deg_rate * year
            eff_at_year = max(70, eff_at_year)  # Floor at 70%

            # Extra fuel cost
            actual_fuel = annual_heat / (eff_at_year / 100)
            extra_fuel = actual_fuel - design_fuel
            extra_cost = extra_fuel * params.fuel_cost_per_mmbtu * (
                (1 + params.fuel_cost_escalation_rate) ** year
            )

            # Discount
            discount = 1 / ((1 + params.discount_rate) ** year)
            total_cost += extra_cost * discount

        return total_cost


# =============================================================================
# GROUP REPLACEMENT STRATEGY
# =============================================================================

class GroupReplacementStrategy:
    """
    Group replacement strategies for economies of scale.

    Implements:
    - Age-based grouping
    - Block replacement scheduling
    - Staggered replacement for continuous operation

    Example:
        >>> strategy = GroupReplacementStrategy()
        >>> result = strategy.analyze_group_replacement(assets, params)
    """

    # Group discount factors by quantity
    GROUP_DISCOUNT_FACTORS = {
        1: 1.00,
        2: 0.95,
        3: 0.92,
        5: 0.88,
        10: 0.82,
        20: 0.75,
    }

    def __init__(self) -> None:
        """Initialize group replacement strategy."""
        logger.info("GroupReplacementStrategy initialized")

    def analyze_group_replacement(
        self,
        assets: List[BurnerAsset],
        params: EconomicParameters,
        grouping_window_years: float = 2.0,
    ) -> GroupReplacementResult:
        """
        Analyze economics of group vs. individual replacement.

        Args:
            assets: List of burner assets
            params: Economic parameters
            grouping_window_years: Window for grouping (years)

        Returns:
            GroupReplacementResult with recommendations
        """
        logger.info(f"Analyzing group replacement for {len(assets)} assets")

        if not assets:
            raise ValueError("At least one asset required")

        # Calculate individual replacement costs
        individual_costs = []
        replacement_dates = []

        timing_calc = OptimalTimingCalculator()
        econ_model = EconomicReplacementModel()

        for asset in assets:
            timing = timing_calc.find_optimal_replacement_time(asset, params)
            individual_costs.append(asset.replacement_cost + asset.installation_cost)
            replacement_dates.append(timing["optimal_date"])

        total_individual_cost = sum(individual_costs)

        # Calculate group replacement cost with discount
        n_assets = len(assets)
        discount_factor = self._get_group_discount(n_assets)
        total_group_cost = total_individual_cost * discount_factor

        # Calculate savings
        group_savings = total_individual_cost - total_group_cost
        savings_pct = (group_savings / total_individual_cost * 100) if total_individual_cost > 0 else 0

        # Determine recommended replacement date (weighted by urgency)
        avg_date = sum(
            (d - datetime.now(timezone.utc)).total_seconds() for d in replacement_dates
        ) / len(replacement_dates)
        recommended_date = datetime.now(timezone.utc) + timedelta(seconds=avg_date)

        # Determine strategy
        if n_assets >= 5:
            strategy = ReplacementStrategy.BLOCK_REPLACEMENT
            recommendation = (
                f"Block replacement of {n_assets} burners recommended. "
                f"Group savings: ${group_savings:,.0f} ({savings_pct:.1f}%). "
                f"Coordinate with next turnaround."
            )
        elif n_assets >= 2:
            strategy = ReplacementStrategy.GROUP_REPLACEMENT
            recommendation = (
                f"Group replacement of {n_assets} burners recommended. "
                f"Savings: ${group_savings:,.0f} ({savings_pct:.1f}%)."
            )
        else:
            strategy = ReplacementStrategy.CONDITION_BASED
            recommendation = (
                f"Individual condition-based replacement recommended. "
                f"Continue monitoring."
            )

        return GroupReplacementResult(
            asset_ids=[a.asset_id for a in assets],
            total_assets=n_assets,
            individual_replacement_cost=total_individual_cost,
            group_replacement_cost=total_group_cost,
            group_savings=group_savings,
            savings_percentage=savings_pct,
            recommended_replacement_date=recommended_date,
            strategy=strategy,
            recommendation=recommendation,
        )

    def _get_group_discount(self, n: int) -> float:
        """Get discount factor for group size."""
        for threshold, factor in sorted(
            self.GROUP_DISCOUNT_FACTORS.items(), reverse=True
        ):
            if n >= threshold:
                return factor
        return 1.0

    def create_staggered_schedule(
        self,
        assets: List[BurnerAsset],
        params: EconomicParameters,
        max_simultaneous: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Create staggered replacement schedule for continuous operation.

        Args:
            assets: List of assets
            params: Economic parameters
            max_simultaneous: Maximum simultaneous replacements

        Returns:
            List of scheduled replacements
        """
        logger.info(f"Creating staggered schedule for {len(assets)} assets")

        # Sort assets by urgency (failure probability)
        def urgency_score(asset: BurnerAsset) -> float:
            p_fail = 1 - math.exp(
                -((asset.current_age_hours / asset.weibull_eta) ** asset.weibull_beta)
            )
            return p_fail

        sorted_assets = sorted(assets, key=urgency_score, reverse=True)

        schedule = []
        current_date = datetime.now(timezone.utc)
        batch_size = max_simultaneous

        for i, asset in enumerate(sorted_assets):
            batch_number = i // batch_size
            # Space batches 3 months apart
            scheduled_date = current_date + timedelta(days=90 * batch_number)

            schedule.append({
                "asset_id": asset.asset_id,
                "tag_number": asset.tag_number,
                "scheduled_date": scheduled_date,
                "batch_number": batch_number + 1,
                "urgency_score": urgency_score(asset),
                "current_age_hours": asset.current_age_hours,
            })

        return schedule


# =============================================================================
# INVENTORY OPTIMIZER
# =============================================================================

class InventoryOptimizer:
    """
    Spare parts inventory optimization using ABC analysis.

    Implements:
    - ABC classification by value/criticality
    - Safety stock calculation
    - Reorder point optimization
    - Lead time consideration

    Example:
        >>> optimizer = InventoryOptimizer()
        >>> result = optimizer.optimize_inventory(parts, params)
    """

    # Service level to safety factor mapping
    SERVICE_LEVEL_Z = {
        0.90: 1.282,
        0.95: 1.645,
        0.99: 2.326,
    }

    def __init__(self) -> None:
        """Initialize inventory optimizer."""
        logger.info("InventoryOptimizer initialized")

    def optimize_inventory(
        self,
        parts: List[Dict[str, Any]],
        demand_history: Dict[str, List[float]],
        target_service_level: float = 0.95,
        holding_cost_rate: float = 0.25,
    ) -> InventoryOptimizationResult:
        """
        Optimize spare parts inventory.

        Args:
            parts: List of part information dictionaries
            demand_history: Historical demand by part number
            target_service_level: Target service level (0-1)
            holding_cost_rate: Annual holding cost as fraction of value

        Returns:
            InventoryOptimizationResult
        """
        logger.info(f"Optimizing inventory for {len(parts)} parts")

        # Classify parts using ABC analysis
        classified_parts = self._abc_classification(parts)

        recommendations = []
        total_current_value = 0.0
        total_recommended_value = 0.0

        for part in classified_parts:
            part_number = part["part_number"]
            unit_cost = part["unit_cost"]
            current_qty = part.get("current_quantity", 0)
            lead_time = part.get("lead_time_days", 30)
            criticality = part.get("criticality", CriticalityLevel.MEDIUM)

            # Get demand statistics
            demand = demand_history.get(part_number, [1])
            avg_demand = sum(demand) / len(demand) if demand else 1
            std_demand = (
                math.sqrt(sum((d - avg_demand) ** 2 for d in demand) / len(demand))
                if len(demand) > 1 else avg_demand * 0.3
            )

            # Calculate safety stock
            z = self._get_service_z(target_service_level)
            safety_stock = z * std_demand * math.sqrt(lead_time / 30)

            # Reorder point
            lead_time_demand = avg_demand * (lead_time / 30)
            reorder_point = int(math.ceil(lead_time_demand + safety_stock))

            # EOQ (Economic Order Quantity)
            # Simplified: assume annual demand = 12 * avg_demand
            annual_demand = 12 * avg_demand
            order_cost = 100  # Fixed order cost
            if unit_cost > 0:
                eoq = math.sqrt(2 * annual_demand * order_cost / (unit_cost * holding_cost_rate))
            else:
                eoq = 10
            order_qty = max(1, int(round(eoq)))

            # Recommended quantity
            recommended_qty = max(reorder_point, current_qty)
            if part["category"] == SparePartCategory.A_CRITICAL:
                recommended_qty = max(recommended_qty, int(safety_stock * 2))

            # Holding cost
            holding_cost = recommended_qty * unit_cost * holding_cost_rate

            # Stockout risk
            if current_qty > reorder_point:
                stockout_risk = 0.01
            elif current_qty > 0:
                stockout_risk = 0.1
            else:
                stockout_risk = 0.5

            recommendations.append(SparePartRecommendation(
                part_number=part_number,
                description=part.get("description", ""),
                category=part["category"],
                current_quantity=current_qty,
                recommended_quantity=recommended_qty,
                reorder_point=reorder_point,
                order_quantity=order_qty,
                unit_cost=unit_cost,
                annual_holding_cost=holding_cost,
                stockout_risk=stockout_risk,
                lead_time_days=lead_time,
                criticality=criticality,
            ))

            total_current_value += current_qty * unit_cost
            total_recommended_value += recommended_qty * unit_cost

        # Sort by category then value
        recommendations.sort(key=lambda x: (x.category.value, -x.unit_cost))

        return InventoryOptimizationResult(
            recommendations=recommendations,
            total_current_inventory_value=total_current_value,
            total_recommended_inventory_value=total_recommended_value,
            inventory_value_change=total_recommended_value - total_current_value,
            current_service_level=self._estimate_current_service_level(recommendations),
            target_service_level=target_service_level,
        )

    def _abc_classification(
        self,
        parts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Classify parts using ABC analysis."""
        # Calculate annual value for each part
        for part in parts:
            annual_demand = part.get("annual_demand", 12)
            unit_cost = part.get("unit_cost", 0)
            part["annual_value"] = annual_demand * unit_cost

        # Sort by annual value descending
        sorted_parts = sorted(parts, key=lambda x: x["annual_value"], reverse=True)

        # Calculate cumulative percentage
        total_value = sum(p["annual_value"] for p in sorted_parts)
        if total_value <= 0:
            for p in sorted_parts:
                p["category"] = SparePartCategory.C_ROUTINE
            return sorted_parts

        cumulative = 0
        for part in sorted_parts:
            cumulative += part["annual_value"]
            pct = cumulative / total_value

            if pct <= 0.80:
                part["category"] = SparePartCategory.A_CRITICAL
            elif pct <= 0.95:
                part["category"] = SparePartCategory.B_IMPORTANT
            else:
                part["category"] = SparePartCategory.C_ROUTINE

        return sorted_parts

    def _get_service_z(self, service_level: float) -> float:
        """Get z-score for service level."""
        for level, z in sorted(self.SERVICE_LEVEL_Z.items()):
            if service_level <= level:
                return z
        return 2.326  # 99%

    def _estimate_current_service_level(
        self,
        recommendations: List[SparePartRecommendation],
    ) -> float:
        """Estimate current service level from stockout risks."""
        if not recommendations:
            return 0.9

        avg_stockout = sum(r.stockout_risk for r in recommendations) / len(recommendations)
        return 1 - avg_stockout


# =============================================================================
# MONTE CARLO SIMULATOR
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for replacement analysis uncertainty.

    Simulates uncertainty in:
    - Fuel prices
    - Failure rates
    - Efficiency degradation
    - Maintenance costs

    Example:
        >>> simulator = MonteCarloSimulator()
        >>> results = simulator.simulate_npv(asset, params, n_simulations=1000)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize Monte Carlo simulator."""
        if seed is not None:
            random.seed(seed)
        logger.info("MonteCarloSimulator initialized")

    def simulate_npv(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        n_simulations: int = 1000,
        fuel_price_std: float = 0.2,
        failure_rate_std: float = 0.3,
        efficiency_std: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for NPV uncertainty.

        Args:
            asset: Burner asset
            params: Economic parameters
            n_simulations: Number of simulations
            fuel_price_std: Fuel price standard deviation (fraction)
            failure_rate_std: Failure rate std (fraction)
            efficiency_std: Efficiency std (absolute %)

        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations")

        econ_model = EconomicReplacementModel()
        npv_results = []

        base_npv, _ = econ_model.calculate_npv_replacement(asset, params)

        for _ in range(n_simulations):
            # Perturb parameters
            perturbed_params = params.model_copy()

            # Fuel price variation (lognormal)
            fuel_factor = random.lognormvariate(0, fuel_price_std)
            perturbed_params.fuel_cost_per_mmbtu = params.fuel_cost_per_mmbtu * fuel_factor

            # Failure rate variation
            failure_factor = random.lognormvariate(0, failure_rate_std)
            perturbed_params.unplanned_failure_cost = (
                params.unplanned_failure_cost * failure_factor
            )

            # Efficiency variation
            eff_delta = random.gauss(0, efficiency_std)
            perturbed_params.new_efficiency_pct = params.new_efficiency_pct + eff_delta

            # Calculate NPV with perturbed parameters
            npv, _ = econ_model.calculate_npv_replacement(asset, perturbed_params)
            npv_results.append(npv)

        # Calculate statistics
        npv_results.sort()
        n = len(npv_results)

        results = {
            "n_simulations": n_simulations,
            "mean_npv": sum(npv_results) / n,
            "std_npv": math.sqrt(sum((x - sum(npv_results)/n)**2 for x in npv_results) / n),
            "min_npv": npv_results[0],
            "max_npv": npv_results[-1],
            "p10_npv": npv_results[int(n * 0.1)],
            "p50_npv": npv_results[int(n * 0.5)],
            "p90_npv": npv_results[int(n * 0.9)],
            "prob_positive": sum(1 for x in npv_results if x > 0) / n,
            "base_npv": base_npv,
        }

        logger.info(
            f"Monte Carlo complete: Mean NPV=${results['mean_npv']:,.0f}, "
            f"P10=${results['p10_npv']:,.0f}, P90=${results['p90_npv']:,.0f}"
        )

        return results


# =============================================================================
# REPLACEMENT PLANNER - MAIN CLASS
# =============================================================================

class ReplacementPlanner:
    """
    Optimal burner replacement scheduling with economics.

    Implements replacement optimization:
    - Economic replacement analysis (NPV, IRR)
    - Optimal replacement timing (minimize total cost)
    - Group replacement strategies
    - Spare parts inventory optimization
    - Turnaround/outage coordination

    This class integrates all replacement planning components into
    a unified interface for burner maintenance decision support.

    All calculations are DETERMINISTIC with full provenance tracking.
    No ML in calculation path - ZERO HALLUCINATION compliance.

    Attributes:
        economic_model: Economic analysis model
        timing_calculator: Optimal timing calculator
        group_strategy: Group replacement strategy
        inventory_optimizer: Spare parts optimizer
        monte_carlo: Monte Carlo simulator

    Example:
        >>> planner = ReplacementPlanner()
        >>> result = planner.analyze_replacement(burner, econ_params)
        >>> print(f"NPV: ${result.npv_replacement:,.0f}")
        >>> print(f"Optimal replacement: {result.optimal_replacement_date}")
        >>> print(f"Recommendation: {result.recommendation}")
    """

    def __init__(
        self,
        enable_monte_carlo: bool = True,
        monte_carlo_simulations: int = 1000,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize ReplacementPlanner.

        Args:
            enable_monte_carlo: Enable Monte Carlo uncertainty analysis
            monte_carlo_simulations: Number of MC simulations
            random_seed: Random seed for reproducibility
        """
        self.economic_model = EconomicReplacementModel()
        self.timing_calculator = OptimalTimingCalculator()
        self.group_strategy = GroupReplacementStrategy()
        self.inventory_optimizer = InventoryOptimizer()

        self.enable_monte_carlo = enable_monte_carlo
        self.monte_carlo_simulations = monte_carlo_simulations

        if enable_monte_carlo:
            self.monte_carlo = MonteCarloSimulator(seed=random_seed)
        else:
            self.monte_carlo = None

        logger.info(
            f"ReplacementPlanner initialized (MC={'enabled' if enable_monte_carlo else 'disabled'})"
        )

    def analyze_replacement(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        include_sensitivity: bool = True,
    ) -> ReplacementAnalysisResult:
        """
        Perform comprehensive replacement analysis.

        Args:
            asset: Burner asset to analyze
            params: Economic parameters
            include_sensitivity: Include sensitivity analysis

        Returns:
            ReplacementAnalysisResult with comprehensive analysis
        """
        logger.info(f"Analyzing replacement for {asset.asset_id}")
        start_time = datetime.now(timezone.utc)

        # Economic analysis
        npv_replacement, cost_breakdown = self.economic_model.calculate_npv_replacement(
            asset, params
        )
        irr = self.economic_model.calculate_irr(asset, params)
        payback = self.economic_model.calculate_payback_period(asset, params)

        # NPV of continuing (negative of savings stream)
        npv_continue = -npv_replacement + (asset.replacement_cost + asset.installation_cost)

        # Optimal timing
        timing_result = self.timing_calculator.find_optimal_replacement_time(
            asset, params
        )

        # Failure probability
        current_failure_prob = 1 - math.exp(
            -((asset.current_age_hours / asset.weibull_eta) ** asset.weibull_beta)
        )
        expected_failures = current_failure_prob * (
            params.operating_hours_per_year / asset.expected_life_hours
        )

        # Monte Carlo analysis
        mc_results = None
        if self.enable_monte_carlo and self.monte_carlo:
            mc_results = self.monte_carlo.simulate_npv(
                asset, params, n_simulations=self.monte_carlo_simulations
            )

        # Sensitivity analysis
        sensitivity = {}
        if include_sensitivity:
            sensitivity = self._sensitivity_analysis(asset, params, npv_replacement)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            npv_replacement, irr, payback, current_failure_prob, asset
        )

        # Calculate confidence level
        confidence = self._calculate_confidence(
            asset, params, mc_results
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            asset, params, npv_replacement, timing_result
        )

        result = ReplacementAnalysisResult(
            asset_id=asset.asset_id,
            npv_continue=npv_continue,
            npv_replacement=npv_replacement,
            npv_advantage=npv_replacement,
            irr_replacement=irr,
            payback_period_years=payback,
            optimal_replacement_date=timing_result["optimal_date"],
            optimal_replacement_hours=timing_result["optimal_hours"],
            annual_savings=cost_breakdown["total_annual_savings"],
            fuel_savings_annual=cost_breakdown["fuel_savings_annual"],
            maintenance_savings_annual=cost_breakdown["maintenance_savings_annual"],
            emissions_savings_annual=cost_breakdown["emissions_savings_annual"],
            avoided_failure_cost_annual=cost_breakdown["avoided_failure_cost_annual"],
            current_failure_probability=current_failure_prob,
            expected_failures_next_year=expected_failures,
            recommendation=recommendation,
            confidence_level=confidence,
            sensitivity_results=sensitivity,
            monte_carlo_npv_mean=mc_results["mean_npv"] if mc_results else None,
            monte_carlo_npv_p10=mc_results["p10_npv"] if mc_results else None,
            monte_carlo_npv_p90=mc_results["p90_npv"] if mc_results else None,
            provenance_hash=provenance_hash,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Replacement analysis complete in {processing_time:.2f}s: "
            f"NPV=${npv_replacement:,.0f}, {recommendation[:50]}..."
        )

        return result

    def analyze_group_replacement(
        self,
        assets: List[BurnerAsset],
        params: EconomicParameters,
    ) -> GroupReplacementResult:
        """
        Analyze group replacement for multiple burners.

        Args:
            assets: List of burner assets
            params: Economic parameters

        Returns:
            GroupReplacementResult
        """
        return self.group_strategy.analyze_group_replacement(assets, params)

    def create_replacement_schedule(
        self,
        assets: List[BurnerAsset],
        params: EconomicParameters,
        max_simultaneous: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Create staggered replacement schedule.

        Args:
            assets: List of assets
            params: Economic parameters
            max_simultaneous: Max simultaneous replacements

        Returns:
            Replacement schedule
        """
        return self.group_strategy.create_staggered_schedule(
            assets, params, max_simultaneous
        )

    def optimize_spare_inventory(
        self,
        parts: List[Dict[str, Any]],
        demand_history: Dict[str, List[float]],
        target_service_level: float = 0.95,
    ) -> InventoryOptimizationResult:
        """
        Optimize spare parts inventory.

        Args:
            parts: Part information
            demand_history: Historical demand
            target_service_level: Target service level

        Returns:
            InventoryOptimizationResult
        """
        return self.inventory_optimizer.optimize_inventory(
            parts, demand_history, target_service_level
        )

    def _sensitivity_analysis(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        base_npv: float,
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity = {}

        # Fuel price sensitivity (+/- 20%)
        params_high_fuel = params.model_copy()
        params_high_fuel.fuel_cost_per_mmbtu = params.fuel_cost_per_mmbtu * 1.2
        npv_high_fuel, _ = self.economic_model.calculate_npv_replacement(
            asset, params_high_fuel
        )
        sensitivity["fuel_price_+20%"] = (npv_high_fuel - base_npv) / base_npv if base_npv != 0 else 0

        params_low_fuel = params.model_copy()
        params_low_fuel.fuel_cost_per_mmbtu = params.fuel_cost_per_mmbtu * 0.8
        npv_low_fuel, _ = self.economic_model.calculate_npv_replacement(
            asset, params_low_fuel
        )
        sensitivity["fuel_price_-20%"] = (npv_low_fuel - base_npv) / base_npv if base_npv != 0 else 0

        # Discount rate sensitivity
        params_high_rate = params.model_copy()
        params_high_rate.discount_rate = params.discount_rate + 0.02
        npv_high_rate, _ = self.economic_model.calculate_npv_replacement(
            asset, params_high_rate
        )
        sensitivity["discount_rate_+2%"] = (npv_high_rate - base_npv) / base_npv if base_npv != 0 else 0

        # Replacement cost sensitivity
        asset_high_cost = asset.model_copy()
        asset_high_cost.replacement_cost = asset.replacement_cost * 1.15
        npv_high_cost, _ = self.economic_model.calculate_npv_replacement(
            asset_high_cost, params
        )
        sensitivity["replacement_cost_+15%"] = (npv_high_cost - base_npv) / base_npv if base_npv != 0 else 0

        return sensitivity

    def _generate_recommendation(
        self,
        npv: float,
        irr: Optional[float],
        payback: Optional[float],
        failure_prob: float,
        asset: BurnerAsset,
    ) -> str:
        """Generate replacement recommendation."""
        recommendations = []

        # NPV-based
        if npv > 50000:
            recommendations.append(
                f"Strong economic case for replacement (NPV=${npv:,.0f})"
            )
        elif npv > 0:
            recommendations.append(
                f"Positive economic case for replacement (NPV=${npv:,.0f})"
            )
        else:
            recommendations.append(
                f"Marginal economic case (NPV=${npv:,.0f}). Consider deferring."
            )

        # IRR-based
        if irr and irr > 0.20:
            recommendations.append(f"Excellent return (IRR={irr*100:.1f}%)")
        elif irr and irr > 0.10:
            recommendations.append(f"Good return (IRR={irr*100:.1f}%)")

        # Payback-based
        if payback and payback < 3:
            recommendations.append(f"Quick payback ({payback:.1f} years)")
        elif payback and payback < 5:
            recommendations.append(f"Reasonable payback ({payback:.1f} years)")

        # Risk-based
        if failure_prob > 0.5:
            recommendations.append(
                "HIGH FAILURE RISK. Urgent replacement recommended."
            )
        elif failure_prob > 0.3:
            recommendations.append(
                "Elevated failure risk. Schedule replacement soon."
            )

        # Efficiency-based
        if asset.efficiency_degradation_pct > 5:
            recommendations.append(
                f"Significant efficiency loss ({asset.efficiency_degradation_pct:.1f}%)"
            )

        return " ".join(recommendations)

    def _calculate_confidence(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        mc_results: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate analysis confidence level."""
        confidence = 0.7  # Base confidence

        # Increase if Monte Carlo shows consistent results
        if mc_results:
            prob_positive = mc_results.get("prob_positive", 0)
            if prob_positive > 0.9 or prob_positive < 0.1:
                confidence += 0.15
            elif prob_positive > 0.7 or prob_positive < 0.3:
                confidence += 0.10

        # Increase if asset has good data
        if asset.weibull_beta > 0 and asset.weibull_eta > 0:
            confidence += 0.10

        # Decrease if efficiency data is missing
        if asset.current_efficiency_pct == asset.design_efficiency_pct:
            confidence -= 0.05

        return min(0.95, max(0.5, confidence))

    def _calculate_provenance(
        self,
        asset: BurnerAsset,
        params: EconomicParameters,
        npv: float,
        timing: Dict[str, Any],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"replacement_planner|{asset.asset_id}|"
            f"{asset.current_age_hours:.6f}|{params.discount_rate:.6f}|"
            f"{npv:.2f}|{timing['optimal_hours']:.2f}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_replacement_planner(
    enable_monte_carlo: bool = True,
    simulations: int = 1000,
) -> ReplacementPlanner:
    """
    Factory function to create ReplacementPlanner.

    Args:
        enable_monte_carlo: Enable Monte Carlo analysis
        simulations: Number of MC simulations

    Returns:
        Configured ReplacementPlanner
    """
    return ReplacementPlanner(
        enable_monte_carlo=enable_monte_carlo,
        monte_carlo_simulations=simulations,
    )


def create_burner_asset(
    asset_id: str,
    tag_number: str,
    burner_type: BurnerType,
    installation_date: datetime,
    current_age_hours: float,
    heat_input_mmbtu_hr: float,
    replacement_cost: float,
    **kwargs: Any,
) -> BurnerAsset:
    """
    Factory function to create BurnerAsset.

    Args:
        asset_id: Unique identifier
        tag_number: Plant tag
        burner_type: Type of burner
        installation_date: Installation date
        current_age_hours: Current operating hours
        heat_input_mmbtu_hr: Heat input capacity
        replacement_cost: Replacement cost
        **kwargs: Additional parameters

    Returns:
        BurnerAsset instance
    """
    return BurnerAsset(
        asset_id=asset_id,
        tag_number=tag_number,
        burner_type=burner_type,
        installation_date=installation_date,
        current_age_hours=current_age_hours,
        heat_input_mmbtu_hr=heat_input_mmbtu_hr,
        replacement_cost=replacement_cost,
        original_cost=kwargs.get("original_cost", replacement_cost * 0.8),
        **kwargs,
    )
